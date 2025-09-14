import torch
from torch.nn import Module
import einx

from models.model_utils import center_structure
from models.rigid_utils import Rigid, Rotation


class TmLossModule(Module):
    def __init__(self):
        super().__init__()
        self.name = "TmLossModule"

    def forward(self, P: torch.Tensor, Q: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Calculate squared distances
        d = torch.sum((P - Q) ** 2, dim=-1)  # [batch, length]

        if mask is not None:
            # Count valid residues per sample
            N = mask.sum(dim=-1)  # [batch,]
            # Apply mask to distance matrix
            d0 = 0.6 * torch.pow(N - 0.5, 1 / 2) - 2.5
            d0 = torch.clamp(d0, min=0.5) ** 2
            # Compute TM score by evaluating scaled distances, focusing only on masked elements.
            tm_score = torch.sum((1 / (1 + (d / d0.unsqueeze(-1)))) * mask, dim=-1) / N
        else:
            N = P.shape[1] * torch.ones(P.size(0), device=P.device)
            d0 = 0.6 * torch.pow(N - 0.5, 1 / 2) - 2.5
            d0 = torch.clamp(d0, min=0.5) ** 2
            # Compute TM score considering all elements without masking.
            tm_score = torch.sum((1 / (1 + (d / d0.unsqueeze(-1)))), dim=-1) / N

        return 1 - tm_score.mean()


class InterAtomDistance(Module):
    def __init__(self, c):
        super(InterAtomDistance, self).__init__()
        self.name = "InterAtomDistanceLoss"

    def forward(self, P, Q, mask_remove, idx):
        """
        Compute the Inter-Atom Distance loss for a batch, updating the batch dictionary.

        Args:
            batch (Dict): Dictionary containing input data, including predicted and target coordinates, and optional masks.

        Returns:
            Dict: Updated batch dictionary with the calculated inter-atom distance loss.
        """
        # Retrieve batch size (B), sequence length (L), and channel size (C) from the predictions
        B, L, C = P.shape

        # Determine residue indices if provided

        idx = P.new_zeros(B, L, dtype=torch.long)  # Default to zero indices

        # Initialize variables for loss computation
        loss = P.new_zeros(B)
        n = P.new_zeros(B)  # Counter for valid interactions per batch

        # Calculate the inter-atom distance differences and store in loss
        for b in range(B):
            # Apply the mask to include only specified interactions
            idx_b = idx[b][mask_remove[b]]
            mask_b = torch.tril((idx_b[:, None] - idx_b[None, :]) == 0, diagonal=-1)  # Mask for unique interactions

            # Compute target and predicted inter-atomic distances for masked atoms
            q_b = Q[b][mask_remove[b]]
            p_b = P[b][mask_remove[b]]

            # Compute norm differences for distances
            q_b = torch.linalg.vector_norm((q_b[:, None] - q_b[None, :])[mask_b], dim=-1)
            q_b = q_b - torch.linalg.vector_norm((p_b[:, None] - p_b[None, :])[mask_b], dim=-1)

            # Compute loss per batch entry
            loss[b] = torch.sum((q_b ** 2))
            n[b] = mask_b.sum()  # Number of valid interactions

        # Normalize the loss by the number of interactions and handle numerical stability
        loss = loss / (n + 1e-6)

        # Optionally take the square root of the computed loss
        if self.config.root:
            loss = torch.sqrt(loss + 1e-6)  # Adding small constant for numerical stability

        # Store the computed loss in the batch under the given name
        return loss


def compute_total_loss(outputs, targets, mask, losses, loss_weights):
    total_loss = 0.0
    loss_components = {}

    for loss_module, weight in zip(losses, loss_weights):
        loss_val = loss_module(outputs, targets, mask)

        total_loss += weight * loss_val
        loss_components[loss_module.name] = loss_val.item()

    loss_components["total_loss"] = total_loss.item()

    return total_loss, loss_components


# everything bewlow this line is stolen from
# https://github.com/lucidrains/alphafold3-pytorch/blob/main/alphafold3_pytorch/alphafold3.py#L226


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def to_pairwise_mask(
        mask_i,
        mask_j=None
):
    mask_j = default(mask_j, mask_i)
    assert mask_i.shape == mask_j.shape
    return einx.logical_and('... i, ... j -> ... i j', mask_i, mask_j)


def masked_average(
        t,
        mask,
        *,
        dim,
        eps=1.
):
    num = (t * mask).sum(dim=dim)
    den = mask.sum(dim=dim)
    return num / den.clamp(min=eps)


class SmoothLDDTLoss(Module):
    """ Algorithm 27 """

    def __init__(
            self,
            nucleic_acid_cutoff: float = 30.0,
            other_cutoff: float = 15.0
    ):
        super().__init__()
        self.nucleic_acid_cutoff = nucleic_acid_cutoff
        self.other_cutoff = other_cutoff
        self.name = "SmoothLDDTLoss"

        self.register_buffer('lddt_thresholds', torch.tensor([0.5, 1.0, 2.0, 4.0]))

    def forward(
            self,
            pred_coords,
            true_coords,
            coords_mask
    ):
        """
        pred_coords: predicted coordinates
        true_coords: true coordinates
        is_dna: boolean tensor indicating DNA atoms
        is_rna: boolean tensor indicating RNA atoms
        """
        # make dna and rna masks
        B, L, _ = pred_coords.shape
        is_dna = torch.zeros((B, L), dtype=torch.bool, device=pred_coords.device)
        is_rna = torch.zeros((B, L), dtype=torch.bool, device=pred_coords.device)
        # Compute distances between all pairs of atoms

        pred_dists = torch.cdist(pred_coords, pred_coords)
        true_dists = torch.cdist(true_coords, true_coords)

        # Compute distance difference for all pairs of atoms
        dist_diff = torch.abs(true_dists - pred_dists)

        # Compute epsilon values

        eps = einx.subtract('thresholds, ... -> ... thresholds', self.lddt_thresholds, dist_diff)
        eps = eps.sigmoid().mean(dim=-1)

        # Restrict to bespoke inclusion radius
        is_nucleotide = is_dna | is_rna
        is_nucleotide_pair = to_pairwise_mask(is_nucleotide)

        inclusion_radius = torch.where(
            is_nucleotide_pair,
            true_dists < self.nucleic_acid_cutoff,
            true_dists < self.other_cutoff
        )

        # Compute mean, avoiding self term
        mask = inclusion_radius & ~torch.eye(pred_coords.shape[1], dtype=torch.bool, device=pred_coords.device)

        # Take into account variable lengthed atoms in batch
        if exists(coords_mask):
            paired_coords_mask = to_pairwise_mask(coords_mask)
            mask = mask & paired_coords_mask

        # Calculate masked averaging
        lddt = masked_average(eps, mask=mask, dim=(-1, -2), eps=1)

        return 1. - lddt.mean()


class RmseLoss(Module):
    def __init__(self):
        super().__init__()
        self.root = False
        self.name="RmseLoss"

    def forward(self, pred_coords, true_coords, coords_mask):
        true_coords = center_structure(true_coords, coords_mask)
        P = pred_coords
        Q = true_coords

        # Compute the squared differences between predicted and target coordinates
        squared_diff = torch.sum((P - Q) ** 2, dim=-1)  # Calculate squared Euclidean distances

        # Compute the RMSD using the mask if provided, otherwise compute mean squared difference
        rmsd = torch.sum(squared_diff * coords_mask, dim=1) / (coords_mask.sum(dim=1) + 1e-6)

        # Optionally take the square root of the mean squared deviation to get RMSD
        # ATTENTION: Square root is usually not recommanded when training because not differentiable at 0, and
        # leads to large gradients near 0. Use at your own risk.
        if self.root:
            rmsd = torch.sqrt(rmsd + 1e-6)  # Add small value to prevent divisions by zero

        return rmsd.mean()


class InterAtomDistanceLoss(Module):
    def __init__(self):
        super().__init__()
        self.c_alpha_only = True
        self.name="InterAtomDistanceLoss"

    def forward(self, pred_coords, true_coords, coords_mask):
        """
        Compute the Inter-Atom Distance loss for a batch, updating the batch dictionary.

        """
        # Retrieve batch size (B), sequence length (L), and channel size (C) from the predictions
        B, L, C = pred_coords.shape

        # center structure:
        true_coords = center_structure(true_coords, coords_mask)

        # Extract predicted (P) and target (Q) coordinates
        P = pred_coords
        Q = true_coords

        # get residue ids
        idx = torch.arange(L).unsqueeze(0).expand(B, -1)
        idx = torch.where(coords_mask, idx, torch.full_like(idx, -1))  # Default to zero indices

        # Initialize variables for loss computation
        loss = P.new_zeros(B)
        n = P.new_zeros(B)  # Counter for valid interactions per batch

        # Calculate the inter-atom distance differences and store in loss
        for b in range(B):
            # Apply the mask to include only specified interactions
            idx_b = idx[b][coords_mask[b]]
            mask_b = torch.tril((idx_b[:, None] - idx_b[None, :]) == 0, diagonal=-1)  # Mask for unique interactions

            # Compute target and predicted inter-atomic distances for masked atoms
            q_b = Q[b][coords_mask[b]]
            p_b = P[b][coords_mask[b]]

            # Compute norm differences for distances
            q_b = torch.linalg.vector_norm((q_b[:, None] - q_b[None, :])[mask_b], dim=-1)
            q_b = q_b - torch.linalg.vector_norm((p_b[:, None] - p_b[None, :])[mask_b], dim=-1)

            # Compute loss per batch entry
            loss[b] = torch.sum((q_b ** 2))
            n[b] = mask_b.sum()  # Number of valid interactions

        # Normalize the loss by the number of interactions and handle numerical stability
        loss = loss / (n + 1e-6)

        # Optionally take the square root of the computed loss
        loss = torch.sqrt(loss + 1e-6)  # Adding small constant for numerical stability
        return loss


class FapeLoss(Module):
    def __init__(self):
        super().__init__()
        self.name="FapeLoss"

    @staticmethod
    def create_ca_frames(ca_positions):
        """
        Create simplified frames for C-alpha only structure.
        ca_positions: [batch_size, num_residues, 3] tensor of Cα coordinates
        Returns: Rigid objects representing frames at each Cα
        """
        batch_size, num_residues, _ = ca_positions.shape

        # Use identity rotation for all frames
        # Create identity rotation matrices [3, 3]
        identity_rot = torch.eye(3, device=ca_positions.device, dtype=ca_positions.dtype)
        # Expand to match batch and residue dimensions
        rotations = identity_rot.reshape(1, 1, 3, 3).expand(batch_size, num_residues, 3, 3)

        return Rigid(Rotation(rot_mats=rotations), ca_positions)

    @staticmethod
    def compute_fape(
            pred_frames: Rigid,
            target_frames: Rigid,
            frames_mask: torch.Tensor,
            pred_positions: torch.Tensor,
            target_positions: torch.Tensor,
            positions_mask: torch.Tensor,
            length_scale: float,
            pair_mask = None,
            l1_clamp_distance = None,
            eps=1e-8,
    ) -> torch.Tensor:
        """
            Computes FAPE loss.

            Args:
                pred_frames:
                    [*, N_frames] Rigid object of predicted frames
                target_frames:
                    [*, N_frames] Rigid object of ground truth frames
                frames_mask:
                    [*, N_frames] binary mask for the frames
                pred_positions:
                    [*, N_pts, 3] predicted atom positions
                target_positions:
                    [*, N_pts, 3] ground truth positions
                positions_mask:
                    [*, N_pts] positions mask
                length_scale:
                    Length scale by which the loss is divided
                pair_mask:
                    [*,  N_frames, N_pts] mask to use for
                    separating intra- from inter-chain losses.
                l1_clamp_distance:
                    Cutoff above which distance errors are disregarded
                eps:
                    Small value used to regularize denominators
            Returns:
                [*] loss tensor
        """
        # [*, N_frames, N_pts, 3]
        local_pred_pos = pred_frames.invert()[..., None].apply(
            pred_positions[..., None, :, :],
        )
        local_target_pos = target_frames.invert()[..., None].apply(
            target_positions[..., None, :, :],
        )

        error_dist = torch.sqrt(
            torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
        )

        if l1_clamp_distance is not None:
            error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

        normed_error = error_dist / length_scale
        normed_error = normed_error * frames_mask[..., None]
        normed_error = normed_error * positions_mask[..., None, :]

        if pair_mask is not None:
            normed_error = normed_error * pair_mask
            normed_error = torch.sum(normed_error, dim=(-1, -2))

            mask = frames_mask[..., None] * positions_mask[..., None, :] * pair_mask
            norm_factor = torch.sum(mask, dim=(-2, -1))

            normed_error = normed_error / (eps + norm_factor)
        else:
            # FP16-friendly averaging. Roughly equivalent to:
            #
            # norm_factor = (
            #     torch.sum(frames_mask, dim=-1) *
            #     torch.sum(positions_mask, dim=-1)
            # )
            # normed_error = torch.sum(normed_error, dim=(-1, -2)) / (eps + norm_factor)
            #
            # ("roughly" because eps is necessarily duplicated in the latter)
            normed_error = torch.sum(normed_error, dim=-1)
            normed_error = (
                    normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
            )
            normed_error = torch.sum(normed_error, dim=-1)
            normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))

        return normed_error

    def forward(self, pred_coords, true_coords, coords_mask):
        # For your prediction and ground truth
        pred_frames = self.create_ca_frames(pred_coords)
        true_frames = self.create_ca_frames(true_coords)

        loss = self.compute_fape(
            pred_frames,
            true_frames,
            coords_mask,  # Your residue mask
            pred_coords,
            true_coords,
            coords_mask,  # Same mask for positions
            length_scale=10.0,
            l1_clamp_distance=10.0,
        )

        return loss