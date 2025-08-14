import numpy as np
import torch
from biotite.structure import lddt, AtomArray
from biotite.structure.io.pdb import PDBFile
from torch.nn import Module
import einx

from models.bio2token.data.utils.tokens import PAD_CLASS
from models.bio2token.data.utils.utils import filter_batch, pad_and_stack_batch, compute_masks, pdb_2_dict, \
    uniform_dataframe

def model_prediction_to_atom_array(sequences, model_prediction, final_mask):
    for i in range(len(sequences)):
        seq = sequences[i]
        protein_length = len(seq)

        # init atom array
        atom_arrray = AtomArray(protein_length * 4)

        # set structure with predictions
        atom_arrray.coord = model_prediction[i][final_mask[i]].detach().cpu().numpy().astype(np.float32)

        # set atom names, residues and sequences
        atom_names = np.array(["N", "CA", "C", "O"])
        atom_arrray.atom_name = np.tile(atom_names, protein_length)
        atom_arrray.res_id = np.repeat(np.arange(1, protein_length + 1), 4)
        seq_array = np.array(list(seq))
        rep_seq_array = np.repeat(seq_array, 4)
        atom_arrray.res_name = rep_seq_array
        atom_arrray.chain_id = np.repeat(np.array(["A"]), protein_length * 4)
        atom_arrray.element = np.tile(np.array(["N", "C", "C", "O"]), protein_length)

        return atom_arrray


def print_tensor(tensor, name):
    print(f"{name}-{tensor.shape}:\n{tensor}")


def _masked_accuracy(logits, tgt, mask):
    pred = logits.argmax(dim=-1)
    correct = ((pred == tgt) & mask).sum().item()
    total = mask.sum().item()
    return correct / total if total else 0.0


def calc_lddt_scores(protein_predictions, ref_protein):
    lddt_scores = []
    for protein_prediction, protein_ref in zip(protein_predictions, ref_protein):
        X, _, _ = protein_prediction.to_XCS(all_atom=False)
        X = X.detach().squeeze(0).reshape(-1, 3).cpu().numpy()
        lddt_scores.append(lddt(protein_ref, X))
    return lddt_scores


def masked_mse_loss(prediction: torch.Tensor,
                    target: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    # (B, L, D)
    squared_error = (prediction - target) ** 2
    # (B, L)
    mse_per_position = squared_error.mean(dim=-1)

    # Maske als float für Gewichtung
    weight = mask.float()

    # Gewichteter Mittelwert der Fehler
    masked_loss = (mse_per_position * weight).sum() / weight.sum()

    return masked_loss


def calc_token_loss(criterion, tokens_predictions, tokens_reference):
    return criterion(tokens_predictions.transpose(1, 2), tokens_reference)


def batch_pdbs_for_bio2token(pdbs, device):
    # read to dicts
    dicts = [pdb_2_dict(pdb) for pdb in pdbs]
    return batch_pdb_dicts(dicts, device)


def batch_pdb_dicts(dicts, device):
    batch = []
    for pdb_dict in dicts:
        structure, unknown_structure, residue_name, residue_ids, token_class, atom_names_reordered = uniform_dataframe(
            pdb_dict["seq"],
            pdb_dict["res_types"],
            pdb_dict["coords_groundtruth"],
            pdb_dict["atom_names"],
            pdb_dict["res_atom_start"],
            pdb_dict["res_atom_end"],
        )
        batch_item = {
            "structure": torch.tensor(structure).float(),
            "unknown_structure": torch.tensor(unknown_structure).bool(),
            "residue_ids": torch.tensor(residue_ids).long(),
            "token_class": torch.tensor(token_class).long(),
        }
        batch_item = {k: v[~batch_item["unknown_structure"]] for k, v in batch_item.items()}

        batch.append(batch_item)

    # taken from config
    sequences_to_pad = {
        "structure": 0,
        "unknown_structure": True,
        "residue_ids": -1,
        "token_class": PAD_CLASS,
        "eos_pad_mask": 1,
        "structure_known_all_atom_mask": 0,
        "bb_atom_known_structure_mask": 0,
        "sc_atom_known_structure_mask": 0,
        "cref_atom_known_structure_mask": 0,
    }
    # batch = [pdb_2_dict(pdb) for pdb in test_pdbs]
    batch = filter_batch(batch, sequences_to_pad.keys())
    batch = pad_and_stack_batch(
        batch,
        sequences_to_pad,
        1
    )

    batch = compute_masks(batch, structure_track=True)
    batch = {k: v.to(device) for k, v in batch.items()}
    return batch


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
            coords_mask,
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


# thanks, David :)
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


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

        loss_val=loss_module(outputs,targets,mask)

        total_loss += weight * loss_val
        loss_components[loss_module.name] = loss_val.item()

    loss_components["total_loss"] = total_loss.item()

    return total_loss, loss_components