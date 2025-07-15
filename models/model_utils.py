import torch
from biotite.structure import lddt
from torch.nn import Module
import einx


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


def calc_token_loss(criterion, tokens_predictions, tokens_reference):
    return criterion(tokens_predictions.transpose(1, 2), tokens_reference)


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

        self.register_buffer('lddt_thresholds', torch.tensor([0.5, 1.0, 2.0, 4.0]))

    def forward(
            self,
            pred_coords,
            true_coords,
            is_dna,
            is_rna,
            coords_mask,
    ):
        """
        pred_coords: predicted coordinates
        true_coords: true coordinates
        is_dna: boolean tensor indicating DNA atoms
        is_rna: boolean tensor indicating RNA atoms
        """
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
