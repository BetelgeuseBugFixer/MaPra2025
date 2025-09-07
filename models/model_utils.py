from turtledemo.clock import current_day

import numpy as np
import torch
from biotite.structure import lddt, AtomArray


from models.bio2token.data.utils.tokens import PAD_CLASS
from models.bio2token.data.utils.utils import filter_batch, pad_and_stack_batch, compute_masks, pdb_2_dict, \
    uniform_dataframe

AA1_TO_AA3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    # Sonderfälle / Ambiguitäten
    "U": "SEC",  # Selenocystein
    "O": "PYL",  # Pyrrolysin
    "B": "ASX",  # Asn/Asp
    "Z": "GLX",  # Gln/Glu
    "J": "XLE",  # Leu/Ile
    "X": "UNK",  # unbekannt
}


def seq_to_aa3(seq: str):
    return [AA1_TO_AA3.get(a.upper(), "UNK") for a in seq]

def model_prediction_to_atom_array(sequences, model_prediction, final_mask, only_c_alpha=False):
    atom_arrays = []
    for i in range(len(sequences)):
        seq = sequences[i]
        protein_length = len(seq)

        # init atom array
        if only_c_alpha:
            atoms_per_residue = 1
        else:
            atoms_per_residue=4
        current_atom_arrray = AtomArray(protein_length * atoms_per_residue)


        # set structure with predictions
        current_atom_arrray.coord = model_prediction[i][final_mask[i]].detach().cpu().numpy().astype(np.float32)

        # set atom names
        if only_c_alpha:
            atom_names = np.array(["CA"])
        else:
            atom_names = np.array(["N", "CA", "C", "O"])

        current_atom_arrray.atom_name = np.tile(atom_names, protein_length)
        current_atom_arrray.res_id = np.repeat(np.arange(1, protein_length + 1), atoms_per_residue)
        # seq
        seq_array = np.array(list(seq_to_aa3(seq)))
        rep_seq_array = np.repeat(seq_array, atoms_per_residue)
        current_atom_arrray.res_name = rep_seq_array
        # chain
        current_atom_arrray.chain_id = np.repeat(np.array(["A"]), protein_length * atoms_per_residue)
        # atom types
        if only_c_alpha:
            current_atom_arrray.element = np.tile(np.array(["C"]), protein_length)
        else:
            current_atom_arrray.element = np.tile(np.array(["N", "C", "C", "O"]), protein_length)
        atom_arrays.append(current_atom_arrray)

    return atom_arrays


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


def center_structure(structure, coords_mask):
    barycenter = structure[coords_mask].mean(axis=0)
    structure[coords_mask] = structure[coords_mask] - barycenter
    return structure