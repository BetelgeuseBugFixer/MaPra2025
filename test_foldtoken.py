import json

import torch
# import biotite
import biotite.structure.io.pdb as pdb
from biotite.structure.io.pdb import PDBFile
from biotite.structure import lddt, tm_score, filter_canonical_amino_acids
from biotite.structure.filter import _filter_atom_names

from models.foldtoken_decoder.foldtoken_decoder import FoldDecoder


def load_prot_from_pdb(pdb_file):
    # load
    file = PDBFile.read(pdb_file)
    array_stack = file.get_structure(model=1)
    # filter canonical atoms
    return array_stack[_filter_atom_names(array_stack, ["N", "CA", "C", "O"])]


def load_casp_tokens(token_jsonl):
    token_data = {}

    with open(token_jsonl, "r") as f:
        for line in f:
            entry = json.loads(line)
            token_data.update(entry)
    return token_data


def decode_atom_coordinates(vq_codes):
    protein = model.decode_single_prot(vq_codes, "test.pdb")
    X, _, _ = protein.to_XCS(all_atom=False)
    return X


if __name__ == '__main__':
    device = "cuda"
    model = FoldDecoder(device=device)
    test_pdb = "tokenizer_benchmark/casps/casp14/T1024-D1.pdb"

    # get ref
    ref_protein = load_prot_from_pdb(test_pdb)

    # encode and decode model
    locally_encoded_vq_codes = model.encode_pdb(test_pdb)
    locally_encoded_coords = decode_atom_coordinates(locally_encoded_vq_codes).detach().squeeze(0).reshape(-1, 3).cpu().numpy()

    # also load prev computed tokens
    pre_encoded_vq_codes = torch.tensor(load_casp_tokens("data/casp14_test/casp14_tokens.jsonl")["T1024-D1"]["vqid"],dtype=torch.long,device=device)
    pre_encoded_coords = decode_atom_coordinates(pre_encoded_vq_codes).detach().squeeze(0).reshape(-1, 3).cpu().numpy()

    # get scores
    print(f"locally encoded: {lddt(ref_protein, locally_encoded_coords)}")
    print(f"pre encoded: {lddt(ref_protein, pre_encoded_coords)}")
