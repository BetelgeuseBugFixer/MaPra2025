import os

from models.bio2token.data.utils.utils import pdb_2_dict
from models.bio2token.decoder import load_bio2_token_decoder_and_quantizer
from models.foldtoken_decoder.foldtoken import FoldToken
from models.model_utils import batch_pdb_dicts
from models.prot_t5.prot_t5 import ProtT5
from utils.prepare_data import DEVICE

BACKBONE_ATOMS = ["N", "CA", "C", "O"]

MAX_LENGTH = 800

TARGET_BATCH_SIZE = 128

DEVICE = "cuda"


def get_backbone_indices(atom_list):
    return [i for i, atom in enumerate(atom_list) if atom in BACKBONE_ATOMS]


def filter_list_with_indices(list_to_filter, indices):
    return [x for i, x in enumerate(list_to_filter) if i in indices]


def filter_pdb_dict(pdb_dict):
    # filter important data for backbone atoms
    backbone_atom_indices = get_backbone_indices(pdb_dict["atom_names"])
    new_atoms = filter_list_with_indices(pdb_dict["atom_names"], backbone_atom_indices)
    new_coords = filter_list_with_indices(pdb_dict[""], backbone_atom_indices)
    new_res_atom_start = [i * 4 for i in range(len(pdb_dict["seq"]))]
    new_res_atom_end = [i + 4 for i in new_res_atom_start]
    # create ne dict
    new_dict = {
        "seq": pdb_dict["seq"],
        "res_types": pdb_dict["res_types"],
        "coords_groundtruth": new_coords,
        "atom_names": new_atoms,
        "res_atom_start": new_res_atom_start,
        "res_atom_end": new_res_atom_end
    }
    return new_dict


def get_bio2token(filtered_pdb_dicts,bio2token_model):
    # create tmp pdbs with only backbone
    batch = batch_pdb_dicts(filtered_pdb_dicts,DEVICE)
    batch = bio2token_model.encoder(batch)
    tokens = []
    encodings=[]
    lengths = [len(pdb_dict["seq"]) for pdb_dict in filtered_pdb_dicts]
    for i, length in enumerate(lengths):
        tokens.append(batch["indices"][i, :length * 4])
        encodings.append(batch["encoding"])
    return tokens,encodings


def get_foldtoken(pdb_paths):
    foldtokens = []
    for pdb_path in pdb_paths:
        foldtokens.append(foldtoken_model.encode_pdb(pdb_path))
    return foldtokens


def process_batch(pdb_dicts,pdb_paths, plm, bio2token, foldtoken):
    seqs = [pdb_dict["seq"] for pdb_dict in pdb_dicts]
    embeddings = plm.encode_list_of_seqs(seqs, TARGET_BATCH_SIZE)
    bio2token = get_bio2token(pdb_dicts)
    fold_token = get_foldtoken(pdb_paths)
    return embeddings, bio2token, fold_token


def main():
    # init models
    plm = ProtT5(device=DEVICE).to(DEVICE)
    _, _, encoder = load_bio2_token_decoder_and_quantizer()
    encoder.to(DEVICE)
    foldtoken = FoldToken(device=DEVICE).to(DEVICE)
    # input_dir = "/mnt/data/large/zip_file/final_data_PDB/val/"
    input_dir = "tokenizer_benchmark/casps/casp14"
    pdb_dicts_batch = []
    pdb_paths_dicts = []
    current_batch_size = 0
    for pdb in os.listdir(input_dir):
        pdb_path = os.path.join(input_dir, pdb)
        pdb_dict = pdb_2_dict(pdb_path)
        if len(pdb_dict["seq"]) >= MAX_LENGTH:
            continue
        pdb_dicts_batch.append(pdb_dict)
        pdb_paths_dicts.append(pdb_path)
        current_batch_size += 1
        if current_batch_size == TARGET_BATCH_SIZE:
            sequences, structure, embeddings, bio2token, fold_tokens = process_batch(pdb_dicts_batch,pdb_paths_dicts, plm, bio2token, foldtoken)


if __name__ == '__main__':
    main()
