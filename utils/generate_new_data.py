import os
import time

import torch

from models.bio2token.data.utils.utils import pdb_2_dict
from models.bio2token.decoder import load_bio2token_decoder_and_quantizer, load_bio2token_encoder
from models.foldtoken_decoder.foldtoken import FoldToken
from models.model_utils import batch_pdb_dicts
from models.prot_t5.prot_t5 import ProtT5

BACKBONE_ATOMS = ["N", "CA", "C", "O"]

MAX_LENGTH = 800

TARGET_BATCH_SIZE = 128

DEVICE = "cuda"


def save_simple(all_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # save sequences as txt
    with open(os.path.join(output_dir, "sequences.txt"), "w") as f:
        f.write("\n".join(all_data["sequences"]))

    # save tensors
    for name in ["structures", "embeddings", "bio2tokens", "encodings", "fold_tokens"]:
        # Liste von Tensoren → Liste in Tensor speichern
        torch.save(all_data[name], os.path.join(output_dir, f"{name}.pt"))


def get_backbone_indices(atom_list):
    return [i for i, atom in enumerate(atom_list) if atom in BACKBONE_ATOMS]


def filter_list_with_indices(list_to_filter, indices):
    return [list_to_filter[i] for i in indices]


def filter_pdb_dict(pdb_dict):
    # filter important data for backbone atoms
    backbone_atom_indices = get_backbone_indices(pdb_dict["atom_names"])
    new_atoms = filter_list_with_indices(pdb_dict["atom_names"], backbone_atom_indices)
    new_coords = filter_list_with_indices(pdb_dict["coords_groundtruth"], backbone_atom_indices)
    new_res_atom_start = [i * 4 for i in range(len(pdb_dict["seq"]))]
    new_res_atom_end = [i + 4 for i in new_res_atom_start]
    # create new dict
    new_dict = {
        "seq": pdb_dict["seq"],
        "res_types": pdb_dict["res_types"],
        "coords_groundtruth": new_coords,
        "atom_names": new_atoms,
        "res_atom_start": new_res_atom_start,
        "res_atom_end": new_res_atom_end
    }
    return new_dict


def get_bio2token(filtered_pdb_dicts, bio2token_model):
    # create tmp pdbs with only backbone
    batch = batch_pdb_dicts(filtered_pdb_dicts, DEVICE)
    batch = bio2token_model.encoder(batch).detach()
    tokens = []
    encodings = []
    lengths = [len(pdb_dict["seq"]) for pdb_dict in filtered_pdb_dicts]
    for i, length in enumerate(lengths):
        tokens.append(batch["indices"][i, :length * 4].cpu())
        encodings.append(batch["encoding"][i, :length * 4].cpu)
    return tokens, encodings


def process_batch(pdb_dicts, pdb_paths, plm, bio2token_model, foldtoken_model):
    # here we filter the backbone atoms
    pdb_dicts = [filter_pdb_dict(pdb_dict) for pdb_dict in pdb_dicts]
    seqs = [pdb_dict["seq"] for pdb_dict in pdb_dicts]
    embeddings = plm.encode_list_of_seqs(seqs, TARGET_BATCH_SIZE)
    bio2tokens, encodings = get_bio2token(pdb_dicts, bio2token_model)
    fold_tokens = foldtoken_model.encode_lists_of_pdbs(pdb_paths, DEVICE).cpu()
    structure = [pdb_dict["coords_groundtruth"] for pdb_dict in pdb_dicts]
    return seqs, structure, embeddings, bio2tokens, encodings, fold_tokens

def get_pid_from_file_name(file_name):
    return file_name.split(".")[0].split("-")[-1]


def main():
    # prepare singltons to be skipped
    with open("/mnt/data/large/prostt5_IDs/afdb50best_foldseek_clu_singleton.ids") as f:
        singleton_ids = {line.strip().split("-")[1] for line in f if "-" in line}
    # init models
    plm = ProtT5(device=DEVICE).to(DEVICE)
    bio2token_model = load_bio2token_encoder()
    bio2token_model.to(DEVICE)
    foldtoken = FoldToken(device=DEVICE).to(DEVICE)

    # data path
    # input_dir = "tokenizer_benchmark/casps/casp14"
    input_dir = "/mnt/data/large/zip_file/final_data_PDB/val/"
    output_dir = "/mnt/data/large/subset2/val"
    # save our data in lists
    all_sequences, all_structures = [], []
    all_embeddings, all_bio2tokens = [], []
    all_encodings, all_fold_tokens = [], []

    # count basic statistics
    error_skipped = max_length_skipped = singletons_skipped = 0
    processed = 0
    # init loop
    pdb_dicts_batch = []
    pdb_paths_dicts = []
    current_batch_size = 0
    # start timer
    start=time.time()
    batch_start = time.time()
    for pdb in os.listdir(input_dir):
        if not pdb.endswith(".pdb"):
            continue
        if get_pid_from_file_name(pdb)  in singleton_ids:
            singletons_skipped += 1
            continue
        pdb_path = os.path.join(input_dir, pdb)
        try:
            pdb_dict = pdb_2_dict(pdb_path)
        except Exception as e:
            print(f"error with {pdb}: {str(e)}")
            error_skipped += 1
            continue
        if len(pdb_dict["seq"]) >= MAX_LENGTH:
            max_length_skipped += 1
            continue
        pdb_dicts_batch.append(pdb_dict)
        pdb_paths_dicts.append(pdb_path)
        current_batch_size += 1
        if current_batch_size == TARGET_BATCH_SIZE:
            sequences, structure, embeddings, bio2tokens, encodings, fold_tokens = process_batch(pdb_dicts_batch,
                                                                                                 pdb_paths_dicts,
                                                                                                 plm, bio2token_model,
                                                                                                 foldtoken)
            all_sequences.extend(sequences)
            all_structures.extend(structure)
            all_embeddings.extend(embeddings)
            all_bio2tokens.extend(bio2tokens)
            all_encodings.extend(encodings)
            all_fold_tokens.extend(fold_tokens)
            # reset everything
            pdb_dicts_batch = []
            pdb_paths_dicts = []
            processed += current_batch_size
            current_batch_size = 0
            # print time and start timer anew
            print(f"processed new batch! total processed:{processed} | time for batch: {time.time() - batch_start}")
            batch_start = time.time()

    if current_batch_size > 0:
        sequences, structure, embeddings, bio2tokens, encodings, fold_tokens = process_batch(pdb_dicts_batch,
                                                                                             pdb_paths_dicts,
                                                                                             plm, bio2token_model,
                                                                                             foldtoken)

        all_sequences.extend(sequences)
        all_structures.extend(structure)
        all_embeddings.extend(embeddings)
        all_bio2tokens.extend(bio2tokens)
        all_encodings.extend(encodings)
        all_fold_tokens.extend(fold_tokens)

    all_data = {
        "sequences": all_sequences,
        "structures": all_structures,
        "embeddings": all_embeddings,
        "bio2tokens": all_bio2tokens,
        "encodings": all_encodings,
        "fold_tokens": all_fold_tokens
    }

    save_simple(all_data, output_dir)
    print(f"data was saved in {output_dir}")
    print(f"error skipped: {error_skipped}, max length skipped {max_length_skipped}, singletons skipped {singletons_skipped}")
    print(f"processed: {processed}")
    print(f"process took {time.time() - start}")


if __name__ == '__main__':
    main()
