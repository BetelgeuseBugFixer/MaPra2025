import shutil
import time
from collections import defaultdict

import torch

from models.bio2token.data.utils.utils import pdb_2_dict
from models.bio2token.decoder import load_bio2token_encoder
from models.foldtoken_decoder.foldtoken import FoldToken
from models.model_utils import batch_pdb_dicts
from models.prot_t5.prot_t5 import ProtT5
import os
import tarfile
import psutil


BACKBONE_ATOMS = ["N", "CA", "C", "O"]

MAX_LENGTH = 800

TARGET_BATCH_SIZE = 128

DEVICE = "cuda"

NUM_OF_TARGET_PROTEINS=100_000
CHUNK_SIZE = 50_000
TMP_DIR = "/mnt/data/large/new_tmp"


def save_data(all_data, output_dir):
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
    with torch.no_grad():
        batch = bio2token_model(batch)
    tokens = []
    encodings = []
    lengths = [len(pdb_dict["seq"]) for pdb_dict in filtered_pdb_dicts]
    for i, length in enumerate(lengths):
        tokens.append(batch["indices"][i, :length * 4].detach().cpu())
        encodings.append(batch["encoding"][i, :length * 4].detach().cpu())
    return tokens, encodings


def reset_batch():
    return [], [], 0  # pdb_dicts, pdb_paths, current_batch_size


def append_batch_data(seqs, structure, embeddings, bio2tokens, encodings, fold_tokens, all_data):
    all_data["sequences"].extend(seqs)
    all_data["structures"].extend(structure)
    all_data["embeddings"].extend(embeddings)
    all_data["bio2tokens"].extend(bio2tokens)
    all_data["encodings"].extend(encodings)
    all_data["fold_tokens"].extend(fold_tokens)


def process_batch(pdb_dicts, pdb_paths, plm, bio2token_model, foldtoken_model, all_data):
    # here we filter the backbone atoms
    pdb_dicts = [filter_pdb_dict(pdb_dict) for pdb_dict in pdb_dicts]
    seqs = [pdb_dict["seq"] for pdb_dict in pdb_dicts]
    embeddings = plm.encode_list_of_seqs(seqs, TARGET_BATCH_SIZE)
    bio2tokens, encodings = get_bio2token(pdb_dicts, bio2token_model)
    fold_tokens = foldtoken_model.encode_lists_of_pdbs(pdb_paths, DEVICE)
    # load to cpu
    fold_tokens = [token_vector.cpu() for token_vector in fold_tokens]
    structure = [pdb_dict["coords_groundtruth"] for pdb_dict in pdb_dicts]
    # update all data
    append_batch_data(seqs, structure, embeddings, bio2tokens, encodings, fold_tokens, all_data)


def get_pid_from_file_name(file_name):
    return file_name.split("-")[1]


def init_models():
    plm = ProtT5(device=DEVICE).to(DEVICE).eval()
    bio2token_model = load_bio2token_encoder().to(DEVICE).eval()
    foldtoken = FoldToken(device=DEVICE).to(DEVICE).eval()
    return plm, bio2token_model, foldtoken


def get_pdb_dict(pdb_path, pdb_file_name, singleton_ids, skipped_statistics):
    if not pdb_file_name.endswith(".pdb"):
        skipped_statistics["skipped - non pdb"] += 1
        return None
    if get_pid_from_file_name(pdb_file_name) in singleton_ids:
        skipped_statistics["skipped - singleton"] += 1
        return None
    try:
        pdb_dict = pdb_2_dict(pdb_path)
    except Exception as e:
        print(f"[ERROR] {pdb_path}: {str(e)}")
        skipped_statistics["skipped - error"] += 1
        return None
    if len(pdb_dict["seq"]) >= MAX_LENGTH:
        skipped_statistics["skipped - too long"] += 1
        return None
    return pdb_dict  # valid dict

def empty_dir(tmp_dir):
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

def handle_full_dir(full_dir, singleton_ids, statistics, plm, bio2token_model, foldtoken, all_data):
    pdb_dicts_batch = []
    pdb_paths_batch = []
    current_batch_size = 0
    batch_start = time.time()
    for pdb in os.listdir(full_dir):
        pdb_path = os.path.join(full_dir, pdb)
        pdb_dict = get_pdb_dict(pdb_path, pdb, singleton_ids, statistics)
        if pdb_dict is None:
            continue

        pdb_dicts_batch.append(pdb_dict)
        pdb_paths_batch.append(pdb_path)
        current_batch_size += 1
        if current_batch_size == TARGET_BATCH_SIZE:
            process_batch(pdb_dicts_batch, pdb_paths_batch, plm, bio2token_model, foldtoken, all_data)
            statistics["processed"] += current_batch_size
            # reset everything
            pdb_dicts_batch, pdb_paths_batch, current_batch_size = reset_batch()
            # print time and start timer anew
            print(
                f"processed new batch! total processed:{statistics["processed"]} | time for batch: {time.time() - batch_start:.2f}s")
            batch_start = time.time()

    if current_batch_size > 0:
        process_batch(pdb_dicts_batch, pdb_paths_batch, plm, bio2token_model, foldtoken, all_data)
        statistics["processed"] += current_batch_size
        print(
            f"processed new batch! total processed:{statistics["processed"]} | time for batch: {time.time() - batch_start}")

def write_to_temp(tar_path, tmp_dir, pdb_chunk):
    writing_start=time.time()
    os.makedirs(tmp_dir, exist_ok=True)
    with tarfile.open(tar_path, "r") as tar:
        for member in pdb_chunk:
            tar.extract(member, tmp_dir)

    print(f"Extracted {len(pdb_chunk)} PDB files to {tmp_dir} in {time.time() - writing_start:.2f}s")


def extract_chunks_from_tar(tar_path, tmp_dir, chunk_size, file_filter=lambda name: name.endswith('.pdb')):
    os.makedirs(tmp_dir, exist_ok=True)
    chunk = []
    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            if member.isfile() and file_filter(member.name):
                chunk.append(member)
                if len(chunk) == chunk_size:
                    tar.extractall(path=tmp_dir, members=chunk)
                    yield chunk
                    chunk = []
        if chunk:
            tar.extractall(path=tmp_dir, members=chunk)
            yield chunk



def log_memory_usage(prefix=""):
    process = psutil.Process()
    mem_bytes = process.memory_info().rss
    mem_gb = mem_bytes / (1024 ** 3)
    print(f"{prefix} Memory Usage: {mem_gb:.3f} GB")



def main(input_dir="/mnt/data/large/zip_file/final_data_PDB/val/val_pdb", output_dir="/mnt/data/large/subset2/val/"):
    # start timer
    start = time.time()
    # prepare singletons to be skipped
    with open("/mnt/data/large/prostt5_IDs/afdb50best_foldseek_clu_singleton.ids") as f:
        singleton_ids = {line.strip().split("-")[1] for line in f if "-" in line}
    log_memory_usage("loading singletons")
    # init models
    plm, bio2token_model, foldtoken = init_models()
    # data path
    # input_dir = "tokenizer_benchmark/casps/casp14"

    # save our data in lists
    all_data = defaultdict(list)

    # count basic statistics
    statistics = defaultdict(int)
    if input_dir.endswith(".tar"):
        for _ in extract_chunks_from_tar(input_dir, TMP_DIR, CHUNK_SIZE):
            handle_full_dir(TMP_DIR, singleton_ids, statistics, plm, bio2token_model, foldtoken, all_data)
            empty_dir(TMP_DIR)
            log_memory_usage("completed next chunk")
    else:
        handle_full_dir(input_dir, singleton_ids, statistics, plm, bio2token_model, foldtoken, all_data)

    log_memory_usage("processed data")
    save_data(all_data, output_dir)
    print(f"data was saved in {output_dir}")
    print("statistics:")
    for key, value in statistics.items():
        print(f"{key}: {value}")
    print(f"process took {time.time() - start}")


if __name__ == '__main__':
    main("/mnt/data/large/zip_file/final_data_PDB/train/rostlab_subset.tar","/mnt/data/large/subset2/val/")
