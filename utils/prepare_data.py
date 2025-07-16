#!/usr/bin/env python3
"""
prepare_data_memory_efficient_compressed.py
--------------------------------
Speicher- und zeitoptimierte Version mit JSONL-Kompression für das Train-Set und BATCH_LIMIT=50k.
"""

import contextlib
import gc
import glob
import json
import gzip
import os
import pickle
import tarfile
import tempfile
import time
from collections import defaultdict
from os import mkdir
from pathlib import Path

import h5py
import torch
from biotite.structure.filter import _filter_atom_names
from biotite.structure.io.pdb import PDBFile
from hydra_zen import load_from_yaml

from models.foldtoken_decoder.foldtoken import FoldToken
from models.bio2token.data.utils.utils import pdb_2_dict, uniform_dataframe, compute_masks
from models.bio2token.models.autoencoder import AutoencoderConfig, Autoencoder
from models.bio2token.utils.configs import pi_instantiate
from models.model_utils import batch_pdbs_for_bio2token
from models.prot_t5.prot_t5 import ProtT5
from tokenizer_benchmark.extract_ca_atoms import rewrite_pdb

# ----------------------------------------------------------------------------
# Konfiguration
# ----------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_LIMIT = 50_000  # Max. Proteine je Split (nur für train)
CHUNK_SIZE = 1_000  # Fortschrittsintervalle

INPUT_DIR = Path("/mnt/data/large/zip_file/final_data_PDB")
SINGLETON_ID_PATH = Path(
    "/mnt/data/large/prostt5_IDs/afdb50best_foldseek_clu_singleton.ids"
)
OUTPUT_BASE = Path("/mnt/data/large/subset")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# Einmaliger devnull-Handle
DEVNULL = open(os.devnull, 'w')

# Aminosäure-Mapping
three_to_one = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}


# ----------------------------------------------------------------------------
# Hilfsfunktionen
# ----------------------------------------------------------------------------
def get_pdb_structure_and_seq(pdb_path: str):
    pdb_dict = pdb_2_dict(pdb_path)
    return pdb_dict["coords_groundtruth"], pdb_dict["sequence"]


def get_seq_from_lines(lines):
    seq = []
    chain_id = None
    for line in lines:
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            current_chain = line[21]
            if chain_id is None:
                chain_id = current_chain
            elif current_chain != chain_id:
                return None
            one = three_to_one.get(line[17:20].strip())
            if one is None:
                return None
            seq.append(one)
        elif line.startswith("ENDMDL"):
            break
    return "".join(seq) if seq else None


def load_prot_from_pdb(pdb_file):
    pdb = PDBFile.read(pdb_file)
    arr = pdb.get_structure(model=1)
    return arr[_filter_atom_names(arr, ["N", "CA", "C", "O"])]


# ----------------------------------------------------------------------------
# Modelle laden
# ----------------------------------------------------------------------------
print("[INIT] FoldToken laden …", flush=True)
foldtoken_model = FoldToken(device=DEVICE)
foldtoken_model.eval()

print("[INIT] Bio2Token laden …", flush=True)
yaml_cfg = load_from_yaml("models/bio2token/files/model.yaml")["model"]
model_cfg = pi_instantiate(AutoencoderConfig, yaml_cfg)
bio2token_model = Autoencoder(model_cfg)
state = torch.load(
    "models/bio2token/files/epoch=0243-val_loss_epoch=0.71-best-checkpoint.ckpt",
    map_location="cpu"
)["state_dict"]
bio2token_model.load_state_dict({k.replace("model.", ""): v for k, v in state.items()})
bio2token_model.eval().to(DEVICE)
print("[INIT] Modelle bereit.", flush=True)
plm = ProtT5()

# Singleton-IDs laden
with open(SINGLETON_ID_PATH) as f:
    singleton_ids = {line.strip().split("-")[1] for line in f if "-" in line}
print(f"[INIT] {len(singleton_ids):,} Singleton-IDs geladen", flush=True)

BATCH_SIZE = 64

TMP_DIR = OUTPUT_BASE / "tmp"
os.makedirs(TMP_DIR,exist_ok=True)


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
def get_bio2token(pdb_paths, seq_lengths):
    # create tmp pdbs with only backbone
    new_pdbs = []
    for pdb_path in pdb_paths:
        pdb_filename = pdb_path.split("/")[-1]
        new_path = TMP_DIR / pdb_filename
        rewrite_pdb(pdb_path, new_path)
        new_pdbs.append(new_path)

    batch = batch_pdbs_for_bio2token(new_pdbs, device=DEVICE)
    batch = bio2token_model.encoder(batch)
    tokens = []
    for i, length in enumerate(seq_lengths):
        tokens.append(batch["indices"][i:length * 4])
    # delete all pdbs in folder in python
    temp_pdbs = glob.glob(TMP_DIR / "*.pdb")
    for pdb_file in temp_pdbs:
        os.remove(pdb_file)
    return tokens


def get_foldtoken(pdb_paths):
    foldtokens=[]
    for pdb_path in pdb_paths:
        foldtokens.append(foldtoken_model.encode(pdb_path))
    return foldtokens


def process_batch(pdb_paths, seqs):
    seq_lengths = [len(seq) for seq in seqs]
    embeddings = plm.encode_list_of_seqs(seqs, BATCH_SIZE)
    bio2token = get_bio2token(pdb_paths, seq_lengths)
    fold_token=get_foldtoken(pdb_paths)
    return embeddings, bio2token, fold_token


@torch.inference_mode()
def process_split(split: str):
    print("=" * 60, flush=True)
    print(f"[{split.upper()}] begin", flush=True)

    out_dir = OUTPUT_BASE / split
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_file = h5py.File("embeddings.h5")
    seq_file = h5py.File("sequences.h5")
    structure_file = h5py.File("structures.h5")
    bio2token_file = h5py.File("bio2tokens.h5")
    foldtoken_file = h5py.File("foldtokens.h5")

    processed = skipped = 0
    start = time.time()
    chunk_start = time.time()

    # def handle_protein(pid, seq, pdb_path):
    #     nonlocal processed, chunk_start
    #     vq = foldtoken_model.encode_pdb(pdb_path)
    #     with contextlib.redirect_stdout(DEVNULL):
    #         pdb_d = pdb_2_dict(pdb_path, None)
    #     st, unk, _, res_ids, tok, _ = uniform_dataframe(
    #         pdb_d['seq'], pdb_d['res_types'], pdb_d['coords_groundtruth'],
    #         pdb_d['atom_names'], pdb_d['res_atom_start'], pdb_d['res_atom_end']
    #     )
    #     cpu_batch = {
    #         'structure': torch.tensor(st, dtype=torch.float32),
    #         'unknown_structure': torch.tensor(unk, dtype=torch.bool),
    #         'residue_ids': torch.tensor(res_ids, dtype=torch.int64),
    #         'token_class': torch.tensor(tok, dtype=torch.int64),
    #     }
    #     cpu_batch = {k: v[~cpu_batch['unknown_structure']] for k, v in cpu_batch.items()}
    #     cpu_batch = compute_masks(cpu_batch, structure_track=True)
    #     batch = {k: v.unsqueeze(0).to(DEVICE) for k, v in cpu_batch.items()}
    #     enc = bio2token_model.encoder(batch)
    #     encoding = enc['encoding'].squeeze(0).cpu().tolist()
    #     indices = enc['indices'].squeeze(0).cpu().tolist()
    #     entry = {'id': pid, 'sequence': seq, 'vq_ids': vq.tolist(),
    #              'bio2tokens': indices, 'bio2token_encoding': encoding}
    #     struct = load_prot_from_pdb(pdb_path)
    #
    #     if is_train:
    #         json_f.write(json.dumps(entry) + "\n")
    #         pickle.dump(struct, pickle_f, protocol=pickle.HIGHEST_PROTOCOL)
    #     else:
    #         buffer_json.append(entry)
    #         buffer_structs.append(struct)
    #
    #     processed += 1
    #     if processed % CHUNK_SIZE == 0:
    #         dur = time.time() - chunk_start
    #         print(f"[{split}] {processed:,} in {dur:.1f}s", flush=True)
    #         chunk_start = time.time()
    #         gc.collect()
    #         if torch.cuda.is_available(): torch.cuda.empty_cache()
    #     del cpu_batch, batch, enc, struct, vq

    pid_batch = []
    seq_batch = []
    structure_batch = []
    pbd_path_batch = []
    for pdb_file in (INPUT_DIR / split / f"{split}_pdb").glob('*.pdb'):
        pid = pdb_file.stem.split('-')[1] if '-' in pdb_file.stem else None
        if not pid or pid in singleton_ids:
            skipped += bool(pid)
            continue
        try:
            pdb_path = INPUT_DIR / split / f"{pid}.pdb"
            sequence, structure = get_pdb_structure_and_seq(pdb_path)
            if len(sequence) < 800:
                pid_batch.append(pid)
                seq_batch.append(sequence)
                structure_batch.append(structure)
                pbd_path_batch.append(pdb_path)
            else:
                skipped += 1
                continue

            if len(pid_batch) == BATCH_SIZE:
                embeddings, foldtoken, bio2token = process_batch(pbd_path_batch, seq_batch)
                pid_batch = []
                seq_batch = []
                structure_batch = []
                pbd_path_batch = []

        except Exception as e:
            print(f"[WARN] {pdb_file.name} -> {e}", flush=True)




if __name__ == '__main__':
    for s in ['val', 'test', 'train']:
        process_split(s)
