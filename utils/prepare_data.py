#!/usr/bin/env python3
"""
--------------------------------
Speicher- und zeitoptimierte Version mit JSONL-Kompression für das Train-Set und MAX_PROT=100k.
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
# Configuration
# ----------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_PROT = 100_000  # Max. Proteine je Split (nur für train)

INPUT_DIR = Path("/mnt/data/large/zip_file/final_data_PDB")
SINGLETON_ID_PATH = Path(
    "/mnt/data/large/prostt5_IDs/afdb50best_foldseek_clu_singleton.ids"
)
OUTPUT_BASE = Path("/mnt/data/large/subset")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64

TMP_DIR = OUTPUT_BASE / "tmp"
os.makedirs(TMP_DIR,exist_ok=True)

# ----------------------------------------------------------------------------
# load models
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


# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------
def get_bio2token(pdb_paths, seq_lengths):
    # create tmp pdbs with only backbone
    new_pdbs = []
    for pdb_path in pdb_paths:
        pdb_filename = pdb_path.split("/")[-1]
        new_path = TMP_DIR / pdb_filename
        rewrite_pdb(pdb_path, new_path,allowed_atom_list=["N", "CA", "C", "O"])
        new_pdbs.append(new_path)

    batch = batch_pdbs_for_bio2token(new_pdbs, device=DEVICE)
    batch = bio2token_model.encoder(batch)
    tokens = []
    for i, length in enumerate(seq_lengths):
        tokens.append(batch["indices"][i,:length * 4])
    # delete all pdbs in folder in python
    temp_pdbs = glob.glob(os.path.join(TMP_DIR,"*.pdb"))
    for pdb_file in temp_pdbs:
        os.remove(pdb_file)
    return tokens


def get_foldtoken(pdb_paths):
    foldtokens=[]
    for pdb_path in pdb_paths:
        foldtokens.append(foldtoken_model.encode_pdb(pdb_path))
    return foldtokens


def process_batch(pdb_paths, seqs):
    seq_lengths = [len(seq) for seq in seqs]
    embeddings = plm.encode_list_of_seqs(seqs, BATCH_SIZE)
    bio2token = get_bio2token(pdb_paths, seq_lengths)
    fold_token = get_foldtoken(pdb_paths)
    return embeddings, bio2token, fold_token


def get_pdb_structure_and_seq(pdb_path: str):
    pdb_dict = pdb_2_dict(pdb_path)
    return pdb_dict["coords_groundtruth"], pdb_dict["seq"]


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        # 1) Gradienten abschneiden, 2) auf CPU ziehen, 3) zu NumPy konvertieren
        return x.detach().cpu().numpy()
    else:
        # z.B. NumPy-Array oder String bleibt unverändert
        return x

# ----------------------------------------------------------------------------
# Iterators
# ----------------------------------------------------------------------------

def iterate_train_pdbs(input_dir, singleton_ids, max_prot):
    """
    Yields (pid, tmp_pdb_path, cleanup_fn) for each .pdb in the rostlab_subset.tar.
    """
    processed = 0
    tar_path = input_dir / "train" / "rostlab_subset.tar"
    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            if processed >= max_prot:
                break
            if not member.name.endswith(".pdb"):
                continue

            pid = member.name.split("-")[1]
            if pid in singleton_ids:
                continue

            fobj = tar.extractfile(member)
            if fobj is None:
                continue

            lines = fobj.read().decode().splitlines()
            # write to temp file
            tmp = tempfile.NamedTemporaryFile("w+", suffix=".pdb", delete=False)
            tmp.write("\n".join(lines))
            tmp.flush()
            tmp_path = tmp.name
            tmp.close()

            # cleanup for tmp files
            def _cleanup(path=tmp_path):
                os.remove(path)

            yield pid, tmp_path, _cleanup
            processed += 1

def iterate_split_pdbs(input_dir, split, singleton_ids):
    """
    Yields (pid, pdb_path, no_op_cleanup) for each .pdb on disk in val/test.
    """
    pdb_dir = input_dir / split / f"{split}_pdb"
    for pdb_file in pdb_dir.glob("*.pdb"):
        stem = pdb_file.stem
        pid = stem.split("-", 1)[1] if "-" in stem else None
        if not pid or pid in singleton_ids:
            continue

        # no cleanup needed for non-tmp files
        yield pid, str(pdb_file), lambda: None


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------

@torch.inference_mode()
def process_split(split: str):
    print("=" * 60, flush=True)
    print(f"[{split.upper()}] begin", flush=True)

    out_dir = OUTPUT_BASE / split
    out_dir.mkdir(parents=True, exist_ok=True)

    # open HDF5 files
    emb_f    = h5py.File(out_dir / "embeddings.h5", mode="w")
    seq_f    = h5py.File(out_dir / "sequences.h5",  mode="w")
    struct_f = h5py.File(out_dir / "structures.h5", mode="w")
    bio2t_f  = h5py.File(out_dir / "bio2tokens.h5", mode="w")
    foldt_f  = h5py.File(out_dir / "foldtokens.h5", mode="w")

    processed = skipped = 0
    pid_batch, seq_batch, struct_batch, pdb_paths = [], [], [], []
    cleanup_fns = []   # Cleanup-Funktionen

    # pick the right iterator
    if split == "train":
        iterator = iterate_train_pdbs(INPUT_DIR, singleton_ids, MAX_PROT)
    else:
        iterator = iterate_split_pdbs(INPUT_DIR, split, singleton_ids)

    for pid, pdb_path, cleanup in iterator:
        try:
            structure, sequence = get_pdb_structure_and_seq(pdb_path)
        except Exception as e:
            print(f"[WARN] {pid} -> {e}", flush=True)
            cleanup()      # nur hier löschen, wenn wir gar nicht weitermachen
            continue


        if not sequence or len(sequence) >= 800:
            skipped += 1
            cleanup()      # hier löschen, wenn wir skippen
            continue

        # sammeln statt sofort löschen
        pid_batch.append(pid)
        seq_batch.append(sequence)
        struct_batch.append(structure)
        pdb_paths.append(pdb_path)
        cleanup_fns.append(cleanup)
        processed += 1

        if len(pid_batch) >= BATCH_SIZE:
            embeddings, bio2token, foldtoken = process_batch(pdb_paths, seq_batch)
            for pid_i, emb_i, seq_i, struct_i, b2t_i, ft_i in zip(
                pid_batch, embeddings, seq_batch, struct_batch, bio2token, foldtoken
            ):
                emb_f   .create_dataset(pid_i, data=to_numpy(emb_i))
                seq_f   .create_dataset(pid_i, data=to_numpy(seq_i))
                struct_f.create_dataset(pid_i, data=to_numpy(struct_i))
                bio2t_f .create_dataset(pid_i, data=to_numpy(b2t_i))
                foldt_f .create_dataset(pid_i, data=to_numpy(ft_i))

            #  erst jetzt löschen wir alle PDBs
            for fn in cleanup_fns:
                fn()
            cleanup_fns.clear()

            pid_batch.clear()
            seq_batch.clear()
            struct_batch.clear()
            pdb_paths.clear()

    # flush any leftovers
    if pid_batch:
        embeddings, bio2token, foldtoken = process_batch(pdb_paths, seq_batch)
        for pid_i, emb_i, seq_i, struct_i, b2t_i, ft_i in zip(
            pid_batch, embeddings, seq_batch, struct_batch, bio2token, foldtoken
        ):
            emb_f   .create_dataset(pid_i, data=to_numpy(emb_i))
            seq_f   .create_dataset(pid_i, data=to_numpy(seq_i))
            struct_f.create_dataset(pid_i, data=to_numpy(struct_i))
            bio2t_f .create_dataset(pid_i, data=to_numpy(b2t_i))
            foldt_f .create_dataset(pid_i, data=to_numpy(ft_i))

        # hier cleanup
        for fn in cleanup_fns:
            fn()

    # close files
    for f in (emb_f, seq_f, struct_f, bio2t_f, foldt_f):
        f.close()

    print(f"[{split.upper()}] done — processed {processed}, skipped {skipped}", flush=True)



if __name__ == '__main__':
    for s in ['train']: #only train for debug + because we already processed val and test
        process_split(s)
