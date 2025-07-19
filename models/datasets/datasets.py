import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

PAD_LABEL = -100


def load_pt(path):
    data = torch.load(path, weights_only=False)
    if "structures.pt" in path:
        return [torch.as_tensor(np.array(structure)) for structure in data]
    return data



def load_model_in(file_dir, precomputed_embeddings):
    if precomputed_embeddings:
        return load_pt(os.path.join(file_dir, "embeddings.pt"))
    else:
        return open(os.path.join(file_dir, "sequences.txt")).read().splitlines()


def load_tokens(file_dir, token_type):
    match token_type:
        case "bio2token":
            return load_pt(os.path.join(file_dir, "bio2tokens.pt"))
        case "foldtoken":
            return load_pt(os.path.join(file_dir, "foldtokens.pt"))
        case "encoding":
            return load_pt(os.path.join(file_dir, "encodings.pt"))
        case _:
            raise RuntimeError(f"{token_type} is not supported, please use bio2token or foldtoken")


class StructureSet(Dataset):
    def __init__(self, file_dir, precomputed_embeddings=False):
        self.model_in = load_model_in(file_dir, precomputed_embeddings)
        self.structures = load_pt(os.path.join(file_dir, "structures.pt"))

    def __len__(self):
        return len(self.model_in)

    def __getitem__(self, idx):
        return self.model_in[idx],self.structures[idx]


class TokenSet(Dataset):
    def __init__(self, file_dir, token_type, precomputed_embeddings=False):
        self.model_in = load_model_in(file_dir, precomputed_embeddings)
        self.tokens = load_tokens(file_dir, token_type)

    def __len__(self):
        return len(self.model_in)

    def __getitem__(self, idx):
        return self.model_in[idx], self.tokens[idx]


class StructureAndTokenSet(Dataset):
    def __init__(self, file_dir, token_type, precomputed_embeddings=False):
        self.model_in = load_model_in(file_dir, precomputed_embeddings)
        self.tokens = load_tokens(file_dir, token_type)
        self.structures = load_pt(os.path.join(file_dir, "structures.pt"))

    def __len__(self):
        return len(self.model_in)

    def __getitem__(self, idx):
        return self.model_in[idx], self.tokens[idx], self.structures[idx]


class ProteinPairJSONL(Dataset):
    """
    Yield `(embedding, vqid_tokens)` pairs from a single HDF5 file containing
    one dataset per protein. Skips proteins with mismatched sequence lengths.
    """

    def __init__(self, emb_h5_path: str, tok_jsonl: str, ids: list):
        # Open the shared HDF5 file
        self.emb_file = h5py.File(emb_h5_path, "r")

        # Load VQ token sequences from JSONL
        self.vqid_map = {}
        for line in open(tok_jsonl, "r"):
            entry = json.loads(line)
            pid, data = next(iter(entry.items()))
            self.vqid_map[pid] = torch.tensor(data["vqid"], dtype=torch.long)

        # Filter valid IDs
        valid_ids = []
        for pid in ids:
            if pid not in self.emb_file or pid not in self.vqid_map:
                continue

            emb = self.emb_file[pid][:]
            tok = self.vqid_map[pid]
            if emb.shape[0] != tok.shape[0]:
                print(f"Skipping {pid}: embedding length = {emb.shape[0]}, token length = {tok.shape[0]}")
                continue

            valid_ids.append(pid)

        if not valid_ids:
            raise ValueError("No valid proteins remain after filtering.")

        self.ids = valid_ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        emb = torch.from_numpy(self.emb_file[pid][:]).float()  # (L, d_emb)
        tok = self.vqid_map[pid]  # (L,)
        return emb, tok


### from directory for casp14 dataset
class ProteinPairJSONL_FromDir(ProteinPairJSONL):
    def __init__(self, emb_dir: str, tok_jsonl: str, ids: list):
        self.emb_map = {}
        for fname in os.listdir(emb_dir):
            if fname.endswith(".h5"):
                pid = os.path.splitext(fname)[0]
                self.emb_map[pid] = os.path.join(emb_dir, fname)

        self.vqid_map = {}
        for line in open(tok_jsonl, "r"):
            entry = json.loads(line)
            pid, data = next(iter(entry.items()))
            self.vqid_map[pid] = torch.tensor(data["vqid"], dtype=torch.long)

        valid_ids = []
        for pid in ids:
            if pid not in self.emb_map or pid not in self.vqid_map:
                continue

            with h5py.File(self.emb_map[pid], "r") as f:
                ds = f[list(f.keys())[0]][:]
            if ds.shape[0] != self.vqid_map[pid].shape[0]:
                print(f"Skipping {pid}: length mismatch")
                continue
            valid_ids.append(pid)

        if not valid_ids:
            raise ValueError("No valid proteins remain after filtering.")
        self.ids = valid_ids

    def __getitem__(self, idx):
        pid = self.ids[idx]
        with h5py.File(self.emb_map[pid], "r") as f:
            emb = torch.from_numpy(f[list(f.keys())[0]][:]).float()
        tok = self.vqid_map[pid]
        return emb, tok
