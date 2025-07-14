import json
import os
import pickle
import gzip
import builtins
import h5py
import torch
from torch.utils.data import Dataset

from models.prot_t5.prot_t5 import ProtT5

PAD_LABEL = -100




class EmbTokSet(Dataset):
    def __init__(self, token_and_seq_file,batch_size,device):
        plm=ProtT5(device=device).to(device)
        sequences=[]
        self.vq_ids = []
        open_func = gzip.open if token_and_seq_file.endswith('.gz') else builtins.open
        with open_func(token_and_seq_file, 'rt') as f:
            for line in f.readlines():
                values = json.loads(line)
                sequences.append(values['sequence'])
                self.vq_ids.append(torch.tensor(values['vq_ids'], dtype=torch.long))

        self.embeddings = plm.encode_list_of_seqs(sequences,batch_size)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.vq_ids[idx]

class EmbStrucTokSet(Dataset):
    def __init__(self, token_and_seq_file, structure_file,batch_size,device):
        plm=ProtT5(device=device).to(device)
        sequences = []
        self.vq_ids = []
        self.structures = pickle.load(open(structure_file, 'rb'))
        open_func = gzip.open if token_and_seq_file.endswith('.gz') else builtins.open
        with open_func(token_and_seq_file, 'rt') as f:
            for line in f.readlines():
                values = json.loads(line)
                sequences.append(values['sequence'])
                self.vq_ids.append(torch.tensor(values['vq_ids'], dtype=torch.long))

        self.embeddings = plm.encode_list_of_seqs(sequences, batch_size)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.vq_ids[idx], self.structures[idx]


class SeqTokSet(Dataset):
    def __init__(self, token_and_seq_file):
        self.sequences = []
        self.vq_ids = []
        open_func = gzip.open if token_and_seq_file.endswith('.gz') else builtins.open
        with open_func(token_and_seq_file, 'rt') as f:
            for line in f.readlines():
                values = json.loads(line)
                self.sequences.append(values['sequence'])
                self.vq_ids.append(torch.tensor(values['vq_ids'], dtype=torch.long))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.vq_ids[idx]


class SeqStrucTokSet(Dataset):
    def __init__(self, token_and_seq_file, structure_file):
        self.sequences = []
        self.vq_ids = []
        self.structures = pickle.load(open(structure_file, 'rb'))
        open_func = gzip.open if token_and_seq_file.endswith('.gz') else builtins.open
        with open_func(token_and_seq_file, 'rt') as f:
            for line in f.readlines():
                values = json.loads(line)
                self.sequences.append(values['sequence'])
                self.vq_ids.append(torch.tensor(values['vq_ids'], dtype=torch.long))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.vq_ids[idx], self.structures[idx]


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
