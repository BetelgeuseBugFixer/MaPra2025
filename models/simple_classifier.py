import argparse
import json
import os
import random

import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ------------------------------------------------------------
#  Hyper-parameters & constants
# ------------------------------------------------------------

PAD_LABEL = -100         # ignored in the loss on padded positions
SPLIT_SEED = 42          # for reproducible splits

# ------------------------------------------------------------
#  Dataset
# ------------------------------------------------------------

class ProteinPairJSONL(Dataset):
    """
    Yield `(embedding, vqid_tokens)` pairs, skipping any protein where the
    embedding length ≠ token length (and printing those mismatches).
    """
    def __init__(self, emb_dir: str, tok_jsonl: str, ids: list):
        # Build a map: protein ID → path to its .h5 file
        self.emb_map = {}
        for fname in os.listdir(emb_dir):
            if not fname.endswith(".h5"):
                continue
            pid = os.path.splitext(fname)[0]
            self.emb_map[pid] = os.path.join(emb_dir, fname)

        # Load JSONL once into a dict: {protein_id: LongTensor(vqid_list)}
        self.vqid_map = {}
        with open(tok_jsonl, "r") as fh:
            for line in fh:
                entry = json.loads(line)
                pid, data = next(iter(entry.items()))
                self.vqid_map[pid] = torch.tensor(data["vqid"], dtype=torch.long)

        # Filter `ids` to only those that exist in both maps AND have matching lengths
        valid_ids = []
        for pid in ids:
            if pid not in self.emb_map or pid not in self.vqid_map:
                # If you really want to catch missing files/tokens, you could print here,
                # but your original code already raised a KeyError if anything was missing.
                continue

            # Fetch embedding once just to check its length
            emb_np = self._fetch_emb_from_file(self.emb_map[pid])  # (L_emb, d_emb)
            L_emb = emb_np.shape[0]
            L_tok = self.vqid_map[pid].shape[0]

            if L_emb != L_tok:
                print(f"Skipping {pid}: embedding length = {L_emb}, token length = {L_tok}")
                continue

            valid_ids.append(pid)

        if len(valid_ids) < len(ids):
            print(f"  → {len(ids)-len(valid_ids)} protein(s) skipped due to length mismatch.")

        if not valid_ids:
            raise ValueError("No valid proteins remain after filtering length mismatches.")

        self.ids = valid_ids

    @staticmethod
    def _fetch_emb_from_file(h5_path: str):
        """
        Given a path to a .h5 file containing a single dataset at root,
        return that dataset as a NumPy array.
        """
        with h5py.File(h5_path, "r", swmr=True) as f:
            ds_names = list(f.keys())
            if len(ds_names) != 1:
                raise ValueError(f"Expected exactly one dataset in {h5_path}, found: {ds_names}")
            return f[ds_names[0]][:]    # shape: (L, d_emb)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        emb_np = self._fetch_emb_from_file(self.emb_map[pid])  # shape: (L, d_emb)
        emb = torch.from_numpy(emb_np).float()                  # → FloatTensor (L, d_emb)
        tok = self.vqid_map[pid]                                # → LongTensor (L,)
        return emb, tok


# ------------------------------------------------------------
#  Model
# ------------------------------------------------------------

class ResidueTokenCNN(nn.Module):
    """1D‐CNN head replacing the MLP."""
    def __init__(self, d_emb: int, hidden: int, vocab_size: int, dropout: float = 0.3):
        super().__init__()
        # Conv1: in_channels=d_emb, out_channels=hidden, kernel_size=3, padding=1
        self.conv1 = nn.Conv1d(in_channels=d_emb,
                               out_channels=hidden,
                               kernel_size=3,
                               padding=1)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(dropout)
        # Conv2: in_channels=hidden, out_channels=vocab_size, kernel_size=1
        self.conv2 = nn.Conv1d(in_channels=hidden,
                               out_channels=vocab_size,
                               kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_emb)
        B, L, D = x.shape
        x = x.permute(0, 2, 1)       # → (B, d_emb, L)
        h = self.conv1(x)            # → (B, hidden, L)
        h = self.relu(h)
        h = self.drop(h)
        h = self.conv2(h)            # → (B, vocab_size, L)
        out = h.permute(0, 2, 1)     # → (B, L, vocab_size)
        return out


# ------------------------------------------------------------
#  Utility functions
# ------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train per-residue token classifier (CNN variant) without padding, using per-protein .h5 files"
    )
    parser.add_argument("--emb_dir",      required=True,
                        help="Directory containing per-protein .h5 files (filename stem = protein ID)")
    parser.add_argument("--tok_jsonl",    required=True,
                        help="JSONL with per-protein {'<ID>': {..., 'vqid': [...]}}")
    parser.add_argument("--codebook_size", type=int, default=1024,
                        help="Integer size of VQ vocabulary (e.g. 1024)")
    parser.add_argument("--d_emb",        type=int,   default=1024)
    parser.add_argument("--hidden",       type=int,   default=256)
    parser.add_argument("--dropout",      type=float, default=0.3)
    parser.add_argument("--batch",        type=int,   default=1,
                        help="Batch size (set to 1 if you want truly no padding)")
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_common_ids(emb_dir: str, tok_jsonl_path: str) -> list:
    """
    Return the sorted list of protein IDs common to both:
      - The set of .h5 filenames (without .h5 extension) in `emb_dir`, and
      - The set of IDs in the JSONL at `tok_jsonl_path`.
    """
    emb_ids = set()
    for fname in os.listdir(emb_dir):
        if not fname.endswith(".h5"):
            continue
        pid = os.path.splitext(fname)[0]
        emb_ids.add(pid)

    tok_ids = set()
    with open(tok_jsonl_path, "r") as fh:
        for line in fh:
            entry = json.loads(line)
            pid = next(iter(entry.keys()))
            tok_ids.add(pid)

    common = sorted(emb_ids & tok_ids)
    if not common:
        raise ValueError("No overlapping protein IDs between emb_dir and tok_jsonl.")
    return common


def split_ids(ids: list, seed: int = SPLIT_SEED):
    """
    Shuffle and split IDs into 70% train, 15% val, 15% test.
    Returns (train_ids, val_ids, test_ids).
    """
    random.seed(seed)
    random.shuffle(ids)
    n_total = len(ids)
    n_train = int(0.70 * n_total)
    n_val   = int(0.15 * n_total)
    n_val   = max(n_val, 1) if n_total > 2 else n_val
    n_test  = n_total - n_train - n_val
    train_ids = ids[:n_train]
    val_ids   = ids[n_train:n_train + n_val]
    test_ids  = ids[n_train + n_val:]
    return train_ids, val_ids, test_ids


def create_data_loaders(emb_dir: str, tok_jsonl: str,
                        train_ids: list, val_ids: list, test_ids: list,
                        batch_size: int):
    """
    Create DataLoader objects for train/val/test splits, each dataset opening
    per-protein .h5 files on the fly (batch_size=1 → no explicit padding).
    """
    train_ds = ProteinPairJSONL(emb_dir, tok_jsonl, train_ids)
    val_ds   = ProteinPairJSONL(emb_dir, tok_jsonl, val_ids)
    test_ds  = ProteinPairJSONL(emb_dir, tok_jsonl, test_ids)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def build_model(d_emb: int, hidden: int, vocab_size: int, dropout: float, lr: float, device: str):
    """
    Instantiate the ResidueTokenCNN and move it to the specified device.
    Returns (model, optimizer, criterion).
    """
    model = ResidueTokenCNN(d_emb, hidden, vocab_size, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_LABEL)
    return model, optimizer, criterion


def _masked_accuracy(logits: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> float:
    """
    Compute accuracy ignoring PAD_LABEL positions (but if batch_size=1 and no padding,
    `mask = (tgt != PAD_LABEL)` will be all True).
    """
    pred = logits.argmax(dim=-1)      # (B, L)
    correct = ((pred == tgt) & mask).sum().item()
    total = mask.sum().item()
    return correct / total if total else 0.0


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    """
    Run a single training or validation epoch.
    If `optimizer` is not None, performs training; otherwise, runs evaluation.
    Returns averaged (loss, accuracy) over all batches.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = total_acc = total_samples = 0
    torch.set_grad_enabled(is_train)

    for emb, tok in tqdm(loader, desc="train" if is_train else "val", leave=False):
        # emb: shape (B, L, d_emb), tok: shape (B, L); often B=1 when no padding
        emb, tok = emb.to(device), tok.to(device)
        mask = (tok != PAD_LABEL)           # Will be all True if you never padded
        logits = model(emb)                 # (B, L, V)
        loss = criterion(logits.transpose(1, 2), tok)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bsz = emb.size(0)
        total_loss += loss.item() * bsz
        total_acc  += _masked_accuracy(logits, tok, mask) * bsz
        total_samples += bsz

    avg_loss = total_loss / total_samples
    avg_acc  = total_acc / total_samples
    return avg_loss, avg_acc


# ------------------------------------------------------------
#  Main workflow
# ------------------------------------------------------------

def main(args):
    # 1) Vocabulary size is passed explicitly
    vocab = args.codebook_size
    print(f"VQ‐token vocabulary size V = {vocab}")

    # 2) Find common protein IDs
    common_ids = load_common_ids(args.emb_dir, args.tok_jsonl)
    print(f"Total proteins: {len(common_ids)}")

    # 3) Split IDs
    train_ids, val_ids, test_ids = split_ids(common_ids, seed=SPLIT_SEED)
    print(f"Split sizes: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    # 4) Create DataLoaders (often batch_size=1 → no padding needed)
    train_loader, val_loader, test_loader = create_data_loaders(
        args.emb_dir, args.tok_jsonl,
        train_ids, val_ids, test_ids,
        batch_size=args.batch
    )

    # 5) Build CNN model
    model, optimizer, criterion = build_model(
        d_emb=args.d_emb,
        hidden=args.hidden,
        vocab_size=vocab,
        dropout=args.dropout,
        lr=args.lr,
        device=args.device
    )

    # 6) Training loop
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device=args.device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device=args.device)
        print(f"Epoch {epoch:02d} | train {tr_loss:.4f}/{tr_acc:.4f} | val {val_loss:.4f}/{val_acc:.4f}")

    # 7) Test evaluation
    test_loss, test_acc = run_epoch(model, test_loader, criterion, device=args.device)
    print(f"Test {test_loss:.4f}/{test_acc:.4f}")

    # 8) Save the trained model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/simple_cnn_classifier.pt")
    print("Saved → models/simple_cnn_classifier.pt")


if __name__ == "__main__":
    args = parse_args()
    main(args)
