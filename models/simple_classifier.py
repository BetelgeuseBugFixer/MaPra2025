import argparse
import json
import os
import random

import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt  # for plotting

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
                continue

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
        self.conv1 = nn.Conv1d(in_channels=d_emb,
                               out_channels=hidden,
                               kernel_size=3,
                               padding=1)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(in_channels=hidden,
                               out_channels=vocab_size,
                               kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        description="Train per-residue token classifier (CNN) with early stopping and plots"
    )
    parser.add_argument("--emb_dir",      required=True,
                        help="Directory containing per-protein .h5 files")
    parser.add_argument("--tok_jsonl",    required=True,
                        help="JSONL with per-protein {'<ID>': {..., 'vqid': [...]}}")
    parser.add_argument("--codebook_size", type=int, default=1024,
                        help="Size of VQ vocabulary")
    parser.add_argument("--d_emb",        type=int, default=1024)
    parser.add_argument("--hidden",       type=int, default=256)
    parser.add_argument("--dropout",      type=float, default=0.3)
    parser.add_argument("--batch",        type=int, default=1,
                        help="Batch size (1 → no padding)")
    parser.add_argument("--epochs",       type=int, default=10)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patience",     type=int, default=3,
                        help="Early stopping patience (in epochs)")
    return parser.parse_args()


def load_common_ids(emb_dir: str, tok_jsonl_path: str) -> list:
    emb_ids = {os.path.splitext(f)[0] for f in os.listdir(emb_dir) if f.endswith(".h5")}
    tok_ids = set()
    with open(tok_jsonl_path, "r") as fh:
        for line in fh:
            entry = json.loads(line)
            tok_ids.add(next(iter(entry.keys())))
    common = sorted(emb_ids & tok_ids)
    if not common:
        raise ValueError("No overlapping protein IDs between emb_dir and tok_jsonl.")
    return common


def split_ids(ids: list, seed: int = SPLIT_SEED):
    random.seed(seed)
    random.shuffle(ids)
    n_total = len(ids)
    n_train = int(0.70 * n_total)
    n_val   = max(int(0.15 * n_total), 1) if n_total > 2 else 0
    n_test  = n_total - n_train - n_val
    return ids[:n_train], ids[n_train:n_train+n_val], ids[n_train+n_val:]


def create_data_loaders(emb_dir, tok_jsonl, train_ids, val_ids, test_ids, batch_size):
    train_ds = ProteinPairJSONL(emb_dir, tok_jsonl, train_ids)
    val_ds   = ProteinPairJSONL(emb_dir, tok_jsonl, val_ids)
    test_ds  = ProteinPairJSONL(emb_dir, tok_jsonl, test_ids)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    )


def build_model(d_emb, hidden, vocab_size, dropout, lr, device):
    model     = ResidueTokenCNN(d_emb, hidden, vocab_size, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_LABEL)
    return model, optimizer, criterion


def _masked_accuracy(logits, tgt, mask):
    pred    = logits.argmax(dim=-1)
    correct = ((pred == tgt) & mask).sum().item()
    total   = mask.sum().item()
    return correct/total if total else 0.0


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = total_acc = total_samples = 0
    torch.set_grad_enabled(is_train)

    for emb, tok in tqdm(loader, desc="train" if is_train else "val", leave=False):
        emb, tok = emb.to(device), tok.to(device)
        mask = (tok != PAD_LABEL)
        logits = model(emb)
        loss   = criterion(logits.transpose(1,2), tok)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bsz = emb.size(0)
        total_loss    += loss.item() * bsz
        total_acc     += _masked_accuracy(logits, tok, mask) * bsz
        total_samples += bsz

    return total_loss/total_samples, total_acc/total_samples

# ------------------------------------------------------------
# Main workflow with early stopping & plots
# ------------------------------------------------------------

def main(args):
    vocab = args.codebook_size
    print(f"VQ-token vocabulary size V = {vocab}")

    common_ids = load_common_ids(args.emb_dir, args.tok_jsonl)
    print(f"Total proteins: {len(common_ids)}")

    train_ids, val_ids, test_ids = split_ids(common_ids, seed=SPLIT_SEED)
    print(f"Split sizes: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    train_loader, val_loader, test_loader = create_data_loaders(
        args.emb_dir, args.tok_jsonl, train_ids, val_ids, test_ids, args.batch
    )

    model, optimizer, criterion = build_model(
        args.d_emb, args.hidden, vocab, args.dropout, args.lr, args.device
    )

    # Tracking metrics
    train_losses, train_accs = [], []
    val_losses,   val_accs   = [], []
    best_val_loss = float('inf')
    patience_ctr  = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, args.device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device=args.device)

        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch:02d} | train {tr_loss:.4f}/{tr_acc:.4f} | val {val_loss:.4f}/{val_acc:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr = 0
            # Save best model
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/simple_cnn_classifier.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"Stopping early at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    # Final test evaluation
    test_loss, test_acc = run_epoch(model, test_loader, criterion, device=args.device)
    print(f"Test {test_loss:.4f}/{test_acc:.4f}")

    # Plotting training curves
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.savefig('models/model_out/simple/loss_curve.png')
    plt.show()

    plt.figure()
    plt.plot(epochs_range, train_accs, label='Train Acc')
    plt.plot(epochs_range, val_accs,   label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')
    plt.savefig('models/model_out/simple/accuracy_curve.png')
    plt.show()

    print("Saved best model → models/simple_cnn_classifier.pt")

if __name__ == "__main__":
    args = parse_args()
    main(args)
