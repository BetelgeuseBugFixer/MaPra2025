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
        tok = self.vqid_map[pid]                               # (L,)
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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--emb_file", help="HDF5 file with one dataset per protein ID")
    group.add_argument("--emb_dir",  help="Directory with per-protein .h5 files")

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



def load_common_ids(emb_source: str, tok_jsonl_path: str, use_single_file: bool) -> list:
    if use_single_file:
        with h5py.File(emb_source, "r") as f:
            emb_ids = set(f.keys())
    else:
        emb_ids = {os.path.splitext(f)[0] for f in os.listdir(emb_source) if f.endswith(".h5")}

    tok_ids = set()
    with open(tok_jsonl_path, "r") as fh:
        for line in fh:
            tok_ids.add(next(iter(json.loads(line).keys())))

    common = sorted(emb_ids & tok_ids)
    if not common:
        raise ValueError("No overlapping protein IDs.")
    return common




def split_ids(ids: list, seed: int = SPLIT_SEED):
    random.seed(seed)
    random.shuffle(ids)
    n_total = len(ids)
    n_train = int(0.70 * n_total)
    n_val   = max(int(0.15 * n_total), 1) if n_total > 2 else 0
    n_test  = n_total - n_train - n_val
    return ids[:n_train], ids[n_train:n_train+n_val], ids[n_train+n_val:]


def create_data_loaders(emb_source, tok_jsonl, train_ids, val_ids, test_ids, batch_size, use_single_file):
    DSClass = ProteinPairJSONL if use_single_file else ProteinPairJSONL_FromDir
    train_ds = DSClass(emb_source, tok_jsonl, train_ids)
    val_ds   = DSClass(emb_source, tok_jsonl, val_ids)
    test_ds  = DSClass(emb_source, tok_jsonl, test_ids)
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
    use_file = args.emb_file is not None
    emb_source = args.emb_file if use_file else args.emb_dir

    common_ids = load_common_ids(emb_source, args.tok_jsonl, use_file)
    print(f"Total proteins: {len(common_ids)}")

    train_ids, val_ids, test_ids = split_ids(common_ids, seed=SPLIT_SEED)

    train_loader, val_loader, test_loader = create_data_loaders(
        emb_source, args.tok_jsonl, train_ids, val_ids, test_ids, args.batch, use_file
    )

    model, optimizer, criterion = build_model(
        args.d_emb, args.hidden, args.codebook_size, args.dropout, args.lr, args.device
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
