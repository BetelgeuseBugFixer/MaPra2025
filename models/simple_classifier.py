import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

"""token_classifier.py – two‑file variant
========================================
Embeddings and tokens live in **separate** HDF5 files but share the **same
protein IDs** (group names).

```
embeddings.h5
├── T1027            ← float32 dataset (L × d_emb)
├── T1029
└── …

tokens.h5
├── T1027            ← int16 dataset   (L,)
├── T1029
└── …
```

Train/val splits are text files with one ID per line:
``train_ids.txt``, ``val_ids.txt``.
"""

# ------------------------------------------------------------
#  Hyper‑parameters & constants
# ------------------------------------------------------------

PAD_LABEL = -100          # ignored in the loss on padded positions

# ------------------------------------------------------------
#  Dataset
# ------------------------------------------------------------


class ProteinPairH5(Dataset):
    """Yield `(embedding, tokens)` pairs from two parallel HDF5 stores."""

    def __init__(self, emb_h5: str, tok_h5: str, id_list_txt: str):
        self.emb_h5 = h5py.File(emb_h5, "r", swmr=True)
        self.tok_h5 = h5py.File(tok_h5, "r", swmr=True)
        with open(id_list_txt) as fh:
            self.ids = [ln.strip() for ln in fh if ln.strip()]

    @staticmethod
    def _fetch(root, pid, fallback):
        """Return numpy array no matter if data sit at `/pid` or `/pid/<fallback>`"""
        node = root[pid]
        if isinstance(node, h5py.Dataset):
            return node[:]                     # dataset directly at /pid
        return node[fallback][:]               # inside a subgroup

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        emb = torch.from_numpy(self._fetch(self.emb_h5, pid, "emb")).float()
        tok = torch.from_numpy(self._fetch(self.tok_h5, pid, "tokens")).long()
        return emb, tok


def collate_fn(batch):
    """Right‑pad variable‑length proteins so tensors are rectangular."""
    embs, toks = zip(*batch)
    L_max = max(e.size(0) for e in embs)
    d_emb = embs[0].size(1)
    B = len(batch)

    emb_pad = torch.zeros(B, L_max, d_emb, dtype=torch.float32)
    tok_pad = torch.full((B, L_max), PAD_LABEL, dtype=torch.long)
    mask    = torch.zeros(B, L_max, dtype=torch.bool)

    for i, (e, t) in enumerate(zip(embs, toks)):
        L = e.size(0)
        emb_pad[i, :L] = e
        tok_pad[i, :L] = t
        mask[i, :L] = True
    return emb_pad, tok_pad, mask

# ------------------------------------------------------------
#  Model
# ------------------------------------------------------------


class ResidueTokenClassifier(nn.Module):
    """Simple position‑wise 2‑layer MLP head."""

    def __init__(self, d_emb: int, hidden: int, vocab_size: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_emb, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L, d_emb)
        B, L, D = x.shape
        logits = self.net(x.view(-1, D))                 # (B*L, V)
        return logits.view(B, L, -1)                     # (B, L, V)

# ------------------------------------------------------------
#  Training helpers
# ------------------------------------------------------------


def _masked_accuracy(logits: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    correct = ((pred == tgt) & mask).sum().item()
    total = mask.sum().item()
    return correct / total if total else 0.0


def _run_epoch(model, loader, criterion, optimizer=None, *, device):
    train = optimizer is not None
    model.train() if train else model.eval()
    agg_loss = agg_acc = n = 0
    torch.set_grad_enabled(train)

    for emb, tok, mask in tqdm(loader, desc="train" if train else "val", leave=False):
        emb, tok, mask = emb.to(device), tok.to(device), mask.to(device)
        logits = model(emb)
        loss = criterion(logits.transpose(1, 2), tok)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bs = emb.size(0)
        agg_loss += loss.item() * bs
        agg_acc  += _masked_accuracy(logits, tok, mask) * bs
        n += bs
    return agg_loss / n, agg_acc / n

# ------------------------------------------------------------
#  CLI & main
# ------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser("Train per‑residue token classifier (emb.h5 + tokens.h5)")
    p.add_argument("--emb_h5", required=True, help="HDF5 with ProtT5 embeddings")
    p.add_argument("--tok_h5", required=True, help="HDF5 with bio2token labels")
    p.add_argument("--train_ids", required=True, help="txt file with protein IDs for training")
    p.add_argument("--val_ids", required=True, help="txt file with protein IDs for validation")
    p.add_argument("--codebook", required=True, help=".npy centroids → derive vocab size V")
    p.add_argument("--d_emb", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main(cfg):
    vocab = np.load(cfg.codebook).shape[0]
    print(f"Codebook size V = {vocab}")

    train_ds = ProteinPairH5(cfg.emb_h5, cfg.tok_h5, cfg.train_ids)
    val_ds   = ProteinPairH5(cfg.emb_h5, cfg.tok_h5, cfg.val_ids)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True,  collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch, shuffle=False, collate_fn=collate_fn, pin_memory=True)

    model = ResidueTokenClassifier(cfg.d_emb, cfg.hidden, vocab, cfg.dropout).to(cfg.device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    crit  = nn.CrossEntropyLoss(ignore_index=PAD_LABEL)

    for ep in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, crit, optim, device=cfg.device)
        va_loss, va_acc = _run_epoch(model, val_loader,   crit, device=cfg.device)
        print(f"Epoch {ep:02d} | train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f}")

    torch.save(model.state_dict(), "token_classifier.pt")
    print("Saved → token_classifier.pt")


if __name__ == "__main__":
    main(parse_args())
