import os.path

import torch
import torch.nn as nn
from models.simple_classifier.datasets import ProteinPairJSONL, ProteinPairJSONL_FromDir, PAD_LABEL
from models.model_utils import _masked_accuracy


class ResidueTokenCNN(nn.Module):
    """1D‐CNN head."""

    def __init__(self, d_emb: int, hidden: list, vocab_size: int, kernel_sizes=[5], dropout: float = 0.3):
        super().__init__()

        # input conv
        self.conv_in = nn.Conv1d(in_channels=d_emb,
                                 out_channels=hidden[0],
                                 kernel_size=kernel_sizes[0],
                                 padding=(kernel_sizes[0] - 1) // 2)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden) - 1):
            k = kernel_sizes[i + 1]
            self.hidden_layers.append(
                nn.Conv1d(in_channels=hidden[i],
                          out_channels=hidden[i + 1],
                          kernel_size=k,
                          padding=(k - 1) // 2
                          )
            )

        # Output projection
        self.conv_out = nn.Conv1d(in_channels=hidden[-1],
                                  out_channels=vocab_size,
                                  kernel_size=1)

        # save args
        self.args = {
            "d_emb": d_emb,
            "hidden": hidden,
            "vocab_size": vocab_size,
            "kernel_sizes": kernel_sizes,
            "dropout": dropout,
        }

        hidden_layers_string = "_".join(str(i) for i in hidden)
        kernel_sizes_string = "_".join(str(i) for i in kernel_sizes)
        self.model_name = f"cnn_k{kernel_sizes_string}_h{hidden_layers_string}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B, L, D = x.shape
        x = x.permute(0, 2, 1)  # → (B, d_emb, L)
        h = self.conv_in(x)  # → (B, hidden, L)
        h = self.relu(h)
        h = self.drop(h)

        for layer in self.hidden_layers:
            h = layer(h)
            h = self.relu(h)
            h = self.drop(h)

        h = self.conv_out(h)  # → (B, vocab_size, L)
        out = h.permute(0, 2, 1)  # → (B, L, vocab_size)
        return out

    def save(self, out_put_dir):
        torch.save({
            "model_args": self.args,
            "state_dict": self.state_dict()
        }, os.path.join(out_put_dir, f"{self.model_name}.pt"))

    @staticmethod
    def load_cnn(path: str):
        checkpoint = torch.load(path)
        model = ResidueTokenCNN(**checkpoint["model_args"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model


class ResidueTokenLNN(nn.Module):
    """Linear head."""

    def __init__(self, d_emb: int, hidden: int, vocab_size: int, dropout: float = 0.3):
        super().__init__()
        self.lin1 = nn.Linear(in_features=d_emb,
                              out_features=hidden)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.lin2 = nn.Linear(in_features=hidden,
                              out_features=vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_LABEL)

        # define most important metric and whether it needs to be minimized or maximized
        self.key_metric = "val_acc"
        self.maximize = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B, L, D = x.shape
        x = x.permute(0, 2, 1)  # → (B, d_emb, L)
        h = self.lin1(x)  # → (B, hidden, L)
        h = self.relu(h)
        h = self.drop(h)
        h = self.lin2(h)  # → (B, vocab_size, L)
        out = h.permute(0, 2, 1)  # → (B, L, vocab_size)
        return out

    def run_epoch(self, loader, optimizer=None, device="cpu"):
        is_train = optimizer is not None
        self.train() if is_train else self.eval()
        total_loss = total_acc = total_samples = 0
        torch.set_grad_enabled(is_train)

        for emb, tok in loader:
            emb, tok = emb.to(device), tok.to(device)
            mask = (tok != PAD_LABEL)
            logits = self(emb)
            loss = self.criterion(logits.transpose(1, 2), tok)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bsz = emb.size(0)
            total_loss += loss.item() * bsz
            total_acc += _masked_accuracy(logits, tok, mask) * bsz
            total_samples += bsz
        set_prefix = ""
        if not is_train:
            set_prefix = "val_"
        score_dict = {
            f"{set_prefix}acc": total_acc / total_samples,
            f"{set_prefix}loss": total_loss / total_samples,
        }
        return score_dict
