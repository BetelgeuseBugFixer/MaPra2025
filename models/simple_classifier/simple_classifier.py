import torch
import torch.nn as nn


class ResidueTokenCNN(nn.Module):
    """1D‐CNN head."""

    def __init__(self, d_emb: int, hidden: list, vocab_size: int, kernel_size=5, dropout: float = 0.3):
        super().__init__()
        self.conv_in = nn.Conv1d(in_channels=d_emb,
                                 out_channels=hidden[0],
                                 kernel_size=kernel_size,
                                 padding=(kernel_size - 1) // 2)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden) - 1):
            self.hidden_layers.append(
                nn.Conv1d(in_channels=hidden[i],
                          out_channels=hidden[i + 1],
                          kernel_size=1
                          )
            )

        self.conv_out = nn.Conv1d(in_channels=hidden[-1],
                                  out_channels=vocab_size,
                                  kernel_size=1)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B, L, D = x.shape
        x = x.permute(0, 2, 1)  # → (B, d_emb, L)
        h = self.lin1(x)  # → (B, hidden, L)
        h = self.relu(h)
        h = self.drop(h)
        h = self.lin2(h)  # → (B, vocab_size, L)
        out = h.permute(0, 2, 1)  # → (B, L, vocab_size)
        return out
