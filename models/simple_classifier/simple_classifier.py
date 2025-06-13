import torch
import torch.nn as nn

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