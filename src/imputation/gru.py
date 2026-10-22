"""
Bidirectional GRU imputer.

Input per frame: [x, y, dx, dy, is_hidden], with coordinates centred on the mean of the
observed frames (in units of SCALE) and dx, dy the per-frame displacement (x 25, i.e. m/s). Hidden frames are pre-filled with linear interpolation, and
the network predicts a residual on top of it, so at initialisation it is exactly the
linear baseline and it only has to learn the correction (curvature, noise filtering).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

SCALE = 1.0  # metres


def linear_fill(y: torch.Tensor, start: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    """y: (B, W, 2); contiguous hidden block [start, start+length) per sample."""
    B, W, _ = y.shape
    t = torch.arange(W, device=y.device)[None, :]
    s, L = start[:, None], length[:, None]
    left = y[torch.arange(B), start - 1]           # (B, 2)
    right = y[torch.arange(B), start + length]     # (B, 2)
    alpha = ((t - s + 1).float() / (L + 1).float()).clamp(0, 1)  # (B, W)
    interp = left[:, None] + alpha[..., None] * (right - left)[:, None]
    hidden = (t >= s) & (t < s + L)
    return torch.where(hidden[..., None], interp, y), hidden


class GRUImputer(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(5, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.out = nn.Linear(2 * hidden, 2)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, filled: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """filled: (B, W, 2) metres (linear-filled); hidden: (B, W) bool. Returns metres."""
        obs = (~hidden).float()[..., None]
        anchor = (filled * obs).sum(1, keepdim=True) / obs.sum(1, keepdim=True)
        z = (filled - anchor) / SCALE
        vel = torch.diff(filled, dim=1, prepend=filled[:, :1]) * 25.0
        feats = torch.cat([z, vel, hidden.float()[..., None]], -1)
        h, _ = self.gru(torch.relu(self.inp(feats)))
        return filled + self.out(h) * SCALE


@torch.no_grad()
def predict_numpy(model: GRUImputer, Y: np.ndarray, mask: np.ndarray, batch: int = 2048) -> np.ndarray:
    """Shared-mask prediction (for evaluate()). mask must be one contiguous block."""
    model.eval()
    idx = np.flatnonzero(mask)
    s, L = int(idx[0]), len(idx)
    out = []
    for i in range(0, len(Y), batch):
        y = torch.from_numpy(np.ascontiguousarray(Y[i:i + batch])).float()
        n = len(y)
        filled, hid = linear_fill(y, torch.full((n,), s), torch.full((n,), L))
        out.append(model(filled, hid).numpy())
    return np.concatenate(out)
