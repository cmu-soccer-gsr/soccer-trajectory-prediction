"""
Short-term trajectory forecasting: from a player's last 2 s of pitch positions, predict the
next 1 s. The model outputs K candidate paths with probabilities, trained winner-takes-all
(only the closest candidate gets the position loss), so the candidates spread over the
plausible routes instead of averaging them.

Single player only: no ball, teammates or opponents (yet), so it predicts how the
movement continues, not whether a pass or dribble happens.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

PAST, FUTURE, K = 50, 25, 3


def features(past: torch.Tensor) -> torch.Tensor:
    """past: (B, PAST, 2) metres -> (B, PAST, 4): position relative to the last frame, velocity (m/s)."""
    rel = past - past[:, -1:]
    vel = torch.diff(past, dim=1, prepend=past[:, :1]) * 25.0
    return torch.cat([rel, vel], -1)


class Forecaster(nn.Module):
    def __init__(self, hidden: int = 96, layers: int = 2, k: int = K):
        super().__init__()
        self.k = k
        self.inp = nn.Linear(4, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden + 2, 256), nn.ReLU(), nn.Linear(256, k * FUTURE * 2 + k))

    def forward(self, past: torch.Tensor):
        """Returns paths (B, K, FUTURE, 2) as offsets from the last observed position, and logits (B, K)."""
        h, _ = self.gru(torch.relu(self.inp(features(past))))
        v = (past[:, -1] - past[:, -6]) * 5.0  # mean velocity over the last 0.2 s, m/s
        out = self.head(torch.cat([h[:, -1], v], -1))
        B = past.shape[0]
        paths = out[:, : self.k * FUTURE * 2].view(B, self.k, FUTURE, 2)
        # residual over constant velocity: candidate k = CV extrapolation + learned correction
        t = torch.arange(1, FUTURE + 1, device=past.device).float()[None, None, :, None] / 25.0
        paths = paths + v[:, None, None, :] * t
        return paths, out[:, self.k * FUTURE * 2:]


def wta_loss(paths, logits, fut_rel):
    """paths (B,K,T,2), fut_rel (B,T,2) offsets from the last observed position."""
    d = torch.linalg.norm(paths - fut_rel[:, None], dim=-1).mean(-1)  # (B, K) ADE per candidate
    best = d.argmin(1)
    reg = nn.functional.smooth_l1_loss(paths[torch.arange(len(best)), best], fut_rel, beta=0.05)
    cls = nn.functional.cross_entropy(logits, best)
    return reg + 0.1 * cls, d


@torch.no_grad()
def predict(model: Forecaster, past: np.ndarray, batch: int = 4096):
    model.eval()
    P, L = [], []
    for i in range(0, len(past), batch):
        p, l = model(torch.from_numpy(past[i:i + batch]).float())
        P.append(p.numpy())
        L.append(torch.softmax(l, -1).numpy())
    return np.concatenate(P), np.concatenate(L)
