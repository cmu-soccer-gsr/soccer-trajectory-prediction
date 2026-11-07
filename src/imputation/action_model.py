"""
Context-aware model for the ball carrier's next second (samples from events.build_samples).

Inputs (canonicalised so the carrier's team attacks +x):
  - carrier and ball over the last 2 s (relative to the carrier's current position)
  - every other visible player now: position and velocity relative to the carrier,
    teammate / goalkeeper flags
  - the carrier's absolute pitch position (distance to goal, touchline)
Outputs:
  - action: dribble / pass / lost within 1 s
  - receiver: a score per teammate (softmax over visible teammates)
  - the carrier's path over the next 1 s: 3 candidates + probabilities (winner-takes-all)

Architecture: a GRU encodes the carrier + ball history into a carrier token; agent tokens
come from an MLP; two transformer layers let all tokens attend to each other.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .events import HORIZON

K = 3


def history_features(car, ball):
    """car, ball: (B, T, 2) relative to the carrier's current position (ball may be nan)."""
    bmask = torch.isnan(ball).any(-1, keepdim=True)
    ball = torch.where(bmask, torch.zeros_like(ball), ball)
    vc = torch.diff(car, dim=1, prepend=car[:, :1]) * 25.0
    vb = torch.diff(ball, dim=1, prepend=ball[:, :1]) * 25.0
    vb = torch.where(bmask, torch.zeros_like(vb), vb)
    return torch.cat([car / 10, vc / 5, ball / 10, vb / 10, (~bmask).float()], -1)  # 9


def agent_features(ag):
    """ag: (B, A, 6) = rel x, y, vx, vy, teammate, goalkeeper."""
    d = torch.linalg.norm(ag[..., :2], dim=-1, keepdim=True)
    return torch.cat([ag[..., :2] / 10, ag[..., 2:4] / 5, ag[..., 4:6], d / 10], -1)  # 7


class ActionModel(nn.Module):
    def __init__(self, h: int = 96, layers: int = 2, k: int = K):
        super().__init__()
        self.k = k
        self.hist = nn.Linear(9, h)
        self.gru = nn.GRU(h, h, num_layers=2, batch_first=True, dropout=0.1)
        self.agent = nn.Sequential(nn.Linear(7, h), nn.ReLU(), nn.Linear(h, h))
        self.kind = nn.Embedding(2, h)  # 0 = carrier token, 1 = other player
        enc = nn.TransformerEncoderLayer(h, 4, dim_feedforward=2 * h, dropout=0.1, batch_first=True)
        self.tf = nn.TransformerEncoder(enc, layers)
        self.act = nn.Sequential(nn.Linear(h + 2, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, 3))
        self.rec = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))
        self.path = nn.Sequential(nn.Linear(h + 4, 2 * h), nn.ReLU(), nn.Linear(2 * h, k * HORIZON * 2 + k))

    def forward(self, car, ball, ag, mask, absxy):
        B = car.shape[0]
        hseq, _ = self.gru(torch.relu(self.hist(history_features(car, ball))))
        ctok = hseq[:, -1:] + self.kind.weight[0]
        atok = self.agent(agent_features(ag)) + self.kind.weight[1]
        x = torch.cat([ctok, atok], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=car.device), ~mask], 1)
        x = self.tf(x, src_key_padding_mask=pad)
        c, a = x[:, 0], x[:, 1:]
        act = self.act(torch.cat([c, absxy / 50], -1))
        rec = self.rec(torch.cat([a, c[:, None].expand_as(a)], -1)).squeeze(-1)
        mates = mask & (ag[..., 4] > 0.5)
        rec = rec.masked_fill(~mates, -1e4)
        v = (car[:, -1] - car[:, -6]) * 5.0
        out = self.path(torch.cat([c, absxy / 50, v / 5], -1))
        paths = out[:, : self.k * HORIZON * 2].view(B, self.k, HORIZON, 2)
        t = torch.arange(1, HORIZON + 1, device=car.device).float()[None, None, :, None] / 25.0
        paths = paths + v[:, None, None, :] * t
        return act, rec, paths, out[:, self.k * HORIZON * 2:]


def tensors(D, idx=None):
    sel = slice(None) if idx is None else idx
    f = lambda k: torch.from_numpy(np.ascontiguousarray(D[k][sel]))
    return f("car"), f("ball"), f("ag"), f("mask"), f("abs")


@torch.no_grad()
def predict(model, D, batch=2048):
    model.eval()
    outs = {"act": [], "rec": [], "paths": [], "pprob": []}
    n = len(D["act"])
    for i in range(0, n, batch):
        idx = np.arange(i, min(n, i + batch))
        a, r, p, l = model(*tensors(D, idx))
        outs["act"].append(torch.softmax(a, -1).numpy())
        outs["rec"].append(torch.softmax(r, -1).numpy())
        outs["paths"].append(p.numpy())
        outs["pprob"].append(torch.softmax(l, -1).numpy())
    return {k: np.concatenate(v) for k, v in outs.items()}
