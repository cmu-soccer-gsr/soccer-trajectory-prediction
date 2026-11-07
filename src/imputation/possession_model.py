"""
Possession model for live tracking feeds (samples from skillcorner.build_samples).

Per frame, for whoever has the ball:
  - outcome of the possession: pass / loss / shot
  - does the carrier release the ball within 1 s
  - for a pass: a score per teammate (who gets it) and the chance it arrives

A GRU encodes the carrier's and the ball's last 2 s into a carrier token, an MLP turns each
other player into a token, and two transformer layers let them attend to each other.
Small on purpose: one forward pass takes about a millisecond on a laptop CPU.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def history_features(car, ball):
    """car (B,T,2), ball (B,T,3) relative to the carrier now; ball may be nan."""
    bm = torch.isnan(ball).any(-1, keepdim=True)
    ball = torch.where(bm, torch.zeros_like(ball), ball)
    vc = torch.diff(car, dim=1, prepend=car[:, :1]) * 10.0
    vb = torch.diff(ball[..., :2], dim=1, prepend=ball[:, :1, :2]) * 10.0
    vb = torch.where(bm, torch.zeros_like(vb), vb)
    return torch.cat([car / 10, vc / 5, ball[..., :2] / 10, vb / 10, ball[..., 2:3] / 3, (~bm).float()], -1)  # 10


def agent_features(ag):
    d = torch.linalg.norm(ag[..., :2], dim=-1, keepdim=True)
    return torch.cat([ag[..., :2] / 10, ag[..., 2:4] / 5, ag[..., 4:7], d / 10], -1)  # 8


class PossessionModel(nn.Module):
    def __init__(self, h: int = 96, layers: int = 2):
        super().__init__()
        self.hist = nn.Linear(10, h)
        self.gru = nn.GRU(h, h, num_layers=2, batch_first=True, dropout=0.1)
        self.agent = nn.Sequential(nn.Linear(8, h), nn.ReLU(), nn.Linear(h, h))
        self.kind = nn.Embedding(2, h)
        enc = nn.TransformerEncoderLayer(h, 4, dim_feedforward=2 * h, dropout=0.1, batch_first=True)
        self.tf = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.Linear(h + 2, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, 4))  # 3 outcomes + soon
        self.pair = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 2))  # receiver score, arrives

    def forward(self, car, ball, ag, mask, absxy):
        B = car.shape[0]
        hs, _ = self.gru(torch.relu(self.hist(history_features(car, ball))))
        x = torch.cat([hs[:, -1:] + self.kind.weight[0], self.agent(agent_features(ag)) + self.kind.weight[1]], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=car.device), ~mask], 1)
        x = self.tf(x, src_key_padding_mask=pad)
        c, a = x[:, 0], x[:, 1:]
        o = self.out(torch.cat([c, absxy / 50], -1))
        pr = self.pair(torch.cat([a, c[:, None].expand_as(a)], -1))
        mates = mask & (ag[..., 4] > 0.5)
        rec = pr[..., 0].masked_fill(~mates, -1e4)
        return o[:, :3], o[:, 3], rec, pr[..., 1]


def tensors(D, idx=None):
    sel = slice(None) if idx is None else idx
    f = lambda k: torch.from_numpy(np.ascontiguousarray(D[k][sel]))
    return f("car"), f("ball"), f("ag"), f("mask"), f("abs")


@torch.no_grad()
def predict(model, D, batch=4096):
    model.eval()
    out = {"outcome": [], "soon": [], "rec": [], "arrive": []}
    n = len(D["outcome"]) if "outcome" in D else len(D["car"])
    for i in range(0, n, batch):
        idx = np.arange(i, min(n, i + batch))
        o, s, r, a = model(*tensors(D, idx))
        out["outcome"].append(torch.softmax(o, -1).numpy())
        out["soon"].append(torch.sigmoid(s).numpy())
        out["rec"].append(torch.softmax(r, -1).numpy())
        out["arrive"].append(torch.sigmoid(a).numpy())
    return {k: np.concatenate(v) for k, v in out.items()}
