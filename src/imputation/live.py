"""
Live inference over a tracking feed, one frame at a time.

LiveEngine keeps the last 2 s of the feed in a ring buffer. For each incoming frame it
works out who has the ball (the provider's possession field when present, otherwise the
nearest player within 1.5 m of the ball, kept while within 2.5 m), builds the same
features as training, runs the possession model and returns the predictions plus how long
that took. Nothing looks ahead: a frame's output only uses frames up to that frame.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .skillcorner import FPS, MAX_AGENTS, PAST

ACQUIRE, KEEP = 1.5, 2.5


class LiveEngine:
    def __init__(self, model, pids, team, gk, home_team, home_side_by_period, threads: int = 1):
        torch.set_num_threads(threads)
        self.model = model.eval()
        self.pids = np.asarray(pids)
        self.index = {int(p): i for i, p in enumerate(self.pids)}
        self.team, self.gk = np.asarray(team), np.asarray(gk)
        self.home, self.side = home_team, home_side_by_period
        P = len(self.pids)
        self.pos = np.full((PAST, P, 2), np.nan, np.float32)
        self.ball = np.full((PAST, 3), np.nan, np.float32)
        self.det = np.zeros(P, bool)
        self.n = 0
        self.carrier = -1

    def _push(self, fr):
        self.pos = np.roll(self.pos, -1, axis=0); self.pos[-1] = np.nan
        self.ball = np.roll(self.ball, -1, axis=0); self.ball[-1] = np.nan
        self.det[:] = False
        b = fr.get("ball_data") or {}
        if b.get("x") is not None:
            self.ball[-1] = (b["x"], b["y"], b.get("z") or 0.0)
        for p in fr.get("player_data") or []:
            j = self.index.get(p["player_id"])
            if j is not None and p.get("x") is not None:
                self.pos[-1, j] = (p["x"], p["y"]); self.det[j] = bool(p.get("is_detected"))
        self.n += 1

    def _who_has_ball(self, fr):
        pid = (fr.get("possession") or {}).get("player_id")
        if pid is not None and pid in self.index:
            self.carrier = self.index[pid]
            return self.carrier
        b = self.ball[-1]
        if np.isnan(b[0]) or b[2] > 1.5:   # unknown or in the air
            self.carrier = -1
            return -1
        d = np.linalg.norm(self.pos[-1] - b[:2], axis=1)
        d = np.where(np.isnan(d), np.inf, d)
        if self.carrier >= 0 and d[self.carrier] < KEEP:
            return self.carrier
        j = int(np.argmin(d))
        self.carrier = j if d[j] < ACQUIRE else -1
        return self.carrier

    def step(self, fr):
        t0 = time.perf_counter()
        self._push(fr)
        c = self._who_has_ball(fr)
        out = {"frame": fr["frame"], "carrier": None}
        if c < 0 or self.n < PAST or fr.get("period") is None:
            out["ms"] = (time.perf_counter() - t0) * 1000
            return out
        hist = self.pos[:, c]
        if np.isnan(hist).any():
            out["ms"] = (time.perf_counter() - t0) * 1000
            return out
        home_ltr = self.side[int(fr["period"]) - 1] == "left_to_right"
        k = 1.0 if (self.team[c] == self.home) == home_ltr else -1.0
        o = hist[-1]
        car = (hist - o) * k
        ball = self.ball.copy(); ball[:, :2] = (ball[:, :2] - o) * k
        ag = np.zeros((MAX_AGENTS, 7), np.float32); mask = np.zeros(MAX_AGENTS, bool); idx = np.full(MAX_AGENTS, -1)
        now, prev = self.pos[-1], self.pos[-3]
        j = 0
        for q in np.flatnonzero(~np.isnan(now[:, 0])):
            if q == c or j >= MAX_AGENTS:
                continue
            v = (now[q] - prev[q]) * (FPS / 2) if not np.isnan(prev[q, 0]) else np.zeros(2)
            ag[j] = [*((now[q] - o) * k), *(v * k), float(self.team[q] == self.team[c]), float(self.gk[q]), float(self.det[q])]
            mask[j], idx[j] = True, q
            j += 1
        T = lambda a: torch.from_numpy(np.ascontiguousarray(a))[None]
        with torch.no_grad():
            oc, s, r, a = self.model(T(car), T(ball), T(ag), T(mask), T((o * k).astype(np.float32)))
        po = torch.softmax(oc, -1)[0].numpy(); pr = torch.softmax(r, -1)[0].numpy(); pa = torch.sigmoid(a)[0].numpy()
        mates = [(int(self.pids[idx[i]]), float(pr[i]), float(pa[i])) for i in range(MAX_AGENTS) if mask[i] and ag[i, 4] > 0.5]
        mates.sort(key=lambda m: -m[1])
        out.update(carrier=int(self.pids[c]), outcome=po.tolist(), soon=float(torch.sigmoid(s)[0]), options=mates)
        out["ms"] = (time.perf_counter() - t0) * 1000
        return out
