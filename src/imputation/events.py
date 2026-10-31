"""
Ball possession, pass and dribble labels derived from SoccerNet-GSR game states, and the
samples used by the action model (src/imputation/action_model.py).

Ball carrier: the nearest player or goalkeeper within ACQUIRE m of the ball; the current
carrier keeps the ball while within KEEP m (hysteresis against flicker between two nearby
players). Gaps of up to FILL frames between the same carrier are filled.

Label for a frame t where player c has the ball, looking HORIZON frames (1 s) ahead:
    pass      c releases the ball within 1 s and the next carrier (within 3 s) is a teammate
    lost      same, but the next carrier is an opponent
    dribble   c still has the ball 1 s later
Frames that fit none of these (ball out of view, no clear next carrier) are dropped.
For passes, the receiver is the next carrier, kept only if visible at time t.

Coordinates are canonicalised so the carrier's team always attacks towards +x
(GSR "left" team attacks +x); the whole scene is rotated 180 degrees otherwise.
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass

import numpy as np
import pandas as pd

ACQUIRE, KEEP, FILL = 1.5, 2.5, 3
PAST, HORIZON, LOOK = 50, 25, 75
MAX_AGENTS = 22
ACTIONS = ["dribble", "pass", "lost"]


@dataclass
class Seq:
    name: str
    n: int
    ball: np.ndarray            # (n, 2) nan if unknown
    pos: dict                   # tid -> (n, 2) nan if not labeled
    team: dict                  # tid -> "left"/"right"/""
    role: dict                  # tid -> role
    jersey: dict
    carrier: np.ndarray         # (n,) int, -1 = nobody


def load_states(path) -> list[Seq]:
    df = pd.read_csv(path, dtype={"jersey": str, "team": str})
    out = []
    for name, g in df.groupby("sequence", sort=True):
        n = int(g["frame"].max())
        ball = np.full((n, 2), np.nan, np.float32)
        b = g[(g.role == "ball") & g.x.notna()].drop_duplicates("frame")
        ball[b.frame.values - 1] = b[["x", "y"]].values
        pos, team, role, jersey = {}, {}, {}, {}
        for t, h in g[g.role.isin(["player", "goalkeeper"]) & g.x.notna()].groupby("track_id"):
            a = np.full((n, 2), np.nan, np.float32)
            a[h.frame.values - 1] = h[["x", "y"]].values
            pos[int(t)] = a
            team[int(t)] = (h.team.dropna().iloc[0] if h.team.notna().any() else "")
            role[int(t)] = h.role.iloc[0]
            jersey[int(t)] = h.jersey.dropna().iloc[0] if h.jersey.notna().any() else ""
        s = Seq(name, n, ball, pos, team, role, jersey, np.full(n, -1))
        s.carrier = compute_carrier(s)
        out.append(s)
    return out


def compute_carrier(s: Seq) -> np.ndarray:
    tids = list(s.pos)
    P = np.stack([s.pos[t] for t in tids]) if tids else np.zeros((0, s.n, 2))
    car, cur = np.full(s.n, -1), -1
    for f in range(s.n):
        b = s.ball[f]
        if np.isnan(b).any() or not len(tids):
            cur = -1
            continue
        d = np.linalg.norm(P[:, f] - b, axis=1)
        d = np.where(np.isnan(d), np.inf, d)
        if cur in tids and d[tids.index(cur)] < KEEP:
            car[f] = cur
            continue
        j = int(np.argmin(d))
        cur = tids[j] if d[j] < ACQUIRE else -1
        car[f] = cur
    # fill short holes between the same carrier
    f = 0
    while f < s.n:
        if car[f] == -1:
            e = f
            while e < s.n and car[e] == -1:
                e += 1
            if 0 < f and e < s.n and e - f <= FILL and car[f - 1] == car[e]:
                car[f:e] = car[e]
            f = e
        else:
            f += 1
    return car


def label_frame(s: Seq, t: int):
    """Returns (action, receiver_tid or -1, release_frame) or None."""
    c = s.carrier[t]
    if c < 0:
        return None
    end = min(s.n, t + LOOK + 1)
    nxt = next((f for f in range(t + 1, end) if s.carrier[f] not in (-1, c)), None)
    held = [f for f in range(t, min(s.n, t + HORIZON + 1)) if s.carrier[f] == c]
    release = max(held) if held else t
    if t + HORIZON < s.n and s.carrier[t + HORIZON] == c and (nxt is None or nxt > t + HORIZON):
        return "dribble", -1, -1
    if nxt is not None and release < t + HORIZON:
        r = s.carrier[nxt]
        same = s.team.get(r) == s.team.get(c) and s.team.get(c) != ""
        return ("pass" if same else "lost"), (r if same else -1), release
    return None


def canon(team: str) -> float:
    return 1.0 if team == "left" else -1.0


def build_samples(seqs: list[Seq], stride: int = 1, require_label: bool = True):
    """Feature arrays for every labelled frame with 2 s of carrier history.

    With require_label=False every frame with a carrier is kept (act = -1 when unlabelled),
    which is what the explorer export uses."""
    X_car, X_ball, X_ag, M_ag, T_ag, Y_act, Y_rec, Y_fut, ABS, META = [], [], [], [], [], [], [], [], [], []
    for s in seqs:
        for t in range(PAST - 1, s.n - 1, stride):
            if s.carrier[t] < 0:
                continue
            lab = label_frame(s, t)
            if lab is None:
                if require_label:
                    continue
                lab = (None, -1, -1)
            c = s.carrier[t]
            hist = s.pos[c][t - PAST + 1:t + 1]
            if np.isnan(hist).any() or np.linalg.norm(np.diff(hist, axis=0), axis=1).max() > 1.0:
                continue
            k = canon(s.team.get(c, "left"))
            o = hist[-1]
            car = (hist - o) * k
            ball = (s.ball[t - PAST + 1:t + 1] - o) * k
            fut = s.pos[c][t + 1:t + 1 + HORIZON]
            if len(fut) < HORIZON:
                fut = np.full((HORIZON, 2), np.nan, np.float32)
            fut = (fut - o) * k
            ag = np.zeros((MAX_AGENTS, 6), np.float32)
            msk = np.zeros(MAX_AGENTS, bool)
            tids = np.full(MAX_AGENTS, -1)
            j = 0
            for tid, p in s.pos.items():
                if tid == c or np.isnan(p[t]).any() or j >= MAX_AGENTS:
                    continue
                v = (p[t] - p[t - 5]) * 5.0 if not np.isnan(p[t - 5]).any() else np.zeros(2)
                mate = float(s.team.get(tid) == s.team.get(c))
                ag[j] = [*((p[t] - o) * k), *(v * k), mate, float(s.role.get(tid) == "goalkeeper")]
                msk[j], tids[j] = True, tid
                j += 1
            rec = -1
            if lab[0] == "pass" and lab[1] in tids:
                rec = int(np.flatnonzero(tids == lab[1])[0])
            X_car.append(car); X_ball.append(ball); X_ag.append(ag); M_ag.append(msk); T_ag.append(tids)
            Y_act.append(ACTIONS.index(lab[0]) if lab[0] else -1); Y_rec.append(rec); Y_fut.append(fut)
            ABS.append(o * k); META.append((s.name, t, c, lab[2]))
    f32 = lambda a: np.asarray(a, np.float32)
    return {"car": f32(X_car), "ball": f32(X_ball), "ag": f32(X_ag), "mask": np.asarray(M_ag), "tids": np.asarray(T_ag),
            "act": np.asarray(Y_act), "rec": np.asarray(Y_rec), "fut": f32(Y_fut), "abs": f32(ABS), "meta": META}
