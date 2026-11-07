"""
SkillCorner open data (full matches, 10 fps broadcast tracking with extrapolated off-screen
players, plus dynamic events) turned into samples for the possession model.

For every frame t inside a player possession (SkillCorner `player_possession` event) the
sample holds what a live feed would have at t:
  - carrier and ball over the last 2 s (PAST frames), relative to the carrier now
  - every other player now: position and velocity relative to the carrier, teammate,
    goalkeeper and "seen by the camera" flags
and the labels:
  - outcome of the possession: pass / loss / shot (other endings are dropped)
  - whether the carrier releases the ball within 1 s
  - for passes: the targeted teammate and whether the pass arrived
Coordinates are canonicalised so the carrier's team attacks +x.
SkillCorner's own passing-option scores are kept as a reference baseline.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FPS = 10
PAST = 20          # 2 s of history
HORIZON = 10       # 1 s
MAX_AGENTS = 22
OUTCOMES = ["pass", "loss", "shot"]
END_MAP = {"pass": 0, "possession_loss": 1, "shot": 2}


@dataclass
class Match:
    mid: str
    n: int
    pids: np.ndarray        # (P,) player ids
    pos: np.ndarray         # (n, P, 2) nan when absent
    det: np.ndarray         # (n, P) bool, seen by the camera
    ball: np.ndarray        # (n, 3) nan when unknown
    team: np.ndarray        # (P,) team id
    gk: np.ndarray          # (P,) bool
    number: np.ndarray      # (P,) shirt number
    name: list
    home: int
    teams: dict             # team id -> short name
    events: pd.DataFrame


def _parse_tracking(path: Path, pid_index: dict):
    try:
        import orjson as js
    except ImportError:  # optional; orjson only makes the first parse about 3x faster
        import json as js
    lines = path.read_bytes().splitlines()
    n = len(lines)
    P = len(pid_index)
    pos = np.full((n, P, 2), np.nan, np.float32)
    det = np.zeros((n, P), bool)
    ball = np.full((n, 3), np.nan, np.float32)
    for line in lines:
        fr = js.loads(line)
        f = fr["frame"]
        if f >= n:
            continue
        b = fr.get("ball_data") or {}
        if b.get("x") is not None:
            ball[f] = (b["x"], b["y"], b.get("z") or 0.0)
        for p in fr.get("player_data") or []:
            j = pid_index.get(p["player_id"])
            if j is None or p.get("x") is None:
                continue
            pos[f, j] = (p["x"], p["y"])
            det[f, j] = bool(p.get("is_detected"))
    return pos, det, ball


def load_match(d: Path, cache_dir: Path | None = None) -> Match:
    d = Path(d)
    mid = d.name
    meta = json.loads((d / f"{mid}_match.json").read_text())
    players = meta["players"]
    pids = np.array([p["id"] for p in players])
    pid_index = {int(p): i for i, p in enumerate(pids)}
    cache = (cache_dir / f"{mid}.npz") if cache_dir else None
    if cache and cache.exists():
        z = np.load(cache)
        pos, det, ball = z["pos"], z["det"], z["ball"]
    else:
        pos, det, ball = _parse_tracking(d / f"{mid}_tracking_extrapolated.jsonl", pid_index)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, pos=pos, det=det, ball=ball)
    ev = pd.read_csv(d / f"{mid}_dynamic_events.csv", low_memory=False)
    return Match(
        mid=mid, n=len(pos), pids=pids, pos=pos, det=det, ball=ball,
        team=np.array([p["team_id"] for p in players]),
        gk=np.array([(p.get("player_role") or {}).get("acronym") == "GK" for p in players]),
        number=np.array([p.get("number") or -1 for p in players]),
        name=[p.get("short_name") or "" for p in players],
        home=meta["home_team"]["id"],
        teams={meta["home_team"]["id"]: meta["home_team"].get("short_name", "Home"),
               meta["away_team"]["id"]: meta["away_team"].get("short_name", "Away")},
        events=ev,
    )


def possessions(m: Match) -> pd.DataFrame:
    """Player possessions with a usable ending, plus SkillCorner's passing-option reference."""
    ev = m.events
    p = ev[(ev.event_type == "player_possession") & ev.end_type.isin(END_MAP)].copy()
    p["outcome"] = p.end_type.map(END_MAP)
    p["success"] = np.where(p.pass_outcome == "successful", 1, np.where(p.pass_outcome.isin(["unsuccessful", "offside"]), 0, -1))
    o = ev[ev.event_type == "passing_option"]
    sc_pred, sc_xpass = {}, {}
    for pid_, g in o.groupby("associated_player_possession_event_id"):
        g = g.dropna(subset=["passing_option_score"])
        if len(g):
            sc_pred[pid_] = int(g.loc[g.passing_option_score.idxmax(), "player_id"])
        t = g[g.targeted == True]  # noqa: E712
        if len(t) and pd.notna(t.xpass_completion.iloc[0]):
            sc_xpass[pid_] = float(t.xpass_completion.iloc[0])
    p["sc_pred"] = p.event_id.map(sc_pred)
    p["sc_xpass"] = p.event_id.map(sc_xpass)
    return p


def frame_features(m: Match, t: int, c: int, k: float):
    """Features at frame t for carrier index c (k = +1 / -1 canonical direction).
    Returns car (PAST,2), ball (PAST,3), ag (A,7), mask (A,), idx (A,) or None."""
    if t < PAST - 1:
        return None
    hist = m.pos[t - PAST + 1:t + 1, c]
    if np.isnan(hist).any():
        return None
    o = hist[-1]
    car = (hist - o) * k
    b = m.ball[t - PAST + 1:t + 1].copy()
    b[:, :2] = (b[:, :2] - o) * k
    ag = np.zeros((MAX_AGENTS, 7), np.float32)
    mask = np.zeros(MAX_AGENTS, bool)
    idx = np.full(MAX_AGENTS, -1)
    j = 0
    now, prev = m.pos[t], m.pos[t - 2]
    for q in np.flatnonzero(~np.isnan(now[:, 0])):
        if q == c or j >= MAX_AGENTS:
            continue
        v = (now[q] - prev[q]) * (FPS / 2) if not np.isnan(prev[q, 0]) else np.zeros(2)
        ag[j] = [*((now[q] - o) * k), *(v * k), float(m.team[q] == m.team[c]), float(m.gk[q]), float(m.det[t, q])]
        mask[j], idx[j] = True, q
        j += 1
    return car.astype(np.float32), b.astype(np.float32), ag, mask, idx


def build_samples(m: Match, stride: int = 1):
    out = {k: [] for k in ["car", "ball", "ag", "mask", "idx", "outcome", "soon", "rec", "success", "abs", "meta"]}
    for _, r in possessions(m).iterrows():
        if pd.isna(r.player_id):
            continue
        c = np.flatnonzero(m.pids == int(r.player_id))
        if not len(c):
            continue
        c = int(c[0])
        k = 1.0 if r.attacking_side == "left_to_right" else -1.0
        tgt = np.flatnonzero(m.pids == int(r.player_targeted_id)) if pd.notna(r.player_targeted_id) else []
        tgt = int(tgt[0]) if len(tgt) else -1
        for t in range(int(r.frame_start), int(r.frame_end) + 1, stride):
            if t >= m.n:
                break
            f = frame_features(m, t, c, k)
            if f is None:
                continue
            car, ball, ag, mask, idx = f
            rec = int(np.flatnonzero(idx == tgt)[0]) if r.outcome == 0 and tgt in idx else -1
            out["car"].append(car); out["ball"].append(ball); out["ag"].append(ag); out["mask"].append(mask); out["idx"].append(idx)
            out["outcome"].append(int(r.outcome)); out["soon"].append(int(r.frame_end - t <= HORIZON))
            out["rec"].append(rec); out["success"].append(int(r.success) if r.outcome == 0 else -1)
            out["abs"].append(m.pos[t, c] * k)
            out["meta"].append((m.mid, t, c, r.event_id, int(r.frame_start), int(r.frame_end),
                                -1 if pd.isna(r.sc_pred) else int(r.sc_pred), np.nan if pd.isna(r.sc_xpass) else float(r.sc_xpass)))
    D = {}
    for key, v in out.items():
        if key == "meta":
            D[key] = v
        elif key in ("mask",):
            D[key] = np.asarray(v, bool)
        elif key in ("idx", "outcome", "soon", "rec", "success"):
            D[key] = np.asarray(v, np.int64)
        else:
            D[key] = np.asarray(v, np.float32)
    return D


def concat(Ds):
    return {k: (sum((D[k] for D in Ds), []) if k == "meta" else np.concatenate([D[k] for D in Ds])) for k in Ds[0]}
