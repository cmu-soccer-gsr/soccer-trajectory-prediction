#!/usr/bin/env python3
"""
Run a SoccerNet GSR clip through the same live engine as the SkillCorner matches, so the
Live Board can show the possession model over the real broadcast video.

    python scripts/record_gsr_clip.py --explorer explorer --out live/replay_gsr.json --video broadcast_gsr.mp4

Uses the explorer's clip.json (labeled pitch positions, 25 fps) and forecast.json (one
pitch-to-image homography per frame). Positions are resampled to the engine's 10 Hz.
Differences from the SkillCorner feed the model was trained on: only players the camera
sees are present (no estimates for the others), and the ball has no height.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from imputation.live import LiveEngine  # noqa: E402
from imputation.possession_model import PossessionModel  # noqa: E402

FPS_OUT = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explorer", required=True, type=Path, help="folder with clip.json, forecast.json, broadcast.mp4")
    ap.add_argument("--model", default=REPO / "results" / "possession_model.pt", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--video", default=None, help="copy broadcast.mp4 next to --out under this name")
    a = ap.parse_args()
    C = json.loads((a.explorer / "clip.json").read_text())
    H = json.loads((a.explorer / "forecast.json").read_text())["homography"]
    fps_in = C["fps"]
    tracks = {int(t): m for t, m in C["tracks"].items()}
    people = [t for t, m in tracks.items() if m["role"] in ("player", "goalkeeper")]
    ball_ids = [t for t, m in tracks.items() if m["role"] == "ball"]
    # GSR's y grows towards the camera; flip it so the board has the camera at the bottom, as in the video
    model = PossessionModel()
    model.load_state_dict(torch.load(a.model, map_location="cpu")["state_dict"])
    eng = LiveEngine(model, people, [tracks[t]["team"] for t in people], [tracks[t]["role"] == "goalkeeper" for t in people],
                     "left", ["left_to_right"])  # GSR's left team attacks +x
    eng.model(*[torch.zeros(1, *sh) for sh in [(20, 2), (20, 3), (22, 7)]], torch.ones(1, 22, dtype=torch.bool), torch.zeros(1, 2))  # warm-up
    n_out = int((C["n"] - 1) / fps_in * FPS_OUT) + 1
    frames, ms = [], []
    r1 = lambda v: round(v, 1)
    for i in range(n_out):
        f = min(C["n"] - 1, round(i * fps_in / FPS_OUT))
        rows = C["frames"][f]
        ball = next(([x, -y] for t, *_, x, y in rows if t in ball_ids and x is not None), None)
        pl = [{"player_id": t, "x": x, "y": -y, "is_detected": True} for t, *_, x, y in rows if t in tracks and tracks[t]["role"] in ("player", "goalkeeper") and x is not None]
        fr = {"frame": i, "period": 1, "timestamp": None, "ball_data": {"x": None if ball is None else ball[0], "y": None if ball is None else ball[1], "z": 0.0},
              "player_data": pl, "possession": {"player_id": None}}
        out = eng.step(fr)
        msg = {"f": i, "vt": round(f / fps_in, 3), "b": None if ball is None else [r1(ball[0]), r1(ball[1]), 0],
               "pl": [[p["player_id"], r1(p["x"]), r1(p["y"]), 1] for p in pl], "ms": round(out["ms"], 2)}
        h = H[f]
        if h:
            h = list(h); h[1], h[4], h[7] = -h[1], -h[4], -h[7]   # same flip of y
            msg["h"] = [round(v, 7) for v in h]
        if out.get("carrier") is not None:
            ms.append(out["ms"])
            msg.update(c=out["carrier"], o=[round(v, 3) for v in out["outcome"]], s=round(out["soon"], 3),
                       op=[[pid, round(p, 3), round(q, 3)] for pid, p, q in out["options"][:5]])
        frames.append(msg)
    info = C["info"]
    players = {str(t): {"n": int(tracks[t]["jersey"]) if str(tracks[t].get("jersey") or "").isdigit() else None,
                        "name": ("No. " + tracks[t]["jersey"]) if tracks[t].get("jersey") else "No. not labeled",
                        "team": "home" if tracks[t]["team"] == "left" else "away", "gk": tracks[t]["role"] == "goalkeeper"} for t in people}
    out = {"roster": {"match": info["name"], "teams": {"home": {"name": "Left team", "short": "Left"}, "away": {"name": "Right team", "short": "Right"}},
                      "pitch": [105, 68], "players": players, "clip": f"{info['name']} · game clock {info['game_time_start']}"},
           "fps": FPS_OUT, "kind": "video", "image": [C["width"], C["height"]],
           "latency": {"n": len(ms), "p50": float(np.percentile(ms, 50)), "p95": float(np.percentile(ms, 95)), "max": float(np.max(ms)), "budget_ms": 100.0, "cpu": "1 thread"},
           "frames": frames, "events": []}
    a.out.write_text(json.dumps(out, separators=(",", ":")))
    if a.video:
        shutil.copy(a.explorer / "broadcast.mp4", a.out.parent / a.video)
    print(f"{len(frames)} frames ({sum('c' in m for m in frames)} with a carrier) -> {a.out}")


if __name__ == "__main__":
    main()
