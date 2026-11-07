#!/usr/bin/env python3
"""
Export 1 s forecasts and per-frame homographies for the explorer (adds forecast.json).

    python scripts/export_forecasts.py --clip-dir data/SNGS-021 --results results --out explorer/site

For every 2nd frame and every player or goalkeeper with 2 s of continuous history, the
forecaster's 3 candidate paths (points every 0.2 s up to 1 s, cm offsets from the current
position) and their probabilities. Each frame also gets a pitch-to-image homography,
fitted to the labeled boxes (bottom middle of the box vs pitch position), so paths can be
drawn on the broadcast frame.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-dir", required=True, type=Path)
    ap.add_argument("--results", default=REPO / "results", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--every", type=int, default=2)
    a = ap.parse_args()
    import torch
    from imputation.forecast import FUTURE, PAST, Forecaster, predict

    clip = json.loads((a.out / "clip.json").read_text())
    tracks, frames = clip["tracks"], clip["frames"]
    n = clip["n"]
    pos = {}
    for f, fs in enumerate(frames):
        for t, bx, by, bw, bh, x, y in fs:
            if x is not None and tracks[str(t)]["role"] in ("player", "goalkeeper"):
                pos.setdefault(t, {})[f] = (x, y)

    homs = []
    for fs in frames:
        P = np.array([[x, y] for t, bx, by, bw, bh, x, y in fs if x is not None], np.float64)
        I = np.array([[bx + bw / 2, by + bh] for t, bx, by, bw, bh, x, y in fs if x is not None], np.float64)
        H = cv2.findHomography(P, I, cv2.RANSAC, 8.0)[0] if len(P) >= 4 else None
        homs.append(None if H is None else [round(v, 7) for v in (H / H[2, 2]).ravel()[:8].tolist()])

    keys, pasts = [], []
    for f in range(PAST - 1, n, a.every):
        for t, d in pos.items():
            if all((f - k) in d for k in range(PAST)):
                p = np.array([d[f - PAST + 1 + k] for k in range(PAST)], np.float32)
                if np.linalg.norm(np.diff(p, axis=0), axis=1).max() <= 1.0:
                    keys.append((f, t))
                    pasts.append(p)
    st = torch.load(a.results / "forecaster.pt", map_location="cpu")
    model = Forecaster()
    model.load_state_dict(st["state_dict"])
    paths, prob = predict(model, np.stack(pasts))
    idx = np.arange(4, FUTURE, 5)  # 0.2, 0.4, ..., 1.0 s
    by_frame = {}
    for (f, t), P, pr in zip(keys, paths, prob):
        by_frame.setdefault(f, []).append(
            [int(t), [int(round(v * 100)) for v in pr]] + [np.round(P[k, idx] * 100).astype(int).ravel().tolist() for k in range(len(pr))])
    out = {"every": a.every, "past": PAST, "future": FUTURE, "step": 5, "homography": homs,
           "frames": {str(f): v for f, v in by_frame.items()}}
    (a.out / "forecast.json").write_text(json.dumps(out, separators=(",", ":")))
    print(len(keys), "forecasts,", (a.out / "forecast.json").stat().st_size, "bytes")


if __name__ == "__main__":
    main()
