#!/usr/bin/env python3
"""
Export one SoccerNet-GSR clip for the browser explorer (explorer/index.html).

    python scripts/export_explorer_data.py --clip-dir data/SNGS-021 --results results --out explorer/site

Writes:
    broadcast.mp4   the clip at 960x540, 25 fps, a keyframe every 5 frames (fast seeking)
    clip.json       per-frame labeled people (box, pitch position), track metadata, pitch lines
    impute.json     for every fully observed 3 s window of every player (stride 5 frames):
                    the centred 1 s gap filled by linear / spline / Kalman / BiGRU, with clean
                    and noisy (sigma = 0.25 m) context; stored as cm offsets from ground truth
    results.json    test-set ADE by method, gap length and condition (from results/metrics.csv)
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
WINDOW, GAP, SIGMA = 75, 25, 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-dir", required=True, type=Path)
    ap.add_argument("--results", default=REPO / "results", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--skip-video", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    lab = json.loads((a.clip_dir / "Labels-GameState.json").read_text())
    images = sorted(lab["images"], key=lambda im: im["file_name"])
    frame_of = {str(im["image_id"]): i for i, im in enumerate(images)}
    n = len(images)
    frames = [[] for _ in range(n)]
    lines = [None] * n
    meta = {}
    for o in lab["annotations"]:
        f = frame_of.get(str(o["image_id"]))
        if f is None:
            continue
        if o.get("supercategory") == "pitch":
            lines[f] = [[round(v, 3) for q in pts for v in (q["x"], q["y"])] for pts in o["lines"].values()]
            continue
        if o.get("supercategory") != "object":
            continue
        at = o.get("attributes") or {}
        role = at.get("role")
        if role in (None, "other"):
            continue
        b, p = o["bbox_image"], o.get("bbox_pitch") or {}
        x, y = p.get("x_bottom_middle"), p.get("y_bottom_middle")
        tid = o["track_id"]
        meta.setdefault(tid, {"role": role, "team": at.get("team"), "jersey": at.get("jersey")})
        frames[f].append([tid, int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"]),
                          None if x is None else round(x, 3), None if y is None else round(y, 3)])
    clip = {"info": lab["info"], "fps": 25, "n": n, "width": 1920, "height": 1080,
            "tracks": meta, "frames": frames, "lines": lines}
    (a.out / "clip.json").write_text(json.dumps(clip, separators=(",", ":")))

    # ---- imputation windows
    from imputation.baselines import kalman, linear, spline
    from imputation.data import centred_mask
    params = json.loads((a.results / "baseline_params.json").read_text())
    tr = {}
    for f, fs in enumerate(frames):
        for t, *_r, x, y in fs:
            if x is not None and meta[t]["role"] in ("player", "goalkeeper"):
                tr.setdefault(t, {})[f] = (x, y)
    wins, keys = [], []
    for t, d in tr.items():
        fr = np.array(sorted(d))
        xy = np.array([d[f] for f in fr], np.float32)
        brk = np.flatnonzero(np.diff(fr) != 1) + 1
        for s, e in zip(np.r_[0, brk], np.r_[brk, len(fr)]):
            for k in range(s, e - WINDOW + 1, 5):
                w = xy[k:k + WINDOW]
                if np.linalg.norm(np.diff(w, axis=0), axis=1).max() <= 1.0:
                    wins.append(w)
                    keys.append((int(t), int(fr[k])))
    G = np.stack(wins)
    m = centred_mask(GAP, WINDOW)
    noise = np.random.default_rng(7).normal(0, SIGMA, G.shape).astype(np.float32)
    model = None
    ck = a.results / "gru.pt"
    if ck.exists():
        import torch
        from imputation.gru import GRUImputer, predict_numpy
        st = torch.load(ck, map_location="cpu")
        model = GRUImputer(hidden=st["hidden"])
        model.load_state_dict(st["state_dict"])
    out = {"window": WINDOW, "gap": GAP, "gapStart": int(np.flatnonzero(m)[0]), "sigma": SIGMA,
           "methods": ["linear", "spline", "kalman"] + (["gru"] if model else []), "windows": []}
    preds = {}
    for cond, Y in (("clean", G), ("noisy", G + noise)):
        p = params["0.0" if cond == "clean" else str(SIGMA)]
        preds[cond] = {"linear": linear(Y, m), "spline": spline(Y, m, **p["spline"]),
                       "kalman": kalman(Y, m, **p["kalman"])}
        if model:
            preds[cond]["gru"] = predict_numpy(model, Y, m)
    cm = lambda arr: np.round(arr * 100).astype(int).ravel().tolist()
    for i, (t, f0) in enumerate(keys):
        w = {"t": t, "f0": f0, "noise": cm(noise[i]), "p": {}}
        for cond in preds:
            w["p"][cond] = {k: cm(v[i, m] - G[i, m]) for k, v in preds[cond].items()}
        out["windows"].append(w)
    (a.out / "impute.json").write_text(json.dumps(out, separators=(",", ":")))

    # ---- test results
    rows = [r for r in csv.DictReader(open(a.results / "metrics.csv")) if r["split"] == "test"]
    res = [{"noise": float(r["noise_m"]), "method": r["method"], "gap_s": float(r["gap_s"]),
            "ade_cm": round(float(r["ade_m"]) * 100, 2), "ci_cm": round(float(r["ade_ci95_m"]) * 100, 2),
            "n": int(r["n_windows"])} for r in rows]
    (a.out / "results.json").write_text(json.dumps(res, separators=(",", ":")))

    if not a.skip_video:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "25", "-i",
                        str(a.clip_dir / "img1" / "%06d.jpg"), "-vf", "scale=960:540",
                        "-c:v", "libx264", "-crf", "27", "-preset", "slow", "-g", "5",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
                        str(a.out / "broadcast.mp4")], check=True)
    print(len(keys), "windows;", {p.name: p.stat().st_size for p in a.out.iterdir()})


if __name__ == "__main__":
    main()
