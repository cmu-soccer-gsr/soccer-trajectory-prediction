#!/usr/bin/env python3
"""
Masking protocol + linear / cubic P-spline / Kalman baselines.

    python scripts/run_baselines.py --processed-dir ~/data/processed --out-dir results

1. Builds windows from trajectories_{train,valid,test}.csv (see src/imputation/data.py).
2. Tunes spline (lambda) and Kalman (q, r) on train windows, separately for each
   context-noise level.
3. Evaluates every method on valid and test for gap lengths 5..25 frames.

Outputs (in --out-dir):
    baseline_params.json      tuned hyper-parameters
    metrics.csv               one row per split x noise x method x gap length
    per_window_ade.npz        per-window errors (for paired comparisons / figures)
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from imputation.baselines import kalman, linear, spline  # noqa: E402
from imputation.data import GAP_LENGTHS, WINDOW, load_runs, make_windows  # noqa: E402
from imputation.evaluate import evaluate  # noqa: E402

SPLIT_SEED = {"train": 0, "valid": 1, "test": 2}


def mean_ade(rows):
    return float(np.mean([r["ade_m"] for r in rows]))


def tune(train_w, sigma):
    best = {}
    # spline: smoothing penalty
    res = []
    for lam in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]:
        f = lambda Y, m, lam=lam: spline(Y, m, lam=lam)
        res.append((mean_ade(evaluate(f, train_w, sigma, SPLIT_SEED["train"])), {"lam": lam}))
    best["spline"] = min(res, key=lambda x: x[0])[1]
    # kalman: process / measurement noise
    res = []
    for q, r in itertools.product([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
                                  [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1]):
        f = lambda Y, m, q=q, r=r: kalman(Y, m, q=q, r=r)
        res.append((mean_ade(evaluate(f, train_w, sigma, SPLIT_SEED["train"])), {"q": q, "r": r}))
    best["kalman"] = min(res, key=lambda x: x[0])[1]
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", required=True, type=Path)
    ap.add_argument("--out-dir", default="results", type=Path)
    ap.add_argument("--noise", nargs="+", type=float, default=[0.0, 0.25],
                    help="Std (m) of Gaussian noise on observed context frames")
    ap.add_argument("--tune-windows", type=int, default=20000)
    args = ap.parse_args()
    pdir, out = args.processed_dir.expanduser(), args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    windows = {}
    for split in ["train", "valid", "test"]:
        runs = load_runs(str(pdir / f"trajectories_{split}.csv"))
        windows[split] = make_windows(runs, WINDOW, stride=WINDOW)
        print(f"[{split}] {len(runs)} fully observed runs -> {len(windows[split])} eval windows")

    rng = np.random.default_rng(0)
    tw = windows["train"]
    if len(tw) > args.tune_windows:
        tw = tw[rng.choice(len(tw), args.tune_windows, replace=False)]

    params, rows, raw = {}, [], {}
    for sigma in args.noise:
        p = tune(tw, sigma)
        params[str(sigma)] = p
        print(f"[tune sigma={sigma}] {p}")
        methods = {
            "linear": linear,
            "spline": lambda Y, m, p=p: spline(Y, m, **p["spline"]),
            "kalman": lambda Y, m, p=p: kalman(Y, m, **p["kalman"]),
        }
        for split in ["valid", "test"]:
            for name, f in methods.items():
                for r in evaluate(f, windows[split], sigma, SPLIT_SEED[split]):
                    raw[f"{split}|{sigma}|{name}|{r['gap_frames']}"] = r.pop("_per_window")
                    rows.append({"split": split, "noise_m": sigma, "method": name, **r})

    (out / "baseline_params.json").write_text(json.dumps(params, indent=2))
    df = pd.DataFrame(rows)
    mpath = out / "metrics.csv"
    if mpath.exists():  # keep rows from other methods (e.g. gru)
        old = pd.read_csv(mpath)
        df = pd.concat([old[~old["method"].isin(df["method"].unique())], df])
    df.to_csv(mpath, index=False)
    np.savez_compressed(out / "per_window_ade.npz", **raw)

    view = df[df.split == "test"].pivot_table(index=["noise_m", "method"], columns="gap_frames", values="ade_m")
    print("\nTest ADE (m):\n", view.round(3).to_string())


if __name__ == "__main__":
    main()
