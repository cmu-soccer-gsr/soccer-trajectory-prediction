#!/usr/bin/env python3
"""
Combine the pipeline-noise re-run into one table and one figure.

    python scripts/summarize_pipeline_eval.py --processed-dir ~/data/processed \
        --eval-dir results/pipeline_eval

Reads base/, gru_old/, gru_pn/ metrics.csv under --eval-dir, adds a bias-corrected
Kalman (observations shifted by the measured mean offset before filtering, same tuned
q and r), and writes summary.csv (test ADE averaged over gap lengths) and
pipeline_noise_sweep.png.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from imputation.baselines import kalman  # noqa: E402
from imputation.data import WINDOW, load_runs, make_windows  # noqa: E402
from imputation.evaluate import evaluate  # noqa: E402
from imputation.noise import load_model, parse  # noqa: E402

NAMES = {"linear": "Linear", "spline": "Spline", "kalman": "Kalman", "kalman_debiased": "Kalman, bias-corrected",
         "gru": "GRU (trained on iid noise)", "gru_pn": "GRU (trained on pipeline noise)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", required=True, type=Path)
    ap.add_argument("--eval-dir", default="results/pipeline_eval", type=Path)
    a = ap.parse_args()
    d = pd.concat([pd.read_csv(a.eval_dir / s / "metrics.csv") for s in ["base", "gru_old", "gru_pn"]])
    d = d[d.split == "test"]
    params = json.loads((a.eval_dir / "base" / "baseline_params.json").read_text())
    W = make_windows(load_runs(str(a.processed_dir.expanduser() / "trajectories_test.csv")), WINDOW, WINDOW)
    mu = np.array(load_model()["offset_mean"], np.float32)
    rows = []
    for spec in [s for s in params if s.startswith("pipeline")]:
        sc = parse(spec)[1]
        for r in evaluate(lambda Y, m: kalman(Y - sc * mu, m, **params[spec]["kalman"]), W, spec, 2):
            r.pop("_per_window")
            rows.append({"split": "test", "noise_m": spec, "method": "kalman_debiased", **r})
    d = pd.concat([d, pd.DataFrame(rows)])
    d.to_csv(a.eval_dir / "metrics_test_all.csv", index=False)

    s = d.groupby(["noise_m", "method"])[["ade_m", "ade_track_m"]].mean().reset_index()
    s.to_csv(a.eval_dir / "summary.csv", index=False)
    print(s.pivot(index="method", columns="noise_m", values="ade_m").round(3).to_string())

    scales = {"pipeline:0.5": 0.5, "pipeline": 1.0, "pipeline:1.5": 1.5}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=200)
    for ax, col, title in [(axes[0], "ade_m", "Error vs clean ground truth"),
                           (axes[1], "ade_track_m", "Error vs ground truth + window offset")]:
        for m in ["linear", "spline", "kalman", "kalman_debiased", "gru", "gru_pn"]:
            v = s[(s.method == m) & s.noise_m.isin(scales)]
            if not len(v):
                continue
            x = v.noise_m.map(scales)
            o = np.argsort(x.to_numpy())
            ax.plot(x.to_numpy()[o], v[col].to_numpy()[o], marker="o", label=NAMES[m],
                    ls="--" if m == "kalman_debiased" else "-")
        ax.set_xlabel("Scale of measured pipeline noise"); ax.set_ylabel("Test ADE (m), mean over 0.2-1 s gaps")
        ax.set_title(title, fontsize=10); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(a.eval_dir / "pipeline_noise_sweep.png")
    print("wrote", a.eval_dir / "summary.csv", a.eval_dir / "pipeline_noise_sweep.png")


if __name__ == "__main__":
    main()
