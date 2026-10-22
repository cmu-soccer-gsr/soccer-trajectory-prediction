#!/usr/bin/env python3
"""
Final tables and figures from results/ (run run_baselines.py and train_gru.py first).

    python scripts/make_figures.py --processed-dir ~/data/processed --results results

Writes to results/figures/:
    error_vs_gap.png      test ADE vs gap length, clean vs noisy context
    gru_loss.png          GRU training loss and validation ADE per epoch
    example.png           one test window: context, hidden ground truth, predictions
    results_table.md/.csv test ADE (cm) per method and gap length
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from imputation.baselines import kalman, linear, spline  # noqa: E402
from imputation.data import WINDOW, centred_mask, load_runs, make_windows  # noqa: E402
from imputation.evaluate import context_noise  # noqa: E402

# fixed categorical order (colour follows the method, never its rank)
STYLE = {
    "gru":    dict(color="#2a78d6", marker="o", label="BiGRU (ours)"),
    "kalman": dict(color="#eb6834", marker="s", label="Kalman (CV + RTS)"),
    "spline": dict(color="#1baf7a", marker="^", label="Cubic P-spline"),
    "linear": dict(color="#8a8984", marker="D", label="Linear"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def style_ax(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def error_vs_gap(df, out):
    noises = sorted(df.noise_m.unique())
    fig, axes = plt.subplots(1, len(noises), figsize=(5.2 * len(noises), 4), dpi=200)
    axes = np.atleast_1d(axes)
    for ax, sigma in zip(axes, noises):
        d = df[(df.split == "test") & (df.noise_m == sigma)]
        for m, st in STYLE.items():
            r = d[d.method == m].sort_values("gap_frames")
            if r.empty:
                continue
            x, y, ci = r.gap_s, r.ade_m * 100, r.ade_ci95_m * 100
            ax.plot(x, y, color=st["color"], marker=st["marker"], ms=6, lw=2, label=st["label"])
            ax.fill_between(x, y - ci, y + ci, color=st["color"], alpha=0.15, lw=0)
        ax.set_title("Clean context" if sigma == 0 else f"Noisy context (σ = {sigma:g} m)",
                     color=INK, fontsize=11)
        ax.set_xlabel("Gap length (s)", color=MUTED)
        ax.set_ylabel("Mean displacement error (cm)", color=MUTED)
        ax.set_ylim(bottom=0)
        style_ax(ax)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Imputation error vs gap length (test split, 95% CI)", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "error_vs_gap.png")
    plt.close(fig)


def loss_curve(loss, df, out):
    cols = [c for c in loss.columns if c.startswith("val_ade_noise")]
    fig, axes = plt.subplots(1, 1 + len(cols), figsize=(4.2 * (1 + len(cols)), 3.6), dpi=200)
    tr = loss.dropna(subset=["train_loss"])
    axes[0].plot(tr.epoch, tr.train_loss, color=STYLE["gru"]["color"], lw=2, marker="o", ms=4)
    axes[0].set_title("Training loss (Smooth L1)", color=INK, fontsize=11)
    axes[0].set_xlabel("Epoch", color=MUTED)
    for ax, c in zip(axes[1:], cols):
        sigma = float(c.replace("val_ade_noise", ""))
        ax.plot(loss.epoch, loss[c] * 100, color=STYLE["gru"]["color"], lw=2, marker="o", ms=4,
                label="BiGRU (epoch 0 = linear)")
        v = df[(df.split == "valid") & (df.noise_m == sigma)]
        for m in ["kalman", "spline"]:
            ref = v[v.method == m].ade_m.mean() * 100
            if np.isfinite(ref):
                ax.axhline(ref, color=STYLE[m]["color"], lw=1.5, ls="--", label=STYLE[m]["label"])
        ax.set_title(f"Valid ADE, {'clean' if sigma == 0 else f'σ = {sigma:g} m'}", color=INK, fontsize=11)
        ax.set_xlabel("Epoch", color=MUTED)
        ax.set_ylabel("cm (mean over gap lengths)", color=MUTED)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=8)
    for ax in axes:
        style_ax(ax)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out / "gru_loss.png")
    plt.close(fig)


def example(pdir, res, out, sigma=0.25, gap=25):
    W = make_windows(load_runs(str(pdir / "trajectories_test.csv")), WINDOW, WINDOW)
    noisy = W + context_noise(W.shape, sigma, 2)  # same noise as the test evaluation
    m = centred_mask(gap)
    params = json.loads((res / "baseline_params.json").read_text())[str(sigma)]
    preds = {"linear": linear(noisy, m), "spline": spline(noisy, m, **params["spline"]),
             "kalman": kalman(noisy, m, **params["kalman"])}
    if (res / "gru.pt").exists():
        import torch
        from imputation.gru import GRUImputer, predict_numpy
        ck = torch.load(res / "gru.pt")
        model = GRUImputer(hidden=ck["hidden"])
        model.load_state_dict(ck["state_dict"])
        preds["gru"] = predict_numpy(model, noisy, m)
    # pick a representative moving player with a curved path in the gap: among windows
    # with > 2 m displacement, take the one at the 75th percentile of curvature
    # (a typical turn, not a pathological outlier)
    gt_gap = W[:, m]
    chord = np.linspace(gt_gap[:, 0], gt_gap[:, -1], gap, axis=1)
    curvature = np.linalg.norm(gt_gap - chord, axis=-1).max(1)
    speed = np.linalg.norm(gt_gap[:, -1] - gt_gap[:, 0], axis=-1)
    cand = np.flatnonzero(speed > 2)
    target = np.quantile(curvature[cand], 0.75)
    i = int(cand[np.argmin(np.abs(curvature[cand] - target))])

    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    ax.plot(noisy[i, ~m, 0], noisy[i, ~m, 1], ".", color=MUTED, ms=4, label="Observed context (noisy)")
    ax.plot(W[i, m, 0], W[i, m, 1], color=INK, lw=2.5, label="Hidden ground truth")
    for k, p in preds.items():
        err = np.linalg.norm(p[i, m] - W[i, m], axis=-1).mean() * 100
        st = STYLE[k]
        ax.plot(p[i, m, 0], p[i, m, 1], color=st["color"], lw=2, ls="--", label=f"{st['label']} ({err:.0f} cm)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Pitch x (m)", color=MUTED)
    ax.set_ylabel("Pitch y (m)", color=MUTED)
    ax.set_title(f"Example: {gap / 25:g} s gap, σ = {sigma:g} m context noise", color=INK, fontsize=11)
    style_ax(ax)
    ax.grid(color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out / "example.png")
    plt.close(fig)


def table(df, out):
    d = df[df.split == "test"].copy()
    d["ade_cm"] = d.ade_m * 100
    order = [m for m in STYLE if m in d.method.unique()][::-1]  # linear first
    lines, csv_rows = [], []
    for sigma in sorted(d.noise_m.unique()):
        t = d[d.noise_m == sigma].pivot_table(index="method", columns="gap_s", values="ade_cm").reindex(order)
        t["mean"] = t.mean(axis=1)
        t["vs linear"] = (t["mean"] / t.loc["linear", "mean"] - 1) * 100
        title = "Clean context" if sigma == 0 else f"Noisy context (σ = {sigma:g} m)"
        lines += [f"### {title}", "", "| Method | " + " | ".join(f"{c:g} s" for c in t.columns[:-2])
                  + " | Mean | vs linear |", "|---" * (len(t.columns) + 1) + "|"]
        best = t.drop(columns="vs linear").idxmin()
        for m, row in t.iterrows():
            cells = []
            for c in t.columns[:-1]:
                s = f"{row[c]:.1f}"
                cells.append(f"**{s}**" if best[c] == m else s)
            vs = "" if m == "linear" else f"{row['vs linear']:+.0f}%"
            lines.append(f"| {STYLE[m]['label']} | " + " | ".join(cells) + f" | {vs} |")
            csv_rows.append({"noise_m": sigma, "method": m, **{f"{c:g}" if isinstance(c, float) else c: row[c]
                                                                for c in t.columns}})
        lines.append("")
    n = int(d.n_windows.iloc[0])
    lines.append(f"Test ADE in cm (mean displacement over hidden frames), {n} windows of 3 s, "
                 "centred gap. Bold = best per column.")
    (out / "results_table.md").write_text("\n".join(lines))
    pd.DataFrame(csv_rows).to_csv(out / "results_table.csv", index=False)
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", required=True, type=Path)
    ap.add_argument("--results", default="results", type=Path)
    args = ap.parse_args()
    res, pdir = args.results, args.processed_dir.expanduser()
    out = res / "figures"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(res / "metrics.csv")
    error_vs_gap(df, out)
    if (res / "gru_loss.csv").exists():
        loss_curve(pd.read_csv(res / "gru_loss.csv"), df, out)
    example(pdir, res, out)
    table(df, out)
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
