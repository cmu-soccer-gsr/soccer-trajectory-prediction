#!/usr/bin/env python3
"""
Train the 1 s trajectory forecaster (src/imputation/forecast.py) on the GSR train split.

    python scripts/train_forecaster.py --processed-dir data/processed --out-dir results

Windows: 3 s of fully observed positions (2 s past + 1 s future), stride 5 (train) and
non-overlapping (valid). Reports valid ADE / FDE of the most likely candidate and the
best of K candidates (minADE_K), next to constant velocity. Saves results/forecaster.pt
and results/forecaster_log.csv.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from imputation.data import load_runs, make_windows  # noqa: E402
from imputation.forecast import FUTURE, PAST, Forecaster, predict, wta_loss  # noqa: E402


def evaluate(model, W):
    past, fut = W[:, :PAST], W[:, PAST:]
    rel = fut - past[:, -1:]
    paths, prob = predict(model, past)
    d = np.linalg.norm(paths - rel[:, None], axis=-1)  # (N, K, T)
    top = prob.argmax(1)
    v = (past[:, -1] - past[:, -6]) * 5.0
    cv = v[:, None] * (np.arange(1, FUTURE + 1)[None, :, None] / 25.0)
    dcv = np.linalg.norm(cv - rel, axis=-1)
    return {"ade_top": d[np.arange(len(top)), top].mean(), "fde_top": d[np.arange(len(top)), top, -1].mean(),
            "minade_k": d.mean(-1).min(1).mean(), "ade_cv": dcv.mean(), "fde_cv": dcv[:, -1].mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", required=True, type=Path)
    ap.add_argument("--out-dir", default="results", type=Path)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    a = ap.parse_args()
    torch.manual_seed(0)
    pdir = a.processed_dir.expanduser()
    Wtr = make_windows(load_runs(str(pdir / "trajectories_train.csv")), PAST + FUTURE, stride=5)
    Wva = make_windows(load_runs(str(pdir / "trajectories_valid.csv")), PAST + FUTURE, stride=PAST + FUTURE)
    print(f"train {len(Wtr)} windows, valid {len(Wva)}")
    Wt = torch.from_numpy(Wtr)
    model = Forecaster()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = len(Wt) // a.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * steps)
    g = torch.Generator().manual_seed(0)
    log, best = [], np.inf
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, ls = time.time(), []
        perm = torch.randperm(len(Wt), generator=g)
        for i in range(steps):
            w = Wt[perm[i * a.batch:(i + 1) * a.batch]]
            flip = torch.where(torch.rand(len(w), 1, 2, generator=g) < 0.5, -1.0, 1.0)
            w = w * flip
            past, fut = w[:, :PAST], w[:, PAST:]
            paths, logits = model(past)
            loss, _ = wta_loss(paths, logits, fut - past[:, -1:])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ls.append(loss.item())
        m = evaluate(model, Wva)
        log.append({"epoch": ep, "train_loss": float(np.mean(ls)), **{k: float(v) for k, v in m.items()}})
        print(f"epoch {ep} loss {np.mean(ls):.4f} " + " ".join(f"{k} {v:.3f}" for k, v in m.items())
              + f" ({time.time() - t0:.0f}s)", flush=True)
        if m["ade_top"] < best:
            best = m["ade_top"]
            torch.save({"state_dict": model.state_dict()}, a.out_dir / "forecaster.pt")
    pd.DataFrame(log).to_csv(a.out_dir / "forecaster_log.csv", index=False)


if __name__ == "__main__":
    main()
