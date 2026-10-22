#!/usr/bin/env python3
"""
Train the bidirectional GRU imputer (CPU is fine, ~1-2 min per epoch on a laptop).

    python scripts/train_gru.py --processed-dir ~/data/processed --out-dir results

Training samples: overlapping train windows (stride 5) with a random hidden block
(1..25 frames, at least 5 observed frames on each side), random context noise
(0..0.4 m) and random pitch flips. Loss: Smooth L1 on hidden frames, in metres.
Model selection: mean valid ADE over gap lengths and noise levels.

Outputs (in --out-dir): gru.pt, gru_loss.csv, and rows with method = "gru" appended to
metrics.csv / per_window_ade.npz (run scripts/run_baselines.py first).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from imputation.data import GAP_LENGTHS, WINDOW, load_runs, make_windows  # noqa: E402
from imputation.evaluate import evaluate  # noqa: E402
from imputation.gru import GRUImputer, linear_fill, predict_numpy  # noqa: E402

SPLIT_SEED = {"train": 0, "valid": 1, "test": 2}
MIN_CONTEXT = 5


def sample_batch(W: torch.Tensor, max_noise: float, gen: torch.Generator):
    B, T, _ = W.shape
    L = torch.randint(1, max(GAP_LENGTHS) + 1, (B,), generator=gen)
    hi = T - MIN_CONTEXT - L  # exclusive upper bound for start
    start = MIN_CONTEXT + (torch.rand(B, generator=gen) * (hi - MIN_CONTEXT + 1).float()).long()
    flip = torch.where(torch.rand(B, 2, generator=gen) < 0.5, -1.0, 1.0)[:, None, :]
    gt = W * flip
    sigma = torch.rand(B, 1, 1, generator=gen) * max_noise
    noisy = gt + sigma * torch.randn(gt.shape, generator=gen)
    filled, hidden = linear_fill(noisy, start, L)
    return filled, hidden, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", required=True, type=Path)
    ap.add_argument("--out-dir", default="results", type=Path)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--max-noise", type=float, default=0.4)
    ap.add_argument("--eval-noise", nargs="+", type=float, default=[0.0, 0.25])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    pdir, out = args.processed_dir.expanduser(), args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    train = make_windows(load_runs(str(pdir / "trajectories_train.csv")), WINDOW, stride=5)
    evalw = {s: make_windows(load_runs(str(pdir / f"trajectories_{s}.csv")), WINDOW, stride=WINDOW)
             for s in ["valid", "test"]}
    print(f"train windows {len(train)}, valid {len(evalw['valid'])}, test {len(evalw['test'])}")
    Wt = torch.from_numpy(train)

    model = GRUImputer(hidden=args.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = len(Wt) // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * steps_per_epoch)
    gen = torch.Generator().manual_seed(args.seed)

    def val_score(split):
        f = lambda Y, m: predict_numpy(model, Y, m)
        return {s: float(np.mean([r["ade_m"] for r in evaluate(f, evalw[split], s, SPLIT_SEED[split])]))
                for s in args.eval_noise}

    log, best = [], (np.inf, None)
    v0 = val_score("valid")  # epoch 0 == linear baseline (zero-initialised output)
    log.append({"epoch": 0, "train_loss": np.nan, **{f"val_ade_noise{s}": v for s, v in v0.items()}})
    print(f"epoch 0 (= linear) val ADE {v0}")
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        perm = torch.randperm(len(Wt), generator=gen)
        for i in range(steps_per_epoch):
            filled, hidden, gt = sample_batch(Wt[perm[i * args.batch:(i + 1) * args.batch]], args.max_noise, gen)
            pred = model(filled, hidden)
            loss = F.smooth_l1_loss(pred[hidden], gt[hidden], beta=0.05)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        v = val_score("valid")
        log.append({"epoch": ep, "train_loss": float(np.mean(losses)),
                    **{f"val_ade_noise{s}": x for s, x in v.items()}})
        score = float(np.mean(list(v.values())))
        if score < best[0]:
            best = (score, ep)
            torch.save({"state_dict": model.state_dict(), "hidden": args.hidden}, out / "gru.pt")
        print(f"epoch {ep:2d} loss {np.mean(losses):.4f} val ADE {v} ({time.time() - t0:.0f}s)")

    pd.DataFrame(log).to_csv(out / "gru_loss.csv", index=False)
    print(f"best epoch {best[1]} (mean val ADE {best[0]:.4f} m)")

    # final evaluation with the best checkpoint
    model.load_state_dict(torch.load(out / "gru.pt")["state_dict"])
    f = lambda Y, m: predict_numpy(model, Y, m)
    rows, raw = [], {}
    for sigma in args.eval_noise:
        for split in ["valid", "test"]:
            for r in evaluate(f, evalw[split], sigma, SPLIT_SEED[split]):
                raw[f"{split}|{sigma}|gru|{r['gap_frames']}"] = r.pop("_per_window")
                rows.append({"split": split, "noise_m": sigma, "method": "gru", **r})
    mpath, npath = out / "metrics.csv", out / "per_window_ade.npz"
    df = pd.DataFrame(rows)
    if mpath.exists():
        old = pd.read_csv(mpath)
        df = pd.concat([old[old["method"] != "gru"], df])
    df.to_csv(mpath, index=False)
    if npath.exists():
        prev = dict(np.load(npath))
        prev = {k: v for k, v in prev.items() if "|gru|" not in k}
        raw = {**prev, **raw}
    np.savez_compressed(npath, **raw)
    view = df[df.split == "test"].pivot_table(index=["noise_m", "method"], columns="gap_frames", values="ade_m")
    print("\nTest ADE (m):\n", view.round(3).to_string())


if __name__ == "__main__":
    main()
