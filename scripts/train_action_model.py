#!/usr/bin/env python3
"""
Train the context-aware action model (src/imputation/action_model.py): will the ball
carrier dribble, pass or lose the ball in the next second, to whom would a pass go, and
which path will they take.

    python scripts/train_action_model.py --states-dir data/processed --out-dir results

Needs game_states_{train,valid,test}.csv.gz from scripts/extract_game_states.py.
Saves results/action_model.pt, results/action_log.csv and results/action_metrics.json
(test split, with simple baselines for comparison).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from imputation.action_model import ActionModel, predict, tensors  # noqa: E402
from imputation.events import ACTIONS, build_samples, load_states  # noqa: E402


def find(d, split):
    for name in (f"game_states_{split}.csv.gz", f"game_states_{split}.csv"):
        if (d / name).exists():
            return d / name
    raise SystemExit(f"missing game_states_{split} in {d}")


def metrics(D, P):
    y, pa = D["act"], P["act"]
    pred = pa.argmax(1)
    out = {"n": int(len(y)), "accuracy": float((pred == y).mean())}
    rec = [float((pred[y == c] == c).mean()) for c in range(3)]
    out["recall"] = dict(zip(ACTIONS, rec))
    out["balanced_accuracy"] = float(np.mean(rec))
    out["majority_accuracy"] = float((y == np.bincount(y).argmax()).mean())
    # pass vs not-pass AUC
    s, t = pa[:, 1], (y == 1).astype(int)
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    npos, nneg = t.sum(), len(t) - t.sum()
    out["pass_auc"] = float((ranks[t == 1].sum() - npos * (npos + 1) / 2) / max(1, npos * nneg))
    # receiver
    m = D["rec"] >= 0
    if m.any():
        pr = P["rec"][m]
        rank = (pr > pr[np.arange(m.sum()), D["rec"][m]][:, None]).sum(1)
        ag = D["ag"][m]
        mates = D["mask"][m] & (ag[..., 4] > 0.5)
        dist = np.where(mates, np.linalg.norm(ag[..., :2], axis=-1), np.inf)
        out["receiver"] = {"n": int(m.sum()), "top1": float((rank == 0).mean()), "top3": float((rank < 3).mean()),
                           "nearest_teammate_top1": float((dist.argmin(1) == D["rec"][m]).mean()),
                           "mean_visible_teammates": float(mates.sum(1).mean())}
    # path
    ok = ~np.isnan(D["fut"]).any((1, 2))
    d = np.linalg.norm(P["paths"][ok] - D["fut"][ok][:, None], axis=-1)
    top = P["pprob"][ok].argmax(1)
    v = (D["car"][ok][:, -1] - D["car"][ok][:, -6]) * 5.0
    cv = v[:, None] * (np.arange(1, 26)[None, :, None] / 25.0)
    out["path"] = {"ade_top": float(d[np.arange(len(top)), top].mean()), "minade3": float(d.mean(-1).min(1).mean()),
                   "fde_top": float(d[np.arange(len(top)), top, -1].mean()),
                   "ade_cv": float(np.linalg.norm(cv - D["fut"][ok], axis=-1).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states-dir", required=True, type=Path)
    ap.add_argument("--out-dir", default="results", type=Path)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    torch.manual_seed(0)
    d = a.states_dir.expanduser()
    data = {s: build_samples(load_states(find(d, s))) for s in ["train", "valid", "test"]}
    for s, D in data.items():
        print(f"[{s}] {len(D['act'])} samples, actions {dict(zip(ACTIONS, np.bincount(D['act'], minlength=3).tolist()))}, "
              f"passes with visible receiver {(D['rec'] >= 0).sum()}")
    Dt = data["train"]
    w = np.bincount(Dt["act"], minlength=3).astype(float)
    cw = torch.tensor((w.sum() / (3 * w)) ** 0.5, dtype=torch.float32)
    model = ActionModel()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-3)
    n = len(Dt["act"])
    steps = n // a.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * steps)
    g = np.random.default_rng(0)
    yact, yrec = torch.from_numpy(Dt["act"]), torch.from_numpy(Dt["rec"])
    fut = torch.from_numpy(Dt["fut"])
    log, best = [], -1
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, ls = time.time(), []
        perm = g.permutation(n)
        for i in range(steps):
            idx = perm[i * a.batch:(i + 1) * a.batch]
            car, ball, ag, mask, absxy = tensors(Dt, idx)
            if g.random() < 0.5:  # mirror across the long axis (attack direction unchanged)
                flip = torch.tensor([1.0, -1.0])
                car, ball, absxy = car * flip, ball * flip, absxy * flip
                ag = ag.clone(); ag[..., [1, 3]] *= -1
                f_ = fut[idx] * flip
            else:
                f_ = fut[idx]
            act, rec, paths, pl = model(car, ball, ag, mask, absxy)
            loss = F.cross_entropy(act, yact[idx], weight=cw)
            r = yrec[idx]
            if (r >= 0).any():
                loss = loss + F.cross_entropy(rec[r >= 0], r[r >= 0])
            ok = ~torch.isnan(f_).any(-1).any(-1)
            if ok.any():
                dd = torch.linalg.norm(paths[ok] - f_[ok][:, None], dim=-1).mean(-1)
                b = dd.argmin(1)
                loss = loss + F.smooth_l1_loss(paths[ok][torch.arange(len(b)), b], f_[ok], beta=0.05) \
                    + 0.1 * F.cross_entropy(pl[ok], b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ls.append(loss.item())
        mv = metrics(data["valid"], predict(model, data["valid"]))
        score = mv["balanced_accuracy"] + mv.get("receiver", {}).get("top1", 0)
        log.append({"epoch": ep, "loss": float(np.mean(ls)), "val_bal_acc": mv["balanced_accuracy"],
                    "val_pass_auc": mv["pass_auc"], "val_rec_top1": mv.get("receiver", {}).get("top1", np.nan),
                    "val_ade_top": mv["path"]["ade_top"]})
        print(f"epoch {ep:2d} loss {np.mean(ls):.3f} val bal-acc {mv['balanced_accuracy']:.3f} pass-AUC {mv['pass_auc']:.3f} "
              f"rec-top1 {mv.get('receiver', {}).get('top1', float('nan')):.3f} ADE {mv['path']['ade_top']:.3f} ({time.time() - t0:.0f}s)",
              flush=True)
        if score > best:
            best = score
            torch.save({"state_dict": model.state_dict()}, a.out_dir / "action_model.pt")
    pd.DataFrame(log).to_csv(a.out_dir / "action_log.csv", index=False)
    model.load_state_dict(torch.load(a.out_dir / "action_model.pt")["state_dict"])
    res = {s: metrics(data[s], predict(model, data[s])) for s in ["valid", "test"]}
    (a.out_dir / "action_metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["test"], indent=2))


if __name__ == "__main__":
    main()
