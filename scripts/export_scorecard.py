#!/usr/bin/env python3
"""
Test-set scorecard for the Live Board (writes live/scorecard.json).

    python scripts/export_scorecard.py --cache-dir results/skillcorner_cache --out live/scorecard.json

Needs the cached test samples from train_possession_model.py and results/possession_model.pt.
Contents: headline metrics with baselines, receiver accuracy by time before the pass,
calibration of "pass arrives" (model and SkillCorner xPass), the outcome confusion matrix,
and a per-match breakdown.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from imputation.possession_model import PossessionModel, predict  # noqa: E402
from train_possession_model import auc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--results", default=REPO / "results", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    D = pickle.loads((a.cache_dir / "samples_test.pkl").read_bytes())
    model = PossessionModel()
    model.load_state_dict(torch.load(a.results / "possession_model.pt", map_location="cpu", weights_only=False)["state_dict"])
    P = predict(model, D)
    meta = D["meta"]
    mid = np.array([m[0] for m in meta]); t = np.array([m[1] for m in meta]); fe = np.array([m[5] for m in meta])
    sc = np.array([m[6] for m in meta]); xp = np.array([m[7] for m in meta])
    has = D["rec"] >= 0
    rec = np.maximum(D["rec"], 0)
    rank = (P["rec"] > P["rec"][np.arange(len(rec)), rec][:, None]).sum(1)
    mates = D["mask"] & (D["ag"][..., 4] > 0.5)
    nearest = np.where(mates, np.linalg.norm(D["ag"][..., :2], axis=-1), np.inf).argmin(1)
    tgt_pid = D["pids_of"][np.arange(len(rec)), rec]
    sc_hit = (sc == tgt_pid)

    # receiver accuracy by time before the pass (0 to 2 s)
    by_time = []
    for d in range(0, 21):
        s = has & (fe - t == d)
        s2 = s & (sc >= 0)
        if s.sum() < 30:
            continue
        by_time.append({"s": d / 10, "n": int(s.sum()), "model": float((rank[s] == 0).mean()), "model_top3": float((rank[s] < 3).mean()),
                        "nearest": float((nearest[s] == D["rec"][s]).mean()), "skillcorner": float(sc_hit[s2].mean()) if s2.any() else None})

    # calibration of "pass arrives" at the release frame
    rel = has & (t == fe) & (D["success"] >= 0)
    pa = P["arrive"][np.flatnonzero(rel), D["rec"][rel]]
    ya = D["success"][rel]
    xr = xp[rel]
    def calib(p, y):
        out = []
        for lo in np.arange(0, 1, 0.1):
            s = (p >= lo) & (p < lo + 0.1 if lo < 0.9 else p <= 1.0)
            if s.sum() >= 10:
                out.append({"lo": round(float(lo), 1), "pred": float(p[s].mean()), "actual": float(y[s].mean()), "n": int(s.sum())})
        return out
    ok = ~np.isnan(xr)
    calibration = {"model": calib(pa, ya), "skillcorner": calib(xr[ok], ya[ok]), "n": int(rel.sum()), "base_rate": float(ya.mean())}

    # outcome confusion matrix (per frame) and per possession at its first frame
    y = D["outcome"]; pred = P["outcome"].argmax(1)
    conf = [[int(((y == i) & (pred == j)).sum()) for j in range(3)] for i in range(3)]

    # per match
    per_match = []
    for m in sorted(set(mid)):
        s = mid == m
        sr = rel & s
        rr = has & s & (t == fe)
        yy, pp = y[s], pred[s]
        bal = np.mean([(pp[yy == c] == c).mean() for c in range(3) if (yy == c).any()])
        per_match.append({"match": m, "frames": int(s.sum()), "passes": int(rr.sum()), "receiver_top1": float((rank[rr] == 0).mean()),
                          "arrive_auc": auc(P["arrive"][np.flatnonzero(sr), D["rec"][sr]], D["success"][sr]), "balanced_accuracy": float(bal)})
    metrics = json.loads((a.results / "possession_metrics.json").read_text())
    log = (a.results / "possession_log.csv").read_text().strip().splitlines()
    out = {"metrics": metrics["test"], "split": metrics["split"], "receiver_by_time": by_time, "calibration": calibration,
           "confusion": {"labels": ["Pass", "Loss", "Shot"], "counts": conf}, "per_match": per_match,
           "training": {"params": int(sum(p.numel() for p in model.parameters())), "train_frames": 215619,
                        "log": [dict(zip(log[0].split(","), map(float, r.split(",")))) for r in log[1:]]}}
    a.out.write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
