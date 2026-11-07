#!/usr/bin/env python3
"""
Train and evaluate the possession model on SkillCorner open data (20 full A-League matches).

    git clone https://github.com/SkillCorner/opendata  (tracking files are Git LFS)
    python scripts/train_possession_model.py --data-dir opendata/data/matches --out-dir results

Matches are split by match (14 train / 3 valid / 3 test, fixed seed), so no test match is
seen in training. Writes results/possession_model.pt, possession_log.csv and
possession_metrics.json (test matches, with baselines including SkillCorner's own
passing-option model).
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from imputation.possession_model import PossessionModel, predict, tensors  # noqa: E402
from imputation.skillcorner import OUTCOMES, build_samples, concat, load_match  # noqa: E402


def auc(score, y):
    y = np.asarray(y).astype(int)
    if y.min() == y.max():
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    r = np.empty(len(score)); r[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    s = np.asarray(score)[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    npos = y.sum()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos)))


def metrics(D, P):
    y = D["outcome"]
    pred = P["outcome"].argmax(1)
    rec = [float((pred[y == c] == c).mean()) for c in range(3)]
    out = {"frames": int(len(y)), "possessions": int(len({m[3] for m in D["meta"]})),
           "outcome": {"balanced_accuracy": float(np.mean(rec)), "recall": dict(zip(OUTCOMES, rec)),
                       "auc": {OUTCOMES[c]: auc(P["outcome"][:, c], y == c) for c in range(3)},
                       "always_pass_accuracy": float((y == 0).mean()), "accuracy": float((pred == y).mean())},
           "release_within_1s_auc": auc(P["soon"], D["soon"])}
    t = np.array([m[1] for m in D["meta"]]); fs = np.array([m[4] for m in D["meta"]]); fe = np.array([m[5] for m in D["meta"]])
    sc = np.array([m[6] for m in D["meta"]])
    tgt_pid = np.full(len(y), -1)
    has = D["rec"] >= 0
    tgt_pid[has] = D["idx"][has, D["rec"][has]]
    ag = D["ag"]
    mates = D["mask"] & (ag[..., 4] > 0.5)
    dist = np.where(mates, np.linalg.norm(ag[..., :2], axis=-1), np.inf)
    nearest = dist.argmin(1)
    rk = (P["rec"] > P["rec"][np.arange(len(y)), np.maximum(D["rec"], 0)][:, None]).sum(1)
    res = {}
    for name, sel in [("all_frames", has), ("possession_start", has & (t == fs)), ("release_frame", has & (t == fe))]:
        e = {"n": int(sel.sum()), "top1": float((rk[sel] == 0).mean()), "top3": float((rk[sel] < 3).mean()),
             "nearest_teammate_top1": float((nearest[sel] == D["rec"][sel]).mean()),
             "random_top1": float((1 / mates[sel].sum(1)).mean())}
        s2 = sel & (sc >= 0)
        # SkillCorner's highest-scoring passing option for the possession (their own model;
        # one guess per possession, so it is the same for every frame of that possession)
        ids = [sc[i] == D["pids_of"][i][D["rec"][i]] for i in np.flatnonzero(s2)]
        e["skillcorner_option_n"] = int(s2.sum())
        e["skillcorner_option_top1"] = float(np.mean(ids)) if ids else float("nan")
        e["model_top1_same_frames"] = float((rk[s2] == 0).mean()) if s2.any() else float("nan")
        res[name] = e
    out["receiver"] = res
    # pass success at the release frame, true target
    sel = has & (t == fe) & (D["success"] >= 0)
    arr = P["arrive"][np.flatnonzero(sel), D["rec"][sel]]
    xp = np.array([D["meta"][i][7] for i in np.flatnonzero(sel)])
    ok = ~np.isnan(xp)
    out["pass_arrives"] = {"n": int(sel.sum()), "completion_rate": float(D["success"][sel].mean()),
                           "model_auc": auc(arr, D["success"][sel]),
                           "n_with_skillcorner_xpass": int(ok.sum()),
                           "model_auc_same_passes": auc(arr[ok], D["success"][sel][ok]),
                           "skillcorner_xpass_auc": auc(xp[ok], D["success"][sel][ok])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out-dir", default="results", type=Path)
    ap.add_argument("--cache-dir", default=None, type=Path)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    torch.manual_seed(0)
    cache = a.cache_dir or (a.out_dir / "skillcorner_cache")
    mids = sorted(p.name for p in a.data_dir.iterdir() if (p / f"{p.name}_tracking_extrapolated.jsonl").exists())
    perm = np.random.default_rng(0).permutation(mids)
    perm = [str(x) for x in perm]
    split = {"train": sorted(perm[:-6]), "valid": sorted(perm[-6:-3]), "test": sorted(perm[-3:])}
    print({k: list(v) for k, v in split.items()})
    data = {}
    for s, ms in split.items():
        pk = cache / f"samples_{s}.pkl"
        if pk.exists():
            data[s] = pickle.loads(pk.read_bytes())
            continue
        Ds = []
        for mid in ms:
            t0 = time.time()
            m = load_match(a.data_dir / mid, cache)
            D = build_samples(m)
            D["pids_of"] = m.pids[np.where(D["idx"] >= 0, D["idx"], 0)]
            Ds.append(D)
            print(f"  {s} {mid}: {len(D['outcome'])} frames ({time.time() - t0:.0f}s)", flush=True)
        data[s] = concat(Ds)
        pk.parent.mkdir(parents=True, exist_ok=True)
        pk.write_bytes(pickle.dumps(data[s]))
    for s, D in data.items():
        print(f"[{s}] {len(D['outcome'])} frames, outcomes {dict(zip(OUTCOMES, np.bincount(D['outcome'], minlength=3).tolist()))}")
    Dt = data["train"]
    w = np.bincount(Dt["outcome"], minlength=3).astype(float)
    cw = torch.tensor((w.sum() / (3 * w)) ** 0.5, dtype=torch.float32)
    model = PossessionModel()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-3)
    n = len(Dt["outcome"]); steps = n // a.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * steps)
    g = np.random.default_rng(0)
    Y = {k: torch.from_numpy(Dt[k]) for k in ["outcome", "soon", "rec", "success"]}
    log, best = [], -1
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); ls = []
        perm_ = g.permutation(n)
        for i in range(steps):
            idx = perm_[i * a.batch:(i + 1) * a.batch]
            car, ball, ag, mask, absxy = tensors(Dt, idx)
            if g.random() < 0.5:  # mirror across the long axis
                car = car * torch.tensor([1.0, -1.0]); absxy = absxy * torch.tensor([1.0, -1.0])
                ball = ball.clone(); ball[..., 1] *= -1
                ag = ag.clone(); ag[..., [1, 3]] *= -1
            o, s, r, arr = model(car, ball, ag, mask, absxy)
            loss = F.cross_entropy(o, Y["outcome"][idx], weight=cw) + 0.5 * F.binary_cross_entropy_with_logits(s, Y["soon"][idx].float())
            rr = Y["rec"][idx]; m_ = rr >= 0
            if m_.any():
                loss = loss + F.cross_entropy(r[m_], rr[m_])
                su = Y["success"][idx]; m2 = m_ & (su >= 0)
                if m2.any():
                    loss = loss + 0.5 * F.binary_cross_entropy_with_logits(arr[m2, rr[m2]], su[m2].float())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); ls.append(loss.item())
        mv = metrics(data["valid"], predict(model, data["valid"]))
        score = mv["outcome"]["balanced_accuracy"] + mv["receiver"]["all_frames"]["top1"] + mv["pass_arrives"]["model_auc"]
        log.append({"epoch": ep, "loss": float(np.mean(ls)), "val_bal_acc": mv["outcome"]["balanced_accuracy"],
                    "val_soon_auc": mv["release_within_1s_auc"], "val_rec_top1": mv["receiver"]["all_frames"]["top1"],
                    "val_rec_top1_release": mv["receiver"]["release_frame"]["top1"], "val_arrive_auc": mv["pass_arrives"]["model_auc"]})
        print(f"epoch {ep:2d} loss {np.mean(ls):.3f} | val bal-acc {mv['outcome']['balanced_accuracy']:.3f} soon-AUC {mv['release_within_1s_auc']:.3f} "
              f"rec-top1 {mv['receiver']['all_frames']['top1']:.3f} (release {mv['receiver']['release_frame']['top1']:.3f}) "
              f"arrive-AUC {mv['pass_arrives']['model_auc']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if score > best:
            best = score
            torch.save({"state_dict": model.state_dict(), "split": split}, a.out_dir / "possession_model.pt")
    pd.DataFrame(log).to_csv(a.out_dir / "possession_log.csv", index=False)
    model.load_state_dict(torch.load(a.out_dir / "possession_model.pt", weights_only=False)["state_dict"])
    res = {"split": split, **{s: metrics(data[s], predict(model, data[s])) for s in ["valid", "test"]}}
    (a.out_dir / "possession_metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["test"], indent=2))


if __name__ == "__main__":
    main()
