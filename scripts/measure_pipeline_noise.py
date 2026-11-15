#!/usr/bin/env python3
"""
Measure the real position noise of the GSR pipeline against ground truth.

    python scripts/measure_pipeline_noise.py --state baseline-valid-slim.pklz \
        --gt game_states_valid.csv.gz --out results/pipeline_noise

Pipeline detections are matched to GT boxes per frame in image space (Hungarian on
1 - IoU, IoU >= 0.5), so association errors do not leak into the position error.
For every matched pair the error is pipeline bbox_pitch bottom-middle minus GT (x, y).

Outputs (in --out):
    pairs.csv.gz        one row per matched pair (sequence, frame, gt id, pred track, error)
    detection.csv       per-sequence box counts
    noise_model.json    fitted generator used by src/imputation/noise.py
    noise.json          calibration-failure frames (rate, bursts, a GT-free detector), and on
                        the remaining frames: bias, shared per-frame shift, per-track offset,
                        AR residual (std, jitter, autocorrelation), error by field position and
                        box height; plus detection recall and identity continuity

Add --pairs DIR to re-run the analysis from an earlier pairs.csv.gz without matching.
"""
import argparse
import json
import zipfile
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

FPS = 25


def iou(a, b):
    """a: (n,4) ltwh, b: (m,4) ltwh -> (n,m)"""
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, 0, None], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, 1, None], b[None, :, 1]), 0, None)
    inter = iw * ih
    return inter / (a[:, 2:].prod(1)[:, None] + b[:, 2:].prod(1)[None] - inter + 1e-9)


def load_state(path):
    z = zipfile.ZipFile(path)
    out = {}
    for n in z.namelist():
        if n.endswith(".pkl") and not n.endswith("_image.pkl"):
            out[n[:-4]] = pickle.loads(z.read(n))
    return out


def match_sequence(det, gt):
    det = det[(det.role != "ball") & det.track_id.notna() & ~det.ignored.astype(bool)].copy()
    det["frame"] = (det.image_id.astype(np.int64) % 1_000_000).astype(int)
    det["px"] = [p["x_bottom_middle"] if isinstance(p, dict) else np.nan for p in det.bbox_pitch]
    det["py"] = [p["y_bottom_middle"] if isinstance(p, dict) else np.nan for p in det.bbox_pitch]
    gt = gt[gt.role != "ball"]
    rows, n_gt, n_det = [], 0, 0
    dg = dict(tuple(det.groupby("frame")))
    for f, g in gt.groupby("frame"):
        n_gt += len(g)
        d = dg.get(f)
        if d is None or not len(d):
            continue
        n_det += len(d)
        M = iou(g[["bx", "by", "bw", "bh"]].to_numpy(float), np.stack(d.bbox_ltwh.to_numpy()).astype(float))
        r, c = linear_sum_assignment(-M)
        for i, j in zip(r, c):
            if M[i, j] >= 0.5:
                gi, dj = g.iloc[i], d.iloc[j]
                rows.append((f, int(gi.track_id), int(dj.track_id), gi.role, gi.x, gi.y, dj.px, dj.py, gi.bh, M[i, j]))
    df = pd.DataFrame(rows, columns=["frame", "gt_id", "pred_id", "role", "gx", "gy", "px", "py", "bh", "iou"])
    return df, n_gt, n_det, len(det)


def acf(series_list, max_lag):
    """Pooled autocorrelation of demeaned 1-D series at lags 0..max_lag (consecutive frames only)."""
    num = np.zeros(max_lag + 1); den = np.zeros(max_lag + 1); var_n = 0; var_s = 0.0
    for s in series_list:
        s = s - s.mean()
        var_s += (s ** 2).sum(); var_n += len(s)
        for k in range(max_lag + 1):
            if len(s) > k:
                num[k] += (s[: len(s) - k] * s[k:]).sum(); den[k] += len(s) - k
    v = var_s / var_n
    return (num / np.maximum(den, 1)) / v


def runs(df):
    """Split matched pairs into runs of consecutive frames per (sequence, gt_id)."""
    out = []
    for _, g in df.sort_values("frame").groupby(["seq", "gt_id"]):
        f = g.frame.to_numpy()
        cut = np.flatnonzero(np.diff(f) != 1) + 1
        for idx in np.split(np.arange(len(g)), cut):
            out.append(g.iloc[idx])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path)
    ap.add_argument("--gt", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pairs", type=Path, help="reuse pairs.csv.gz and detection.csv from this dir")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.pairs:
        P = pd.read_csv(a.pairs / "pairs.csv.gz"); D = pd.read_csv(a.pairs / "detection.csv")
        out = analyse(P, D)
        (a.out / "noise.json").write_text(json.dumps(out, indent=1, default=float))
        (a.out / "noise_model.json").write_text(json.dumps(out["noise_model"], indent=1))
        print(json.dumps(out["summary"], indent=1, default=float))
        return

    state = load_state(a.state)
    G = pd.read_csv(a.gt)
    G["vid"] = G.sequence.str[-3:]
    parts, det_stats = [], []
    for vid, det in sorted(state.items()):
        gt = G[G.vid == vid]
        if not len(gt):
            continue
        df, n_gt, n_det, n_tr = match_sequence(det, gt)
        df["seq"] = f"SNGS-{vid}"
        parts.append(df)
        det_stats.append({"seq": f"SNGS-{vid}", "gt_boxes": n_gt, "pred_boxes": n_tr, "matched": len(df)})
        print(vid, n_gt, n_tr, len(df), flush=True)
    P = pd.concat(parts, ignore_index=True)
    P = P[np.isfinite(P.px) & np.isfinite(P.py) & np.isfinite(P.gx) & np.isfinite(P.gy)]
    P["ex"] = P.px - P.gx; P["ey"] = P.py - P.gy
    P["err"] = np.hypot(P.ex, P.ey)
    P.to_csv(a.out / "pairs.csv.gz", index=False)
    D = pd.DataFrame(det_stats)
    D.to_csv(a.out / "detection.csv", index=False)
    out = analyse(P, D)
    (a.out / "noise.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out["summary"], indent=1, default=float))


def fit_noise_model(P, bad_m=3.0, min_run=3 * FPS, lags=15):
    """Per-axis generator for good-frame noise, fitted on runs of >= min_run frames:
    e(t) = offset + slow(t) + white(t), offset ~ N(mu, s_off) constant per run,
    slow ~ AR(1) with coefficient rho, white ~ iid. The AR weight w and rho come from
    a least-squares fit of acf(k) = w * rho^k, k = 1..lags."""
    from scipy.optimize import curve_fit
    med = P.groupby(["seq", "frame"]).err.median().rename("med_")
    G = P.join(med, on=["seq", "frame"])
    G = G[G.med_ <= bad_m]
    R = [r for r in runs(G) if len(r) >= min_run]
    off = np.array([[r.ex.mean(), r.ey.mean()] for r in R])
    res = [np.stack([r.ex.to_numpy() - r.ex.mean(), r.ey.to_numpy() - r.ey.mean()], 1) for r in R]
    sd = np.concatenate(res).std(0)
    m = {"fps": FPS, "runs": len(R), "offset_mean": off.mean(0).tolist(), "offset_std": off.std(0).tolist(),
         "ar_std": [], "ar_rho": [], "white_std": []}
    k = np.arange(1, lags + 1)
    for ax in range(2):
        ac = acf([x[:, ax] for x in res], lags)
        (w, rho), _ = curve_fit(lambda k, w, r: w * r ** k, k, ac[1:], p0=[0.6, 0.9], bounds=([0, 0], [1, 0.999]))
        m["ar_std"].append(float(np.sqrt(w) * sd[ax])); m["ar_rho"].append(float(rho))
        m["white_std"].append(float(np.sqrt(1 - w) * sd[ax]))
    return m


def bursts(flags):
    b = np.r_[0, flags.astype(int), 0]
    d = np.diff(b)
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def tau(ac):
    k = np.flatnonzero(ac < np.exp(-1))
    return int(k[0]) if len(k) else None


def analyse(P, D, bad_m=3.0):
    q = lambda v, p: float(np.percentile(v, p))
    max_lag = 2 * FPS

    # 1. calibration failures: frames where the median player error exceeds bad_m
    F = P.groupby(["seq", "frame"]).agg(med=("err", "median"), sx=("ex", "median"), sy=("ey", "median"),
                                        off=("px", lambda s: 0.0)).reset_index()
    F["off"] = P.assign(o=(P.px.abs() > 56) | (P.py.abs() > 37)).groupby(["seq", "frame"]).o.mean().to_numpy()
    F["bad"] = F.med > bad_m
    B = np.concatenate([bursts(g.sort_values("frame").bad.to_numpy()) for _, g in F.groupby("seq")])
    flag = F.off > 0
    seq_bad = F.groupby("seq").bad.mean()
    calib = {"bad_frame_threshold_m": bad_m, "bad_frame_rate": float(F.bad.mean()),
             "bad_pair_rate": float(P.merge(F[["seq", "frame", "bad"]]).bad.mean()),
             "bad_error_median_m": float(P.merge(F[F.bad][["seq", "frame"]]).err.median()),
             "bursts": {"n": int(len(B)), "median_frames": q(B, 50), "p90_frames": q(B, 90),
                        "share_of_bad_frames_in_bursts_over_1s": float(B[B > FPS].sum() / B.sum())},
             "sequences_over_half_bad": int((seq_bad > 0.5).sum()),
             "sequences_under_5pct_bad": int((seq_bad < 0.05).sum()),
             "detector_any_player_off_pitch": {"flag_rate": float(flag.mean()),
                                               "recall": float((flag & F.bad).sum() / F.bad.sum()),
                                               "precision": float((flag & F.bad).sum() / max(flag.sum(), 1))}}

    # 2. good frames: shared per-frame shift + per-player residual
    G = P.merge(F[~F.bad][["seq", "frame", "sx", "sy"]])
    G["rx"] = G.ex - G.sx; G["ry"] = G.ey - G.sy
    shift_runs = []
    for _, g in F[~F.bad].sort_values("frame").groupby("seq"):
        f = g.frame.to_numpy(); cut = np.flatnonzero(np.diff(f) != 1) + 1
        shift_runs += [x for x in np.split(g[["sx", "sy"]].to_numpy(), cut) if len(x) >= max_lag + 1]
    ac_sx = acf([x[:, 0] for x in shift_runs], max_lag); ac_sy = acf([x[:, 1] for x in shift_runs], max_lag)
    sh = F[~F.bad][["sx", "sy"]].to_numpy()

    R = [r for r in runs(G) if len(r) >= 2 * FPS]
    off = np.array([[r.rx.mean(), r.ry.mean()] for r in R])
    res = [np.stack([r.rx.to_numpy() - r.rx.mean(), r.ry.to_numpy() - r.ry.mean()], 1) for r in R]
    ra = np.concatenate(res)
    jit = np.concatenate([np.diff(x, axis=0) for x in res])
    ac_x = acf([x[:, 0] for x in res], max_lag); ac_y = acf([x[:, 1] for x in res], max_lag)
    rho = float((ac_x[1] + ac_y[1]) / 2)
    tot = sh.var(0).sum() + off.var(0).sum() + ra.var(0).sum()

    G["y_bin"] = pd.cut(G.gy, [-40, -20, -10, 0, 10, 20, 40])
    by_y = G.groupby("y_bin", observed=True).agg(n=("err", "size"), err_median=("err", "median"),
                                                  bias_y=("ey", "median"), resid_std_x=("rx", "std"),
                                                  resid_std_y=("ry", "std")).reset_index()
    by_y["y_bin"] = by_y.y_bin.astype(str)
    G["h_bin"] = pd.cut(G.bh, [0, 60, 80, 100, 130, 1000])
    by_h = G.groupby("h_bin", observed=True).agg(n=("err", "size"), err_median=("err", "median")).reset_index()
    by_h["h_bin"] = by_h.h_bin.astype(str)
    good = {"pairs": int(len(G)),
            "error_m": {"median": q(G.err, 50), "p90": q(G.err, 90), "rmse": float(np.sqrt((G.err ** 2).mean()))},
            "bias_m": [float(G.ex.mean()), float(G.ey.mean())],
            "frame_shift": {"std_m": sh.std(0).tolist(), "lag1_acf": [float(ac_sx[1]), float(ac_sy[1])],
                            "decorrelation_frames": [tau(ac_sx), tau(ac_sy)],
                            "acf_x": ac_sx.round(4).tolist(), "acf_y": ac_sy.round(4).tolist()},
            "track_offset": {"runs": len(R), "std_m": off.std(0).tolist()},
            "residual": {"frames": int(len(ra)), "std_m": ra.std(0).tolist(),
                         "jitter_std_m_per_frame": jit.std(0).tolist(), "ar1_rho": rho,
                         "decorrelation_frames": [tau(ac_x), tau(ac_y)],
                         "acf_x": ac_x.round(4).tolist(), "acf_y": ac_y.round(4).tolist()},
            "variance_share": {"frame_shift": float(sh.var(0).sum() / tot), "track_offset": float(off.var(0).sum() / tot),
                               "residual": float(ra.var(0).sum() / tot)},
            "by_gt_y": by_y.to_dict("records"), "by_box_height": by_h.to_dict("records")}

    # 3. coverage and identity continuity (GT view)
    gaps, sw = [], 0
    for _, g in P.sort_values("frame").groupby(["seq", "gt_id"]):
        d = np.diff(g.frame.to_numpy()); gaps += list(d[d > 1] - 1)
        sw += int((np.diff(g.pred_id.to_numpy()) != 0).sum())
    gaps = np.array(gaps); L = np.array([len(r) for r in runs(P)])
    tracks = {"recall_iou50": float(D.matched.sum() / D.gt_boxes.sum()),
              "precision_iou50": float(D.matched.sum() / D.pred_boxes.sum()),
              "pred_id_changes_per_gt_track": float(sw / P.groupby(["seq", "gt_id"]).ngroups),
              "observed_run_frames": {"median": q(L, 50), "p90": q(L, 90)},
              "gap_frames": {"n": int(len(gaps)), "median": q(gaps, 50), "p90": q(gaps, 90),
                             "share_le_1s": float((gaps <= FPS).mean())}}
    summary = {"pairs": int(len(P)), "sequences": int(P.seq.nunique()),
               "all_frames_error_median_m": q(P.err, 50), "all_frames_error_p90_m": q(P.err, 90),
               "bad_frame_rate": calib["bad_frame_rate"], "good_error_median_m": good["error_m"]["median"],
               "good_bias_m": good["bias_m"], "frame_shift_std_m": good["frame_shift"]["std_m"],
               "track_offset_std_m": good["track_offset"]["std_m"], "residual_std_m": good["residual"]["std_m"],
               "residual_ar1_rho": rho, "residual_jitter_m": good["residual"]["jitter_std_m_per_frame"],
               "variance_share": good["variance_share"], "recall": tracks["recall_iou50"],
               "pred_id_changes_per_gt_track": tracks["pred_id_changes_per_gt_track"]}
    return {"summary": summary, "noise_model": fit_noise_model(P, bad_m), "calibration": calib, "good_frames": good, "tracks": tracks,
            "per_sequence": D.assign(bad_rate=D.seq.map(seq_bad)).to_dict("records")}


if __name__ == "__main__":
    main()
