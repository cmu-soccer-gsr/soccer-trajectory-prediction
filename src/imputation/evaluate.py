"""Shared evaluation: fixed eval windows, optional context noise, per-gap metrics."""

from __future__ import annotations

import numpy as np

from .data import GAP_LENGTHS, ade, centred_mask, max_err


def context_noise(shape, sigma: float, seed: int) -> np.ndarray:
    """Fixed Gaussian noise (m) added to the observed frames.

    The GSR labels are very smooth (median per-frame acceleration ~4 mm/frame^2), much
    smoother than positions a tracker + camera calibration would produce. With sigma = 0
    the task is 'impute clean ground truth'; with sigma > 0 the observed context is
    perturbed (as a tracker output would be) while the error is still measured against
    the clean ground truth. The noise is fixed per split, so all methods see the same
    inputs.
    """
    if sigma == 0:
        return np.zeros(shape, np.float32)
    rng = np.random.default_rng(seed)
    return (sigma * rng.standard_normal(shape)).astype(np.float32)


def evaluate(predict, gt: np.ndarray, sigma: float, seed: int = 0,
             gaps=GAP_LENGTHS) -> list[dict]:
    """predict(Y_noisy, mask) -> (N, W, 2). Returns one row per gap length."""
    noisy = gt + context_noise(gt.shape, sigma, seed)
    rows = []
    for L in gaps:
        m = centred_mask(L, gt.shape[1])
        pred = predict(noisy, m)
        a = ade(pred, gt, m)
        rows.append({
            "gap_frames": L,
            "gap_s": L / 25,
            "ade_m": float(a.mean()),
            "ade_ci95_m": float(1.96 * a.std(ddof=1) / np.sqrt(len(a))),
            "ade_median_m": float(np.median(a)),
            "max_err_m": float(max_err(pred, gt, m).mean()),
            "n_windows": int(len(a)),
            "_per_window": a,
        })
    return rows
