"""
Data loading and the masking protocol for trajectory imputation.

Protocol (shared by every method so the numbers are comparable):

1. Start from the on-screen segments produced by scripts/extract_trajectories.py.
2. Split each segment into fully observed runs: consecutive frames with finite pitch
   coordinates. Real short gaps (gap_before > 0) and frames without pitch coordinates
   break a run, so ground truth exists for every frame we evaluate.
3. Cut runs into fixed windows of WINDOW frames (default 75 = 3 s at 25 fps).
   Windows containing a physically impossible jump (> max_step metres in one frame,
   i.e. annotation glitches) are dropped.
4. Evaluation: in every window, hide a centred block of L frames, L in GAP_LENGTHS
   (default 5, 10, 15, 20, 25 frames = 0.2 to 1 s, matching the <= 1 s gaps we keep).
   The same windows are used for every L, so error-vs-gap curves are paired.
   Eval windows are non-overlapping (stride = WINDOW).
5. Training (GRU, hyper-parameter tuning) uses overlapping windows from the train
   split with random gap lengths and positions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FPS = 25
WINDOW = 75
GAP_LENGTHS = (5, 10, 15, 20, 25)
MAX_STEP_M = 1.0  # 1 m per frame = 25 m/s, faster than any sprint


def load_runs(csv_path: str) -> list[np.ndarray]:
    """Return a list of fully observed runs, each an array (T, 2) of pitch x, y in metres."""
    df = pd.read_csv(csv_path, usecols=["segment_uid", "frame", "x_pitch", "y_pitch"])
    df = df.sort_values(["segment_uid", "frame"])
    runs = []
    for _, g in df.groupby("segment_uid", sort=False):
        frames = g["frame"].to_numpy()
        xy = g[["x_pitch", "y_pitch"]].to_numpy(dtype=np.float32)
        ok = np.isfinite(xy).all(axis=1)
        # new run whenever a frame is skipped or coordinates are missing
        brk = np.ones(len(frames), dtype=bool)
        brk[1:] = (np.diff(frames) != 1) | ~ok[1:] | ~ok[:-1]
        starts = np.flatnonzero(brk)
        ends = np.append(starts[1:], len(frames))
        for s, e in zip(starts, ends):
            if ok[s:e].all() and e - s >= 2:
                runs.append(xy[s:e])
    return runs


def make_windows(runs: list[np.ndarray], window: int = WINDOW, stride: int = WINDOW,
                 max_step: float = MAX_STEP_M) -> np.ndarray:
    """Slice runs into windows (N, window, 2), dropping windows with glitch jumps."""
    out = []
    for r in runs:
        if len(r) < window:
            continue
        steps = np.linalg.norm(np.diff(r, axis=0), axis=1)
        for s in range(0, len(r) - window + 1, stride):
            if steps[s:s + window - 1].max() <= max_step:
                out.append(r[s:s + window])
    return np.stack(out) if out else np.empty((0, window, 2), np.float32)


def centred_mask(gap: int, window: int = WINDOW) -> np.ndarray:
    """Boolean mask (window,), True = hidden frame."""
    m = np.zeros(window, dtype=bool)
    s = (window - gap) // 2
    m[s:s + gap] = True
    return m


def ade(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-window mean Euclidean error (m) over hidden frames. pred, gt: (N, W, 2)."""
    err = np.linalg.norm(pred[:, mask] - gt[:, mask], axis=-1)
    return err.mean(axis=1)


def max_err(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    err = np.linalg.norm(pred[:, mask] - gt[:, mask], axis=-1)
    return err.max(axis=1)
