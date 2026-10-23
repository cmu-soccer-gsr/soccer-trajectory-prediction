"""
Non-learned imputation baselines. All functions take

    Y    : (N, W, 2) windows (hidden frames may contain anything, they are ignored)
    mask : (W,) bool, True = hidden, identical for the whole batch

and return a full (N, W, 2) prediction. Because the mask is shared by the batch, every
method is a fixed linear operator and runs fully vectorised.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline


# ---------------------------------------------------------------- linear
def linear(Y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    W = len(mask)
    t = np.arange(W)
    obs = np.flatnonzero(~mask)
    out = Y.copy()
    for d in range(Y.shape[-1]):
        # np.interp per window; mask is shared, so build the weight matrix once
        A = np.stack([np.interp(t, obs, np.eye(len(obs))[i]) for i in range(len(obs))], axis=1)
        out[..., d] = Y[:, obs, d] @ A.T
    return out


# ---------------------------------------------------------------- cubic P-spline
def _pspline_operator(mask: np.ndarray, knot_every: int, lam: float) -> np.ndarray:
    """Matrix M (W, n_obs) such that pred = M @ y_obs for a penalised cubic B-spline."""
    W = len(mask)
    k = 3
    inner = np.arange(0, W - 1 + 1e-9, knot_every, dtype=float)
    if inner[-1] < W - 1:
        inner = np.append(inner, W - 1)
    knots = np.concatenate([[inner[0]] * k, inner, [inner[-1]] * k])
    t = np.arange(W, dtype=float)
    B = BSpline.design_matrix(t, knots, k).toarray()  # (W, n_basis)
    obs = ~mask
    Bo = B[obs]
    D = np.diff(np.eye(B.shape[1]), n=2, axis=0)  # 2nd-difference penalty
    coef_op = np.linalg.solve(Bo.T @ Bo + lam * D.T @ D, Bo.T)  # (n_basis, n_obs)
    return B @ coef_op


def spline(Y: np.ndarray, mask: np.ndarray, knot_every: int = 5, lam: float = 1.0) -> np.ndarray:
    M = _pspline_operator(mask, knot_every, lam)
    obs = np.flatnonzero(~mask)
    out = Y.copy()
    out[:] = np.einsum("wo,nod->nwd", M, Y[:, obs])
    return out


# ---------------------------------------------------------------- Kalman (CV + RTS)
def kalman(Y: np.ndarray, mask: np.ndarray, q: float = 1e-3, r: float = 1e-3) -> np.ndarray:
    """
    Constant-velocity Kalman filter + Rauch-Tung-Striebel smoother per axis.
    q: white-acceleration noise (m^2/frame^3), r: measurement noise (m^2).
    Hidden frames are prediction-only steps. The covariances do not depend on the data,
    so all gains are computed once and applied to the whole batch.
    """
    N, W, D = Y.shape
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = q * np.array([[1 / 3, 1 / 2], [1 / 2, 1.0]])
    H = np.array([[1.0, 0.0]])

    # covariance recursion (shared)
    P_pred, P_filt, K = [], [], []
    P = np.diag([r, 1.0])
    for t in range(W):
        if t > 0:
            P = F @ P @ F.T + Q
        P_pred.append(P)
        if not mask[t]:
            S = H @ P @ H.T + r
            Kt = P @ H.T / S
            P = (np.eye(2) - Kt @ H) @ P
        else:
            Kt = np.zeros((2, 1))
        K.append(Kt)
        P_filt.append(P)
    G = [P_filt[t] @ F.T @ np.linalg.inv(P_pred[t + 1]) for t in range(W - 1)]

    first_obs = np.flatnonzero(~mask)[0]
    out = np.empty_like(Y)
    for d in range(D):
        y = Y[:, :, d]
        x = np.zeros((N, 2))
        x[:, 0] = y[:, first_obs]
        xs_pred, xs_filt = [], []
        for t in range(W):
            if t > 0:
                x = x @ F.T
            xs_pred.append(x)
            if not mask[t]:
                innov = y[:, t] - x[:, 0]
                x = x + innov[:, None] * K[t][:, 0][None, :]
            xs_filt.append(x)
        xs = xs_filt[-1]
        res = [xs]
        for t in range(W - 2, -1, -1):
            xs = xs_filt[t] + (xs - xs_pred[t + 1]) @ G[t].T
            res.append(xs)
        res = res[::-1]
        out[:, :, d] = np.stack([s[:, 0] for s in res], axis=1)
    return out


METHODS = {"linear": linear, "spline": spline, "kalman": kalman}
