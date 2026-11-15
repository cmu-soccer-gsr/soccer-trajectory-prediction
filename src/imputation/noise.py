"""Context-noise models for the imputation benchmark.

"gauss:<sigma>"      iid Gaussian, sigma metres per axis (the original protocol)
"pipeline[:<scale>]" noise measured on the real GSR pipeline (valid split, good frames,
                     see scripts/measure_pipeline_noise.py), optionally scaled:
                     e(t) = offset + slow(t) + white(t) per axis, offset constant per
                     window, slow an AR(1) process, white iid.

A plain float is read as gauss:<float>, so old calls keep working.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

MODEL_PATH = Path(__file__).resolve().parents[2] / "results" / "pipeline_noise" / "noise_model.json"
DEFAULT = {"offset_mean": [0.057, 0.298], "offset_std": [0.453, 0.439], "ar_std": [0.327, 0.415],
           "ar_rho": [0.936, 0.923], "white_std": [0.202, 0.306]}


def load_model(path: Path = MODEL_PATH) -> dict:
    return json.loads(path.read_text()) if path.exists() else DEFAULT


def parse(spec) -> tuple[str, float]:
    if isinstance(spec, (int, float)):
        return "gauss", float(spec)
    kind, _, v = str(spec).partition(":")
    if kind == "gauss":
        return "gauss", float(v)
    if kind == "pipeline":
        return "pipeline", float(v) if v else 1.0
    return "gauss", float(spec)


def label(spec) -> str:
    kind, v = parse(spec)
    return f"{v:g}" if kind == "gauss" else f"pipeline x{v:g}"


def sample(shape, spec, rng: np.random.Generator, model: dict | None = None):
    """Return (noise, offset): noise (N, T, 2); offset (N, 1, 2) is its constant part."""
    N, T, _ = shape
    kind, v = parse(spec)
    if v == 0:
        z = np.zeros(shape, np.float32)
        return z, z[:, :1]
    if kind == "gauss":
        n = (v * rng.standard_normal(shape)).astype(np.float32)
        return n, np.zeros((N, 1, 2), np.float32)
    m = model or load_model()
    mu, so = np.array(m["offset_mean"]), np.array(m["offset_std"])
    sa, rho, sw = np.array(m["ar_std"]), np.array(m["ar_rho"]), np.array(m["white_std"])
    off = mu + so * rng.standard_normal((N, 1, 2))
    ar = np.empty((N, T, 2))
    ar[:, 0] = sa * rng.standard_normal((N, 2))
    inno = sa * np.sqrt(1 - rho ** 2)
    for t in range(1, T):
        ar[:, t] = rho * ar[:, t - 1] + inno * rng.standard_normal((N, 2))
    n = off + ar + sw * rng.standard_normal((N, T, 2))
    return (v * n).astype(np.float32), (v * off).astype(np.float32)


def sample_torch(shape, scale: torch.Tensor, gen: torch.Generator, model: dict | None = None) -> torch.Tensor:
    """Pipeline noise for training batches; scale (B, 1, 1) per window."""
    B, T, _ = shape
    m = model or load_model()
    t = lambda k: torch.tensor(m[k], dtype=torch.float32)
    mu, so, sa, rho, sw = t("offset_mean"), t("offset_std"), t("ar_std"), t("ar_rho"), t("white_std")
    off = mu + so * torch.randn(B, 1, 2, generator=gen)
    e = torch.randn(B, T, 2, generator=gen)
    ar = torch.empty(B, T, 2)
    ar[:, 0] = sa * e[:, 0]
    inno = sa * torch.sqrt(1 - rho ** 2)
    for i in range(1, T):
        ar[:, i] = rho * ar[:, i - 1] + inno * e[:, i]
    return scale * (off + ar + sw * torch.randn(B, T, 2, generator=gen))
