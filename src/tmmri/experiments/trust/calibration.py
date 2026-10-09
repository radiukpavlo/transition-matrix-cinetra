"""S4 calibration transfer: one temperature per predictor fitted on ``acdc_dev`` out-of-fold logits (NLL), ECE, Brier
score, log loss and reliability-diagram data before and after scaling.

When a predictor records probabilities but no logits, ``log p`` is used: ``softmax(log p / T) == softmax(logit / T)``
because ``log p`` differs from the logits by a per-row constant, so the fitted temperature is the same.
The primary ECE is the v6 equal-mass ECE (:func:`tmmri.classification.expected_calibration_error_equal_mass`); the
equal-width ECE is a companion (reliability diagrams, NEURAL-TM-1.0 comparability).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from ...classification import expected_calibration_error_equal_mass

LOG_LOSS_CLIP = 1e-15


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    shifted = z - z.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def softmax(z: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(z))


def logits_from_probabilities(proba: np.ndarray) -> np.ndarray:
    """``log p`` (rows of finite probabilities; a zero probability becomes the log of the log-loss clip)."""

    return np.log(np.clip(np.asarray(proba, dtype=float), LOG_LOSS_CLIP, 1.0))


def nll(logits: np.ndarray, y: np.ndarray, temperature: float = 1.0) -> float:
    lp = log_softmax(np.asarray(logits, dtype=float) / float(temperature))
    return float(-np.mean(lp[np.arange(lp.shape[0]), np.asarray(y, dtype=int)]))


def fit_temperature(logits: np.ndarray, y: np.ndarray, *, bounds: tuple[float, float] = (0.05, 20.0), xatol: float = 1e-10) -> dict[str, Any]:
    """The temperature minimizing the mean NLL (bounded search over log T); ``y`` are class indices."""

    z = np.asarray(logits, dtype=float)
    t = np.asarray(y, dtype=int)
    ok = np.all(np.isfinite(z), axis=1)
    z, t = z[ok], t[ok]
    if z.shape[0] == 0:
        return {"temperature": None, "status": "NE", "reason": "no_finite_logits", "n": 0}
    lo, hi = math.log(bounds[0]), math.log(bounds[1])
    res = minimize_scalar(lambda s: nll(z, t, math.exp(s)), bounds=(lo, hi), method="bounded", options={"xatol": float(xatol)})
    temp = float(math.exp(res.x))
    at_bound = bool(abs(res.x - lo) < 1e-6 or abs(res.x - hi) < 1e-6)
    return {"temperature": temp, "log_temperature": float(res.x), "nll_before": nll(z, t, 1.0), "nll_after": nll(z, t, temp), "n": int(z.shape[0]),
            "converged": bool(res.success), "at_bound": at_bound, "bounds": [float(bounds[0]), float(bounds[1])], "status": "FITTED", "reason": None}


def scaled_probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = np.asarray(logits, dtype=float)
    out = np.full(z.shape, np.nan)
    ok = np.all(np.isfinite(z), axis=1)
    if ok.any():
        out[ok] = softmax(z[ok] / float(temperature))
    return out


def brier(proba: np.ndarray, y: np.ndarray) -> float:
    """Multiclass Brier score: mean over cases of the squared distance to the one-hot reference."""

    p = np.asarray(proba, dtype=float)
    onehot = np.zeros_like(p)
    onehot[np.arange(p.shape[0]), np.asarray(y, dtype=int)] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def log_loss(proba: np.ndarray, y: np.ndarray, clip: float = LOG_LOSS_CLIP) -> float:
    p = np.asarray(proba, dtype=float)
    return float(np.mean(-np.log(np.clip(p[np.arange(p.shape[0]), np.asarray(y, dtype=int)], clip, 1.0))))


def ece_equal_width(proba: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    table = reliability_table(proba, y, n_bins=n_bins, binning="equal_width")
    n = float(table["count"].sum())
    gaps = table["abs_gap"].fillna(0.0).to_numpy(dtype=float)
    return float(np.sum(table["count"].to_numpy(dtype=float) * gaps) / n) if n else float("nan")


def ece(proba: np.ndarray, y: np.ndarray, *, n_bins: int = 10, binning: str = "equal_mass") -> float:
    if binning == "equal_mass":
        return expected_calibration_error_equal_mass(np.asarray(y, dtype=int), np.asarray(proba, dtype=float), n_bins=int(n_bins))[0]
    if binning == "equal_width":
        return ece_equal_width(proba, y, n_bins=n_bins)
    raise ValueError(f"unknown ECE binning {binning!r}")


def reliability_table(proba: np.ndarray, y: np.ndarray, *, n_bins: int = 10, binning: str = "equal_width") -> pd.DataFrame:
    """Reliability-diagram data of the top-label confidence: per bin the count, mean confidence and accuracy."""

    p = np.asarray(proba, dtype=float)
    t = np.asarray(y, dtype=int)
    conf = p.max(axis=1)
    correct = (np.argmax(p, axis=1) == t).astype(float)
    rows = []
    if binning == "equal_width":
        edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
        idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, int(n_bins) - 1)
        members = [np.flatnonzero(idx == b) for b in range(int(n_bins))]
        bounds = [(float(edges[b]), float(edges[b + 1])) for b in range(int(n_bins))]
    elif binning == "equal_mass":
        members = list(np.array_split(np.argsort(conf, kind="mergesort"), int(n_bins)))
        bounds = [(float(conf[m].min()), float(conf[m].max())) if m.size else (None, None) for m in members]
    else:
        raise ValueError(f"unknown binning {binning!r}")
    for b, (m, (lo, hi)) in enumerate(zip(members, bounds)):
        if m.size == 0:
            rows.append({"bin_id": b, "lower": lo, "upper": hi, "count": 0, "mean_confidence": np.nan, "accuracy": np.nan, "abs_gap": np.nan})
            continue
        mc, acc = float(conf[m].mean()), float(correct[m].mean())
        rows.append({"bin_id": b, "lower": lo, "upper": hi, "count": int(m.size), "mean_confidence": mc, "accuracy": acc, "abs_gap": abs(mc - acc)})
    return pd.DataFrame(rows)
