"""S1 selective prediction: development thresholds on the maximum softmax and their realised behaviour under shift.

``tau_c`` is the k-th largest ``acdc_dev`` out-of-fold confidence with ``k = ceil(c * n)``; a case is accepted when its
confidence is at least ``tau_c``, so the development coverage is exactly ``k / n`` without ties (configs/trust.yaml).
The risk-coverage curve has one point per distinct confidence; AURC is the step integral of the selective risk over
coverage (the mean selective risk over k = 1..n when confidences are distinct).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ...metrics import wilson_interval


def max_softmax(proba: np.ndarray) -> np.ndarray:
    """Row maximum of a probability matrix; NaN for rows without finite probabilities."""

    p = np.asarray(proba, dtype=float)
    finite = np.all(np.isfinite(p), axis=1)
    out = np.full(p.shape[0], np.nan)
    if finite.any():
        out[finite] = p[finite].max(axis=1)
    return out


def coverage_threshold(confidence: np.ndarray, coverage: float) -> dict[str, Any]:
    """``tau`` giving ``coverage`` on the development confidences (k-th largest, k = ceil(c n))."""

    c = np.asarray(confidence, dtype=float)
    c = c[np.isfinite(c)]
    if not 0.0 < float(coverage) <= 1.0:
        raise ValueError("coverage must lie in (0, 1]")
    if c.size == 0:
        return {"coverage_target": float(coverage), "tau": None, "n": 0, "k": 0, "status": "NE", "reason": "no_finite_confidences"}
    k = min(max(int(math.ceil(float(coverage) * c.size - 1e-12)), 1), int(c.size))
    tau = float(np.sort(c)[::-1][k - 1])
    accepted = int(np.sum(c >= tau))
    return {"coverage_target": float(coverage), "tau": tau, "n": int(c.size), "k": k, "development_coverage": accepted / c.size,
            "n_accepted_development": accepted, "ties_at_tau": int(np.sum(c == tau)), "status": "FITTED", "reason": None}


def realised_at_threshold(confidence: np.ndarray, correct: np.ndarray, tau: float | None) -> dict[str, Any]:
    """Realised coverage (accepted / evaluable) and accuracy among accepted cases at ``tau`` with Wilson intervals.

    ``correct`` is 1/0 with NaN for cases without a known reference; both proportions use the reference-known cases
    with a finite confidence (the denominators are reported with every proportion)."""

    conf = np.asarray(confidence, dtype=float)
    corr = np.asarray(correct, dtype=float)
    ok = np.isfinite(conf) & np.isfinite(corr)
    n = int(ok.sum())
    if tau is None or n == 0:
        return {"tau": tau, "n": n, "status": "NE", "reason": "no_threshold" if tau is None else "no_evaluable_cases"}
    acc = ok & (conf >= float(tau))
    n_acc = int(acc.sum())
    k_correct = int(np.sum(corr[acc] == 1.0))
    cov_lo, cov_hi = wilson_interval(n_acc, n)
    out: dict[str, Any] = {"tau": float(tau), "n": n, "n_accepted": n_acc, "coverage": n_acc / n, "coverage_wilson_95": [cov_lo, cov_hi],
                           "n_correct_accepted": k_correct, "status": "EVALUATED", "reason": None}
    if n_acc:
        lo, hi = wilson_interval(k_correct, n_acc)
        out.update({"accuracy_accepted": k_correct / n_acc, "accuracy_accepted_wilson_95": [lo, hi]})
    else:
        out.update({"accuracy_accepted": None, "accuracy_accepted_wilson_95": [None, None], "accuracy_reason": "no_accepted_cases"})
    n_all_correct = int(np.sum(corr[ok] == 1.0))
    lo, hi = wilson_interval(n_all_correct, n)
    out.update({"accuracy_full_coverage": n_all_correct / n, "accuracy_full_coverage_wilson_95": [lo, hi]})
    return out


def risk_coverage_curve(confidence: np.ndarray, correct: np.ndarray) -> pd.DataFrame:
    """One row per distinct confidence (descending): threshold, accepted count, coverage and selective risk."""

    conf = np.asarray(confidence, dtype=float)
    corr = np.asarray(correct, dtype=float)
    ok = np.isfinite(conf) & np.isfinite(corr)
    conf, corr = conf[ok], corr[ok]
    n = conf.size
    if n == 0:
        return pd.DataFrame(columns=["threshold", "n_accepted", "coverage", "risk", "n_errors"])
    order = np.argsort(-conf, kind="mergesort")
    conf_sorted, err_sorted = conf[order], 1.0 - corr[order]
    cum_err = np.cumsum(err_sorted)
    # last position of every distinct confidence (ties enter together)
    last = np.flatnonzero(np.r_[conf_sorted[1:] != conf_sorted[:-1], True])
    n_acc = last + 1
    return pd.DataFrame({"threshold": conf_sorted[last], "n_accepted": n_acc.astype(int), "coverage": n_acc / n,
                         "risk": cum_err[last] / n_acc, "n_errors": cum_err[last].astype(int)})


def aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve (step integral; NaN without evaluable cases)."""

    curve = risk_coverage_curve(confidence, correct)
    if curve.empty:
        return float("nan")
    cov = curve["coverage"].to_numpy(dtype=float)
    widths = np.diff(np.r_[0.0, cov])
    return float(np.sum(curve["risk"].to_numpy(dtype=float) * widths))
