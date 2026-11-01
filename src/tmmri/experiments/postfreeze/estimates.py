"""Effect sizes and descriptive intervals of the post-freeze block (``configs/exploratory_postfreeze.yaml`` statistics).

* :func:`paired_rho` - rho, rho_EF, rho of the volumes, rho_j and the normalized MAE of several arms (prediction, null)
  on the same rows with one paired cluster bootstrap: 2,000 replicates, seed 20260928, unit ``cluster_id``, replicate r
  drawn with ``SeedSequence(seed).spawn(n)[r]`` and each patient weighed once, exactly as
  :func:`tmmri.metrics.primary_rho`, so a recomputed primary arm reproduces the recorded rho and interval.  Paired
  differences against a reference arm use the same draws.
* :func:`wilson` (proportions with numerator and denominator), :func:`auroc_ci` (cluster bootstrap), :func:`risk_ratio`
  (Katz log interval).

Effect sizes with percentile or Wilson intervals only; no p-values.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from ...metrics import (
    _cluster_codes,
    _cluster_draw_weights,
    _patient_weighted_mae,
    _resampling_clusters,
    per_target_errors,
    wilson_interval,
)

BELOW_MIN_CASES = "STRATUM_BELOW_MIN_CASES"


def families(lane_cfg: Mapping[str, Any]) -> dict[str, list[str]]:
    """``{rho: 7 primary targets, rho_ef: EF targets, rho_volumes: volume targets}`` (lane ``statistics``)."""

    s = lane_cfg["statistics"]
    return {"rho": [str(t) for t in s["primary_targets"]], "rho_ef": [str(t) for t in s["ef_targets"]],
            "rho_volumes": [str(t) for t in s["volume_targets"]]}


def min_estimable(study: Any, family: list[str]) -> int:
    """rho is withdrawn below five estimable targets (``bootstrap.yaml`` primary_test); a smaller family needs all its
    targets (rho_EF: both EF targets, as H4a)."""

    return min(int(study.bootstrap["primary_test"]["min_estimable_targets"]), len(family))


def wilson(k: int, n: int, level: float = 0.95) -> dict[str, Any]:
    k, n = int(k), int(n)
    if n <= 0:
        return {"k": k, "n": n, "estimate": None, "ci_lower": None, "ci_upper": None, "interval": "wilson", "status": "NE", "reason": "NO_CASES"}
    lo, hi = wilson_interval(k, n, level)
    return {"k": k, "n": n, "estimate": k / n, "ci_lower": lo, "ci_upper": hi, "interval": "wilson", "status": "EVALUATED", "reason": None}


def _terms(err_model: np.ndarray, err_null: np.ndarray, sd: np.ndarray, codes: np.ndarray, n_units: int,
           weights: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(nMAE model, nMAE null, rho_j) per target as :func:`tmmri.metrics._rho_from_errors` (patient-weighted MAE / s_j on
    the rows where both errors are finite)."""

    k = err_model.shape[1]
    m, z, rho = np.full(k, np.nan), np.full(k, np.nan), np.full(k, np.nan)
    for j in range(k):
        if not (np.isfinite(sd[j]) and sd[j] > 0):
            continue
        both = np.isfinite(err_model[:, j]) & np.isfinite(err_null[:, j])
        m[j] = _patient_weighted_mae(np.where(both, err_model[:, j], np.nan), codes, n_units, weights) / sd[j]
        z[j] = _patient_weighted_mae(np.where(both, err_null[:, j], np.nan), codes, n_units, weights) / sd[j]
        if np.isfinite(m[j]) and np.isfinite(z[j]) and z[j] > 0:
            rho[j] = 1.0 - m[j] / z[j]
    return m, z, rho


def _aggregate(rho_j: np.ndarray, columns: list[int], need: int) -> float:
    vals = rho_j[columns]
    ok = np.isfinite(vals)
    return float(np.mean(vals[ok])) if ok.sum() >= need else float("nan")


def _interval(point: float, draws: np.ndarray, level: float, min_valid_fraction: float, below: bool) -> dict[str, Any]:
    draws = np.asarray(draws, dtype=float)
    valid = draws[np.isfinite(draws)]
    frac = float(valid.size / draws.size) if draws.size else 0.0
    out = {"estimate": float(point) if np.isfinite(point) else None, "n_boot_valid": int(valid.size), "valid_fraction": frac}
    if not np.isfinite(point):
        return {**out, "ci_lower": None, "ci_upper": None, "status": "NE", "reason": "fewer_than_min_estimable_targets"}
    if below:
        return {**out, "ci_lower": None, "ci_upper": None, "status": "NE", "reason": BELOW_MIN_CASES}
    if frac < min_valid_fraction:
        return {**out, "ci_lower": None, "ci_upper": None, "status": "NE", "reason": "fewer_than_min_valid_bootstrap_replicates"}
    alpha = (1.0 - level) / 2.0
    return {**out, "ci_lower": float(np.quantile(valid, alpha)), "ci_upper": float(np.quantile(valid, 1.0 - alpha)), "status": "EVALUATED",
            "reason": None}


def paired_rho(y_true: pd.DataFrame, arms: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]], sd: pd.Series, fams: Mapping[str, list[str]], *,
               units: pd.Series, clusters: pd.Series, boot: Mapping[str, Any], study: Any, min_cases: int, reference: str | None = None,
               min_valid_fraction: float | None = None) -> dict[str, Any]:
    """rho per family, rho_j and nMAE (model and null) of every arm ``{name: (prediction, null)}`` on the rows of
    ``y_true`` (measured B, OBSERVED cells), with one paired cluster bootstrap; with ``reference``, the paired
    differences ``arm - reference`` of rho, rho_j and nMAE, each target on the cases where both arms are scored (the
    same cases and the same resampled clusters; ``n_common`` per target).  Each arm's own estimate uses every row where
    it is scored (``n_scored`` per target).  Fewer than ``min_cases`` rows: point estimates with NE intervals
    (STRATUM_BELOW_MIN_CASES)."""

    targets = list(dict.fromkeys(t for fam in fams.values() for t in fam))
    idx = y_true.index
    t = y_true.reindex(columns=targets).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    sdv = np.array([float(sd.get(c, np.nan)) for c in targets])
    codes, n_units = _cluster_codes(units.reindex(idx).to_numpy(dtype=object), len(idx))
    unit_cluster, n_resample = _resampling_clusters(codes, n_units, clusters.reindex(idx).to_numpy(dtype=object))
    errs = {}
    for name, (pred, null) in arms.items():
        p = pred.reindex(index=idx, columns=targets).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        z = null.reindex(index=idx, columns=targets).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        errs[name] = (np.abs(p - t), np.abs(z - t))
    scored = {name: np.isfinite(em) & np.isfinite(en) for name, (em, en) in errs.items()}
    # Paired differences: both arms of a pair restricted, per target, to the cases where both are scored.
    common = {name: scored[name] & scored[reference] for name in arms if reference is not None and name != reference}
    series: dict[Any, tuple[np.ndarray, np.ndarray]] = dict(errs)
    for name, mask in common.items():
        for side, src in (("arm", name), ("ref", reference)):
            em, en = errs[src]
            series[(name, side)] = (np.where(mask, em, np.nan), np.where(mask, en, np.nan))
    cols = {f: [targets.index(c) for c in fam] for f, fam in fams.items()}
    need = {f: min_estimable(study, fam) for f, fam in fams.items()}
    below = len(idx) < int(min_cases)
    n_boot = 0 if below or len(idx) == 0 else int(boot["replicates"])
    level = float(boot.get("level", 0.95))
    mvf = float(study.bootstrap["fixed_model_bootstrap"]["min_valid_fraction"] if min_valid_fraction is None else min_valid_fraction)
    point = {key: _terms(em, en, sdv, codes, n_units, None) for key, (em, en) in series.items()}
    draws = {key: (np.full((n_boot, len(targets)), np.nan), np.full((n_boot, len(targets)), np.nan), np.full((n_boot, len(targets)), np.nan))
             for key in series}
    for r, ss in enumerate(np.random.SeedSequence(int(boot["seed"])).spawn(n_boot)):
        w = _cluster_draw_weights(np.random.default_rng(ss), unit_cluster, n_resample)
        for key, (em, en) in series.items():
            m, z, rho = _terms(em, en, sdv, codes, n_units, w)
            draws[key][0][r], draws[key][1][r], draws[key][2][r] = m, z, rho

    def fam_draws(name: Any, f: str) -> np.ndarray:
        return np.array([_aggregate(draws[name][2][r], cols[f], need[f]) for r in range(n_boot)], dtype=float)

    out: dict[str, Any] = {"n_cases": int(len(idx)), "n_units": int(n_units), "n_clusters": int(n_resample), "targets": targets,
                           "bootstrap": {"replicates": n_boot, "seed": int(boot["seed"]), "unit": "cluster_id", "weighting": "patient",
                                         "interval": f"percentile_{int(round(level * 100))}", "paired": True},
                           "min_cases": int(min_cases), "descriptive_only": bool(below), "arms": {}, "differences": {}}
    for name in arms:
        m, z, rho = point[name]
        out["arms"][name] = {
            "rho": {f: {**_interval(_aggregate(rho, cols[f], need[f]), fam_draws(name, f), level, mvf, below),
                        "family": fams[f], "min_estimable": need[f], "n_estimable": int(np.isfinite(rho[cols[f]]).sum())} for f in fams},
            "rho_j": {c: _interval(rho[j], draws[name][2][:, j], level, 0.0, below) for j, c in enumerate(targets)},
            "nmae": {c: _interval(m[j], draws[name][0][:, j], level, 0.0, below) for j, c in enumerate(targets)},
            "nmae_null": {c: _interval(z[j], draws[name][1][:, j], level, 0.0, below) for j, c in enumerate(targets)},
            "n_scored": {c: int(scored[name][:, j].sum()) for j, c in enumerate(targets)}}
    for name, mask in common.items():
        a_key, r_key = (name, "arm"), (name, "ref")
        m, _, rho = point[a_key]
        rm, _, rrho = point[r_key]
        diff = {"versus": reference, "direction": f"{name} minus {reference}", "rows": "cases where both arms are scored, per target",
                "n_common": {c: int(mask[:, j].sum()) for j, c in enumerate(targets)}, "rho": {}, "rho_j": {}, "nmae": {}}
        for f in fams:
            a, b = _aggregate(rho, cols[f], need[f]), _aggregate(rrho, cols[f], need[f])
            diff["rho"][f] = _interval(a - b, fam_draws(a_key, f) - fam_draws(r_key, f), level, mvf, below)
        for j, c in enumerate(targets):
            diff["rho_j"][c] = _interval(rho[j] - rrho[j], draws[a_key][2][:, j] - draws[r_key][2][:, j], level, 0.0, below)
            diff["nmae"][c] = _interval(m[j] - rm[j], draws[a_key][0][:, j] - draws[r_key][0][:, j], level, 0.0, below)
        out["differences"][name] = diff
    return out


def nmae_draws(y_true: pd.DataFrame, pred: pd.DataFrame, sd: pd.Series, targets: list[str], *, units: pd.Series, clusters: pd.Series,
               boot: Mapping[str, Any], stream: int) -> tuple[np.ndarray, np.ndarray]:
    """(point nMAE per target, replicate matrix) of one arm under the cluster bootstrap; replicate r is seeded by
    ``SeedSequence(seed, spawn_key=(stream,)).spawn(n)[r]``, so partitions with different ``stream`` are independent draws
    (unpaired partition differences)."""

    idx = y_true.index
    t = y_true.reindex(columns=targets).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    p = pred.reindex(index=idx, columns=targets).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    err = np.abs(p - t)
    sdv = np.array([float(sd.get(c, np.nan)) for c in targets])
    codes, n_units = _cluster_codes(units.reindex(idx).to_numpy(dtype=object), len(idx))
    unit_cluster, n_resample = _resampling_clusters(codes, n_units, clusters.reindex(idx).to_numpy(dtype=object))

    def one(w: np.ndarray | None) -> np.ndarray:
        return np.array([_patient_weighted_mae(err[:, j], codes, n_units, w) / sdv[j] if np.isfinite(sdv[j]) and sdv[j] > 0 else np.nan
                         for j in range(len(targets))])

    n_boot = int(boot["replicates"]) if len(idx) else 0
    reps = np.array([one(_cluster_draw_weights(np.random.default_rng(ss), unit_cluster, n_resample))
                     for ss in np.random.SeedSequence(int(boot["seed"]), spawn_key=(int(stream),)).spawn(n_boot)]).reshape(n_boot, len(targets))
    return one(None), reps


def interval_from_draws(point: float, draws: np.ndarray, level: float = 0.95, *, below: bool = False) -> dict[str, Any]:
    return _interval(point, draws, level, 0.0, below)


def target_rows(y_true: pd.DataFrame, pred: pd.DataFrame, sd: pd.Series, targets: list[str], arm_result: Mapping[str, Any] | None, *,
                units: pd.Series, **labels: Any) -> pd.DataFrame:
    """Per-target MAE, bias and 95 % limits of agreement in physical units (:func:`tmmri.metrics.per_target_errors`) with
    nMAE and rho_j and their paired-bootstrap intervals from ``arm_result``."""

    frame = per_target_errors(y_true.reindex(columns=targets), pred.reindex(index=y_true.index, columns=targets), sd,
                              cluster=units.reindex(y_true.index).to_numpy(dtype=object))
    keep = [c for c in ("target", "n", "reference_sd", "status", "reason", "mae", "mae_patient_weighted", "median_ae", "bias", "residual_sd",
                        "loa_lower", "loa_upper", "nmae_fit_sd", "nmae_patient_weighted_fit_sd") if c in frame.columns]
    frame = frame[keep].copy()
    if arm_result is not None:
        for key in ("rho_j", "nmae", "nmae_null"):
            frame[key] = [(arm_result[key].get(t) or {}).get("estimate") for t in frame["target"]]
            frame[f"{key}_ci_lower"] = [(arm_result[key].get(t) or {}).get("ci_lower") for t in frame["target"]]
            frame[f"{key}_ci_upper"] = [(arm_result[key].get(t) or {}).get("ci_upper") for t in frame["target"]]
    for name, value in reversed(list(labels.items())):
        frame.insert(0, name, value)
    return frame


def rho_rows(result: Mapping[str, Any], **labels: Any) -> pd.DataFrame:
    """Flat rows (arm, family, estimate, interval) of a :func:`paired_rho` result, with the paired differences."""

    rows = []
    for arm, entry in result["arms"].items():
        for fam, r in entry["rho"].items():
            rows.append({**labels, "arm": arm, "quantity": fam, "kind": "estimate", **{k: r.get(k) for k in ("estimate", "ci_lower", "ci_upper", "status", "reason")},
                         "n_estimable": r.get("n_estimable"), "n_cases": result["n_cases"], "n_clusters": result["n_clusters"]})
    for arm, diff in result.get("differences", {}).items():
        for fam, r in diff["rho"].items():
            rows.append({**labels, "arm": arm, "quantity": fam, "kind": f"difference_vs_{diff['versus']}",
                         **{k: r.get(k) for k in ("estimate", "ci_lower", "ci_upper", "status", "reason")}, "n_estimable": None,
                         "n_cases": result["n_cases"], "n_clusters": result["n_clusters"]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- classification signals


def weighted_auroc(score: np.ndarray, label: np.ndarray, weights: np.ndarray | None = None) -> float:
    """AUROC (ties count one half) with row multiplicities ``weights``; NaN without both classes."""

    s = np.asarray(score, dtype=float)
    y = np.asarray(label, dtype=bool)
    w = np.ones(len(s)) if weights is None else np.asarray(weights, dtype=float)
    ok = np.isfinite(s) & (w > 0)
    s, y, w = s[ok], y[ok], w[ok]
    wp, wn = w[y].sum(), w[~y].sum()
    if wp <= 0 or wn <= 0:
        return float("nan")
    values, inverse = np.unique(s, return_inverse=True)
    neg = np.bincount(inverse, weights=np.where(~y, w, 0.0), minlength=len(values))
    below = np.concatenate([[0.0], np.cumsum(neg)[:-1]])
    pos_terms = w[y] * (below[inverse[y]] + 0.5 * neg[inverse[y]])
    return float(pos_terms.sum() / (wp * wn))


def auroc_ci(score: Iterable[float], label: Iterable[Any], clusters: Iterable[Any], boot: Mapping[str, Any], *, min_cases: int) -> dict[str, Any]:
    """AUROC of ``score`` for the binary ``label`` (rows with both finite) with a cluster-bootstrap percentile interval."""

    s = np.asarray(list(score), dtype=float)
    lab = pd.to_numeric(pd.Series(list(label), dtype=object), errors="coerce").to_numpy(dtype=float)
    cl = np.asarray(list(clusters), dtype=object)
    ok = np.isfinite(s) & np.isfinite(lab)
    s, y, cl = s[ok], lab[ok] > 0.5, cl[ok]
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    base = {"n": int(ok.sum()), "n_positive": n_pos, "n_negative": n_neg}
    if n_pos == 0 or n_neg == 0:
        return {**base, "estimate": None, "ci_lower": None, "ci_upper": None, "status": "NE", "reason": "ONE_CLASS_ONLY"}
    point = weighted_auroc(s, y)
    below = int(ok.sum()) < int(min_cases)
    n_boot = 0 if below else int(boot["replicates"])
    codes, n_units = _cluster_codes(cl, len(cl))
    draws = np.array([weighted_auroc(s, y, _cluster_draw_weights(np.random.default_rng(ss), np.arange(n_units), n_units)[codes])
                      for ss in np.random.SeedSequence(int(boot["seed"])).spawn(n_boot)], dtype=float)
    return {**base, **_interval(point, draws, float(boot.get("level", 0.95)), 0.0, below)}


def binary_signal(flag: Iterable[Any], label: Iterable[Any]) -> dict[str, Any]:
    """Sensitivity, specificity and positive predictive value of a binary signal for a binary label (Wilson)."""

    f = pd.to_numeric(pd.Series(list(flag), dtype=object), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(list(label), dtype=object), errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(f) & np.isfinite(y)
    f, y = f[ok] > 0.5, y[ok] > 0.5
    tp, fp, fn, tn = int((f & y).sum()), int((f & ~y).sum()), int((~f & y).sum()), int((~f & ~y).sum())
    return {"n": int(ok.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "sensitivity": wilson(tp, tp + fn), "specificity": wilson(tn, tn + fp),
            "ppv": wilson(tp, tp + fp)}


def risk_ratio(a: int, b: int, c: int, d: int, level: float = 0.95) -> dict[str, Any]:
    """Risk ratio (a/(a+b)) / (c/(c+d)) with the Katz log interval; NE when a row is empty or a cell with an event is 0."""

    from scipy import stats

    a, b, c, d = (int(v) for v in (a, b, c, d))
    base = {"exposed_events": a, "exposed_n": a + b, "unexposed_events": c, "unexposed_n": c + d, "interval": "katz_log"}
    if a + b == 0 or c + d == 0 or a == 0 or c == 0:
        return {**base, "estimate": None, "ci_lower": None, "ci_upper": None, "status": "NE", "reason": "ZERO_CELL"}
    rr = (a / (a + b)) / (c / (c + d))
    se = float(np.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d)))
    zq = float(stats.norm.ppf(1.0 - (1.0 - level) / 2.0))
    return {**base, "estimate": rr, "ci_lower": float(np.exp(np.log(rr) - zq * se)), "ci_upper": float(np.exp(np.log(rr) + zq * se)),
            "status": "EVALUATED", "reason": None}
