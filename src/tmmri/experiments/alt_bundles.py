"""Comparative estimands of the alternative bundles (protocol-v6.0.1 item 12; docs/MANUSCRIPT_PROTOCOL_V2_2026-10-02.md D.12, J).

Readers of the C4 prediction and activation tables, the predictor metrics (the v6 code of
:func:`tmmri.classification.predictor_summary_v6` plus the multiclass Brier score) and the paired cluster-bootstrap
differences against the descriptor pipeline: the same ``SeedSequence(seed).spawn(n_boot)`` cluster draws for both
members of a pair, percentile intervals, ``NE`` below ``min_valid_fraction`` valid replicates.  The representation
estimand is the frozen v6 rho (:func:`tmmri.metrics.primary_rho` internals; patient weighting, ``cluster_id``
resampling).  Estimation only: no p-values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..classification import PREDICTOR_INTERVAL_METRICS, _replicate_predictor_metrics, predictor_summary_v6
from ..metrics import _aligned, _cluster_codes, _cluster_draw_weights, _resampling_clusters, _rho_from_errors
from .trust.calibration import LOG_LOSS_CLIP, brier, ece_equal_width

C4_KEYS = ("sample_id", "cluster_id", "partition", "model_id", "seed", "encoder_id", "is_oof", "y_true", "y_pred")
PAIRED_METRICS = (*PREDICTOR_INTERVAL_METRICS, "brier", "log_loss")
NO_SEED = -1


# ----------------------------------------------------------------------------- C4 tables


def read_c4(path: str | Path, *, model_id: str, classes: Sequence[str] | None = None, partition: str | None = None,
            seed: int | None = None, encoder: str | None = None) -> pd.DataFrame:
    """Rows of a C4 ``predictions.parquet`` / ``activations.parquet`` for one model (and partition, seed, encoder).

    ``encoder`` is ``"final"``, ``"oof"`` (``is_oof`` rows of the fold encoders) or an explicit ``encoder_id``.  The
    selection must have one row per sample_id; ``classes`` (registry order) requires every ``prob_<CLASS>`` column."""

    frame = pd.read_parquet(path)
    missing = [c for c in C4_KEYS if c not in frame.columns and c not in ("y_true", "y_pred")]
    if missing:
        raise ValueError(f"{path}: not a C4 table (missing {missing})")
    sel = frame["model_id"].astype(str) == str(model_id)
    if partition is not None:
        sel &= frame["partition"].astype(str) == str(partition)
    if seed is not None:
        sel &= pd.to_numeric(frame["seed"], errors="coerce").fillna(NO_SEED).astype(int) == int(seed)
    if encoder == "oof":
        sel &= frame["is_oof"].astype(bool)
    elif encoder is not None:
        sel &= frame["encoder_id"].astype(str) == str(encoder)
    out = frame.loc[sel].copy()
    if out["sample_id"].duplicated().any():
        dup = out.loc[out["sample_id"].duplicated(), "sample_id"].astype(str).tolist()[:3]
        raise ValueError(f"{path}: {model_id} has several rows per sample_id for this selection (e.g. {dup})")
    if classes is not None:
        absent = [f"prob_{c}" for c in classes if f"prob_{c}" not in out.columns]
        if absent:
            raise ValueError(f"{path}: missing probability columns {absent}")
    out["sample_id"] = out["sample_id"].astype(str)
    return out.sort_values("sample_id").reset_index(drop=True)


def c4_probabilities(frame: pd.DataFrame, classes: Sequence[str]) -> pd.DataFrame:
    """``prob_<CLASS>`` columns as a frame with the class names as columns (index = sample_id)."""

    out = frame[[f"prob_{c}" for c in classes]].astype(float)
    out.columns = list(classes)
    out.index = frame["sample_id"].astype(str).to_numpy()
    return out


def c4_logits(frame: pd.DataFrame, classes: Sequence[str]) -> tuple[pd.DataFrame, str]:
    """Recorded ``logit_<CLASS>`` columns when complete, else ``log p`` (same temperature; module calibration)."""

    from .trust.calibration import logits_from_probabilities

    cols = [f"logit_{c}" for c in classes]
    if all(c in frame.columns for c in cols):
        z = frame[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if np.all(np.isfinite(z)):
            return pd.DataFrame(z, index=frame["sample_id"].astype(str).to_numpy(), columns=list(classes)), "recorded_logits"
    p = c4_probabilities(frame, classes)
    return pd.DataFrame(logits_from_probabilities(p.to_numpy()), index=p.index, columns=list(classes)), "log_probabilities"


def activation_columns(frame: pd.DataFrame, prefix: str = "a_") -> list[str]:
    """Activation / feature columns (``<prefix><digits>``) in file order (the order the frozen models were fitted in)."""

    cols = [str(c) for c in frame.columns if str(c).startswith(prefix) and str(c)[len(prefix):].isdigit()]
    if not cols:
        raise ValueError(f"no activation columns with prefix {prefix!r}")
    return cols


def argmax_labels(proba: pd.DataFrame) -> np.ndarray:
    """Argmax class (ties by class order) of the finite rows; None elsewhere (as the v6 predictor)."""

    p = proba.to_numpy(dtype=float)
    out = np.full(p.shape[0], None, dtype=object)
    finite = np.all(np.isfinite(p), axis=1)
    out[finite] = np.asarray(list(proba.columns), dtype=object)[np.argmax(p[finite], axis=1)]
    return out


# ----------------------------------------------------------------------------- bootstrap helpers


def bootstrap_rows(clusters: np.ndarray, n_boot: int, seed: int) -> list[np.ndarray]:
    """Row indices of every replicate: replicate r draws clusters with ``SeedSequence(seed).spawn(n_boot)[r]``, each
    drawn cluster bringing all its rows (the draws of :func:`tmmri.classification.predictor_cluster_bootstrap`)."""

    cl = np.asarray(clusters, dtype=object)
    n = cl.size
    unit_cluster, n_resample = _resampling_clusters(np.arange(n), n, cl)
    rows = np.arange(n)
    return [np.repeat(rows, _cluster_draw_weights(np.random.default_rng(ss), unit_cluster, n_resample).astype(int))
            for ss in np.random.SeedSequence(int(seed)).spawn(int(n_boot))]


def percentile_interval(values: np.ndarray, *, level: float = 0.95, min_valid_fraction: float = 0.95) -> dict[str, Any]:
    v = np.asarray(values, dtype=float)
    valid = v[np.isfinite(v)]
    frac = float(valid.size / v.size) if v.size else 0.0
    out: dict[str, Any] = {"n_boot_valid": int(valid.size), "valid_fraction": frac}
    if not valid.size or frac < float(min_valid_fraction):
        out.update({"ci_lower": None, "ci_upper": None, "status": "NE", "reason": "fewer_than_min_valid_bootstrap_replicates"})
        return out
    alpha = (1.0 - float(level)) / 2.0
    out.update({"ci_lower": float(np.quantile(valid, alpha)), "ci_upper": float(np.quantile(valid, 1.0 - alpha)), "status": "EVALUATED", "reason": None})
    return out


def bootstrap_statistic(statistic, n_rows: int, clusters: np.ndarray, *, n_boot: int, seed: int, level: float = 0.95,
                        min_valid_fraction: float = 0.95) -> dict[str, Any]:
    """Point estimate and percentile interval of ``statistic(rows)`` (a function of row indices) under the cluster bootstrap."""

    point = statistic(np.arange(int(n_rows)))
    boot = np.array([statistic(r) for r in bootstrap_rows(clusters, n_boot, seed)], dtype=float)
    out = {"estimate": float(point) if point is not None and np.isfinite(point) else None,
           **percentile_interval(boot, level=level, min_valid_fraction=min_valid_fraction)}
    if out["estimate"] is None:
        out.update({"ci_lower": None, "ci_upper": None, "status": "NE", "reason": "metric_undefined_on_sample"})
    return out


# ----------------------------------------------------------------------------- predictor metrics


def evaluable_rows(truth: Sequence[Any], proba: pd.DataFrame, classes: Sequence[str]) -> np.ndarray:
    t = pd.Series(np.asarray(truth, dtype=object))
    known = t.map(lambda v: v is not None and not (isinstance(v, float) and np.isnan(v)) and str(v) not in ("", "nan", "None", "<NA>")).to_numpy()
    in_classes = t.astype(str).isin([str(c) for c in classes]).to_numpy()
    finite = np.all(np.isfinite(proba[list(classes)].to_numpy(dtype=float)), axis=1)
    return known & in_classes & finite


def predictor_metrics(truth: Sequence[Any], proba: pd.DataFrame, classes: Sequence[str], clusters: Sequence[Any], *, n_boot: int, seed: int,
                      n_bins: int, min_valid_fraction: float, n_jobs: int = 1, level: float = 0.95) -> dict[str, Any]:
    """v6 predictor summary (accuracy, balanced accuracy, macro-F1, macro OvR AUROC, equal-mass ECE with cluster-bootstrap
    CIs; log loss) on the argmax decisions of ``proba``, plus the multiclass Brier score (same draws) and the
    equal-width ECE companion."""

    classes = [str(c) for c in classes]
    proba = proba[classes].reset_index(drop=True)
    truth_arr = np.asarray(truth, dtype=object)
    cl = np.asarray(clusters, dtype=object)
    summary = predictor_summary_v6(truth_arr, argmax_labels(proba), proba, classes, clusters=cl, n_boot=n_boot, seed=seed, n_bins=n_bins,
                                   min_valid_fraction=min_valid_fraction, level=level, n_jobs=n_jobs)
    keep = evaluable_rows(truth_arr, proba, classes)
    if keep.any():
        t = np.array([classes.index(str(v)) for v in truth_arr[keep]], dtype=int)
        p = proba.to_numpy(dtype=float)[keep]
        summary["metrics"]["brier"] = {**bootstrap_statistic(lambda r: brier(p[r], t[r]), int(keep.sum()), cl[keep], n_boot=n_boot, seed=seed,
                                                             level=level, min_valid_fraction=min_valid_fraction), "definition": "multiclass"}
        summary["metrics"]["ece_equal_width"] = {"estimate": ece_equal_width(p, t, n_bins=n_bins), "role": "companion", "bins": int(n_bins)}
    else:
        summary["metrics"]["brier"] = {"estimate": None, "status": "NE", "reason": "no_evaluable_rows"}
        summary["metrics"]["ece_equal_width"] = {"estimate": None, "status": "NE", "reason": "no_evaluable_rows"}
    return summary


def _metric_vector(t: np.ndarray, p: np.ndarray, n_classes: int, n_bins: int, present_ref: np.ndarray) -> np.ndarray:
    pred = np.argmax(p, axis=1)
    base = _replicate_predictor_metrics(t, pred, p, n_classes, n_bins, present_ref)
    loss = float(np.mean(-np.log(np.clip(p[np.arange(t.size), t], LOG_LOSS_CLIP, 1.0)))) if t.size else np.nan
    return np.r_[base, brier(p, t) if t.size else np.nan, loss]


def paired_predictor_differences(truth: Sequence[Any], proba_reference: pd.DataFrame, proba_comparator: pd.DataFrame, classes: Sequence[str],
                                 clusters: Sequence[Any], *, n_boot: int, seed: int, n_bins: int, min_valid_fraction: float,
                                 level: float = 0.95) -> dict[str, Any]:
    """Comparator minus reference for every metric of :data:`PAIRED_METRICS` on the rows evaluable for both, with the
    same cluster draws for both members (percentile CI of the replicate differences)."""

    classes = [str(c) for c in classes]
    pr = proba_reference[classes].reset_index(drop=True)
    pc = proba_comparator[classes].reset_index(drop=True)
    truth_arr = np.asarray(truth, dtype=object)
    keep = evaluable_rows(truth_arr, pr, classes) & evaluable_rows(truth_arr, pc, classes)
    cl = np.asarray(clusters, dtype=object)[keep]
    out: dict[str, Any] = {"n_paired": int(keep.sum()), "n_clusters": int(pd.Series(cl).astype(str).nunique()), "direction": "comparator_minus_reference",
                           "bootstrap": {"replicates": int(n_boot), "seed": int(seed), "unit": "cluster_id", "paired": "same cluster draw for both"}}
    if not keep.any():
        out.update({"status": "NE", "reason": "no_paired_evaluable_rows", "metrics": {}})
        return out
    t = np.array([classes.index(str(v)) for v in truth_arr[keep]], dtype=int)
    a, b = pr.to_numpy(dtype=float)[keep], pc.to_numpy(dtype=float)[keep]
    present = np.unique(t)
    ref_pt, cmp_pt = _metric_vector(t, a, len(classes), n_bins, present), _metric_vector(t, b, len(classes), n_bins, present)
    boot = np.array([_metric_vector(t[r], b[r], len(classes), n_bins, present) - _metric_vector(t[r], a[r], len(classes), n_bins, present)
                     for r in bootstrap_rows(cl, n_boot, seed)])
    metrics = {}
    for j, m in enumerate(PAIRED_METRICS):
        diff = cmp_pt[j] - ref_pt[j]
        entry = {"reference": float(ref_pt[j]) if np.isfinite(ref_pt[j]) else None, "comparator": float(cmp_pt[j]) if np.isfinite(cmp_pt[j]) else None,
                 "difference": float(diff) if np.isfinite(diff) else None, **percentile_interval(boot[:, j], level=level, min_valid_fraction=min_valid_fraction)}
        if entry["difference"] is None:
            entry.update({"ci_lower": None, "ci_upper": None, "status": "NE", "reason": "metric_undefined_on_sample"})
        metrics[m] = entry
    out.update({"status": "EVALUATED", "reason": None, "metrics": metrics})
    return out


# ----------------------------------------------------------------------------- representation estimand (rho)


def paired_rho_difference(y_true: pd.DataFrame, pred_reference: pd.DataFrame, pred_comparator: pd.DataFrame, y_null: pd.DataFrame,
                          reference_sd: pd.Series, family: Sequence[str], *, cluster: Sequence[Any], resample_cluster: Sequence[Any],
                          n_boot: int, seed: int, min_estimable: int, level: float = 0.95, min_valid_fraction: float = 0.95) -> dict[str, Any]:
    """rho(comparator) - rho(reference) on the rows and targets where truth and both reconstructions are finite.

    Each rho_j is the v6 ``1 - MAE_model,j / MAE_mean,j`` (patient-weighted, fit-SD normalized); the difference averages
    rho_j over the targets estimable for both members (``min_estimable`` or NE), with the same cluster draw for both."""

    family = list(family)
    t, r = _aligned(y_true, pred_reference, family)
    _, c = _aligned(y_true, pred_comparator, family)
    _, z = _aligned(y_true, y_null, family)
    common = np.isfinite(t) & np.isfinite(r) & np.isfinite(c) & np.isfinite(z)
    err_r = np.where(common, np.abs(r - t), np.nan)
    err_c = np.where(common, np.abs(c - t), np.nan)
    err_z = np.where(common, np.abs(z - t), np.nan)
    sd = np.array([float(reference_sd.get(f, np.nan)) for f in family])
    codes, n_units = _cluster_codes(np.asarray(cluster, dtype=object), len(t))
    unit_cluster, n_resample = _resampling_clusters(codes, n_units, np.asarray(resample_cluster, dtype=object))

    def both(weights: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rj_r, e_r = _rho_from_errors(err_r, err_z, sd, codes, n_units, weights)
        rj_c, e_c = _rho_from_errors(err_c, err_z, sd, codes, n_units, weights)
        return rj_r, rj_c, e_r & e_c

    rj_r, rj_c, est = both(None)
    out: dict[str, Any] = {"family": family, "direction": "comparator_minus_reference", "n_rows": int(len(t)), "n_units": int(n_units),
                           "n_clusters": int(n_resample), "estimable_targets": [f for f, e in zip(family, est) if e],
                           "rows_rule": "per target, rows where the reference value and both reconstructions are finite",
                           "bootstrap": {"replicates": int(n_boot), "seed": int(seed), "unit": "cluster_id", "weighting": "patient"}}
    if est.sum() < int(min_estimable):
        out.update({"rho_reference": None, "rho_comparator": None, "difference": None, "ci_lower": None, "ci_upper": None, "status": "NE",
                    "reason": "fewer_than_min_estimable_targets", "rho_j": {}})
        return out
    boot, boot_j = [], []
    for ss in np.random.SeedSequence(int(seed)).spawn(int(n_boot)):
        w = _cluster_draw_weights(np.random.default_rng(ss), unit_cluster, n_resample)
        br, bc, be = both(w)
        boot.append(float(np.mean(bc[be]) - np.mean(br[be])) if be.sum() >= int(min_estimable) else np.nan)
        boot_j.append(bc - br)
    boot_j = np.array(boot_j, dtype=float)
    out.update({"rho_reference": float(np.mean(rj_r[est])), "rho_comparator": float(np.mean(rj_c[est])),
                "difference": float(np.mean(rj_c[est]) - np.mean(rj_r[est])),
                **percentile_interval(np.array(boot), level=level, min_valid_fraction=min_valid_fraction)})
    per_target = {}
    for j, f in enumerate(family):
        if not (np.isfinite(rj_r[j]) and np.isfinite(rj_c[j])):
            per_target[f] = {"reference": None, "comparator": None, "difference": None, "status": "NE", "reason": "target_not_estimable"}
            continue
        per_target[f] = {"reference": float(rj_r[j]), "comparator": float(rj_c[j]), "difference": float(rj_c[j] - rj_r[j]),
                         **percentile_interval(boot_j[:, j], level=level, min_valid_fraction=0.0)}
    out["rho_j"] = per_target
    return out


def seed_spread(values: dict[int, float | None]) -> dict[str, Any]:
    """Range and SD (ddof = 1) over the seeds of a metric's point estimates; no best-seed choice."""

    v = np.array([x for x in values.values() if x is not None and np.isfinite(x)], dtype=float)
    return {"n_seeds": int(v.size), "min": float(v.min()) if v.size else None, "max": float(v.max()) if v.size else None,
            "range": float(v.max() - v.min()) if v.size else None, "sd": float(v.std(ddof=1)) if v.size > 1 else None,
            "selection": "none (no best-seed choice; the primary seed defines the headline)"}
