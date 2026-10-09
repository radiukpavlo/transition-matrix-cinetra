from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy import stats

from .io import resolve_path, utc_now_iso, write_table_pair

HIGH_IMPACT_FEATURES = {
    "lv_edv_ml",
    "lv_esv_ml",
    "lv_sv_ml",
    "lv_ef_percent",
    "rv_edv_ml",
    "rv_esv_ml",
    "rv_sv_ml",
    "rv_ef_percent",
    "lv_myocardial_mass_ed_g",
    "global_wall_thickness_ed_mm",
    "global_wall_thickening_percent",
}


def benjamini_hochberg(p_values: Iterable[float], alpha: float = 0.05) -> pd.DataFrame:
    vals = pd.Series(list(p_values), dtype="float64")
    out = pd.DataFrame({"p_value": vals, "rank": np.nan, "q_value": np.nan, "rejected": False})
    valid = vals.dropna()
    if valid.empty:
        return out
    order = valid.sort_values(kind="mergesort").index.to_list()
    m = len(order)
    q = np.empty(m, dtype=float)
    prev = 1.0
    for rev_rank, idx in enumerate(reversed(order), start=1):
        rank = m - rev_rank + 1
        p = float(vals.loc[idx])
        val = min(prev, p * m / rank)
        q[rank - 1] = val
        prev = val
    for rank, idx in enumerate(order, start=1):
        out.loc[idx, "rank"] = rank
        out.loc[idx, "q_value"] = q[rank - 1]
        out.loc[idx, "rejected"] = bool(q[rank - 1] <= alpha)
    return out


def _safe_corr(x: np.ndarray, y: np.ndarray, kind: str) -> tuple[float, float]:
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, np.nan
    try:
        if kind == "pearson":
            r, p = stats.pearsonr(x, y)
        else:
            r, p = stats.spearmanr(x, y)
        return float(r), float(p)
    except Exception:
        return np.nan, np.nan


def metric_summary(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[mask]
    predicted = predicted[mask]
    if len(observed) == 0:
        return {k: np.nan for k in ["rmse", "mae", "median_absolute_error", "pearson_r", "pearson_p", "spearman_r", "spearman_p", "bias", "residual_sd", "loa_lower", "loa_upper", "n"]}
    residual = predicted - observed
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    medae = float(np.median(np.abs(residual)))
    pr, pp = _safe_corr(observed, predicted, "pearson")
    sr, sp = _safe_corr(observed, predicted, "spearman")
    bias = float(np.mean(residual))
    sd = float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0
    return {
        "rmse": rmse,
        "mae": mae,
        "median_absolute_error": medae,
        "pearson_r": pr,
        "pearson_p": pp,
        "spearman_r": sr,
        "spearman_p": sp,
        "bias": bias,
        "residual_sd": sd,
        "loa_lower": bias - 1.96 * sd,
        "loa_upper": bias + 1.96 * sd,
        "n": float(len(observed)),
    }


def percentile_bootstrap_ci(values: np.ndarray, statistic: Callable[[np.ndarray], float], iterations: int = 1000, seed: int = 20260611, confidence: float = 0.95) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    stats_arr = np.empty(iterations, dtype=float)
    n = len(values)
    for i in range(iterations):
        sample = values[rng.integers(0, n, size=n)]
        stats_arr[i] = statistic(sample)
    alpha = (1.0 - confidence) / 2.0
    return float(np.nanquantile(stats_arr, alpha)), float(np.nanquantile(stats_arr, 1.0 - alpha))


def bca_bootstrap_ci(values: np.ndarray, statistic: Callable[[np.ndarray], float], iterations: int = 1000, seed: int = 20260611, confidence: float = 0.95) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    theta_hat = float(statistic(values))
    boot = np.empty(iterations, dtype=float)
    for i in range(iterations):
        boot[i] = statistic(values[rng.integers(0, n, size=n)])
    boot = boot[np.isfinite(boot)]
    if boot.size < 10:
        return np.nan, np.nan
    prop_less = np.clip(np.mean(boot < theta_hat), 1e-6, 1 - 1e-6)
    z0 = stats.norm.ppf(prop_less)
    jack = np.empty(n, dtype=float)
    for i in range(n):
        jack[i] = statistic(np.delete(values, i))
    jack_mean = np.nanmean(jack)
    numerator = np.nansum((jack_mean - jack) ** 3)
    denominator = 6.0 * (np.nansum((jack_mean - jack) ** 2) ** 1.5)
    acceleration = 0.0 if denominator == 0 or not np.isfinite(denominator) else float(numerator / denominator)
    alpha = (1.0 - confidence) / 2.0
    z_low = stats.norm.ppf(alpha)
    z_high = stats.norm.ppf(1.0 - alpha)

    def adjusted_alpha(z_alpha: float) -> float:
        denom = 1.0 - acceleration * (z0 + z_alpha)
        if denom == 0:
            return stats.norm.cdf(z0 + z_alpha)
        return float(stats.norm.cdf(z0 + (z0 + z_alpha) / denom))

    a1 = np.clip(adjusted_alpha(z_low), 0.0, 1.0)
    a2 = np.clip(adjusted_alpha(z_high), 0.0, 1.0)
    if not np.isfinite(a1) or not np.isfinite(a2) or a1 >= a2:
        return percentile_bootstrap_ci(values, statistic, iterations=iterations, seed=seed, confidence=confidence)
    return float(np.nanquantile(boot, a1)), float(np.nanquantile(boot, a2))


def evaluate_residuals(
    residuals: pd.DataFrame,
    experiment_id: str,
    bootstrap_iterations: int = 1000,
    bootstrap_seed: int = 20260611,
    fdr_alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    group_cols = ["dataset_source", "split", "feature_name"]
    for keys, group in residuals.groupby(group_cols, dropna=False):
        dataset, split, feature = keys
        observed = pd.to_numeric(group["observed_value"], errors="coerce").to_numpy(dtype=float)
        predicted = pd.to_numeric(group["predicted_value"], errors="coerce").to_numpy(dtype=float)
        summary = metric_summary(observed, predicted)
        valid = np.isfinite(observed) & np.isfinite(predicted)
        residual = (predicted - observed)[valid]
        abs_err = np.abs(residual)
        rmse_ci_low, rmse_ci_high = bca_bootstrap_ci(residual, lambda x: np.sqrt(np.nanmean(x**2)), iterations=bootstrap_iterations, seed=bootstrap_seed, confidence=0.95)
        mae_ci_low, mae_ci_high = bca_bootstrap_ci(abs_err, np.nanmean, iterations=bootstrap_iterations, seed=bootstrap_seed, confidence=0.95)
        bias_ci_low, bias_ci_high = bca_bootstrap_ci(residual, np.nanmean, iterations=bootstrap_iterations, seed=bootstrap_seed, confidence=0.95)
        train_sd = np.nanstd(observed)
        summary.update(
            {
                "dataset_source": dataset,
                "split": split,
                "feature_name": feature,
                "experiment_id": experiment_id,
                "rmse_ci_lower": rmse_ci_low,
                "rmse_ci_upper": rmse_ci_high,
                "mae_ci_lower": mae_ci_low,
                "mae_ci_upper": mae_ci_high,
                "bias_ci_lower": bias_ci_low,
                "bias_ci_upper": bias_ci_high,
                "ci_method": "bca_bootstrap",
                "bootstrap_iterations": int(bootstrap_iterations),
                "bootstrap_seed": int(bootstrap_seed),
                "normalized_rmse_by_split_sd": summary["rmse"] / train_sd if train_sd and np.isfinite(train_sd) else np.nan,
                "manual_review_rows": int(group.get("manual_review_flag", pd.Series(False, index=group.index)).astype(bool).sum()),
            }
        )
        rows.append(summary)
    metrics = pd.DataFrame(rows)
    if not metrics.empty:
        metrics["pearson_q"] = np.nan
        metrics["pearson_fdr_rejected"] = False
        for keys, idx in metrics.groupby(["dataset_source", "split"], dropna=False).groups.items():
            bh = benjamini_hochberg(metrics.loc[idx, "pearson_p"].to_list(), alpha=fdr_alpha)
            metrics.loc[list(idx), "pearson_q"] = bh["q_value"].to_numpy()
            metrics.loc[list(idx), "pearson_fdr_rejected"] = bh["rejected"].to_numpy(dtype=bool)
    failure_rows = []
    for _, row in metrics.iterrows():
        reasons = []
        if not np.isfinite(row.get("pearson_r", np.nan)):
            reasons.append("correlation_not_testable")
        elif not bool(row.get("pearson_fdr_rejected", False)):
            reasons.append("correlation_not_significant_after_fdr")
        if np.isfinite(row.get("normalized_rmse_by_split_sd", np.nan)) and row["normalized_rmse_by_split_sd"] > 1.0:
            reasons.append("normalized_rmse_gt_1sd")
        if int(row.get("n", 0)) < 10:
            reasons.append("low_evaluation_n")
        if reasons:
            failure_rows.append(
                {
                    "dataset_source": row["dataset_source"],
                    "split": row["split"],
                    "feature_name": row["feature_name"],
                    "failure_reasons": ";".join(reasons),
                    "experiment_id": experiment_id,
                }
            )
    failures = pd.DataFrame(failure_rows)
    return metrics, failures


def write_transition_evaluation_report(
    metrics: pd.DataFrame,
    failures: pd.DataFrame,
    experiment_id: str,
    output_path: str | Path | None = None,
) -> Path:
    path = resolve_path(output_path or f"artifacts/reports/transition/transition_evaluation_{experiment_id}.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Transition Evaluation Report - {experiment_id}",
        "",
        "Clinical status: research use only. This project is not clinically validated for diagnosis, triage, treatment selection, or patient management.",
        "",
        "Transition convention: row-sample right multiplication, `B_hat = zscore(A) @ T + mu_B`.",
        "",
        "This report evaluates numerical reconstruction fidelity only. Transition weights are not causal, and poor or high-conflict features must not be converted into diagnostic claims.",
        "",
        f"Generated at UTC: {utc_now_iso()}",
        "",
    ]
    if metrics.empty:
        lines.append("No metrics were computed.")
    else:
        lines.extend(["## Aggregate metrics by dataset/split", ""])
        agg = metrics.groupby(["dataset_source", "split"], dropna=False).agg(rmse_mean=("rmse", "mean"), mae_mean=("mae", "mean"), features=("feature_name", "count")).reset_index()
        lines.append(agg.to_markdown(index=False))
        lines.extend(["", "## High-impact feature metrics", ""])
        hi = metrics[metrics["feature_name"].isin(HIGH_IMPACT_FEATURES)].copy()
        cols = ["dataset_source", "split", "feature_name", "n", "rmse", "mae", "pearson_r", "pearson_q", "bias", "loa_lower", "loa_upper"]
        if len(hi):
            lines.append(hi[cols].sort_values(["dataset_source", "split", "feature_name"]).to_markdown(index=False))
        else:
            lines.append("No high-impact features were present in the target list.")
        lines.extend(["", "## Failed or unstable features", ""])
        if failures.empty:
            lines.append("No feature failures were flagged by the implemented criteria.")
        else:
            lines.append(failures.head(100).to_markdown(index=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_transition_evaluation_artifacts(metrics: pd.DataFrame, failures: pd.DataFrame, experiment_id: str) -> list[Path]:
    artifacts: list[Path] = []
    artifacts.extend(write_table_pair(metrics, f"artifacts/reports/transition/transition_metrics_{experiment_id}.parquet", f"artifacts/reports/transition/transition_metrics_{experiment_id}.csv"))
    artifacts.extend(write_table_pair(failures, f"artifacts/reports/transition/transition_feature_failures_{experiment_id}.parquet", f"artifacts/reports/transition/transition_feature_failures_{experiment_id}.csv"))
    artifacts.append(write_transition_evaluation_report(metrics, failures, experiment_id))
    return artifacts


# ===============================================================================================================
# Protocol v6 metrics (SPECIFICATION.md 6.0.0, Sections 3.3 and 13; Decision D18; configs/bootstrap.yaml).
# New functions only; evaluate_residuals keeps its legacy (misnamed) split-SD normalization for the legacy flows.
# In v6 every normalization uses the fit-partition SD passed as ``reference_sd``, never the evaluated split's SD.
# ===============================================================================================================
from typing import Any, Sequence


def _aligned(y_true: pd.DataFrame, y_pred: pd.DataFrame, columns: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    t = y_true[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    p = y_pred[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if t.shape != p.shape:
        raise ValueError("y_true and y_pred must be row-aligned")
    return t, p


def _cluster_codes(cluster: pd.Series | np.ndarray | None, n: int) -> tuple[np.ndarray, int]:
    if cluster is None:
        return np.arange(n), n
    labels = pd.Series(np.asarray(cluster, dtype=object)).astype(str)
    codes, uniques = pd.factorize(labels, sort=True)
    return codes.astype(int), int(len(uniques))


def _resampling_clusters(codes: np.ndarray, n_units: int, resample_cluster: pd.Series | np.ndarray | None) -> tuple[np.ndarray, int]:
    """Bootstrap cluster of every weighting unit (``codes`` from :func:`_cluster_codes`) and the number of clusters.

    ``resample_cluster`` is row-aligned (the ``cluster_id`` of the split manifest: a declared duplicate-image pair is
    one cluster, every other case its own).  Clusters are numbered by their smallest unit code, so with singleton
    clusters every replicate draws exactly the units of the unit (case) bootstrap: the cluster bootstrap reduces to the
    case bootstrap.  A unit that spans two clusters is an error."""
    if resample_cluster is None:
        return np.arange(n_units), int(n_units)
    labels = pd.Series(np.asarray(resample_cluster, dtype=object)).astype(str).to_numpy()
    if labels.size != np.asarray(codes).size:
        raise ValueError("resample_cluster must be row-aligned with the data")
    unit_label = np.full(n_units, None, dtype=object)
    for unit, label in zip(np.asarray(codes, dtype=int), labels):
        if unit_label[unit] is None:
            unit_label[unit] = label
        elif unit_label[unit] != label:
            raise ValueError("a weighting unit spans two bootstrap clusters")
    index: dict[str, int] = {}
    for label in unit_label:
        index.setdefault(label, len(index))
    return np.array([index[label] for label in unit_label], dtype=int), len(index)


def _cluster_draw_weights(rng: np.random.Generator, unit_cluster: np.ndarray, n_resample: int) -> np.ndarray:
    """Per-unit bootstrap multiplicities of one cluster draw (each drawn cluster brings all its units)."""
    counts = np.bincount(rng.integers(0, n_resample, size=n_resample), minlength=n_resample).astype(float)
    return counts[unit_cluster]


def cluster_count(cluster: pd.Series | np.ndarray | None, n: int) -> int:
    """Independent clusters among ``n`` rows (``n`` when ``cluster`` is None)."""
    return int(n) if cluster is None else int(pd.Series(np.asarray(cluster, dtype=object)).astype(str).nunique())


def _patient_weighted_mae(abs_err: np.ndarray, codes: np.ndarray, n_clusters: int, weights: np.ndarray | None = None) -> float:
    """Mean over clusters of the within-cluster mean absolute error (observed rows only); ``weights`` are bootstrap
    multiplicities per cluster."""
    ok = np.isfinite(abs_err)
    if not ok.any():
        return float("nan")
    sums = np.bincount(codes[ok], weights=abs_err[ok], minlength=n_clusters)
    counts = np.bincount(codes[ok], minlength=n_clusters)
    has = counts > 0
    per = np.zeros(n_clusters)
    per[has] = sums[has] / counts[has]
    w = np.ones(n_clusters) if weights is None else np.asarray(weights, dtype=float)
    w = np.where(has, w, 0.0)
    total = w.sum()
    return float((per * w).sum() / total) if total > 0 else float("nan")


def per_target_errors(y_true: pd.DataFrame, y_pred: pd.DataFrame, reference_sd: pd.Series, *, cluster: pd.Series | np.ndarray | None = None) -> pd.DataFrame:
    """Per-target errors on rows where both reference and prediction are finite, in original units, with MAE and
    RMSE also divided by the fit-partition SD (``reference_sd``; configs/transition.yaml evaluation.normalization)."""
    columns = [c for c in y_pred.columns if c in y_true.columns and c != "outer_fold"]
    t, p = _aligned(y_true, y_pred, columns)
    codes, n_clusters = _cluster_codes(cluster, len(t))
    rows = []
    for j, col in enumerate(columns):
        ok = np.isfinite(t[:, j]) & np.isfinite(p[:, j])
        sd = float(reference_sd.get(col, np.nan)) if hasattr(reference_sd, "get") else float("nan")
        base = {"target": col, "n": int(ok.sum()), "reference_sd": sd, "normalization": "fit_partition_sd"}
        if not ok.any():
            rows.append({**base, "status": "NE", "reason": "no_observed_rows"})
            continue
        s = metric_summary(t[ok, j], p[ok, j])
        abs_err = np.where(ok, np.abs(p[:, j] - t[:, j]), np.nan)
        pw_mae = _patient_weighted_mae(abs_err, codes, n_clusters)
        valid_sd = np.isfinite(sd) and sd > 0
        rows.append(
            {
                **base,
                "status": "EVALUATED",
                "reason": None,
                "mae": s["mae"],
                "mae_patient_weighted": pw_mae,
                "rmse": s["rmse"],
                "median_ae": s["median_absolute_error"],
                "bias": s["bias"],
                "residual_sd": s["residual_sd"],
                "pearson_r": s["pearson_r"],
                "pearson_p": s["pearson_p"],
                "spearman_rho": s["spearman_r"],
                "spearman_p": s["spearman_p"],
                "loa_lower": s["loa_lower"],
                "loa_upper": s["loa_upper"],
                "nmae_fit_sd": s["mae"] / sd if valid_sd else np.nan,
                "nmae_patient_weighted_fit_sd": pw_mae / sd if valid_sd else np.nan,
                "nrmse_fit_sd": s["rmse"] / sd if valid_sd else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _rho_from_errors(err_model: np.ndarray, err_null: np.ndarray, sd: np.ndarray, codes: np.ndarray, n_clusters: int, weights: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """rho_j per target and an estimability mask for one (possibly bootstrap-weighted) sample."""
    l = err_model.shape[1]
    rho = np.full(l, np.nan)
    for j in range(l):
        if not (np.isfinite(sd[j]) and sd[j] > 0):
            continue
        both = np.isfinite(err_model[:, j]) & np.isfinite(err_null[:, j])
        m = _patient_weighted_mae(np.where(both, err_model[:, j], np.nan), codes, n_clusters, weights) / sd[j]
        z = _patient_weighted_mae(np.where(both, err_null[:, j], np.nan), codes, n_clusters, weights) / sd[j]
        if np.isfinite(m) and np.isfinite(z) and z > 0:
            rho[j] = 1.0 - m / z
    return rho, np.isfinite(rho)


def _boot_chunk(err_model: np.ndarray, err_null: np.ndarray, sd: np.ndarray, codes: np.ndarray, n_clusters: int, seeds: list[np.random.SeedSequence], min_estimable: int,
                unit_cluster: np.ndarray | None = None, n_resample: int | None = None) -> list[tuple[float, np.ndarray]]:
    out = []
    unit_cluster = np.arange(n_clusters) if unit_cluster is None else unit_cluster
    n_resample = n_clusters if n_resample is None else int(n_resample)
    for ss in seeds:
        rng = np.random.default_rng(ss)
        weights = _cluster_draw_weights(rng, unit_cluster, n_resample)
        rho_j, est = _rho_from_errors(err_model, err_null, sd, codes, n_clusters, weights)
        out.append((float(np.mean(rho_j[est])) if est.sum() >= min_estimable else float("nan"), rho_j))
    return out


def primary_rho(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    y_null: pd.DataFrame,
    reference_sd: pd.Series,
    family: list[str],
    n_boot: int,
    seed: int,
    cluster: pd.Series | np.ndarray | None,
    min_estimable: int,
    *,
    level: float = 0.95,
    min_valid_fraction: float = 0.95,
    n_jobs: int = 1,
    resample_cluster: pd.Series | np.ndarray | None = None,
) -> dict[str, Any]:
    """Primary estimand rho = mean over estimable family targets of rho_j = 1 - MAE_ridge,j / MAE_mean,j (each MAE
    patient-weighted and divided by the fit-partition SD) with a paired cluster bootstrap (the same cluster draw for
    ridge and mean; replicate r seeded by ``SeedSequence(seed).spawn(n_boot)[r]``, so the result does not depend on
    ``n_jobs``) and a percentile interval. Fewer than ``min_estimable`` estimable targets withdraws the aggregate (NE).

    ``cluster`` is the weighting unit (patient; each weighs once in the point estimate) and ``resample_cluster`` the
    bootstrap cluster (``cluster_id``; default: the weighting unit).  A drawn cluster brings all its units, so a declared
    duplicate-image pair is resampled together; with singleton clusters the result equals the patient bootstrap."""
    t, p = _aligned(y_true, y_pred, family)
    _, z = _aligned(y_true, y_null, family)
    err_model = np.abs(p - t)
    err_null = np.abs(z - t)
    sd = np.array([float(reference_sd.get(c, np.nan)) for c in family])
    codes, n_clusters = _cluster_codes(cluster, len(t))
    unit_cluster, n_resample = _resampling_clusters(codes, n_clusters, resample_cluster)
    rho_j, est = _rho_from_errors(err_model, err_null, sd, codes, n_clusters, None)
    result: dict[str, Any] = {
        "family": list(family),
        "rho_j": {c: (float(v) if np.isfinite(v) else None) for c, v in zip(family, rho_j)},
        "estimable_targets": [c for c, e in zip(family, est) if e],
        "n_estimable": int(est.sum()),
        "min_estimable": int(min_estimable),
        "n_units": int(n_clusters),
        "n_clusters": int(n_resample),
        "resampling_unit": "patient" if resample_cluster is None else "cluster_id",
        "n_boot": int(n_boot),
        "seed": int(seed),
        "interval": f"percentile_{int(round(level * 100))}",
        "normalization": "fit_partition_sd",
        "weighting": "patient",
    }
    if est.sum() < min_estimable:
        result.update({"rho": None, "ci_lower": None, "ci_upper": None, "status": "NE", "reason": "fewer_than_min_estimable_targets", "n_boot_valid": 0, "valid_fraction": 0.0})
        return result
    rho = float(np.mean(rho_j[est]))
    children = np.random.SeedSequence(int(seed)).spawn(int(n_boot))
    n_chunks = max(1, min(int(n_boot), max(1, int(n_jobs)) * 4))
    chunks = [c for c in np.array_split(np.arange(int(n_boot)), n_chunks) if c.size]
    if int(n_jobs) == 1:
        parts = [_boot_chunk(err_model, err_null, sd, codes, n_clusters, [children[i] for i in c], min_estimable, unit_cluster, n_resample) for c in chunks]
    else:
        from joblib import Parallel, delayed

        parts = Parallel(n_jobs=int(n_jobs), backend="loky")(delayed(_boot_chunk)(err_model, err_null, sd, codes, n_clusters, [children[i] for i in c], min_estimable,
                                                                                  unit_cluster, n_resample) for c in chunks)
    boot = np.array([r for part in parts for r, _ in part], dtype=float)
    boot_j = np.array([rj for part in parts for _, rj in part], dtype=float)
    valid = np.isfinite(boot)
    frac = float(valid.mean()) if boot.size else 0.0
    alpha = (1.0 - level) / 2.0
    result.update({"rho": rho, "n_boot_valid": int(valid.sum()), "valid_fraction": frac})
    if frac < min_valid_fraction:
        result.update({"ci_lower": None, "ci_upper": None, "status": "NE", "reason": "fewer_than_min_valid_bootstrap_replicates"})
    else:
        result.update({"ci_lower": float(np.quantile(boot[valid], alpha)), "ci_upper": float(np.quantile(boot[valid], 1.0 - alpha)), "status": "EVALUATED", "reason": None})
    ci_j = {}
    for j, c in enumerate(family):
        col = boot_j[:, j]
        col = col[np.isfinite(col)]
        ci_j[c] = (float(np.quantile(col, alpha)), float(np.quantile(col, 1.0 - alpha))) if col.size else (None, None)
    result["rho_j_ci"] = ci_j
    return result


def fixed_sequence_test(results: list[dict[str, Any]], sesoi: float) -> list[dict[str, Any]]:
    """Fixed-sequence (hierarchical) testing without alpha splitting: hypothesis k is tested only if every earlier
    hypothesis in the hierarchy passed; it passes when its CI lower bound exceeds ``sesoi``. Items with
    ``in_hierarchy: False`` (sensitivities such as the 220-case stratum) are evaluated on their own and never gate."""
    out = []
    gate_open = True
    for res in results:
        item = dict(res)
        lb = item.get("ci_lower")
        in_hierarchy = bool(item.get("in_hierarchy", True))
        tested = gate_open or not in_hierarchy
        if not tested:
            item.update({"tested": False, "passed": None, "status": "NOT_TESTED", "plain_superiority": None})
        elif lb is None or not np.isfinite(float(lb)):
            item.update({"tested": True, "passed": False, "status": "NE", "plain_superiority": None})
        else:
            passed = float(lb) > float(sesoi)
            item.update({"tested": True, "passed": passed, "status": "PASSED" if passed else "FAILED", "plain_superiority": float(lb) > 0.0})
        item["sesoi"] = float(sesoi)
        if in_hierarchy and tested and not item["passed"]:
            gate_open = False
        out.append(item)
    return out


def wilson_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials (defined for k = 0 and k = n); (nan, nan) when n = 0."""
    k, n = int(k), int(n)
    if n <= 0:
        return float("nan"), float("nan")
    if not 0 <= k <= n:
        raise ValueError("wilson_interval requires 0 <= k <= n")
    z = float(stats.norm.ppf(1.0 - (1.0 - level) / 2.0))
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if k == 0 else float(max(0.0, centre - half))
    hi = 1.0 if k == n else float(min(1.0, centre + half))
    return lo, hi


def bh_within_families(tests: pd.DataFrame, *, q: float, families: Sequence[str] | None = None, family_col: str = "family", p_col: str = "p_value") -> pd.DataFrame:
    """Benjamini-Hochberg within each prespecified family (configs/bootstrap.yaml multiplicity); a merged ranking
    across families is forbidden, and unknown family names are rejected."""
    out = tests.copy()
    if families is not None:
        unknown = sorted(set(out[family_col].astype(str)) - set(families))
        if unknown:
            raise ValueError(f"Tests outside the prespecified families: {unknown}")
    out["q_value"] = np.nan
    out["rejected"] = False
    for _, idx in out.groupby(family_col, sort=True).groups.items():
        bh = benjamini_hochberg(out.loc[idx, p_col].to_list(), alpha=q)
        out.loc[idx, "q_value"] = bh["q_value"].to_numpy()
        out.loc[idx, "rejected"] = bh["rejected"].to_numpy(dtype=bool)
    return out


def route_metrics(routes: pd.DataFrame, reference: pd.Series | None, d_model: pd.Series | None, *, cluster: pd.Series | np.ndarray | None = None) -> dict[str, Any]:
    """Route metrics (E18-E20) over the attempted inputs of ``routes`` (tmmri.inference.route output).

    ``reference`` and ``d_model`` are aligned with ``routes`` rows (index or sample_id). Covered fidelity compares the
    accepted label with d_model; covered reference accuracy uses rows with a known reference only; the audit tax
    compares rule and base (d_model) reference accuracy on exactly the same accepted, reference-known subset.
    Undefined covered metrics are None (NE).
    """
    n = int(len(routes))
    route = routes["route"].astype(str).to_numpy()
    accepted = route == "EXACT_ACCEPT"
    label = routes["label"].astype(object).to_numpy()

    def _align(s: pd.Series | None) -> np.ndarray:
        if s is None:
            return np.full(n, None, dtype=object)
        s = pd.Series(s)
        if "sample_id" in routes.columns and set(routes["sample_id"].astype(str)) <= set(s.index.astype(str)):
            s.index = s.index.astype(str)
            s = s.reindex(routes["sample_id"].astype(str))
        return s.astype(object).where(s.notna(), None).to_numpy()

    ref = _align(reference)
    dm = _align(d_model)
    n_acc = int(accepted.sum())
    fired = routes["n_fired"].to_numpy(dtype=int) > 0 if "n_fired" in routes.columns else accepted
    qc = route == "QC_ABSTAIN"
    out: dict[str, Any] = {
        "n_attempted": n,
        "n_clusters": cluster_count(cluster, n),
        "n_qc_abstain": int(qc.sum()),
        "n_processed": int((~qc).sum()),
        "n_accepted": n_acc,
        "activation": float(fired.mean()) if n else None,
        "accepted_coverage": n_acc / n if n else None,
        "abstention": 1.0 - n_acc / n if n else None,
        "conflict_rate": float((route == "CONFLICT_ABSTAIN").mean()) if n else None,
        "conflict_encounter_rate": float(routes["conflict_encountered"].astype(bool).mean()) if n and "conflict_encountered" in routes.columns else None,
        "soft_candidates_rate": float((route == "SOFT_CANDIDATES_ONLY").mean()) if n else None,
        "routes": {r: int((route == r).sum()) for r in sorted(set(route))},
        "abstention_by_reason": {},
    }
    for code, count in routes.loc[~accepted, "reason_code"].astype(str).value_counts().sort_index().items():
        out["abstention_by_reason"][code] = {"n": int(count), "rate": count / n}
    fid_rows = accepted & np.array([v is not None for v in dm])
    out["covered_fidelity"] = float(np.mean([label[i] == dm[i] for i in np.flatnonzero(fid_rows)])) if fid_rows.any() else None
    out["faithful_coverage"] = float(sum(label[i] == dm[i] for i in np.flatnonzero(fid_rows)) / n) if n else None
    ref_rows = accepted & np.array([v is not None for v in ref])
    out["n_accepted_reference_known"] = int(ref_rows.sum())
    out["covered_reference_accuracy"] = float(np.mean([label[i] == ref[i] for i in np.flatnonzero(ref_rows)])) if ref_rows.any() else None
    both = ref_rows & np.array([v is not None for v in dm])
    if both.any():
        idx = np.flatnonzero(both)
        rule_acc = float(np.mean([label[i] == ref[i] for i in idx]))
        base_acc = float(np.mean([dm[i] == ref[i] for i in idx]))
        out["audit_tax"] = {"n_same_subset": int(idx.size), "rule_reference_accuracy": rule_acc, "base_reference_accuracy": base_acc,
                            "difference_rule_minus_base": rule_acc - base_acc, "accepted_coverage": out["accepted_coverage"]}
    else:
        out["audit_tax"] = None
    out["route_has_empirical_support"] = bool(n_acc > 0)
    return out


def stratified_route_metrics(routes: pd.DataFrame, reference: pd.Series | None, d_model: pd.Series | None, strata: pd.Series, *, min_cases: int,
                             cluster: pd.Series | np.ndarray | None = None) -> pd.DataFrame:
    """Route metrics per stratum (vendor, centre, release split, pathology, mms100_member); strata below
    ``min_cases`` (configs/bootstrap.yaml subgroup_reporting_thresholds) are flagged descriptive-only.  ``cluster``
    (row-aligned ``cluster_id``) adds the number of independent clusters per stratum."""
    s = pd.Series(np.asarray(strata, dtype=object), index=routes.index).astype("string").fillna("<missing>")
    ref = None if reference is None else pd.Series(np.asarray(reference, dtype=object), index=routes.index)
    dm = None if d_model is None else pd.Series(np.asarray(d_model, dtype=object), index=routes.index)
    cl = None if cluster is None else pd.Series(np.asarray(cluster, dtype=object), index=routes.index)
    rows = []
    for value in sorted(s.unique()):
        idx = s.index[s == value]
        sub = routes.loc[idx].drop(columns=["sample_id"], errors="ignore")
        m = route_metrics(sub, None if ref is None else ref.loc[idx], None if dm is None else dm.loc[idx], cluster=None if cl is None else cl.loc[idx])
        rows.append({"stratum": value, "n": int(len(idx)), "n_clusters": m["n_clusters"], "descriptive_only": bool(len(idx) < int(min_cases)), "min_cases": int(min_cases),
                     "accepted_coverage": m["accepted_coverage"], "covered_fidelity": m["covered_fidelity"],
                     "covered_reference_accuracy": m["covered_reference_accuracy"], "conflict_rate": m["conflict_rate"], "routes": m["routes"]})
    return pd.DataFrame(rows)


def stratified_errors(y_true: pd.DataFrame, y_pred: pd.DataFrame, reference_sd: pd.Series, strata: pd.Series, *, min_cases: int,
                      cluster: pd.Series | np.ndarray | None = None) -> pd.DataFrame:
    """Per-target errors per stratum; strata below ``min_cases`` are descriptive-only (point estimates, no tests)."""
    s = pd.Series(np.asarray(strata, dtype=object), index=y_true.index).astype("string").fillna("<missing>")
    cl = None if cluster is None else pd.Series(np.asarray(cluster, dtype=object), index=y_true.index)
    frames = []
    for value in sorted(s.unique()):
        idx = s.index[s == value]
        e = per_target_errors(y_true.loc[idx], y_pred.loc[idx], reference_sd)
        e.insert(0, "stratum", value)
        e["stratum_n"] = int(len(idx))
        e["stratum_clusters"] = cluster_count(None if cl is None else cl.loc[idx], len(idx))
        e["descriptive_only"] = bool(len(idx) < int(min_cases))
        frames.append(e)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------------------------------------------
# Secondary tests and the within-family Benjamini-Hochberg step (SPECIFICATION.md 13.3; configs/bootstrap.yaml
# multiplicity.tests and family_members). Effect sizes precede p-values; NE rows are kept, never dropped.
# ---------------------------------------------------------------------------------------------------------------
BELOW_MIN_CASES = "STRATUM_BELOW_MIN_CASES"
TEST_COLUMNS = ["family", "partition", "comparison", "stratum_variable", "stratum", "target", "variant", "n", "effect",
                "effect_normalized", "statistic", "p_value", "test", "status", "reason", "n_clusters"]


def family_name(short: str, families: Sequence[str]) -> str:
    """The configured family whose name starts with ``short`` + ``_`` (for example ``F1`` -> ``F1_reconstruction...``)."""
    matches = [f for f in families if str(f).split("_", 1)[0] == short]
    if len(matches) != 1:
        raise KeyError(f"family {short!r} is not configured exactly once in configs/bootstrap.yaml multiplicity.families")
    return str(matches[0])


def paired_error_tests(y_true: pd.DataFrame, y_primary: pd.DataFrame, y_comparator: pd.DataFrame, targets: Sequence[str],
                       reference_sd: pd.Series, *, family: str, partition: str, comparison: str, stratum_variable: str = "all",
                       stratum: str = "all", min_cases: int | None = None, zero_method: str = "wilcox",
                       cluster: pd.Series | np.ndarray | None = None) -> pd.DataFrame:
    """Per-target two-sided Wilcoxon signed-rank test of d_i = |error_comparator,i| - |error_primary,i| (positive: the
    primary transition is better) on the rows with a finite reference and both predictions. ``effect`` is the MAE
    difference (comparator minus primary, original units) and ``effect_normalized`` divides it by the fit-partition SD.
    A stratum below ``min_cases`` keeps its point estimate with status NE and reason STRATUM_BELOW_MIN_CASES.

    With ``cluster`` (row-aligned ``cluster_id``) the test is cluster-level: it ranks the per-cluster means of d_i, so a
    declared duplicate-image pair contributes one observation (identical to the case-level test for singletons)."""
    rows = []
    cl = None if cluster is None else pd.Series(np.asarray(cluster, dtype=object)).astype(str).to_numpy()
    for t in targets:
        y = pd.to_numeric(y_true[t], errors="coerce").to_numpy(dtype=float) if t in y_true.columns else np.full(len(y_true), np.nan)
        p = pd.to_numeric(y_primary[t], errors="coerce").to_numpy(dtype=float) if t in y_primary.columns else np.full(len(y_true), np.nan)
        c = pd.to_numeric(y_comparator[t], errors="coerce").to_numpy(dtype=float) if t in y_comparator.columns else np.full(len(y_true), np.nan)
        ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(c)
        sd = float(reference_sd.get(t, np.nan)) if hasattr(reference_sd, "get") else float("nan")
        base = {"family": family, "partition": partition, "comparison": comparison, "stratum_variable": stratum_variable,
                "stratum": str(stratum), "target": t, "variant": None, "n": int(ok.sum()), "test": f"wilcoxon_signed_rank_{zero_method}_two_sided",
                "n_clusters": cluster_count(None if cl is None else cl[ok], int(ok.sum()))}
        if not ok.any():
            rows.append({**base, "effect": np.nan, "effect_normalized": np.nan, "statistic": np.nan, "p_value": np.nan,
                         "status": "NE", "reason": "no_paired_rows"})
            continue
        d = np.abs(c[ok] - y[ok]) - np.abs(p[ok] - y[ok])
        effect = float(np.mean(d))
        eff_n = effect / sd if np.isfinite(sd) and sd > 0 else np.nan
        if min_cases is not None and ok.sum() < int(min_cases):
            rows.append({**base, "effect": effect, "effect_normalized": eff_n, "statistic": np.nan, "p_value": np.nan,
                         "status": "NE", "reason": BELOW_MIN_CASES})
            continue
        if cl is not None:   # cluster-level test: one observation (the mean difference) per cluster
            codes, uniques = pd.factorize(cl[ok])
            d = np.bincount(codes, weights=d, minlength=len(uniques)) / np.bincount(codes, minlength=len(uniques))
        nonzero = d[d != 0]
        if nonzero.size == 0:
            stat, pv, reason = 0.0, 1.0, "all_differences_zero"
        else:
            res = stats.wilcoxon(nonzero, zero_method=zero_method, alternative="two-sided")
            stat, pv, reason = float(res.statistic), float(res.pvalue), None
        rows.append({**base, "effect": effect, "effect_normalized": eff_n, "statistic": stat, "p_value": pv, "status": "EVALUATED", "reason": reason})
    return pd.DataFrame(rows, columns=TEST_COLUMNS)


def mcnemar_exact(primary: Sequence[bool], variant: Sequence[bool]) -> dict[str, Any]:
    """Exact (binomial) McNemar test on paired binary outcomes; ``b`` = variant only, ``c`` = primary only."""
    x = np.asarray(primary, dtype=bool)
    y = np.asarray(variant, dtype=bool)
    if x.shape != y.shape:
        raise ValueError("paired outcomes must be aligned")
    b, c = int((y & ~x).sum()), int((x & ~y).sum())
    pv = 1.0 if b + c == 0 else float(stats.binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue)
    return {"n": int(x.size), "variant_only": b, "primary_only": c, "n_discordant": b + c, "p_value": pv,
            "effect": (b - c) / x.size if x.size else np.nan}


def variant_test_row(family: str, partition: str, variant: str, accepted_primary: Sequence[bool], accepted_variant: Sequence[bool]) -> dict[str, Any]:
    """One F4/F5 row: the exact McNemar test of the EXACT_ACCEPT indicator (variant versus primary configuration)."""
    m = mcnemar_exact(accepted_primary, accepted_variant)
    return {"family": family, "partition": partition, "comparison": "accepted_coverage_vs_primary", "stratum_variable": "all", "stratum": "all",
            "target": None, "variant": variant, "n": m["n"], "effect": m["effect"], "effect_normalized": np.nan,
            "statistic": float(m["n_discordant"]), "p_value": m["p_value"], "test": "exact_mcnemar", "status": "EVALUATED",
            "reason": f"variant_only={m['variant_only']};primary_only={m['primary_only']}"}


def bh_family_table(tests: pd.DataFrame, *, q: float, families: Sequence[str]) -> pd.DataFrame:
    """Benjamini-Hochberg within each family over the rows with a finite p-value (merged ranking forbidden); NE rows
    are kept with ``q_value`` NaN and ``rejected`` False."""
    out = tests.reset_index(drop=True).copy()
    out["q_value"] = np.nan
    out["rejected"] = False
    out["bh_q"] = float(q)
    tested = out.index[np.isfinite(pd.to_numeric(out["p_value"], errors="coerce").to_numpy(dtype=float))]
    if len(tested):
        adj = bh_within_families(out.loc[tested], q=q, families=families)
        out.loc[tested, "q_value"] = adj["q_value"].to_numpy()
        out.loc[tested, "rejected"] = adj["rejected"].to_numpy(dtype=bool)
    unknown = sorted(set(out["family"].astype(str)) - set(families))
    if unknown:
        raise ValueError(f"Tests outside the prespecified families: {unknown}")
    return out


def paired_nmae_difference(y_true: pd.DataFrame, y_reference: pd.DataFrame, y_comparator: pd.DataFrame, targets: Sequence[str],
                           reference_sd: pd.Series, *, n_boot: int, seed: int, cluster: pd.Series | np.ndarray | None = None,
                           level: float = 0.95, resample_cluster: pd.Series | np.ndarray | None = None) -> pd.DataFrame:
    """Per-target paired cluster-bootstrap percentile CI of nMAE(comparator) - nMAE(reference) (same draws for both;
    replicate r seeded by ``SeedSequence(seed).spawn(n_boot)[r]``; ``cluster`` / ``resample_cluster`` as in
    :func:`primary_rho`)."""
    t, r = _aligned(y_true, y_reference, targets)
    _, c = _aligned(y_true, y_comparator, targets)
    codes, n_clusters = _cluster_codes(cluster, len(t))
    unit_cluster, n_resample = _resampling_clusters(codes, n_clusters, resample_cluster)
    children = np.random.SeedSequence(int(seed)).spawn(int(n_boot))
    weights = [_cluster_draw_weights(np.random.default_rng(ss), unit_cluster, n_resample) for ss in children]
    alpha = (1.0 - level) / 2.0
    rows = []
    for j, name in enumerate(targets):
        sd = float(reference_sd.get(name, np.nan))
        ok = np.isfinite(t[:, j]) & np.isfinite(r[:, j]) & np.isfinite(c[:, j])
        er = np.where(ok, np.abs(r[:, j] - t[:, j]), np.nan)
        ec = np.where(ok, np.abs(c[:, j] - t[:, j]), np.nan)
        if not (ok.any() and np.isfinite(sd) and sd > 0):
            rows.append({"target": name, "n": int(ok.sum()), "nmae_reference": np.nan, "nmae_comparator": np.nan, "difference": np.nan,
                         "ci_lower": np.nan, "ci_upper": np.nan, "status": "NE"})
            continue
        point = (_patient_weighted_mae(ec, codes, n_clusters) - _patient_weighted_mae(er, codes, n_clusters)) / sd
        boot = np.array([(_patient_weighted_mae(ec, codes, n_clusters, w) - _patient_weighted_mae(er, codes, n_clusters, w)) / sd for w in weights])
        boot = boot[np.isfinite(boot)]
        rows.append({"target": name, "n": int(ok.sum()), "nmae_reference": _patient_weighted_mae(er, codes, n_clusters) / sd,
                     "nmae_comparator": _patient_weighted_mae(ec, codes, n_clusters) / sd, "difference": float(point),
                     "ci_lower": float(np.quantile(boot, alpha)) if boot.size else np.nan,
                     "ci_upper": float(np.quantile(boot, 1.0 - alpha)) if boot.size else np.nan,
                     "n_boot_valid": int(boot.size), "n_clusters": int(n_resample), "status": "EVALUATED" if boot.size else "NE"})
    return pd.DataFrame(rows)


def pooled_nmae_by_fold(y_true: pd.DataFrame, y_pred: pd.DataFrame, targets: Sequence[str], reference_sd: pd.Series, folds: np.ndarray) -> dict[int, float]:
    """Per outer fold: the mean over targets (finite, positive fit SD) of MAE_j / s_j on the fold's rows."""
    t, p = _aligned(y_true, y_pred, targets)
    sd = np.array([float(reference_sd.get(c, np.nan)) for c in targets])
    folds = np.asarray(folds, dtype=float)
    out = {}
    for f in sorted({int(v) for v in folds[np.isfinite(folds)]}):
        sel = folds == f
        vals = []
        for j in range(len(targets)):
            ok = sel & np.isfinite(t[:, j]) & np.isfinite(p[:, j])
            if ok.any() and np.isfinite(sd[j]) and sd[j] > 0:
                vals.append(float(np.mean(np.abs(p[ok, j] - t[ok, j]))) / sd[j])
        out[f] = float(np.mean(vals)) if vals else float("nan")
    return out
