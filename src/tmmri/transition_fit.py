from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from .io import resolve_path, sha256_file, utc_now_iso, write_table_pair, write_yaml
from .semantics.manual_review import parse_qc_reasons, reason_key, severity_for_reasons
from .target_transforms import TargetTransformSet, get_transform, observed_values, registry_feature_map, registry_transform_kind
from .transition import (
    LAMBDA_GRID,
    FormalStandardizer,
    PreprocessingState,
    fit_ridge_transition,
    fit_train_only_preprocessing,
    inverse_b_centered,
    ridge_from_svd,
    ridge_penalty,
    svd_rank_cutoff,
    transform_a,
    transform_b_centered,
)

DEFAULT_EXCLUDED_TARGETS = {
    "bmi_kg_m2",
    "bsa_m2",
    "myocardial_density_g_per_ml",
    "lv_sphericity_index_3d",
}


def _as_numeric_finite_count(df: pd.DataFrame, col: str, mask: pd.Series | np.ndarray | None = None) -> int:
    values = pd.to_numeric(df[col], errors="coerce")
    if mask is not None:
        values = values[pd.Series(mask, index=df.index).to_numpy()]
    return int(np.isfinite(values.to_numpy(dtype=float)).sum())


def select_transition_targets(
    b_df: pd.DataFrame,
    feature_dictionary: pd.DataFrame,
    min_train_observed: int = 10,
    exclude_targets: Iterable[str] | None = None,
) -> list[str]:
    exclude = set(DEFAULT_EXCLUDED_TARGETS)
    if exclude_targets:
        exclude.update(exclude_targets)
    eligible = feature_dictionary.loc[feature_dictionary.get("transition_target_eligible", False).astype(bool), "column_name"].astype(str).tolist()
    train_mask = b_df.get("split", pd.Series(["train"] * len(b_df))).astype(str).eq("train")
    targets: list[str] = []
    for col in eligible:
        if col in exclude or col not in b_df.columns:
            continue
        if _as_numeric_finite_count(b_df, col, train_mask) < min_train_observed:
            continue
        values = pd.to_numeric(b_df.loc[train_mask, col], errors="coerce")
        if values.dropna().nunique() <= 1:
            continue
        targets.append(col)
    return targets


def build_transition_target_manifest(
    b_acdc: pd.DataFrame,
    b_mms: pd.DataFrame,
    feature_dictionary: pd.DataFrame,
    included_targets: list[str],
) -> pd.DataFrame:
    rows = []
    included = set(included_targets)
    for _, row in feature_dictionary.iterrows():
        col = str(row["column_name"])
        if not bool(row.get("transition_target_eligible", False)):
            reason = "not_marked_transition_eligible"
        elif col in included:
            reason = "included"
        elif col in DEFAULT_EXCLUDED_TARGETS:
            reason = "excluded_by_default_policy"
        elif col not in b_acdc.columns:
            reason = "missing_from_B_ACDC"
        else:
            reason = "insufficient_or_degenerate_training_values"
        train_mask = b_acdc.get("split", pd.Series(["train"] * len(b_acdc))).astype(str).eq("train")
        val_mask = b_acdc.get("split", pd.Series([""] * len(b_acdc))).astype(str).eq("val")
        test_mask = b_acdc.get("split", pd.Series([""] * len(b_acdc))).astype(str).eq("test")
        rows.append(
            {
                "feature_name": col,
                "unit": row.get("unit"),
                "derivation_formula": row.get("derivation_formula"),
                "default_transition_target_eligible": bool(row.get("transition_target_eligible", False)),
                "included_in_experiment": col in included,
                "exclusion_reason": reason,
                "missing_count_train": int(pd.to_numeric(b_acdc.loc[train_mask, col], errors="coerce").isna().sum()) if col in b_acdc.columns else None,
                "missing_count_val": int(pd.to_numeric(b_acdc.loc[val_mask, col], errors="coerce").isna().sum()) if col in b_acdc.columns else None,
                "missing_count_test": int(pd.to_numeric(b_acdc.loc[test_mask, col], errors="coerce").isna().sum()) if col in b_acdc.columns else None,
                "missing_count_mms_100": int(pd.to_numeric(b_mms[col], errors="coerce").isna().sum()) if col in b_mms.columns else None,
                "manual_review_sensitive": any(token in col for token in ["ef", "sv", "volume", "wall", "mass", "edv", "esv"]),
                "affected_by_qc_reasons": "lv_rv_sv_discrepancy_gt_15_ml;extreme_robust_outlier" if col in included else "",
            }
        )
    return pd.DataFrame(rows)


def row_allowed_for_transition(row: pd.Series, allow_high_moderate: bool = True) -> bool:
    reasons = parse_qc_reasons(row.get("qc_failure_reason"))
    severity = severity_for_reasons(reasons)
    if severity == "critical":
        return False
    if severity in {"high", "moderate"} and not allow_high_moderate:
        return False
    return True


def build_row_inclusion_table(b_acdc: pd.DataFrame, allow_high_moderate: bool = True) -> pd.DataFrame:
    rows = []
    for _, row in b_acdc.iterrows():
        reasons = parse_qc_reasons(row.get("qc_failure_reason"))
        severity = severity_for_reasons(reasons)
        rows.append(
            {
                "sample_id": row.get("sample_id"),
                "patient_id": row.get("patient_id"),
                "split": row.get("split"),
                "disease_label": row.get("disease_label"),
                "qc_failure_reason": row.get("qc_failure_reason"),
                "max_qc_severity": severity,
                "included_for_transition": row_allowed_for_transition(row, allow_high_moderate=allow_high_moderate),
                "inclusion_policy": "exclude_critical_only" if allow_high_moderate else "exclude_high_moderate_and_critical",
            }
        )
    return pd.DataFrame(rows)


def _state_to_dict(state: PreprocessingState) -> dict[str, object]:
    return {
        "b_medians": state.b_medians,
        "b_clip_lower": state.b_clip_lower,
        "b_clip_upper": state.b_clip_upper,
        "a_mu": state.a_mu,
        "a_sigma": state.a_sigma,
        "b_mu": state.b_mu,
        "a_columns": state.a_columns,
        "b_columns": state.b_columns,
    }


def preprocessing_to_artifacts(state: PreprocessingState, experiment_id: str, imputed_counts: dict[str, dict[str, int]] | None = None) -> list[Path]:
    payload = _state_to_dict(state)
    payload.update({"experiment_id": experiment_id, "clinical_status": "research_use_only", "generated_at_utc": utc_now_iso()})
    artifacts = [
        write_yaml({"experiment_id": experiment_id, "target_feature_names": state.b_columns, "training_medians": state.b_medians, "imputed_counts_by_split": imputed_counts or {}}, f"artifacts/transforms/B_imputation_metadata_{experiment_id}.yaml"),
        write_yaml({"experiment_id": experiment_id, "clip_lower_q01": state.b_clip_lower, "clip_upper_q99": state.b_clip_upper}, f"artifacts/transforms/B_clipping_metadata_{experiment_id}.yaml"),
        write_yaml({"experiment_id": experiment_id, "mu_A": state.a_mu, "sigma_A": state.a_sigma, "mu_B": state.b_mu, "feature_order_A": state.a_columns, "feature_order_B": state.b_columns}, f"artifacts/transforms/scaling_metadata_{experiment_id}.yaml"),
    ]
    return artifacts


def _cv_splitter(labels: pd.Series, n_samples: int, folds_default: int, seed: int):
    folds = max(2, min(folds_default, n_samples))
    if labels.notna().nunique() > 1 and labels.value_counts().min() >= folds:
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(np.zeros(n_samples), labels.astype(str))
    return KFold(n_splits=folds, shuffle=True, random_state=seed).split(np.zeros(n_samples))


def cross_validate_lambdas(
    a_acdc: pd.DataFrame,
    b_acdc: pd.DataFrame,
    a_columns: list[str],
    b_columns: list[str],
    train_mask: pd.Series,
    lambda_grid: Iterable[float] = LAMBDA_GRID,
    cv_folds: int = 5,
    seed: int = 20260611,
) -> pd.DataFrame:
    train_idx = np.flatnonzero(train_mask.to_numpy())
    labels = b_acdc.loc[train_mask, "disease_label"] if "disease_label" in b_acdc.columns else pd.Series([None] * len(train_idx))
    rows = []
    splits = list(_cv_splitter(labels.reset_index(drop=True), len(train_idx), cv_folds, seed))
    for lam in lambda_grid:
        fold_scores = []
        fold_mae = []
        fold_normalized = []
        for fold_id, (tr_rel, va_rel) in enumerate(splits):
            tr_idx = train_idx[tr_rel]
            va_idx = train_idx[va_rel]
            state = fit_train_only_preprocessing(a_acdc.iloc[tr_idx], b_acdc.iloc[tr_idx], a_columns, b_columns)
            az_tr = transform_a(a_acdc.iloc[tr_idx], state)
            bc_tr = transform_b_centered(b_acdc.iloc[tr_idx], state)
            t = fit_ridge_transition(az_tr, bc_tr, float(lam))
            az_va = transform_a(a_acdc.iloc[va_idx], state)
            observed = inverse_b_centered(transform_b_centered(b_acdc.iloc[va_idx], state), state)
            pred = inverse_b_centered(az_va @ t, state)
            residual = pred - observed
            fold_scores.append(float(np.sqrt(np.nanmean(residual**2))))
            fold_mae.append(float(np.nanmean(np.abs(residual))))
            train_scale = np.nanstd(bc_tr, axis=0, ddof=0)
            train_scale = np.where(np.isfinite(train_scale) & (train_scale > 1e-12), train_scale, 1.0)
            per_target_rmse = np.sqrt(np.nanmean(residual**2, axis=0))
            fold_normalized.append(float(np.nanmean(per_target_rmse / train_scale)))
        rows.append(
            {
                "lambda_value": float(lam),
                "cv_rmse_mean": float(np.mean(fold_scores)),
                "cv_rmse_std": float(np.std(fold_scores, ddof=0)),
                "cv_mae_mean": float(np.mean(fold_mae)),
                "cv_macro_train_sd_normalized_rmse_mean": float(np.mean(fold_normalized)),
                "cv_macro_train_sd_normalized_rmse_std": float(np.std(fold_normalized, ddof=0)),
                "selection_objective": "mean_per_target_RMSE_divided_by_training_fold_target_SD",
                "cv_folds": len(splits),
            }
        )
    out = pd.DataFrame(rows).sort_values(
        ["cv_macro_train_sd_normalized_rmse_mean", "cv_rmse_mean", "lambda_value"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    out["selected"] = False
    if len(out):
        # If tied within 0.1 percent, choose the larger lambda for stability.
        best = float(out.loc[0, "cv_macro_train_sd_normalized_rmse_mean"])
        tied = out[out["cv_macro_train_sd_normalized_rmse_mean"] <= best * 1.001]
        selected_idx = tied["lambda_value"].idxmax()
        out.loc[selected_idx, "selected"] = True
    return out.sort_values("lambda_value").reset_index(drop=True)


def selected_lambda(cv_results: pd.DataFrame) -> float:
    chosen = cv_results[cv_results["selected"].astype(bool)]
    if chosen.empty:
        raise ValueError("No selected lambda in CV results.")
    return float(chosen.iloc[0]["lambda_value"])


def fit_final_transition(
    a_acdc: pd.DataFrame,
    b_acdc: pd.DataFrame,
    a_columns: list[str],
    b_columns: list[str],
    train_mask: pd.Series,
    lambda_value: float,
) -> tuple[np.ndarray, PreprocessingState]:
    state = fit_train_only_preprocessing(a_acdc.loc[train_mask], b_acdc.loc[train_mask], a_columns, b_columns)
    az = transform_a(a_acdc.loc[train_mask], state)
    bc = transform_b_centered(b_acdc.loc[train_mask], state)
    t = fit_ridge_transition(az, bc, lambda_value)
    if (az @ t).shape != bc.shape:
        raise AssertionError("A @ T shape contract failed during final fit.")
    return t, state


def write_transition_artifacts(
    t_matrix: np.ndarray,
    state: PreprocessingState,
    cv_results: pd.DataFrame,
    target_manifest: pd.DataFrame,
    row_inclusion: pd.DataFrame,
    experiment_id: str,
    selected_lambda_value: float,
) -> list[Path]:
    artifacts: list[Path] = []
    coef_rows = []
    for i, a_col in enumerate(state.a_columns):
        for j, b_col in enumerate(state.b_columns):
            coef_rows.append(
                {
                    "latent_feature": a_col,
                    "semantic_feature": b_col,
                    "coefficient": float(t_matrix[i, j]),
                    "experiment_id": experiment_id,
                    "lambda_value": float(selected_lambda_value),
                    "schema_version": "T_v4.0",
                }
            )
    artifacts.extend(write_table_pair(pd.DataFrame(coef_rows), f"artifacts/models/transition/T_{experiment_id}.parquet", f"artifacts/models/transition/T_{experiment_id}.csv"))
    t_wide = pd.DataFrame(t_matrix, columns=state.b_columns)
    t_wide.insert(0, "latent_feature", state.a_columns)
    t_wide.insert(0, "schema_version", "T_wide_v4.0")
    t_wide.insert(1, "experiment_id", experiment_id)
    artifacts.extend(write_table_pair(t_wide, f"artifacts/models/transition/T_wide_{experiment_id}.parquet", f"artifacts/models/transition/T_wide_{experiment_id}.csv"))
    artifacts.extend(write_table_pair(cv_results, f"artifacts/models/transition/transition_cv_results_{experiment_id}.parquet", f"artifacts/models/transition/transition_cv_results_{experiment_id}.csv"))
    artifacts.extend(write_table_pair(target_manifest, f"artifacts/transforms/transition_target_manifest_{experiment_id}.parquet", f"artifacts/transforms/transition_target_manifest_{experiment_id}.csv"))
    artifacts.extend(write_table_pair(row_inclusion, f"artifacts/transforms/transition_row_inclusion_{experiment_id}.parquet", f"artifacts/transforms/transition_row_inclusion_{experiment_id}.csv"))
    artifacts.extend(preprocessing_to_artifacts(state, experiment_id))
    metadata = {
        "schema_version": "T_v4.0",
        "experiment_id": experiment_id,
        "clinical_status": "research_use_only",
        "causal_interpretation_allowed": False,
        "orientation": "row_sample_right_multiplication: B_hat = zscore(A_train_only_transform) @ T + mu_B_train; T has shape k_by_l",
        "primary_matrix_artifact": f"artifacts/models/transition/T_wide_{experiment_id}.parquet",
        "long_coefficient_artifact": f"artifacts/models/transition/T_{experiment_id}.parquet",
        "t_shape": [int(t_matrix.shape[0]), int(t_matrix.shape[1])],
        "selected_lambda": float(selected_lambda_value),
        "feature_order_A": state.a_columns,
        "feature_order_B": state.b_columns,
        "generated_at_utc": utc_now_iso(),
    }
    artifacts.append(write_yaml(metadata, f"artifacts/models/transition/T_{experiment_id}_metadata.yaml"))
    return artifacts


def reconstruct_bhat(
    a_df: pd.DataFrame,
    b_df: pd.DataFrame,
    t_matrix: np.ndarray,
    state: PreprocessingState,
    experiment_id: str,
    dataset_stem: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    az = transform_a(a_df, state)
    pred = inverse_b_centered(az @ t_matrix, state)
    pred_df = pd.DataFrame(pred, columns=[f"{c}_hat" for c in state.b_columns])
    for col in ["schema_version", "experiment_id", "sample_id", "patient_id", "dataset_source", "split", "disease_label"]:
        if col == "schema_version":
            pred_df.insert(0, col, "Bhat_v3.0")
        elif col == "experiment_id":
            pred_df.insert(1, col, experiment_id)
        elif col in b_df.columns:
            pred_df.insert(len([c for c in pred_df.columns if c in ["schema_version", "experiment_id", "sample_id", "patient_id", "dataset_source", "split", "disease_label"]]), col, b_df[col].to_numpy())
    residual_rows = []
    for idx, b_row in b_df.reset_index(drop=True).iterrows():
        for j, feature in enumerate(state.b_columns):
            observed = pd.to_numeric(pd.Series([b_row.get(feature)]), errors="coerce").iloc[0]
            predicted = float(pred[idx, j])
            residual = predicted - float(observed) if pd.notna(observed) else np.nan
            residual_rows.append(
                {
                    "sample_id": b_row.get("sample_id"),
                    "patient_id": b_row.get("patient_id"),
                    "dataset_source": b_row.get("dataset_source"),
                    "split": b_row.get("split"),
                    "disease_label": b_row.get("disease_label"),
                    "feature_name": feature,
                    "observed_value": float(observed) if pd.notna(observed) else np.nan,
                    "predicted_value": predicted,
                    "residual": residual,
                    "absolute_error": abs(residual) if pd.notna(residual) else np.nan,
                    "manual_review_flag": bool(b_row.get("qc_manual_review_required", False)),
                    "feature_valid_flag": pd.notna(observed),
                    "unit": None,
                    "experiment_id": experiment_id,
                }
            )
    residuals = pd.DataFrame(residual_rows)
    artifacts = []
    artifacts.extend(write_table_pair(pred_df, f"artifacts/models/transition/Bhat_{dataset_stem}_{experiment_id}.parquet", f"artifacts/models/transition/Bhat_{dataset_stem}_{experiment_id}.csv"))
    artifacts.extend(write_table_pair(residuals, f"artifacts/models/transition/transition_residuals_{dataset_stem}_{experiment_id}.parquet", f"artifacts/models/transition/transition_residuals_{dataset_stem}_{experiment_id}.csv"))
    return pred_df, residuals, artifacts


# ===============================================================================================================
# Protocol v6 transition (SPECIFICATION.md 6.0.0, Sections 9.1-9.5; Decision D10). New functions only: the legacy
# functions above (select_transition_targets, cross_validate_lambdas, fit_final_transition, ...) are unchanged.
# Targets are chosen by registry transition_role, never by column-name substring.
# ===============================================================================================================

V6_TRANSITION_ROLES = ("candidate", "candidate_secondary")
LAMBDA_AT_GRID_EDGE = "LAMBDA_AT_GRID_EDGE"
RECONSTRUCTED_ESTIMATED = "ESTIMATED"
RECONSTRUCTED_INPUT_QC_BLOCKED = "INPUT_QC_BLOCKED"
RECONSTRUCTED_TARGET_INELIGIBLE = "TARGET_INELIGIBLE"
RECONSTRUCTED_NUMERIC_DOMAIN_BLOCKED = "NUMERIC_DOMAIN_BLOCKED"
_TIE_RTOL = 1e-9  # numerical tie in the inner-CV objective; ties go to the larger lambda (configs/transition.yaml)


def transition_candidates_v6(registry: dict[str, Any]) -> list[str]:
    """Registry features with transition_role candidate or candidate_secondary, in registry order."""
    return [str(e["name"]) for e in registry.get("features", []) if e.get("transition_role") in V6_TRANSITION_ROLES]


def lambda_grid_v6(cfg: dict[str, Any]) -> np.ndarray:
    """The binding grid ``estimator.lambda_grid_rule`` (checked against ``lambda_grid_display``)."""
    est = cfg["estimator"]
    match = re.fullmatch(r"\s*numpy\.logspace\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*(\d+)\s*\)\s*", str(est["lambda_grid_rule"]))
    if not match:
        raise ValueError(f"Unrecognized lambda_grid_rule {est['lambda_grid_rule']!r}")
    grid = np.logspace(float(match.group(1)), float(match.group(2)), int(match.group(3)))
    display = est.get("lambda_grid_display")
    if display is not None and (len(display) != grid.size or not np.allclose(grid, np.asarray(display, dtype=float), rtol=1e-4)):
        raise ValueError("lambda_grid_rule and lambda_grid_display disagree in configs/transition.yaml")
    return grid


def _grid_edges(cfg: dict[str, Any], grid: np.ndarray) -> tuple[float, float]:
    edges = ((cfg.get("estimator") or {}).get("grid_edge_policy") or {}).get("edges")
    if edges:
        return float(edges[0]), float(edges[1])
    return float(grid[0]), float(grid[-1])


def _is_edge(value: float, edges: tuple[float, float]) -> bool:
    if not np.isfinite(value):
        return False
    return bool(np.isclose(value, edges[0], rtol=1e-9) or np.isclose(value, edges[1], rtol=1e-9))


def eligible_targets_v6(
    b_fit: pd.DataFrame,
    targets: list[str],
    cfg: dict[str, Any],
    registry: dict[str, Any],
    *,
    identity: bool = False,
) -> tuple[list[str], dict[str, str]]:
    """Targets with >= min_observed_fit_rows observed, in-domain fit rows and a positive fit SD (transformed)."""
    candidates = set(transition_candidates_v6(registry))
    tcfg = cfg.get("targets") or {}
    min_rows = int(tcfg["min_observed_fit_rows"])
    require_sd = bool(tcfg.get("require_fit_sd_gt_zero", True))
    eligible: list[str] = []
    excluded: dict[str, str] = {}
    for target in targets:
        if target not in candidates:
            excluded[target] = "TARGET_INELIGIBLE:not_a_transition_candidate"
            continue
        kind = "identity" if identity else registry_transform_kind(registry, target)
        z = get_transform(kind, cfg).forward(observed_values(b_fit, target))
        z = z[np.isfinite(z)]
        if z.size < min_rows:
            excluded[target] = f"TARGET_INELIGIBLE:fewer_than_{min_rows}_observed_fit_rows"
        elif require_sd and not float(np.std(z, ddof=0)) > 0:
            excluded[target] = "TARGET_INELIGIBLE:zero_fit_sd"
        else:
            eligible.append(target)
    return eligible, excluded


def _mask_groups(y: np.ndarray) -> list[tuple[np.ndarray, list[int]]]:
    """Group target columns sharing one observation mask so that each group needs one SVD."""
    groups: dict[bytes, tuple[np.ndarray, list[int]]] = {}
    finite = np.isfinite(y)
    for j in range(y.shape[1]):
        key = np.packbits(finite[:, j]).tobytes() + int(finite.shape[0]).to_bytes(8, "little")
        if key not in groups:
            groups[key] = (finite[:, j].copy(), [])
        groups[key][1].append(j)
    return list(groups.values())


def _fit_ridge_targets(z: np.ndarray, y: np.ndarray, lambdas: list[float], convention: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-target masked ridge (unpenalized intercept) on standardized z and standardized transformed y."""
    coef = np.zeros((z.shape[1], y.shape[1]), dtype=float)
    intercept = np.full(y.shape[1], np.nan, dtype=float)
    for rows, cols in _mask_groups(y):
        if not rows.any():
            continue
        zr = z[rows]
        z_mean = zr.mean(axis=0)
        zc = zr - z_mean
        u, s, vt = np.linalg.svd(zc, full_matrices=False)
        keep = s > svd_rank_cutoff(s, zc.shape[0], zc.shape[1])
        for j in cols:
            yr = y[rows, j]
            y_mean = float(yr.mean())
            w = ridge_from_svd(s, vt, u.T @ (yr - y_mean), ridge_penalty(lambdas[j], zc.shape[0], convention), keep)
            coef[:, j] = w
            intercept[j] = y_mean - float(z_mean @ w)
    return coef, intercept


def select_lambdas_inner_cv(
    a_fit: np.ndarray,
    b_fit: pd.DataFrame,
    targets: list[str],
    cfg: dict[str, Any],
    registry: dict[str, Any],
    inner_folds: np.ndarray,
    *,
    identity: bool = False,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Per-target lambda by inner CV: mean over inner folds of the MAE in original units after the inverse transform.

    Every preprocessing object (zero-variance filter, A scaler, target transforms, clipping) is refitted on each inner
    training set. Ties (relative 1e-9) go to the larger lambda; selections at a grid edge are flagged, never extended.
    """
    grid = lambda_grid_v6(cfg)
    edges = _grid_edges(cfg, grid)
    convention = str(cfg["estimator"]["objective_convention"])
    min_rows = int(cfg["targets"]["min_observed_fit_rows"])
    inner = pd.to_numeric(pd.Series(np.asarray(inner_folds)), errors="coerce").to_numpy(dtype=float)
    labelled = np.isfinite(inner)
    folds = sorted({int(v) for v in inner[labelled]})
    if len(folds) < 2:
        raise ValueError("Inner cross-validation needs at least two inner folds")
    scores = np.full((len(folds), len(targets), grid.size), np.nan)
    n_val = np.zeros((len(folds), len(targets)), dtype=int)
    for fi, k in enumerate(folds):
        va = inner == k
        tr = labelled & ~va
        scaler = FormalStandardizer.fit(a_fit[tr])
        z_tr = scaler.transform(a_fit[tr])
        z_va = scaler.transform(a_fit[va])
        b_tr = b_fit.iloc[np.flatnonzero(tr)]
        b_va = b_fit.iloc[np.flatnonzero(va)]
        tts = TargetTransformSet.fit(b_tr, registry, targets, cfg=cfg, identity=identity)
        y_tr = tts.forward(b_tr, clip=True).to_numpy(dtype=float)
        for rows, cols in _mask_groups(y_tr):
            if rows.sum() < min_rows:
                continue
            zr = z_tr[rows]
            z_mean = zr.mean(axis=0)
            zc = zr - z_mean
            u, s, vt = np.linalg.svd(zc, full_matrices=False)
            keep = s > svd_rank_cutoff(s, zc.shape[0], zc.shape[1])
            penalties = np.array([ridge_penalty(lam, zc.shape[0], convention) for lam in grid])
            with np.errstate(divide="ignore", invalid="ignore"):
                factors = s[:, None] / (s[:, None] ** 2 + penalties[None, :])
            factors = np.where(keep[:, None], factors, 0.0)
            for j in cols:
                fit = tts.fits[targets[j]]
                if not fit.usable:
                    continue
                x_va = observed_values(b_va, targets[j])
                obs = np.isfinite(x_va)
                n_val[fi, j] = int(obs.sum())
                if not obs.any():
                    continue
                yr = y_tr[rows, j]
                y_mean = float(yr.mean())
                proj = u.T @ (yr - y_mean)
                w = vt.T @ (factors * proj[:, None])            # (p, n_lambda)
                c = y_mean - z_mean @ w                          # (n_lambda,)
                pred = z_va[obs] @ w + c                         # (n_obs, n_lambda)
                x_hat = fit.transform.inverse(pred * fit.sd + fit.mu)
                scores[fi, j, :] = np.mean(np.abs(x_hat - x_va[obs][:, None]), axis=0)
    return _lambda_selection_table(scores, n_val, targets, grid, edges, cfg)


def _lambda_selection_table(scores: np.ndarray, n_val: np.ndarray, targets: list[str], grid: np.ndarray, edges: tuple[float, float],
                            cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, float]]:
    """Per-target lambda from an inner-CV score cube (fold x target x lambda): mean over the scored folds, ties to the
    larger lambda, grid-edge flags (shared by the ridge and the RBF kernel-ridge comparator)."""
    rows_out: list[dict[str, Any]] = []
    selected: dict[str, float] = {}
    for j, target in enumerate(targets):
        per_fold = scores[:, j, :]
        scored = np.isfinite(per_fold).all(axis=1)
        s_j = per_fold[scored].mean(axis=0) if scored.any() else np.full(grid.size, np.nan)
        finite = np.isfinite(s_j)
        lam = float("nan")
        best = float("nan")
        edge = False
        if finite.any():
            best = float(np.min(s_j[finite]))
            tied = finite & np.isclose(s_j, best, rtol=_TIE_RTOL, atol=0.0)
            lam = float(np.max(grid[tied]))
            edge = _is_edge(lam, edges)
            selected[target] = lam
        rows_out.append(
            {
                "target": target,
                "selected_lambda": lam,
                "selected_cv_mae_original_units": best,
                "lambda_at_grid_edge": edge,
                "flag": LAMBDA_AT_GRID_EDGE if edge else "",
                "inner_folds_scored": int(scored.sum()),
                "inner_validation_rows": int(n_val[:, j].sum()),
                "selection_metric": str(cfg["estimator"].get("selection_metric", "mae_original_units_after_inverse_transform")),
                "cv_mae_by_lambda": [float(v) if np.isfinite(v) else None for v in s_j],
                "lambda_grid": [float(g) for g in grid],
            }
        )
    return pd.DataFrame(rows_out), selected


@dataclass
class TransitionModel:
    """Typed transforms + standardization + per-target ridge; ``predict`` returns original units."""

    targets: list[str]
    fitted_targets: list[str]
    excluded: dict[str, str]
    standardizer: FormalStandardizer
    transforms: TargetTransformSet
    coef: np.ndarray
    intercept: np.ndarray
    lambdas: dict[str, float]
    edge_flags: dict[str, bool]
    convention: str
    identity_transforms: bool = False
    representation: str = "formal_descriptors_v3"
    lambda_selection: list[dict[str, Any]] = field(default_factory=list)
    n_fit_rows: int = 0

    def predict_standardized(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        finite = np.all(np.isfinite(a[:, self.standardizer.keep]), axis=1)
        out = np.full((a.shape[0], len(self.fitted_targets)), np.nan)
        if finite.any() and self.fitted_targets:
            out[finite] = self.standardizer.transform(a[finite]) @ self.coef + self.intercept
        return out

    def predict(self, a: np.ndarray, index: pd.Index | None = None) -> pd.DataFrame:
        """Reconstructed B-hat in original units; excluded targets and non-finite A rows are NaN."""
        a = np.asarray(a, dtype=float)
        idx = index if index is not None else pd.RangeIndex(a.shape[0])
        out = pd.DataFrame(np.nan, index=idx, columns=list(self.targets), dtype=float)
        if self.fitted_targets:
            y = pd.DataFrame(self.predict_standardized(a), columns=self.fitted_targets, index=idx)
            decoded = self.transforms.inverse(y)
            for col in self.fitted_targets:
                out[col] = decoded[col].to_numpy()
        return out

    def decode(self, a: np.ndarray, registry: dict[str, Any], cfg: dict[str, Any], index: pd.Index | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Predictions after the registry-domain guard (configs/transition.yaml decoders) and reconstructed states."""
        raw = self.predict(a, index=index)
        a = np.asarray(a, dtype=float)
        finite_rows = np.all(np.isfinite(a[:, self.standardizer.keep]), axis=1)
        return decode_with_domain_guard(raw, registry, self.transforms.fit_range_original(), cfg, excluded=self.excluded, input_ok=finite_rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "transition_model_v6",
            "targets": list(self.targets),
            "fitted_targets": list(self.fitted_targets),
            "excluded": dict(self.excluded),
            "standardizer": self.standardizer.to_dict(),
            "transforms": self.transforms.to_dict(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept.tolist(),
            "lambdas": {k: float(v) for k, v in self.lambdas.items()},
            "edge_flags": {k: bool(v) for k, v in self.edge_flags.items()},
            "convention": self.convention,
            "identity_transforms": bool(self.identity_transforms),
            "representation": self.representation,
            "lambda_selection": self.lambda_selection,
            "n_fit_rows": int(self.n_fit_rows),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TransitionModel":
        standardizer = FormalStandardizer.from_dict(d["standardizer"])
        coef = np.asarray(d["coef"], dtype=float).reshape(standardizer.n_kept, len(d["fitted_targets"]))
        return cls(
            targets=list(d["targets"]),
            fitted_targets=list(d["fitted_targets"]),
            excluded=dict(d["excluded"]),
            standardizer=standardizer,
            transforms=TargetTransformSet.from_dict(d["transforms"]),
            coef=coef,
            intercept=np.asarray(d["intercept"], dtype=float),
            lambdas={k: float(v) for k, v in d["lambdas"].items()},
            edge_flags={k: bool(v) for k, v in d["edge_flags"].items()},
            convention=str(d["convention"]),
            identity_transforms=bool(d.get("identity_transforms", False)),
            representation=str(d.get("representation", "formal_descriptors_v3")),
            lambda_selection=list(d.get("lambda_selection", [])),
            n_fit_rows=int(d.get("n_fit_rows", 0)),
        )


def _fit_rows_mask(a: np.ndarray, n: int, fit_mask: np.ndarray | pd.Series | None) -> np.ndarray:
    rows = np.ones(n, dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool).copy()
    if rows.shape != (n,):
        raise ValueError("fit_mask must be a boolean vector aligned with the rows of A and B")
    return rows & np.all(np.isfinite(a), axis=1)


def fit_transition_v6(
    A: np.ndarray,
    B: pd.DataFrame,
    targets: list[str],
    cfg: dict[str, Any],
    registry: dict[str, Any],
    inner_folds: np.ndarray | None,
    *,
    fit_mask: np.ndarray | pd.Series | None = None,
    identity_transforms: bool = False,
    lambdas: dict[str, float] | None = None,
    representation: str = "formal_descriptors_v3",
) -> TransitionModel:
    """Fit the v6 transition on the fit rows (``fit_mask``, default all; critical-QC rows must be False).

    Per-target lambda is selected by inner CV over ``inner_folds`` (aligned with the rows of A/B) unless ``lambdas``
    fixes it (lambda = 0 gives the P1 pseudoinverse comparator). A and B rows must already be aligned by sample_id.
    """
    a = np.asarray(A, dtype=float)
    if a.ndim != 2 or a.shape[0] != len(B):
        raise ValueError(f"A has {a.shape[0] if a.ndim == 2 else '?'} rows but B has {len(B)}; rows must be aligned")
    rows = _fit_rows_mask(a, len(B), fit_mask)
    pos = np.flatnonzero(rows)
    b_fit = B.iloc[pos]
    a_fit = a[pos]
    eligible, excluded = eligible_targets_v6(b_fit, list(targets), cfg, registry, identity=identity_transforms)
    selection_rows: list[dict[str, Any]] = []
    if lambdas is None:
        if inner_folds is None:
            raise ValueError("inner_folds are required when lambdas are not fixed")
        inner = np.asarray(inner_folds)[pos]
        table, selected = select_lambdas_inner_cv(a_fit, b_fit, eligible, cfg, registry, inner, identity=identity_transforms)
        selection_rows = table.to_dict("records")
        for target in list(eligible):
            if target not in selected:
                excluded[target] = "TARGET_INELIGIBLE:no_inner_cv_score"
        eligible = [t for t in eligible if t in selected]
        lam_by_target = {t: selected[t] for t in eligible}
        grid_edges = _grid_edges(cfg, lambda_grid_v6(cfg))
    else:
        lam_by_target = {t: float(lambdas[t]) for t in eligible}
        grid_edges = (float("nan"), float("nan"))
    scaler = FormalStandardizer.fit(a_fit)
    z = scaler.transform(a_fit)
    tts = TargetTransformSet.fit(b_fit, registry, eligible, cfg=cfg, identity=identity_transforms)
    y = tts.forward(b_fit, clip=True).to_numpy(dtype=float)
    convention = str(cfg["estimator"]["objective_convention"])
    coef, intercept = _fit_ridge_targets(z, y, [lam_by_target[t] for t in eligible], convention)
    return TransitionModel(
        targets=list(targets),
        fitted_targets=eligible,
        excluded=excluded,
        standardizer=scaler,
        transforms=tts,
        coef=coef,
        intercept=intercept,
        lambdas=lam_by_target,
        edge_flags={t: _is_edge(lam_by_target[t], grid_edges) for t in eligible},
        convention=convention,
        identity_transforms=identity_transforms,
        representation=representation,
        lambda_selection=selection_rows,
        n_fit_rows=int(pos.size),
    )


def positional_inner_folds(n_inner: int) -> Callable[[int, np.ndarray], np.ndarray]:
    """Fallback inner-fold function ``(outer_fold, train_positions) -> labels`` (position modulo n_inner, 1-based).

    Protocol runs pass the configs/splits.yaml hash rule (src/tmmri/splits.py) instead; this fallback exists for
    synthetic controls and callers without patient identifiers.
    """

    def _fn(outer_fold: int, train_positions: np.ndarray) -> np.ndarray:
        return (np.arange(len(train_positions)) % int(n_inner)) + 1

    return _fn


def nested_oof_transition(
    A: np.ndarray,
    B: pd.DataFrame,
    targets: list[str],
    cfg: dict[str, Any],
    registry: dict[str, Any],
    outer_folds: np.ndarray,
    inner_fold_fn: Callable[[int, np.ndarray], np.ndarray] | None,
    *,
    fit_mask: np.ndarray | pd.Series | None = None,
    identity_transforms: bool = False,
    lambdas: dict[str, float] | None = None,
    return_details: bool = False,
    fitter: Callable[..., Any] | None = None,
):
    """Nested out-of-fold B-hat (original units) for every row with an outer fold label (1..K; NaN elsewhere).

    For outer fold f the transition (preprocessing, transforms, per-target lambda by inner CV over
    ``inner_fold_fn(f, train_positions)``) is fitted on the other folds' fit rows and predicts fold f. The returned
    frame has one column per target plus ``outer_fold``; ``return_details=True`` also returns the per-fold lambda
    ledger (with LAMBDA_AT_GRID_EDGE flags) and the fold models. ``fitter`` replaces :func:`fit_transition_v6` with a
    comparator of the same signature (:func:`fit_transition_global_lambda_v6`, :func:`fit_rbf_kernel_ridge_v6`).
    """
    a = np.asarray(A, dtype=float)
    outer = pd.to_numeric(pd.Series(np.asarray(outer_folds)), errors="coerce").to_numpy(dtype=float)
    if outer.shape[0] != len(B) or a.shape[0] != len(B):
        raise ValueError("A, B and outer_folds must be aligned")
    dev = np.isfinite(outer)
    fit_rows = np.ones(len(B), dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    out = pd.DataFrame(np.nan, index=B.index, columns=list(targets), dtype=float)
    lambda_rows: list[dict[str, Any]] = []
    models: dict[int, TransitionModel] = {}
    for f in sorted({int(v) for v in outer[dev]}):
        train_pos = np.flatnonzero(dev & (outer != f))
        held_pos = np.flatnonzero(outer == f)
        inner = None if lambdas is not None else np.asarray(inner_fold_fn(int(f), train_pos))
        model = (fitter or fit_transition_v6)(
            a[train_pos], B.iloc[train_pos], list(targets), cfg, registry, inner,
            fit_mask=fit_rows[train_pos], identity_transforms=identity_transforms, lambdas=lambdas,
        )
        models[f] = model
        out.iloc[held_pos, :] = model.predict(a[held_pos]).to_numpy()
        for target in targets:
            lambda_rows.append(
                {
                    "outer_fold": f,
                    "target": target,
                    "lambda": model.lambdas.get(target, np.nan),
                    "lambda_at_grid_edge": bool(model.edge_flags.get(target, False)),
                    "flag": LAMBDA_AT_GRID_EDGE if model.edge_flags.get(target, False) else "",
                    "eligible": target in model.fitted_targets,
                    "exclusion_reason": model.excluded.get(target, ""),
                    "n_fit_rows": model.n_fit_rows,
                }
            )
    out["outer_fold"] = pd.array([int(v) if np.isfinite(v) else pd.NA for v in outer], dtype="Int64")
    if return_details:
        return out, {"lambda_selection": pd.DataFrame(lambda_rows), "models": models}
    return out


@dataclass
class MeanBaseline:
    """Null comparator: fit-partition mean of the observed values in original units (configs/transition.yaml)."""

    targets: list[str]
    means: dict[str, float]

    @classmethod
    def fit(cls, b_fit: pd.DataFrame, targets: list[str]) -> "MeanBaseline":
        means = {}
        for t in targets:
            x = observed_values(b_fit, t)
            x = x[np.isfinite(x)]
            means[t] = float(np.mean(x)) if x.size else float("nan")
        return cls(list(targets), means)

    def predict(self, n_rows: int, index: pd.Index | None = None) -> pd.DataFrame:
        idx = index if index is not None else pd.RangeIndex(int(n_rows))
        return pd.DataFrame({t: np.full(len(idx), self.means[t]) for t in self.targets}, index=idx, columns=self.targets)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "mean_baseline_v6", "targets": list(self.targets), "means": dict(self.means)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MeanBaseline":
        return cls(list(d["targets"]), {k: float(v) for k, v in d["means"].items()})


def comparators(
    A: np.ndarray,
    B: pd.DataFrame,
    targets: list[str],
    cfg: dict[str, Any],
    registry: dict[str, Any],
    folds: np.ndarray,
    *,
    inner_fold_fn: Callable[[int, np.ndarray], np.ndarray] | None = None,
    fit_mask: np.ndarray | pd.Series | None = None,
    alt_representations: dict[str, np.ndarray] | None = None,
    global_lambda: bool = False,
    rbf: bool = False,
) -> dict[str, pd.DataFrame]:
    """Out-of-fold comparators on the same rows, masks and folds as the primary ridge (configs/transition.yaml):
    ``mean`` (null), ``pseudoinverse_p1`` (lambda = 0), ``identity_transform_ridge`` and ``ridge_<name>`` for each
    alternative representation (for example the P6 128-d projection); ``global_lambda`` adds ``ridge_global_lambda``
    (one lambda for all targets, SPECIFICATION.md 9.4 f) and ``rbf`` adds ``rbf_kernel_ridge`` (the nonlinear
    comparator of the linearity criterion, 9.4 g), both with the same inner folds."""
    a = np.asarray(A, dtype=float)
    outer = pd.to_numeric(pd.Series(np.asarray(folds)), errors="coerce").to_numpy(dtype=float)
    dev = np.isfinite(outer)
    fit_rows = np.ones(len(B), dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    if inner_fold_fn is None:
        inner_fold_fn = positional_inner_folds(int((cfg.get("nested_folds") or {}).get("inner", cfg["estimator"].get("inner_folds", 5))))
    mean = pd.DataFrame(np.nan, index=B.index, columns=list(targets), dtype=float)
    for f in sorted({int(v) for v in outer[dev]}):
        train_pos = np.flatnonzero(dev & (outer != f) & fit_rows & np.all(np.isfinite(a), axis=1))
        held_pos = np.flatnonzero(outer == f)
        eligible, _ = eligible_targets_v6(B.iloc[train_pos], list(targets), cfg, registry)
        pred = MeanBaseline.fit(B.iloc[train_pos], list(targets)).predict(len(held_pos))
        for t in targets:
            if t not in eligible:
                pred[t] = np.nan
        mean.iloc[held_pos, :] = pred.to_numpy()
    mean["outer_fold"] = pd.array([int(v) if np.isfinite(v) else pd.NA for v in outer], dtype="Int64")
    out = {
        "mean": mean,
        "pseudoinverse_p1": nested_oof_transition(a, B, targets, cfg, registry, outer, None, fit_mask=fit_rows, lambdas={t: 0.0 for t in targets}),
        "identity_transform_ridge": nested_oof_transition(a, B, targets, cfg, registry, outer, inner_fold_fn, fit_mask=fit_rows, identity_transforms=True),
    }
    for name, alt in (alt_representations or {}).items():
        out[f"ridge_{name}"] = nested_oof_transition(np.asarray(alt, dtype=float), B, targets, cfg, registry, outer, inner_fold_fn, fit_mask=fit_rows)
    if global_lambda:
        out["ridge_global_lambda"] = nested_oof_transition(a, B, targets, cfg, registry, outer, inner_fold_fn, fit_mask=fit_rows,
                                                           fitter=fit_transition_global_lambda_v6)
    if rbf:
        out["rbf_kernel_ridge"] = nested_oof_transition(a, B, targets, cfg, registry, outer, inner_fold_fn, fit_mask=fit_rows,
                                                        fitter=fit_rbf_kernel_ridge_v6)
    return out


def decode_with_domain_guard(
    predictions: pd.DataFrame,
    registry: dict[str, Any],
    fit_ranges: dict[str, tuple[float, float]],
    cfg: dict[str, Any],
    *,
    excluded: dict[str, str] | None = None,
    input_ok: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Registry-domain guard: a value outside the domain by more than ``relative_width`` x fit range is
    NUMERIC_DOMAIN_BLOCKED (value NaN); values inside the guard are clipped to the domain when configured.
    Returns ``(values, states)`` with reconstructed states ESTIMATED, INPUT_QC_BLOCKED, TARGET_INELIGIBLE or
    NUMERIC_DOMAIN_BLOCKED."""
    dec = cfg.get("decoders") or {}
    rel = float((dec.get("domain_guard") or {})["relative_width"])
    clip = bool(dec.get("clip_to_domain_within_guard", True))
    features = registry_feature_map(registry)
    excluded = excluded or {}
    values = predictions.copy()
    states = pd.DataFrame(index=predictions.index)
    ok = np.ones(len(predictions), dtype=bool) if input_ok is None else np.asarray(input_ok, dtype=bool)
    for col in predictions.columns:
        if col == "outer_fold":
            continue
        v = pd.to_numeric(predictions[col], errors="coerce").to_numpy(dtype=float)
        state = np.where(np.isfinite(v), RECONSTRUCTED_ESTIMATED, RECONSTRUCTED_INPUT_QC_BLOCKED).astype(object)
        if col in excluded:
            state[:] = RECONSTRUCTED_TARGET_INELIGIBLE
        state[~ok] = RECONSTRUCTED_INPUT_QC_BLOCKED
        domain = (features.get(col) or {}).get("domain")
        if isinstance(domain, (list, tuple)) and len(domain) == 2:
            lo = -np.inf if domain[0] is None else float(domain[0])
            hi = np.inf if domain[1] is None else float(domain[1])
            fmin, fmax = fit_ranges.get(col, (np.nan, np.nan))
            width = rel * (fmax - fmin) if np.isfinite(fmax - fmin) else 0.0
            blocked = np.isfinite(v) & ((v < lo - width) | (v > hi + width))
            state[blocked] = RECONSTRUCTED_NUMERIC_DOMAIN_BLOCKED
            v = np.where(blocked, np.nan, v)
            if clip:
                v = np.where(np.isfinite(v), np.clip(v, lo, hi), v)
        values[col] = v
        states[col] = state
    return values, states


# ---------------------------------------------------------------------------------------------------------------
# Named comparators of SPECIFICATION.md 9.4 (f) and (g): one global lambda, and the RBF kernel ridge of the frozen
# linearity criterion (configs/transition.yaml comparators.ridge_global_lambda, comparators.rbf_kernel_ridge).
# ---------------------------------------------------------------------------------------------------------------


def _fit_sd_original(b_fit: pd.DataFrame, target: str) -> float:
    x = observed_values(b_fit, target)
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=1)) if x.size > 1 else float("nan")


def fit_transition_global_lambda_v6(
    A: np.ndarray,
    B: pd.DataFrame,
    targets: list[str],
    cfg: dict[str, Any],
    registry: dict[str, Any],
    inner_folds: np.ndarray | None,
    *,
    fit_mask: np.ndarray | pd.Series | None = None,
    identity_transforms: bool = False,
    lambdas: dict[str, float] | None = None,
    representation: str = "formal_descriptors_v3",
) -> TransitionModel:
    """Ridge with one lambda for every target: the grid value minimizing the mean over targets of the inner-CV MAE
    (original units) divided by each target's fit-row SD; ties to the larger lambda; grid edges flagged."""
    if lambdas is not None:
        raise ValueError("the global-lambda comparator selects its own lambda")
    if inner_folds is None:
        raise ValueError("inner_folds are required for the global-lambda comparator")
    a = np.asarray(A, dtype=float)
    rows = _fit_rows_mask(a, len(B), fit_mask)
    pos = np.flatnonzero(rows)
    b_fit, a_fit = B.iloc[pos], a[pos]
    eligible, _ = eligible_targets_v6(b_fit, list(targets), cfg, registry, identity=identity_transforms)
    table, selected = select_lambdas_inner_cv(a_fit, b_fit, eligible, cfg, registry, np.asarray(inner_folds)[pos], identity=identity_transforms)
    grid = lambda_grid_v6(cfg)
    curves, pooled_targets = [], []
    for rec in table.to_dict("records"):
        sd = _fit_sd_original(b_fit, rec["target"])
        curve = np.array([np.nan if v is None else float(v) for v in rec["cv_mae_by_lambda"]], dtype=float)
        if rec["target"] in selected and np.isfinite(sd) and sd > 0 and np.isfinite(curve).all():
            curves.append(curve / sd)
            pooled_targets.append(rec["target"])
    if not curves:
        raise ValueError("global-lambda comparator: no target has a complete inner-CV curve")
    pooled = np.mean(np.vstack(curves), axis=0)
    best = float(np.min(pooled))
    lam = float(np.max(grid[np.isclose(pooled, best, rtol=_TIE_RTOL, atol=0.0)]))
    usable = [t for t in targets if t in selected]
    model = fit_transition_v6(A, B, usable, cfg, registry, None, fit_mask=fit_mask, identity_transforms=identity_transforms,
                              lambdas={t: lam for t in usable}, representation=representation)
    edges = _grid_edges(cfg, grid)
    model.targets = list(targets)
    model.excluded = {**{t: "TARGET_INELIGIBLE:no_inner_cv_score" for t in targets if t not in usable}, **model.excluded}
    model.edge_flags = {t: _is_edge(lam, edges) for t in model.fitted_targets}
    model.lambda_selection = [{"scope": "global", "selected_lambda": lam, "lambda_at_grid_edge": _is_edge(lam, edges),
                               "flag": LAMBDA_AT_GRID_EDGE if _is_edge(lam, edges) else "", "n_targets_pooled": len(pooled_targets),
                               "pooled_targets": pooled_targets, "pooled_cv_nmae_by_lambda": [float(v) for v in pooled],
                               "lambda_grid": [float(g) for g in grid]}]
    return model


def _sq_distances(za: np.ndarray, zb: np.ndarray) -> np.ndarray:
    d2 = np.sum(za**2, axis=1)[:, None] + np.sum(zb**2, axis=1)[None, :] - 2.0 * (za @ zb.T)
    return np.maximum(d2, 0.0)


def median_pairwise_bandwidth(z: np.ndarray, fallback: float = 1.0) -> float:
    """Median heuristic: the median Euclidean distance over distinct pairs of fitting rows (``fallback`` when zero)."""
    z = np.asarray(z, dtype=float)
    if z.shape[0] < 2:
        return float(fallback)
    iu = np.triu_indices(z.shape[0], 1)
    med = float(np.median(np.sqrt(_sq_distances(z, z)[iu])))
    return med if np.isfinite(med) and med > 0 else float(fallback)


def _rbf_kernel(za: np.ndarray, zb: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-_sq_distances(za, zb) / (2.0 * float(sigma) ** 2))


def _krr_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    krr = dict((cfg.get("comparators") or {}).get("rbf_kernel_ridge") or {})
    if not krr:
        raise KeyError("configs/transition.yaml comparators.rbf_kernel_ridge is not configured")
    if str(krr.get("kernel", "rbf")) != "rbf" or str(krr.get("bandwidth")) != "median_pairwise_euclidean_distance_fit_rows":
        raise ValueError("only the RBF kernel with the median pairwise-distance bandwidth is implemented (fail closed)")
    return krr


def _eig_solve(kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    evals, evecs = np.linalg.eigh(kernel)
    return np.clip(evals, 0.0, None), evecs


@dataclass
class KernelRidgeModel:
    """RBF kernel ridge on standardized formal A with typed target transforms (a comparator, never the audit map)."""

    targets: list[str]
    fitted_targets: list[str]
    excluded: dict[str, str]
    standardizer: FormalStandardizer
    transforms: TargetTransformSet
    z_fit: np.ndarray
    sigma: float
    weights: dict[str, tuple[np.ndarray, np.ndarray, float]]
    lambdas: dict[str, float]
    edge_flags: dict[str, bool]
    lambda_selection: list[dict[str, Any]] = field(default_factory=list)
    n_fit_rows: int = 0
    representation: str = "formal_descriptors_v3"

    def predict_standardized(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        finite = np.all(np.isfinite(a[:, self.standardizer.keep]), axis=1)
        out = np.full((a.shape[0], len(self.fitted_targets)), np.nan)
        if finite.any() and self.fitted_targets:
            kernel = _rbf_kernel(self.standardizer.transform(a[finite]), self.z_fit, self.sigma)
            for j, t in enumerate(self.fitted_targets):
                idx, alpha, intercept = self.weights[t]
                out[finite, j] = kernel[:, idx] @ alpha + intercept
        return out

    def predict(self, a: np.ndarray, index: pd.Index | None = None) -> pd.DataFrame:
        a = np.asarray(a, dtype=float)
        idx = index if index is not None else pd.RangeIndex(a.shape[0])
        out = pd.DataFrame(np.nan, index=idx, columns=list(self.targets), dtype=float)
        if self.fitted_targets:
            decoded = self.transforms.inverse(pd.DataFrame(self.predict_standardized(a), columns=self.fitted_targets, index=idx))
            for col in self.fitted_targets:
                out[col] = decoded[col].to_numpy()
        return out


def _krr_inner_scores(a_fit: np.ndarray, b_fit: pd.DataFrame, targets: list[str], cfg: dict[str, Any], registry: dict[str, Any],
                      inner: np.ndarray, fallback: float) -> tuple[pd.DataFrame, dict[str, float]]:
    grid = lambda_grid_v6(cfg)
    edges = _grid_edges(cfg, grid)
    min_rows = int(cfg["targets"]["min_observed_fit_rows"])
    inner = pd.to_numeric(pd.Series(np.asarray(inner)), errors="coerce").to_numpy(dtype=float)
    labelled = np.isfinite(inner)
    folds = sorted({int(v) for v in inner[labelled]})
    if len(folds) < 2:
        raise ValueError("Inner cross-validation needs at least two inner folds")
    scores = np.full((len(folds), len(targets), grid.size), np.nan)
    n_val = np.zeros((len(folds), len(targets)), dtype=int)
    for fi, k in enumerate(folds):
        va = inner == k
        tr = labelled & ~va
        scaler = FormalStandardizer.fit(a_fit[tr])
        z_tr, z_va = scaler.transform(a_fit[tr]), scaler.transform(a_fit[va])
        sigma = median_pairwise_bandwidth(z_tr, fallback)
        k_tr, k_va = _rbf_kernel(z_tr, z_tr, sigma), _rbf_kernel(z_va, z_tr, sigma)
        b_tr, b_va = b_fit.iloc[np.flatnonzero(tr)], b_fit.iloc[np.flatnonzero(va)]
        tts = TargetTransformSet.fit(b_tr, registry, targets, cfg=cfg)
        y_tr = tts.forward(b_tr, clip=True).to_numpy(dtype=float)
        for rows, cols in _mask_groups(y_tr):
            if rows.sum() < min_rows:
                continue
            evals, evecs = _eig_solve(k_tr[np.ix_(rows, rows)])
            k_va_r = k_va[:, rows]
            for j in cols:
                fit = tts.fits[targets[j]]
                if not fit.usable:
                    continue
                x_va = observed_values(b_va, targets[j])
                obs = np.isfinite(x_va)
                n_val[fi, j] = int(obs.sum())
                if not obs.any():
                    continue
                yr = y_tr[rows, j]
                y_mean = float(yr.mean())
                proj = evecs.T @ (yr - y_mean)
                alphas = evecs @ (proj[:, None] / (evals[:, None] + grid[None, :]))   # sum loss: (K + lambda I)^-1 y_c
                pred = k_va_r[obs] @ alphas + y_mean
                x_hat = fit.transform.inverse(pred * fit.sd + fit.mu)
                scores[fi, j, :] = np.mean(np.abs(x_hat - x_va[obs][:, None]), axis=0)
    return _lambda_selection_table(scores, n_val, targets, grid, edges, cfg)


def fit_rbf_kernel_ridge_v6(
    A: np.ndarray,
    B: pd.DataFrame,
    targets: list[str],
    cfg: dict[str, Any],
    registry: dict[str, Any],
    inner_folds: np.ndarray | None,
    *,
    fit_mask: np.ndarray | pd.Series | None = None,
    identity_transforms: bool = False,
    lambdas: dict[str, float] | None = None,
    representation: str = "formal_descriptors_v3",
) -> KernelRidgeModel:
    """RBF kernel ridge (configs/transition.yaml comparators.rbf_kernel_ridge): standardized A, typed transforms and
    clipping as the primary ridge, bandwidth by the median pairwise distance of the fitting rows, per-target penalty by
    inner CV over the primary lambda grid (sum loss, unpenalized mean intercept)."""
    if identity_transforms or lambdas is not None:
        raise ValueError("the RBF comparator uses the registry transforms and selects its own penalties")
    if inner_folds is None:
        raise ValueError("inner_folds are required for the RBF comparator")
    krr = _krr_settings(cfg)
    fallback = float(krr.get("bandwidth_zero_fallback", 1.0))
    a = np.asarray(A, dtype=float)
    rows = _fit_rows_mask(a, len(B), fit_mask)
    pos = np.flatnonzero(rows)
    b_fit, a_fit = B.iloc[pos], a[pos]
    eligible, excluded = eligible_targets_v6(b_fit, list(targets), cfg, registry)
    table, selected = _krr_inner_scores(a_fit, b_fit, eligible, cfg, registry, np.asarray(inner_folds)[pos], fallback)
    for t in eligible:
        if t not in selected:
            excluded[t] = "TARGET_INELIGIBLE:no_inner_cv_score"
    eligible = [t for t in eligible if t in selected]
    scaler = FormalStandardizer.fit(a_fit)
    z = scaler.transform(a_fit)
    sigma = median_pairwise_bandwidth(z, fallback)
    kernel = _rbf_kernel(z, z, sigma)
    tts = TargetTransformSet.fit(b_fit, registry, eligible, cfg=cfg)
    y = tts.forward(b_fit, clip=True).to_numpy(dtype=float)
    weights: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for j, t in enumerate(eligible):
        idx = np.flatnonzero(np.isfinite(y[:, j]))
        yr = y[idx, j]
        y_mean = float(yr.mean())
        evals, evecs = _eig_solve(kernel[np.ix_(idx, idx)])
        alpha = evecs @ ((evecs.T @ (yr - y_mean)) / (evals + selected[t]))
        weights[t] = (idx, alpha, y_mean)
    edges = _grid_edges(cfg, lambda_grid_v6(cfg))
    return KernelRidgeModel(targets=list(targets), fitted_targets=eligible, excluded=excluded, standardizer=scaler, transforms=tts,
                            z_fit=z, sigma=float(sigma), weights=weights, lambdas={t: selected[t] for t in eligible},
                            edge_flags={t: _is_edge(selected[t], edges) for t in eligible},
                            lambda_selection=[{**r, "bandwidth_sigma": float(sigma)} for r in table.to_dict("records")],
                            n_fit_rows=int(pos.size), representation=representation)
