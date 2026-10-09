from __future__ import annotations

"""Fail-closed Phase-5 transition experiment grid.

The functions in this module operate only on explicit, already-frozen Matrix A,
Matrix B, feature-dictionary, and extractor-metadata artifacts.  They never read
native images and never use ACDC test or M&Ms labels for model selection.

All outputs are research artifacts.  They do not establish clinical validity or
authorize diagnosis, triage, treatment selection, or patient management.
"""

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold, StratifiedKFold
import yaml

from ..formal_features import _matrix_checksum, _projection_matrix
from ..metrics import benjamini_hochberg
from ..io import sha256_file, utc_now_iso
from ..semantics.manual_review import parse_qc_reasons, severity_for_reasons
from ..transition import (
    LAMBDA_GRID,
    PreprocessingState,
    fit_train_only_preprocessing,
    inverse_b_centered,
    ridge_from_svd,
    transform_a,
    transform_b_centered,
)
from ..transition_fit import DEFAULT_EXCLUDED_TARGETS, select_transition_targets


RESEARCH_WARNING = (
    "RESEARCH USE ONLY. Not authorized for autonomous diagnosis, triage, "
    "treatment selection, or patient management."
)

VENTRICULAR_FUNCTION_TARGETS = (
    "lv_edv_ml",
    "lv_esv_ml",
    "lv_sv_ml",
    "lv_ef_percent",
    "rv_edv_ml",
    "rv_esv_ml",
    "rv_sv_ml",
    "rv_ef_percent",
)

PHYSICIAN_PRIORITY_CANDIDATES = VENTRICULAR_FUNCTION_TARGETS + (
    "lv_myocardial_mass_ed_g",
    "global_wall_thickness_ed_mm",
    "global_wall_thickening_percent",
)

MORPHOLOGY_TOKENS = (
    "myo",
    "mass",
    "wall",
    "area",
    "compactness",
    "eccentricity",
    "long_axis",
)

QC_TARGET_CANDIDATES = (
    "wall_thickness_invalid_slice_count_ed",
    "wall_thickness_invalid_slice_count_es",
    "qc_extreme_robust_outlier_flag",
    "qc_manual_review_required",
)

DEFAULT_LAMBDAS = tuple(float(value) for value in LAMBDA_GRID)
IDENTIFIER_COLUMNS = ("sample_id", "patient_id", "dataset_source", "split")


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    experiment_family: str
    a_branch: str
    target_family: str
    qc_policy: str
    selection_eligible: bool = True
    status: str = "scheduled"
    notes: str = ""


@dataclass
class FittedTransition:
    coefficients: np.ndarray
    state: PreprocessingState
    lambda_value: float
    train_sd: dict[str, float]
    design_rank: int
    regularized_condition_number: float


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_json(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Required frozen table does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(source)
    if suffix == ".csv":
        return pd.read_csv(source)
    raise ValueError(f"Unsupported table format for {source}; expected parquet or CSV")


def _write_frame(frame: pd.DataFrame, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    parquet = stem.with_suffix(".parquet")
    csv = stem.with_suffix(".csv")
    frame.to_parquet(parquet, engine="pyarrow", compression="snappy", index=False)
    frame.to_csv(csv, index=False)
    return [parquet, csv]


def _write_json(value: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def _load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Required extractor metadata does not exist: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Extractor metadata must be a mapping: {source}")
    return payload


def _a_columns(frame: pd.DataFrame) -> list[str]:
    columns = sorted(
        (column for column in frame.columns if str(column).startswith("a_")),
        key=lambda value: int(str(value).split("_")[-1]),
    )
    if not columns:
        raise ValueError("Frozen Matrix A has no a_* feature columns")
    return columns


def _align_a_to_b(a: pd.DataFrame, b: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    for name, frame in (("A", a), ("B", b)):
        missing = {"sample_id", "patient_id"}.difference(frame.columns)
        if missing:
            raise ValueError(f"{dataset_name} Matrix {name} lacks identifiers: {sorted(missing)}")
        if frame["sample_id"].astype(str).duplicated().any():
            raise ValueError(f"{dataset_name} Matrix {name} has duplicate sample_id values")
        if frame["patient_id"].astype(str).duplicated().any():
            raise ValueError(f"{dataset_name} Matrix {name} is not one-row-per-patient")
    a_ids = set(a["sample_id"].astype(str))
    b_ids = set(b["sample_id"].astype(str))
    if a_ids != b_ids:
        raise ValueError(
            f"{dataset_name} Matrix A/B sample_id mismatch: "
            f"A-only={len(a_ids - b_ids)}, B-only={len(b_ids - a_ids)}"
        )
    indexed = a.assign(__sample_id=a["sample_id"].astype(str)).set_index("__sample_id", drop=True)
    aligned = indexed.loc[b["sample_id"].astype(str).tolist()].reset_index(drop=True)
    if not np.array_equal(aligned["patient_id"].astype(str), b["patient_id"].astype(str)):
        raise ValueError(f"{dataset_name} Matrix A/B patient_id alignment failed")
    return aligned


def native_descriptor_names() -> list[str]:
    """Return the documented order of the 57 native deterministic descriptors."""

    names: list[str] = []
    regions = ("background", "lv_blood_pool", "lv_myocardium", "rv_blood_pool")
    summaries = ("mean", "sd", "median", "p25", "p75", "voxel_fraction")
    for phase in ("ed", "es"):
        for region in regions:
            names.extend(f"{phase}_{region}_intensity_{summary}" for summary in summaries)
    for region in regions[1:]:
        names.extend(
            (
                f"{region}_mask_fraction_ed",
                f"{region}_mask_fraction_es",
                f"{region}_mask_fraction_ed_minus_es",
            )
        )
    if len(names) != 57:
        raise AssertionError(f"Native descriptor schema must contain 57 entries, found {len(names)}")
    return names


def recover_native_descriptors(
    frozen_a: pd.DataFrame,
    extractor_metadata: Mapping[str, Any],
    *,
    tolerance: float = 1e-10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Recover A0 exactly from the untouched portion of an overcomplete A1.

    The native extractor replaces the first ``min(32, input_dim, output_dim)``
    projected columns with direct descriptors.  The remaining 96 columns of the
    frozen 128-dimensional branch are an injective linear map of all 57 inputs.
    The function verifies the recorded basis checksum, full row rank, and a
    complete A1 round trip before returning A0.
    """

    a_columns = _a_columns(frozen_a)
    latent_dim = int(extractor_metadata.get("latent_dim", len(a_columns)))
    input_dim = int(extractor_metadata.get("projection_input_dimension", 0))
    seed = int(extractor_metadata.get("seed", 0))
    recorded_checksum = str(extractor_metadata.get("projection_checksum", ""))
    if latent_dim != len(a_columns):
        raise ValueError(f"Metadata latent_dim={latent_dim} but Matrix A has {len(a_columns)} features")
    if input_dim <= 0:
        raise ValueError("projection_input_dimension must be positive")
    basis = _projection_matrix(input_dim, latent_dim, seed)
    observed_checksum = _matrix_checksum(basis)
    if not recorded_checksum or observed_checksum != recorded_checksum:
        raise ValueError(
            "Frozen projection checksum mismatch; A0 recovery refused "
            f"(recorded={recorded_checksum!r}, regenerated={observed_checksum!r})"
        )
    direct_n = min(latent_dim, input_dim, 32)
    untouched_basis = basis[:, direct_n:]
    rank = int(np.linalg.matrix_rank(untouched_basis))
    if untouched_basis.shape[1] < input_dim or rank != input_dim:
        raise ValueError(
            "Frozen projected tail is not injective; native descriptors cannot be recovered "
            f"(shape={untouched_basis.shape}, rank={rank}, required_rank={input_dim})"
        )
    projected = frozen_a[a_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(projected).all():
        raise ValueError("Frozen Matrix A contains non-finite projected values")
    raw = projected[:, direct_n:] @ np.linalg.pinv(untouched_basis)
    roundtrip = raw @ basis
    if direct_n:
        roundtrip[:, :direct_n] = raw[:, :direct_n]
    max_error = float(np.max(np.abs(roundtrip - projected))) if projected.size else 0.0
    if not np.isfinite(max_error) or max_error > tolerance:
        raise ValueError(
            f"A0 recovery round-trip error {max_error:.6g} exceeds tolerance {tolerance:.6g}"
        )
    names = native_descriptor_names()
    if len(names) != input_dim:
        names = [f"native_descriptor_{index:03d}" for index in range(input_dim)]
    recovered = pd.DataFrame(raw, columns=[f"a0_{name}" for name in names])
    for column in reversed(IDENTIFIER_COLUMNS):
        if column in frozen_a.columns:
            recovered.insert(0, column, frozen_a[column].to_numpy())
    audit = {
        "projection_input_dimension": input_dim,
        "frozen_latent_dimension": latent_dim,
        "direct_descriptor_columns": direct_n,
        "untouched_projection_shape": [int(value) for value in untouched_basis.shape],
        "untouched_projection_rank": rank,
        "untouched_projection_condition_number": float(np.linalg.cond(untouched_basis)),
        "projection_checksum": observed_checksum,
        "roundtrip_max_absolute_error": max_error,
        "roundtrip_tolerance": float(tolerance),
        "recovery_status": "verified_exact_within_tolerance",
    }
    return recovered, audit


def build_formal_branches(
    a_acdc: pd.DataFrame,
    a_mms: pd.DataFrame,
    metadata_acdc: Mapping[str, Any],
    metadata_mms: Mapping[str, Any],
    *,
    extension_seed_offset: int = 256,
) -> tuple[dict[str, tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame, pd.DataFrame]:
    """Build nested A0/A2/A1/A3 branches without reading native data."""

    for key in ("seed", "latent_dim", "projection_input_dimension", "projection_checksum"):
        if metadata_acdc.get(key) != metadata_mms.get(key):
            raise ValueError(f"ACDC and M&Ms extractor metadata disagree on {key}")
    a0_acdc, audit_acdc = recover_native_descriptors(a_acdc, metadata_acdc)
    a0_mms, audit_mms = recover_native_descriptors(a_mms, metadata_mms)
    a1_columns = _a_columns(a_acdc)
    if a1_columns != _a_columns(a_mms):
        raise ValueError("ACDC and M&Ms Matrix A feature orders differ")
    if len(a1_columns) != 128:
        raise ValueError(f"The frozen primary A1 branch must have 128 features, found {len(a1_columns)}")

    def prefix(frame: pd.DataFrame, dimension: int) -> pd.DataFrame:
        keep = [column for column in IDENTIFIER_COLUMNS if column in frame.columns]
        return frame[keep + a1_columns[:dimension]].copy()

    raw_columns = [column for column in a0_acdc.columns if column.startswith("a0_")]
    extension_seed = int(metadata_acdc["seed"]) + int(extension_seed_offset)
    extension_basis = _projection_matrix(len(raw_columns), 128, extension_seed)

    def expanded(a1: pd.DataFrame, a0: pd.DataFrame) -> pd.DataFrame:
        identifiers = [column for column in IDENTIFIER_COLUMNS if column in a1.columns]
        raw = a0[raw_columns].to_numpy(dtype=float)
        extra = pd.DataFrame(raw @ extension_basis, columns=[f"a_{index:04d}" for index in range(128, 256)])
        return pd.concat([a1[identifiers + a1_columns].reset_index(drop=True), extra], axis=1)

    branches = {
        "A0_native_unprojected_57": (a0_acdc, a0_mms),
        "A2_nested_32": (prefix(a_acdc, 32), prefix(a_mms, 32)),
        "A2_nested_64": (prefix(a_acdc, 64), prefix(a_mms, 64)),
        "A2_nested_96": (prefix(a_acdc, 96), prefix(a_mms, 96)),
        "A1_primary_128": (a_acdc.copy(), a_mms.copy()),
        "A3_nested_expansion_256": (expanded(a_acdc, a0_acdc), expanded(a_mms, a0_mms)),
    }
    rows: list[dict[str, Any]] = []
    train_mask = a_acdc["split"].astype(str).eq("train")
    for name, (acdc, _) in branches.items():
        columns = [column for column in acdc.columns if column.startswith(("a_", "a0_"))]
        values = acdc.loc[train_mask, columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        centered = values - np.nanmean(values, axis=0, keepdims=True)
        rows.append(
            {
                "a_branch": name,
                "declared_dimension": len(columns),
                "training_effective_rank": int(np.linalg.matrix_rank(np.nan_to_num(centered))),
                "nested_prefix_of_A1": name in {"A2_nested_32", "A2_nested_64", "A2_nested_96", "A1_primary_128"},
                "native_image_reread": False,
                "feature_order_sha256": _sha256_json(columns),
                "extension_projection_checksum": _matrix_checksum(extension_basis) if name == "A3_nested_expansion_256" else "",
                "notes": (
                    "A3 preserves all 128 frozen A1 columns and appends a domain-separated deterministic projection."
                    if name == "A3_nested_expansion_256"
                    else ""
                ),
            }
        )
    audit = pd.DataFrame(
        [
            {"dataset_source": "ACDC", **audit_acdc},
            {"dataset_source": "MMS_100", **audit_mms},
        ]
    )
    return branches, pd.DataFrame(rows), audit


def _numeric_non_degenerate_targets(b_acdc: pd.DataFrame, candidates: Iterable[str]) -> list[str]:
    train = b_acdc["split"].astype(str).eq("train")
    output: list[str] = []
    for column in candidates:
        if column not in b_acdc.columns:
            continue
        values = pd.to_numeric(b_acdc.loc[train, column], errors="coerce")
        if int(values.notna().sum()) >= 10 and int(values.nunique(dropna=True)) > 1:
            output.append(column)
    return output


def build_target_families(b_acdc: pd.DataFrame, feature_dictionary: pd.DataFrame) -> tuple[dict[str, list[str]], pd.DataFrame]:
    all_targets = select_transition_targets(b_acdc, feature_dictionary, min_train_observed=10)
    families = {
        "all_transition_eligible": all_targets,
        "ventricular_function": [feature for feature in VENTRICULAR_FUNCTION_TARGETS if feature in all_targets],
        "morphology_and_wall": [feature for feature in all_targets if any(token in feature for token in MORPHOLOGY_TOKENS)],
        "qc_targets": _numeric_non_degenerate_targets(b_acdc, QC_TARGET_CANDIDATES),
        "physician_prioritized_pending": [feature for feature in PHYSICIAN_PRIORITY_CANDIDATES if feature in all_targets],
    }
    rows: list[dict[str, Any]] = []
    for family, targets in families.items():
        status = "pending_physician_signoff_no_fit" if family == "physician_prioritized_pending" else ("ready" if len(targets) >= 2 else "insufficient_targets")
        for ordinal, feature in enumerate(targets, start=1):
            rows.append(
                {
                    "target_family": family,
                    "target_ordinal": ordinal,
                    "feature_name": feature,
                    "status": status,
                    "excluded_by_default_policy": feature in DEFAULT_EXCLUDED_TARGETS,
                }
            )
        if not targets:
            rows.append(
                {
                    "target_family": family,
                    "target_ordinal": np.nan,
                    "feature_name": "",
                    "status": status,
                    "excluded_by_default_policy": False,
                }
            )
    return families, pd.DataFrame(rows)


def _qc_severity(frame: pd.DataFrame) -> pd.Series:
    reasons = frame.get("qc_failure_reason", pd.Series([None] * len(frame), index=frame.index))
    return reasons.map(lambda value: severity_for_reasons(parse_qc_reasons(value)))


def qc_policy_mask(b_acdc: pd.DataFrame, policy: str) -> pd.Series:
    train = b_acdc["split"].astype(str).eq("train")
    severity = _qc_severity(b_acdc)
    if policy == "ungated_ablation":
        return train
    if policy == "exclude_critical_only":
        return train & ~severity.eq("critical")
    if policy == "exclude_high_moderate_critical":
        return train & severity.eq("none")
    raise ValueError(f"Unknown QC policy: {policy}")


def _cv_splits(labels: pd.Series, n_samples: int, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    n_splits = max(2, min(int(folds), int(n_samples)))
    labels = labels.reset_index(drop=True).astype(str)
    if labels.nunique() > 1 and int(labels.value_counts().min()) >= n_splits:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(n_samples), labels))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(n_samples)))


def _ridge_svd(z: np.ndarray, b_centered: np.ndarray, lambda_value: float) -> tuple[np.ndarray, np.ndarray]:
    if z.ndim != 2 or b_centered.ndim != 2 or z.shape[0] != b_centered.shape[0]:
        raise ValueError("Ridge SVD requires aligned two-dimensional A and B matrices")
    if float(lambda_value) <= 0:
        raise ValueError("Ridge lambda must be positive")
    u, singular, vt = np.linalg.svd(z, full_matrices=False)
    coefficients = ridge_from_svd(singular, vt, u.T @ b_centered, float(lambda_value))
    return coefficients, singular


def _regularized_condition(singular_values: np.ndarray, n_features: int, lambda_value: float) -> float:
    squared = np.asarray(singular_values, dtype=float) ** 2
    maximum = float(np.max(squared)) if squared.size else 0.0
    minimum = float(np.min(squared)) if squared.size and len(squared) == n_features else 0.0
    return float((maximum + lambda_value) / (minimum + lambda_value))


def _training_target_scale(b_centered: np.ndarray) -> np.ndarray:
    scale = np.std(b_centered, axis=0, ddof=0)
    return np.where(np.isfinite(scale) & (scale > 1e-12), scale, np.nan)


def _coefficient_cosine_summary(coefficient_matrices: Sequence[np.ndarray]) -> tuple[float, float]:
    similarities: list[float] = []
    for left, right in combinations(coefficient_matrices, 2):
        x = left.ravel()
        y = right.ravel()
        denom = float(np.linalg.norm(x) * np.linalg.norm(y))
        similarities.append(float(np.dot(x, y) / denom) if denom > 0 else np.nan)
    finite = np.asarray([value for value in similarities if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return np.nan, np.nan
    return float(np.median(finite)), float(np.min(finite))


def normalized_cv_lambda_grid(
    a_acdc: pd.DataFrame,
    b_acdc: pd.DataFrame,
    a_columns: Sequence[str],
    b_columns: Sequence[str],
    train_mask: pd.Series,
    *,
    lambda_grid: Iterable[float] = DEFAULT_LAMBDAS,
    cv_folds: int = 5,
    seed: int = 20260611,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Select lambda using train-fold, per-target standardized RMSE."""

    lambdas = tuple(float(value) for value in lambda_grid)
    if not lambdas or any(value <= 0 for value in lambdas):
        raise ValueError("lambda_grid must contain positive values")
    train_indices = np.flatnonzero(train_mask.to_numpy(dtype=bool))
    if len(train_indices) < 4:
        raise ValueError("At least four training patients are required for cross-validation")
    labels = b_acdc.loc[train_mask, "disease_label"] if "disease_label" in b_acdc.columns else pd.Series(["unknown"] * len(train_indices))
    splits = _cv_splits(labels, len(train_indices), cv_folds, seed)
    fold_rows: list[dict[str, Any]] = []
    coefficient_cache: dict[float, list[np.ndarray]] = {value: [] for value in lambdas}
    for fold_id, (train_relative, validation_relative) in enumerate(splits, start=1):
        fit_indices = train_indices[train_relative]
        held_indices = train_indices[validation_relative]
        state = fit_train_only_preprocessing(
            a_acdc.iloc[fit_indices], b_acdc.iloc[fit_indices], list(a_columns), list(b_columns)
        )
        z_train = transform_a(a_acdc.iloc[fit_indices], state)
        b_train_centered = transform_b_centered(b_acdc.iloc[fit_indices], state)
        z_held = transform_a(a_acdc.iloc[held_indices], state)
        held_observed = inverse_b_centered(transform_b_centered(b_acdc.iloc[held_indices], state), state)
        train_scale = _training_target_scale(b_train_centered)
        u, singular, vt = np.linalg.svd(z_train, full_matrices=False)
        projected_targets = u.T @ b_train_centered
        for lambda_value in lambdas:
            coefficients = ridge_from_svd(singular, vt, projected_targets, lambda_value)
            coefficient_cache[lambda_value].append(coefficients)
            predicted = inverse_b_centered(z_held @ coefficients, state)
            residual = predicted - held_observed
            per_target_rmse = np.sqrt(np.nanmean(residual**2, axis=0))
            standardized = per_target_rmse / train_scale
            fold_rows.append(
                {
                    "fold_id": fold_id,
                    "lambda_value": lambda_value,
                    "macro_train_sd_normalized_rmse": float(np.nanmean(standardized)),
                    "mixed_unit_rmse_descriptive": float(np.sqrt(np.nanmean(residual**2))),
                    "mixed_unit_mae_descriptive": float(np.nanmean(np.abs(residual))),
                    "coefficient_frobenius_norm": float(np.linalg.norm(coefficients, ord="fro")),
                    "design_effective_rank": int(np.linalg.matrix_rank(z_train)),
                    "regularized_condition_number": _regularized_condition(singular, z_train.shape[1], lambda_value),
                    "fit_patients": int(len(fit_indices)),
                    "held_out_patients": int(len(held_indices)),
                    "selection_partition": "ACDC_train_internal_CV",
                }
            )
    folds = pd.DataFrame(fold_rows)
    rows: list[dict[str, Any]] = []
    for lambda_value, group in folds.groupby("lambda_value", sort=True):
        cosine_median, cosine_minimum = _coefficient_cosine_summary(coefficient_cache[float(lambda_value)])
        rows.append(
            {
                "lambda_value": float(lambda_value),
                "cv_macro_train_sd_normalized_rmse_mean": float(group["macro_train_sd_normalized_rmse"].mean()),
                "cv_macro_train_sd_normalized_rmse_sd": float(group["macro_train_sd_normalized_rmse"].std(ddof=0)),
                "cv_mixed_unit_rmse_mean_descriptive": float(group["mixed_unit_rmse_descriptive"].mean()),
                "cv_mixed_unit_mae_mean_descriptive": float(group["mixed_unit_mae_descriptive"].mean()),
                "cv_coefficient_norm_mean": float(group["coefficient_frobenius_norm"].mean()),
                "cv_regularized_condition_number_mean": float(group["regularized_condition_number"].mean()),
                "cv_effective_rank_min": int(group["design_effective_rank"].min()),
                "fold_coefficient_cosine_median": cosine_median,
                "fold_coefficient_cosine_minimum": cosine_minimum,
                "cv_folds": int(group["fold_id"].nunique()),
                "selection_objective": "mean_per_target_RMSE_divided_by_training_fold_target_SD",
            }
        )
    summary = pd.DataFrame(rows).sort_values("lambda_value").reset_index(drop=True)
    best = float(summary["cv_macro_train_sd_normalized_rmse_mean"].min())
    tied = summary[summary["cv_macro_train_sd_normalized_rmse_mean"] <= best * 1.001]
    selected = float(tied["lambda_value"].max())
    summary["selected"] = summary["lambda_value"].eq(selected)
    return summary, folds, selected


def fit_selected_transition(
    a_acdc: pd.DataFrame,
    b_acdc: pd.DataFrame,
    a_columns: Sequence[str],
    b_columns: Sequence[str],
    train_mask: pd.Series,
    lambda_value: float,
) -> FittedTransition:
    state = fit_train_only_preprocessing(
        a_acdc.loc[train_mask], b_acdc.loc[train_mask], list(a_columns), list(b_columns)
    )
    z = transform_a(a_acdc.loc[train_mask], state)
    b_centered = transform_b_centered(b_acdc.loc[train_mask], state)
    coefficients, singular = _ridge_svd(z, b_centered, lambda_value)
    train_scale = _training_target_scale(b_centered)
    return FittedTransition(
        coefficients=coefficients,
        state=state,
        lambda_value=float(lambda_value),
        train_sd={feature: float(train_scale[index]) for index, feature in enumerate(b_columns)},
        design_rank=int(np.linalg.matrix_rank(z)),
        regularized_condition_number=_regularized_condition(singular, z.shape[1], lambda_value),
    )


def _stable_seed(base_seed: int, *parts: object) -> int:
    digest = sha256("|".join(str(part) for part in (base_seed, *parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def _bootstrap_intervals(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    if iterations <= 0 or len(observed) < 3:
        return {
            "rmse_ci_lower": np.nan,
            "rmse_ci_upper": np.nan,
            "mae_ci_lower": np.nan,
            "mae_ci_upper": np.nan,
            "bias_ci_lower": np.nan,
            "bias_ci_upper": np.nan,
            "pearson_ci_lower": np.nan,
            "pearson_ci_upper": np.nan,
            "spearman_ci_lower": np.nan,
            "spearman_ci_upper": np.nan,
        }
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(observed), size=(iterations, len(observed)))
    obs = observed[indices]
    pred = predicted[indices]
    residual = pred - obs
    rmse = np.sqrt(np.mean(residual**2, axis=1))
    mae = np.mean(np.abs(residual), axis=1)
    bias = np.mean(residual, axis=1)

    def row_corr(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left_centered = left - left.mean(axis=1, keepdims=True)
        right_centered = right - right.mean(axis=1, keepdims=True)
        denominator = np.sqrt(np.sum(left_centered**2, axis=1) * np.sum(right_centered**2, axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.sum(left_centered * right_centered, axis=1) / denominator

    pearson = row_corr(obs, pred)
    observed_ranks = stats.rankdata(observed)
    predicted_ranks = stats.rankdata(predicted)
    spearman = row_corr(observed_ranks[indices], predicted_ranks[indices])

    def limits(values: np.ndarray) -> tuple[float, float]:
        finite = values[np.isfinite(values)]
        if finite.size < max(10, iterations // 10):
            return np.nan, np.nan
        return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))

    output: dict[str, float] = {}
    for name, values in (("rmse", rmse), ("mae", mae), ("bias", bias), ("pearson", pearson), ("spearman", spearman)):
        lower, upper = limits(values)
        output[f"{name}_ci_lower"] = lower
        output[f"{name}_ci_upper"] = upper
    return output


def _metric_row(
    *,
    observed_series: pd.Series,
    predicted: np.ndarray,
    qc_valid: pd.Series,
    train_sd: float,
    tolerance_sd_fraction: float,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    observed_all = pd.to_numeric(observed_series, errors="coerce").to_numpy(dtype=float)
    predicted_all = np.asarray(predicted, dtype=float)
    finite = np.isfinite(observed_all) & np.isfinite(predicted_all)
    observed = observed_all[finite]
    estimated = predicted_all[finite]
    n = int(len(observed))
    output: dict[str, Any] = {
        "n_total": int(len(observed_all)),
        "n_observed": int(np.isfinite(observed_all).sum()),
        "n_missing": int((~np.isfinite(observed_all)).sum()),
        "n_qc_valid": int(qc_valid.astype(bool).sum()),
        "n_evaluated": n,
        "train_target_sd": float(train_sd) if np.isfinite(train_sd) else np.nan,
        "statistical_tolerance_sd_fraction": float(tolerance_sd_fraction),
        "statistical_tolerance_native_units": float(train_sd * tolerance_sd_fraction) if np.isfinite(train_sd) else np.nan,
        "clinical_tolerance_status": "pending_physician_approval",
        "bootstrap_method": "patient_level_percentile_paired; Spearman CI resamples frozen full-sample ranks",
        "bootstrap_iterations": int(bootstrap_iterations),
    }
    if n == 0:
        for name in (
            "rmse",
            "mae",
            "median_absolute_error",
            "normalized_rmse_by_train_sd",
            "pearson_r",
            "pearson_p",
            "spearman_r",
            "spearman_p",
            "bias",
            "residual_sd",
            "loa_lower",
            "loa_upper",
            "calibration_slope_observed_on_predicted",
            "calibration_intercept_observed_on_predicted",
            "proportion_within_statistical_tolerance",
        ):
            output[name] = np.nan
        output.update(_bootstrap_intervals(observed, estimated, iterations=0, seed=bootstrap_seed))
        return output
    residual = estimated - observed
    rmse = float(np.sqrt(np.mean(residual**2)))
    bias = float(np.mean(residual))
    residual_sd = float(np.std(residual, ddof=1)) if n > 1 else 0.0
    if n >= 3 and np.std(observed) > 0 and np.std(estimated) > 0:
        pearson_r, pearson_p = stats.pearsonr(observed, estimated)
        spearman_r, spearman_p = stats.spearmanr(observed, estimated)
    else:
        pearson_r = pearson_p = spearman_r = spearman_p = np.nan
    if n >= 2 and np.std(estimated) > 0:
        calibration_slope, calibration_intercept = np.polyfit(estimated, observed, 1)
    else:
        calibration_slope = calibration_intercept = np.nan
    tolerance = train_sd * tolerance_sd_fraction if np.isfinite(train_sd) else np.nan
    output.update(
        {
            "rmse": rmse,
            "mae": float(np.mean(np.abs(residual))),
            "median_absolute_error": float(np.median(np.abs(residual))),
            "normalized_rmse_by_train_sd": rmse / train_sd if np.isfinite(train_sd) and train_sd > 0 else np.nan,
            "pearson_r": float(pearson_r) if np.isfinite(pearson_r) else np.nan,
            "pearson_p": float(pearson_p) if np.isfinite(pearson_p) else np.nan,
            "spearman_r": float(spearman_r) if np.isfinite(spearman_r) else np.nan,
            "spearman_p": float(spearman_p) if np.isfinite(spearman_p) else np.nan,
            "bias": bias,
            "residual_sd": residual_sd,
            "loa_lower": bias - 1.96 * residual_sd,
            "loa_upper": bias + 1.96 * residual_sd,
            "calibration_slope_observed_on_predicted": float(calibration_slope) if np.isfinite(calibration_slope) else np.nan,
            "calibration_intercept_observed_on_predicted": float(calibration_intercept) if np.isfinite(calibration_intercept) else np.nan,
            "proportion_within_statistical_tolerance": float(np.mean(np.abs(residual) <= tolerance)) if np.isfinite(tolerance) else np.nan,
        }
    )
    output.update(
        _bootstrap_intervals(
            observed,
            estimated,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        )
    )
    return output


def evaluate_transition_predictions(
    *,
    experiment_id: str,
    dataset_source: str,
    split_name: str,
    b_frame: pd.DataFrame,
    predicted: np.ndarray,
    b_columns: Sequence[str],
    train_sd: Mapping[str, float],
    units: Mapping[str, str],
    lambda_value: float,
    external_preprocessing: str,
    selection_eligible: bool,
    tolerance_sd_fraction: float,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    if predicted.shape != (len(b_frame), len(b_columns)):
        raise ValueError(
            f"Prediction shape {predicted.shape} does not match expected {(len(b_frame), len(b_columns))}"
        )
    qc_valid = ~b_frame.get("qc_manual_review_required", pd.Series(False, index=b_frame.index)).fillna(True).astype(bool)
    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(b_columns):
        metric = _metric_row(
            observed_series=b_frame[feature],
            predicted=predicted[:, index],
            qc_valid=qc_valid,
            train_sd=float(train_sd.get(feature, np.nan)),
            tolerance_sd_fraction=tolerance_sd_fraction,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=_stable_seed(bootstrap_seed, experiment_id, dataset_source, split_name, feature),
        )
        rows.append(
            {
                "experiment_id": experiment_id,
                "dataset_source": dataset_source,
                "split": split_name,
                "feature_name": feature,
                "unit": units.get(feature, ""),
                "lambda_value": float(lambda_value),
                "external_preprocessing": external_preprocessing,
                "selection_eligible": bool(selection_eligible),
                **metric,
            }
        )
    return pd.DataFrame(rows)


def _apply_fdr(metrics: pd.DataFrame) -> pd.DataFrame:
    output = metrics.copy()
    output["pearson_q"] = np.nan
    output["pearson_fdr_rejected"] = False
    output["spearman_q"] = np.nan
    output["spearman_fdr_rejected"] = False
    grouping = ["experiment_id", "dataset_source", "split", "external_preprocessing"]
    for _, indices in output.groupby(grouping, dropna=False).groups.items():
        positions = list(indices)
        for kind in ("pearson", "spearman"):
            adjusted = benjamini_hochberg(output.loc[positions, f"{kind}_p"].tolist(), alpha=0.05)
            output.loc[positions, f"{kind}_q"] = adjusted["q_value"].to_numpy()
            output.loc[positions, f"{kind}_fdr_rejected"] = adjusted["rejected"].to_numpy(dtype=bool)
    return output


def _feature_stability(
    *,
    experiment_id: str,
    a_acdc: pd.DataFrame,
    b_acdc: pd.DataFrame,
    a_columns: Sequence[str],
    b_columns: Sequence[str],
    train_mask: pd.Series,
    lambda_value: float,
    cv_folds: int,
    seed: int,
    active_relative_threshold: float = 0.01,
) -> pd.DataFrame:
    train_indices = np.flatnonzero(train_mask.to_numpy(dtype=bool))
    labels = b_acdc.loc[train_mask, "disease_label"] if "disease_label" in b_acdc.columns else pd.Series(["unknown"] * len(train_indices))
    matrices: list[np.ndarray] = []
    for train_relative, _ in _cv_splits(labels, len(train_indices), cv_folds, seed):
        indices = train_indices[train_relative]
        state = fit_train_only_preprocessing(
            a_acdc.iloc[indices], b_acdc.iloc[indices], list(a_columns), list(b_columns)
        )
        z = transform_a(a_acdc.iloc[indices], state)
        centered = transform_b_centered(b_acdc.iloc[indices], state)
        coefficients, _ = _ridge_svd(z, centered, lambda_value)
        matrices.append(coefficients)
    stack = np.stack(matrices, axis=0)
    rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(b_columns):
        target_coefficients = stack[:, :, target_index]
        median_absolute = np.median(np.abs(target_coefficients), axis=0)
        maximum = float(np.max(median_absolute)) if median_absolute.size else 0.0
        active = median_absolute >= maximum * active_relative_threshold if maximum > 0 else np.zeros_like(median_absolute, dtype=bool)
        signs = np.sign(target_coefficients[:, active]) if np.any(active) else np.empty((len(matrices), 0))
        if signs.size:
            agreement = np.maximum((signs > 0).mean(axis=0), (signs < 0).mean(axis=0))
            sign_mean = float(np.mean(agreement))
            sign_min = float(np.min(agreement))
        else:
            sign_mean = sign_min = np.nan
        pairwise: list[float] = []
        for left, right in combinations(target_coefficients, 2):
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            pairwise.append(float(np.dot(left, right) / denominator) if denominator > 0 else np.nan)
        finite_pairwise = np.asarray([value for value in pairwise if np.isfinite(value)], dtype=float)
        rows.append(
            {
                "experiment_id": experiment_id,
                "feature_name": target,
                "lambda_value": float(lambda_value),
                "cv_folds": len(matrices),
                "active_coefficient_count": int(np.sum(active)),
                "active_relative_magnitude_threshold": float(active_relative_threshold),
                "coefficient_sign_agreement_mean": sign_mean,
                "coefficient_sign_agreement_minimum": sign_min,
                "fold_coefficient_cosine_median": float(np.median(finite_pairwise)) if finite_pairwise.size else np.nan,
                "fold_coefficient_cosine_minimum": float(np.min(finite_pairwise)) if finite_pairwise.size else np.nan,
                "causal_interpretation_allowed": False,
            }
        )
    return pd.DataFrame(rows)


def _coefficient_frame(
    experiment_id: str,
    coefficients: np.ndarray,
    a_columns: Sequence[str],
    b_columns: Sequence[str],
    lambda_value: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(b_columns):
        values = coefficients[:, target_index]
        rank_abs = pd.Series(np.abs(values)).rank(method="first", ascending=False).astype(int).to_numpy()
        for feature_index, formal_feature in enumerate(a_columns):
            coefficient = float(values[feature_index])
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "formal_feature": formal_feature,
                    "semantic_feature": target,
                    "coefficient": coefficient,
                    "coefficient_rank_absolute": int(rank_abs[feature_index]),
                    "direction": "positive" if coefficient > 0 else ("negative" if coefficient < 0 else "zero"),
                    "lambda_value": float(lambda_value),
                    "causal_interpretation_allowed": False,
                }
            )
    return pd.DataFrame(rows)


def _summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    return (
        metrics.groupby(
            ["experiment_id", "dataset_source", "split", "external_preprocessing", "lambda_value", "selection_eligible"],
            dropna=False,
        )
        .agg(
            target_count=("feature_name", "nunique"),
            macro_train_sd_normalized_rmse=("normalized_rmse_by_train_sd", "mean"),
            mixed_unit_rmse_mean_descriptive=("rmse", "mean"),
            mixed_unit_mae_mean_descriptive=("mae", "mean"),
            median_pearson_r=("pearson_r", "median"),
            median_spearman_r=("spearman_r", "median"),
            mean_absolute_bias_native_units_descriptive=("bias", lambda values: float(np.nanmean(np.abs(values)))),
            evaluated_values=("n_evaluated", "sum"),
            observed_values=("n_observed", "sum"),
            missing_values=("n_missing", "sum"),
            qc_valid_patient_feature_values=("n_qc_valid", "sum"),
        )
        .reset_index()
    )


def _state_payload(state: PreprocessingState) -> dict[str, Any]:
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


def _transition_residual_frame(
    observed: pd.DataFrame,
    reconstructed: pd.DataFrame,
    features: Sequence[str],
    *,
    dataset_source: str,
) -> pd.DataFrame:
    """Return one auditable measured/reconstructed residual per case and target."""

    identifiers = [column for column in ("sample_id", "patient_id", "split") if column in observed.columns]
    if "sample_id" not in identifiers:
        raise ValueError("Residual export requires sample_id")
    if observed["sample_id"].astype(str).duplicated().any():
        raise ValueError(f"{dataset_source} observed semantics contain duplicate sample_id values")
    if reconstructed["sample_id"].astype(str).duplicated().any():
        raise ValueError(f"{dataset_source} reconstructed semantics contain duplicate sample_id values")
    observed_by_id = observed.assign(__sample_id=observed["sample_id"].astype(str)).set_index("__sample_id")
    reconstructed_by_id = reconstructed.assign(
        __sample_id=reconstructed["sample_id"].astype(str)
    ).set_index("__sample_id")
    if set(observed_by_id.index) != set(reconstructed_by_id.index):
        raise ValueError(f"{dataset_source} observed/reconstructed sample sets differ")
    sample_order = observed["sample_id"].astype(str).tolist()
    rows: list[pd.DataFrame] = []
    base = observed_by_id.loc[sample_order, identifiers].reset_index(drop=True)
    for feature in features:
        predicted_column = f"{feature}_hat"
        if feature not in observed_by_id.columns or predicted_column not in reconstructed_by_id.columns:
            raise ValueError(f"Residual export lacks {feature!r} or {predicted_column!r}")
        observed_values = pd.to_numeric(
            observed_by_id.loc[sample_order, feature], errors="coerce"
        ).to_numpy(dtype=float)
        predicted_values = pd.to_numeric(
            reconstructed_by_id.loc[sample_order, predicted_column], errors="coerce"
        ).to_numpy(dtype=float)
        frame = base.copy()
        frame["dataset_source"] = dataset_source
        frame["feature_name"] = feature
        frame["observed_value"] = observed_values
        frame["predicted_value"] = predicted_values
        frame["residual_predicted_minus_observed"] = predicted_values - observed_values
        frame["absolute_error"] = np.abs(predicted_values - observed_values)
        frame["finite_pair"] = np.isfinite(observed_values) & np.isfinite(predicted_values)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _infer_synthetic_rules(
    semantics: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
    *,
    minimum_support: int,
    minimum_confidence: float,
) -> set[tuple[int, str, int, str, int]]:
    states = np.where(semantics[:, :4] > thresholds[:4], "high", "low")
    rules: set[tuple[int, str, int, str, int]] = set()
    for left, right in ((0, 1), (2, 3)):
        for left_state in ("low", "high"):
            for right_state in ("low", "high"):
                selected = (states[:, left] == left_state) & (states[:, right] == right_state)
                support = int(selected.sum())
                if support < minimum_support:
                    continue
                counts = pd.Series(labels[selected]).value_counts()
                predicted_label = int(counts.index[0])
                confidence = float(counts.iloc[0] / support)
                if predicted_label != 0 and confidence >= minimum_confidence:
                    rules.add((left, left_state, right, right_state, predicted_label))
    return rules


def _apply_synthetic_rules(
    semantics: np.ndarray,
    thresholds: np.ndarray,
    rules: set[tuple[int, str, int, str, int]],
) -> np.ndarray:
    states = np.where(semantics[:, :4] > thresholds[:4], "high", "low")
    predictions = np.full(len(semantics), -1, dtype=int)
    for row_index in range(len(semantics)):
        fired = {
            label
            for left, left_state, right, right_state, label in rules
            if states[row_index, left] == left_state and states[row_index, right] == right_state
        }
        if len(fired) == 1:
            predictions[row_index] = int(next(iter(fired)))
    return predictions


def _synthetic_labels(semantics: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    first = (semantics[:, 0] > thresholds[0]) & (semantics[:, 1] > thresholds[1])
    second = (semantics[:, 2] <= thresholds[2]) & (semantics[:, 3] > thresholds[3])
    return np.where(first, 1, np.where(second, 2, 0)).astype(int)


def run_synthetic_recovery_benchmark(
    *,
    seed: int = 20260611,
    replicates: int = 5,
    n_train: int = 360,
    n_test: int = 220,
    lambda_value: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exercise T, threshold, rule-set, and abstention recovery deterministically."""

    if replicates <= 0 or n_train < 40 or n_test < 20:
        raise ValueError("Synthetic benchmark requires positive replicates, n_train>=40, and n_test>=20")
    scenarios = [
        ("baseline", "baseline", 0.0, 0.0, 0.0, 0.0, 0.0),
        ("noise_0.05", "noise", 0.05, 0.0, 0.0, 0.0, 0.0),
        ("noise_0.15", "noise", 0.15, 0.0, 0.0, 0.0, 0.0),
        ("noise_0.30", "noise", 0.30, 0.0, 0.0, 0.0, 0.0),
        ("collinearity_0.50", "collinearity", 0.0, 0.50, 0.0, 0.0, 0.0),
        ("collinearity_0.90", "collinearity", 0.0, 0.90, 0.0, 0.0, 0.0),
        ("collinearity_0.99", "collinearity", 0.0, 0.99, 0.0, 0.0, 0.0),
        ("missingness_0.10", "missingness", 0.0, 0.0, 0.10, 0.0, 0.0),
        ("missingness_0.30", "missingness", 0.0, 0.0, 0.30, 0.0, 0.0),
        ("outliers_0.02", "outliers", 0.0, 0.0, 0.0, 0.02, 8.0),
        ("outliers_0.05", "outliers", 0.0, 0.0, 0.0, 0.05, 8.0),
        ("combined_stress", "combined", 0.20, 0.90, 0.20, 0.05, 8.0),
    ]
    k = 12
    l = 8
    true_thresholds = np.zeros(l, dtype=float)
    rows: list[dict[str, Any]] = []
    true_rules = {
        (0, "high", 1, "high", 1),
        (2, "low", 3, "high", 2),
    }
    for scenario_id, family, noise, rho, missingness, outlier_fraction, outlier_magnitude in scenarios:
        for replicate in range(replicates):
            rng = np.random.default_rng(_stable_seed(seed, "synthetic", scenario_id, replicate))
            true_transition = rng.normal(0.0, 0.45, size=(k, l))

            def design(n: int) -> np.ndarray:
                independent = rng.normal(size=(n, k))
                if rho <= 0:
                    return independent
                common = rng.normal(size=(n, 1))
                return np.sqrt(rho) * common + np.sqrt(1.0 - rho) * independent

            a_train = design(n_train)
            a_test = design(n_test)
            b_train_clean = a_train @ true_transition
            b_test_clean = a_test @ true_transition
            b_train_observed = b_train_clean + rng.normal(0.0, noise, size=b_train_clean.shape)
            if outlier_fraction > 0:
                count = max(1, int(round(n_train * outlier_fraction)))
                selected_rows = rng.choice(n_train, size=count, replace=False)
                selected_columns = rng.integers(0, l, size=count)
                b_train_observed[selected_rows, selected_columns] += (
                    rng.choice((-1.0, 1.0), size=count) * outlier_magnitude
                )
            if missingness > 0:
                missing = rng.random(size=b_train_observed.shape) < missingness
                b_train_observed[missing] = np.nan

            a_columns = [f"a_{index:03d}" for index in range(k)]
            b_columns = [f"b_{index:03d}" for index in range(l)]
            a_train_frame = pd.DataFrame(a_train, columns=a_columns)
            b_train_frame = pd.DataFrame(b_train_observed, columns=b_columns)
            state = fit_train_only_preprocessing(a_train_frame, b_train_frame, a_columns, b_columns)
            z_train = transform_a(a_train_frame, state)
            centered = transform_b_centered(b_train_frame, state)
            estimated_transition, _ = _ridge_svd(z_train, centered, lambda_value)
            predicted_train = inverse_b_centered(z_train @ estimated_transition, state)
            predicted_test = inverse_b_centered(
                transform_a(pd.DataFrame(a_test, columns=a_columns), state) @ estimated_transition,
                state,
            )
            expected_standardized_transition = np.diag(
                [float(state.a_sigma[column]) for column in a_columns]
            ) @ true_transition
            transition_error = float(
                np.linalg.norm(estimated_transition - expected_standardized_transition, ord="fro")
                / max(np.linalg.norm(expected_standardized_transition, ord="fro"), 1e-12)
            )
            semantic_scale = np.std(b_test_clean, axis=0, ddof=0)
            semantic_nrmse = float(
                np.mean(
                    np.sqrt(np.mean((predicted_test - b_test_clean) ** 2, axis=0))
                    / np.where(semantic_scale > 1e-12, semantic_scale, np.nan)
                )
            )
            learned_thresholds = np.nanmedian(predicted_train, axis=0)
            threshold_error = float(
                np.mean(
                    np.abs(learned_thresholds - true_thresholds)
                    / np.where(np.std(b_train_clean, axis=0, ddof=0) > 1e-12, np.std(b_train_clean, axis=0, ddof=0), np.nan)
                )
            )
            train_labels = _synthetic_labels(b_train_clean, true_thresholds)
            test_labels = _synthetic_labels(b_test_clean, true_thresholds)
            inferred_rules = _infer_synthetic_rules(
                predicted_train,
                train_labels,
                learned_thresholds,
                minimum_support=max(10, int(round(0.03 * n_train))),
                minimum_confidence=0.80,
            )
            union = true_rules | inferred_rules
            rule_jaccard = float(len(true_rules & inferred_rules) / len(union)) if union else 1.0
            predictions = _apply_synthetic_rules(predicted_test, learned_thresholds, inferred_rules)
            expected_abstention = test_labels == 0
            observed_abstention = predictions < 0
            covered = ~observed_abstention
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_family": family,
                    "replicate": replicate,
                    "noise_sigma": noise,
                    "collinearity_rho": rho,
                    "missingness_fraction": missingness,
                    "outlier_row_fraction": outlier_fraction,
                    "outlier_magnitude_sd": outlier_magnitude,
                    "transition_relative_frobenius_error": transition_error,
                    "semantic_macro_train_scale_normalized_rmse": semantic_nrmse,
                    "threshold_normalized_absolute_error": threshold_error,
                    "rule_set_jaccard": rule_jaccard,
                    "true_rule_count": len(true_rules),
                    "recovered_rule_count": len(inferred_rules),
                    "abstention_routing_accuracy": float(np.mean(expected_abstention == observed_abstention)),
                    "coverage": float(np.mean(covered)),
                    "covered_accuracy": float(np.mean(predictions[covered] == test_labels[covered])) if np.any(covered) else np.nan,
                    "seed": _stable_seed(seed, "synthetic", scenario_id, replicate),
                    "clinical_validity_claim_allowed": False,
                }
            )
    detail = pd.DataFrame(rows)
    numeric = [
        "transition_relative_frobenius_error",
        "semantic_macro_train_scale_normalized_rmse",
        "threshold_normalized_absolute_error",
        "rule_set_jaccard",
        "abstention_routing_accuracy",
        "coverage",
        "covered_accuracy",
    ]
    summary = detail.groupby(["scenario_id", "scenario_family"], as_index=False)[numeric].agg(["mean", "std"])
    summary.columns = [
        "_".join(str(part) for part in column if str(part)).rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    return detail, summary


def _fidelity_gate(
    primary_metrics: pd.DataFrame,
    primary_stability: pd.DataFrame,
) -> pd.DataFrame:
    validation = primary_metrics[
        primary_metrics["dataset_source"].astype(str).eq("ACDC")
        & primary_metrics["split"].astype(str).eq("val")
        & primary_metrics["external_preprocessing"].astype(str).eq("frozen_acdc_train_preprocessing")
    ].copy()
    external = primary_metrics[
        primary_metrics["dataset_source"].astype(str).eq("MMS_100")
        & primary_metrics["external_preprocessing"].astype(str).eq("frozen_acdc_train_preprocessing")
    ][["feature_name", "pearson_r", "normalized_rmse_by_train_sd", "bias"]].rename(
        columns={
            "pearson_r": "external_pearson_r",
            "normalized_rmse_by_train_sd": "external_normalized_rmse_by_train_sd",
            "bias": "external_bias",
        }
    )
    stability = primary_stability[
        [
            "feature_name",
            "coefficient_sign_agreement_mean",
            "fold_coefficient_cosine_median",
        ]
    ]
    merged = validation.merge(stability, on="feature_name", how="left", validate="one_to_one").merge(
        external, on="feature_name", how="left", validate="one_to_one"
    )
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        r = float(row["pearson_r"]) if pd.notna(row["pearson_r"]) else np.nan
        nrmse = float(row["normalized_rmse_by_train_sd"]) if pd.notna(row["normalized_rmse_by_train_sd"]) else np.nan
        train_sd = float(row["train_target_sd"]) if pd.notna(row["train_target_sd"]) else np.nan
        bias = float(row["bias"]) if pd.notna(row["bias"]) else np.nan
        relative_bias = abs(bias) / train_sd if np.isfinite(train_sd) and train_sd > 0 and np.isfinite(bias) else np.inf
        sign_agreement = float(row["coefficient_sign_agreement_mean"]) if pd.notna(row["coefficient_sign_agreement_mean"]) else np.nan
        if (
            np.isfinite(r)
            and np.isfinite(nrmse)
            and np.isfinite(sign_agreement)
            and r >= 0.70
            and nrmse <= 1.00
            and relative_bias <= 0.50
            and sign_agreement >= 0.75
        ):
            validation_status = "green_reliable_for_explanation"
        elif np.isfinite(r) and np.isfinite(nrmse) and r >= 0.40 and nrmse <= 1.50:
            validation_status = "amber_display_with_residual_warning"
        else:
            validation_status = "red_not_for_case_level_use"
        external_nrmse = float(row["external_normalized_rmse_by_train_sd"]) if pd.notna(row["external_normalized_rmse_by_train_sd"]) else np.nan
        external_r = float(row["external_pearson_r"]) if pd.notna(row["external_pearson_r"]) else np.nan
        external_instability = bool(
            (np.isfinite(external_nrmse) and external_nrmse > 2.0)
            or (np.isfinite(external_r) and external_r < 0.0)
        )
        final_status = validation_status
        if external_instability and validation_status == "green_reliable_for_explanation":
            final_status = "amber_display_with_external_instability_warning"
        technical_candidate = final_status == "green_reliable_for_explanation"
        rows.append(
            {
                "feature_name": row["feature_name"],
                "unit": row.get("unit", ""),
                "validation_fidelity_status": validation_status,
                "fidelity_status": final_status,
                "validation_pearson_r": r,
                "validation_normalized_rmse_by_train_sd": nrmse,
                "validation_bias": bias,
                "validation_absolute_bias_by_train_sd": relative_bias,
                "coefficient_sign_agreement": sign_agreement,
                "coefficient_cosine_stability": row["fold_coefficient_cosine_median"],
                "external_pearson_r": external_r,
                "external_normalized_rmse_by_train_sd": external_nrmse,
                "external_bias": row["external_bias"],
                "external_instability_flag": external_instability,
                "technical_rule_antecedent_candidate": technical_candidate,
                "physician_tolerance_status": "pending_physician_approval",
                "rule_antecedent_eligible_final": False,
                "threshold_source": "Phase-5 pre-specified technical gate; physician tolerance pending",
            }
        )
    return pd.DataFrame(rows).sort_values("feature_name").reset_index(drop=True)


def _predict_with_fitted(a_frame: pd.DataFrame, fitted: FittedTransition) -> np.ndarray:
    return inverse_b_centered(transform_a(a_frame, fitted.state) @ fitted.coefficients, fitted.state)


def _evaluate_fitted_splits(
    *,
    experiment_id: str,
    fitted: FittedTransition,
    a_acdc: pd.DataFrame,
    b_acdc: pd.DataFrame,
    a_mms: pd.DataFrame,
    b_mms: pd.DataFrame,
    b_columns: Sequence[str],
    units: Mapping[str, str],
    tolerance_sd_fraction: float,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    predictions_acdc = _predict_with_fitted(a_acdc, fitted)
    for split_name in ("train", "val", "test"):
        mask = b_acdc["split"].astype(str).eq(split_name).to_numpy()
        frames.append(
            evaluate_transition_predictions(
                experiment_id=experiment_id,
                dataset_source="ACDC",
                split_name=split_name,
                b_frame=b_acdc.loc[mask].reset_index(drop=True),
                predicted=predictions_acdc[mask],
                b_columns=b_columns,
                train_sd=fitted.train_sd,
                units=units,
                lambda_value=fitted.lambda_value,
                external_preprocessing="frozen_acdc_train_preprocessing",
                selection_eligible=split_name == "train",
                tolerance_sd_fraction=tolerance_sd_fraction,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_seed=bootstrap_seed,
            )
        )
    predictions_mms = _predict_with_fitted(a_mms, fitted)
    frames.append(
        evaluate_transition_predictions(
            experiment_id=experiment_id,
            dataset_source="MMS_100",
            split_name="mms_100_external",
            b_frame=b_mms.reset_index(drop=True),
            predicted=predictions_mms,
            b_columns=b_columns,
            train_sd=fitted.train_sd,
            units=units,
            lambda_value=fitted.lambda_value,
            external_preprocessing="frozen_acdc_train_preprocessing",
            selection_eligible=False,
            tolerance_sd_fraction=tolerance_sd_fraction,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed,
        )
    )
    bhat_acdc = pd.DataFrame(predictions_acdc, columns=[f"{feature}_hat" for feature in b_columns])
    bhat_mms = pd.DataFrame(predictions_mms, columns=[f"{feature}_hat" for feature in b_columns])
    for source, output, metadata_columns in (
        (b_acdc, bhat_acdc, ("sample_id", "patient_id", "dataset_source", "split", "disease_label")),
        (b_mms, bhat_mms, ("sample_id", "patient_id", "dataset_source", "split")),
    ):
        for column in reversed(metadata_columns):
            if column in source.columns:
                output.insert(0, column, source[column].to_numpy())
    return pd.concat(frames, ignore_index=True), bhat_acdc, bhat_mms


def _regularization_post_selection_diagnostic(
    *,
    a_acdc: pd.DataFrame,
    b_acdc: pd.DataFrame,
    a_mms: pd.DataFrame,
    b_mms: pd.DataFrame,
    a_columns: Sequence[str],
    b_columns: Sequence[str],
    train_mask: pd.Series,
    lambda_grid: Sequence[float],
    selected_lambda: float,
) -> pd.DataFrame:
    """Report every lambda on every split after train-only selection is locked."""

    rows: list[dict[str, Any]] = []
    for lambda_value in lambda_grid:
        fitted = fit_selected_transition(
            a_acdc,
            b_acdc,
            a_columns,
            b_columns,
            train_mask,
            lambda_value,
        )
        datasets = [
            ("ACDC", "train", a_acdc[b_acdc["split"].astype(str).eq("train")], b_acdc[b_acdc["split"].astype(str).eq("train")]),
            ("ACDC", "val", a_acdc[b_acdc["split"].astype(str).eq("val")], b_acdc[b_acdc["split"].astype(str).eq("val")]),
            ("ACDC", "test", a_acdc[b_acdc["split"].astype(str).eq("test")], b_acdc[b_acdc["split"].astype(str).eq("test")]),
            ("MMS_100", "mms_100_external", a_mms, b_mms),
        ]
        for dataset_source, split_name, a_frame, b_frame in datasets:
            predicted = _predict_with_fitted(a_frame.reset_index(drop=True), fitted)
            normalized: list[float] = []
            native_rmse: list[float] = []
            native_mae: list[float] = []
            for target_index, target in enumerate(b_columns):
                observed = pd.to_numeric(b_frame[target], errors="coerce").to_numpy(dtype=float)
                valid = np.isfinite(observed) & np.isfinite(predicted[:, target_index])
                if not np.any(valid):
                    continue
                residual = predicted[valid, target_index] - observed[valid]
                rmse = float(np.sqrt(np.mean(residual**2)))
                native_rmse.append(rmse)
                native_mae.append(float(np.mean(np.abs(residual))))
                scale = fitted.train_sd[target]
                normalized.append(rmse / scale if np.isfinite(scale) and scale > 0 else np.nan)
            rows.append(
                {
                    "lambda_value": float(lambda_value),
                    "selected_by_train_cv": bool(math.isclose(lambda_value, selected_lambda)),
                    "dataset_source": dataset_source,
                    "split": split_name,
                    "selection_eligible": False,
                    "diagnostic_timing": "post_train_CV_selection_lock",
                    "macro_train_sd_normalized_rmse": float(np.nanmean(normalized)),
                    "mixed_unit_rmse_mean_descriptive": float(np.nanmean(native_rmse)),
                    "mixed_unit_mae_mean_descriptive": float(np.nanmean(native_mae)),
                    "coefficient_frobenius_norm": float(np.linalg.norm(fitted.coefficients, ord="fro")),
                    "design_effective_rank": fitted.design_rank,
                    "regularized_condition_number": fitted.regularized_condition_number,
                    "target_count": len(b_columns),
                }
            )
    return pd.DataFrame(rows)


def _estimated_peak_matrix_memory_mb(
    n_train: int,
    a_dimension: int,
    target_count: int,
) -> float:
    elements = (
        n_train * a_dimension
        + n_train * target_count
        + a_dimension * target_count
        + a_dimension * a_dimension
        + 4 * max(n_train, a_dimension) * max(a_dimension, target_count)
    )
    return float(elements * 8 / (1024**2))


def _a0_dictionary() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ordinal, name in enumerate(native_descriptor_names()):
        if "intensity" in name and not name.endswith("voxel_fraction"):
            unit = "normalized_intensity_fraction_of_phase_p01_p99_range"
        else:
            unit = "fraction"
        rows.append(
            {
                "column_name": f"a0_{name}",
                "ordinal": ordinal,
                "descriptor_name": name,
                "unit": unit,
                "source": "native_ED_ES_image_and_project_normalized_mask",
                "clinical_status": "research_use_only_formal_descriptor",
            }
        )
    return pd.DataFrame(rows)


def run_transition_experiment_grid(
    *,
    a_acdc_path: str | Path,
    a_mms_path: str | Path,
    b_acdc_path: str | Path,
    b_mms_path: str | Path,
    feature_dictionary_path: str | Path,
    a_acdc_metadata_path: str | Path,
    a_mms_metadata_path: str | Path,
    output_dir: str | Path,
    experiment_version: str = "phase5_transition_grid_v1",
    lambda_grid: Iterable[float] = DEFAULT_LAMBDAS,
    cv_folds: int = 5,
    seed: int = 20260611,
    bootstrap_iterations: int = 1000,
    tolerance_sd_fraction: float = 0.50,
    synthetic_replicates: int = 5,
    synthetic_n_train: int = 360,
    synthetic_n_test: int = 220,
) -> dict[str, Any]:
    """Execute and persist the complete computationally available Phase-5 grid."""

    started = time.perf_counter()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    input_paths = {
        "A_ACDC": Path(a_acdc_path).resolve(),
        "A_MMS_100": Path(a_mms_path).resolve(),
        "B_ACDC": Path(b_acdc_path).resolve(),
        "B_MMS_100": Path(b_mms_path).resolve(),
        "feature_dictionary": Path(feature_dictionary_path).resolve(),
        "A_ACDC_extractor_metadata": Path(a_acdc_metadata_path).resolve(),
        "A_MMS_100_extractor_metadata": Path(a_mms_metadata_path).resolve(),
    }
    missing_inputs = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"Missing frozen inputs: {missing_inputs}")
    input_hashes = {name: sha256_file(path) for name, path in input_paths.items()}

    a_acdc = _read_table(input_paths["A_ACDC"])
    a_mms = _read_table(input_paths["A_MMS_100"])
    b_acdc = _read_table(input_paths["B_ACDC"])
    b_mms = _read_table(input_paths["B_MMS_100"])
    feature_dictionary = _read_table(input_paths["feature_dictionary"])
    a_acdc = _align_a_to_b(a_acdc, b_acdc, "ACDC")
    a_mms = _align_a_to_b(a_mms, b_mms, "MMS_100")
    if set(b_acdc["split"].astype(str)) != {"train", "val", "test"}:
        raise ValueError("ACDC Matrix B must contain exactly train, val, and test splits")
    if int(b_acdc["patient_id"].nunique()) != len(b_acdc) or int(b_mms["patient_id"].nunique()) != len(b_mms):
        raise ValueError("All transition-grid rows must be patient-unique")
    metadata_acdc = _load_yaml(input_paths["A_ACDC_extractor_metadata"])
    metadata_mms = _load_yaml(input_paths["A_MMS_100_extractor_metadata"])
    branches, branch_manifest, recovery_audit = build_formal_branches(
        a_acdc, a_mms, metadata_acdc, metadata_mms
    )
    target_families, target_manifest = build_target_families(b_acdc, feature_dictionary)
    lambda_values = tuple(float(value) for value in lambda_grid)
    if not lambda_values:
        raise ValueError("At least one lambda value is required")
    units = {
        str(row["column_name"]): "" if pd.isna(row.get("unit")) else str(row.get("unit"))
        for _, row in feature_dictionary.iterrows()
        if "column_name" in row
    }

    specs: list[ExperimentSpec] = [
        ExperimentSpec(
            "primary_A1_all_targets",
            "primary_and_regularization",
            "A1_primary_128",
            "all_transition_eligible",
            "exclude_critical_only",
        )
    ]
    for branch in (
        "A0_native_unprojected_57",
        "A2_nested_32",
        "A2_nested_64",
        "A2_nested_96",
        "A3_nested_expansion_256",
    ):
        specs.append(
            ExperimentSpec(
                f"dimension_{branch}",
                "formal_dimension_sensitivity",
                branch,
                "all_transition_eligible",
                "exclude_critical_only",
            )
        )
    for target_family in ("ventricular_function", "morphology_and_wall", "qc_targets"):
        specs.append(
            ExperimentSpec(
                f"target_{target_family}",
                "target_family_sensitivity",
                "A1_primary_128",
                target_family,
                "exclude_critical_only",
            )
        )
    specs.extend(
        [
            ExperimentSpec(
                "qc_strict_exclusion",
                "qc_gating_ablation",
                "A1_primary_128",
                "all_transition_eligible",
                "exclude_high_moderate_critical",
            ),
            ExperimentSpec(
                "qc_ungated_ablation",
                "qc_gating_ablation",
                "A1_primary_128",
                "all_transition_eligible",
                "ungated_ablation",
                notes="Ablation only; never a recommended configuration.",
            ),
        ]
    )
    registry_rows: list[dict[str, Any]] = [
        {
            "experiment_id": "target_physician_prioritized_pending",
            "experiment_family": "target_family_sensitivity",
            "a_branch": "A1_primary_128",
            "target_family": "physician_prioritized_pending",
            "qc_policy": "exclude_critical_only",
            "status": "blocked_pending_physician_signoff_no_fit",
            "selection_eligible": False,
            "selected_lambda": np.nan,
            "target_count": len(target_families["physician_prioritized_pending"]),
            "training_patient_count": np.nan,
            "compute_seconds": 0.0,
            "estimated_peak_matrix_memory_mb": 0.0,
            "notes": "Candidate list is recorded but cannot be treated as physician-prioritized without physician sign-off.",
        }
    ]
    cv_summaries: list[pd.DataFrame] = []
    cv_folds_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    stability_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []
    primary_fitted: FittedTransition | None = None
    primary_cv: pd.DataFrame | None = None
    primary_bhat_acdc: pd.DataFrame | None = None
    primary_bhat_mms: pd.DataFrame | None = None
    primary_a_acdc: pd.DataFrame | None = None
    primary_a_mms: pd.DataFrame | None = None
    primary_train_mask: pd.Series | None = None

    for spec in specs:
        branch_acdc, branch_mms = branches[spec.a_branch]
        a_columns = [column for column in branch_acdc.columns if column.startswith(("a_", "a0_"))]
        b_columns = target_families[spec.target_family]
        train_mask = qc_policy_mask(b_acdc, spec.qc_policy)
        if len(b_columns) < 2 or int(train_mask.sum()) < 10:
            registry_rows.append(
                {
                    "experiment_id": spec.experiment_id,
                    "experiment_family": spec.experiment_family,
                    "a_branch": spec.a_branch,
                    "target_family": spec.target_family,
                    "qc_policy": spec.qc_policy,
                    "status": "skipped_insufficient_targets_or_training_rows",
                    "selection_eligible": False,
                    "selected_lambda": np.nan,
                    "target_count": len(b_columns),
                    "training_patient_count": int(train_mask.sum()),
                    "compute_seconds": 0.0,
                    "estimated_peak_matrix_memory_mb": 0.0,
                    "notes": spec.notes,
                }
            )
            continue
        experiment_started = time.perf_counter()
        cv_summary, cv_fold_rows, selected_lambda = normalized_cv_lambda_grid(
            branch_acdc,
            b_acdc,
            a_columns,
            b_columns,
            train_mask,
            lambda_grid=lambda_values,
            cv_folds=cv_folds,
            seed=seed,
        )
        cv_summary.insert(0, "experiment_id", spec.experiment_id)
        cv_fold_rows.insert(0, "experiment_id", spec.experiment_id)
        cv_summaries.append(cv_summary)
        cv_folds_frames.append(cv_fold_rows)
        fitted = fit_selected_transition(
            branch_acdc,
            b_acdc,
            a_columns,
            b_columns,
            train_mask,
            selected_lambda,
        )
        metrics, bhat_acdc, bhat_mms = _evaluate_fitted_splits(
            experiment_id=spec.experiment_id,
            fitted=fitted,
            a_acdc=branch_acdc,
            b_acdc=b_acdc,
            a_mms=branch_mms,
            b_mms=b_mms,
            b_columns=b_columns,
            units=units,
            tolerance_sd_fraction=tolerance_sd_fraction,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=seed,
        )
        metric_frames.append(metrics)
        stability = _feature_stability(
            experiment_id=spec.experiment_id,
            a_acdc=branch_acdc,
            b_acdc=b_acdc,
            a_columns=a_columns,
            b_columns=b_columns,
            train_mask=train_mask,
            lambda_value=selected_lambda,
            cv_folds=cv_folds,
            seed=seed,
        )
        stability_frames.append(stability)
        coefficient_frames.append(
            _coefficient_frame(spec.experiment_id, fitted.coefficients, a_columns, b_columns, selected_lambda)
        )
        elapsed = time.perf_counter() - experiment_started
        registry_rows.append(
            {
                "experiment_id": spec.experiment_id,
                "experiment_family": spec.experiment_family,
                "a_branch": spec.a_branch,
                "target_family": spec.target_family,
                "qc_policy": spec.qc_policy,
                "status": "completed",
                "selection_eligible": spec.selection_eligible,
                "selected_lambda": selected_lambda,
                "target_count": len(b_columns),
                "training_patient_count": int(train_mask.sum()),
                "compute_seconds": float(elapsed),
                "estimated_peak_matrix_memory_mb": _estimated_peak_matrix_memory_mb(
                    int(train_mask.sum()), len(a_columns), len(b_columns)
                ),
                "notes": spec.notes,
            }
        )
        if spec.experiment_id == "primary_A1_all_targets":
            primary_fitted = fitted
            primary_cv = cv_summary
            primary_bhat_acdc = bhat_acdc
            primary_bhat_mms = bhat_mms
            primary_a_acdc = branch_acdc
            primary_a_mms = branch_mms
            primary_train_mask = train_mask

    if primary_fitted is None or primary_cv is None or primary_a_acdc is None or primary_a_mms is None or primary_train_mask is None:
        raise RuntimeError("Primary transition experiment did not complete")
    metrics = pd.concat(metric_frames, ignore_index=True)
    stability = pd.concat(stability_frames, ignore_index=True)
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    cv_summary_all = pd.concat(cv_summaries, ignore_index=True)
    cv_folds_all = pd.concat(cv_folds_frames, ignore_index=True)

    primary_targets = target_families["all_transition_eligible"]
    primary_stability = stability[stability["experiment_id"].eq("primary_A1_all_targets")].copy()
    primary_metrics = metrics[metrics["experiment_id"].eq("primary_A1_all_targets")].copy()
    primary_gate = _fidelity_gate(primary_metrics, primary_stability)

    centered_z = transform_a(primary_a_mms, primary_fitted.state)
    centered_z = centered_z - np.mean(centered_z, axis=0, keepdims=True)
    centered_prediction = inverse_b_centered(centered_z @ primary_fitted.coefficients, primary_fitted.state)
    centered_metrics = evaluate_transition_predictions(
        experiment_id="external_unsupervised_input_centering_diagnostic",
        dataset_source="MMS_100",
        split_name="mms_100_external",
        b_frame=b_mms,
        predicted=centered_prediction,
        b_columns=primary_targets,
        train_sd=primary_fitted.train_sd,
        units=units,
        lambda_value=primary_fitted.lambda_value,
        external_preprocessing="unsupervised_external_input_mean_centering_no_labels",
        selection_eligible=False,
        tolerance_sd_fraction=tolerance_sd_fraction,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=seed,
    )
    metrics = pd.concat([metrics, centered_metrics], ignore_index=True)
    registry_rows.append(
        {
            "experiment_id": "external_unsupervised_input_centering_diagnostic",
            "experiment_family": "external_recalibration_diagnostic",
            "a_branch": "A1_primary_128",
            "target_family": "all_transition_eligible",
            "qc_policy": "not_applicable_external_evaluation",
            "status": "completed_exploratory_never_selection_eligible",
            "selection_eligible": False,
            "selected_lambda": primary_fitted.lambda_value,
            "target_count": len(primary_targets),
            "training_patient_count": int(primary_train_mask.sum()),
            "compute_seconds": np.nan,
            "estimated_peak_matrix_memory_mb": _estimated_peak_matrix_memory_mb(
                len(b_mms), len(primary_fitted.state.a_columns), len(primary_targets)
            ),
            "notes": "Uses only the unlabeled external Matrix A mean; external disease labels are not read or used.",
        }
    )

    stability_lookup = stability.rename(
        columns={
            "coefficient_sign_agreement_mean": "coefficient_sign_agreement",
            "fold_coefficient_cosine_median": "coefficient_cosine_stability",
        }
    )[["experiment_id", "feature_name", "coefficient_sign_agreement", "coefficient_cosine_stability"]]
    metrics = metrics.merge(stability_lookup, on=["experiment_id", "feature_name"], how="left", validate="many_to_one")
    external_flags = primary_gate.set_index("feature_name")["external_instability_flag"].to_dict()
    metrics["external_instability_flag"] = metrics.apply(
        lambda row: bool(external_flags.get(row["feature_name"], False))
        if row["experiment_id"] == "primary_A1_all_targets"
        else False,
        axis=1,
    )
    metrics = _apply_fdr(metrics)
    metric_summary = _summarize_metrics(metrics)

    primary_cv_selected = float(primary_cv.loc[primary_cv["selected"].astype(bool), "lambda_value"].iloc[0])
    regularization = _regularization_post_selection_diagnostic(
        a_acdc=primary_a_acdc,
        b_acdc=b_acdc,
        a_mms=primary_a_mms,
        b_mms=b_mms,
        a_columns=primary_fitted.state.a_columns,
        b_columns=primary_targets,
        train_mask=primary_train_mask,
        lambda_grid=lambda_values,
        selected_lambda=primary_cv_selected,
    )

    synthetic_detail, synthetic_summary = run_synthetic_recovery_benchmark(
        seed=seed,
        replicates=synthetic_replicates,
        n_train=synthetic_n_train,
        n_test=synthetic_n_test,
    )

    artifacts: list[Path] = []
    artifacts.extend(_write_frame(pd.DataFrame(registry_rows), output / "transition_experiment_registry"))
    artifacts.extend(_write_frame(metrics, output / "transition_metrics_by_feature"))
    artifacts.extend(_write_frame(metric_summary, output / "transition_metrics_summary"))
    artifacts.extend(_write_frame(coefficients, output / "transition_coefficients"))
    artifacts.extend(_write_frame(stability, output / "transition_stability"))
    artifacts.extend(_write_frame(primary_gate, output / "transition_fidelity_gate"))
    artifacts.extend(_write_frame(cv_summary_all, output / "transition_lambda_cv_summary"))
    artifacts.extend(_write_frame(cv_folds_all, output / "transition_lambda_cv_folds"))
    artifacts.extend(_write_frame(regularization, output / "regularization_sensitivity"))
    artifacts.extend(_write_frame(branch_manifest, output / "formal_branch_manifest"))
    artifacts.extend(_write_frame(recovery_audit, output / "A0_recovery_audit"))
    artifacts.extend(_write_frame(_a0_dictionary(), output / "A0_feature_dictionary"))
    artifacts.extend(_write_frame(target_manifest, output / "transition_target_family_manifest"))
    artifacts.extend(_write_frame(synthetic_detail, output / "synthetic_recovery_benchmark"))
    artifacts.extend(_write_frame(synthetic_summary, output / "synthetic_recovery_summary"))

    if primary_bhat_acdc is None or primary_bhat_mms is None:
        raise RuntimeError("Primary reconstructed semantics were not retained")
    artifacts.extend(_write_frame(primary_bhat_acdc, output / "selected_primary_Bhat_ACDC"))
    artifacts.extend(_write_frame(primary_bhat_mms, output / "selected_primary_Bhat_MMS_100"))
    residuals_acdc = _transition_residual_frame(
        b_acdc,
        primary_bhat_acdc,
        primary_targets,
        dataset_source="ACDC",
    )
    residuals_mms = _transition_residual_frame(
        b_mms,
        primary_bhat_mms,
        primary_targets,
        dataset_source="MMS_100",
    )
    artifacts.extend(
        _write_frame(residuals_acdc, output / "selected_primary_transition_residuals_ACDC")
    )
    artifacts.extend(
        _write_frame(residuals_mms, output / "selected_primary_transition_residuals_MMS_100")
    )
    primary_t_wide = pd.DataFrame(
        primary_fitted.coefficients,
        columns=primary_targets,
    )
    primary_t_wide.insert(0, "formal_feature", primary_fitted.state.a_columns)
    artifacts.extend(_write_frame(primary_t_wide, output / "selected_primary_T_wide"))
    primary_t_long = coefficients[coefficients["experiment_id"].eq("primary_A1_all_targets")].copy()
    artifacts.extend(_write_frame(primary_t_long, output / "selected_primary_T_long"))
    preprocessing_path = _write_json(
        {
            "schema": "tmmri.phase5_transition_preprocessing/1.0",
            "experiment_version": experiment_version,
            "clinical_status": "research_use_only",
            "state": _state_payload(primary_fitted.state),
        },
        output / "selected_primary_preprocessing_state.json",
    )
    artifacts.append(preprocessing_path)

    selected_hash_inputs = {
        "source_input_hashes": input_hashes,
        "target_feature_order_sha256": _sha256_json(primary_targets),
        "formal_feature_order_sha256": _sha256_json(primary_fitted.state.a_columns),
        "preprocessing_state_sha256": sha256_file(preprocessing_path),
        "coefficient_matrix_sha256": sha256_file(output / "selected_primary_T_wide.parquet"),
        "fidelity_gate_sha256": sha256_file(output / "transition_fidelity_gate.parquet"),
        "residuals_acdc_sha256": sha256_file(
            output / "selected_primary_transition_residuals_ACDC.parquet"
        ),
        "residuals_mms_sha256": sha256_file(
            output / "selected_primary_transition_residuals_MMS_100.parquet"
        ),
        "module_sha256": sha256_file(Path(__file__).resolve()),
    }
    selected_manifest = {
        "schema": "tmmri.phase5_selected_transition_model/1.0",
        "experiment_version": experiment_version,
        "generated_at_utc": utc_now_iso(),
        "clinical_status": "research_use_only_preclinical_physician_tolerance_pending",
        "warning": RESEARCH_WARNING,
        "primary_experiment_id": "primary_A1_all_targets",
        "a_branch": "A1_primary_128",
        "selected_lambda": primary_fitted.lambda_value,
        "lambda_selection_partition": "ACDC_train_internal_patient_level_CV_only",
        "lambda_selection_objective": "mean_per_target_RMSE_divided_by_training_fold_target_SD",
        "test_or_external_used_for_selection": False,
        "target_features": primary_targets,
        "target_count": len(primary_targets),
        "formal_dimension": len(primary_fitted.state.a_columns),
        "training_patient_count": int(primary_train_mask.sum()),
        "validation_patient_count": int(b_acdc["split"].astype(str).eq("val").sum()),
        "test_patient_count": int(b_acdc["split"].astype(str).eq("test").sum()),
        "external_patient_count": len(b_mms),
        "physician_prioritized_target_set_status": "pending_not_fit",
        "physician_tolerance_status": "pending",
        "fidelity_gate_status": "technical_only_not_final_until_physician_approval",
        "stable_artifacts": {
            "Bhat_ACDC": "selected_primary_Bhat_ACDC.parquet",
            "Bhat_MMS_100": "selected_primary_Bhat_MMS_100.parquet",
            "T_wide": "selected_primary_T_wide.parquet",
            "T_long": "selected_primary_T_long.parquet",
            "preprocessing": "selected_primary_preprocessing_state.json",
            "fidelity_gate": "transition_fidelity_gate.parquet",
            "residuals_ACDC": "selected_primary_transition_residuals_ACDC.parquet",
            "residuals_MMS_100": "selected_primary_transition_residuals_MMS_100.parquet",
        },
        "hashes": selected_hash_inputs,
    }
    selected_manifest["selected_model_bundle_sha256"] = _sha256_json(selected_hash_inputs)
    selected_manifest_path = _write_json(selected_manifest, output / "selected_model_manifest.json")
    artifacts.append(selected_manifest_path)

    artifact_hashes = [
        {
            "path": path.name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(set(artifacts), key=lambda item: item.name)
    ]
    artifacts.extend(_write_frame(pd.DataFrame(artifact_hashes), output / "artifact_sha256_manifest"))
    run_manifest = {
        "schema": "tmmri.phase5_transition_grid_run/1.0",
        "experiment_version": experiment_version,
        "generated_at_utc": utc_now_iso(),
        "status": "COMPUTATIONAL_GRID_COMPLETE_PHYSICIAN_PRIORITY_AND_TOLERANCE_PENDING",
        "warning": RESEARCH_WARNING,
        "input_paths": {name: str(path) for name, path in input_paths.items()},
        "input_sha256": input_hashes,
        "executed_experiment_count": int(sum(row.get("status") == "completed" for row in registry_rows)),
        "blocked_experiment_count": int(sum(str(row.get("status", "")).startswith("blocked") for row in registry_rows)),
        "selected_primary_lambda": primary_fitted.lambda_value,
        "selected_model_manifest": selected_manifest_path.name,
        "artifact_manifest": "artifact_sha256_manifest.csv",
        "elapsed_seconds": float(time.perf_counter() - started),
        "figures_generated": 0,
    }
    run_manifest_path = _write_json(run_manifest, output / "transition_grid_run_manifest.json")
    return {
        "output_dir": str(output),
        "run_manifest": str(run_manifest_path),
        "selected_model_manifest": str(selected_manifest_path),
        "selected_lambda": primary_fitted.lambda_value,
        "executed_experiments": run_manifest["executed_experiment_count"],
        "blocked_experiments": run_manifest["blocked_experiment_count"],
    }


__all__ = [
    "DEFAULT_LAMBDAS",
    "ExperimentSpec",
    "build_formal_branches",
    "build_target_families",
    "native_descriptor_names",
    "normalized_cv_lambda_grid",
    "qc_policy_mask",
    "recover_native_descriptors",
    "run_synthetic_recovery_benchmark",
    "run_transition_experiment_grid",
]
