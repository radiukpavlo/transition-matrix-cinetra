from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import rotate

from .formal_features import _intensity_summaries, _load_native_phase_arrays, _projection_matrix
from .io import resolve_path, write_table_pair
from .transition import PreprocessingState, inverse_b_centered, transform_a


def _native_vector_from_arrays(image_ed: np.ndarray, image_es: np.ndarray, mask_ed: np.ndarray, mask_es: np.ndarray) -> np.ndarray:
    vals = _intensity_summaries(image_ed, mask_ed) + _intensity_summaries(image_es, mask_es)
    for label in [1, 2, 3]:
        ed_frac = float(np.mean(mask_ed == label))
        es_frac = float(np.mean(mask_es == label))
        vals.extend([ed_frac, es_frac, ed_frac - es_frac])
    return np.asarray(vals, dtype=float)


def _project_native_vector(vec: np.ndarray, latent_dim: int, seed: int) -> np.ndarray:
    mat = _projection_matrix(int(vec.size), int(latent_dim), int(seed))
    projected = vec.reshape(1, -1) @ mat
    direct_n = min(int(latent_dim), int(vec.size), 32)
    if direct_n:
        projected[0, :direct_n] = vec[:direct_n]
    return projected.reshape(-1)


def _shift_zero(arr: np.ndarray, shift_xy: tuple[int, int]) -> np.ndarray:
    sx, sy = shift_xy
    shifted = np.roll(arr, shift=(sx, sy), axis=(0, 1)).copy()
    if sx > 0:
        shifted[:sx, ...] = 0
    elif sx < 0:
        shifted[sx:, ...] = 0
    if sy > 0:
        shifted[:, :sy, ...] = 0
    elif sy < 0:
        shifted[:, sy:, ...] = 0
    return shifted


def _rotate_inplane(arr: np.ndarray, angle_deg: float, *, is_mask: bool) -> np.ndarray:
    order = 0 if is_mask else 1
    rotated = rotate(arr, angle=float(angle_deg), axes=(0, 1), reshape=False, order=order, mode="constant", cval=0.0, prefilter=not is_mask)
    if is_mask:
        return np.clip(np.rint(rotated), 0, 3).astype(int)
    return rotated


def _mask_count_vector(mask_ed: np.ndarray, mask_es: np.ndarray) -> np.ndarray:
    counts: list[float] = []
    for mask in [mask_ed, mask_es]:
        for label in [1, 2, 3]:
            counts.append(float(np.count_nonzero(mask == label)))
    return np.asarray(counts, dtype=float)


def _predict_from_latent(latent: np.ndarray, t_matrix: np.ndarray, transition_state: PreprocessingState) -> np.ndarray:
    row = pd.DataFrame([latent], columns=transition_state.a_columns)
    return inverse_b_centered(transform_a(row, transition_state) @ t_matrix, transition_state).reshape(-1)


def _feature_drift(values_a: np.ndarray, values_b: np.ndarray) -> tuple[float, float, float]:
    diff = np.asarray(values_b, dtype=float) - np.asarray(values_a, dtype=float)
    base = np.asarray(values_a, dtype=float)
    denom = max(float(np.linalg.norm(base)), 1e-12)
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    return float(np.linalg.norm(diff)), float(np.linalg.norm(diff) / denom), max_abs


def _safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else 0.0


def _perturbation_pairs(
    image_ed: np.ndarray,
    image_es: np.ndarray,
    mask_ed: np.ndarray,
    mask_es: np.ndarray,
) -> list[tuple[str, str, float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    return [
        (
            "translation_x_1px_central",
            "translation",
            1.0,
            (_shift_zero(image_ed, (1, 0)), _shift_zero(image_es, (1, 0)), _shift_zero(mask_ed, (1, 0)), _shift_zero(mask_es, (1, 0))),
            (_shift_zero(image_ed, (-1, 0)), _shift_zero(image_es, (-1, 0)), _shift_zero(mask_ed, (-1, 0)), _shift_zero(mask_es, (-1, 0))),
        ),
        (
            "translation_y_1px_central",
            "translation",
            1.0,
            (_shift_zero(image_ed, (0, 1)), _shift_zero(image_es, (0, 1)), _shift_zero(mask_ed, (0, 1)), _shift_zero(mask_es, (0, 1))),
            (_shift_zero(image_ed, (0, -1)), _shift_zero(image_es, (0, -1)), _shift_zero(mask_ed, (0, -1)), _shift_zero(mask_es, (0, -1))),
        ),
        (
            "rotation_2deg_central",
            "rotation",
            2.0,
            (
                _rotate_inplane(image_ed, 2.0, is_mask=False),
                _rotate_inplane(image_es, 2.0, is_mask=False),
                _rotate_inplane(mask_ed, 2.0, is_mask=True),
                _rotate_inplane(mask_es, 2.0, is_mask=True),
            ),
            (
                _rotate_inplane(image_ed, -2.0, is_mask=False),
                _rotate_inplane(image_es, -2.0, is_mask=False),
                _rotate_inplane(mask_ed, -2.0, is_mask=True),
                _rotate_inplane(mask_es, -2.0, is_mask=True),
            ),
        ),
        (
            "intensity_scale_5pct_central",
            "magnitude",
            0.05,
            (image_ed * 1.05, image_es * 1.05, mask_ed, mask_es),
            (image_ed * 0.95, image_es * 0.95, mask_ed, mask_es),
        ),
    ]


def run_equivariance_diagnostics(
    experiment_id: str,
    native_images_available: bool,
    b_acdc: pd.DataFrame | None = None,
    a_acdc: pd.DataFrame | None = None,
    t_matrix: np.ndarray | None = None,
    transition_state: PreprocessingState | None = None,
    seed: int = 20260611,
    max_cases: int = 12,
) -> list[Path]:
    _ = a_acdc
    can_run = native_images_available and b_acdc is not None and t_matrix is not None and transition_state is not None
    if not can_run:
        metrics = pd.DataFrame(
            [
                {
                    "experiment_id": experiment_id,
                    "diagnostic": "small_translation_rotation_magnitude_stability",
                    "status": "skipped_native_images_or_transition_state_unavailable",
                    "clinical_status": "research_use_only_exploratory_only",
                }
            ]
        )
        artifacts = write_table_pair(metrics, f"artifacts/reports/equivariance/equivariance_metrics_{experiment_id}.parquet", f"artifacts/reports/equivariance/equivariance_metrics_{experiment_id}.csv")
        summary = metrics.copy()
    else:
        latent_dim = len(transition_state.a_columns)
        rows: list[dict[str, object]] = []
        eligible = b_acdc.reset_index(drop=True)
        if "split" in eligible.columns:
            parts = []
            for split in ["train", "val", "test"]:
                split_df = eligible[eligible["split"].astype(str).eq(split)].head(max(1, int(max_cases) // 3))
                parts.append(split_df)
            eligible = pd.concat(parts, ignore_index=True).drop_duplicates("sample_id").head(int(max_cases))
        else:
            eligible = eligible.head(int(max_cases))

        for _, row in eligible.iterrows():
            try:
                image_ed, image_es, mask_ed, mask_es = _load_native_phase_arrays(row)
                vec0 = _native_vector_from_arrays(image_ed, image_es, mask_ed, mask_es)
                latent0 = _project_native_vector(vec0, latent_dim=latent_dim, seed=seed)
                pred0 = _predict_from_latent(latent0, t_matrix, transition_state)
                counts0 = _mask_count_vector(mask_ed, mask_es)
                volume_idx = [i for i, name in enumerate(transition_state.b_columns) if any(token in name for token in ["edv", "esv", "sv_ml", "volume", "mass"])]
                for diagnostic, family, magnitude, plus_arrays, minus_arrays in _perturbation_pairs(image_ed, image_es, mask_ed, mask_es):
                    vec_plus = _native_vector_from_arrays(*plus_arrays)
                    vec_minus = _native_vector_from_arrays(*minus_arrays)
                    latent_plus = _project_native_vector(vec_plus, latent_dim=latent_dim, seed=seed)
                    latent_minus = _project_native_vector(vec_minus, latent_dim=latent_dim, seed=seed)
                    latent_central = (latent_plus + latent_minus) / 2.0
                    pred_central = _predict_from_latent(latent_central, t_matrix, transition_state)
                    counts_plus = _mask_count_vector(plus_arrays[2], plus_arrays[3])
                    counts_minus = _mask_count_vector(minus_arrays[2], minus_arrays[3])
                    counts_central = (counts_plus + counts_minus) / 2.0
                    latent_l2, latent_rel, latent_max = _feature_drift(latent0, latent_central)
                    semantic_abs = np.abs(pred_central - pred0)
                    count_abs = np.abs(counts_central - counts0)
                    count_rel = count_abs / np.maximum(counts0, 1.0)
                    finite_difference_norm = float(np.linalg.norm(latent_plus - latent_minus) / (2.0 * max(float(magnitude), 1e-12)))
                    rows.append(
                        {
                            "experiment_id": experiment_id,
                            "sample_id": row.get("sample_id"),
                            "patient_id": row.get("patient_id"),
                            "case_id": row.get("case_id"),
                            "split": row.get("split"),
                            "disease_label": row.get("disease_label"),
                            "diagnostic": diagnostic,
                            "perturbation_family": family,
                            "perturbation_magnitude": float(magnitude),
                            "status": "completed_exploratory",
                            "latent_l2_change": latent_l2,
                            "latent_relative_l2_change": latent_rel,
                            "latent_max_abs_change": latent_max,
                            "central_finite_difference_norm": finite_difference_norm,
                            "reconstructed_semantic_mean_abs_change": _safe_mean(semantic_abs),
                            "reconstructed_semantic_max_abs_change": float(np.max(semantic_abs)) if semantic_abs.size else 0.0,
                            "reconstructed_volume_feature_mean_abs_change": _safe_mean(semantic_abs[volume_idx]) if volume_idx else 0.0,
                            "measured_mask_count_mean_abs_change_voxels": _safe_mean(count_abs),
                            "measured_mask_count_max_abs_change_voxels": float(np.max(count_abs)) if count_abs.size else 0.0,
                            "measured_mask_count_mean_relative_change": _safe_mean(count_rel),
                            "clinical_status": "research_use_only_exploratory_only",
                        }
                    )
            except Exception as exc:
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "sample_id": row.get("sample_id"),
                        "patient_id": row.get("patient_id"),
                        "case_id": row.get("case_id"),
                        "split": row.get("split"),
                        "disease_label": row.get("disease_label"),
                        "diagnostic": "native_case_equivariance",
                        "status": f"failed:{type(exc).__name__}:{exc}",
                        "clinical_status": "research_use_only_exploratory_only",
                    }
                )

        if not rows:
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "diagnostic": "small_translation_rotation_magnitude_stability",
                    "status": "skipped_no_eligible_cases",
                    "clinical_status": "research_use_only_exploratory_only",
                }
            )
        metrics = pd.DataFrame(rows)
        artifacts = write_table_pair(metrics, f"artifacts/reports/equivariance/equivariance_metrics_{experiment_id}.parquet", f"artifacts/reports/equivariance/equivariance_metrics_{experiment_id}.csv")
        numeric_cols = [
            "latent_relative_l2_change",
            "latent_max_abs_change",
            "central_finite_difference_norm",
            "reconstructed_semantic_mean_abs_change",
            "reconstructed_volume_feature_mean_abs_change",
            "measured_mask_count_mean_relative_change",
        ]
        completed = metrics[metrics["status"].astype(str).eq("completed_exploratory")].copy()
        if len(completed):
            summary = completed.groupby(["diagnostic", "perturbation_family"], dropna=False)[numeric_cols].agg(["mean", "max"]).reset_index()
            summary.columns = ["_".join([str(x) for x in col if str(x)]) for col in summary.columns.to_flat_index()]
            summary.insert(0, "experiment_id", experiment_id)
            summary["status"] = "completed_exploratory"
            artifacts.extend(write_table_pair(summary, f"artifacts/reports/equivariance/equivariance_summary_{experiment_id}.parquet", f"artifacts/reports/equivariance/equivariance_summary_{experiment_id}.csv"))
        else:
            summary = metrics.copy()
            artifacts.extend(write_table_pair(summary, f"artifacts/reports/equivariance/equivariance_summary_{experiment_id}.parquet", f"artifacts/reports/equivariance/equivariance_summary_{experiment_id}.csv"))

    md = resolve_path(f"artifacts/reports/equivariance/equivariant_transition_report_{experiment_id}.md")
    md.parent.mkdir(parents=True, exist_ok=True)
    status_counts = metrics["status"].value_counts().to_dict() if len(metrics) and "status" in metrics.columns else {}
    status_lines = "\n".join(f"- {status}: {count}" for status, count in status_counts.items()) or "- none: 0"
    summary_text = summary.to_markdown(index=False) if len(summary) else "No summary rows were generated."
    md.write_text(
        f"# Exploratory Equivariance Diagnostics - {experiment_id}\n\n"
        "Clinical status: research use only. These diagnostics do not validate deployment robustness, diagnosis, triage, treatment selection, or patient management.\n\n"
        "The diagnostic applies small in-plane translations, small rotations, and magnitude perturbations to native ED/ES images and masks. "
        "It reports native latent-feature stability, transition-reconstructed semantic stability, and measured mask-count invariance for volume-sensitive labels.\n\n"
        "## Status Counts\n\n"
        f"{status_lines}\n\n"
        "## Summary\n\n"
        f"{summary_text}\n",
        encoding="utf-8",
    )
    artifacts.append(md)
    return artifacts


# ===============================================================================================================
# Protocol v6 equivariance diagnostics (SPECIFICATION.md 6.0.0, Section 10; Decision D19; configs/equivariance.yaml).
# Optional, diagnostic only; never feeds the primary rulebook. run_equivariance_diagnostics above is unchanged.
# ===============================================================================================================
import re as _re
from typing import Any, Callable, Iterable

from scipy import ndimage as _ndimage

_INTERPOLATION_ORDER = {"nearest": 0, "linear": 1}


def _check_equivariance_cfg(cfg: dict[str, Any]) -> None:
    if bool(cfg.get("feeds_primary_rulebook", False)):
        raise ValueError("configs/equivariance.yaml feeds_primary_rulebook must be false")


def select_equivariance_cases(patient_ids: Iterable[str], cfg: dict[str, Any]) -> list[str]:
    """Seeded case selection (configs/equivariance.yaml case_selection): ``default_rng(seed).choice(sorted ids,
    size=n_cases, replace=False)``, in draw order. Replaces the v5 ``head()`` selection."""
    _check_equivariance_cfg(cfg)
    sel = cfg["case_selection"]
    seed, n = int(sel["seed"]), int(sel["n_cases"])
    rule = str(sel.get("rule", ""))
    m = _re.search(r"default_rng\((\d+)\).*size=(\d+)", rule)
    if m and (int(m.group(1)) != seed or int(m.group(2)) != n):
        raise ValueError("configs/equivariance.yaml case_selection rule disagrees with seed or n_cases")
    ids = sorted(str(p) for p in patient_ids)
    if n > len(ids):
        raise ValueError(f"Cannot select {n} cases from {len(ids)}")
    chosen = np.random.default_rng(seed).choice(np.array(ids, dtype=object), size=n, replace=False)
    return [str(c) for c in chosen]


def transformation_steps(cfg: dict[str, Any]) -> list[tuple[str, float, float, float]]:
    """(name, minus step, plus step, epsilon) per configured transformation; epsilons come from the config."""
    out = []
    for name, spec in cfg["transformations"].items():
        if not isinstance(spec, dict):
            continue
        if "steps_px" in spec:
            lo, hi = (float(v) for v in spec["steps_px"])
        elif "steps_deg" in spec:
            lo, hi = (float(v) for v in spec["steps_deg"])
        elif "factors" in spec:
            lo, hi = (float(v) - 1.0 for v in spec["factors"])
        else:
            raise ValueError(f"Transformation {name} has no steps")
        if not np.isclose(-lo, hi):
            raise ValueError(f"Transformation {name} steps must be symmetric for central differences")
        out.append((name, lo, hi, hi))
    return out


def apply_transformation_v6(image: np.ndarray, mask: np.ndarray, name: str, step: float, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Apply one configured in-plane action jointly to a 3D image and mask (zero padding; image linear, mask
    nearest interpolation; intensity scaling leaves the mask unchanged)."""
    spec = cfg["transformations"][name]
    if cfg["transformations"].get("boundary", "zero_padding") != "zero_padding":
        raise ValueError("Only zero_padding boundaries are implemented")
    img = np.asarray(image, dtype=float)
    msk = np.asarray(mask)
    if name.startswith("translation_"):
        axis = 0 if name.endswith("_x") else 1
        shift = [0.0] * img.ndim
        shift[axis] = float(step)
        io = _INTERPOLATION_ORDER[spec.get("image_interpolation", "linear")]
        mo = _INTERPOLATION_ORDER[spec.get("mask_interpolation", "nearest")]
        return (_ndimage.shift(img, shift, order=io, mode="constant", cval=0.0),
                _ndimage.shift(msk.astype(float), shift, order=mo, mode="constant", cval=0.0).round().astype(msk.dtype))
    if name == "rotation":
        if spec.get("center", "slice_center") != "slice_center":
            raise ValueError("Only slice_center rotation is implemented")
        io = _INTERPOLATION_ORDER[spec.get("image_interpolation", "linear")]
        mo = _INTERPOLATION_ORDER[spec.get("mask_interpolation", "nearest")]
        return (_ndimage.rotate(img, float(step), axes=(0, 1), reshape=False, order=io, mode="constant", cval=0.0),
                _ndimage.rotate(msk.astype(float), float(step), axes=(0, 1), reshape=False, order=mo, mode="constant", cval=0.0).round().astype(msk.dtype))
    if name == "intensity_scale":
        return img * (1.0 + float(step)), msk.copy()
    raise ValueError(f"Unknown transformation {name}")


def equivariance_diagnostics_v6(
    cases: Iterable[Any],
    formal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    semantic_fn: Callable[[Any, np.ndarray, np.ndarray], np.ndarray],
    predict_fn: Callable[[np.ndarray], np.ndarray],
    cfg: dict[str, Any],
    *,
    target_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Central-difference generators of formal A (``formal_fn(ed_image, es_image, ed_mask, es_mask)``), of measured B
    on the transformed masks (``semantic_fn(case, ed_mask, es_mask)``) and of the reconstruction
    (``predict_fn(A_rows) -> B-hat rows``). Per case: generator residual ||g_Bhat - g_B|| / ||g_B|| and finite-action
    defect ||(T(A+) - T(A0)) - (B+ - B0)|| / ||B+ - B0||. Per transformation and target: least-squares slope of the
    measured on the reconstructed generator over the selected cases, and its relative residual. Diagnostic only."""
    _check_equivariance_cfg(cfg)
    steps = transformation_steps(cfg)
    case_rows: list[dict[str, Any]] = []
    gens: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {name: [] for name, *_ in steps}
    for case in cases:
        a0 = np.asarray(formal_fn(case.ed_image, case.es_image, case.ed_mask, case.es_mask), dtype=float)
        b0 = np.asarray(semantic_fn(case, case.ed_mask, case.es_mask), dtype=float)
        bh0 = np.asarray(predict_fn(a0[None, :]), dtype=float).reshape(-1)
        for name, lo, hi, eps in steps:
            plus = [apply_transformation_v6(case.ed_image, case.ed_mask, name, hi, cfg), apply_transformation_v6(case.es_image, case.es_mask, name, hi, cfg)]
            minus = [apply_transformation_v6(case.ed_image, case.ed_mask, name, lo, cfg), apply_transformation_v6(case.es_image, case.es_mask, name, lo, cfg)]
            a_p = np.asarray(formal_fn(plus[0][0], plus[1][0], plus[0][1], plus[1][1]), dtype=float)
            a_m = np.asarray(formal_fn(minus[0][0], minus[1][0], minus[0][1], minus[1][1]), dtype=float)
            b_p = np.asarray(semantic_fn(case, plus[0][1], plus[1][1]), dtype=float)
            b_m = np.asarray(semantic_fn(case, minus[0][1], minus[1][1]), dtype=float)
            bh_p = np.asarray(predict_fn(a_p[None, :]), dtype=float).reshape(-1)
            bh_m = np.asarray(predict_fn(a_m[None, :]), dtype=float).reshape(-1)
            g_a = (a_p - a_m) / (2.0 * eps)
            g_b = (b_p - b_m) / (2.0 * eps)
            g_bh = (bh_p - bh_m) / (2.0 * eps)
            ok = np.isfinite(g_b) & np.isfinite(g_bh)
            gb_norm = float(np.linalg.norm(g_b[ok]))
            fa_meas = (b_p - b0)[ok]
            fa_def = (bh_p - bh0)[ok] - fa_meas
            case_rows.append(
                {
                    "sample_id": getattr(case, "sample_id", None),
                    "transformation": name,
                    "epsilon": eps,
                    "formal_generator_norm": float(np.linalg.norm(g_a[np.isfinite(g_a)])),
                    "semantic_generator_norm": gb_norm,
                    "reconstructed_generator_norm": float(np.linalg.norm(g_bh[ok])),
                    "generator_residual_rel": float(np.linalg.norm(g_bh[ok] - g_b[ok]) / gb_norm) if gb_norm > 0 else np.nan,
                    "finite_action_defect_rel": float(np.linalg.norm(fa_def) / np.linalg.norm(fa_meas)) if np.linalg.norm(fa_meas) > 0 else np.nan,
                    "status": "completed_diagnostic",
                    "feeds_primary_rulebook": False,
                }
            )
            gens[name].append((g_b, g_bh))
    target_rows = []
    for name, pairs in gens.items():
        if not pairs:
            continue
        gb = np.vstack([p[0] for p in pairs])
        gbh = np.vstack([p[1] for p in pairs])
        names = target_names or [f"target_{j}" for j in range(gb.shape[1])]
        for j, tname in enumerate(names):
            ok = np.isfinite(gb[:, j]) & np.isfinite(gbh[:, j])
            x, y = gbh[ok, j], gb[ok, j]
            denom = float(x @ x)
            slope = float(x @ y / denom) if denom > 0 else np.nan
            resid = float(np.linalg.norm(y - slope * x) / np.linalg.norm(y)) if np.isfinite(slope) and np.linalg.norm(y) > 0 else np.nan
            target_rows.append({"transformation": name, "target": tname, "n_cases": int(ok.sum()), "generator_ls_slope": slope, "generator_ls_residual_rel": resid, "feeds_primary_rulebook": False})
    return pd.DataFrame(case_rows), pd.DataFrame(target_rows)
