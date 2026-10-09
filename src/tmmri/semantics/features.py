from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd

from ..ingest.geometry import geometry_from_image, select_frame, serialize_geometry
from ..io import resolve_path, write_json
from ..ingest.labels import LV_BLOOD_POOL, LV_MYOCARDIUM, RV_BLOOD_POOL, normalize_labels, validate_project_labels
from .measurements import (
    bmi_and_bsa,
    finite_or_none,
    global_wall_thickness_mm,
    indexed,
    lv_sphericity_index,
    mask_volume_ml,
    safe_ef,
    safe_sv,
    shape_summary,
    short_axis_long_axis_approx_mm,
    wall_thickening_percent,
)


def _none_if_na(value: object) -> object | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def _float_or_none(value: object) -> float | None:
    value = _none_if_na(value)
    if value is None:
        return None
    try:
        return finite_or_none(float(value))
    except (TypeError, ValueError):
        return None


def _int_or_none(value: object) -> int | None:
    value = _none_if_na(value)
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _load_normalized_mask(path: str | Path, dataset_source: str, frame_index: int | None = None) -> tuple[np.ndarray, set[int], nib.spatialimages.SpatialImage]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    frame = select_frame(data, frame_index)
    labels = np.rint(frame).astype(np.int16, copy=False)
    normalized, unexpected = normalize_labels(labels, dataset_source)
    return normalized, unexpected, img


def _load_nifti_frame(path: str | Path, frame_index: int | None = None) -> tuple[np.ndarray, nib.spatialimages.SpatialImage]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    return select_frame(data, frame_index), img


def _save_3d_nifti(data: np.ndarray, source_img: nib.spatialimages.SpatialImage, out_path: Path, dtype: np.dtype | type | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = data.astype(dtype) if dtype is not None else data
    out = nib.Nifti1Image(arr, source_img.affine)
    zooms = source_img.header.get_zooms()[:3]
    if len(zooms) == 3:
        out.header.set_zooms(zooms)
    nib.save(out, str(out_path))


def extract_mms_ed_es_derivatives(manifest: pd.DataFrame, derived_root: str | Path) -> list[Path]:
    root = resolve_path(derived_root)
    image_dir = root / "images"
    mask_dir = root / "masks"
    meta_dir = root / "metadata"
    outputs: list[Path] = []
    for row in manifest.itertuples(index=False):
        case_id = str(row.case_id)
        ed_frame = int(row.ed_frame)
        es_frame = int(row.es_frame)
        image_img = nib.load(str(row.image_path))
        mask_img = nib.load(str(row.mask_path))
        image_data = np.asanyarray(image_img.dataobj)
        mask_data = np.asanyarray(mask_img.dataobj)
        for phase, frame in [("ED", ed_frame), ("ES", es_frame)]:
            phase_image = select_frame(image_data, frame)
            phase_mask_raw = select_frame(mask_data, frame)
            phase_mask, unexpected = normalize_labels(np.rint(phase_mask_raw).astype(np.int16, copy=False), "MMS_100")
            if unexpected:
                raise ValueError(f"{case_id} {phase} has unexpected labels: {sorted(unexpected)}")
            image_out = image_dir / f"{case_id}_{phase}.nii.gz"
            mask_out = mask_dir / f"{case_id}_{phase}_mask.nii.gz"
            _save_3d_nifti(phase_image, image_img, image_out)
            _save_3d_nifti(phase_mask, mask_img, mask_out, dtype=np.uint8)
            outputs.extend([image_out, mask_out])
        geom = geometry_from_image(image_img)
        meta_out = meta_dir / f"{case_id}_geometry.json"
        write_json(
            {
                "case_id": case_id,
                "ed_frame": ed_frame,
                "es_frame": es_frame,
                "source_image_path": str(row.image_path),
                "source_mask_path": str(row.mask_path),
                "geometry": serialize_geometry(geom),
                "label_convention": {"background": 0, "lv_blood_pool": 1, "lv_myocardium": 2, "rv_blood_pool": 3},
            },
            meta_out,
        )
        outputs.append(meta_out)
    return outputs


def _aha_null_columns() -> dict[str, object]:
    out: dict[str, object] = {
        "aha_mapping_feasible_flag": False,
        "aha_mapping_invalid_reason": "orientation_metadata_not_validated_short_axis_only",
    }
    for idx in range(1, 17):
        out[f"aha16_seg{idx:02d}_wall_thickness_ed_mm"] = None
        out[f"aha16_seg{idx:02d}_wall_thickness_es_mm"] = None
        out[f"aha16_seg{idx:02d}_wall_thickening_percent"] = None
    return out


def _base_feature_row(source: str, row: pd.Series, split: str | None) -> dict[str, object]:
    case_id = str(row["case_id"])
    patient_id = str(row.get("patient_id", case_id))
    study_id = _none_if_na(row.get("study_id"))
    sample_id = f"{source}_{patient_id}_{study_id or 'na'}"
    height = _float_or_none(row.get("height_cm"))
    weight = _float_or_none(row.get("weight_kg"))
    bmi, bsa = bmi_and_bsa(height, weight)
    return {
        "schema_version": "B_v2.0",
        "sample_id": sample_id,
        "case_id": case_id,
        "patient_id": patient_id,
        "study_id": study_id,
        "dataset_source": source,
        "subset_name": split,
        "split": split,
        "disease_label": _none_if_na(row.get("disease_label")),
        "ed_frame_index": _int_or_none(row.get("ed_frame")),
        "es_frame_index": _int_or_none(row.get("es_frame")),
        "ed_es_source": "metadata",
        "heart_rate_bpm": None,
        "temporal_resolution_ms": None,
        "scanner_vendor": _none_if_na(row.get("vendor_name")) or _none_if_na(row.get("vendor")),
        "scanner_model": None,
        "center_id": _none_if_na(row.get("center")),
        "field_strength_t": None,
        "height_cm": height,
        "weight_kg": weight,
        "bmi_kg_m2": bmi,
        "bsa_m2": bsa,
        "bsa_formula": "Mosteller" if bsa is not None else None,
        "bsa_available_flag": bsa is not None,
        "bsa_mosteller_bmi_gt_40_caution_flag": bool(bmi is not None and bmi > 40),
        "papillary_inclusion_status": "unknown",
    }


def compute_b_row(
    source: str,
    row: pd.Series,
    split: str | None,
    mask_ed: np.ndarray,
    mask_es: np.ndarray,
    geometry: Any,
    unexpected_labels: set[int],
    raw_image_path: str | None,
    raw_mask_path: str | None,
) -> dict[str, object]:
    dx = float(geometry.spacing_x_mm)
    dy = float(geometry.spacing_y_mm)
    dz = geometry.slice_increment_mm
    near_zero_edv = 1.0
    row_out = _base_feature_row(source, row, split)
    row_out.update(
        {
            "spacing_x_mm": dx,
            "spacing_y_mm": dy,
            "slice_thickness_mm": geometry.slice_thickness_mm,
            "slice_gap_mm": geometry.slice_gap_mm,
            "slice_increment_mm": dz,
            "slice_increment_variability_mm": geometry.slice_increment_variability_mm,
            "z_spacing_method": geometry.z_spacing_method,
            "z_spacing_estimated_flag": geometry.z_spacing_method in {"slice_thickness_plus_gap_fallback", "slice_thickness_only_fallback"},
            "affine_matrix": geometry.affine_json(),
            "raw_image_path": raw_image_path,
            "raw_mask_path": raw_mask_path,
            "raw_ed_image_path": _none_if_na(row.get("ed_image_path")) or raw_image_path,
            "raw_es_image_path": _none_if_na(row.get("es_image_path")) or raw_image_path,
            "raw_ed_mask_path": _none_if_na(row.get("ed_mask_path")) or raw_mask_path,
            "raw_es_mask_path": _none_if_na(row.get("es_mask_path")) or raw_mask_path,
        }
    )

    project_label_unexpected = validate_project_labels(mask_ed).union(validate_project_labels(mask_es))
    all_unexpected = set(unexpected_labels).union(project_label_unexpected)
    labels_valid = len(all_unexpected) == 0

    lv_edv = mask_volume_ml(mask_ed, LV_BLOOD_POOL, dx, dy, dz)
    lv_esv = mask_volume_ml(mask_es, LV_BLOOD_POOL, dx, dy, dz)
    rv_edv = mask_volume_ml(mask_ed, RV_BLOOD_POOL, dx, dy, dz)
    rv_esv = mask_volume_ml(mask_es, RV_BLOOD_POOL, dx, dy, dz)
    lv_sv = safe_sv(lv_edv, lv_esv)
    rv_sv = safe_sv(rv_edv, rv_esv)
    lv_ef = safe_ef(lv_edv, lv_esv, near_zero_edv)
    rv_ef = safe_ef(rv_edv, rv_esv, near_zero_edv)
    myo_ed = mask_volume_ml(mask_ed, LV_MYOCARDIUM, dx, dy, dz)
    myo_es = mask_volume_ml(mask_es, LV_MYOCARDIUM, dx, dy, dz)
    density = 1.050
    mass_ed = float(myo_ed * density) if myo_ed is not None else None
    bsa = row_out["bsa_m2"]
    sv_discrepancy = abs(lv_sv - rv_sv) if lv_sv is not None and rv_sv is not None else None

    lv_shape_ed = shape_summary(mask_ed, LV_BLOOD_POOL, dx, dy)
    lv_shape_es = shape_summary(mask_es, LV_BLOOD_POOL, dx, dy)
    rv_shape_ed = shape_summary(mask_ed, RV_BLOOD_POOL, dx, dy)
    rv_shape_es = shape_summary(mask_es, RV_BLOOD_POOL, dx, dy)
    wall_ed, wall_invalid_ed, wall_reliable_ed = global_wall_thickness_mm(mask_ed, dx, dy)
    wall_es, wall_invalid_es, wall_reliable_es = global_wall_thickness_mm(mask_es, dx, dy)
    wall_thickening = wall_thickening_percent(wall_ed, wall_es)
    long_axis_approx = short_axis_long_axis_approx_mm(mask_ed, dz)
    sphericity = lv_sphericity_index(lv_edv, long_axis_approx)

    qc_failure_reasons: list[str] = []
    present_flags = {
        "qc_lv_ed_present": bool(np.any(mask_ed == LV_BLOOD_POOL)),
        "qc_lv_es_present": bool(np.any(mask_es == LV_BLOOD_POOL)),
        "qc_rv_ed_present": bool(np.any(mask_ed == RV_BLOOD_POOL)),
        "qc_rv_es_present": bool(np.any(mask_es == RV_BLOOD_POOL)),
        "qc_myo_ed_present": bool(np.any(mask_ed == LV_MYOCARDIUM)),
        "qc_myo_es_present": bool(np.any(mask_es == LV_MYOCARDIUM)),
    }
    for key, value in present_flags.items():
        if not value:
            qc_failure_reasons.append(key.replace("qc_", "missing_"))
    if not labels_valid:
        qc_failure_reasons.append(f"unexpected_label_value:{sorted(all_unexpected)}")
    if dz is None or dz <= 0:
        qc_failure_reasons.append("z_spacing_unavailable_for_volume")
    if lv_edv is not None and lv_esv is not None and lv_edv <= lv_esv:
        qc_failure_reasons.append("lv_edv_le_lvesv")
    if rv_edv is not None and rv_esv is not None and rv_edv <= rv_esv:
        qc_failure_reasons.append("rv_edv_le_rvesv")
    if lv_ef is not None and not (0 <= lv_ef <= 100):
        qc_failure_reasons.append("lv_ef_outside_0_100")
    if rv_ef is not None and not (0 <= rv_ef <= 100):
        qc_failure_reasons.append("rv_ef_outside_0_100")
    if mass_ed is not None and mass_ed <= 0:
        qc_failure_reasons.append("myocardial_mass_nonpositive")
    if sv_discrepancy is not None and sv_discrepancy > 15:
        qc_failure_reasons.append("lv_rv_sv_discrepancy_gt_15_ml")
    if np.array_equal(mask_ed, mask_es):
        qc_failure_reasons.append("ed_es_masks_identical")
    if not wall_reliable_ed:
        qc_failure_reasons.append("wall_thickness_invalid_global")

    row_out.update(
        {
            "lv_edv_ml": lv_edv,
            "lv_esv_ml": lv_esv,
            "lv_sv_ml": lv_sv,
            "lv_ef_percent": lv_ef,
            "rv_edv_ml": rv_edv,
            "rv_esv_ml": rv_esv,
            "rv_sv_ml": rv_sv,
            "rv_ef_percent": rv_ef,
            "lv_edv_indexed_ml_m2": indexed(lv_edv, bsa),
            "lv_esv_indexed_ml_m2": indexed(lv_esv, bsa),
            "rv_edv_indexed_ml_m2": indexed(rv_edv, bsa),
            "rv_esv_indexed_ml_m2": indexed(rv_esv, bsa),
            "lv_myo_volume_ed_ml": myo_ed,
            "lv_myo_volume_es_ml": myo_es,
            "lv_myocardial_mass_ed_g": mass_ed,
            "lv_myocardial_mass_indexed_ed_g_m2": indexed(mass_ed, bsa),
            "myocardial_density_g_per_ml": density,
            "lv_rv_sv_discrepancy_ml": sv_discrepancy,
            "lv_rv_sv_discrepancy_review_flag": bool(sv_discrepancy is not None and sv_discrepancy > 15),
            "global_wall_thickness_ed_mm": wall_ed,
            "global_wall_thickness_es_mm": wall_es,
            "global_wall_thickening_percent": wall_thickening,
            "wall_thickness_invalid_slice_count_ed": int(wall_invalid_ed),
            "wall_thickness_invalid_slice_count_es": int(wall_invalid_es),
            "wall_thickness_reliable_flag": bool(wall_reliable_ed and wall_reliable_es),
            "wall_thickness_method": "radial_ray_lv_center_area_weighted",
            "lv_area_mean_ed_mm2": lv_shape_ed["area_mean_mm2"],
            "lv_area_mean_es_mm2": lv_shape_es["area_mean_mm2"],
            "rv_area_mean_ed_mm2": rv_shape_ed["area_mean_mm2"],
            "rv_area_mean_es_mm2": rv_shape_es["area_mean_mm2"],
            "lv_compactness_mean_ed": lv_shape_ed["compactness_mean"],
            "lv_compactness_mean_es": lv_shape_es["compactness_mean"],
            "rv_compactness_mean_ed": rv_shape_ed["compactness_mean"],
            "rv_compactness_mean_es": rv_shape_es["compactness_mean"],
            "lv_eccentricity_mean_ed": lv_shape_ed["eccentricity_mean"],
            "lv_eccentricity_mean_es": lv_shape_es["eccentricity_mean"],
            "rv_eccentricity_mean_ed": rv_shape_ed["eccentricity_mean"],
            "rv_eccentricity_mean_es": rv_shape_es["eccentricity_mean"],
            "myocardium_eccentricity_valid_flag": False,
            "myocardium_eccentricity_invalid_reason": "annular_lv_myocardium_not_valid_for_filled_blood_pool_eccentricity",
            "perimeter_algorithm": "marching_squares_subpixel",
            "perimeter_spacing_applied_flag": True,
            "perimeter_valid_flag": bool(
                lv_shape_ed["perimeter_valid_flag"]
                or lv_shape_es["perimeter_valid_flag"]
                or rv_shape_ed["perimeter_valid_flag"]
                or rv_shape_es["perimeter_valid_flag"]
            ),
            "lv_long_axis_length_mm": long_axis_approx,
            "lv_long_axis_source": "short_axis_stack_approximation",
            "lv_sphericity_index_3d": sphericity,
            "lv_sphericity_valid_flag": False,
            "lv_sphericity_invalid_reason": "short_axis_stack_approximation_not_validated_for_clinical_interpretation",
            "qc_spacing_valid": bool(dx > 0 and dy > 0),
            "qc_z_spacing_valid": bool(dz is not None and dz > 0),
            "qc_mask_labels_valid": labels_valid,
            "qc_overlap_resolution_valid": True,
            "qc_temporal_metadata_review_flag": False,
            "qc_ed_es_masks_identical_flag": bool(np.array_equal(mask_ed, mask_es)),
            "qc_extreme_robust_outlier_flag": False,
            "qc_extreme_outlier_features": None,
            "qc_failure_reason": ";".join(qc_failure_reasons) if qc_failure_reasons else None,
        }
    )
    row_out.update(present_flags)
    row_out["qc_lv_ef_valid"] = bool(lv_ef is not None and 0 <= lv_ef <= 100)
    row_out["qc_rv_ef_valid"] = bool(rv_ef is not None and 0 <= rv_ef <= 100)
    row_out["qc_manual_review_required"] = bool(qc_failure_reasons)
    row_out.update(_aha_null_columns())
    return row_out


def build_acdc_b_matrix(acdc_inventory: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    split_map = dict(zip(split_df["patient_id"].astype(str), split_df["split"].astype(str)))
    rows: list[dict[str, object]] = []
    for _, inv_row in acdc_inventory[acdc_inventory["eligible_for_b"]].sort_values("patient_id").iterrows():
        ed_mask, unexpected_ed, ed_img = _load_normalized_mask(inv_row["ed_mask_path"], "ACDC")
        es_mask, unexpected_es, _ = _load_normalized_mask(inv_row["es_mask_path"], "ACDC")
        geom = geometry_from_image(ed_img)
        split = split_map.get(str(inv_row["patient_id"]))
        rows.append(
            compute_b_row(
                "ACDC",
                inv_row,
                split,
                ed_mask,
                es_mask,
                geom,
                unexpected_ed.union(unexpected_es),
                raw_image_path=inv_row["ed_image_path"],
                raw_mask_path=inv_row["ed_mask_path"],
            )
        )
    return pd.DataFrame(rows)


def build_mms_b_matrix(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, inv_row in manifest.sort_values("selected_rank").iterrows():
        ed = int(inv_row["ed_frame"])
        es = int(inv_row["es_frame"])
        mask_ed, unexpected_ed, mask_img = _load_normalized_mask(inv_row["mask_path"], "MMS_100", ed)
        mask_es, unexpected_es, _ = _load_normalized_mask(inv_row["mask_path"], "MMS_100", es)
        geom = geometry_from_image(mask_img)
        rows.append(
            compute_b_row(
                "MMS_100",
                inv_row,
                "mms_100_external",
                mask_ed,
                mask_es,
                geom,
                unexpected_ed.union(unexpected_es),
                raw_image_path=inv_row["image_path"],
                raw_mask_path=inv_row["mask_path"],
            )
        )
    return pd.DataFrame(rows)


def add_robust_outlier_flags(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["qc_extreme_robust_outlier_flag"] = False
    out["qc_extreme_outlier_features"] = None
    flagged: dict[int, list[str]] = {idx: [] for idx in out.index}
    for col in feature_cols:
        values = pd.to_numeric(out[col], errors="coerce")
        valid = values.dropna()
        if len(valid) < 5:
            continue
        med = float(valid.median())
        mad = float((valid - med).abs().median())
        if mad <= 0 or not np.isfinite(mad):
            continue
        robust_z = 0.6745 * (values - med) / mad
        for idx in out.index[robust_z.abs() > 3.5]:
            flagged[idx].append(col)
    for idx, cols in flagged.items():
        if cols:
            out.at[idx, "qc_extreme_robust_outlier_flag"] = True
            out.at[idx, "qc_extreme_outlier_features"] = ",".join(cols)
            reason = out.at[idx, "qc_failure_reason"]
            addition = "extreme_robust_outlier"
            out.at[idx, "qc_failure_reason"] = f"{reason};{addition}" if isinstance(reason, str) and reason else addition
            out.at[idx, "qc_manual_review_required"] = True
    return out


TRANSITION_ELIGIBLE_PREFIXES = (
    "lv_",
    "rv_",
    "global_wall_",
)


def default_transition_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded_substrings = (
        "flag",
        "reason",
        "path",
        "index",
        "source",
        "schema_version",
        "sample_id",
        "case_id",
        "patient_id",
        "study_id",
        "dataset",
        "split",
        "label",
        "affine",
    )
    cols: list[str] = []
    for col in df.columns:
        if any(s in col for s in excluded_substrings):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if col.startswith(TRANSITION_ELIGIBLE_PREFIXES) or col in {
            "bmi_kg_m2",
            "bsa_m2",
            "lv_myo_volume_ed_ml",
            "lv_myo_volume_es_ml",
            "myocardial_density_g_per_ml",
        }:
            cols.append(col)
    return cols


def feature_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    transition_cols = set(default_transition_feature_columns(df))
    units = {
        "height_cm": "cm",
        "weight_kg": "kg",
        "bmi_kg_m2": "kg/m2",
        "bsa_m2": "m2",
        "spacing_x_mm": "mm",
        "spacing_y_mm": "mm",
        "slice_thickness_mm": "mm",
        "slice_gap_mm": "mm",
        "slice_increment_mm": "mm",
        "lv_edv_ml": "mL",
        "lv_esv_ml": "mL",
        "lv_sv_ml": "mL",
        "lv_ef_percent": "%",
        "rv_edv_ml": "mL",
        "rv_esv_ml": "mL",
        "rv_sv_ml": "mL",
        "rv_ef_percent": "%",
        "lv_myo_volume_ed_ml": "mL",
        "lv_myo_volume_es_ml": "mL",
        "lv_myocardial_mass_ed_g": "g",
        "myocardial_density_g_per_ml": "g/mL",
        "global_wall_thickness_ed_mm": "mm",
        "global_wall_thickness_es_mm": "mm",
        "global_wall_thickening_percent": "%",
        "lv_long_axis_length_mm": "mm",
    }
    formulas = {
        "bsa_m2": "sqrt(height_cm * weight_kg / 3600)",
        "lv_sv_ml": "lv_edv_ml - lv_esv_ml",
        "rv_sv_ml": "rv_edv_ml - rv_esv_ml",
        "lv_ef_percent": "100 * lv_sv_ml / lv_edv_ml",
        "rv_ef_percent": "100 * rv_sv_ml / rv_edv_ml",
        "lv_myocardial_mass_ed_g": "lv_myo_volume_ed_ml * 1.050",
        "lv_rv_sv_discrepancy_ml": "abs(lv_sv_ml - rv_sv_ml)",
        "global_wall_thickening_percent": "100 * (global_wall_thickness_es_mm - global_wall_thickness_ed_mm) / global_wall_thickness_ed_mm",
    }
    rows = []
    for col in df.columns:
        rows.append(
            {
                "column_name": col,
                "dtype": str(df[col].dtype),
                "unit": units.get(col),
                "nullable": bool(df[col].isna().any()),
                "valid_range": None,
                "derivation_formula": formulas.get(col),
                "transition_target_eligible": col in transition_cols,
            }
        )
    return pd.DataFrame(rows)


# ===============================================================================================================
# Protocol v6 semantics (SPECIFICATION.md 6.0.0, Sections 6-7 and 12.4; Decisions D6, D7, D21). New functions only:
# compute_b_row and the legacy builders above are unchanged and are reused for the measurement arithmetic.
# ===============================================================================================================
from dataclasses import dataclass as _dataclass

from .qc import assess_qc_v6, parse_reason_tokens, reason_prefix, repo_config

V6_STATES = ("OBSERVED", "NOT_APPLICABLE", "UNSUPPORTED", "QC_BLOCKED", "MISSING_SOURCE", "CONFLICTING")
_Z_FALLBACK_METHODS = {"nifti_header_pixdim_unscaled_affine", "slice_thickness_only_fallback"}


@_dataclass(frozen=True)
class StagedGeometry:
    """Geometry adapter with the attributes compute_b_row reads, built from a staged case."""

    spacing_x_mm: float
    spacing_y_mm: float
    slice_thickness_mm: float | None
    slice_gap_mm: float | None
    slice_increment_mm: float | None
    slice_increment_variability_mm: float | None
    z_spacing_method: str
    affine_matrix: list

    def affine_json(self) -> str:
        return json.dumps(self.affine_matrix, separators=(",", ":"))


def _finite_positive(value: object) -> float | None:
    v = _float_or_none(value)
    return v if v is not None and v > 0 else None


def typed_columns_v6(registry: dict[str, Any]) -> list[str]:
    """Column order of the typed table B_v6.0: metadata, QC, then each feature with __state and __evidence."""
    suffixes = registry.get("typed_companions", ["__state", "__evidence"])
    cols = list(registry["metadata_columns"]) + list(registry["qc_columns"])
    for entry in registry["features"]:
        cols.append(str(entry["name"]))
        cols.extend(f"{entry['name']}{s}" for s in suffixes)
    return cols


# Registry evidence profile of a cohort without its own ``evidence_sources`` key: M&Ms-2 (protocol-v6.0.1) releases the
# same evidence tiers as M&Ms-1 (expert masks, header geometry, release CSV); configs/cohorts.yaml mms2.evidence_profile.
EVIDENCE_PROFILE_ALIASES = {"mms2": "mms"}


def _cohort_key(dataset: str) -> str:
    key = str(dataset).strip().lower()
    return EVIDENCE_PROFILE_ALIASES.get(key, key)


def _mask_int(mask: np.ndarray | None) -> tuple[np.ndarray | None, set]:
    if mask is None:
        return None, set()
    arr = np.asarray(mask)
    if not np.all(np.isfinite(arr)):
        return np.where(np.isfinite(arr), np.rint(arr), 0).astype(np.int16), {"nonfinite"}
    return np.rint(arr).astype(np.int16), set()


def _outside_registry_domain(value: float | None, domain: object) -> bool:
    """True when ``value`` lies outside the registry domain ``[low, high]`` (null bound = unbounded)."""

    if value is None or not isinstance(domain, (list, tuple)) or len(domain) != 2:
        return False
    low, high = domain
    return (low is not None and value < float(low)) or (high is not None and value > float(high))


def case_semantics_v6(case: Any, registry: dict[str, Any], *, cohorts: dict[str, Any] | None = None, reason_codes: dict[str, Any] | None = None) -> dict[str, Any]:
    """Every registry metadata, QC and feature column (with ``__state``/``__evidence``) for one staged case.

    ``case`` provides ``sample_id``, ``dataset`` (ACDC or MMS), ``case_id``, ``ed_mask``/``es_mask`` (canonical labels
    LV = 1, MYO = 2, RV = 3), ``spacing_mm`` (dx, dy, dz), ``geometry_method``, ``affine`` and ``meta`` (release
    metadata: patient_id, subset_name, split, disease_label, 0-based ed/es_frame_index, scanner_vendor, center_id,
    height_cm, weight_kg, slice_thickness_mm, raw_*_path relative to the source mount, optional upstream qc_reasons and
    qc_temporal_metadata_review_flag). Measurements reuse :func:`compute_b_row`. QC severities are the v6 severities of
    configs/cohorts.yaml; robust-outlier flags are applied later against the acdc_dev reference
    (:func:`tmmri.semantics.qc.apply_outlier_reference`).
    """
    cohorts = cohorts if cohorts is not None else repo_config("configs/cohorts.yaml")
    reason_codes = reason_codes if reason_codes is not None else repo_config("registry/reason_codes.yaml")
    meta = dict(getattr(case, "meta", None) or {})
    dataset = str(case.dataset)
    cohort = _cohort_key(dataset)
    dx, dy, dz = (tuple(case.spacing_mm) + (None, None, None))[:3]
    dx_ok, dy_ok = _finite_positive(dx), _finite_positive(dy)
    dz_ok = _finite_positive(dz)
    inplane_valid = dx_ok is not None and dy_ok is not None
    geometry = StagedGeometry(
        spacing_x_mm=float(dx_ok if inplane_valid else 1.0),
        spacing_y_mm=float(dy_ok if inplane_valid else 1.0),
        slice_thickness_mm=_finite_positive(meta.get("slice_thickness_mm")),
        slice_gap_mm=None,
        slice_increment_mm=dz_ok,
        slice_increment_variability_mm=_float_or_none(meta.get("slice_increment_variability_mm")) if dz_ok is not None else None,
        z_spacing_method=str(getattr(case, "geometry_method", None) or meta.get("z_spacing_method") or ("unavailable" if dz_ok is None else "nifti_affine_projection")),
        affine_matrix=np.asarray(getattr(case, "affine", np.eye(4)), dtype=float).tolist(),
    )
    upstream = [str(t) for t in (meta.get("qc_reasons") or [])]
    mask_ed, bad_ed = _mask_int(getattr(case, "ed_mask", None))
    mask_es, bad_es = _mask_int(getattr(case, "es_mask", None))
    if mask_ed is None or mask_es is None:
        shape = (mask_es if mask_ed is None else mask_ed)
        shape = shape.shape if shape is not None else (1, 1, 1)
        mask_ed = np.zeros(shape, dtype=np.int16) if mask_ed is None else mask_ed
        mask_es = np.zeros(shape, dtype=np.int16) if mask_es is None else mask_es
        upstream.append("missing_ed_or_es_mask")
    row = pd.Series(
        {
            "case_id": str(case.case_id),
            "patient_id": meta.get("patient_id", str(case.case_id)),
            "study_id": meta.get("study_id"),
            "disease_label": meta.get("disease_label"),
            "ed_frame": meta.get("ed_frame_index"),
            "es_frame": meta.get("es_frame_index"),
            "vendor_name": meta.get("scanner_vendor", meta.get("vendor_name")),
            "center": meta.get("center_id", meta.get("centre")),
            "height_cm": meta.get("height_cm"),
            "weight_kg": meta.get("weight_kg"),
            "ed_image_path": meta.get("raw_ed_image_path"),
            "es_image_path": meta.get("raw_es_image_path"),
            "ed_mask_path": meta.get("raw_ed_mask_path"),
            "es_mask_path": meta.get("raw_es_mask_path"),
        },
        dtype=object,
    )
    unexpected = {int(v) for v in np.unique(mask_ed)} | {int(v) for v in np.unique(mask_es)}
    b = compute_b_row(dataset, row, meta.get("split"), mask_ed, mask_es, geometry, set(), meta.get("raw_image_path"), meta.get("raw_mask_path"))
    # ---- v6 metadata corrections ------------------------------------------------------------------------------
    b["schema_version"] = str(registry.get("typed_schema_version", "B_v6.0"))
    b["sample_id"] = str(getattr(case, "sample_id", None) or f"{dataset}:{case.case_id}")
    b["dataset_source"] = dataset
    b["subset_name"] = meta.get("subset_name", meta.get("release_split"))
    b["split"] = meta.get("split")
    b["ed_frame_index"] = _int_or_none(meta.get("ed_frame_index"))
    b["es_frame_index"] = _int_or_none(meta.get("es_frame_index"))
    b["center_id"] = _int_or_none(meta.get("center_id", meta.get("centre")))
    b["scanner_vendor"] = _none_if_na(meta.get("scanner_vendor", meta.get("vendor_name")))
    for key in ("raw_image_path", "raw_mask_path", "raw_ed_image_path", "raw_es_image_path", "raw_ed_mask_path", "raw_es_mask_path"):
        b[key] = _none_if_na(meta.get(key)) if key in meta else b.get(key)
    b["spacing_x_mm"] = _float_or_none(dx)
    b["spacing_y_mm"] = _float_or_none(dy)
    b["qc_spacing_valid"] = bool(inplane_valid)
    b["qc_temporal_metadata_review_flag"] = bool(meta.get("qc_temporal_metadata_review_flag", False))
    # ---- v6 QC tokens ----------------------------------------------------------------------------------------
    tokens: list[str] = []
    for tok in parse_reason_tokens(b.get("qc_failure_reason")):
        if reason_prefix(tok) == "unexpected_label_value":
            continue
        tokens.append(tok)
    bad_labels = sorted(v for v in unexpected if v not in (0, 1, 2, 3))
    if bad_labels or bad_ed or bad_es:
        payload = "|".join([str(v) for v in bad_labels] + sorted(bad_ed | bad_es))
        tokens.append(f"unexpected_label_value:[{payload}]")
        b["qc_mask_labels_valid"] = False
    if not inplane_valid:
        tokens.append("spacing_nonpositive")
    if "wall_thickness_invalid_global" not in tokens and not bool(b.get("wall_thickness_reliable_flag", True)):
        tokens.append("wall_thickness_invalid_es")
    if b.get("bsa_m2") is None:
        tokens.append("bsa_unavailable")
    if bool(b.get("bsa_mosteller_bmi_gt_40_caution_flag")):
        tokens.append("bmi_gt_40_mosteller_caution")
    if geometry.z_spacing_method in _Z_FALLBACK_METHODS and dz_ok is not None:
        tokens.append("z_spacing_header_pixdim_fallback")
    if b["qc_temporal_metadata_review_flag"]:
        tokens.append("qc_temporal_metadata_review_flag")
    for tok in upstream:
        if tok not in tokens:
            tokens.append(tok)
    assessment = assess_qc_v6(tokens, cohorts, reason_codes)
    b["qc_failure_reason"] = ";".join(tokens) if tokens else None
    b["qc_manual_review_required"] = bool(assessment.manual_review)
    b["qc_extreme_robust_outlier_flag"] = False
    b["qc_extreme_outlier_features"] = None
    # ---- typed feature states ----------------------------------------------------------------------------------
    masked = set(assessment.masked_features)
    critical_row = bool(assessment.critical)
    bsa_ok = b.get("bsa_m2") is not None
    out: dict[str, Any] = {c: b.get(c) for c in list(registry["metadata_columns"]) + list(registry["qc_columns"])}
    for entry in registry["features"]:
        name = str(entry["name"])
        value = _float_or_none(b.get(name))
        sources = (entry.get("evidence_sources") or {}).get(cohort)
        deps = [str(d) for d in (entry.get("dependencies") or [])]
        if sources == "UNSUPPORTED" or sources is None or entry.get("family") == "reserved_unsupported":
            state, value = "UNSUPPORTED", None
        elif value is None:
            body_size = any(d in ("height_cm", "weight_kg", "bsa_m2") for d in deps)
            if body_size and not bsa_ok:
                state = "MISSING_SOURCE"
            elif "header_geometry" in deps and (not inplane_valid or (dz_ok is None and entry.get("scale_dependence") in ("physical_volume", "physical_length"))):
                state = "MISSING_SOURCE"
            else:
                state = "NOT_APPLICABLE"
        elif name in masked or (critical_row and _outside_registry_domain(value, entry.get("domain"))):
            # High-severity feature masks, and out-of-domain values on critical-QC rows (for example an EF outside
            # 0-100 % that is itself the critical reason): the value is never observed (the contract is strict).
            state, value = "QC_BLOCKED", None
        else:
            state = "OBSERVED"
        out[name] = value
        out[f"{name}__state"] = state
        out[f"{name}__evidence"] = "+".join(str(s) for s in sources) if state == "OBSERVED" and isinstance(sources, list) else None
    return out


def _semantics_task(item: Any, registry: dict[str, Any], cohorts: dict[str, Any], reason_codes: dict[str, Any], loader: Any) -> dict[str, Any]:
    case = loader(item) if loader is not None else item
    return case_semantics_v6(case, registry, cohorts=cohorts, reason_codes=reason_codes)


def build_semantics_table_v6(
    cases_iterable_or_loader: Any,
    registry: dict[str, Any],
    n_jobs: int = 1,
    *,
    cohorts: dict[str, Any] | None = None,
    reason_codes: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Typed B_v6.0 table (one row per case, sorted by sample_id) computed per case in parallel.

    ``cases_iterable_or_loader`` is an iterable of staged cases or a ``(loader, keys)`` pair; with a loader each worker
    loads its own case (``loader(key)``), so images are never pickled. Results do not depend on ``n_jobs``: rows are
    gathered and then sorted by sample_id. Robust-outlier flags are applied separately against the fit-partition
    reference.
    """
    cohorts = cohorts if cohorts is not None else repo_config("configs/cohorts.yaml")
    reason_codes = reason_codes if reason_codes is not None else repo_config("registry/reason_codes.yaml")
    loader = None
    items: Any = cases_iterable_or_loader
    if isinstance(items, tuple) and len(items) == 2 and callable(items[0]):
        loader, items = items
    items = list(items)
    if int(n_jobs) == 1:
        rows = [_semantics_task(it, registry, cohorts, reason_codes, loader) for it in items]
    else:
        from joblib import Parallel, delayed, parallel_config

        with parallel_config(backend="loky", inner_max_num_threads=1):
            rows = Parallel(n_jobs=int(n_jobs))(delayed(_semantics_task)(it, registry, cohorts, reason_codes, loader) for it in items)
    table = pd.DataFrame(rows, columns=typed_columns_v6(registry))
    if table["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample_id values: {table.loc[table['sample_id'].duplicated(), 'sample_id'].tolist()[:5]}")
    for col in table.columns:
        if col.endswith("__state") or col.endswith("__evidence"):
            table[col] = table[col].astype("string")
    return table.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
