from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .geometry import load_nifti_header
from ..io import file_manifest_sha256, git_commit, resolve_path, utc_now_iso


_MMS_INVENTORY_COLUMNS = [
    "case_id",
    "patient_id",
    "vendor",
    "vendor_name",
    "center",
    "disease_label",
    "split",
    "has_image",
    "has_mask",
    "has_ed_phase",
    "has_es_phase",
    "ed_frame",
    "es_frame",
    "image_path",
    "mask_path",
    "image_shape",
    "mask_shape",
    "spacing_x_mm",
    "spacing_y_mm",
    "slice_thickness_mm",
    "slice_gap_mm",
    "slice_increment_mm",
    "slice_increment_variability_mm",
    "z_spacing_method",
    "affine_matrix",
    "num_slices",
    "num_frames",
    "height_cm",
    "weight_kg",
    "age_years",
    "sex",
    "eligible_for_mms_100",
    "exclusion_reason",
]

_MMS_BOOLEAN_COLUMNS = [
    "has_image",
    "has_mask",
    "has_ed_phase",
    "has_es_phase",
    "eligible_for_mms_100",
]


def _parse_cfg(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _as_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        out = int(float(value))
    except (TypeError, ValueError):
        return None
    return out


def build_acdc_inventory(acdc_root: str | Path) -> pd.DataFrame:
    root = resolve_path(acdc_root)
    rows: list[dict[str, Any]] = []
    for split in ["training", "testing"]:
        for patient_dir in sorted((root / split).glob("patient*")):
            if not patient_dir.is_dir():
                continue
            patient_id = patient_dir.name
            cfg = _parse_cfg(patient_dir / "Info.cfg")
            ed_frame = _as_int(cfg.get("ED"))
            es_frame = _as_int(cfg.get("ES"))
            ed_name = f"{patient_id}_frame{ed_frame:02d}" if ed_frame is not None else None
            es_name = f"{patient_id}_frame{es_frame:02d}" if es_frame is not None else None
            ed_image = patient_dir / f"{ed_name}.nii.gz" if ed_name else None
            es_image = patient_dir / f"{es_name}.nii.gz" if es_name else None
            ed_mask = patient_dir / f"{ed_name}_gt.nii.gz" if ed_name else None
            es_mask = patient_dir / f"{es_name}_gt.nii.gz" if es_name else None
            geom = load_nifti_header(ed_image) if ed_image and ed_image.exists() else None
            mask_geom = load_nifti_header(ed_mask) if ed_mask and ed_mask.exists() else None
            has_all = all(p is not None and p.exists() for p in [ed_image, es_image, ed_mask, es_mask])
            spacing_valid = bool(geom and geom.spacing_x_mm > 0 and geom.spacing_y_mm > 0)
            z_valid = bool(geom and geom.slice_increment_mm and geom.slice_increment_mm > 0)
            exclusion = []
            if not has_all:
                exclusion.append("missing_ed_es_image_or_mask")
            if ed_frame is None or es_frame is None:
                exclusion.append("missing_ed_or_es_metadata")
            if not spacing_valid:
                exclusion.append("invalid_inplane_spacing")
            if not z_valid:
                exclusion.append("invalid_z_spacing")
            rows.append(
                {
                    "case_id": patient_id,
                    "patient_id": patient_id,
                    "study_id": None,
                    "dataset_source": "ACDC",
                    "source_split": split,
                    "disease_label": cfg.get("Group"),
                    "height_cm": _as_float(cfg.get("Height")),
                    "weight_kg": _as_float(cfg.get("Weight")),
                    "ed_frame": ed_frame,
                    "es_frame": es_frame,
                    "ed_frame_index": ed_frame,
                    "es_frame_index": es_frame,
                    "ed_image_path": str(ed_image) if ed_image else None,
                    "es_image_path": str(es_image) if es_image else None,
                    "ed_mask_path": str(ed_mask) if ed_mask else None,
                    "es_mask_path": str(es_mask) if es_mask else None,
                    "image_shape": geom.as_json_shape() if geom else None,
                    "mask_shape": mask_geom.as_json_shape() if mask_geom else None,
                    "spacing_x_mm": geom.spacing_x_mm if geom else None,
                    "spacing_y_mm": geom.spacing_y_mm if geom else None,
                    "slice_thickness_mm": geom.slice_thickness_mm if geom else None,
                    "slice_gap_mm": geom.slice_gap_mm if geom else None,
                    "slice_increment_mm": geom.slice_increment_mm if geom else None,
                    "slice_increment_variability_mm": geom.slice_increment_variability_mm if geom else None,
                    "z_spacing_method": geom.z_spacing_method if geom else "unavailable",
                    "affine_matrix": geom.affine_json() if geom else None,
                    "num_slices": geom.num_slices if geom else None,
                    "num_frames": geom.num_frames if geom else None,
                    "eligible_for_b": len(exclusion) == 0,
                    "exclusion_reason": ";".join(exclusion) if exclusion else None,
                }
            )
    return pd.DataFrame(rows)


def load_mms_metadata(csv_path: str | Path) -> pd.DataFrame:
    csv = resolve_path(csv_path)
    meta = pd.read_csv(csv)
    if "External code" not in meta.columns:
        raise ValueError(f"M&Ms metadata file missing 'External code': {csv}")
    meta = meta.rename(
        columns={
            "External code": "case_id",
            "VendorName": "vendor_name",
            "Vendor": "vendor",
            "Centre": "center",
            "ED": "ed_frame",
            "ES": "es_frame",
            "Pathology": "disease_label",
            "Height": "height_cm",
            "Weight": "weight_kg",
            "Age": "age_years",
            "Sex": "sex",
        }
    )
    for col in ["ed_frame", "es_frame", "height_cm", "weight_kg", "age_years"]:
        if col in meta.columns:
            meta[col] = pd.to_numeric(meta[col], errors="coerce")
    meta["case_id"] = meta["case_id"].astype(str)
    return meta


def _mms_split_from_path(case_dir: Path, root: Path) -> str:
    rel = case_dir.relative_to(root)
    if len(rel.parts) >= 2 and rel.parts[0] == "Training":
        return f"Training/{rel.parts[1]}"
    return rel.parts[0]


def build_mms_inventory(mms_root: str | Path, metadata_csv: str | Path) -> pd.DataFrame:
    root = resolve_path(mms_root)
    meta = load_mms_metadata(metadata_csv).set_index("case_id", drop=False)
    rows: list[dict[str, Any]] = []
    case_dirs = [p for p in root.rglob("*") if p.is_dir() and re.match(r"^[A-Z0-9]{6}$", p.name)]
    for case_dir in sorted(case_dirs):
        case_id = case_dir.name
        image_path = case_dir / f"{case_id}_sa.nii"
        mask_path = case_dir / f"{case_id}_sa_gt.nii"
        meta_row = meta.loc[case_id].to_dict() if case_id in meta.index else {}
        geom = load_nifti_header(image_path) if image_path.exists() else None
        mask_geom = load_nifti_header(mask_path) if mask_path.exists() else None
        ed_frame = _as_int(meta_row.get("ed_frame"))
        es_frame = _as_int(meta_row.get("es_frame"))
        split = _mms_split_from_path(case_dir, root)
        has_image = image_path.exists()
        has_mask = mask_path.exists()
        num_frames = geom.num_frames if geom else None
        has_ed = ed_frame is not None and num_frames is not None and 0 <= ed_frame < num_frames
        has_es = es_frame is not None and num_frames is not None and 0 <= es_frame < num_frames
        shape_ok = geom is not None and mask_geom is not None and geom.image_shape == mask_geom.image_shape
        spacing_valid = bool(geom and geom.spacing_x_mm > 0 and geom.spacing_y_mm > 0)
        z_valid = bool(geom and geom.slice_increment_mm and geom.slice_increment_mm > 0)
        is_labeled_release = split != "Training/Unlabeled"
        exclusion = []
        if not is_labeled_release:
            exclusion.append("release_split_marked_unlabeled")
        if not has_image:
            exclusion.append("missing_image")
        if not has_mask:
            exclusion.append("missing_mask")
        if not has_ed:
            exclusion.append("missing_or_invalid_ed_phase")
        if not has_es:
            exclusion.append("missing_or_invalid_es_phase")
        if not shape_ok:
            exclusion.append("image_mask_shape_mismatch")
        if not spacing_valid:
            exclusion.append("invalid_inplane_spacing")
        if not z_valid:
            exclusion.append("invalid_z_spacing")
        rows.append(
            {
                "case_id": case_id,
                "patient_id": case_id,
                "vendor": meta_row.get("vendor"),
                "vendor_name": meta_row.get("vendor_name"),
                "center": meta_row.get("center"),
                "disease_label": meta_row.get("disease_label"),
                "split": split,
                "has_image": has_image,
                "has_mask": has_mask,
                "has_ed_phase": has_ed,
                "has_es_phase": has_es,
                "ed_frame": ed_frame,
                "es_frame": es_frame,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "image_shape": geom.as_json_shape() if geom else None,
                "mask_shape": mask_geom.as_json_shape() if mask_geom else None,
                "spacing_x_mm": geom.spacing_x_mm if geom else None,
                "spacing_y_mm": geom.spacing_y_mm if geom else None,
                "slice_thickness_mm": geom.slice_thickness_mm if geom else None,
                "slice_gap_mm": geom.slice_gap_mm if geom else None,
                "slice_increment_mm": geom.slice_increment_mm if geom else None,
                "slice_increment_variability_mm": geom.slice_increment_variability_mm if geom else None,
                "z_spacing_method": geom.z_spacing_method if geom else "unavailable",
                "affine_matrix": geom.affine_json() if geom else None,
                "num_slices": geom.num_slices if geom else None,
                "num_frames": geom.num_frames if geom else None,
                "height_cm": _as_float(meta_row.get("height_cm")),
                "weight_kg": _as_float(meta_row.get("weight_kg")),
                "age_years": _as_float(meta_row.get("age_years")),
                "sex": meta_row.get("sex"),
                "eligible_for_mms_100": len(exclusion) == 0,
                "exclusion_reason": ";".join(exclusion) if exclusion else None,
            }
        )
    inventory = pd.DataFrame(rows, columns=_MMS_INVENTORY_COLUMNS)
    if inventory.empty:
        inventory = inventory.astype({column: bool for column in _MMS_BOOLEAN_COLUMNS})
    return inventory


def dataset_provenance(
    dataset_id: str,
    release_name: str,
    release_url_or_access_note: str,
    local_raw_root: str | Path,
    raw_paths: list[str | Path],
    number_of_raw_cases: int,
    number_of_eligible_cases: int,
    number_of_selected_cases: int,
    ed_es_source: str,
    label_mapping_file: str,
    created_by_script: str,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "release_name": release_name,
        "release_url_or_access_note": release_url_or_access_note,
        "local_raw_root": str(resolve_path(local_raw_root)),
        "raw_file_manifest_sha256": file_manifest_sha256(raw_paths),
        "license_or_terms_summary": "See local release terms and docs/references/external_sources.bib.",
        "download_date_utc": utc_now_iso(),
        "number_of_raw_cases": int(number_of_raw_cases),
        "number_of_eligible_cases": int(number_of_eligible_cases),
        "number_of_selected_cases": int(number_of_selected_cases),
        "ed_es_source": ed_es_source,
        "label_mapping_file": label_mapping_file,
        "geometry_source_priority": [
            "dicom_ipp_projection",
            "nifti_affine_projection",
            "nifti_header_pixdim_unscaled_affine",
            "slice_thickness_plus_gap_fallback",
        ],
        "created_by_script": created_by_script,
        "git_commit": git_commit(),
    }
