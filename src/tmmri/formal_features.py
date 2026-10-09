from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

from .ingest.geometry import select_frame
from .io import resolve_path, sha256_text, utc_now_iso, write_table_pair, write_yaml
from .ingest.labels import normalize_labels

IDENTIFIER_OR_LABEL_SUBSTRINGS = (
    "schema_version",
    "sample_id",
    "case_id",
    "patient_id",
    "study_id",
    "dataset",
    "subset",
    "split",
    "label",
    "path",
    "reason",
    "affine",
    "source",
)

TRANSITION_TARGET_HINTS = (
    "lv_edv_ml",
    "lv_esv_ml",
    "lv_sv_ml",
    "lv_ef_percent",
    "rv_edv_ml",
    "rv_esv_ml",
    "rv_sv_ml",
    "rv_ef_percent",
    "lv_myo_volume_ed_ml",
    "lv_myo_volume_es_ml",
    "lv_myocardial_mass_ed_g",
    "lv_myocardial_mass_indexed_ed_g_m2",
    "lv_rv_sv_discrepancy_ml",
    "global_wall_thickness_ed_mm",
    "global_wall_thickness_es_mm",
    "global_wall_thickening_percent",
    "lv_area_mean_ed_mm2",
    "lv_area_mean_es_mm2",
    "rv_area_mean_ed_mm2",
    "rv_area_mean_es_mm2",
    "lv_compactness_mean_ed",
    "lv_compactness_mean_es",
    "rv_compactness_mean_ed",
    "rv_compactness_mean_es",
    "lv_eccentricity_mean_ed",
    "lv_eccentricity_mean_es",
    "rv_eccentricity_mean_ed",
    "rv_eccentricity_mean_es",
    "lv_long_axis_length_mm",
)

STRICT_PROXY_ALLOWLIST_SUBSTRINGS = (
    "ed_frame_index",
    "es_frame_index",
    "heart_rate",
    "temporal_resolution",
    "height_cm",
    "weight_kg",
    "bmi_kg_m2",
    "bsa_",
    "spacing_",
    "slice_",
    "field_strength",
    "z_spacing",
    "qc_",
    "perimeter_valid_flag",
    "perimeter_spacing_applied_flag",
    "myocardium_eccentricity_valid_flag",
    "lv_sphericity_valid_flag",
    "aha_mapping_feasible_flag",
)

STRICT_PROXY_EXCLUDE_SUBSTRINGS = (
    "edv",
    "esv",
    "sv_ml",
    "ef_percent",
    "volume",
    "mass",
    "density",
    "discrepancy",
    "wall_thickness",
    "wall_thickening",
    "area_mean",
    "compactness",
    "eccentricity_mean",
    "long_axis",
    "sphericity_index",
    "aha16_seg",
)

def _resolve_existing_path(path_value: object) -> Path | None:
    if path_value is None or (isinstance(path_value, float) and pd.isna(path_value)):
        return None
    text = str(path_value)
    candidates = [Path(text)]
    normalized = text.replace("\\", "/")
    marker = "/data/"
    if marker in normalized:
        candidates.append(resolve_path("data") / normalized.split(marker, 1)[1])
    candidates.append(resolve_path(text))
    for c in candidates:
        if c.exists():
            return c
    return None


def native_inputs_available(b_df: pd.DataFrame) -> bool:
    if not ({"raw_ed_image_path", "raw_es_image_path", "raw_ed_mask_path", "raw_es_mask_path"}.issubset(b_df.columns) or {"raw_image_path", "raw_mask_path"}.issubset(b_df.columns)):
        return False
    for _, row in b_df.iterrows():
        explicit = all(
            _resolve_existing_path(row.get(col)) is not None
            for col in ["raw_ed_image_path", "raw_es_image_path", "raw_ed_mask_path", "raw_es_mask_path"]
            if col in b_df.columns
        )
        fallback = _resolve_existing_path(row.get("raw_image_path")) is not None and _resolve_existing_path(row.get("raw_mask_path")) is not None
        if not (explicit or fallback):
            return False
    return True


def _numeric_source_columns(
    b_df: pd.DataFrame,
    include_transition_targets: bool = False,
    proxy_policy: str = "strict_metadata_qc_no_targets",
) -> list[str]:
    """Return numeric source columns for the explicit no-native fallback.

    The default policy is intentionally conservative: it excludes direct clinical
    measurements and derived target families so the fallback cannot trivially
    reconstruct Matrix B from the same Matrix B targets. This fallback is still
    weaker than native image/mask extraction and is documented as a research-only
    smoke-test formal feature baseline.
    """
    policy = str(proxy_policy or "strict_metadata_qc_no_targets")
    cols: list[str] = []
    for col in b_df.columns:
        if any(s in col for s in IDENTIFIER_OR_LABEL_SUBSTRINGS):
            continue
        if not include_transition_targets:
            if col in TRANSITION_TARGET_HINTS:
                continue
            if policy == "strict_metadata_qc_no_targets" and any(s in col for s in STRICT_PROXY_EXCLUDE_SUBSTRINGS):
                continue
        if policy == "strict_metadata_qc_no_targets" and not any(s in col for s in STRICT_PROXY_ALLOWLIST_SUBSTRINGS):
            continue
        numeric = pd.to_numeric(b_df[col], errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        cols.append(col)
    if not cols:
        # Last resort for strongly reduced synthetic tests: allow non-target numeric columns
        # by dtype only.
        cols = [c for c in b_df.columns if pd.api.types.is_numeric_dtype(b_df[c])]
    return cols


def _stable_hash_feature(value: object, seed: int) -> float:
    payload = f"{seed}|{value}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    integer = int.from_bytes(digest[:8], "big", signed=False)
    return (integer / float(2**64 - 1)) * 2.0 - 1.0


def _projection_matrix(n_in: int, n_out: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mat = rng.normal(loc=0.0, scale=1.0 / np.sqrt(max(n_in, 1)), size=(n_in, n_out))
    return mat.astype(np.float64)


def _matrix_checksum(mat: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(mat).view(np.uint8)).hexdigest()


def _semantic_proxy_features(
    b_df: pd.DataFrame,
    latent_dim: int,
    seed: int,
    include_transition_targets: bool = False,
    proxy_policy: str = "strict_metadata_qc_no_targets",
) -> tuple[pd.DataFrame, dict[str, object]]:
    source_cols = _numeric_source_columns(b_df, include_transition_targets=include_transition_targets, proxy_policy=proxy_policy)
    values = b_df[source_cols].apply(pd.to_numeric, errors="coerce").astype(float) if source_cols else pd.DataFrame(index=b_df.index)
    med = values.median(skipna=True).fillna(0.0) if len(source_cols) else pd.Series(dtype=float)
    iqr = (values.quantile(0.75) - values.quantile(0.25)).replace(0, 1.0).fillna(1.0) if len(source_cols) else pd.Series(dtype=float)
    z = ((values.fillna(med) - med) / iqr).to_numpy(dtype=float) if len(source_cols) else np.empty((len(b_df), 0))
    hashed_cols = []
    for col in ["case_id", "patient_id", "scanner_vendor", "center_id", "ed_frame_index", "es_frame_index"]:
        if col in b_df.columns:
            hashed_cols.append(np.array([_stable_hash_feature(v, seed + i) for i, v in enumerate(b_df[col].astype(str))], dtype=float)[:, None])
    if hashed_cols:
        z = np.hstack([z] + hashed_cols)
    if z.shape[1] == 0:
        z = np.zeros((len(b_df), 1), dtype=float)
    mat = _projection_matrix(z.shape[1], latent_dim, seed)
    projected = z @ mat
    # Preserve a few direct non-target metadata channels at the front for auditability.
    direct_n = min(latent_dim, z.shape[1], 16)
    if direct_n:
        projected[:, :direct_n] = z[:, :direct_n]
    a_cols = [f"a_{i:04d}" for i in range(latent_dim)]
    out = pd.DataFrame(projected, columns=a_cols)
    for col in ["split", "dataset_source", "patient_id", "sample_id"]:
        if col in b_df.columns:
            out.insert(0, col, b_df[col].to_numpy())
    meta = {
        "extractor_id": "deterministic_semantic_proxy_v1",
        "clinical_status": "research_use_only",
        "native_images_used": False,
        "circularity_warning": "Packaged archive has no native NIfTI inputs. This explicit fallback uses only configured metadata/QC numeric columns by default and is suitable for pipeline smoke testing, not confirmatory transition-fidelity claims.",
        "include_transition_targets_in_proxy": bool(include_transition_targets),
        "proxy_source_policy": proxy_policy,
        "source_columns": source_cols,
        "latent_dim": int(latent_dim),
        "seed": int(seed),
        "projection_input_dimension": int(z.shape[1]),
        "projection_checksum": _matrix_checksum(mat),
        "generated_at_utc": utc_now_iso(),
    }
    return out, meta


def _intensity_summaries(image: np.ndarray, mask: np.ndarray) -> list[float]:
    vals: list[float] = []
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        finite = np.array([0.0])
    lo, hi = np.percentile(finite, [1, 99])
    scale = hi - lo if hi > lo else 1.0
    image_z = np.clip((image - lo) / scale, 0, 1)
    for label in [0, 1, 2, 3]:
        arr = image_z[mask == label]
        if arr.size == 0:
            vals.extend([0.0] * 6)
            continue
        vals.extend([float(np.mean(arr)), float(np.std(arr)), float(np.median(arr)), float(np.percentile(arr, 25)), float(np.percentile(arr, 75)), float(arr.size / mask.size)])
    return vals


def _load_native_phase_arrays(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import nibabel as nib

    dataset = str(row.get("dataset_source", "ACDC"))
    ed_image_path = _resolve_existing_path(row.get("raw_ed_image_path"))
    es_image_path = _resolve_existing_path(row.get("raw_es_image_path"))
    ed_mask_path = _resolve_existing_path(row.get("raw_ed_mask_path"))
    es_mask_path = _resolve_existing_path(row.get("raw_es_mask_path"))
    if all(p is not None for p in [ed_image_path, es_image_path, ed_mask_path, es_mask_path]):
        ed = int(row.get("ed_frame_index", 0))
        es = int(row.get("es_frame_index", ed))
        image_ed_full = np.asanyarray(nib.load(str(ed_image_path)).dataobj)
        image_es_full = np.asanyarray(nib.load(str(es_image_path)).dataobj)
        mask_ed_full = np.rint(np.asanyarray(nib.load(str(ed_mask_path)).dataobj)).astype(int)
        mask_es_full = np.rint(np.asanyarray(nib.load(str(es_mask_path)).dataobj)).astype(int)
        image_ed = select_frame(image_ed_full, ed) if image_ed_full.ndim == 4 else image_ed_full
        image_es = select_frame(image_es_full, es) if image_es_full.ndim == 4 else image_es_full
        mask_ed_raw = select_frame(mask_ed_full, ed) if mask_ed_full.ndim == 4 else mask_ed_full
        mask_es_raw = select_frame(mask_es_full, es) if mask_es_full.ndim == 4 else mask_es_full
        mask_ed, unexpected_ed = normalize_labels(mask_ed_raw, dataset)
        mask_es, unexpected_es = normalize_labels(mask_es_raw, dataset)
        if unexpected_ed or unexpected_es:
            raise ValueError(f"Unexpected native labels for {row.get('sample_id')}: {sorted(unexpected_ed.union(unexpected_es))}")
        return image_ed, image_es, mask_ed, mask_es

    image_path = _resolve_existing_path(row.get("raw_image_path"))
    mask_path = _resolve_existing_path(row.get("raw_mask_path"))
    if image_path is None or mask_path is None:
        raise FileNotFoundError("native image or mask path missing")
    image = np.asanyarray(nib.load(str(image_path)).dataobj)
    mask_raw = np.rint(np.asanyarray(nib.load(str(mask_path)).dataobj)).astype(int)
    if image.ndim == 4:
        ed = int(row.get("ed_frame_index", 0))
        es = int(row.get("es_frame_index", ed))
        image_ed = select_frame(image, ed)
        image_es = select_frame(image, es)
    else:
        image_ed = image
        image_es = image
    if mask_raw.ndim == 4:
        ed = int(row.get("ed_frame_index", 0))
        es = int(row.get("es_frame_index", ed))
        mask_ed_raw = select_frame(mask_raw, ed)
        mask_es_raw = select_frame(mask_raw, es)
    else:
        mask_ed_raw = mask_raw
        mask_es_raw = mask_raw
    mask_ed, unexpected_ed = normalize_labels(mask_ed_raw, dataset)
    mask_es, unexpected_es = normalize_labels(mask_es_raw, dataset)
    if unexpected_ed or unexpected_es:
        raise ValueError(f"Unexpected native labels for {row.get('sample_id')}: {sorted(unexpected_ed.union(unexpected_es))}")
    return image_ed, image_es, mask_ed, mask_es


def _descriptor_values(image_ed: np.ndarray, image_es: np.ndarray, mask_ed: np.ndarray, mask_es: np.ndarray) -> list[float]:
    """The 57 deterministic descriptors of ``native_deterministic_baseline_v2`` (shared by v5 and v6)."""

    vals = _intensity_summaries(image_ed, mask_ed) + _intensity_summaries(image_es, mask_es)
    for label in [1, 2, 3]:
        ed_frac = float(np.mean(mask_ed == label))
        es_frac = float(np.mean(mask_es == label))
        vals.extend([ed_frac, es_frac, ed_frac - es_frac])
    return vals


def _native_features_for_row(row: pd.Series) -> list[float]:
    image_ed, image_es, mask_ed, mask_es = _load_native_phase_arrays(row)
    return _descriptor_values(image_ed, image_es, mask_ed, mask_es)


def _native_a_features(b_df: pd.DataFrame, latent_dim: int, seed: int) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = [_native_features_for_row(row) for _, row in b_df.iterrows()]
    max_len = max(len(r) for r in rows) if rows else 1
    arr = np.zeros((len(rows), max_len), dtype=float)
    for i, r in enumerate(rows):
        arr[i, : len(r)] = r
    mat = _projection_matrix(arr.shape[1], latent_dim, seed)
    projected = arr @ mat
    direct_n = min(latent_dim, arr.shape[1], 32)
    if direct_n:
        projected[:, :direct_n] = arr[:, :direct_n]
    a_cols = [f"a_{i:04d}" for i in range(latent_dim)]
    out = pd.DataFrame(projected, columns=a_cols)
    for col in ["split", "dataset_source", "patient_id", "sample_id"]:
        if col in b_df.columns:
            out.insert(0, col, b_df[col].to_numpy())
    meta = {
        "extractor_id": "native_deterministic_baseline_v2",
        "clinical_status": "research_use_only",
        "native_images_used": True,
        "source": "native_ed_es_images_and_project_normalized_masks",
        "semantic_proxy_fallback_used": False,
        "latent_dim": int(latent_dim),
        "seed": int(seed),
        "projection_input_dimension": int(arr.shape[1]),
        "projection_checksum": _matrix_checksum(mat),
        "generated_at_utc": utc_now_iso(),
    }
    return out, meta


def build_a_matrix(
    b_df: pd.DataFrame,
    latent_dim: int = 128,
    seed: int = 20260611,
    allow_semantic_proxy_without_native: bool = False,
    include_transition_targets_in_proxy: bool = False,
    proxy_policy: str = "strict_metadata_qc_no_targets",
) -> tuple[pd.DataFrame, dict[str, object]]:
    if "sample_id" not in b_df.columns:
        raise ValueError("B matrix must contain sample_id for A extraction.")
    if native_inputs_available(b_df):
        a_df, meta = _native_a_features(b_df, latent_dim=latent_dim, seed=seed)
    elif allow_semantic_proxy_without_native:
        a_df, meta = _semantic_proxy_features(
            b_df,
            latent_dim=latent_dim,
            seed=seed,
            include_transition_targets=include_transition_targets_in_proxy,
            proxy_policy=proxy_policy,
        )
    else:
        raise FileNotFoundError(
            "Native images/masks are unavailable. Re-run with --allow-semantic-proxy-a for an explicitly flagged smoke-test fallback, or provide raw NIfTI inputs."
        )
    a_cols = [c for c in a_df.columns if c.startswith("a_")]
    if len(a_cols) != latent_dim:
        raise AssertionError(f"Expected {latent_dim} latent columns, found {len(a_cols)}")
    if not np.isfinite(a_df[a_cols].to_numpy(dtype=float)).all():
        raise AssertionError("A matrix contains NaN or infinite latent values.")
    if list(a_df["sample_id"].astype(str)) != list(b_df["sample_id"].astype(str)):
        raise AssertionError("A/B sample_id row alignment failed.")
    return a_df, meta


def write_a_artifacts(
    a_df: pd.DataFrame,
    meta: dict[str, object],
    dataset_stem: str,
    experiment_id: str,
) -> list[Path]:
    artifacts: list[Path] = []
    artifacts.extend(write_table_pair(a_df, f"data/features/A_{dataset_stem}.parquet", f"data/features/A_{dataset_stem}.csv"))
    feature_rows = []
    for col in [c for c in a_df.columns if c.startswith("a_")]:
        feature_rows.append(
            {
                "column_name": col,
                "dtype": str(a_df[col].dtype),
                "feature_family": meta.get("extractor_id"),
                "unit": "arbitrary_formal_feature_unit",
                "source": "native_image_mask" if meta.get("native_images_used") else "explicit_semantic_proxy_fallback",
                "clinical_status": "research_use_only",
            }
        )
    artifacts.extend(write_table_pair(pd.DataFrame(feature_rows), f"data/features/A_{dataset_stem}_feature_dictionary.parquet", f"data/features/A_{dataset_stem}_feature_dictionary.csv"))
    artifacts.append(write_yaml(meta, f"artifacts/transforms/A_{dataset_stem}_extractor_metadata_{experiment_id}.yaml"))
    return artifacts


def build_aligned_a_matrices(
    b_acdc: pd.DataFrame,
    b_mms: pd.DataFrame,
    latent_dim: int = 128,
    seed: int = 20260611,
    allow_semantic_proxy_without_native: bool = False,
    include_transition_targets_in_proxy: bool = False,
    proxy_policy: str = "strict_metadata_qc_no_targets",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, object]]:
    """Build ACDC and M&Ms A matrices with identical formal-feature coordinates.

    Native image features are extracted independently because the hand-crafted feature
    coordinates are fixed by code. For the explicit semantic-proxy fallback, this function
    fits source-column normalization and random projection on ACDC training rows, then
    applies the exact same source columns and projection to ACDC and M&Ms. This prevents
    cross-dataset coordinate drift.
    """
    if native_inputs_available(b_acdc) and native_inputs_available(b_mms):
        a_acdc, meta_acdc = _native_a_features(b_acdc, latent_dim=latent_dim, seed=seed)
        a_mms, meta_mms = _native_a_features(b_mms, latent_dim=latent_dim, seed=seed)
        return a_acdc, a_mms, meta_acdc, meta_mms
    if not allow_semantic_proxy_without_native:
        raise FileNotFoundError(
            "Native images/masks are unavailable. Re-run with --allow-semantic-proxy-a for an explicitly flagged smoke-test fallback, or provide raw NIfTI inputs."
        )
    source_cols = [
        c
        for c in _numeric_source_columns(
            b_acdc,
            include_transition_targets=include_transition_targets_in_proxy,
            proxy_policy=proxy_policy,
        )
        if c in b_mms.columns
    ]
    if not source_cols:
        source_cols = [c for c in b_acdc.columns if c in b_mms.columns and pd.api.types.is_numeric_dtype(b_acdc[c])]
    train_mask = b_acdc.get("split", pd.Series(["train"] * len(b_acdc))).astype(str).eq("train")
    train_values = b_acdc.loc[train_mask, source_cols].apply(pd.to_numeric, errors="coerce").astype(float)
    med = train_values.median(skipna=True).fillna(0.0)
    iqr = (train_values.quantile(0.75) - train_values.quantile(0.25)).replace(0, 1.0).fillna(1.0)
    hash_cols = [col for col in ["case_id", "patient_id", "scanner_vendor", "center_id", "ed_frame_index", "es_frame_index"] if col in b_acdc.columns and col in b_mms.columns]

    def transform(df: pd.DataFrame) -> pd.DataFrame:
        values = df[source_cols].apply(pd.to_numeric, errors="coerce").astype(float)
        z = ((values.fillna(med) - med) / iqr).to_numpy(dtype=float)
        hashed = [np.array([_stable_hash_feature(v, seed + i) for i, v in enumerate(df[col].astype(str))], dtype=float)[:, None] for col in hash_cols]
        if hashed:
            z2 = np.hstack([z] + hashed)
        else:
            z2 = z if z.shape[1] else np.zeros((len(df), 1), dtype=float)
        return pd.DataFrame(z2)

    z_acdc = transform(b_acdc).to_numpy(dtype=float)
    z_mms = transform(b_mms).to_numpy(dtype=float)
    mat = _projection_matrix(z_acdc.shape[1], latent_dim, seed)

    def project(df: pd.DataFrame, z: np.ndarray) -> pd.DataFrame:
        projected = z @ mat
        direct_n = min(latent_dim, z.shape[1], 16)
        if direct_n:
            projected[:, :direct_n] = z[:, :direct_n]
        a_cols = [f"a_{i:04d}" for i in range(latent_dim)]
        out = pd.DataFrame(projected, columns=a_cols)
        for col in ["split", "dataset_source", "patient_id", "sample_id"]:
            if col in df.columns:
                out.insert(0, col, df[col].to_numpy())
        if not np.isfinite(out[a_cols].to_numpy(dtype=float)).all():
            raise AssertionError("A matrix contains NaN or infinite latent values.")
        if list(out["sample_id"].astype(str)) != list(df["sample_id"].astype(str)):
            raise AssertionError("A/B sample_id row alignment failed.")
        return out

    a_acdc = project(b_acdc, z_acdc)
    a_mms = project(b_mms, z_mms)
    common = {
        "extractor_id": "deterministic_semantic_proxy_v1_aligned",
        "clinical_status": "research_use_only",
        "native_images_used": False,
        "circularity_warning": "Packaged archive has no native NIfTI inputs. This explicit fallback uses only configured metadata/QC numeric columns by default and is suitable for pipeline smoke testing, not confirmatory transition-fidelity claims.",
        "include_transition_targets_in_proxy": bool(include_transition_targets_in_proxy),
        "proxy_source_policy": proxy_policy,
        "source_columns": source_cols,
        "hash_columns": hash_cols,
        "latent_dim": int(latent_dim),
        "seed": int(seed),
        "projection_input_dimension": int(z_acdc.shape[1]),
        "projection_checksum": _matrix_checksum(mat),
        "normalization_fitted_on": "ACDC_train_rows_only",
        "training_medians": {k: float(v) for k, v in med.to_dict().items()},
        "training_iqr": {k: float(v) for k, v in iqr.to_dict().items()},
        "generated_at_utc": utc_now_iso(),
    }
    return a_acdc, a_mms, {**common, "dataset_stem": "ACDC"}, {**common, "dataset_stem": "MMS_100"}


# ----------------------------------------------------------------------------- protocol 6.0.0 formal A
# configs/formal.yaml primary: formal_descriptors_v3 = the 57 unprojected descriptors of
# native_deterministic_baseline_v2, computed from a staged case (tmmri.ingest.staging.load_staged_case) with the
# same descriptor math as the legacy path (_descriptor_values).  The P6 128-d Gaussian projection is a separate
# continuity comparator (comparators.p6_projection_128).  Standardization happens on the fit partition, downstream.

FORMAL_V3_NAME = "formal_descriptors_v3"
FORMAL_V3_EXTRACTOR = "native_deterministic_baseline_v2"
FORMAL_V3_STATISTICS = ("mean", "sd_population", "median", "p25", "p75", "label_voxel_fraction")
FORMAL_V3_LABELS = (0, 1, 2, 3)


def formal_descriptor_names() -> list[str]:
    """Column names in ``configs/formal.yaml`` order (ED labels 0-3 x 6, ES labels 0-3 x 6, labels 1-3 x 3)."""

    names = [f"fd_{phase}_l{label}_{stat}" for phase in ("ed", "es") for label in FORMAL_V3_LABELS for stat in FORMAL_V3_STATISTICS]
    for label in (1, 2, 3):
        names.extend([f"fd_l{label}_fraction_ed", f"fd_l{label}_fraction_es", f"fd_l{label}_fraction_ed_minus_es"])
    return names


def formal_descriptors_from_arrays(image_ed: np.ndarray, image_es: np.ndarray, mask_ed: np.ndarray, mask_es: np.ndarray) -> np.ndarray:
    return np.asarray(_descriptor_values(image_ed, image_es, mask_ed, mask_es), dtype=np.float64)


def formal_descriptors_v3(case) -> np.ndarray:
    """57-vector of a :class:`tmmri.ingest.staging.StagedCase` (project labels, float32 images)."""

    return formal_descriptors_from_arrays(case.ed_image, case.es_image, case.ed_mask, case.es_mask)


def _formal_worker(job: dict) -> dict:
    from .ingest.staging import load_staged_case

    out = {"sample_id": f"{job['dataset']}:{job['case_id']}", "case_id": job["case_id"], "values": None,
           "reason_code": None, "detail": None}
    try:
        case = load_staged_case(job["dataset"], job["case_id"], job["staged_root"], label_map=job.get("label_map"))
    except (FileNotFoundError, OSError, ValueError) as exc:
        out.update(reason_code="INPUT_DECODE_FAILURE", detail=f"{type(exc).__name__}: {exc}")
        return out
    values = formal_descriptors_v3(case)
    if not np.isfinite(values).all() or not np.isfinite(case.ed_image).all() or not np.isfinite(case.es_image).all():
        out.update(reason_code="INPUT_NONFINITE", detail="non-finite voxel or descriptor")
        return out
    out["values"] = values.tolist()
    return out


def extract_formal_table(
    dataset: str,
    case_ids: Iterable[str],
    staged_root: str | Path,
    *,
    n_jobs: int = 1,
    label_map: dict | None = None,
    expected_dimension: int = 57,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Formal A for staged cases: (table sorted by ``sample_id``, failures with reason codes).

    Results are gathered in ``case_id`` order, so the table is identical for every ``n_jobs``.
    """

    from .runs import parallel_map

    names = formal_descriptor_names()
    if len(names) != int(expected_dimension):
        raise AssertionError(f"formal_descriptors_v3 has {len(names)} columns, configs/formal.yaml says {expected_dimension}")
    jobs = [{"dataset": dataset, "case_id": str(c), "staged_root": str(staged_root), "label_map": label_map} for c in sorted(set(case_ids))]
    results = parallel_map(_formal_worker, jobs, n_jobs=n_jobs)
    ok = [r for r in results if r["values"] is not None]
    table = pd.DataFrame([r["values"] for r in ok], columns=names, dtype=np.float64)
    table.insert(0, "case_id", [r["case_id"] for r in ok])
    table.insert(0, "dataset", dataset)
    table.insert(0, "sample_id", [r["sample_id"] for r in ok])
    failures = pd.DataFrame([{k: r[k] for k in ("sample_id", "case_id", "reason_code", "detail")} for r in results if r["values"] is None],
                            columns=["sample_id", "case_id", "reason_code", "detail"])
    return table.sort_values("sample_id").reset_index(drop=True), failures.sort_values("sample_id").reset_index(drop=True)


def p6_settings(formal_cfg: dict) -> dict[str, int]:
    """Seed, output dimension and retained raw columns of ``configs/formal.yaml`` ``comparators.p6_projection_128``."""

    import re

    cfg = formal_cfg["comparators"]["p6_projection_128"]
    match = re.search(r"first\s+(\d+)\s+raw", str(cfg.get("raw_retained", "")))
    if match is None:
        raise ValueError("configs/formal.yaml comparators.p6_projection_128.raw_retained must name 'the first N raw descriptors'")
    return {"seed": int(cfg["seed"]), "output_dimension": int(cfg["output_dimension"]), "raw_retained": int(match.group(1))}


def p6_projection_128(descriptors: np.ndarray, *, seed: int, output_dimension: int = 128, raw_retained: int = 32) -> np.ndarray:
    """P6 continuity comparator: Gaussian N(0, 1/n_in) projection, first ``raw_retained`` columns replaced by raw A."""

    x = np.asarray(descriptors, dtype=np.float64)
    projected = x @ _projection_matrix(x.shape[1], int(output_dimension), int(seed))
    direct = min(int(output_dimension), x.shape[1], int(raw_retained))
    projected[:, :direct] = x[:, :direct]
    return projected


def p6_projection_table(formal: pd.DataFrame, *, seed: int, output_dimension: int = 128, raw_retained: int = 32) -> pd.DataFrame:
    values = p6_projection_128(formal[formal_descriptor_names()].to_numpy(dtype=np.float64), seed=seed,
                               output_dimension=output_dimension, raw_retained=raw_retained)
    out = pd.DataFrame(values, columns=[f"a_{i:04d}" for i in range(int(output_dimension))])
    for column in ("case_id", "dataset", "sample_id"):
        out.insert(0, column, formal[column].to_numpy())
    return out