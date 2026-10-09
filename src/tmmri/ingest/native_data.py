from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd

from ..config import cfg_path, load_config
from .inventory import build_acdc_inventory, build_mms_inventory
from ..io import file_manifest_sha256, resolve_path, sha256_file, utc_now_iso, write_table_pair, write_yaml


PROJECT_LABEL_VALUES = {0, 1, 2, 3}


def _scan_mask_labels(paths: list[str | Path], max_failures: int = 20) -> tuple[set[int], list[str]]:
    labels: set[int] = set()
    failures: list[str] = []
    for raw_path in paths:
        path = resolve_path(raw_path)
        try:
            data = np.asanyarray(nib.load(str(path)).dataobj)
            found = {int(x) for x in np.unique(np.rint(data).astype(np.int16))}
            labels.update(found)
            unexpected = sorted(found.difference(PROJECT_LABEL_VALUES))
            if unexpected:
                failures.append(f"{path}:unexpected_labels:{unexpected}")
        except Exception as exc:
            failures.append(f"{path}:mask_read_failed:{type(exc).__name__}:{exc}")
        if len(failures) >= max_failures:
            break
    return labels, failures


def _checksum_manifest(paths: list[str | Path], output_stem: str) -> tuple[Path, Path]:
    rows = []
    for raw_path in sorted({str(resolve_path(p)) for p in paths}):
        path = Path(raw_path)
        if path.exists() and path.is_file():
            rows.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = pd.DataFrame(rows)
    outputs = write_table_pair(manifest, f"data/raw_registry/{output_stem}_checksum_manifest.parquet", f"data/raw_registry/{output_stem}_checksum_manifest.csv")
    return outputs[0], outputs[1]


def _acdc_registry(cfg: dict[str, Any]) -> tuple[dict[str, Any], list[Path], list[str]]:
    root = cfg_path(cfg, "acdc.raw_root")
    inv = build_acdc_inventory(root)
    image_paths = list(inv["ed_image_path"].dropna()) + list(inv["es_image_path"].dropna())
    mask_paths = list(inv["ed_mask_path"].dropna()) + list(inv["es_mask_path"].dropna())
    labels, label_failures = _scan_mask_labels(mask_paths)
    checksum_parquet, checksum_csv = _checksum_manifest(image_paths + mask_paths, "acdc_native")
    failures: list[str] = []
    if len(inv) == 0:
        failures.append("ACDC:no_cases_found")
    if not inv["eligible_for_b"].all():
        failures.append(f"ACDC:ineligible_cases:{int((~inv['eligible_for_b']).sum())}")
    failures.extend(label_failures)
    registry = {
        "dataset_id": "ACDC",
        "registry_version": "native_v5",
        "created_at_utc": utc_now_iso(),
        "local_path": str(root),
        "file_count": int(len(image_paths) + len(mask_paths)),
        "sample_count": int(len(inv)),
        "eligible_sample_count": int(inv["eligible_for_b"].sum()),
        "checksum_manifest": str(checksum_parquet),
        "checksum_manifest_sha256": sha256_file(checksum_parquet),
        "raw_file_manifest_sha256": file_manifest_sha256(image_paths + mask_paths),
        "dataset_license_or_access_note": cfg.get("acdc", {}).get("release_url_or_access_note"),
        "expected_label_values": sorted(PROJECT_LABEL_VALUES),
        "observed_label_values": sorted(labels),
        "mask_label_convention": "ACDC source labels are remapped by configs/legacy/labels/label_mapping_ACDC.yaml",
        "voxel_spacing_metadata": "NIfTI affine/header",
        "ed_es_frame_source": "Info.cfg metadata with frame-specific native files",
        "known_preprocessing_requirements": ["project label normalization", "native geometry measurement"],
        "validation_failures": failures,
    }
    return registry, [checksum_parquet, checksum_csv], failures


def _mms_registry(cfg: dict[str, Any]) -> tuple[dict[str, Any], list[Path], list[str]]:
    root = cfg_path(cfg, "mms.raw_root")
    csv_path = cfg_path(cfg, "mms.diagnosis_csv")
    inv = build_mms_inventory(root, csv_path)
    image_paths = list(inv["image_path"].dropna())
    mask_paths = list(inv["mask_path"].dropna())
    eligible = inv[inv["eligible_for_mms_100"]].copy()
    selected_count = int(cfg.get("mms", {}).get("selected_case_count", 100))
    labels, label_failures = _scan_mask_labels(mask_paths)
    checksum_parquet, checksum_csv = _checksum_manifest(image_paths + mask_paths + [csv_path], "mms_native")
    failures: list[str] = []
    if len(inv) == 0:
        failures.append("MMS:no_cases_found")
    if len(eligible) < selected_count:
        failures.append(f"MMS:fewer_than_required_eligible_cases:{len(eligible)}<{selected_count}")
    failures.extend(label_failures)
    registry = {
        "dataset_id": "MMS_100",
        "registry_version": "native_v5",
        "created_at_utc": utc_now_iso(),
        "local_path": str(root),
        "metadata_csv": str(csv_path),
        "file_count": int(len(image_paths) + len(mask_paths) + 1),
        "sample_count": int(len(inv)),
        "eligible_sample_count": int(len(eligible)),
        "selected_sample_count_required": selected_count,
        "checksum_manifest": str(checksum_parquet),
        "checksum_manifest_sha256": sha256_file(checksum_parquet),
        "raw_file_manifest_sha256": file_manifest_sha256(image_paths + mask_paths + [csv_path]),
        "dataset_license_or_access_note": cfg.get("mms", {}).get("release_url_or_access_note"),
        "expected_label_values": sorted(PROJECT_LABEL_VALUES),
        "observed_label_values": sorted(labels),
        "mask_label_convention": "M&Ms labels match project convention in configs/legacy/labels/label_mapping_MMS_100.yaml",
        "voxel_spacing_metadata": "NIfTI affine/header",
        "ed_es_frame_source": "M&Ms diagnosis metadata CSV",
        "known_preprocessing_requirements": ["selected 100-case manifest", "project label validation", "native geometry measurement"],
        "validation_failures": failures,
    }
    return registry, [checksum_parquet, checksum_csv], failures


def validate_native_data(config_path: str | Path = "configs/legacy/pipeline/default.yaml", dataset: str = "all") -> list[Path]:
    # Writes go to the git-ignored runtime data/raw_registry; the tracked
    # records live in registry/sources and are never overwritten (ADR-0004).
    cfg = load_config(config_path)
    resolve_path("data/raw_registry").mkdir(parents=True, exist_ok=True)
    dataset = dataset.lower()
    artifacts: list[Path] = []
    failures: list[str] = []
    if dataset in {"all", "acdc"}:
        registry, registry_artifacts, registry_failures = _acdc_registry(cfg)
        artifacts.extend(registry_artifacts)
        artifacts.append(write_yaml(registry, "data/raw_registry/acdc_native.yaml"))
        failures.extend(registry_failures)
    if dataset in {"all", "mms", "mms_100"}:
        registry, registry_artifacts, registry_failures = _mms_registry(cfg)
        artifacts.extend(registry_artifacts)
        artifacts.append(write_yaml(registry, "data/raw_registry/mms_100_native.yaml"))
        failures.extend(registry_failures)
    if dataset not in {"all", "acdc", "mms", "mms_100"}:
        raise ValueError(f"Unknown dataset for native validation: {dataset}")
    if failures:
        raise RuntimeError("Native data validation failed: " + "; ".join(failures[:20]))
    return artifacts
