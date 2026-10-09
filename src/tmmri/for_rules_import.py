from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .semantics.features import add_robust_outlier_flags, default_transition_feature_columns, feature_dictionary
from .io import resolve_path, sha256_file, utc_now_iso, write_checkpoint, write_table_pair, write_yaml
from .paths import resolve_recorded_path
from .semantics.qc import qc_case_report, validate_b_matrix, write_qc_markdown

EXPECTED_FOR_RULES_FILES = {
    "acdc_train": {"rows": 85, "split": "train"},
    "acdc_val": {"rows": 15, "split": "val"},
    "acdc_test": {"rows": 50, "split": "test"},
    "mms_100": {"rows": 100, "split": "mms_100_external"},
}

REQUIRED_COLUMNS = {
    "schema_version",
    "sample_id",
    "patient_id",
    "dataset_source",
    "split",
    "disease_label",
    "lv_edv_ml",
    "lv_esv_ml",
    "lv_ef_percent",
    "rv_edv_ml",
    "rv_esv_ml",
    "rv_ef_percent",
    "qc_manual_review_required",
    "qc_failure_reason",
}


@dataclass(frozen=True)
class ForRulesImportResult:
    b_acdc: pd.DataFrame
    b_mms: pd.DataFrame
    artifacts: list[Path]
    manifest_path: Path
    validation_failures: list[str]


def _artifact(paths: list[Path], new_paths: list[Path] | Path | None) -> None:
    if new_paths is None:
        return
    if isinstance(new_paths, Path):
        paths.append(new_paths)
    else:
        paths.extend(new_paths)


def _safe_extract_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        for info in zf.infolist():
            name = Path(info.filename)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"Unsafe path in archive: {info.filename}")
        zf.extractall(destination)


def _locate_table(extracted_root: Path, stem: str) -> Path:
    candidates = [extracted_root / "for_rules" / f"{stem}.parquet", extracted_root / f"{stem}.parquet"]
    candidates += sorted(extracted_root.glob(f"**/{stem}.parquet"))
    candidates += [extracted_root / "for_rules" / f"{stem}.csv", extracted_root / f"{stem}.csv"]
    candidates += sorted(extracted_root.glob(f"**/{stem}.csv"))
    seen: set[Path] = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if c.exists() and c.is_file():
            return c
    raise FileNotFoundError(f"for_rules archive is missing {stem}.parquet or {stem}.csv")


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _normalize_bool_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.endswith("_flag") or col.startswith("qc_") or col in {"bsa_available_flag", "myocardium_eccentricity_valid_flag", "perimeter_valid_flag", "lv_sphericity_valid_flag", "aha_mapping_feasible_flag"}:
            if out[col].dtype == object:
                low = out[col].astype(str).str.lower()
                if low.isin(["true", "false", "nan", "none", "", "0", "1"]).all():
                    out[col] = low.map({"true": True, "false": False, "1": True, "0": False, "nan": pd.NA, "none": pd.NA, "": pd.NA}).astype("boolean")
    return out


def validate_for_rules_frames(frames: dict[str, pd.DataFrame], strict_rows: bool = True) -> list[str]:
    failures: list[str] = []
    expected_cols: list[str] | None = None
    all_sample_ids: list[str] = []
    for stem, spec in EXPECTED_FOR_RULES_FILES.items():
        if stem not in frames:
            failures.append(f"missing_frame:{stem}")
            continue
        df = frames[stem]
        missing_cols = sorted(REQUIRED_COLUMNS - set(df.columns))
        if missing_cols:
            failures.append(f"{stem}:missing_required_columns:{','.join(missing_cols)}")
        if strict_rows and len(df) != int(spec["rows"]):
            failures.append(f"{stem}:unexpected_row_count:{len(df)}!= {spec['rows']}")
        if "split" in df.columns:
            split_values = set(df["split"].astype(str).dropna().unique().tolist())
            if split_values != {str(spec["split"])}:
                failures.append(f"{stem}:unexpected_split_values:{sorted(split_values)}")
        if expected_cols is None:
            expected_cols = list(df.columns)
        elif list(df.columns) != expected_cols:
            failures.append(f"{stem}:schema_order_mismatch_vs_acdc_train")
        if "sample_id" in df.columns:
            if df["sample_id"].astype(str).duplicated().any():
                failures.append(f"{stem}:duplicate_sample_id_within_frame")
            all_sample_ids.extend(df["sample_id"].astype(str).tolist())
    if pd.Series(all_sample_ids).duplicated().any():
        failures.append("duplicate_sample_id_across_for_rules_frames")
    return failures


def _copy_archive_payload(extracted_root: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for stem in EXPECTED_FOR_RULES_FILES:
        for suffix in [".parquet", ".csv", "_schema.csv"]:
            matches = sorted(extracted_root.glob(f"**/{stem}{suffix}"))
            if not matches:
                continue
            src = matches[0]
            dst = output_dir / src.name
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            copied.append(dst)
    return copied


def _write_schema(df: pd.DataFrame, path: str | Path) -> Path:
    schema = pd.DataFrame(
        [
            {
                "column_name": col,
                "dtype": str(df[col].dtype),
                "unit": None,
                "nullable": bool(df[col].isna().any()),
                "valid_range": None,
                "derivation_formula": None,
                "transition_target_eligible": False,
            }
            for col in df.columns
        ]
    )
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    schema.to_csv(out, index=False)
    return out


def _write_validation_artifacts(df: pd.DataFrame, stem: str, artifacts: list[Path]) -> pd.DataFrame:
    validation = validate_b_matrix(df)
    case_report = qc_case_report(df)
    _artifact(artifacts, write_table_pair(case_report, f"data/features/{stem}_qc_report.parquet", f"data/features/{stem}_qc_report.csv"))
    _artifact(artifacts, write_table_pair(validation, f"artifacts/reports/qc/{stem}_validation_gates.parquet", f"artifacts/reports/qc/{stem}_validation_gates.csv"))
    _artifact(artifacts, write_qc_markdown(df, validation, resolve_path(f"artifacts/reports/qc/{stem}_qc_report.md")))
    _artifact(artifacts, _write_schema(case_report, f"data/features/{stem}_qc_report_schema.csv"))
    _artifact(artifacts, _write_schema(validation, f"artifacts/reports/qc/{stem}_validation_gates_schema.csv"))
    return validation


def import_for_rules_archive(
    archive_path: str | Path,
    experiment_id: str,
    strict_rows: bool = True,
    extracted_output_dir: str | Path = "data/derived/for_rules",
    overwrite_b_matrices: bool = True,
) -> ForRulesImportResult:
    """Import the provided for_rules.zip as the authoritative derived ACDC/M&Ms substrate.

    The archive contains already-derived Matrix-B rows for ACDC train/val/test and the
    100-case M&Ms subset. It does not contain native NIfTI volumes in the supplied task
    archive, so native visual overlays and native image formal features remain conditional
    on separate raw-data availability.
    """
    # Read-only input: the documented data/derived/for_rules.zip resolves to
    # legacy/v5/data/derived/for_rules.zip (ADR-0004); outputs stay in the runtime roots.
    archive = resolve_path(resolve_recorded_path(archive_path))
    if not archive.exists():
        raise FileNotFoundError(f"for_rules archive not found: {archive}")
    extract_root = resolve_path("data/interim/for_rules_import")
    if extract_root.exists():
        shutil.rmtree(extract_root)
    _safe_extract_archive(archive, extract_root)

    frames: dict[str, pd.DataFrame] = {}
    source_paths: dict[str, Path] = {}
    for stem in EXPECTED_FOR_RULES_FILES:
        table_path = _locate_table(extract_root, stem)
        df = _normalize_bool_columns(_read_table(table_path))
        frames[stem] = df
        source_paths[stem] = table_path

    failures = validate_for_rules_frames(frames, strict_rows=strict_rows)
    if failures:
        raise ValueError("for_rules archive validation failed: " + "; ".join(failures))

    copied_payload = _copy_archive_payload(extract_root, resolve_path(extracted_output_dir))
    b_acdc = pd.concat([frames["acdc_train"], frames["acdc_val"], frames["acdc_test"]], ignore_index=True)
    b_mms = frames["mms_100"].copy().reset_index(drop=True)
    b_acdc = add_robust_outlier_flags(b_acdc, default_transition_feature_columns(b_acdc))
    b_mms = add_robust_outlier_flags(b_mms, default_transition_feature_columns(b_mms))

    artifacts: list[Path] = []
    artifacts.extend(copied_payload)
    if overwrite_b_matrices:
        artifacts.extend(write_table_pair(b_acdc, "data/features/B_ACDC.parquet", "data/features/B_ACDC.csv"))
        artifacts.extend(write_table_pair(b_mms, "data/features/B_MMS_100.parquet", "data/features/B_MMS_100.csv"))
        fd_acdc = feature_dictionary(b_acdc)
        fd_mms = feature_dictionary(b_mms)
        artifacts.extend(write_table_pair(fd_acdc, "data/features/B_ACDC_feature_dictionary.parquet", "data/features/B_ACDC_feature_dictionary.csv"))
        artifacts.extend(write_table_pair(fd_mms, "data/features/B_MMS_100_feature_dictionary.parquet", "data/features/B_MMS_100_feature_dictionary.csv"))
        artifacts.append(_write_schema(b_acdc, "data/features/B_ACDC_schema.csv"))
        artifacts.append(_write_schema(fd_acdc, "data/features/B_ACDC_feature_dictionary_schema.csv"))
        artifacts.append(_write_schema(b_mms, "data/features/B_MMS_100_schema.csv"))
        artifacts.append(_write_schema(fd_mms, "data/features/B_MMS_100_feature_dictionary_schema.csv"))
        acdc_val = _write_validation_artifacts(b_acdc, "B_ACDC", artifacts)
        mms_val = _write_validation_artifacts(b_mms, "B_MMS_100", artifacts)
        if not acdc_val["passed"].all():
            failures.append("B_ACDC_validation_gate_failure_after_for_rules_import")
        if not mms_val["passed"].all():
            failures.append("B_MMS_100_validation_gate_failure_after_for_rules_import")

    source_manifest_rows: list[dict[str, Any]] = []
    for stem, df in frames.items():
        for idx, row in df.iterrows():
            source_manifest_rows.append(
                {
                    "source_frame": stem,
                    "source_row_index": int(idx),
                    "sample_id": row.get("sample_id"),
                    "patient_id": row.get("patient_id"),
                    "dataset_source": row.get("dataset_source"),
                    "split": row.get("split"),
                    "disease_label": row.get("disease_label"),
                    "source_table_name": source_paths[stem].name,
                }
            )
    artifacts.extend(write_table_pair(pd.DataFrame(source_manifest_rows), f"artifacts/provenance/for_rules_row_manifest_{experiment_id}.parquet", f"artifacts/provenance/for_rules_row_manifest_{experiment_id}.csv"))

    archive_sha = sha256_file(archive)
    payload = {
        "schema_version": "for_rules_import_manifest_v1.0",
        "clinical_status": "research_use_only_not_clinically_validated",
        "experiment_id": experiment_id,
        "generated_at_utc": utc_now_iso(),
        "source_archive": str(archive),
        "source_archive_sha256": archive_sha,
        "source_archive_size_bytes": archive.stat().st_size,
        "authoritative_use": "B_ACDC and B_MMS_100 were regenerated from for_rules.zip before full-validation DSS stages.",
        "native_image_payload_present": False,
        "native_image_payload_note": "The supplied for_rules.zip contains derived Matrix-B tables and schemas, not native NIfTI image/mask volumes.",
        "frames": {
            stem: {
                "rows": int(len(df)),
                "columns": int(df.shape[1]),
                "split_values": sorted(df["split"].astype(str).unique().tolist()) if "split" in df.columns else [],
                "labels": {str(k): int(v) for k, v in df.get("disease_label", pd.Series(dtype=object)).value_counts(dropna=False).to_dict().items()},
                "source_table": str(source_paths[stem]),
                "source_table_sha256": sha256_file(source_paths[stem]),
            }
            for stem, df in frames.items()
        },
        "canonical_outputs": {
            "B_ACDC_rows": int(len(b_acdc)),
            "B_MMS_100_rows": int(len(b_mms)),
            "B_ACDC_split_counts": {str(k): int(v) for k, v in b_acdc["split"].value_counts().to_dict().items()},
            "B_MMS_100_split_counts": {str(k): int(v) for k, v in b_mms["split"].value_counts().to_dict().items()},
        },
        "validation_failures_or_warnings": failures,
    }
    manifest = write_yaml(payload, f"artifacts/provenance/for_rules_import_manifest_{experiment_id}.yaml")
    artifacts.append(manifest)
    write_checkpoint(
        "06_for_rules_import_done.json",
        "tmmri.for_rules_import import_for_rules_archive",
        [archive],
        [manifest, "data/features/B_ACDC.parquet", "data/features/B_MMS_100.parquet"],
        int(len(b_acdc) + len(b_mms)),
        failures,
    )
    return ForRulesImportResult(b_acdc=b_acdc, b_mms=b_mms, artifacts=artifacts, manifest_path=manifest, validation_failures=failures)
