"""M&Ms cohort table for protocol CMR-TRANS-AUDIT 6.0.0 (Decisions D1, D2, D8, D16, D17; ``configs/cohorts.yaml`` ``mms``).

One row per case of the complete M&Ms-1 release (345).  Identifiers, release split, hashes, eligibility,
``mms100_member`` and duplicates come from the tracked manifest (``registry/sources/mms_full_release_manifest.csv``).
Vendor, centre, pathology, height, weight and the 0-based ED/ES frame indices are read **at run time** from the
mounted information CSV, which is verified against ``mms_full_release_metadata_manifest.csv`` first (the M&Ms
Research Use Agreement forbids redistributing them, so they are never tracked).  The table lives only in the
runtime volume.  The CSV ED/ES are already 0-based.  Partition columns: ``mms_external`` (labelled 320),
``never_analysed`` (``mms_external`` minus the 100 MMS_100 members), ``refit_partition`` (``mms_refit_train`` =
Training/Labeled, ``mms_refit_eval`` = Validation + Testing).  The legacy inventory is unchanged.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from .sources import (
    MMS_INFO_CSV,
    SourceError,
    SourceFile,
    load_mms_manifest,
    load_mms_metadata_manifest,
    read_verified_bytes,
    sample_id,
)

DATASET = "MMS"
REFIT_PARTITIONS = {"Training/Labeled": "mms_refit_train", "Validation": "mms_refit_eval", "Testing": "mms_refit_eval"}
CSV_COLUMNS = {"External code": "case_id", "VendorName": "vendor_name", "Vendor": "vendor", "Centre": "centre",
               "ED": "ed_frame_index", "ES": "es_frame_index", "Pathology": "pathology", "Height": "height_cm",
               "Weight": "weight_kg", "Age": "age_years", "Sex": "sex"}


def read_info_csv(source_root: str | Path, metadata: list[SourceFile] | None = None) -> pd.DataFrame:
    """Read the mounted information CSV once, verified by SHA-256 (only the columns the protocol uses).

    ``metadata`` overrides the tracked ``mms_full_release_metadata_manifest.csv`` (synthetic fixtures).
    """

    entries = [e for e in (load_mms_metadata_manifest() if metadata is None else metadata) if e.role == "info_csv"]
    if len(entries) != 1:
        raise SourceError("mms_full_release_metadata_manifest.csv must list exactly one info_csv")
    data = read_verified_bytes(source_root, entries[0])
    frame = pd.read_csv(io.BytesIO(data), dtype={"External code": str})
    missing = set(CSV_COLUMNS).difference(frame.columns)
    if missing:
        raise SourceError(f"{MMS_INFO_CSV}: missing columns {sorted(missing)}")
    frame = frame[list(CSV_COLUMNS)].rename(columns=CSV_COLUMNS)
    for column in ("ed_frame_index", "es_frame_index", "centre"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    for column in ("height_cm", "weight_kg", "age_years"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    frame["sex"] = frame["sex"].astype("string").str.strip().str.upper()   # descriptive demographic strata only (Q28)
    return frame


def crosswalk(pathology: object, registry: dict | None) -> tuple[object, object]:
    """ACDC class and compatibility of an M&Ms pathology (registry ``label_tables.mms_to_acdc``; evaluation-only)."""

    if registry is None or pathology is None or pd.isna(pathology):
        return None, None
    entry = registry.get("label_tables", {}).get("mms_to_acdc", {}).get("crosswalk", {}).get(str(pathology))
    if entry is None:
        return None, "OFF_ONTOLOGY"
    return entry.get("acdc"), entry.get("compatibility")


def mms_cohort_table(
    source_root: str | Path,
    manifest: pd.DataFrame | None = None,
    *,
    info: pd.DataFrame | None = None,
    registry: dict | None = None,
    metadata: list[SourceFile] | None = None,
) -> pd.DataFrame:
    """Build the M&Ms cohort table (sorted by ``sample_id``); ``info`` overrides the mounted CSV (tests)."""

    manifest = load_mms_manifest() if manifest is None else manifest
    info = read_info_csv(source_root, metadata) if info is None else info
    table = manifest.merge(info, on="case_id", how="left", validate="one_to_one", indicator="merge_status")
    reasons: list[object] = []
    eligible: list[bool] = []
    for row in table.itertuples(index=False):
        reason = str(row.exclusion_reason or "")
        ok = bool(row.eligible_v6)
        if row.merge_status != "both":
            ok, reason = False, reason or "EXCLUDED_UPSTREAM"
        elif ok and (pd.isna(row.ed_frame_index) or pd.isna(row.es_frame_index) or row.ed_frame_index == row.es_frame_index
                     or row.ed_frame_index < 0 or row.es_frame_index < 0):
            ok, reason = False, "INPUT_MISSING_ED_ES_FRAME"
        eligible.append(ok)
        reasons.append(reason or None)
    table = table.drop(columns=["merge_status"])
    table["eligible_v6"] = eligible
    table["exclusion_reason"] = reasons
    table.insert(0, "sample_id", [sample_id(DATASET, c) for c in table["case_id"]])
    table.insert(1, "dataset", DATASET)
    table.insert(3, "patient_id", table["case_id"])
    table["duplicate_of"] = table["duplicate_of"].replace("", None)
    table["mms_external"] = table["eligible_v6"]
    table["never_analysed"] = table["eligible_v6"] & ~table["mms100_member"]
    table["refit_partition"] = [REFIT_PARTITIONS.get(s) if e else None for s, e in zip(table["release_split"], table["eligible_v6"])]
    walked = [crosswalk(p, registry) for p in table["pathology"]]
    table["crosswalk_acdc_class"] = [w[0] for w in walked]
    table["crosswalk_compatibility"] = [w[1] for w in walked]
    columns = ["sample_id", "dataset", "case_id", "patient_id", "release_split", "vendor", "vendor_name", "centre",
               "pathology", "crosswalk_acdc_class", "crosswalk_compatibility", "ed_frame_index", "es_frame_index",
               "height_cm", "weight_kg", "age_years", "sex", "eligible_v6", "exclusion_reason", "mms100_member", "never_analysed",
               "mms_external", "refit_partition", "duplicate_of", "image_relpath", "image_sha256", "image_bytes",
               "mask_relpath", "mask_sha256", "mask_bytes"]
    return table.reindex(columns=columns).sort_values("sample_id").reset_index(drop=True)
