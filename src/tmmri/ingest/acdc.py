"""ACDC cohort table for protocol CMR-TRANS-AUDIT 6.0.0 (Decisions D1, D15; ``configs/cohorts.yaml`` ``acdc``).

One row per patient of the mounted release (150): official split, ACDC class from ``Info.cfg`` ``Group``, 1-based
``Info.cfg`` ED/ES and the **0-based** ``ed_frame_index``/``es_frame_index`` (ED - 1, ES - 1; the same convention as
M&Ms), eligibility with a registered reason code, and the source paths of the released ED/ES frame files relative
to the source mount.  ``Info.cfg`` is verified against the tracked release manifest before it is parsed.  The 4D
cine (``patientNNN_4d.nii.gz``) is never read.  The legacy inventory (:mod:`tmmri.ingest.inventory`) is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .sources import SourceError, SourceFile, acdc_patient_files, load_acdc_manifest, read_verified_bytes, sample_id

DATASET = "ACDC"
SPLIT_TO_PARTITION = {"training": "acdc_dev", "testing": "acdc_test"}   # configs/cohorts.yaml acdc.roles


def parse_info_cfg(text: str) -> dict[str, str]:
    """Parse ``key: value`` lines of an ACDC ``Info.cfg``."""

    values: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def _int(value: object) -> int | None:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _float(value: object) -> float | None:
    try:
        out = float(str(value))
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def frame_file_names(patient_id: str, frame_number: int) -> tuple[str, str]:
    """Image and mask file names of a released frame (``frameNN`` uses the 1-based Info.cfg number)."""

    stem = f"{patient_id}_frame{int(frame_number):02d}"
    return f"{stem}.nii.gz", f"{stem}_gt.nii.gz"


def case_source_files(patient_id: str, files: dict[str, SourceFile], ed: int, es: int) -> dict[str, SourceFile]:
    """Map the roles ``info_cfg, ed_image, ed_mask, es_image, es_mask`` to tracked files (KeyError if absent)."""

    ed_image, ed_mask = frame_file_names(patient_id, ed)
    es_image, es_mask = frame_file_names(patient_id, es)
    names = {"info_cfg": "Info.cfg", "ed_image": ed_image, "ed_mask": ed_mask, "es_image": es_image, "es_mask": es_mask}
    return {role: SourceFile(role, files[name].relpath, files[name].sha256, files[name].bytes) for role, name in names.items()}


def evaluate_info(info: dict[str, str]) -> dict[str, object]:
    """Frame numbers, 0-based indices and the eligibility decision from a parsed ``Info.cfg``."""

    ed, es, n_frames = _int(info.get("ED")), _int(info.get("ES")), _int(info.get("NbFrame"))
    reason = ""
    if ed is None or es is None:
        reason = "INPUT_MISSING_ED_ES_FRAME"
    elif ed == es or ed < 1 or es < 1 or (n_frames is not None and (ed > n_frames or es > n_frames)):
        reason = "INPUT_MISSING_ED_ES_FRAME"
    return {
        "ed_frame_number": ed,
        "es_frame_number": es,
        "ed_frame_index": None if ed is None else ed - 1,
        "es_frame_index": None if es is None else es - 1,
        "n_frames": n_frames,
        "acdc_class": info.get("Group") or None,
        "height_cm": _float(info.get("Height")),
        "weight_kg": _float(info.get("Weight")),
        "reason": reason,
    }


def acdc_cohort_table(source_root: str | Path, manifest: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the ACDC cohort table from the mounted release (sorted by ``sample_id``)."""

    root = Path(source_root)
    manifest = load_acdc_manifest() if manifest is None else manifest
    rows: list[dict[str, object]] = []
    for patient_id, files in acdc_patient_files(manifest).items():
        split = files[next(iter(files))].relpath.split("/")[0]
        row: dict[str, object] = {
            "sample_id": sample_id(DATASET, patient_id),
            "dataset": DATASET,
            "case_id": patient_id,
            "patient_id": patient_id,
            "release_split": split,
            "partition": SPLIT_TO_PARTITION.get(split),
        }
        reason = ""
        info: dict[str, object] = {}
        if "Info.cfg" not in files:
            reason = "INPUT_DECODE_FAILURE"
        else:
            try:
                text = read_verified_bytes(root, files["Info.cfg"]).decode("utf-8", errors="replace")
                info = evaluate_info(parse_info_cfg(text))
                reason = str(info.pop("reason"))
            except SourceError:
                reason = "SOURCE_HASH_MISMATCH"
        row.update({k: info.get(k) for k in ("acdc_class", "ed_frame_number", "es_frame_number", "ed_frame_index",
                                               "es_frame_index", "n_frames", "height_cm", "weight_kg")})
        if not reason:
            try:
                sources = case_source_files(patient_id, files, int(info["ed_frame_number"]), int(info["es_frame_number"]))
                row.update({f"{role}_relpath": sf.relpath for role, sf in sources.items()})
            except KeyError:
                reason = "INPUT_MISSING_ED_ES_FRAME"
        row["eligible_v6"] = not reason
        row["exclusion_reason"] = reason or None
        rows.append(row)
    columns = ["sample_id", "dataset", "case_id", "patient_id", "release_split", "partition", "acdc_class",
               "ed_frame_number", "es_frame_number", "ed_frame_index", "es_frame_index", "n_frames", "height_cm",
               "weight_kg", "eligible_v6", "exclusion_reason", "info_cfg_relpath", "ed_image_relpath",
               "ed_mask_relpath", "es_image_relpath", "es_mask_relpath"]
    frame = pd.DataFrame(rows).reindex(columns=columns)
    for column in ("ed_frame_number", "es_frame_number", "ed_frame_index", "es_frame_index", "n_frames"):
        frame[column] = frame[column].astype("Int64")
    frame["eligible_v6"] = frame["eligible_v6"].astype(bool)
    return frame.sort_values("sample_id").reset_index(drop=True)
