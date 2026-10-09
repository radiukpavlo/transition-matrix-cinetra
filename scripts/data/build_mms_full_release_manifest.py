#!/usr/bin/env python
"""Build the tracked source manifest of the complete M&Ms-1 release (protocol CMR-TRANS-AUDIT 6.0.0, Decision D2).

Standard library only.  Inputs are read-only:

* the release folder (default ``TMMRI_MMS_SOURCE`` or ``<local path>``): the
  folder listing gives every file's size;
* ``BACKUP_MANIFEST.sha256`` beside it: SHA-256 of all 691 release files (``<sha>  <prefix>/<relpath>`` lines);
* the release information CSV: only ``External code``, ``ED`` and ``ES`` are read, to cross-check the folder list
  and the ED/ES placeholder rule.  Pathology, demographics, vendor and centre are never written: the M&Ms Research
  Use Agreement forbids redistributing any portion of the release, so v6 code reads them at run time from the
  mounted CSV (``src/tmmri/ingest/mms.py``);
* ``registry/sources/mms_100_source_normalization_manifest.csv`` for ``mms100_member``.

Outputs (LF line endings, sorted by ``case_id``):

* ``registry/sources/mms_full_release_manifest.csv`` with the columns ``case_id, release_split, image_relpath,
  image_sha256, image_bytes, mask_relpath, mask_sha256, mask_bytes, eligible_v6, exclusion_reason, mms100_member,
  duplicate_of`` (345 rows);
* ``registry/sources/mms_full_release_metadata_manifest.csv`` with the information CSV as a verified source entry.

``eligible_v6`` applies the metadata part of ``configs/cohorts.yaml`` ``mms.eligibility`` (release split and
CSV ED != ES); the image-level conditions (frames inside the frame count, mask annotated on exactly ED and ES) are
re-checked by ``tmmri.cli stage-sources`` / ``validate-cohort`` on every run.  ``--check`` compares with the tracked
files and writes nothing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import sys
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(os.environ.get("TMMRI_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
REGISTRY = PROJECT_ROOT / "registry" / "sources"
DEFAULT_SOURCE = "<local path>"
INFO_CSV = "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"
SPLITS = ("Training/Labeled", "Training/Unlabeled", "Validation", "Testing")
UNLABELED = "Training/Unlabeled"
# configs/splits.yaml duplicate_control.known_duplicates (voxel-identical image, mask, affine and metadata;
# REPORT_MMS_DATA 2026-09-28).  All four cases are Training/Unlabeled and already excluded.
KNOWN_DUPLICATES = (("C5Q2Y5", "E9L4N2"), ("A8C9H8", "Q0Q1Y4"))
COLUMNS = [
    "case_id", "release_split", "image_relpath", "image_sha256", "image_bytes", "mask_relpath", "mask_sha256",
    "mask_bytes", "eligible_v6", "exclusion_reason", "mms100_member", "duplicate_of",
]
METADATA_COLUMNS = ["relpath", "sha256", "bytes", "role"]


def read_backup_manifest(path: Path, folder_name: str) -> dict[str, str]:
    marker = f"/{folder_name}/"
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.strip().lstrip("*").replace("\\", "/")
        if marker in f"/{name}":
            out[f"/{name}".split(marker, 1)[1]] = digest.lower()
    return out


def listing(root: Path) -> dict[str, int]:
    return {p.relative_to(root).as_posix(): p.stat().st_size for p in sorted(root.rglob("*")) if p.is_file()}


def csv_frames(path: Path) -> dict[str, tuple[int, int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["External code"]: (int(float(row["ED"])), int(float(row["ES"]))) for row in csv.DictReader(handle)}


def mms100_members(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["case_id"] for row in csv.DictReader(handle)}


def render(rows: list[dict[str, object]], columns: list[str]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def build(source: Path, backup_manifest: Path) -> tuple[bytes, bytes, dict[str, object]]:
    hashes = read_backup_manifest(backup_manifest, source.name)
    sizes = listing(source)
    if set(hashes) != set(sizes):
        raise SystemExit(f"folder listing and {backup_manifest.name} differ: "
                         f"{sorted(set(hashes) ^ set(sizes))[:5]}")
    frames = csv_frames(source / INFO_CSV)
    members = mms100_members(REGISTRY / "mms_100_source_normalization_manifest.csv")
    duplicate_of = {a: b for a, b in KNOWN_DUPLICATES} | {b: a for a, b in KNOWN_DUPLICATES}
    rows: list[dict[str, object]] = []
    for rel in sizes:
        parts = PurePosixPath(rel).parts
        if not rel.endswith("_sa.nii.gz"):
            continue
        split = "/".join(parts[:-2])
        case_id = parts[-2]
        if split not in SPLITS or parts[-1] != f"{case_id}_sa.nii.gz":
            raise SystemExit(f"unexpected release layout: {rel}")
        mask_rel = f"{split}/{case_id}/{case_id}_sa_gt.nii.gz"
        if mask_rel not in sizes:
            raise SystemExit(f"mask missing for {case_id}")
        if case_id not in frames:
            raise SystemExit(f"{case_id} missing from {INFO_CSV}")
        ed, es = frames[case_id]
        if split == UNLABELED:
            eligible, reason = False, "MMS_RELEASE_UNLABELED"
        elif ed == es:
            eligible, reason = False, "INPUT_MISSING_ED_ES_FRAME"
        else:
            eligible, reason = True, ""
        rows.append({
            "case_id": case_id, "release_split": split,
            "image_relpath": rel, "image_sha256": hashes[rel], "image_bytes": sizes[rel],
            "mask_relpath": mask_rel, "mask_sha256": hashes[mask_rel], "mask_bytes": sizes[mask_rel],
            "eligible_v6": "true" if eligible else "false", "exclusion_reason": reason,
            "mms100_member": "true" if case_id in members else "false",
            "duplicate_of": duplicate_of.get(case_id, ""),
        })
    rows.sort(key=lambda r: str(r["case_id"]))
    if set(frames) != {r["case_id"] for r in rows}:
        raise SystemExit("CSV rows and case folders differ")
    if not members <= {r["case_id"] for r in rows if r["eligible_v6"] == "true"}:
        raise SystemExit("an MMS_100 member is not eligible in the full release")
    meta_rows = [{"relpath": INFO_CSV, "sha256": hashes[INFO_CSV], "bytes": sizes[INFO_CSV], "role": "info_csv"}]
    manifest = render(rows, COLUMNS)
    summary = {
        "cases": len(rows),
        "files": len(sizes),
        "total_bytes": sum(sizes.values()),
        "eligible_v6": sum(r["eligible_v6"] == "true" for r in rows),
        "by_split": {s: sum(r["release_split"] == s for r in rows) for s in SPLITS},
        "exclusions": {k: sum(r["exclusion_reason"] == k for r in rows) for k in sorted({str(r["exclusion_reason"]) for r in rows} - {""})},
        "mms100_members": sum(r["mms100_member"] == "true" for r in rows),
        "never_analysed_eligible": sum(r["eligible_v6"] == "true" and r["mms100_member"] == "false" for r in rows),
        "duplicate_rows": sum(bool(r["duplicate_of"]) for r in rows),
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "backup_manifest_sha256": hashlib.sha256(backup_manifest.read_bytes()).hexdigest(),
    }
    return manifest, render(meta_rows, METADATA_COLUMNS), summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=os.environ.get("TMMRI_MMS_SOURCE", DEFAULT_SOURCE))
    parser.add_argument("--backup-manifest", default=None, help="default: <source>/../BACKUP_MANIFEST.sha256")
    parser.add_argument("--out-dir", default=str(REGISTRY))
    parser.add_argument("--check", action="store_true", help="compare with the tracked files; write nothing")
    args = parser.parse_args(argv)
    source = Path(args.source).resolve()
    backup = Path(args.backup_manifest) if args.backup_manifest else source.parent / "BACKUP_MANIFEST.sha256"
    manifest, metadata, summary = build(source, backup)
    out = Path(args.out_dir)
    targets = {out / "mms_full_release_manifest.csv": manifest, out / "mms_full_release_metadata_manifest.csv": metadata}
    status = 0
    for path, content in targets.items():
        if args.check:
            same = path.is_file() and path.read_bytes() == content
            print(f"{'OK' if same else 'DIFFERS'}: {path}")
            status |= 0 if same else 1
        else:
            path.write_bytes(content)
            print(f"wrote {path}")
    for key, value in summary.items():
        print(f"{key}: {value}")
    return status


if __name__ == "__main__":
    sys.exit(main())
