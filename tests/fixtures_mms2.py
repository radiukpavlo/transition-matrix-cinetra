"""Synthetic M&Ms-2 fixtures for the protocol-v6.0.1 tests (never cohort data; every root carries SYNTHETIC_FIXTURE).

* :func:`make_mms2_root` writes an extracted M&Ms-2 source area directly: ``MnM2/dataset/<ID>/<ID>_SA_{ED,ES}[_gt].nii.gz``
  (3D, project labels 1 LV, 2 MYO, 3 RV), ``MMS2_INFO_EXTERNAL.csv`` and the extraction manifest.
* :func:`make_mms2_archive` writes a small uncompressed tar with the declared member layout for all 360 subjects
  (short-axis ED/ES files, long-axis and cine decoys, directory entries, ``dataset_information.csv``) plus the
  ``SYNTHETIC_FIXTURE.archive.json`` descriptor; bsdtar reads it exactly like the RAR release.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

import fixtures_v6 as fx
from tmmri.ingest import mms2

INFO_COLUMNS = ["SUBJECT_CODE", "DISEASE", "VENDOR", "SCANNER", "FIELD"]   # the release header (read before the tag; no SEX, no AGE)
VOXEL_AFFINE = np.array([[1.25, 0.0, 0.0, -10.0], [0.0, 1.25, 0.0, 5.0], [0.0, 0.0, 8.0, 30.0], [0.0, 0.0, 0.0, 1.0]])
UNIT_AFFINE = np.diag([-1.0, -1.0, 1.0, 1.0])          # orientation only; voxel size in the header pixdim (ACDC-like)
UNIT_ZOOMS = (1.5625, 1.5625, 10.0)
PATHOLOGIES = ("NOR", "HCM", "LV", "ARR", "FALL", "CIA", "TRI", "DRV")
VENDORS = ("SIEMENS", "GE", "PHILIPS")


def find_tar() -> str | None:
    for candidate in ("C:/Windows/System32/tar.exe", shutil.which("bsdtar"), shutil.which("tar")):
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def info_row(sid: str, i: int, pathology: str | None = None) -> dict:
    return {"SUBJECT_CODE": str(int(sid)), "DISEASE": pathology or PATHOLOGIES[i % len(PATHOLOGIES)], "VENDOR": VENDORS[i % 3],
            "SCANNER": f"Model{i % 4}", "FIELD": "1.5" if i % 2 else "3.0"}


def info_csv_bytes(rows: list[dict], columns: list[str] = INFO_COLUMNS, *, line_end: str = "\r\n") -> bytes:
    """The release CSV is comma-separated with CRLF line endings (header observed before the tag)."""

    return pd.DataFrame(rows, columns=columns).to_csv(index=False, lineterminator=line_end).encode("utf-8")


def case_arrays(sid: str, shape=(26, 24, 3)) -> dict[str, np.ndarray]:
    out = {}
    for phase, radius in (("ed", 6.0), ("es", 4.0)):
        proj = fx.project_mask(shape, radius, f"mms2|{sid}|{phase}")
        out[f"{phase}_mask"] = proj.astype(np.uint8)
        out[f"{phase}_image"] = fx.image_from_mask(proj, f"mms2|{sid}|{phase}")
    return out


def write_case(root: Path, sid: str, arrays: dict[str, np.ndarray], *, geometry: str = "voxel", mask_affine=None) -> dict[str, str]:
    """Write the four files of a subject; ``geometry`` voxel | unit | inconsistent (affine norms 2.0, pixdim 1.25)."""

    folder = root / "MnM2" / "dataset" / sid
    rels = {}
    for role, name in mms2.case_file_names(sid).items():
        if geometry == "unit":
            affine, zooms = UNIT_AFFINE, UNIT_ZOOMS
        elif geometry == "inconsistent":
            affine, zooms = np.diag([2.0, 2.0, 8.0, 1.0]), (1.25, 1.25, 8.0)
        else:
            affine, zooms = VOXEL_AFFINE, None
        if mask_affine is not None and role.endswith("mask"):
            affine = mask_affine
        data = arrays[role]
        dtype = np.uint8 if role.endswith("mask") else np.float32
        fx.write_nifti(folder / name, data.astype(dtype), affine, zooms=zooms, dtype=dtype)
        rels[role] = (folder / name).relative_to(root).as_posix()
    return rels


def finish_root(root: Path, cases: dict[str, dict[str, str]], info: list[dict]) -> Path:
    """Write ``MMS2_INFO_EXTERNAL.csv``, the extraction manifest and the marker for ``cases`` ({sid: {role: relpath}})."""

    data = info_csv_bytes(info)
    (root / mms2.INFO_EXTERNAL).write_bytes(data)
    rows = []
    for sid, rels in cases.items():
        for role, rel in rels.items():
            payload = (root / rel).read_bytes()
            rows.append({"case_id": mms2.case_key(sid), "role": role, "relpath": rel, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                         "archive_member": rel})
    rows.append({"case_id": mms2.INFO_CASE, "role": mms2.INFO_ROLE, "relpath": mms2.INFO_EXTERNAL, "bytes": len(data),
                 "sha256": hashlib.sha256(data).hexdigest(), "archive_member": "dataset_information.csv (external rows only)"})
    mms2.write_file_manifest(root, rows)
    fx.mark_synthetic(root)
    return root


def make_mms2_root(root: Path, subjects: list[str], *, pathologies: dict[str, str] | None = None, geometry: dict[str, str] | None = None,
                   arrays: dict[str, dict[str, np.ndarray]] | None = None, shape=(26, 24, 3)) -> Path:
    pathologies, geometry, arrays = pathologies or {}, geometry or {}, arrays or {}
    cases = {}
    for sid in subjects:
        cases[sid] = write_case(root, sid, arrays.get(sid) or case_arrays(sid, shape), geometry=geometry.get(sid, "voxel"))
    info = [info_row(sid, i, pathologies.get(sid)) for i, sid in enumerate(subjects)]
    return finish_root(root, cases, info)


def make_mms2_archive(folder: Path, *, info_rows: list[dict] | None = None, info_columns: list[str] = INFO_COLUMNS) -> Path:
    """A tar with the M&Ms-2 member layout for all 360 subjects (tiny placeholder payloads; tests of the extraction only)."""

    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / "MnM2_synthetic.tar"
    rows = info_rows if info_rows is not None else [info_row(sid, i) for i, sid in enumerate(mms2.SUBJECT_IDS)]
    with tarfile.open(archive, "w", format=tarfile.USTAR_FORMAT) as tar:
        def add(name: str, payload: bytes | None) -> None:
            info = tarfile.TarInfo(name)
            if payload is None:
                info.type, info.mode = tarfile.DIRTYPE, 0o755
                tar.addfile(info)
                return
            info.size, info.mode = len(payload), 0o644
            tar.addfile(info, io.BytesIO(payload))

        add("MnM2/", None)
        add("MnM2/dataset/", None)
        add("MnM2/dataset_information.csv", info_csv_bytes(rows, info_columns))
        add("MnM2/readme.txt", b"synthetic readme\r\n")
        for sid in mms2.SUBJECT_IDS:
            add(f"MnM2/dataset/{sid}/", None)
            for suffix in ("SA_ED", "SA_ED_gt", "SA_ES", "SA_ES_gt", "SA_CINE", "LA_ED", "LA_ED_gt", "LA_ES", "LA_ES_gt", "LA_CINE"):
                add(f"MnM2/dataset/{sid}/{sid}_{suffix}.nii.gz", f"synthetic {sid} {suffix} ".encode("ascii") * (1 + int(sid) % 7))
    fx.mark_synthetic(folder)
    data = archive.read_bytes()
    (folder / mms2.SYNTHETIC_ARCHIVE_DESCRIPTOR).write_text(
        json.dumps({"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}), encoding="utf-8")
    return archive
