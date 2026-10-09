"""Synthetic NIfTI fixture builders for the protocol-6.0.0 tests (never cohort data).

Every source root written here carries the ``SYNTHETIC_FIXTURE`` marker.  ACDC-style fixtures use the release
layout (``<split>/<patient>/Info.cfg`` + ``patientNNN_frameNN[_gt].nii.gz``), the release label convention
(1 RV, 2 MYO, 3 LV) and the unit-scaled affine with the voxel size only in the header pixdim.  M&Ms-style fixtures
use ``<split>/<ID>/<ID>_sa[_gt].nii.gz`` 4D files, masks either float32 or uint8 with ``scl_slope`` 3/255, an
unlabeled case (CSV ED = ES = 0) and a voxel-identical duplicate.  Seeds derive from ``SEED_SYNTHETIC``.
:func:`mms_refit_cases` adds enough Training/Labeled vendor A/B cases for the D17 refit arm, and
:func:`write_fixture_manifests` makes a fixture root self-describing (``SYNTHETIC_FIXTURE.manifest.csv``), so the
CLI can run the whole DAG on it without test-only overrides.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

SEED_SYNTHETIC = 20260613
MARKER = "SYNTHETIC_FIXTURE"
ACDC_TO_RELEASE = {0: 0, 1: 3, 2: 2, 3: 1}   # project -> ACDC release labels (the swap is an involution)
MMS_INFO_CSV = "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"


def rng_for(key: str) -> np.random.Generator:
    digest = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    return np.random.default_rng(np.random.SeedSequence([SEED_SYNTHETIC, digest]))


def project_mask(shape: tuple[int, int, int], lv_radius: float, key: str) -> np.ndarray:
    """Project-convention labels (1 LV, 2 MYO ring, 3 RV crescent) on every slice."""

    nx, ny, nz = shape
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    cx, cy = nx / 2.0, ny / 2.0
    r = np.hypot(x - cx, y - cy)
    rv = np.hypot(x - (cx + lv_radius + 3.5), y - cy)
    out = np.zeros(shape, dtype=np.uint8)
    rng = rng_for(key)
    for z in range(nz):
        radius = lv_radius * (1.0 - 0.08 * abs(z - nz / 2.0) / max(nz, 1)) + rng.uniform(-0.2, 0.2)
        sl = np.zeros((nx, ny), dtype=np.uint8)
        sl[(rv < radius * 0.9) & (r > radius + 2.0)] = 3
        sl[(r <= radius + 2.0) & (r > radius)] = 2
        sl[r <= radius] = 1
        out[..., z] = sl
    return out


def image_from_mask(mask: np.ndarray, key: str) -> np.ndarray:
    rng = rng_for("img|" + key)
    base = np.array([40.0, 220.0, 90.0, 180.0])[mask.astype(int)]
    return (base + rng.normal(0.0, 8.0, size=mask.shape)).astype(np.float32)


def write_nifti(path: Path, data: np.ndarray, affine: np.ndarray, *, zooms=None, dtype=None, slope=None) -> Path:
    img = nib.Nifti1Image(data, affine)
    if zooms is not None:
        img.header.set_zooms(zooms)
    if dtype is not None:
        img.header.set_data_dtype(dtype)
    if slope is not None:
        img.header.set_slope_inter(slope, 0.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))
    return path


def mark_synthetic(root: Path) -> None:
    (root / MARKER).write_text("synthetic fixture for tests; not cohort data\n", encoding="utf-8")


# ----------------------------------------------------------------------------- ACDC


def acdc_patients(n_per_class: int = 1, classes=("DCM", "HCM", "MINF", "NOR", "RV"), testing_per_class: int = 0):
    rows, number = [], 1
    for split, count in (("training", n_per_class), ("testing", testing_per_class)):
        for cls in classes:
            for _ in range(count):
                rows.append({"patient_id": f"patient{number:03d}", "split": split, "group": cls, "ed": 1, "es": 3, "n_frames": 4})
                number += 1
    return rows


def make_acdc_source(root: Path, patients: list[dict], shape=(24, 22, 4)) -> Path:
    """Write an ACDC-style release below ``root`` (the ``database`` folder)."""

    affine = np.diag([-1.0, -1.0, 1.0, 1.0])     # orientation only; voxel size lives in pixdim
    zooms = (1.5625, 1.5625, 10.0)
    (root / "MANDATORY_CITATION.md").parent.mkdir(parents=True, exist_ok=True)
    (root / "MANDATORY_CITATION.md").write_text("synthetic\n", encoding="utf-8")
    for p in patients:
        pid, folder = p["patient_id"], root / p["split"] / p["patient_id"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "Info.cfg").write_text(
            f"ED: {p['ed']}\nES: {p['es']}\nGroup: {p['group']}\nHeight: 175.0\nNbFrame: {p['n_frames']}\nWeight: 72.0\n",
            encoding="utf-8")
        (folder / "MANDATORY_CITATION.md").write_text("synthetic\n", encoding="utf-8")
        cine = np.zeros(shape + (p["n_frames"],), dtype=np.int16)
        for phase, frame, radius in (("ed", p["ed"], 6.0), ("es", p["es"], 4.0)):
            proj = project_mask(shape, radius, f"{pid}|{phase}")
            released = np.vectorize(ACDC_TO_RELEASE.get)(proj).astype(np.uint8)
            image = np.rint(image_from_mask(proj, f"{pid}|{phase}")).astype(np.int16)
            cine[..., frame - 1] = image
            write_nifti(folder / f"{pid}_frame{frame:02d}.nii.gz", image, affine, zooms=zooms, dtype=np.int16)
            write_nifti(folder / f"{pid}_frame{frame:02d}_gt.nii.gz", released, affine, zooms=zooms, dtype=np.uint8)
        write_nifti(folder / f"{pid}_4d.nii.gz", cine, affine, zooms=zooms + (1.0,), dtype=np.int16)
    mark_synthetic(root)
    return root


def acdc_manifest(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.name.startswith(MARKER):
            data = path.read_bytes()
            rows.append({"relative_path": "database/" + path.relative_to(root).as_posix(), "size_bytes": len(data),
                         "sha256": hashlib.sha256(data).hexdigest()})
    frame = pd.DataFrame(rows)
    frame["relpath"] = frame["relative_path"].str.slice(len("database/"))
    return frame


# ----------------------------------------------------------------------------- M&Ms


def mms_cases() -> list[dict]:
    """Four labelled cases (two mask encodings), one unlabeled case and a duplicate of it."""

    return [
        {"case_id": "A1A1A1", "split": "Training/Labeled", "vendor": "A", "vendor_name": "Siemens", "centre": 1, "ed": 0, "es": 2, "n_frames": 5, "pathology": "NOR", "encoding": "uint8_scaled"},
        {"case_id": "B2B2B2", "split": "Validation", "vendor": "B", "vendor_name": "Philips", "centre": 2, "ed": 1, "es": 3, "n_frames": 5, "pathology": "DCM", "encoding": "float32"},
        {"case_id": "C3C3C3", "split": "Testing", "vendor": "C", "vendor_name": "GE", "centre": 4, "ed": 0, "es": 3, "n_frames": 4, "pathology": "HHD", "encoding": "uint8_scaled"},
        {"case_id": "D4D4D4", "split": "Testing", "vendor": "D", "vendor_name": "Canon", "centre": 5, "ed": 4, "es": 1, "n_frames": 6, "pathology": "HCM", "encoding": "float32"},
        {"case_id": "U5U5U5", "split": "Training/Unlabeled", "vendor": "C", "vendor_name": "GE", "centre": 4, "ed": 0, "es": 0, "n_frames": 4, "pathology": "Other", "encoding": "float32", "annotated": (1, 3)},
        {"case_id": "U6U6U6", "split": "Training/Unlabeled", "vendor": "C", "vendor_name": "GE", "centre": 4, "ed": 0, "es": 0, "n_frames": 4, "pathology": "Other", "encoding": "float32", "annotated": (1, 3), "duplicate_of": "U5U5U5"},
    ]


def mms_refit_cases(per_vendor: int = 15) -> list[dict]:
    """:func:`mms_cases` plus ``per_vendor`` Training/Labeled cases for each of vendors A and B (NOR/DCM/HCM/IHD/Other
    cycle) and four Validation/Testing cases (vendors A-D), so the refit arm has five refit folds, at least five cases
    in three predictor classes, unlabelled (IHD, off-ontology) cases that stay in transition fitting only, and seen and
    unseen evaluation vendors.  The added cases use ES = 3 of 4 frames (physically consistent volumes)."""

    cycle = ("NOR", "DCM", "HCM", "NOR", "DCM", "HCM", "IHD", "NOR", "DCM", "HCM", "Other", "HCM")
    extra = []
    for v, (vendor, name, centre) in enumerate((("A", "Siemens", 1), ("B", "Philips", 2))):
        for i in range(per_vendor):
            extra.append({"case_id": f"{vendor}{i:02d}R{v}{i:02d}", "split": "Training/Labeled", "vendor": vendor, "vendor_name": name,
                          "centre": centre, "ed": 0, "es": 3, "n_frames": 4, "pathology": cycle[i % len(cycle)],
                          "encoding": "float32" if i % 2 else "uint8_scaled"})
    for case_id, split, vendor, name, centre, pathology in (("V0A0V0", "Validation", "A", "Siemens", 1, "NOR"),
                                                            ("V1B1V1", "Validation", "B", "Philips", 3, "DCM"),
                                                            ("T2C2T2", "Testing", "C", "GE", 4, "HCM"),
                                                            ("T3D3T3", "Testing", "D", "Canon", 5, "ARV")):
        extra.append({"case_id": case_id, "split": split, "vendor": vendor, "vendor_name": name, "centre": centre, "ed": 0, "es": 3,
                      "n_frames": 4, "pathology": pathology, "encoding": "float32"})
    return mms_cases() + extra


def write_fixture_manifests(root: Path, manifest: pd.DataFrame, metadata=None) -> Path:
    """Write a fixture root's own manifests (the stage context reads them only for SYNTHETIC_FIXTURE roots)."""

    from tmmri.ingest.sources import SYNTHETIC_MANIFEST, SYNTHETIC_METADATA_MANIFEST

    frame = manifest.drop(columns=[c for c in ("relpath",) if c in manifest.columns])
    frame.to_csv(root / SYNTHETIC_MANIFEST, index=False, lineterminator="\n")
    if metadata is not None:
        pd.DataFrame([{"relpath": e.relpath, "sha256": e.sha256, "bytes": e.bytes, "role": e.role} for e in metadata]).to_csv(
            root / SYNTHETIC_METADATA_MANIFEST, index=False, lineterminator="\n")
    return root / SYNTHETIC_MANIFEST


def make_mms_source(root: Path, cases: list[dict] | None = None, shape=(20, 18, 3)) -> Path:
    cases = mms_cases() if cases is None else cases
    affine = np.array([[1.25, 0.0, 0.0, -10.0], [0.0, 1.25, 0.0, 5.0], [0.0, 0.0, 8.0, 30.0], [0.0, 0.0, 0.0, 1.0]])
    rows = []
    for c in cases:
        key = c.get("duplicate_of") or c["case_id"]
        folder = root / c["split"] / c["case_id"]
        n = c["n_frames"]
        cine = np.zeros(shape + (n,), dtype=np.float32)
        labels = np.zeros(shape + (n,), dtype=np.uint8)
        annotated = c.get("annotated", (c["ed"], c["es"]))
        for t in range(n):
            radius = 6.0 - 2.0 * (abs(t - c["ed"]) / max(n - 1, 1))   # largest LV at ED
            proj = project_mask(shape, radius, f"{key}|{t}")
            cine[..., t] = image_from_mask(proj, f"{key}|{t}")
            if t in annotated:
                labels[..., t] = proj
        write_nifti(folder / f"{c['case_id']}_sa.nii.gz", cine, affine, dtype=np.float32)
        if c["encoding"] == "uint8_scaled":
            raw = (labels.astype(np.uint16) * 85).astype(np.uint8)   # 3/255 slope decodes 85 -> 1, 170 -> 2, 255 -> 3
            write_nifti(folder / f"{c['case_id']}_sa_gt.nii.gz", raw, affine, dtype=np.uint8, slope=3.0 / 255.0)
        else:
            write_nifti(folder / f"{c['case_id']}_sa_gt.nii.gz", labels.astype(np.float32), affine, dtype=np.float32)
        rows.append({"External code": c["case_id"], "VendorName": c["vendor_name"], "Vendor": c["vendor"],
                     "Centre": c["centre"], "ED": c["ed"], "ES": c["es"], "Age": 50, "Pathology": c["pathology"],
                     "Sex": "M", "Height": 170.0 if c["vendor"] in "AD" else None, "Weight": 70.0})
    info = pd.DataFrame(rows)
    info.index.name = None
    info.to_csv(root / MMS_INFO_CSV, index=True, lineterminator="\n")
    mark_synthetic(root)
    return root


def _file_entry(root: Path, rel: str) -> tuple[str, int]:
    data = (root / rel).read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data)


def mms_manifest(root: Path, cases: list[dict] | None = None, mms100: set[str] = frozenset({"A1A1A1"})) -> pd.DataFrame:
    cases = mms_cases() if cases is None else cases
    rows = []
    for c in cases:
        image_rel = f"{c['split']}/{c['case_id']}/{c['case_id']}_sa.nii.gz"
        mask_rel = f"{c['split']}/{c['case_id']}/{c['case_id']}_sa_gt.nii.gz"
        (ish, ib), (msh, mb) = _file_entry(root, image_rel), _file_entry(root, mask_rel)
        unlabeled = c["split"] == "Training/Unlabeled"
        rows.append({"case_id": c["case_id"], "release_split": c["split"], "image_relpath": image_rel, "image_sha256": ish,
                     "image_bytes": ib, "mask_relpath": mask_rel, "mask_sha256": msh, "mask_bytes": mb,
                     "eligible_v6": not unlabeled, "exclusion_reason": "MMS_RELEASE_UNLABELED" if unlabeled else "",
                     "mms100_member": c["case_id"] in mms100,
                     "duplicate_of": c.get("duplicate_of") or next((d["case_id"] for d in cases if d.get("duplicate_of") == c["case_id"]), "")})
    return pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)


def mms_metadata(root: Path):
    from tmmri.ingest.sources import SourceFile

    sha, size = _file_entry(root, MMS_INFO_CSV)
    return [SourceFile("info_csv", MMS_INFO_CSV, sha, size)]


def corrupt_one_byte(path: Path, offset: int = -20) -> None:
    data = bytearray(path.read_bytes())
    data[offset] ^= 0xFF
    path.write_bytes(bytes(data))
