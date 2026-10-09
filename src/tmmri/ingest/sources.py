"""Tracked source manifests and source-root resolution (protocol CMR-TRANS-AUDIT 6.0.0, Decisions D2 and D23).

The tracked manifests in ``registry/sources`` carry identifiers, release split, sizes and SHA-256 only:

* ACDC: ``acdc_release_file_manifest.csv`` (1,056 release files; ``relative_path`` below ``data/raw/acdc``, i.e.
  ``database/<split>/<patient>/<file>``; the source mount ``/sources/acdc`` is the ``database`` folder).
* M&Ms: ``mms_full_release_manifest.csv`` (345 cases, image and mask per case) and
  ``mms_full_release_metadata_manifest.csv`` (the information CSV); paths are relative to the release root
  (``/sources/mms``).

Source roots are resolved in this order: an explicit path, ``TMMRI_SOURCES_ROOT/<cohort>`` when it exists (the
container binds), the host variable ``TMMRI_<COHORT>_SOURCE``, then ``mount.host_default`` of
``configs/source_bindings/<cohort>.json``.  Sources are always read-only.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from ..paths import PROJECT_ROOT, SOURCE_REGISTRY_DIR, SOURCES_ROOT

DATASET_IDS = {"acdc": "ACDC", "mms": "MMS", "mms2": "MMS2"}   # mms2: M&Ms-2 (protocol-v6.0.1 amendment; ingest.mms2)
LEGACY_ALIASES = {"MMS_100": "MMS"}
SYNTHETIC_MARKER = "SYNTHETIC_FIXTURE"
CHUNK_BYTES = 4 * 1024 * 1024

ACDC_RELEASE_MANIFEST = SOURCE_REGISTRY_DIR / "acdc_release_file_manifest.csv"
MMS_FULL_MANIFEST = SOURCE_REGISTRY_DIR / "mms_full_release_manifest.csv"
MMS_METADATA_MANIFEST = SOURCE_REGISTRY_DIR / "mms_full_release_metadata_manifest.csv"
MMS_100_MANIFEST = SOURCE_REGISTRY_DIR / "mms_100_source_normalization_manifest.csv"
MMS_INFO_CSV = "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"
ACDC_ROOT_PREFIX = "database/"

_BOOL_COLUMNS = ("eligible_v6", "mms100_member")


class SourceError(RuntimeError):
    """A source root, manifest or file is unusable (never silently worked around)."""


@dataclass(frozen=True)
class SourceFile:
    """One tracked source file: its role in a case, path relative to the source root, size and SHA-256."""

    role: str
    relpath: str
    sha256: str
    bytes: int

    def as_dict(self) -> dict[str, object]:
        return {"role": self.role, "relpath": self.relpath, "sha256": self.sha256, "bytes": int(self.bytes)}


def canonical_dataset(name: str) -> str:
    """Map ``acdc``/``ACDC``/``mms``/``MMS``/``MMS_100`` to the protocol-6.0.0 dataset id (``ACDC`` or ``MMS``), and
    ``mms2``/``MMS2`` to the M&Ms-2 id of the protocol-v6.0.1 amendment (``MMS2``)."""

    text = str(name).strip()
    if text in LEGACY_ALIASES:
        return LEGACY_ALIASES[text]
    upper = text.upper()
    if upper in DATASET_IDS.values():
        return upper
    lower = text.lower()
    if lower in DATASET_IDS:
        return DATASET_IDS[lower]
    raise ValueError(f"unknown dataset {name!r}; expected one of acdc, mms, mms2 (legacy alias MMS_100)")


def cohort_key(dataset: str) -> str:
    """``ACDC`` -> ``acdc``, ``MMS`` -> ``mms``, ``MMS2`` -> ``mms2`` (config and source-binding key)."""

    return {v: k for k, v in DATASET_IDS.items()}[canonical_dataset(dataset)]


def sample_id(dataset: str, case_id: str) -> str:
    return f"{canonical_dataset(dataset)}:{case_id}"


def source_binding(dataset: str, root: Path = PROJECT_ROOT) -> dict:
    path = Path(root) / "configs" / "source_bindings" / f"{cohort_key(dataset)}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_source_root(dataset: str, explicit: str | Path | None = None) -> Path:
    """Resolve the read-only source root of a dataset (see the module docstring for the order)."""

    key = cohort_key(dataset)
    if explicit:
        return Path(explicit).resolve()
    mounted = SOURCES_ROOT / key
    if mounted.is_dir():
        return mounted.resolve()
    env = os.environ.get(f"TMMRI_{key.upper()}_SOURCE", "").strip()
    value = env or source_binding(dataset)["mount"]["host_default"]
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def is_synthetic_fixture(root: str | Path) -> bool:
    """True when the source root carries the ``SYNTHETIC_FIXTURE`` marker file (tests and benchmarks only)."""

    return (Path(root) / SYNTHETIC_MARKER).is_file()


SYNTHETIC_MANIFEST = f"{SYNTHETIC_MARKER}.manifest.csv"             # case/file manifest of a synthetic fixture root
SYNTHETIC_METADATA_MANIFEST = f"{SYNTHETIC_MARKER}.metadata.csv"    # M&Ms information-CSV manifest of a fixture root


def synthetic_manifest_path(root: str | Path | None, *, metadata: bool = False) -> Path | None:
    """The self-describing manifest of a synthetic fixture root (never used for a root without the marker)."""

    if root is None or not is_synthetic_fixture(root):
        return None
    path = Path(root) / (SYNTHETIC_METADATA_MANIFEST if metadata else SYNTHETIC_MANIFEST)
    return path if path.is_file() else None


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class HashingReader:
    """File wrapper that hashes and counts every byte read (single pass over a source)."""

    def __init__(self, raw: BinaryIO) -> None:
        self.raw = raw
        self.hasher = hashlib.sha256()
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self.raw.read(size)
        self.hasher.update(chunk)
        self.size += len(chunk)
        return chunk

    def hexdigest(self) -> str:
        return self.hasher.hexdigest()


def _strip_acdc_prefix(relpath: str) -> str:
    text = str(relpath).replace("\\", "/")
    return text[len(ACDC_ROOT_PREFIX):] if text.startswith(ACDC_ROOT_PREFIX) else text


def load_acdc_manifest(path: str | Path = ACDC_RELEASE_MANIFEST) -> pd.DataFrame:
    """ACDC release manifest with ``relpath`` relative to the source mount (the ``database`` folder)."""

    frame = pd.read_csv(path, dtype={"relative_path": str, "sha256": str})
    if not {"relative_path", "size_bytes", "sha256"}.issubset(frame.columns):
        raise SourceError(f"{path}: expected columns relative_path, size_bytes, sha256")
    frame["relpath"] = frame["relative_path"].map(_strip_acdc_prefix)
    frame["sha256"] = frame["sha256"].str.strip().str.lower()
    frame["size_bytes"] = frame["size_bytes"].astype("int64")
    return frame.sort_values("relpath").reset_index(drop=True)


def load_mms_manifest(path: str | Path = MMS_FULL_MANIFEST) -> pd.DataFrame:
    """Full-release case manifest (one row per case) with typed boolean columns, sorted by ``case_id``."""

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"case_id", "release_split", "image_relpath", "image_sha256", "image_bytes", "mask_relpath",
                "mask_sha256", "mask_bytes", "eligible_v6", "exclusion_reason", "mms100_member", "duplicate_of"}
    missing = required.difference(frame.columns)
    if missing:
        raise SourceError(f"{path}: missing columns {sorted(missing)}")
    for column in _BOOL_COLUMNS:
        frame[column] = frame[column].str.lower().map({"true": True, "false": False}).astype(bool)
    for column in ("image_bytes", "mask_bytes"):
        frame[column] = frame[column].astype("int64")
    for column in ("image_sha256", "mask_sha256"):
        frame[column] = frame[column].str.strip().str.lower()
    if frame["case_id"].duplicated().any():
        raise SourceError(f"{path}: duplicate case_id")
    return frame.sort_values("case_id").reset_index(drop=True)


def load_mms_metadata_manifest(path: str | Path = MMS_METADATA_MANIFEST) -> list[SourceFile]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return [SourceFile(str(r["role"]), str(r["relpath"]), str(r["sha256"]).strip().lower(), int(r["bytes"]))
            for _, r in frame.iterrows()]


def mms100_case_ids(path: str | Path = MMS_100_MANIFEST) -> set[str]:
    return set(pd.read_csv(path, dtype=str)["case_id"].astype(str))


def acdc_patient_files(manifest: pd.DataFrame) -> dict[str, dict[str, SourceFile]]:
    """Group the ACDC manifest by patient: ``{patient_id: {file name: SourceFile}}`` (roles filled later)."""

    out: dict[str, dict[str, SourceFile]] = {}
    for row in manifest.itertuples(index=False):
        parts = str(row.relpath).split("/")
        if len(parts) != 3 or parts[0] not in {"training", "testing"} or not parts[1].startswith("patient"):
            continue
        out.setdefault(parts[1], {})[parts[2]] = SourceFile("", str(row.relpath), str(row.sha256), int(row.size_bytes))
    return dict(sorted(out.items()))


def mms_case_files(row: pd.Series | dict) -> list[SourceFile]:
    return [
        SourceFile("image", str(row["image_relpath"]), str(row["image_sha256"]), int(row["image_bytes"])),
        SourceFile("mask", str(row["mask_relpath"]), str(row["mask_sha256"]), int(row["mask_bytes"])),
    ]


def verify_source_file(root: str | Path, expected: SourceFile) -> str | None:
    """Return ``None`` when the file matches size and SHA-256, else a short problem string."""

    path = Path(root) / expected.relpath
    if not path.is_file():
        return f"missing: {expected.relpath}"
    size = path.stat().st_size
    if size != expected.bytes:
        return f"size {size} != {expected.bytes}: {expected.relpath}"
    if sha256_file(path) != expected.sha256:
        return f"sha256 mismatch: {expected.relpath}"
    return None


def read_verified_bytes(root: str | Path, expected: SourceFile) -> bytes:
    """Read a small source file once, verify size and SHA-256, return its bytes (raises :class:`SourceError`)."""

    path = Path(root) / expected.relpath
    if not path.is_file():
        raise SourceError(f"missing: {expected.relpath}")
    data = path.read_bytes()
    if len(data) != expected.bytes or sha256_bytes(data) != expected.sha256:
        raise SourceError(f"sha256 mismatch: {expected.relpath}")
    return data
