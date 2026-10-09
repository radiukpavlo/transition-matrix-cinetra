"""Verify-and-stage (protocol CMR-TRANS-AUDIT 6.0.0, Decision D23).

:func:`stage_dataset` reads every source file of a case **exactly once**: the compressed stream is hashed with
SHA-256 while it is decompressed, the hash is checked against the tracked manifest, and the ED and ES frames are
cut out of the stream frame by frame (M&Ms 4D ``_sa.nii.gz``/``_sa_gt.nii.gz``; the released ACDC frame files).
Memory per worker is bounded by the two retained frames plus one decompression buffer; the full 4D volume is
never materialized.  The frames are written as uncompressed 3D NIfTI files

    <out_root>/<DATASET>/<case_id>/{ed,es}_{image,mask}.nii   +   case.json

with the source header geometry preserved (affine, sform/qform codes, pixdim; the ACDC unit-scaled affine keeps its
header slice thickness, so :mod:`tmmri.ingest.geometry` applies the same pixdim fallback).  Images are stored as
float32 (the decoded values), masks as the **released** label values in uint8 (after ``np.rint``; the dataset
label map is applied only when a case is loaded).  Writes are atomic (temporary file, then rename; ``case.json``
last).  A case whose tracked source hashes equal the recorded ones and whose staged files still hash to the
recorded values is skipped without reading its sources (content addressing).  Cases run in parallel with
joblib/loky; the worker count is ``min(n_jobs, source_readers)`` (``TMMRI_N_JOBS``, ``TMMRI_SOURCE_READERS``).
A failure is recorded with a registered reason code (``SOURCE_HASH_MISMATCH``, ``INPUT_DECODE_FAILURE``,
``INPUT_MISSING_ED_ES_FRAME``) and never aborts the other cases.  The per-dataset staging manifest
``<manifests>/staging_<DATASET>.json`` holds no timestamps or absolute paths, so it can key downstream stages.
Staging and hash verification of M&Ms are allowed before the protocol freeze; nothing here computes an endpoint.
M&Ms-2 (``MMS2``, protocol-v6.0.1) stages the four 3D short-axis ED/ES files of each extracted ``mms2_external``
case listed by the extraction manifest of :mod:`tmmri.ingest.mms2`, after the v6 geometry rule.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import struct
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Mapping

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.volumeutils import apply_read_scaling

from ..runs import atomic_write_bytes, atomic_write_json, parallel_map, peak_rss_bytes, resolve_n_jobs, resolve_source_readers, sha256_json
from .acdc import case_source_files, evaluate_info, parse_info_cfg
from .geometry import NiftiGeometry, geometry_from_image, serialize_geometry
from .labels import DATASET_LABEL_MAPPINGS
from .sources import (
    SourceError,
    SourceFile,
    acdc_patient_files,
    canonical_dataset,
    is_synthetic_fixture,
    load_acdc_manifest,
    load_mms_manifest,
    load_mms_metadata_manifest,
    mms_case_files,
    read_verified_bytes,
    sample_id,
    sha256_file,
)

STAGER_VERSION = "verify_and_stage_v1"
CASE_RECORD = "case.json"
PHASE_FILES = ("ed_image", "ed_mask", "es_image", "es_mask")
RAW_CHUNK = 4 * 1024 * 1024
OUT_CHUNK = 8 * 1024 * 1024
MANIFEST_SCHEMA = "tmmri.staging_manifest/1"


class StagingFailure(Exception):
    """A case-level failure with a registered reason code (``registry/reason_codes.yaml``)."""

    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = reason_code
        self.detail = detail


# ----------------------------------------------------------------------------- single-pass NIfTI stream


class _Stream:
    """Hash the raw bytes and decompress (gzip) or pass through (plain ``.nii``) in one forward pass."""

    def __init__(self, raw: BinaryIO) -> None:
        self.raw = raw
        self.hasher = hashlib.sha256()
        self.nbytes = 0
        self._buf = bytearray()
        self._pos = 0
        first = self._read_raw()
        self.gzip = first[:2] == b"\x1f\x8b"
        self._pending = first
        self._d = zlib.decompressobj(31) if self.gzip else None
        self._member_open = self.gzip

    def _read_raw(self) -> bytes:
        chunk = self.raw.read(RAW_CHUNK)
        if chunk:
            self.hasher.update(chunk)
            self.nbytes += len(chunk)
        return chunk

    def _fill(self) -> bool:
        while True:
            if not self.gzip:
                data = self._pending or self._read_raw()
                self._pending = b""
                if not data:
                    return False
                self._buf += data
                return True
            if not self._pending:
                self._pending = self._read_raw()
                if not self._pending:
                    if self._member_open:
                        raise StagingFailure("INPUT_DECODE_FAILURE", "truncated gzip stream")
                    return False
            if self._d is None:                      # after a complete member: new member or zero padding
                if self._pending.strip(b"\x00"):
                    self._d = zlib.decompressobj(31)
                    self._member_open = True
                else:
                    self._pending = b""
                continue
            out = self._d.decompress(self._pending, OUT_CHUNK)
            self._pending = self._d.unconsumed_tail
            if self._d.eof:
                self._pending = self._d.unused_data + self._pending
                self._d = None
                self._member_open = False
            if out:
                self._buf += out
                return True

    def _compact(self) -> None:
        if self._pos and (self._pos >= len(self._buf) or self._pos > OUT_CHUNK):
            del self._buf[: self._pos]
            self._pos = 0

    def read(self, n: int) -> bytes:
        while len(self._buf) - self._pos < n:
            if not self._fill():
                break
        end = min(self._pos + n, len(self._buf))
        out = bytes(memoryview(self._buf)[self._pos:end])
        self._pos = end
        self._compact()
        return out

    def skip(self, n: int) -> None:
        remaining = n
        while remaining:
            available = len(self._buf) - self._pos
            if available == 0:
                self._buf.clear()
                self._pos = 0
                if not self._fill():
                    raise StagingFailure("INPUT_DECODE_FAILURE", "truncated voxel data")
                continue
            take = min(available, remaining)
            self._pos += take
            remaining -= take
        self._compact()

    def drain(self) -> None:
        """Consume the rest of the decompressed stream (validates the gzip CRC), then the raw tail."""

        self._buf.clear()
        self._pos = 0
        while self._fill():
            self._buf.clear()
        self.drain_raw()

    def drain_raw(self) -> None:
        while self._read_raw():
            pass


@dataclass
class _NiftiRead:
    sha256: str
    nbytes: int
    header: nib.Nifti1Header | None
    frames: dict[int, np.ndarray]
    n_frames: int | None
    annotated: list[int]
    failure: StagingFailure | None


def _read_nifti_stream(path: Path, wanted: set[int], scan_annotation: bool = False) -> _NiftiRead:
    """Stream one NIfTI-1 file: hash it, keep the ``wanted`` frames (decoded), optionally list annotated frames."""

    header: nib.Nifti1Header | None = None
    frames: dict[int, np.ndarray] = {}
    annotated: list[int] = []
    n_frames: int | None = None
    failure: StagingFailure | None = None
    with path.open("rb") as handle:
        stream = _Stream(handle)
        try:
            head = stream.read(348)
            if len(head) < 348:
                raise StagingFailure("INPUT_DECODE_FAILURE", "short NIfTI header")
            if struct.unpack("<i", head[:4])[0] == 348:
                endian = "<"
            elif struct.unpack(">i", head[:4])[0] == 348:
                endian = ">"
            else:
                raise StagingFailure("INPUT_DECODE_FAILURE", "not a NIfTI-1 header")
            vox_offset = int(struct.unpack(endian + "f", head[108:112])[0])
            if vox_offset < 352:
                raise StagingFailure("INPUT_DECODE_FAILURE", f"unsupported vox_offset {vox_offset}")
            prefix = head + stream.read(vox_offset - 348)
            header = nib.Nifti1Header.from_fileobj(io.BytesIO(prefix))
            shape = tuple(int(x) for x in header.get_data_shape())
            if len(shape) < 3 or any(x != 1 for x in shape[4:]):
                raise StagingFailure("INPUT_DECODE_FAILURE", f"unsupported data shape {shape}")
            dtype = header.get_data_dtype()
            n_frames = shape[3] if len(shape) >= 4 else 1
            frame_shape = shape[:3]
            frame_bytes = int(np.prod(frame_shape)) * dtype.itemsize
            slope, inter = header.get_slope_inter()
            for index in range(n_frames):
                if index in wanted or scan_annotation:
                    block = stream.read(frame_bytes)
                    if len(block) != frame_bytes:
                        raise StagingFailure("INPUT_DECODE_FAILURE", "truncated voxel data")
                    raw = np.frombuffer(block, dtype=dtype).reshape(frame_shape, order="F")
                    decoded = apply_read_scaling(raw, slope, inter)
                    if scan_annotation and np.any(np.rint(decoded) != 0):
                        annotated.append(index)
                    if index in wanted:
                        frames[index] = decoded
                else:
                    stream.skip(frame_bytes)
            stream.drain()
        except StagingFailure as exc:
            failure = exc
            stream.drain_raw()
        except (zlib.error, EOFError, ValueError, struct.error, nib.spatialimages.HeaderDataError) as exc:
            failure = StagingFailure("INPUT_DECODE_FAILURE", f"{type(exc).__name__}: {exc}")
            stream.drain_raw()
    return _NiftiRead(stream.hasher.hexdigest(), stream.nbytes, header, frames, n_frames, annotated, failure)


# ----------------------------------------------------------------------------- staged files


def _voxel_sha256(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    hasher = hashlib.sha256(f"{arr.dtype.str}|{arr.shape}|".encode("ascii"))
    hasher.update(arr.tobytes())
    return hasher.hexdigest()


def _decoded_image(frame: np.ndarray) -> np.ndarray:
    # Non-finite voxels are staged unchanged and counted in case.json; INPUT_NONFINITE is routed downstream.
    return np.asarray(frame, dtype=np.float32)


def _decoded_mask(frame: np.ndarray) -> np.ndarray:
    values = np.rint(np.asarray(frame, dtype=np.float64))
    if not np.isfinite(values).all() or values.min(initial=0) < 0 or values.max(initial=0) > 255:
        raise StagingFailure("INPUT_DECODE_FAILURE", "mask values outside 0-255 after rounding")
    return values.astype(np.uint8)


def _nifti_bytes(array: np.ndarray, source_header: nib.Nifti1Header) -> tuple[bytes, nib.Nifti1Image]:
    header = source_header.copy()
    header.set_data_shape(array.shape)
    header.set_data_dtype(array.dtype)
    image = nib.Nifti1Image(array, header.get_best_affine(), header=header)
    data = image.to_bytes()
    return data, nib.Nifti1Image.from_bytes(data)


def _check_preserved(staged: nib.Nifti1Image, source_header: nib.Nifti1Header, array: np.ndarray, name: str) -> None:
    if not np.allclose(staged.affine, source_header.get_best_affine(), rtol=0, atol=1e-6):
        raise StagingFailure("INPUT_DECODE_FAILURE", f"{name}: affine not preserved")
    if not np.allclose(staged.header.get_zooms()[:3], source_header.get_zooms()[:3], rtol=0, atol=1e-6):
        raise StagingFailure("INPUT_DECODE_FAILURE", f"{name}: zooms not preserved")
    if not np.array_equal(np.asanyarray(staged.dataobj), array):
        raise StagingFailure("INPUT_DECODE_FAILURE", f"{name}: voxel data not preserved")


def _staged_matches(case_dir: Path, record: Mapping[str, Any]) -> bool:
    staged = record.get("staged", {})
    if set(staged) != set(PHASE_FILES):
        return False
    for entry in staged.values():
        path = case_dir / entry["file"]
        if not path.is_file() or path.stat().st_size != int(entry["bytes"]) or sha256_file(path) != entry["sha256"]:
            return False
    return True


def _read_record(case_dir: Path) -> dict[str, Any] | None:
    try:
        return json.loads((case_dir / CASE_RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ----------------------------------------------------------------------------- case workers


def _result(job: Mapping[str, Any], status: str, action: str, *, reason: str | None = None, detail: str | None = None,
            record: dict[str, Any] | None = None, sources: dict[str, Any] | None = None, started: float = 0.0) -> dict[str, Any]:
    return {
        "case_id": job["case_id"], "sample_id": sample_id(job["dataset"], job["case_id"]), "status": status,
        "action": action, "reason_code": reason, "detail": detail, "record": record, "sources": sources,
        "seconds": round(time.perf_counter() - started, 4) if started else 0.0, "peak_rss_bytes": peak_rss_bytes(),
        "source_bytes": int(sum(s["bytes"] for s in (sources or {}).values())) if sources else 0,
    }


def _verify_only(job: Mapping[str, Any], started: float) -> dict[str, Any]:
    root = Path(job["source_root"])
    problems: list[str] = []
    checked: dict[str, Any] = {}
    for role, entry in job["files"].items():
        path = root / entry["relpath"]
        if not path.is_file():
            problems.append(f"missing: {entry['relpath']}")
            continue
        digest = sha256_file(path)
        checked[role] = {**entry, "observed_sha256": digest}
        if path.stat().st_size != int(entry["bytes"]) or digest != entry["sha256"]:
            problems.append(f"sha256 mismatch: {entry['relpath']}")
    if problems:
        return _result(job, "FAILED", "verified", reason="SOURCE_HASH_MISMATCH", detail="; ".join(problems),
                       sources=checked, started=started)
    return _result(job, "VERIFIED", "verified", sources=checked, started=started)


def _stream_verified(root: Path, entry: Mapping[str, Any], wanted: set[int], scan: bool) -> _NiftiRead:
    path = root / entry["relpath"]
    if not path.is_file():
        raise StagingFailure("SOURCE_HASH_MISMATCH", f"missing: {entry['relpath']}")
    read = _read_nifti_stream(path, wanted, scan)
    if read.nbytes != int(entry["bytes"]) or read.sha256 != entry["sha256"]:
        raise StagingFailure("SOURCE_HASH_MISMATCH", f"sha256 mismatch: {entry['relpath']}")
    if read.failure is not None:
        raise read.failure
    return read


def _write_case(job: Mapping[str, Any], case_dir: Path, arrays: dict[str, np.ndarray],
                headers: dict[str, nib.Nifti1Header], base: dict[str, Any]) -> dict[str, Any]:
    staged: dict[str, Any] = {}
    blobs: dict[str, bytes] = {}
    images: dict[str, nib.Nifti1Image] = {}
    for name in PHASE_FILES:
        data, reloaded = _nifti_bytes(arrays[name], headers[name])
        _check_preserved(reloaded, headers[name], arrays[name], name)
        blobs[name], images[name] = data, reloaded
        staged[name] = {"file": f"{name}.nii", "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                        "dtype": str(arrays[name].dtype), "voxel_sha256": _voxel_sha256(arrays[name])}
    geometry = geometry_from_image(images["ed_mask"])
    image_geometry = geometry_from_image(images["ed_image"])
    record = {
        **base,
        "staged": staged,
        "geometry": serialize_geometry(geometry),
        "image_geometry_matches_mask": bool(
            np.allclose(image_geometry.affine_matrix, geometry.affine_matrix, atol=1e-5)
            and image_geometry.image_shape == geometry.image_shape
            and np.isclose(image_geometry.slice_increment_mm or 0.0, geometry.slice_increment_mm or 0.0)),
        "mask_values_released": sorted({int(v) for name in ("ed_mask", "es_mask") for v in np.unique(arrays[name])}),
        "nonfinite_image_voxels": int(sum(int((~np.isfinite(arrays[n])).sum()) for n in ("ed_image", "es_image"))),
    }
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / CASE_RECORD).unlink(missing_ok=True)
    for name in PHASE_FILES:
        atomic_write_bytes(case_dir / f"{name}.nii", blobs[name])
    atomic_write_json(case_dir / CASE_RECORD, record)
    return record


def _base_record(job: Mapping[str, Any], sources: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "tmmri.staged_case/1", "stager_version": STAGER_VERSION, "dataset": job["dataset"],
        "case_id": job["case_id"], "sample_id": sample_id(job["dataset"], job["case_id"]),
        "release_split": job.get("release_split"), "status": "STAGED", "sources": sources,
        "synthetic_fixture": bool(job.get("synthetic_fixture", False)),
    }


def _stage_acdc(job: Mapping[str, Any], started: float) -> dict[str, Any]:
    root, case_dir = Path(job["source_root"]), Path(job["case_dir"])
    files = {name: SourceFile("", e["relpath"], e["sha256"], int(e["bytes"])) for name, e in job["files"].items()}
    if "Info.cfg" not in files:
        raise StagingFailure("INPUT_DECODE_FAILURE", "Info.cfg not in the tracked manifest")
    prior = _read_record(case_dir)
    if not job.get("force") and prior and prior.get("status") == "STAGED" and prior.get("stager_version") == STAGER_VERSION:
        expected = {role: files.get(Path(src["relpath"]).name) for role, src in prior.get("sources", {}).items()}
        if all(sf is not None and sf.sha256 == prior["sources"][role]["sha256"] for role, sf in expected.items()) \
                and prior["sources"].get("info_cfg", {}).get("sha256") == files["Info.cfg"].sha256 \
                and _staged_matches(case_dir, prior):
            return _result(job, "STAGED", "skipped_identical", record=prior, started=started)
    try:
        text = read_verified_bytes(root, files["Info.cfg"]).decode("utf-8", errors="replace")
    except SourceError as exc:
        raise StagingFailure("SOURCE_HASH_MISMATCH", str(exc)) from exc
    info = evaluate_info(parse_info_cfg(text))
    if info["reason"]:
        raise StagingFailure(str(info["reason"]), "Info.cfg ED/ES invalid")
    try:
        roles = case_source_files(job["case_id"], files, int(info["ed_frame_number"]), int(info["es_frame_number"]))
    except KeyError as exc:
        raise StagingFailure("INPUT_MISSING_ED_ES_FRAME", f"frame file not in the tracked manifest: {exc}") from exc
    sources = {role: sf.as_dict() for role, sf in roles.items()}
    arrays: dict[str, np.ndarray] = {}
    headers: dict[str, nib.Nifti1Header] = {}
    for name in PHASE_FILES:
        read = _stream_verified(root, sources[name], {0}, False)
        if read.n_frames != 1:
            raise StagingFailure("INPUT_DECODE_FAILURE", f"{name}: expected a 3D frame file")
        arrays[name] = _decoded_mask(read.frames[0]) if name.endswith("mask") else _decoded_image(read.frames[0])
        headers[name] = read.header
    for phase in ("ed", "es"):
        if arrays[f"{phase}_image"].shape != arrays[f"{phase}_mask"].shape:
            raise StagingFailure("INPUT_DECODE_FAILURE", f"{phase}: image and mask shapes differ")
    base = _base_record(job, sources)
    base.update({
        "ed_frame_index": int(info["ed_frame_index"]), "es_frame_index": int(info["es_frame_index"]),
        "frame_numbers_1based": {"ed": int(info["ed_frame_number"]), "es": int(info["es_frame_number"])},
        "n_frames": info["n_frames"], "mask_annotated_frames": None, "annotation_matches_metadata": None,
    })
    return _result(job, "STAGED", "written", record=_write_case(job, case_dir, arrays, headers, base),
                   sources=sources, started=started)


def _stage_mms(job: Mapping[str, Any], started: float) -> dict[str, Any]:
    root, case_dir = Path(job["source_root"]), Path(job["case_dir"])
    sources = {role: dict(entry) for role, entry in job["files"].items()}
    if job.get("verify_only"):
        return _verify_only(job, started)
    ed, es = int(job["ed_frame_index"]), int(job["es_frame_index"])
    prior = _read_record(case_dir)
    if not job.get("force") and prior and prior.get("status") == "STAGED" and prior.get("stager_version") == STAGER_VERSION \
            and {r: s["sha256"] for r, s in prior.get("sources", {}).items()} == {r: s["sha256"] for r, s in sources.items()} \
            and (prior.get("ed_frame_index"), prior.get("es_frame_index")) == (ed, es) and _staged_matches(case_dir, prior):
        return _result(job, "STAGED", "skipped_identical", record=prior, started=started)
    if ed == es or ed < 0 or es < 0:
        raise StagingFailure("INPUT_MISSING_ED_ES_FRAME", f"invalid ED/ES {ed}/{es}")
    image = _stream_verified(root, sources["image"], {ed, es}, False)
    mask = _stream_verified(root, sources["mask"], {ed, es}, True)
    if image.n_frames != mask.n_frames or image.header.get_data_shape()[:3] != mask.header.get_data_shape()[:3]:
        raise StagingFailure("INPUT_DECODE_FAILURE", "image and mask shapes differ")
    if ed >= image.n_frames or es >= image.n_frames:
        raise StagingFailure("INPUT_MISSING_ED_ES_FRAME", f"ED/ES {ed}/{es} outside {image.n_frames} frames")
    if ed not in mask.annotated or es not in mask.annotated:
        raise StagingFailure("INPUT_MISSING_ED_ES_FRAME", f"mask not annotated on ED/ES {ed}/{es}; annotated {mask.annotated}")
    arrays = {"ed_image": _decoded_image(image.frames[ed]), "es_image": _decoded_image(image.frames[es]),
              "ed_mask": _decoded_mask(mask.frames[ed]), "es_mask": _decoded_mask(mask.frames[es])}
    headers = {"ed_image": image.header, "es_image": image.header, "ed_mask": mask.header, "es_mask": mask.header}
    base = _base_record(job, sources)
    base.update({
        "ed_frame_index": ed, "es_frame_index": es, "frame_numbers_1based": None, "n_frames": int(image.n_frames),
        "mask_annotated_frames": [int(f) for f in mask.annotated],
        "annotation_matches_metadata": sorted(mask.annotated) == sorted({ed, es}),
        "source_dtypes": {"image": str(image.header.get_data_dtype()), "mask": str(mask.header.get_data_dtype())},
    })
    return _result(job, "STAGED", "written", record=_write_case(job, case_dir, arrays, headers, base),
                   sources=sources, started=started)


def _stage_mms2(job: Mapping[str, Any], started: float) -> dict[str, Any]:
    """M&Ms-2 (protocol-v6.0.1): four 3D short-axis files per case (phases from the file names; no frame index), the
    v6 geometry rule of :func:`tmmri.ingest.mms2.check_geometry` (``INPUT_GEOMETRY_INCONSISTENT`` otherwise)."""

    from .mms2 import GeometryInconsistent, check_geometry

    root, case_dir = Path(job["source_root"]), Path(job["case_dir"])
    sources = {role: dict(entry) for role, entry in job["files"].items()}
    if job.get("verify_only"):
        return _verify_only(job, started)
    if set(sources) != set(PHASE_FILES):
        raise StagingFailure("INPUT_MISSING_ED_ES_FRAME", f"short-axis ED/ES inputs listed: {sorted(sources)}")
    prior = _read_record(case_dir)
    if not job.get("force") and prior and prior.get("status") == "STAGED" and prior.get("stager_version") == STAGER_VERSION \
            and {r: s["sha256"] for r, s in prior.get("sources", {}).items()} == {r: s["sha256"] for r, s in sources.items()} \
            and _staged_matches(case_dir, prior):
        return _result(job, "STAGED", "skipped_identical", record=prior, started=started)
    arrays: dict[str, np.ndarray] = {}
    headers: dict[str, nib.Nifti1Header] = {}
    for name in PHASE_FILES:
        read = _stream_verified(root, sources[name], {0}, False)
        if read.n_frames != 1:
            raise StagingFailure("INPUT_DECODE_FAILURE", f"{name}: expected a 3D short-axis file, found {read.n_frames} frames")
        arrays[name] = _decoded_mask(read.frames[0]) if name.endswith("mask") else _decoded_image(read.frames[0])
        headers[name] = read.header
    try:
        geometry_check = check_geometry(headers)
    except GeometryInconsistent as exc:
        raise StagingFailure(exc.reason_code, str(exc)) from exc
    base = _base_record(job, sources)
    base.update({
        "ed_frame_index": None, "es_frame_index": None, "phase_source": "file_name", "frame_numbers_1based": None, "n_frames": None,
        "mask_annotated_frames": None, "annotation_matches_metadata": None, "geometry_check": geometry_check,
        "source_dtypes": {name: str(headers[name].get_data_dtype()) for name in PHASE_FILES},
    })
    return _result(job, "STAGED", "written", record=_write_case(job, case_dir, arrays, headers, base),
                   sources=sources, started=started)


def _stage_case(job: Mapping[str, Any]) -> dict[str, Any]:
    """Worker entry point (module-level for pickling); never raises for a case-level problem."""

    started = time.perf_counter()
    try:
        if job["dataset"] == "ACDC":
            return _stage_acdc(job, started)
        if job["dataset"] == "MMS2":
            return _stage_mms2(job, started)
        return _stage_mms(job, started)
    except StagingFailure as exc:
        return _result(job, "FAILED", "failed", reason=exc.reason_code, detail=exc.detail, started=started)
    except (OSError, MemoryError, ValueError, KeyError) as exc:
        return _result(job, "FAILED", "failed", reason="INPUT_DECODE_FAILURE", detail=f"{type(exc).__name__}: {exc}",
                       started=started)


# ----------------------------------------------------------------------------- dataset driver


@dataclass
class StagingReport:
    dataset: str
    source_root: str
    out_root: str
    manifest_path: str | None
    verify_only: bool
    synthetic_fixture: bool
    n_workers: int
    attempted: int
    actions: dict[str, int]
    failures: list[dict[str, Any]]
    excluded: list[dict[str, Any]]
    seconds: float
    source_bytes_read: int
    peak_worker_rss_bytes: int | None
    case_seconds: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict[str, Any]:
        out = dict(self.__dict__)
        out["ok"] = self.ok
        return out


def _acdc_jobs(manifest: pd.DataFrame, cases: set[str] | None, verify_only: bool) -> list[dict[str, Any]]:
    jobs = []
    patients = acdc_patient_files(manifest)
    for patient_id, files in patients.items():
        if cases is not None and patient_id not in cases:
            continue
        split = next(iter(files.values())).relpath.split("/")[0]
        jobs.append({"case_id": patient_id, "release_split": split,
                     "files": {name: {"relpath": sf.relpath, "sha256": sf.sha256, "bytes": sf.bytes} for name, sf in files.items()}})
    if verify_only and cases is None:
        in_patients = {sf.relpath for files in patients.values() for sf in files.values()}
        release = {str(r.relpath): {"relpath": str(r.relpath), "sha256": str(r.sha256), "bytes": int(r.size_bytes)}
                   for r in manifest.itertuples(index=False) if str(r.relpath) not in in_patients}
        if release:
            jobs.append({"case_id": "_release_files", "release_split": None, "files": release})
    return jobs


def _mms_jobs(manifest: pd.DataFrame, cohort: pd.DataFrame | None, cases: set[str] | None, verify_only: bool):
    jobs, excluded = [], []
    frames = {} if cohort is None else {r.case_id: r for r in cohort.itertuples(index=False)}
    for row in manifest.itertuples(index=False):
        if cases is not None and row.case_id not in cases:
            continue
        files = {sf.role: {"relpath": sf.relpath, "sha256": sf.sha256, "bytes": sf.bytes} for sf in mms_case_files(row._asdict())}
        entry = {"case_id": row.case_id, "release_split": row.release_split, "files": files}
        if verify_only:
            jobs.append(entry)
            continue
        info = frames.get(row.case_id)
        eligible = bool(row.eligible_v6) and info is not None and bool(info.eligible_v6)
        if not eligible:
            reason = (info.exclusion_reason if info is not None else None) or row.exclusion_reason or "EXCLUDED_UPSTREAM"
            excluded.append({"case_id": row.case_id, "reason_code": reason})
            continue
        entry.update({"ed_frame_index": int(info.ed_frame_index), "es_frame_index": int(info.es_frame_index)})
        jobs.append(entry)
    return jobs, excluded


def _mms2_jobs(manifest: pd.DataFrame, cohort: pd.DataFrame | None, cases: set[str] | None, verify_only: bool):
    """Jobs of the extracted ``mms2_external`` cases (the file manifest never lists a sealed subject)."""

    from .mms2 import case_files

    jobs, excluded = [], []
    frames = {} if cohort is None else {r.case_id: r for r in cohort.itertuples(index=False)}
    for case_id, files in case_files(manifest).items():
        if cases is not None and case_id not in cases:
            continue
        entry = {"case_id": case_id, "release_split": None, "files": files}
        if verify_only:
            jobs.append(entry)
            continue
        info = frames.get(case_id)
        if info is None or not bool(info.eligible_v6):
            excluded.append({"case_id": case_id, "reason_code": (info.exclusion_reason if info is not None else None) or "EXCLUDED_UPSTREAM"})
            continue
        jobs.append(entry)
    return jobs, excluded


def manifest_path_for(manifests_dir: str | Path, dataset: str, verify_only: bool = False) -> Path:
    stem = "verification" if verify_only else "staging"
    return Path(manifests_dir) / f"{stem}_{canonical_dataset(dataset)}.json"


def load_staging_manifest(manifests_dir: str | Path, dataset: str) -> dict[str, Any] | None:
    path = manifest_path_for(manifests_dir, dataset)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def stage_dataset(
    dataset: str,
    source_root: str | Path,
    out_root: str | Path,
    manifest: pd.DataFrame | None = None,
    n_jobs: int | None = None,
    verify_only: bool = False,
    *,
    cases: list[str] | set[str] | None = None,
    force: bool = False,
    source_readers: int | None = None,
    manifests_dir: str | Path | None = None,
    cohort: pd.DataFrame | None = None,
    source_manifest_sha256: str | None = None,
    metadata: list[SourceFile] | None = None,
) -> StagingReport:
    """Verify and stage one dataset (see the module docstring); returns a :class:`StagingReport`.

    ``manifest`` is the tracked source manifest (default: the registry file of the dataset).  For M&Ms the ED/ES
    indices and the eligibility come from ``cohort`` (default: :func:`tmmri.ingest.mms.mms_cohort_table`, which
    reads and verifies the mounted information CSV).  ``out_root`` is the staging root (``DATA_ROOT/staged``);
    ``manifests_dir`` defaults to ``<out_root>/../manifests``.
    """

    ds = canonical_dataset(dataset)
    root = Path(source_root).resolve()
    out = Path(out_root).resolve()
    mdir = Path(manifests_dir).resolve() if manifests_dir is not None else out.parent / "manifests"
    if not root.is_dir():
        raise SourceError(f"{ds} source root not found: {root}")
    started = time.perf_counter()
    synthetic = is_synthetic_fixture(root)
    wanted = set(cases) if cases is not None else None
    excluded: list[dict[str, Any]] = []
    if ds == "ACDC":
        manifest = load_acdc_manifest() if manifest is None else manifest
        jobs = _acdc_jobs(manifest, wanted, verify_only)
    elif ds == "MMS2":
        from .mms2 import INFO_CASE, default_cohort_config, info_entry, load_file_manifest, mms2_cohort_table

        manifest = load_file_manifest(root) if manifest is None else manifest
        if verify_only:
            jobs, excluded = _mms2_jobs(manifest, None, wanted, True)
            if wanted is None:
                jobs.append({"case_id": INFO_CASE, "release_split": None, "files": {"info_csv": info_entry(manifest).as_dict()}})
        else:
            if cohort is None:
                cohort = mms2_cohort_table(root, manifest, cohort_cfg=default_cohort_config())
            jobs, excluded = _mms2_jobs(manifest, cohort, wanted, False)
    else:
        manifest = load_mms_manifest() if manifest is None else manifest
        if verify_only:
            jobs, excluded = _mms_jobs(manifest, None, wanted, True)
            entries = load_mms_metadata_manifest() if metadata is None else metadata
            if wanted is None:
                jobs.append({"case_id": "_release_metadata", "release_split": None,
                             "files": {e.role: {"relpath": e.relpath, "sha256": e.sha256, "bytes": e.bytes} for e in entries}})
        else:
            if cohort is None:
                from .mms import mms_cohort_table

                cohort = mms_cohort_table(root, manifest, metadata=metadata)
            jobs, excluded = _mms_jobs(manifest, cohort, wanted, False)
    for job in jobs:
        job.update({"dataset": ds, "source_root": str(root), "case_dir": str(out / ds / job["case_id"]),
                    "verify_only": verify_only, "force": force, "synthetic_fixture": synthetic})
    workers = max(1, min(resolve_n_jobs(n_jobs), resolve_source_readers(source_readers), max(len(jobs), 1)))
    results = parallel_map(_stage_case_dispatch, jobs, n_jobs=workers)
    results.sort(key=lambda r: r["case_id"])
    actions: dict[str, int] = {}
    for r in results:
        actions[r["action"]] = actions.get(r["action"], 0) + 1
    failures = [{"case_id": r["case_id"], "reason_code": r["reason_code"], "detail": r["detail"]} for r in results if r["status"] == "FAILED"]
    peaks = [r["peak_rss_bytes"] for r in results if r.get("peak_rss_bytes")]
    manifest_file: Path | None = None
    if verify_only:
        manifest_file = manifest_path_for(mdir, ds, verify_only=True)
        atomic_write_json(manifest_file, {
            "schema": "tmmri.verification_manifest/1", "dataset": ds, "synthetic_fixture": synthetic,
            "source_manifest_sha256": source_manifest_sha256,
            "cases": {r["case_id"]: {"status": r["status"], "reason_code": r["reason_code"], "detail": r["detail"],
                                     "sources": r["sources"]} for r in results},
        })
    else:
        manifest_file = _write_staging_manifest(mdir, ds, results, excluded, synthetic, source_manifest_sha256)
    report = StagingReport(
        dataset=ds, source_root=str(root), out_root=str(out), manifest_path=str(manifest_file) if manifest_file else None,
        verify_only=verify_only, synthetic_fixture=synthetic, n_workers=workers, attempted=len(results), actions=actions,
        failures=failures, excluded=excluded, seconds=round(time.perf_counter() - started, 3),
        source_bytes_read=int(sum(r.get("source_bytes", 0) for r in results)), peak_worker_rss_bytes=max(peaks) if peaks else None,
        case_seconds={r["case_id"]: r["seconds"] for r in results},
    )
    mdir.mkdir(parents=True, exist_ok=True)
    with (mdir / f"staging_{ds}.log.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        summary = {k: v for k, v in report.as_dict().items() if k != "case_seconds"}
        handle.write(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **summary}, sort_keys=True, default=str) + "\n")
    return report


def _stage_case_dispatch(job: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    if job.get("verify_only") and job["dataset"] == "ACDC":
        return _verify_only(job, started)
    return _stage_case(job)


def _write_staging_manifest(mdir: Path, ds: str, results: list[dict[str, Any]], excluded: list[dict[str, Any]],
                            synthetic: bool, source_manifest_sha256: str | None) -> Path:
    path = manifest_path_for(mdir, ds)
    existing = load_staging_manifest(mdir, ds) or {}
    if existing.get("stager_version") != STAGER_VERSION or existing.get("synthetic_fixture") != synthetic:
        existing = {}
    cases = dict(existing.get("cases", {}))
    failures = dict(existing.get("failures", {}))
    for r in results:
        if r["status"] == "STAGED":
            cases[r["case_id"]] = r["record"]
            failures.pop(r["case_id"], None)
        else:
            cases.pop(r["case_id"], None)
            failures[r["case_id"]] = {"reason_code": r["reason_code"], "detail": r["detail"]}
    excluded_map = dict(existing.get("excluded", {}))
    excluded_map.update({e["case_id"]: e["reason_code"] for e in excluded})
    by_source = {src["sha256"]: f"{cid}/{role}" for cid, rec in sorted(cases.items()) for role, src in sorted(rec["sources"].items())}
    payload = {
        "schema": MANIFEST_SCHEMA, "dataset": ds, "stager_version": STAGER_VERSION, "synthetic_fixture": synthetic,
        "source_manifest_sha256": source_manifest_sha256 or existing.get("source_manifest_sha256"),
        "counts": {"staged": len(cases), "failed": len(failures), "excluded": len(excluded_map)},
        "cases": dict(sorted(cases.items())), "failures": dict(sorted(failures.items())),
        "excluded": dict(sorted(excluded_map.items())), "by_source_sha256": dict(sorted(by_source.items())),
    }
    payload["content_sha256"] = sha256_json({k: v for k, v in payload.items() if k != "content_sha256"})
    return atomic_write_json(path, payload)


# ----------------------------------------------------------------------------- loading staged cases


@dataclass
class StagedCase:
    """One staged case, loaded for B and A computation (project labels 1 LV, 2 MYO, 3 RV)."""

    sample_id: str
    dataset: str
    case_id: str
    ed_image: np.ndarray            # float32
    es_image: np.ndarray            # float32
    ed_mask: np.ndarray             # int16, canonical labels after the dataset map
    es_mask: np.ndarray             # int16
    spacing_mm: tuple[float, float, float | None]   # (dx, dy, dz); dz = slice increment (header-pixdim fallback)
    geometry_method: str            # z_spacing_method of tmmri.ingest.geometry
    affine: np.ndarray
    geometry: NiftiGeometry
    ed_frame_index: int | None      # None for M&Ms-2 (separate ED and ES files; phases from the file names)
    es_frame_index: int | None
    meta: dict[str, Any]


def _label_lut(label_map: Mapping[int, int]) -> np.ndarray:
    lut = np.arange(256, dtype=np.int16)
    for source, target in label_map.items():
        lut[int(source)] = int(target)
    return lut


def load_staged_case(dataset: str, case_id: str, root: str | Path, *, label_map: Mapping[int, int] | None = None,
                     verify: bool = False) -> StagedCase:
    """Load ``<root>/<DATASET>/<case_id>`` (``root`` = staging root).  ``verify`` re-hashes the staged files.

    Released mask values are rounded (they are stored rounded) and mapped with the dataset label map (ACDC 1<->3
    swap, M&Ms identity; ``configs/cohorts.yaml`` ``label_map``).  Values outside the map are kept unchanged and
    listed in ``meta["unexpected_labels"]`` so QC can flag them (``unexpected_label_value``).
    """

    ds = canonical_dataset(dataset)
    case_dir = Path(root) / ds / str(case_id)
    record = _read_record(case_dir)
    if record is None or record.get("status") != "STAGED":
        raise FileNotFoundError(f"no staged case at {case_dir}")
    if verify and not _staged_matches(case_dir, record):
        raise SourceError(f"staged files of {ds}:{case_id} differ from case.json")
    mapping = dict(label_map) if label_map is not None else DATASET_LABEL_MAPPINGS[ds]
    lut = _label_lut(mapping)
    images = {name: nib.load(str(case_dir / f"{name}.nii")) for name in PHASE_FILES}
    arrays: dict[str, np.ndarray] = {}
    unexpected: set[int] = set()
    for name, img in images.items():
        if name.endswith("image"):
            arrays[name] = np.asarray(img.dataobj, dtype=np.float32)
        else:
            released = np.rint(np.asanyarray(img.dataobj)).astype(np.int64)
            unexpected |= {int(v) for v in np.unique(released)} - {int(k) for k in mapping}
            clipped = np.clip(released, 0, 255)
            mapped = lut[clipped]
            outside = released != clipped
            if outside.any():
                mapped = np.where(outside, released, mapped)
            arrays[name] = mapped.astype(np.int16)
    geometry = geometry_from_image(images["ed_mask"])
    meta = dict(record)
    meta["unexpected_labels"] = sorted(unexpected)
    meta["label_map"] = {int(k): int(v) for k, v in mapping.items()}
    return StagedCase(
        sample_id=str(record["sample_id"]), dataset=ds, case_id=str(case_id),
        ed_image=arrays["ed_image"], es_image=arrays["es_image"], ed_mask=arrays["ed_mask"], es_mask=arrays["es_mask"],
        spacing_mm=(float(geometry.spacing_x_mm), float(geometry.spacing_y_mm),
                    None if geometry.slice_increment_mm is None else float(geometry.slice_increment_mm)),
        geometry_method=geometry.z_spacing_method, affine=np.asarray(images["ed_mask"].affine, dtype=float),
        geometry=geometry, ed_frame_index=None if record.get("ed_frame_index") is None else int(record["ed_frame_index"]),
        es_frame_index=None if record.get("es_frame_index") is None else int(record["es_frame_index"]), meta=meta,
    )


def staged_case_ids(root: str | Path, dataset: str) -> list[str]:
    base = Path(root) / canonical_dataset(dataset)
    return sorted(p.parent.name for p in base.glob(f"*/{CASE_RECORD}")) if base.is_dir() else []
