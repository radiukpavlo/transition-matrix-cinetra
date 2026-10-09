"""Image input path of the protocol-v6.0.1 neural branches: the tensor cache (workflow contract C3).

NEURAL-TM-1.0 Section 4.1 and ``configs/neural_input.yaml``.  For each case the staged ED and ES image frames
(``<staged_root>/<DATASET>/<case_id>/{ed,es}_image.nii``; masks are never read) are

1. checked: the geometry comes from the *image* header through :func:`tmmri.ingest.geometry.geometry_from_image`
   (the ACDC unit-scaled affine falls back to the header pixdim, the slice-increment correction of 8ed5a40).
   Non-positive or non-finite spacing, an unavailable increment, non-orthogonal or sheared voxel axes, an affine whose
   in-plane scale disagrees with pixdim, or ED/ES geometry that differ reject the case (``INPUT_GEOMETRY_INVALID``)
   instead of substituting unit spacing; a non-finite voxel rejects it (``INPUT_NONFINITE``), as does a constant phase
   (``INPUT_CONSTANT_INTENSITY``, a branch-specific input-QC code);
2. reoriented in plane only (a lossless transpose/flip chosen from the direction cosines); the short-axis plane and
   the stored slice order are preserved;
3. scaled per phase: the 1st/99th percentiles of the finite voxels of all slices of the phase, clipped to [0, 1];
4. resampled per slice to 1.5 mm with separable bilinear interpolation on a physical grid centred on the field of view
   and cropped or zero-padded to 224 x 224 pixels (336 mm) in the same step; no through-plane interpolation;
5. written as ``neural_cache/<partition>/<file>.npz`` with float32 ``ed``/``es`` [n, 224, 224] and ``ed_geom``/
   ``es_geom`` [n, 2] = (centred slice position mm / 100, corrected increment mm / 10).  The zip writer is
   deterministic, so identical inputs give identical bytes and ``npz_sha256``.  ``manifest.csv`` lists every written
   case (the contract columns first, then the transform metadata), ``failures.csv`` every rejected case and
   ``cache_record.json`` the transform version and the configuration hash.

Grey-to-3-channel replication and the ImageNet channel mean/SD are applied at load time and never stored.  Labels are
written only for partitions whose label access ``configs/neural_input.yaml`` ``cache.labels.allowed_partitions``
grants (``acdc_dev``).  :func:`build_tensor_cache_from_volumes` builds a cache from in-memory volumes
(:class:`CaseVolumes`), so the simulated acquisition shift (S6) reuses exactly this transform.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..runs import atomic_write_bytes, atomic_write_json, parallel_map, sha256_json

NEURAL_INPUT_CONFIG = "configs/neural_input.yaml"
PHASES = ("ed", "es")
CACHE_ARRAYS = ("ed", "es", "ed_geom", "es_geom")
MANIFEST_COLUMNS = ("sample_id", "cluster_id", "partition", "label", "n_ed", "n_es", "npz_sha256", "crop_fraction_ed",
                    "crop_fraction_es", "inplane_spacing_mm", "transform_version")
EXTRA_COLUMNS = ("dataset", "case_id", "npz_file", "npz_bytes", "target_spacing_mm", "source_spacing_row_mm",
                 "source_spacing_col_mm", "slice_increment_mm", "source_rows", "source_cols", "orientation",
                 "z_spacing_method", "intensity_lo_ed", "intensity_hi_ed", "intensity_lo_es", "intensity_hi_es")
FAILURE_COLUMNS = ("sample_id", "cluster_id", "partition", "dataset", "case_id", "reason_code", "detail")
CACHE_RECORD = "cache_record.json"
CACHE_FILES = ("manifest.csv", "failures.csv", CACHE_RECORD)
GEOMETRY_INVALID = "INPUT_GEOMETRY_INVALID"
NONFINITE = "INPUT_NONFINITE"
DECODE_FAILURE = "INPUT_DECODE_FAILURE"
CONSTANT_PHASE = "INPUT_CONSTANT_INTENSITY"
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)


class InputRejected(Exception):
    """A case-level input-QC rejection with a reason code (the case is listed in ``failures.csv``)."""

    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = reason_code
        self.detail = detail


# ----------------------------------------------------------------------------- configuration


def load_neural_input_config(root: str | Path | None = None) -> dict[str, Any]:
    """``configs/neural_input.yaml`` below ``root`` (default: the project root)."""

    from ..io import read_yaml
    from ..paths import PROJECT_ROOT

    return read_yaml(Path(root if root is not None else PROJECT_ROOT) / NEURAL_INPUT_CONFIG)


def config_digest(cfg: Mapping[str, Any]) -> str:
    """SHA-256 of the blocks that define the transform (input, cache, geometry, input_qc)."""

    return sha256_json({k: cfg.get(k) for k in ("input", "cache", "geometry", "input_qc")})


def transform_version(cfg: Mapping[str, Any]) -> str:
    return f"{cfg['cache']['transform_id']}:{config_digest(cfg)[:12]}"


def _qc(cfg: Mapping[str, Any], key: str, default: str) -> str:
    return str((cfg.get("input_qc") or {}).get(key, default))


def labels_allowed(cfg: Mapping[str, Any], partition: str) -> bool:
    return partition in {str(p) for p in cfg["cache"]["labels"]["allowed_partitions"]}


def cache_file_name(sample_id: str) -> str:
    """File name of a case (``:`` is not portable to Windows file systems)."""

    return str(sample_id).replace(":", "__").replace("/", "_") + ".npz"


# ----------------------------------------------------------------------------- geometry


@dataclass(frozen=True)
class PhaseGeometry:
    """Validated geometry of one phase volume in staged voxel order (X, Y, Z); directions are unit RAS vectors."""

    shape: tuple[int, int, int]
    spacing_mm: tuple[float, float]
    slice_increment_mm: float
    directions: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    z_spacing_method: str = "in_memory"

    def axis(self, index: int) -> np.ndarray:
        return np.asarray(self.directions[index], dtype=float)


def geometry_from_parameters(shape: Sequence[int], spacing_mm: Sequence[float], slice_increment_mm: float | None,
                             directions: np.ndarray | Sequence[Sequence[float]] | None = None, *,
                             z_spacing_method: str = "in_memory", cfg: Mapping[str, Any] | None = None) -> PhaseGeometry:
    """Validate and freeze a phase geometry (``directions``: rows are the unit directions of voxel axes 0, 1, 2)."""

    gcfg = (cfg or {}).get("geometry") or {}
    orth_tol = float(gcfg.get("orthogonality_tolerance", 1e-3))
    min_cos = float(gcfg.get("slice_axis_min_abs_cos", 0.99))
    dims = tuple(int(x) for x in shape)
    if len(dims) != 3 or min(dims) < 1:
        raise InputRejected(DECODE_FAILURE, f"expected a non-empty 3D phase volume, got shape {dims}")
    sx, sy = (float(v) for v in spacing_mm)
    if not (np.isfinite(sx) and np.isfinite(sy) and sx > 0 and sy > 0):
        raise InputRejected(GEOMETRY_INVALID, f"in-plane spacing ({sx}, {sy}) mm is non-positive or non-finite")
    dz = float("nan") if slice_increment_mm is None else float(slice_increment_mm)
    if not (np.isfinite(dz) and dz > 0):
        raise InputRejected(GEOMETRY_INVALID, f"slice increment {slice_increment_mm} mm is unavailable or non-positive")
    d = np.eye(3) if directions is None else np.asarray(directions, dtype=float)
    if d.shape != (3, 3) or not np.isfinite(d).all():
        raise InputRejected(GEOMETRY_INVALID, "direction cosines are not a finite 3 x 3 matrix")
    norms = np.linalg.norm(d, axis=1)
    if np.any(norms <= 1e-9):
        raise InputRejected(GEOMETRY_INVALID, "a voxel axis has zero length in the affine")
    d = d / norms[:, None]
    if abs(float(d[0] @ d[1])) > orth_tol:
        raise InputRejected(GEOMETRY_INVALID, f"in-plane voxel axes are not orthogonal (|cos| = {abs(float(d[0] @ d[1])):.2e})")
    normal = np.cross(d[0], d[1])
    normal /= np.linalg.norm(normal)
    if abs(float(normal @ d[2])) < min_cos:
        raise InputRejected(GEOMETRY_INVALID, f"slice axis is sheared against the slice normal (|cos| = {abs(float(normal @ d[2])):.4f})")
    return PhaseGeometry(shape=dims, spacing_mm=(sx, sy), slice_increment_mm=dz,
                         directions=tuple(tuple(float(v) for v in row) for row in d), z_spacing_method=str(z_spacing_method))


def phase_geometry(img: Any, cfg: Mapping[str, Any] | None = None) -> PhaseGeometry:
    """Geometry of a staged image through :func:`tmmri.ingest.geometry.geometry_from_image` (ACDC pixdim fallback)."""

    from ..ingest.geometry import affine_is_unscaled, geometry_from_image

    geom = geometry_from_image(img)
    if len(geom.image_shape) != 3:
        raise InputRejected(DECODE_FAILURE, f"expected a 3D frame, got shape {geom.image_shape}")
    if geom.z_spacing_method == "unavailable" or geom.slice_increment_mm is None:
        raise InputRejected(GEOMETRY_INVALID, "slice increment unavailable")
    affine = np.asarray(geom.affine_matrix, dtype=float)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise InputRejected(GEOMETRY_INVALID, "affine is not a finite 4 x 4 matrix")
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    norms = np.linalg.norm(affine[:3, :3], axis=0)
    if np.any(norms <= 1e-9):
        raise InputRejected(GEOMETRY_INVALID, "a voxel axis has zero length in the affine")
    unit_affine = bool(np.allclose(norms, 1.0, atol=1e-4) and not np.allclose(zooms, 1.0, atol=1e-4))
    if affine_is_unscaled(affine, zooms) and not unit_affine:
        raise InputRejected(GEOMETRY_INVALID, f"affine in-plane scale {norms[:2].tolist()} disagrees with pixdim {list(zooms[:2])}")
    return geometry_from_parameters(geom.image_shape, (geom.spacing_x_mm, geom.spacing_y_mm), geom.slice_increment_mm,
                                    (affine[:3, :3] / norms).T, z_spacing_method=geom.z_spacing_method, cfg=cfg)


def check_phase_pair(ed: PhaseGeometry, es: PhaseGeometry, cfg: Mapping[str, Any] | None = None) -> None:
    """ED and ES of one case must share in-plane spacing, slice increment and direction cosines."""

    tol = float(((cfg or {}).get("geometry") or {}).get("ed_es_tolerance_mm", 1e-4))
    same = (np.allclose(ed.spacing_mm, es.spacing_mm, rtol=0, atol=tol)
            and abs(ed.slice_increment_mm - es.slice_increment_mm) <= tol
            and np.allclose(np.asarray(ed.directions), np.asarray(es.directions), rtol=0, atol=1e-4)
            and ed.shape[:2] == es.shape[:2])
    if not same:
        raise InputRejected(GEOMETRY_INVALID, "ED and ES geometry differ (spacing, increment, orientation or in-plane shape)")


# ----------------------------------------------------------------------------- orientation


@dataclass(frozen=True)
class Orientation:
    """Lossless in-plane reorientation: output rows/columns come from voxel axes ``row_axis``/``col_axis`` (0 or 1)."""

    row_axis: int
    row_flip: bool
    col_axis: int
    col_flip: bool

    @property
    def code(self) -> str:
        return f"rows={'-' if self.row_flip else '+'}{self.row_axis},cols={'-' if self.col_flip else '+'}{self.col_axis}"

    def spacing(self, spacing_mm: Sequence[float]) -> tuple[float, float]:
        return float(spacing_mm[self.row_axis]), float(spacing_mm[self.col_axis])


def canonical_orientation(geom: PhaseGeometry) -> Orientation:
    """Columns: the signed in-plane voxel axis pointing most toward patient left (RAS -x; ties to the earlier
    candidate in the order +0, -0, +1, -1).  Rows: the other in-plane axis, signed toward posterior (RAS -y), or toward
    inferior (RAS -z) when it has no anterior-posterior component."""

    axes = (geom.axis(0), geom.axis(1))
    candidates = ((0, False), (0, True), (1, False), (1, True))
    scores = [-(-axes[a] if flip else axes[a])[0] for a, flip in candidates]
    best = 0
    for i in range(1, len(candidates)):
        if scores[i] > scores[best] + 1e-9:
            best = i
    col_axis, col_flip = candidates[best]
    row_axis = 1 - col_axis
    r = axes[row_axis]
    toward = -r[1] if abs(r[1]) > 1e-9 else -r[2]
    return Orientation(row_axis=row_axis, row_flip=bool(toward < 0), col_axis=col_axis, col_flip=bool(col_flip))


def oriented_slices(volume: np.ndarray, orientation: Orientation) -> np.ndarray:
    """``(n_slices, rows, cols)`` view of an (X, Y, Z) volume; slice order unchanged, no interpolation."""

    arr = np.moveaxis(np.asarray(volume), 2, 0)            # (Z, X, Y)
    if orientation.row_axis == 1:
        arr = arr.transpose(0, 2, 1)                        # rows from voxel axis 1
    if orientation.row_flip:
        arr = arr[:, ::-1, :]
    if orientation.col_flip:
        arr = arr[:, :, ::-1]
    return np.ascontiguousarray(arr)


# ----------------------------------------------------------------------------- intensity and resampling


def phase_window(slices: np.ndarray, percentiles: Sequence[float], cfg: Mapping[str, Any] | None = None) -> tuple[float, float]:
    """(lo, hi) intensity percentiles of a phase; non-finite voxels and constant phases are rejected."""

    values = np.asarray(slices, dtype=np.float64)
    if values.size == 0:
        raise InputRejected(_qc(cfg or {}, "empty_or_not_3d", DECODE_FAILURE), "empty phase volume")
    if not np.isfinite(values).all():
        raise InputRejected(_qc(cfg or {}, "nonfinite_voxel", NONFINITE), f"{int((~np.isfinite(values)).sum())} non-finite voxels")
    lo, hi = (float(v) for v in np.percentile(values, [float(percentiles[0]), float(percentiles[1])]))
    if not hi > lo:
        raise InputRejected(_qc(cfg or {}, "constant_phase", CONSTANT_PHASE), f"constant phase intensity (p{percentiles[0]} = p{percentiles[1]} = {lo})")
    return lo, hi


def scale_phase(slices: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((np.asarray(slices, dtype=np.float64) - lo) / (hi - lo), 0.0, 1.0)


def axis_weights(n_source: int, spacing_mm: float, target_mm: float, size: int) -> np.ndarray:
    """``(size, n_source)`` bilinear weights from a source axis to the centred ``target_mm`` grid of ``size`` pixels.

    Output pixel j sits at ``(j - (size - 1) / 2) * target_mm`` from the field-of-view centre; the source index is
    ``offset / spacing_mm + (n_source - 1) / 2``.  Positions inside the source field of view ([-0.5, n - 0.5]
    pixels) interpolate between the two nearest pixel centres (edge pixels replicate up to the FOV border); positions
    outside it are zero (zero-padding)."""

    j = np.arange(int(size), dtype=np.float64)
    u = (j - (size - 1) / 2.0) * (float(target_mm) / float(spacing_mm)) + (n_source - 1) / 2.0
    inside = (u >= -0.5) & (u <= n_source - 0.5)
    uc = np.clip(u, 0.0, n_source - 1.0)
    i0 = np.minimum(np.floor(uc).astype(int), n_source - 1)
    i1 = np.minimum(i0 + 1, n_source - 1)
    w1 = uc - i0
    weights = np.zeros((int(size), int(n_source)), dtype=np.float64)
    rows = np.flatnonzero(inside)
    np.add.at(weights, (rows, i0[rows]), 1.0 - w1[rows])
    np.add.at(weights, (rows, i1[rows]), w1[rows])
    return weights


def crop_fraction(shape_rc: Sequence[int], spacing_rc: Sequence[float], target_mm: float, size_rc: Sequence[int]) -> float:
    """Fraction of the source field of view outside the ``size x target_mm`` window (0 when it fits)."""

    extent = [float(n) * float(s) for n, s in zip(shape_rc, spacing_rc)]
    window = [float(n) * float(target_mm) for n in size_rc]
    kept = min(extent[0], window[0]) * min(extent[1], window[1])
    return float(max(0.0, 1.0 - kept / (extent[0] * extent[1])))


def resample_slices(slices: np.ndarray, spacing_rc: Sequence[float], target_mm: float,
                    size_rc: Sequence[int]) -> tuple[np.ndarray, float]:
    """Physical in-plane resampling plus centre crop/zero-pad of ``(n, rows, cols)`` slices -> float32 ``(n, H, W)``."""

    arr = np.asarray(slices, dtype=np.float64)
    wr = axis_weights(arr.shape[1], spacing_rc[0], target_mm, size_rc[0])
    wc = axis_weights(arr.shape[2], spacing_rc[1], target_mm, size_rc[1])
    out = np.matmul(np.matmul(wr, arr), wc.T)
    return out.astype(np.float32), crop_fraction(arr.shape[1:], spacing_rc, target_mm, size_rc)


def slice_geometry(n_slices: int, increment_mm: float) -> np.ndarray:
    """``[n, 2]`` float32: centred physical slice position (mm) / 100 and the corrected increment (mm) / 10."""

    k = np.arange(int(n_slices), dtype=np.float64)
    position = (k - (n_slices - 1) / 2.0) * float(increment_mm)
    return np.column_stack([position / 100.0, np.full(int(n_slices), float(increment_mm) / 10.0)]).astype(np.float32)


# ----------------------------------------------------------------------------- cases and cache entries


@dataclass
class CaseVolumes:
    """ED and ES image volumes of one case in staged voxel order (X, Y, Z) with their validated geometry."""

    sample_id: str
    ed_image: np.ndarray
    es_image: np.ndarray
    ed_geometry: PhaseGeometry
    es_geometry: PhaseGeometry
    cluster_id: str | None = None
    label: str | None = None
    dataset: str | None = None
    case_id: str | None = None


@dataclass
class CacheEntry:
    arrays: dict[str, np.ndarray]
    row: dict[str, Any] = field(default_factory=dict)


def preprocess_case(case: CaseVolumes, cfg: Mapping[str, Any]) -> CacheEntry:
    """The C3 arrays and the manifest metadata of one case (raises :class:`InputRejected`)."""

    icfg = cfg["input"]
    target = float(icfg["in_plane_spacing_mm"])
    size = tuple(int(v) for v in icfg["size_pixels"])
    percentiles = [float(v) for v in icfg["percentile_clip_per_phase"]]
    check_phase_pair(case.ed_geometry, case.es_geometry, cfg)
    orientation = canonical_orientation(case.ed_geometry)
    arrays: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {}
    for phase in PHASES:
        volume = np.asarray(getattr(case, f"{phase}_image"))
        geom: PhaseGeometry = getattr(case, f"{phase}_geometry")
        if volume.ndim != 3 or volume.shape[2] < 1 or tuple(volume.shape) != tuple(geom.shape):
            raise InputRejected(_qc(cfg, "empty_or_not_3d", DECODE_FAILURE),
                                f"{phase}: volume shape {volume.shape} does not match its geometry {geom.shape}")
        slices = oriented_slices(volume, orientation)
        lo, hi = phase_window(slices, percentiles, cfg)
        spacing_rc = orientation.spacing(geom.spacing_mm)
        arrays[phase], meta[f"crop_fraction_{phase}"] = resample_slices(scale_phase(slices, lo, hi), spacing_rc, target, size)
        arrays[f"{phase}_geom"] = slice_geometry(slices.shape[0], geom.slice_increment_mm)
        meta[f"intensity_lo_{phase}"], meta[f"intensity_hi_{phase}"] = lo, hi
        if phase == "ed":
            meta.update({"source_spacing_row_mm": spacing_rc[0], "source_spacing_col_mm": spacing_rc[1],
                         "source_rows": int(slices.shape[1]), "source_cols": int(slices.shape[2])})
    ed = case.ed_geometry
    row = {
        "sample_id": str(case.sample_id), "cluster_id": str(case.cluster_id or case.sample_id),
        "label": case.label, "n_ed": int(arrays["ed"].shape[0]), "n_es": int(arrays["es"].shape[0]),
        "inplane_spacing_mm": float(np.sqrt(ed.spacing_mm[0] * ed.spacing_mm[1])), "transform_version": transform_version(cfg),
        "dataset": case.dataset, "case_id": case.case_id, "target_spacing_mm": target,
        "slice_increment_mm": float(ed.slice_increment_mm), "orientation": orientation.code,
        "z_spacing_method": ed.z_spacing_method, **meta,
    }
    return CacheEntry(arrays=arrays, row=row)


# ----------------------------------------------------------------------------- npz files


def npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Deterministic ``.npz`` (stored zip, fixed timestamps and attributes) of the C3 arrays in contract order."""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name in CACHE_ARRAYS:
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_DATE)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload.getvalue())
    return buffer.getvalue()


def write_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> tuple[str, int]:
    data = npz_bytes(arrays)
    atomic_write_bytes(path, data)
    return hashlib.sha256(data).hexdigest(), len(data)


def load_cache_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Read one cache file (no pickles); the trainer's own loader is ``tmmri.neural.data.load_npz``."""

    with np.load(str(path), allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in CACHE_ARRAYS}


# ----------------------------------------------------------------------------- staged sources


def load_staged_volumes(staged_root: str | Path, dataset: str, case_id: str, *, sample_id: str | None = None,
                        cluster_id: str | None = None, label: str | None = None, cfg: Mapping[str, Any] | None = None,
                        verify: bool = True) -> CaseVolumes:
    """ED and ES images of a staged case (``tmmri.ingest.staging`` layout); masks are never opened."""

    import nibabel as nib

    from ..ingest.staging import _read_record, _staged_matches

    case_dir = Path(staged_root) / str(dataset) / str(case_id)
    record = _read_record(case_dir)
    if record is None or record.get("status") != "STAGED":
        raise FileNotFoundError(f"no staged case at {case_dir}")
    if verify and not _staged_matches(case_dir, record):
        raise InputRejected("SOURCE_HASH_MISMATCH", f"staged files of {dataset}:{case_id} differ from case.json")
    volumes: dict[str, np.ndarray] = {}
    geometries: dict[str, PhaseGeometry] = {}
    for phase in PHASES:
        img = nib.load(str(case_dir / f"{phase}_image.nii"))
        geometries[phase] = phase_geometry(img, cfg)
        volumes[phase] = np.asarray(img.dataobj, dtype=np.float32)
    return CaseVolumes(sample_id=str(sample_id or record["sample_id"]), ed_image=volumes["ed"], es_image=volumes["es"],
                       ed_geometry=geometries["ed"], es_geometry=geometries["es"], cluster_id=cluster_id, label=label,
                       dataset=str(dataset), case_id=str(case_id))


# ----------------------------------------------------------------------------- cache directory


def _failure(job: Mapping[str, Any], reason: str, detail: str) -> dict[str, Any]:
    return {"status": "FAILED", "row": {k: job.get(k) for k in ("sample_id", "cluster_id", "partition", "dataset", "case_id")}
            | {"reason_code": reason, "detail": detail}}


def _write_entry(entry: CacheEntry, out_dir: Path, partition: str) -> dict[str, Any]:
    name = cache_file_name(entry.row["sample_id"])
    sha, nbytes = write_npz(out_dir / name, entry.arrays)
    return {**entry.row, "partition": partition, "npz_sha256": sha, "npz_file": name, "npz_bytes": nbytes}


def _cache_job(job: Mapping[str, Any]) -> dict[str, Any]:
    """Worker (module level for pickling): load one staged case, preprocess it and write its npz."""

    try:
        case = load_staged_volumes(job["staged_root"], job["dataset"], job["case_id"], sample_id=job["sample_id"],
                                   cluster_id=job.get("cluster_id"), label=job.get("label"), cfg=job["cfg"])
        return {"status": "WRITTEN", "row": _write_entry(preprocess_case(case, job["cfg"]), Path(job["out_dir"]), job["partition"])}
    except InputRejected as exc:
        return _failure(job, exc.reason_code, exc.detail)
    except (OSError, ValueError) as exc:
        return _failure(job, DECODE_FAILURE, f"{type(exc).__name__}: {exc}")


def _reset_cache_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in list(out_dir.glob("*.npz")) + [out_dir / name for name in CACHE_FILES]:
        path.unlink(missing_ok=True)


def write_cache_tables(out_dir: str | Path, partition: str, cfg: Mapping[str, Any], results: Iterable[Mapping[str, Any]],
                       extra_record: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``manifest.csv``, ``failures.csv`` and ``cache_record.json`` from worker results (sorted by sample_id)."""

    out = Path(out_dir)
    results = list(results)
    rows = sorted((r["row"] for r in results if r["status"] == "WRITTEN"), key=lambda r: str(r["sample_id"]))
    failures = sorted((r["row"] for r in results if r["status"] != "WRITTEN"), key=lambda r: str(r["sample_id"]))
    manifest = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS + EXTRA_COLUMNS))
    if not labels_allowed(cfg, partition):
        manifest["label"] = None
    failed = pd.DataFrame(failures, columns=list(FAILURE_COLUMNS))
    manifest_bytes = manifest.to_csv(index=False, lineterminator="\n").encode("utf-8")
    atomic_write_bytes(out / "manifest.csv", manifest_bytes)
    atomic_write_bytes(out / "failures.csv", failed.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    record = {
        "schema": "tmmri.neural_cache/1", "partition": partition, "transform_version": transform_version(cfg),
        "config_sha256": config_digest(cfg), "input": cfg["input"], "labels_written": labels_allowed(cfg, partition),
        "n_cases": int(len(manifest)), "n_failures": int(len(failed)),
        "failure_reasons": {str(k): int(v) for k, v in failed["reason_code"].value_counts().sort_index().items()},
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), **dict(extra_record or {}),
    }
    atomic_write_json(out / CACHE_RECORD, record)
    return {"manifest": manifest, "failures": failed, "record": record}


def build_tensor_cache_from_staged(jobs: Sequence[Mapping[str, Any]], out_dir: str | Path, partition: str,
                                   cfg: Mapping[str, Any], *, staged_root: str | Path, n_jobs: int = 1,
                                   extra_record: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Cache of staged cases; ``jobs`` carry ``sample_id, cluster_id, dataset, case_id, label`` (label is dropped
    unless the partition grants label access)."""

    out = Path(out_dir)
    _reset_cache_dir(out)
    allowed = labels_allowed(cfg, partition)
    work = [{**dict(job), "label": job.get("label") if allowed else None, "partition": partition, "cfg": dict(cfg),
             "staged_root": str(staged_root), "out_dir": str(out)} for job in sorted(jobs, key=lambda j: str(j["sample_id"]))]
    return write_cache_tables(out, partition, cfg, parallel_map(_cache_job, work, n_jobs=n_jobs), extra_record)


def build_tensor_cache_from_volumes(cases: Iterable[CaseVolumes], out_dir: str | Path, partition: str,
                                    cfg: Mapping[str, Any] | None = None, *, extra_record: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The same cache from in-memory volumes (for example the S6 resampled ``acdc_test`` images)."""

    cfg = load_neural_input_config() if cfg is None else cfg
    out = Path(out_dir)
    _reset_cache_dir(out)
    allowed = labels_allowed(cfg, partition)
    results = []
    for case in sorted(cases, key=lambda c: str(c.sample_id)):
        job = {"sample_id": case.sample_id, "cluster_id": case.cluster_id or case.sample_id, "partition": partition,
               "dataset": case.dataset, "case_id": case.case_id}
        try:
            if not allowed:
                case = CaseVolumes(**{**case.__dict__, "label": None})
            results.append({"status": "WRITTEN", "row": _write_entry(preprocess_case(case, cfg), out, partition)})
        except InputRejected as exc:
            results.append(_failure(job, exc.reason_code, exc.detail))
    return write_cache_tables(out, partition, cfg, results, extra_record)


def case_tensors(case: Any, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Integration of L89 (S6) and L4a: the C3 arrays and manifest fields of an in-memory staged case
    (:class:`tmmri.ingest.staging.StagedCase`, for example a resampled ``acdc_test`` case), by exactly the transform of
    ``build-tensor-cache`` (:func:`preprocess_case`).  This is the hook ``configs/data_centric.yaml``
    ``simulated_shift.tensor_inputs.hook`` names; geometry comes from the case affine (direction cosines) and its
    ``spacing_mm`` (in-plane spacing and corrected slice increment)."""

    entry = preprocess_case(case_volumes(case, cfg), cfg)
    fields = ("crop_fraction_ed", "crop_fraction_es", "inplane_spacing_mm", "transform_version")
    return {**entry.arrays, **{k: entry.row[k] for k in fields}}


def case_volumes(case: Any, cfg: Mapping[str, Any]) -> CaseVolumes:
    """The :class:`CaseVolumes` of an in-memory staged case (geometry from its affine and ``spacing_mm``), the input of
    :func:`preprocess_case` (C3) and of the D20 extractor (S6 runs both on resampled ``acdc_test`` cases)."""

    affine = np.asarray(case.affine, dtype=float)
    norms = np.linalg.norm(affine[:3, :3], axis=0)
    if affine.shape != (4, 4) or np.any(norms <= 1e-9):
        raise InputRejected(GEOMETRY_INVALID, "case affine is not a 4 x 4 matrix with non-zero voxel axes")
    directions = (affine[:3, :3] / norms).T
    sx, sy, dz = (None if v is None else float(v) for v in case.spacing_mm)
    method = str(getattr(case, "geometry_method", "in_memory"))
    geoms = {phase: geometry_from_parameters(np.asarray(getattr(case, f"{phase}_image")).shape, (sx, sy), dz, directions,
                                             z_spacing_method=method, cfg=cfg) for phase in PHASES}
    return CaseVolumes(sample_id=str(case.sample_id), ed_image=np.asarray(case.ed_image, dtype=np.float32),
                       es_image=np.asarray(case.es_image, dtype=np.float32), ed_geometry=geoms["ed"], es_geometry=geoms["es"],
                       dataset=getattr(case, "dataset", None), case_id=getattr(case, "case_id", None))


def read_cache_manifest(cache_dir: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(Path(cache_dir) / "manifest.csv", dtype={"sample_id": str, "cluster_id": str, "label": "string"})
    return frame.sort_values("sample_id").reset_index(drop=True)


def verify_cache_entry(cache_dir: str | Path, row: Mapping[str, Any]) -> Path:
    """Path of a manifest row's npz after checking its SHA-256 (raises ``ValueError`` on a mismatch)."""

    from ..runs import sha256_file

    path = Path(cache_dir) / str(row.get("npz_file") or cache_file_name(str(row["sample_id"])))
    if not path.is_file():
        raise FileNotFoundError(f"cache file missing: {path}")
    if sha256_file(path) != str(row["npz_sha256"]):
        raise ValueError(f"cache file {path.name} does not match npz_sha256 in manifest.csv")
    return path
