"""F9 input pipeline on the contract-C3 tensor cache (contract C8 ``tmmri.neural.data``).

Cache layout: ``neural_cache/<partition>/<sample_id>.npz`` with float32 ``ed [n_ed,H,W]`` and ``es [n_es,H,W]`` in
[0, 1] (H = W = 224 for real caches) and ``ed_geom [n_ed,2]``, ``es_geom [n_es,2]`` = (centred physical slice position
mm / 100, corrected slice increment mm / 10), plus ``manifest.csv`` (:data:`MANIFEST_COLUMNS`).  Slice order is kept.

* :func:`load_npz` reads and checks one patient (optionally verifying the manifest SHA-256).
* :func:`collate_patients` pads a list of patients to ``[B, S, H, W]`` per phase with **prefix** boolean validity
  masks (valid slices first, zeros after); :class:`tmmri.neural.model.F9Net` replicates grey to three channels and applies
  the ImageNet mean/SD at its input, so neither is stored.
* :func:`augment_batch` is the training augmentation of NEURAL-TM-1.0 4.2 with the 6.0.1 D4 step (owner issue O1;
  :mod:`tmmri.neural.dihedral`): one transform per patient shared by both phases and all slices: first one element of
  the in-plane dihedral group D4 drawn uniformly (``augmentation.dihedral: d4_uniform_per_patient``; exact pixel
  permutation), then rotation U[-10, 10] degrees, translation U[-10, 10] mm per axis at 1.5 mm per pixel, intensity
  x U[0.9, 1.1], additive Gaussian noise SD 0.01, then clipping to [0, 1]; no other reflection, no elastic
  deformation, scaling, phase exchange or mixup; geometry scalars unchanged.  Per patient the numpy ``Generator``
  (platform independent) draws, in this order, the D4 element, angle, x and y translation, intensity factor, ED noise
  and ES noise; the spatial resampling is a forward-only ``grid_sample`` on the training device.

torch is imported inside the functions that need it, so manifest and npz helpers run in the CPU container.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TRAINING_PARTITION = "acdc_dev"
ARRAY_KEYS = ("ed", "es", "ed_geom", "es_geom")
MANIFEST_COLUMNS = ("sample_id", "cluster_id", "partition", "label", "n_ed", "n_es", "npz_sha256", "crop_fraction_ed",
                    "crop_fraction_es", "inplane_spacing_mm", "transform_version")
RANGE_TOLERANCE = 1e-6


def read_manifest(path: str | Path) -> pd.DataFrame:
    """``manifest.csv`` with string identifiers (an empty label stays ``""``)."""

    frame = pd.read_csv(path, dtype={"sample_id": str, "cluster_id": str, "partition": str, "label": str, "npz_sha256": str,
                                     "transform_version": str}, keep_default_na=False)
    missing = [c for c in MANIFEST_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: manifest columns missing {missing}")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate sample ids")
    return frame


def npz_path(cache_dir: str | Path, row: Mapping[str, Any] | str) -> Path:
    """The C3 file of a case (integration of L4a and L4b): the manifest ``npz_file`` column when present, else
    :func:`tmmri.neural.preprocess.cache_file_name` of the sample id (``ACDC:patient001`` -> ``ACDC__patient001.npz``;
    ``:`` is not portable to Windows file systems)."""

    from .preprocess import cache_file_name

    if isinstance(row, Mapping):
        name = str(row.get("npz_file") or "").strip()
        return Path(cache_dir) / (name if name and name.lower() != "nan" else cache_file_name(str(row["sample_id"])))
    return Path(cache_dir) / cache_file_name(str(row))


def sha256_bytes_of(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_npz(path: str | Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    """One patient: ``sample_id`` (file stem) and the four float32 arrays, checked for shape, finiteness and range."""

    path = Path(path)
    if expected_sha256 is not None:
        digest = sha256_bytes_of(path)
        if digest != str(expected_sha256).strip().lower():
            raise ValueError(f"{path}: SHA-256 {digest} differs from the manifest ({expected_sha256})")
    with np.load(path, allow_pickle=False) as handle:
        missing = [k for k in ARRAY_KEYS if k not in handle.files]
        if missing:
            raise ValueError(f"{path}: arrays missing {missing}")
        out: dict[str, Any] = {"sample_id": path.stem}
        for key in ARRAY_KEYS:
            out[key] = np.ascontiguousarray(handle[key], dtype=np.float32)
    for phase in ("ed", "es"):
        img, geom = out[phase], out[f"{phase}_geom"]
        if img.ndim != 3 or img.shape[0] < 1:
            raise ValueError(f"{path}: {phase} must be [n_slices, H, W] with n_slices >= 1, got {img.shape}")
        if geom.shape != (img.shape[0], 2):
            raise ValueError(f"{path}: {phase}_geom must be [{img.shape[0]}, 2], got {geom.shape}")
        if not (np.isfinite(img).all() and np.isfinite(geom).all()):
            raise ValueError(f"{path}: non-finite values in {phase}")
        if img.min() < -RANGE_TOLERANCE or img.max() > 1.0 + RANGE_TOLERANCE:
            raise ValueError(f"{path}: {phase} intensities outside [0, 1]")
    if out["ed"].shape[1:] != out["es"].shape[1:]:
        raise ValueError(f"{path}: ED and ES slice sizes differ")
    return out


def collate_patients(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pad patients to ``ed/es [B,S,H,W]``, ``ed_geom/es_geom [B,S,2]`` and prefix masks ``ed_mask/es_mask [B,S]``."""

    import torch

    if not items:
        raise ValueError("collate_patients needs at least one patient")
    size = tuple(np.asarray(items[0]["ed"]).shape[1:])
    out: dict[str, Any] = {"sample_id": [str(item.get("sample_id", "")) for item in items]}
    for phase in ("ed", "es"):
        counts = [int(np.asarray(item[phase]).shape[0]) for item in items]
        if min(counts) < 1:
            raise ValueError(f"every patient needs at least one {phase.upper()} slice")
        n_max = max(counts)
        images = torch.zeros((len(items), n_max, *size), dtype=torch.float32)
        geoms = torch.zeros((len(items), n_max, 2), dtype=torch.float32)
        mask = torch.zeros((len(items), n_max), dtype=torch.bool)
        for i, item in enumerate(items):
            arr = np.asarray(item[phase], dtype=np.float32)
            if tuple(arr.shape[1:]) != size:
                raise ValueError(f"slice size {arr.shape[1:]} differs from {size} in one batch")
            n = arr.shape[0]
            images[i, :n] = torch.from_numpy(np.ascontiguousarray(arr))
            geoms[i, :n] = torch.from_numpy(np.ascontiguousarray(np.asarray(item[f"{phase}_geom"], dtype=np.float32)))
            mask[i, :n] = True
        out[phase], out[f"{phase}_geom"], out[f"{phase}_mask"] = images, geoms, mask
    return out


def batch_to(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}


def augmentation_parameters(rng: np.random.Generator, cfg: Mapping[str, Any]) -> dict[str, float]:
    """One patient's transform, drawn in a fixed order: angle, x and y translation (mm), intensity factor."""

    lo, hi = cfg["rotation_degrees"]
    t_lo, t_hi = cfg["translation_mm_each_axis"]
    i_lo, i_hi = cfg["intensity_multiplier"]
    return {"angle_deg": float(rng.uniform(lo, hi)), "tx_mm": float(rng.uniform(t_lo, t_hi)),
            "ty_mm": float(rng.uniform(t_lo, t_hi)), "intensity": float(rng.uniform(i_lo, i_hi))}


def _theta(params: Mapping[str, float], spacing_mm: float, size: tuple[int, int]) -> np.ndarray:
    """Inverse affine of ``F.affine_grid`` (normalized coordinates) for a rotation about the centre and a shift."""

    h, w = size
    a = math.radians(params["angle_deg"])
    tx = params["tx_mm"] / spacing_mm * 2.0 / w
    ty = params["ty_mm"] / spacing_mm * 2.0 / h
    c, s = math.cos(a), math.sin(a)
    # output -> input: x_in = R(-a) (x_out - t)
    return np.array([[c, s, -(c * tx + s * ty)], [-s, c, -(-s * tx + c * ty)]], dtype=np.float32)


def augment_batch(batch: Mapping[str, Any], rng: np.random.Generator, cfg: Mapping[str, Any], *, spacing_mm: float) -> dict[str, Any]:
    """Training augmentation of a collated batch (tensors on the training device); padding stays zero."""

    import torch
    import torch.nn.functional as F

    from .dihedral import D4_ORDER, d4_apply, uses_d4_augmentation

    out = dict(batch)
    ed, es = batch["ed"], batch["es"]
    device = ed.device
    size = (int(ed.shape[-2]), int(ed.shape[-1]))
    noise_sd = float(cfg["noise_sd"])
    n_b = int(ed.shape[0])
    dihedral = uses_d4_augmentation(cfg)
    thetas, factors, elements = [], [], []
    noise = {"ed": np.zeros(tuple(ed.shape), dtype=np.float32), "es": np.zeros(tuple(es.shape), dtype=np.float32)}
    for i in range(n_b):
        elements.append(int(rng.integers(D4_ORDER)) if dihedral else 0)
        params = augmentation_parameters(rng, cfg)
        thetas.append(_theta(params, spacing_mm, size))
        factors.append(params["intensity"])
        for phase in ("ed", "es"):
            n = int(batch[f"{phase}_mask"][i].sum().item())
            noise[phase][i, :n] = rng.standard_normal((n, *size), dtype=np.float32) * np.float32(noise_sd)
    theta = torch.from_numpy(np.stack(thetas)).to(device)
    factor = torch.tensor(factors, dtype=torch.float32, device=device)
    with torch.no_grad():
        if any(elements):                                   # one D4 element per patient, before the affine steps
            ed = torch.stack([d4_apply(ed[i], g) for i, g in enumerate(elements)])
            es = torch.stack([d4_apply(es[i], g) for i, g in enumerate(elements)])
        for phase, images in (("ed", ed), ("es", es)):
            b, s = int(images.shape[0]), int(images.shape[1])
            flat = images.reshape(b * s, 1, *size)
            grid = F.affine_grid(theta.repeat_interleave(s, dim=0), [b * s, 1, *size], align_corners=False)
            moved = F.grid_sample(flat, grid, mode=str(cfg.get("interpolation", "bilinear")), padding_mode=str(cfg.get("padding", "zeros")),
                                  align_corners=False).reshape(b, s, *size)
            moved = moved * factor.view(b, 1, 1, 1) + torch.from_numpy(noise[phase]).to(device)
            moved = moved.clamp(0.0, 1.0) * batch[f"{phase}_mask"].to(moved.dtype).view(b, s, 1, 1)
            out[phase] = moved
    return out
