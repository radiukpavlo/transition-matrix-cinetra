"""Synthetic contract-C3 tensor caches for tests and the Modal qualification (algorithm controls, never cohort data).

:func:`write_synthetic_cache` writes ``<root>/<partition>/<sample_id>.npz`` and ``manifest.csv`` in the C3 format,
with a ``SYNTHETIC_FIXTURE`` marker beside the partition folder (outside the CPE-1 upload set).  Images are smooth
disks whose radius and brightness depend on the class, plus noise, so a few epochs can separate the classes.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .data import MANIFEST_COLUMNS
from .preprocess import cache_file_name
from .selection import CLASSES

SEED_SYNTHETIC = 20260613     # configs/study_v6.yaml seeds.synthetic_controls
MARKER = "SYNTHETIC_FIXTURE"


def _phase(rng: np.random.Generator, n: int, size: int, radius: float, level: float) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    centre = (size - 1) / 2.0
    out = np.empty((n, size, size), dtype=np.float32)
    for s in range(n):
        r = radius * (0.8 + 0.4 * s / max(n - 1, 1))
        disk = 1.0 / (1.0 + np.exp((np.hypot(yy - centre, xx - centre) - r) / 1.5))
        out[s] = np.clip(level * disk + 0.05 * rng.standard_normal((size, size), dtype=np.float32), 0.0, 1.0)
    return out


def write_synthetic_cache(root: str | Path, *, n_per_class: int = 3, slices: tuple[int, int] = (3, 5), size: int = 64,
                          partition: str = "acdc_dev", classes: Sequence[str] = CLASSES, seed: int = SEED_SYNTHETIC,
                          label: bool = True) -> Path:
    """Write a synthetic cache; returns the manifest path.  Sample ids are ``synth<class><k>``; one cluster each."""

    root = Path(root)
    base = root / partition
    base.mkdir(parents=True, exist_ok=True)
    (root / MARKER).write_text("synthetic tensors only; never cohort data\n", encoding="utf-8")
    rng = np.random.default_rng(seed)
    rows = []
    for ci, cls in enumerate(classes):
        for k in range(n_per_class):
            sid = f"synth{cls.lower()}{k:03d}"
            n_ed, n_es = (int(v) for v in rng.integers(slices[0], slices[1] + 1, size=2))
            radius = size * (0.12 + 0.04 * ci)
            arrays = {"ed": _phase(rng, n_ed, size, radius, 0.6 + 0.08 * ci),
                      "es": _phase(rng, n_es, size, radius * (0.9 - 0.08 * ci), 0.6 + 0.08 * ci)}
            for phase in ("ed", "es"):
                n = arrays[phase].shape[0]
                position = (np.arange(n, dtype=np.float32) - (n - 1) / 2.0) * 10.0 / 100.0
                arrays[f"{phase}_geom"] = np.stack([position, np.full(n, 1.0, dtype=np.float32)], axis=1).astype(np.float32)
            path = base / cache_file_name(sid)
            np.savez(path, **arrays)
            rows.append({"sample_id": sid, "cluster_id": sid, "partition": partition, "label": cls if label else "",
                         "n_ed": n_ed, "n_es": n_es, "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "crop_fraction_ed": 0.0, "crop_fraction_es": 0.0, "inplane_spacing_mm": 1.5,
                         "transform_version": "synthetic-v1"})
    manifest = base / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def write_random_resnet18(path: str | Path, seed: int = SEED_SYNTHETIC) -> tuple[Path, str]:
    """A seeded random-initialization ResNet-18 state dict (stands in for the pinned weights in synthetic runs)."""

    import torch
    import torchvision

    torch.manual_seed(int(seed))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torchvision.models.resnet18(weights=None).state_dict(), path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()
