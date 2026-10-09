from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class NiftiGeometry:
    image_shape: tuple[int, ...]
    spacing_x_mm: float
    spacing_y_mm: float
    slice_thickness_mm: float | None
    slice_gap_mm: float | None
    slice_increment_mm: float | None
    slice_increment_variability_mm: float | None
    z_spacing_method: str
    affine_matrix: list[list[float]]
    num_slices: int | None
    num_frames: int | None

    def as_json_shape(self) -> str:
        return json.dumps(list(self.image_shape), separators=(",", ":"))

    def affine_json(self) -> str:
        return json.dumps(self.affine_matrix, separators=(",", ":"))


def load_nifti_header(path: str | Path) -> NiftiGeometry:
    img = nib.load(str(path))
    return geometry_from_image(img)


def affine_is_unscaled(affine: np.ndarray, zooms: tuple[float, ...]) -> bool:
    """True when the affine encodes orientation only, not voxel size.

    ACDC ships such files: sform diag(-1, -1, 1) while the header pixdim holds 1.5625 x 1.5625 x 10 mm.
    Projecting the slice axis of that affine yields 1 mm and shrinks every volume by the true slice spacing.
    """
    if affine.shape != (4, 4) or len(zooms) < 3:
        return False
    column_norms = np.linalg.norm(affine[:3, :3], axis=0)
    header = np.asarray(zooms[:3], dtype=float)
    in_plane_mismatch = not np.allclose(column_norms[:2], header[:2], rtol=1e-3, atol=1e-4)
    unit_affine = np.allclose(column_norms, 1.0, atol=1e-4) and not np.allclose(header, 1.0, atol=1e-4)
    return bool(in_plane_mismatch or unit_affine)


def geometry_from_image(img: nib.spatialimages.SpatialImage) -> NiftiGeometry:
    shape = tuple(int(x) for x in img.shape)
    zooms = img.header.get_zooms()
    spacing_x = float(zooms[0]) if len(zooms) > 0 else np.nan
    spacing_y = float(zooms[1]) if len(zooms) > 1 else np.nan
    slice_thickness = float(zooms[2]) if len(zooms) > 2 and zooms[2] > 0 else None
    affine = np.asarray(img.affine, dtype=float)
    unscaled_affine = slice_thickness is not None and affine_is_unscaled(affine, zooms)
    if unscaled_affine:
        z_increment = slice_thickness
    elif affine.shape == (4, 4):
        z_vector = affine[:3, 2]
        z_increment = float(np.linalg.norm(z_vector))
    else:
        z_increment = float(slice_thickness or np.nan)
    if unscaled_affine:
        z_method = "nifti_header_pixdim_unscaled_affine"
    elif np.isfinite(z_increment) and z_increment > 0:
        z_method = "nifti_affine_projection"
    elif slice_thickness is not None and slice_thickness > 0:
        z_increment = slice_thickness
        z_method = "slice_thickness_only_fallback"
    else:
        z_increment = None
        z_method = "unavailable"
    num_slices = shape[2] if len(shape) >= 3 else None
    num_frames = shape[3] if len(shape) >= 4 else 1 if len(shape) == 3 else None
    return NiftiGeometry(
        image_shape=shape,
        spacing_x_mm=spacing_x,
        spacing_y_mm=spacing_y,
        slice_thickness_mm=slice_thickness,
        slice_gap_mm=None,
        slice_increment_mm=z_increment,
        slice_increment_variability_mm=0.0 if z_increment is not None else None,
        z_spacing_method=z_method,
        affine_matrix=affine.tolist(),
        num_slices=int(num_slices) if num_slices is not None else None,
        num_frames=int(num_frames) if num_frames is not None else None,
    )


def z_spacing_from_positions(
    image_position_patient: np.ndarray,
    slice_normal: np.ndarray,
) -> tuple[float | None, float | None]:
    positions = np.asarray(image_position_patient, dtype=float)
    normal = np.asarray(slice_normal, dtype=float)
    norm = np.linalg.norm(normal)
    if positions.ndim != 2 or positions.shape[1] != 3 or norm <= 0:
        return None, None
    normal = normal / norm
    projected = np.sort(positions @ normal)
    if projected.size < 2:
        return None, None
    deltas = np.abs(np.diff(projected))
    valid = deltas[deltas > 0]
    if valid.size == 0:
        return None, None
    return float(np.median(valid)), float(np.std(valid))


def select_frame(data: np.ndarray, frame_index: int | None) -> np.ndarray:
    if data.ndim == 3:
        return data
    if data.ndim != 4:
        raise ValueError(f"Expected 3D or 4D image data, got shape {data.shape}.")
    if frame_index is None:
        raise ValueError("Cannot select a 4D frame without a frame index.")
    if frame_index < 0 or frame_index >= data.shape[3]:
        raise IndexError(f"Frame index {frame_index} outside image with {data.shape[3]} frames.")
    return data[..., frame_index]


def serialize_geometry(geom: NiftiGeometry) -> dict[str, Any]:
    return {
        "image_shape": list(geom.image_shape),
        "spacing_x_mm": geom.spacing_x_mm,
        "spacing_y_mm": geom.spacing_y_mm,
        "slice_thickness_mm": geom.slice_thickness_mm,
        "slice_gap_mm": geom.slice_gap_mm,
        "slice_increment_mm": geom.slice_increment_mm,
        "slice_increment_variability_mm": geom.slice_increment_variability_mm,
        "z_spacing_method": geom.z_spacing_method,
        "affine_matrix": geom.affine_matrix,
        "num_slices": geom.num_slices,
        "num_frames": geom.num_frames,
    }
