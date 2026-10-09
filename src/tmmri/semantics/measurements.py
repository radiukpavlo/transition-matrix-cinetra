from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from scipy import ndimage
from skimage import measure

from ..ingest.labels import LV_BLOOD_POOL, LV_MYOCARDIUM, RV_BLOOD_POOL


def finite_or_none(value: float | int | np.floating | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def mask_volume_ml(mask: np.ndarray, label: int, dx_mm: float, dy_mm: float, dz_mm: float | None) -> float | None:
    if dz_mm is None or dz_mm <= 0 or dx_mm <= 0 or dy_mm <= 0:
        return None
    return float(np.count_nonzero(mask == label) * dx_mm * dy_mm * dz_mm / 1000.0)


def slice_areas_mm2(mask: np.ndarray, label: int, dx_mm: float, dy_mm: float) -> np.ndarray:
    binary = mask == label
    return binary.sum(axis=(0, 1)).astype(float) * dx_mm * dy_mm


def safe_sv(edv_ml: float | None, esv_ml: float | None) -> float | None:
    if edv_ml is None or esv_ml is None:
        return None
    return float(edv_ml - esv_ml)


def safe_ef(edv_ml: float | None, esv_ml: float | None, near_zero_guard_ml: float = 1.0) -> float | None:
    if edv_ml is None or esv_ml is None or edv_ml < near_zero_guard_ml:
        return None
    return float(100.0 * (edv_ml - esv_ml) / edv_ml)


def bmi_and_bsa(height_cm: float | None, weight_kg: float | None) -> tuple[float | None, float | None]:
    if height_cm is None or weight_kg is None or height_cm <= 0 or weight_kg <= 0:
        return None, None
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m * height_m)
    bsa = math.sqrt(height_cm * weight_kg / 3600.0)
    return float(bmi), float(bsa)


def indexed(value: float | None, bsa_m2: float | None) -> float | None:
    if value is None or bsa_m2 is None or bsa_m2 <= 0:
        return None
    return float(value / bsa_m2)


def _contour_perimeter_mm(binary_2d: np.ndarray, dx_mm: float, dy_mm: float) -> float | None:
    if not binary_2d.any():
        return None
    contours = measure.find_contours(binary_2d.astype(float), 0.5)
    if not contours:
        return None
    total = 0.0
    for contour in contours:
        if contour.shape[0] < 2:
            continue
        coords = contour.copy()
        coords[:, 0] *= dy_mm
        coords[:, 1] *= dx_mm
        total += float(np.linalg.norm(np.diff(coords, axis=0), axis=1).sum())
    return total if total > 0 else None


def _eccentricity_physical(binary_2d: np.ndarray, dx_mm: float, dy_mm: float) -> float | None:
    filled = ndimage.binary_fill_holes(binary_2d.astype(bool))
    rows, cols = np.nonzero(filled)
    if rows.size < 3:
        return None
    coords = np.column_stack([cols.astype(float) * dx_mm, rows.astype(float) * dy_mm])
    cov = np.cov(coords, rowvar=False)
    vals = np.linalg.eigvalsh(cov)
    vals = np.sort(np.maximum(vals, 0.0))
    if vals[-1] <= 0:
        return None
    return float(math.sqrt(max(0.0, 1.0 - vals[0] / vals[-1])))


def shape_summary(mask: np.ndarray, label: int, dx_mm: float, dy_mm: float) -> dict[str, float | bool | None | str]:
    areas = slice_areas_mm2(mask, label, dx_mm, dy_mm)
    area_values: list[float] = []
    compactness_values: list[float] = []
    eccentricity_values: list[float] = []
    for z in range(mask.shape[2]):
        binary = mask[:, :, z] == label
        area = float(binary.sum() * dx_mm * dy_mm)
        if area <= 0:
            continue
        perimeter = _contour_perimeter_mm(binary, dx_mm, dy_mm)
        area_values.append(area)
        if perimeter is not None and perimeter > 0:
            compactness_values.append(float(4.0 * math.pi * area / (perimeter * perimeter)))
        ecc = _eccentricity_physical(binary, dx_mm, dy_mm)
        if ecc is not None:
            eccentricity_values.append(ecc)
    return {
        "area_mean_mm2": float(np.mean(area_values)) if area_values else None,
        "compactness_mean": float(np.mean(compactness_values)) if compactness_values else None,
        "eccentricity_mean": float(np.mean(eccentricity_values)) if eccentricity_values else None,
        "perimeter_algorithm": "marching_squares_subpixel",
        "perimeter_spacing_applied_flag": True,
        "perimeter_valid_flag": bool(compactness_values),
        "nonzero_slice_count": int(np.count_nonzero(areas > 0)),
    }


def _radial_slice_thickness_mm(
    lv_binary: np.ndarray,
    myo_binary: np.ndarray,
    dx_mm: float,
    dy_mm: float,
    rays: int = 72,
) -> float | None:
    if not lv_binary.any() or not myo_binary.any():
        return None
    rows, cols = np.nonzero(lv_binary)
    center_row = float(rows.mean())
    center_col = float(cols.mean())
    if not lv_binary[int(round(center_row)), int(round(center_col))]:
        return None
    epi_binary = np.logical_or(lv_binary, myo_binary)
    max_radius_mm = math.hypot(lv_binary.shape[0] * dy_mm, lv_binary.shape[1] * dx_mm)
    step_mm = max(min(dx_mm, dy_mm) / 2.0, 0.25)
    radii = np.arange(0.0, max_radius_mm + step_mm, step_mm)
    thickness_values: list[float] = []
    for theta in np.linspace(0.0, 2.0 * math.pi, rays, endpoint=False):
        row_coords = center_row + (radii * math.sin(theta) / dy_mm)
        col_coords = center_col + (radii * math.cos(theta) / dx_mm)
        coords = np.vstack([row_coords, col_coords])
        lv_samples = ndimage.map_coordinates(lv_binary.astype(np.uint8), coords, order=0, mode="constant", cval=0) > 0
        epi_samples = ndimage.map_coordinates(epi_binary.astype(np.uint8), coords, order=0, mode="constant", cval=0) > 0
        lv_true = np.flatnonzero(lv_samples)
        epi_true = np.flatnonzero(epi_samples)
        if lv_true.size == 0 or epi_true.size == 0:
            continue
        endo_idx = int(lv_true.max())
        epi_after = epi_true[epi_true >= endo_idx]
        if epi_after.size == 0:
            continue
        epi_idx = int(epi_after.max())
        thickness = (epi_idx - endo_idx) * step_mm
        if thickness > 0:
            thickness_values.append(float(thickness))
    return float(np.mean(thickness_values)) if thickness_values else None


def global_wall_thickness_mm(mask: np.ndarray, dx_mm: float, dy_mm: float) -> tuple[float | None, int, bool]:
    weighted_sum = 0.0
    weight_total = 0.0
    invalid_count = 0
    for z in range(mask.shape[2]):
        lv = mask[:, :, z] == LV_BLOOD_POOL
        myo = mask[:, :, z] == LV_MYOCARDIUM
        myo_area = float(myo.sum() * dx_mm * dy_mm)
        if myo_area <= 0:
            continue
        thickness = _radial_slice_thickness_mm(lv, myo, dx_mm, dy_mm)
        if thickness is None:
            invalid_count += 1
            continue
        weighted_sum += thickness * myo_area
        weight_total += myo_area
    if weight_total <= 0:
        return None, invalid_count, False
    return float(weighted_sum / weight_total), invalid_count, True


def wall_thickening_percent(ed_mm: float | None, es_mm: float | None, near_zero_guard_mm: float = 0.5) -> float | None:
    if ed_mm is None or es_mm is None or ed_mm < near_zero_guard_mm:
        return None
    return float(100.0 * (es_mm - ed_mm) / ed_mm)


def short_axis_long_axis_approx_mm(mask: np.ndarray, dz_mm: float | None) -> float | None:
    if dz_mm is None or dz_mm <= 0:
        return None
    lv_areas = slice_areas_mm2(mask, LV_BLOOD_POOL, 1.0, 1.0)
    return float(np.count_nonzero(lv_areas > 0) * dz_mm) if np.any(lv_areas > 0) else None


def lv_sphericity_index(lv_edv_ml: float | None, long_axis_length_mm: float | None) -> float | None:
    if lv_edv_ml is None or long_axis_length_mm is None or long_axis_length_mm <= 0:
        return None
    sphere_volume_ml = (math.pi / 6.0) * (long_axis_length_mm**3) / 1000.0
    if sphere_volume_ml <= 0:
        return None
    return float(lv_edv_ml / sphere_volume_ml)


def any_missing(values: Iterable[object]) -> bool:
    return any(v is None for v in values)
