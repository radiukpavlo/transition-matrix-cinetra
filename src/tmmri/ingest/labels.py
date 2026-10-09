from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BACKGROUND = 0
LV_BLOOD_POOL = 1
LV_MYOCARDIUM = 2
RV_BLOOD_POOL = 3
VALID_PROJECT_LABELS = {BACKGROUND, LV_BLOOD_POOL, LV_MYOCARDIUM, RV_BLOOD_POOL}


DATASET_LABEL_MAPPINGS: dict[str, dict[int, int]] = {
    # ACDC release: 0 background, 1 RV cavity, 2 myocardium, 3 LV cavity.
    "ACDC": {0: 0, 1: 3, 2: 2, 3: 1},
    # M&Ms release follows the project convention: 1 LV, 2 MYO, 3 RV.  "MMS" is the protocol-6.0.0 dataset id
    # (complete labelled release); "MMS_100" is the legacy alias kept for the governed and legacy flows.
    "MMS": {0: 0, 1: 1, 2: 2, 3: 3},
    "MMS_100": {0: 0, 1: 1, 2: 2, 3: 3},
    # M&Ms-2 (protocol-v6.0.1 amendment; configs/cohorts.yaml mms2.label_map): identity, 1 LV, 2 MYO, 3 RV.
    "MMS2": {0: 0, 1: 1, 2: 2, 3: 3},
}


@dataclass(frozen=True)
class OverlapDiagnostics:
    overlap_lv_rv_voxels: int
    overlap_lv_myo_voxels: int
    overlap_rv_myo_voxels: int
    overlap_any_voxels: int
    overlap_resolution_applied_flag: bool


def normalize_labels(mask: np.ndarray, dataset_source: str) -> tuple[np.ndarray, set[int]]:
    mapping = DATASET_LABEL_MAPPINGS[dataset_source]
    labels = {int(x) for x in np.unique(mask)}
    unexpected = labels.difference(mapping.keys())
    out = np.zeros(mask.shape, dtype=np.uint8)
    for source_label, target_label in mapping.items():
        out[mask == source_label] = target_label
    return out, unexpected


def merge_structure_masks(
    lv_blood_pool_mask: np.ndarray,
    rv_blood_pool_mask: np.ndarray,
    myocardium_mask: np.ndarray,
) -> tuple[np.ndarray, OverlapDiagnostics]:
    lv = lv_blood_pool_mask.astype(bool)
    rv = rv_blood_pool_mask.astype(bool)
    myo = myocardium_mask.astype(bool)
    overlap_lv_rv = int(np.logical_and(lv, rv).sum())
    overlap_lv_myo = int(np.logical_and(lv, myo).sum())
    overlap_rv_myo = int(np.logical_and(rv, myo).sum())
    overlap_any = int(((lv.astype(int) + rv.astype(int) + myo.astype(int)) > 1).sum())

    final = np.zeros(lv.shape, dtype=np.uint8)
    final[myo] = LV_MYOCARDIUM
    final[rv] = RV_BLOOD_POOL
    final[lv] = LV_BLOOD_POOL
    diagnostics = OverlapDiagnostics(
        overlap_lv_rv_voxels=overlap_lv_rv,
        overlap_lv_myo_voxels=overlap_lv_myo,
        overlap_rv_myo_voxels=overlap_rv_myo,
        overlap_any_voxels=overlap_any,
        overlap_resolution_applied_flag=overlap_any > 0,
    )
    return final, diagnostics


def validate_project_labels(mask: np.ndarray) -> set[int]:
    return {int(x) for x in np.unique(mask)}.difference(VALID_PROJECT_LABELS)


def prohibit_gaussian_on_integer_labels(mask: np.ndarray) -> None:
    if np.issubdtype(mask.dtype, np.integer):
        raise ValueError("Gaussian smoothing is prohibited for integer label masks.")
