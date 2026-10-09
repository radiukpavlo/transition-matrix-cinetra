from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from tmmri.semantics.features import compute_b_row, feature_dictionary
from tmmri.ingest.geometry import NiftiGeometry, z_spacing_from_positions
from tmmri.ingest.inventory import build_mms_inventory
from tmmri.io import write_parquet
from tmmri.ingest.labels import LV_BLOOD_POOL, LV_MYOCARDIUM, RV_BLOOD_POOL, merge_structure_masks, prohibit_gaussian_on_integer_labels
from tmmri.semantics.measurements import (
    lv_sphericity_index,
    mask_volume_ml,
    shape_summary,
    wall_thickening_percent,
)
from tmmri.transition import assert_transition_shape, fit_train_only_preprocessing, transform_b_centered
from tmmri.splits import select_mms_100


def _geom() -> NiftiGeometry:
    return NiftiGeometry(
        image_shape=(8, 8, 3),
        spacing_x_mm=2.0,
        spacing_y_mm=2.0,
        slice_thickness_mm=5.0,
        slice_gap_mm=None,
        slice_increment_mm=5.0,
        slice_increment_variability_mm=0.0,
        z_spacing_method="nifti_affine_projection",
        affine_matrix=[[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 5, 0], [0, 0, 0, 1]],
        num_slices=3,
        num_frames=1,
    )


def test_synthetic_cuboid_volume_known_spacing() -> None:
    mask = np.zeros((4, 4, 2), dtype=np.uint8)
    mask[1:3, 1:3, :] = LV_BLOOD_POOL
    assert mask_volume_ml(mask, LV_BLOOD_POOL, 2.0, 3.0, 5.0) == pytest.approx(0.24)


def test_z_axis_distance_from_positions() -> None:
    positions = np.array([[0, 0, 0], [0, 0, 6], [0, 0, 12]], dtype=float)
    dz, variability = z_spacing_from_positions(positions, np.array([0, 0, 1], dtype=float))
    assert dz == pytest.approx(6.0)
    assert variability == pytest.approx(0.0)


def test_mask_conflict_resolution_priority() -> None:
    lv = np.ones((2, 2), dtype=bool)
    rv = np.ones((2, 2), dtype=bool)
    myo = np.ones((2, 2), dtype=bool)
    final, diag = merge_structure_masks(lv, rv, myo)
    assert np.all(final == LV_BLOOD_POOL)
    assert diag.overlap_any_voxels == 4
    assert diag.overlap_resolution_applied_flag


def test_gaussian_smoothing_prohibited_for_integer_labels() -> None:
    with pytest.raises(ValueError):
        prohibit_gaussian_on_integer_labels(np.zeros((3, 3), dtype=np.uint8))


def test_wall_thickening_uses_ed_denominator() -> None:
    assert wall_thickening_percent(10.0, 15.0) == pytest.approx(50.0)
    assert wall_thickening_percent(0.1, 15.0) is None


def test_aha_lv_only_policy_defaults_to_null_segments() -> None:
    mask_ed = np.zeros((8, 8, 3), dtype=np.uint8)
    mask_es = np.zeros((8, 8, 3), dtype=np.uint8)
    mask_ed[2:6, 2:6, :] = LV_BLOOD_POOL
    mask_es[3:5, 3:5, :] = LV_BLOOD_POOL
    mask_ed[1:7, 1:7, :] = np.where(mask_ed[1:7, 1:7, :] == 0, LV_MYOCARDIUM, mask_ed[1:7, 1:7, :])
    mask_es[2:6, 2:6, :] = np.where(mask_es[2:6, 2:6, :] == 0, LV_MYOCARDIUM, mask_es[2:6, 2:6, :])
    row = pd.Series({"case_id": "case", "patient_id": "case", "ed_frame": 0, "es_frame": 1})
    out = compute_b_row("MMS_100", row, "mms_100_external", mask_ed, mask_es, _geom(), set(), None, None)
    assert out["aha_mapping_feasible_flag"] is False
    assert out["aha16_seg01_wall_thickness_ed_mm"] is None


def test_eccentricity_invalid_for_annular_myocardium_in_feature_dictionary() -> None:
    df = pd.DataFrame({"myocardium_eccentricity_valid_flag": [False]})
    assert feature_dictionary(df).loc[0, "transition_target_eligible"] == np.False_


def test_perimeter_circle_approximation() -> None:
    y, x = np.ogrid[:80, :80]
    radius = 20
    circle = ((x - 40) ** 2 + (y - 40) ** 2) <= radius**2
    mask = np.zeros((80, 80, 1), dtype=np.uint8)
    mask[:, :, 0] = circle.astype(np.uint8) * LV_BLOOD_POOL
    summary = shape_summary(mask, LV_BLOOD_POOL, 1.0, 1.0)
    compactness = summary["compactness_mean"]
    assert compactness is not None
    assert 0.75 <= compactness <= 1.1


def test_lv_sphericity_formula() -> None:
    assert lv_sphericity_index(100.0, 60.0) == pytest.approx(100.0 / ((math.pi / 6.0) * 60.0**3 / 1000.0))


def test_transition_shape_contract() -> None:
    a = np.ones((5, 3))
    t = np.ones((3, 2))
    b = np.ones((5, 2))
    assert_transition_shape(a, t, b)
    with pytest.raises(AssertionError):
        assert_transition_shape(a, t, np.ones((5, 3)))


def test_train_only_preprocessing_does_not_leak_validation_values() -> None:
    a_train = pd.DataFrame({"a_0000": [0.0, 1.0, 2.0]})
    b_train = pd.DataFrame({"lv_edv_ml": [10.0, None, 30.0]})
    state = fit_train_only_preprocessing(a_train, b_train, ["a_0000"], ["lv_edv_ml"])
    assert state.b_medians["lv_edv_ml"] == pytest.approx(20.0)
    validation = pd.DataFrame({"lv_edv_ml": [1000.0]})
    centered = transform_b_centered(validation, state)
    assert centered.shape == (1, 1)
    assert state.b_clip_upper["lv_edv_ml"] < 1000.0


def test_parquet_serialization_uses_snappy(tmp_path) -> None:
    out = write_parquet(pd.DataFrame({"x": [1, 2]}), tmp_path / "x.parquet")
    meta = pq.ParquetFile(out).metadata
    assert meta.row_group(0).column(0).compression == "SNAPPY"


def test_empty_mms_inventory_preserves_fail_closed_contract(tmp_path) -> None:
    mms_root = tmp_path / "mms"
    mms_root.mkdir()
    metadata = mms_root / "metadata.csv"
    metadata.write_text(
        "External code,VendorName,Vendor,Centre,ED,ES,Pathology,Height,Weight,Age,Sex\n",
        encoding="utf-8",
    )

    inventory = build_mms_inventory(mms_root, metadata)

    assert inventory.empty
    assert {"image_path", "mask_path", "eligible_for_mms_100"}.issubset(inventory.columns)
    assert inventory["eligible_for_mms_100"].dtype == bool
    assert select_mms_100(inventory).empty
