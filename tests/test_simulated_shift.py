"""S6 resampler, perturbed-case geometry, C3 writer and paired cluster bootstraps (protocol-v6.0.1 item 15).

Synthetic volumes only (algorithm controls, never cohort data): geometry round trip, label integrity, physical volume
and centre preservation, identity resampling, and the paired differences of rho and accuracy."""
from __future__ import annotations

import dataclasses

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from tmmri.experiments import simulated_shift as ss
from tmmri.experiments.data_centric import paired_accuracy_difference, paired_rho_difference
from tmmri.ingest.geometry import NiftiGeometry, affine_is_unscaled
from tmmri.ingest.staging import StagedCase
from tmmri.metrics import primary_rho

ACDC_SPACING = (1.5625, 1.5625, 10.0)
VENDOR_SPACING = (1.25, 1.25, 8.0)


def ellipsoid_mask(shape, spacing, centre_mm=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Project labels 1 LV (inner ellipsoid), 2 MYO (shell), 3 RV (offset ellipsoid) on a physical grid centred at 0."""
    axes = [(np.arange(n) - (n - 1) / 2.0) * s for n, s in zip(shape, spacing)]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    x, y, z = x - centre_mm[0], y - centre_mm[1], z - centre_mm[2]
    lv = (x / 22.0) ** 2 + (y / 20.0) ** 2 + (z / 38.0) ** 2
    rv = ((x - 30.0) / 16.0) ** 2 + (y / 24.0) ** 2 + (z / 34.0) ** 2
    mask = np.zeros(shape, dtype=np.int16)
    mask[rv <= 1.0] = 3
    mask[lv <= 1.6] = 2
    mask[lv <= 1.0] = 1
    return mask


def smooth_image(shape, spacing) -> np.ndarray:
    axes = [(np.arange(n) - (n - 1) / 2.0) * s for n, s in zip(shape, spacing)]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    return (100.0 + 40.0 * np.cos(x / 30.0) * np.sin(y / 25.0 + 0.3) + 0.5 * z).astype(np.float32)


def scaled_affine(spacing, origin=(-50.0, 20.0, 5.0)) -> np.ndarray:
    a = np.diag([*spacing, 1.0])
    a[:3, 3] = origin
    return a


def test_round_trip_restores_geometry_and_masks_exactly_through_a_finer_grid():
    shape = (64, 60, 10)
    mask = ellipsoid_mask(shape, ACDC_SPACING)
    fine_shape = ss.resampled_shape(shape, ACDC_SPACING, VENDOR_SPACING)
    assert fine_shape == (80, 75, 13)                                    # floor(n * s_in / s_out + 0.5)
    fine = ss.resample_volume(mask, ACDC_SPACING, VENDOR_SPACING, interpolation="nearest")
    back = ss.resample_volume(fine, VENDOR_SPACING, ACDC_SPACING, interpolation="nearest", shape_out=shape)
    assert fine.shape == fine_shape and fine.dtype == mask.dtype and back.shape == shape
    assert set(np.unique(fine)) <= {0, 1, 2, 3} and set(np.unique(fine)) == set(np.unique(mask))   # labels copied, never blended
    np.testing.assert_array_equal(back, mask)                             # nearest neighbour through a finer grid is exact
    for affine in (scaled_affine(ACDC_SPACING), np.diag([-1.0, -1.0, 1.0, 1.0])):
        moved = ss.resampled_affine(affine, ACDC_SPACING, shape, VENDOR_SPACING, fine_shape)
        restored = ss.resampled_affine(moved, VENDOR_SPACING, fine_shape, ACDC_SPACING, shape)
        np.testing.assert_allclose(restored, affine, atol=1e-9)
        assert affine_is_unscaled(moved, VENDOR_SPACING) == affine_is_unscaled(affine, ACDC_SPACING)
    image = smooth_image(shape, ACDC_SPACING)
    image_back = ss.resample_volume(ss.resample_volume(image, ACDC_SPACING, VENDOR_SPACING, interpolation="linear"), VENDOR_SPACING,
                                    ACDC_SPACING, interpolation="linear", shape_out=shape)
    assert image_back.dtype == np.float32 and np.abs(image_back - image)[2:-2, 2:-2, 1:-1].max() < 0.02 * float(np.ptp(image))


def test_physical_volume_and_centre_are_preserved():
    shape = (64, 60, 10)
    mask = ellipsoid_mask(shape, ACDC_SPACING)
    before = ss.label_volumes_ml(mask, ACDC_SPACING)
    affine_in = scaled_affine(ACDC_SPACING)

    def centroid(m, affine):
        return affine[:3, :3] @ np.argwhere(m == 1).mean(axis=0) + affine[:3, 3]

    # in-plane resampling preserves label volumes within 3 %; through-plane nearest neighbour of 10 mm slices duplicates
    # or drops whole slices, so a changed slice increment is held to 10 %
    for target, tol in ((VENDOR_SPACING, 0.10), ((1.25, 1.25, 10.0), 0.03), ((1.8, 1.8, 10.0), 0.03), ((1.8, 1.8, 9.0), 0.10)):
        out = ss.resample_volume(mask, ACDC_SPACING, target, interpolation="nearest")
        after = ss.label_volumes_ml(out, target)
        for label in (1, 2, 3):
            assert after[label] == pytest.approx(before[label], rel=tol), (target, label)
        affine_out = ss.resampled_affine(affine_in, ACDC_SPACING, shape, target, out.shape)
        np.testing.assert_allclose(centroid(out, affine_out), centroid(mask, affine_in), atol=0.5 * max(target))   # physical centre kept
    # the field of view (physical extent) is preserved up to the rounding of the voxel count
    for n, s_in, s_out in zip(shape, ACDC_SPACING, VENDOR_SPACING):
        n_out = ss.resampled_shape((n,), (s_in,), (s_out,))[0]
        assert abs(n_out * s_out - n * s_in) <= s_out / 2 + 1e-9


def test_identity_resampling_is_exact_and_bad_geometry_is_rejected():
    shape = (24, 22, 4)
    mask = ellipsoid_mask(shape, ACDC_SPACING)
    image = smooth_image(shape, ACDC_SPACING)
    np.testing.assert_array_equal(ss.resample_volume(mask, ACDC_SPACING, ACDC_SPACING, interpolation="nearest"), mask)
    np.testing.assert_array_equal(ss.resample_volume(image, ACDC_SPACING, ACDC_SPACING, interpolation="linear"), image)
    with pytest.raises(ValueError, match="inconsistent geometry"):
        ss.resample_volume(mask, (1.5, 1.5, 0.0), VENDOR_SPACING, interpolation="nearest")
    with pytest.raises(ValueError, match="3-D"):
        ss.resample_volume(mask[..., 0], ACDC_SPACING[:2], VENDOR_SPACING[:2], interpolation="nearest")


def staged_case(shape=(24, 22, 4), spacing=ACDC_SPACING, dz=10.0) -> StagedCase:
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])                             # ACDC: orientation-only affine, voxel size in pixdim
    geom = NiftiGeometry(image_shape=shape, spacing_x_mm=spacing[0], spacing_y_mm=spacing[1], slice_thickness_mm=spacing[2], slice_gap_mm=None,
                         slice_increment_mm=dz, slice_increment_variability_mm=0.0, z_spacing_method="nifti_header_pixdim_unscaled_affine",
                         affine_matrix=affine.tolist(), num_slices=shape[2], num_frames=1)
    ed, es = ellipsoid_mask(shape, spacing), ellipsoid_mask(shape, spacing, centre_mm=(1.0, 0.0, 0.0))
    return StagedCase(sample_id="ACDC:patient999", dataset="ACDC", case_id="patient999", ed_image=smooth_image(shape, spacing),
                      es_image=smooth_image(shape, spacing) * 0.9, ed_mask=ed, es_mask=es, spacing_mm=(spacing[0], spacing[1], dz),
                      geometry_method=geom.z_spacing_method, affine=affine, geometry=geom, ed_frame_index=0, es_frame_index=2,
                      meta={"slice_thickness_mm": spacing[2]})


def test_perturbed_case_carries_consistent_geometry():
    case = staged_case()
    moved = ss.perturb_case(case, VENDOR_SPACING, vendor="A")
    assert moved.spacing_mm == VENDOR_SPACING and moved.ed_mask.shape == (30, 28, 5) == moved.es_image.shape
    assert moved.geometry_method == case.geometry_method and moved.geometry.z_spacing_method == case.geometry.z_spacing_method
    assert moved.geometry.slice_increment_mm == 8.0 and moved.geometry.image_shape == (30, 28, 5) and moved.meta["slice_thickness_mm"] == 8.0
    assert moved.meta["simulated_shift"]["vendor"] == "A" and moved.meta["simulated_shift"]["shape_in"] == [24, 22, 4]
    assert case.ed_mask.shape == (24, 22, 4)                              # the original is untouched
    np.testing.assert_allclose(np.linalg.norm(moved.affine[:3, :3], axis=0), 1.0)
    same = ss.perturb_case(case, case.spacing_mm)
    for name in ("ed_image", "es_image", "ed_mask", "es_mask"):
        np.testing.assert_array_equal(getattr(same, name), getattr(case, name))
    with pytest.raises(ValueError, match="slice increment"):
        ss.perturb_case(dataclasses.replace(case, spacing_mm=(1.5625, 1.5625, None)), VENDOR_SPACING)


def test_header_geometry_reads_spacing_and_increment(tmp_path):
    affine = np.array([[1.25, 0, 0, -10.0], [0, 1.25, 0, 5.0], [0, 0, 8.0, 30.0], [0, 0, 0, 1.0]])
    path = tmp_path / "sa.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((20, 18, 3, 4), dtype=np.float32), affine), str(path))
    geom = ss.header_geometry(path)
    assert (geom.spacing_x_mm, geom.spacing_y_mm, geom.slice_increment_mm) == (1.25, 1.25, 8.0)
    summary = {"A": {"vendor_name": "Siemens", "status": "EVALUATED", "median": {"spacing_x_mm": 1.25, "spacing_y_mm": 1.3, "slice_increment_mm": 8.0}},
               "B": {"vendor_name": "Philips", "status": "NE", "median": {}}}
    assert ss.target_spacing(summary, "A") == (1.25, 1.3, 8.0)
    with pytest.raises(ValueError, match="no header geometry"):
        ss.target_spacing(summary, "B")


def _family_frames(n=40, seed=5):
    rng = np.random.default_rng(seed)
    family = ["lv_edv_ml", "lv_esv_ml", "lv_ef_percent", "rv_edv_ml", "rv_esv_ml", "rv_ef_percent", "lv_myocardial_mass_ed_g"]
    ids = [f"ACDC:patient{i:03d}" for i in range(n)]
    truth = pd.DataFrame(rng.normal(100, 20, size=(n, len(family))), index=ids, columns=family)
    good = truth + rng.normal(0, 5, size=truth.shape)
    worse = truth + rng.normal(0, 15, size=truth.shape)
    return family, ids, truth, good, worse


def test_paired_rho_difference_is_zero_for_identical_conditions_and_matches_primary_rho():
    family, ids, truth, good, worse = _family_frames()
    mean, sd = truth.mean(), truth.std(ddof=1)
    cluster = np.array([s.split(":")[1] for s in ids], dtype=object)
    kw = dict(cluster=cluster, resample_cluster=np.array(ids, dtype=object), n_boot=300, seed=20260928, min_estimable=5)
    same = paired_rho_difference(truth, good, truth, good, mean, sd, family, **kw)
    assert same["delta_rho"] == 0.0 and same["ci_lower"] == 0.0 and same["ci_upper"] == 0.0 and same["status"] == "EVALUATED"
    ref = primary_rho(truth, good, pd.DataFrame([mean[family]] * len(ids), index=ids), sd, family, 300, 20260928, cluster, 5,
                      resample_cluster=np.array(ids, dtype=object))
    assert same["rho_reference"] == pytest.approx(ref["rho"], abs=1e-12)
    drop = paired_rho_difference(truth, good, truth, worse, mean, sd, family, **kw)
    assert drop["delta_rho"] < 0 and drop["ci_upper"] < 0 and set(drop["delta_rho_j"]) == set(family)
    assert drop["delta_rho"] == pytest.approx(drop["rho_alternative"] - drop["rho_reference"])
    holes = worse.copy()
    holes.iloc[:, :3] = np.nan                                           # fewer than 5 estimable targets in one condition
    ne = paired_rho_difference(truth, good, truth, holes, mean, sd, family, **kw)
    assert ne["status"] == "NE" and ne["delta_rho"] is None


def test_paired_accuracy_difference():
    truth = np.array(["NOR", "DCM", "HCM", "RV", "MINF"] * 8, dtype=object)
    ref = truth.copy()
    alt = truth.copy()
    alt[:8] = "NOR"
    alt[39] = None                                                       # no decision: excluded from the paired rows
    clusters = np.array([f"c{i}" for i in range(40)], dtype=object)
    out = paired_accuracy_difference(truth, ref, alt, clusters=clusters, n_boot=400, seed=20260928)
    assert out["counts"]["n_evaluable_paired"] == 39 and out["accuracy_reference"] == 1.0
    assert out["delta_accuracy"] == pytest.approx(-6 / 39) and out["ci_upper"] < 0 and out["n_changed_decisions"] == 6
    zero = paired_accuracy_difference(truth, ref, ref, clusters=clusters, n_boot=50, seed=1)
    assert zero["delta_accuracy"] == 0.0 and zero["ci_lower"] == 0.0 and zero["ci_upper"] == 0.0


def test_c3_writer_is_deterministic_and_validates_shapes(tmp_path):
    tensors = {"ed": np.full((3, 224, 224), 0.5, dtype=np.float32), "es": np.zeros((2, 224, 224), dtype=np.float32),
               "ed_geom": np.array([[-0.1, 1.0], [0.0, 1.0], [0.1, 1.0]]), "es_geom": np.array([[-0.05, 1.0], [0.05, 1.0]]),
               "crop_fraction_ed": 0.0, "crop_fraction_es": 0.0, "inplane_spacing_mm": 1.5, "transform_version": "synthetic"}
    row = ss.c3_entry("ACDC:patient999", tensors, tmp_path / "a", cluster_id="ACDC:patient999", partition="acdc_test", label=None)
    again = ss.c3_entry("ACDC:patient999", tensors, tmp_path / "b", cluster_id="ACDC:patient999", partition="acdc_test", label=None)
    assert row["npz_sha256"] == again["npz_sha256"] and list(row) == ss.C3_MANIFEST_COLUMNS and row["label"] == ""
    with np.load(tmp_path / "a" / ss.npz_filename("ACDC:patient999")) as data:
        assert data["ed"].dtype == np.float32 and data["ed"].shape == (3, 224, 224) and data["es_geom"].shape == (2, 2)
    with pytest.raises(ValueError, match="224"):
        ss.c3_entry("x", {**tensors, "ed": np.zeros((3, 100, 100))}, tmp_path / "c", cluster_id="x", partition="acdc_test", label=None)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ss.c3_entry("x", {**tensors, "es": np.full((2, 224, 224), 2.0)}, tmp_path / "c", cluster_id="x", partition="acdc_test", label=None)
    assert ss.resolve_tensor_hook("tmmri.no_such_module:fn") is None
