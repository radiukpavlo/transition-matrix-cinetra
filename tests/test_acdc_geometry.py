from __future__ import annotations

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from tmmri.semantics.features import build_acdc_b_matrix
from tmmri.ingest.geometry import geometry_from_image
from tmmri.ingest.inventory import build_acdc_inventory
from tmmri.paths import DATA_DIR

ACDC_ROOT = DATA_DIR / "raw" / "acdc" / "database"

# Plausibility limits for adult CMR including ACDC's DCM, HCM, MINF and abnormal-RV groups; they are
# deliberately wider than normal reference ranges. With the former 1 mm slice increment every LVEDV was < 39 mL.
# Ejection fractions are scale-free and not bounded here (patient098, RV group, has a ground-truth RVEF of 1.9%).
PHYSIOLOGICAL_LIMITS = {
    "lv_edv_ml": (40.0, 600.0),
    "lv_esv_ml": (5.0, 550.0),
    "rv_edv_ml": (40.0, 600.0),
    "rv_esv_ml": (5.0, 550.0),
    "lv_myocardial_mass_ed_g": (30.0, 450.0),
}
COHORT_MEDIAN_LIMITS = {
    "lv_edv_ml": (100.0, 250.0),
    "lv_myocardial_mass_ed_g": (80.0, 200.0),
}


def _image(affine: np.ndarray, zooms: tuple[float, float, float]) -> nib.Nifti1Image:
    img = nib.Nifti1Image(np.zeros((4, 4, 3), dtype=np.uint8), affine)
    img.header.set_zooms(zooms)
    return img


def test_unit_scaled_affine_uses_header_slice_spacing() -> None:
    # ACDC layout: sform diag(-1, -1, 1) carries orientation only; pixdim holds the voxel size.
    geom = geometry_from_image(_image(np.diag([-1.0, -1.0, 1.0, 1.0]), (1.5625, 1.5625, 10.0)))
    assert geom.slice_increment_mm == pytest.approx(10.0)
    assert geom.z_spacing_method == "nifti_header_pixdim_unscaled_affine"


def test_voxel_scaled_oblique_affine_keeps_affine_projection() -> None:
    # M&Ms layout: an oblique affine whose column norms equal the header zooms.
    affine = np.array(
        [
            [-0.382, 0.370, -7.443, 36.761],
            [-0.686, -0.073, 4.987, 138.483],
            [-0.130, -0.701, -4.442, 231.941],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    norms = np.linalg.norm(affine[:3, :3], axis=0)
    geom = geometry_from_image(_image(affine, tuple(float(n) for n in norms)))
    assert geom.slice_increment_mm == pytest.approx(float(np.linalg.norm(affine[:3, 2])))
    assert geom.z_spacing_method == "nifti_affine_projection"


def test_affine_slice_increment_wins_over_header_thickness_when_in_plane_matches() -> None:
    # A metric affine may encode the slice increment (thickness plus gap) while pixdim holds the thickness.
    geom = geometry_from_image(_image(np.diag([1.25, 1.25, 10.0, 1.0]), (1.25, 1.25, 8.0)))
    assert geom.slice_increment_mm == pytest.approx(10.0)
    assert geom.z_spacing_method == "nifti_affine_projection"


@pytest.fixture(scope="module")
def acdc_b_matrix() -> pd.DataFrame:
    if not (ACDC_ROOT / "training" / "patient001" / "Info.cfg").is_file():
        pytest.skip("ACDC raw data are not restored (docs/DATA_ACQUISITION_AGENTS_PROMPT.md; v5 archive: docs/history/experiment_machine_v5/transfer_archive.md)")
    inventory = build_acdc_inventory(ACDC_ROOT)
    return build_acdc_b_matrix(inventory, pd.DataFrame({"patient_id": [], "split": []}))


def test_acdc_slice_increment_matches_header_pixdim(acdc_b_matrix: pd.DataFrame) -> None:
    assert len(acdc_b_matrix) == 150
    header_z = [float(nib.load(str(path)).header.get_zooms()[2]) for path in acdc_b_matrix["raw_ed_image_path"]]
    assert acdc_b_matrix["slice_increment_mm"].to_numpy() == pytest.approx(header_z)
    assert set(acdc_b_matrix["slice_increment_mm"]) <= {5.0, 6.5, 7.0, 10.0}


def test_acdc_volumes_are_physiological(acdc_b_matrix: pd.DataFrame) -> None:
    for column, (low, high) in PHYSIOLOGICAL_LIMITS.items():
        values = acdc_b_matrix.set_index("patient_id")[column]
        outside = values[(values < low) | (values > high) | values.isna()]
        assert outside.empty, f"{column} outside [{low}, {high}]: {outside.round(1).to_dict()}"
    for column, (low, high) in COHORT_MEDIAN_LIMITS.items():
        assert low <= acdc_b_matrix[column].median() <= high, column
    assert (acdc_b_matrix["lv_esv_ml"] < acdc_b_matrix["lv_edv_ml"]).all()
    assert (acdc_b_matrix["rv_esv_ml"] < acdc_b_matrix["rv_edv_ml"]).all()
