"""v6 semantics: typed B_v6.0 rows from staged cases, v6 QC severities, fit-partition outlier references
(SPECIFICATION.md 6.0.0, Sections 6, 7 and 12.4; Decisions D6, D7, D21). Synthetic phantoms only."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import REPO

from tmmri.ingest.geometry import NiftiGeometry
from tmmri.semantics import features as feat
from tmmri.semantics import qc as qcv6
from tmmri.semantics.manual_review import build_manual_review_table_v6


@pytest.fixture(scope="module")
def cohorts():
    with (REPO / "configs" / "cohorts.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass
class StagedCase:  # the attributes of DATA's load_staged_case result
    sample_id: str
    dataset: str
    case_id: str
    ed_image: np.ndarray
    ed_mask: np.ndarray
    es_image: np.ndarray
    es_mask: np.ndarray
    spacing_mm: tuple
    geometry_method: str
    affine: np.ndarray
    meta: dict = field(default_factory=dict)


DX, DY, DZ = 1.25, 1.25, 10.0  # 10 mm slice increment (ACDC-like header pixdim)


def _disk(shape, cy, cx, r):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def _phantom_mask(lv_r, myo_r, rv_r, shape=(64, 64, 10), slices=range(2, 8)):
    """Cylinders: LV blood pool disk, myocardial annulus around it, RV disk; identical on every heart slice."""
    m = np.zeros(shape, dtype=np.uint8)
    none = np.zeros(shape[:2], dtype=bool)
    lv = _disk(shape, 32, 26, lv_r) if lv_r else none
    myo = (_disk(shape, 32, 26, myo_r) & ~lv) if myo_r else none
    rv = (_disk(shape, 32, 50, rv_r) & ~_disk(shape, 32, 26, myo_r or 0)) if rv_r else none
    for z in slices:
        m[..., z][lv] = 1
        m[..., z][myo] = 2
        m[..., z][rv] = 3
    return m


def _case(case_id="patient001", dataset="ACDC", ed=None, es=None, meta=None):
    ed = _phantom_mask(10, 14, 8) if ed is None else ed
    es = _phantom_mask(7, 13, 6) if es is None else es
    base_meta = {"patient_id": case_id, "subset_name": "training", "split": "acdc_dev", "disease_label": "NOR", "ed_frame_index": 0, "es_frame_index": 9,
                 "height_cm": 172.0, "weight_kg": 70.0, "slice_thickness_mm": DZ, "raw_ed_image_path": f"training/{case_id}/{case_id}_frame01.nii.gz"}
    base_meta.update(meta or {})
    img = np.random.default_rng(7).normal(size=ed.shape)
    return StagedCase(f"{dataset}:{case_id}", dataset, case_id, img, ed, img, es, (DX, DY, DZ), "nifti_header_pixdim_unscaled_affine", np.diag([-1.0, -1.0, 1.0, 1.0]), base_meta)


def test_typed_columns_cover_every_b_v2_column(registry):
    cols = feat.typed_columns_v6(registry)
    assert len(cols) == registry["counts"]["total_typed_B_columns"] == 323
    legacy = feat.compute_b_row("ACDC", pd.Series({"case_id": "p"}), "train", _phantom_mask(10, 14, 8), _phantom_mask(7, 13, 6), _legacy_geom(), set(), None, None)
    classified = set(registry["metadata_columns"]) | set(registry["qc_columns"]) | {e["name"] for e in registry["features"]}
    assert set(legacy) == classified and len(legacy) == registry["counts"]["total_b_v2_columns"]


def _legacy_geom():
    return NiftiGeometry((64, 64, 10), DX, DY, DZ, None, DZ, 0.0, "nifti_header_pixdim_unscaled_affine", np.diag([-1.0, -1.0, 1.0, 1.0]).tolist(), 10, 1)


def test_cylinder_phantom_volumes_ef_and_legacy_parity(registry, cohorts):
    case = _case()
    row = feat.case_semantics_v6(case, registry, cohorts=cohorts)
    voxel_ml = DX * DY * DZ / 1000.0
    n_lv_ed = int(_disk((64, 64), 32, 26, 10).sum()) * 6
    n_lv_es = int(_disk((64, 64), 32, 26, 7).sum()) * 6
    assert row["lv_edv_ml"] == pytest.approx(n_lv_ed * voxel_ml) and row["lv_esv_ml"] == pytest.approx(n_lv_es * voxel_ml)
    assert row["lv_edv_ml"] == pytest.approx(6 * math.pi * 100 * voxel_ml, rel=0.02)  # the cylinder volume, 10 mm slices
    assert row["lv_ef_percent"] == pytest.approx(100.0 * (n_lv_ed - n_lv_es) / n_lv_ed)
    assert row["lv_myocardial_mass_ed_g"] == pytest.approx(1.050 * int(np.count_nonzero(case.ed_mask == 2)) * voxel_ml)
    assert row["sample_id"] == "ACDC:patient001" and row["schema_version"] == "B_v6.0" and row["slice_increment_mm"] == DZ
    assert row["lv_edv_ml__state"] == "OBSERVED" and row["lv_edv_ml__evidence"] == "E1_EXPERT_MASK_GEOMETRY+E1H_HEADER_GEOMETRY"
    assert row["aha16_seg01_wall_thickness_ed_mm"] is None and row["aha16_seg01_wall_thickness_ed_mm__state"] == "UNSUPPORTED"
    assert qcv6.assess_qc_v6(qcv6.parse_reason_tokens(row["qc_failure_reason"]), cohorts).max_severity in ("none", "informational")
    legacy = feat.compute_b_row("ACDC", pd.Series({"case_id": "patient001", "height_cm": 172.0, "weight_kg": 70.0}), "train", case.ed_mask, case.es_mask, _legacy_geom(), set(), None, None)
    for entry in registry["features"]:
        name = entry["name"]
        if row[f"{name}__state"] == "OBSERVED":
            assert row[name] == pytest.approx(legacy[name]), name
        else:
            assert legacy[name] is None, name


def test_feature_states(registry, cohorts):
    mms = _case("A1B2C3", "MMS", meta={"height_cm": None, "split": "mms_external", "subset_name": "Testing", "scanner_vendor": "Philips", "center_id": 2})
    row = feat.case_semantics_v6(mms, registry, cohorts=cohorts)
    assert row["bsa_m2__state"] == "MISSING_SOURCE" and row["lv_edv_indexed_ml_m2__state"] == "MISSING_SOURCE"
    assert "bsa_unavailable" in row["qc_failure_reason"] and row["center_id"] == 2
    tiny = _phantom_mask(0, 0, 8)  # LV blood pool absent at ED: EDV 0 < 1 mL, EF undefined
    row = feat.case_semantics_v6(_case(ed=tiny), registry, cohorts=cohorts)
    assert row["lv_ef_percent"] is None and row["lv_ef_percent__state"] == "NOT_APPLICABLE"
    same = _case(ed=_phantom_mask(10, 14, 8), es=_phantom_mask(10, 14, 8))
    row = feat.case_semantics_v6(same, registry, cohorts=cohorts)
    masked = cohorts["qc_severity"]["feature_masks"]["ed_es_masks_identical"]
    assert all(row[f"{f}__state"] == "QC_BLOCKED" and row[f] is None for f in masked)
    assert row["lv_edv_ml__state"] == "OBSERVED"


def test_missing_structure_is_critical_and_unknown_reason_fails_closed(registry, cohorts, reason_codes):
    no_rv_es = _phantom_mask(7, 13, 0)
    row = feat.case_semantics_v6(_case(es=no_rv_es), registry, cohorts=cohorts)
    a = qcv6.assess_qc_v6(qcv6.parse_reason_tokens(row["qc_failure_reason"]), cohorts, reason_codes)
    assert "missing_rv_es_present" in a.by_severity["critical"] and a.route == "QC_ABSTAIN" and "QC_MISSING_RV_BLOOD_POOL" in a.abstain_codes
    assert row["qc_manual_review_required"] is True
    for reason in ["missing_lv_ed_present", "missing_lv_es_present", "missing_myo_ed_present", "missing_myo_es_present", "missing_rv_ed_present"]:
        assert qcv6.assess_qc_v6([reason], cohorts, reason_codes).max_severity == "critical"
    bad = qcv6.assess_qc_v6(["zz_unknown_reason"], cohorts, reason_codes)
    assert bad.max_severity == "critical" and bad.abstain_codes == ("QC_UNREGISTERED_REASON",)
    with pytest.raises(qcv6.UnregisteredQCReasonError):
        qcv6.assess_qc_v6(["zz_unknown_reason"], cohorts, reason_codes, strict=True)
    assert qcv6.assess_qc_v6(["unexpected_label_value:[4|5]"], cohorts, reason_codes).abstain_codes == ("QC_UNEXPECTED_LABEL_VALUE",)
    sv = qcv6.assess_qc_v6(["lv_rv_sv_discrepancy_gt_15_ml"], cohorts, reason_codes)
    assert sv.max_severity == "moderate" and sv.route is None and sv.masked_features == ()
    labelled = _phantom_mask(10, 14, 8)
    labelled[0, 0, 0] = 5
    row = feat.case_semantics_v6(_case(ed=labelled), registry, cohorts=cohorts)
    assert "unexpected_label_value:[5]" in row["qc_failure_reason"] and row["qc_mask_labels_valid"] is False
    table = pd.DataFrame([row, feat.case_semantics_v6(_case("patient002"), registry, cohorts=cohorts)])
    review = build_manual_review_table_v6(table, cohorts, reason_codes)
    assert review["sample_id"].tolist() == ["ACDC:patient001"] and review["adjudication_status"].iloc[0] == "pending_review"


def test_outlier_reference_is_fitted_on_the_fit_partition_only(registry, cohorts, reason_codes):
    rng = np.random.default_rng(3)
    n_dev, n_ext = 40, 20  # deterministic dev spread; external rows far outside it
    cands = ["lv_edv_ml", "rv_edv_ml"]
    frame = pd.DataFrame({c: np.concatenate([150.0 + np.linspace(-15, 15, n_dev), rng.normal(400, 10, n_ext)]) for c in cands})
    for c in cands:
        frame[f"{c}__state"] = "OBSERVED"
    frame["qc_failure_reason"] = None
    frame.loc[0, "qc_failure_reason"] = "missing_lv_ed_present"
    frame.loc[0, "lv_edv_ml"] = 5000.0  # a critical row never shapes the reference
    dev = np.arange(n_dev + n_ext) < n_dev
    ref = qcv6.fit_outlier_reference(frame, registry, cohorts, fit_mask=dev, reason_codes=reason_codes)
    assert ref["n_fit_rows"] == n_dev - 1 and ref["threshold"] == 3.5 and ref["scale"] == 0.6745
    assert ref["features"]["lv_edv_ml"]["median"] == pytest.approx(np.median(frame.loc[1:n_dev - 1, "lv_edv_ml"]))
    shifted = frame.copy()
    shifted.loc[~dev, cands] += 1000.0
    assert qcv6.fit_outlier_reference(shifted, registry, cohorts, fit_mask=dev, reason_codes=reason_codes) == ref
    flagged = qcv6.apply_outlier_reference(frame, ref, cohorts, reason_codes=reason_codes)
    assert flagged.loc[~dev, "qc_extreme_robust_outlier_flag"].all() and not flagged.loc[1:n_dev - 1, "qc_extreme_robust_outlier_flag"].any()
    assert flagged.loc[n_dev, "qc_failure_reason"] == "extreme_robust_outlier" and flagged.loc[n_dev, "qc_extreme_outlier_features"] == "lv_edv_ml,rv_edv_ml"
    again = qcv6.apply_outlier_reference(flagged, ref, cohorts, reason_codes=reason_codes)
    pd.testing.assert_frame_equal(again, flagged)  # idempotent


def test_semantics_table_is_sorted_and_invariant_to_n_jobs(registry, cohorts):
    cases = [_case("patient003"), _case("patient001"), _case("A1B2C3", "MMS", meta={"split": "mms_external"})]
    one = feat.build_semantics_table_v6(cases, registry, n_jobs=1, cohorts=cohorts)
    two = feat.build_semantics_table_v6(cases[::-1], registry, n_jobs=2, cohorts=cohorts)
    assert one["sample_id"].tolist() == ["ACDC:patient001", "ACDC:patient003", "MMS:A1B2C3"] and one.shape[1] == 323
    pd.testing.assert_frame_equal(one, two)
    by_key = {c.case_id: c for c in cases}
    loaded = feat.build_semantics_table_v6((by_key.__getitem__, list(by_key)), registry, n_jobs=1, cohorts=cohorts)
    pd.testing.assert_frame_equal(loaded, one)


def test_equivariance_v6_selection_and_config_epsilons(registry, cohorts):
    from tmmri import equivariance as eq

    with (REPO / "configs" / "equivariance.yaml").open(encoding="utf-8") as fh:
        ecfg = yaml.safe_load(fh)
    ids = [f"patient{i:03d}" for i in range(1, 101)]
    chosen = eq.select_equivariance_cases(ids[::-1], ecfg)
    assert chosen == eq.select_equivariance_cases(ids, ecfg) and len(set(chosen)) == ecfg["case_selection"]["n_cases"]
    assert chosen != ids[: len(chosen)]  # not the v5 head() selection
    steps = {name: eps for name, _, _, eps in eq.transformation_steps(ecfg)}
    assert steps["translation_x"] == 1.0 and steps["rotation"] == 2.0 and steps["intensity_scale"] == pytest.approx(0.05)
    with pytest.raises(ValueError):
        eq.select_equivariance_cases(ids, {**ecfg, "feeds_primary_rulebook": True})
    case = _case()
    names = ["lv_edv_ml", "lv_ef_percent"]

    def semantic_fn(c, ed, es):
        row = feat.case_semantics_v6(StagedCase(c.sample_id, c.dataset, c.case_id, c.ed_image, ed, c.es_image, es, c.spacing_mm, c.geometry_method, c.affine, c.meta), registry, cohorts=cohorts)
        return np.array([row[n] if row[n] is not None else np.nan for n in names])

    def formal_fn(ed_img, es_img, ed_m, es_m):
        return np.array([np.mean(ed_m == 1), np.mean(es_m == 1)])

    def predict_fn(a):  # an exact linear map for these two descriptors: finite-action defect of LV EDV is zero
        voxels = np.prod(case.ed_mask.shape) * DX * DY * DZ / 1000.0
        return np.column_stack([a[:, 0] * voxels, 100.0 * (a[:, 0] - a[:, 1]) / a[:, 0]])

    cases_df, targets_df = eq.equivariance_diagnostics_v6([case], formal_fn, semantic_fn, predict_fn, ecfg, target_names=names)
    assert set(cases_df["transformation"]) == set(steps) and not cases_df["feeds_primary_rulebook"].any()
    intensity = cases_df.set_index("transformation").loc["intensity_scale"]
    assert intensity["semantic_generator_norm"] == 0.0  # the mask is unchanged by intensity scaling
    assert set(targets_df["target"]) == set(names)
