"""Post-freeze exploratory block B10-B20 (protocol-v6.0.2): hand-checked pure functions, the source-file stage map and the
worked-example slot map on a fake repository (synthetic values only; never cohort data)."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
import yaml

from test_v6_cli import REPO
from tmmri.config import load_study
from tmmri.experiments.postfreeze import b10_analytical_ef as b10
from tmmri.experiments.postfreeze import b15_rule_ledger as b15
from tmmri.experiments.postfreeze import b16_coverage_fidelity as b16
from tmmri.experiments.postfreeze import b17_concept_reliability as b17
from tmmri.experiments.postfreeze import b19_majority_class as b19
from tmmri.experiments.postfreeze import b20_worked_example as b20
from tmmri.experiments.postfreeze.common import stage_of
from tmmri.experiments.postfreeze.estimates import paired_rho, risk_ratio, weighted_auroc, wilson


@pytest.fixture(scope="module")
def study():
    return load_study(REPO / "configs" / "study_v6.yaml")


def test_stage_of_maps_every_source_file_to_its_record():
    expected = {"semantics/B_ACDC.parquet": "build-semantics:ACDC", "formal/A_MMS.parquet": "extract-formal:MMS",
                "oof/bhat_oof_ACDC_dev.parquet": "fit-transition", "oof/inner_oof_fold1.parquet": "gate-rules",
                "reconstructed/bhat_acdc_test_decoded.parquet": "evaluate:acdc_test",
                "reconstructed/stability_d_model_mms_external.parquet": "evaluate:mms_external",
                "inference/routes_measured_oracle_acdc_test.parquet": "evaluate:acdc_test",
                "inference/audited_predictor/acdc_test/predictions.parquet": "summarize-predictor-extended:acdc_test",
                "metrics/primary_rho_mms2_external.json": "evaluate:mms2_external", "metrics/h_primary_3.json": "evaluate:mms2_external",
                "metrics/predictor_acdc_test.json": "summarize-predictor", "metrics/predictor_mms2_external.json": "summarize-predictor:mms2_external",
                "metrics/confirmatory_h4/h4_mms_external.json": "evaluate-h4:mms_external", "metrics/ablations/transition_variants.json": "run-ablations",
                "rules/d_model/rulebook_induced.json": "induce-rules", "rules/d_model/gate.parquet": "gate-rules",
                "rules/semantic_eligibility.parquet": "check-semantic-eligibility", "thresholds/d_model_wedd_gaussian_mm_v1.json": "discretize",
                "comparators/cart_f8/evaluation/predictions_acdc_test.parquet": "evaluate-cart-surrogate:acdc_test",
                "comparators/cart_f8/nested_oof_ledger.parquet": "fit-cart-surrogate", "trust/acdc_test/s2_scores.parquet": "evaluate-trust:acdc_test",
                "trust/s1_thresholds.json": "fit-trust", "contracts/split_manifest.parquet": "validate-cohort",
                "contracts/split_manifest_MMS2.parquet": "validate-cohort:MMS2", "release_manifest.json": None}
    assert {rel: stage_of(rel) for rel in expected} == expected


def test_analytical_ef_by_hand():
    a = pd.DataFrame({"fd_l1_fraction_ed": [0.5, 0.0, np.nan, 0.4], "fd_l1_fraction_es": [0.2, 0.1, 0.1, 0.1],
                      "fd_l3_fraction_ed": [0.25, 0.3, 0.3, 0.3], "fd_l3_fraction_es": [0.05, 0.15, 0.15, 0.15]}, index=list("abcd"))
    b = pd.DataFrame({"lv_edv_ml": [120.0, 100.0, 100.0, 0.5], "lv_edv_ml__state": "OBSERVED", "rv_edv_ml": [110.0] * 4,
                      "rv_edv_ml__state": ["OBSERVED", "OBSERVED", "OBSERVED", "MISSING_SOURCE"]}, index=list("abcd"))
    ef, states = b10.analytical_ef(a, b)
    assert ef.loc["a", "lv_ef_percent"] == pytest.approx(60.0) and ef.loc["a", "rv_ef_percent"] == pytest.approx(80.0)
    assert list(states["lv_ef_percent"]) == ["ANALYTICAL", "NE_ZERO_ED_FRACTION", "NE_NONFINITE_FRACTION", "NOT_APPLICABLE"]
    assert ef["lv_ef_percent"].iloc[1:].isna().all() and states.loc["d", "rv_ef_percent"] == "ANALYTICAL"   # unobserved EDV: no veto


def test_geometry_covariates_by_hand():
    from tmmri.experiments.postfreeze import b12_geometry as b12

    b = pd.DataFrame({"spacing_x_mm": [1.5, 2.0], "spacing_y_mm": [1.5, 2.0], "slice_increment_mm": [10.0, 5.0]}, index=["ACDC:p1", "ACDC:p2"])
    cases = {"p1": {"image_shape": [200, 100, 10, 30]}, "p2": {"num_slices": 8}}
    geo = b12.geometry(b, cases, list(b.index))
    assert geo.loc["ACDC:p1", "voxel_volume_ml"] == pytest.approx(0.0225) and geo.loc["ACDC:p1", "n_voxels"] == 200_000
    assert geo.loc["ACDC:p1", "log_imaged_volume_ml"] == pytest.approx(math.log(4500.0)) and geo.loc["ACDC:p1", "log_slice_count"] == pytest.approx(math.log(10))
    assert np.isnan(geo.loc["ACDC:p2", "log_imaged_volume_ml"]) and geo.loc["ACDC:p2", "log_slice_count"] == pytest.approx(math.log(8))
    a = pd.DataFrame({column: [0.01, 0.02] for column, _ in b12.VOLUMES.values()}, index=b.index)
    vol = b12.analytical_volumes(a, geo)
    target, (_, density) = next(iter(b12.VOLUMES.items()))
    assert vol.loc["ACDC:p1", target] == pytest.approx(0.01 * 4500.0 * density)


def test_wilson_risk_ratio_and_auroc_by_hand():
    w = wilson(8, 10)
    z = 1.959963984540054
    centre, half = (0.8 + z * z / 20) / (1 + z * z / 10), z * math.sqrt(0.8 * 0.2 / 10 + z * z / 400) / (1 + z * z / 10)
    assert (w["estimate"], w["ci_lower"], w["ci_upper"]) == pytest.approx((0.8, centre - half, centre + half))
    assert wilson(0, 0)["estimate"] is None
    rr = risk_ratio(10, 10, 5, 15)
    se = math.sqrt(1 / 10 - 1 / 20 + 1 / 5 - 1 / 20)
    assert (rr["estimate"], rr["ci_lower"], rr["ci_upper"]) == pytest.approx((2.0, 2 * math.exp(-z * se), 2 * math.exp(z * se)))
    assert risk_ratio(0, 5, 3, 7)["status"] == "NE"
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(3)
    s, y = rng.integers(0, 5, 60).astype(float), rng.integers(0, 2, 60).astype(bool)
    assert weighted_auroc(s, y) == pytest.approx(roc_auc_score(y, s))
    assert weighted_auroc(s, y, np.full(60, 2.0)) == pytest.approx(weighted_auroc(np.repeat(s, 2), np.repeat(y, 2)))
    assert math.isnan(weighted_auroc(s, np.zeros(60, dtype=bool)))


def test_paired_rho_reproduces_the_recorded_primary_estimand(study):
    from tmmri.metrics import primary_rho

    rng = np.random.default_rng(20261028)
    targets = list(study.primary_family())
    ids = [f"ACDC:patient{i:03d}" for i in range(40)]
    y = pd.DataFrame(rng.normal(50, 10, (40, len(targets))), index=ids, columns=targets)
    pred = y + rng.normal(0, 4, y.shape)
    mean = y.mean() + 1.0
    sd = y.std()
    null = pd.DataFrame([mean] * len(ids), index=ids)
    units = pd.Series([s.split(":", 1)[1] for s in ids], index=ids)
    boot = {"replicates": 60, "seed": 20260928, "level": 0.95}
    fixed = study.bootstrap["fixed_model_bootstrap"]
    ours = paired_rho(y, {"ridge": (pred, null)}, sd, {"rho": targets}, units=units, clusters=pd.Series(ids, index=ids), boot=boot, study=study,
                      min_cases=20)["arms"]["ridge"]["rho"]["rho"]
    ref = primary_rho(y, pred, null, sd, targets, 60, 20260928, units.to_numpy(dtype=object), int(study.bootstrap["primary_test"]["min_estimable_targets"]),
                      min_valid_fraction=float(fixed["min_valid_fraction"]), resample_cluster=np.asarray(ids, dtype=object))
    assert (ours["estimate"], ours["ci_lower"], ours["ci_upper"]) == pytest.approx((ref["rho"], ref["ci_lower"], ref["ci_upper"]))
    # Paired differences use the cases both arms score: an identical arm missing half its rows differs by exactly 0.
    holes = pred.copy()
    holes.iloc[::2] = np.nan
    res = paired_rho(y, {"ridge": (pred, null), "holes": (holes, null)}, sd, {"rho": targets}, units=units, clusters=pd.Series(ids, index=ids),
                     boot=boot, study=study, min_cases=20, reference="ridge")
    diff = res["differences"]["holes"]
    assert set(diff["n_common"].values()) == {20} and set(res["arms"]["ridge"]["n_scored"].values()) == {40}
    assert all(diff["nmae"][t]["estimate"] == 0 and diff["nmae"][t]["ci_lower"] == 0 == diff["nmae"][t]["ci_upper"] for t in targets)
    assert diff["rho"]["rho"]["estimate"] == 0


def test_rule_ledger_thresholds_are_the_frozen_policy(study):
    order = ["concept_eligibility", "granule_support_min_5", "granule_confidence_min_0.90", "crossfit_reproduction", "pooled_matches_min_5",
             "wilson_lower_min_0.60"]
    assert b15.pipeline_thresholds(order, study.rule_policy) == {"granule_support": 5.0, "granule_confidence": 0.9, "pooled_matches": 5.0,
                                                                 "wilson_lower": 0.6}
    with pytest.raises(ValueError, match="differs"):
        b15.pipeline_thresholds([*order[:5], "wilson_lower_min_0.50"], study.rule_policy)
    with pytest.raises(ValueError, match="pipeline_order"):
        b15.pipeline_thresholds(order[::-1], study.rule_policy)


def test_coverage_fidelity_reliability_and_majority_by_hand():
    idx = list("abcdef")
    label = pd.Series(["NOR", None, "DCM", "DCM", None, "HCM"], index=idx)
    processed = pd.Series([True, True, True, True, False, True], index=idx)
    d_model = pd.Series(["NOR", "NOR", "DCM", "HCM", "NOR", None], index=idx)
    reference = pd.Series(["NOR", "DCM", "MINF", "DCM", None, "HCM"], index=idx)
    m = b16.system_metrics(label, processed, d_model, reference)
    assert (m["n_processed"], m["n_accepted"]) == (5, 4)
    assert (m["coverage"]["k"], m["coverage"]["n"], m["fidelity"]["k"], m["fidelity"]["n"]) == (4, 5, 2, 3)
    assert (m["accuracy"]["k"], m["accuracy"]["n"], m["d_model_accuracy_same_rows"]["k"], m["d_model_accuracy_same_rows"]["n"]) == (3, 4, 1, 3)
    err = pd.DataFrame({"t1": [0.1, 0.5, np.nan], "t2": [0.2, np.nan, np.nan]}, index=list("xyz"))
    lab = b17.large_error_labels(err, pd.Series({"t1": 0.3, "t2": 0.1}))
    assert lab["t1"].tolist()[:2] == [0.0, 1.0] and lab["any_large"].tolist()[:2] == [1.0, 1.0] and np.isnan(lab.loc["z", "any_large"])
    cells = pd.DataFrame({"symbol_error": [True, True, False, True, False, False], "unstable": [True, False, True, False, False, False]})
    t = b17.two_by_two(cells)
    assert (t["unstable_error"], t["unstable_no_error"], t["stable_error"], t["stable_no_error"]) == (1, 1, 2, 2)
    assert t["sensitivity"] == pytest.approx(1 / 3) and t["specificity"] == pytest.approx(2 / 3) and t["risk_ratio"] == pytest.approx(0.5 / 0.5)
    maj = b19.majority(["DCM", "HCM", "NOR"], [[3, 1, 0], [2, 2, 1], [0, 0, 5]])
    assert maj["majority_class"] == "HCM" and maj["n_evaluable"] == 14 and maj["majority_rate"]["k"] == 5   # ties by class order
    assert maj["uniform_chance_rate"] == pytest.approx(1 / 3)


def _fake_repo(root, folders):
    spec = {"cases": [{"key": f"example_{k}"} for k in (2, 1, 3, 4, 5)]}
    scans = {"dataset": "ACDC test release only; no M&Ms data",
             "examples": [{"example": k, "acdc_folder": f} for k, f in enumerate(folders, start=1)]}
    (root / b20.SPEC).parent.mkdir(parents=True)
    (root / b20.SPEC).write_text(yaml.safe_dump(spec), encoding="utf-8")
    (root / b20.SCANS).parent.mkdir(parents=True)
    (root / b20.SCANS).write_text(json.dumps(scans), encoding="utf-8")


def test_worked_example_slots_follow_the_figure_order(tmp_path):
    folders = [f"data/raw/acdc/database/testing/patient{900 + k}" for k in range(1, 7)]
    _fake_repo(tmp_path, folders)
    slots = b20.slot_cases(tmp_path)
    assert list(slots) == ["a", "b1", "b2", "b3", "b4"]
    assert slots == {"a": "ACDC:patient902", "b1": "ACDC:patient901", "b2": "ACDC:patient903", "b3": "ACDC:patient904", "b4": "ACDC:patient905"}
    bad = tmp_path / "bad"
    _fake_repo(bad, [f.replace("testing", "training") for f in folders])
    with pytest.raises(ValueError, match="testing folder"):
        b20.slot_cases(bad)
