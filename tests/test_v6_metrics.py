"""v6 metrics, primary estimand, multiplicity, route metrics and the audited predictor (SPECIFICATION.md 6.0.0,
Sections 3.3, 8 and 13; Decisions D5, D18). Synthetic algorithm controls only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import REPO, SEED_SYNTHETIC

from tmmri import classification as cl
from tmmri import metrics as mt


def _load(name):
    with (REPO / "configs" / name).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def bcfg():
    return _load("bootstrap.yaml")


FAMILY = ["lv_edv_ml", "lv_esv_ml", "lv_ef_percent", "rv_edv_ml", "rv_esv_ml", "rv_ef_percent", "lv_myocardial_mass_ed_g"]


def _family_fixture(n=50, seed=SEED_SYNTHETIC):
    rng = np.random.default_rng(seed)
    truth = pd.DataFrame({c: 100 + 20 * rng.normal(size=n) for c in FAMILY})
    mean = pd.DataFrame({c: np.full(n, 100.0) for c in FAMILY})
    ridge = truth + 8 * rng.normal(size=truth.shape)
    sd = pd.Series({c: 20.0 for c in FAMILY})
    return truth, ridge, mean, sd


def test_wilson_interval_including_zero_events():
    lo, hi = mt.wilson_interval(28, 30)
    assert round(lo, 2) == 0.79 and hi < 1.0
    assert mt.wilson_interval(0, 10)[0] == 0.0 and mt.wilson_interval(10, 10)[1] == 1.0
    assert all(np.isnan(v) for v in mt.wilson_interval(0, 0))


def test_per_target_errors_use_the_fit_partition_sd():
    truth = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan], "b": [10.0, 10.0, 10.0, 10.0]})
    pred = pd.DataFrame({"a": [1.5, 2.0, 2.0, 4.0], "b": [11.0, 9.0, 10.0, 12.0]})
    out = mt.per_target_errors(truth, pred, pd.Series({"a": 2.0, "b": 4.0})).set_index("target")
    assert out.loc["a", "n"] == 3 and out.loc["a", "mae"] == pytest.approx(0.5) and out.loc["a", "nmae_fit_sd"] == pytest.approx(0.25)
    assert out.loc["b", "nmae_fit_sd"] == pytest.approx(1.0 / 4.0)  # the evaluated split has SD 0 for b: never used


def test_primary_rho_point_ci_and_n_jobs_invariance(bcfg):
    truth, ridge, mean, sd = _family_fixture()
    pb = bcfg["primary_test"]
    kw = dict(family=FAMILY, n_boot=200, seed=bcfg["fixed_model_bootstrap"]["seed"], cluster=None, min_estimable=pb["min_estimable_targets"])
    one = mt.primary_rho(truth, ridge, mean, sd, n_jobs=1, **kw)
    three = mt.primary_rho(truth, ridge, mean, sd, n_jobs=3, **kw)
    assert one == three
    assert one["status"] == "EVALUATED" and one["ci_lower"] <= one["rho"] <= one["ci_upper"]
    rho_j = [1 - np.mean(np.abs(ridge[c] - truth[c])) / np.mean(np.abs(mean[c] - truth[c])) for c in FAMILY]
    assert one["rho"] == pytest.approx(np.mean(rho_j))
    perfect = mt.primary_rho(truth, truth, mean, sd, **kw)
    assert perfect["rho"] == pytest.approx(1.0) and perfect["ci_lower"] == pytest.approx(1.0)
    assert mt.primary_rho(truth, mean, mean, sd, **kw)["rho"] == pytest.approx(0.0)


def test_primary_rho_withdraws_the_aggregate_below_min_estimable(bcfg):
    truth, ridge, mean, sd = _family_fixture()
    truth.loc[:, FAMILY[:3]] = np.nan
    out = mt.primary_rho(truth, ridge, mean, sd, FAMILY, 50, 1, None, bcfg["primary_test"]["min_estimable_targets"])
    assert out["status"] == "NE" and out["rho"] is None and out["n_estimable"] == 4


def test_primary_rho_resamples_patients_not_rows():
    truth, ridge, mean, sd = _family_fixture(n=40)
    cluster = np.repeat(np.arange(20), 2)
    a = mt.primary_rho(truth, ridge, mean, sd, FAMILY, 100, 3, cluster, 5)
    assert a["n_clusters"] == 20 and a["status"] == "EVALUATED"


def test_fixed_sequence_test(bcfg):
    sesoi = bcfg["primary_test"]["smallest_effect_of_interest"]
    res = mt.fixed_sequence_test([{"id": "H-primary-1", "ci_lower": 0.05}, {"id": "H-primary-2", "ci_lower": 0.5},
                                  {"id": "H-primary-2-never-analysed", "ci_lower": 0.4, "in_hierarchy": False}], sesoi)
    assert [r["status"] for r in res] == ["FAILED", "NOT_TESTED", "PASSED"]
    res = mt.fixed_sequence_test([{"id": "H-primary-1", "ci_lower": 0.2}, {"id": "H-primary-2", "ci_lower": 0.1}], sesoi)
    assert [r["status"] for r in res] == ["PASSED", "FAILED"] and res[1]["plain_superiority"] is True  # 0.1 is not > 0.1


def test_bh_within_families(bcfg):
    tests = pd.DataFrame({"family": ["F1_x"] * 3 + ["F3_y"] * 2, "p_value": [0.01, 0.04, 0.03, 0.02, 0.6]})
    out = mt.bh_within_families(tests, q=bcfg["multiplicity"]["q"])
    f1 = out[out["family"] == "F1_x"]
    assert f1["q_value"].tolist() == pytest.approx([0.03, 0.04, 0.04]) and f1["rejected"].all()
    assert out[out["family"] == "F3_y"]["q_value"].tolist() == pytest.approx([0.04, 0.6])
    with pytest.raises(ValueError):
        mt.bh_within_families(tests, q=0.05, families=bcfg["multiplicity"]["families"])


def test_route_metrics_and_audit_tax(bcfg):
    routes = pd.DataFrame(
        {
            "sample_id": [f"S{i}" for i in range(6)],
            "route": ["EXACT_ACCEPT", "EXACT_ACCEPT", "EXACT_ACCEPT", "CONFLICT_ABSTAIN", "QC_ABSTAIN", "NO_MATCH_ABSTAIN"],
            "reason_code": [None, None, None, "MATCH_CONFLICTING_CONSEQUENTS", "QC_MISSING_LV_BLOOD_POOL", "MATCH_NONE"],
            "label": ["DCM", "NOR", "HCM", None, None, None],
            "n_fired": [1, 1, 2, 2, 0, 0],
            "conflict_encountered": [False, False, False, True, False, False],
        }
    )
    reference = pd.Series(["DCM", "HCM", None, "NOR", "NOR", "RV"], index=routes["sample_id"])
    d_model = pd.Series(["DCM", "NOR", "NOR", "NOR", "RV", "RV"], index=routes["sample_id"])
    m = mt.route_metrics(routes, reference, d_model)
    assert m["activation"] == pytest.approx(4 / 6) and m["accepted_coverage"] == pytest.approx(0.5) and m["abstention"] == pytest.approx(0.5)
    assert m["covered_fidelity"] == pytest.approx(2 / 3) and m["faithful_coverage"] == pytest.approx(2 / 6)
    assert m["covered_reference_accuracy"] == pytest.approx(0.5) and m["n_accepted_reference_known"] == 2
    assert m["audit_tax"]["n_same_subset"] == 2 and m["audit_tax"]["base_reference_accuracy"] == pytest.approx(0.5)
    assert m["abstention_by_reason"]["MATCH_NONE"]["n"] == 1 and m["conflict_rate"] == pytest.approx(1 / 6)
    strata = pd.Series(["A", "A", "B", "B", "B", "B"])
    s = mt.stratified_route_metrics(routes, reference.to_numpy(), d_model.to_numpy(), strata, min_cases=bcfg["subgroup_reporting_thresholds"]["min_cases_per_stratum"])
    assert s["descriptive_only"].all() and s["n"].sum() == 6


def test_formal_predictor_d_model(bcfg):
    fcfg = _load("formal.yaml")
    rng = np.random.default_rng(SEED_SYNTHETIC)
    classes = fcfg["audited_predictor"]["classes"]
    centers = rng.normal(scale=6.0, size=(len(classes), 10))
    y = np.repeat(classes, 20)
    A = centers[np.repeat(np.arange(len(classes)), 20)] + rng.normal(size=(100, 10))
    A[:, 3] = 1.0  # zero-variance column is removed on the fit rows
    pred = cl.fit_formal_predictor(A, pd.Series(y), fcfg)
    assert pred.classes == classes and pred.standardizer.n_kept == 9
    proba = pred.predict_proba(A)
    assert np.allclose(proba.sum(axis=1), 1.0) and (pred.predict(A) == y).mean() > 0.95
    again = cl.FormalPredictor.from_dict(pred.to_dict())
    assert np.array_equal(again.predict_proba(A).to_numpy(), proba.to_numpy())
    metrics, bins = cl.classification_metrics_v6(y, proba, n_bins=bcfg["calibration"]["ece_bins"])
    assert set(bcfg["classification_metrics"]) <= set(metrics) and len(bins) == 10 and metrics["accuracy"] > 0.95
    tie = cl.FormalPredictor(classes=classes, standardizer=pred.standardizer, coef=np.zeros_like(pred.coef), intercept=np.zeros(len(classes)))
    assert set(tie.predict(A[:3])) == {classes[0]}  # ties broken by the fixed class order
