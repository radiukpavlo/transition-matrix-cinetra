"""WEDD edge behaviour, planted-cut recovery, stability gate and rough-set definitions (SPECIFICATION.md 6.0.0,
Sections 11.3-11.6; Decisions D11-D13). Synthetic algorithm controls only."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import REPO, SEED_SYNTHETIC

from tmmri import rough_sets as rs
from tmmri import wedd


def _load(name):
    with (REPO / "configs" / name).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def wcfg():
    return _load("wedd.yaml")


@pytest.fixture(scope="module")
def rcfg():
    return _load("rule_policy.yaml")


def _two_clusters(n_half, seed, mirrored=False):
    rng = np.random.default_rng(seed)
    left = rng.normal(-2.0, 0.3, n_half)
    right = -left if mirrored else rng.normal(2.0, 0.3, n_half)
    x = np.concatenate([left, right])
    return x, np.where(x > 0, "HIGH", "LOW")


def test_planted_cut_recovered_at_zero_noise(wcfg):
    x, d = _two_clusters(300, SEED_SYNTHETIC, mirrored=True)
    cs = wedd.fit_cuts(pd.DataFrame({"f": x}), d, "wedd_gaussian_mm_v1", wcfg)
    assert cs.features["f"].cuts == [pytest.approx(0.0, abs=1e-12)]  # the gap midpoint; pure children stop growth
    x, d = _two_clusters(300, SEED_SYNTHETIC + 1)
    cut = wedd.fit_cuts(pd.DataFrame({"f": x}), d, "wedd_gaussian_mm_v1", wcfg).features["f"].cuts
    assert len(cut) == 1 and x[x < 0].max() < cut[0] < x[x > 0].min()


def test_bins_are_right_closed_at_every_cut():
    fc = wedd.FeatureCuts(feature="f", profile="p", cuts=[0.25, 0.5], node_paths=["root", "root.R"], status="CUT", n_fit=0, fit_iqr=1.0, fit_sd=1.0)
    x = np.array([0.25, np.nextafter(0.25, 1), 0.5, np.nextafter(0.5, 1), np.nextafter(0.25, 0), np.nan])
    assert fc.bin_index(x).tolist() == [0, 1, 1, 2, 0, -1]
    states = wedd.CutSet({"f": fc}, "p").transform(pd.DataFrame({"f": x}))
    assert states["f"].tolist() == ["b0of3", "b1of3", "b1of3", "b2of3", "b0of3", wedd.MISSING_STATE]


@pytest.mark.parametrize("profile", ["wedd_gaussian_mm_v1", "wedd_uniform_mm_v1", "three_quantile_v1"])
def test_constant_attribute_produces_no_cut(wcfg, profile):
    cs = wedd.fit_cuts(pd.DataFrame({"f": np.ones(200)}), np.repeat(["A", "B"], 100), profile, wcfg)
    assert cs.features["f"].cuts == [] and cs.features["f"].status == "NO_CUT_CONSTANT"
    assert set(cs.transform(pd.DataFrame({"f": [1.0, np.nan]}))["f"]) == {"b0of1", wedd.MISSING_STATE}


def test_uniform_and_tertile_profiles_run_on_same_fixture(wcfg):
    x, d = _two_clusters(300, SEED_SYNTHETIC + 2)
    frame = pd.DataFrame({"f": x})
    uni = wedd.fit_cuts(frame, d, "wedd_uniform_mm_v1", wcfg).features["f"]
    assert 1 <= len(uni.cuts) <= 2 and any(x[x < 0].max() < c < x[x > 0].min() for c in uni.cuts)
    ter = wedd.fit_cuts(frame, None, "three_quantile_v1", wcfg).features["f"]
    assert ter.cuts == pytest.approx(list(np.quantile(x, [1 / 3, 2 / 3])))
    assert wedd.resolve_profile("wedd_uniform_mm_v1", wcfg)["weight_entropy"] == wcfg["profiles"]["wedd_gaussian_mm_v1"]["weight_entropy"]


def test_min_leaf_and_max_bins_are_respected(wcfg):
    rng = np.random.default_rng(SEED_SYNTHETIC + 3)
    x = rng.normal(size=500)
    d = np.where(x < -1.0, "A", np.where(x < 0.7, "B", "C"))
    fc = wedd.fit_cuts(pd.DataFrame({"f": x}), d, "wedd_gaussian_mm_v1", wcfg).features["f"]
    assert len(fc.cuts) == 2 and fc.cuts == pytest.approx([-1.0, 0.7], abs=0.05)
    assert fc.min_leaf == max(10, math.ceil(0.02 * 500))
    counts = np.bincount(fc.bin_index(x))
    assert counts.min() >= fc.min_leaf and len(counts) <= wcfg["profiles"]["wedd_gaussian_mm_v1"]["max_bins"]
    assert {r["node_path"] for r in fc.records} <= {"root", "root.L", "root.R"}
    assert wedd.CutSet.from_dict(wedd.fit_cuts(pd.DataFrame({"f": x}), d, "wedd_gaussian_mm_v1", wcfg).to_dict()).cuts() == {"f": fc.cuts}


def test_stability_gate_is_invariant_to_n_jobs(wcfg):
    x, d = _two_clusters(60, SEED_SYNTHETIC + 4)
    rng = np.random.default_rng(SEED_SYNTHETIC + 5)
    frame = pd.DataFrame({"f": x, "g": rng.normal(size=x.size)})
    one = wedd.stability_gate(frame, d, "wedd_gaussian_mm_v1", wcfg, seed=SEED_SYNTHETIC, n_replicates=24, n_jobs=1)
    three = wedd.stability_gate(frame, d, "wedd_gaussian_mm_v1", wcfg, seed=SEED_SYNTHETIC, n_replicates=24, n_jobs=3)
    pd.testing.assert_frame_equal(one, three)
    planted = one[one["feature"] == "f"].iloc[0]
    assert planted["recurrence"] >= wcfg["stability_gate"]["recurrence_min"] and bool(planted["retained"])
    assert planted["tolerance"] == pytest.approx(0.10 * np.subtract(*np.percentile(x, [75, 25])))


def test_lower_upper_boundary_match_hand_computation():
    table = {"a": ("s0", 1), "b": ("s0", 1), "c": ("s1", 0), "d": ("s1", 1), "e": ("s2", 0), "f": ("missing", 1)}
    approx = rs.approximations({k: v[0] for k, v in table.items()}, {k: v[1] for k, v in table.items()}, decision_value=1)
    assert approx.lower == {"a", "b"}
    assert approx.upper == {"a", "b", "c", "d"}
    assert approx.boundary == {"c", "d"}
    res = rs.lower_upper_boundary(pd.DataFrame({"s": [v[0] for v in table.values()]}, index=list(table)), pd.Series([v[1] for v in table.values()], index=list(table)), ["s"])
    assert res["excluded"] == ["f"] and res["classes"]["0"]["lower"] == {"e"}


def test_variable_precision_granules_are_not_strict():
    states = pd.DataFrame({"s": ["b0of2"] * 10}, index=[f"x{i}" for i in range(10)])
    decisions = pd.Series(["Y"] * 9 + ["N"], index=states.index)
    (g,) = rs.granules(states, decisions, ["s"], beta=0.10)
    assert g.tier == rs.VARIABLE_PRECISION and abs(g.confidence - 0.9) < 1e-12
    assert rs.granules(states, decisions, ["s"], beta=0.0) == []


def test_tautology_guard(registry):
    allowed, groups = rs.antecedent_policy(registry)
    assert allowed == registry["primary_rulebook_antecedent_candidates"]
    assert set(groups) == {frozenset({"lv_ef_percent", "lv_edv_ml", "lv_esv_ml"}), frozenset({"rv_ef_percent", "rv_edv_ml", "rv_esv_ml"})}
    with pytest.raises(rs.TautologyError):
        rs.check_no_tautology(["lv_ef_percent", "lv_edv_ml", "lv_esv_ml"], registry)
    with pytest.raises(rs.TautologyError):
        rs.check_no_tautology(["qc_failure_reason", "lv_edv_ml"], registry)
    with pytest.raises(rs.TautologyError):
        rs.check_no_tautology(["lv_sv_ml"], registry)  # reconstruction-only target
    rs.check_no_tautology(["lv_ef_percent", "lv_edv_ml"], registry)


def _ef_states(n=120, seed=SEED_SYNTHETIC):
    rng = np.random.default_rng(seed)
    edv = rng.integers(0, 2, n)
    esv = rng.integers(0, 2, n)
    ef = rng.integers(0, 2, n)
    d = np.where((edv == 1) & (esv == 1) & (ef == 1), "DCM", "NOR")  # only the forbidden triple is informative
    states = pd.DataFrame({"lv_edv_ml": [f"b{v}of2" for v in edv], "lv_esv_ml": [f"b{v}of2" for v in esv], "lv_ef_percent": [f"b{v}of2" for v in ef]})
    return states, d, pd.Series([f"patient{i:03d}" for i in range(n)])


def test_vprs_never_uses_an_ef_with_both_of_its_volumes(registry, rcfg):
    states, d, anchors = _ef_states()
    rb = rs.induce_vprs_rules(states, d, anchors, rcfg, registry=registry, consequent_kind="d_model")
    assert all(not {"lv_ef_percent", "lv_edv_ml", "lv_esv_ml"} <= set(r.features) for r in rb.rules)
    assert all(r.consequent != "DCM" or r.confidence >= 0.9 for r in rb.rules)


def test_vprs_discovery_is_exhaustive_deterministic_and_uncapped(registry, rcfg):
    rng = np.random.default_rng(SEED_SYNTHETIC + 6)
    n = 150
    cols = registry["primary_rulebook_antecedent_candidates"]
    states = pd.DataFrame({c: [f"b{v}of3" for v in rng.integers(0, 3, n)] for c in cols})
    d = np.where(states["global_wall_thickness_ed_mm"] == "b2of3", "HCM", np.where(states["rv_ef_percent"] == "b0of3", "RV", "NOR"))
    d = np.where(rng.random(n) < 0.03, "MINF", d)
    anchors = pd.Series([f"p{i:03d}" for i in range(n)])
    rb = rs.induce_vprs_rules(states, d, anchors, rcfg, registry=registry, consequent_kind="dataset_label")
    ranks = [(-r.confidence, -r.antecedent_n, r.length) for r in rb.rules]
    assert ranks == sorted(ranks) and len(rb.rules) >= 2
    assert all(r.antecedent_n >= rcfg["discovery"]["min_antecedent_n_patients"] and r.length <= rcfg["discovery"]["max_length"] for r in rb.rules)
    assert all(r.strict == (r.confidence == 1.0) for r in rb.rules)
    assert all(r.consequent != "MINF" for r in rb.rules)  # class support gate (5 patients)
    shuffled = rs.induce_vprs_rules(states[cols[::-1]], d, anchors, rcfg, registry=registry, consequent_kind="dataset_label")
    assert shuffled.to_records() == rb.to_records()  # independent of column order
    assert rs.Rulebook.from_records(rb.to_records()).to_records() == rb.to_records()
    hcm = [r for r in rb.rules if r.consequent == "HCM"]
    assert any(r.antecedent == (("global_wall_thickness_ed_mm", "b2of3"),) for r in hcm)


def test_cross_fitted_gate(registry, rcfg, wcfg):
    rng = np.random.default_rng(SEED_SYNTHETIC + 7)
    n = 100
    x = np.concatenate([rng.normal(-2, 0.3, n // 2), rng.normal(2, 0.3, n // 2)])
    y = np.where(x > 0, "HCM", "NOR")
    values = pd.DataFrame({"global_wall_thickness_ed_mm": x})
    folds = np.arange(n) % 5 + 1
    anchors = pd.Series([f"p{i:03d}" for i in range(n)])

    def cutset_fn(train_values, train_decision, n_train):
        return wedd.fit_cuts(train_values, train_decision, "wedd_gaussian_mm_v1", wcfg, n_fit_patients=n_train)

    final_cuts = cutset_fn(values, y, n)
    final = rs.induce_vprs_rules(final_cuts.transform(values), y, anchors, rcfg, registry=registry, consequent_kind="dataset_label")
    gate = rs.cross_fitted_gate(values, lambda tr, ap: y[ap], folds, rcfg, final_rulebook=final, cutset_fn=cutset_fn, anchors=anchors,
                                induce_kwargs={"registry": registry})
    assert len(gate) == 2 and gate["eligible"].all() and (gate["folds_reproduced"] == 5).all()
    assert gate["oof_matched"].sum() == n and (gate["oof_agree"] == gate["oof_matched"]).all()
    assert {f"eligible_at_{s:.2f}" for s in rcfg["validation_gate"]["gate_sensitivities"]} <= set(gate.columns)
    flipped = rs.cross_fitted_gate(values, lambda tr, ap: np.where(np.isin(ap, tr), y[ap], np.where(y[ap] == "HCM", "NOR", "HCM")), folds, rcfg,
                                   final_rulebook=final, cutset_fn=cutset_fn, anchors=anchors, induce_kwargs={"registry": registry})
    assert not flipped["eligible"].any()  # held-out consequent disagrees: agreement 0
    gated = rs.apply_gate(final, gate)
    assert all(r.eligibility == rs.ELIGIBLE and r.crossfit["folds_reproduced"] == 5 for r in gated.rules)
    empty = rs.apply_gate(final, flipped).eligible()
    assert empty.is_empty and empty.empty_record()["reason_code"] == "RULEBOOK_EMPTY"


def test_retain_and_per_object_stability(wcfg):
    x, d = _two_clusters(60, SEED_SYNTHETIC + 8)
    frame = pd.DataFrame({"f": x})
    cs = wedd.fit_cuts(frame, d, "wedd_gaussian_mm_v1", wcfg)
    stab = wedd.stability_gate(frame, d, "wedd_gaussian_mm_v1", wcfg, seed=SEED_SYNTHETIC, n_replicates=10, reference=cs)
    assert cs.retain(stab).cuts() == cs.cuts() and cs.retain(stab, min_recurrence=1.01).cuts() == {"f": []}
    c = cs.features["f"].cuts[0]
    final = pd.DataFrame({"f": [-1.5, 1.5, c + 0.01]})
    folds = [pd.DataFrame({"f": [-1.6, 1.7, c + v]}) for v in (0.02, -0.02, 0.03, -0.03, 0.04)]
    st = wedd.per_object_stability(folds, final, cs, wcfg)
    assert st["f"].tolist() == ["ESTIMATED", "ESTIMATED", wcfg["per_object_stability"]["failure_state"]]  # 3 of 5 agree
    tokens = cs.transform(final, states=st)["f"].tolist()
    assert tokens[2] == "UNAVAILABLE:FEATURE_UNSTABLE" and tokens[0] == "b0of2"


def test_antecedent_eligibility_gates(registry, rcfg):
    from tmmri.rules import antecedent_eligibility_v6

    rng = np.random.default_rng(SEED_SYNTHETIC + 9)
    n = 100
    cands = registry["primary_rulebook_antecedent_candidates"]
    measured = pd.DataFrame({c: 50 + 10 * rng.normal(size=n) for c in cands})
    for c in cands:
        measured[f"{c}__state"] = "OBSERVED"
    measured.loc[:29, "rv_esv_ml__state"] = "MISSING_SOURCE"   # 70 % observed < 80 %
    measured["rv_ef_percent"] = 55.0                             # constant: variability gate
    oof = measured[cands] + rng.normal(scale=2.0, size=(n, len(cands)))
    oof["lv_esv_ml"] = 50 + 10 * rng.normal(size=n)              # uninformative reconstruction
    mean = pd.DataFrame({c: np.full(n, measured[c].mean()) for c in cands})
    folds = np.arange(n) % 5 + 1
    out = antecedent_eligibility_v6(measured, oof, mean, folds, rcfg, registry, rulebook="d_model").set_index("feature")
    assert out.loc["lv_edv_ml", "eligible"] and out.loc["lv_edv_ml", "folds_improved"] == 5
    assert out.loc["rv_esv_ml", "failing_gates"] == "reference_availability"
    assert "variability" in out.loc["rv_ef_percent", "failing_gates"]
    assert out.loc["lv_esv_ml", "failing_gates"] == "reconstruction"
    oracle = antecedent_eligibility_v6(measured, oof, mean, folds, rcfg, registry, rulebook="measured_oracle").set_index("feature")
    assert oracle.loc["lv_esv_ml", "eligible"]  # the reconstruction gate does not apply to the measured oracle
