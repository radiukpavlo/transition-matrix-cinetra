"""Inference routes, reason codes and descriptive templates (SPECIFICATION.md 6.0.0, Sections 11.7-11.8; Decision D14)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import REPO, SEED_SYNTHETIC

from tmmri import inference as inf
from tmmri import rough_sets as rs


def _load(rel):
    with (REPO / rel).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def rcfg():
    return _load("configs/rule_policy.yaml")


@pytest.fixture(scope="module")
def cohorts():
    return _load("configs/cohorts.yaml")


def _rule(rule_id, antecedent, consequent, support=40):
    ant = tuple(sorted(antecedent.items()))
    return rs.Rule(rule_id=rule_id, antecedent=ant, consequent=consequent, support=support, confidence=0.95, strict=False,
                   anchors=(), antecedent_n=support, eligibility=rs.ELIGIBLE, consequent_kind="d_model")


def _book(*rules):
    return rs.Rulebook(list(rules), "d_model", "d_model")


def _states(*rows):
    return pd.DataFrame([{"sample_id": f"S{i}", **r} for i, r in enumerate(rows)])


def _one(rulebook, row, rcfg, reason_codes, qc=None, cohorts=None):
    return inf.route(rulebook, _states(row), qc, rcfg, reason_codes, qc_policy=cohorts).iloc[0]


def test_single_consequent_exact_accept(rcfg, reason_codes):
    out = _one(_book(_rule("r1", {"lv_edv_ml": "b2of3", "lv_ef_percent": "b0of3"}, "DCM")), {"lv_edv_ml": "b2of3", "lv_ef_percent": "b0of3"}, rcfg, reason_codes)
    assert out.route == "EXACT_ACCEPT" and out.label == "DCM" and out.fired_rule_ids == "r1"


def test_conflict_never_emits_a_label(rcfg, reason_codes):
    rb = _book(_rule("r1", {"lv_edv_ml": "b2of3"}, "DCM"), _rule("r2", {"lv_ef_percent": "b0of3"}, "MINF"))
    out = _one(rb, {"lv_edv_ml": "b2of3", "lv_ef_percent": "b0of3"}, rcfg, reason_codes)
    assert out.route == "CONFLICT_ABSTAIN" and out.label is None
    assert out.reason_code == "MATCH_CONFLICTING_CONSEQUENTS" and bool(out.conflict_encountered)
    assert out.candidate_labels == "DCM;MINF"


def test_soft_only_returns_candidates_not_label(rcfg, reason_codes):
    rb = _book(_rule("r1", {"lv_edv_ml": "b2of3", "lv_ef_percent": "b0of3", "rv_edv_ml": "b0of3"}, "DCM"))
    out = _one(rb, {"lv_edv_ml": "b2of3", "lv_ef_percent": "b0of3", "rv_edv_ml": "b2of3"}, rcfg, reason_codes)
    assert out.route == "SOFT_CANDIDATES_ONLY" and out.label is None
    assert out.candidate_labels == "DCM" and out.candidate_rule_ids == "r1" and out.reason_code == "MATCH_SOFT_ONLY"
    two = _book(_rule("r2", {"lv_edv_ml": "b2of3", "lv_ef_percent": "b0of3"}, "DCM"))  # L_r = 2 < min_antecedents
    assert _one(two, {"lv_edv_ml": "b2of3", "lv_ef_percent": "b1of3"}, rcfg, reason_codes).route == "NO_MATCH_ABSTAIN"


def test_unavailable_feature_and_no_match(rcfg, reason_codes):
    rb = _book(_rule("r1", {"lv_edv_ml": "b1of2", "lv_ef_percent": "b1of2"}, "DCM"))
    out = _one(rb, {"lv_edv_ml": "missing", "lv_ef_percent": "b1of2"}, rcfg, reason_codes)
    assert out.route == "UNAVAILABLE_FEATURE_ABSTAIN" and out.reason_code == "MATCH_MISSING_REQUIRED_SYMBOL" and out.label is None
    out = _one(rb, {"lv_edv_ml": "UNAVAILABLE:FEATURE_UNSTABLE", "lv_ef_percent": "b1of2"}, rcfg, reason_codes)
    assert out.route == "UNAVAILABLE_FEATURE_ABSTAIN" and out.reason_code == "FEATURE_UNSTABLE"
    assert "MATCH_MISSING_REQUIRED_SYMBOL" in out.reason_codes.split(";")
    out = _one(rb, {"lv_edv_ml": "missing", "lv_ef_percent": "b0of2"}, rcfg, reason_codes)  # mismatch decides the rule
    assert out.route == "NO_MATCH_ABSTAIN" and out.reason_code == "MATCH_NONE"


def test_empty_rulebook_route(rcfg, reason_codes):
    out = _one(_book(), {"lv_edv_ml": "b0of2"}, rcfg, reason_codes)
    assert out.route == "EMPTY_RULEBOOK" and out.reason_code == "RULEBOOK_EMPTY"
    candidate_only = rs.Rulebook([rs.Rule("r", (("lv_edv_ml", "b0of2"),), "NOR", 9, 1.0, True, ())], "d_model", "d_model")
    assert _one(candidate_only, {"lv_edv_ml": "b0of2"}, rcfg, reason_codes).route == "EMPTY_RULEBOOK"  # ungated rules never route


def test_qc_abstain_precedes_rules_and_fails_closed(rcfg, reason_codes, cohorts):
    rb = _book(_rule("r1", {"lv_edv_ml": "b1of2"}, "DCM"))
    qc = pd.DataFrame({"sample_id": ["S0"], "qc_failure_reason": ["missing_lv_ed_present;lv_edv_le_lvesv"]})
    out = _one(rb, {"lv_edv_ml": "b1of2"}, rcfg, reason_codes, qc=qc, cohorts=cohorts)
    assert out.route == "QC_ABSTAIN" and out.reason_code == "QC_MISSING_LV_BLOOD_POOL" and out.label is None
    qc = pd.DataFrame({"sample_id": ["S0"], "qc_failure_reason": ["not_a_registered_reason"]})
    assert _one(rb, {"lv_edv_ml": "b1of2"}, rcfg, reason_codes, qc=qc, cohorts=cohorts).reason_code == "QC_UNREGISTERED_REASON"
    qc = pd.DataFrame({"sample_id": ["S0"], "qc_failure_reason": ["lv_rv_sv_discrepancy_gt_15_ml"]})  # moderate: routed
    assert _one(rb, {"lv_edv_ml": "b1of2"}, rcfg, reason_codes, qc=qc, cohorts=cohorts).route == "EXACT_ACCEPT"


def test_routes_partition_and_codes_are_registered(rcfg, reason_codes, cohorts):
    rng = np.random.default_rng(SEED_SYNTHETIC)
    feats = ["lv_edv_ml", "lv_ef_percent", "rv_edv_ml"]
    rb = _book(
        _rule("r1", {"lv_edv_ml": "b2of3", "lv_ef_percent": "b0of3", "rv_edv_ml": "b1of3"}, "DCM"),
        _rule("r2", {"lv_ef_percent": "b2of3"}, "NOR"),
        _rule("r3", {"rv_edv_ml": "b2of3"}, "RV"),
    )
    tokens = ["b0of3", "b1of3", "b2of3", "missing", "UNAVAILABLE:FEATURE_DOMAIN_BLOCKED"]
    rows = [{f: tokens[rng.integers(0, len(tokens))] for f in feats} for _ in range(300)]
    qc = pd.DataFrame({"sample_id": [f"S{i}" for i in range(0, 300, 37)], "qc_failure_reason": "missing_myo_es_present"})
    out = inf.route(rb, _states(*rows), qc, rcfg, reason_codes, qc_policy=cohorts)
    counts = inf.route_counts(out, reason_codes)
    assert sum(counts.values()) == len(rows) and set(counts) == set(reason_codes["routes"])
    assert out["label"].notna().sum() == counts["EXACT_ACCEPT"]  # only EXACT_ACCEPT emits a label
    registered = {c["code"]: c["route"] for c in reason_codes["codes"]}
    for r in out.itertuples():
        for code in [c for c in (r.reason_codes or "").split(";") if c]:
            assert registered[code] == r.route


def test_templates_are_descriptive(rcfg):
    templates = _load("registry/pattern_templates.yaml")
    states = _states({"lv_myocardial_mass_ed_g": "b2of3", "lv_edv_ml": "b0of3", "lv_ef_percent": "b1of3", "rv_edv_ml": "b0of2"})
    m = inf.match_templates(states, templates).set_index("template_id")
    assert bool(m.loc["PT_INCREASED_LV_MASS", "matched"]) and bool(m.loc["PT_NORMAL_SIZE_FUNCTION", "matched"])
    assert not m.loc[m["support_status"] == "unsupported", "matched"].any() and m["descriptive_only"].all()
    rb = _book(_rule("r1", {"lv_edv_ml": "b2of3", "lv_ef_percent": "b0of3"}, "DCM"), _rule("r2", {"lv_myocardial_mass_ed_g": "b1of3"}, "HCM"))
    t = inf.template_rule_matches(rb, templates).set_index("template_id")
    assert t.loc["PT_DILATED_LV_REDUCED_EF", "implying_rule_ids"] == "r1" and t.loc["PT_INCREASED_LV_MASS", "n_implying_rules"] == 0
