"""Protocol 6.0.0 acceptance controls on a synthetic ACDC run (SPECIFICATION.md 6.0.0, Sections 9.5, 11.3.1, 11.4,
12.3, 12.4, 13.3 and 15.2): reference substitution at inference (acceptance test 13), QC panels and the manual review
table, the WEDD candidate table, decoders and per-object stability in routing, the within-family BH tables, the shift
audit classifier and the WEDD settings grid.  Synthetic algorithm controls only; never cohort results."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import fixtures_v6 as fx
from test_v6_cli import _study_copy
from tmmri import stages
from tmmri.config import load_study
from tmmri.rough_sets import ELIGIBLE, Rule, Rulebook
from tmmri.rule_pipeline import qc_route_frame, route_records, routing_substrate
from tmmri.wedd import fit_cuts

RULEBOOKS_FROM_A = ("d_model", "dataset_label")   # the measured-B oracle consumes measured B by definition (Section 11.1)
SENTINEL = 123456.789
SENTINEL_CLASS = "SENTINEL_CLASS"


@pytest.fixture(scope="module")
def acdc_run(tmp_path_factory):
    """``run --until evaluate:acdc_test`` on a synthetic ACDC fixture (25 development, 10 test patients)."""
    tmp = tmp_path_factory.mktemp("acceptance")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(stages, "protocol_frozen", lambda *a, **k: False)
        acdc = fx.make_acdc_source(tmp / "src" / "acdc", fx.acdc_patients(5, testing_per_class=2))
        fx.write_fixture_manifests(acdc, fx.acdc_manifest(acdc))
        mms = fx.make_mms_source(tmp / "src" / "mms")
        fx.write_fixture_manifests(mms, fx.mms_manifest(mms), fx.mms_metadata(mms))
        study_path = _study_copy(tmp)
        ctx = stages.make_context(study_path, "v6-acceptance-20260928-1700", artifact_root=tmp / "artifacts", data_root=tmp / "data",
                                  n_jobs=1, allow_unfrozen_synthetic=True, sources={"acdc": acdc, "mms": mms})
        stages.run_pipeline(ctx, until="evaluate:acdc_test")
        yield ctx


def _snapshot(run) -> dict[str, pd.DataFrame]:
    out = {"d_model": pd.read_parquet(run / "inference" / "d_model_acdc_test.parquet"),
           "bhat": pd.read_parquet(run / "reconstructed" / "bhat_acdc_test.parquet"),
           "bhat_decoded": pd.read_parquet(run / "reconstructed" / "bhat_acdc_test_decoded.parquet"),
           "per_target": pd.read_parquet(run / "metrics" / "per_target_acdc_test.parquet")}
    for rb in RULEBOOKS_FROM_A:
        out[f"routes_{rb}"] = pd.read_parquet(run / "inference" / f"routes_{rb}_acdc_test.parquet")
        out[f"stability_{rb}"] = pd.read_parquet(run / "reconstructed" / f"stability_{rb}_acdc_test.parquet")
    return out


def test_qc_panels_and_manual_review_table(acdc_run):
    """Section 12.4: all critical and high cases plus 20 hash-ranked cases per partition; ACDC only before M0."""
    from tmmri.semantics.visual_qc import select_qc_panels_v6

    run, study = acdc_run.run.path, acdc_run.study
    manifest = json.loads((run / "qc" / "panels_ACDC.json").read_text(encoding="utf-8"))
    assert manifest["partitions"] == ["acdc_dev", "acdc_test"] and manifest["selection_salt"] == "qc-panel-v6|"
    panels = manifest["panels"]
    b = pd.read_parquet(run / "semantics" / "B_ACDC.parquet")
    for part, n_part in (("acdc_dev", 25), ("acdc_test", 10)):
        chosen = [e for e in panels.values() if e["partition"] == part]
        flagged = [e for e in chosen if e["selection_reason"] in ("qc_critical", "qc_high")]
        assert len(chosen) - len(flagged) == min(20, n_part - len(flagged)), part
        assert int((b["split"] == part).sum()) == n_part
    for entry in panels.values():
        png = run / entry["file"]
        assert png.is_file() and png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    review = pd.read_parquet(run / "qc" / "manual_review.parquet")
    assert set(panels) <= set(review["sample_id"]) and (review["adjudication_status"] == "pending_review").all()
    assert set(review.loc[review["panel_file"] != "", "sample_id"]) == set(panels)
    assert not (run / "qc" / "panels_MMS.json").exists()              # M&Ms panels only through the guarded MMS stage
    parts = ["acdc_dev", "acdc_test"]
    first = select_qc_panels_v6(b, study.cohorts, study.reason_codes, parts)
    shuffled = select_qc_panels_v6(b.sample(frac=1.0, random_state=3), study.cohorts, study.reason_codes, parts)
    assert first.equals(shuffled) and set(first["sample_id"]) == set(panels)   # independent of row order
    salted = dict(study.cohorts, qc_panels={**study.cohorts["qc_panels"], "random_selection_salt": "other-salt|"})
    other = select_qc_panels_v6(b, salted, study.reason_codes, parts)
    assert list(other.loc[other["partition"] == "acdc_dev", "sample_id"]) != list(first.loc[first["partition"] == "acdc_dev", "sample_id"]) or \
        list(other["selection_rank"]) != list(first["selection_rank"])


def test_wedd_candidates_table_records_every_candidate(acdc_run):
    """Section 11.3.1: every valid candidate cut of every split node with H, density, normalization, J and supports."""
    run, study = acdc_run.run.path, acdc_run.study
    table = pd.read_parquet(run / "thresholds" / "wedd_candidates.parquet")
    assert len(table) and set(table["rulebook"]) <= {"d_model", "dataset_label", "measured_oracle"}
    profile = study.wedd["profiles"][study.wedd["primary_profile"]]
    j = profile["weight_entropy"] * table["entropy_normalized"] + profile["weight_density"] * table["density_normalized"]
    assert np.allclose(j, table["objective"], atol=1e-12)
    assert (table["support_left"] >= table["min_leaf"]).all() and (table["support_right"] >= table["min_leaf"]).all()
    for (rb, feature, node), group in table.groupby(["rulebook", "feature", "node_path"]):
        assert group["selected"].sum() == 1, (rb, feature, node)
        chosen = float(group.loc[group["selected"], "cut"].iloc[0])
        assert group.loc[group["selected"], "objective"].iloc[0] == pytest.approx(group["objective"].min())
        cuts = json.loads((run / "thresholds" / f"{rb}_{study.wedd['primary_profile']}.json").read_text(encoding="utf-8"))
        assert chosen in cuts["features"][feature]["cuts"]


def test_evaluation_writes_decoder_log_stability_states_and_f1_tests(acdc_run):
    run, study = acdc_run.run.path, acdc_run.study
    decoded = pd.read_parquet(run / "reconstructed" / "bhat_acdc_test_decoded.parquet")
    targets = study.transition_targets()
    assert {f"{t}__state" for t in targets} <= set(decoded.columns) and len(decoded) == 10
    assert set(decoded[[f"{t}__state" for t in targets]].stack().unique()) <= {"ESTIMATED", "NUMERIC_DOMAIN_BLOCKED", "TARGET_INELIGIBLE", "INPUT_QC_BLOCKED"}
    log = json.loads((run / "metrics" / "decoder_acdc_test.json").read_text(encoding="utf-8"))
    assert set(log["targets"]) == set(targets)
    assert all(v["estimated"] + v["numeric_domain_blocked"] + v["target_ineligible"] + v["input_qc_blocked"] == 10 for v in log["targets"].values())
    for f in range(1, 6):
        assert (run / "transitions" / "fold_models" / f"transition_fold{f}.json").is_file()   # frozen with the bundle
    frozen = json.loads((run / ".stages" / "FROZEN.json").read_text(encoding="utf-8"))
    assert "transitions/fold_models/transition_fold1.json" in frozen["bundle"]
    f1 = pd.read_parquet(run / "metrics" / "multiplicity_F1.parquet")
    assert set(f1["target"]) == set(targets) and set(f1["family"]) == {"F1_reconstruction_ridge_vs_mean_acdc_test"}
    tested = f1["p_value"].notna()
    assert (f1.loc[tested, "q_value"] >= f1.loc[tested, "p_value"] - 1e-12).all()


def _single_rule_book(feature: str, state: str) -> list[dict]:
    from tmmri.rough_sets import apply_gate
    from tmmri.rule_pipeline import rulebook_payload

    rule = Rule(rule_id="R_unit", antecedent=((feature, state),), consequent="NOR", support=6, confidence=1.0, strict=True,
                anchors=tuple(f"p{i}" for i in range(6)), antecedent_n=6, tier="STRICT", consequent_kind="d_model")
    gate = pd.DataFrame([{"rule_id": "R_unit", "consequent": "NOR", "length": 1, "oof_matched": 6, "oof_agree": 6, "agreement": 1.0,
                          "wilson_lb": 0.61, "wilson_ub": 1.0, "folds_reproduced": 5, "reproduced_in_folds": "1;2;3;4;5",
                          "eligible": True, "eligibility": ELIGIBLE}])
    return rulebook_payload("d_model", apply_gate(Rulebook([rule], "d_model", "d_model"), gate), gate, {})[1]["records"]


def test_domain_guard_and_unstable_estimates_route_with_registered_codes():
    """Sections 9.5, 11.4 and 11.7: a domain-blocked or per-object-unstable reconstruction cannot satisfy an antecedent;
    the case routes UNAVAILABLE_FEATURE_ABSTAIN with FEATURE_DOMAIN_BLOCKED or FEATURE_UNSTABLE."""
    study = load_study()
    feature = "lv_edv_ml"
    cuts = fit_cuts(pd.DataFrame({feature: np.r_[np.linspace(40, 90, 10), np.linspace(110, 160, 10)]}),
                    pd.Series(["NOR"] * 10 + ["DCM"] * 10), "wedd_gaussian_mm_v1", study.wedd)
    records = _single_rule_book(feature, "b0of2")
    ids = pd.Index(["ACDC:stable", "ACDC:unstable", "ACDC:blocked"], name="sample_id")
    values = pd.DataFrame({feature: [60.0, 60.0, np.nan]}, index=ids)
    states = pd.DataFrame({feature: ["ESTIMATED", "ESTIMATED", "NUMERIC_DOMAIN_BLOCKED"]}, index=ids)
    folds = [pd.DataFrame({feature: [60.0, 60.0 if k < 3 else 150.0, np.nan]}, index=ids) for k in range(5)]
    folds[3].loc["ACDC:unstable", feature] = 150.0            # three of five fold symbols agree: below 4 of 5
    substrate, stability = routing_substrate(values, states, folds, cuts, study.wedd, [feature])
    assert list(substrate[f"{feature}__state"]) == ["ESTIMATED", "UNSTABLE_ESTIMATE", "NUMERIC_DOMAIN_BLOCKED"]
    qc = qc_route_frame(pd.DataFrame({"qc_failure_reason": [None] * 3}, index=ids), np.ones(3, dtype=bool), study)
    routes = route_records(records, "d_model", cuts, substrate, qc, study).set_index("sample_id")
    assert routes.loc["ACDC:stable", "route"] == "EXACT_ACCEPT"
    assert routes.loc["ACDC:unstable", ["route", "reason_code"]].tolist() == ["UNAVAILABLE_FEATURE_ABSTAIN", "FEATURE_UNSTABLE"]
    assert routes.loc["ACDC:blocked", ["route", "reason_code"]].tolist() == ["UNAVAILABLE_FEATURE_ABSTAIN", "FEATURE_DOMAIN_BLOCKED"]


def test_bh_is_applied_within_each_family_never_merged():
    from tmmri.metrics import benjamini_hochberg, bh_family_table

    p = [0.001, 0.02, 0.04, 0.03, 0.2]
    tests = pd.DataFrame({"family": ["F1_a"] * 3 + ["F2_b"] * 2, "p_value": p})
    out = bh_family_table(tests, q=0.05, families=["F1_a", "F2_b"])
    assert np.allclose(out["q_value"][:3], benjamini_hochberg(p[:3])["q_value"]) and np.allclose(out["q_value"][3:], benjamini_hochberg(p[3:])["q_value"])
    assert not np.allclose(out["q_value"], benjamini_hochberg(p)["q_value"])           # a merged ranking would differ
    with pytest.raises(ValueError, match="prespecified families"):
        bh_family_table(tests.assign(family="F9_unknown"), q=0.05, families=["F1_a"])


def test_shift_audit_classifier_is_cross_fitted_and_label_free():
    """Section 12.3: out-of-fold AUROC near 0.5 without shift, near 1 with shift; hash folds within domain, order-free."""
    from tmmri.classification import domain_shift_audit, hash_rule_folds

    cfg = load_study().bootstrap["shift_audit"]
    rng = np.random.default_rng(fx.SEED_SYNTHETIC)
    ref, same, moved = rng.normal(size=(60, 6)), rng.normal(size=(40, 6)), rng.normal(size=(40, 6)) + 1.5
    ids_ref, ids_cmp = [f"ACDC:p{i:03d}" for i in range(60)], [f"MMS:c{i:03d}" for i in range(40)]
    null, _ = domain_shift_audit(ref, same, ids_ref, ids_cmp, cfg, n_boot=200, seed=20260928, min_cases=20)
    shifted, scores = domain_shift_audit(ref, moved, ids_ref, ids_cmp, cfg, n_boot=200, seed=20260928, min_cases=20)
    assert abs(null["auroc"] - 0.5) < 0.2 and shifted["auroc"] > 0.9 and shifted["ci_lower"] > 0.8 and shifted["status"] == "EVALUATED"
    assert scores["score"].notna().all() and set(scores["fold"]) == set(range(1, 6))
    small, _ = domain_shift_audit(ref, moved[:10], ids_ref, ids_cmp[:10], cfg, n_boot=50, seed=1, min_cases=20)
    assert small["status"] == "NE" and small["reason"] == "STRATUM_BELOW_MIN_CASES" and small["auroc"] is not None
    tiny, _ = domain_shift_audit(ref, moved[:3], ids_ref, ids_cmp[:3], cfg, n_boot=50, seed=1, min_cases=20)
    assert tiny["reason"] == "fewer_cases_than_folds"
    domains = np.r_[np.zeros(60), np.ones(40)]
    folds = hash_rule_folds(ids_ref + ids_cmp, domains, cfg["cross_fitting"]["hash_salt"], 5)
    perm = rng.permutation(100)
    again = hash_rule_folds([(ids_ref + ids_cmp)[i] for i in perm], domains[perm], cfg["cross_fitting"]["hash_salt"], 5)
    assert np.array_equal(again, folds[perm])


def test_wedd_settings_grid_includes_depth_three_at_four_bins():
    """configs/wedd.yaml sensitivity_grid: one factor at a time; depth three is evaluated at four bins (Section 11.3.1)."""
    from tmmri.wedd import WeddParams, resolve_profile

    study = load_study()
    variants = dict(stages.wedd_settings_variants(study))
    assert variants["max_depth_3_max_bins_4"] == {"max_depth": 3, "max_bins": 4}
    assert "weight_entropy_0.65" not in variants and variants["weight_entropy_0.25"]["weight_density"] == pytest.approx(0.75)
    primary = study.wedd["primary_profile"]
    params = WeddParams.from_profile(resolve_profile(primary, stages._wedd_cfg_variant(study.wedd, primary, variants["max_depth_3_max_bins_4"])))
    assert (params.max_depth, params.max_bins) == (3, 4)
    assert resolve_profile(primary, study.wedd)["max_depth"] == 2          # the primary configuration is unchanged


def test_reference_substitution_leaves_predictions_and_routes_unchanged(acdc_run):
    """Acceptance test 13 (Sections 12.3 and 15.2): replacing every acdc_test reference B value and every class label Y
    with sentinels leaves d_model, the reconstructions and the routes of the rulebooks routed from A unchanged; the
    evaluation-only metrics change, which proves the sentinels reached the evaluator.  QC tokens are kept: they are
    inference inputs derived from the staged image and mask at build-semantics (Section 12.4), not reference values.
    The measured-B oracle is excluded because it consumes measured B by definition (Section 11.1).

    The frozen run itself may not re-execute its recorded evaluation (once-only guard), so the substituted evaluation
    runs on a copy of the run root under a new run id, without the copied evaluation record."""
    import dataclasses
    import shutil

    from tmmri.runs import RunRoot

    ctx = acdc_run
    run = ctx.run.path
    before = _snapshot(run)
    copy_id = "v6-acceptance-20260928-1701"
    shutil.copytree(run, run.parent / copy_id)
    copy = RunRoot(copy_id, run.parent, subfolders=ctx.study.study["artifact_subfolders"])
    copy.record_path("evaluate:acdc_test").unlink()
    with pytest.raises(stages.StageRecordLockedError, match="new run id"):
        stages.evaluate(dataclasses.replace(ctx, force=True), "acdc_test", no_fit=True)   # the original stays evaluated once
    b_path, cohort_path = copy.path / "semantics" / "B_ACDC.parquet", copy.path / "contracts" / "cohort_ACDC.parquet"
    b = pd.read_parquet(b_path)
    test_rows = b["split"] == "acdc_test"
    features = [f["name"] for f in ctx.study.registry["features"] if f["name"] in b.columns]
    for f in features:
        if pd.api.types.is_numeric_dtype(b[f]):
            b.loc[test_rows, f] = SENTINEL
    if "disease_label" in b.columns:
        b.loc[test_rows, "disease_label"] = SENTINEL_CLASS
    b.to_parquet(b_path, engine="pyarrow", index=False)
    cohort = pd.read_parquet(cohort_path)
    cohort.loc[cohort["partition"] == "acdc_test", "acdc_class"] = SENTINEL_CLASS
    cohort.to_parquet(cohort_path, engine="pyarrow", index=False)
    result = stages.evaluate(dataclasses.replace(ctx, run=copy), "acdc_test", no_fit=True)
    assert result.action == "RAN"                                    # the substituted references are stage inputs
    after = _snapshot(copy.path)
    for key in ("d_model", "bhat", "bhat_decoded", *[f"routes_{rb}" for rb in RULEBOOKS_FROM_A], *[f"stability_{rb}" for rb in RULEBOOKS_FROM_A]):
        pd.testing.assert_frame_equal(after[key], before[key], check_exact=True, obj=key)
    assert not after["per_target"]["mae"].equals(before["per_target"]["mae"])   # the evaluator did read the sentinels
    route_metrics = json.loads((copy.path / "metrics" / "routes_d_model_acdc_test.json").read_text(encoding="utf-8"))
    assert route_metrics["covered_reference_accuracy"] in (None, 0.0)
    assert _snapshot(run)["per_target"].equals(before["per_target"])           # the frozen original is untouched
