"""Integration pieces of the protocol-6.0.0 stage DAG (SPECIFICATION.md 6.0.0, Sections 11.2, 11.6, 11.7, 12.4, 13.3
and 14.2): per-section configuration hashes, rulebook payloads routed at inference, non-finite and critical-QC
abstention, M7 eligibility gates, the downstream refit bootstrap, and QC_BLOCKED out-of-domain values on critical rows.
Synthetic inputs only (algorithm controls)."""
from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest

from conftest import REPO

from tmmri.config import load_study
from tmmri.rough_sets import ELIGIBLE, Rule, Rulebook, apply_gate, induce_vprs_rules
from tmmri.rule_pipeline import (
    downstream_refit_bootstrap,
    k_of_n,
    qc_route_frame,
    route_records,
    rulebook_payload,
    semantic_eligibility,
)
from tmmri.stages import SAME_DEFINITION, STAGE_CONFIG_SECTIONS, bundle_sections, reference_labels
from tmmri.wedd import fit_cuts


@pytest.fixture(scope="module")
def study():
    return load_study()


def test_stage_sections_resolve_and_bundle_excludes_evaluation_only_sections(study):
    for stage, sections in STAGE_CONFIG_SECTIONS.items():
        assert study.sections_hash(sections), stage          # every declared section exists (KeyError otherwise)
    bundle = set(bundle_sections())
    assert "bootstrap.threshold_bootstrap" in bundle and "bootstrap.fixed_model_bootstrap" not in bundle and "equivariance" not in bundle
    with pytest.raises(KeyError):
        study.section("bootstrap.no_such_section")


def test_study_copy_outside_repo_and_comment_only_edits(tmp_path, study):
    shutil.copytree(REPO / "configs", tmp_path / "configs", ignore=shutil.ignore_patterns("clinical", "legacy"))
    (tmp_path / "registry").mkdir()
    for path in (REPO / "registry").glob("*.*"):
        shutil.copy2(path, tmp_path / "registry" / path.name)
    copy = load_study(tmp_path / "configs" / "study_v6.yaml")
    assert copy.root == tmp_path and copy.sections_hash(["wedd"]) == study.sections_hash(["wedd"])
    wedd = tmp_path / "configs" / "wedd.yaml"
    wedd.write_text(wedd.read_text(encoding="utf-8") + "# a comment-only edit\n", encoding="utf-8")
    edited = load_study(tmp_path / "configs" / "study_v6.yaml")
    assert edited.config_hash != study.config_hash                       # the file hash changes ...
    assert edited.sections_hash(["wedd"]) == study.sections_hash(["wedd"])   # ... the parsed section does not


def _rule(consequent="NOR", state="b0of2") -> Rule:
    return Rule(rule_id="R_test", antecedent=(("lv_edv_ml", state),), consequent=consequent, support=6, confidence=1.0, strict=True,
                anchors=tuple(f"p{i}" for i in range(6)), antecedent_n=6, tier="STRICT", consequent_kind="d_model")


def test_gated_rulebook_payload_routes_exact_accept_and_qc_first(study):
    book = Rulebook([_rule()], "d_model", "d_model")
    gate = pd.DataFrame([{"rule_id": "R_test", "consequent": "NOR", "length": 1, "oof_matched": 6, "oof_agree": 6, "agreement": 1.0,
                          "wilson_lb": 0.61, "wilson_ub": 1.0, "folds_reproduced": 5, "reproduced_in_folds": "1;2;3;4;5",
                          "eligible": True, "eligibility": ELIGIBLE}])
    name, payload = rulebook_payload("d_model", apply_gate(book, gate), gate, {"gate_substrate": "nested_inner_oof"})
    assert name == "rulebook.json" and payload["records"][0]["eligibility"] == ELIGIBLE      # evaluate routes only ELIGIBLE rules
    assert payload["records"][0]["crossfit_wilson_lower"] == pytest.approx(0.61)
    values = pd.DataFrame({"lv_edv_ml": [50.0, 150.0, 60.0, 55.0]}, index=["ACDC:a", "ACDC:b", "ACDC:c", "ACDC:d"])
    cuts = fit_cuts(pd.DataFrame({"lv_edv_ml": np.r_[np.linspace(40, 90, 10), np.linspace(110, 160, 10)]}),
                    pd.Series(["NOR"] * 10 + ["DCM"] * 10), "wedd_gaussian_mm_v1", study.wedd)
    assert len(cuts.features["lv_edv_ml"].cuts) == 1
    b_rows = pd.DataFrame({"qc_failure_reason": [None, None, "lv_edv_le_lvesv", None]}, index=values.index)
    qc = qc_route_frame(b_rows, np.array([True, True, True, False]), study)
    assert list(qc["qc_abstain"]) == [False, False, True, True]
    assert qc.loc[3, "qc_reason_code"].split(";")[0] == "INPUT_NONFINITE"
    routes = route_records(payload["records"], "d_model", cuts, values, qc, study).set_index("sample_id")
    assert routes.loc["ACDC:a", "route"] == "EXACT_ACCEPT" and routes.loc["ACDC:a", "label"] == "NOR"
    assert routes.loc["ACDC:b", "route"] == "NO_MATCH_ABSTAIN"
    assert routes.loc["ACDC:c", "route"] == "QC_ABSTAIN" and routes.loc["ACDC:d", "reason_code"] == "INPUT_NONFINITE"
    empty_name, empty = rulebook_payload("d_model", Rulebook([], "d_model", "d_model"), gate.iloc[0:0], {})
    assert empty_name == "EMPTY.json" and empty["reason_code"] == "RULEBOOK_EMPTY"


def test_semantic_eligibility_gates(study):
    rng = np.random.default_rng(20260613)
    n = 40
    folds = np.tile(np.arange(1, 6), n // 5).astype(float)
    y = rng.normal(100.0, 20.0, n)
    frame = pd.DataFrame({"lv_edv_ml": y, "lv_edv_ml__state": "OBSERVED", "rv_edv_ml": np.full(n, 80.0), "rv_edv_ml__state": "OBSERVED"})
    oof = pd.DataFrame({"lv_edv_ml": y + rng.normal(0, 2.0, n), "rv_edv_ml": np.full(n, 80.0)})
    mean_oof = pd.DataFrame({"lv_edv_ml": np.full(n, y.mean()), "rv_edv_ml": np.full(n, 80.0)})
    stability = pd.DataFrame([{"feature": "lv_edv_ml", "cut": 100.0, "recurrence": 0.9}, {"feature": "rv_edv_ml", "cut": np.nan, "recurrence": np.nan}])
    table = semantic_eligibility(rulebook="d_model", reference=frame, fit_rows=frame, oof=oof, mean_oof=mean_oof, folds=folds, stability=stability,
                                 candidates=["lv_edv_ml", "rv_edv_ml"], rule_policy=study.rule_policy, registry=study.registry).set_index("feature")
    assert bool(table.loc["lv_edv_ml", "eligible"]) and table.loc["lv_edv_ml", "folds_improved"] == 5
    assert not bool(table.loc["rv_edv_ml", "eligible"]) and "INELIGIBLE_VARIABILITY" in table.loc["rv_edv_ml", "ineligibility"]
    unstable = stability.assign(recurrence=[0.4, np.nan])
    table = semantic_eligibility(rulebook="measured_oracle", reference=frame, fit_rows=frame, oof=None, mean_oof=None, folds=None,
                                 stability=unstable, candidates=["lv_edv_ml"], rule_policy=study.rule_policy, registry=study.registry)
    assert table.loc[0, "ineligibility"] == "INELIGIBLE_UNSTABLE_CUT" and table.loc[0, "reconstruction_pass"] is None
    assert k_of_n("4_of_5") == (4, 5)


def test_downstream_refit_bootstrap_recovers_a_planted_rule(study):
    rng = np.random.default_rng(20260613)
    n = 60
    targets = study.transition_targets()
    b = pd.DataFrame({t: rng.normal(50.0, 5.0, n) for t in targets})
    b["lv_edv_ml"] = np.r_[rng.normal(80.0, 3.0, n // 2), rng.normal(160.0, 3.0, n // 2)]
    for t in targets:
        b[f"{t}__state"] = "OBSERVED"
    decision = np.array(["NOR"] * (n // 2) + ["DCM"] * (n // 2), dtype=object)
    ids = [f"ACDC:p{i:03d}" for i in range(n)]
    values = b[["lv_edv_ml", "lv_edv_ml__state"]].set_axis(ids)
    cuts = fit_cuts(values, pd.Series(decision, index=ids), "wedd_gaussian_mm_v1", study.wedd)
    reference = induce_vprs_rules(cuts.transform(values), pd.Series(decision, index=ids), pd.Series(ids, index=ids), study.rule_policy,
                                  registry=study.registry, attributes=["lv_edv_ml"], consequent_kind="measured_oracle", rulebook_id="measured_oracle")
    assert reference.rules
    folds = np.tile(np.arange(1, 6), n // 5).astype(float)
    table = downstream_refit_bootstrap(A=rng.normal(size=(n, 57)), B=b, folds=folds, inner_labels=lambda f: np.where(folds == f, np.nan, folds),
                                       row_ids=ids, study=study, rulebooks={"measured_oracle": {"substrate": "measured", "decision": decision,
                                                                                              "attributes": ["lv_edv_ml"], "reference": reference, "cuts": cuts}},
                                       profile="wedd_gaussian_mm_v1", n_replicates=4, seed=20260928, n_jobs=1)
    assert (table["n_valid_replicates"] == 4).all() and (table["exact_recurrence"] == 1.0).all() and (table["relaxed_recurrence"] == 1.0).all()


def test_mms_reference_labels_use_same_definition_crosswalk_only():
    cohort = pd.DataFrame({"crosswalk_acdc_class": ["NOR", "MINF", None], "crosswalk_compatibility": [SAME_DEFINITION, "HARMONIZABLE_NOT_EXACT", "OFF_ONTOLOGY"]})
    out = reference_labels(cohort, "MMS")
    assert out.iloc[0] == "NOR" and pd.isna(out.iloc[1]) and pd.isna(out.iloc[2])


def test_out_of_domain_value_on_critical_row_is_qc_blocked(registry, study):
    from tmmri.semantics import features as feat
    from tmmri.semantics.contract import validate_typed_b
    import test_v6_semantics as sem

    case = sem._case(ed=sem._phantom_mask(7, 13, 6), es=sem._phantom_mask(10, 14, 8))   # ESV > EDV: EF < 0 and critical QC
    row = feat.case_semantics_v6(case, registry, cohorts=study.cohorts, reason_codes=study.reason_codes)
    assert "lv_edv_le_lvesv" in row["qc_failure_reason"]
    assert row["lv_ef_percent__state"] == "QC_BLOCKED" and row["lv_ef_percent"] is None and row["lv_ef_percent__evidence"] is None
    table = feat.build_semantics_table_v6([case], registry, 1, cohorts=study.cohorts, reason_codes=study.reason_codes)
    validate_typed_b(table, registry, cohorts_cfg=study.cohorts)          # strict contract: no out-of-domain OBSERVED value


def test_whole_pixel_translation_leaves_b_identical(registry, study):
    """Acceptance test 21 (SPECIFICATION.md 15.2): a +-1 px translation of image and mask leaves B unchanged."""
    from tmmri.equivariance import apply_transformation_v6
    from tmmri.semantics import features as feat
    import test_v6_semantics as sem

    case = sem._case()
    base = feat.case_semantics_v6(case, registry, cohorts=study.cohorts, reason_codes=study.reason_codes)
    cfg = study.configs["equivariance"]
    for name in ("translation_x", "translation_y"):
        for step in (-1.0, 1.0):
            _, ed = apply_transformation_v6(case.ed_image, case.ed_mask, name, step, cfg)
            _, es = apply_transformation_v6(case.es_image, case.es_mask, name, step, cfg)
            moved = sem.StagedCase(case.sample_id, case.dataset, case.case_id, case.ed_image, ed, case.es_image, es, case.spacing_mm,
                                   case.geometry_method, case.affine, case.meta)
            row = feat.case_semantics_v6(moved, registry, cohorts=study.cohorts, reason_codes=study.reason_codes)
            for entry in registry["features"]:
                f = entry["name"]
                assert row[f"{f}__state"] == base[f"{f}__state"], (name, step, f)
                if base[f"{f}__state"] == "OBSERVED":
                    assert row[f] == pytest.approx(base[f], rel=1e-9, abs=1e-9), (name, step, f)


def test_blocked_staging_writes_no_data_bearing_artifact(tmp_path, monkeypatch):
    """Acceptance test 20: a source that fails verification blocks the run; no B, A, T or rulebook is written."""
    import fixtures_v6 as fx
    from tmmri import stages
    from tmmri.runs import RunRoot

    monkeypatch.setattr(stages, "protocol_frozen", lambda *a, **k: False)
    acdc = fx.make_acdc_source(tmp_path / "src" / "acdc", fx.acdc_patients(1))
    manifest = fx.acdc_manifest(acdc)
    fx.corrupt_one_byte(acdc / "training" / "patient001" / "patient001_frame01_gt.nii.gz")
    study = load_study()
    run = RunRoot("v6-blocked-20260928-1600", tmp_path / "artifacts", subfolders=study.study["artifact_subfolders"])
    ctx = stages.StageContext(study=study, run=run, data_root=tmp_path / "data", n_jobs=1, sources={"acdc": acdc, "mms": tmp_path / "absent"},
                              source_manifests={"ACDC": manifest})
    with pytest.raises(RuntimeError, match="stage-sources:acdc failed"):
        stages.run_pipeline(ctx)
    status = run.status()["stages"]
    assert status["stage-sources:ACDC"]["status"] == "BLOCKED_DATA_ACCESS"
    written = [p.name for p in run.path.rglob("*") if p.is_file()]
    assert not [n for n in written if n.startswith(("B_", "A_", "T_", "transition_model", "rulebook"))], written
