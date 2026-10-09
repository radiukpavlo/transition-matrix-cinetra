"""Protocol 6.0.0 CLI surface, M&Ms freeze guard, cohort validation, bundle freeze and orchestration."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pandas as pd
import pytest

import fixtures_v6 as fx
from tmmri import cli, stages
from tmmri.config import load_study
from tmmri.stages import (
    BundleError,
    CohortValidationError,
    ProtocolFreezeError,
    StageContext,
    extract_formal,
    freeze_analysis_bundle,
    methods_api,
    run_pipeline,
    stage_sources,
    validate_cohort,
    verify_frozen_bundle,
)
from tmmri.runs import RunRoot

V6_SUBCOMMANDS = ["stage-sources", "validate-cohort", "build-semantics", "extract-formal", "fit-transition", "fit-predictor",
                  "discretize", "check-semantic-eligibility", "induce-rules", "gate-rules", "freeze-analysis-bundle", "evaluate",
                  "audit-shift", "fit-refit-arm", "summarize-predictor", "run-ablations", "run-equivariance", "build-manuscript-tables",
                  "write-release-manifest", "run", "bench", "build-manuscript-figures"]
V6_SUBCOMMANDS += ["extract-mms2"]   # protocol-v6.0.1 (M&Ms-2)
V6_SUBCOMMANDS += ["evaluate-h4"]    # protocol-v6.0.1 (Decision D36: the H4 family)


@pytest.mark.parametrize("command", V6_SUBCOMMANDS)
def test_every_v6_subcommand_has_help(command, capsys):
    with pytest.raises(SystemExit) as info:
        cli.main([command, "--help"])
    assert info.value.code == 0
    assert command in capsys.readouterr().out
    assert set(V6_SUBCOMMANDS) == set(cli.V6_COMMANDS)


def test_legacy_subcommands_still_parse(capsys):
    for command in ("init-layout", "run-minimal", "validate-native-data", "run-full-validation", "prepare-clinical-validation",
                    "analyze-reader-study", "package-clinical-release"):
        with pytest.raises(SystemExit) as info:
            cli.main([command, "--help"])
        assert info.value.code == 0


def test_init_layout_creates_v6_roots_not_project_history():
    assert "docs/project_history" not in cli.LAYOUT_DIRS
    roots = [p.as_posix() for p in cli.v6_runtime_roots()]
    assert any(r.endswith("/staged") for r in roots) and any(r.endswith("/manifests") for r in roots)


@pytest.fixture()
def synthetic(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "protocol_frozen", lambda *a, **k: False)
    acdc = fx.make_acdc_source(tmp_path / "src" / "acdc", fx.acdc_patients(1))
    mms = fx.make_mms_source(tmp_path / "src" / "mms")
    study = load_study()
    run = RunRoot("v6-synthetic-20260928-1300", tmp_path / "artifacts", subfolders=study.study["artifact_subfolders"])
    ctx = StageContext(study=study, run=run, data_root=tmp_path / "data", n_jobs=1, sources={"acdc": acdc, "mms": mms},
                       source_manifests={"ACDC": fx.acdc_manifest(acdc), "MMS": fx.mms_manifest(mms)}, mms_metadata=fx.mms_metadata(mms))
    return ctx


def test_mms_endpoint_stages_refuse_before_the_freeze(synthetic):
    ctx = synthetic
    assert stage_sources(ctx, "mms").ok                          # staging M&Ms is always allowed
    validate_cohort(ctx)
    for call in (lambda: extract_formal(ctx, "mms"), lambda: stages.build_semantics(ctx, "mms"),
                 lambda: stages.evaluate(ctx, "mms_external", no_fit=True), lambda: stages.audit_shift(ctx), lambda: stages.fit_refit_arm(ctx)):
        with pytest.raises(ProtocolFreezeError, match="protocol freeze"):
            call()
    ctx.allow_unfrozen_synthetic = True                          # synthetic fixture + explicit override
    result = extract_formal(ctx, "mms")
    assert result.action == "RAN"
    table = pd.read_parquet(ctx.run.path / "formal" / "A_MMS.parquet")
    assert list(table["case_id"]) == ["A1A1A1", "B2B2B2", "C3C3C3", "D4D4D4"]
    manifest_path = ctx.manifests_dir / "staging_MMS.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["synthetic_fixture"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ProtocolFreezeError, match="synthetic"):
        extract_formal(ctx, "mms")


def test_cli_exit_code_for_refused_mms_stage(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(stages, "protocol_frozen", lambda *a, **k: False)
    for argv in (["extract-formal", "--dataset", "mms"], ["build-semantics", "--dataset", "mms"],
                 ["evaluate", "--split", "mms_external", "--no-fit"], ["audit-shift"], ["fit-refit-arm"]):
        with pytest.raises(SystemExit) as info:
            cli.main(argv + ["--run-id", "v6-guard-20260928-1301", "--artifact-root", str(tmp_path / "a"), "--data-root", str(tmp_path / "d")])
        assert info.value.code == cli.EXIT_PROTOCOL_NOT_FROZEN
    assert "REFUSED" in capsys.readouterr().out


def test_validate_cohort_and_extract_formal_on_synthetic_data(synthetic):
    ctx = synthetic
    assert stage_sources(ctx, "acdc").ok and stage_sources(ctx, "mms").ok
    report = validate_cohort(ctx)
    checks = {c["check"]: c for c in report["checks"]}
    for name in ("mms_duplicate_pairs", "mms_duplicates_excluded", "acdc_ed_es_zero_based", "acdc_staged", "mms_staged",
                 "mms_staged_sources_match_tracked_manifest", "mms_mask_annotated_exactly_on_ed_es",
                 "decoded_voxel_duplicates_within_and_across_cohorts", "sample_id_disjoint_across_cohorts"):
        assert name in checks
    assert checks["acdc_staged"]["passed"] and checks["mms_staged_sources_match_tracked_manifest"]["passed"]
    assert checks["mms_mask_annotated_exactly_on_ed_es"]["passed"]
    assert not checks["mms_external_size"]["passed"]              # synthetic sizes differ from configs/cohorts.yaml
    with pytest.raises(CohortValidationError):
        validate_cohort(ctx, strict=True)
    split = pd.read_parquet(ctx.run.path / "contracts" / "split_manifest.parquet")
    assert set(split["partition"]) == {"acdc_dev", "mms_external", "excluded"}
    first = extract_formal(ctx, "acdc")
    assert first.action == "RAN" and extract_formal(ctx, "acdc").action == "SKIPPED"
    a = pd.read_parquet(ctx.run.path / "formal" / "A_ACDC.parquet")
    p6 = pd.read_parquet(ctx.run.path / "formal" / "A_ACDC_p6_128.parquet")
    assert a.shape == (5, 60) and p6.shape == (5, 131)
    extractor = json.loads((ctx.run.path / "formal" / "A_ACDC_extractor.json").read_text(encoding="utf-8"))
    assert extractor["representation"] == "formal_descriptors_v3" and extractor["dimension"] == 57


def test_methods_api_names_missing_functions():
    with pytest.raises(NotImplementedError, match="tmmri.no_such_module.fn"):
        methods_api("tmmri.no_such_module", "fn")
    with pytest.raises(NotImplementedError, match="tmmri.runs.no_such_function"):
        methods_api("tmmri.runs", "no_such_function")


def _bundle(run):
    (run.path / "transitions" / "transition_model.json").write_text("{}", encoding="utf-8")
    (run.path / "formal" / "d_model_predictor.joblib").write_bytes(b"model")
    for rb in ("d_model", "dataset_label", "measured_oracle"):
        (run.path / "rules" / rb).mkdir(parents=True, exist_ok=True)
        (run.path / "rules" / rb / "rulebook.json").write_text(json.dumps({"rulebook": rb, "records": []}), encoding="utf-8")


def test_freeze_analysis_bundle(synthetic):
    ctx = synthetic
    with pytest.raises(BundleError, match="incomplete"):
        freeze_analysis_bundle(ctx)
    _bundle(ctx.run)
    frozen = freeze_analysis_bundle(ctx)
    assert set(frozen["bundle"]) >= {"transitions/transition_model.json", "formal/d_model_predictor.joblib", "rules/d_model/rulebook.json"}
    assert frozen["config"]["config_hash"] == ctx.study.config_hash and "code_sha256" in frozen["code"]
    assert "utc" not in json.dumps(frozen)
    assert freeze_analysis_bundle(ctx)["bundle_sha256"] == frozen["bundle_sha256"]
    assert verify_frozen_bundle(ctx)["bundle_sha256"] == frozen["bundle_sha256"]
    (ctx.run.path / "rules" / "d_model" / "rulebook.json").write_text("{}", encoding="utf-8")
    with pytest.raises(BundleError, match="changed after the freeze"):
        verify_frozen_bundle(ctx)
    with pytest.raises(BundleError, match="already frozen"):
        freeze_analysis_bundle(ctx)


def test_run_pipeline_selection_and_blocking(synthetic):
    ctx = synthetic
    done = run_pipeline(ctx, until="stage-sources:mms")
    assert [d["step"] for d in done] == ["stage-sources:acdc", "stage-sources:mms"]
    assert [s for s, _ in stages._select_steps("extract-formal", "build-semantics")] == ["build-semantics", "extract-formal"]
    assert stages._select_steps("extract-formal:mms", "extract-formal:mms") == [("extract-formal", "mms")]
    with pytest.raises(ValueError):
        stages._select_steps("validate-cohort", "fit-transition")
    with pytest.raises(CohortValidationError):
        run_pipeline(ctx, start="validate-cohort", until="validate-cohort")
    status = json.loads(ctx.run.status_path.read_text(encoding="utf-8"))
    assert status["stages"]["validate-cohort"]["status"] == "BLOCKED_COMPUTE"


METHODS_MODULES = ("tmmri.target_transforms", "tmmri.wedd", "tmmri.rough_sets", "tmmri.inference")
REPO = Path(__file__).resolve().parents[1]
# Speed-only overrides of the copied study (algorithm control): identical code paths, fewer bootstrap replicates.
FAST_BOOTSTRAP = {"threshold_bootstrap": 20, "fixed_model_bootstrap": 200, "downstream_refit_bootstrap": 3}


def _study_copy(tmp_path: Path) -> Path:
    """Copy of configs/ and the registry root files with reduced bootstrap replicates; returns its study_v6.yaml."""

    import yaml

    root = tmp_path / "study"
    shutil.copytree(REPO / "configs", root / "configs", ignore=shutil.ignore_patterns("clinical", "legacy"))
    (root / "registry").mkdir(parents=True)
    for path in (REPO / "registry").glob("*.*"):
        shutil.copy2(path, root / "registry" / path.name)
    boot_path = root / "configs" / "bootstrap.yaml"
    boot = yaml.safe_load(boot_path.read_text(encoding="utf-8"))
    for key, replicates in FAST_BOOTSTRAP.items():
        boot[key]["replicates"] = replicates
    boot_path.write_text(yaml.safe_dump(boot, sort_keys=False), encoding="utf-8")
    return root / "configs" / "study_v6.yaml"


def _edit_yaml(path: Path, edit) -> None:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    edit(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _run(argv: list[str], capsys) -> tuple[int, dict | str]:
    """(exit code, parsed JSON summary on success, else the captured output)."""

    code = 0
    try:
        cli.main(argv)
    except SystemExit as exc:
        code = int(exc.code or 0)
    out = capsys.readouterr().out
    start = out.find("{")
    return code, (json.loads(out[start:]) if code == 0 and start >= 0 else out)


def test_cli_run_synthetic_end_to_end(tmp_path, monkeypatch, capsys):
    """``tmmri.cli run --study .../configs/study_v6.yaml`` over every v6 stage on synthetic fixtures (algorithm control).

    Covers the refit arm, the predictor summary, the ablation matrix, equivariance, the manuscript tables and the release
    manifest; a second run skips every stage (content addressing); in the frozen run an evaluation-only configuration
    change makes the recorded evaluation refuse (once-only guard, exit 7) while a stage without a record still runs and
    the release manifest re-runs."""

    from conftest import require_module

    for module in METHODS_MODULES:
        require_module(module)
    monkeypatch.setattr(stages, "protocol_frozen", lambda *a, **k: False)
    acdc = fx.make_acdc_source(tmp_path / "src" / "acdc", fx.acdc_patients(5, testing_per_class=2))
    fx.write_fixture_manifests(acdc, fx.acdc_manifest(acdc))
    cases = fx.mms_refit_cases()
    mms = fx.make_mms_source(tmp_path / "src" / "mms", cases, shape=(26, 24, 3))   # room for the RV crescent at ED
    fx.write_fixture_manifests(mms, fx.mms_manifest(mms, cases), fx.mms_metadata(mms))
    study_path = _study_copy(tmp_path)
    run_id = "v6-synthetic-20260928-1500"
    base = ["run", "--study", str(study_path), "--run-id", run_id, "--artifact-root", str(tmp_path / "artifacts"),
            "--data-root", str(tmp_path / "data"), "--acdc-source", str(acdc), "--mms-source", str(mms), "--n-jobs", "1"]
    # Without the override the synthetic cohort sizes are not waived: strict cohort validation stops the run.
    code, _ = _run(base, capsys)
    assert code == cli.EXIT_COHORT_VALIDATION
    code, first = _run(base + ["--allow-unfrozen-synthetic"], capsys)
    assert code == 0, first
    steps = [stages.step_id(*s) for s in stages.PIPELINE_STEPS]
    assert [d["step"] for d in first["steps"]] == steps
    for name in ("check-semantic-eligibility", "run-ablations", "run-equivariance", "write-release-manifest", "fit-refit-arm"):
        assert name in {stage for stage, _ in stages.PIPELINE_STEPS}
    run = tmp_path / "artifacts" / run_id
    status = json.loads((run / "artifact_status.json").read_text(encoding="utf-8"))
    assert status["data"]["status"] == "UNIT_TESTED_SYNTHETIC"
    recorded = {"stage-sources:acdc": "stage-sources:ACDC", "stage-sources:mms": "stage-sources:MMS", "build-semantics:acdc": "build-semantics:ACDC",
                "build-semantics:mms": "build-semantics:MMS", "extract-formal:acdc": "extract-formal:ACDC", "extract-formal:mms": "extract-formal:MMS",
                "run-equivariance:acdc": "run-equivariance"}
    for sid in steps:
        entry = status["stages"][recorded.get(sid, sid)]
        assert entry["status"] in {"BUILT", "EVALUATED"}, (sid, entry)
    content_addressed = [s for s in steps if not s.startswith("stage-sources") and s not in {"freeze-analysis-bundle", "write-release-manifest"}]
    for sid in content_addressed:
        record = json.loads((run / ".stages" / f"{recorded.get(sid, sid).replace(':', '__')}.json").read_text(encoding="utf-8"))
        assert record["payload"]["params"]["config_sections"], sid
    gate = json.loads((run / ".stages" / "gate-rules.json").read_text(encoding="utf-8"))
    assert gate["payload"]["params"]["substrate"] == "nested_inner_oof"
    assert all((run / "oof" / f"inner_oof_ACDC_dev_fold{f}.parquet").is_file() for f in range(1, 6))
    for rb in ("d_model", "dataset_label", "measured_oracle"):
        pointer = json.loads((run / "rules" / rb / "rulebook_or_empty.json").read_text(encoding="utf-8"))
        book = json.loads((run / "rules" / rb / pointer["file"]).read_text(encoding="utf-8"))
        assert book["provenance"]["gate_substrate"] == ("measured_B" if rb == "measured_oracle" else "nested_inner_oof")
        assert all(r["eligibility"] == "ELIGIBLE" for r in book.get("records", []))
    eligibility = json.loads((run / "rules" / "semantic_eligibility.json").read_text(encoding="utf-8"))
    assert set(eligibility["rulebooks"]) == {"d_model", "dataset_label", "measured_oracle"}
    frozen = json.loads((run / ".stages" / "FROZEN.json").read_text(encoding="utf-8"))
    assert not any("refit_arm" in rel or "ablations" in rel for rel in frozen["bundle"])
    assert "bootstrap.fixed_model_bootstrap" not in frozen["bundle_config"]["sections"]
    refit = json.loads((run / "metrics" / "refit_arm" / "primary_rho_mms_refit_eval.json").read_text(encoding="utf-8"))
    assert refit["pooled_with_primary"] is False and set(refit["refit_arm"]) == {"all", "seen_vendors", "unseen_vendors"}
    assert refit["refit_arm"]["all"]["n_cases"] == refit["frozen_acdc_bundle"]["all"]["n_cases"] == 7
    summary = json.loads((run / "metrics" / "refit_arm" / "refit_arm_summary.json").read_text(encoding="utf-8"))
    assert summary["n_train_partition"] == 31 and summary["n_train_fit_rows"] >= 30 and summary["predictor_classes"] == ["DCM", "HCM", "NOR"]
    boot = pd.read_parquet(run / "rules" / "refit_arm" / "downstream_refit_bootstrap.parquet")
    assert set(boot.columns) >= {"exact_recurrence", "relaxed_recurrence", "n_valid_replicates"}
    for name in ("transition_variants.json", "discretizers.json", "rule_policy.json", "gate_thresholds.json", "ablation_index.json"):
        payload = json.loads((run / "metrics" / "ablations" / name).read_text(encoding="utf-8"))
        assert payload["role"] == "secondary_ablation" and payload["replaces_primary"] is False
    assert set(json.loads((run / "metrics" / "ablations" / "discretizers.json").read_text(encoding="utf-8"))["profiles"]) == {
        "wedd_gaussian_mm_v1", "wedd_uniform_mm_v1", "three_quantile_v1"}
    assert set(json.loads((run / "metrics" / "ablations" / "rule_policy.json").read_text(encoding="utf-8"))["variants"]) >= {
        "beta_0.05", "beta_0.10", "beta_0.20"}
    selection = json.loads((run / "equivariance" / "equivariance_selection_ACDC.json").read_text(encoding="utf-8"))
    assert len(selection["selected_patients"]) == 12 and selection["feeds_primary_rulebook"] is False
    manifest = json.loads((run / "release_manifest.json").read_text(encoding="utf-8"))
    assert manifest["synthetic_fixture"] is True and "configs/bootstrap.yaml" in manifest["config"]["files"]
    assert "artifact_status.json" not in manifest["files"] and "rules/semantic_eligibility.json" in manifest["files"]
    routes = pd.read_parquet(run / "inference" / "routes_d_model_acdc_test.parquet")
    assert len(routes) == 10 and set(routes["route"]) <= set(load_study(study_path).reason_codes["routes"])
    _assert_protocol_outputs(run, load_study(study_path))
    _assert_predictor_and_tables(run, load_study(study_path))
    # Second run: every stage is content-addressed and skipped.
    code, second = _run(base + ["--allow-unfrozen-synthetic"], capsys)
    assert code == 0, second
    assert {d["action"] for d in second["steps"]} == {"SKIPPED"}, second["steps"]
    # Once-only guard: in the frozen run an evaluation-only change (fixed-model bootstrap) makes the recorded evaluation
    # refuse (exit 7) instead of re-executing; its record and status stay as they were.
    record = run / ".stages" / "evaluate__acdc_test.json"
    record_bytes = record.read_bytes()
    _edit_yaml(study_path.parent / "bootstrap.yaml", lambda d: d["fixed_model_bootstrap"].update(replicates=150))
    code, out = _run(base + ["--allow-unfrozen-synthetic"], capsys)
    assert code == cli.EXIT_BUNDLE and "evaluate:acdc_test" in out and "once-only guard" in out and "new run id" in out, out
    assert record.read_bytes() == record_bytes
    assert json.loads((run / "artifact_status.json").read_text(encoding="utf-8"))["stages"]["evaluate:acdc_test"]["status"] == "EVALUATED"
    code, out = _run(base + ["--allow-unfrozen-synthetic", "--from", "summarize-predictor", "--until", "summarize-predictor"], capsys)
    assert code == cli.EXIT_BUNDLE and "summarize-predictor" in out, out                  # --from/--until obey the guard
    # A stage without a record still runs in the frozen run; the release manifest (not a recorded stage) re-runs.
    options = base[1:] + ["--allow-unfrozen-synthetic"]
    (run / ".stages" / "build-manuscript-tables.json").unlink()
    code, rebuilt = _run(["build-manuscript-tables", *options], capsys)
    assert code == 0 and rebuilt["action"] == "RAN", rebuilt
    code, manifest_run = _run(["write-release-manifest", *options], capsys)
    assert code == 0 and manifest_run["action"] == "RAN", manifest_run
    code, tail = _run(base + ["--allow-unfrozen-synthetic", "--from", "build-manuscript-tables"], capsys)
    expected_tail = [s for s in steps if steps.index(s) >= steps.index("build-manuscript-tables")]   # v6.0.1: + build-manuscript-figures
    assert code == 0 and [d["action"] for d in tail["steps"]] == ["SKIPPED"] * len(expected_tail), tail
    # A change to a fitting stage's configuration after the freeze is refused (a new run id is required).
    _edit_yaml(study_path.parent / "transition.yaml", lambda d: d["targets"].update(min_observed_fit_rows=11))
    code, _ = _run(base + ["--allow-unfrozen-synthetic"], capsys)
    assert code == cli.EXIT_BUNDLE


def _assert_predictor_and_tables(run: Path, study) -> None:
    """summarize-predictor (SPECIFICATION.md 13.2) and build-manuscript-tables (Section 16) on the synthetic run: every
    partition summarized, the ten tables with complete provenance, no case identifier anywhere, small strata withheld."""

    from tmmri.manuscript_tables import TABLE_NAMES

    metrics = run / "metrics"
    index = json.loads((metrics / "predictor_summary.json").read_text(encoding="utf-8"))
    assert set(index["partitions"]) == {"acdc_dev_oof", "acdc_test", "mms_external", "mms_refit_eval"}
    assert index["semantic_comparator"]["status"] == "NE" and index["semantic_comparator"]["reason"] == "COMPARATOR_NOT_FITTED_IN_RUN"
    oof = json.loads((metrics / "predictor_acdc_dev_oof.json").read_text(encoding="utf-8"))
    assert oof["counts"]["n_evaluable"] == 25 and oof["metrics"]["macro_ovr_auroc"]["reason"] == "OOF_PROBABILITIES_NOT_RECORDED"
    ext = json.loads((metrics / "predictor_mms_external.json").read_text(encoding="utf-8"))
    cohort = pd.read_parquet(run / "contracts" / "cohort_MMS.parquet")
    same = cohort["mms_external"] & (cohort["crosswalk_compatibility"] == "HARMONIZABLE_SAME_DEFINITION")
    assert ext["counts"]["n_same_definition"] == int(same.sum()) and ext["ihd_not_exact"]["pooled"] is False
    assert ext["ihd_not_exact"]["n"] == int((cohort["mms_external"] & (cohort["pathology"] == "IHD")).sum())
    assert sum(ext["off_ontology"]["d_model_distribution"].values()) == ext["off_ontology"]["n"]
    assert len(ext["ece_realized_bins"]) == study.bootstrap["calibration"]["ece_bins"]
    assert (metrics / "refit_arm" / "predictor_mms_refit_eval.json").is_file() and (metrics / "predictor_confusion.parquet").is_file()
    tables = run / "manuscript_tables"
    provenance = json.loads((tables / "provenance.json").read_text(encoding="utf-8"))
    assert set(provenance["tables"]) == set(TABLE_NAMES) and len(TABLE_NAMES) == 10
    split = pd.read_parquet(run / "contracts" / "split_manifest.parquet")
    identifiers = set(split["case_id"].astype(str)) | set(split["patient_id"].astype(str))
    for name in TABLE_NAMES:
        entry = provenance["tables"][name]
        for ext_name in ("csv", "tex"):
            text = (tables / f"{name}.{ext_name}").read_text(encoding="utf-8")
            assert not set(re.findall(r"[A-Za-z0-9]+", text)) & identifiers, (name, ext_name)
        assert entry["sources"] and all(s["sha256"] for s in entry["sources"]), name
        for column in entry["numeric_columns"]:
            assert entry["columns"][column] and all(p["sha256"] and p["keys"] for p in entry["columns"][column]), (name, column)
    assert not set(re.findall(r"[A-Za-z0-9]+", (tables / "provenance.json").read_text(encoding="utf-8"))) & identifiers
    min_cases = study.bootstrap["subgroup_reporting_thresholds"]["min_cases_per_stratum"]
    t5 = pd.read_csv(tables / "table05_external_reconstruction_mms_external.csv", dtype=str, keep_default_na=False)
    strata = t5[t5["panel"] == "strata"]
    small = strata[strata["n"].astype(int) < min_cases]
    assert len(small) and (small[["rho", "ci_lower", "ci_upper"]] == f"NE (n < {min_cases})").all().all()
    assert (strata.loc[strata["stratum_variable"].isin(["sex", "age_band"]), "role"].str.contains("descriptive")).all()
    t6 = pd.read_csv(tables / "table06_audited_predictor.csv", dtype=str, keep_default_na=False)
    assert (t6.loc[t6["panel"] == "semantic_comparator", "note"] == "COMPARATOR_NOT_FITTED_IN_RUN").all()
    t1 = pd.read_csv(tables / "table01_methodological_reconciliation.csv")
    assert len(t1) == 20 and provenance["specification"]["matches_protocol_tag"] is not False
    assert provenance["manuscript_directory_written"] is False


def _assert_protocol_outputs(run: Path, study) -> None:
    """Prespecified outputs of evaluate, audit-shift and run-ablations on the synthetic run (SPECIFICATION.md 12.3,
    13.2-13.3, 16): external strata with NE below the reporting threshold, the fixed-sequence decision with the 220-case
    sensitivity, BH tables within F1-F5, the shift audit, the linearity criterion and the WEDD settings grid."""

    metrics = run / "metrics"
    min_cases = study.bootstrap["subgroup_reporting_thresholds"]["min_cases_per_stratum"]
    errors = pd.read_parquet(metrics / "strata_errors_mms_external.parquet")
    thresholds = study.bootstrap["subgroup_reporting_thresholds"]
    descriptive = set(thresholds["descriptive_strata"])
    assert set(errors["stratum_variable"]) == set(thresholds["external_strata"]) | descriptive
    demo = errors[errors["stratum_variable"].isin(descriptive)]
    assert (demo["stratum_role"] == "descriptive_demographic").all() and demo["descriptive_only"].all()
    assert demo[["pearson_p", "spearman_p"]].isna().all().all() and set(demo["inference_status"]) <= {"DESCRIPTIVE", "NE"}
    assert set(errors.loc[errors["stratum_variable"] == "refit_vendor_group", "stratum"]) == {"seen_vendors", "unseen_vendors"}
    small = errors[errors["stratum_n"] < min_cases]
    assert len(small) and (small["inference_status"] == "NE").all() and (small["inference_reason"] == "STRATUM_BELOW_MIN_CASES").all()
    assert small["mae"].notna().any()                                   # point estimates kept, never dropped
    routes = pd.read_parquet(metrics / "strata_routes_mms_external.parquet")
    assert set(routes["rulebook"]) == {"d_model", "dataset_label", "measured_oracle"}
    assert (routes.groupby(["rulebook", "stratum_variable"])["n"].sum() == 38).all()
    rho = json.loads((metrics / "strata_rho_mms_external.json").read_text(encoding="utf-8"))
    vendor_c = rho["strata"]["vendor"]["C"]
    assert vendor_c["status"] == "NE" and vendor_c["reason"] == "STRATUM_BELOW_MIN_CASES" and vendor_c["ci_lower"] is None
    sequence = json.loads((metrics / "fixed_sequence.json").read_text(encoding="utf-8"))["results"]
    assert [r["hypothesis_id"] for r in sequence] == ["H-primary-1", "H-primary-2", "H-primary-2-never-analysed"]
    assert sequence[2]["in_hierarchy"] is False and sequence[2]["tested"] is True
    families = study.bootstrap["multiplicity"]["families"]
    for path, short in ((metrics / "multiplicity_F1.parquet", "F1"), (metrics / "multiplicity_F3.parquet", "F3"),
                        (metrics / "ablations" / "multiplicity_F2.parquet", "F2"), (metrics / "ablations" / "multiplicity_F4.parquet", "F4"),
                        (metrics / "ablations" / "multiplicity_F5.parquet", "F5")):
        table = pd.read_parquet(path)
        assert len(table) and set(table["family"]) == {next(f for f in families if f.startswith(short + "_"))}, path
        tested = table["p_value"].notna()
        assert (table.loc[tested, "q_value"] >= table.loc[tested, "p_value"] - 1e-12).all() and table.loc[~tested, "q_value"].isna().all()
    f3 = pd.read_parquet(metrics / "multiplicity_F3.parquet")
    assert "all" in set(f3["stratum"]) and (f3.loc[f3["reason"] == "STRATUM_BELOW_MIN_CASES", "status"] == "NE").all()
    assert not set(f3.get("stratum_variable", pd.Series(dtype=object)).dropna()) & descriptive   # demographic strata are never tested
    shift = json.loads((metrics / "shift_audit_mms_external.json").read_text(encoding="utf-8"))
    assert shift["label_free"] is True and {r["comparison"] for r in shift["results"]} == {"pooled", "vendor"}
    pooled = next(r for r in shift["results"] if r["comparison"] == "pooled")
    assert 0.0 <= pooled["auroc"] <= 1.0 and pooled["n_comparison"] == 38
    assert {r["group"] for r in shift["results"] if r["comparison"] == "vendor"} == {"A", "B", "C", "D"}
    decoder = json.loads((metrics / "decoder_mms_external.json").read_text(encoding="utf-8"))
    assert set(decoder["targets"]) == set(study.transition_targets())
    for split in ("acdc_test", "mms_external"):
        stab = pd.read_parquet(run / "reconstructed" / f"stability_d_model_{split}.parquet")
        assert set(stab.drop(columns="sample_id").stack().unique()) <= {"ESTIMATED", "UNSTABLE_ESTIMATE", ""}
    ablations = metrics / "ablations"
    lin = json.loads((ablations / "linearity_criterion.json").read_text(encoding="utf-8"))
    assert lin["comparator"] == "rbf_kernel_ridge" and isinstance(lin["nonlinear_declared_better"], bool) and len(lin["per_fold"]) == 5
    assert lin["executed_audit_pipeline"] == "affine"
    variants = json.loads((ablations / "transition_variants.json").read_text(encoding="utf-8"))
    assert {"ridge_global_lambda", "rbf_kernel_ridge"} <= set(variants["acdc_test"]) and variants["global_lambda"][0]["scope"] == "global"
    wedd = json.loads((ablations / "wedd_settings.json").read_text(encoding="utf-8"))
    assert "max_depth_3_max_bins_4" in wedd["variants"] and "weight_entropy_0.65" not in wedd["variants"] and len(wedd["variants"]) == 10
    assert json.loads((ablations / "ablation_index.json").read_text(encoding="utf-8"))["not_run"] == []
    for f in range(1, 6):
        assert (run / "transitions" / "fold_models" / f"transition_fold{f}.json").is_file()
    for name in ("ridge_global_lambda", "rbf_kernel_ridge"):
        assert (run / "transitions" / f"comparator_{name}.parquet").is_file()
    candidates = pd.read_parquet(run / "thresholds" / "wedd_candidates.parquet")
    assert set(candidates["rulebook"]) <= {"d_model", "dataset_label", "measured_oracle"} and candidates["objective"].notna().all()
    assert json.loads((run / "qc" / "panels_MMS.json").read_text(encoding="utf-8"))["partitions"] == ["mms_external"]
    assert (run / "qc" / "manual_review_MMS.parquet").is_file() and (run / "qc" / "manual_review.parquet").is_file()
