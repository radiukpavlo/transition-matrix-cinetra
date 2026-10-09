"""Protocol-v6.0.1 M&Ms-2 stages through the CLI on a frozen synthetic v6 run (algorithm control; never cohort data).

``stage-sources``, ``validate-cohort``, ``build-semantics``, ``extract-formal``, ``evaluate`` (once-only),
``audit-shift`` and ``summarize-predictor`` accept M&Ms-2 and keep the frozen v6.0.0 bundle unchanged; a staging
rejection (inconsistent geometry) and a duplicate of an ACDC case leave ``mms2_external`` with their reason codes.
"""
from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pandas as pd

import fixtures_mms2 as f2
import fixtures_v6 as fx
import test_v6_cli as v6cli
from tmmri import cli, stages
from tmmri.ingest import mms2

PATHOLOGIES = ["NOR", "HCM", "LV", "ARR", "NOR", "HCM", "DRV", "NOR", "HCM", "LV", "TRI", "NOR"]


def _mms2_source(root, acdc_root):
    sids = mms2.external_ids()[:12]
    cases = {}
    for sid in sids[:10]:
        cases[sid] = f2.write_case(root, sid, f2.case_arrays(sid), geometry="unit" if int(sid) % 2 else "voxel")
    cases[sids[10]] = f2.write_case(root, sids[10], f2.case_arrays(sids[10]), geometry="inconsistent")
    frames = {}                                   # an exact copy of the ACDC patient001 ED/ES frame files
    for phase, frame in (("ed", 1), ("es", 3)):
        for kind, suffix in (("image", ""), ("mask", "_gt")):
            frames[f"{phase}_{kind}"] = np.asanyarray(nib.load(str(acdc_root / "training" / "patient001" / f"patient001_frame{frame:02d}{suffix}.nii.gz")).dataobj)
    cases[sids[11]] = f2.write_case(root, sids[11], {k: v.astype(np.float32) if k.endswith("image") else v for k, v in frames.items()}, geometry="unit")
    f2.finish_root(root, cases, [f2.info_row(s, i, PATHOLOGIES[i]) for i, s in enumerate(sids)])
    return root, sids


def test_mms2_stages_on_a_frozen_synthetic_run(tmp_path, monkeypatch, capsys):
    from conftest import require_module

    for module in v6cli.METHODS_MODULES:
        require_module(module)
    monkeypatch.setattr(stages, "protocol_frozen", lambda *a, **k: False)
    acdc = fx.make_acdc_source(tmp_path / "src" / "acdc", fx.acdc_patients(5, testing_per_class=2))
    fx.write_fixture_manifests(acdc, fx.acdc_manifest(acdc))
    mms = fx.make_mms_source(tmp_path / "src" / "mms", shape=(26, 24, 3))
    fx.write_fixture_manifests(mms, fx.mms_manifest(mms), fx.mms_metadata(mms))
    mms2_root, sids = _mms2_source(tmp_path / "src" / "mms2", acdc)
    study_path = v6cli._study_copy(tmp_path)
    run_id = "v6-synthetic-20261003-1000"
    common = ["--study", str(study_path), "--run-id", run_id, "--artifact-root", str(tmp_path / "artifacts"), "--data-root", str(tmp_path / "data"),
              "--acdc-source", str(acdc), "--mms-source", str(mms), "--mms2-source", str(mms2_root), "--n-jobs", "1", "--allow-unfrozen-synthetic"]
    code, out = v6cli._run(["run", *common, "--until", "freeze-analysis-bundle"], capsys)
    assert code == 0, out
    run = tmp_path / "artifacts" / run_id
    frozen_bytes = (run / ".stages" / "FROZEN.json").read_bytes()
    # Staging: the inconsistent-geometry case is rejected with its reason code (exit 2); the others are staged.
    code, out = v6cli._run(["stage-sources", "--dataset", "mms2", *common], capsys)
    assert code == cli.EXIT_STAGING_FAILURES and mms2.GEOMETRY_REASON in out, out
    code, report = v6cli._run(["validate-cohort", "--dataset", "mms2", "--strict", *common], capsys)
    assert code == 0 and report["passed"], report
    cohort = pd.read_parquet(run / "contracts" / "cohort_MMS2.parquet").set_index("case_id")
    keys = [mms2.case_key(s) for s in sids]                       # owner decision C: case ids MMS2_<code>
    assert cohort.at[keys[10], "exclusion_reason"] == mms2.GEOMETRY_REASON
    assert cohort.at[keys[11], "exclusion_reason"] == "DUPLICATE_IMAGE_OF_ACDC_CASE" and cohort.at[keys[11], "duplicate_of"] == "ACDC:patient001"
    assert sorted(cohort.index[cohort["mms2_external"]]) == sorted(keys[:10])
    assert not any(c.isdigit() for c in cohort.index) and set(cohort["sample_id"]) == {f"MMS2:{k}" for k in keys}
    for argv in (["build-semantics", "--dataset", "mms2"], ["extract-formal", "--dataset", "mms2"],
                 ["evaluate", "--split", "mms2_external", "--no-fit"], ["audit-shift", "--dataset", "mms2"],
                 ["summarize-predictor", "--split", "mms2_external"]):
        code, result = v6cli._run([*argv, *common], capsys)
        assert code == 0 and result["action"] == "RAN", (argv, result)
    b = pd.read_parquet(run / "semantics" / "B_MMS2.parquet")
    assert sorted(b["case_id"]) == sorted(keys[:10]) and set(b["dataset_source"]) == {"MMS2"}
    assert (run / "formal" / "A_MMS2.parquet").is_file() and (run / "qc" / "manual_review_MMS2.parquet").is_file()
    h3 = json.loads((run / "metrics" / "h_primary_3.json").read_text(encoding="utf-8"))
    assert h3["hypothesis"]["id"] == "H-primary-3" and h3["result"]["sesoi"] == 0.10 and h3["family"] == "own_one_test_family"
    assert h3["result"]["status"] in {"PASSED", "FAILED", "NE"} and h3["n_cases"] == 10
    rho = json.loads((run / "metrics" / "primary_rho_mms2_external.json").read_text(encoding="utf-8"))
    assert rho["hypothesis_id"] == "H-primary-3" and rho["partition"] == "mms2_external"
    strata = json.loads((run / "metrics" / "strata_rho_mms2_external.json").read_text(encoding="utf-8"))
    assert set(strata["strata"]["pathology_group"]) == {"NOR", "HCM", "DLV", "RV_FAMILY"}
    tests = pd.read_parquet(run / "metrics" / "multiplicity_MMS2.parquet")
    assert set(tests["family"]) == {"F_MMS2_external_reconstruction"}
    predictor = json.loads((run / "metrics" / "predictor_mms2_external.json").read_text(encoding="utf-8"))
    assert predictor["counts"]["n_same_definition"] == 6 and set(predictor["non_equivalent_strata"]) == {"DLV", "RV_FAMILY"}   # NOR 3 + HCM 3
    dlv, rv = predictor["non_equivalent_strata"]["DLV"], predictor["non_equivalent_strata"]["RV_FAMILY"]
    assert rv["mapped_acdc_class"] == ["RV"] and rv["pathologies"] == ["ARR", "DRV"] and rv["pooled"] is False
    assert dlv["mapped_acdc_class"] == ["DCM"] and dlv["n"] + dlv["n_excluded_critical_qc"] == 2
    shift = json.loads((run / "metrics" / "shift_audit_mms2_external.json").read_text(encoding="utf-8"))
    assert shift["comparison_partition"] == "mms2_external" and shift["counts"]["comparison"]["n_partition"] == 10
    assert not (run / "metrics" / "shift_audit_mms_external.json").exists() and not (run / "metrics" / "primary_rho_mms_external.json").exists()
    # Decision D36: the H4 family, confirmatory on mms2_external, registered after unblinding on acdc_test, from recorded artifacts.
    code, out = v6cli._run(["evaluate", "--split", "acdc_test", "--no-fit", *common], capsys)
    assert code == 0, out
    for split in ("mms2_external", "acdc_test"):
        code, result = v6cli._run(["evaluate-h4", "--split", split, *common], capsys)
        assert code == 0 and result["action"] == "RAN" and result["status"] == "EVALUATED", (split, result)
    h4 = json.loads((run / "metrics" / "confirmatory_h4" / "h4_mms2_external.json").read_text(encoding="utf-8"))
    assert h4["confirmatory"] is True and h4["evidence"].startswith("confirmatory; prospective") and h4["n_cases"] == 10
    assert [r["hypothesis_id"] for r in h4["results"]] == ["H4a", "H4b"] and all(r["status"] in {"PASSED", "FAILED", "NE"} for r in h4["results"])
    assert h4["bootstrap"]["sub_seeds"] == {"H4a": 1958159060, "H4b": 1591327988} and h4["bootstrap"]["level"] == 0.975
    assert h4["h4a"]["family"] == ["lv_ef_percent", "rv_ef_percent"] and h4["bundle_sha256"] == json.loads(frozen_bytes)["bundle_sha256"]
    assert h4["h4b"]["harmonization"]["variant"] == "H-a"
    desc = json.loads((run / "metrics" / "confirmatory_h4" / "h4_acdc_test.json").read_text(encoding="utf-8"))
    assert desc["confirmatory"] is False and desc["analysis_label"] == "registered after unblinding"
    assert all(r["status"] in {"DESCRIPTIVE", "NE"} and r["tested"] is False for r in desc["results"])
    code, out = v6cli._run(["evaluate-h4", "--split", "mms_external", *common], capsys)
    assert code == cli.EXIT_BUNDLE and "evaluate:mms_external" in out, out           # never before its evaluation is recorded
    code, again = v6cli._run(["evaluate-h4", "--split", "acdc_test", *common], capsys)
    assert code == 0 and again["action"] == "SKIPPED", again
    (run / ".stages" / "evaluate-h4__acdc_test.json").unlink()                       # a record without its stage record is never overwritten
    code, out = v6cli._run(["evaluate-h4", "--split", "acdc_test", *common], capsys)
    assert code == cli.EXIT_BUNDLE and "never overwritten" in out, out
    # The frozen v6.0.0 bundle is untouched; the evaluation runs once (a changed M&Ms-2 endpoint is refused, exit 7).
    assert (run / ".stages" / "FROZEN.json").read_bytes() == frozen_bytes
    code, again = v6cli._run(["evaluate", "--split", "mms2_external", "--no-fit", *common], capsys)
    assert code == 0 and again["action"] == "SKIPPED", again
    v6cli._edit_yaml(study_path.parent / "cohorts.yaml", lambda d: d["mms2"]["endpoints"]["h_primary_3"].update(threshold=0.2))
    code, out = v6cli._run(["evaluate", "--split", "mms2_external", "--no-fit", *common], capsys)
    assert code == cli.EXIT_BUNDLE and "once-only guard" in out, out
    code, out = v6cli._run(["evaluate-h4", "--split", "mms2_external", *common], capsys)
    assert code == cli.EXIT_BUNDLE and "once-only guard" in out, out                 # the H4 decision is recorded once
    code, out = v6cli._run(["evaluate", "--split", "acdc_test", "--no-fit", *common], capsys)
    assert code == 0, out                         # the M&Ms-2 section never enters the v6.0.0 records or bundle hash
