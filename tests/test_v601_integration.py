"""Protocol-v6.0.1 integration checks that run on the host (no torch; synthetic fixtures under tmp_path only).

* Owner decision F: Table 1 is checked against ``SPECIFICATION.md`` of the protocol tag a run was executed under.
* Owner decision C: the case-identifier guard of tables and figures covers prefixed M&Ms-2 ids and never trips on counts.
* C3 (L4a with L89): the S6 hook ``tmmri.neural.preprocess.case_tensors`` is the ``build-tensor-cache`` transform.
* C5 (L4b with L89): purpose-s5 spec files import after the F9 freeze; F9 spec files do not.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import fixtures_v6 as fx
from tmmri import manuscript_tables as mt
from tmmri import stages
from tmmri.config import load_study
from tmmri.runs import RunRoot, atomic_write_json


def _run_root(tmp_path, run_id="v6-integration-20261003-1300") -> RunRoot:
    study = load_study()
    return RunRoot(run_id, tmp_path / "artifacts", subfolders=study.study["artifact_subfolders"])


def test_run_protocol_tag_follows_the_protocol_the_run_was_executed_under(tmp_path, monkeypatch):
    study = load_study()
    assert study.protocol_tags == ["protocol-v6.0.0", "protocol-v6.0.1"] and study.current_protocol_tag == "protocol-v6.0.1"
    run = _run_root(tmp_path)
    atomic_write_json(run.frozen_path, {"protocol_tag": "protocol-v6.0.0", "bundle_sha256": "0" * 64})
    atomic_write_json(run.record_path("evaluate:acdc_test"), {"stage": "evaluate:acdc_test"})
    assert mt.run_protocol_tag(run, study) == "protocol-v6.0.0"              # a v6.0.0 run
    atomic_write_json(run.record_path("evaluate:mms2_external"), {"stage": "evaluate:mms2_external"})
    assert mt.run_protocol_tag(run, study) == "protocol-v6.0.1"              # the v6.0.0 bundle executed under v6.0.1
    other = _run_root(tmp_path, "v6-integration-20261003-1301")
    atomic_write_json(other.frozen_path, {"protocol_tag": "protocol-v6.0.0", "protocol_version_tag": "protocol-v6.0.1"})
    assert mt.run_protocol_tag(other, study) == "protocol-v6.0.1"            # frozen with protocol-v6.0.1 code
    seen = []
    monkeypatch.setattr(mt, "spec_matches_tag", lambda path, tag: seen.append(tag) or False)
    with pytest.raises(mt.TableBuildError, match="protocol-v6.0.0"):
        mt.build_tables(_run_root(tmp_path, "v6-integration-20261003-1302"), study, include_mms=False, out_dir=tmp_path / "tables")
    assert seen == ["protocol-v6.0.0"]                                        # no FROZEN.json and no record: the freeze tag


def test_freeze_records_only_a_protocol_tag_that_exists(tmp_path, monkeypatch):
    """FROZEN.json never names an unsigned amendment tag as the protocol the run was executed under."""
    study = load_study()
    run = _run_root(tmp_path, "v6-integration-20261003-1310")
    ctx = stages.StageContext(study=study, run=run, data_root=tmp_path / "data", n_jobs=1)
    for rel in ("transitions/transition_model.json", "formal/d_model_predictor.joblib",
                *[f"rules/{rb}/EMPTY.json" for rb in stages._rulebooks(ctx)]):
        (run.path / rel).parent.mkdir(parents=True, exist_ok=True)
        (run.path / rel).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(stages, "protocol_frozen", lambda tag="protocol-v6.0.0", root=None, **k: tag == "protocol-v6.0.0")
    payload = stages.freeze_analysis_bundle(ctx)                              # protocol-v6.0.1 not signed yet
    assert payload["protocol_version_tag"] == "protocol-v6.0.0" and payload["protocol_version_tag_present_at_freeze"] is False
    assert payload["protocol_version_tag_configured"] == "protocol-v6.0.1" and payload["protocol_tag_present_at_freeze"] is True
    assert mt.run_protocol_tag(run, study) == "protocol-v6.0.0"
    atomic_write_json(run.record_path("evaluate:mms2_external"), {"stage": "evaluate:mms2_external"})
    assert mt.run_protocol_tag(run, study) == "protocol-v6.0.1"              # later executed under the amendment
    signed = _run_root(tmp_path, "v6-integration-20261003-1311")
    ctx_signed = stages.StageContext(study=study, run=signed, data_root=tmp_path / "data", n_jobs=1)
    for rel in ("transitions/transition_model.json", "formal/d_model_predictor.joblib",
                *[f"rules/{rb}/EMPTY.json" for rb in stages._rulebooks(ctx_signed)]):
        (signed.path / rel).parent.mkdir(parents=True, exist_ok=True)
        (signed.path / rel).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(stages, "protocol_frozen", lambda tag="protocol-v6.0.0", root=None, **k: True)
    payload = stages.freeze_analysis_bundle(ctx_signed)
    assert payload["protocol_version_tag"] == "protocol-v6.0.1" and payload["protocol_version_tag_present_at_freeze"] is True
    assert mt.run_protocol_tag(signed, study) == "protocol-v6.0.1"


def test_identifier_guard_covers_prefixed_mms2_ids_and_never_trips_on_counts():
    split = pd.DataFrame({"sample_id": ["ACDC:patient001", "MMS2:MMS2_180"], "case_id": ["patient001", "MMS2_180"],
                          "patient_id": ["patient001", "MMS2_180"]})
    ids = mt.case_identifiers([split, None])
    assert {"patient001", "MMS2_180"} <= ids and "180" not in ids
    mt.assert_no_case_identifiers({"table.csv": "n = 180 clusters; MMS2 180 cases"}, ids)
    for text in ("row MMS2_180 shown", "MMS2_180,0.3", "patient001 listed"):
        with pytest.raises(mt.TableBuildError, match="case identifier"):
            mt.assert_no_case_identifiers({"table.csv": text}, ids)


def test_s6_hook_is_the_build_tensor_cache_transform(tmp_path, monkeypatch):
    from tmmri.ingest.staging import load_staged_case
    from tmmri.neural import preprocess as pp

    monkeypatch.setattr(stages, "protocol_frozen", lambda *a, **k: False)
    acdc = fx.make_acdc_source(tmp_path / "src" / "acdc", fx.acdc_patients(1))
    study = load_study()
    ctx = stages.StageContext(study=study, run=None, data_root=tmp_path / "data", n_jobs=1, sources={"acdc": acdc},
                              source_manifests={"ACDC": fx.acdc_manifest(acdc)})
    assert stages.stage_sources(ctx, "acdc").ok
    cfg = pp.load_neural_input_config()
    for case_id in ("patient001", "patient002"):
        volumes = pp.load_staged_volumes(ctx.staged_root, "ACDC", case_id, cfg=cfg)
        expected = pp.preprocess_case(volumes, cfg)
        case = load_staged_case("ACDC", case_id, ctx.staged_root, label_map=study.label_map("acdc"))
        got = pp.case_tensors(case, cfg)
        for key in pp.CACHE_ARRAYS:
            np.testing.assert_array_equal(got[key], expected.arrays[key])
        assert got["transform_version"] == expected.row["transform_version"] and got["crop_fraction_ed"] == expected.row["crop_fraction_ed"]
    assert pp.cache_file_name("ACDC:patient001") == "ACDC__patient001.npz"


def test_s5_spec_files_import_after_the_f9_freeze(tmp_path, monkeypatch):
    import test_f9_stages as f9t
    from tmmri.neural import f9_stages
    from tmmri.neural.fit_spec import Candidate, FitSpec, read_spec_file, write_spec_file

    monkeypatch.setattr(f9_stages, "resolve_weights", lambda cfg, root=None: dict(f9t.FAKE_WEIGHTS))
    ctx = f9t.make_run(tmp_path)
    run = ctx.run
    f9_stages.plan_f9(ctx)
    atomic_write_json(f9_stages.f9_paths(run).bundle, {"schema": f9_stages.BUNDLE_SCHEMA, "fits": []})   # F9 is frozen
    with pytest.raises(stages.BundleError, match="frozen"):
        f9_stages.import_f9_fits(ctx, tmp_path, wave=1)                       # F9 fitting stages refuse after the freeze
    dev = sorted(pd.read_csv(run.path / "neural_cache" / "acdc_dev" / "manifest.csv")["sample_id"])
    candidate = Candidate.coerce({"unfreeze": "layer4", "lr_backbone": 1e-5})
    specs = [FitSpec(purpose="s5", seed=20261011, outer_fold=1, fitting_patients=tuple(dev[:60]), selection_fold=s, candidate=candidate,
                     refit_epochs=None, fraction=0.75, repeat=0) for s in (1, 2, 3)]
    f9_cfg = f9_stages.load_f9_config(ctx.study.root)
    spec_file = run.path / "data_efficiency" / "f9" / "train_wave1_specs.json"
    write_spec_file(spec_file, specs, run_id=run.run_id, wave=1, config=f9_stages.training_config(f9_cfg),
                    cache_manifest_sha256="0" * 64, weights={k: f9t.FAKE_WEIGHTS[k] for k in ("name", "sha256", "file", "url")}, purpose="s5")
    payload = read_spec_file(spec_file)
    download = tmp_path / "download"
    for spec in specs:
        folder = download / spec.fit_id
        folder.mkdir(parents=True)
        (folder / "record.json").write_text(json.dumps(f9t.fake_record(spec, payload)), encoding="utf-8")
    result = f9_stages.import_f9_fits(ctx, download, wave=1, specs="data_efficiency/f9/train_wave1_specs.json")
    assert result.action == "RAN" and result.stage == "import-f9-fits:train_wave1_specs"
    assert all((f9_stages.f9_paths(run).fit_dir(s.fit_id) / "record.json").is_file() for s in specs)
    with pytest.raises(ValueError, match="wave-1 spec file"):
        f9_stages.import_f9_fits(ctx, download, wave=2, specs="data_efficiency/f9/train_wave1_specs.json")


def test_mms2_freeze_order_labels_bundles_frozen_after_the_extraction_post_hoc(tmp_path):
    """SPECIFICATION.md 14.6 (D33): freeze records must be dated before the M&Ms-2 extraction record, else post hoc."""
    from tmmri.comparators import cart_f8
    from tmmri.experiments import l57_stages as l57
    from tmmri.runs import StageRecord

    study = load_study()
    run = _run_root(tmp_path, "v6-integration-20261003-1320")
    ctx = stages.StageContext(study=study, run=run, data_root=tmp_path / "data", n_jobs=1, sources={"mms2": tmp_path / "src" / "mms2"})

    def record(name, started, seconds=5.0):
        rec = StageRecord(stage=name, inputs={}, outputs={}, config_hash="0", code={}, library_versions={}, n_jobs=1,
                          run_log={"started_utc": started, "seconds": seconds})
        atomic_write_json(run.record_path(name), rec.to_dict())

    for name, started in (("fit-cart-surrogate", "2026-10-04T08:00:00+00:00"), ("freeze-f9", "2026-10-04T08:10:00+00:00"),
                          ("fit-probes", "2026-10-04T08:20:00+00:00")):
        record(name, started)
    order = stages.mms2_freeze_order(ctx, ["F8", "F9b"])            # no extraction record: the order cannot be shown
    assert not order["before_extraction"] and order["label"] == "post hoc" and order["evidence"] == "post_hoc"
    assert not cart_f8._mms2_staged(ctx)
    atomic_write_json(ctx.manifests_dir / stages.MMS2_EXTRACTION_MANIFEST,
                      {"action": "RAN", "extraction_started_utc": "2026-10-04T09:00:00+00:00", "first_extraction_started_utc": "2026-10-04T09:00:00+00:00"})
    assert cart_f8._mms2_staged(ctx)                                 # F8 refuses to freeze once M&Ms-2 is extracted
    order = stages.mms2_freeze_order(ctx, ["F8", "F9", "F9b"])
    assert order["before_extraction"] and order["label"] == "prospective"
    assert order["extraction"]["first_extraction_started_utc"] == "2026-10-04T09:00:00+00:00"
    record("fit-trust", "2026-10-04T09:30:00+00:00")                 # S1-S4 fitted after the extraction
    late = stages.mms2_freeze_order(ctx, ["F9", "S1-S4"])
    assert late["label"] == "post hoc" and late["reason"] == "FROZEN_AFTER_MMS2_EXTRACTION"
    assert late["bundles"]["F9"]["before_extraction"] and not late["bundles"]["S1-S4"]["before_extraction"]
    record("fit-deep-encoder", "2026-10-04T08:59:59+00:00", seconds=0.2)   # may have finished after the extraction started
    assert not stages.mms2_freeze_order(ctx, ["D20"])["before_extraction"]
    assert not stages.mms2_freeze_order(ctx, ["D20"])["bundles"]["D20"]["freezes"][0]["before_extraction"]
    info, order = l57.partition_evidence(ctx, "mms2_external", ["S1-S4"])
    assert info["evidence"] == "post_hoc" and info["label"] == "post hoc" and order["label"] == "post hoc"
    info, order = l57.partition_evidence(ctx, "mms2_external", ["F9b"])
    assert info["evidence"] == "prospective" and order["before_extraction"]
    info, order = l57.partition_evidence(ctx, "acdc_test", ["S1-S4"])
    assert order is None and info["label"] == "registered after unblinding"


def test_table4_rows_carry_the_post_hoc_label_of_a_late_frozen_bundle(tmp_path):
    from tmmri.experiments import l57_stages as l57
    from tmmri.experiments.comparison_tables import build_comparison

    study = load_study()
    run = _run_root(tmp_path, "v6-integration-20261003-1330")
    pcfg = l57.lane_config(study, "probes")
    docs = {"d20": "post hoc", "f9b": "prospective"}
    for bundle, label in docs.items():
        atomic_write_json(run.path / "probes" / "evaluations" / bundle / "mms2_external.json",
                          {"bundle": bundle, "partition": "mms2_external", "label": label, "representations": {}, "predictors": {}})
    sources = [run.path / "probes" / "evaluations" / b / f"{p}.json" for b in pcfg["bundles"] for p in pcfg["table4"]["partitions"]]
    build_comparison(run, pcfg, [s for s in sources if s.is_file()], tmp_path / "out", min_clusters=20)
    table = pd.read_csv(tmp_path / "out" / "table04_representation_comparison.csv", dtype=str, keep_default_na=False)
    rows = table[table["panel"].str.startswith("mms2_external")].set_index("representation")["evidence"]
    assert rows.str.contains("post hoc").sum() == 1 and "post hoc" in rows.iloc[[i for i, r in enumerate(rows.index) if "20" in r][0]]
    assert (table.loc[table["panel"].str.startswith("acdc_test"), "evidence"] == "registered after unblinding").all()
