"""Post-freeze exploratory block B10-B20 (protocol-v6.0.2) on a synthetic frozen v6 run (algorithm controls only; never
cohort results): the ``run-postfreeze`` infrastructure (lock, SHA-256 refusal, no write below the source run, the
EXPLORATORY_POST_HOC label, the repeat policy, B21/B22 absent from the build) and every analysis through the CLI."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import shutil
from types import SimpleNamespace

import pandas as pd
import pytest

import fixtures_v6 as fx
from test_v6_cli import METHODS_MODULES, _edit_yaml, _run, _study_copy
from tmmri import cli, stages
from tmmri.experiments.postfreeze import b20_worked_example as b20
from tmmri.experiments.postfreeze import common as pf
from tmmri.experiments.postfreeze.b15_rule_ledger import PASSED

SOURCE = "v6-pfsource-20261003-1241"
RUN = "v6-postfreeze-20261028-1300"


def _main(argv: list[str]) -> int:
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            cli.main(argv)
    except SystemExit as exc:
        if exc.code:
            raise AssertionError(f"{argv[0]} failed ({exc.code}): {out.getvalue()[-2000:]}") from None
    return 0


def _fast(d: dict) -> None:
    d["statistics"]["bootstrap"]["replicates"] = 40


def _tree(root) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def lane(tmp_path_factory):
    """A frozen synthetic source run through ``run-ablations``, F8, the trust bundle and the release manifest."""

    from conftest import require_module

    for module in METHODS_MODULES:
        require_module(module)
    mp = pytest.MonkeyPatch()
    mp.setattr(stages, "protocol_frozen", lambda *a, **k: False)
    mp.delattr(stages, "require_mms2_unlocked", raising=False)   # the M&Ms-2 lock is lane L1's; without it M&Ms-2 is not executed
    tmp = tmp_path_factory.mktemp("postfreeze")
    acdc = fx.make_acdc_source(tmp / "src" / "acdc", fx.acdc_patients(5, testing_per_class=2))
    fx.write_fixture_manifests(acdc, fx.acdc_manifest(acdc))
    cases = fx.mms_refit_cases()
    mms = fx.make_mms_source(tmp / "src" / "mms", cases, shape=(26, 24, 3))
    fx.write_fixture_manifests(mms, fx.mms_manifest(mms, cases), fx.mms_metadata(mms))
    study = _study_copy(tmp)
    _edit_yaml(study.parent / "exploratory_postfreeze.yaml", _fast)
    data_args = ["--artifact-root", str(tmp / "artifacts"), "--data-root", str(tmp / "data"), "--acdc-source", str(acdc), "--mms-source", str(mms),
                 "--mms2-source", str(tmp / "src" / "mms2"), "--n-jobs", "1"]
    common = ["--study", str(study), "--run-id", SOURCE, *data_args, "--allow-unfrozen-synthetic"]
    _main(["run", *common, "--until", "run-ablations"])
    _main(["fit-cart-surrogate", *common])
    for split in ("acdc_test", "mms_external"):
        _main(["evaluate-cart-surrogate", *common, "--split", split])
    for partition in ("acdc_dev_oof", "acdc_test", "mms_external"):
        _main(["summarize-predictor", *common, "--partition", partition])
    _main(["fit-trust", *common, "--allow-missing"])
    for split in ("acdc_test", "mms_external"):
        _main(["evaluate-trust", *common, "--split", split, "--allow-missing"])
    _main(["write-release-manifest", *common])
    source = tmp / "artifacts" / SOURCE
    yield SimpleNamespace(tmp=tmp, study=study, source=source, artifacts=tmp / "artifacts", data_args=data_args,
                          args=["--study", str(study), "--source-run", SOURCE, *data_args])
    mp.undo()


def _postfreeze(lane, analysis: str, run_id: str, capsys, *extra: str, synthetic: bool = True):
    argv = ["run-postfreeze", "--analysis", analysis, *lane.args, "--run-id", run_id, *extra]
    return _run([*argv, "--allow-unfrozen-synthetic"] if synthetic else argv, capsys)


def test_run_postfreeze_infrastructure(lane, capsys, monkeypatch):
    """Run-id template, lock, B21/B22 not built, label, no write below the source run, repeat refusal and repeat report."""

    code, out = _postfreeze(lane, "B19", "v6-other-20261028-1300", capsys)
    assert code == cli.EXIT_USAGE_OR_INPUT and "must match" in out
    code, out = _postfreeze(lane, "B19", RUN, capsys, synthetic=False)
    assert code == cli.EXIT_PROTOCOL_NOT_FROZEN and "source run is fixed" in out                  # lane source_run, then the signed tag
    _edit_yaml(lane.study.parent / "exploratory_postfreeze.yaml", lambda d: d["source_run"].update(run_id=SOURCE))
    try:
        code, out = _postfreeze(lane, "B19", RUN, capsys, synthetic=False)
        assert code == cli.EXIT_PROTOCOL_NOT_FROZEN and "protocol-v6.0.2" in out
    finally:
        _edit_yaml(lane.study.parent / "exploratory_postfreeze.yaml", lambda d: d["source_run"].update(run_id="v6-replication-20261003-1241"))
    monkeypatch.setitem(pf.EXTERNAL_MODULES, "B21", "tmmri.experiments.postfreeze._not_built_b21")
    code, out = _postfreeze(lane, "B21", RUN, capsys)
    assert code == cli.EXIT_METHODS_API_MISSING and "not part of this build" in out
    before = _tree(lane.source)
    code, res = _postfreeze(lane, "B19", RUN, capsys)
    assert code == 0 and res["label"] == pf.LABEL and res["protocol_tag_state"] == "synthetic_override" and res["repeat"] is False, res
    assert _tree(lane.source) == before                                                          # nothing written below the source run
    run = lane.artifacts / RUN
    prov = json.loads((run / "postfreeze" / "B19" / "provenance.json").read_text(encoding="utf-8"))
    assert prov["label"] == pf.LABEL and prov["confirmatory"] is False and prov["source_run"]["run_id"] == SOURCE
    assert prov["inputs"] and all(v["sha256"] for v in prov["inputs"].values()) and all(k.startswith("run:postfreeze/B19/") for k in prov["outputs"])
    code, out = _postfreeze(lane, "B19", RUN, capsys)
    assert code == cli.EXIT_BUNDLE and "a repeat needs a new run id" in out
    code, res = _postfreeze(lane, "B19", "v6-postfreeze-20261028-1301", capsys)
    assert code == 0 and res["repeat"] is True
    again = json.loads((lane.artifacts / "v6-postfreeze-20261028-1301" / "postfreeze" / "B19" / "provenance.json").read_text(encoding="utf-8"))
    assert again["repeat_of"] == [RUN]


def test_changed_source_file_and_non_synthetic_override_refuse(lane, capsys):
    tampered = lane.tmp / "tampered"
    shutil.copytree(lane.source, tampered / SOURCE)
    args = [a if a != str(lane.artifacts) else str(tampered) for a in lane.args]
    path = tampered / SOURCE / "metrics" / "predictor_acdc_test.json"
    path.write_bytes(path.read_bytes() + b" ")
    code, out = _run(["run-postfreeze", "--analysis", "B19", *args, "--run-id", RUN, "--allow-unfrozen-synthetic"], capsys)
    assert code == cli.EXIT_BUNDLE and "differs from the SHA-256" in out
    manifest = tampered / SOURCE / "release_manifest.json"
    manifest.write_text(json.dumps({**json.loads(manifest.read_text(encoding="utf-8")), "synthetic_fixture": False}), encoding="utf-8")
    code, out = _run(["run-postfreeze", "--analysis", "B19", *args, "--run-id", "v6-postfreeze-20261028-1302", "--allow-unfrozen-synthetic"], capsys)
    assert code == cli.EXIT_PROTOCOL_NOT_FROZEN and "synthetic" in out


def _read(lane, analysis: str, name: str):
    path = lane.artifacts / "v6-postfreeze-20261028-1400" / "postfreeze" / analysis / name
    return pd.read_parquet(path) if name.endswith(".parquet") else json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("analysis", [f"B{k}" for k in range(10, 21)])
def test_every_analysis_on_the_synthetic_run(lane, capsys, monkeypatch, analysis):
    if analysis == "B20":
        split = pd.read_parquet(lane.source / "contracts" / "split_manifest.parquet")
        test_ids = sorted(split.loc[split["partition"] == "acdc_test", "sample_id"].astype(str))[:5]
        monkeypatch.setattr(b20, "slot_cases", lambda repo: dict(zip(b20.SLOTS, test_ids)))
    before = _tree(lane.source)
    code, res = _postfreeze(lane, analysis, "v6-postfreeze-20261028-1400", capsys)
    assert code == 0, res
    assert _tree(lane.source) == before
    prov = _read(lane, analysis, "provenance.json")
    assert prov["label"] == pf.LABEL and prov["analysis"] == analysis and prov["outputs"]
    summary = _read(lane, analysis, "summary.json")
    if analysis not in ("B15", "B20"):                                                       # B15: acdc_dev only; B20: acdc_test only
        assert "mms2_external" in (summary.get("not_executed") or {}), summary.get("not_executed")
    check = CHECKS[analysis]
    check(lane, summary)


def _b10(lane, summary):
    exact = _read(lane, "B10", "exactness_counts.parquet")
    assert set(exact["target"]) >= {"lv_ef_percent", "rv_ef_percent"}
    rho = _read(lane, "B10", "rho.parquet")
    assert {"analytical", "primary_ridge"} <= set(rho["arm"]) and set(rho["partition"]) >= {"acdc_dev_oof", "acdc_test", "mms_external"}


def _b11(lane, summary):
    run = lane.artifacts / "v6-postfreeze-20261028-1400" / "postfreeze" / "B11"
    for estimator in ("ridge_typed_fractions17", "ridge_typed_intensities40"):
        frozen = json.loads((run / "models" / estimator / "FROZEN.json").read_text(encoding="utf-8"))
        assert frozen["fit_partition"] == "acdc_dev" and frozen["n_columns"] in (17, 40)
    rho = _read(lane, "B11", "rho.parquet")
    assert {"ridge_typed_fractions17", "ridge_typed_intensities40", "primary_ridge"} <= set(rho["arm"])


def _b12(lane, summary):
    table = _read(lane, "B12", "decomposition_table.parquet")
    assert len(table) and set(table["component"]) >= {"approximation", "missing_information", "shift"}


def _b13(lane, summary):
    side = _read(lane, "B13", "side_by_side_with_mean_null.parquet")
    recorded = json.loads((lane.source / "metrics" / "primary_rho_acdc_test.json").read_text(encoding="utf-8"))
    row = side[(side["partition"] == "acdc_test") & (side["quantity"] == "rho")].iloc[0]
    assert row["recorded_mean_null"] == pytest.approx(recorded["rho"]) and pd.notna(row["median_null"])


def _b14(lane, summary):
    table = _read(lane, "B14", "translation_table.parquet")
    assert (table["threshold_physical_units"] - 0.10 * table["mae_null"]).abs().max() < 1e-12
    sweep = _read(lane, "B14", "sweep_table.parquet")
    assert set(sweep["threshold"]) == {0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30} or len(sweep) == 0


def _b15(lane, summary):
    ledger = _read(lane, "B15", "candidate_ledger.parquet")
    import yaml

    order = yaml.safe_load((lane.study.parent / "exploratory_postfreeze.yaml").read_text(encoding="utf-8"))["analyses"]["B15"]["pipeline_order"]
    assert len(ledger) and set(ledger["first_failed_condition"]) <= {*order, PASSED}
    counts = _read(lane, "B15", "rejection_counts_by_reason.parquet")
    assert int(counts["n_candidates"].sum()) == len(ledger)


def _b16(lane, summary):
    table = _read(lane, "B16", "coverage_fidelity_table.parquet")
    evaluated = table[table["status"] == "EVALUATED"]
    assert {"primary_d_model", "companion_dataset_label", "f8_cart_full", "f5_gate_bounds_0.50_0.70"} <= set(evaluated["system"])
    assert evaluated["coverage"].dropna().between(0, 1).all()
    assert all(c["recomputed_routes_equal_recorded"] for c in summary["reproduction_checks"]), summary["reproduction_checks"]


def _b17(lane, summary):
    auroc = _read(lane, "B17", "signal_auroc_table.parquet")
    assert set(auroc["signal"]) == {"uncertainty", "ood_score", "unstable_bins"} and set(auroc["partition"]) == {"acdc_test", "mms_external"}
    symbols = _read(lane, "B17", "symbol_error_table.parquet")
    assert "pooled" in set(symbols["concept"])


def _b18(lane, summary):
    table = _read(lane, "B18", "decision_link_table.parquet")
    assert set(table["input"]) == {"bhat7", "bhat9", "measured7"} and set(table["partition"]) == {"acdc_dev_oof", "acdc_test", "mms_external"}
    assert table["fidelity"].dropna().between(0, 1).all()
    for key, frozen in summary["frozen_models"].items():
        path = lane.artifacts / "v6-postfreeze-20261028-1400" / frozen["file"].split(":", 1)[1]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == frozen["sha256"], key


def _b19(lane, summary):
    table = _read(lane, "B19", "majority_class_table.parquet").set_index("partition")
    doc = json.loads((lane.source / "metrics" / "predictor_acdc_test.json").read_text(encoding="utf-8"))
    assert table.loc["acdc_test", "n_evaluable"] == doc["counts"]["n_evaluable"] and 0 < table.loc["acdc_test", "majority_rate"] <= 1


def _b20(lane, summary):
    run = lane.artifacts / "v6-postfreeze-20261028-1400" / "postfreeze" / "B20"
    table = pd.read_parquet(run / "case_slots.parquet")
    slots = json.loads((run / "restricted" / "slot_map.json").read_text(encoding="utf-8"))["slots"]
    assert list(table["slot"]) == ["a", "b1", "b2", "b3", "b4"] and "sample_id" not in table.columns
    assert {"d_model_decision", "reference_label", "route_d_model__route", "f8__label_full", "ood_score"} <= set(table.columns)
    public = (run / "case_slots.parquet").read_bytes() + (run / "summary.json").read_bytes() + (run / "provenance.json").read_bytes()
    assert not any(s.encode() in public or s.split(":", 1)[1].encode() in public for s in slots.values())


CHECKS = {"B10": _b10, "B11": _b11, "B12": _b12, "B13": _b13, "B14": _b14, "B15": _b15, "B16": _b16, "B17": _b17, "B18": _b18, "B19": _b19,
          "B20": _b20}
