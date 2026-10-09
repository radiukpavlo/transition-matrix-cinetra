"""F9 stages on a synthetic run root: plan-f9 (324), select-f9 (18), import-f9-fits, freeze-f9, the CPE-1 upload guard
and ledger, the CLI surface and the Modal script's static contract.  Synthetic identifiers and tensors only; no torch,
no network, nothing outside pytest's tmp_path."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tmmri import cli
from tmmri.neural import f9_stages
from tmmri.neural.cloud import (
    CloudPolicyError,
    backend_block,
    cpe1_upload_set,
    parse_block_mapping,
    read_backend_config,
)
from tmmri.neural.data import MANIFEST_COLUMNS
from tmmri.neural.f9_config import load_f9_config
from tmmri.neural.fit_spec import read_spec_file
from tmmri.splits import hash_rank_folds
from tmmri.stages import BundleError, make_context

REPO = Path(__file__).resolve().parents[1]
RUN_ID = "v6-f9test-20261003-0900"
PRIMARY = 20261011
CLASS_SIZES = {"DCM": 20, "HCM": 20, "MINF": 19, "NOR": 19, "RV": 20}
FAKE_WEIGHTS = {"name": "resnet18_imagenet1k_v1", "sha256": "a" * 64, "file": "resnet18-f37072fd.pth", "url": None, "path": "/nonexistent"}


def _npz(path: Path, seed: int) -> str:
    rng = np.random.default_rng(seed)
    np.savez(path, ed=rng.random((2, 8, 8), dtype=np.float32), es=rng.random((2, 8, 8), dtype=np.float32),
             ed_geom=np.zeros((2, 2), np.float32), es_geom=np.zeros((2, 2), np.float32))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_run(tmp_path: Path, *, n_npz: int | None = None):
    ctx = make_context("configs/study_v6.yaml", RUN_ID, artifact_root=tmp_path / "artifacts")
    run = ctx.run
    rows, k = [], 1
    for cls, n in CLASS_SIZES.items():
        for _ in range(n):
            rows.append({"sample_id": f"synthetic{k:03d}", "label": cls})
            k += 1
    dev = pd.DataFrame(rows)
    dev["outer_fold"] = hash_rank_folds(dev, "sample_id", "label", "acdc-fold-v6|", 5).astype("Int64")
    test = pd.DataFrame({"sample_id": ["heldout001", "heldout002"], "label": ["DCM", "RV"], "outer_fold": pd.array([pd.NA, pd.NA], dtype="Int64")})
    split = pd.concat([dev.assign(partition="acdc_dev"), test.assign(partition="acdc_test")], ignore_index=True)
    split["cluster_id"] = split["sample_id"]
    (run.path / "contracts").mkdir(parents=True, exist_ok=True)
    split[["sample_id", "partition", "outer_fold", "cluster_id"]].to_parquet(run.path / "contracts" / "split_manifest.parquet", index=False)
    cache = run.path / "neural_cache" / "acdc_dev"
    cache.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, row in enumerate(dev.itertuples()):
        sha = _npz(cache / f"{row.sample_id}.npz", i) if n_npz is None or i < n_npz else "0" * 64
        manifest.append({"sample_id": row.sample_id, "cluster_id": row.sample_id, "partition": "acdc_dev", "label": row.label, "n_ed": 2,
                         "n_es": 2, "npz_sha256": sha, "crop_fraction_ed": 0.0, "crop_fraction_es": 0.0, "inplane_spacing_mm": 1.5,
                         "transform_version": "synthetic-v1"})
    pd.DataFrame(manifest, columns=list(MANIFEST_COLUMNS)).to_csv(cache / "manifest.csv", index=False, lineterminator="\n")
    return ctx


def fake_record(spec, payload, *, epoch: int = 25) -> dict:
    h = int(hashlib.sha256(spec.fit_id.encode()).hexdigest()[:8], 16)
    return {"fit_id": spec.fit_id, "spec": spec.to_dict(), "status": "COMPLETE", "config_sha256": payload["config_sha256"],
            "selected_epoch": epoch + h % 7, "val_metrics": {"macro_f1": (h % 101) / 101.0, "cross_entropy": 1.0 + (h % 13) / 13.0},
            "checkpoint_sha256": None, "backend": {"name": "synthetic"}, "environment": {"gpu_name": None},
            "epochs": [{"epoch": e, "train_loss": 1.6 - 0.02 * e} for e in range(1, 4)],         # the training check of tmmri.neural.train
            "parameters": {"unfrozen_conv_max_abs_change": {"backbone.layer4.1.conv2.weight": 1e-3}, "bn_running_statistics_unchanged": True,
                           "frozen_weights_unchanged": True}}


def write_oof(ctx, wave2: dict, *, correct_per_10: int = 8, seed: int = PRIMARY) -> Path:
    """Synthetic ``infer-f9 --split acdc_dev`` output (contract C4 is_oof rows of the fold encoders) and its stage record;
    patient i is predicted correctly when ``i % 10 < correct_per_10``, else as the next class."""

    from tmmri.neural.prediction_table import prediction_frame
    from tmmri.runs import run_stage

    run = ctx.run
    manifest = pd.read_csv(run.path / "neural_cache" / "acdc_dev" / "manifest.csv")
    split = pd.read_parquet(run.path / "contracts" / "split_manifest.parquet").set_index("sample_id")
    classes = list(CLASS_SIZES)
    ids, labels = manifest["sample_id"].tolist(), manifest["label"].tolist()
    proba = np.zeros((len(ids), len(classes)))
    for i, label in enumerate(labels):
        k = classes.index(label)
        proba[i, k if i % 10 < correct_per_10 else (k + 1) % len(classes)] = 1.0
    frame = prediction_frame(sample_id=ids, probabilities=proba, model_id="f9", partition="acdc_dev", seed=seed, cluster_id=ids, y_true=labels,
                             encoder_id=[f"fold{int(split.at[s, 'outer_fold'])}" for s in ids], is_oof=True)
    path = run.path / "neural_f9" / "predictions_acdc_dev.parquet"
    encoders = sorted(f"{s.seed}:{'final' if s.outer_fold == 'final' else f'fold{s.outer_fold}'}:{s.fit_id}" for s in wave2["fit_specs"])
    record = run.record_path("infer-f9:acdc_dev")
    if record.is_file():
        record.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    run_stage(run, "infer-f9:acdc_dev", lambda: frame.to_parquet(path, index=False), [], [path], config_hash="synthetic",
              params={"partition": "acdc_dev", "encoders": encoders})
    return path


@pytest.fixture()
def planned(tmp_path, monkeypatch):
    monkeypatch.setattr(f9_stages, "resolve_weights", lambda cfg, root=None: dict(FAKE_WEIGHTS))
    ctx = make_run(tmp_path)
    result = f9_stages.plan_f9(ctx)
    return ctx, result


def test_plan_f9_writes_324_specs_and_reads_only_acdc_dev(planned):
    ctx, result = planned
    paths = f9_stages.f9_paths(ctx.run)
    payload = read_spec_file(paths.wave_specs(1))
    assert result.value["wave1_specs"] == 324 and payload["n_specs"] == 324 and payload["wave"] == 1
    assert payload["fitting_sets"]["final"] == sorted(payload["fitting_sets"]["final"]) and len(payload["fitting_sets"]["final"]) == 98
    assert all("heldout" not in sid for s in payload["fit_specs"] for sid in s.fitting_patients)
    assert {s.seed for s in payload["fit_specs"]} == {20261011, 20261012, 20261013}
    assert payload["weights"]["sha256"] == "a" * 64
    folds = pd.read_parquet(paths.plan / "selection_folds.parquet")
    assert set(folds["fitting_set"]) == {"1", "2", "3", "4", "5", "final"} and len(folds) == 4 * 78 + 80 + 98
    again = f9_stages.plan_f9(ctx)
    assert again.action == "SKIPPED"


def _select_and_import(ctx, tmp_path):
    """Wave-1 records, select-f9 and the verified import of the 18 refits (with the import guards)."""

    paths = f9_stages.f9_paths(ctx.run)
    wave1 = read_spec_file(paths.wave_specs(1))
    with pytest.raises(FileNotFoundError, match="324 of 324"):
        f9_stages.select_f9(ctx)
    for spec in wave1["fit_specs"]:
        (paths.fit_dir(spec.fit_id)).mkdir(parents=True, exist_ok=True)
        (paths.fit_dir(spec.fit_id) / "record.json").write_text(json.dumps(fake_record(spec, wave1)), encoding="utf-8")
    result = f9_stages.select_f9(ctx)
    wave2 = read_spec_file(paths.wave_specs(2))
    assert result.value["refits"] == 18 and wave2["n_specs"] == 18 and wave2["wave"] == 2
    assert all(s.refit_epochs >= 25 for s in wave2["fit_specs"])
    table = pd.read_parquet(paths.selection / "selection_table.parquet")
    assert len(table) == 108 and int(table["selected"].sum()) == 18

    download = tmp_path / "download" / RUN_ID
    for spec in wave2["fit_specs"]:
        folder = download / spec.fit_id
        folder.mkdir(parents=True)
        (folder / "model.pt").write_bytes(spec.fit_id.encode() * 100)
        record = {**fake_record(spec, wave2), "checkpoint_sha256": hashlib.sha256(spec.fit_id.encode() * 100).hexdigest()}
        (folder / "record.json").write_text(json.dumps(record), encoding="utf-8")
        (folder / "resume").mkdir()
    (download / "ffffffffffffffff").mkdir()
    with pytest.raises(ValueError, match="belong to no planned wave"):
        f9_stages.import_f9_fits(ctx, download, wave=2)
    (download / "ffffffffffffffff").rmdir()
    first = wave2["fit_specs"][0].fit_id
    good = (download / first / "model.pt").read_bytes()
    (download / first / "model.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="differs from its record"):
        f9_stages.import_f9_fits(ctx, download, wave=2)
    (download / first / "model.pt").write_bytes(good)
    imported = f9_stages.import_f9_fits(ctx, download, wave=2)
    assert imported.value["files"] == {"imported": 36}
    assert not (paths.fit_dir(first) / "resume").exists()
    ledger = pd.read_csv(paths.root / "import_ledger.csv")
    assert len(ledger) == 36 and set(ledger["action"]) == {"imported"}
    return paths, wave1, wave2, download


def test_select_import_freeze_and_once_only(planned, tmp_path):
    ctx, _ = planned
    paths, _, wave2, download = _select_and_import(ctx, tmp_path)
    with pytest.raises(FileNotFoundError, match="before freeze-f9"):          # the gate needs the recorded OOF predictions
        f9_stages.freeze_f9(ctx)
    write_oof(ctx, wave2)
    frozen = f9_stages.freeze_f9(ctx)
    bundle = json.loads(paths.bundle.read_text(encoding="utf-8"))
    assert frozen.value["fits"] == 18 and frozen.value["primary_fits"] == 6
    assert bundle["seed_selection"] == "forbidden" and bundle["primary_seed"] == 20261011 and not bundle["ensemble"]
    assert {f["encoder_id"] for f in bundle["fits"]} == {"fold1", "fold2", "fold3", "fold4", "fold5", "final"}
    report_path = paths.root / "validity_gate.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] and bundle["validity_gate"]["passed"]
    assert bundle["validity_gate"]["report_sha256"] == hashlib.sha256(report_path.read_bytes()).hexdigest()
    assert report["checks"][0]["observed"] == pytest.approx(0.8, abs=0.05) and len(report["refits"]) == 6 and len(report["selected_fits"]) == 18
    for stage in (lambda: f9_stages.select_f9(ctx), lambda: f9_stages.import_f9_fits(ctx, download, wave=2),
                  lambda: f9_stages.train_f9(ctx, "neural_f9/plan/wave2_specs.json"), lambda: f9_stages.plan_f9(ctx)):
        with pytest.raises(BundleError, match="frozen"):
            stage()


# ----------------------------------------------------------------------------- technical-validity gate (pre-evaluation bug check)


def test_validity_gate_refuses_with_a_report_and_never_changes_hyperparameters(planned, tmp_path, monkeypatch):
    ctx, _ = planned
    paths, _, wave2, _ = _select_and_import(ctx, tmp_path)
    write_oof(ctx, wave2, correct_per_10=2)                     # development OOF macro-F1 near chance (0.20)
    with pytest.raises(f9_stages.ValidityGateError, match="PRE_EVALUATION_BUG_CHECK") as info:
        f9_stages.freeze_f9(ctx)
    assert info.value.reason == "PRE_EVALUATION_BUG_CHECK" and isinstance(info.value, BundleError)      # exit 7 through the CLI
    assert not paths.bundle.exists() and ctx.run.read_record("freeze-f9") is None
    report = json.loads((paths.root / "validity_gate.json").read_text(encoding="utf-8"))
    assert report["passed"] is False and report["failed_checks"] == ["oof_macro_f1"] and report["reason"] == "PRE_EVALUATION_BUG_CHECK"
    assert report["checks"][0]["observed"] < 0.30 and "invalidate the affected development artifacts" in report["rule"]
    assert any(a.endswith("predictions_acdc_dev.parquet") for a in report["affected_development_artifacts"])
    # A failure never changes a hyperparameter or a value of the gate: a configuration other than the plan's is refused.
    cfg = load_f9_config()
    cfg["validity_gate"]["min_oof_macro_f1"] = 0.10
    monkeypatch.setattr(f9_stages, "_config", lambda ctx: cfg)
    with pytest.raises(BundleError, match="never changes after planning"):
        f9_stages.freeze_f9(ctx)
    assert not paths.bundle.exists()


def _gate_inputs(n: int = 50):
    from tmmri.neural.prediction_table import prediction_frame

    classes = list(CLASS_SIZES)
    ids = [f"synthetic{i:03d}" for i in range(n)]
    labels = [classes[i % 5] for i in range(n)]
    frame = prediction_frame(sample_id=ids, probabilities=np.eye(5)[[classes.index(c) for c in labels]], model_id="f9", partition="acdc_dev",
                             seed=PRIMARY, encoder_id="fold1", is_oof=True, cluster_id=ids, y_true=labels)
    good = {"epochs": [{"epoch": 1, "train_loss": 1.6}, {"epoch": 2, "train_loss": 1.2}],
            "parameters": {"unfrozen_conv_max_abs_change": {"backbone.layer4.1.conv2.weight": 0.01}, "bn_running_statistics_unchanged": True,
                           "frozen_weights_unchanged": True}}
    return classes, ids, frame, good


@pytest.mark.parametrize("case, failed", [
    ("pass", []),
    ("chance_level_oof", ["oof_macro_f1"]),
    ("oof_patient_missing", ["oof_macro_f1"]),
    ("refit_loss_rose", ["refit_training_loss_decreased"]),
    ("single_epoch_refit", ["refit_training_loss_decreased"]),
    ("zero_change_selected_fit", ["nonzero_parameter_change"]),
    ("frozen_weights_changed", ["nonzero_parameter_change"]),
])
def test_validity_gate_checks(case, failed):
    gate = load_f9_config()["validity_gate"]
    assert gate["min_oof_macro_f1"] == 0.30 and gate["on_failure"]["reason"] == "PRE_EVALUATION_BUG_CHECK"
    classes, ids, frame, good = _gate_inputs()
    refits, selected = {"r1": dict(good), "r2": dict(good)}, {"s1": dict(good)}
    if case == "chance_level_oof":
        frame = frame.assign(y_pred=frame["y_true"].map(lambda c: classes[(classes.index(c) + 1) % 5]))
    elif case == "oof_patient_missing":
        frame = frame.iloc[1:]
    elif case == "refit_loss_rose":
        refits["r2"] = {**good, "epochs": [{"epoch": 1, "train_loss": 1.2}, {"epoch": 2, "train_loss": 1.3}]}
    elif case == "single_epoch_refit":
        refits["r1"] = {**good, "epochs": [{"epoch": 1, "train_loss": 1.2}]}
    elif case == "zero_change_selected_fit":
        selected["s1"] = {**good, "parameters": {**good["parameters"], "unfrozen_conv_max_abs_change": {"backbone.layer4.1.conv2.weight": 0.0}}}
    elif case == "frozen_weights_changed":
        refits["r1"] = {**good, "parameters": {**good["parameters"], "frozen_weights_unchanged": False}}
    report = f9_stages.validity_gate(gate, classes=classes, primary=PRIMARY, refits=refits, selected_fits=selected, predictions=frame, dev_ids=ids)
    assert report["failed_checks"] == failed and report["passed"] is (not failed)
    other_seed = frame.assign(seed=PRIMARY + 1)                  # only the primary seed is judged
    assert f9_stages.oof_macro_f1(other_seed, classes, seed=PRIMARY, dev_ids=ids)["n_missing"] == len(ids)


# ----------------------------------------------------------------------------- CPE-1


def _pinned(monkeypatch, pin="1.2.3"):
    cfg = load_f9_config()
    cfg["backend"]["modal"]["client_version"] = pin
    monkeypatch.setattr(f9_stages, "_config", lambda ctx: cfg)
    return cfg


def test_upload_cache_refuses_files_outside_cpe1(tmp_path, monkeypatch):
    ctx = make_run(tmp_path, n_npz=None)
    run = ctx.run
    cache = run.path / "neural_cache" / "acdc_dev"
    assert len(cpe1_upload_set(run.path)) == 99
    (cache / "notes.txt").write_text("stray", encoding="utf-8")
    with pytest.raises(CloudPolicyError, match="outside the permitted set"):
        f9_stages.upload_cache(ctx, dry_run=True)
    (cache / "notes.txt").unlink()
    (cache / "nested").mkdir()
    with pytest.raises(CloudPolicyError):
        cpe1_upload_set(run.path)
    (cache / "nested").rmdir()
    with pytest.raises(CloudPolicyError, match="not permitted"):
        cpe1_upload_set(run.path, requested=[run.path / "contracts" / "split_manifest.parquet"])
    manifest = pd.read_csv(cache / "manifest.csv", dtype=str, keep_default_na=False)
    manifest.loc[0, "partition"] = "acdc_test"
    manifest.to_csv(cache / "manifest.csv", index=False, lineterminator="\n")
    with pytest.raises(CloudPolicyError, match="only acdc_dev"):
        cpe1_upload_set(run.path)
    manifest.loc[0, "partition"] = "acdc_dev"
    manifest.loc[1, "npz_sha256"] = "0" * 64
    manifest.to_csv(cache / "manifest.csv", index=False, lineterminator="\n")
    with pytest.raises(CloudPolicyError, match="differs from the manifest"):
        cpe1_upload_set(run.path)
    assert not (run.path / "cloud" / "modal_ledger.csv").exists()
    code = None
    try:
        cli.main(["upload-cache", "--run-id", RUN_ID, "--artifact-root", str(tmp_path / "artifacts"), "--dry-run"])
    except SystemExit as exc:
        code = exc.code
    assert code == cli.EXIT_CONTRACT_VIOLATION


def test_upload_cache_ledgers_before_each_upload(tmp_path, monkeypatch):
    ctx = make_run(tmp_path)
    run = ctx.run
    ledger = run.path / "cloud" / "modal_ledger.csv"
    _pinned(monkeypatch, pin=None)                                     # an unpinned client is refused (independent of the repository pin)
    with pytest.raises(ValueError, match="not pinned"):
        f9_stages.upload_cache(ctx, dry_run=True)
    _pinned(monkeypatch)
    dry = f9_stages.upload_cache(ctx, dry_run=True)
    assert dry["files"] == 99 and not ledger.exists()
    calls = []

    def runner(command):
        rows = pd.read_csv(ledger, dtype=str, keep_default_na=False)
        assert rows.iloc[-1]["remote_path"] == command[-1]           # the ledger row precedes the upload
        calls.append(command)

    result = f9_stages.upload_cache(ctx, runner=runner)
    rows = pd.read_csv(ledger, dtype=str, keep_default_na=False)
    assert len(calls) == 99 and len(rows) == 99 and set(rows["action"]) == {"upload"} and set(rows["volume"]) == {"tmmri-neural-cache"}
    assert calls[0][:5] == ["uvx", "--from", "modal==1.2.3", "modal", "volume"] and calls[0][-1] == f"/{RUN_ID}/neural_cache/acdc_dev/manifest.csv"
    assert rows.iloc[1]["sha256"] == hashlib.sha256((run.path / rows.iloc[1]["local_path"]).read_bytes()).hexdigest()
    assert result.record.outputs
    with pytest.raises(CloudPolicyError):
        f9_stages.record_volume_deletion(ctx, "some-other-volume")
    f9_stages.record_volume_deletion(ctx, "tmmri-neural-runs")
    assert pd.read_csv(ledger).iloc[-1]["action"] == "delete_volume"


def test_backend_block_parser_matches_yaml():
    yaml = pytest.importorskip("yaml")
    text = (REPO / "configs" / "f9_cnn.yaml").read_text(encoding="utf-8")
    assert parse_block_mapping(backend_block(text))["backend"] == yaml.safe_load(text)["backend"]
    modal_cfg = read_backend_config(REPO / "configs" / "f9_cnn.yaml")["modal"]
    assert modal_cfg["gpu"] == "L40S" and modal_cfg["max_containers"] == 10
    assert (modal_cfg["cache_volume"], modal_cfg["runs_volume"]) == ("tmmri-neural-cache", "tmmri-neural-runs")


def test_f9_config_matches_the_protocol():
    cfg = load_f9_config()
    assert cfg["seeds"]["values"] == [20261011, 20261012, 20261013] and cfg["seeds"]["primary"] == 20261011
    assert cfg["fit_counts"] == {"wave1_selection": 324, "wave2_refit": 18, "total": 342, "s5_total": 570}
    assert cfg["selection"]["salt"] == "f9-select-v6.0.1|" and cfg["epochs"]["max"] == 80 and cfg["batch"]["effective_patients"] == 4
    assert cfg["early_stopping"] == {"metric": "validation_cross_entropy", "min_epoch": 20, "patience": 12, "min_delta": 0.0001,
                                     "training_metrics_select_checkpoints": False}


def test_modal_script_static_contract():
    source = (REPO / "scripts" / "cloud" / "modal_train.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_imports = {alias.name.split(".")[0] for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
                   for alias in node.names} | {node.module.split(".")[0] for node in tree.body if isinstance(node, ast.ImportFrom) and node.module}
    assert "torch" not in top_imports and "tmmri" not in top_imports
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    nested = {(a.name if isinstance(n, ast.Import) else (n.module or "")).split(".")[0] for n in ast.walk(main)
              if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    assert "torch" not in nested and not any("torch" in m for n in ast.walk(main) if isinstance(n, ast.ImportFrom) for m in [n.module or ""])
    assert "order_outputs=False" in source and '"max_containers"' in source and "modal.Retries(" in source
    assert "add_local_python_source(\"tmmri\")" in source and "uv sync --locked --no-install-project" in source


@pytest.mark.parametrize("command", ["plan-f9", "train-f9", "select-f9", "import-f9-fits", "freeze-f9", "upload-cache"])
def test_every_f9_subcommand_has_help(command, capsys):
    from tmmri.neural.f9_cli import F9_COMMANDS

    with pytest.raises(SystemExit) as info:
        cli.main([command, "--help"])
    assert info.value.code == 0 and command in capsys.readouterr().out and command in F9_COMMANDS
