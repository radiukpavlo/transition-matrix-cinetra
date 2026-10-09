"""train_fit (contract C5) on CPU with synthetic C3 caches: identical fits are bitwise equal, an interrupted and
resumed fit equals the uninterrupted one, the fit record, refits, NE_INSUFFICIENT_CLASS_SUPPORT and the local
train-f9 stage.  Runs in the GPU image with CUDA hidden; never reads cohort data."""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tmmri.neural import f9_stages
from tmmri.neural.data import read_manifest
from tmmri.neural.f9_config import load_f9_config, training_config
from tmmri.neural.fit_spec import (
    CANDIDATES,
    PRIMARY_SEED,
    Candidate,
    FitSpec,
    read_spec_file,
    write_spec_file,
)
from tmmri.neural.model import load_f9
from tmmri.neural.qualify import run_qualification
from tmmri.neural.selection import NE_INSUFFICIENT_CLASS_SUPPORT
from tmmri.neural.synthetic_cache import (
    write_random_resnet18,
    write_synthetic_cache,
)
from tmmri.neural.train import (
    STATUS_COMPLETE,
    FitRecord,
    WeightsRef,
    predict_logits,
    train_fit,
)
from tmmri.runs import sha256_file
from tmmri.stages import make_context


def tiny_config(epochs: int = 3) -> dict:
    cfg = training_config(load_f9_config())
    cfg = copy.deepcopy(cfg)
    cfg["epochs"] = {"max": epochs}
    cfg["early_stopping"] = {**cfg["early_stopping"], "min_epoch": 1, "patience": 2}
    return cfg


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("f9")
    manifest = write_synthetic_cache(root / "cache", n_per_class=3, slices=(2, 3), size=32)
    path, sha = write_random_resnet18(root / "weights.pt")
    ids = tuple(sorted(read_manifest(manifest)["sample_id"]))
    return root, ids, WeightsRef(path=str(path), sha256=sha, name="synthetic_random_init")


def test_qualification_bitwise_repeat_resume_and_ieee(tmp_path):
    report = run_qualification(tmp_path, "cpu", config=tiny_config(), size=32, n_per_class=3, slices=(2, 3), epochs=2, bench_steps=1)
    assert report["synthetic_only"] and report["tf32_disabled"] and report["ieee_fp32"]["passed"]
    assert report["determinism"]["passed"], report["determinism"]
    assert report["resume"]["passed"] and report["resume"]["resumed_from_epoch"] == 2, report["resume"]
    assert report["benchmark"]["layer4"]["train_slices_per_s"] > 0 and report["passed"]
    assert report["environment"]["numerics"]["deterministic_algorithms"] and not report["environment"]["numerics"]["tf32_enabled"]


def test_selection_fit_record_and_idempotence(setup):
    root, ids, weights = setup
    spec = FitSpec(purpose="f9", seed=PRIMARY_SEED, outer_fold="final", fitting_patients=ids, selection_fold=2,
                   candidate=Candidate("layer4", 0.0001), refit_epochs=None)
    out = root / "fits" / spec.fit_id
    record = train_fit(spec, root / "cache", out, "cpu", config=tiny_config(), weights=weights, backend={"name": "test"})
    assert record.status == STATUS_COMPLETE and 1 <= record.selected_epoch <= record.stopped_epoch <= 3
    assert record.support_val == {c: 1 for c in record.class_order} and sum(record.support_train.values()) == 10
    assert np.mean(record.class_weights) == pytest.approx(1.0)
    assert record.parameters["bn_running_statistics_unchanged"] and record.parameters["frozen_weights_unchanged"]
    assert record.parameters["unfrozen_conv_max_abs_change"]["backbone.layer4.1.conv2.weight"] > 0
    assert record.parameters["initial_state_sha256"] != record.state_sha256
    assert record.checkpoint_sha256 == sha256_file(out / "model.pt") and not (out / "resume").exists()
    assert record.environment["numerics"]["tf32_enabled"] is False and record.environment["device"] == "cpu"
    assert set(record.val_metrics) >= {"cross_entropy", "macro_f1", "accuracy", "ece_equal_width_10", "brier", "log_loss", "confusion"}
    assert len(record.val_predictions) == 5 and all(abs(sum(p[f"prob_{c}"] for c in record.class_order) - 1) < 1e-9 for p in record.val_predictions)
    assert record.epochs[record.selected_epoch - 1]["val_cross_entropy"] == pytest.approx(record.val_metrics["cross_entropy"])
    assert FitRecord.load(out / "record.json").deterministic_view() == record.deterministic_view()
    again = train_fit(spec, root / "cache", out, "cpu", config=tiny_config(), weights=weights)
    assert again.timings == record.timings                               # completed fits are returned, never retrained
    model = load_f9(out)
    from tmmri.neural.data import load_npz

    items = [load_npz(root / "cache" / "acdc_dev" / f"{sid}.npz") for sid in record.val_ids]
    logits, _ = predict_logits(model, items, torch.device("cpu"))
    np.testing.assert_allclose(logits[:, 0], [p["logit_DCM"] for p in record.val_predictions], rtol=1e-5, atol=1e-5)


def test_refit_trains_exactly_refit_epochs_without_validation(setup):
    root, ids, weights = setup
    spec = FitSpec(purpose="f9", seed=20261012, outer_fold=3, fitting_patients=ids, selection_fold=None, candidate=CANDIDATES[4], refit_epochs=2)
    record = train_fit(spec, root / "cache", root / "refits" / spec.fit_id, "cpu", config=tiny_config(), weights=weights)
    assert [e["epoch"] for e in record.epochs] == [1, 2] and record.selected_epoch == 2 and record.val_ids == []
    assert "val_cross_entropy" not in record.epochs[0] and record.stop_reason == "refit_epochs"
    assert record.epochs[1]["lr_backbone"] < record.epochs[0]["lr_backbone"] == pytest.approx(0.00003)   # cosine over 80 epochs
    assert record.epochs[0]["lr_new"] == pytest.approx(0.0003)


def test_insufficient_class_support_is_recorded(tmp_path):
    manifest = write_synthetic_cache(tmp_path / "cache", n_per_class=2, slices=(1, 2), size=32)
    ids = tuple(sorted(read_manifest(manifest)["sample_id"]))
    spec = FitSpec(purpose="f9", seed=PRIMARY_SEED, outer_fold=1, fitting_patients=ids, selection_fold=1, candidate=CANDIDATES[0], refit_epochs=None)
    record = train_fit(spec, tmp_path / "cache", tmp_path / "fit", "cpu", config=tiny_config(), weights={"path": None, "sha256": "0" * 64})
    assert record.status == NE_INSUFFICIENT_CLASS_SUPPORT and "fewer than 3 clusters" in record.reason
    assert json.loads((tmp_path / "fit" / "record.json").read_text())["status"] == NE_INSUFFICIENT_CLASS_SUPPORT
    assert not (tmp_path / "fit" / "model.pt").exists()


def test_train_fit_refuses_rows_outside_acdc_dev_and_wrong_hashes(tmp_path, setup):
    root, ids, weights = setup
    spec = FitSpec(purpose="f9", seed=PRIMARY_SEED, outer_fold=1, fitting_patients=ids[:-1] + ("heldout999",), selection_fold=1,
                   candidate=CANDIDATES[0], refit_epochs=None)
    with pytest.raises(ValueError, match="not in"):
        train_fit(spec, root / "cache", tmp_path / "a", "cpu", config=tiny_config(), weights=weights)
    good = FitSpec(purpose="f9", seed=PRIMARY_SEED, outer_fold=1, fitting_patients=ids, selection_fold=1, candidate=CANDIDATES[0], refit_epochs=None)
    with pytest.raises(ValueError, match="SHA-256"):
        train_fit(good, root / "cache", tmp_path / "b", "cpu", config=tiny_config(), weights=weights, expected_manifest_sha256="0" * 64)


def test_train_f9_local_stage(tmp_path, monkeypatch, setup):
    _, ids, weights = setup
    ctx = make_context("configs/study_v6.yaml", "v6-f9local-20261003-1000", artifact_root=tmp_path / "artifacts")
    run = ctx.run
    write_synthetic_cache(run.path / "neural_cache", n_per_class=3, slices=(2, 3), size=32)
    cfg = load_f9_config()
    cfg.update(tiny_config(2))
    monkeypatch.setattr(f9_stages, "_config", lambda ctx: cfg)
    monkeypatch.setattr(f9_stages, "local_weights", lambda payload: {"name": weights.name, "sha256": weights.sha256, "path": weights.path})
    specs = [FitSpec(purpose="f9", seed=PRIMARY_SEED, outer_fold="final", fitting_patients=ids, selection_fold=f, candidate=CANDIDATES[2],
                     refit_epochs=None) for f in (1, 2)]
    spec_file = run.path / "neural_f9" / "plan" / "wave1_specs.json"
    write_spec_file(spec_file, specs, run_id=run.run_id, wave=1, config=training_config(cfg),
                    cache_manifest_sha256=sha256_file(run.path / "neural_cache" / "acdc_dev" / "manifest.csv"),
                    weights={"name": weights.name, "sha256": weights.sha256, "file": "x.pth"})
    result = f9_stages.train_f9(ctx, "neural_f9/plan/wave1_specs.json")
    assert result.action == "RAN" and result.value == {"fits": 2, "statuses": {STATUS_COMPLETE: 2}}
    for spec in read_spec_file(spec_file)["fit_specs"]:
        record = json.loads((run.path / "neural_f9" / "checkpoints" / spec.fit_id / "record.json").read_text())
        assert record["backend"]["name"] == "local" and record["fit_id"] == spec.fit_id
    assert f9_stages.train_f9(ctx, "neural_f9/plan/wave1_specs.json").action == "SKIPPED"
    assert pd.read_csv(run.path / "neural_cache" / "acdc_dev" / "manifest.csv")["partition"].eq("acdc_dev").all()
