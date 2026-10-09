"""Local F9 inference (``infer-f9``; contracts C4, C8, C9) with a stub model and synthetic tensors only.

``tmmri.neural.model`` and ``tmmri.neural.data`` belong to the trainer lane; these tests replace both modules by
stubs that follow contract C8, so they exercise checkpoint discovery, the leakage checks, the C4 tables, the stage
record and the evaluation-partition freeze requirement without any trained network.  Run in the GPU image on CPU."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from tmmri import stages
from tmmri.config import load_study
from tmmri.neural import inference as inf
from tmmri.neural import preprocess as pp
from tmmri.neural.image_stages import infer_f9
from tmmri.neural.prediction_table import (
    activation_columns,
    validate_prediction_frame,
)
from tmmri.runs import RunRoot

SEEDS = (20261011, 20261012)
CLASSES = ("DCM", "HCM", "MINF", "NOR", "RV")


class StubF9(torch.nn.Module):
    """C8 signature: forward(ed, es, ed_geom, es_geom, ed_mask, es_mask) -> (logits [B, 5], a [B, 64])."""

    def __init__(self, seed: int, fold: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed * 10 + fold)
        self.proj = torch.nn.Parameter(torch.randn(6, 64, generator=generator))
        self.head = torch.nn.Parameter(torch.randn(64, 5, generator=generator))

    def forward(self, ed, es, ed_geom, es_geom, ed_mask, es_mask):
        def pooled(x, g, m):
            w = m.to(x.dtype)
            mean = (x.mean(dim=(-1, -2)) * w).sum(1) / w.sum(1)
            return torch.stack([mean, (g[..., 0] * w).sum(1) / w.sum(1), (g[..., 1] * w).sum(1) / w.sum(1)], dim=1)

        a = torch.nn.functional.gelu(torch.cat([pooled(ed, ed_geom, ed_mask), pooled(es, es_geom, es_mask)], dim=1) @ self.proj)
        return a @ self.head, a


def _stub_modules():
    model = types.ModuleType("tmmri.neural.model")
    data = types.ModuleType("tmmri.neural.data")

    def load_f9(checkpoint_dir):
        record = json.loads((Path(checkpoint_dir) / "record.json").read_text(encoding="utf-8"))["spec"]
        fold = 0 if record["outer_fold"] == "final" else int(record["outer_fold"])
        return StubF9(int(record["seed"]), fold).eval()

    def load_npz(path):
        with np.load(path) as arrays:
            return {k: torch.from_numpy(np.asarray(arrays[k])) for k in ("ed", "es", "ed_geom", "es_geom")}

    def collate_patients(samples):
        out = {}
        for phase in ("ed", "es"):
            n = max(s[phase].shape[0] for s in samples)
            out[phase] = torch.zeros((len(samples), n) + tuple(samples[0][phase].shape[1:]))
            out[f"{phase}_geom"] = torch.zeros((len(samples), n, 2))
            out[f"{phase}_mask"] = torch.zeros((len(samples), n), dtype=torch.bool)
            for i, s in enumerate(samples):
                k = s[phase].shape[0]
                out[phase][i, :k], out[f"{phase}_geom"][i, :k], out[f"{phase}_mask"][i, :k] = s[phase], s[f"{phase}_geom"], True
        return out

    model.load_f9, data.load_npz, data.collate_patients = load_f9, load_npz, collate_patients
    return model, data


@pytest.fixture()
def stubs(monkeypatch):
    model, data = _stub_modules()
    monkeypatch.setitem(sys.modules, "tmmri.neural.model", model)
    monkeypatch.setitem(sys.modules, "tmmri.neural.data", data)


def _cases(ids, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i, sid in enumerate(ids):
        shape = (32, 30, 3 + i % 2)
        vol = rng.normal(100.0, 15.0, size=shape).astype(np.float32)
        geom = pp.geometry_from_parameters(shape, (1.4, 1.4), 9.0, np.eye(3))
        out.append(pp.CaseVolumes(sample_id=sid, ed_image=vol, es_image=vol * 0.8 + 5.0, ed_geometry=geom, es_geometry=geom,
                                  cluster_id=sid, label=CLASSES[i % 5], dataset="ACDC", case_id=sid.split(":")[1]))
    return out


def _write_checkpoints(root, dev_folds, *, leak=False):
    ids = sorted(dev_folds)
    for seed in SEEDS:
        for fold in (1, 2, 3, 4, 5, "final"):
            fitting = ids if fold == "final" else [s for s in ids if dev_folds[s] != fold or (leak and fold == 1)]
            fit_id = f"{seed}-{fold}"
            spec = {"fit_id": fit_id, "purpose": "f9", "seed": seed, "outer_fold": fold, "fitting_patients": fitting,
                    "selection_fold": None, "candidate": {"unfreeze": "layer4", "lr_backbone": 3e-5}, "refit_epochs": 31,
                    "fraction": 1.0, "repeat": 0}
            path = root / fit_id
            path.mkdir(parents=True)
            torch.save({}, path / "model.pt")
            (path / "record.json").write_text(json.dumps({"spec": spec}), encoding="utf-8")
    selection = root / "selection-fit"                              # a selection fit is never used for inference
    selection.mkdir()
    torch.save({}, selection / "model.pt")
    (selection / "record.json").write_text(json.dumps({"spec": {"purpose": "f9", "seed": SEEDS[0], "outer_fold": 1, "selection_fold": 2,
                                                                 "fitting_patients": ids, "fraction": 1.0, "repeat": 0}}), encoding="utf-8")


DEV_IDS = [f"ACDC:patient{i:03d}" for i in range(1, 11)]
DEV_FOLDS = {sid: (i % 5) + 1 for i, sid in enumerate(DEV_IDS)}


def test_run_f9_inference_writes_c4_tables(tmp_path, stubs):
    from tmmri.neural.torch_numerics import configure_torch_numerics

    configure_torch_numerics()
    cache = tmp_path / "cache"
    built = pp.build_tensor_cache_from_volumes(_cases(DEV_IDS), cache, "acdc_dev")
    assert built["failures"].empty
    _write_checkpoints(tmp_path / "ckpt", DEV_FOLDS)
    encoders = inf.select_encoders(inf.discover_checkpoints(tmp_path / "ckpt"), ["fold1", "fold2", "fold3", "fold4", "fold5", "final"])
    assert len(encoders) == 12 and {e.seed for e in encoders} == set(SEEDS)
    inf.check_leakage(encoders, "acdc_dev", set(DEV_IDS), DEV_FOLDS)
    labels = dict(zip(built["manifest"]["sample_id"], built["manifest"]["label"]))
    cpu = torch.device("cpu")
    predictions, activations, details = inf.run_f9_inference(cache, encoders, partition="acdc_dev", device=cpu, dev_folds=DEV_FOLDS,
                                                             y_true=labels)
    validate_prediction_frame(predictions)
    assert len(predictions) == len(activations) == 10 * 12
    assert list(activations.columns[-64:]) == activation_columns(64) and activations.columns[-1] == "a_063"
    oof = predictions[predictions["is_oof"]]
    assert len(oof) == 10 * len(SEEDS)                                 # every patient once per seed, by its own fold encoder
    assert all(e == f"fold{DEV_FOLDS[s]}" for s, e in zip(oof["sample_id"], oof["encoder_id"]))
    assert set(predictions["model_id"]) == {"f9"} and set(predictions["seed"]) == set(SEEDS)
    assert (predictions["y_true"].astype(str).to_numpy() == predictions["sample_id"].map(labels).to_numpy()).all()
    probs = predictions[[f"prob_{c}" for c in CLASSES]].to_numpy()
    logits = predictions[[f"logit_{c}" for c in CLASSES]].to_numpy()
    np.testing.assert_allclose(probs, np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True), rtol=1e-10)
    again, again_act, _ = inf.run_f9_inference(cache, encoders, partition="acdc_dev", device=cpu, dev_folds=DEV_FOLDS, y_true=labels)
    pd.testing.assert_frame_equal(predictions, again)                 # deterministic
    pd.testing.assert_frame_equal(activations, again_act)
    assert len(details["encoders"]) == 12 and all(len(e["model_pt_sha256"]) == 64 for e in details["encoders"])


def test_leaking_checkpoints_are_refused(tmp_path):
    _write_checkpoints(tmp_path / "ckpt", DEV_FOLDS, leak=True)
    encoders = inf.select_encoders(inf.discover_checkpoints(tmp_path / "ckpt"), ["fold1", "final"])
    with pytest.raises(inf.CheckpointError, match="held-out outer fold"):
        inf.check_leakage(encoders, "acdc_dev", set(DEV_IDS), DEV_FOLDS)
    final = [e for e in encoders if e.encoder_id == "final"]
    with pytest.raises(inf.CheckpointError, match="acdc_test patients"):
        inf.check_leakage(final, "acdc_test", {DEV_IDS[0]}, DEV_FOLDS)
    with pytest.raises(inf.CheckpointError, match="missing"):
        inf.select_encoders(final, ["fold1", "final"])


def test_infer_f9_stage_records_and_requires_the_f9_freeze_for_evaluation(tmp_path, stubs):
    study = load_study()
    run = RunRoot("v6-f9infer-20261003-1100", tmp_path / "artifacts", subfolders=study.study["artifact_subfolders"])
    test_ids = ["ACDC:patient101", "ACDC:patient102"]
    rows = [{"sample_id": s, "dataset": "ACDC", "case_id": s.split(":")[1], "patient_id": s.split(":")[1], "partition": "acdc_dev",
             "outer_fold": DEV_FOLDS[s], "cluster_id": s} for s in DEV_IDS]
    rows += [{"sample_id": s, "dataset": "ACDC", "case_id": s.split(":")[1], "patient_id": s.split(":")[1], "partition": "acdc_test",
              "outer_fold": None, "cluster_id": s} for s in test_ids]
    (run.path / "contracts").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(run.path / "contracts" / "split_manifest.parquet", index=False)
    pp.build_tensor_cache_from_volumes(_cases(DEV_IDS), run.path / "neural_cache" / "acdc_dev", "acdc_dev")
    _write_checkpoints(run.path / "neural_f9" / "checkpoints", DEV_FOLDS)
    ctx = stages.StageContext(study=study, run=run, data_root=tmp_path / "data", n_jobs=1)
    first = infer_f9(ctx, "acdc_dev", device="cpu")
    assert first.action == "RAN" and infer_f9(ctx, "acdc_dev", device="cpu").action == "SKIPPED"
    out = run.path / "neural_f9"
    combined = pd.read_parquet(out / "predictions.parquet")
    pd.testing.assert_frame_equal(combined, pd.read_parquet(out / "predictions_acdc_dev.parquet"))
    record = json.loads((out / "inference_acdc_dev.json").read_text(encoding="utf-8"))
    assert record["environment"]["numerics"]["tf32_enabled"] is False and record["environment"]["device"] == "cpu"
    assert record["numerics_at_start"]["deterministic_algorithms"] is True and len(record["encoders"]) == 12
    with pytest.raises(stages.BundleError, match="f9_bundle"):
        infer_f9(ctx, "acdc_test", device="cpu")
