"""D20 deep-encoder branch (SPECIFICATION.md 8.4; protocol-v6.0.1 D item 9) on synthetic data only.

The fitting test runs on the host (no torch): random 2048-d vectors stand in for the frozen encoder.  The extraction
tests need torch and run in the GPU image on the CPU with a random-weight ResNet-50 (no ImageNet weights)."""
from __future__ import annotations

import json

import fixtures_v6 as fx
import numpy as np
import pandas as pd
import pytest

from tmmri import deep_encoder as de
from tmmri import stages
from tmmri.config import load_study
from tmmri.neural import preprocess as pp
from tmmri.neural.image_stages import fit_deep_encoder
from tmmri.neural.prediction_table import validate_prediction_frame
from tmmri.runs import RunRoot


def _volume_case(sample_id, shape=(30, 28, 3), seed=0, constant_slice=None):
    rng = np.random.default_rng(seed)
    vol = rng.normal(100.0, 20.0, size=shape).astype(np.float32)
    if constant_slice is not None:
        vol[..., constant_slice] = 7.0
    geom = pp.geometry_from_parameters(shape, (1.25, 1.25), 8.0, np.eye(3))
    return pp.CaseVolumes(sample_id=sample_id, ed_image=vol, es_image=vol[..., :2].copy() + 1.0, ed_geometry=geom,
                          es_geometry=pp.geometry_from_parameters(shape[:2] + (2,), (1.25, 1.25), 8.0, np.eye(3)),
                          cluster_id=sample_id, dataset="SYN", case_id=sample_id.split(":")[1])


def test_per_slice_normalization_follows_the_frozen_rule():
    flat = de.normalize_slice(np.full((5, 5), 3.0))
    assert np.array_equal(flat, np.zeros((5, 5)))                 # hi <= lo -> scale 1 (configs/formal.yaml convention)
    ramp = de.normalize_slice(np.arange(100.0).reshape(10, 10))
    assert ramp.min() == 0.0 and ramp.max() == 1.0
    np.testing.assert_allclose(ramp.reshape(-1)[50], (50 - 0.99) / (98.01 - 0.99))
    nan_case = _volume_case("SYN:nan")
    nan_case.ed_image[0, 0, 0] = np.inf
    with pytest.raises(pp.InputRejected) as info:
        de.d20_phase_slices(nan_case)
    assert info.value.reason_code == "INPUT_NONFINITE"


# ----------------------------------------------------------------------------- fitting (host, no torch)


@pytest.fixture()
def dev_run(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "protocol_frozen", lambda *a, **k: False)
    acdc = fx.make_acdc_source(tmp_path / "src" / "acdc", fx.acdc_patients(5, testing_per_class=1))
    mms = fx.make_mms_source(tmp_path / "src" / "mms")
    study = load_study()
    run = RunRoot("v6-d20-20261003-1000", tmp_path / "artifacts", subfolders=study.study["artifact_subfolders"])
    ctx = stages.StageContext(study=study, run=run, data_root=tmp_path / "data", n_jobs=1, sources={"acdc": acdc, "mms": mms},
                              source_manifests={"ACDC": fx.acdc_manifest(acdc), "MMS": fx.mms_manifest(mms)},
                              mms_metadata=fx.mms_metadata(mms))
    assert stages.stage_sources(ctx, "acdc").ok and stages.stage_sources(ctx, "mms").ok
    stages.validate_cohort(ctx)
    stages.build_semantics(ctx, "acdc")
    return ctx


def _features(ids, partition, seed):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(len(ids), de.FEATURE_DIM))
    frame = pd.DataFrame(values, columns=de.feature_columns())
    frame.insert(0, "sample_id", list(ids))
    frame.insert(1, "cluster_id", list(ids))
    frame.insert(2, "partition", partition)
    return frame


def test_fit_deep_encoder_writes_frozen_d20_bundle(dev_run):
    ctx = dev_run
    split = pd.read_parquet(ctx.run.path / "contracts" / "split_manifest.parquet")
    dev_ids = sorted(split.loc[split["partition"] == "acdc_dev", "sample_id"])
    test_ids = sorted(split.loc[split["partition"] == "acdc_test", "sample_id"])
    assert len(dev_ids) == 25 and len(test_ids) == 5
    deep = ctx.run.path / "deep_encoder"
    (deep / "features").mkdir(parents=True)
    _features(dev_ids, "acdc_dev", 1).to_parquet(deep / "features" / "A_D20_acdc_dev.parquet", index=False)
    first = fit_deep_encoder(ctx)
    assert first.action == "RAN" and first.value["predictor_rows"] == 25
    predictions = pd.read_parquet(deep / "predictions.parquet")
    validate_prediction_frame(predictions)
    assert set(predictions["model_id"]) == {"d20_predictor"} and set(predictions["seed"]) == {-1}
    oof = predictions[predictions["is_oof"]]
    assert sorted(oof["sample_id"]) == dev_ids                      # every development patient once out of fold
    folds = split.set_index("sample_id").loc[oof["sample_id"], "outer_fold"].to_numpy()
    assert list(oof["encoder_id"]) == [f"fold{f}" for f in folds]
    final = predictions[predictions["encoder_id"] == "final"]
    assert sorted(final["sample_id"]) == dev_ids and not final["is_oof"].any()
    cohort = pd.read_parquet(ctx.run.path / "contracts" / "cohort_ACDC.parquet").set_index("sample_id")
    assert (final["y_true"].astype(str).to_numpy() == cohort.loc[final["sample_id"], "acdc_class"].to_numpy()).all()
    fold_model = json.loads((deep / "transitions" / "fold_models" / "transition_fold1.json").read_text(encoding="utf-8"))
    assert fold_model["representation"] == de.D20_REPRESENTATION
    oof_b = pd.read_parquet(deep / "oof" / "bhat_oof_ACDC_dev.parquet")
    assert set(oof_b["sample_id"]) <= set(dev_ids) and oof_b["outer_fold"].notna().all()
    model, predictor, frozen = de.load_d20_bundle(deep)
    assert frozen["files"]["features/A_D20_acdc_dev.parquet"] and len(frozen["bundle_sha256"]) == 64
    assert predictor.classes == ["DCM", "HCM", "MINF", "NOR", "RV"] and model.representation == de.D20_REPRESENTATION
    test_frame = _features(test_ids, "acdc_test", 2)
    held = de.d20_predictions(deep, test_frame, "acdc_test")
    validate_prediction_frame(held)
    assert len(held) == 5 and set(held["encoder_id"]) == {"final"} and held["y_true"].isna().all()
    assert de.d20_reconstructions(deep, test_frame).shape[0] == 5
    # Content addressing, then the freeze: an identical call skips, a forced re-execution is refused.
    assert fit_deep_encoder(ctx).action == "SKIPPED"
    ctx.force = True
    with pytest.raises(stages.StageRecordLockedError, match="FROZEN_D20"):
        fit_deep_encoder(ctx)
    ctx.force = False
    (deep / "formal" / "d_model_ACDC_dev.parquet").write_bytes(b"tampered")
    with pytest.raises(de.D20BundleError, match="changed after the freeze"):
        de.load_d20_bundle(deep)


def test_d20_fit_refuses_rows_outside_acdc_dev():
    dev = pd.DataFrame({"sample_id": ["ACDC:patient001"], "outer_fold": [1], "cluster_id": ["ACDC:patient001"]})
    features = _features(["ACDC:patient001", "ACDC:patient101"], "acdc_dev", 3)
    with pytest.raises(ValueError, match="outside acdc_dev"):
        de.fit_d20_models(features, dev, pd.DataFrame({"sample_id": []}), pd.Series(dtype=object), set(), targets=[],
                          transition_cfg={}, registry={}, predictor_cfg={}, make_inner_fn=lambda ids: None)


# ----------------------------------------------------------------------------- extraction (torch, CPU)


def test_d20_extraction_is_deterministic_on_a_random_resnet50():
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")
    from tmmri.neural.torch_numerics import configure_torch_numerics, environment_record

    state = configure_torch_numerics()
    assert not state["tf32_enabled"] and state["deterministic_algorithms"] and state["cublas_workspace_config"] == ":4096:8"
    torch.manual_seed(20260613)
    encoder = de.build_d20_encoder(model=torchvision.models.resnet50(weights=None))
    assert not any(p.requires_grad for p in encoder.parameters()) and not encoder.training
    cpu = torch.device("cpu")
    cases = [_volume_case("SYN:a", seed=1), _volume_case("SYN:b", seed=2, constant_slice=1)]
    kwargs = {"size": (64, 64)}
    first, failures = de.extract_d20_features(cases, encoder, device=cpu, partition="acdc_dev", batch_size=32, input_kwargs=kwargs)
    second, _ = de.extract_d20_features(list(reversed(cases)), encoder, device=cpu, partition="acdc_dev", batch_size=32, input_kwargs=kwargs)
    assert failures.empty and list(first["sample_id"]) == ["SYN:a", "SYN:b"]
    pd.testing.assert_frame_equal(first, second)                   # bitwise identical, order independent
    cols = de.feature_columns()
    assert len(cols) == 2048 and np.isfinite(first[cols].to_numpy()).all()
    assert list(first["n_slices_ed"]) == [3, 3] and list(first["n_slices_es"]) == [2, 2]
    x, counts = de.d20_input_tensor(cases[0], size=(64, 64))
    assert tuple(x.shape) == (5, 3, 64, 64) and x.dtype == torch.float32 and counts == {"n_ed": 3, "n_es": 2}
    per_slice = de.encode_slices(encoder, x, device=cpu, batch_size=32)
    assert per_slice.shape == (5, 2048) and per_slice.dtype == np.float32
    symmetrized = de.encode_slices_tta(encoder, x, device=cpu, batch_size=32)       # 6.0.1: 8 D4 transforms x 5 slices
    assert symmetrized.shape == (40, 2048)
    np.testing.assert_array_equal(symmetrized[:5], per_slice)                       # element 0 is the identity
    np.testing.assert_array_equal(first.loc[0, cols].to_numpy(dtype=np.float64), de.patient_vector(symmetrized))
    np.testing.assert_allclose(de.encode_slices(encoder, x, device=cpu, batch_size=2), per_slice, rtol=1e-4, atol=1e-5)
    full, _ = de.d20_input_tensor(cases[1])
    assert tuple(full.shape) == (5, 3, 224, 224)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    sd = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    torch.testing.assert_close(full[1], ((torch.zeros(3, 224, 224) - mean) / sd))   # constant ED slice -> zeros
    record = environment_record(cpu)
    assert record["device"] == "cpu" and record["numerics"]["tf32_enabled"] is False and record["torch"].startswith("2.")
