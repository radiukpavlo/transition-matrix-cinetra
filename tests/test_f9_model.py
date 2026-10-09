"""F9Net (contract C8; NEURAL-TM-1.0 4.1): shapes, padding masks, freezing, BN statistics, parameter groups,
augmentation and checkpoint loading.  CPU, synthetic tensors, random-initialization weights; runs in the GPU image."""
from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tmmri.neural.data import augment_batch, collate_patients
from tmmri.neural.f9_config import load_f9_config
from tmmri.neural.fit_spec import Candidate, FitSpec
from tmmri.neural.model import (
    F9Net,
    build_model,
    cpu_state,
    file_sha256,
    load_f9,
    state_sha256,
)
from tmmri.neural.numerics import configure_numerics
from tmmri.neural.synthetic_cache import write_random_resnet18


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    configure_numerics()
    return write_random_resnet18(tmp_path_factory.mktemp("weights") / "resnet18_random.pt")


def patient(n_ed: int, n_es: int, size: int = 32, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {"sample_id": f"p{seed}", "ed": rng.random((n_ed, size, size), dtype=np.float32), "es": rng.random((n_es, size, size), dtype=np.float32),
            "ed_geom": rng.normal(size=(n_ed, 2)).astype(np.float32), "es_geom": rng.normal(size=(n_es, 2)).astype(np.float32)}


def model_for(weights, unfreeze="layer3+layer4", seed=20261011) -> F9Net:
    path, sha = weights
    return build_model(Candidate(unfreeze, 0.0001), path, seed=seed, expected_sha256=sha)


def test_shapes_head_and_activations(weights):
    model = model_for(weights).eval()
    batch = collate_patients([patient(2, 3, seed=1), patient(5, 4, seed=2), patient(1, 1, seed=3)])
    assert batch["ed"].shape == (3, 5, 32, 32) and batch["es_mask"].tolist()[2] == [True, False, False, False]
    with torch.no_grad():
        logits, a = model.forward_batch(batch)
    assert logits.shape == (3, 5) and a.shape == (3, 64) and torch.isfinite(logits).all()
    assert isinstance(model.head, torch.nn.Linear) and (model.head.in_features, model.head.out_features) == (64, 5)
    assert (model.slice_proj.in_features, model.slice_proj.out_features) == (514, 128)
    assert (model.att_hidden.in_features, model.att_hidden.out_features, model.att_score.out_features) == (128, 64, 1)
    assert (model.fc1.in_features, model.fc1.out_features, model.fc2.out_features) == (768, 256, 64) and model.dropout.p == 0.25
    torch.testing.assert_close(logits, model.head(a))                 # the head reads A only


def test_padding_never_contributes(weights):
    model = model_for(weights).eval()
    a, b = patient(2, 3, seed=4), patient(6, 5, seed=5)
    batch = collate_patients([a, b])
    noisy = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    for phase in ("ed", "es"):
        pad = ~noisy[f"{phase}_mask"]
        noisy[phase][pad] = torch.rand_like(noisy[phase][pad]) * 1e3
        noisy[f"{phase}_geom"][pad] = 1e6
    with torch.no_grad():
        out, act = model.forward_batch(batch)
        out_noisy, act_noisy = model.forward_batch(noisy)
        alone, _ = model.forward_batch(collate_patients([a]))
    assert torch.equal(out, out_noisy) and torch.equal(act, act_noisy)   # identical shapes: bitwise
    torch.testing.assert_close(out[0:1], alone, rtol=1e-5, atol=1e-5)    # another padding width: same values
    with pytest.raises(ValueError, match="prefix"):
        bad = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        bad["ed_mask"][0] = torch.tensor([False, True, True, False, False, False])
        model.forward_batch(bad)


@pytest.mark.parametrize("unfreeze", ["layer4", "layer3+layer4"])
def test_freezing_bn_statistics_and_groups(weights, unfreeze):
    model = model_for(weights, unfreeze)
    trainable = {n.split(".")[1] for n, p in model.backbone.named_parameters(prefix="backbone") if p.requires_grad}
    assert trainable == set(unfreeze.split("+"))
    assert all(p.requires_grad for p in model.new_parameters())
    groups = model.parameter_groups(0.00003, 10)
    assert groups[0]["lr"] == pytest.approx(0.00003) and groups[1]["lr"] == pytest.approx(0.0003)
    assert {id(p) for p in groups[0]["params"]} == {id(p) for p in model.backbone.parameters() if p.requires_grad}
    before = cpu_state(model)
    model.train()
    assert not any(m.training for m in model.backbone.modules() if isinstance(m, torch.nn.BatchNorm2d))
    assert model.dropout.training
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0001)
    logits, _ = model.forward_batch(collate_patients([patient(3, 2, seed=6), patient(2, 2, seed=7)]))
    torch.log_softmax(logits, dim=1)[:, 0].sum().neg().backward()
    optimizer.step()
    after = cpu_state(model)
    running = [k for k in before if "running_" in k]
    assert running and all(torch.equal(before[k], after[k]) for k in running)
    assert torch.equal(before["backbone.layer1.0.conv1.weight"], after["backbone.layer1.0.conv1.weight"])
    assert not torch.equal(before["backbone.layer4.1.conv2.weight"], after["backbone.layer4.1.conv2.weight"])
    assert torch.equal(before["backbone.layer3.1.conv2.weight"], after["backbone.layer3.1.conv2.weight"]) == (unfreeze == "layer4")


def test_build_model_checks_the_pinned_sha256(weights):
    path, _ = weights
    with pytest.raises(ValueError, match="SHA-256"):
        build_model(Candidate("layer4", 0.0001), path, expected_sha256="0" * 64)
    one, two = model_for(weights, seed=1), model_for(weights, seed=1)
    assert state_sha256(cpu_state(one)) == state_sha256(cpu_state(two))
    assert state_sha256(cpu_state(one)) != state_sha256(cpu_state(model_for(weights, seed=2)))


def test_augmentation_is_one_transform_per_patient(weights):
    cfg = dict(load_f9_config()["augmentation"])
    img = np.zeros((32, 32), dtype=np.float32)
    img[10:20, 12:16] = 0.8
    p = {"sample_id": "p", "ed": np.stack([img] * 3), "es": np.stack([img] * 3), "ed_geom": np.zeros((3, 2), np.float32),
         "es_geom": np.zeros((3, 2), np.float32)}
    q = {**p, "ed": p["ed"][:2], "ed_geom": p["ed_geom"][:2]}
    batch = collate_patients([p, q])
    quiet = {**cfg, "noise_sd": 0.0}
    out = augment_batch(batch, np.random.default_rng(0), quiet, spacing_mm=1.5)
    for k in range(3):
        assert torch.equal(out["ed"][0, k], out["es"][0, k]) and torch.equal(out["ed"][0, k], out["ed"][0, 0])
    assert not torch.equal(out["ed"][0, 0], batch["ed"][0, 0])            # the transform moved the image
    assert torch.equal(out["ed"][1, 2], torch.zeros(32, 32))              # padding stays zero
    again = augment_batch(batch, np.random.default_rng(0), cfg, spacing_mm=1.5)
    twice = augment_batch(batch, np.random.default_rng(0), cfg, spacing_mm=1.5)
    assert torch.equal(again["ed"], twice["ed"]) and float(again["ed"].min()) >= 0.0 and float(again["ed"].max()) <= 1.0


def test_load_f9_round_trip(weights, tmp_path):
    model = model_for(weights).eval()
    spec = FitSpec(purpose="f9", seed=20261011, outer_fold="final", fitting_patients=("p1",), selection_fold=None,
                   candidate=Candidate("layer3+layer4", 0.0001), refit_epochs=3)
    folder = tmp_path / spec.fit_id
    folder.mkdir()
    torch.save(cpu_state(model), folder / "model.pt")
    (folder / "record.json").write_text(json.dumps({"fit_id": spec.fit_id, "spec": spec.to_dict(),
                                                    "checkpoint_sha256": file_sha256(folder / "model.pt"),
                                                    "architecture": {"dropout": 0.25}}), encoding="utf-8")
    loaded = load_f9(folder)
    batch = collate_patients([patient(2, 2, seed=8)])
    with torch.no_grad():
        assert torch.equal(loaded.forward_batch(batch)[0], model.forward_batch(batch)[0]) and not loaded.training
    (folder / "model.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        load_f9(folder)
