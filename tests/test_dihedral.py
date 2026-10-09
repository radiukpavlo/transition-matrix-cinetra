"""In-plane D4 symmetrization of D20 and F9 (SPECIFICATION.md 6.0.1, Sections 8.4 and 8.6; owner issue O1).

Synthetic tensors only.  The group and configuration tests run on the host; the torch tests (TTA invariance of a tiny
random F9 and of the D20 path, augmentation determinism and draw order) run in the GPU image with CUDA hidden."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from tmmri.config import load_study
from tmmri.neural import dihedral as d4
from tmmri.neural.f9_config import load_f9_config

REPO = Path(__file__).resolve().parents[1]
# sections_digest(["deep_encoder"]) of configs/deep_encoder.yaml at 4d3b999, before the amendment-scoped key `tta`.
DEEP_ENCODER_DIGEST_BEFORE_TTA = "1632dc382e935db648d56bae98c6de813df7c0e6d956fbcd73c447de172c1747"


def _pattern(size: int = 5) -> np.ndarray:
    return np.arange(size * size, dtype=np.float64).reshape(size, size)       # no symmetry: the 8 images all differ


def _element_of(image: np.ndarray, base: np.ndarray) -> int:
    hits = [g for g in d4.D4_ELEMENTS if np.array_equal(d4.d4_apply(base, g), image)]
    assert len(hits) == 1, hits
    return hits[0]


# ----------------------------------------------------------------------------- the group (host)


def test_d4_closure_identity_and_inverses():
    base = _pattern()
    images = [d4.d4_apply(base, g) for g in d4.D4_ELEMENTS]
    assert len({im.tobytes() for im in images}) == 8 and np.array_equal(images[0], base)
    table = np.array([[_element_of(d4.d4_apply(d4.d4_apply(base, h), g), base) for h in d4.D4_ELEMENTS] for g in d4.D4_ELEMENTS])
    for row in table:                                                   # closure: a Latin square over the eight elements
        assert sorted(row) == list(d4.D4_ELEMENTS)
    for col in table.T:
        assert sorted(col) == list(d4.D4_ELEMENTS)
    assert all(0 in table[g] for g in d4.D4_ELEMENTS)                   # every element has an inverse
    batch = np.stack([base, base + 100.0])[None]                        # leading axes ride along
    for g in d4.D4_ELEMENTS:
        out = d4.d4_apply(batch, g)
        assert out.shape == batch.shape and np.array_equal(out[0, 1], d4.d4_apply(base + 100.0, g))
    for g in range(4, 8):                                               # the four reflections are involutions
        assert np.array_equal(d4.d4_apply(d4.d4_apply(base, g), g), base)


def test_d4_settings_and_refusals():
    assert d4.tta_elements("d4_mean_8") == tuple(range(8)) and d4.tta_elements("none") == (0,)
    with pytest.raises(ValueError):
        d4.tta_elements("d4_mean_4")
    with pytest.raises(ValueError, match="square"):
        d4.d4_apply(np.zeros((4, 5)), 1)
    with pytest.raises(ValueError):
        d4.d4_apply(np.zeros((4, 4)), 8)
    assert d4.uses_d4_augmentation({"dihedral": "d4_uniform_per_patient"}) and not d4.uses_d4_augmentation({})
    with pytest.raises(ValueError):
        d4.uses_d4_augmentation({"dihedral": "d4_some"})


def test_configs_declare_the_code_settings():
    f9 = load_f9_config()
    assert f9["augmentation"]["dihedral"] == d4.D4_AUGMENTATION and f9["inference"]["tta"] == d4.D4_TTA
    study = load_study()
    assert study.configs["deep_encoder"]["tta"] == d4.D4_TTA


def test_deep_encoder_tta_is_amendment_scoped(tmp_path):
    """The v6.0.0 whole-file digest of deep_encoder is unchanged by the 6.0.1 key; the D20 stages hash it explicitly."""

    from tmmri.neural.image_stages import STAGE_SECTIONS

    study = load_study()
    digest = study.sections_digest(["deep_encoder"])["deep_encoder"]
    assert digest == DEEP_ENCODER_DIGEST_BEFORE_TTA
    root = tmp_path / "study"
    shutil.copytree(REPO / "configs", root / "configs", ignore=shutil.ignore_patterns("clinical", "legacy"))
    (root / "registry").mkdir()
    for path in (REPO / "registry").glob("*.*"):
        shutil.copy2(path, root / "registry" / path.name)
    path = root / "configs" / "deep_encoder.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data.pop("tta") == d4.D4_TTA
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    without = load_study(root / "configs" / "study_v6.yaml")
    assert without.sections_digest(["deep_encoder"])["deep_encoder"] == digest
    data["tta"] = "none"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    other = load_study(root / "configs" / "study_v6.yaml")
    assert other.sections_digest(["deep_encoder"])["deep_encoder"] == digest
    assert other.sections_hash(STAGE_SECTIONS["extract-deep-features"]) != study.sections_hash(STAGE_SECTIONS["extract-deep-features"])
    assert all("deep_encoder.tta" in STAGE_SECTIONS[s] for s in ("extract-deep-features", "fit-deep-encoder"))


# ----------------------------------------------------------------------------- torch (GPU image, CPU)


def _torch():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from tmmri.neural.numerics import configure_numerics

    configure_numerics()
    return torch


def test_torch_and_numpy_elements_agree():
    torch = _torch()
    base = _pattern(6)
    for g in d4.D4_ELEMENTS:
        out = d4.d4_apply(torch.from_numpy(base)[None, None], g)
        assert out.is_contiguous() and np.array_equal(out[0, 0].numpy(), d4.d4_apply(base, g))


def _patient(n_ed: int, n_es: int, size: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {"sample_id": f"p{seed}", "ed": rng.random((n_ed, size, size), dtype=np.float32), "es": rng.random((n_es, size, size), dtype=np.float32),
            "ed_geom": rng.normal(size=(n_ed, 2)).astype(np.float32), "es_geom": rng.normal(size=(n_es, 2)).astype(np.float32)}


def test_f9_tta_is_invariant_to_d4_transforms_of_the_input(tmp_path):
    torch = _torch()
    from tmmri.neural.data import collate_patients
    from tmmri.neural.fit_spec import Candidate
    from tmmri.neural.model import build_model
    from tmmri.neural.synthetic_cache import write_random_resnet18
    from tmmri.neural.train import predict_logits

    path, sha = write_random_resnet18(tmp_path / "resnet18_random.pt")
    model = build_model(Candidate("layer3+layer4", 0.0001), path, seed=20261011, expected_sha256=sha).eval()
    items = [_patient(3, 2, 32, seed=1), _patient(2, 4, 32, seed=2)]
    batch = collate_patients(items)
    with torch.no_grad():
        logits, acts = d4.f9_forward_tta(model, batch)
        assert logits.dtype == torch.float64 and tuple(logits.shape) == (2, 5) and tuple(acts.shape) == (2, 64)
        plain, _ = d4.f9_forward_tta(model, batch, tta="none")
        moved_plain = []
        for h in d4.D4_ELEMENTS:
            moved = {**batch, "ed": d4.d4_apply(batch["ed"], h), "es": d4.d4_apply(batch["es"], h)}
            logits_h, acts_h = d4.f9_forward_tta(model, moved)
            assert float((logits_h - logits).abs().max()) <= 1e-6 and float((acts_h - acts).abs().max()) <= 1e-6, h
            moved_plain.append(float((d4.f9_forward_tta(model, moved, tta="none")[0] - plain).abs().max()))
    assert max(moved_plain) > 1e-5                                       # the control: without TTA the output changes
    # The trainer's evaluation helper is the same symmetrization (validation, selection, cross-hardware QC).
    val_logits, val_acts = predict_logits(model, items, torch.device("cpu"))
    np.testing.assert_allclose(val_logits, logits.numpy().astype(np.float32), rtol=0, atol=1e-6)
    moved_items = [{**it, "ed": d4.d4_apply(it["ed"], 5), "es": d4.d4_apply(it["es"], 5)} for it in items]
    np.testing.assert_allclose(predict_logits(model, moved_items, torch.device("cpu"))[1], val_acts, rtol=0, atol=1e-6)


def test_d20_tta_is_invariant_to_d4_transforms_of_the_input():
    torch = _torch()
    import torchvision

    from tmmri import deep_encoder as de
    from tmmri.neural import preprocess as pp

    torch.manual_seed(20261003)
    encoder = de.build_d20_encoder(model=torchvision.models.resnet18(weights=None))      # tiny stand-in: 512-d penultimate
    cpu = torch.device("cpu")
    rng = np.random.default_rng(3)
    vol = rng.normal(100.0, 20.0, size=(32, 32, 3)).astype(np.float32)

    def case(volume: np.ndarray, sid: str) -> pp.CaseVolumes:
        geom = pp.geometry_from_parameters(volume.shape, (1.25, 1.25), 8.0, np.eye(3))
        return pp.CaseVolumes(sample_id=sid, ed_image=volume, es_image=volume[..., :2] * 0.7 + 3.0, ed_geometry=geom,
                              es_geometry=pp.geometry_from_parameters(volume.shape[:2] + (2,), (1.25, 1.25), 8.0, np.eye(3)),
                              cluster_id=sid, dataset="SYN", case_id=sid.split(":")[1])

    x, _ = de.d20_input_tensor(case(vol, "SYN:a"), size=(32, 32))
    reference = de.patient_vector(de.encode_slices_tta(encoder, x, device=cpu, batch_size=32))
    plain = de.patient_vector(de.encode_slices(encoder, x, device=cpu, batch_size=32))
    controls = []
    for h in d4.D4_ELEMENTS:
        moved = de.patient_vector(de.encode_slices_tta(encoder, d4.d4_apply(x, h), device=cpu, batch_size=32))
        np.testing.assert_allclose(moved, reference, rtol=0, atol=1e-6)
        controls.append(float(np.abs(de.patient_vector(de.encode_slices(encoder, d4.d4_apply(x, h), device=cpu, batch_size=32)) - plain).max()))
    assert max(controls) > 1e-5
    # End to end from the volumes: an in-plane D4 transform of the stored voxels gives the same feature row.
    cases = [case(vol, "SYN:a")] + [case(np.ascontiguousarray(d4.d4_apply(np.moveaxis(vol, 2, 0), h).transpose(1, 2, 0)), f"SYN:h{h}")
                                    for h in (1, 4, 6)]
    features, failures = de.extract_d20_features(cases, encoder, device=cpu, partition="acdc_dev", batch_size=32, dimension=512,
                                                 input_kwargs={"size": (32, 32)})
    assert failures.empty
    a = features.set_index("sample_id")[[c for c in features.columns if c.startswith("a_")]].to_numpy(dtype=np.float64)
    np.testing.assert_allclose(a, np.repeat(a[:1], len(a), axis=0), rtol=0, atol=1e-6)
    np.testing.assert_allclose(a[0], reference, rtol=0, atol=1e-6)


def test_augmentation_d4_draw_is_seeded_first_and_shared_by_phases():
    torch = _torch()
    from tmmri.neural.data import augment_batch, collate_patients

    cfg = dict(load_f9_config()["augmentation"])
    assert cfg["dihedral"] == d4.D4_AUGMENTATION
    quiet = {**cfg, "rotation_degrees": [0.0, 0.0], "translation_mm_each_axis": [0.0, 0.0], "intensity_multiplier": [1.0, 1.0], "noise_sd": 0.0}
    items = [_patient(3, 2, 16, seed=k) for k in range(6)]
    batch = collate_patients(items)
    out = augment_batch(batch, np.random.default_rng(11), quiet, spacing_mm=1.5)
    rng = np.random.default_rng(11)                                     # the documented per-patient draw order
    for i, item in enumerate(items):
        g = int(rng.integers(8))
        for _ in range(4):                                              # angle, x and y translation, intensity
            rng.uniform(0.0, 1.0)
        rng.standard_normal((item["ed"].shape[0], 16, 16), dtype=np.float32)
        rng.standard_normal((item["es"].shape[0], 16, 16), dtype=np.float32)
        for phase in ("ed", "es"):
            n = item[phase].shape[0]
            expected = d4.d4_apply(torch.from_numpy(item[phase]), g)
            torch.testing.assert_close(out[phase][i, :n], expected, rtol=0, atol=1e-5)
            assert float(out[phase][i, n:].abs().sum()) == 0.0         # padding stays zero
    again = augment_batch(batch, np.random.default_rng(11), cfg, spacing_mm=1.5)
    twice = augment_batch(batch, np.random.default_rng(11), cfg, spacing_mm=1.5)
    other = augment_batch(batch, np.random.default_rng(12), cfg, spacing_mm=1.5)
    assert torch.equal(again["ed"], twice["ed"]) and torch.equal(again["es"], twice["es"]) and not torch.equal(again["ed"], other["ed"])
    assert torch.equal(again["ed_geom"], batch["ed_geom"]) and torch.equal(again["es_mask"], batch["es_mask"])   # geometry unchanged
    no_d4 = augment_batch(batch, np.random.default_rng(11), {k: v for k, v in quiet.items() if k != "dihedral"}, spacing_mm=1.5)
    torch.testing.assert_close(no_d4["ed"], batch["ed"], rtol=0, atol=1e-5)  # without the key: the 6.0.0 stream, no D4 draw
