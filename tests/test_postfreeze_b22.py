"""B22 of the post-freeze block (protocol-v6.0.2): F9-M (F9 with three reference-mask channels) and F9b-M on synthetic
tensors and a tiny synthetic source run only (tmp_path; never cohort data).  CPU, random-initialization weights; runs
in the GPU image."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import yaml

from tmmri import stages
from tmmri.config import load_study
from tmmri.experiments.postfreeze import b22_f9m as b22
from tmmri.neural import f9m
from tmmri.neural import preprocess as pp
from tmmri.neural.data import augment_batch, collate_patients, load_npz, read_manifest
from tmmri.neural.f9_config import load_f9_config, training_config
from tmmri.neural.fit_spec import CANDIDATES, PRIMARY_SEED, Candidate, FitSpec
from tmmri.neural.model import build_model
from tmmri.neural.numerics import configure_numerics
from tmmri.neural.synthetic_cache import write_random_resnet18, write_synthetic_cache
from tmmri.neural.train import WeightsRef, train_fit
from tmmri.paths import PROJECT_ROOT
from tmmri.runs import atomic_write_json, sha256_json

CLASSES = ["DCM", "HCM", "MINF", "NOR", "RV"]
SEEDS = (20261011, 20261012, 20261013)
NI_CFG = pp.load_neural_input_config()


def tiny_config() -> dict:
    return copy.deepcopy(training_config(load_f9_config()))


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    configure_numerics()
    path, sha = write_random_resnet18(tmp_path_factory.mktemp("weights") / "resnet18_random.pt")
    return WeightsRef(path=str(path), sha256=sha, name="synthetic_random_init")


def _batch(n_patients=2, size=32, seed=0, *, slices=(3, 2)) -> list[dict]:
    rng = np.random.default_rng(seed)
    items = []
    for p in range(n_patients):
        item = {"sample_id": f"p{p}"}
        for phase, n in zip(("ed", "es"), (slices[0] + p, slices[1] + p)):
            item[phase] = rng.random((n, size, size), dtype=np.float32)
            item[f"{phase}_geom"] = rng.normal(size=(n, 2)).astype(np.float32)
            item[f"{phase}_labels"] = rng.integers(0, 4, size=(n, size, size)).astype(np.uint8)
        items.append(item)
    return items


def _pair(weights, candidate=CANDIDATES[3], seed=20261011):
    f9 = build_model(candidate, weights.path, seed=seed, expected_sha256=weights.sha256)
    m = f9m.build_f9m(candidate, weights.path, seed=seed, expected_sha256=weights.sha256, mask_channels=True)
    off = f9m.build_f9m(candidate, weights.path, seed=seed, expected_sha256=weights.sha256, mask_channels=False)
    return f9.eval(), m.eval(), off.eval()


# ----------------------------------------------------------------------------- network


def test_mask_path_off_and_zero_stem_reproduce_f9_exactly(weights):
    f9, masked, off = _pair(weights)
    reference = f9.state_dict()
    assert set(off.state_dict()) == set(reference) and all(torch.equal(off.state_dict()[k], v) for k, v in reference.items())
    assert set(masked.state_dict()) == set(reference) | {f9m.MASK_STEM_KEY}
    assert all(torch.equal(masked.state_dict()[k], v) for k, v in reference.items())   # the stem draws nothing from the RNG
    assert not masked.mask_stem.weight.any()
    batch = f9m.collate_with_masks(_batch())
    assert batch["ed_seg"].sum() > 0                                                    # nonzero masks
    with torch.no_grad():
        ref = f9.forward_batch(batch)
        for model in (off, masked):
            out = model.forward_batch(batch)
            assert torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1])
        tta_ref = f9m.f9m_forward_tta(off, batch)
        tta_m = f9m.f9m_forward_tta(masked, batch)
    from tmmri.neural.dihedral import f9_forward_tta

    tta_f9 = f9_forward_tta(f9, batch)
    assert torch.equal(tta_ref[0], tta_f9[0]) and torch.equal(tta_m[0], tta_f9[0]) and torch.equal(tta_m[1], tta_f9[1])
    with pytest.raises(ValueError, match="mask channels"):
        masked(batch["ed"], batch["es"], batch["ed_geom"], batch["es_geom"], batch["ed_mask"], batch["es_mask"])


def test_mask_stem_trains_in_the_new_layer_group_and_conv1_stays_frozen(weights):
    _, masked, _ = _pair(weights)
    groups = masked.parameter_groups(1e-4, 10.0)
    new = next(g for g in groups if g["name"] == "new_layers_and_head")
    assert new["lr"] == pytest.approx(1e-3) and any(p is masked.mask_stem.weight for p in new["params"])
    assert not any(p is masked.mask_stem.weight for p in groups[0]["params"])
    assert not masked.backbone.conv1.weight.requires_grad and masked.mask_stem.weight.requires_grad
    masked.train()
    logits, _ = masked.forward_batch(f9m.collate_with_masks(_batch(seed=3)))
    logits.sum().backward()
    assert masked.mask_stem.weight.grad is not None and masked.mask_stem.weight.grad.abs().max() > 0
    assert masked.backbone.conv1.weight.grad is None


# ----------------------------------------------------------------------------- mask grid


def _label_volume(shape, spacing, *, shift=0.0):
    nx, ny, nz = shape
    x = (np.arange(nx) - (nx - 1) / 2.0) * spacing
    y = (np.arange(ny) - (ny - 1) / 2.0) * spacing
    xx, yy = np.meshgrid(x, y, indexing="ij")
    vol = np.zeros(shape, dtype=np.int16)
    for k in range(nz):
        r = np.hypot(xx - 3.0 - shift, yy + 2.0)
        vol[..., k][np.hypot(xx + 9.0, yy - 4.0) <= 6.0 + k] = 3          # RV cavity
        vol[..., k][r <= 9.0 + k] = 2                                       # myocardium
        vol[..., k][r <= 5.0 + k] = 1                                       # LV cavity
    return vol


def _case_from_labels(sample_id, labels_ed, labels_es, spacing, *, directions=None, label="NOR", increment=8.0):
    geom = pp.geometry_from_parameters(labels_ed.shape, (spacing, spacing), increment, directions, cfg=NI_CFG)
    levels = np.array([10.0, 160.0, 60.0, 110.0], dtype=np.float32)      # background, LV, myocardium, RV -> 0, 1, 1/3, 2/3
    return pp.CaseVolumes(sample_id=sample_id, ed_image=levels[labels_ed], es_image=levels[labels_es], ed_geometry=geom, es_geometry=geom,
                          cluster_id=sample_id, label=label, dataset="ACDC", case_id=sample_id.split(":")[1])


@pytest.mark.parametrize("directions", [None, ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0))])   # transpose+flip; flip only
def test_mask_labels_land_on_the_image_grid(directions):
    cfg = copy.deepcopy(NI_CFG)
    cfg["input"]["size_pixels"] = [64, 64]
    labels = _label_volume((40, 36, 3), 1.5)                  # source spacing = target spacing, (n - size) even: exact pixels
    case = _case_from_labels("ACDC:patient900", labels, labels, 1.5, directions=directions)
    image = pp.preprocess_case(case, cfg).arrays["ed"]
    out = f9m.mask_label_slices(case, {"ed": labels, "es": labels}, cfg)
    assert out["ed_labels"].shape == image.shape and out["ed_labels"].dtype == np.uint8
    decoded = np.array([0, 2, 3, 1], dtype=np.uint8)[np.rint(image * 3.0).astype(int)]    # scaled level -> label; padding -> 0
    np.testing.assert_array_equal(decoded, out["ed_labels"])
    assert set(np.unique(out["ed_labels"])) == {0, 1, 2, 3} and (out["ed_labels"][:, 0, :] == 0).all()   # zero padding outside the FOV


def test_mask_labels_follow_the_image_on_a_non_integer_grid():
    labels = _label_volume((48, 44, 2), 1.25)
    case = _case_from_labels("ACDC:patient901", labels, labels, 1.25)
    image = pp.preprocess_case(case, NI_CFG).arrays["ed"]
    out = f9m.mask_label_slices(case, {"ed": labels, "es": labels}, NI_CFG)["ed_labels"]
    lv_image = image > 2.0 / 3.0                                # halfway between myocardium (1/3) and LV cavity (1)
    lv_mask = out == 1
    iou = (lv_image & lv_mask).sum() / (lv_image | lv_mask).sum()
    assert iou > 0.85


def test_mask_label_slices_refuse_a_mask_of_another_shape():
    labels = _label_volume((20, 20, 2), 1.5)
    case = _case_from_labels("ACDC:patient902", labels, labels, 1.5)
    with pytest.raises(pp.InputRejected):
        f9m.mask_label_slices(case, {"ed": labels[:, :, :1], "es": labels}, NI_CFG)


# ----------------------------------------------------------------------------- augmentation


def test_mask_augmentation_keeps_f9s_image_stream_and_moves_masks_with_the_images():
    cfg = copy.deepcopy(tiny_config()["augmentation"])
    items = _batch(n_patients=4, seed=7)
    for item in items:                                         # binary images whose labels are the images themselves
        for phase in ("ed", "es"):
            item[phase] = (item[f"{phase}_labels"] == 1).astype(np.float32)
    batch = f9m.collate_with_masks(items)
    plain = collate_patients(items)
    rng_a, rng_b = np.random.default_rng([5, 1]), np.random.default_rng([5, 1])
    ref = augment_batch(plain, rng_a, cfg, spacing_mm=1.5)
    out = f9m.augment_with_masks(batch, rng_b, cfg, spacing_mm=1.5)
    assert torch.equal(ref["ed"], out["ed"]) and torch.equal(ref["es"], out["es"])
    assert rng_a.bit_generator.state == rng_b.bit_generator.state
    rigid = {**cfg, "rotation_degrees": [0.0, 0.0], "translation_mm_each_axis": [0.0, 0.0], "intensity_multiplier": [1.0, 1.0], "noise_sd": 0.0}
    out = f9m.augment_with_masks(batch, np.random.default_rng([5, 2]), rigid, spacing_mm=1.5)
    for phase in ("ed", "es"):                                  # D4 only: exact permutations of both
        torch.testing.assert_close(out[f"{phase}_seg"][:, :, 0], (out[phase] > 0.5).float() * batch[f"{phase}_mask"].float()[..., None, None],
                                   rtol=0, atol=0)
    moved = f9m.augment_with_masks(batch, np.random.default_rng([5, 3]), {**cfg, "noise_sd": 0.0, "intensity_multiplier": [1.0, 1.0]}, spacing_mm=1.5)
    for phase in ("ed", "es"):                                  # rotation and translation: masks follow the images
        a, b = moved[f"{phase}_seg"][:, :, 0] > 0.5, moved[phase] > 0.5
        assert (a & b).sum() / (a | b).sum() > 0.8
        assert set(torch.unique(moved[f"{phase}_seg"]).tolist()) <= {0.0, 1.0}


# ----------------------------------------------------------------------------- refits


def _labelled_items(manifest_path: Path) -> tuple[dict, dict]:
    manifest = read_manifest(manifest_path)
    items, labels = {}, {}
    for row in manifest.to_dict("records"):
        item = load_npz(manifest_path.parent / f"{row['sample_id']}.npz")
        for phase in ("ed", "es"):
            item[f"{phase}_labels"] = np.where(item[phase] > 0.5, 1, np.where(item[phase] > 0.25, 2, 0)).astype(np.uint8)
        items[row["sample_id"]], labels[row["sample_id"]] = item, row["label"]
    return items, labels


def test_refit_trainer_reproduces_train_fit_with_the_mask_path_off(tmp_path, weights):
    manifest = write_synthetic_cache(tmp_path / "cache", n_per_class=3, slices=(2, 3), size=32)
    items, labels = _labelled_items(manifest)
    spec = FitSpec(purpose="f9", seed=20261012, outer_fold=3, fitting_patients=tuple(sorted(items)), selection_fold=None,
                   candidate=CANDIDATES[4], refit_epochs=2)
    ref = train_fit(spec, tmp_path / "cache", tmp_path / "f9", "cpu", config=tiny_config(), weights=weights)
    rec = f9m.train_refit(spec, items, labels, tmp_path / "off", "cpu", config=tiny_config(), weights=weights, mask_channels=False)
    assert rec.state_sha256 == ref.state_sha256 and rec.class_weights == ref.class_weights
    assert [e["train_loss"] for e in rec.epochs] == [e["train_loss"] for e in ref.epochs]


def test_masked_refit_trains_the_stem_resumes_bitwise_and_loads(tmp_path, weights):
    manifest = write_synthetic_cache(tmp_path / "cache", n_per_class=3, slices=(2, 3), size=32)
    items, labels = _labelled_items(manifest)
    spec = FitSpec(purpose="f9", seed=PRIMARY_SEED, outer_fold="final", fitting_patients=tuple(sorted(items)), selection_fold=None,
                   candidate=Candidate("layer4", 0.0001), refit_epochs=2)
    full = f9m.train_refit(spec, items, labels, tmp_path / "full", "cpu", config=tiny_config(), weights=weights)
    assert full.status == "COMPLETE" and full.parameters["mask_stem_trained"] and full.architecture["mask_channels"]
    assert full.parameters["frozen_weights_unchanged"] and full.parameters["bn_running_statistics_unchanged"]
    assert not (tmp_path / "full" / "resume").exists()
    part = f9m.train_refit(spec, items, labels, tmp_path / "resumed", "cpu", config=tiny_config(), weights=weights, stop_after_epoch=1)
    assert part.status == "INTERRUPTED"
    resumed = f9m.train_refit(spec, items, labels, tmp_path / "resumed", "cpu", config=tiny_config(), weights=weights)
    assert resumed.state_sha256 == full.state_sha256 and resumed.resume["resumed_from_epoch"] == 2
    again = f9m.train_refit(spec, items, labels, tmp_path / "full", "cpu", config=tiny_config(), weights=weights)
    assert again.timings == full.timings                                    # completed refits are returned, never retrained
    model = f9m.load_f9m(tmp_path / "full")
    logits, acts = f9m.predict_patients(model, [items[s] for s in sorted(items)[:3]], torch.device("cpu"))
    assert logits.shape == (3, 5) and acts.shape == (3, 64) and np.isfinite(logits).all()
    with pytest.raises(ValueError, match="selection fit"):
        f9m.train_refit(FitSpec(purpose="f9", seed=PRIMARY_SEED, outer_fold=1, fitting_patients=spec.fitting_patients, selection_fold=1,
                                candidate=CANDIDATES[0], refit_epochs=None), items, labels, tmp_path / "sel", "cpu", config=tiny_config(),
                        weights=weights)


# ----------------------------------------------------------------------------- synthetic source run and run()


class FakeContext:
    """The PostfreezeContext contract (src/tmmri/experiments/postfreeze/common.py) on a synthetic source run: relative or
    absolute ``verified`` paths, ``write_provenance(analysis, {label: Path}, [existing Paths], summary)`` refusing an
    existing ``provenance.json``, and ``require_new_output`` raising ``BundleError``."""

    def __init__(self, source_root: Path, run_root: Path, study, lane: dict) -> None:
        self.source_root, self.run_root, self.staged_data_root, self.study, self.lane_cfg = source_root, run_root, run_root / "data", study, lane
        self.verified_calls: list[tuple[str, str]] = []
        self.provenance: dict | None = None

    def analysis_cfg(self, analysis: str) -> dict:
        return dict(self.lane_cfg["analyses"][analysis])

    def out_dir(self, analysis: str) -> Path:
        path = self.run_root / "postfreeze" / analysis
        path.mkdir(parents=True, exist_ok=True)
        return path

    def verified(self, path, stage) -> Path:
        full = Path(path) if Path(path).is_absolute() else self.source_root / path
        assert full.is_file(), path
        self.verified_calls.append((Path(path).as_posix(), stage))
        return full

    def write_provenance(self, analysis, inputs, outputs, summary) -> Path:
        path = self.out_dir(analysis) / "provenance.json"
        self.require_new_output(path)
        assert all(isinstance(k, str) and Path(v).is_file() for k, v in inputs.items())
        missing = [str(p) for p in outputs if not Path(p).exists()]
        if missing:
            raise RuntimeError(f"declared outputs missing {missing}")
        self.provenance = {"analysis": analysis, "inputs": {k: str(v) for k, v in inputs.items()}, "outputs": [str(p) for p in outputs],
                           "summary": summary}
        atomic_write_json(path, self.provenance, allow_nan=True)
        return path

    def require_new_output(self, path: Path) -> None:
        if Path(path).exists():
            raise stages.BundleError(f"{path} exists; a repeat needs a new run id")


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


@pytest.fixture()
def source_run(tmp_path, weights):
    study = load_study()
    small = copy.deepcopy(NI_CFG)
    small["input"]["size_pixels"] = [32, 32]
    study.configs["neural_input"] = small
    lane = yaml.safe_load((PROJECT_ROOT / "configs" / "exploratory_postfreeze.yaml").read_text(encoding="utf-8"))
    lane["policy"]["evaluated_partitions"] = ["acdc_test"]
    lane["statistics"]["bootstrap"]["replicates"] = 30
    src = tmp_path / "source"
    rng = np.random.default_rng(20260613)
    dev = [f"ACDC:patient{i:03d}" for i in range(1, 21)]
    test = [f"ACDC:patient{i:03d}" for i in range(101, 106)]
    cls = {s: CLASSES[i // 4] for i, s in enumerate(dev)} | {s: CLASSES[i] for i, s in enumerate(test)}
    fold = {s: i % 5 + 1 for i, s in enumerate(dev)}
    allids = dev + test
    _write(pd.DataFrame({"sample_id": allids, "dataset": "ACDC", "case_id": [s.split(":")[1] for s in allids],
                         "partition": ["acdc_dev"] * len(dev) + ["acdc_test"] * len(test),
                         "outer_fold": pd.array([fold[s] for s in dev] + [pd.NA] * len(test), dtype="Int64"), "cluster_id": allids}),
           src / "contracts" / "split_manifest.parquet")
    inner = [{"outer_fold_heldout": f, "sample_id": s, "inner_fold": j % 4 + 1}
             for f in range(1, 6) for j, s in enumerate([s for s in dev if fold[s] != f])]
    _write(pd.DataFrame(inner), src / "contracts" / "inner_folds.parquet")
    _write(pd.DataFrame({"sample_id": allids, "acdc_class": [cls[s] for s in allids]}), src / "contracts" / "cohort_ACDC.parquet")
    targets = study.transition_targets()
    latent = rng.normal(size=(len(allids), 4)) + np.array([[CLASSES.index(cls[s])] for s in allids])
    b = pd.DataFrame(50.0 + 10.0 * np.tanh(latent @ rng.normal(size=(4, len(targets))) / 3.0), columns=targets)
    b.insert(0, "sample_id", allids)
    b["qc_failure_reason"] = None
    _write(b, src / "semantics" / "B_ACDC.parquet")
    _write(b.iloc[:len(dev)][["sample_id", *targets]].assign(**{t: b[t].iloc[:len(dev)] + 1.0 for t in targets}), src / "oof" / "bhat_oof_ACDC_dev.parquet")
    _write(b.iloc[len(dev):][["sample_id", *targets]].assign(**{t: b[t].iloc[len(dev):] - 1.0 for t in targets}),
           src / "reconstructed" / "bhat_acdc_test.parquet")
    cases = {}
    for i, s in enumerate(allids):
        k = CLASSES.index(cls[s])
        ed = _label_volume((24, 22, 3), 1.5, shift=0.6 * k)
        es = _label_volume((24, 22, 3), 1.5, shift=-0.6 * k)
        cases[s] = (_case_from_labels(s, ed, es, 1.5, label=cls[s]), {"ed": ed, "es": es})
    for part, ids in (("acdc_dev", dev), ("acdc_test", test)):
        result = pp.build_tensor_cache_from_volumes([cases[s][0] for s in ids], src / "neural_cache" / part, part, small)
        assert result["failures"].empty
    cfg = tiny_config()
    winners, fits = [], []
    for seed in SEEDS:
        for j, outer in enumerate([1, 2, 3, 4, 5, "final"]):
            patients = tuple(s for s in dev if outer == "final" or fold[s] != outer)
            spec = FitSpec(purpose="f9", seed=seed, outer_fold=outer, fitting_patients=patients, selection_fold=None,
                           candidate=CANDIDATES[j], refit_epochs=1)
            atomic_write_json(src / "neural_f9" / "checkpoints" / spec.fit_id / "record.json",
                              {"fit_id": spec.fit_id, "spec": spec.to_dict(), "config_sha256": sha256_json(cfg)})
            winners.append({"outer_fold": spec.outer_fold, "seed": seed, "candidate": spec.candidate.label, "refit_epochs": 1, "fit_id": spec.fit_id})
            fits.append({"fit_id": spec.fit_id, "seed": seed, "outer_fold": spec.outer_fold, "candidate": spec.candidate.to_dict(), "refit_epochs": 1})
    atomic_write_json(src / "neural_f9" / "selection" / "selection_summary.json", {"groups": 18, "refits": 18, "not_estimable": [], "winners": winners})
    atomic_write_json(src / "neural_f9" / "f9_bundle.json", {"schema": "tmmri.f9_bundle/1", "fits": fits, "weights": {"sha256": weights.sha256}})
    ctx = FakeContext(src, tmp_path / "artifacts" / "v6-postfreeze-20261028-0900", study, lane)
    return ctx, cases, cfg


def test_refit_plan_checks_every_recorded_winner(source_run):
    ctx, _, cfg = source_run
    plan, meta = b22.refit_plan(ctx, cfg)
    assert len(plan) == 18 and meta["total_refit_epochs"] == 18 and meta["seeds"] == list(SEEDS)
    assert [p["encoder_id"] for p in plan[:6]] == ["fold1", "fold2", "fold3", "fold4", "fold5", "final"]
    b22.check_fitting_sets(plan, {s: int(f) for s, f in pd.read_parquet(ctx.source_root / "contracts" / "split_manifest.parquet")
                                  .dropna(subset=["outer_fold"])[["sample_id", "outer_fold"]].itertuples(index=False)})
    other = copy.deepcopy(cfg)
    other["gradient_clip_norm"] = 1.0
    with pytest.raises(ValueError, match="config_sha256"):
        b22.refit_plan(ctx, other)
    path = ctx.source_root / "neural_f9" / "selection" / "selection_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["winners"][0]["refit_epochs"] = 2
    atomic_write_json(path, summary)
    with pytest.raises(ValueError, match="refit_epochs"):
        b22.refit_plan(ctx, cfg)


def test_run_end_to_end_on_a_synthetic_source_run(source_run, weights):
    ctx, cases, cfg = source_run
    summary = b22.run(ctx, device="cpu", loader=lambda row: cases[row["sample_id"]], config=cfg, weights=weights)
    out = ctx.run_root / "postfreeze" / "B22"
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert summary["completed_seeds"] == list(SEEDS) and summary["not_executed_seeds"] == []
    assert results["label"] == "EXPLORATORY_POST_HOC" and results["model_id"] == "f9m"
    assert results["inputs"]["acdc_dev"]["image_alignment_max_abs_difference"] <= b22.ALIGNMENT_TOLERANCE
    for seed in SEEDS:
        per = results["per_seed"][str(seed)]
        assert set(per) == {"acdc_dev_oof", "acdc_test"}
        assert {"rho", "rho_ef", "rho_volumes", "paired_vs_primary_ridge"} <= set(per["acdc_dev_oof"]["representation"])
        assert {"accuracy", "macro_f1", "macro_ovr_auroc"} <= set(per["acdc_test"]["predictor"]["metrics"])
        assert per["acdc_test"]["predictor"]["chance_and_majority_rates"]["chance"] == pytest.approx(0.2)
        gate = results["validity_gate"][str(seed)]
        assert gate["oof_rows_complete"] and gate["mask_stem_trained"] and isinstance(gate["passed"], bool)
        assert (out / "probes" / f"seed{seed}" / "FROZEN.json").is_file()
    assert set(results["seed_mean"]) == {"acdc_dev_oof", "acdc_test"} and results["seed_mean"]["acdc_test"]["macro_f1"]["n_seeds"] in (0, 3)
    frozen = json.loads((out / "fits" / "FROZEN.json").read_text(encoding="utf-8"))
    assert len(frozen["files"]) == 36 and frozen["label"] == "EXPLORATORY_POST_HOC"
    preds = pd.read_parquet(out / "inference" / "predictions_acdc_dev.parquet")
    assert len(preds) == 3 * 6 * 20 and preds.loc[preds["is_oof"], "sample_id"].value_counts().eq(3).all()
    assert set(pd.read_parquet(out / "inference" / "predictions_acdc_test.parquet")["encoder_id"]) == {"final"}
    verified = {p for p, _ in ctx.verified_calls}
    assert "neural_f9/selection/selection_summary.json" in verified and any(p.endswith(".npz") for p in verified)
    assert (out / "mask_cache" / "mask_manifest_acdc_test.parquet").is_file()
    assert ctx.provenance["analysis"] == "B22" and ctx.provenance["summary"]["label"] == "EXPLORATORY_POST_HOC"
    assert "source:neural_f9/selection/selection_summary.json" in ctx.provenance["inputs"]
    assert f"weights:{weights.name}" in ctx.provenance["inputs"] and "source:neural_cache/acdc_dev/manifest.csv" in ctx.provenance["inputs"]
    with pytest.raises(stages.BundleError):                          # a repeat needs a new run root
        b22.run(ctx, device="cpu", loader=lambda row: cases[row["sample_id"]], config=cfg, weights=weights)


def test_a_failing_seed_is_reported_not_executed(source_run, weights, monkeypatch):
    ctx, cases, cfg = source_run
    real = f9m.train_refit

    def flaky(spec, *args, **kwargs):
        if spec.seed == SEEDS[1] and spec.outer_fold == 2:
            raise RuntimeError("synthetic out-of-memory")
        return real(spec, *args, **kwargs)

    monkeypatch.setattr(f9m, "train_refit", flaky)
    summary = b22.run(ctx, device="cpu", loader=lambda row: cases[row["sample_id"]], config=cfg, weights=weights)
    results = json.loads((ctx.run_root / "postfreeze" / "B22" / "results.json").read_text(encoding="utf-8"))
    assert summary["not_executed_seeds"] == [SEEDS[1]] and summary["completed_seeds"] == [SEEDS[0], SEEDS[2]]
    assert results["seeds"][str(SEEDS[1])]["status"] == b22.NOT_EXECUTED and "synthetic out-of-memory" in results["seeds"][str(SEEDS[1])]["reason"]
    assert str(SEEDS[1]) not in results["per_seed"] and results["seed_mean"]["acdc_dev_oof"]["rho"]["n_seeds"] <= 2


def test_run_refuses_without_cuda_before_any_work(source_run, monkeypatch):
    ctx, cases, cfg = source_run
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="local GPU"):
        b22.run(ctx, loader=lambda row: cases[row["sample_id"]], config=cfg)
    assert not (ctx.run_root / "postfreeze" / "B22" / "fits").exists()
