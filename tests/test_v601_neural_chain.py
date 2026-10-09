"""Protocol-v6.0.1 integration: the cross-lane chain on synthetic fixtures, on the CPU (GPU image; never cohort data).

One frozen synthetic v6 run (ACDC and M&Ms-1 fixtures, ``run --until audit-shift``) feeds, through the CLI:
``build-tensor-cache`` (L4a, contract C3) -> ``plan-f9`` on a tiny F9 configuration -> ``train-f9 --backend local``
-> ``select-f9`` -> the refits -> ``infer-f9 --split acdc_dev`` -> ``freeze-f9`` (L4b, C5/C8; technical-validity gate)
-> ``infer-f9`` (L4a, C4) -> D20 with random-weight
encoders (``extract-deep-features``, ``fit-deep-encoder``) -> ``fit-probes``, ``fit-trust``, ``evaluate-alt-bundles``
and ``evaluate-trust`` on the held-out ``acdc_test`` and the synthetic external ``mms_external`` partition and
``build-comparison-tables`` (L57); then the S5 F9 path after the F9 freeze (L89 with L4b and L4a): ``run-data-efficiency
--step f9-wave1`` -> ``train-f9`` on its purpose-s5 spec file -> ``f9-wave2`` -> ``train-f9`` -> ``f9-infer`` ->
``f9-aggregate``.  Random-weight ResNet files stand in for the pinned ImageNet weights (their SHA-256
is written into the study copy's registry/weights.yaml); everything is written below pytest's tmp_path.
"""
from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

import fixtures_v6 as fx
from test_v6_cli import METHODS_MODULES, _edit_yaml, _run, _study_copy
from tmmri import stages

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")

SEED = 20261011
SIZE = 64          # cache slices of 64 x 64 at 1.5 mm keep the CPU run short (real caches: 224 x 224)


def _random_weights(home) -> dict[str, str]:
    """Random-weight ResNet-18 and ResNet-50 state dicts under the torchvision file names; returns their SHA-256."""

    folder = home / "hub" / "checkpoints"
    folder.mkdir(parents=True)
    torch.manual_seed(20261003)
    out = {}
    for key, ctor, name in (("resnet18_imagenet1k_v1", torchvision.models.resnet18, "resnet18-f37072fd.pth"),
                            ("resnet50_imagenet1k_v2", torchvision.models.resnet50, "resnet50-11ad3fa6.pth")):
        path = folder / name
        torch.save(ctor(weights=None).state_dict(), path)
        out[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _tiny_f9(cfg: dict) -> None:
    cfg["input"]["size_pixels"] = [SIZE, SIZE]
    cfg["seeds"]["values"] = [SEED]
    cfg["epochs"]["max"] = 2
    cfg["schedule"]["t_max_epochs"] = 2
    cfg["early_stopping"].update(min_epoch=1, patience=1)
    cfg["fit_counts"].update(wave1_selection=18, wave2_refit=6, total=24)     # 6 fitting sets x 1 seed x 1 candidate x 3 folds
    cfg["validity_gate"]["min_oof_macro_f1"] = 0.0      # a 2-epoch fit on 25 synthetic patients cannot learn; the other gate checks stay


def _tiny_s5(cfg: dict) -> None:
    section = cfg["data_efficiency"]
    section["repeats"]["below_one"] = 1
    section["f9"]["fractions"] = [0.75]                  # 3 of 4 per class: every selection fold keeps every class
    section["f9"]["candidates"] = section["f9"]["candidates"][:1]


def test_neural_chain_on_a_frozen_synthetic_run(tmp_path, monkeypatch, capsys):
    from conftest import require_module

    for module in METHODS_MODULES:
        require_module(module)
    from tmmri.neural import f9_stages
    from tmmri.neural.fit_spec import candidates_from_config

    monkeypatch.setattr(stages, "protocol_frozen", lambda *a, **k: False)
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))
    # Tiny plan: the configuration keeps the six protocol candidates (validated), the plan uses the first one only.
    monkeypatch.setattr(f9_stages, "candidates_from_config", lambda cfg: candidates_from_config(cfg)[:1])
    shas = _random_weights(tmp_path / "torch")
    acdc = fx.make_acdc_source(tmp_path / "src" / "acdc", fx.acdc_patients(5, testing_per_class=2))
    fx.write_fixture_manifests(acdc, fx.acdc_manifest(acdc))
    cases = fx.mms_refit_cases()
    mms = fx.make_mms_source(tmp_path / "src" / "mms", cases, shape=(26, 24, 3))
    fx.write_fixture_manifests(mms, fx.mms_manifest(mms, cases), fx.mms_metadata(mms))
    study_path = _study_copy(tmp_path)
    configs, registry = study_path.parent, study_path.parent.parent / "registry"
    _edit_yaml(registry / "weights.yaml", lambda d: [d["weights"][k].update(sha256=v) for k, v in shas.items()])
    _edit_yaml(configs / "deep_encoder.yaml", lambda d: (d["encoder"].update(weights_sha256=shas["resnet50_imagenet1k_v2"]),
                                                         d["input"].update(resize=[SIZE, SIZE])))
    _edit_yaml(configs / "neural_input.yaml", lambda d: d["input"].update(size_pixels=[SIZE, SIZE]))
    _edit_yaml(configs / "f9_cnn.yaml", _tiny_f9)
    _edit_yaml(configs / "probes.yaml", lambda d: d["f9b"].update(seeds=[SEED]))
    _edit_yaml(configs / "data_centric.yaml", _tiny_s5)
    run_id = "v6-chain-20261003-1200"
    common = ["--study", str(study_path), "--run-id", run_id, "--artifact-root", str(tmp_path / "artifacts"), "--data-root", str(tmp_path / "data"),
              "--acdc-source", str(acdc), "--mms-source", str(mms), "--n-jobs", "1", "--allow-unfrozen-synthetic"]
    run = tmp_path / "artifacts" / run_id

    def ok(*argv: str) -> dict:
        code, out = _run([*argv, *common], capsys)
        assert code == 0, (argv, out)
        return out

    ok("run", "--until", "audit-shift")
    # C3: the tensor cache of the development, test and synthetic external partitions (':' -> '__' in file names).
    for split in ("acdc_dev", "acdc_test", "mms_external"):
        ok("build-tensor-cache", "--split", split)
    dev_manifest = pd.read_csv(run / "neural_cache" / "acdc_dev" / "manifest.csv", dtype=str, keep_default_na=False)
    assert len(dev_manifest) == 25 and set(dev_manifest["label"]) == {"DCM", "HCM", "MINF", "NOR", "RV"}
    assert all(f == s.replace(":", "__") + ".npz" for f, s in zip(dev_manifest["npz_file"], dev_manifest["sample_id"]))
    assert set(pd.read_csv(run / "neural_cache" / "acdc_test" / "manifest.csv", keep_default_na=False)["label"]) == {""}
    # C5/C8: plan, train (local backend, CPU), select, refit and freeze the tiny F9.
    plan = ok("plan-f9")
    assert plan["action"] == "RAN"
    ok("train-f9", "--backend", "local", "--specs", "neural_f9/plan/wave1_specs.json", "--device", "cpu")
    ok("select-f9")
    ok("train-f9", "--backend", "local", "--specs", "neural_f9/plan/wave2_specs.json", "--device", "cpu")
    # Technical-validity gate of freeze-f9: the development out-of-fold predictions (fold encoders) are recorded first.
    ok("infer-f9", "--split", "acdc_dev", "--device", "cpu")
    ok("freeze-f9")
    bundle = json.loads((run / "neural_f9" / "f9_bundle.json").read_text(encoding="utf-8"))
    assert len(bundle["fits"]) == 6 and {f["encoder_id"] for f in bundle["fits"]} == {"fold1", "fold2", "fold3", "fold4", "fold5", "final"}
    gate = json.loads((run / "neural_f9" / "validity_gate.json").read_text(encoding="utf-8"))
    assert gate["passed"] and bundle["validity_gate"]["passed"] and len(gate["refits"]) == 6
    record = json.loads((run / "neural_f9" / "checkpoints" / bundle["fits"][0]["fit_id"] / "record.json").read_text(encoding="utf-8"))
    assert record["environment"]["device"] == "cpu" and record["status"] == "COMPLETE"
    # C4: local inference of the frozen encoders (fold encoders on acdc_dev, recorded above; the final encoder elsewhere).
    for split in ("acdc_test", "mms_external"):
        ok("infer-f9", "--split", split, "--device", "cpu")
    preds = pd.read_parquet(run / "neural_f9" / "predictions.parquet")
    assert set(preds["partition"]) == {"acdc_dev", "acdc_test", "mms_external"} and set(preds["seed"]) == {SEED}
    dev = preds[preds["partition"] == "acdc_dev"]
    assert dev.groupby("sample_id")["is_oof"].sum().eq(1).all()             # each development patient is out-of-fold exactly once
    assert set(preds.loc[preds["partition"] != "acdc_dev", "encoder_id"]) == {"final"}
    acts = pd.read_parquet(run / "neural_f9" / "activations.parquet")
    assert sum(c.startswith("a_") for c in acts.columns) == 64
    # D20 with random-weight ResNet-50 files (the pinned SHA-256 of the study copy).
    for split in ("acdc_dev", "acdc_test", "mms_external"):
        ok("extract-deep-features", "--split", split, "--device", "cpu")
    ok("fit-deep-encoder")
    assert (run / "deep_encoder" / "FROZEN_D20.json").is_file()
    # L57 on the real L4a/L4b outputs.
    ok("fit-probes", "--seed", str(SEED))
    ok("fit-trust")
    for split in ("acdc_dev", "acdc_test", "mms_external"):
        result = ok("evaluate-alt-bundles", "--split", split)
        assert {k: v["action"] for k, v in result.items()} == {"d20": "RAN", "f9": "RAN", "f9b": "RAN"}, (split, result)
    evaluation = json.loads((run / "probes" / "evaluations" / "f9b" / "mms_external.json").read_text(encoding="utf-8"))
    assert {f["record"] for f in evaluation["alt_bundle_freeze"]} == {"probes/f9b/bundle.json", "neural_f9/f9_bundle.json"}
    d20 = json.loads((run / "probes" / "evaluations" / "d20" / "acdc_test.json").read_text(encoding="utf-8"))
    assert {f["record"] for f in d20["alt_bundle_freeze"]} == {"deep_encoder/FROZEN_D20.json"}
    for split in ("acdc_test", "mms_external"):
        ok("evaluate-trust", "--split", split)
    assert (run / "trust" / "mms_external" / "s2_ood.json").is_file()
    ok("build-comparison-tables")
    table = pd.read_csv(run / "manuscript_tables" / "comparison" / "table04_representation_comparison.csv")
    assert not table.empty
    # Once-only: a recorded evaluation of the frozen run is skipped, never recomputed.
    again = ok("evaluate-alt-bundles", "--split", "acdc_test", "--bundle", "f9")
    assert again["f9"]["action"] == "SKIPPED"
    # S5 F9 after the F9 freeze: purpose-s5 spec files train locally; their fits never enter the frozen F9 bundle.
    fdir = run / "data_efficiency" / "f9"
    ok("run-data-efficiency", "--step", "f9-wave1")
    ok("train-f9", "--backend", "local", "--specs", "data_efficiency/f9/train_wave1_specs.json", "--device", "cpu")
    ok("run-data-efficiency", "--step", "f9-wave2")
    ok("train-f9", "--backend", "local", "--specs", "data_efficiency/f9/train_wave2_specs.json", "--device", "cpu")
    ok("run-data-efficiency", "--step", "f9-infer", "--device", "cpu")
    refits = [json.loads(line) for line in (fdir / "specs_wave2.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert len(refits) == 5
    oof = pd.read_parquet(fdir / "oof" / refits[0]["fit_id"] / "predictions.parquet")
    assert set(oof["partition"]) == {"acdc_dev"} and int(oof["is_oof"].sum()) == 5
    ok("run-data-efficiency", "--step", "f9-aggregate")
    curves = json.loads((fdir / "curves.json").read_text(encoding="utf-8"))
    assert {(p["fraction"], p["repeat"]) for p in curves["points"]} == {(0.75, 0), (1.0, 0)}
    # the S5 probe decodes every held-out fold (is_oof rows of encoder fold<k>): a rho estimate at every point
    assert all(p["rho"].get("reason") != "INPUTS_NOT_RECORDED" and p["rho"]["rho"] is not None for p in curves["points"]), curves["points"]
    assert json.loads((run / "neural_f9" / "f9_bundle.json").read_text(encoding="utf-8")) == bundle   # the F9 freeze is untouched"
    # S6 D20 and F9 legs on the real producer outputs: the frozen D20 extractor and F9 encoder run on the perturbed inputs.
    ok("run-simulated-shift", "--vendor", "A", "--step", "descriptors")
    ok("run-simulated-shift", "--vendor", "A", "--step", "tensors")
    vdir = run / "simulated_shift" / "A"
    for step in ("d20", "f9"):
        result = ok("run-simulated-shift", "--vendor", "A", "--step", step, "--device", "cpu")
        assert result["conditions"][0]["status"] == "EVALUATED" and result["conditions"][0]["action"] == "RAN", result
        delta = json.loads((vdir / f"delta_{step}.json").read_text(encoding="utf-8"))
        assert delta["representation"] == step and delta["n_cases"] == 10 and delta["analysis_label"] == "registered after unblinding"
        assert "delta_rho" in delta["rho"] and "delta_accuracy" in delta["accuracy"]
        assert ok("run-simulated-shift", "--vendor", "A", "--step", step, "--device", "cpu")["conditions"][0]["action"] == "SKIPPED"
    perturbed = pd.read_parquet(vdir / "neural_f9" / "predictions.parquet")
    assert len(perturbed) == 10 and set(perturbed["partition"]) == {"acdc_test"} and set(perturbed["encoder_id"]) == {"final"}
    recorded = pd.read_parquet(run / "neural_f9" / "predictions_acdc_test.parquet")
    unperturbed = pd.read_parquet(vdir / "probes" / "f9b" / "bhat_acdc_test_unperturbed.parquet")
    assert sorted(unperturbed["sample_id"]) == sorted(recorded["sample_id"].unique())
    assert len(pd.read_parquet(vdir / "deep_encoder" / "features" / "A_D20_acdc_test.parquet")) == 10
    d20_pred = pd.read_parquet(vdir / "deep_encoder" / "predictions_acdc_test_unperturbed.parquet")
    assert set(d20_pred["model_id"]) == {"d20_predictor"} and len(d20_pred) == 10
