"""S5 and S6 stages on a synthetic run (protocol-v6.0.1 item 15; configs/data_centric.yaml).

``run --until evaluate:acdc_test`` on a synthetic ACDC fixture (25 development, 10 test patients) and the synthetic M&Ms
fixture (vendors A-D, 1.25 mm x 1.25 mm x 8 mm headers), then the data-centric stages.  Every number here is an
algorithm control on synthetic data, never a cohort result; F9 records and C4 outputs are synthetic stand-ins."""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
import yaml

import fixtures_v6 as fx
from test_v6_cli import _study_copy
from tmmri import stages
from tmmri.experiments import data_efficiency as de
from tmmri.experiments import simulated_shift as ss
from tmmri.experiments.data_centric import DevelopmentOnlyError, data_centric_config

CLASSES = ["DCM", "HCM", "MINF", "NOR", "RV"]


@pytest.fixture(scope="module")
def run_ctx(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("data_centric")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(stages, "protocol_frozen", lambda *a, **k: False)
        acdc = fx.make_acdc_source(tmp / "src" / "acdc", fx.acdc_patients(5, testing_per_class=2))
        fx.write_fixture_manifests(acdc, fx.acdc_manifest(acdc))
        mms = fx.make_mms_source(tmp / "src" / "mms")
        fx.write_fixture_manifests(mms, fx.mms_manifest(mms), fx.mms_metadata(mms))
        study_path = _study_copy(tmp)
        dc_path = study_path.parent / "data_centric.yaml"
        dc = yaml.safe_load(dc_path.read_text(encoding="utf-8"))
        dc["common"]["bootstrap"]["replicates"] = 200
        dc_path.write_text(yaml.safe_dump(dc, sort_keys=False), encoding="utf-8")
        ctx = stages.make_context(study_path, "v6-datacentric-20261003-0900", artifact_root=tmp / "artifacts", data_root=tmp / "data",
                                  n_jobs=1, allow_unfrozen_synthetic=True, sources={"acdc": acdc, "mms": mms})
        stages.run_pipeline(ctx, until="evaluate:acdc_test")
        yield ctx


def _dev_ids(ctx) -> set[str]:
    split = pd.read_parquet(ctx.run.path / "contracts" / "split_manifest.parquet")
    return set(split.loc[split["partition"] == "acdc_dev", "sample_id"])


# ----------------------------------------------------------------------------- S5


def test_s5_descriptor_curves_reproduce_the_v6_oof_at_full_fraction(run_ctx):
    ctx = run_ctx
    result = de.run_data_efficiency(ctx, "descriptor")
    assert result.action == "RAN" and de.run_data_efficiency(ctx, "descriptor").action == "SKIPPED"
    out = ctx.run.path / "data_efficiency" / "descriptor"
    targets = ctx.study.transition_targets()
    bhat = pd.read_parquet(out / "oof_bhat.parquet")
    full = bhat[(bhat["fraction"] == 1.0) & (bhat["repeat"] == 0)].set_index("sample_id")[targets]
    v6 = pd.read_parquet(ctx.run.path / "oof" / "bhat_oof_ACDC_dev.parquet").set_index("sample_id").loc[full.index, targets]
    pd.testing.assert_frame_equal(full, v6, check_exact=True, check_names=False)          # the v6 code exactly
    pred = pd.read_parquet(out / "oof_d_model.parquet")
    full_pred = pred[pred["fraction"] == 1.0].set_index("sample_id")["d_model"]
    crossfit = pd.read_parquet(ctx.run.path / "formal" / "d_model_crossfit_ACDC_dev.parquet").set_index("sample_id")["d_model_crossfit"]
    assert full_pred.loc[crossfit.index].astype(str).tolist() == crossfit.astype(str).tolist()
    assert set(bhat["sample_id"]) <= _dev_ids(ctx) and set(pred["sample_id"]) <= _dev_ids(ctx)   # development only
    curves = json.loads((out / "curves.json").read_text(encoding="utf-8"))
    assert curves["development_only"] and [(p["fraction"], p["repeat"]) for p in curves["points"]] == \
        [(0.5, 0), (0.5, 1), (0.5, 2), (0.75, 0), (0.75, 1), (0.75, 2), (1.0, 0)]
    units = pd.read_parquet(out / "units.parquet")
    assert len(units) == 35 and (units.loc[units["fraction"] == 1.0, "n_subsample"] == 20).all()
    assert (units.loc[units["fraction"] == 0.5, "n_subsample"] == 10).all() and (units["status"] == "EVALUATED").all()
    subs = pd.read_parquet(out / "subsamples.parquet")
    chosen = subs[subs["selected"]]
    assert not (chosen["outer_fold"] == chosen["heldout_fold"]).any()


def _write_records(ctx, specs):
    rng = np.random.default_rng(11)
    for s in specs:
        path = ctx.run.path / "neural_f9" / "checkpoints" / s["fit_id"] / "record.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fit_id": s["fit_id"], "val_metrics": {"macro_f1": float(rng.uniform(0.2, 0.9)),
                                    "cross_entropy": float(rng.uniform(0.5, 1.5))}, "selected_epoch": int(rng.integers(20, 81))}), encoding="utf-8")


def _c4(ctx, directory, k, fitting, heldout, *, extra_rows=()):
    """Synthetic C4 predictions/activations of one fold encoder: held-out rows correct, random activations."""
    cohort = pd.read_parquet(ctx.run.path / "contracts" / "cohort_ACDC.parquet").set_index("sample_id")
    rng = np.random.default_rng(k)
    rows, acts = [], []
    for sid, oof in [(s, False) for s in fitting] + [(s, True) for s in heldout] + list(extra_rows):
        key = {"sample_id": sid, "cluster_id": sid, "partition": "acdc_dev" if sid in cohort.index and cohort.loc[sid, "partition"] == "acdc_dev" else "acdc_test",
               "model_id": "f9", "seed": 20261011, "encoder_id": f"fold{k}", "is_oof": oof}
        y = str(cohort.loc[sid, "acdc_class"])
        rows.append({**key, "y_true": y, "y_pred": y, **{f"prob_{c}": float(c == y) for c in CLASSES}, **{f"logit_{c}": None for c in CLASSES}})
        acts.append({**key, **{f"a_{j:03d}": float(v) for j, v in enumerate(rng.normal(size=8))}})
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(directory / "predictions.parquet", index=False)
    pd.DataFrame(acts).to_parquet(directory / "activations.parquet", index=False)


def _cache_manifest(ctx) -> None:
    """A C3 manifest of the acdc_dev rows (the S5 spec files pin its SHA-256; no tensors are needed to plan)."""

    from tmmri.neural.data import MANIFEST_COLUMNS

    split = pd.read_parquet(ctx.run.path / "contracts" / "split_manifest.parquet")
    cohort = pd.read_parquet(ctx.run.path / "contracts" / "cohort_ACDC.parquet").set_index("sample_id")
    dev = sorted(split.loc[split["partition"] == "acdc_dev", "sample_id"])
    rows = [{"sample_id": s, "cluster_id": s, "partition": "acdc_dev", "label": cohort.at[s, "acdc_class"], "n_ed": 1, "n_es": 1,
             "npz_sha256": "0" * 64, "crop_fraction_ed": 0.0, "crop_fraction_es": 0.0, "inplane_spacing_mm": 1.5, "transform_version": "synthetic"}
            for s in dev]
    folder = ctx.run.path / "neural_cache" / "acdc_dev"
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS)).to_csv(folder / "manifest.csv", index=False, lineterminator="\n")


def test_s5_f9_waves_specs_selection_and_aggregate(run_ctx):
    from tmmri.neural.fit_spec import read_spec_file

    ctx = run_ctx
    fdir = ctx.run.path / "data_efficiency" / "f9"
    with pytest.raises(FileNotFoundError, match="build-tensor-cache"):    # the train-ready spec file pins the cache manifest
        de.run_data_efficiency(ctx, "f9-wave1")
    assert ctx.run.read_record("run-data-efficiency:f9-wave1") is None
    _cache_manifest(ctx)
    assert de.run_data_efficiency(ctx, "f9-wave1").action == "RAN"
    wave1 = de.read_jsonl(fdir / "specs_wave1.jsonl")
    train1 = read_spec_file(fdir / "train_wave1_specs.json")             # C5 spec file of tmmri.neural.fit_spec (purpose s5)
    assert train1["purpose"] == "s5" and train1["wave"] == 1 and [s.fit_id for s in train1["fit_specs"]] == [s["fit_id"] for s in wave1]
    plan = json.loads((fdir / "plan_wave1.json").read_text(encoding="utf-8"))
    # 4 patients per class in each outer fitting set: f = 0.5 keeps 2 per class (< 3 selection folds, NE); f = 0.75 keeps 3
    half = [u for u in plan["units"] if u["fraction"] == 0.5]
    assert len(half) == 15 and all(u["reason"] == "NE_INSUFFICIENT_CLASS_SUPPORT" for u in half)
    assert len(wave1) == 15 * 18 == plan["n_specs"] and {s["fraction"] for s in wave1} == {0.75}
    dev = _dev_ids(ctx)
    assert all(set(s["fitting_patients"]) <= dev for s in wave1)
    with pytest.raises(FileNotFoundError, match="FitRecord"):
        de.run_data_efficiency(ctx, "f9-wave2")
    assert ctx.run.read_record("run-data-efficiency:f9-wave2") is None     # a refused step records nothing
    _write_records(ctx, wave1)
    assert de.run_data_efficiency(ctx, "f9-wave2").action == "RAN"
    refits = de.read_jsonl(fdir / "specs_wave2.jsonl")
    assert len(refits) == 15 and all(s["selection_fold"] is None and s["refit_epochs"] >= 20 for s in refits)
    assert [s.fit_id for s in read_spec_file(fdir / "train_wave2_specs.json")["fit_specs"]] == [s["fit_id"] for s in refits]
    with pytest.raises(FileNotFoundError, match="refit checkpoints"):
        de.run_data_efficiency(ctx, "f9-infer")                          # local inference needs the trained refits
    with pytest.raises(FileNotFoundError, match="C4 outputs not recorded"):
        de.run_data_efficiency(ctx, "f9-aggregate")
    split = pd.read_parquet(ctx.run.path / "contracts" / "split_manifest.parquet")
    folds = split[split["partition"] == "acdc_dev"].set_index("sample_id")["outer_fold"].astype(int)
    test_id = sorted(split.loc[split["partition"] == "acdc_test", "sample_id"])[0]
    for s in refits:
        held = sorted(folds.index[folds == s["outer_fold"]])
        _c4(ctx, ctx.run.path / "data_efficiency" / "f9" / "oof" / s["fit_id"], s["outer_fold"], s["fitting_patients"], held)
    full = ctx.run.path / "neural_f9"
    for k in range(1, 6):
        _c4(ctx, full / f"tmp{k}", k, sorted(folds.index[folds != k]), sorted(folds.index[folds == k]), extra_rows=[(test_id, False)])
    for name in ("predictions.parquet", "activations.parquet"):      # one C4 file for the five primary-seed fold encoders
        pd.concat([pd.read_parquet(full / f"tmp{k}" / name) for k in range(1, 6)]).to_parquet(full / name, index=False)
    bad = ctx.run.path / "data_efficiency" / "f9" / "oof" / refits[0]["fit_id"]
    good = pd.read_parquet(bad / "predictions.parquet")
    pd.concat([good, good.iloc[[0]].assign(sample_id=test_id, partition="acdc_test")]).to_parquet(bad / "predictions.parquet", index=False)
    with pytest.raises(DevelopmentOnlyError):
        de.run_data_efficiency(ctx, "f9-aggregate")                   # an S5 refit output may never name a test row
    good.to_parquet(bad / "predictions.parquet", index=False)
    assert de.run_data_efficiency(ctx, "f9-aggregate").action == "RAN"
    curves = json.loads((fdir / "curves.json").read_text(encoding="utf-8"))
    points = {(p["fraction"], p["repeat"]): p for p in curves["points"]}
    assert set(points) == {(0.75, 0), (0.75, 1), (0.75, 2), (1.0, 0)}
    assert all(p["macro_f1"]["estimate"] == 1.0 for p in points.values())   # synthetic held-out predictions are correct
    # Every probe point decodes its held-out folds (is_oof rows of encoder fold<k>): a finite rho, never INPUTS_NOT_RECORDED.
    assert all(p["rho"]["status"] == "EVALUATED" and p["rho"]["rho"] is not None and np.isfinite(p["rho"]["rho"]) for p in points.values()), points
    assert (pd.read_parquet(fdir / "aggregate_units.parquet")["probe"] == "EVALUATED").all()
    oof = pd.read_parquet(fdir / "aggregate_oof_predictions.parquet")
    assert set(oof["sample_id"]) <= dev and test_id not in set(oof["sample_id"])   # the test row of the shared file is never read


# ----------------------------------------------------------------------------- S6


def test_s6_vendor_geometry_descriptor_deltas_and_once_only(run_ctx):
    ctx = run_ctx
    with pytest.raises(ValueError, match="unknown vendor"):
        ss.run_simulated_shift(ctx, "Z")
    result = ss.run_simulated_shift(ctx, "A", "descriptors")
    assert result["conditions"][0]["action"] == "RAN" and result["conditions"][0]["status"] == "EVALUATED"
    sdir = ctx.run.path / "simulated_shift"
    geometry = json.loads((sdir / "vendor_geometry.json").read_text(encoding="utf-8"))
    assert geometry["metadata_only"] and set(geometry["vendors"]) == {"A", "B", "C", "D"}
    for entry in geometry["vendors"].values():
        assert entry["n_cases"] == 1 and entry["median"] == {"spacing_x_mm": 1.25, "spacing_y_mm": 1.25, "slice_increment_mm": 8.0}
    cases = pd.read_parquet(sdir / "vendor_geometry_cases.parquet")
    assert set(cases["header_source"]) == {"staged_ed_image_header"} and len(cases) == 4
    vdir = sdir / "A"
    delta = json.loads((vdir / "delta_descriptors.json").read_text(encoding="utf-8"))
    assert delta["analysis_label"] == "registered after unblinding" and delta["n_cases"] == 10 and delta["partition"] == "acdc_test"
    assert delta["target_spacing_mm"] == [1.25, 1.25, 8.0] and delta["representation"] == "descriptors"
    assert "delta_accuracy" in delta["accuracy"] and "delta_rho" in delta["rho"] and set(delta["rho"]["family"]) == set(ctx.study.primary_family())
    log = pd.read_parquet(vdir / "resample_log.parquet")
    assert len(log) == 10 and set(log["shape_out"]) == {"[30, 28, 5]"} and set(log["shape_in"]) == {"[24, 22, 4]"}
    assert all(set(json.loads(v)) <= {0, 1, 2, 3} for v in log["mask_labels_out"])
    b1 = pd.read_parquet(vdir / "B_acdc_test.parquet")
    assert len(b1) == 10 and set(b1["split"]) == {"acdc_test"} and set(b1["spacing_x_mm"]) == {1.25}
    assert ss.run_simulated_shift(ctx, "A", "descriptors")["conditions"][0]["action"] == "SKIPPED"
    with pytest.raises(stages.StageRecordLockedError, match="once-only"):   # frozen run: a vendor condition runs once
        ss.run_simulated_shift(dataclasses.replace(ctx, force=True), "A", "descriptors")


def test_s6_identity_condition_reproduces_recorded_a_and_b(run_ctx):
    """Resampling to the case's own geometry is the identity, so the recomputed A and B equal the recorded ones."""
    from tmmri.formal_features import formal_descriptor_names, formal_descriptors_v3
    from tmmri.semantics.features import case_semantics_v6

    ctx = run_ctx
    _, case_ids, metadata = ss._test_cases(ctx, data_centric_config(ctx.study)["simulated_shift"])
    a = pd.read_parquet(ctx.run.path / "formal" / "A_ACDC.parquet").set_index("sample_id")
    b = pd.read_parquet(ctx.run.path / "semantics" / "B_ACDC.parquet").set_index("sample_id")
    for cid in case_ids[:3]:
        case = stages.load_case_for_semantics(cid, dataset="ACDC", staged_root=str(ctx.staged_root), label_map=ctx.study.label_map("acdc"),
                                              metadata=metadata)
        same = ss.perturb_case(case, case.spacing_mm)
        np.testing.assert_array_equal(formal_descriptors_v3(same), a.loc[case.sample_id, formal_descriptor_names()].to_numpy(float))
        row = case_semantics_v6(same, ctx.study.registry, cohorts=ctx.study.cohorts, reason_codes=ctx.study.reason_codes)
        for t in ctx.study.transition_targets():
            recorded = b.loc[case.sample_id, t]
            assert (row[t] is None and pd.isna(recorded)) or row[t] == pytest.approx(float(recorded), rel=0, abs=1e-12), t


def test_s6_tensor_inputs_and_deep_representation_collection(run_ctx, monkeypatch):
    ctx = run_ctx
    ss.run_simulated_shift(ctx, "A", "descriptors")                    # recorded (skips)
    monkeypatch.setattr(ss, "resolve_tensor_hook", lambda spec: None)
    with pytest.raises(NotImplementedError, match="hook"):
        ss.run_simulated_shift(ctx, "A", "tensors")
    assert ctx.run.read_record("run-simulated-shift:A:tensors") is None
    monkeypatch.undo()
    # The configured hook is L4a's own transform (tmmri.neural.preprocess:case_tensors), as build-tensor-cache.
    assert ss.resolve_tensor_hook(data_centric_config(ctx.study)["simulated_shift"]["tensor_inputs"]["hook"]) is not None
    assert ss.run_simulated_shift(ctx, "A", "tensors")["conditions"][0]["action"] == "RAN"
    cache = ctx.run.path / "simulated_shift" / "A" / "neural_cache" / "acdc_test"
    manifest = pd.read_csv(cache / "manifest.csv", keep_default_na=False)
    assert list(manifest.columns) == ss.C3_MANIFEST_COLUMNS and len(manifest) == 10
    assert set(manifest["n_ed"]) == {5} and set(manifest["partition"]) == {"acdc_test"} and set(manifest["label"]) == {""}
    from tmmri.runs import sha256_file
    for row in manifest.itertuples():
        assert sha256_file(cache / ss.npz_filename(row.sample_id)) == row.npz_sha256
    # The D20 and F9 legs run the frozen extractor / encoder on the perturbed inputs themselves (GPU image; covered on
    # real producer outputs by test_v601_neural_chain.py).  Without the D20 freeze, the recorded acdc_test features, the
    # F9 freeze and infer-f9 --split acdc_test they refuse and record nothing; no stand-in file is ever read.
    for step, needed in (("d20", "deep_encoder/FROZEN_D20.json"), ("f9", "neural_f9/f9_bundle.json")):
        with pytest.raises(FileNotFoundError, match="inputs not recorded") as info:
            ss.run_simulated_shift(ctx, "A", step)
        assert needed in str(info.value)
        assert ctx.run.read_record(f"run-simulated-shift:A:{step}") is None
    cfg = data_centric_config(ctx.study)["simulated_shift"]["representations"]
    paths = {p for rep in ("d20", "f9") for kind in ("features", "predictions", "activations") for p in (cfg[rep].get(kind) or {}).values()}
    assert {"deep_encoder/features/A_D20_acdc_test.parquet", "neural_f9/predictions_acdc_test.parquet",
            "neural_f9/activations_acdc_test.parquet"} <= paths                 # the unperturbed inputs are producer outputs
