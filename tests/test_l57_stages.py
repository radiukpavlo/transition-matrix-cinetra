"""Lane L57 stages (fit-probes, evaluate-alt-bundles, fit-trust, evaluate-trust, build-comparison-tables) on a synthetic,
frozen run root: synthetic A, B, D20 features and C4 predictions/activations only (tmp_path; no cohort data)."""

from __future__ import annotations

import json
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest

from tmmri import stages
from tmmri.classification import fit_formal_predictor
from tmmri.config import load_study
from tmmri.experiments import l57_stages as l57
from tmmri.experiments.probes import fit_f9b_probes
from tmmri.experiments.trust.calibration import softmax
from tmmri.formal_features import formal_descriptor_names
from tmmri.runs import RunRoot, atomic_write_json, hash_path
from tmmri.transition_fit import fit_transition_v6, nested_oof_transition

CLASSES = ["DCM", "HCM", "MINF", "NOR", "RV"]
SEEDS = (20261011, 20261012, 20261013)
N_DEV, N_TEST = 30, 10


def _ids(start: int, n: int) -> list[str]:
    return [f"ACDC:patient{i:03d}" for i in range(start, start + n)]


def _b(a: np.ndarray, targets: list[str], rng: np.random.Generator) -> np.ndarray:
    return np.column_stack([50.0 + 15.0 * np.tanh(0.5 * (a[:, j % 16] + a[:, (j + 3) % 16]) + 0.1 * rng.normal(size=len(a))) for j in range(len(targets))])


def _c4(ids: list[str], partition: str, model_id: str, seed: int, encoder: list[str], is_oof: list[bool], y: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"sample_id": ids, "cluster_id": ids, "partition": partition, "model_id": model_id, "seed": seed, "encoder_id": encoder,
                         "is_oof": is_oof, "y_true": y})


def _proba_rows(frame: pd.DataFrame, z: np.ndarray, *, logits: bool = True) -> pd.DataFrame:
    p = softmax(z)
    frame = frame.copy()
    frame["y_pred"] = np.asarray(CLASSES, dtype=object)[p.argmax(axis=1)]
    for j, c in enumerate(CLASSES):
        frame[f"prob_{c}"] = p[:, j]
    for j, c in enumerate(CLASSES):
        frame[f"logit_{c}"] = z[:, j] if logits else np.nan
    return frame


@pytest.fixture()
def l57_run(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "protocol_frozen", lambda *a, **k: True)
    study = load_study()
    study.bootstrap["fixed_model_bootstrap"]["replicates"] = 48       # fast; not a bundle section
    run = RunRoot("v6-l57-20261003-0900", tmp_path / "artifacts", subfolders=study.study["artifact_subfolders"])
    ctx = stages.StageContext(study=study, run=run, data_root=tmp_path / "data", n_jobs=1)
    rng = np.random.default_rng(20261003)
    p = run.path
    dev, test = _ids(1, N_DEV), _ids(101, N_TEST)
    allids = dev + test
    cls = {s: CLASSES[i % 5] for i, s in enumerate(dev)} | {s: CLASSES[i % 5] for i, s in enumerate(test)}
    fold = {s: (i // 5) % 5 + 1 for i, s in enumerate(dev)}
    split = pd.DataFrame({"sample_id": allids, "dataset": "ACDC", "case_id": [s.split(":")[1] for s in allids],
                          "patient_id": [s.split(":")[1] for s in allids], "partition": ["acdc_dev"] * N_DEV + ["acdc_test"] * N_TEST,
                          "outer_fold": pd.array([fold[s] for s in dev] + [pd.NA] * N_TEST, dtype="Int64"), "cluster_id": allids,
                          "refit_partition": [None] * len(allids)})
    split.to_parquet(p / "contracts" / "split_manifest.parquet", index=False)
    inner = []
    for f in range(1, 6):
        train = [s for s in dev if fold[s] != f]
        inner += [{"outer_fold_heldout": f, "sample_id": s, "inner_fold": j % 5 + 1} for j, s in enumerate(train)]
    pd.DataFrame(inner).to_parquet(p / "contracts" / "inner_folds.parquet", index=False)
    pd.DataFrame({"sample_id": allids, "case_id": [s.split(":")[1] for s in allids], "acdc_class": [cls[s] for s in allids],
                  "partition": split["partition"], "eligible_v6": True}).to_parquet(p / "contracts" / "cohort_ACDC.parquet", index=False)
    names = formal_descriptor_names()
    signal = np.array([[2.0 if CLASSES.index(cls[s]) == k else 0.0 for k in range(5)] for s in allids])
    latent = rng.normal(size=(len(allids), 16))
    latent[:, :5] += signal
    a = np.column_stack([latent, rng.normal(size=(len(allids), 57 - 16))])
    a_frame = pd.DataFrame(a, columns=names)
    a_frame.insert(0, "sample_id", allids)
    a_frame.to_parquet(p / "formal" / "A_ACDC.parquet", index=False)
    targets = study.transition_targets()
    critical = sorted(str(c) for c in study.cohorts["qc_severity"]["critical"])[0]
    b = pd.DataFrame(_b(latent, targets, rng), columns=targets)
    b.insert(0, "sample_id", allids)
    b["qc_failure_reason"] = [None] * (len(allids) - 1) + [critical]          # one critical test row (excluded from class metrics)
    b.to_parquet(p / "semantics" / "B_ACDC.parquet", index=False)
    ids_frame = pd.DataFrame({"sample_id": dev, "outer_fold": [fold[s] for s in dev]})
    folds = ids_frame["outer_fold"].to_numpy(dtype=int)
    b_dev = b.iloc[:N_DEV].reset_index(drop=True)
    inner_fn = stages._inner_fold_fn(ctx, ids_frame)
    cfg, reg = study.transition, study.registry
    model = fit_transition_v6(a[:N_DEV], b_dev, targets, cfg, reg, folds)
    atomic_write_json(p / "transitions" / "transition_model.json", model.to_dict(), allow_nan=True)
    oof = nested_oof_transition(a[:N_DEV], b_dev, targets, cfg, reg, folds, inner_fn)
    oof.assign(sample_id=dev).to_parquet(p / "oof" / "bhat_oof_ACDC_dev.parquet", index=False)
    pcfg = study.formal["audited_predictor"]
    y_dev = pd.Series([cls[s] for s in dev])
    predictor = fit_formal_predictor(a[:N_DEV], y_dev, pcfg)
    joblib.dump(predictor, p / "formal" / "d_model_predictor.joblib")
    crossfit = np.empty(N_DEV, dtype=object)
    for f in range(1, 6):
        held = folds == f
        crossfit[held] = stages.predict_labels(fit_formal_predictor(a[:N_DEV], y_dev, pcfg, fit_mask=~held), a[:N_DEV][held])
    pd.DataFrame({"sample_id": dev, "outer_fold": folds, "d_model_crossfit": crossfit}).to_parquet(p / "formal" / "d_model_crossfit_ACDC_dev.parquet", index=False)
    bhat_test = model.predict(a[N_DEV:])
    bhat_test.insert(0, "sample_id", test)
    bhat_test.to_parquet(p / "reconstructed" / "bhat_acdc_test.parquet", index=False)
    pd.DataFrame({"sample_id": test, "d_model": stages.predict_labels(predictor, a[N_DEV:])}).to_parquet(p / "inference" / "d_model_acdc_test.parquet", index=False)
    atomic_write_json(p / "metrics" / "routes_d_model_acdc_test.json", {"accepted_coverage": 0.0, "covered_fidelity": None, "n_accepted": 0, "n_attempted": N_TEST})
    for rb in ("d_model", "dataset_label", "measured_oracle"):
        (p / "rules" / rb).mkdir(parents=True, exist_ok=True)
        (p / "rules" / rb / "EMPTY.json").write_text(json.dumps({"rulebook": rb, "records": []}), encoding="utf-8")
    stages.freeze_analysis_bundle(ctx)
    stages._stage(ctx, "evaluate:acdc_test", lambda: {}, [run.frozen_path],
                  [p / "reconstructed" / "bhat_acdc_test.parquet", p / "inference" / "d_model_acdc_test.parquet", p / "metrics" / "routes_d_model_acdc_test.json"],
                  {"split": "acdc_test"}, status="EVALUATED")
    # ---- D20 (L4a outputs, the layout of tmmri.deep_encoder.bundle_paths): 20 features per partition, T, predictor, C4
    de = p / "deep_encoder"
    for sub in ("features", "transitions", "oof", "formal"):
        (de / sub).mkdir(parents=True)
    feats = np.column_stack([latent[:, :8] + 0.3 * rng.normal(size=(len(allids), 8)), rng.normal(size=(len(allids), 12))])
    fcols = [f"a_{j:04d}" for j in range(20)]
    for part, sl in (("acdc_dev", slice(0, N_DEV)), ("acdc_test", slice(N_DEV, None))):
        frame = pd.DataFrame(feats[sl], columns=fcols)
        frame.insert(0, "sample_id", allids[sl])
        frame.insert(1, "cluster_id", allids[sl])
        frame.insert(2, "partition", part)
        frame.to_parquet(de / "features" / f"A_D20_{part}.parquet", index=False)
    d20_t = fit_transition_v6(feats[:N_DEV], b_dev, targets, cfg, reg, folds, representation="d20")
    atomic_write_json(de / "transitions" / "transition_model.json", d20_t.to_dict(), allow_nan=True)
    nested_oof_transition(feats[:N_DEV], b_dev, targets, cfg, reg, folds, inner_fn).assign(sample_id=dev).to_parquet(
        de / "oof" / "bhat_oof_ACDC_dev.parquet", index=False)
    d20_pred = fit_formal_predictor(feats[:N_DEV], y_dev, pcfg)
    joblib.dump(d20_pred, de / "formal" / "d_model_predictor.joblib")
    oof_z = np.zeros((N_DEV, 5))
    for f in range(1, 6):
        held = folds == f
        oof_z[held] = fit_formal_predictor(feats[:N_DEV], y_dev, pcfg, fit_mask=~held).decision_function(feats[:N_DEV][held])
    c4 = pd.concat([_proba_rows(_c4(dev, "acdc_dev", "d20_predictor", -1, [f"fold{fold[s]}" for s in dev], [True] * N_DEV, [cls[s] for s in dev]), oof_z),
                    _proba_rows(_c4(test, "acdc_test", "d20_predictor", -1, ["final"] * N_TEST, [False] * N_TEST, [cls[s] for s in test]),
                                d20_pred.decision_function(feats[N_DEV:]))], ignore_index=True)
    c4.to_parquet(de / "predictions.parquet", index=False)
    atomic_write_json(de / "FROZEN_D20.json", {"schema": "tmmri.d20_bundle/1",       # L4a freeze record: files relative to deep_encoder/
                                               "files": {"transitions/transition_model.json": hash_path(de / "transitions" / "transition_model.json"),
                                                         "formal/d_model_predictor.joblib": hash_path(de / "formal" / "d_model_predictor.joblib")}})
    # ---- F9 (L4a/L4b outputs): C4 predictions and 64-d activations per encoder, three seeds
    nf = p / "neural_f9"
    (nf / "checkpoints" / "abc").mkdir(parents=True)
    (nf / "checkpoints" / "abc" / "model.pt").write_bytes(b"synthetic checkpoint stand-in")
    atomic_write_json(nf / "f9_bundle.json", {"schema": "tmmri.f9_bundle/1",          # L4b freeze-f9 record
                                              "fits": [{"fit_id": "abc", "checkpoint": "neural_f9/checkpoints/abc/model.pt",
                                                        "checkpoint_sha256": hash_path(nf / "checkpoints" / "abc" / "model.pt")}]})
    preds, acts = [], []
    acols = [f"a_{j:03d}" for j in range(64)]
    for seed in SEEDS:
        srng = np.random.default_rng(seed)
        onehot = np.array([[1.0 if CLASSES.index(cls[s]) == k else 0.0 for k in range(5)] for s in allids])
        z = 2.0 * onehot + srng.normal(size=(len(allids), 5))
        preds.append(_proba_rows(_c4(dev, "acdc_dev", "f9", seed, [f"fold{fold[s]}" for s in dev], [True] * N_DEV, [cls[s] for s in dev]), z[:N_DEV]))
        preds.append(_proba_rows(_c4(test, "acdc_test", "f9", seed, ["final"] * N_TEST, [False] * N_TEST, [cls[s] for s in test]), z[N_DEV:],
                                 logits=seed != SEEDS[1]))
        for enc in [f"fold{k}" for k in range(1, 6)] + ["final"]:
            rot = srng.normal(size=(16, 64)) / 4.0                    # every encoder has its own coordinate system
            vals = latent @ rot + 0.05 * srng.normal(size=(len(allids), 64))
            members = list(range(N_DEV)) + (list(range(N_DEV, len(allids))) if enc == "final" else [])
            frame = _c4([allids[i] for i in members], "acdc_dev", "f9", seed, [enc] * len(members),
                        [enc != "final" and fold.get(allids[i]) == int(enc[4:]) for i in members], [cls[allids[i]] for i in members])
            frame["partition"] = ["acdc_dev" if i < N_DEV else "acdc_test" for i in members]
            acts.append(pd.concat([frame, pd.DataFrame(vals[members], columns=acols)], axis=1))
    pd.concat(preds, ignore_index=True).to_parquet(nf / "predictions.parquet", index=False)
    pd.concat(acts, ignore_index=True).to_parquet(nf / "activations.parquet", index=False)
    return ctx


def test_l57_stages_end_to_end_and_once_only(l57_run, monkeypatch):
    ctx = l57_run
    p = ctx.run.path
    assert l57.fit_probes(ctx).action == "RAN"
    rho = json.loads((p / "probes" / "f9b" / "rho_oof_acdc_dev.json").read_text(encoding="utf-8"))
    assert list(rho["seeds"]) == [str(s) for s in SEEDS] and rho["primary_seed"] == SEEDS[0] and rho["seed_spread"]["rho"]["n_seeds"] == 3
    bundle = json.loads((p / "probes" / "f9b" / "bundle.json").read_text(encoding="utf-8"))
    assert all((p / rel).is_file() for rel in bundle["bundle"]) and bundle["pool_latent_across_encoders"] is False
    fold1 = json.loads((p / "probes" / "f9b" / f"seed{SEEDS[0]}" / "transition_fold1.json").read_text(encoding="utf-8"))
    assert fold1["representation"].endswith(":fold1") and fold1["n_fit_rows"] == N_DEV - 10      # fold 1 holds 10 patients

    for split in ("acdc_dev", "acdc_test"):
        results = l57.evaluate_alt_bundles(ctx, split)
        assert {k: v.action for k, v in results.items()} == {"d20": "RAN", "f9": "RAN", "f9b": "RAN"}
    d20 = json.loads((p / "probes" / "evaluations" / "d20" / "acdc_test.json").read_text(encoding="utf-8"))
    assert d20["label"] == "registered after unblinding" and d20["rows"]["n_excluded_critical_qc"] == 1
    desc = d20["descriptor_pipeline"]["representation"]["rho"]
    recorded_bhat = pd.read_parquet(p / "reconstructed" / "bhat_acdc_test.parquet").set_index("sample_id")
    b = pd.read_parquet(p / "semantics" / "B_ACDC.parquet").set_index("sample_id")
    sd, mean = stages._dev_reference(ctx)
    targets = ctx.study.transition_targets()
    expected = stages._rho(ctx, b.loc[recorded_bhat.index, targets], recorded_bhat, mean, sd)
    assert desc["rho"] == pytest.approx(expected["rho"])                       # the frozen v6 estimand on the recorded rows
    rep = d20["representations"]["d20"]["seeds"]["-1"]
    paired = rep["paired_vs_descriptors"]
    if paired["status"] == "EVALUATED" and set(paired["estimable_targets"]) == set(desc["estimable_targets"]) == set(rep["rho"]["estimable_targets"]):
        assert paired["difference"] == pytest.approx(rep["rho"]["rho"] - desc["rho"])
    pm = d20["predictors"]["d20_predictor"]["seeds"]["-1"]["metrics"]
    assert pm["counts"]["n_evaluable"] == N_TEST - 1 and "brier" in pm["metrics"]
    f9 = json.loads((p / "probes" / "evaluations" / "f9" / "acdc_test.json").read_text(encoding="utf-8"))
    assert list(f9["predictors"]["f9"]["seeds"]) == [str(s) for s in SEEDS] and f9["predictors"]["f9"]["seed_spread"]["accuracy"]["n_seeds"] == 3
    assert f9["predictors"]["f9"]["seeds"][str(SEEDS[1])]["provenance"]["logits"] == "log_probabilities"
    assert {f["record"] for f in f9["alt_bundle_freeze"]} == {"neural_f9/f9_bundle.json"}

    assert l57.fit_trust(ctx).action == "RAN"
    s1 = json.loads((p / "trust" / "s1_thresholds.json").read_text(encoding="utf-8"))
    for predictor in ("audited_predictor", "d20_predictor", "f9"):
        for name, c in (("tau50", 0.5), ("tau80", 0.8)):
            fit = s1["predictors"][predictor][name]
            assert fit["development_coverage"] == pytest.approx(int(np.ceil(c * fit["n"])) / fit["n"]) and fit["development_coverage"] >= c
    temps = json.loads((p / "trust" / "s4_temperatures.json").read_text(encoding="utf-8"))
    assert all(temps["predictors"][k]["status"] == "FITTED" for k in ("audited_predictor", "d20_predictor", "f9"))
    assert set(json.loads((p / "trust" / "trust_bundle.json").read_text(encoding="utf-8"))["representations"].values()) == {"FITTED"}

    assert l57.evaluate_trust(ctx, "acdc_test").action == "RAN"
    out = p / "trust" / "acdc_test"
    s3 = json.loads((out / "s3_harmonization.json").read_text(encoding="utf-8"))
    assert s3["representations"]["descriptors"]["variants"]["H-a"]["label_free_assertion"].startswith("passed")
    assert s3["representations"]["descriptors"]["variants"]["H-b"]["reason"] == "VENDOR_METADATA_NOT_RECORDED"
    assert s3["representations"]["f9b"]["predictor_status"]["reason"] == l57.NO_A_STANDARDIZATION
    s2 = json.loads((out / "s2_ood.json").read_text(encoding="utf-8"))
    assert set(s2["representations"]) == {"descriptors", "d20", "f9b"} and s2["representations"]["d20"]["n_scored"] == N_TEST
    s1e = json.loads((out / "s1_selective.json").read_text(encoding="utf-8"))
    assert s1e["rule_routes"]["accepted_coverage"] == 0.0 and s1e["predictors"]["f9"]["tau50"]["realised"]["n"] == N_TEST - 1
    s4 = json.loads((out / "s4_calibration.json").read_text(encoding="utf-8"))
    assert s4["predictors"]["audited_predictor"]["before_after"]["metrics"]["accuracy"]["difference"] == 0.0
    assert not pd.read_parquet(out / "s4_reliability.parquet").empty

    assert l57.build_comparison_tables(ctx).action == "RAN"
    table = pd.read_csv(p / "manuscript_tables" / "comparison" / "table04_representation_comparison.csv")
    assert len(table) == 12 and table["panel"].nunique() == 4
    mms = table[table["panel"].str.startswith("mms_external")]
    assert mms["rho"].str.startswith("NE").all()                                   # not evaluated: NE with a reason
    assert table.loc[table["panel"].str.startswith("acdc_test"), "rho"].str.startswith("NE (n < 20)").all()   # 10 clusters < 20
    fig = pd.read_parquet(p / "manuscript_tables" / "comparison" / "figure05_representation_panel.parquet")
    assert {"estimate", "difference_vs_descriptors"} <= set(fig["kind"]) and set(fig.loc[fig["predictor"] == "f9", "seed"]) == set(SEEDS)

    # once-only: an identical second call is skipped; any re-execution of a recorded stage of the frozen run is refused
    assert l57.evaluate_trust(ctx, "acdc_test").action == "SKIPPED"
    with pytest.raises(stages.StageRecordLockedError):
        l57.evaluate_trust(replace(ctx, force=True), "acdc_test")
    with pytest.raises(stages.StageRecordLockedError):
        l57.evaluate_alt_bundles(replace(ctx, force=True), "acdc_test", bundles=["d20"])
    (p / "probes" / "evaluations" / "f9" / "acdc_test.json").write_text("{}", encoding="utf-8")   # a changed recorded output
    with pytest.raises(stages.StageRecordLockedError):
        l57.evaluate_alt_bundles(ctx, "acdc_test", bundles=["f9"])

    # CLI: the same refusal maps to the bundle exit code 7
    from tmmri import cli

    with pytest.raises(SystemExit) as exc:
        cli.main(["evaluate-trust", "--split", "acdc_test", "--run-id", ctx.run.run_id, "--artifact-root", str(ctx.run.artifact_root), "--force"])
    assert exc.value.code == cli.EXIT_BUNDLE


@pytest.mark.parametrize("command", ["fit-probes", "evaluate-alt-bundles", "fit-trust", "evaluate-trust", "build-comparison-tables"])
def test_l57_subcommands_have_help(command, capsys):
    from tmmri import cli
    from tmmri.experiments.l57_cli import L57_COMMANDS

    assert command in L57_COMMANDS and not set(L57_COMMANDS) & set(cli.V6_COMMANDS)
    with pytest.raises(SystemExit) as info:
        cli.main([command, "--help"])
    assert info.value.code == 0 and command in capsys.readouterr().out


def test_held_out_evaluation_needs_the_alternative_bundle_freeze_record(l57_run):
    ctx = l57_run
    record = ctx.run.path / "deep_encoder" / "FROZEN_D20.json"
    frozen = json.loads(record.read_text(encoding="utf-8"))
    record.unlink()
    with pytest.raises(stages.BundleError, match="freeze record"):
        l57.evaluate_alt_bundles(ctx, "acdc_test", bundles=["d20"])
    atomic_write_json(record, frozen)
    joblib.dump("tampered", ctx.run.path / "deep_encoder" / "formal" / "d_model_predictor.joblib")   # keys relative to deep_encoder/
    with pytest.raises(stages.BundleError, match="changed after the freeze"):
        l57.evaluate_alt_bundles(ctx, "acdc_test", bundles=["d20"])


def test_mms_partitions_refuse_before_the_protocol_freeze(l57_run, monkeypatch):
    ctx = l57_run
    monkeypatch.setattr(stages, "protocol_frozen", lambda *a, **k: False)
    for call in (lambda: l57.evaluate_alt_bundles(ctx, "mms_external"), lambda: l57.evaluate_alt_bundles(ctx, "mms2_external"),
                 lambda: l57.evaluate_trust(ctx, "mms2_external")):
        with pytest.raises(stages.ProtocolFreezeError):
            call()
    monkeypatch.setattr(stages, "protocol_frozen", lambda tag="protocol-v6.0.0", *a, **k: tag == "protocol-v6.0.0")   # v6.0.0 only
    with pytest.raises(stages.ProtocolFreezeError, match="protocol-v6.0.1"):
        l57.evaluate_alt_bundles(ctx, "mms2_external")


def test_probes_never_pool_latent_coordinates_across_encoders(l57_run):
    ctx = l57_run
    ids, b, _ = l57._dev_fit_ids(ctx)
    acts = pd.read_parquet(ctx.run.path / "neural_f9" / "activations.parquet")
    acts = acts[(acts["seed"] == SEEDS[0]) & (acts["partition"] == "acdc_dev")]
    cols = [c for c in acts.columns if c.startswith("a_")]
    missing = acts[~((acts["encoder_id"] == "fold2") & (acts["sample_id"] == ids["sample_id"].iloc[0]))]   # fold-2 encoder lacks one fitting patient
    with pytest.raises(ValueError, match="encoder fold2"):
        fit_f9b_probes(ids, b.reset_index(), missing, cols, ctx.study.transition_targets(), ctx.study.transition, ctx.study.registry,
                       stages._inner_fold_fn(ctx, ids))
