"""Second post-freeze exploratory block B23-B27 (protocol-v6.0.3) on a synthetic frozen v6 run and a synthetic B10-B22 run
(algorithm controls only; never cohort results): the D43 reading, the B18 reproduction refusal, the B25 fold assignment
and frozen cuts, the B26 cell suppression and identifier policy, the B27 spread, the protocol-v6.0.3 lock, the
second-source hash check and that nothing is written outside the new run root; every analysis through the CLI.

The synthetic F9 and F9-M outputs (predictions, activations, F9b and F9b-M probes) are generated here: activations are
a fixed random map of standardized measured B plus noise and the decisions a noisy linear read-out of it (no network is
trained; the probes are fitted with :func:`tmmri.experiments.probes.fit_f9b_probes`)."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import fixtures_v6 as fx
from test_v6_cli import METHODS_MODULES, _edit_yaml, _run, _study_copy
from tmmri import cli, stages
from tmmri.experiments.postfreeze import b23_decision_link_nulls as b23
from tmmri.experiments.postfreeze import b24_network_rule_audit as b24
from tmmri.experiments.postfreeze import b25_pooled_admission as b25
from tmmri.experiments.postfreeze import b26_robust_agreement as b26
from tmmri.experiments.postfreeze import b27_fold_spread as b27
from tmmri.experiments.postfreeze import common as pf
from tmmri.experiments.postfreeze import lane_v603 as v603
from tmmri.experiments.postfreeze.b15_rule_ledger import PASSED
from tmmri.runs import atomic_write_json, hash_path

SOURCE = "v6-pfsource-20261003-1241"
SECOND = "v6-postfreeze-20261028-0201"          # the lane's second source run id, rebuilt synthetically
RUN = "v6-postfreeze-20261029-1300"
SEED = 20261011
CLASSES = ("DCM", "HCM", "MINF", "NOR", "RV")
ORDER = ("B23", "B24", "B26", "B25", "B27")     # D45 run order


def _main(argv: list[str]) -> None:
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            cli.main(argv)
    except SystemExit as exc:
        if exc.code:
            raise AssertionError(f"{argv[0]} failed ({exc.code}): {out.getvalue()[-2000:]}") from None


def _tree(root) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def _network(pctx, rng: np.random.Generator, splits: dict[str, str]) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """(predictions, activations) of one synthetic network, seed 20261011: six encoders on acdc_dev, the final one elsewhere."""

    targets = pctx.study.transition_targets()
    manifest = pd.read_parquet(pctx.source_root / "contracts" / "split_manifest.parquet")
    _, _, b_dev, _ = stages._dev_matrices(pctx.stage_ctx)
    ref = b_dev[targets].apply(pd.to_numeric, errors="coerce")
    mean, sd = ref.mean(), ref.std().replace(0, 1).fillna(1)
    w, v = rng.normal(size=(len(targets), 64)), rng.normal(size=(len(targets), len(CLASSES)))
    out = {}
    for split, ds in splits.items():
        rows = manifest[manifest["partition"].astype(str) == split]
        b = pd.read_parquet(pctx.source_root / "semantics" / f"B_{ds}.parquet").set_index("sample_id")
        ids = [s for s in rows["sample_id"].astype(str) if s in b.index]
        folds = rows.set_index("sample_id")["outer_fold"].reindex(ids)
        clusters = rows.set_index("sample_id")["cluster_id"].reindex(ids).astype(str)
        z = ((b.loc[ids, targets].apply(pd.to_numeric, errors="coerce") - mean) / sd).fillna(0.0).to_numpy()
        preds, acts = [], []
        for enc in ([f"fold{k}" for k in range(1, 6)] + ["final"]) if split == "acdc_dev" else ["final"]:
            a = z @ w + rng.normal(scale=0.3, size=(len(ids), 64))
            logits = 1.5 * z @ v + rng.normal(scale=0.5, size=(len(ids), len(CLASSES)))
            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            is_oof = (folds.astype(float).to_numpy() == float(enc[4:])) if enc.startswith("fold") else np.zeros(len(ids), dtype=bool)
            meta = pd.DataFrame({"sample_id": ids, "cluster_id": clusters.to_numpy(), "partition": split, "model_id": "f9_synthetic", "seed": SEED,
                                 "encoder_id": enc, "is_oof": is_oof})
            preds.append(meta.assign(y_true=None, y_pred=np.array(CLASSES)[probs.argmax(axis=1)],
                                     **{f"prob_{c}": probs[:, i] for i, c in enumerate(CLASSES)}, **{f"logit_{c}": logits[:, i] for i, c in enumerate(CLASSES)}))
            acts.append(pd.concat([meta, pd.DataFrame(a, columns=[f"a_{i:03d}" for i in range(64)])], axis=1))
        out[split] = (pd.concat(preds, ignore_index=True), pd.concat(acts, ignore_index=True))
    return out


def _write_network(pctx, tables, pred_dir, probe_dir, evaluated_dir=None, evaluations_dir=None) -> list:
    """Predictions and activations per split, the F9b-type probe of seed 20261011 and (F9-M) its evaluated B-hat or (F9)
    its recorded evaluation metrics, written by the metric code of evaluate-alt-bundles (``l57_stages.rho_block``)."""

    from tmmri.experiments import l57_stages as l57
    from tmmri.experiments.probes import encoder_matrix, fit_f9b_probes

    written = []
    pred_dir.mkdir(parents=True, exist_ok=True)
    for split, (pred, act) in tables.items():
        for name, frame in (("predictions", pred), ("activations", act)):
            path = pred_dir / f"{name}_{split}.parquet"
            frame.to_parquet(path, index=False)
            written.append(path)
    ids, _, b_dev, _ = stages._dev_matrices(pctx.stage_ctx)
    cols = [f"a_{i:03d}" for i in range(64)]
    res = fit_f9b_probes(ids, b_dev, tables["acdc_dev"][1], cols, pctx.study.transition_targets(), pctx.study.transition, pctx.study.registry,
                         stages._inner_fold_fn(pctx.stage_ctx, ids))
    probe_dir.mkdir(parents=True, exist_ok=True)
    res["oof"].to_parquet(probe_dir / "bhat_oof_ACDC_dev.parquet", index=False)
    atomic_write_json(probe_dir / "transition_final.json", res["final"].to_dict(), allow_nan=True)
    written += [probe_dir / "bhat_oof_ACDC_dev.parquet", probe_dir / "transition_final.json"]
    for k, model in res["fold_models"].items():
        atomic_write_json(probe_dir / f"transition_fold{k}.json", model.to_dict(), allow_nan=True)
        written.append(probe_dir / f"transition_fold{k}.json")
    if evaluated_dir is not None:
        evaluated_dir.mkdir(parents=True, exist_ok=True)
        for split, (_, act) in tables.items():
            if split == "acdc_dev":
                continue
            sids = act["sample_id"].astype(str).tolist()
            bhat = res["final"].predict(encoder_matrix(act, "final", sids, cols), index=pd.Index(sids, name="sample_id")).reset_index()
            path = evaluated_dir / f"bhat_{split}_seed{SEED}.parquet"
            bhat.to_parquet(path, index=False)
            written.append(path)
    if evaluations_dir is not None:
        ctx = pctx.stage_ctx
        sd, mean = stages._dev_reference(ctx)
        for split, (_, act) in tables.items():
            if split == "acdc_dev":
                continue
            sids = act["sample_id"].astype(str).tolist()
            bhat = res["final"].predict(encoder_matrix(act, "final", sids, cols), index=pd.Index(sids, name="sample_id"))
            pdata = l57.partition_data(ctx, split)
            rho = l57.rho_block(ctx, pdata, bhat, l57._rho_rows(ctx, pdata), mean, sd)
            path = evaluations_dir / f"{split}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            l57._write_json(path, {"schema": "tmmri.alt_bundle_evaluation/1", "bundle": "f9b", "partition": split,
                                   "representations": {"f9b": {"primary_seed": SEED, "seeds": {str(SEED): {"rho": rho}}}}})
            written.append(path)
    return written


@pytest.fixture(scope="module")
def lane(tmp_path_factory):
    """A frozen synthetic source run (through run-ablations, F8, the trust bundle, a synthetic F9 and F9b, and the release
    manifest) and a synthetic second source run with B12, B17, B18 and a synthetic B22."""

    from conftest import require_module

    for module in METHODS_MODULES:
        require_module(module)
    mp = pytest.MonkeyPatch()
    mp.setattr(stages, "protocol_frozen", lambda *a, **k: False)
    mp.delattr(stages, "require_mms2_unlocked", raising=False)   # without the L1 lock M&Ms-2 is not executed
    tmp = tmp_path_factory.mktemp("postfreeze_v603")
    acdc = fx.make_acdc_source(tmp / "src" / "acdc", fx.acdc_patients(5, testing_per_class=2))
    fx.write_fixture_manifests(acdc, fx.acdc_manifest(acdc))
    cases = fx.mms_refit_cases()
    mms = fx.make_mms_source(tmp / "src" / "mms", cases, shape=(26, 24, 3))
    fx.write_fixture_manifests(mms, fx.mms_manifest(mms, cases), fx.mms_metadata(mms))
    study = _study_copy(tmp)
    _edit_yaml(study.parent / "exploratory_postfreeze.yaml", lambda d: d["statistics"]["bootstrap"].update(replicates=40))

    def fast(d: dict) -> None:
        d["statistics"]["bootstrap"]["replicates"] = 40
        d["statistics"]["random_projections"]["draws"] = 25
        d["statistics"]["permutations"]["draws"] = 25

    _edit_yaml(study.parent / "exploratory_postfreeze_v603.yaml", fast)
    data_args = ["--artifact-root", str(tmp / "artifacts"), "--data-root", str(tmp / "data"), "--acdc-source", str(acdc), "--mms-source", str(mms),
                 "--mms2-source", str(tmp / "src" / "mms2"), "--n-jobs", "1"]
    common = ["--study", str(study), "--run-id", SOURCE, *data_args, "--allow-unfrozen-synthetic"]
    _main(["run", *common, "--until", "run-ablations"])
    _main(["fit-cart-surrogate", *common])
    for split in ("acdc_test", "mms_external"):
        _main(["evaluate-cart-surrogate", *common, "--split", split])
    for partition in ("acdc_dev_oof", "acdc_test", "mms_external"):
        _main(["summarize-predictor", *common, "--partition", partition])
    _main(["fit-trust", *common, "--allow-missing"])
    for split in ("acdc_test", "mms_external"):
        _main(["evaluate-trust", *common, "--split", split, "--allow-missing"])
    source = tmp / "artifacts" / SOURCE
    pctx = pf.make_postfreeze_context(study, SOURCE, "v6-postfreeze-20261028-0100", artifact_root=tmp / "artifacts", data_root=tmp / "data")
    splits = {"acdc_dev": "ACDC", "acdc_test": "ACDC", "mms_external": "MMS"}
    _write_network(pctx, _network(pctx, np.random.default_rng(1), splits), source / "neural_f9", source / "probes" / "f9b" / f"seed{SEED}",
                   evaluations_dir=source / "probes" / "evaluations" / "f9b")
    atomic_write_json(source / "neural_f9" / "f9_bundle.json", {"primary_seed": SEED, "seeds": [SEED], "seed_selection": "forbidden"})
    _main(["write-release-manifest", *common])
    args = ["--study", str(study), "--source-run", SOURCE, *data_args]
    for analysis in ("B12", "B17", "B18"):
        _main(["run-postfreeze", "--analysis", analysis, *args, "--run-id", SECOND, "--allow-unfrozen-synthetic"])
    second = tmp / "artifacts" / SECOND
    b22 = second / "postfreeze" / "B22"
    probe_dir = b22 / "probes" / f"seed{SEED}"
    written = _write_network(pctx, _network(pctx, np.random.default_rng(2), splits), b22 / "inference", probe_dir, b22 / "evaluated")
    probes = [p for p in written if probe_dir in p.parents]                 # as recorded: listed through probes/seed<k>/FROZEN.json only
    atomic_write_json(probe_dir / "FROZEN.json", {"schema": "tmmri.postfreeze_b22_probe/1", "files": {p.name: hash_path(p) for p in probes}})
    atomic_write_json(b22 / "fits" / "FROZEN.json", {"provenance": {"completed_seeds": [SEED]}, "files": {}})
    written = [p for p in written if p not in probes] + [probe_dir / "FROZEN.json", b22 / "fits" / "FROZEN.json"]
    atomic_write_json(b22 / "provenance.json", {
        "label": pf.LABEL, "analysis": "B22", "specification_version": "6.0.2", "run_id": SECOND, "protocol_tag_state": "synthetic_override",
        "source_run": {"run_id": SOURCE, "release_manifest_sha256": hash_path(source / "release_manifest.json")},
        "outputs": {f"run:{p.relative_to(second).as_posix()}": hash_path(p) for p in written}})
    yield SimpleNamespace(tmp=tmp, study=study, source=source, second=second, artifacts=tmp / "artifacts", data=tmp / "data", data_args=data_args,
                          args=args)
    mp.undo()


def _postfreeze(lane, analysis: str, run_id: str, capsys, *extra: str, synthetic: bool = True, args=None):
    argv = ["run-postfreeze", "--analysis", analysis, *(args or lane.args), "--run-id", run_id, *extra]
    return _run([*argv, "--allow-unfrozen-synthetic"] if synthetic else argv, capsys)


def _read(lane, analysis: str, name: str):
    path = lane.artifacts / RUN / "postfreeze" / analysis / name
    return pd.read_parquet(path) if name.endswith(".parquet") else json.loads(path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------------------------- unit checks


def test_d43_outcomes_and_the_lower_outcome():
    assert b23.outcome(0.90, 0.02, 0.10, 0.80) == "O1"
    assert b23.outcome(0.90, -0.02, 0.10, 0.80) == "O2"
    assert b23.outcome(0.90, 0.0, 0.10, 0.80) == "O2"                    # a lower bound of exactly 0 is inside the interval
    assert b23.outcome(0.90, -0.10, -0.01, 0.80) == "O2b"
    assert b23.outcome(0.80, 0.02, 0.10, 0.80) == "O3"                    # at P95 is not above P95
    assert b23.outcome(0.70, -0.10, -0.01, 0.80) == "O3"
    assert b23.outcome(0.90, None, None, 0.80) is None and b23.outcome(None, 0.1, 0.2, 0.8) is None
    assert b23.outcome(0.70, None, None, 0.80) == "O3"                    # O3 needs no interval
    assert b23.lower_outcome({"mms_external": "O1", "mms2_external": "O2b"}) == "O2b"
    assert b23.lower_outcome({"mms_external": "O3", "mms2_external": "O1"}) == "O3"
    assert b23.lower_outcome({"mms_external": "O2", "mms2_external": "O1", "acdc_test": "O3"}) == "O2"   # acdc_test never enters
    assert b23.lower_outcome({"mms_external": "O1"}) is None
    r = b23.reading("acdc_test", 0.9, {"ci_lower": 0.1, "ci_upper": 0.2, "estimate": 0.15, "n_common": 9, "n_pca7": 10},
                    {"p95": 0.5, "fidelity_bhat7_common": 0.85, "n_bhat7": 10, "n_null": 9, "n_common": 9})
    assert r["outcome"] == "O1" and r["enters_reading"] is False and r["fidelity_bhat7"] == 0.85 and r["fidelity_bhat7_own_cases"] == 0.9
    assert (r["n_bhat7"], r["n_common_random7"], r["n_pca7"], r["n_common_delta"]) == (10, 9, 10, 9)


def test_b23_null_comparison_uses_the_cases_both_inputs_predict():
    """bhat7 predicts cases 0-9 and the projections cases 2-11: fidelity(bhat7) and every draw are compared on 2-9."""

    ids = [f"MMS:s{i:02d}" for i in range(12)]
    decision = pd.Series(["A"] * 12, index=ids, dtype=object)
    bhat7 = pd.Series(["A", "A"] + ["B"] * 2 + ["A"] * 6 + [None, None], index=ids, dtype=object)          # 6 of 8 on 2-9; 8 of 10 own
    draw = pd.Series([None, None] + ["A"] * 4 + ["B"] * 6, index=ids, dtype=object)                       # 4 of 8 on 2-9
    other = pd.Series([None, None] + ["A"] * 8 + ["B"] * 2, index=ids, dtype=object)                      # 8 of 8 on 2-9
    cmp = b23.null_comparison(bhat7, [draw, other], decision)
    assert (cmp["n_bhat7"], cmp["n_null"], cmp["n_common"]) == (10, 10, 8)
    assert cmp["fidelity_bhat7_common"] == pytest.approx(6 / 8) and cmp["k"] == [4, 8] and cmp["n"] == [8, 8]
    assert cmp["p95"] == pytest.approx(float(np.quantile([0.5, 1.0], 0.95)))
    with pytest.raises(ValueError, match="different cases"):
        b23.null_comparison(bhat7, [draw, draw.where(draw.index != ids[5], None)], decision)


def test_reconstruction_and_recorded_metric_checks():
    idx = pd.Index(["a", "b", "c"], name="sample_id")
    rec = pd.DataFrame({"t1": [1.0, 2.0, np.nan], "t2": [0.5, 0.25, 1.0], "extra": [0, 0, 0]}, index=idx)
    ok = v603.check_reconstruction(rec[["t1", "t2"]] + 5e-7, rec, "x")
    assert ok["reproduced"] and ok["max_abs_difference"] == pytest.approx(5e-7) and ok["n_recorded_rows"] == 3
    for bad, message in ((rec[["t1", "t2"]] + 2e-6, "maximum absolute difference"), (rec[["t1", "t2"]].fillna(0.0), "missing"),
                         (rec[["t1", "t2"]].iloc[:2], "rows are not recomputed"), (rec[["t1", "t2"]].assign(t3=1.0), "not recorded")):
        with pytest.raises(stages.BundleError, match=message):
            v603.check_reconstruction(bad, rec, "x")
    block = {"rho": 0.5, "rho_j": {"t1": 0.4, "t2": None}, "rho_j_ci": {"t1": [0.1, 0.6]}, "n_cases": 20, "status": "ESTIMATED"}
    assert v603.compare_recorded(block, json.loads(json.dumps(block))) == []
    assert v603.compare_recorded(block, {**block, "rho": 0.5 + 5e-7}) == []
    assert v603.compare_recorded(block, {**block, "rho": 0.5 + 2e-6}) == ["rho: differs by more than 1e-06"]
    assert v603.compare_recorded(block, {**block, "rho_j": {"t1": 0.4, "t2": 0.1}})[0].startswith("rho_j.t2: differs")
    assert v603.compare_recorded(block, {**block, "n_cases": 21}) and v603.compare_recorded(block, {**block, "rho_j_ci": {"t1": [0.1]}})
    assert v603.compare_recorded(block, {k: v for k, v in block.items() if k != "status"})[0].startswith("<root>: keys differ")


def test_b18_reproduction_refuses_any_difference():
    rec = pd.DataFrame([{"input": "bhat7", "partition": "acdc_test", "fidelity": 0.8, "fidelity_k": 8, "fidelity_n": 10},
                        {"input": "bhat9", "partition": "acdc_test", "fidelity": 0.5, "fidelity_k": 5, "fidelity_n": 10}])
    same = {"acdc_test": {"fidelity": {"k": 8, "n": 10, "estimate": 0.8}}}
    assert b23.check_reproduction(rec, same)[0]["reproduced"] is True
    for bad in ({"acdc_test": {"fidelity": {"k": 7, "n": 10, "estimate": 0.7}}}, {"mms_external": {"fidelity": {"k": 8, "n": 10, "estimate": 0.8}}}):
        with pytest.raises(stages.BundleError, match="does not reproduce"):
            b23.check_reproduction(rec, bad)


def test_b25_fold_assignment_is_deterministic():
    clusters = [f"MMS:c{i:02d}" for i in range(23)] + ["MMS:c03", "MMS:c07"]           # repeated clusters keep one fold
    salt = "protocol-v6.0.3/B25"
    folds = b25.fold_assignment(clusters, salt, 5)
    expected = sorted(set(clusters), key=lambda c: hashlib.sha256((salt + c).encode("utf-8")).hexdigest())
    assert folds == {c: i % 5 + 1 for i, c in enumerate(expected)}
    assert b25.fold_assignment(list(reversed(clusters)), salt, 5) == folds
    assert sorted(pd.Series(folds).value_counts().tolist()) == [4, 4, 5, 5, 5]
    assert b25.fold_assignment(clusters, "another salt", 5) != folds


def test_b26_outliers_suppression_and_no_identifiers():
    d = pd.Series([0.0, 1.0, -1.0, 2.0, -2.0, 0.5, 30.0], index=[f"MMS:s{i}" for i in range(7)])
    flag, rule = b26.outliers(d)
    med = float(np.median(d))
    assert rule["threshold"] == pytest.approx(3 * 1.4826 * float(np.median(np.abs(d - med))))
    assert flag.tolist() == [False] * 6 + [True]
    ids = [f"MMS:s{i}" for i in range(14)]
    flag = pd.Series([True] * 7 + [False] * 7, index=ids)
    vendor = pd.Series(["A"] * 5 + ["B"] * 2 + ["A"] * 3 + ["C"] * 2 + ["D"] * 2, index=ids)
    rows = pd.DataFrame(b26.describe_categorical(vendor, flag, 5, partition="mms_external", target="lv_edv_ml"))
    cells = rows.set_index(["group", "category"])["n"].to_dict()
    assert cells == {("outlier", "other"): 7, ("other_cases", "other"): 7}             # A is 3 among the others: merged in both groups
    cont = pd.DataFrame(b26.describe_continuous(pd.Series(np.arange(14.0), index=ids), flag.where(flag.index.isin(ids[:4]), False), 5,
                                                partition="mms_external", target="lv_edv_ml"))
    assert cont.set_index("group").loc["outlier", "status"] == "SUPPRESSED_BELOW_MIN_CELL"
    assert cont.set_index("group").loc["other_cases", "status"] == "DESCRIBED"
    b26.assert_no_identifiers(pd.concat([rows, cont]), ids)
    with pytest.raises(RuntimeError, match="identifier"):
        b26.assert_no_identifiers(pd.DataFrame({"sample_id": ids}), ids)
    with pytest.raises(RuntimeError, match="identifier"):
        b26.assert_no_identifiers(pd.DataFrame({"case": ["s3"]}), ids)


def test_b26_both_groups_share_the_merged_categories_and_the_suppression():
    def table(outlier_values: str, other_values: str) -> pd.DataFrame:
        vals = list(outlier_values) + list(other_values)
        ids = [f"MMS:s{i:03d}" for i in range(len(vals))]
        flag = pd.Series([True] * len(outlier_values) + [False] * len(other_values), index=ids)
        return pd.DataFrame(b26.describe_categorical(pd.Series(vals, index=ids), flag, 5, partition="mms_external", target="lv_edv_ml"))

    # B (2) and C (3) are small among the outliers: merged in both groups, so the outliers' B is not total B minus the others' B.
    rows = table("A" * 5 + "B" * 2 + "C" * 3, "A" * 10 + "B" * 10 + "C" * 10)
    assert rows.set_index(["group", "category"])["n"].to_dict() == {("outlier", "A"): 5, ("outlier", "other"): 5, ("other_cases", "A"): 10,
                                                                    ("other_cases", "other"): 20}
    # 'other' is small in the comparison group only: the attribute is suppressed for both groups.
    rows = table("A" * 5 + "B" * 6, "A" * 10 + "B" * 10 + "C" * 3)
    assert set(rows["status"]) == {"SUPPRESSED_BELOW_MIN_CELL"} and rows["n"].isna().all() and set(rows["group"]) == {"outlier", "other_cases"}
    # A zero cell is not a small cell, and it is not listed.
    rows = table("A" * 5, "A" * 7 + "B" * 9)
    assert rows.set_index(["group", "category"])["n"].to_dict() == {("outlier", "A"): 5, ("other_cases", "A"): 7, ("other_cases", "B"): 9}


def test_b25_folds_cover_every_mms_cluster_before_exclusions():
    salt = "protocol-v6.0.3/B25"
    ids = [f"MMS:p{i:02d}" for i in range(12)]
    manifest = pd.DataFrame({"sample_id": [*ids, "ACDC:patient001"], "partition": ["mms_external"] * 12 + ["acdc_dev"],
                             "cluster_id": ["", *ids[1:10], "MMS:pair", "MMS:pair", "ACDC:patient001"]})
    folds = b25.mms_fold_assignment(manifest, salt, 5)
    assert folds == b25.fold_assignment([*ids[:10], "MMS:pair"], salt, 5)          # a case without cluster_id is its own cluster
    assert "ACDC:patient001" not in folds and len(folds) == 11


def test_paired_bootstrap_resamples_both_inputs_together():
    ids = [f"MMS:s{i:02d}" for i in range(30)]
    clusters = pd.Series(ids, index=ids)
    decision = pd.Series(["A"] * 30, index=ids, dtype=object)
    boot = {"replicates": 200, "seed": 20260928, "level": 0.95}
    same = v603.paired_difference(decision, decision.copy(), decision, clusters, boot, 5)
    assert same["estimate"] == 0 and same["ci_lower"] == 0 and same["ci_upper"] == 0                # unpaired draws would spread around 0
    one_off = decision.copy()
    one_off.iloc[0] = "B"
    diff = v603.paired_difference(decision, one_off, decision, clusters, boot, 5)
    assert diff["estimate"] == pytest.approx(1 / 30) and diff["ci_lower"] >= 0                     # every paired replicate is >= 0


def test_b27_fold_spread_on_a_known_example():
    idx = pd.Index(["a", "b"], name="sample_id")
    folds = [pd.DataFrame({"t1": [v, 1.0], "t2": [2.0 * v, np.nan if k == 2 else 1.0]}, index=idx) for k, v in enumerate([1.0, 2.0, 3.0, 4.0, 5.0])]
    spread = b27.fold_spread(folds, pd.Series({"t1": 2.0, "t2": 1.0}), ["t1", "t2"])
    sd = float(np.std([1, 2, 3, 4, 5], ddof=1))
    assert spread.loc["a", "t1"] == pytest.approx(sd / 2.0) and spread.loc["a", "t2"] == pytest.approx(2 * sd)
    assert spread.loc["b", "t1"] == 0.0 and np.isnan(spread.loc["b", "t2"])                   # a missing fold value: no signal
    assert spread.loc["a", "any_large"] == pytest.approx(2 * sd) and spread.loc["b", "any_large"] == 0.0
    err = pd.DataFrame({"t1": np.arange(10.0), "t2": np.arange(10.0)[::-1]})
    labels, thresholds = b27.within_partition_labels(err)
    assert labels["t1"].sum() == 1 and thresholds["t1"] == pytest.approx(8.1) and labels["any_large"].sum() == 2


# ----------------------------------------------------------------------------------------------- lane on the synthetic runs


def test_v603_lock_and_second_source(lane, capsys):
    """Run-id template, the fixed source runs, the protocol-v6.0.3 tag, no write below either source run."""

    code, out = _postfreeze(lane, "B26", "v6-other-20261029-1300", capsys)
    assert code == cli.EXIT_USAGE_OR_INPUT and "must match" in out
    code, out = _postfreeze(lane, "B26", SECOND, capsys)
    assert code == cli.EXIT_USAGE_OR_INPUT and "new run root" in out
    code, out = _postfreeze(lane, "B26", RUN, capsys, synthetic=False)
    assert code == cli.EXIT_PROTOCOL_NOT_FROZEN and "source run is fixed" in out
    path = lane.study.parent / "exploratory_postfreeze_v603.yaml"
    _edit_yaml(path, lambda d: d["source_runs"]["frozen"].update(run_id=SOURCE))
    try:
        code, out = _postfreeze(lane, "B26", RUN, capsys, synthetic=False)
        assert code == cli.EXIT_PROTOCOL_NOT_FROZEN and "protocol-v6.0.3" in out
    finally:
        _edit_yaml(path, lambda d: d["source_runs"]["frozen"].update(run_id="v6-replication-20261003-1241"))
    assert not (lane.artifacts / RUN).exists()
    pctx = v603.make_v603_context(lane.study, SOURCE, RUN, artifact_root=lane.artifacts, data_root=lane.data, allow_unfrozen_synthetic=True)
    for target in (lane.second / "postfreeze" / "B99" / "x.json", lane.source / "x.json", lane.artifacts / "elsewhere" / "x.json"):
        with pytest.raises(PermissionError):
            pctx.require_new_output(target)
    assert pctx.portable(lane.second / "postfreeze" / "B18" / "summary.json") == "postfreeze_v602:postfreeze/B18/summary.json"


def test_changed_second_source_file_refuses(lane, capsys):
    tampered = lane.tmp / "tampered_second"
    for name in (SOURCE, SECOND):
        shutil.copytree(lane.artifacts / name, tampered / name)
    args = [a if a != str(lane.artifacts) else str(tampered) for a in lane.args]
    path = tampered / SECOND / "postfreeze" / "B17" / "summary.json"
    path.write_bytes(path.read_bytes() + b" ")
    code, out = _postfreeze(lane, "B27", RUN, capsys, args=args)
    assert code == cli.EXIT_BUNDLE and "differs from the SHA-256" in out and "postfreeze_v602" in out
    probe = f"postfreeze/B22/probes/seed{SEED}/transition_final.json"           # recorded through probes/seed<k>/FROZEN.json
    good = v603.make_v603_context(lane.study, SOURCE, RUN, artifact_root=lane.artifacts, data_root=lane.data, allow_unfrozen_synthetic=True)
    assert good.second_file(probe) == lane.second / probe and f"postfreeze_v602:postfreeze/B22/probes/seed{SEED}/FROZEN.json" in good.verified_log
    with pytest.raises(stages.BundleError, match="recorded neither"):
        good.second_file("postfreeze/B22/unlisted.json")
    (tampered / SECOND / probe).write_bytes((tampered / SECOND / probe).read_bytes() + b" ")
    bad = v603.make_v603_context(lane.study, SOURCE, RUN, artifact_root=tampered, data_root=lane.data, allow_unfrozen_synthetic=True)
    with pytest.raises(stages.BundleError, match="FROZEN.json"):
        bad.second_file(probe)


def test_b23_refuses_when_b18_is_not_reproduced(lane, capsys):
    tampered = lane.tmp / "tampered_b18"
    for name in (SOURCE, SECOND):
        shutil.copytree(lane.artifacts / name, tampered / name)
    args = [a if a != str(lane.artifacts) else str(tampered) for a in lane.args]
    table_path = tampered / SECOND / "postfreeze" / "B18" / "decision_link_table.parquet"
    table = pd.read_parquet(table_path)
    row = (table["input"] == "bhat7") & (table["partition"] == "acdc_test")
    table.loc[row, "fidelity_k"] = table.loc[row, "fidelity_k"] - 1
    table.loc[row, "fidelity"] = table.loc[row, "fidelity_k"] / table.loc[row, "fidelity_n"]
    table.to_parquet(table_path, index=False)
    prov_path = tampered / SECOND / "postfreeze" / "B18" / "provenance.json"
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    prov["outputs"]["run:postfreeze/B18/decision_link_table.parquet"] = hash_path(table_path)
    prov_path.write_text(json.dumps(prov), encoding="utf-8")
    code, out = _postfreeze(lane, "B23", RUN, capsys, args=args)
    assert code == cli.EXIT_BUNDLE and "does not reproduce the recorded B18" in out
    assert not (tampered / RUN / "postfreeze" / "B23" / "provenance.json").exists()
    assert not (tampered / RUN / "postfreeze" / "B23" / "reading.json").exists()


def _tampered(lane, name: str):
    """A copy of both source runs under ``name`` and the CLI arguments that point at it."""

    tampered = lane.tmp / name
    for run_id in (SOURCE, SECOND):
        shutil.copytree(lane.artifacts / run_id, tampered / run_id)
    return tampered, [a if a != str(lane.artifacts) else str(tampered) for a in lane.args]


def _rehash_second(tampered, analysis: str, rel: str) -> None:
    prov_path = tampered / SECOND / "postfreeze" / analysis / "provenance.json"
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    prov["outputs"][f"run:{rel}"] = hash_path(tampered / SECOND / rel)
    prov_path.write_text(json.dumps(prov), encoding="utf-8")


def _rehash_source(tampered, rel: str) -> None:
    """Record a changed source-run file in release_manifest.json and the new manifest hash in every second-run provenance."""

    manifest_path = tampered / SOURCE / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert rel in manifest["files"]
    manifest["files"][rel] = hash_path(tampered / SOURCE / rel)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    for prov_path in (tampered / SECOND / "postfreeze").glob("*/provenance.json"):
        prov = json.loads(prov_path.read_text(encoding="utf-8"))
        prov["source_run"]["release_manifest_sha256"] = hash_path(manifest_path)
        prov_path.write_text(json.dumps(prov), encoding="utf-8")


@pytest.mark.parametrize("record", ["f9b_metrics", "f9bm_bhat"])
def test_b24_refuses_when_the_recorded_reconstruction_is_not_reproduced(lane, capsys, record):
    """analyses.B24.reconstruction_check: a recorded F9b metric or F9b-M B-hat off by 1e-5 (above 1e-6) refuses."""

    tampered, args = _tampered(lane, f"tampered_{record}")
    if record == "f9b_metrics":
        rel = "probes/evaluations/f9b/acdc_test.json"
        payload = json.loads((tampered / SOURCE / rel).read_text(encoding="utf-8"))
        rho = payload["representations"]["f9b"]["seeds"][str(SEED)]["rho"]
        if isinstance(rho.get("rho"), float):
            rho["rho"] += 1e-5
        else:
            rho["n_cases"] += 1
        (tampered / SOURCE / rel).write_text(json.dumps(payload), encoding="utf-8")
        _rehash_source(tampered, rel)
        expected = "does not reproduce the recorded evaluation metrics"
    else:
        rel = f"postfreeze/B22/evaluated/bhat_acdc_test_seed{SEED}.parquet"
        frame = pd.read_parquet(tampered / SECOND / rel)
        frame.loc[0, "lv_edv_ml"] = frame.loc[0, "lv_edv_ml"] + 1e-5
        frame.to_parquet(tampered / SECOND / rel, index=False)
        _rehash_second(tampered, "B22", rel)
        expected = "does not reproduce the recorded one"
    code, out = _postfreeze(lane, "B24", RUN, capsys, args=args)
    assert code == cli.EXIT_BUNDLE and expected in out and "refused" in out
    run = tampered / RUN / "postfreeze" / "B24"
    assert not (run / "provenance.json").exists() and not (run / "coverage_fidelity_table.parquet").exists()


def test_b27_refuses_when_the_b17_thresholds_are_not_reproduced(lane, capsys):
    tampered, args = _tampered(lane, "tampered_b17")
    rel = "postfreeze/B17/summary.json"
    summary = json.loads((tampered / SECOND / rel).read_text(encoding="utf-8"))
    first = sorted(summary["development_thresholds"])[0]
    summary["development_thresholds"][first] += 1e-9
    (tampered / SECOND / rel).write_text(json.dumps(summary), encoding="utf-8")
    _rehash_second(tampered, "B17", rel)
    code, out = _postfreeze(lane, "B27", RUN, capsys, args=args)
    assert code == cli.EXIT_BUNDLE and "B17 development thresholds" in out and "refused" in out
    assert not (tampered / RUN / "postfreeze" / "B27" / "spread_auroc_table.parquet").exists()


def _ctx(lane, artifacts=None, run_id: str = "v6-postfreeze-20261029-1900"):
    return v603.make_v603_context(lane.study, SOURCE, run_id, artifact_root=artifacts or lane.artifacts, data_root=lane.data,
                                  allow_unfrozen_synthetic=True)


def _candidates(lane, pctx) -> list[str]:
    from tmmri.wedd import CutSet

    cuts = CutSet.from_dict(json.loads((lane.source / "thresholds" / "d_model_wedd_gaussian_mm_v1.json").read_text(encoding="utf-8")))
    return [c for c in pctx.study.rule_candidates() if c in cuts.features]


def test_b25_acdc_dev_folds_are_the_frozen_gate(lane):
    """In every outer fold the acdc_dev rows take the recorded nested inner OOF B-hat (training folds), the outer OOF
    B-hat (held out) and the decisions of the predictor without fold f, which reproduce the recorded cross-fitted ones."""

    pctx = _ctx(lane)
    candidates = _candidates(lane, pctx)
    dev, fold_predictor, substrate = b25.frozen_gate_parts(pctx, candidates, 5)
    ids, folds = dev.index.to_numpy(), dev["fold"].to_numpy()
    oof = pd.read_parquet(lane.source / "oof" / "bhat_oof_ACDC_dev.parquet").set_index("sample_id")
    cross = pd.read_parquet(lane.source / "formal" / "d_model_crossfit_ACDC_dev.parquet").set_index("sample_id")["d_model_crossfit"]
    final = pd.read_parquet(lane.source / "formal" / "d_model_ACDC_dev.parquet").set_index("sample_id")["d_model"]
    assert dev["decision"].tolist() == final.reindex(ids).tolist()                          # the consequent of the frozen rule layer
    for f in range(1, 6):
        train, held = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        inner = pd.read_parquet(lane.source / "oof" / f"inner_oof_ACDC_dev_fold{f}.parquet").set_index("sample_id")
        tr, he = substrate(f, train, held)
        assert tr.index.tolist() == ids[train].tolist() and he.index.tolist() == ids[held].tolist()
        assert np.array_equal(tr[candidates].to_numpy(float), inner.loc[ids[train], candidates].to_numpy(float), equal_nan=True)
        assert not np.allclose(tr[candidates].to_numpy(float), oof.loc[ids[train], candidates].to_numpy(float), equal_nan=True)
        assert np.array_equal(he[candidates].to_numpy(float), oof.loc[ids[held], candidates].to_numpy(float), equal_nan=True)
        assert fold_predictor(train, held).tolist() == cross.reindex(ids[held]).tolist()
    tampered, _ = _tampered(lane, "tampered_inner_oof")
    (tampered / SOURCE / "oof" / "inner_oof_ACDC_dev_fold3.parquet").unlink()
    with pytest.raises(stages.BundleError, match="nothing is refitted"):
        b25.frozen_gate_parts(_ctx(lane, tampered), candidates, 5)


def test_b25_gate_wiring_and_unstable_cells(lane, capsys, monkeypatch):
    """The pooled gate in fold 1: acdc_dev rows from the frozen gate, mms_external rows with their frozen-bundle B-hat and
    final d_model; an UNSTABLE_ESTIMATE cell satisfies no literal."""

    import tmmri.rough_sets as rough_sets
    from tmmri.wedd import CutSet, is_available_state

    seen: dict = {}
    original = rough_sets.cross_fitted_gate

    def spy(values, decision_fn, folds, *args, substrate_fn=None, **kwargs):
        if not seen:
            train, held = np.flatnonzero(np.isfinite(folds) & (folds != 1)), np.flatnonzero(folds == 1)
            seen.update(values=values, sub=substrate_fn(1, train, held), d_train=decision_fn(train, train), d_held=decision_fn(train, held),
                        train=train, held=held)
        return original(values, decision_fn, folds, *args, substrate_fn=substrate_fn, **kwargs)

    consequent: dict = {}
    original_induce, original_ledger = rough_sets.induce_vprs_rules, b25.ledger

    def induce_spy(states, decision, *args, **kwargs):
        consequent.setdefault("discovery", pd.Series(np.asarray(decision, dtype=object), index=states.index))    # the first call: discovery
        return original_induce(states, decision, *args, **kwargs)

    def ledger_spy(**kwargs):
        consequent["ledger"] = kwargs["decision"]
        return original_ledger(**kwargs)

    monkeypatch.setattr(rough_sets, "cross_fitted_gate", spy)
    monkeypatch.setattr(rough_sets, "induce_vprs_rules", induce_spy)
    monkeypatch.setattr(b25, "ledger", ledger_spy)
    code, res = _postfreeze(lane, "B25", "v6-postfreeze-20261029-1910", capsys)
    assert code == 0, res
    # One consequent, as the frozen rule layer: the final d_model (acdc_dev in sample) in discovery and in the ledger.
    final = pd.read_parquet(lane.source / "formal" / "d_model_ACDC_dev.parquet").set_index("sample_id")["d_model"]
    d_final_mms = pd.read_parquet(lane.source / "inference" / "d_model_mms_external.parquet").set_index("sample_id")["d_model"]
    for name in ("discovery", "ledger"):
        dec = consequent[name]
        dev_rows = dec.index.str.startswith("ACDC:")
        assert dev_rows.any() and (~dev_rows).any(), name
        assert dec[dev_rows].astype(str).tolist() == final.reindex(dec.index[dev_rows]).astype(str).tolist(), name
        assert dec[~dev_rows].astype(str).tolist() == d_final_mms.reindex(dec.index[~dev_rows]).astype(str).tolist(), name
    values, (tr, he) = seen["values"], seen["sub"]
    candidates = [c for c in values.columns if not str(c).endswith("__state")]
    is_dev = lambda frame: frame.index.str.startswith("ACDC:")                                        # noqa: E731
    inner = pd.read_parquet(lane.source / "oof" / "inner_oof_ACDC_dev_fold1.parquet").set_index("sample_id")
    oof = pd.read_parquet(lane.source / "oof" / "bhat_oof_ACDC_dev.parquet").set_index("sample_id")
    assert is_dev(tr).any() and (~is_dev(tr)).any() and is_dev(he).any() and (~is_dev(he)).any()
    assert np.array_equal(tr.loc[is_dev(tr), candidates].to_numpy(float), inner.loc[tr.index[is_dev(tr)], candidates].to_numpy(float), equal_nan=True)
    assert np.array_equal(he.loc[is_dev(he), candidates].to_numpy(float), oof.loc[he.index[is_dev(he)], candidates].to_numpy(float), equal_nan=True)
    for frame in (tr, he):
        mms_rows = frame.loc[~is_dev(frame)]
        assert mms_rows.astype(str).equals(values.loc[mms_rows.index].astype(str))
    cross = pd.read_parquet(lane.source / "formal" / "d_model_crossfit_ACDC_dev.parquet").set_index("sample_id")["d_model_crossfit"]
    d_mms = pd.read_parquet(lane.source / "inference" / "d_model_mms_external.parquet").set_index("sample_id")["d_model"]
    held_ids = values.index[seen["held"]]
    assert [str(d) for d in seen["d_held"]] == [str(cross[s] if s.startswith("ACDC:") else d_mms[s]) for s in held_ids]
    train_ids = values.index[seen["train"]]
    assert [str(d) for d, s in zip(seen["d_train"], train_ids) if not s.startswith("ACDC:")] == [str(d_mms[s]) for s in train_ids if not s.startswith("ACDC:")]
    # An UNSTABLE_ESTIMATE cell gives no literal; the same value without the flag does.
    cuts = CutSet.from_dict(json.loads((lane.source / "thresholds" / "d_model_wedd_gaussian_mm_v1.json").read_text(encoding="utf-8")))
    c = candidates[0]
    idx = pd.Index(["MMS:a", "MMS:b"], name="sample_id")
    v = float(values[c].dropna().median())
    states = cuts.transform(b25.admission_states(pd.DataFrame({c: [v, v]}, index=idx),
                                                 pd.DataFrame({c: ["UNSTABLE_ESTIMATE", "STABLE"]}, index=idx), [c]))
    assert not is_available_state(states.loc["MMS:a", c]) and is_available_state(states.loc["MMS:b", c])


def test_b25_evaluation_path_on_a_stand_in_partition(lane, capsys, monkeypatch):
    """M&Ms-2 is locked in the synthetic run, so the B25 evaluation code runs once on acdc_test as a stand-in (algorithm
    control only; the lane keeps mms2_external as the only evaluation partition)."""

    path = lane.study.parent / "exploratory_postfreeze_v603.yaml"
    monkeypatch.setattr(b25, "EVALUATION", ("acdc_test",))
    _edit_yaml(path, lambda d: d["analyses"]["B25"].update(partitions=["acdc_test"]))
    try:
        code, res = _postfreeze(lane, "B25", "v6-postfreeze-20261029-1400", capsys)
    finally:
        _edit_yaml(path, lambda d: d["analyses"]["B25"].update(partitions=["mms2_external"]))
    assert code == 0, res
    run = lane.artifacts / "v6-postfreeze-20261029-1400" / "postfreeze" / "B25"
    table = pd.read_parquet(run / "coverage_fidelity_mms2.parquet")
    assert table["partition"].tolist() == ["acdc_test"] and table["status"].iloc[0] in ("EVALUATED", "EMPTY_RULEBOOK")
    assert table["n_processed"].iloc[0] > 0 and 0 <= table["coverage"].iloc[0] <= 1
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["checks"] == [{"partition": "acdc_test", "per_object_stability_equals_recorded": True}]


@pytest.mark.parametrize("analysis", ORDER)
def test_every_analysis_on_the_synthetic_runs(lane, capsys, analysis):
    before = {name: _tree(lane.artifacts / name) for name in (SOURCE, SECOND)}
    others = {p.relative_to(lane.artifacts).as_posix() for p in lane.artifacts.rglob("*") if p.is_file() and RUN not in p.relative_to(lane.artifacts).parts}
    data_before = _tree(lane.data)
    code, res = _postfreeze(lane, analysis, RUN, capsys)
    assert code == 0, res
    assert {name: _tree(lane.artifacts / name) for name in (SOURCE, SECOND)} == before                 # both source runs untouched
    after = {p.relative_to(lane.artifacts).as_posix() for p in lane.artifacts.rglob("*") if p.is_file() and RUN not in p.relative_to(lane.artifacts).parts}
    assert after == others and _tree(lane.data) == data_before                                      # nothing written outside the new run root
    prov = _read(lane, analysis, "provenance.json")
    assert prov["label"] == pf.LABEL and prov["analysis"] == analysis and prov["specification_version"] == "6.0.3"
    assert prov["protocol_tag"] == "protocol-v6.0.3" and prov["config"]["file"] == "configs/exploratory_postfreeze_v603.yaml"
    assert prov["source_run"]["run_id"] == SOURCE and prov["source_runs"]["postfreeze_v602"]["run_id"] == SECOND
    assert all(k.startswith(f"run:postfreeze/{analysis}/") for k in prov["outputs"])
    for out_path, sha in prov["outputs"].items():
        assert hashlib.sha256((lane.artifacts / RUN / out_path.split(":", 1)[1]).read_bytes()).hexdigest() == sha
    summary = _read(lane, analysis, "summary.json")
    CHECKS[analysis](lane, summary, prov)


def _b23(lane, summary, prov):
    assert any(k.startswith("postfreeze_v602:postfreeze/B18/") for k in prov["inputs"])
    reproduction = _read(lane, "B23", "b18_reproduction.parquet")
    assert reproduction["reproduced"].all() and set(reproduction["partition"]) == {"acdc_dev_oof", "acdc_test", "mms_external"}
    table = _read(lane, "B23", "null_fidelity_table.parquet")
    assert set(table["input"]) == {"bhat7", "pca7", "formal57", "constant", "random7", "permuted_bhat7"}
    assert set(table["partition"]) == {"acdc_test", "mms_external"}
    nulls = _read(lane, "B23", "null_distributions.parquet")
    assert nulls.groupby(["partition", "null"]).size().eq(25).all()
    reading = _read(lane, "B23", "reading.json")
    assert reading["decision"] == "D43" and reading["status"] == "NOT_DETERMINED" and "mms2_external" in reading["not_executed"]
    assert reading["partitions"]["mms_external"]["outcome"] in b23.ORDER and "acdc_test" in reading["descriptive_only"]
    row = reading["partitions"]["mms_external"]
    assert 0 < row["n_common_random7"] <= min(row["n_bhat7"], row["n_random7"]) and 0 < row["n_common_delta"] <= min(row["n_bhat7"], row["n_pca7"])
    delta = _read(lane, "B23", "delta_table.parquet")
    assert set(delta["partition"]) == {"acdc_test", "mms_external"} and delta["n_common"].gt(0).all()
    run = lane.artifacts / RUN / "postfreeze" / "B23" / "models"
    for estimator, frozen in summary["frozen"].items():
        for name, sha in frozen["files"].items():
            assert hashlib.sha256((run / estimator / name).read_bytes()).hexdigest() == sha
    projections = np.load(run / "lr_decision_link_random7_x1000" / "projections.npy")
    rng = np.random.default_rng(20261028)                                              # D46: drawn in order
    assert np.array_equal(projections, np.stack([rng.standard_normal((57, 7)) for _ in range(25)]))
    permutations = np.load(run / "lr_decision_link_permuted_bhat7_x1000" / "permutations.npy")
    rng = np.random.default_rng(20261029)
    assert np.array_equal(permutations, np.stack([rng.permutation(permutations.shape[1]) for _ in range(25)]))
    # PCA-7 from the acdc_dev rows only.
    from tmmri.formal_features import formal_descriptor_names

    pctx = _ctx(lane)
    dev = pf.load_partition(pctx, "acdc_dev")
    a_dev = pf.formal_a(pctx, "ACDC").reindex(dev.ids)[formal_descriptor_names()].to_numpy(dtype=float)
    scaler = v603.standardizer(a_dev)
    expected = pf.clean({"standardizer": scaler, "pca": v603.principal_components(v603.standardize(a_dev, scaler), 7)})
    frozen_pca = json.loads((run / "pca7_formal57" / "pca.json").read_text(encoding="utf-8"))
    assert v603.compare_recorded({k: frozen_pca[k] for k in expected}, expected, 1e-12) == []


def _b24(lane, summary, prov):
    coverage = _read(lane, "B24", "coverage_fidelity_table.parquet")
    assert set(coverage["network"]) == {"f9", "f9m"} and set(coverage["partition"]) == {"acdc_test", "mms_external"}
    assert coverage["coverage"].dropna().between(0, 1).all() and (coverage["n_processed"] > 0).all()
    ledger = _read(lane, "B24", "candidate_ledger.parquet")
    order = summary["ledger"]["f9"]["pipeline_order"]
    assert len(ledger) and set(ledger["rulebook"]) <= {"f9", "f9m"} and set(ledger["first_failed_condition"]) <= {*order, PASSED}
    counts = _read(lane, "B24", "rejection_counts_by_reason.parquet")
    assert int(counts["n_candidates"].sum()) == len(ledger)
    concepts = _read(lane, "B24", "concept_table.parquet")
    assert set(concepts["network"]) == {"f9", "f9m"}
    link = _read(lane, "B24", "decision_link_table.parquet")
    assert {"probe7", "pca7", "random7", "probe7_minus_pca7"} <= set(link["input"])
    assert (coverage["per_object_stability"] == b24.NO_PER_OBJECT_STABILITY).all()                   # every coverage row states it
    checks = {(c["network"], c["partition"]): c["reconstruction_check"] for c in summary["checks"]}
    assert set(checks) == {(n, p) for n in ("f9", "f9m") for p in ("acdc_test", "mms_external")} and all(c["reproduced"] for c in checks.values())
    assert checks[("f9m", "mms_external")]["max_abs_difference"] < 1e-9 and checks[("f9", "acdc_test")]["metric_code"].endswith("rho_block")
    assert any(k == "source:probes/evaluations/f9b/acdc_test.json" for k in prov["inputs"])
    assert any(k.startswith("postfreeze_v602:postfreeze/B22/") for k in prov["inputs"]) and any(k.startswith("source:neural_f9/") for k in prov["inputs"])


def _b25(lane, summary, prov):
    profile = "wedd_gaussian_mm_v1"
    assert summary["frozen"]["cuts_sha256"] == hash_path(lane.source / "thresholds" / f"d_model_{profile}.json")
    assert summary["frozen"]["cuts_equal_recorded"] is True
    counts = _read(lane, "B25", "admission_counts.parquet").set_index("source")
    assert counts.loc["acdc_dev", "n_cases"] > 0 and counts.loc["mms_external", "n_cases"] > 0
    assert counts.loc["pooled", "n_cases"] == counts.loc["acdc_dev", "n_cases"] + counts.loc["mms_external", "n_cases"]
    coverage = _read(lane, "B25", "coverage_fidelity_mms2.parquet")
    assert set(coverage["partition"]) == {"mms2_external"} and set(coverage["status"]) == {"NOT_EXECUTED"}   # M&Ms-2 locked here
    assert not any("routes_" in k or "_decoded" in k for k in prov["inputs"])                                     # mms_external never evaluated
    # D46 folds over every mms_external cluster of the split manifest; the per-fold counts of the admitted cases follow them.
    pctx = _ctx(lane)
    folds = b25.mms_fold_assignment(pd.read_parquet(lane.source / "contracts" / "split_manifest.parquet"), "protocol-v6.0.3/B25", 5)
    mms = pf.load_partition(pctx, "mms_external")
    d_mms = pd.read_parquet(lane.source / "inference" / "d_model_mms_external.parquet").set_index("sample_id")["d_model"]
    admitted = [s for s in mms.ids if pd.notna(d_mms.get(s))]
    expected = pd.Series([folds[str(mms.clusters[s])] for s in admitted]).value_counts()
    assert all(int(counts.loc["mms_external", f"n_fold{f}"]) == int(expected.get(f, 0)) for f in range(1, 6))
    assert counts.loc["mms_external", "n_cases"] == len(admitted)
    # Recorded UNSTABLE_ESTIMATE cells of mms_external are counted (and satisfy no literal: test_b25_gate_wiring_and_unstable_cells).
    unstable = pd.read_parquet(lane.source / "reconstructed" / "stability_d_model_mms_external.parquet").set_index("sample_id")
    cols = [c for c in _candidates(lane, pctx) if c in unstable.columns]
    assert counts.loc["mms_external", "n_unstable_cells"] == int((unstable.reindex(admitted)[cols] == "UNSTABLE_ESTIMATE").to_numpy().sum())
    assert counts.loc["acdc_dev", "n_unstable_cells"] == 0
    # acdc_dev in the gate folds: the frozen gate's substrate and decisions.
    assert summary["gate_source"]["fold_decisions_reproduced"] is True
    assert {f"source:oof/inner_oof_ACDC_dev_fold{f}.parquet" for f in range(1, 6)} <= set(prov["inputs"])
    ledger = _read(lane, "B25", "candidate_ledger.parquet")
    assert set(ledger["first_failed_condition"]) <= {*summary["ledger"]["pipeline_order"], PASSED}


def _b26(lane, summary, prov):
    split = pd.read_parquet(lane.source / "contracts" / "split_manifest.parquet")
    ids = split["sample_id"].astype(str).tolist()
    tokens = {t for i in ids for t in (i, i.split(":", 1)[-1])}
    run = lane.artifacts / RUN / "postfreeze" / "B26"
    for name in ("robust_agreement_table.parquet", "outlier_counts.parquet", "outlier_description.parquet", "geometry_check.parquet"):
        frame = pd.read_parquet(run / name)
        assert not set(frame.columns) & set(b26.IDENTIFIER_COLUMNS), name
        assert not set(frame.astype(str).to_numpy().ravel()) & tokens, name
    assert not any(json.dumps(t) in (run / "summary.json").read_text(encoding="utf-8") for t in tokens)
    agreement = pd.read_parquet(run / "robust_agreement_table.parquet")
    assert set(agreement["partition"]) == {"acdc_test", "mms_external"} and agreement["median_bias"].notna().all()
    description = pd.read_parquet(run / "outlier_description.parquet")
    described = description[description["status"] == "DESCRIBED"]
    assert (described["n_with_value"] >= 5).all() and described.loc[described["kind"] == "categorical", "n"].ge(5).all()
    geometry = pd.read_parquet(run / "geometry_check.parquet")
    assert (geometry["n_still_outlier_under_geometry"] + geometry["n_no_longer_outlier"] + geometry["n_not_scored_under_geometry"]
            == geometry["n_outliers"]).all()


def _b27(lane, summary, prov):
    auroc = _read(lane, "B27", "spread_auroc_table.parquet")
    assert set(auroc["definition"]) == {"development_b17", "within_partition_top_decile"} and set(auroc["partition"]) == {"acdc_test", "mms_external"}
    assert summary["development_thresholds_reproduced"] is True
    spearman = _read(lane, "B27", "spread_spearman_table.parquet")
    assert len(spearman) == 2 * 7
    reread = _read(lane, "B27", "b17_reread_table.parquet")
    assert set(reread["signal"]) == {"uncertainty", "ood_score", "unstable_bins"}


CHECKS = {"B23": _b23, "B24": _b24, "B25": _b25, "B26": _b26, "B27": _b27}


# ----------------------------------------------------------------------------------------------- B24 decision link (after RUN)


def test_b24_decision_link_is_fitted_to_the_out_of_fold_decisions(lane, capsys, monkeypatch):
    """On acdc_dev the probe7, PCA-7 and random7 links are fitted to the network's out-of-fold decisions (the consequent),
    not to the final network's in-sample decisions."""

    calls = []
    original = b24.fit_link
    monkeypatch.setattr(b24, "fit_link", lambda x, y, pcfg: calls.append(pd.Series(np.asarray(y, dtype=object)).astype(str).tolist())
                        or original(x, y, pcfg))
    code, res = _postfreeze(lane, "B24", "v6-postfreeze-20261029-1920", capsys)
    assert code == 0, res
    dev_ids = pf.load_partition(_ctx(lane), "acdc_dev").ids
    expected = {}
    for key, path in (("f9", lane.source / "neural_f9"), ("f9m", lane.second / "postfreeze" / "B22" / "inference")):
        pred = pd.read_parquet(path / "predictions_acdc_dev.parquet")
        pred = pred[pred["seed"] == SEED]
        oof, _ = v603.argmax_decisions(pred[pred["is_oof"].astype(bool)])
        final, _ = v603.argmax_decisions(pred[pred["encoder_id"] == "final"])
        expected[key] = oof.reindex(dev_ids).astype(object).astype(str).tolist()
        assert expected[key] != final.reindex(dev_ids).astype(object).astype(str).tolist()           # the two targets differ here
    per_network = 2 + 25                                                                              # probe7, pca7, 25 random draws
    assert len(calls) == 2 * per_network
    assert all(y == expected["f9"] for y in calls[:per_network]) and all(y == expected["f9m"] for y in calls[per_network:])


def test_b24_decision_link_excludes_critical_qc_cases(lane, capsys, monkeypatch):
    """One more case outside load_partition (as a critical-QC case would be) leaves the link one case shorter and the rule
    coverage unchanged."""

    import dataclasses

    original = b24.load_partition

    def fewer(pctx, split):
        part = original(pctx, split)
        return part if split == "acdc_dev" else dataclasses.replace(part, ids=list(part.ids)[1:])

    monkeypatch.setattr(b24, "load_partition", fewer)
    run_id = "v6-postfreeze-20261029-1930"
    code, res = _postfreeze(lane, "B24", run_id, capsys)
    assert code == 0, res
    out = lane.artifacts / run_id / "postfreeze" / "B24"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    base = {(c["network"], c["partition"]): c for c in _read(lane, "B24", "summary.json")["checks"]}
    link = pd.read_parquet(out / "decision_link_table.parquet")
    for c in summary["checks"]:
        ref = base[(c["network"], c["partition"])]
        assert c["n_rule_cases"] == ref["n_rule_cases"] and c["n_link_cases"] == ref["n_link_cases"] - 1 <= c["n_rule_cases"] - 1
        rows = link[(link["network"] == c["network"]) & (link["partition"] == c["partition"])]
        assert (rows["n_cases"] == c["n_link_cases"]).all()
    pd.testing.assert_frame_equal(pd.read_parquet(out / "coverage_fidelity_table.parquet"), _read(lane, "B24", "coverage_fidelity_table.parquet"))
