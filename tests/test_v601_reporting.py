"""Protocol v6.0.1 reporting additions (lane L2): A2 stored outer-fold predictors with byte-identical v6.0.0 outputs, A4
the measured-B comparator, A5 vendor/centre strata and Bland-Altman, the v6.0.1 manuscript tables and the
build-manuscript-figures stage.  Synthetic fixtures and synthetic metric payloads only; never cohort results."""
from __future__ import annotations

import json
import shutil
import struct
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from test_v6_cli import _study_copy
from conftest import REPO
from tmmri import manuscript_figures as mf
from tmmri import manuscript_tables as mt
from tmmri import predictor_reporting as pr
from tmmri import stages
from tmmri.classification import fit_formal_predictor
from tmmri.config import load_study
from tmmri.formal_features import formal_descriptor_names
from tmmri.metrics import _cluster_draw_weights, _resampling_clusters, metric_summary
from tmmri.runs import RunRoot

CLASSES = ["DCM", "HCM", "MINF", "NOR", "RV"]
V600_OUTPUTS = ("formal/d_model_predictor.joblib", "formal/d_model_ACDC_dev.parquet", "formal/d_model_crossfit_ACDC_dev.parquet")
CROSSWALK = {"NOR": ("NOR", "HARMONIZABLE_SAME_DEFINITION"), "DCM": ("DCM", "HARMONIZABLE_SAME_DEFINITION"),
             "HCM": ("HCM", "HARMONIZABLE_SAME_DEFINITION"), "ARV": ("RV", "HARMONIZABLE_SAME_DEFINITION"), "IHD": ("MINF", "HARMONIZABLE_NOT_EXACT")}


# ----------------------------------------------------------------------------- synthetic run roots


def _context(tmp, study, run_id: str) -> stages.StageContext:
    run = RunRoot(run_id, tmp / "artifacts", subfolders=study.study["artifact_subfolders"])
    return stages.StageContext(study=study, run=run, data_root=tmp / "data", n_jobs=1)


def _write_inputs(run: RunRoot, study) -> None:
    """ACDC development (30) and test (10) rows, M&Ms external (50; vendors A 30, B 15, C 5); class-structured A and B."""

    rng = np.random.default_rng(20260613)
    names, family = formal_descriptor_names(), study.primary_family()
    critical = sorted(str(c) for c in study.cohorts["qc_severity"]["critical"])[0]
    base = 100.0 + 10.0 * np.arange(len(family))

    def a_row(k: int | None, shift: float = 0.0) -> np.ndarray:
        x = rng.normal(0.0, 1.0, len(names)) + shift
        if k is not None:
            x[k] += 2.5
            x[5 + k] += 1.0
        return x

    def b_row(k: int | None) -> np.ndarray:
        return base * (1.0 + 0.12 * ((2 if k is None else k) - 2)) + rng.normal(0.0, 6.0, len(family))

    dev = [f"ACDC:patient{i:03d}" for i in range(1, 31)]
    test = [f"ACDC:patient{i:03d}" for i in range(101, 111)]
    acdc_class = {s: CLASSES[i % 5] for i, s in enumerate(dev)} | {s: CLASSES[i % 5] for i, s in enumerate(test)}
    folds = {s: (i // 5) % 5 + 1 for i, s in enumerate(dev)}           # every fold holds all five classes
    acdc = dev + test
    a = np.vstack([a_row(CLASSES.index(acdc_class[s])) for s in acdc])
    a[3, 10] = np.nan                                                  # INPUT_NONFINITE development row
    b = np.vstack([b_row(CLASSES.index(acdc_class[s])) for s in acdc])
    b[7, 0] = np.nan                                                   # a missing measured target (median-imputed)
    vendors = ["A"] * 30 + ["B"] * 15 + ["C"] * 5
    cycles = {"A": ["NOR", "DCM", "HCM", "ARV"], "B": ["NOR", "IHD", "Other", "DCM", "HCM"], "C": ["HCM", "NOR", "DCM", "Other", "IHD"]}
    counters = {"A": 0, "B": 0, "C": 0}
    pathology = []
    for v in vendors:
        pathology.append(cycles[v][counters[v] % len(cycles[v])])
        counters[v] += 1
    mms = [f"MMS:Q{i:03d}" for i in range(50)]
    xclass = [CROSSWALK.get(p, (None, None))[0] for p in pathology]
    a_mms = np.vstack([a_row(None if c is None else CLASSES.index(c), 0.3) for c in xclass])
    b_mms = np.vstack([b_row(None if c is None else CLASSES.index(c)) for c in xclass])
    p = run.path
    clusters = {s: s for s in acdc + mms} | {dev[25]: dev[0], mms[1]: mms[0]}   # one declared duplicate-image pair each
    split = pd.DataFrame({"sample_id": acdc + mms, "dataset": ["ACDC"] * 40 + ["MMS"] * 50, "case_id": [s.split(":")[1] for s in acdc + mms],
                          "patient_id": [s.split(":")[1] for s in acdc + mms], "partition": ["acdc_dev"] * 30 + ["acdc_test"] * 10 + ["mms_external"] * 50,
                          "outer_fold": pd.array([folds.get(s) for s in acdc + mms], dtype="Int64"), "cluster_id": [clusters[s] for s in acdc + mms],
                          "never_analysed": [False] * 40 + [i >= 20 for i in range(50)], "refit_partition": [None] * 90})
    split.to_parquet(p / "contracts" / "split_manifest.parquet", index=False)
    pd.DataFrame({"sample_id": acdc, "case_id": [s.split(":")[1] for s in acdc], "acdc_class": [acdc_class[s] for s in acdc],
                  "partition": ["acdc_dev"] * 30 + ["acdc_test"] * 10, "eligible_v6": True, "exclusion_reason": None,
                  "release_split": ["training"] * 30 + ["testing"] * 10}).to_parquet(p / "contracts" / "cohort_ACDC.parquet", index=False)
    pd.DataFrame({"sample_id": mms, "case_id": [s.split(":")[1] for s in mms], "pathology": pathology, "vendor": vendors,
                  "centre": [1 if v == "A" else (4 if v == "C" else 2 + i % 2) for i, v in enumerate(vendors)], "mms_external": True, "eligible_v6": True,
                  "exclusion_reason": None, "release_split": "Testing", "crosswalk_acdc_class": xclass, "mms100_member": [i < 20 for i in range(50)],
                  "never_analysed": [i >= 20 for i in range(50)], "refit_partition": None,
                  "crosswalk_compatibility": [CROSSWALK.get(x, (None, "OFF_ONTOLOGY"))[1] for x in pathology]}).to_parquet(
        p / "contracts" / "cohort_MMS.parquet", index=False)
    for ds, ids, av, bv, crit in (("ACDC", acdc, a, b, {9}), ("MMS", mms, a_mms, b_mms, {40})):
        frame = pd.DataFrame(av, columns=names)
        frame.insert(0, "sample_id", ids)
        frame.to_parquet(p / "formal" / f"A_{ds}.parquet", index=False)
        bt = pd.DataFrame(bv, columns=family)
        bt.insert(0, "sample_id", ids)
        bt["qc_failure_reason"] = [critical if i in crit else None for i in range(len(ids))]
        bt.to_parquet(p / "semantics" / f"B_{ds}.parquet", index=False)


def _freeze(ctx) -> dict:
    p = ctx.run.path
    (p / "transitions" / "transition_model.json").write_text("{}", encoding="utf-8")
    for rb in ("d_model", "dataset_label", "measured_oracle"):
        (p / "rules" / rb).mkdir(parents=True, exist_ok=True)
        (p / "rules" / rb / "rulebook.json").write_text(json.dumps({"rulebook": rb, "records": []}), encoding="utf-8")
    return stages.freeze_analysis_bundle(ctx)


def _rho_payload(partition: str, rho: float, n: int, rho_j: dict, status: str = "EVALUATED") -> dict:
    """A synthetic primary-estimand payload (algorithm control; not derived from any cohort)."""
    return {"partition": partition, "rho": rho, "ci_lower": rho - 0.08, "ci_upper": rho + 0.07, "n_cases": n, "n_clusters": n, "status": status, "rho_j": rho_j}


def _write_evaluations(ctx, study) -> None:
    """Recorded evaluation outputs of the frozen predictor and synthetic reconstructions and metric payloads."""

    rng = np.random.default_rng(20261003)
    p, family, names = ctx.run.path, study.primary_family(), formal_descriptor_names()
    predictor = joblib.load(p / "formal" / "d_model_predictor.joblib")
    split = pd.read_parquet(p / "contracts" / "split_manifest.parquet")
    for split_name, ds in (("acdc_test", "ACDC"), ("mms_external", "MMS")):
        ids = sorted(split.loc[split["partition"] == split_name, "sample_id"])
        a = pd.read_parquet(p / "formal" / f"A_{ds}.parquet").set_index("sample_id").loc[ids]
        b = pd.read_parquet(p / "semantics" / f"B_{ds}.parquet").set_index("sample_id").loc[ids, family]
        pd.DataFrame({"sample_id": ids, "d_model": stages.predict_labels(predictor, a[names].to_numpy())}).to_parquet(
            p / "inference" / f"d_model_{split_name}.parquet", index=False)
        (b + 4.0 + rng.normal(0.0, 5.0, b.shape)).reset_index(names="sample_id").to_parquet(p / "reconstructed" / f"bhat_{split_name}.parquet", index=False)
    dev = split[split["partition"] == "acdc_dev"]["sample_id"].tolist()
    b_dev = pd.read_parquet(p / "semantics" / "B_ACDC.parquet").set_index("sample_id").loc[dev, family]
    (b_dev + rng.normal(0.0, 7.0, b_dev.shape)).reset_index(names="sample_id").to_parquet(p / "oof" / "bhat_oof_ACDC_dev.parquet", index=False)
    rho_j = {t: round(0.6 - 0.1 * j, 2) for j, t in enumerate(family)}
    m = p / "metrics"
    payloads = {"primary_rho_acdc_test.json": _rho_payload("acdc_test", 0.55, 50, rho_j),
                "primary_rho_mms_external.json": _rho_payload("mms_external", 0.19, 318, {t: v - 0.3 for t, v in rho_j.items()}),
                "primary_rho_mms_external_never_analysed.json": _rho_payload("mms_external_never_analysed", 0.15, 220, rho_j),
                "strata_rho_mms_external.json": {"min_cases_per_stratum": 20, "strata": {
                    "vendor": {"A": _rho_payload("mms_external", 0.3, 30, rho_j), "B": _rho_payload("mms_external", 0.1, 24, rho_j),
                               "C": {**_rho_payload("mms_external", 0.4, 5, rho_j), "ci_lower": None, "ci_upper": None, "status": "NE",
                                     "reason": "STRATUM_BELOW_MIN_CASES"}},
                    "centre": {"1": _rho_payload("mms_external", 0.3, 30, rho_j), "4": {**_rho_payload("mms_external", 0.4, 5, rho_j), "status": "NE"}}}},
                "fixed_sequence.json": {"results": [{"hypothesis_id": "H-primary-1", "tested": True, "status": "PASSED"},
                                                    {"hypothesis_id": "H-primary-2", "tested": True, "status": "FAILED"},
                                                    {"hypothesis_id": "H-primary-2-never-analysed", "tested": True, "status": "FAILED"}]},
                "shift_audit_mms_external.json": {"results": [{"comparison": "pooled", "group": "all", "auroc": 0.9, "ci_lower": 0.85, "ci_upper": 0.95,
                                                               "n_comparison": 50, "n_comparison_clusters": 49},
                                                              {"comparison": "vendor", "group": "A", "auroc": 0.95, "ci_lower": 0.9, "ci_upper": 0.99,
                                                               "n_comparison": 30, "n_comparison_clusters": 29},
                                                              {"comparison": "vendor", "group": "C", "auroc": 0.7, "n_comparison": 5, "n_comparison_clusters": 5}]}}
    (m / "ablations").mkdir(parents=True, exist_ok=True)
    (m / "ablations" / "transition_variants.json").write_text(json.dumps({"acdc_dev_oof": {"ridge_typed_formal57": _rho_payload("acdc_dev_oof", 0.5, 98, rho_j)}}),
                                                              encoding="utf-8")
    for name, payload in payloads.items():
        (m / name).write_text(json.dumps(payload), encoding="utf-8")
    for split_name in ("acdc_test", "mms_external"):
        outs = [p / "inference" / f"d_model_{split_name}.parquet", p / "reconstructed" / f"bhat_{split_name}.parquet", m / f"primary_rho_{split_name}.json"]
        stages._stage(ctx, f"evaluate:{split_name}", lambda: {}, [ctx.run.frozen_path], outs, {"split": split_name}, status="EVALUATED")


def _add_mms2(ctx, study) -> None:
    """Synthetic lane-L1 outputs of a recorded mms2_external evaluation (payloads only; idempotent)."""

    m = ctx.run.path / "metrics"
    if (m / "primary_rho_mms2_external.json").is_file():
        return
    rho_j = {t: 0.3 for t in study.primary_family()}
    (m / "primary_rho_mms2_external.json").write_text(json.dumps({**_rho_payload("mms2_external", 0.31, 176, rho_j), "hypothesis_id": "H-primary-3"}),
                                                      encoding="utf-8")
    (m / "strata_rho_mms2_external.json").write_text(json.dumps({"strata": {"vendor": {"A": _rho_payload("mms2_external", 0.3, 60, rho_j),
                                                                                         "D": _rho_payload("mms2_external", 0.2, 12, rho_j)}}}), encoding="utf-8")
    (m / "h_primary_3_mms2_external.json").write_text(json.dumps({"results": [{"hypothesis_id": "H-primary-3", "tested": True, "status": "PASSED",
                                                                                "in_hierarchy": False}]}), encoding="utf-8")
    stages._stage(ctx, "evaluate:mms2_external", lambda: {}, [ctx.run.frozen_path], [m / "primary_rho_mms2_external.json"], {"split": "mms2_external"},
                  status="EVALUATED")


def _other_lane_tables(run: RunRoot) -> None:
    """Synthetic outputs in the long-table conventions of tmmri.manuscript_figures (lanes L3, L57, L89)."""

    p = run.path
    rows = [{"representation": rep, "metric": metric, "partition": part, "estimate": est, "ci_lower": est - 0.1, "ci_upper": est + 0.1, "n_clusters": n}
            for rep, shift in (("descriptors", 0.0), ("d20", -0.1), ("f9b", 0.05)) for metric, base in (("rho", 0.5), ("macro_f1", 0.7))
            for part, n, est in (("acdc_test", 50, base + shift), ("mms_external", 272, base + shift - 0.3), ("mms2_external", 150, base + shift - 0.2))]
    for folder in ("probes/comparison_tables", "exploratory/B5", "trust/s1", "trust/s2", "trust/s3", "simulated_shift"):
        (p / folder).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p / "probes" / "comparison_tables" / "comparison_long.csv", index=False)
    (p / "probes" / "d20").mkdir(parents=True, exist_ok=True)
    (p / "probes" / "d20" / "primary_rho_acdc_test.json").write_text(json.dumps({**_rho_payload("acdc_test", 0.42, 50, {}), "representation": "d20"}),
                                                                     encoding="utf-8")
    pd.DataFrame([{"gate": g, "agreement": a, "support": s, "p_pass": min(1.0, (a - 0.75) * s / 4), "false_pass_rate": 0.02}
                  for g in ("v6", "relaxed") for a in (0.8, 0.85, 0.9, 0.95) for s in (5, 10, 20)]).to_parquet(p / "exploratory" / "B5" / "gate_power.parquet", index=False)
    pd.DataFrame([{"partition": part, "model": "audited_predictor", "coverage": c, "risk": r * c}
                  for part, r in (("acdc_test", 0.1), ("mms_external", 0.4)) for c in (0.2, 0.5, 0.8, 1.0)]).to_csv(p / "trust" / "s1" / "risk_coverage.csv", index=False)
    (p / "trust" / "s2" / "ood_flags.json").write_text(json.dumps({"rows": [
        {"partition": "acdc_test", "representation": "descriptors", "proportion_flagged": 0.06, "ci_lower": 0.01, "ci_upper": 0.15, "n_clusters": 50},
        {"partition": "mms_external", "representation": "descriptors", "proportion_flagged": 0.7, "ci_lower": 0.6, "ci_upper": 0.8, "n_clusters": 318},
        {"partition": "mms_external", "vendor": "C", "representation": "descriptors", "proportion_flagged": 0.9, "n_clusters": 5}]}), encoding="utf-8")
    pd.DataFrame([{"variant": v, "partition": "mms_external", "metric": "rho", "estimate": e, "ci_lower": e - 0.1, "ci_upper": e + 0.1}
                  for v, e in (("frozen", 0.19), ("H-a", 0.25), ("H-b", 0.3))]).to_csv(p / "trust" / "s3" / "harmonization_rho.csv", index=False)
    pd.DataFrame([{"vendor": v, "representation": r, "metric": "delta_rho", "estimate": -0.1 * k, "ci_lower": -0.1 * k - 0.05, "ci_upper": -0.1 * k + 0.05}
                  for k, v in enumerate("ABCD") for r in ("descriptors", "d20")]).to_csv(p / "simulated_shift" / "delta_rho.csv", index=False)


@pytest.fixture(scope="module")
def reporting(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("v601")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(stages, "protocol_frozen", lambda *a, **k: True)
        study = load_study(_study_copy(tmp))
        plain, ext = _context(tmp, study, "v6-plain-20261003-0900"), _context(tmp, study, "v6-extended-20261003-0900")
        for ctx in (plain, ext):
            _write_inputs(ctx.run, study)
        assert stages.fit_predictor(plain, extended=False).action == "RAN"
        assert stages.fit_predictor(ext).action == "RAN"
        frozen = {"plain": _freeze(plain), "ext": _freeze(ext)}
        _write_evaluations(ext, study)
        yield SimpleNamespace(tmp=tmp, study=study, plain=plain, ext=ext, frozen=frozen)


# ----------------------------------------------------------------------------- A2 and A4: fit-predictor


def test_fit_predictor_extension_keeps_v600_outputs_byte_identical(reporting):
    plain, ext = reporting.plain.run, reporting.ext.run
    for rel in V600_OUTPUTS:
        assert (plain.path / rel).read_bytes() == (ext.path / rel).read_bytes(), rel
    assert reporting.frozen["plain"]["bundle_sha256"] == reporting.frozen["ext"]["bundle_sha256"]   # the bundle hash is unchanged
    assert not any("audited_predictor" in k or "measured_b_comparator" in k for k in reporting.frozen["ext"]["bundle"])
    plain_record, ext_record = plain.read_record("fit-predictor"), ext.read_record("fit-predictor")
    assert sorted(plain_record.outputs) == sorted(plain.portable(plain.path / rel) for rel in V600_OUTPUTS)
    assert set(plain_record.params) == {"config_sections"}                       # the v6.0.0 record parameters
    assert {ext.portable(q) for q in pr.fit_predictor_extension_paths(ext, reporting.study)} <= set(ext_record.outputs)
    assert not (plain.path / "formal" / "audited_predictor").exists()


def test_stored_fold_predictors_reproduce_the_crossfit_in_the_c4_schema(reporting):
    run, study = reporting.ext.run, reporting.study
    pred = pd.read_parquet(run.path / "formal" / "audited_predictor" / "predictions.parquet")
    assert list(pred.columns) == list(pr.C4_KEYS) + [f"prob_{c}" for c in CLASSES] + [f"logit_{c}" for c in CLASSES]
    assert pred["is_oof"].all() and set(pred["partition"]) == {"acdc_dev"} and set(pred["model_id"]) == {"audited_predictor"} and (pred["seed"] == -1).all()
    split = pd.read_parquet(run.path / "contracts" / "split_manifest.parquet").set_index("sample_id")
    assert len(pred) == 28                                                    # 30 minus the non-finite and the critical-QC row
    assert "ACDC:patient004" not in set(pred["sample_id"]) and "ACDC:patient010" not in set(pred["sample_id"])
    assert (pred["encoder_id"] == "fold" + split.loc[pred["sample_id"], "outer_fold"].astype(str).to_numpy()).all()
    assert pred.loc[pred["sample_id"] == "ACDC:patient026", "cluster_id"].item() == "ACDC:patient001"
    crossfit = pd.read_parquet(run.path / "formal" / "d_model_crossfit_ACDC_dev.parquet").set_index("sample_id")["d_model_crossfit"]
    assert (pred["y_pred"].to_numpy() == crossfit.loc[pred["sample_id"]].to_numpy()).all()
    probs = pred[[f"prob_{c}" for c in CLASSES]].to_numpy()
    logits = pred[[f"logit_{c}" for c in CLASSES]].to_numpy()
    assert np.allclose(probs.sum(axis=1), 1.0)
    soft = np.exp(logits - logits.max(axis=1, keepdims=True))
    assert np.allclose(soft / soft.sum(axis=1, keepdims=True), probs)
    manifest = json.loads((run.path / "formal" / "audited_predictor" / "fold_models.json").read_text(encoding="utf-8"))
    a = pd.read_parquet(run.path / "formal" / "A_ACDC.parquet").set_index("sample_id")
    dev = split[split["partition"] == "acdc_dev"].index.tolist()
    x = a.loc[dev, formal_descriptor_names()].to_numpy()
    y = pd.read_parquet(run.path / "contracts" / "cohort_ACDC.parquet").set_index("sample_id").loc[dev, "acdc_class"].reset_index(drop=True)
    fit_rows = np.isfinite(x).all(axis=1) & ~np.isin(dev, ["ACDC:patient010"])
    folds = split.loc[dev, "outer_fold"].to_numpy(dtype=int)
    for f in range(1, 6):
        stored = joblib.load(run.path / "formal" / "audited_predictor" / "fold_models" / f"fold{f}.joblib")
        refit = fit_formal_predictor(x, y, study.formal["audited_predictor"], fit_mask=fit_rows & (folds != f))
        assert np.array_equal(stored.coef, refit.coef) and stored.n_fit_rows == int((fit_rows & (folds != f)).sum())   # never the held-out fold
        assert manifest["folds"][str(f)]["n_fit_rows"] == stored.n_fit_rows
    bundle = json.loads((run.path / "formal" / "measured_b_comparator" / "bundle.json").read_text(encoding="utf-8"))
    from tmmri.runs import hash_path

    assert all(hash_path(run.path / "formal" / "measured_b_comparator" / rel) == digest for rel, digest in bundle["files"].items())
    assert bundle["features"] == study.primary_family() and bundle["n_fit_rows"] == 29 and bundle["fitted_on"] == "acdc_dev"
    comp = joblib.load(run.path / "formal" / "measured_b_comparator" / "final_model.joblib")
    b = pd.read_parquet(run.path / "semantics" / "B_ACDC.parquet").set_index("sample_id").loc[[s for s in dev if s != "ACDC:patient010"], study.primary_family()]
    assert np.allclose(comp.medians, np.nanmedian(b.to_numpy(), axis=0))
    cpred = pd.read_parquet(run.path / "formal" / "measured_b_comparator" / "predictions.parquet")
    assert (cpred.groupby("encoder_id").size().loc["final"] == 29) and cpred.loc[cpred["is_oof"], "encoder_id"].str.startswith("fold").all()


def test_measured_b_predictor_imputes_with_fit_row_medians_only(reporting):
    cfg = reporting.study.formal["audited_predictor"]
    b = pd.DataFrame({"f1": [1.0, 2.0, np.nan, 10.0, 11.0, 1000.0], "f2": [5.0, 6.0, 7.0, np.nan, 1.0, -1000.0]})
    y = pd.Series(["NOR", "NOR", "NOR", "DCM", "DCM", "DCM"])
    model = pr.fit_measured_b_predictor(b, y, cfg, ["f1", "f2"], fit_mask=np.array([True, True, True, True, True, False]))
    assert model.medians.tolist() == [6.0, 5.5]                             # the last (held-out) row never enters a median
    restored = pr.MeasuredBPredictor.from_dict(json.loads(json.dumps(model.to_dict())))
    assert np.allclose(restored.predict_proba(b).to_numpy(), model.predict_proba(b).to_numpy())
    assert model.impute(pd.DataFrame({"f1": [np.nan], "f2": [np.nan]})).tolist() == [[6.0, 5.5]]


# ----------------------------------------------------------------------------- A2 and A5 statistics


def test_equal_width_ece_brier_and_paired_cluster_draws():
    t = np.array([0, 1, 1, 0])
    p = np.array([[0.95, 0.05], [0.35, 0.65], [0.55, 0.45], [0.62, 0.38]])
    ece, bins = pr.ece_equal_width(t, p, 10)
    # bins (0.6, 0.7] holds 0.65 and 0.62 (both correct), (0.5, 0.6] holds 0.55 (wrong), (0.9, 1.0] holds 0.95 (correct)
    hand = 2 / 4 * abs((0.65 + 0.62) / 2 - 1.0) + 1 / 4 * abs(0.55 - 0.0) + 1 / 4 * abs(0.95 - 1.0)
    assert ece == pytest.approx(hand, abs=1e-12) and bins["count"].sum() == 4 and len(bins) == 10
    brier = np.mean([(0.95 - 1) ** 2 + 0.05 ** 2, 0.35 ** 2 + (0.65 - 1) ** 2, 0.55 ** 2 + (0.45 - 1) ** 2, (0.62 - 1) ** 2 + 0.38 ** 2])
    assert pr.brier_multiclass(t, p) == pytest.approx(brier, abs=1e-12)
    clusters = np.array(["c1", "c1", "p3", "p4"], dtype=object)
    boot = pr.calibration_bootstrap(t, p, clusters=clusters, n_boot=16, seed=20260928, n_bins=10)
    unit_cluster, n_resample = _resampling_clusters(np.arange(4), 4, clusters)
    for r, ss in enumerate(np.random.SeedSequence(20260928).spawn(16)):
        rows = np.repeat(np.arange(4), _cluster_draw_weights(np.random.default_rng(ss), unit_cluster, n_resample).astype(int))
        assert boot[r, 1] == pytest.approx(pr.brier_multiclass(t[rows], p[rows]), abs=1e-12)
    small = pr.extended_predictor_summary(np.array(["A", "B", "B", "A"], dtype=object), np.array(["A", "B", "A", "A"], dtype=object),
                                          pd.DataFrame(p, columns=["A", "B"]), ["A", "B"], clusters=clusters, n_boot=50, seed=1, n_bins=10,
                                          min_valid_fraction=0.95, min_clusters=20)
    assert small["descriptive_only"] is True
    for m in ("accuracy", "ece_equal_width", "brier", "log_loss"):
        assert small["metrics"][m]["estimate"] is not None and small["metrics"][m]["reason"] == "STRATUM_BELOW_MIN_CASES", m
        assert small["metrics"][m].get("ci_lower") is None


def test_bland_altman_matches_metric_summary_and_marks_small_strata():
    rng = np.random.default_rng(3)
    obs = pd.DataFrame({"t1": rng.normal(100, 10, 40), "t2": rng.normal(50, 5, 40)})
    pred = obs + rng.normal(2.0, 4.0, obs.shape)
    pred.loc[3, "t2"] = np.nan
    clusters = np.array([f"c{i // 2}" for i in range(40)], dtype=object)
    out = pr.bland_altman(obs, pred, ["t1", "t2"], clusters, n_boot=200, seed=20260928)
    for t in ("t1", "t2"):
        ref = metric_summary(obs[t].to_numpy(), pred[t].to_numpy())
        assert out[t]["bias"] == pytest.approx(ref["bias"]) and out[t]["loa_lower"] == pytest.approx(ref["loa_lower"])
        assert out[t]["loa_upper"] == pytest.approx(ref["loa_upper"]) and out[t]["status"] == "EVALUATED"
        lo, hi = out[t]["ci"]["bias"]
        assert lo < out[t]["bias"] < hi
    assert out["t2"]["n"] == 39 and out["t1"]["n_clusters"] == 20
    again = pr.bland_altman(obs, pred, ["t1", "t2"], clusters, n_boot=200, seed=20260928)
    assert json.dumps(again, sort_keys=True) == json.dumps(out, sort_keys=True)
    small = pr.bland_altman(obs.iloc[:10], pred.iloc[:10], ["t1"], clusters[:10], n_boot=200, seed=1, min_clusters=20)["t1"]
    assert small["status"] == "NE" and small["reason"] == "STRATUM_BELOW_MIN_CASES" and small["bias"] is not None and small["ci"] is None


# ----------------------------------------------------------------------------- A2, A4, A5: extended summaries


def test_summarize_predictor_extended_fills_development_metrics_and_strata(reporting):
    ctx, study = reporting.ext, reporting.study
    results = pr.summarize_predictor_extended(ctx)
    assert [r.stage for r in results] == [f"summarize-predictor-extended:{p}" for p in ("acdc_dev_oof", "acdc_test", "mms_external")]
    assert {r.action for r in results} == {"RAN"}
    assert stages.summarize_predictor(ctx).action == "RAN"                  # the v6.0.0 summary is untouched
    m = ctx.run.path / "metrics"
    dev = json.loads((m / "predictor_extended" / "acdc_dev_oof.json").read_text(encoding="utf-8"))
    audited = dev["models"]["audited_predictor"]
    for metric in ("macro_ovr_auroc", "ece", "log_loss", "ece_equal_width", "brier"):
        assert audited["metrics"][metric]["status"] == "EVALUATED" and audited["metrics"][metric]["estimate"] is not None, metric
    base_dev = json.loads((m / "predictor_acdc_dev_oof.json").read_text(encoding="utf-8"))
    assert base_dev["metrics"]["macro_ovr_auroc"]["reason"] == "OOF_PROBABILITIES_NOT_RECORDED"     # v6.0.0 file unchanged in kind
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):                               # same rows, same cluster draws
        assert audited["metrics"][metric] == base_dev["metrics"][metric], metric
    assert dev["models"]["measured_b_comparator"]["status"] == "EVALUATED"
    for part in ("acdc_test", "mms_external"):
        ext = json.loads((m / "predictor_extended" / f"{part}.json").read_text(encoding="utf-8"))
        base = json.loads((m / f"predictor_{part}.json").read_text(encoding="utf-8"))
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "macro_ovr_auroc", "ece", "log_loss"):
            assert ext["models"]["audited_predictor"]["metrics"][metric] == base["metrics"][metric], (part, metric)
        assert ext["models"]["measured_b_comparator"]["status"] == "EVALUATED" and ext["evidence"] == "registered after unblinding"
        c4 = pd.read_parquet(ctx.run.path / "inference" / "audited_predictor" / part / "predictions.parquet")
        d_model = pd.read_parquet(ctx.run.path / "inference" / f"d_model_{part}.parquet").set_index("sample_id")["d_model"]
        assert (c4["y_pred"].to_numpy() == d_model.loc[c4["sample_id"]].to_numpy()).all() and (c4["encoder_id"] == "final").all()
        assert not c4["is_oof"].any()
    mms = json.loads((m / "predictor_extended" / "mms_external.json").read_text(encoding="utf-8"))
    vendor = mms["strata"]["vendor"]
    assert vendor["A"]["audited_predictor"]["metrics"]["accuracy"]["status"] == "EVALUATED"
    for v in ("B", "C"):
        acc = vendor[v]["audited_predictor"]["metrics"]["accuracy"]
        assert acc["status"] == "NE" and acc["reason"] == "STRATUM_BELOW_MIN_CASES" and acc["estimate"] is not None and acc["ci_lower"] is None
    assert set(mms["strata"]["centre"]) == {"1", "2", "3", "4"}
    ba = mms["bland_altman"]
    bhat = pd.read_parquet(ctx.run.path / "reconstructed" / "bhat_mms_external.parquet").set_index("sample_id")
    b = pd.read_parquet(ctx.run.path / "semantics" / "B_MMS.parquet").set_index("sample_id").loc[bhat.index]
    target = study.primary_family()[0]
    assert ba["targets"][target]["bias"] == pytest.approx(metric_summary(b[target].to_numpy(), bhat[target].to_numpy())["bias"])
    assert ba["strata"]["vendor"]["C"][target]["reason"] == "STRATUM_BELOW_MIN_CASES"
    assert ba["strata"]["vendor"]["A"][target]["status"] == "EVALUATED"
    table = pd.read_parquet(m / "predictor_extended" / "bland_altman_mms_external.parquet")
    assert set(table["stratum_variable"]) == {"all", "vendor", "centre"}
    assert {r.action for r in pr.summarize_predictor_extended(ctx)} == {"SKIPPED"}
    with pytest.raises(stages.BundleError, match="not evaluated"):
        pr.summarize_predictor_extended(ctx, ["mms2_external"])
    plain = pr.summarize_predictor_extended(reporting.plain, ["acdc_dev_oof"])[0]
    assert plain.action == "RAN"
    payload = json.loads((reporting.plain.run.path / "metrics" / "predictor_extended" / "acdc_dev_oof.json").read_text(encoding="utf-8"))
    assert payload["models"]["audited_predictor"]["reason"] == pr.FOLD_PREDICTORS_NOT_RECORDED


def test_extended_summary_refuses_a_changed_comparator_and_obeys_the_mms2_lock(reporting, monkeypatch, tmp_path):
    ctx = reporting.ext
    other = RunRoot("v6-tampered-20261003-0901", tmp_path, subfolders=reporting.study.study["artifact_subfolders"], create=False)
    shutil.copytree(ctx.run.path, other.path)
    for record in other.stages_dir.glob("summarize-predictor-extended__*.json"):
        record.unlink()
    with (other.path / "formal" / "measured_b_comparator" / "final_model.joblib").open("ab") as handle:
        handle.write(b"\0")
    with pytest.raises(stages.BundleError, match="changed after fit-predictor"):
        pr.summarize_predictor_extended(stages.StageContext(study=reporting.study, run=other, n_jobs=1), ["acdc_test"])

    def refuse(*_a, **_k):
        raise stages.ProtocolFreezeError("protocol-v6.0.1 tag absent")

    monkeypatch.setattr(stages, "require_mms2_unlocked", refuse, raising=False)
    with pytest.raises(stages.ProtocolFreezeError, match="v6.0.1"):
        pr.require_partition_unlocked(ctx, "mms2_external", "test")
    pr.require_partition_unlocked(ctx, "acdc_test", "test")                  # ACDC partitions need no lock


# ----------------------------------------------------------------------------- v6.0.1 tables


def _build_tables(reporting, out_dir, *, include_mms2: bool):
    spec = reporting.tmp / "spec" / "SPECIFICATION.md"
    if not spec.is_file():
        spec.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / "SPECIFICATION.md", spec)                       # outside git: Table 1 from the copied frozen text
    return mt.build_tables(reporting.ext.run, reporting.study, include_mms=True, out_dir=out_dir, spec_path=spec, include_mms2=include_mms2)


def test_v601_tables_beside_byte_identical_tables_1_to_10(reporting, tmp_path):
    run = reporting.ext.run
    if not (run.path / "metrics" / "predictor_extended").is_dir():
        pr.summarize_predictor_extended(reporting.ext)
    moved = tmp_path / "moved"
    shutil.move(str(run.path / "metrics" / "predictor_extended"), str(moved))
    try:
        _build_tables(reporting, tmp_path / "without", include_mms2=False)
    finally:
        shutil.move(str(moved), str(run.path / "metrics" / "predictor_extended"))
    _build_tables(reporting, tmp_path / "with", include_mms2=False)
    for name in mt.TABLE_NAMES:
        for ext in ("csv", "tex"):
            assert (tmp_path / "without" / f"{name}.{ext}").read_bytes() == (tmp_path / "with" / f"{name}.{ext}").read_bytes(), (name, ext)
    assert (tmp_path / "without" / "provenance.json").read_bytes() == (tmp_path / "with" / "provenance.json").read_bytes()
    t3 = pd.read_csv(tmp_path / "with" / "v601_table03_hypotheses.csv", dtype=str, keep_default_na=False).set_index("hypothesis")
    assert t3.loc["H-primary-1", "decision"] == "PASSED" and t3.loc["H-primary-2", "decision"] == "FAILED"
    assert t3.loc["H-primary-3", "status"] == "NOT_RECORDED"                 # mms2_external not evaluated: never derived
    _add_mms2(reporting.ext, reporting.study)
    _build_tables(reporting, tmp_path / "mms2", include_mms2=True)
    out = tmp_path / "mms2"
    t3 = pd.read_csv(out / "v601_table03_hypotheses.csv", dtype=str, keep_default_na=False).set_index("hypothesis")
    assert t3.loc["H-primary-3", "decision"] == "PASSED" and float(t3.loc["H-primary-3", "rho"]) == pytest.approx(0.31)
    assert t3.loc["H-primary-3", "evidence"].startswith("confirmatory; prospective")
    t5 = pd.read_csv(out / "v601_table05_vendor_strata.csv", dtype=str, keep_default_na=False)
    small = f"NE (n < {reporting.study.bootstrap['subgroup_reporting_thresholds']['min_cases_per_stratum']})"
    vend = t5[(t5["partition"] == "mms_external") & (t5["stratum_variable"] == "vendor")].set_index("stratum")
    assert vend.loc["C", "rho"] == small and vend.loc["A", "rho"] != small and vend.loc["A", "accuracy"] not in ("", small)
    assert vend.loc["B", "accuracy"] == small and vend.loc["A", "domain_auroc"] != "" and vend.loc["C", "domain_auroc"] == small
    mms2 = t5[(t5["partition"] == "mms2_external") & (t5["stratum_variable"] == "vendor")].set_index("stratum")
    assert mms2.loc["D", "rho"] == small and mms2.loc["A", "evidence"] == "prospective"
    t6 = pd.read_csv(out / "v601_table06_predictors.csv", dtype=str, keep_default_na=False)
    dev = t6[(t6["partition"] == "acdc_dev_oof") & (t6["panel"] == "overall")]
    assert set(dev["model"]) == {"audited_predictor", "measured_b_comparator"}
    auroc = dev[(dev["model"] == "audited_predictor") & (dev["metric"] == "macro_ovr_auroc")]
    assert auroc["status"].item() == "EVALUATED" and 0.0 <= float(auroc["estimate"].item()) <= 1.0
    assert (t6.loc[t6["partition"] == "mms2_external", "status"] == "NOT_RECORDED").all()
    strata = t6[t6["panel"] == "strata"]
    assert (strata.loc[strata["stratum"] == "C", "estimate"] == small).all() and len(strata)
    ba = pd.read_csv(out / "v601_tableS1_bland_altman.csv", dtype=str, keep_default_na=False)
    assert set(ba.loc[ba["panel"] == "all", "partition"]) >= {"acdc_dev_oof", "acdc_test", "mms_external"}
    assert (ba.loc[(ba["stratum"] == "C"), "bias"] == small).all()
    prov = json.loads((out / mt.PROVENANCE_V601).read_text(encoding="utf-8"))
    assert set(prov["tables"]) == set(mt.TABLE_NAMES_V601) and prov["include_mms2"] is True
    for name in mt.TABLE_NAMES_V601:
        assert prov["tables"][name]["sources"] and all(s["sha256"] for s in prov["tables"][name]["sources"]), name
    split = pd.read_parquet(run.path / "contracts" / "split_manifest.parquet")
    identifiers = set(split["case_id"].astype(str))
    import re

    for path in out.glob("*.*"):
        assert not set(re.findall(r"[A-Za-z0-9]+", path.read_text(encoding="utf-8"))) & identifiers, path.name


def _h4_record(partition: str, confirmatory: bool) -> dict:
    """Synthetic once-only H4 decision record (tmmri.confirmatory_h4 schema; payload values only)."""

    a = {"hypothesis_id": "H4a", "estimand": "rho_EF", "estimate": 0.42, "ci_lower": 0.21, "ci_upper": 0.58, "n_cases": 176, "n_clusters": 176,
         "tested": confirmatory, "status": "PASSED" if confirmatory else "DESCRIPTIVE", "criterion_met": True, "reason": None}
    b = {"hypothesis_id": "H4b", "estimand": "delta_rho", "estimate": 0.03, "ci_lower": -0.02, "ci_upper": 0.08, "n_cases": 176, "n_clusters": 176,
         "tested": confirmatory, "status": "FAILED" if confirmatory else "DESCRIPTIVE", "criterion_met": False, "reason": None}
    return {"schema": "tmmri.h4_family/1", "family": "H4", "partition": partition, "confirmatory": confirmatory, "results": [a, b]}


def test_v601_table03_h4_rows_from_the_once_only_records(reporting, tmp_path):
    run = reporting.ext.run
    _add_mms2(reporting.ext, reporting.study)
    folder = run.path / "metrics" / mt.H4_METRICS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    for part, confirmatory in (("mms2_external", True), ("acdc_test", False)):         # mms_external: no record (NOT_RECORDED)
        if not (folder / f"h4_{part}.json").is_file():
            (folder / f"h4_{part}.json").write_text(json.dumps(_h4_record(part, confirmatory)), encoding="utf-8")
    assert not [q for q in mt.metric_paths(run, include_mms=True) if mt.H4_METRICS_DIR in q.parts]         # never read by Tables 1-10
    gated = {q.name for q in mt.extended_source_paths(run, include_mms=True, include_mms2=False)}
    assert "h4_acdc_test.json" in gated and "h4_mms2_external.json" not in gated
    _build_tables(reporting, tmp_path / "h4", include_mms2=True)
    t3 = pd.read_csv(tmp_path / "h4" / "v601_table03_hypotheses.csv", dtype=str, keep_default_na=False)
    rows = {(r["hypothesis"], r["partition"]): r for r in t3.to_dict("records")}
    assert {h for h, _ in rows} >= {"H-primary-1", "H-primary-3", "H4a", "H4b"}
    a, b = rows[("H4a", "mms2_external")], rows[("H4b", "mms2_external")]
    assert a["decision"] == "PASSED" and float(a["rho"]) == pytest.approx(0.42) and a["estimand"] == "rho_EF" and a["status"] == "EVALUATED"
    assert a["evidence"].startswith("confirmatory; prospective") and "97.5 %" in a["criterion"] and "> 0.1" in a["criterion"]
    assert b["decision"] == "FAILED" and float(b["ci_lower"]) == pytest.approx(-0.02) and b["estimand"] == "delta_rho" and "> 0" in b["criterion"]
    assert "Bonferroni" in a["family"] and rows[("H-primary-1", "acdc_test")]["estimand"] == "rho"
    for hid in ("H4a", "H4b"):
        assert rows[(hid, "acdc_test")]["decision"] == "DESCRIPTIVE" and rows[(hid, "acdc_test")]["evidence"] == "registered after unblinding"
        assert rows[(hid, "mms_external")]["status"] == "NOT_RECORDED" and "absent" in rows[(hid, "mms_external")]["note"]
    prov = json.loads((tmp_path / "h4" / mt.PROVENANCE_V601).read_text(encoding="utf-8"))
    sources = {s["path"] for s in prov["tables"]["v601_table03_hypotheses"]["sources"]}
    assert any(p.endswith("confirmatory_h4/h4_mms2_external.json") for p in sources)
    _build_tables(reporting, tmp_path / "h4_plain", include_mms2=False)                  # M&Ms-2 not included: never read, never derived
    plain = pd.read_csv(tmp_path / "h4_plain" / "v601_table03_hypotheses.csv", dtype=str, keep_default_na=False)
    assert (plain.loc[(plain["hypothesis"].isin(["H4a", "H4b"])) & (plain["partition"] == "mms2_external"), "status"] == "NOT_RECORDED").all()


# ----------------------------------------------------------------------------- build-manuscript-figures


def _png_dpi(data: bytes) -> float:
    at = data.index(b"pHYs")
    x_ppm = struct.unpack(">I", data[at + 4:at + 8])[0]
    return x_ppm * 0.0254


def test_build_manuscript_figures_from_recorded_artifacts(reporting, tmp_path):
    ctx = reporting.ext
    if not (ctx.run.path / "metrics" / "predictor_extended").is_dir():
        pr.summarize_predictor_extended(ctx)
    _add_mms2(ctx, reporting.study)
    _other_lane_tables(ctx.run)
    result = mf.build_manuscript_figures(ctx)
    assert result.action == "RAN"
    fdir = ctx.run.path / "figures"
    index = json.loads((fdir / "provenance.json").read_text(encoding="utf-8"))
    assert set(index["figures"]) == set(mf.FIGURE_NAMES) and index["include_mms2"] is True
    assert {name: entry["status"] for name, entry in index["figures"].items()} == {name: "DRAWN" for name in mf.FIGURE_NAMES}
    for name in mf.FIGURE_NAMES:
        pdf, png = (fdir / f"{name}.pdf").read_bytes(), (fdir / f"{name}.png").read_bytes()
        assert pdf.startswith(b"%PDF") and _png_dpi(png) == pytest.approx(300, abs=0.5), name
        record = json.loads((fdir / f"{name}.provenance.json").read_text(encoding="utf-8"))
        assert record["files"]["pdf"]["sha256"] and all(p["status"] == "DRAWN" and p["sources"] for p in record["panels"]), name
        assert all(s["sha256"] for p in record["panels"] for s in p["sources"]), name
    forest = json.loads((fdir / "figure03_rho_forest.provenance.json").read_text(encoding="utf-8"))
    paths = {s["path"] for s in forest["panels"][0]["sources"]}
    assert {"run:metrics/primary_rho_mms2_external.json", "run:probes/d20/primary_rho_acdc_test.json"} <= paths
    heat = json.loads((fdir / "figure04_external_target_heatmap.provenance.json").read_text(encoding="utf-8"))
    assert heat["panels"][0]["suppressed_below_min_cases"] >= 2                 # vendor C (M&Ms-1) and vendor D (M&Ms-2)
    ood = json.loads((fdir / "figure07_risk_coverage_ood.provenance.json").read_text(encoding="utf-8"))
    assert ood["panels"][1]["suppressed_below_min_cases"] == 1
    assert mf.build_manuscript_figures(ctx).action == "SKIPPED"
    # Determinism: drawing again gives byte-identical files.
    mf.build_figures(ctx.run, reporting.study, include_mms=True, include_mms2=True, out_dir=tmp_path / "again")
    for name in mf.FIGURE_NAMES:
        for ext in ("pdf", "png"):
            assert (tmp_path / "again" / f"{name}.{ext}").read_bytes() == (fdir / f"{name}.{ext}").read_bytes(), (name, ext)


def test_figures_omit_missing_or_malformed_panels_and_refuse_case_identifiers(reporting, tmp_path):
    study = reporting.study
    other = RunRoot("v6-figures-20261003-0902", tmp_path, subfolders=study.study["artifact_subfolders"], create=False)
    shutil.copytree(reporting.ext.run.path, other.path)
    for folder in ("trust", "simulated_shift", "probes", "exploratory", "figures"):
        shutil.rmtree(other.path / folder, ignore_errors=True)
    (other.path / ".stages" / "build-manuscript-figures.json").unlink(missing_ok=True)
    (other.path / "exploratory" / "B5").mkdir(parents=True)
    pd.DataFrame({"unrelated": [1, 2]}).to_csv(other.path / "exploratory" / "B5" / "grid.csv", index=False)
    ctx = stages.StageContext(study=study, run=other, n_jobs=1)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(stages, "protocol_frozen", lambda *a, **k: True)
        assert mf.build_manuscript_figures(ctx).action == "RAN"
        fdir = other.path / "figures"
        index = json.loads((fdir / "provenance.json").read_text(encoding="utf-8"))["figures"]
        assert index["figure07_risk_coverage_ood"]["status"] == "OMITTED" and not (fdir / "figure07_risk_coverage_ood.pdf").exists()
        assert index["figure06_gate_power"]["status"] == "PLACEHOLDER"
        gate = json.loads((fdir / "figure06_gate_power.provenance.json").read_text(encoding="utf-8"))
        assert gate["panels"][0]["status"] == "OMITTED" and "agreement" in gate["panels"][0]["note"]
        omitted = json.loads((fdir / "figure08_harmonization_simulated_shift.provenance.json").read_text(encoding="utf-8"))
        assert [p["status"] for p in omitted["panels"]] == ["OMITTED", "OMITTED"] and "inputs absent" in omitted["panels"][0]["note"]
        assert index["figure03_rho_forest"]["status"] == "DRAWN"
        # A figure input that would print a case identifier is refused; nothing is written.
        (other.path / "probes" / "comparison_tables").mkdir(parents=True)
        (other.path / ".stages" / "build-manuscript-figures.json").unlink()      # a recorded stage of a frozen run may only skip
        pd.DataFrame([{"representation": "patient001", "metric": "rho", "partition": "acdc_test", "estimate": 0.5, "n_clusters": 50}]).to_csv(
            other.path / "probes" / "comparison_tables" / "comparison.csv", index=False)
        before = (fdir / "provenance.json").read_bytes()
        with pytest.raises(stages.BundleError, match="case identifier"):
            mf.build_manuscript_figures(ctx)
        assert (fdir / "provenance.json").read_bytes() == before
