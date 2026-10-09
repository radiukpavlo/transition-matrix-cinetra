"""summarize-predictor (SPECIFICATION.md 13.1-13.2 "Audited predictor"), build-manuscript-tables (Section 16) and the
once-only guard of a frozen run.  Synthetic fixtures only; never cohort results."""
from __future__ import annotations

import json
import shutil
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest

from test_v6_cli import _edit_yaml, _study_copy
from tmmri import stages
from tmmri.classification import FormalPredictor, classification_metrics_v6, predictor_cluster_bootstrap, predictor_summary_v6
from tmmri.config import load_study
from tmmri.formal_features import formal_descriptor_names
from tmmri.manuscript_tables import TableBuildError, assert_no_case_identifiers, latex_escape
from tmmri.runs import RunRoot
from tmmri.transition import FormalStandardizer

CLASSES = ["DCM", "HCM", "NOR"]
PROBA = [[0.7, 0.2, 0.1], [0.3, 0.6, 0.1], [0.1, 0.8, 0.1], [0.2, 0.5, 0.3], [0.1, 0.1, 0.8], [0.5, 0.1, 0.4],
         [0.2, 0.3, 0.5], [np.nan, np.nan, np.nan]]
TRUTH = ["DCM", "DCM", "HCM", "HCM", "NOR", "NOR", None, "DCM"]          # row 6: unknown reference; row 7: no decision
DECISION = ["DCM", "HCM", "HCM", "HCM", "NOR", "DCM", "NOR", None]
CLUSTERS = ["c01", "c01", "p2", "p3", "p4", "p5", "p6", "p7"]           # rows 0 and 1: one declared duplicate-image cluster


def _hand() -> dict[str, float]:
    """Hand computation on the six evaluable rows (confusion DCM [1, 1, 0], HCM [0, 2, 0], NOR [1, 0, 1])."""
    f1 = {"DCM": 2 * 1 / (2 * 1 + 1 + 1), "HCM": 2 * 2 / (2 * 2 + 1 + 0), "NOR": 2 * 1 / (2 * 1 + 0 + 1)}
    auroc = {"DCM": 7 / 8, "HCM": 7 / 8, "NOR": 8 / 8}   # rank statistic of the one-vs-rest scores
    ece = 0.5 * abs((0.5 + 0.5 + 0.6) / 3 - 1 / 3) + 0.5 * abs((0.7 + 0.8 + 0.8) / 3 - 1.0)   # 2 equal-mass bins
    return {"accuracy": 4 / 6, "balanced_accuracy": (0.5 + 1.0 + 0.5) / 3, "macro_f1": sum(f1.values()) / 3,
            "macro_ovr_auroc": sum(auroc.values()) / 3, "ece": ece,
            "log_loss": -float(np.mean(np.log([0.7, 0.3, 0.8, 0.5, 0.8, 0.4])))}


def _summary(n_jobs: int = 1, proba: bool = True, n_boot: int = 64, **kw):
    frame = pd.DataFrame(PROBA, columns=CLASSES) if proba else None
    return predictor_summary_v6(np.array(TRUTH, dtype=object), np.array(DECISION, dtype=object), frame, CLASSES, clusters=np.array(CLUSTERS, dtype=object),
                                n_boot=n_boot, seed=20260928, n_bins=2, min_valid_fraction=0.95, n_jobs=n_jobs, **kw)


def test_predictor_metrics_match_hand_computation():
    out = _summary()
    hand = _hand()
    for metric, value in hand.items():
        assert out["metrics"][metric]["estimate"] == pytest.approx(value, abs=1e-12), metric
    assert out["counts"] == {"n_rows": 8, "n_no_decision": 1, "n_reference_unknown": 1, "n_reference_outside_predictor_classes": 0,
                             "n_evaluable": 6, "n_clusters": 5}
    assert out["companions"]["confusion_matrix"]["counts"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert {c: r["recall"] for c, r in out["companions"]["per_class_recall"].items()} == {"DCM": 0.5, "HCM": 1.0, "NOR": 0.5}
    assert [b["count"] for b in out["ece_realized_bins"]] == [3, 3] and out["ece_bins"] == 2 and out["ece_binning"] == "equal_mass"
    reference, _ = classification_metrics_v6(pd.Series(TRUTH[:6]), pd.DataFrame(PROBA[:6], columns=CLASSES), n_bins=2)
    for metric in ("accuracy", "balanced_accuracy", "macro_f1", "macro_ovr_auroc", "ece"):
        assert out["metrics"][metric]["estimate"] == reference[metric]
    labels_only = _summary(proba=False, probability_reason="OOF_PROBABILITIES_NOT_RECORDED")
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        assert labels_only["metrics"][metric]["estimate"] == pytest.approx(hand[metric], abs=1e-12)
    for metric in ("macro_ovr_auroc", "ece", "log_loss"):
        assert labels_only["metrics"][metric]["status"] == "NE" and labels_only["metrics"][metric]["reason"] == "OOF_PROBABILITIES_NOT_RECORDED"
    wrong = list(DECISION)
    wrong[0] = "HCM"
    with pytest.raises(ValueError, match="argmax"):
        predictor_summary_v6(np.array(TRUTH, dtype=object), np.array(wrong, dtype=object), pd.DataFrame(PROBA, columns=CLASSES), CLASSES,
                             clusters=np.array(CLUSTERS, dtype=object), n_boot=8, seed=1, n_bins=2, min_valid_fraction=0.95)


def test_predictor_bootstrap_draws_clusters_and_is_invariant_to_n_jobs():
    one, three = _summary(n_jobs=1), _summary(n_jobs=3)
    assert json.dumps(one, sort_keys=True, default=str) == json.dumps(three, sort_keys=True, default=str)
    acc = one["metrics"]["accuracy"]
    assert acc["status"] == "EVALUATED" and acc["n_boot_valid"] == 64 and acc["ci_lower"] <= acc["estimate"] <= acc["ci_upper"]
    # Replicate r draws the five clusters with SeedSequence(seed).spawn(n)[r]; the pair (rows 0, 1) comes as a whole.
    t = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 1, 1, 1, 2, 0])
    boot = predictor_cluster_bootstrap(t, pred, np.array(PROBA[:6]), 3, clusters=np.array(CLUSTERS[:6], dtype=object), n_boot=64, seed=20260928, n_bins=2)
    unit_cluster = np.array([0, 0, 1, 2, 3, 4])
    for r, ss in enumerate(np.random.SeedSequence(20260928).spawn(64)):
        counts = np.bincount(np.random.default_rng(ss).integers(0, 5, size=5), minlength=5)
        w = counts[unit_cluster]
        assert boot[r, 0] == pytest.approx(float(np.sum(w * (t == pred)) / np.sum(w)), abs=1e-12)
    alpha = np.quantile(boot[:, 0], [0.025, 0.975])
    assert (acc["ci_lower"], acc["ci_upper"]) == (float(alpha[0]), float(alpha[1]))


# ----------------------------------------------------------------------------- summarize-predictor on a minimal run root


def _predictor() -> FormalPredictor:
    """Frozen multinomial LR stand-in: class k scores 3 * A[:, k] for the first five descriptors (no fitting)."""
    keep = np.zeros(57, dtype=bool)
    keep[:5] = True
    return FormalPredictor(classes=["DCM", "HCM", "MINF", "NOR", "RV"], standardizer=FormalStandardizer(keep, np.zeros(57), np.ones(57)),
                           coef=3.0 * np.eye(5), intercept=np.zeros(5), config={"name": "test"})


def _softmax_labels(a5: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = 3.0 * a5
    e = np.exp(s - s.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    return p, np.array(["DCM", "HCM", "MINF", "NOR", "RV"], dtype=object)[np.argmax(p, axis=1)]


def _a_rows(ids: list[str], hot: list[int], strength: list[float]) -> pd.DataFrame:
    x = np.zeros((len(ids), 57))
    for i, (k, s) in enumerate(zip(hot, strength)):
        x[i, k] = s
        x[i, (k + 1) % 5] = 0.25 * (i % 3)
    frame = pd.DataFrame(x, columns=formal_descriptor_names())
    frame.insert(0, "sample_id", ids)
    return frame


@pytest.fixture()
def predictor_run(tmp_path, monkeypatch):
    """A frozen run root with recorded d_model predictions (ACDC dev/test, M&Ms external), synthetic values only."""
    monkeypatch.setattr(stages, "protocol_frozen", lambda *a, **k: True)
    study = load_study()
    run = RunRoot("v6-reporting-20260929-1200", tmp_path / "artifacts", subfolders=study.study["artifact_subfolders"])
    ctx = stages.StageContext(study=study, run=run, data_root=tmp_path / "data", n_jobs=1)
    critical = sorted(str(c) for c in study.cohorts["qc_severity"]["critical"])[0]
    p = run.path
    dev = [f"ACDC:patient{i:03d}" for i in range(1, 8)]
    test = [f"ACDC:patient{i:03d}" for i in range(101, 106)]
    mms = [f"MMS:Q{i}X{i}Y{i}" for i in range(10)]
    acdc_class = {**dict(zip(dev, ["DCM", "DCM", "HCM", "HCM", "NOR", "NOR", "RV"])), **dict(zip(test, ["DCM", "HCM", "MINF", "NOR", "RV"]))}
    pathology = ["NOR", "DCM", "HCM", "ARV", "NOR", "IHD", "IHD", "HHD", "Other", "LVNC"]
    crosswalk = {"NOR": ("NOR", "HARMONIZABLE_SAME_DEFINITION"), "DCM": ("DCM", "HARMONIZABLE_SAME_DEFINITION"), "HCM": ("HCM", "HARMONIZABLE_SAME_DEFINITION"),
                 "ARV": ("RV", "HARMONIZABLE_SAME_DEFINITION"), "IHD": ("MINF", "HARMONIZABLE_NOT_EXACT")}
    split = pd.DataFrame({"sample_id": dev + test + mms, "dataset": ["ACDC"] * 12 + ["MMS"] * 10,
                          "case_id": [s.split(":")[1] for s in dev + test + mms], "patient_id": [s.split(":")[1] for s in dev + test + mms],
                          "partition": ["acdc_dev"] * 7 + ["acdc_test"] * 5 + ["mms_external"] * 10,
                          "cluster_id": dev + test + [mms[0], mms[0]] + mms[2:], "refit_partition": [None] * 22})
    split.to_parquet(p / "contracts" / "split_manifest.parquet", index=False)
    pd.DataFrame({"sample_id": dev + test, "case_id": [s.split(":")[1] for s in dev + test], "acdc_class": [acdc_class[s] for s in dev + test],
                  "partition": ["acdc_dev"] * 7 + ["acdc_test"] * 5, "eligible_v6": True}).to_parquet(p / "contracts" / "cohort_ACDC.parquet", index=False)
    pd.DataFrame({"sample_id": mms, "case_id": [s.split(":")[1] for s in mms], "pathology": pathology, "mms_external": True,
                  "crosswalk_acdc_class": [crosswalk.get(x, (None, None))[0] for x in pathology],
                  "crosswalk_compatibility": [crosswalk.get(x, (None, "OFF_ONTOLOGY"))[1] for x in pathology]}).to_parquet(p / "contracts" / "cohort_MMS.parquet", index=False)
    pd.DataFrame({"sample_id": dev + test, "qc_failure_reason": [None] * 6 + [critical] + [None] * 5}).to_parquet(p / "semantics" / "B_ACDC.parquet", index=False)
    pd.DataFrame({"sample_id": mms, "qc_failure_reason": [None] * 4 + [critical] + [None] * 5}).to_parquet(p / "semantics" / "B_MMS.parquet", index=False)
    a_test = _a_rows(test, [0, 1, 1, 3, 4], [1.0, 0.8, 0.9, 0.6, 1.2])
    a_mms = _a_rows(mms, [3, 0, 1, 4, 3, 2, 0, 0, 1, 3], [1.0, 0.7, 0.2, 0.9, 1.0, 1.0, 0.4, 0.5, 0.5, 0.8])
    a_test.to_parquet(p / "formal" / "A_ACDC.parquet", index=False)
    a_mms.to_parquet(p / "formal" / "A_MMS.parquet", index=False)
    predictor = _predictor()
    joblib.dump(predictor, p / "formal" / "d_model_predictor.joblib")
    names = formal_descriptor_names()
    pd.DataFrame({"sample_id": test, "d_model": stages.predict_labels(predictor, a_test[names].to_numpy())}).to_parquet(p / "inference" / "d_model_acdc_test.parquet", index=False)
    pd.DataFrame({"sample_id": mms, "d_model": stages.predict_labels(predictor, a_mms[names].to_numpy())}).to_parquet(p / "inference" / "d_model_mms_external.parquet", index=False)
    pd.DataFrame({"sample_id": dev, "outer_fold": [1, 2, 3, 4, 5, 1, 2], "d_model_crossfit": ["DCM", "HCM", "HCM", "HCM", "NOR", "DCM", "RV"]}).to_parquet(
        p / "formal" / "d_model_crossfit_ACDC_dev.parquet", index=False)
    (p / "transitions" / "transition_model.json").write_text("{}", encoding="utf-8")
    for rb in ("d_model", "dataset_label", "measured_oracle"):
        (p / "rules" / rb).mkdir(parents=True, exist_ok=True)
        (p / "rules" / rb / "rulebook.json").write_text(json.dumps({"rulebook": rb, "records": []}), encoding="utf-8")
    stages.freeze_analysis_bundle(ctx)
    return ctx, a_test, a_mms


def _record_evaluations(ctx) -> None:
    for split, rel in (("acdc_test", "inference/d_model_acdc_test.parquet"), ("mms_external", "inference/d_model_mms_external.parquet")):
        stages._stage(ctx, f"evaluate:{split}", lambda: {}, [ctx.run.frozen_path], [ctx.run.path / rel], {"split": split}, status="EVALUATED")


def test_summarize_predictor_crosswalk_ihd_off_ontology_and_n_jobs(predictor_run, monkeypatch):
    ctx, a_test, a_mms = predictor_run
    with pytest.raises(stages.BundleError, match="evaluate:acdc_test"):
        stages.summarize_predictor(ctx)                              # requires the recorded evaluation
    _record_evaluations(ctx)
    assert stages.summarize_predictor(ctx).action == "RAN"
    metrics = ctx.run.path / "metrics"
    ext = json.loads((metrics / "predictor_mms_external.json").read_text(encoding="utf-8"))
    names = formal_descriptor_names()
    proba, labels = _softmax_labels(a_mms[names].to_numpy()[:, :5])
    same = [0, 1, 2, 3]                                              # row 4 is same-definition but critical QC (excluded)
    truth = np.array(["NOR", "DCM", "HCM", "RV"], dtype=object)
    expected = predictor_summary_v6(truth, labels[same], pd.DataFrame(proba[same], columns=["DCM", "HCM", "MINF", "NOR", "RV"]),
                                    ["DCM", "HCM", "MINF", "NOR", "RV"], clusters=np.array(["MMS:Q0X0Y0", "MMS:Q0X0Y0", "MMS:Q2X2Y2", "MMS:Q3X3Y3"], dtype=object),
                                    n_boot=int(ctx.study.bootstrap["fixed_model_bootstrap"]["replicates"]), seed=20260928, n_bins=10, min_valid_fraction=0.95)
    for metric in ("accuracy", "balanced_accuracy", "macro_f1", "macro_ovr_auroc", "ece", "log_loss"):
        assert ext["metrics"][metric] == expected["metrics"][metric], metric
    assert ext["metrics"]["accuracy"]["estimate"] == pytest.approx(float(np.mean(labels[same] == truth)))
    assert ext["counts"]["n_same_definition"] == 5 and ext["counts"]["n_excluded_critical_qc"] == 1
    assert ext["counts"]["n_evaluable"] == 4 and ext["counts"]["n_clusters"] == 3
    assert ext["crosswalk"]["found"] == {"HARMONIZABLE_SAME_DEFINITION": 5, "HARMONIZABLE_NOT_EXACT": 2, "OFF_ONTOLOGY": 3}
    ihd = ext["ihd_not_exact"]
    assert ihd["n"] == 2 and ihd["pooled"] is False and ihd["mapped_acdc_class"] == ["MINF"]
    assert ihd["n_d_model_equals_mapped_class"] == int(sum(labels[i] == "MINF" for i in (5, 6)))
    assert sum(v for k, v in ihd["d_model_distribution"].items() if k != "no_decision") == 2
    off = ext["off_ontology"]["d_model_distribution"]
    assert {k: v for k, v in off.items() if v and k != "no_decision"} == pd.Series(labels[[7, 8, 9]]).value_counts().to_dict()
    assert "metrics" not in ext["off_ontology"]                      # distribution only, no accuracy
    test = json.loads((metrics / "predictor_acdc_test.json").read_text(encoding="utf-8"))
    t_proba, t_labels = _softmax_labels(a_test[names].to_numpy()[:, :5])
    assert test["metrics"]["accuracy"]["estimate"] == pytest.approx(float(np.mean(t_labels == np.array(["DCM", "HCM", "MINF", "NOR", "RV"]))))
    assert test["metrics"]["log_loss"]["estimate"] == pytest.approx(float(-np.mean(np.log(t_proba[np.arange(5), [0, 1, 2, 3, 4]]))))
    oof = json.loads((metrics / "predictor_acdc_dev_oof.json").read_text(encoding="utf-8"))
    assert oof["counts"]["n_evaluable"] == 6 and oof["counts"]["n_excluded_critical_qc"] == 1
    assert oof["metrics"]["accuracy"]["estimate"] == pytest.approx(4 / 6) and oof["metrics"]["balanced_accuracy"]["estimate"] == pytest.approx(2 / 3)
    assert oof["metrics"]["macro_ovr_auroc"]["reason"] == oof["metrics"]["ece"]["reason"] == "OOF_PROBABILITIES_NOT_RECORDED"
    index = json.loads((metrics / "predictor_summary.json").read_text(encoding="utf-8"))
    assert index["semantic_comparator"]["reason"] == "COMPARATOR_NOT_FITTED_IN_RUN" and "mms_refit_eval" in index["not_included"]
    confusion = pd.read_parquet(metrics / "predictor_confusion.parquet")
    assert confusion.groupby("partition")["n"].sum().to_dict() == {"acdc_dev_oof": 6, "acdc_test": 5, "mms_external": 4}
    # The same summary with three workers (a new run root without the summary record): byte-identical outputs.
    other = RunRoot("v6-reporting-20260929-1201", ctx.run.artifact_root, subfolders=ctx.study.study["artifact_subfolders"], create=False)
    shutil.copytree(ctx.run.path, other.path)
    other.record_path("summarize-predictor").unlink()
    for f in (metrics).glob("predictor_*"):
        (other.path / "metrics" / f.name).unlink()
    assert stages.summarize_predictor(replace(ctx, run=other, n_jobs=3)).action == "RAN"
    for f in sorted(metrics.glob("predictor_*")):
        assert (other.path / "metrics" / f.name).read_bytes() == f.read_bytes(), f.name
    # The M&Ms part obeys the freeze guard like every M&Ms stage.
    monkeypatch.setattr(stages, "protocol_frozen", lambda *a, **k: False)
    with pytest.raises(stages.ProtocolFreezeError):
        stages.summarize_predictor(ctx)


# ----------------------------------------------------------------------------- once-only guard


def test_once_only_guard_refuses_recorded_stages_of_a_frozen_run(tmp_path):
    study_path = _study_copy(tmp_path)
    study = load_study(study_path)
    run = RunRoot("v6-guard-20260929-1300", tmp_path / "artifacts", subfolders=study.study["artifact_subfolders"])
    ctx = stages.StageContext(study=study, run=run, data_root=tmp_path / "data", n_jobs=1,
                              sources={"acdc": tmp_path / "absent", "mms": tmp_path / "absent"})
    source = run.path / "contracts" / "input.json"
    source.write_text("{}", encoding="utf-8")
    calls = []

    def stage(name: str, context) -> str:
        def fn():
            calls.append(name)
            (run.path / "metrics" / f"{name.replace(':', '_')}.json").write_text(json.dumps({"n": len(calls)}), encoding="utf-8")
        return stages._stage(context, name, fn, [source], [run.path / "metrics" / f"{name.replace(':', '_')}.json"], {"split": "acdc_test"},
                             status="EVALUATED").action

    assert stage("evaluate:acdc_test", ctx) == "RAN"
    edited = tmp_path / "study" / "configs" / "bootstrap.yaml"
    _edit_yaml(edited, lambda d: d["fixed_model_bootstrap"].update(replicates=150))
    changed = replace(ctx, study=load_study(study_path))
    assert stage("evaluate:acdc_test", changed) == "RAN"              # not frozen: a changed section re-runs (Section 14.2)
    run.frozen_path.write_text("{}", encoding="utf-8")
    assert stage("evaluate:acdc_test", changed) == "SKIPPED"          # frozen and identical: skip
    record = run.record_path("evaluate:acdc_test").read_bytes()
    _edit_yaml(edited, lambda d: d["fixed_model_bootstrap"].update(replicates=120))
    again = replace(ctx, study=load_study(study_path))
    n_calls = len(calls)
    with pytest.raises(stages.StageRecordLockedError, match=r"evaluate:acdc_test.*config hash.*new run id"):
        stage("evaluate:acdc_test", again)
    with pytest.raises(stages.StageRecordLockedError, match="--force"):
        stage("evaluate:acdc_test", replace(changed, force=True))
    source.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(stages.StageRecordLockedError, match="inputs"):
        stage("evaluate:acdc_test", changed)
    assert len(calls) == n_calls and run.record_path("evaluate:acdc_test").read_bytes() == record
    assert isinstance(stages.StageRecordLockedError("x"), stages.BundleError)   # exit 7 through the CLI's BundleError handler
    assert stage("summarize-predictor", again) == "RAN"               # a stage without a record runs normally
    first = stages.write_release_manifest(again)
    (run.path / "metrics" / "extra.json").write_text("{}", encoding="utf-8")
    assert first["action"] == "RAN" and stages.write_release_manifest(again)["action"] == "RAN"   # re-hashes the run root


# ----------------------------------------------------------------------------- licence guard and LaTeX


def test_case_identifier_guard_and_latex_escape():
    with pytest.raises(TableBuildError, match="case identifier"):
        assert_no_case_identifiers({"t.csv": "stratum,A1B2C3,0.5"}, {"A1B2C3", "patient001"})
    assert_no_case_identifiers({"t.csv": "vendor,A,0.5\nsha,a1b2c3" + "0" * 58}, {"A1B2C3"})   # tokens, not substrings of hashes
    assert latex_escape("`B = TA` & 5 % ≥ λ_j") == r"\texttt{B = TA} \& 5 \% $\geq$ $\lambda$\_j"
    assert latex_escape("B̂") == r"$\hat{B}$" and latex_escape("**bold** 1–2") == "bold 1--2"
