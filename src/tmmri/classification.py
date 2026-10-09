from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from .io import resolve_path, utc_now_iso, write_table_pair, write_yaml


def expected_calibration_error_equal_mass(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> tuple[float, pd.DataFrame]:
    if proba.ndim != 2 or len(y_true) != proba.shape[0]:
        raise ValueError("Invalid probability matrix shape.")
    conf = np.max(proba, axis=1)
    pred = np.argmax(proba, axis=1)
    correct = (pred == y_true).astype(float)
    order = np.argsort(conf, kind="mergesort")
    bins = np.array_split(order, n_bins)
    rows = []
    ece = 0.0
    n = len(y_true)
    for i, idx in enumerate(bins):
        if len(idx) == 0:
            rows.append({"bin_id": i, "count": 0, "mean_confidence": np.nan, "accuracy": np.nan, "abs_gap": np.nan})
            continue
        mean_conf = float(np.mean(conf[idx]))
        acc = float(np.mean(correct[idx]))
        gap = abs(mean_conf - acc)
        ece += len(idx) / max(n, 1) * gap
        rows.append({"bin_id": i, "count": int(len(idx)), "mean_confidence": mean_conf, "accuracy": acc, "abs_gap": gap})
    return float(ece), pd.DataFrame(rows)


def _prepare_xy(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    x = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = df["disease_label"].astype(str)
    return x, y


def run_classification_baseline(b_acdc: pd.DataFrame, feature_cols: list[str], experiment_id: str) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    train = b_acdc[b_acdc["split"].astype(str).eq("train")].copy()
    eval_df = b_acdc[b_acdc["split"].astype(str).isin(["val", "test"])].copy()
    if train.empty or eval_df.empty or train["disease_label"].nunique() < 2:
        metrics = pd.DataFrame([{"status": "skipped", "reason": "insufficient_train_or_eval_labels", "experiment_id": experiment_id}])
        bins = pd.DataFrame()
        return metrics, bins, []
    x_train, y_train = _prepare_xy(train, feature_cols)
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=20260611)),
        ]
    )
    model.fit(x_train, y_train)
    artifacts: list[Path] = []
    rows = []
    bin_frames = []
    classes = list(model.named_steps["clf"].classes_)
    for split_name, split_df in eval_df.groupby("split"):
        x, y = _prepare_xy(split_df, feature_cols)
        pred = model.predict(x)
        proba = model.predict_proba(x)
        labels = np.array([classes.index(v) for v in y.astype(str)], dtype=int)
        pred_labels = np.array([classes.index(v) for v in pred.astype(str)], dtype=int)
        ece, bins = expected_calibration_error_equal_mass(labels, proba, n_bins=10)
        bins["split"] = split_name
        bins["experiment_id"] = experiment_id
        bin_frames.append(bins)
        macro_f1 = f1_score(y, pred, average="macro", zero_division=0)
        bal_acc = balanced_accuracy_score(y, pred)
        acc = accuracy_score(y, pred)
        try:
            auroc = roc_auc_score(pd.get_dummies(y).reindex(columns=classes, fill_value=0), proba, average="macro", multi_class="ovr")
        except Exception:
            auroc = np.nan
        rows.append(
            {
                "split": split_name,
                "accuracy": float(acc),
                "balanced_accuracy": float(bal_acc),
                "macro_f1": float(macro_f1),
                "macro_auroc_ovr": float(auroc) if np.isfinite(auroc) else np.nan,
                "ece_10_equal_mass": float(ece),
                "n": int(len(split_df)),
                "experiment_id": experiment_id,
                "status": "completed",
            }
        )
        cm = confusion_matrix(y, pred, labels=classes)
        cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes]).reset_index(names="true_label")
        artifacts.extend(write_table_pair(cm_df, f"artifacts/reports/classification/confusion_matrix_{split_name}_{experiment_id}.parquet", f"artifacts/reports/classification/confusion_matrix_{split_name}_{experiment_id}.csv"))
    metrics = pd.DataFrame(rows)
    bins_all = pd.concat(bin_frames, ignore_index=True) if bin_frames else pd.DataFrame()
    model_path = resolve_path(f"artifacts/models/classification/classifier_B_measured_{experiment_id}.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    artifacts.append(model_path)
    artifacts.extend(write_table_pair(metrics, f"artifacts/reports/classification/classification_metrics_{experiment_id}.parquet", f"artifacts/reports/classification/classification_metrics_{experiment_id}.csv"))
    artifacts.extend(write_table_pair(bins_all, f"artifacts/reports/classification/calibration_bins_{experiment_id}.parquet", f"artifacts/reports/classification/calibration_bins_{experiment_id}.csv"))
    artifacts.append(write_yaml({"experiment_id": experiment_id, "feature_cols": feature_cols, "model": "median_impute_standardize_multinomial_logistic_regression", "clinical_status": "research_use_only", "generated_at_utc": utc_now_iso()}, f"artifacts/models/classification/classifier_B_measured_{experiment_id}_metadata.yaml"))
    report = resolve_path(f"artifacts/reports/classification/classification_report_{experiment_id}.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Transparent Classification Baseline - {experiment_id}",
        "",
        "Clinical status: research use only. This classifier is not clinically validated for diagnosis, triage, treatment selection, or patient management.",
        "",
        "This baseline uses measured Matrix B features on ACDC only. It is a supporting technical artifact and is not the central DSS.",
        "",
        metrics.to_markdown(index=False) if not metrics.empty else "No metrics.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifacts.append(report)
    return metrics, bins_all, artifacts


# ===============================================================================================================
# Protocol v6 audited predictor d_model (SPECIFICATION.md 6.0.0, Section 8; Decision D5; configs/formal.yaml
# audited_predictor). The legacy measured-B baseline above is unchanged (it is the semantic comparator).
# ===============================================================================================================
import inspect
from dataclasses import dataclass, field
from typing import Any

from .transition import FormalStandardizer


def _predictor_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Accept configs/formal.yaml as a whole or its ``audited_predictor`` block."""
    return dict(cfg["audited_predictor"]) if "audited_predictor" in cfg else dict(cfg)


def _logistic_regression(pcfg: dict[str, Any]) -> LogisticRegression:
    if str(pcfg.get("model", "multinomial_logistic_regression")) != "multinomial_logistic_regression":
        raise ValueError(f"Unsupported audited predictor model {pcfg.get('model')!r}")
    kwargs: dict[str, Any] = {
        "C": float(pcfg["C"]),
        "class_weight": pcfg.get("class_weight"),
        "solver": str(pcfg["solver"]),
        "max_iter": int(pcfg["max_iter"]),
        "tol": float(pcfg["tol"]),
        "random_state": int(pcfg["random_state"]),
    }
    penalty = str(pcfg.get("penalty", "l2"))
    if penalty != "l2":
        raise ValueError("Only the L2 penalty is configured for the audited predictor")
    # scikit-learn >= 1.8 deprecates ``penalty``; L2 is ``l1_ratio = 0`` there (identical objective).
    if inspect.signature(LogisticRegression).parameters["penalty"].default == "deprecated":
        kwargs["l1_ratio"] = 0.0
    else:
        kwargs["penalty"] = "l2"
    return LogisticRegression(**kwargs)


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class FormalPredictor:
    """Frozen multinomial LR on standardized formal A. Probabilities use the stored coefficients (softmax), so a
    deserialized predictor reproduces the fitted one exactly; ``predict`` is argmax with ties by class order."""

    classes: list[str]
    standardizer: FormalStandardizer
    coef: np.ndarray            # (n_classes, n_kept)
    intercept: np.ndarray       # (n_classes,)
    config: dict[str, Any] = field(default_factory=dict)
    missing_classes: list[str] = field(default_factory=list)
    n_fit_rows: int = 0

    def decision_function(self, A: np.ndarray) -> np.ndarray:
        return self.standardizer.transform(np.asarray(A, dtype=float)) @ self.coef.T + self.intercept

    def predict_proba(self, A: np.ndarray, index: pd.Index | None = None) -> pd.DataFrame:
        a = np.asarray(A, dtype=float)
        idx = index if index is not None else pd.RangeIndex(a.shape[0])
        out = np.full((a.shape[0], len(self.classes)), np.nan)
        finite = np.all(np.isfinite(a[:, self.standardizer.keep]), axis=1)
        if finite.any():
            out[finite] = _softmax(self.decision_function(a[finite]))
        return pd.DataFrame(out, index=idx, columns=list(self.classes))

    def predict(self, A: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(A).to_numpy()
        labels = np.array([None] * proba.shape[0], dtype=object)
        finite = np.all(np.isfinite(proba), axis=1)
        labels[finite] = np.asarray(self.classes, dtype=object)[np.argmax(proba[finite], axis=1)]
        return labels

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "formal_predictor_v6",
            "classes": list(self.classes),
            "standardizer": self.standardizer.to_dict(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept.tolist(),
            "config": dict(self.config),
            "missing_classes": list(self.missing_classes),
            "n_fit_rows": int(self.n_fit_rows),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FormalPredictor":
        return cls(
            classes=list(d["classes"]),
            standardizer=FormalStandardizer.from_dict(d["standardizer"]),
            coef=np.asarray(d["coef"], dtype=float),
            intercept=np.asarray(d["intercept"], dtype=float),
            config=dict(d.get("config", {})),
            missing_classes=list(d.get("missing_classes", [])),
            n_fit_rows=int(d.get("n_fit_rows", 0)),
        )


def fit_formal_predictor(A: np.ndarray, y: pd.Series | np.ndarray, cfg: dict[str, Any], *, fit_mask: np.ndarray | None = None) -> FormalPredictor:
    """Fit d_model per configs/formal.yaml audited_predictor on the fit rows (finite A, known class)."""
    pcfg = _predictor_cfg(cfg)
    a = np.asarray(A, dtype=float)
    labels = pd.Series(np.asarray(y, dtype=object)).astype("string")
    classes_cfg = [str(c) for c in pcfg["classes"]]
    rows = np.all(np.isfinite(a), axis=1) & labels.notna().to_numpy()
    if fit_mask is not None:
        rows &= np.asarray(fit_mask, dtype=bool)
    unknown = sorted(set(labels[rows].astype(str)) - set(classes_cfg))
    if unknown:
        raise ValueError(f"Labels outside the configured classes: {unknown}")
    present = [c for c in classes_cfg if c in set(labels[rows].astype(str))]
    if len(present) < 2:
        raise ValueError("The audited predictor needs at least two classes on the fit rows")
    scaler = FormalStandardizer.fit(a[rows])
    z = scaler.transform(a[rows])
    model = _logistic_regression(pcfg)
    model.fit(z, labels[rows].astype(str).to_numpy())
    order = [list(model.classes_).index(c) for c in present]
    coef = np.asarray(model.coef_, dtype=float)
    intercept = np.asarray(model.intercept_, dtype=float)
    if coef.shape[0] == 1 and len(present) == 2:  # binary sklearn parametrization -> two-class softmax
        coef = np.vstack([-coef[0] / 2.0, coef[0] / 2.0])
        intercept = np.array([-intercept[0] / 2.0, intercept[0] / 2.0])
    return FormalPredictor(
        classes=present,
        standardizer=scaler,
        coef=coef[order],
        intercept=intercept[order],
        config=pcfg,
        missing_classes=[c for c in classes_cfg if c not in present],
        n_fit_rows=int(rows.sum()),
    )


def classification_metrics_v6(y_true: pd.Series | np.ndarray, proba: pd.DataFrame, *, n_bins: int) -> tuple[dict[str, Any], pd.DataFrame]:
    """Accuracy, balanced accuracy, macro-F1, macro one-vs-rest AUROC and equal-mass ECE (``n_bins`` from
    configs/bootstrap.yaml calibration). Rows with an unknown reference or non-finite probabilities are dropped."""
    classes = list(proba.columns)
    truth = pd.Series(np.asarray(y_true, dtype=object)).astype("string")
    p = proba.to_numpy(dtype=float)
    keep = truth.notna().to_numpy() & np.all(np.isfinite(p), axis=1) & truth.isin(classes).to_numpy()
    truth = truth[keep].astype(str).to_numpy()
    p = p[keep]
    if truth.size == 0:
        return {"n": 0, "status": "NE", "reason": "no_evaluable_rows"}, pd.DataFrame()
    pred = np.asarray(classes, dtype=object)[np.argmax(p, axis=1)].astype(str)
    idx = np.array([classes.index(t) for t in truth], dtype=int)
    ece, bins = expected_calibration_error_equal_mass(idx, p, n_bins=int(n_bins))
    aurocs = []
    for j, c in enumerate(classes):
        positives = truth == c
        if positives.any() and (~positives).any():
            aurocs.append(float(roc_auc_score(positives.astype(int), p[:, j])))
    metrics = {
        "n": int(truth.size),
        "accuracy": float(accuracy_score(truth, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, labels=[c for c in classes if c in set(truth)], average="macro", zero_division=0)),
        "macro_ovr_auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
        "auroc_classes_used": int(len(aurocs)),
        "ece": float(ece),
        "ece_bins": int(n_bins),
        "ece_binning": "equal_mass",
        "status": "EVALUATED",
    }
    return metrics, bins


# ===============================================================================================================
# Audited-predictor summary (SPECIFICATION.md 13.1-13.2 "Audited predictor"; stage summarize-predictor).  It reads
# recorded decisions (and probabilities of a frozen predictor) only: nothing is fitted here.
# ===============================================================================================================
PREDICTOR_INTERVAL_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "macro_ovr_auroc", "ece")
PROBABILITY_METRICS = ("macro_ovr_auroc", "ece", "log_loss")


def _replicate_predictor_metrics(t: np.ndarray, pred: np.ndarray, p: np.ndarray | None, n_classes: int, n_bins: int,
                                 present_ref: np.ndarray) -> np.ndarray:
    """accuracy, balanced accuracy, macro-F1, macro OvR AUROC and ECE of one (resampled) sample, NaN when a macro
    metric is not comparable (the resample lacks a reference class of the full sample).  The formulas equal
    :func:`classification_metrics_v6` (F1 = 2TP / (2TP + FP + FN) over the classes present in the reference;
    AUROC by the rank statistic, ties averaged)."""
    out = np.full(len(PREDICTOR_INTERVAL_METRICS), np.nan)
    if t.size == 0:
        return out
    out[0] = float(np.mean(t == pred))
    present = np.unique(t)
    if present.size != present_ref.size or not np.array_equal(present, present_ref):
        if p is not None:
            out[4] = expected_calibration_error_equal_mass(t, p, n_bins=n_bins)[0]
        return out
    recalls, f1s = [], []
    for c in present:
        is_c, pred_c = t == c, pred == c
        tp = float(np.sum(is_c & pred_c))
        recalls.append(tp / float(is_c.sum()))
        f1s.append(2.0 * tp / (2.0 * tp + float(np.sum(~is_c & pred_c)) + float(np.sum(is_c & ~pred_c))))
    out[1], out[2] = float(np.mean(recalls)), float(np.mean(f1s))
    if p is not None:
        aurocs = [_auroc_from_scores((t == c).astype(int), p[:, c]) for c in range(n_classes) if (t == c).any() and (t != c).any()]
        out[3] = float(np.mean(aurocs)) if aurocs else np.nan
        out[4] = expected_calibration_error_equal_mass(t, p, n_bins=n_bins)[0]
    return out


def _predictor_boot_chunk(t: np.ndarray, pred: np.ndarray, p: np.ndarray | None, n_classes: int, n_bins: int, present_ref: np.ndarray,
                          unit_cluster: np.ndarray, n_resample: int, seeds: list[np.random.SeedSequence]) -> list[np.ndarray]:
    from .metrics import _cluster_draw_weights

    out = []
    for ss in seeds:
        weights = _cluster_draw_weights(np.random.default_rng(ss), unit_cluster, n_resample).astype(int)
        rows = np.repeat(np.arange(t.size), weights)
        out.append(_replicate_predictor_metrics(t[rows], pred[rows], None if p is None else p[rows], n_classes, n_bins, present_ref))
    return out


def predictor_cluster_bootstrap(t: np.ndarray, pred: np.ndarray, p: np.ndarray | None, n_classes: int, *, clusters: np.ndarray, n_boot: int,
                                seed: int, n_bins: int, n_jobs: int = 1) -> np.ndarray:
    """``(n_boot, 5)`` replicate matrix of :data:`PREDICTOR_INTERVAL_METRICS`: replicate r draws clusters with
    ``SeedSequence(seed).spawn(n_boot)[r]`` (each drawn cluster brings all its rows; clusters numbered by first row),
    so the draws equal those of the fixed-model bootstrap of the primary estimand and never depend on ``n_jobs``."""
    from .metrics import _resampling_clusters

    unit_cluster, n_resample = _resampling_clusters(np.arange(t.size), int(t.size), clusters)
    present_ref = np.unique(t)
    children = np.random.SeedSequence(int(seed)).spawn(int(n_boot))
    n_chunks = max(1, min(int(n_boot), max(1, int(n_jobs)) * 4))
    chunks = [c for c in np.array_split(np.arange(int(n_boot)), n_chunks) if c.size]
    args = (t, pred, p, n_classes, n_bins, present_ref, unit_cluster, n_resample)
    if int(n_jobs) <= 1:
        parts = [_predictor_boot_chunk(*args, [children[i] for i in c]) for c in chunks]
    else:
        from joblib import Parallel, delayed

        parts = Parallel(n_jobs=int(n_jobs), backend="loky")(delayed(_predictor_boot_chunk)(*args, [children[i] for i in c]) for c in chunks)
    rows = [r for part in parts for r in part]
    return np.vstack(rows) if rows else np.empty((0, len(PREDICTOR_INTERVAL_METRICS)))


def predictor_summary_v6(y_true: pd.Series | np.ndarray, d_model: pd.Series | np.ndarray, proba: pd.DataFrame | None, classes: list[str], *,
                         clusters: pd.Series | np.ndarray, n_boot: int, seed: int, n_bins: int, min_valid_fraction: float, level: float = 0.95,
                         n_jobs: int = 1, probability_reason: str | None = None) -> dict[str, Any]:
    """SPECIFICATION.md 13.2 "Audited predictor" on row-aligned recorded decisions ``d_model`` (and, when recorded or
    reproducible from the frozen predictor, class probabilities ``proba`` with columns ``classes``).

    Evaluable rows: a known reference inside ``classes`` and a decision.  Point estimates of accuracy, balanced
    accuracy, macro-F1, macro one-vs-rest AUROC and equal-mass ECE (``n_bins``, realized bin counts) come from
    :func:`classification_metrics_v6` (label metrics directly when no probabilities exist: AUROC, ECE and log loss are
    then ``NE`` with ``probability_reason``); per-class recall, log loss and the confusion matrix are companions.
    Percentile intervals from the fixed-model cluster bootstrap (``clusters`` = ``cluster_id``); an interval is ``NE``
    when fewer than ``min_valid_fraction`` of the replicates are valid.  With ``proba`` the argmax must reproduce
    ``d_model`` exactly (ties by class order), else :class:`ValueError`.
    """
    classes = [str(c) for c in classes]
    truth = pd.Series(np.asarray(y_true, dtype=object)).astype(object)
    decision = pd.Series(np.asarray(d_model, dtype=object)).astype(object)
    cl = pd.Series(np.asarray(clusters, dtype=object)).astype(str)
    if not (len(truth) == len(decision) == len(cl)) or (proba is not None and len(proba) != len(truth)):
        raise ValueError("reference, decisions, clusters and probabilities must be row-aligned")
    known = truth.notna().to_numpy() & truth.map(lambda v: v is not None and str(v) not in ("", "nan", "None")).to_numpy()
    decided = decision.notna().to_numpy() & decision.map(lambda v: v is not None and str(v) not in ("", "nan", "None")).to_numpy()
    in_classes = truth.astype(str).isin(classes).to_numpy()
    p_all = None if proba is None else proba[classes].to_numpy(dtype=float)
    if p_all is not None:
        finite = np.all(np.isfinite(p_all), axis=1)
        if not np.array_equal(finite, decided):
            raise ValueError("finite probabilities and recorded decisions disagree on which rows have a decision")
        argmax = np.full(len(truth), None, dtype=object)
        argmax[finite] = np.asarray(classes, dtype=object)[np.argmax(p_all[finite], axis=1)]
        if not all(str(a) == str(d) for a, d in zip(argmax[finite], decision.to_numpy()[finite])):
            raise ValueError("the frozen predictor's argmax does not reproduce the recorded d_model")
    keep = known & decided & in_classes
    counts = {"n_rows": int(len(truth)), "n_no_decision": int((~decided).sum()), "n_reference_unknown": int((~known).sum()),
              "n_reference_outside_predictor_classes": int((known & ~in_classes).sum()), "n_evaluable": int(keep.sum()),
              "n_clusters": int(cl[keep].nunique())}
    boot_cfg = {"replicates": int(n_boot), "seed": int(seed), "unit": "cluster_id", "interval": f"percentile_{int(round(level * 100))}",
                "min_valid_fraction": float(min_valid_fraction), "draw": "SeedSequence(seed).spawn(replicates)[r]; clusters numbered by first row",
                "valid_replicate": "finite; for balanced accuracy, macro-F1 and macro AUROC the replicate must contain every reference class of "
                                   "the evaluated sample (macro averages over the same classes)"}
    out: dict[str, Any] = {"classes": classes, "counts": counts, "bootstrap": boot_cfg, "ece_bins": int(n_bins), "ece_binning": "equal_mass",
                           "probabilities": "recorded" if p_all is not None else "not_recorded"}
    if not keep.any():
        out.update({"status": "NE", "reason": "no_evaluable_rows", "metrics": {m: {"estimate": None, "status": "NE", "reason": "no_evaluable_rows"}
                                                                               for m in (*PREDICTOR_INTERVAL_METRICS, "log_loss")}})
        return out
    t_lab = truth[keep].astype(str).to_numpy()
    d_lab = decision[keep].astype(str).to_numpy()
    t = np.array([classes.index(v) for v in t_lab], dtype=int)
    pred = np.array([classes.index(v) if v in classes else -1 for v in d_lab], dtype=int)
    p = None if p_all is None else p_all[keep]
    point = _replicate_predictor_metrics(t, pred, p, len(classes), int(n_bins), np.unique(t))
    realized: list[dict[str, Any]] = []
    if p is not None:
        import warnings

        with warnings.catch_warnings():      # balanced accuracy over the reference classes; predicted-only classes are expected
            warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true", category=UserWarning)
            reference, bins = classification_metrics_v6(pd.Series(t_lab), pd.DataFrame(p, columns=classes), n_bins=int(n_bins))
        for j, m in enumerate(PREDICTOR_INTERVAL_METRICS):
            if not np.isclose(point[j], reference[m], rtol=0, atol=1e-12, equal_nan=True):
                raise ValueError(f"bootstrap kernel disagrees with classification_metrics_v6 on {m}")
            point[j] = reference[m]
        realized = [{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()} for r in bins.to_dict("records")]
    else:
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true", category=UserWarning)
            reference = {"accuracy": float(accuracy_score(t_lab, d_lab)), "balanced_accuracy": float(balanced_accuracy_score(t_lab, d_lab)),
                         "macro_f1": float(f1_score(t_lab, d_lab, labels=[c for c in classes if c in set(t_lab)], average="macro", zero_division=0))}
        for j, m in enumerate(PREDICTOR_INTERVAL_METRICS[:3]):
            if not np.isclose(point[j], reference[m], rtol=0, atol=1e-12):
                raise ValueError(f"bootstrap kernel disagrees with scikit-learn on {m}")
            point[j] = reference[m]
    boot = predictor_cluster_bootstrap(t, pred, p, len(classes), clusters=cl[keep].to_numpy(dtype=object), n_boot=n_boot, seed=seed,
                                       n_bins=int(n_bins), n_jobs=n_jobs)
    alpha = (1.0 - level) / 2.0
    metrics: dict[str, Any] = {}
    for j, m in enumerate(PREDICTOR_INTERVAL_METRICS):
        if p is None and m in PROBABILITY_METRICS:
            metrics[m] = {"estimate": None, "ci_lower": None, "ci_upper": None, "status": "NE", "reason": probability_reason or "PROBABILITIES_NOT_RECORDED"}
            continue
        col = boot[:, j] if boot.size else np.empty(0)
        valid = col[np.isfinite(col)]
        frac = float(valid.size / col.size) if col.size else 0.0
        entry = {"estimate": float(point[j]) if np.isfinite(point[j]) else None, "n_boot_valid": int(valid.size), "valid_fraction": frac}
        if entry["estimate"] is None:
            entry.update({"ci_lower": None, "ci_upper": None, "status": "NE", "reason": "metric_undefined_on_sample"})
        elif frac < float(min_valid_fraction):
            entry.update({"ci_lower": None, "ci_upper": None, "status": "NE", "reason": "fewer_than_min_valid_bootstrap_replicates"})
        else:
            entry.update({"ci_lower": float(np.quantile(valid, alpha)), "ci_upper": float(np.quantile(valid, 1.0 - alpha)), "status": "EVALUATED",
                          "reason": None})
        metrics[m] = entry
    if p is None:
        metrics["log_loss"] = {"estimate": None, "status": "NE", "reason": probability_reason or "PROBABILITIES_NOT_RECORDED"}
    else:
        loss = float(np.mean(-np.log(np.clip(p[np.arange(t.size), t], 1e-15, 1.0))))
        metrics["log_loss"] = {"estimate": loss, "status": "EVALUATED", "reason": None, "role": "descriptive_companion", "clip": 1e-15}
    recall = {}
    for k, c in enumerate(classes):
        n_true = int((t == k).sum())
        recall[c] = {"n_reference": n_true, "recall": float(np.mean(pred[t == k] == k)) if n_true else None,
                     "status": "EVALUATED" if n_true else "NE", "reason": None if n_true else "class_absent_from_reference"}
    predicted_labels = classes + sorted({v for v in d_lab if v not in classes})
    matrix = [[int(np.sum((t_lab == a) & (d_lab == b))) for b in predicted_labels] for a in classes]
    out.update({"status": "EVALUATED", "reason": None, "metrics": metrics, "ece_realized_bins": realized,
                "companions": {"per_class_recall": recall, "confusion_matrix": {"rows_reference": classes, "columns_d_model": predicted_labels,
                                                                                 "counts": matrix}}})
    return out


def label_distribution(labels: pd.Series | np.ndarray, classes: list[str]) -> dict[str, int]:
    """Counts of each class among decided rows (``no_decision`` for rows without one); no reference is used."""
    s = pd.Series(np.asarray(labels, dtype=object))
    decided = s.map(lambda v: v is not None and not (isinstance(v, float) and np.isnan(v)) and str(v) not in ("", "None", "nan"))
    out = {str(c): int((s[decided].astype(str) == str(c)).sum()) for c in classes}
    for extra in sorted(set(s[decided].astype(str)) - {str(c) for c in classes}):
        out[extra] = int((s[decided].astype(str) == extra).sum())
    out["no_decision"] = int((~decided).sum())
    return out


# ===============================================================================================================
# Shift audit (SPECIFICATION.md 12.3; configs/bootstrap.yaml shift_audit): a label-free two-sample domain classifier
# on standardized formal A between acdc_dev and M&Ms external cases.  A diagnostic, never part of the bundle.
# ===============================================================================================================


def hash_rule_folds(sample_ids: list[str], domains: np.ndarray, salt: str, n_folds: int,
                    cluster_keys: list[str] | None = None) -> np.ndarray:
    """Within each domain, rank the distinct fold keys by ``SHA256(salt + key)`` hex digest ascending (ties by key);
    ``fold = (rank mod n_folds) + 1`` for every row carrying that key.  The key is ``cluster_id`` when
    ``cluster_keys`` is given (the ``sample_id`` for a singleton), so a declared duplicate-image pair always shares a
    fold; with singleton clusters the folds equal the per-case assignment.  Independent of the input order."""
    import hashlib

    ids = [str(s) for s in sample_ids]
    keys = ids if cluster_keys is None else [str(k) for k in cluster_keys]
    if len(keys) != len(ids):
        raise ValueError("cluster keys must be row-aligned with the sample ids")
    dom = np.asarray(domains)
    folds = np.zeros(len(ids), dtype=int)
    for d in sorted(set(dom.tolist())):
        members = [i for i in range(len(ids)) if dom[i] == d]
        distinct = sorted({keys[i] for i in members}, key=lambda k: (hashlib.sha256((salt + k).encode("utf-8")).hexdigest(), k))
        fold_of = {key: rank % int(n_folds) + 1 for rank, key in enumerate(distinct)}
        for i in members:
            folds[i] = fold_of[keys[i]]
    return folds


def _auroc_from_scores(y: np.ndarray, s: np.ndarray) -> float:
    from scipy.stats import rankdata

    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(s)
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _domain_clusters(idx: np.ndarray, clusters: list[str] | None) -> list[np.ndarray]:
    """Row positions of each cluster of one domain, clusters in order of first appearance."""
    if clusters is None:
        return [np.array([i]) for i in idx]
    labels = pd.Series(np.asarray(clusters, dtype=object)).astype(str).to_numpy()
    if labels.size != idx.size:
        raise ValueError("cluster labels must be row-aligned with the domain's ids")
    codes, uniques = pd.factorize(labels)
    return [idx[codes == k] for k in range(len(uniques))]


def domain_shift_audit(z_reference: np.ndarray, z_comparison: np.ndarray, reference_ids: list[str], comparison_ids: list[str],
                       cfg: dict[str, Any], *, n_boot: int, seed: int, min_cases: int, level: float = 0.95,
                       reference_clusters: list[str] | None = None, comparison_clusters: list[str] | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    """Out-of-fold AUROC of an L2 logistic regression separating ``comparison`` (label 1) from ``reference`` (label 0)
    rows of Z, with hash-rule cross-fitting of clusters within domain (both members of a declared pair share a fold)
    and a percentile bootstrap resampling clusters within domain.

    ``reference_clusters`` / ``comparison_clusters`` are the row-aligned ``cluster_id`` values (default: every case its
    own cluster); a drawn cluster brings all its rows, so a declared duplicate-image pair is resampled together.
    Clusters are ordered by first appearance, so singleton clusters reproduce the case bootstrap draw for draw.

    Returns the result record and the out-of-fold score table (``sample_id``, ``domain``, ``fold``, ``score``).
    A comparison group below ``min_cases`` keeps its point estimate; its interval is NE (STRATUM_BELOW_MIN_CASES).
    """
    cross = cfg["cross_fitting"]
    k = int(cross["n_folds"])
    clf_cfg = dict(cfg["classifier"])
    if str(clf_cfg.get("model")) != "logistic_regression":
        raise ValueError("configs/bootstrap.yaml shift_audit.classifier.model must be logistic_regression")
    z = np.vstack([np.asarray(z_reference, dtype=float), np.asarray(z_comparison, dtype=float)])
    y = np.r_[np.zeros(len(reference_ids), dtype=int), np.ones(len(comparison_ids), dtype=int)]
    ids = [str(s) for s in reference_ids] + [str(s) for s in comparison_ids]
    record: dict[str, Any] = {"n_reference": int(len(reference_ids)), "n_comparison": int(len(comparison_ids)), "n_folds": k,
                              "metric": str(cfg.get("metric", "out_of_fold_auroc")), "min_cases": int(min_cases)}
    if min(len(reference_ids), len(comparison_ids)) < k:
        record.update({"auroc": None, "ci_lower": None, "ci_upper": None, "status": "NE", "reason": "fewer_cases_than_folds"})
        return record, pd.DataFrame(columns=["sample_id", "domain", "fold", "score"])
    fold_keys = None
    if reference_clusters is not None or comparison_clusters is not None:
        fold_keys = (list(reference_clusters) if reference_clusters is not None else [str(s) for s in reference_ids]) + \
                    (list(comparison_clusters) if comparison_clusters is not None else [str(s) for s in comparison_ids])
    folds = hash_rule_folds(ids, y, str(cross["hash_salt"]), k, cluster_keys=fold_keys)
    record["fold_unit"] = "cluster_id"
    scores = np.full(len(ids), np.nan)
    for f in range(1, k + 1):
        held = folds == f
        model = _logistic_regression({**clf_cfg, "model": "multinomial_logistic_regression", "random_state": 0})
        model.fit(z[~held], y[~held])
        scores[held] = model.predict_proba(z[held])[:, list(model.classes_).index(1)]
    auc = _auroc_from_scores(y, scores)
    ref_idx, cmp_idx = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    ref_groups, cmp_groups = _domain_clusters(ref_idx, reference_clusters), _domain_clusters(cmp_idx, comparison_clusters)
    record.update({"n_reference_clusters": len(ref_groups), "n_comparison_clusters": len(cmp_groups), "resampling_unit": "cluster_id"})
    boot = []
    for ss in np.random.SeedSequence(int(seed)).spawn(int(n_boot)):
        rng = np.random.default_rng(ss)
        picks_ref = rng.choice(len(ref_groups), size=len(ref_groups), replace=True)
        picks_cmp = rng.choice(len(cmp_groups), size=len(cmp_groups), replace=True)
        draw = np.concatenate([ref_groups[g] for g in picks_ref] + [cmp_groups[g] for g in picks_cmp])
        boot.append(_auroc_from_scores(y[draw], scores[draw]))
    boot_arr = np.asarray(boot, dtype=float)
    boot_arr = boot_arr[np.isfinite(boot_arr)]
    alpha = (1.0 - level) / 2.0
    record.update({"auroc": auc, "n_boot": int(n_boot), "n_boot_valid": int(boot_arr.size), "seed": int(seed)})
    if len(comparison_ids) < int(min_cases):
        record.update({"ci_lower": None, "ci_upper": None, "status": "NE", "reason": "STRATUM_BELOW_MIN_CASES", "descriptive_only": True})
    else:
        record.update({"ci_lower": float(np.quantile(boot_arr, alpha)), "ci_upper": float(np.quantile(boot_arr, 1.0 - alpha)),
                       "status": "EVALUATED", "reason": None, "descriptive_only": False})
    table = pd.DataFrame({"sample_id": ids, "domain": np.where(y == 1, "comparison", "reference"), "fold": folds, "score": scores})
    return record, table
