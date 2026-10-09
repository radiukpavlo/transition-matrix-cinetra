"""Protocol-v6.0.1 reporting additions A2, A4 and A5 (docs/MANUSCRIPT_PROTOCOL_2026-10-02.md Section D; protocol 2.1
Section D item 12).

A2. ``fit-predictor`` keeps the five outer-fold audited predictors it already fits for the cross-fitted decisions and
writes them with their out-of-fold class probabilities in the shared prediction schema (contract C4, ``model_id``
``audited_predictor``) under ``formal/audited_predictor/``.  The three v6.0.0 outputs of the stage are unchanged byte
for byte (the same fitted objects; nothing is refitted or reordered).

A4. The measured-B comparator (``configs/formal.yaml`` ``comparator_predictor``, ``settings: same_as_audited_predictor``):
the audited-predictor logistic regression fitted on measured B of the registry primary family instead of formal A,
with fit-partition median imputation before the fit-partition standardization.  It is cross-fitted over the v6 outer
folds of ``acdc_dev`` (medians and scaler from the training folds only) and refitted on every fit row of ``acdc_dev``;
the final model, the fold models and their SHA-256 (``bundle.json``) are written by ``fit-predictor``, so the
comparator is frozen with the analysis bundle (a fitting stage refuses after ``.stages/FROZEN.json``) without changing
the bundle hash.  It never reads ``acdc_test`` or an external partition while fitted.

A5 and the extended summaries.  The read-only stage records ``summarize-predictor-extended:<partition>`` (one per
partition, so a later prospective partition never re-executes an earlier record) summarize, from recorded artifacts
only: both predictors on the partition (accuracy, balanced accuracy, macro-F1, macro one-vs-rest AUROC, equal-mass ECE
and log loss exactly as ``summarize-predictor``, plus ECE with 10 equal-width bins and the multiclass Brier score, all
with the fixed-model cluster bootstrap); per-vendor and per-centre strata of the external partitions; and the
Bland-Altman agreement of the recorded B-hat with measured B per primary target (bias and 95 % limits of agreement
with cluster-bootstrap intervals).  A stratum with fewer than ``min_cases_per_stratum`` clusters keeps its counts and
point estimates; its intervals are ``NE`` (``STRATUM_BELOW_MIN_CASES``).  The comparator is evaluated on each
evaluation partition once, through the once-only guard of the frozen run, with its hashes verified first.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .classification import FormalPredictor, expected_calibration_error_equal_mass, fit_formal_predictor, predictor_summary_v6

AUDITED_MODEL_ID = "audited_predictor"
COMPARATOR_MODEL_ID = "measured_b_comparator"
EXTENDED_STAGE = "summarize-predictor-extended"
EXTENDED_PARTITIONS = ("acdc_dev_oof", "acdc_test", "mms_external", "mms2_external")
EVALUATION_DATASETS = {"acdc_test": "ACDC", "mms_external": "MMS", "mms2_external": "MMS2"}   # MMS2: lane L1 dataset id
EVIDENCE_LABELS = {"acdc_dev_oof": "development (cross-fitted out-of-fold)", "acdc_test": "registered after unblinding",
                   "mms_external": "registered after unblinding", "mms2_external": "prospective"}
STRATUM_VARIABLES = ("vendor", "centre")
EXTENDED_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "macro_ovr_auroc", "ece", "ece_equal_width", "brier", "log_loss")
FOLD_PREDICTORS_NOT_RECORDED = "FOLD_PREDICTORS_NOT_RECORDED"
COMPARATOR_NOT_RECORDED = "COMPARATOR_NOT_RECORDED_IN_RUN"
LOA_Z = 1.96
C4_KEYS = ("sample_id", "cluster_id", "partition", "model_id", "seed", "encoder_id", "is_oof", "y_true", "y_pred")


# ----------------------------------------------------------------------------- calibration companions (A2)


def ece_equal_width(t: np.ndarray, p: np.ndarray, n_bins: int = 10) -> tuple[float, pd.DataFrame]:
    """ECE with ``n_bins`` equal-width confidence bins on [0, 1] (right-closed; the first bin includes 0): the
    count-weighted mean of |accuracy - mean confidence| over non-empty bins.  ``t`` holds class indices of ``p``."""

    p = np.asarray(p, dtype=float)
    t = np.asarray(t, dtype=int)
    if p.ndim != 2 or t.size != p.shape[0]:
        raise ValueError("Invalid probability matrix shape.")
    conf = p.max(axis=1)
    correct = (np.argmax(p, axis=1) == t).astype(float)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    which = np.searchsorted(edges[1:-1], conf, side="left")
    rows, ece, n = [], 0.0, max(int(t.size), 1)
    for b in range(int(n_bins)):
        idx = which == b
        k = int(idx.sum())
        if not k:
            rows.append({"bin_id": b, "lower": float(edges[b]), "upper": float(edges[b + 1]), "count": 0, "mean_confidence": None,
                         "accuracy": None, "abs_gap": None})
            continue
        gap = abs(float(conf[idx].mean()) - float(correct[idx].mean()))
        ece += k / n * gap
        rows.append({"bin_id": b, "lower": float(edges[b]), "upper": float(edges[b + 1]), "count": k, "mean_confidence": float(conf[idx].mean()),
                     "accuracy": float(correct[idx].mean()), "abs_gap": gap})
    return float(ece), pd.DataFrame(rows)


def brier_multiclass(t: np.ndarray, p: np.ndarray) -> float:
    """Multiclass Brier score: mean over rows of sum_k (p_k - 1[y = k])^2 (range 0 to 2)."""

    p = np.asarray(p, dtype=float)
    onehot = np.zeros_like(p)
    onehot[np.arange(p.shape[0]), np.asarray(t, dtype=int)] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1))) if p.shape[0] else float("nan")


def _evaluable(truth: Any, decisions: Any, proba: pd.DataFrame | None, classes: list[str]) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """(class index, probabilities, row mask) of the evaluable rows exactly as :func:`predictor_summary_v6` selects them."""

    t_s = pd.Series(np.asarray(truth, dtype=object)).astype(object)
    d_s = pd.Series(np.asarray(decisions, dtype=object)).astype(object)
    known = t_s.notna().to_numpy() & t_s.map(lambda v: v is not None and str(v) not in ("", "nan", "None")).to_numpy()
    decided = d_s.notna().to_numpy() & d_s.map(lambda v: v is not None and str(v) not in ("", "nan", "None")).to_numpy()
    keep = known & decided & t_s.astype(str).isin(classes).to_numpy()
    t = np.array([classes.index(str(v)) for v in t_s[keep]], dtype=int)
    p = None if proba is None else proba[classes].to_numpy(dtype=float)[keep]
    return t, p, keep


def calibration_bootstrap(t: np.ndarray, p: np.ndarray, *, clusters: np.ndarray, n_boot: int, seed: int, n_bins: int) -> np.ndarray:
    """``(n_boot, 2)`` replicates of (equal-width ECE, Brier) with the draws of
    :func:`tmmri.classification.predictor_cluster_bootstrap` (replicate r uses ``SeedSequence(seed).spawn(n_boot)[r]``;
    clusters numbered by first row), so the intervals pair with every other predictor metric."""

    from .metrics import _cluster_draw_weights, _resampling_clusters

    unit_cluster, n_resample = _resampling_clusters(np.arange(t.size), int(t.size), clusters)
    out = np.full((int(n_boot), 2), np.nan)
    for r, ss in enumerate(np.random.SeedSequence(int(seed)).spawn(int(n_boot))):
        weights = _cluster_draw_weights(np.random.default_rng(ss), unit_cluster, n_resample).astype(int)
        rows = np.repeat(np.arange(t.size), weights)
        if rows.size:
            out[r] = (ece_equal_width(t[rows], p[rows], n_bins)[0], brier_multiclass(t[rows], p[rows]))
    return out


def _interval_entry(point: float, col: np.ndarray, *, level: float, min_valid_fraction: float) -> dict[str, Any]:
    valid = col[np.isfinite(col)]
    frac = float(valid.size / col.size) if col.size else 0.0
    alpha = (1.0 - level) / 2.0
    entry: dict[str, Any] = {"estimate": float(point) if np.isfinite(point) else None, "n_boot_valid": int(valid.size), "valid_fraction": frac}
    if entry["estimate"] is None:
        entry.update({"ci_lower": None, "ci_upper": None, "status": "NE", "reason": "metric_undefined_on_sample"})
    elif frac < float(min_valid_fraction):
        entry.update({"ci_lower": None, "ci_upper": None, "status": "NE", "reason": "fewer_than_min_valid_bootstrap_replicates"})
    else:
        entry.update({"ci_lower": float(np.quantile(valid, alpha)), "ci_upper": float(np.quantile(valid, 1.0 - alpha)), "status": "EVALUATED",
                      "reason": None})
    return entry


def _below(entry: dict[str, Any], status: Mapping[str, str]) -> dict[str, Any]:
    out = dict(entry)
    if out.get("estimate") is not None or "ci_lower" in out:
        out.update({"ci_lower": None, "ci_upper": None})
    if out.get("estimate") is not None:
        out.update({"status": str(status["status"]), "reason": str(status["reason"])})
    return out


def extended_predictor_summary(truth: Any, decisions: Any, proba: pd.DataFrame | None, classes: list[str], *, clusters: Any, n_boot: int,
                               seed: int, n_bins: int, min_valid_fraction: float, level: float = 0.95, n_jobs: int = 1,
                               probability_reason: str | None = None, min_clusters: int | None = None,
                               below_status: Mapping[str, str] | None = None) -> dict[str, Any]:
    """:func:`predictor_summary_v6` (unchanged metrics, intervals and counts) plus ``ece_equal_width`` (``n_bins``
    equal-width bins) and ``brier`` with the same cluster draws.  With ``min_clusters``, a sample with fewer evaluable
    clusters keeps its point estimates and gets ``NE`` intervals (``below_status``, default STRATUM_BELOW_MIN_CASES)."""

    classes = [str(c) for c in classes]
    status = dict(below_status or {"status": "NE", "reason": "STRATUM_BELOW_MIN_CASES"})
    kw = dict(clusters=clusters, seed=seed, n_bins=n_bins, min_valid_fraction=min_valid_fraction, level=level, probability_reason=probability_reason)
    probe = predictor_summary_v6(truth, decisions, proba, classes, n_boot=0, n_jobs=1, **kw)
    below = min_clusters is not None and int(probe["counts"]["n_clusters"]) < int(min_clusters)
    out = probe if below or probe["status"] != "EVALUATED" else predictor_summary_v6(truth, decisions, proba, classes, n_boot=n_boot, n_jobs=n_jobs, **kw)
    out = dict(out)
    out["metrics"] = dict(out.get("metrics") or {})
    out.update({"ece_equal_width_bins": int(n_bins), "ece_equal_width_binning": "equal_width", "brier": "multiclass sum over the classes"})
    if out.get("status") != "EVALUATED":
        for m in ("ece_equal_width", "brier"):
            out["metrics"][m] = {"estimate": None, "status": "NE", "reason": out.get("reason") or "no_evaluable_rows"}
    elif proba is None:
        for m in ("ece_equal_width", "brier"):
            out["metrics"][m] = {"estimate": None, "ci_lower": None, "ci_upper": None, "status": "NE", "reason": probability_reason or "PROBABILITIES_NOT_RECORDED"}
    else:
        t, p, keep = _evaluable(truth, decisions, proba, classes)
        ece, bins = ece_equal_width(t, p, n_bins)
        brier = brier_multiclass(t, p)
        cl = pd.Series(np.asarray(clusters, dtype=object)).astype(str).to_numpy()[keep]
        boot = np.empty((0, 2)) if below else calibration_bootstrap(t, p, clusters=cl, n_boot=n_boot, seed=seed, n_bins=n_bins)
        out["metrics"]["ece_equal_width"] = _interval_entry(ece, boot[:, 0], level=level, min_valid_fraction=min_valid_fraction)
        out["metrics"]["brier"] = _interval_entry(brier, boot[:, 1], level=level, min_valid_fraction=min_valid_fraction)
        out["ece_equal_width_realized_bins"] = bins.to_dict("records")
    if below:
        out["metrics"] = {m: _below(e, status) for m, e in out["metrics"].items()}
        out.update({"descriptive_only": True, "min_clusters": int(min_clusters), "below_threshold": status})
    elif min_clusters is not None:
        out.update({"descriptive_only": False, "min_clusters": int(min_clusters)})
    return out


# ----------------------------------------------------------------------------- Bland-Altman (A5)


def bland_altman(observed: pd.DataFrame, predicted: pd.DataFrame, targets: list[str], clusters: Any, *, n_boot: int, seed: int,
                 level: float = 0.95, min_valid_fraction: float = 0.95, min_clusters: int | None = None,
                 below_status: Mapping[str, str] | None = None, z: float = LOA_Z) -> dict[str, dict[str, Any]]:
    """Bland-Altman agreement per target of row-aligned ``predicted`` (B-hat) and ``observed`` (measured B): bias = mean
    of B-hat - B and limits of agreement bias +- ``z`` SD (ddof 1; the definition of :func:`tmmri.metrics.metric_summary`),
    on the rows where both are finite.  Percentile intervals from one cluster draw per replicate shared by every target
    (``SeedSequence(seed).spawn(n_boot)``, clusters numbered by first row).  A target with fewer than ``min_clusters``
    clusters keeps its point estimates; its intervals are NE."""

    from .metrics import _cluster_draw_weights, _resampling_clusters

    status = dict(below_status or {"status": "NE", "reason": "STRATUM_BELOW_MIN_CASES"})
    obs = observed.reindex(columns=targets).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    pred = predicted.reindex(columns=targets).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if obs.shape != pred.shape:
        raise ValueError("observed and predicted must be row-aligned")
    n = obs.shape[0]
    cl = pd.Series(np.asarray(clusters, dtype=object)).astype(str).to_numpy()
    if cl.size != n:
        raise ValueError("clusters must be row-aligned with the rows")
    weights = np.zeros((0, n))
    if n and n_boot:
        unit_cluster, n_resample = _resampling_clusters(np.arange(n), n, cl)
        weights = np.vstack([_cluster_draw_weights(np.random.default_rng(ss), unit_cluster, n_resample)
                             for ss in np.random.SeedSequence(int(seed)).spawn(int(n_boot))])
    alpha = (1.0 - level) / 2.0
    out: dict[str, dict[str, Any]] = {}
    for j, target in enumerate(targets):
        d = pred[:, j] - obs[:, j]
        ok = np.isfinite(d)
        k = int(ok.sum())
        n_cl = int(len(set(cl[ok].tolist())))
        entry: dict[str, Any] = {"n": k, "n_clusters": n_cl, "difference": "B-hat minus measured B", "z": float(z)}
        if k < 2:
            entry.update({"bias": float(d[ok].mean()) if k else None, "sd_difference": None, "loa_lower": None, "loa_upper": None,
                          "ci": None, "status": "NE", "reason": "no_observed_rows" if k == 0 else "fewer_than_two_observed_rows"})
            out[target] = entry
            continue
        dd = d[ok]
        bias, sd = float(dd.mean()), float(dd.std(ddof=1))
        entry.update({"bias": bias, "sd_difference": sd, "loa_lower": bias - z * sd, "loa_upper": bias + z * sd})
        below = min_clusters is not None and n_cl < int(min_clusters)
        if below or not weights.size:
            entry.update({"ci": None, "n_boot_valid": 0, "status": status["status"] if below else "NE",
                          "reason": status["reason"] if below else "no_bootstrap_replicates", "descriptive_only": bool(below)})
            out[target] = entry
            continue
        w = weights[:, ok]
        sw = w.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            b_r = (w @ dd) / sw
            var = (w * (dd[None, :] - b_r[:, None]) ** 2).sum(axis=1) / (sw - 1.0)
        valid = (sw >= 2) & np.isfinite(var)
        sd_r = np.sqrt(np.where(valid, var, np.nan))
        reps = {"bias": b_r, "loa_lower": b_r - z * sd_r, "loa_upper": b_r + z * sd_r}
        frac = float(valid.mean())
        entry["n_boot_valid"] = int(valid.sum())
        if frac < float(min_valid_fraction):
            entry.update({"ci": None, "status": "NE", "reason": "fewer_than_min_valid_bootstrap_replicates"})
        else:
            entry.update({"ci": {name: [float(np.quantile(r[valid], alpha)), float(np.quantile(r[valid], 1.0 - alpha))] for name, r in reps.items()},
                          "status": "EVALUATED", "reason": None})
        entry["descriptive_only"] = False
        out[target] = entry
    return out


def bland_altman_rows(result: Mapping[str, Mapping[str, Any]], **keys: Any) -> list[dict[str, Any]]:
    """Flat rows (one per target) of a :func:`bland_altman` result with the given key columns."""

    rows = []
    for target, e in result.items():
        ci = e.get("ci") or {}
        row = {**keys, "target": target, "n": e.get("n"), "n_clusters": e.get("n_clusters"), "bias": e.get("bias"), "sd_difference": e.get("sd_difference"),
               "loa_lower": e.get("loa_lower"), "loa_upper": e.get("loa_upper"), "status": e.get("status"), "reason": e.get("reason")}
        for name in ("bias", "loa_lower", "loa_upper"):
            lo, hi = (ci.get(name) or [None, None])
            row[f"{name}_ci_lower"], row[f"{name}_ci_upper"] = lo, hi
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------- measured-B comparator (A4)


@dataclass
class MeasuredBPredictor:
    """The audited-predictor model on measured B: fit-partition medians impute missing values, then the frozen
    :class:`FormalPredictor` (fit-partition standardization and multinomial LR with the audited settings) applies."""

    features: list[str]
    medians: np.ndarray
    predictor: FormalPredictor

    @property
    def classes(self) -> list[str]:
        return list(self.predictor.classes)

    def impute(self, b: pd.DataFrame | np.ndarray) -> np.ndarray:
        v = b.reindex(columns=self.features).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float) if isinstance(b, pd.DataFrame) \
            else np.asarray(b, dtype=float)
        return np.where(np.isfinite(v), v, self.medians[None, :])

    def decision_function(self, b: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.predictor.decision_function(self.impute(b))

    def predict_proba(self, b: pd.DataFrame | np.ndarray, index: pd.Index | None = None) -> pd.DataFrame:
        return self.predictor.predict_proba(self.impute(b), index=index)

    def predict(self, b: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.predictor.predict(self.impute(b))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "measured_b_predictor_v6_0_1", "features": list(self.features), "medians": self.medians.tolist(),
                "predictor": self.predictor.to_dict()}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "MeasuredBPredictor":
        return cls(list(d["features"]), np.asarray(d["medians"], dtype=float), FormalPredictor.from_dict(d["predictor"]))


def fit_measured_b_predictor(b: pd.DataFrame, y: Any, cfg: Mapping[str, Any], features: list[str], *,
                             fit_mask: np.ndarray | None = None) -> MeasuredBPredictor:
    """Fit-partition median imputation of ``features`` (medians of the fit rows only; an all-missing column imputes 0
    and is then dropped as zero-variance) followed by :func:`fit_formal_predictor` with the audited settings ``cfg``."""

    v = b.reindex(columns=list(features)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    labels = pd.Series(np.asarray(y, dtype=object)).astype("string")
    rows = np.array(labels.notna().to_numpy(), dtype=bool)
    if fit_mask is not None:
        rows &= np.asarray(fit_mask, dtype=bool)
    if not rows.any():
        raise ValueError("the measured-B comparator has no fit rows")
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            medians = np.nanmedian(v[rows], axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    imputed = np.where(np.isfinite(v), v, medians[None, :])
    return MeasuredBPredictor(list(features), medians, fit_formal_predictor(imputed, labels, dict(cfg), fit_mask=rows))


def comparator_settings(study: Any) -> dict[str, Any]:
    """The comparator definition of ``configs/formal.yaml`` with the resolved registry primary family."""

    ccfg = dict(study.formal.get("comparator_predictor") or {})
    settings = str(ccfg.get("settings", "same_as_audited_predictor"))
    if settings != "same_as_audited_predictor":
        raise ValueError(f"configs/formal.yaml comparator_predictor.settings must be same_as_audited_predictor, not {settings!r}")
    return {"name": str(ccfg.get("name", "measured_b_logistic_semantic_comparator")), "role": str(ccfg.get("role", "semantic_comparator_not_audited")),
            "settings": settings, "features": list(study.primary_family()), "family": str(study.transition["targets"]["primary_family"]),
            "imputation": "fit-partition median (training folds for cross-fitting)", "standardization": "fit-partition mean and SD (ddof 0)"}


# ----------------------------------------------------------------------------- prediction schema (contract C4)


def _expand(columns: list[str], values: np.ndarray, classes: list[str], fill: float) -> np.ndarray:
    out = np.full((values.shape[0], len(classes)), fill, dtype=float)
    for j, c in enumerate(columns):
        if c in classes:
            out[:, classes.index(c)] = values[:, j]
    return out


def c4_frame(ids: list[str], clusters: Any, partition: str | list[str], model_id: str, encoder_id: str | list[str], is_oof: bool | list[bool],
             y_true: Any, y_pred: Any, proba: np.ndarray, logits: np.ndarray | None, classes: list[str]) -> pd.DataFrame:
    """Rows of ``<model_dir>/predictions.parquet`` (contract C4): keys, ``prob_<CLASS>`` and ``logit_<CLASS>`` (nullable)
    for ``classes`` in registry order.  ``proba``/``logits`` are already in ``classes`` order."""

    n = len(ids)

    def col(value: Any) -> list:
        return list(value) if isinstance(value, (list, tuple, np.ndarray, pd.Series)) else [value] * n

    truth = pd.array([None if v is None or (isinstance(v, float) and np.isnan(v)) or str(v) in ("", "nan", "None", "<NA>") else str(v)
                      for v in col(y_true)], dtype="string")
    frame = pd.DataFrame({"sample_id": [str(s) for s in ids], "cluster_id": [str(c) for c in col(clusters)], "partition": [str(p) for p in col(partition)],
                          "model_id": model_id, "seed": np.full(n, -1, dtype=np.int64), "encoder_id": [str(e) for e in col(encoder_id)],
                          "is_oof": np.asarray(col(is_oof), dtype=bool), "y_true": truth, "y_pred": [str(v) for v in col(y_pred)]})
    p = np.asarray(proba, dtype=float).reshape(n, len(classes))
    lg = np.full((n, len(classes)), np.nan) if logits is None else np.asarray(logits, dtype=float).reshape(n, len(classes))
    for j, c in enumerate(classes):
        frame[f"prob_{c}"] = p[:, j]
    for j, c in enumerate(classes):
        frame[f"logit_{c}"] = lg[:, j]
    return frame


def model_outputs(model: Any, x: np.ndarray | pd.DataFrame, classes: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(probabilities, logits, decisions) of a frozen predictor on rows it decides, expanded to ``classes`` (a class the
    model never saw has probability 0 and a null logit)."""

    own = list(model.classes)
    proba = model.predict_proba(x).to_numpy(dtype=float)
    logits = np.asarray(model.decision_function(x), dtype=float)
    return _expand(own, proba, classes, 0.0), _expand(own, logits, classes, np.nan), np.asarray(model.predict(x), dtype=object)


# ----------------------------------------------------------------------------- fit-predictor storage (A2, A4)


def audited_dir(run: Any) -> Path:
    return run.path / "formal" / AUDITED_MODEL_ID


def comparator_dir(run: Any) -> Path:
    return run.path / "formal" / COMPARATOR_MODEL_ID


def n_outer_folds(study: Any) -> int:
    return int(study.splits["acdc"]["acdc_dev"]["outer_folds"]["n_folds"])


def fit_predictor_extension_paths(run: Any, study: Any) -> list[Path]:
    """Declared extra outputs of ``fit-predictor`` (A2 fold predictors and OOF predictions; A4 comparator bundle)."""

    k = range(1, n_outer_folds(study) + 1)
    a, c = audited_dir(run), comparator_dir(run)
    return ([a / "predictions.parquet", a / "fold_models.json", *[a / "fold_models" / f"fold{f}.joblib" for f in k]]
            + [c / "final_model.joblib", c / "predictions.parquet", *[c / "fold_models" / f"fold{f}.joblib" for f in k], c / "bundle.json"])


def fit_predictor_extension_params(study: Any) -> dict[str, Any]:
    """Stage parameters of the extended ``fit-predictor`` (a changed comparator definition re-runs it before the freeze)."""

    return {"v6_0_1_extensions": {"fold_predictors": True, "measured_b_comparator": comparator_settings(study)}}


def store_fit_predictor_extensions(run: Any, study: Any, *, ids: list[str], x: np.ndarray, y: pd.Series, folds: np.ndarray, critical: np.ndarray,
                                   fold_predictors: Mapping[int, FormalPredictor], b_dev: pd.DataFrame, clusters: Any) -> dict[str, Any]:
    """Write the A2/A4 outputs of ``fit-predictor`` (see the module docstring).  ``ids``, ``x`` (formal A), ``y``,
    ``folds`` and ``critical`` (critical QC) are the stage's row-aligned ``acdc_dev`` arrays, ``b_dev`` the measured B
    rows of ``ids`` and ``fold_predictors`` the outer-fold predictors the stage fitted for the cross-fitted decisions.
    Prediction rows: the audited predictor's analysable rows (finite A, no critical QC) and, for the comparator, the
    rows with measured B and no critical QC; critical-QC rows are excluded from every fit (D21)."""

    import joblib

    from .runs import atomic_write_json, hash_path

    pcfg = study.formal["audited_predictor"]
    classes = [str(c) for c in pcfg["classes"]]
    adir, cdir = audited_dir(run), comparator_dir(run)
    (adir / "fold_models").mkdir(parents=True, exist_ok=True)
    (cdir / "fold_models").mkdir(parents=True, exist_ok=True)
    cl = np.asarray([str(c) for c in clusters], dtype=object)
    truth = y.astype(object).to_numpy()
    critical = np.asarray(critical, dtype=bool)
    folds = np.asarray(folds, dtype=int)
    rows = np.flatnonzero(np.isfinite(x).all(axis=1) & ~critical)
    fold_ids = sorted(int(f) for f in fold_predictors)
    # A2: out-of-fold probabilities of the stored outer-fold predictors (each row by the predictor that never saw it).
    parts, manifest = [], {}
    for f in fold_ids:
        model = fold_predictors[f]
        path = adir / "fold_models" / f"fold{f}.joblib"
        joblib.dump(model, path)
        held = rows[folds[rows] == f]
        manifest[str(f)] = {"file": f"fold_models/fold{f}.joblib", "sha256": hash_path(path), "classes": list(model.classes),
                            "missing_classes": list(model.missing_classes), "n_fit_rows": int(model.n_fit_rows),
                            "n_heldout_predictions": int(held.size)}
        if held.size:
            p, lg, pred = model_outputs(model, x[held], classes)
            parts.append(c4_frame([ids[i] for i in held], cl[held], "acdc_dev", AUDITED_MODEL_ID, f"fold{f}", True, truth[held], pred, p, lg, classes))
    frame = pd.concat(parts, ignore_index=True) if parts else c4_frame([], [], "acdc_dev", AUDITED_MODEL_ID, "", True, [], [], np.empty((0, len(classes))), None, classes)
    frame = frame.sort_values(["sample_id", "encoder_id"], kind="mergesort").reset_index(drop=True)
    _write(frame, adir / "predictions.parquet")
    atomic_write_json(adir / "fold_models.json", {
        "schema": "tmmri.fold_predictors/1", "model_id": AUDITED_MODEL_ID, "predictor": str(pcfg["name"]), "partition": "acdc_dev",
        "folds": manifest, "final_predictor": "formal/d_model_predictor.joblib (frozen bundle)",
        "fit_rule": "outer-fold training rows of acdc_dev (finite formal A, no critical QC); the held-out fold is never read",
        "predictions": "predictions.parquet: one out-of-fold row per analysable acdc_dev case (finite A, no critical QC); contract C4",
        "probabilities": "softmax of the stored coefficients; a class absent from a fold's fit rows has probability 0 and a null logit"})
    # A4: the measured-B comparator, cross-fitted over the same outer folds and refitted on every fit row.
    settings = comparator_settings(study)
    features = settings["features"]
    present = b_dev.reindex(columns=features).apply(pd.to_numeric, errors="coerce").notna().any(axis=1).to_numpy()
    comp_fit = present & ~critical
    comp_rows = np.flatnonzero(comp_fit)
    final = fit_measured_b_predictor(b_dev, y, pcfg, features, fit_mask=comp_fit)
    joblib.dump(final, cdir / "final_model.joblib")
    cparts = []
    files = {"final_model.joblib": hash_path(cdir / "final_model.joblib")}
    for f in fold_ids:
        held = folds == f
        model = fit_measured_b_predictor(b_dev, y, pcfg, features, fit_mask=comp_fit & ~held)
        path = cdir / "fold_models" / f"fold{f}.joblib"
        joblib.dump(model, path)
        files[f"fold_models/fold{f}.joblib"] = hash_path(path)
        idx = comp_rows[folds[comp_rows] == f]
        if idx.size:
            p, lg, pred = model_outputs(model, b_dev.iloc[idx], classes)
            cparts.append(c4_frame([ids[i] for i in idx], cl[idx], "acdc_dev", COMPARATOR_MODEL_ID, f"fold{f}", True, truth[idx], pred, p, lg, classes))
    if comp_rows.size:
        p, lg, pred = model_outputs(final, b_dev.iloc[comp_rows], classes)
        cparts.append(c4_frame([ids[i] for i in comp_rows], cl[comp_rows], "acdc_dev", COMPARATOR_MODEL_ID, "final", False, truth[comp_rows], pred, p, lg, classes))
    cframe = pd.concat(cparts, ignore_index=True).sort_values(["sample_id", "encoder_id"], kind="mergesort").reset_index(drop=True)
    _write(cframe, cdir / "predictions.parquet")
    files["predictions.parquet"] = hash_path(cdir / "predictions.parquet")
    atomic_write_json(cdir / "bundle.json", {
        "schema": "tmmri.supplementary_bundle/1", "model_id": COMPARATOR_MODEL_ID, **settings, "fitted_on": "acdc_dev",
        "n_fit_rows": int(comp_fit.sum()), "classes": final.classes, "missing_classes": list(final.predictor.missing_classes),
        "files": files, "fit_rule": "acdc_dev rows with measured B and no critical QC; cross-fitting refits imputation, scaler and model "
        "on the training folds only",
        "freeze": "written by fit-predictor (a fitting stage: refused after .stages/FROZEN.json); never part of the analysis-bundle hash",
        "evaluation": f"{EXTENDED_STAGE}:<partition> (once-only records of the frozen run; hashes verified first)"})
    return {"fold_predictors": len(fold_ids), "oof_rows": int(len(frame)), "comparator_fit_rows": int(comp_fit.sum())}


def _write(frame: pd.DataFrame, path: Path, csv: bool = False) -> Path:
    from .stages import _write_table

    return _write_table(frame, path, csv=csv)


# ----------------------------------------------------------------------------- extended summaries (read-only stage)


def evaluated(run: Any, stage: str) -> bool:
    record = run.read_record(stage)
    return record is not None and record.status == "EVALUATED"


def available_partitions(run: Any) -> list[str]:
    """``acdc_dev_oof`` plus every evaluation partition whose ``evaluate`` record exists with status EVALUATED."""

    return ["acdc_dev_oof"] + [p for p in EXTENDED_PARTITIONS[1:] if evaluated(run, f"evaluate:{p}")]


def require_partition_unlocked(ctx: Any, partition: str, requester: str) -> None:
    """M&Ms locks: ``mms_external`` and ``mms2_external`` obey :func:`tmmri.stages.require_mms_unlocked`; ``mms2_external``
    also obeys the M&Ms-2 lock of lane L1 when it is present (``tmmri.stages.require_mms2_unlocked``)."""

    from . import stages

    if partition in ("mms_external", "mms2_external"):
        stages.require_mms_unlocked(ctx, f"{requester} ({partition})")
    if partition == "mms2_external":
        hook = getattr(stages, "require_mms2_unlocked", None)
        if callable(hook):
            hook(ctx, f"{requester} ({partition})")


def extended_paths(run: Any, partition: str, *, comparator: bool) -> list[Path]:
    mdir = run.path / "metrics" / "predictor_extended"
    out = [mdir / f"{partition}.json", mdir / f"bland_altman_{partition}.parquet"]
    if partition != "acdc_dev_oof":
        out.append(run.path / "inference" / AUDITED_MODEL_ID / partition / "predictions.parquet")
        if comparator:
            out.append(run.path / "inference" / COMPARATOR_MODEL_ID / partition / "predictions.parquet")
    return out


def summarize_predictor_extended(ctx: Any, partitions: Iterable[str] | None = None) -> list[Any]:
    """The v6.0.1 extended predictor summaries, one stage record per partition (default: every available partition;
    see the module docstring).  Requires ``.stages/FROZEN.json``; an evaluation partition requires its ``evaluate``
    record.  Returns the :class:`tmmri.runs.StageResult` of each partition."""

    from . import stages

    run = ctx.require_run(EXTENDED_STAGE)
    if not run.frozen_path.is_file():
        raise stages.BundleError(f"{EXTENDED_STAGE}: run {run.run_id} has no frozen analysis bundle (run freeze-analysis-bundle first)")
    available = available_partitions(run)
    chosen = available if partitions is None else [str(p) for p in partitions]
    unknown = [p for p in chosen if p not in EXTENDED_PARTITIONS]
    if unknown:
        raise ValueError(f"{EXTENDED_STAGE}: unknown partitions {unknown}; expected {EXTENDED_PARTITIONS}")
    missing = [p for p in chosen if p not in available]
    if missing:
        raise stages.BundleError(f"{EXTENDED_STAGE}: {missing} not evaluated in run {run.run_id} (evaluate --split <partition> first)")
    return [_summarize_partition(ctx, p) for p in chosen]


def _summarize_partition(ctx: Any, partition: str) -> Any:
    import joblib

    from . import stages
    from .formal_features import formal_descriptor_names
    from .runs import atomic_write_json, hash_path

    require_partition_unlocked(ctx, partition, EXTENDED_STAGE)
    frozen = stages.verify_frozen_bundle(ctx)
    run = ctx.require_run(EXTENDED_STAGE)
    p = run.path
    dev = partition == "acdc_dev_oof"
    ds = "ACDC" if dev else EVALUATION_DATASETS[partition]
    fit_record = run.read_record("fit-predictor")
    eval_record = None if dev else stages._require_evaluated(run, f"evaluate:{partition}", EXTENDED_STAGE)
    a_pred, c_pred = audited_dir(run) / "predictions.parquet", comparator_dir(run) / "predictions.parquet"
    c_model, c_bundle = comparator_dir(run) / "final_model.joblib", comparator_dir(run) / "bundle.json"
    has_folds = a_pred.is_file()
    has_comparator = all(f.is_file() for f in ((c_pred, c_bundle) if dev else (c_model, c_bundle)))
    contracts = p / "contracts"
    inputs = [run.frozen_path, contracts / "split_manifest.parquet", contracts / f"cohort_{ds}.parquet", p / "semantics" / f"B_{ds}.parquet"]
    if dev:
        inputs += [f for f in (p / "formal" / "d_model_crossfit_ACDC_dev.parquet", p / "oof" / "bhat_oof_ACDC_dev.parquet") if f.is_file()]
        inputs += [a_pred] if has_folds else []
        inputs += [c_pred, c_bundle] if has_comparator else []
    else:
        inputs += [run.record_path(f"evaluate:{partition}"), p / "formal" / "d_model_predictor.joblib", p / "formal" / f"A_{ds}.parquet",
                   p / "inference" / f"d_model_{partition}.parquet"]
        inputs += [f for f in (p / "reconstructed" / f"bhat_{partition}.parquet",) if f.is_file()]
        inputs += [c_model, c_bundle] if has_comparator else []
    if fit_record is not None and (has_folds or has_comparator):
        inputs.append(run.record_path("fit-predictor"))
    outputs = extended_paths(run, partition, comparator=has_comparator)
    for f in ([a_pred] if has_folds else []) + ([c_pred, c_model, c_bundle] if has_comparator else []):
        stages._check_recorded_output(run, fit_record, f, EXTENDED_STAGE)
    if eval_record is not None:
        for f in (p / "inference" / f"d_model_{partition}.parquet", p / "reconstructed" / f"bhat_{partition}.parquet"):
            if f.is_file():
                stages._check_recorded_output(run, eval_record, f, EXTENDED_STAGE)
    if has_comparator:
        listed = json.loads(c_bundle.read_text(encoding="utf-8")).get("files", {})
        changed = [rel for rel, digest in listed.items() if hash_path(comparator_dir(run) / rel) != digest]
        if changed:
            raise stages.BundleError(f"{EXTENDED_STAGE}: the measured-B comparator changed after fit-predictor recorded it: {changed[:3]}")

    def compute() -> dict[str, Any]:
        study = ctx.study
        boot, cal = study.bootstrap["fixed_model_bootstrap"], study.bootstrap["calibration"]
        min_cases = int(study.bootstrap["subgroup_reporting_thresholds"]["min_cases_per_stratum"])
        below_status = stages.below_threshold_status(study)
        kw = dict(n_boot=int(boot["replicates"]), seed=int(boot["seed"]), n_bins=int(cal["ece_bins"]),
                  min_valid_fraction=float(boot["min_valid_fraction"]), n_jobs=ctx.jobs)
        configured = [str(c) for c in study.formal["audited_predictor"]["classes"]]
        split = stages._split_manifest(ctx) if ds != "MMS2" else pd.concat([stages._split_manifest(ctx), stages.split_manifest_for(ctx, dataset="MMS2")],
                                                                          ignore_index=True)   # dev rows plus the M&Ms-2 rows
        cohort = stages._cohort(ctx, ds).set_index("sample_id")
        b = pd.read_parquet(p / "semantics" / f"B_{ds}.parquet").set_index("sample_id")
        critical = set(b.index[stages.critical_qc_mask(b, study).to_numpy(dtype=bool)])
        targets = [t for t in study.primary_family()]
        reference = stages.reference_labels(cohort, "ACDC" if ds == "ACDC" else "MMS").astype(object)
        payload: dict[str, Any] = {
            "schema": "tmmri.predictor_extended/1", "partition": partition, "dataset": ds, "evidence": EVIDENCE_LABELS[partition],
            "bundle_sha256": frozen["bundle_sha256"], "classes": configured, "read_only": True,
            "metrics_definition": "predictor_summary_v6 (SPECIFICATION.md 13.2; equal-mass ECE as frozen) plus ECE with equal-width bins and the "
                                  "multiclass Brier score (docs/MANUSCRIPT_PROTOCOL_2026-10-02.md Section F), one cluster draw per replicate",
            "bootstrap": {"replicates": kw["n_boot"], "seed": kw["seed"], "unit": "cluster_id", "interval": "percentile_95"},
            "ece_bins": kw["n_bins"], "min_clusters_per_stratum": min_cases, "below_threshold": below_status,
            "qc_rule": "critical-QC rows (QC_ABSTAIN) are excluded and counted"}
        models: dict[str, Any] = {}
        strata: dict[str, Any] = {}
        predictions: dict[str, pd.DataFrame] = {}
        ba_rows: list[dict[str, Any]] = []

        def summarize(truth: pd.Series, decisions: pd.Series, proba: pd.DataFrame | None, classes: list[str], ids: list[str], *,
                      stratum: bool = False, reason: str | None = None) -> dict[str, Any]:
            return extended_predictor_summary(truth.reindex(ids).to_numpy(dtype=object), decisions.reindex(ids).to_numpy(dtype=object),
                                              None if proba is None else proba.loc[ids].reset_index(drop=True), classes,
                                              clusters=stages._cluster_ids(split, ids).to_numpy(dtype=object), probability_reason=reason,
                                              min_clusters=min_cases if stratum else None, below_status=below_status, **kw)

        if dev:
            members = set(split.loc[split["partition"] == "acdc_dev", "sample_id"].astype(str))
            truth = cohort["acdc_class"].astype(object)
            crossfit = p / "formal" / "d_model_crossfit_ACDC_dev.parquet"
            recorded = pd.read_parquet(crossfit).set_index("sample_id")["d_model_crossfit"] if crossfit.is_file() else None
            for model_id, path, present, reason in ((AUDITED_MODEL_ID, a_pred, has_folds, FOLD_PREDICTORS_NOT_RECORDED),
                                                    (COMPARATOR_MODEL_ID, c_pred, has_comparator, COMPARATOR_NOT_RECORDED)):
                if not present:
                    models[model_id] = {"status": "NE", "reason": reason, "note": "fit-predictor of this run did not store it (v6.0.0 run)"}
                    continue
                frame = pd.read_parquet(path)
                oof = frame[frame["is_oof"].astype(bool)].set_index("sample_id")
                ids = sorted(s for s in oof.index.astype(str) if s in members and s not in critical)
                if model_id == AUDITED_MODEL_ID and recorded is not None:
                    differ = [s for s in ids if str(recorded.get(s)) != str(oof.loc[s, "y_pred"])]
                    if differ:
                        raise ValueError(f"{EXTENDED_STAGE}: stored out-of-fold predictions disagree with the recorded d_model_crossfit ({len(differ)} rows)")
                proba = oof[[f"prob_{c}" for c in configured]].set_axis(configured, axis=1)
                result = summarize(truth, oof["y_pred"].astype(object), proba, configured, ids)
                result.update({"model_id": model_id, "predictions": run.portable(path), "probability_source": "stored outer-fold predictors (is_oof rows)",
                               "counts_partition": {"n_partition": len(members), "n_stored_oof": int(len(oof)), "n_excluded_critical_qc":
                                                    int(sum(1 for s in oof.index.astype(str) if s in critical))}})
                models[model_id] = result
            bhat_path = p / "oof" / "bhat_oof_ACDC_dev.parquet"
            if bhat_path.is_file():
                bhat = pd.read_parquet(bhat_path).set_index("sample_id")
                ids = [s for s in sorted(bhat.index.astype(str)) if s in members and s in b.index]
                ba = bland_altman(b.loc[ids], bhat.loc[ids], targets, stages._cluster_ids(split, ids).to_numpy(dtype=object), n_boot=kw["n_boot"],
                                  seed=kw["seed"], min_valid_fraction=kw["min_valid_fraction"])
                payload["bland_altman"] = {"source": run.portable(bhat_path), "substrate": "nested out-of-fold B-hat of fit-transition", "targets": ba}
                ba_rows += bland_altman_rows(ba, partition=partition, stratum_variable="all", stratum="all")
            else:
                payload["bland_altman"] = {"status": "NE", "reason": "OOF_BHAT_NOT_RECORDED"}
            payload["strata"] = {"note": "acdc_dev has no vendor or centre strata (single release)"}
        else:
            members = sorted(split.loc[split["partition"] == partition, "sample_id"].astype(str))
            a = pd.read_parquet(p / "formal" / f"A_{ds}.parquet").set_index("sample_id")
            d_model = pd.read_parquet(p / "inference" / f"d_model_{partition}.parquet").set_index("sample_id")["d_model"].astype(object)
            ids = [s for s in members if s in d_model.index]
            kept = [s for s in ids if s not in critical]
            if ds == "ACDC":
                metric_ids = kept
            else:
                compat = cohort.reindex(kept)["crosswalk_compatibility"].astype(str)
                metric_ids = [s for s in kept if compat[s] == stages.SAME_DEFINITION]
            names = formal_descriptor_names()
            predictor = joblib.load(p / "formal" / "d_model_predictor.joblib")
            x = a.reindex(kept)[names].to_numpy(dtype=np.float64)
            decided = np.isfinite(x).all(axis=1)
            # probabilities on exactly the metric rows (as summarize-predictor computes them: identical floating point)
            proba_a = stages._predictor_probabilities(predictor, a.reindex(metric_ids)[names].to_numpy(dtype=np.float64)).set_axis(pd.Index(metric_ids))
            result = summarize(reference, d_model, proba_a, list(predictor.classes), metric_ids)
            result.update({"model_id": AUDITED_MODEL_ID, "probability_source": "formal/d_model_predictor.joblib (frozen bundle, hash-verified) on the "
                           "recorded formal A; its argmax reproduces the recorded d_model", "counts_partition": {
                               "n_partition": len(members), "n_recorded_predictions": len(ids), "n_excluded_critical_qc": len(ids) - len(kept),
                               "n_metric_rows": len(metric_ids)}})
            models[AUDITED_MODEL_ID] = result
            rows = [kept[i] for i in np.flatnonzero(decided)]
            pa, la, pred_a = model_outputs(predictor, x[decided], configured)
            if any(str(u) != str(v) for u, v in zip(pred_a, d_model.reindex(rows).to_numpy(dtype=object))):
                raise ValueError(f"{EXTENDED_STAGE}: the frozen predictor does not reproduce the recorded d_model on {partition}")
            clusters_rows = stages._cluster_ids(split, rows).to_numpy(dtype=object)
            predictions[AUDITED_MODEL_ID] = c4_frame(rows, clusters_rows, partition, AUDITED_MODEL_ID, "final", False,
                                                     reference.reindex(rows).to_numpy(dtype=object), pred_a, pa, la, configured)
            decisions = {AUDITED_MODEL_ID: d_model, COMPARATOR_MODEL_ID: None}
            probas = {AUDITED_MODEL_ID: (proba_a, list(predictor.classes))}
            if has_comparator:
                comparator = joblib.load(c_model)
                b_rows = [s for s in kept if s in b.index]
                pc, lc, pred_c = model_outputs(comparator, b.loc[b_rows], configured)
                pc_own = comparator.predict_proba(b.loc[b_rows]).set_axis(pd.Index(b_rows))
                decisions[COMPARATOR_MODEL_ID] = pd.Series(pred_c, index=b_rows, dtype=object)
                probas[COMPARATOR_MODEL_ID] = (pc_own, list(comparator.classes))
                c_metric = [s for s in metric_ids if s in b.index]
                comp = summarize(reference, decisions[COMPARATOR_MODEL_ID], pc_own, list(comparator.classes), c_metric)
                comp.update({"model_id": COMPARATOR_MODEL_ID, "probability_source": "formal/measured_b_comparator/final_model.joblib (fit-predictor "
                             "record and bundle.json hashes verified) on the recorded measured B", "counts_partition": {
                                 "n_partition": len(members), "n_with_measured_b": len(b_rows), "n_metric_rows": len(c_metric)}})
                models[COMPARATOR_MODEL_ID] = comp
                predictions[COMPARATOR_MODEL_ID] = c4_frame(b_rows, stages._cluster_ids(split, b_rows).to_numpy(dtype=object), partition,
                                                            COMPARATOR_MODEL_ID, "final", False, reference.reindex(b_rows).to_numpy(dtype=object),
                                                            pred_c, pc, lc, configured)
            else:
                models[COMPARATOR_MODEL_ID] = {"status": "NE", "reason": COMPARATOR_NOT_RECORDED,
                                               "note": "fit-predictor of this run did not store the comparator (v6.0.0 run)"}
            bhat_path = p / "reconstructed" / f"bhat_{partition}.parquet"
            bhat = pd.read_parquet(bhat_path).set_index("sample_id") if bhat_path.is_file() else None
            ba_ids = [] if bhat is None else [s for s in sorted(bhat.index.astype(str)) if s in b.index]
            if bhat is not None:
                ba = bland_altman(b.loc[ba_ids], bhat.loc[ba_ids], targets, stages._cluster_ids(split, ba_ids).to_numpy(dtype=object), n_boot=kw["n_boot"],
                                  seed=kw["seed"], min_valid_fraction=kw["min_valid_fraction"])
                payload["bland_altman"] = {"source": run.portable(bhat_path), "substrate": "recorded raw B-hat of evaluate (the error substrate)",
                                           "rows": "the rows of evaluate's per-target errors", "targets": ba, "strata": {}}
                ba_rows += bland_altman_rows(ba, partition=partition, stratum_variable="all", stratum="all")
            else:
                payload["bland_altman"] = {"status": "NE", "reason": "BHAT_NOT_RECORDED"}
            if ds != "ACDC":
                for var in STRATUM_VARIABLES:
                    if var not in cohort.columns:
                        strata[var] = {"status": "NE", "reason": f"COLUMN_ABSENT ({var} not in contracts/cohort_{ds}.parquet)"}
                        continue
                    labels = cohort[var].map(stages._stratum_label)
                    strata[var] = {}
                    for value in sorted(set(labels.reindex(kept).astype(str))):
                        s_kept = [s for s in kept if labels.get(s) == value]
                        s_metric = [s for s in metric_ids if labels.get(s) == value]
                        entry: dict[str, Any] = {"n_cases": len(s_kept), "n_clusters": int(stages._cluster_ids(split, s_kept).nunique()),
                                                 "n_metric_rows": len(s_metric)}
                        for model_id, (proba, classes) in probas.items():
                            usable = [s for s in s_metric if s in proba.index]
                            entry[model_id] = summarize(reference, decisions[model_id], proba, classes, usable, stratum=True)
                        if bhat is not None:
                            s_ba = [s for s in ba_ids if labels.get(s) == value]
                            ba_s = bland_altman(b.loc[s_ba], bhat.loc[s_ba], targets, stages._cluster_ids(split, s_ba).to_numpy(dtype=object),
                                                n_boot=kw["n_boot"], seed=kw["seed"], min_valid_fraction=kw["min_valid_fraction"],
                                                min_clusters=min_cases, below_status=below_status)
                            payload["bland_altman"]["strata"].setdefault(var, {})[value] = ba_s
                            ba_rows += bland_altman_rows(ba_s, partition=partition, stratum_variable=var, stratum=value)
                        strata[var][value] = entry
            payload["strata"] = strata if strata else {"note": "acdc_test has no vendor or centre strata (single release)"}
        payload["models"] = models
        mdir = run.sub("metrics") / "predictor_extended"
        atomic_write_json(mdir / f"{partition}.json", payload, allow_nan=True)
        columns = ["partition", "stratum_variable", "stratum", "target", "n", "n_clusters", "bias", "bias_ci_lower", "bias_ci_upper", "sd_difference",
                   "loa_lower", "loa_lower_ci_lower", "loa_lower_ci_upper", "loa_upper", "loa_upper_ci_lower", "loa_upper_ci_upper", "status", "reason"]
        _write(pd.DataFrame(ba_rows, columns=columns), mdir / f"bland_altman_{partition}.parquet", csv=True)
        for model_id, frame in predictions.items():
            _write(frame, p / "inference" / model_id / partition / "predictions.parquet")
        return {"partition": partition, "models": sorted(models)}

    return stages._stage(ctx, f"{EXTENDED_STAGE}:{partition}", compute, inputs, outputs,
                         {"partition": partition, "fold_predictors_recorded": has_folds, "comparator_recorded": has_comparator}, status="EVALUATED")
