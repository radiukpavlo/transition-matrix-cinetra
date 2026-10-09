"""Shared machinery of the data-centric analyses S5 and S6 (manuscript protocol 2.1, Section E; ``configs/data_centric.yaml``).

* :func:`data_centric_config` reads ``configs/data_centric.yaml`` from the study (``StudyConfig.configs`` when the file
  is listed in ``configs/study_v6.yaml`` ``config_files``, else directly from the study's configuration tree).
* :func:`record_stage` runs a stage through :func:`tmmri.runs.run_stage` with the once-only guard of a frozen run
  (:func:`tmmri.stages.once_only_guard`); its configuration hash covers the named study sections plus the named
  ``data_centric`` section, so the stage records behave like every other v6 stage.
* :func:`paired_rho_difference` and :func:`paired_accuracy_difference` are paired cluster bootstraps of a difference
  between two conditions on the same cases: replicate r draws clusters with ``SeedSequence(seed).spawn(n)[r]`` exactly as
  :func:`tmmri.metrics.primary_rho` and :func:`tmmri.classification.predictor_cluster_bootstrap` do, and both
  conditions are evaluated on the same draw.

Research use only; synthetic fixtures are algorithm controls, never cohort results.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..config import StudyConfig, _normalized_sha256, _read_any
from ..metrics import _aligned, _cluster_codes, _cluster_draw_weights, _resampling_clusters, _rho_from_errors
from ..runs import run_stage, sha256_json

CONFIG_KEY = "data_centric"
CONFIG_PATH = "configs/data_centric.yaml"
NE_INSUFFICIENT_CLASS_SUPPORT = "NE_INSUFFICIENT_CLASS_SUPPORT"
NOT_EXECUTED = "NOT_EXECUTED"
REGISTERED_AFTER_UNBLINDING = "registered after unblinding"


class DevelopmentOnlyError(RuntimeError):
    """A development-only analysis (S5) was handed a row outside ``acdc_dev``."""


def data_centric_config(study: StudyConfig) -> dict[str, Any]:
    """``configs/data_centric.yaml`` of the study's configuration tree (loaded by the study when listed there)."""

    if CONFIG_KEY in study.configs:
        return study.configs[CONFIG_KEY]
    path = Path(study.root) / CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing (S5/S6 configuration)")
    return _read_any(path)


def data_centric_file_sha256(study: StudyConfig) -> str | None:
    rel = CONFIG_PATH
    if rel in study.file_hashes:
        return study.file_hashes[rel]
    path = Path(study.root) / rel
    return _normalized_sha256(path) if path.is_file() else None


def section_digest(value: Any) -> str:
    """SHA-256 of the canonical JSON of a parsed configuration section (comment-only edits do not change it)."""

    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def record_stage(ctx: Any, name: str, compute: Callable[[], Any], inputs: Iterable[Path], outputs: Iterable[Path], *,
                 sections: Sequence[str], dc_section: str, params: Mapping[str, Any] | None = None, status: str = "BUILT"):
    """Content-addressed stage call (as :func:`tmmri.stages._stage`) for the S5/S6 stages.

    The configuration hash covers ``sections`` of the study (``StudyConfig.section`` syntax) and the ``dc_section``
    of ``configs/data_centric.yaml``; both are listed in the record parameters.  In a run with ``.stages/FROZEN.json``
    a recorded stage may only be skipped (once-only guard, exit 7 on any difference).
    """

    from ..stages import once_only_guard

    run = ctx.require_run(name)
    cfg = data_centric_config(ctx.study)
    digest = {**ctx.study.sections_digest(list(sections)), f"{CONFIG_KEY}.{dc_section}": section_digest(cfg[dc_section]),
              f"{CONFIG_KEY}.common": section_digest(cfg.get("common"))}
    config_hash = hashlib.sha256("\n".join(f"{k}\t{v}" for k, v in sorted(digest.items())).encode("utf-8")).hexdigest()
    full = {**(params or {}), "config_sections": sorted([*sections, f"{CONFIG_KEY}.{dc_section}", f"{CONFIG_KEY}.common"])}
    return run_stage(run, name, compute, list(inputs), list(outputs), config_hash=config_hash, params=full, force=ctx.force,
                     n_jobs=ctx.jobs, status=status, rerun_guard=once_only_guard(run, name))


def bootstrap_settings(cfg: Mapping[str, Any]) -> dict[str, Any]:
    boot = cfg["common"]["bootstrap"]
    level = float(str(boot.get("interval", "percentile_95")).rsplit("_", 1)[-1]) / 100.0
    return {"n_boot": int(boot["replicates"]), "seed": int(boot["seed"]), "level": level,
            "min_valid_fraction": float(boot.get("min_valid_fraction", 0.95))}


def _interval(values: np.ndarray, level: float, min_valid_fraction: float, n_total: int) -> dict[str, Any]:
    valid = values[np.isfinite(values)]
    frac = float(valid.size / n_total) if n_total else 0.0
    alpha = (1.0 - level) / 2.0
    if not valid.size or frac < min_valid_fraction:
        return {"ci_lower": None, "ci_upper": None, "n_boot_valid": int(valid.size), "valid_fraction": frac,
                "status": "NE", "reason": "fewer_than_min_valid_bootstrap_replicates"}
    return {"ci_lower": float(np.quantile(valid, alpha)), "ci_upper": float(np.quantile(valid, 1.0 - alpha)),
            "n_boot_valid": int(valid.size), "valid_fraction": frac, "status": "EVALUATED", "reason": None}


def _finite_or_none(value: float) -> float | None:
    return float(value) if value is not None and np.isfinite(value) else None


def paired_rho_difference(y_ref: pd.DataFrame, pred_ref: pd.DataFrame, y_alt: pd.DataFrame, pred_alt: pd.DataFrame, null_mean: pd.Series,
                          reference_sd: pd.Series, family: Sequence[str], *, cluster: np.ndarray, resample_cluster: np.ndarray, n_boot: int,
                          seed: int, min_estimable: int, level: float = 0.95, min_valid_fraction: float = 0.95) -> dict[str, Any]:
    """Paired cluster bootstrap of rho(alt) - rho(ref) and of each rho_j (SPECIFICATION.md 3.3 estimand per condition).

    The four frames are row-aligned (same cases, same order).  Each condition has its own reference (``y_ref`` /
    ``y_alt``, e.g. B measured on the original and on the resampled masks) and its own reconstruction; the null is the
    fit-partition mean ``null_mean`` and the scale the fit-partition SD in both.  A target enters both conditions only
    on the rows where all four absolute errors are finite, and the aggregate uses the targets estimable in both
    conditions (fewer than ``min_estimable`` withdraws it, NE).  Clusters and draws are those of
    :func:`tmmri.metrics.primary_rho` (``cluster`` = weighting unit, ``resample_cluster`` = ``cluster_id``).
    """

    family = list(family)
    t0, p0 = _aligned(y_ref, pred_ref, family)
    t1, p1 = _aligned(y_alt, pred_alt, family)
    if t0.shape != t1.shape:
        raise ValueError("the reference and alternative conditions must be row-aligned")
    z = np.array([float(null_mean.get(c, np.nan)) for c in family], dtype=float)[None, :]
    errs = [np.abs(p0 - t0), np.abs(z - t0), np.abs(p1 - t1), np.abs(z - t1)]
    both = np.logical_and.reduce([np.isfinite(e) for e in errs])
    em0, en0, em1, en1 = (np.where(both, e, np.nan) for e in errs)
    sd = np.array([float(reference_sd.get(c, np.nan)) for c in family], dtype=float)
    codes, n_units = _cluster_codes(cluster, len(t0))
    unit_cluster, n_resample = _resampling_clusters(codes, n_units, resample_cluster)

    def evaluate(weights: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, float, float]:
        r0, e0 = _rho_from_errors(em0, en0, sd, codes, n_units, weights)
        r1, e1 = _rho_from_errors(em1, en1, sd, codes, n_units, weights)
        est = e0 & e1
        if est.sum() < min_estimable:
            return r0, r1, float("nan"), float("nan")
        return r0, r1, float(np.mean(r0[est])), float(np.mean(r1[est]))

    r0, r1, rho0, rho1 = evaluate(None)
    est = np.isfinite(r0) & np.isfinite(r1)
    out: dict[str, Any] = {
        "family": family, "n_rows": int(len(t0)), "n_rows_paired_per_target": {c: int(both[:, j].sum()) for j, c in enumerate(family)},
        "n_units": int(n_units), "n_clusters": int(n_resample), "n_boot": int(n_boot), "seed": int(seed),
        "interval": f"percentile_{int(round(level * 100))}", "paired": "same cluster draw for both conditions",
        "estimable_targets": [c for c, e in zip(family, est) if e], "min_estimable": int(min_estimable),
        "rho_reference": _finite_or_none(rho0), "rho_alternative": _finite_or_none(rho1),
        "rho_j_reference": {c: _finite_or_none(v) for c, v in zip(family, r0)},
        "rho_j_alternative": {c: _finite_or_none(v) for c, v in zip(family, r1)},
    }
    if not np.isfinite(rho0) or not np.isfinite(rho1):
        out.update({"delta_rho": None, "ci_lower": None, "ci_upper": None, "status": "NE", "reason": "fewer_than_min_estimable_targets",
                    "delta_rho_j": {c: {"estimate": _finite_or_none(b - a)} for c, a, b in zip(family, r0, r1)}})
        return out
    boot = np.full(int(n_boot), np.nan)
    boot_j = np.full((int(n_boot), len(family)), np.nan)
    for i, ss in enumerate(np.random.SeedSequence(int(seed)).spawn(int(n_boot))):
        w = _cluster_draw_weights(np.random.default_rng(ss), unit_cluster, n_resample)
        b0, b1, s0, s1 = evaluate(w)
        boot[i] = s1 - s0
        boot_j[i] = b1 - b0
    out.update({"delta_rho": float(rho1 - rho0), **_interval(boot, level, min_valid_fraction, int(n_boot))})
    out["delta_rho_j"] = {c: {"estimate": _finite_or_none(r1[j] - r0[j]),
                              **{k: v for k, v in _interval(boot_j[:, j], level, 0.0, int(n_boot)).items() if k in ("ci_lower", "ci_upper", "n_boot_valid")}}
                          for j, c in enumerate(family)}
    return out


def _decided(values: pd.Series) -> np.ndarray:
    return values.map(lambda v: v is not None and not (isinstance(v, float) and np.isnan(v)) and str(v) not in ("", "nan", "None", "<NA>")).to_numpy(dtype=bool)


def paired_accuracy_difference(y_true: Sequence[Any], pred_ref: Sequence[Any], pred_alt: Sequence[Any], *, clusters: Sequence[Any], n_boot: int,
                               seed: int, level: float = 0.95, min_valid_fraction: float = 0.95) -> dict[str, Any]:
    """Paired cluster bootstrap of accuracy(alt) - accuracy(ref) on the rows with a known reference and a decision in
    both conditions (draws as :func:`tmmri.classification.predictor_cluster_bootstrap`, clusters numbered by first row)."""

    truth = pd.Series(list(y_true), dtype=object)
    d0 = pd.Series(list(pred_ref), dtype=object)
    d1 = pd.Series(list(pred_alt), dtype=object)
    cl = pd.Series(list(clusters), dtype=object).astype(str)
    if not (len(truth) == len(d0) == len(d1) == len(cl)):
        raise ValueError("reference, decisions and clusters must be row-aligned")
    keep = _decided(truth) & _decided(d0) & _decided(d1)
    counts = {"n_rows": int(len(truth)), "n_evaluable_paired": int(keep.sum()), "n_reference_unknown": int((~_decided(truth)).sum()),
              "n_no_decision_reference": int((~_decided(d0)).sum()), "n_no_decision_alternative": int((~_decided(d1)).sum())}
    out: dict[str, Any] = {"counts": counts, "n_boot": int(n_boot), "seed": int(seed), "interval": f"percentile_{int(round(level * 100))}",
                           "paired": "same cluster draw for both conditions", "unit": "cluster_id"}
    if not keep.any():
        return {**out, "accuracy_reference": None, "accuracy_alternative": None, "delta_accuracy": None, "ci_lower": None, "ci_upper": None,
                "status": "NE", "reason": "no_evaluable_rows"}
    t = truth[keep].astype(str).to_numpy()
    c0 = (d0[keep].astype(str).to_numpy() == t).astype(float)
    c1 = (d1[keep].astype(str).to_numpy() == t).astype(float)
    unit_cluster, n_resample = _resampling_clusters(np.arange(t.size), int(t.size), cl[keep].to_numpy(dtype=object))
    boot = np.full(int(n_boot), np.nan)
    for i, ss in enumerate(np.random.SeedSequence(int(seed)).spawn(int(n_boot))):
        w = _cluster_draw_weights(np.random.default_rng(ss), unit_cluster, n_resample)
        total = w.sum()
        if total > 0:
            boot[i] = float((w * c1).sum() / total - (w * c0).sum() / total)
    a0, a1 = float(c0.mean()), float(c1.mean())
    out.update({"accuracy_reference": a0, "accuracy_alternative": a1, "delta_accuracy": a1 - a0, "n_clusters": int(n_resample),
                "n_changed_decisions": int((d0[keep].astype(str).to_numpy() != d1[keep].astype(str).to_numpy()).sum()),
                **_interval(boot, level, min_valid_fraction, int(n_boot))})
    return out


def config_sha256_params(study: StudyConfig) -> dict[str, Any]:
    """Provenance fields of every S5/S6 output: the configuration file hash and the combined study hash."""

    return {"data_centric_sha256": data_centric_file_sha256(study), "study_config_hash": study.config_hash,
            "data_centric_config_digest": sha256_json(data_centric_config(study))}
