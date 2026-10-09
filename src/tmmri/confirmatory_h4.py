"""Second confirmatory family H4 of protocol-v6.0.1 (Decision D36; ``configs/cohorts.yaml`` ``mms2.endpoints.h4``).

The owner's risk-management hook beside H-primary-3.  The hypotheses were generated on the historical M&Ms-1 evaluation
(per-target ``rho_j``: LVEF 0.63 and RVEF 0.50 transferred, the absolute volumes did not) and are tested prospectively on
``mms2_external``, declared before any M&Ms-2 byte is read:

* **H4a, ratio concepts transfer:** ``rho_EF`` is the v6 estimand (E1) of the unchanged v6.0.0 bundle (the code of
  H-primary-3: :func:`tmmri.metrics.primary_rho`, fit-SD-normalized MAE of ridge against the train-mean predictor)
  restricted to the ejection-fraction targets of the registry primary family (``lv_ef_percent``, ``rv_ef_percent``).
  It passes when the lower bound of its two-sided 97.5 % cluster-bootstrap interval exceeds 0.10; both EF targets must
  be estimable, otherwise ``NE``.
* **H4b, label-free harmonization helps:** ``delta_rho = rho(S3 variant H-a) - rho(frozen bundle)`` over the primary
  family on the same rows, where H-a re-standardizes A with the mean and SD of the unlabelled target partition and T,
  lambda and the predictor stay frozen (:mod:`tmmri.experiments.trust.harmonize`); paired cluster bootstrap
  (:func:`tmmri.experiments.alt_bundles.paired_rho_difference`).  It passes when the lower bound of the two-sided
  97.5 % interval exceeds 0.

Multiplicity: Bonferroni within the family (two tests; family alpha 0.05, so each test is judged by its two-sided 97.5 %
interval).  The family is separate from H-primary-3 and from F6.  Bootstrap: 2,000 replicates, seed 20260928 and one
stable sub-seed per hypothesis (:func:`sub_seed`), the cluster unit and the v6 NE rules.

Stage ``evaluate-h4 --split <partition>`` (:func:`evaluate_h4`) reads only recorded artifacts of ``evaluate --split``
and the hash-verified frozen bundle; it fits nothing on the evaluated partition except the label-free H-a statistics.
It is confirmatory on ``mms2_external`` only; on ``acdc_test`` and ``mms_external`` the same code runs descriptively and
the record is labelled "registered after unblinding".  The decision record ``metrics/confirmatory_h4/h4_<partition>.json``
is written once: the v6 once-only guard refuses a changed re-execution in a frozen run, and an existing record without a
stage record is never overwritten.  Research use only; estimation, not diagnosis.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .experiments.l57_stages import LABEL_COLUMNS, _clean, lane_config
from .runs import atomic_write_json, sha256_json
from .stages import (
    MMS2_SPLIT,
    BundleError,
    StageContext,
    _check_recorded_output,
    _cluster_ids,
    _contracts,
    _dev_reference,
    _require_evaluated,
    _split_manifest,
    _stage,
    methods_api,
    require_mms2_unlocked,
    require_mms2_validated,
    require_mms_unlocked,
    verify_frozen_bundle,
)

FAMILY_ID = "H4"
PARTITIONS = ("acdc_test", "mms_external", MMS2_SPLIT)
CONFIRMATORY_PARTITION = MMS2_SPLIT
DATASETS = {"acdc_test": "ACDC", "mms_external": "MMS", MMS2_SPLIT: "MMS2"}
RECORD_DIR = "confirmatory_h4"                     # metrics/confirmatory_h4/ (read by the v6.0.1 Table 3 only)
RECORD_SCHEMA = "tmmri.h4_family/1"
AFTER_UNBLINDING = "registered after unblinding"
PROSPECTIVE = "confirmatory; prospective (protocol-v6.0.1)"
EF_NOT_IN_FAMILY = "EF_TARGET_NOT_IN_PRIMARY_FAMILY"
EF_NOT_RECORDED = "EF_TARGET_NOT_RECORDED"
HARMONIZATION_NOT_ESTIMABLE = "HARMONIZATION_NOT_ESTIMABLE"


class H4RecordExistsError(BundleError):
    """The once-only H4 decision record exists already; it is never overwritten."""


# ----------------------------------------------------------------------------- configuration


def h4_config(study: Any) -> dict[str, Any]:
    """``configs/cohorts.yaml`` ``mms2.endpoints.h4``."""

    return dict(study.cohorts["mms2"]["endpoints"]["h4"])


def interval_level(cfg: Mapping[str, Any]) -> float:
    """Bonferroni within the family: each test is judged by its two-sided ``1 - alpha / n_tests`` interval (0.975)."""

    mult = cfg["multiplicity"]
    return 1.0 - float(mult["family_alpha"]) / int(mult["n_tests"])


def sub_seed(seed: int, hypothesis_id: str) -> int:
    """Stable sub-seed of one hypothesis: the first 4 bytes (big-endian) of ``SHA256("<seed>|<hypothesis id>")``."""

    return int.from_bytes(hashlib.sha256(f"{int(seed)}|{hypothesis_id}".encode("utf-8")).digest()[:4], "big")


def record_path(run: Any, partition: str) -> Path:
    return run.path / "metrics" / RECORD_DIR / f"h4_{partition}.json"


def _interval_label(level: float) -> str:
    return f"percentile_two_sided_{level * 100:g}"


# ----------------------------------------------------------------------------- estimands (pure functions)


def _null(mean: pd.Series, targets: Sequence[str], index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame([mean[list(targets)]] * len(index), index=index)


def rho_ef(y_true: pd.DataFrame, y_pred: pd.DataFrame, mean: pd.Series, sd: pd.Series, targets: Sequence[str], *,
           family: Sequence[str], cluster: Sequence[Any], resample_cluster: Sequence[Any], n_boot: int, seed: int, level: float,
           min_valid_fraction: float, n_jobs: int = 1) -> dict[str, Any]:
    """H4a: the v6 estimand (E1) of the frozen predictions restricted to the EF ``targets`` (all of them must be estimable).

    ``targets`` must belong to the registry primary ``family`` and be columns of both frames; otherwise ``NE``."""

    from .metrics import primary_rho

    targets = [str(t) for t in targets]
    base = {"family": targets, "estimand": "rho_EF", "min_estimable": len(targets), "n_boot": int(n_boot), "seed": int(seed),
            "interval": _interval_label(level), "level": float(level)}
    outside = [t for t in targets if t not in set(family)]
    absent = [t for t in targets if t not in y_true.columns or t not in y_pred.columns]
    if not targets or outside or absent:
        reason = EF_NOT_IN_FAMILY if outside or not targets else EF_NOT_RECORDED
        return {**base, "rho": None, "ci_lower": None, "ci_upper": None, "rho_j": {}, "status": "NE", "reason": reason,
                "targets_outside_family": outside, "targets_not_recorded": absent}
    out = primary_rho(y_true[targets], y_pred[targets], _null(mean, targets, y_true.index), sd, targets, int(n_boot), int(seed),
                      np.asarray(cluster, dtype=object), len(targets), level=float(level), min_valid_fraction=float(min_valid_fraction),
                      n_jobs=int(n_jobs), resample_cluster=np.asarray(resample_cluster, dtype=object))
    return {**out, "estimand": "rho_EF", "interval": _interval_label(level), "level": float(level), "n_cases": int(len(y_true))}


def delta_rho(y_true: pd.DataFrame, frozen: pd.DataFrame, harmonized: pd.DataFrame, mean: pd.Series, sd: pd.Series, family: Sequence[str], *,
              cluster: Sequence[Any], resample_cluster: Sequence[Any], n_boot: int, seed: int, level: float, min_estimable: int,
              min_valid_fraction: float) -> dict[str, Any]:
    """H4b: ``rho(harmonized) - rho(frozen)`` over ``family`` with one paired cluster draw for both members."""

    from .experiments.alt_bundles import paired_rho_difference

    family = [str(f) for f in family]
    out = paired_rho_difference(y_true[family], frozen[family], harmonized[family], _null(mean, family, y_true.index), sd, family,
                                cluster=cluster, resample_cluster=resample_cluster, n_boot=int(n_boot), seed=int(seed),
                                min_estimable=int(min_estimable), level=float(level), min_valid_fraction=float(min_valid_fraction))
    return {**out, "estimand": "delta_rho", "reference": "frozen", "comparator": "H-a", "interval": _interval_label(level), "level": float(level),
            "n_cases": int(len(y_true))}


def harmonize_h_a(model: Any, a_frame: pd.DataFrame, *, forbidden: Sequence[str] = ()) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """B-hat of the frozen ``model`` under S3 variant H-a (the target partition's own mean and SD of A; label-free)."""

    from .experiments.trust.harmonize import apply_harmonized, fit_harmonization

    columns = list(a_frame.columns)
    try:
        harm = fit_harmonization(a_frame, columns, variant="H-a", forbidden=forbidden)
    except ValueError as exc:          # no finite row: the statistics do not exist
        return None, {"variant": "H-a", "status": "NE", "reason": HARMONIZATION_NOT_ESTIMABLE, "detail": str(exc)}
    bhat, record = apply_harmonized(model, a_frame.to_numpy(dtype=np.float64), harm, pd.Index(a_frame.index))
    record.update({"label_free_assertion": "passed (representation columns only; B and labels read only to evaluate)",
                   "statistics": {name: stats.to_dict() for name, stats in harm["groups"].items()}})
    return bhat, record


def decide(hypothesis: Mapping[str, Any], estimate: Mapping[str, Any], value_key: str, *, confirmatory: bool) -> dict[str, Any]:
    """The decision entry of one hypothesis: PASSED or FAILED on the confirmatory partition, DESCRIPTIVE elsewhere, NE
    when the estimand or its interval is not estimable.  The criterion is ``ci_lower > threshold``."""

    lb = estimate.get("ci_lower")
    threshold = float(hypothesis["threshold"])
    entry: dict[str, Any] = {"hypothesis_id": str(hypothesis["id"]), "estimand": str(hypothesis["estimand"]), "estimate": estimate.get(value_key),
                             "ci_lower": lb, "ci_upper": estimate.get("ci_upper"), "interval": estimate.get("interval"), "threshold": threshold,
                             "rule": hypothesis.get("rule"), "tested": bool(confirmatory), "n_cases": estimate.get("n_cases"),
                             "n_clusters": estimate.get("n_clusters")}
    if estimate.get("status") == "NE" or lb is None or not np.isfinite(float(lb)):
        entry.update({"criterion_met": None, "passed": None if not confirmatory else False, "status": "NE",
                      "reason": estimate.get("reason") or "interval_not_estimable"})
        return entry
    met = bool(float(lb) > threshold)
    entry.update({"criterion_met": met, "passed": met if confirmatory else None,
                  "status": ("PASSED" if met else "FAILED") if confirmatory else "DESCRIPTIVE", "reason": None})
    return entry


def family_summary(results: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any], *, confirmatory: bool) -> dict[str, Any]:
    level = interval_level(cfg)
    out = {"family": FAMILY_ID, "multiplicity": {**dict(cfg["multiplicity"]), "per_test_alpha": float(cfg["multiplicity"]["family_alpha"]) / int(cfg["multiplicity"]["n_tests"]),
                                                 "interval_level": level},
           "confirmatory": bool(confirmatory), "scope": cfg.get("family_scope")}
    if confirmatory:
        out.update({"passed": [r["hypothesis_id"] for r in results if r["status"] == "PASSED"],
                    "failed": [r["hypothesis_id"] for r in results if r["status"] == "FAILED"],
                    "not_estimable": [r["hypothesis_id"] for r in results if r["status"] == "NE"]})
    return out


def write_decision_record(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write the H4 decision record once; an existing record is never overwritten (:class:`H4RecordExistsError`)."""

    path = Path(path)
    if path.exists():
        raise H4RecordExistsError(f"{path} exists: the H4 decision record is written once and never overwritten (once-only; a repeat needs a new run id)")
    path.parent.mkdir(parents=True, exist_ok=True)
    return atomic_write_json(path, dict(payload), allow_nan=True)


# ----------------------------------------------------------------------------- stage evaluate-h4


def _guard(ctx: StageContext, split: str, stage: str) -> None:
    ds = DATASETS[split]
    if ds == "MMS":
        require_mms_unlocked(ctx, stage)
    elif ds == "MMS2":
        require_mms2_unlocked(ctx, stage)
        require_mms2_validated(ctx, stage)


def evaluate_h4(ctx: StageContext, split: str):
    """H4a and H4b once on ``split`` from the recorded ``evaluate --split`` outputs and the frozen v6.0.0 bundle."""

    if split not in PARTITIONS:
        raise ValueError(f"evaluate-h4 --split must be one of {PARTITIONS}")
    name = f"evaluate-h4:{split}"
    _guard(ctx, split, f"evaluate-h4 --split {split}")
    run = ctx.require_run(name)
    frozen = verify_frozen_bundle(ctx)
    record = _require_evaluated(run, f"evaluate:{split}", name)
    ds = DATASETS[split]
    p = run.path
    bhat_path = p / "reconstructed" / f"bhat_{split}.parquet"
    _check_recorded_output(run, record, bhat_path, name)
    study = ctx.study
    cfg = h4_config(study)
    confirmatory = split == str(cfg["partition"]) == CONFIRMATORY_PARTITION
    if not confirmatory and split not in [str(s) for s in cfg.get("descriptive_partitions", [])]:
        raise ValueError(f"evaluate-h4: {split} is neither the H4 partition nor one of its descriptive partitions")
    out = record_path(run, split)
    contracts = _contracts(ctx)
    inputs = [run.frozen_path, run.record_path(f"evaluate:{split}"), bhat_path, p / "formal" / f"A_{ds}.parquet", p / "semantics" / f"B_{ds}.parquet",
              p / "semantics" / "B_ACDC.parquet", p / "transitions" / "transition_model.json", contracts / "split_manifest.parquet"]
    if ds == "MMS2":
        inputs.append(contracts / "split_manifest_MMS2.parquet")
    trust_s3 = _trust_s3(study)

    def compute() -> dict[str, Any]:
        if out.exists():
            raise H4RecordExistsError(f"{run.portable(out)} exists without a matching {name} record: the H4 decision record is never overwritten")
        from .formal_features import formal_descriptor_names

        boot_cfg = cfg["bootstrap"]
        n_boot, seed = int(boot_cfg["replicates"]), int(boot_cfg["seed"])
        min_valid = float(study.bootstrap["fixed_model_bootstrap"]["min_valid_fraction"])
        level = interval_level(cfg)
        hyp = {k: dict(v) for k, v in cfg["hypotheses"].items()}
        seeds = {h["id"]: sub_seed(seed, str(h["id"])) for h in hyp.values()}
        family = study.primary_family()
        split_manifest = _split_manifest(ctx, ds if ds == "MMS2" else None)
        ids = split_manifest[split_manifest["partition"] == split].sort_values("sample_id")
        a = pd.read_parquet(p / "formal" / f"A_{ds}.parquet").set_index("sample_id")
        b = pd.read_parquet(p / "semantics" / f"B_{ds}.parquet").set_index("sample_id")
        keep = [s for s in ids["sample_id"] if s in a.index and s in b.index]          # the rows of evaluate (H-primary rho)
        bhat = pd.read_parquet(bhat_path).set_index("sample_id")
        if [str(s) for s in bhat.index] != [str(s) for s in keep]:
            raise BundleError(f"{name}: the recorded reconstructions of evaluate:{split} do not cover exactly its evaluation rows")
        clusters = _cluster_ids(split_manifest, keep)
        weighting = [str(s).split(":", 1)[1] if ":" in str(s) else str(s) for s in keep]     # as tmmri.stages._rho
        resample = clusters.reindex(keep).to_numpy(dtype=object)
        sd, mean = _dev_reference(ctx)
        y_true = b.loc[keep]
        # H4a: the H-primary rho code on the EF targets of the registry primary family.
        h4a = rho_ef(y_true, bhat, mean, sd, hyp["h4a"]["targets"], family=family, cluster=weighting, resample_cluster=resample,
                     n_boot=n_boot, seed=seeds[hyp["h4a"]["id"]], level=level, min_valid_fraction=min_valid, n_jobs=ctx.jobs)
        # H4b: the frozen T (hash-verified bundle) must reproduce the recorded B-hat; then S3 H-a on the same rows.
        model_cls = methods_api("tmmri.transition_fit", "TransitionModel")
        model = model_cls.from_dict(json.loads((p / "transitions" / "transition_model.json").read_text(encoding="utf-8")))
        a_frame = a.loc[keep, formal_descriptor_names()]
        again = model.predict(a_frame.to_numpy(dtype=np.float64))
        again.index = keep
        targets = [t for t in bhat.columns if t in again.columns]
        if not np.allclose(again[targets].to_numpy(dtype=float), bhat[targets].to_numpy(dtype=float), rtol=1e-9, atol=1e-9, equal_nan=True):
            raise BundleError(f"{name}: the frozen transition does not reproduce the recorded reconstructions of evaluate:{split}")
        harmonized, harm_record = harmonize_h_a(model, a_frame, forbidden=sorted({*map(str, b.columns), *LABEL_COLUMNS}))
        boot_b = dict(cluster=weighting, resample_cluster=resample, n_boot=n_boot, seed=seeds[hyp["h4b"]["id"]], level=level,
                      min_estimable=int(study.bootstrap["primary_test"]["min_estimable_targets"]), min_valid_fraction=min_valid)
        if harmonized is None:
            h4b = {"estimand": "delta_rho", "difference": None, "ci_lower": None, "ci_upper": None, "status": "NE", "reason": harm_record["reason"],
                   "interval": _interval_label(level), "n_cases": len(keep)}
        else:
            h4b = delta_rho(y_true, bhat, harmonized, mean, sd, family, **boot_b)
        results = [decide(hyp["h4a"], h4a, "rho", confirmatory=confirmatory),
                   decide(hyp["h4b"], h4b, "difference", confirmatory=confirmatory)]
        payload = {
            "schema": RECORD_SCHEMA, "family": FAMILY_ID, "partition": split, "confirmatory": confirmatory,
            "evidence": PROSPECTIVE if confirmatory else AFTER_UNBLINDING,
            "analysis_label": None if confirmatory else AFTER_UNBLINDING,
            "hypotheses": {h["id"]: {k: v for k, v in h.items()} for h in hyp.values()},
            "origin": cfg.get("origin"), "family_decision": family_summary(results, cfg, confirmatory=confirmatory),
            "bootstrap": {"replicates": n_boot, "seed": seed, "sub_seeds": seeds, "sub_seed_rule": boot_cfg.get("sub_seed_rule"),
                          "unit": str(boot_cfg.get("unit", "cluster_id")), "interval": _interval_label(level), "level": level,
                          "min_valid_fraction": min_valid, "paired": "the same cluster draw for both members of delta_rho"},
            "results": results, "h4a": h4a,
            "h4b": {**h4b, "harmonization": harm_record, "trust_s3_harmonization": trust_s3},
            "n_cases": int(len(keep)), "n_clusters": int(clusters.nunique()), "bundle": cfg.get("bundle"), "bundle_sha256": frozen["bundle_sha256"],
            "reconstructions": f"reconstructed/bhat_{split}.parquet (evaluate:{split}; raw B-hat, the H-primary substrate)",
            "reference": "fit-partition (acdc_dev) mean and SD; never re-estimated on the evaluated partition",
            "fitting": "none on the evaluated partition except the label-free H-a mean and SD of A (transductive, S3)"}
        write_decision_record(out, _clean(payload))
        return {"partition": split, "confirmatory": confirmatory, "results": {r["hypothesis_id"]: r["status"] for r in results}}

    return _stage(ctx, name, compute, inputs, [out], {"split": split, "confirmatory": confirmatory, "trust_s3_sha256": sha256_json(trust_s3)},
                  status="EVALUATED")


def _trust_s3(study: Any) -> dict[str, Any] | None:
    """The S3 section of ``configs/trust.yaml`` (H-a definition; recorded and part of the stage parameters)."""

    try:
        return dict(lane_config(study, "trust").get("s3_harmonization") or {})
    except FileNotFoundError:
        return None
