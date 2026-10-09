"""Protocol-v6.0.1 amendment items 11-14 on a v6 run root (lane L57): F9b probes, alternative bundles, trust S1-S4.

Stages (CLI ``python -m tmmri.cli <stage>``; :mod:`tmmri.experiments.l57_cli`):

* ``fit-probes`` (D.11): per outer fold k the v6 typed ridge T on encoder k's 64-d activations of its fitting patients
  (out-of-fold B-hat and rho on ``acdc_dev``) and the final T on the final encoder's activations of all 98; writes the
  F9b bundle manifest ``probes/f9b/bundle.json`` (its hash freeze).
* ``evaluate-alt-bundles --split`` (D.12-D.13): once-only per (bundle, partition).  Representations d20 and f9b: rho and
  rho_j with the frozen v6 estimand code; predictors d20_predictor and f9 (three seeds, primary first): accuracy,
  macro-F1, one-vs-rest AUROC, ECE, Brier, log loss on the same-definition crosswalk subsets, decision distributions
  of the non-equivalent strata, paired cluster-bootstrap differences against the descriptor pipeline.
* ``fit-trust`` (E, S1/S2/S4): thresholds, Mahalanobis references and temperatures on ``acdc_dev`` out-of-fold outputs
  only, frozen by hash in ``trust/trust_bundle.json``.
* ``evaluate-trust --split`` (E, S1-S4): once-only per partition.
* ``build-comparison-tables``: Table 4 and the tidy Figure-5 parquet from the recorded evaluations (read-only; NE with a
  reason where an input is absent).

Every stage is content-addressed through :func:`tmmri.runs.run_stage`; the evaluation and fitting stages carry the
v6 once-only guard (:func:`tmmri.stages.once_only_guard`): in a frozen run a recorded stage may only be skipped.
The configuration hash covers the named sections of the study configuration plus the named sections of
``configs/probes.yaml`` and ``configs/trust.yaml``.  M&Ms partitions obey :func:`tmmri.stages.require_mms_unlocked`
(``mms2_external``: ``tmmri.stages.require_mms2_unlocked`` when the M&Ms-2 lane provides it, else that guard plus the
annotated tag of configs/probes.yaml ``partitions.mms2_external.requires_tag``, protocol-v6.0.1).  Research use only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import yaml

from .. import stages as v6
from ..classification import _auroc_from_scores, label_distribution
from ..metrics import wilson_interval
from ..runs import RunRoot, atomic_write_json, code_version, hash_path, run_stage, sha256_json
from .alt_bundles import (activation_columns, argmax_labels, bootstrap_statistic, c4_logits, c4_probabilities, evaluable_rows,
                          paired_predictor_differences, paired_rho_difference, predictor_metrics, read_c4, seed_spread)

STAGES = ("fit-probes", "evaluate-alt-bundles", "fit-trust", "evaluate-trust", "build-comparison-tables")
PREDICTORS = ("audited_predictor", "d20_predictor", "f9")
REPRESENTATIONS = ("descriptors", "d20", "f9b")
NOT_RECORDED = "INPUT_NOT_RECORDED"
NO_A_STANDARDIZATION = "END_TO_END_NETWORK_NO_A_STANDARDIZATION"
LABEL_COLUMNS = ("acdc_class", "crosswalk_acdc_class", "crosswalk_compatibility", "pathology", "y_true", "y_pred", "label", "reference",
                 "d_model", "disease_label")

# Configuration read by each stage: study sections (StudyConfig syntax) and lane-config top-level keys.
STUDY_SECTIONS: dict[str, tuple[str, ...]] = {
    "fit-probes": ("transition", "registry", "cohorts.qc_severity", "splits.acdc.acdc_dev.outer_folds", "bootstrap.fixed_model_bootstrap",
                   "bootstrap.primary_test"),
    "evaluate-alt-bundles": ("transition", "registry", "cohorts.qc_severity", "formal.audited_predictor", "bootstrap.fixed_model_bootstrap",
                             "bootstrap.primary_test", "bootstrap.calibration"),
    "fit-trust": ("transition", "registry", "cohorts.qc_severity", "formal.audited_predictor", "splits.acdc.acdc_dev.outer_folds"),
    "evaluate-trust": ("transition", "registry", "cohorts.qc_severity", "formal.audited_predictor", "bootstrap.fixed_model_bootstrap",
                       "bootstrap.primary_test", "bootstrap.calibration", "rule_policy.rulebooks"),
    "build-comparison-tables": ("bootstrap.subgroup_reporting_thresholds",),
}
LANE_SECTIONS: dict[str, dict[str, tuple[str, ...]]] = {
    "fit-probes": {"probes": ("f9b", "inputs")},
    "evaluate-alt-bundles": {"probes": ("partitions", "f9b", "inputs", "require_freeze_record_for_held_out", "bundles", "pairing", "reference",
                                        "estimands", "crosswalk")},
    "fit-trust": {"trust": ("predictors", "representations", "s1_selective_prediction", "s2_ood", "s4_calibration"),
                  "probes": ("f9b", "inputs", "pairing")},
    "evaluate-trust": {"trust": ("predictors", "representations", "evaluation_partitions", "s1_selective_prediction", "s2_ood",
                                 "s3_harmonization", "s4_calibration", "min_clusters"),
                       "probes": ("partitions", "f9b", "inputs", "pairing", "estimands", "crosswalk", "require_freeze_record_for_held_out")},
    "build-comparison-tables": {"probes": ("partitions", "bundles", "pairing", "table4")},
}


# ----------------------------------------------------------------------------- configuration and stage wrapper


def lane_config(study: Any, name: str) -> dict[str, Any]:
    """``configs/<name>.yaml`` of the study's configuration tree (or the parsed copy when the study lists it)."""

    if name in study.configs:
        return study.configs[name]
    path = Path(study.root) / "configs" / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing (protocol-v6.0.1 lane configuration)")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def stage_config_hash(ctx: v6.StageContext, stage: str) -> tuple[str, list[str]]:
    study_sections = STUDY_SECTIONS[stage]
    lane = {f"{cfg}.{key}": lane_config(ctx.study, cfg).get(key) for cfg, keys in LANE_SECTIONS[stage].items() for key in keys}
    digest = sha256_json({"study_sections": ctx.study.sections_hash(study_sections), "lane_sections": lane})
    return digest, [*study_sections, *sorted(lane)]


def _lane_stage(ctx: v6.StageContext, stage: str, name: str, fn: Callable[[], Any], inputs: Iterable[Path], outputs: Iterable[Path],
                params: dict[str, Any] | None = None, *, status: str = "BUILT", once_only: bool = True):
    run = ctx.require_run(name)
    config_hash, sections = stage_config_hash(ctx, stage)
    return run_stage(run, name, fn, list(inputs), list(outputs), config_hash=config_hash, params={**(params or {}), "config_sections": sections},
                     force=ctx.force, n_jobs=ctx.jobs, status=status, rerun_guard=v6.once_only_guard(run, name) if once_only else None)


def _probes(ctx: v6.StageContext) -> dict[str, Any]:
    return lane_config(ctx.study, "probes")


def _trust(ctx: v6.StageContext) -> dict[str, Any]:
    return lane_config(ctx.study, "trust")


def _input_path(ctx: v6.StageContext, group: str, field: str, **fmt: Any) -> Path:
    return ctx.require_run("l57").path / str(_probes(ctx)["inputs"][group][field]).format(**fmt)


def f9_path(ctx: v6.StageContext, field: str, seed: int) -> Path:
    """The F9 C4 table of one seed (``predictions`` or ``activations``; a template may name ``{seed}``)."""

    return _input_path(ctx, "f9", field, seed=int(seed))


def f9_paths(ctx: v6.StageContext, field: str) -> list[Path]:
    return list(dict.fromkeys(f9_path(ctx, field, s) for s in _seeds(ctx)))


def _classes(ctx: v6.StageContext) -> list[str]:
    return [str(c) for c in ctx.study.formal["audited_predictor"]["classes"]]


def _boot(ctx: v6.StageContext) -> dict[str, Any]:
    b = ctx.study.bootstrap
    return {"n_boot": int(b["fixed_model_bootstrap"]["replicates"]), "seed": int(b["fixed_model_bootstrap"]["seed"]),
            "min_valid_fraction": float(b["fixed_model_bootstrap"]["min_valid_fraction"]), "n_bins": int(b["calibration"]["ece_bins"])}


def _seeds(ctx: v6.StageContext) -> list[int]:
    return [int(s) for s in _probes(ctx)["f9b"]["seeds"]]


def _partition(ctx: v6.StageContext, partition: str) -> dict[str, Any]:
    parts = _probes(ctx)["partitions"]
    if partition not in parts:
        raise ValueError(f"unknown partition {partition!r}; configs/probes.yaml partitions: {sorted(parts)}")
    return {"partition": partition, **parts[partition]}


def guard_partition(ctx: v6.StageContext, partition: str, stage: str) -> str | None:
    """M&Ms partitions need the protocol freeze (or the synthetic override), exactly as every v6 M&Ms endpoint stage."""

    info = _partition(ctx, partition)
    if info["dataset"] == "ACDC":
        return None
    if partition == "mms2_external":
        custom = getattr(v6, "require_mms2_unlocked", None)   # provided by the M&Ms-2 lane (exit 3 before its tag)
        if custom is not None:
            return custom(ctx, stage)
    result = v6.require_mms_unlocked(ctx, stage)
    tag = info.get("requires_tag")                             # mms2_external: the protocol-v6.0.1 tag as well
    if tag and result != "synthetic_override" and not v6.protocol_frozen(str(tag)):
        raise v6.ProtocolFreezeError(f"{stage}: {partition} is evaluated only after the annotated tag {tag} is contained in HEAD")
    return result


def split_inputs(run: RunRoot, partition: str) -> list[Path]:
    """The split manifests read for ``partition`` (stage inputs): ``contracts/split_manifest.parquet`` and, for
    ``mms2_external``, ``contracts/split_manifest_MMS2.parquet`` (:func:`tmmri.stages.split_manifest_for`)."""

    contracts = run.path / "contracts"
    return [contracts / "split_manifest.parquet"] + ([contracts / "split_manifest_MMS2.parquet"] if str(partition).startswith("mms2") else [])


def partition_evidence(ctx: v6.StageContext, partition: str, bundles: Iterable[str]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """(partition entry with its evidence label, freeze order).  On ``mms2_external`` SPECIFICATION.md 14.6 (D33) applies:
    a bundle whose freeze record is not dated before the M&Ms-2 extraction record is labelled post hoc
    (:func:`tmmri.stages.mms2_freeze_order`); elsewhere the configured label stands."""

    info = _partition(ctx, partition)
    if partition != "mms2_external":
        return info, None
    order = v6.mms2_freeze_order(ctx, list(dict.fromkeys(bundles)))
    if not order["before_extraction"]:
        info = {**info, "evidence": order["evidence"], "label": order["label"]}
    return info, order


def rel_path(run: RunRoot, path: Path) -> str:
    """Run-relative POSIX path (the key convention of FROZEN.json and of every L57 hash manifest)."""

    return Path(path).resolve().relative_to(run.path.resolve()).as_posix()


def _manifest_files(payload: dict[str, Any], rel: str = "") -> dict[str, str] | None:
    """Run-relative path -> SHA-256 of a freeze record: ``bundle`` or ``files`` (run-relative; ``tmmri.d20_bundle/1``
    lists ``files`` relative to the record's folder, ``deep_encoder/``), or the ``fits`` of ``tmmri.f9_bundle/1``
    (each checkpoint ``model.pt`` and its ``record.json``)."""

    schema = str(payload.get("schema") or "")
    if schema.startswith("tmmri.f9_bundle") and isinstance(payload.get("fits"), list):
        out: dict[str, str] = {}
        for fit in payload["fits"]:
            checkpoint = str(fit["checkpoint"])
            out[checkpoint] = str(fit["checkpoint_sha256"])
            if fit.get("record_sha256"):
                out[str(Path(checkpoint).parent.as_posix()) + "/record.json"] = str(fit["record_sha256"])
        return out or None
    files = payload.get("bundle") if isinstance(payload.get("bundle"), dict) else payload.get("files")
    if not isinstance(files, dict):
        return None
    base = Path(rel).parent.as_posix() if schema.startswith("tmmri.d20_bundle") else ""
    return {(f"{base}/{k}" if base and base != "." else (k[4:] if str(k).startswith("run:") else str(k))): v for k, v in files.items()}


def verify_hash_manifest(run: RunRoot, rel: str, what: str) -> dict[str, Any]:
    """Re-hash every file of a freeze record (``bundle`` or ``files``: run-relative path -> SHA-256)."""

    path = run.path / rel
    if not path.is_file():
        raise v6.BundleError(f"{what}: freeze record {rel} is missing; the bundle hash must be recorded before a held-out evaluation")
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = _manifest_files(payload, rel)
    if not files:
        raise v6.BundleError(f"{what}: {rel} has no 'bundle' or 'files' hash map")
    changed = sorted(r for r, digest in files.items() if hash_path(run.path / r) != digest)
    if changed:
        raise v6.BundleError(f"{what}: files changed after the freeze recorded in {rel}: {changed[:5]}")
    return {"record": rel, "record_sha256": hash_path(path), "bundle_sha256": payload.get("bundle_sha256") or sha256_json(files), "n_files": len(files)}


# ----------------------------------------------------------------------------- partition data


@dataclass
class PartitionData:
    partition: str
    dataset: str
    ids: list[str]                      # partition rows present in B (sorted)
    b: pd.DataFrame                     # typed B, index sample_id
    truth: pd.Series                    # class reference (ACDC class; M&Ms same-definition crosswalk), None elsewhere
    critical: pd.Series                 # critical QC (excluded from predictor metrics)
    clusters: pd.Series                 # cluster_id
    compat: pd.Series | None
    mapped: pd.Series | None
    vendor: pd.Series | None

    @property
    def eval_ids(self) -> list[str]:
        return [s for s in self.ids if not bool(self.critical[s])]


def partition_data(ctx: v6.StageContext, partition: str) -> PartitionData:
    info = _partition(ctx, partition)
    ds = str(info["dataset"])
    run = ctx.require_run("l57")
    split = v6.split_manifest_for(ctx, partition)          # M&Ms-2 rows: contracts/split_manifest_MMS2.parquet
    b = pd.read_parquet(run.path / "semantics" / f"B_{ds}.parquet")
    b["sample_id"] = b["sample_id"].astype(str)
    b = b.set_index("sample_id")
    members = sorted(split.loc[split["partition"] == partition, "sample_id"].astype(str))
    ids = [s for s in members if s in b.index]
    cohort = v6._cohort(ctx, ds)
    cohort["sample_id"] = cohort["sample_id"].astype(str)
    cohort = cohort.set_index("sample_id")
    truth = v6.reference_labels(cohort, "ACDC" if ds == "ACDC" else "MMS").reindex(ids).astype(object)
    truth = truth.where(truth.notna(), None)
    compat = cohort["crosswalk_compatibility"].reindex(ids).astype(object) if ds != "ACDC" and "crosswalk_compatibility" in cohort.columns else None
    mapped = cohort["crosswalk_acdc_class"].reindex(ids).astype(object) if ds != "ACDC" and "crosswalk_acdc_class" in cohort.columns else None
    vendor = cohort["vendor"].reindex(ids).rename("vendor") if "vendor" in cohort.columns else None
    critical = v6.critical_qc_mask(b.loc[ids], ctx.study).astype(bool)
    return PartitionData(partition, ds, ids, b.loc[ids], truth, critical, v6._cluster_ids(split, ids), compat, mapped, vendor)


def _dev_fit_ids(ctx: v6.StageContext) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """(ids frame, B indexed by sample_id, outer folds) of the acdc_dev fit rows (critical QC removed; v6 rows)."""

    ids, _, b, folds = v6._dev_matrices(ctx)
    ids = ids.copy()
    ids["sample_id"] = ids["sample_id"].astype(str)
    b = b.copy()
    b["sample_id"] = b["sample_id"].astype(str)
    return ids, b.set_index("sample_id"), folds


# ----------------------------------------------------------------------------- representations (B-hat) and raw A


def _transition(path: Path):
    from ..transition_fit import TransitionModel

    return TransitionModel.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _indexed(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    return frame.set_index("sample_id")


def f9b_dir(ctx: v6.StageContext, seed: int) -> Path:
    return ctx.require_run("l57").path / str(_probes(ctx)["inputs"]["f9b"]["dir"]) / f"seed{int(seed)}"


def representation_matrix(ctx: v6.StageContext, representation: str, partition: str, *, seed: int | None = None) -> pd.DataFrame:
    """Raw A of a representation on a partition (index sample_id; columns in fitting order)."""

    run = ctx.require_run("l57")
    if representation == "descriptors":
        from ..formal_features import formal_descriptor_names

        ds = _partition(ctx, partition)["dataset"]
        return _indexed(pd.read_parquet(run.path / "formal" / f"A_{ds}.parquet"))[formal_descriptor_names()]
    if representation == "d20":
        prefix = str(_probes(ctx)["inputs"]["d20"]["feature_prefix"])
        frame = _indexed(pd.read_parquet(_input_path(ctx, "d20", "features", partition=partition)))
        return frame[activation_columns(frame.reset_index(), prefix)]
    if representation == "f9b":
        f9b = _probes(ctx)["f9b"]
        seed = int(seed if seed is not None else _seeds(ctx)[0])
        path = f9_path(ctx, "activations", seed)
        frame = read_c4(path, model_id=str(f9b["source_model_id"]), partition=partition, seed=seed, encoder="final")
        if frame.empty:
            raise FileNotFoundError(f"no final-encoder activations of {partition} (seed {seed}) in {path}")
        cols = activation_columns(frame, str(f9b["activation_prefix"]))
        return frame.set_index("sample_id")[cols]
    raise ValueError(f"unknown representation {representation!r}")


def representation_transition_path(ctx: v6.StageContext, representation: str, *, seed: int | None = None) -> Path:
    if representation == "descriptors":
        return _input_path(ctx, "descriptors", "transition")
    if representation == "d20":
        return _input_path(ctx, "d20", "transition")
    if representation == "f9b":
        return f9b_dir(ctx, seed if seed is not None else _seeds(ctx)[0]) / "transition_final.json"
    raise ValueError(f"unknown representation {representation!r}")


def representation_bhat(ctx: v6.StageContext, representation: str, partition: str, *, seed: int | None = None) -> pd.DataFrame:
    """B-hat of a representation (index sample_id): out-of-fold on acdc_dev, the frozen final T elsewhere."""

    targets = ctx.study.transition_targets()
    if partition == "acdc_dev":
        path = {"descriptors": _input_path(ctx, "descriptors", "bhat_oof_dev"), "d20": _input_path(ctx, "d20", "bhat_oof_dev"),
                "f9b": f9b_dir(ctx, seed if seed is not None else _seeds(ctx)[0]) / "bhat_oof_ACDC_dev.parquet"}[representation]
        return _indexed(pd.read_parquet(path))[targets]
    if representation == "descriptors":
        return _indexed(pd.read_parquet(_input_path(ctx, "descriptors", "bhat", partition=partition)))[targets]
    x = representation_matrix(ctx, representation, partition, seed=seed)
    pred = _transition(representation_transition_path(ctx, representation, seed=seed)).predict(x.to_numpy(dtype=np.float64))
    pred.index = x.index
    return pred[targets]


# ----------------------------------------------------------------------------- predictors (probabilities and logits)


def _c4_triplet(frame: pd.DataFrame, classes: list[str], path: Path, run: RunRoot) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    proba = c4_probabilities(frame, classes)
    logits, source = c4_logits(frame, classes)
    if "y_pred" in frame.columns and frame["y_pred"].notna().any():
        derived = argmax_labels(proba)
        recorded = frame["y_pred"].astype(object).to_numpy()
        bad = [i for i in range(len(frame)) if derived[i] is not None and recorded[i] is not None and str(recorded[i]) != str(derived[i])]
        if bad:
            raise v6.BundleError(f"{run.portable(path)}: y_pred disagrees with the argmax of prob_<CLASS> on {len(bad)} rows")
    return proba, logits, {"source": run.portable(path), "sha256": hash_path(path), "logits": source, "n_rows": int(len(frame))}


def _audited_derived(ctx: v6.StageContext, partition: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Audited-predictor probabilities without a C4 table: acdc_dev out-of-fold from the fold refits of ``fit-predictor``
    (same function, rows and folds; must reproduce the recorded cross-fitted decisions), elsewhere the hash-verified frozen
    predictor on the recorded formal A (must reproduce the recorded d_model)."""

    import joblib

    from ..classification import fit_formal_predictor
    from ..formal_features import formal_descriptor_names
    from .trust.calibration import logits_from_probabilities

    run, study, classes = ctx.require_run("l57"), ctx.study, _classes(ctx)
    names = formal_descriptor_names()
    ds = _partition(ctx, partition)["dataset"]
    if partition == "acdc_dev":
        split = v6._split_manifest(ctx)
        dev = split[split["partition"] == "acdc_dev"][["sample_id", "outer_fold"]]
        a = pd.read_parquet(run.path / "formal" / "A_ACDC.parquet").merge(dev, on="sample_id").sort_values("sample_id").reset_index(drop=True)
        cohort = v6._cohort(ctx, "ACDC").set_index("sample_id")
        y = cohort.loc[a["sample_id"], "acdc_class"].astype(str).reset_index(drop=True)
        x = a[names].to_numpy(dtype=np.float64)
        b = pd.read_parquet(run.path / "semantics" / "B_ACDC.parquet").set_index("sample_id")
        critical_ids = set(b.index[v6.critical_qc_mask(b, study).to_numpy()])
        finite = np.isfinite(x).all(axis=1)
        fit_rows = finite & ~a["sample_id"].isin(critical_ids).to_numpy()
        folds = a["outer_fold"].to_numpy(dtype=int)
        proba = np.full((len(a), len(classes)), np.nan)
        logits = np.full((len(a), len(classes)), np.nan)
        logit_source = "decision_function"
        for fold in sorted(set(folds.tolist())):
            held = np.flatnonzero(folds == fold)
            model = fit_formal_predictor(x, y, study.formal["audited_predictor"], fit_mask=fit_rows & (folds != fold))
            p = np.array(model.predict_proba(x[held]).reindex(columns=classes, fill_value=0.0), dtype=float)
            p[~finite[held]] = np.nan
            proba[held] = p
            ok = held[finite[held]]
            if model.missing_classes:
                logit_source = "log_probabilities (a fold predictor lacks a class)"
                logits[ok] = logits_from_probabilities(proba[ok])
            elif ok.size:
                logits[ok] = model.decision_function(x[ok])
        index = a["sample_id"].astype(str).to_numpy()
        proba_df = pd.DataFrame(proba, index=index, columns=classes)
        recorded = pd.read_parquet(run.path / "formal" / "d_model_crossfit_ACDC_dev.parquet").set_index("sample_id")["d_model_crossfit"]
        derived = pd.Series(argmax_labels(proba_df), index=index)
        bad = [s for s in index if s in recorded.index and recorded[s] is not None and pd.notna(recorded[s]) and derived[s] is not None
               and str(recorded[s]) != str(derived[s])]
        if bad:
            raise v6.BundleError(f"audited predictor: the fold refits do not reproduce the recorded cross-fitted d_model on {len(bad)} rows")
        return proba_df, pd.DataFrame(logits, index=index, columns=classes), {
            "source": "derived: fold refits of fit_formal_predictor (fit-predictor rows and folds), checked against formal/d_model_crossfit_ACDC_dev.parquet",
            "logits": logit_source, "n_rows": int(len(index))}
    predictor = joblib.load(_input_path(ctx, "descriptors", "predictor"))
    decisions = _indexed(pd.read_parquet(_input_path(ctx, "descriptors", "decisions", partition=partition)))["d_model"]
    a = _indexed(pd.read_parquet(run.path / "formal" / f"A_{ds}.parquet"))
    ids = [s for s in decisions.index if s in a.index]
    x = a.loc[ids, names].to_numpy(dtype=np.float64)
    proba = v6._predictor_probabilities(predictor, x).reindex(columns=classes, fill_value=0.0)
    finite = np.isfinite(x).all(axis=1)
    proba.loc[~finite] = np.nan
    proba.index = ids
    logits = pd.DataFrame(np.nan, index=ids, columns=classes)
    if finite.any():
        if list(predictor.classes) == classes:
            logits.loc[np.asarray(ids)[finite]] = predictor.decision_function(x[finite])
        else:
            logits.loc[np.asarray(ids)[finite]] = logits_from_probabilities(proba.loc[np.asarray(ids)[finite]].to_numpy())
    derived = pd.Series(argmax_labels(proba), index=ids)
    bad = [s for s in ids if derived[s] is not None and decisions[s] is not None and pd.notna(decisions[s]) and str(decisions[s]) != str(derived[s])]
    if bad:
        raise v6.BundleError(f"audited predictor: the frozen predictor does not reproduce the recorded d_model of {partition} on {len(bad)} rows")
    return proba, logits, {"source": f"derived: {run.portable(_input_path(ctx, 'descriptors', 'predictor'))} on formal/A_{ds}.parquet, "
                                     f"checked against {run.portable(_input_path(ctx, 'descriptors', 'decisions', partition=partition))}",
                           "logits": "decision_function", "n_rows": int(len(ids))}


def predictor_outputs(ctx: v6.StageContext, predictor: str, partition: str, *, seed: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """(probabilities, logits, provenance) of a predictor on a partition, index sample_id, columns the registry classes.

    ``acdc_dev``: out-of-fold rows (``is_oof``); elsewhere the final model.  Raises :class:`FileNotFoundError` when the
    predictions are not recorded."""

    run, classes = ctx.require_run("l57"), _classes(ctx)
    oof = partition == "acdc_dev"
    encoder = "oof" if oof else "final"
    if predictor == "audited_predictor":
        path = _input_path(ctx, "descriptors", "predictions")
        if path.is_file():
            frame = read_c4(path, model_id="audited_predictor", classes=classes, partition=partition, encoder=encoder)
            if len(frame):
                return _c4_triplet(frame, classes, path, run)
        return _audited_derived(ctx, partition)
    if predictor == "d20_predictor":
        path = _input_path(ctx, "d20", "predictions")
        if path.is_file():
            frame = read_c4(path, model_id="d20_predictor", classes=classes, partition=partition, encoder=encoder)
            if len(frame):
                return _c4_triplet(frame, classes, path, run)
        model_path = _input_path(ctx, "d20", "predictor")
        if not oof and model_path.is_file() and _input_path(ctx, "d20", "features", partition=partition).is_file():
            import joblib

            from .trust.calibration import logits_from_probabilities

            model = joblib.load(model_path)
            x = representation_matrix(ctx, "d20", partition)
            proba = model.predict_proba(x.to_numpy(dtype=np.float64)).reindex(columns=classes, fill_value=0.0)
            proba.index = x.index
            proba.loc[~np.isfinite(x.to_numpy(dtype=float)).all(axis=1)] = np.nan
            return proba, pd.DataFrame(logits_from_probabilities(proba.to_numpy()), index=proba.index, columns=classes), {
                "source": f"derived: {run.portable(model_path)} on {run.portable(_input_path(ctx, 'd20', 'features', partition=partition))}",
                "logits": "log_probabilities", "n_rows": int(len(proba))}
        raise FileNotFoundError(f"d20_predictor predictions of {partition} are not recorded ({run.portable(path)})")
    if predictor == "f9":
        seed = int(seed if seed is not None else _seeds(ctx)[0])
        path = f9_path(ctx, "predictions", seed)
        if not path.is_file():
            raise FileNotFoundError(f"f9 predictions are not recorded ({run.portable(path)})")
        frame = read_c4(path, model_id="f9", classes=classes, partition=partition, seed=seed, encoder=encoder)
        if frame.empty:
            raise FileNotFoundError(f"f9 predictions of {partition} (seed {seed}) are not recorded in {run.portable(path)}")
        return _c4_triplet(frame, classes, path, run)
    raise ValueError(f"unknown predictor {predictor!r}")


def predictor_input_files(ctx: v6.StageContext, predictor: str, partition: str) -> list[Path]:
    """Files a predictor's outputs are read from (stage inputs; only those that exist)."""

    run = ctx.require_run("l57")
    ds = _partition(ctx, partition)["dataset"]
    if predictor == "audited_predictor":
        c4 = _input_path(ctx, "descriptors", "predictions")
        if c4.is_file():
            files = [c4]
        elif partition == "acdc_dev":
            files = [run.path / "formal" / "A_ACDC.parquet", _input_path(ctx, "descriptors", "crossfit_dev"), run.path / "semantics" / "B_ACDC.parquet"]
        else:
            files = [_input_path(ctx, "descriptors", "predictor"), _input_path(ctx, "descriptors", "decisions", partition=partition),
                     run.path / "formal" / f"A_{ds}.parquet"]
    elif predictor == "d20_predictor":
        files = [_input_path(ctx, "d20", "predictions"), _input_path(ctx, "d20", "predictor"), _input_path(ctx, "d20", "features", partition=partition)]
    else:
        files = f9_paths(ctx, "predictions")
    return [f for f in files if f.is_file()]


def representation_input_files(ctx: v6.StageContext, representation: str, partition: str) -> list[Path]:
    run = ctx.require_run("l57")
    ds = _partition(ctx, partition)["dataset"]
    if representation == "descriptors":
        files = ([_input_path(ctx, "descriptors", "bhat_oof_dev")] if partition == "acdc_dev" else
                 [_input_path(ctx, "descriptors", "bhat", partition=partition), _input_path(ctx, "descriptors", "transition")]) + [
                     run.path / "formal" / f"A_{ds}.parquet"]
    elif representation == "d20":
        files = [_input_path(ctx, "d20", "bhat_oof_dev") if partition == "acdc_dev" else _input_path(ctx, "d20", "transition"),
                 _input_path(ctx, "d20", "features", partition=partition)]
    else:
        files = [f9b_dir(ctx, s) / name for s in _seeds(ctx) for name in ("bhat_oof_ACDC_dev.parquet", "transition_final.json")]
        files += f9_paths(ctx, "activations")
    return [f for f in files if f.is_file()]


# ----------------------------------------------------------------------------- shared metric blocks


def _rho_kwargs(ctx: v6.StageContext) -> dict[str, Any]:
    boot = _boot(ctx)
    return {"n_boot": boot["n_boot"], "seed": boot["seed"], "min_valid_fraction": boot["min_valid_fraction"],
            "min_estimable": int(ctx.study.bootstrap["primary_test"]["min_estimable_targets"])}


def rho_block(ctx: v6.StageContext, pdata: PartitionData, bhat: pd.DataFrame, rows: list[str], mean: pd.Series, sd: pd.Series,
              *, min_cases: int | None = None) -> dict[str, Any]:
    targets = ctx.study.transition_targets()
    rows = [s for s in rows if s in bhat.index and s in pdata.b.index]
    if not rows:
        return {"rho": None, "status": "NE", "reason": "no_rows_with_reconstructions", "n_cases": 0}
    out = v6._rho(ctx, pdata.b.loc[rows, targets], bhat.loc[rows, targets], mean, sd, clusters=pdata.clusters, min_cases=min_cases)
    n_clusters = int(pdata.clusters.reindex(rows).nunique())
    out["n_clusters_partition_rows"] = n_clusters
    if min_cases is not None and n_clusters < int(min_cases) and out.get("status") != "NE":
        out.update({"ci_lower": None, "ci_upper": None, "rho_j_ci": None, "status": "NE", "reason": "STRATUM_BELOW_MIN_CLUSTERS"})
    return out


def paired_rho_block(ctx: v6.StageContext, pdata: PartitionData, reference: pd.DataFrame, comparator: pd.DataFrame, rows: list[str],
                     mean: pd.Series, sd: pd.Series) -> dict[str, Any]:
    family = ctx.study.primary_family()
    rows = [s for s in rows if s in reference.index and s in comparator.index and s in pdata.b.index]
    if not rows:
        return {"difference": None, "status": "NE", "reason": "no_paired_rows"}
    null = pd.DataFrame([mean[family]] * len(rows), index=rows)
    return paired_rho_difference(pdata.b.loc[rows, family], reference.loc[rows, family], comparator.loc[rows, family], null, sd, family,
                                 cluster=[s.split(":", 1)[1] if ":" in s else s for s in rows],
                                 resample_cluster=pdata.clusters.reindex(rows).to_numpy(dtype=object), **_rho_kwargs(ctx))


def predictor_block(ctx: v6.StageContext, pdata: PartitionData, proba: pd.DataFrame, ids: list[str]) -> dict[str, Any]:
    boot = _boot(ctx)
    return predictor_metrics(pdata.truth.reindex(ids).to_numpy(dtype=object), proba.reindex(ids), _classes(ctx),
                             pdata.clusters.reindex(ids).to_numpy(dtype=object), n_boot=boot["n_boot"], seed=boot["seed"], n_bins=boot["n_bins"],
                             min_valid_fraction=boot["min_valid_fraction"], n_jobs=ctx.jobs)


def paired_predictor_block(ctx: v6.StageContext, pdata: PartitionData, reference: pd.DataFrame, comparator: pd.DataFrame, ids: list[str]) -> dict[str, Any]:
    boot = _boot(ctx)
    return paired_predictor_differences(pdata.truth.reindex(ids).to_numpy(dtype=object), reference.reindex(ids), comparator.reindex(ids), _classes(ctx),
                                        pdata.clusters.reindex(ids).to_numpy(dtype=object), n_boot=boot["n_boot"], seed=boot["seed"],
                                        n_bins=boot["n_bins"], min_valid_fraction=boot["min_valid_fraction"])


def non_equivalent_block(ctx: v6.StageContext, pdata: PartitionData, proba: pd.DataFrame, ids: list[str]) -> dict[str, Any] | None:
    """Decision distributions of the crosswalk groups other than the same-definition group (never pooled)."""

    if pdata.compat is None:
        return None
    same = str(_probes(ctx)["crosswalk"]["same_definition"])
    labels = pd.Series(argmax_labels(proba.reindex(ids)), index=ids, dtype=object)
    compat = pdata.compat.reindex(ids).astype(object).where(pdata.compat.reindex(ids).notna(), "<missing>").astype(str)
    out: dict[str, Any] = {}
    for group in sorted(set(compat) - {same}):
        members = [s for s in ids if compat[s] == group]
        mapped = pdata.mapped.reindex(members) if pdata.mapped is not None else pd.Series(None, index=members, dtype=object)
        entry: dict[str, Any] = {"n": len(members), "pooled": False, "decision_distribution": label_distribution(labels.reindex(members).to_numpy(dtype=object),
                                                                                                                  _classes(ctx)),
                                 "mapped_acdc_class": sorted({str(v) for v in mapped.dropna()})}
        has_map = [s for s in members if mapped.get(s) is not None and pd.notna(mapped.get(s))]
        if has_map:
            hits = int(sum(labels[s] is not None and str(labels[s]) == str(mapped[s]) for s in has_map))
            lo, hi = wilson_interval(hits, len(has_map))
            entry.update({"n_with_mapped_class": len(has_map), "n_decision_equals_mapped_class": hits, "proportion": hits / len(has_map),
                          "wilson_95": [lo, hi]})
        out[group] = entry
    return out


def _clean(value: Any) -> Any:
    """JSON-safe copy (NaN -> None, numpy scalars -> Python)."""

    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, _clean(payload))


# ============================================================================= fit-probes (D.11)


def fit_probes(ctx: v6.StageContext, *, seeds: list[int] | None = None):
    """F9b probes for every configured seed (primary first) on the acdc_dev fit rows; writes the F9b bundle manifest."""

    from .probes import REPRESENTATION, fit_f9b_probes

    run = ctx.require_run("fit-probes")
    pcfg = _probes(ctx)
    f9b = pcfg["f9b"]
    seeds = [int(s) for s in (seeds or f9b["seeds"])]
    n_folds = int(ctx.study.splits["acdc"]["acdc_dev"]["outer_folds"]["n_folds"])
    act_paths = {s: f9_path(ctx, "activations", s) for s in seeds}
    base = run.path / str(pcfg["inputs"]["f9b"]["dir"])
    per_seed = {s: [f9b_dir(ctx, s) / "bhat_oof_ACDC_dev.parquet", f9b_dir(ctx, s) / "transition_final.json", f9b_dir(ctx, s) / "lambda_selection.parquet",
                    *[f9b_dir(ctx, s) / f"transition_fold{k}.json" for k in range(1, n_folds + 1)]] for s in seeds}
    outputs = [p for s in seeds for p in per_seed[s]] + [base / "rho_oof_acdc_dev.json", base / "bundle.json"]
    contracts = run.path / "contracts"
    inputs = [*dict.fromkeys(act_paths.values()), run.path / "semantics" / "B_ACDC.parquet", run.path / "formal" / "A_ACDC.parquet",
              contracts / "split_manifest.parquet", contracts / "inner_folds.parquet", contracts / "cohort_ACDC.parquet"]

    def compute() -> dict[str, Any]:
        ids, b, folds = _dev_fit_ids(ctx)
        tables: dict[Path, pd.DataFrame] = {}
        targets = ctx.study.transition_targets()
        inner = v6._inner_fold_fn(ctx, ids)
        sd, mean = v6._dev_reference(ctx)
        pdata = partition_data(ctx, "acdc_dev")
        rho: dict[int, Any] = {}
        digests: dict[str, str] = {}
        for s in seeds:
            if act_paths[s] not in tables:
                tables[act_paths[s]] = pd.read_parquet(act_paths[s])
                tables[act_paths[s]]["sample_id"] = tables[act_paths[s]]["sample_id"].astype(str)
            acts_all = tables[act_paths[s]]
            sel = ((acts_all["model_id"].astype(str) == str(f9b["source_model_id"])) & (pd.to_numeric(acts_all["seed"], errors="coerce") == s)
                   & (acts_all["partition"].astype(str) == "acdc_dev"))
            acts = acts_all.loc[sel]
            cols = activation_columns(acts, str(f9b["activation_prefix"]))
            if len(cols) != int(f9b["dimension"]):
                raise ValueError(f"seed {s}: {len(cols)} activation columns, configs/probes.yaml f9b.dimension is {f9b['dimension']}")
            res = fit_f9b_probes(ids, b.reset_index(), acts, cols, targets, ctx.study.transition, ctx.study.registry, inner)
            out = f9b_dir(ctx, s)
            out.mkdir(parents=True, exist_ok=True)
            v6._write_table(res["oof"], per_seed[s][0])
            atomic_write_json(per_seed[s][1], res["final"].to_dict(), allow_nan=True)
            v6._write_table(res["lambda_selection"], per_seed[s][2])
            for k, model in sorted(res["fold_models"].items()):
                atomic_write_json(out / f"transition_fold{k}.json", model.to_dict(), allow_nan=True)
            oof = res["oof"].set_index("sample_id")
            rho[s] = rho_block(ctx, pdata, oof, ids["sample_id"].tolist(), mean, sd)
            digests[str(s)] = _rows_digest(acts, cols)
        primary = seeds[0]
        _write_json(base / "rho_oof_acdc_dev.json", {
            "schema": "tmmri.f9b_rho_oof/1", "representation": REPRESENTATION, "partition": "acdc_dev", "evidence": _partition(ctx, "acdc_dev")["label"],
            "primary_seed": primary, "seeds": {str(s): rho[s] for s in seeds},
            "seed_spread": {"rho": seed_spread({s: rho[s].get("rho") for s in seeds})},
            "null": "acdc_dev fit-partition mean (v6 estimand)", "normalization": "acdc_dev fit-partition SD",
            "nesting": "per outer fold k: T fitted on encoder k's activations of its fitting patients, decoding encoder k's activations of fold k"})
        files = {rel_path(run, p): hash_path(p) for s in seeds for p in per_seed[s] if p.suffix == ".json"}
        code = code_version()
        _write_json(base / "bundle.json", {
            "schema": "tmmri.f9b_bundle/1", "representation": REPRESENTATION, "seeds": seeds, "primary_seed": primary, "bundle": files,
            "bundle_sha256": sha256_json(files), "dev_activation_rows_sha256": digests,
            "activations_source": {str(s): run.portable(act_paths[s]) for s in seeds},
            "pool_latent_across_encoders": False, "code": {"commit": code.get("commit"), "code_sha256": code.get("code_sha256")},
            "note": "per-partition activations are local inference of the frozen F9 encoders (neural_f9 freeze record); they are hashed as stage inputs"})
        return {"seeds": seeds, "rho_oof": {str(s): rho[s].get("rho") for s in seeds}, "bundle_sha256": sha256_json(files)}

    return _lane_stage(ctx, "fit-probes", "fit-probes", compute, inputs, outputs, {"seeds": seeds})


def _rows_digest(frame: pd.DataFrame, cols: list[str]) -> str:
    f = frame.sort_values(["encoder_id", "sample_id"])
    h = hashlib.sha256()
    h.update("\n".join(f["encoder_id"].astype(str) + "|" + f["sample_id"].astype(str)).encode("utf-8"))
    h.update(np.ascontiguousarray(f[cols].to_numpy(dtype=np.float64)).tobytes())
    return h.hexdigest()


# ============================================================================= evaluate-alt-bundles (D.12-D.13)


def evaluation_path(ctx: v6.StageContext, bundle: str, partition: str) -> Path:
    return ctx.require_run("l57").path / "probes" / "evaluations" / bundle / f"{partition}.json"


def evaluate_alt_bundles(ctx: v6.StageContext, split: str, *, bundles: list[str] | None = None) -> dict[str, Any]:
    """Evaluate the alternative bundles once on ``split`` (one once-only stage per bundle and partition)."""

    info = _partition(ctx, split)
    configured = list(_probes(ctx)["bundles"])
    bundles = list(bundles or configured)
    unknown = sorted(set(bundles) - set(configured))
    if unknown:
        raise ValueError(f"unknown bundles {unknown}; configs/probes.yaml bundles: {configured}")
    guard_partition(ctx, split, f"evaluate-alt-bundles --split {split}")
    run = ctx.require_run("evaluate-alt-bundles")
    frozen = v6.verify_frozen_bundle(ctx)
    if info["held_out"]:
        v6._require_evaluated(run, f"evaluate:{split}", "evaluate-alt-bundles")
    return {b: _evaluate_one_bundle(ctx, b, split, frozen) for b in bundles}


def _freeze_records(ctx: v6.StageContext, bundle: str) -> list[str]:
    inputs = _probes(ctx)["inputs"]
    return {"d20": [inputs["d20"]["freeze_record"]], "f9": [inputs["f9"]["freeze_record"]],
            "f9b": [inputs["f9b"]["freeze_record"], inputs["f9"]["freeze_record"]]}[bundle]


def _evaluate_one_bundle(ctx: v6.StageContext, bundle: str, split: str, frozen: dict[str, Any]):
    run = ctx.require_run("evaluate-alt-bundles")
    pcfg = _probes(ctx)
    info, order = partition_evidence(ctx, split, {"d20": ["D20"], "f9": ["F9"], "f9b": ["F9b"]}[bundle])
    spec = pcfg["bundles"][bundle]
    reps, preds = list(spec.get("representations", [])), list(spec.get("predictors", []))
    freezes = []
    if info["held_out"] and pcfg.get("require_freeze_record_for_held_out", True):
        freezes = [verify_hash_manifest(run, rel, f"evaluate-alt-bundles {bundle}") for rel in _freeze_records(ctx, bundle)]
    ds = info["dataset"]
    contracts = run.path / "contracts"
    inputs = [run.frozen_path, *split_inputs(run, split), contracts / f"cohort_{ds}.parquet", run.path / "semantics" / f"B_{ds}.parquet"]
    if info["held_out"]:
        inputs.append(run.record_path(f"evaluate:{split}"))
        record = run.read_record(f"evaluate:{split}")
        v6._check_recorded_output(run, record, _input_path(ctx, "descriptors", "bhat", partition=split), "evaluate-alt-bundles")
        v6._check_recorded_output(run, record, _input_path(ctx, "descriptors", "decisions", partition=split), "evaluate-alt-bundles")
    inputs += [run.path / f["record"] for f in freezes]
    inputs += representation_input_files(ctx, "descriptors", split) + predictor_input_files(ctx, "audited_predictor", split)
    for r in reps:
        files = representation_input_files(ctx, r, split)
        if not files:
            raise FileNotFoundError(f"evaluate-alt-bundles {bundle}: no recorded inputs of representation {r} on {split}")
        inputs += files
    for p in preds:
        files = predictor_input_files(ctx, p, split)
        if not files:
            raise FileNotFoundError(f"evaluate-alt-bundles {bundle}: no recorded predictions of {p} on {split}")
        inputs += files
    inputs = list(dict.fromkeys(inputs))
    target = evaluation_path(ctx, bundle, split)

    def compute() -> dict[str, Any]:
        sd, mean = v6._dev_reference(ctx)
        pdata = partition_data(ctx, split)
        rows_rho = _rho_rows(ctx, pdata)
        ids_pred = pdata.eval_ids
        desc_bhat = representation_bhat(ctx, "descriptors", split)
        aud_proba, _, aud_prov = predictor_outputs(ctx, "audited_predictor", split)
        payload: dict[str, Any] = {
            "schema": "tmmri.alt_bundle_evaluation/1", "bundle": bundle, "partition": split, "dataset": pdata.dataset, "evidence": info["evidence"],
            "label": info["label"], "freeze_order": order, "descriptor_bundle_sha256": frozen["bundle_sha256"], "alt_bundle_freeze": freezes,
            "estimation_only": True, "classes": _classes(ctx), "bootstrap": _boot(ctx),
            "rows": {"n_partition_rows": len(pdata.ids), "n_rho_rows": len(rows_rho), "n_predictor_rows": len(ids_pred),
                     "n_excluded_critical_qc": int(pdata.critical.sum()),
                     "class_reference": "acdc_class" if pdata.dataset == "ACDC" else "crosswalk_same_definition_only"},
            "descriptor_pipeline": {"representation": {"name": "descriptors", "rho": rho_block(ctx, pdata, desc_bhat, rows_rho, mean, sd)},
                                    "predictor": {"name": "audited_predictor", "provenance": aud_prov,
                                                  "metrics": predictor_block(ctx, pdata, aud_proba, ids_pred),
                                                  "non_equivalent": non_equivalent_block(ctx, pdata, aud_proba, ids_pred)}},
            "representations": {}, "predictors": {}}
        for r in reps:
            seeds = _seeds(ctx) if r == "f9b" else [None]
            entry: dict[str, Any] = {"seeds": {}, "primary_seed": seeds[0]}
            for s in seeds:
                bhat = representation_bhat(ctx, r, split, seed=s)
                entry["seeds"][str(s if s is not None else -1)] = {
                    "rho": rho_block(ctx, pdata, bhat, rows_rho, mean, sd),
                    "paired_vs_descriptors": paired_rho_block(ctx, pdata, desc_bhat, bhat, rows_rho, mean, sd)}
            if len(seeds) > 1:
                entry["seed_spread"] = {"rho": seed_spread({s: entry["seeds"][str(s)]["rho"].get("rho") for s in seeds})}
            payload["representations"][r] = entry
        for p in preds:
            seeds = _seeds(ctx) if p == "f9" else [None]
            entry = {"seeds": {}, "primary_seed": seeds[0]}
            for s in seeds:
                proba, _, prov = predictor_outputs(ctx, p, split, seed=s)
                entry["seeds"][str(s if s is not None else -1)] = {
                    "provenance": prov, "metrics": predictor_block(ctx, pdata, proba, ids_pred),
                    "paired_vs_audited_predictor": paired_predictor_block(ctx, pdata, aud_proba, proba, ids_pred),
                    "non_equivalent": non_equivalent_block(ctx, pdata, proba, ids_pred)}
            if len(seeds) > 1:
                entry["seed_spread"] = {m: seed_spread({s: (entry["seeds"][str(s)]["metrics"].get("metrics", {}).get(m) or {}).get("estimate")
                                                        for s in seeds})
                                        for m in ("accuracy", "macro_f1", "macro_ovr_auroc", "ece", "brier", "log_loss")}
            payload["predictors"][p] = entry
        _write_json(target, payload)
        return {"bundle": bundle, "partition": split, "evidence": info["evidence"]}

    return _lane_stage(ctx, "evaluate-alt-bundles", f"evaluate-alt-bundles:{bundle}:{split}", compute, inputs, [target],
                       {"bundle": bundle, "split": split, "evidence": info["evidence"]}, status="EVALUATED")


def _rho_rows(ctx: v6.StageContext, pdata: PartitionData) -> list[str]:
    """Rows of the representation estimand: acdc_dev fit rows (as the v6 out-of-fold ledger); elsewhere every partition
    row with recorded formal A and B (as ``evaluate``)."""

    if pdata.partition == "acdc_dev":
        ids, _, _ = _dev_fit_ids(ctx)
        return ids["sample_id"].tolist()
    a = _indexed(pd.read_parquet(ctx.require_run("l57").path / "formal" / f"A_{pdata.dataset}.parquet"))
    return [s for s in pdata.ids if s in a.index]


# ============================================================================= fit-trust (S1, S2, S4)


def trust_dir(ctx: v6.StageContext) -> Path:
    return ctx.require_run("l57").path / "trust"


def has_predictor(ctx: v6.StageContext, predictor: str, partition: str) -> bool:
    """True when the predictor's outputs on ``partition`` are recorded (f9: the primary seed)."""

    try:
        predictor_outputs(ctx, predictor, partition, seed=_seeds(ctx)[0] if predictor == "f9" else None)
        return True
    except FileNotFoundError:
        return False


def has_representation(ctx: v6.StageContext, representation: str, partition: str) -> bool:
    """True when the representation's raw A (and, off acdc_dev, its frozen T) are recorded on ``partition``."""

    seed = _seeds(ctx)[0] if representation == "f9b" else None
    try:
        representation_matrix(ctx, representation, partition, seed=seed)
        if partition != "acdc_dev" and not representation_transition_path(ctx, representation, seed=seed).is_file():
            return False
        return True
    except FileNotFoundError:
        return False


def fit_trust(ctx: v6.StageContext, *, allow_missing: bool = False):
    """S1 thresholds, S2 references and flag thresholds, S4 temperatures from acdc_dev out-of-fold outputs only."""

    from .trust.calibration import fit_temperature
    from .trust.ood import fit_mahalanobis_reference, flag_threshold
    from .trust.selective import coverage_threshold, max_softmax

    run = ctx.require_run("fit-trust")
    tcfg = _trust(ctx)
    predictors = [str(p) for p in tcfg["predictors"]]
    representations = [str(r) for r in tcfg["representations"]]
    have_pred = {p: has_predictor(ctx, p, "acdc_dev") for p in predictors}
    have_rep = {r: has_representation(ctx, r, "acdc_dev") for r in representations}
    missing = [p for p, ok in have_pred.items() if not ok] + [r for r, ok in have_rep.items() if not ok]
    if missing and not allow_missing:
        raise FileNotFoundError(f"fit-trust: acdc_dev out-of-fold inputs not recorded for {missing}; record them first or pass --allow-missing "
                                "(those analyses are then reported NE, not executed)")
    tdir = trust_dir(ctx)
    ref_paths = {r: tdir / "s2_references" / f"{r}.json" for r, ok in have_rep.items() if ok}
    outputs = [tdir / "s1_thresholds.json", tdir / "s2_oof_scores.parquet", tdir / "s4_temperatures.json", tdir / "oof_predictions.parquet",
               *ref_paths.values(), tdir / "trust_bundle.json"]
    contracts = run.path / "contracts"
    inputs = [contracts / "split_manifest.parquet", contracts / "cohort_ACDC.parquet", run.path / "semantics" / "B_ACDC.parquet",
              run.path / "formal" / "A_ACDC.parquet"]
    for p, ok in have_pred.items():
        inputs += predictor_input_files(ctx, p, "acdc_dev") if ok else []
    if have_rep.get("d20"):
        inputs.append(_input_path(ctx, "d20", "features", partition="acdc_dev"))
    if have_rep.get("f9b"):
        inputs.append(f9_path(ctx, "activations", _seeds(ctx)[0]))
    inputs = list(dict.fromkeys(inputs))

    def compute() -> dict[str, Any]:
        classes = _classes(ctx)
        pdata = partition_data(ctx, "acdc_dev")
        ids_frame, _, folds = _dev_fit_ids(ctx)
        dev_ids = ids_frame["sample_id"].tolist()
        fold_of = dict(zip(dev_ids, folds.tolist()))
        s1cfg, s2cfg, s4cfg = tcfg["s1_selective_prediction"], tcfg["s2_ood"], tcfg["s4_calibration"]
        thresholds: dict[str, Any] = {}
        temperatures: dict[str, Any] = {}
        oof_rows: list[pd.DataFrame] = []
        for p in predictors:
            if not have_pred[p]:
                thresholds[p] = temperatures[p] = {"status": "NE", "reason": NOT_RECORDED}
                continue
            proba, logits, prov = predictor_outputs(ctx, p, "acdc_dev", seed=_seeds(ctx)[0] if p == "f9" else None)
            rows = [s for s in pdata.eval_ids if s in proba.index]
            conf = max_softmax(proba.reindex(rows).to_numpy(dtype=float))
            thresholds[p] = {"provenance": prov, "confidence": "max_softmax", "n_oof_rows": len(rows),
                             **{name: coverage_threshold(conf, float(c)) for name, c in s1cfg["coverages"].items()}}
            truth = pdata.truth.reindex(rows)
            known = [i for i, s in enumerate(rows) if truth[s] is not None and str(truth[s]) in classes]
            z = logits.reindex(rows).to_numpy(dtype=float)[known]
            y = np.array([classes.index(str(truth[rows[i]])) for i in known], dtype=int)
            bounds = tuple(float(v) for v in s4cfg["temperature"]["bounds"])
            temperatures[p] = {"provenance": prov, "logits": prov.get("logits"),
                               **fit_temperature(z, y, bounds=bounds, xatol=float(s4cfg["temperature"].get("xatol", 1e-10)))}
            frame = proba.reindex(rows).add_prefix("prob_").join(logits.reindex(rows).add_prefix("logit_"))
            frame.insert(0, "sample_id", rows)
            frame.insert(1, "cluster_id", pdata.clusters.reindex(rows).to_numpy(dtype=object))
            frame.insert(2, "partition", "acdc_dev")
            frame.insert(3, "model_id", p)
            frame.insert(4, "seed", int(_seeds(ctx)[0]) if p == "f9" else -1)
            frame.insert(5, "encoder_id", [f"fold{fold_of.get(s, 0)}" if s in fold_of else None for s in rows])
            frame.insert(6, "is_oof", True)
            frame.insert(7, "y_true", truth.to_numpy(dtype=object))
            frame.insert(8, "y_pred", argmax_labels(proba.reindex(rows)))
            oof_rows.append(frame.reset_index(drop=True))
        refs: dict[str, Any] = {}
        score_rows: list[pd.DataFrame] = []
        quantile, method = float(s2cfg["flag_quantile"]), str(s2cfg.get("quantile_method", "linear"))
        for r in representations:
            if not have_rep[r]:
                refs[r] = {"status": "NE", "reason": NOT_RECORDED}
                continue
            k_pca = (s2cfg.get("pca_components") or {}).get(r)
            scores = pd.Series(np.nan, index=dev_ids, dtype=float)
            fold_refs = {}
            for k in sorted(set(folds.tolist())):
                train = [s for s in dev_ids if fold_of[s] != k]
                held = [s for s in dev_ids if fold_of[s] == k]
                x_train, x_held = _s2_fold_matrices(ctx, r, k, train, held)
                ref = fit_mahalanobis_reference(x_train, pca_components=k_pca)
                scores.loc[held] = ref.score(x_held)
                fold_refs[str(k)] = {"n_fit": ref.n_fit, "shrinkage": ref.shrinkage, "dimension": ref.dimension}
            x_final = _s2_final_matrix(ctx, r, "acdc_dev").reindex(dev_ids).to_numpy(dtype=float)
            final = fit_mahalanobis_reference(x_final, pca_components=k_pca)
            tau = flag_threshold(scores.to_numpy(), quantile, method)
            _write_json(ref_paths[r], {"schema": "tmmri.s2_reference/1", "representation": r, "reference": final.to_dict(), "flag_threshold": tau,
                                       "flag_quantile": quantile, "quantile_method": method, "pca_components": k_pca, "fold_references": fold_refs,
                                       "oof_rule": s2cfg["out_of_fold"]})
            refs[r] = {"reference": run.portable(ref_paths[r]), "flag_threshold": tau, "n_oof_scores": int(scores.notna().sum()),
                       "oof_flag_rate": float(np.mean(scores.dropna().to_numpy() > tau)) if scores.notna().any() else None,
                       "pca_components": k_pca, "dimension": final.dimension, "shrinkage_final": final.shrinkage, "status": "FITTED"}
            score_rows.append(pd.DataFrame({"sample_id": dev_ids, "representation": r, "outer_fold": [fold_of[s] for s in dev_ids],
                                            "score": scores.to_numpy(dtype=float)}))
        _write_json(tdir / "s1_thresholds.json", {"schema": "tmmri.s1_thresholds/1", "fit_partition": "acdc_dev (out-of-fold)",
                                                   "rule": s1cfg["threshold_rule"], "predictors": thresholds})
        _write_json(tdir / "s4_temperatures.json", {"schema": "tmmri.s4_temperatures/1", "fit_partition": "acdc_dev (out-of-fold logits)",
                                                     "objective": s4cfg["temperature"]["objective"], "predictors": temperatures})
        v6._write_table(pd.concat(score_rows, ignore_index=True) if score_rows else
                        pd.DataFrame(columns=["sample_id", "representation", "outer_fold", "score"]), tdir / "s2_oof_scores.parquet")
        v6._write_table(pd.concat(oof_rows, ignore_index=True) if oof_rows else pd.DataFrame(columns=["sample_id"]), tdir / "oof_predictions.parquet")
        files = {rel_path(run, p): hash_path(p) for p in outputs if p.name != "trust_bundle.json"}
        _write_json(tdir / "trust_bundle.json", {"schema": "tmmri.trust_bundle/1", "bundle": files, "bundle_sha256": sha256_json(files),
                                                 "predictors": {p: ("FITTED" if have_pred[p] else "NE") for p in predictors},
                                                 "representations": {r: ("FITTED" if have_rep[r] else "NE") for r in representations},
                                                 "s2_summary": refs, "fit_partition": "acdc_dev out-of-fold outputs only"})
        return {"trust_bundle_sha256": sha256_json(files), "not_executed": missing}

    return _lane_stage(ctx, "fit-trust", "fit-trust", compute, inputs, outputs, {"allow_missing": bool(allow_missing), "not_executed": missing})


def _s2_fold_matrices(ctx: v6.StageContext, representation: str, fold: int, train: list[str], held: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """S2 out-of-fold matrices: descriptors and D20 share one coordinate system; F9b uses encoder ``fold<k>`` only."""

    if representation == "f9b":
        from .probes import encoder_matrix

        f9b = _probes(ctx)["f9b"]
        acts = read_c4(f9_path(ctx, "activations", _seeds(ctx)[0]), model_id=str(f9b["source_model_id"]), partition="acdc_dev", seed=_seeds(ctx)[0],
                       encoder=f"fold{fold}")
        cols = activation_columns(acts, str(f9b["activation_prefix"]))
        return encoder_matrix(acts, f"fold{fold}", train, cols), encoder_matrix(acts, f"fold{fold}", held, cols)
    x = _s2_final_matrix(ctx, representation, "acdc_dev")
    return x.reindex(train).to_numpy(dtype=float), x.reindex(held).to_numpy(dtype=float)


def _s2_final_matrix(ctx: v6.StageContext, representation: str, partition: str) -> pd.DataFrame:
    return representation_matrix(ctx, representation, partition, seed=_seeds(ctx)[0] if representation == "f9b" else None)


# ============================================================================= evaluate-trust (S1-S4)


def evaluate_trust(ctx: v6.StageContext, split: str, *, allow_missing: bool = False):
    """S1-S4 once on ``split`` with the frozen trust bundle (never fits a threshold, reference or temperature)."""

    from .trust.calibration import ece_equal_width, reliability_table, scaled_probabilities
    from .trust.harmonize import VARIANTS, apply_harmonized, fit_harmonization
    from .trust.ood import MahalanobisReference
    from .trust.selective import aurc, max_softmax, realised_at_threshold, risk_coverage_curve

    tcfg = _trust(ctx)
    if split not in [str(p) for p in tcfg["evaluation_partitions"]]:
        raise ValueError(f"evaluate-trust --split must be one of {tcfg['evaluation_partitions']}")
    info = _partition(ctx, split)
    guard_partition(ctx, split, f"evaluate-trust --split {split}")
    run = ctx.require_run("evaluate-trust")
    frozen = v6.verify_frozen_bundle(ctx)
    v6._require_evaluated(run, f"evaluate:{split}", "evaluate-trust")
    tdir = trust_dir(ctx)
    trust_freeze = verify_hash_manifest(run, rel_path(run, tdir / "trust_bundle.json"), "evaluate-trust")
    tb = json.loads((tdir / "trust_bundle.json").read_text(encoding="utf-8"))
    predictors = [p for p in tcfg["predictors"] if tb["predictors"].get(p) == "FITTED"]
    representations = [r for r in tcfg["representations"] if tb["representations"].get(r) == "FITTED"]
    underlying = {"d20_predictor": "D20", "d20": "D20", "f9": "F9", "f9b": "F9b"}
    info, order = partition_evidence(ctx, split, ["S1-S4", *[underlying[x] for x in (*predictors, *representations) if x in underlying]])
    pairing = _probes(ctx)["pairing"]
    have_pred = {p: has_predictor(ctx, p, split) for p in predictors}
    have_rep = {r: has_representation(ctx, r, split) for r in representations}
    missing = [p for p, ok in have_pred.items() if not ok] + [r for r, ok in have_rep.items() if not ok]
    if missing and not allow_missing:
        raise FileNotFoundError(f"evaluate-trust {split}: inputs not recorded for {missing}; record them or pass --allow-missing (reported NE)")
    out_dir = tdir / split
    outputs = [out_dir / n for n in ("s1_selective.json", "s1_risk_coverage.parquet", "s2_ood.json", "s2_scores.parquet", "s3_harmonization.json",
                                     "s4_calibration.json", "s4_reliability.parquet")]
    ds = info["dataset"]
    contracts = run.path / "contracts"
    inputs = [run.frozen_path, run.record_path(f"evaluate:{split}"), tdir / "trust_bundle.json", *split_inputs(run, split),
              contracts / f"cohort_{ds}.parquet", run.path / "semantics" / f"B_{ds}.parquet", run.path / "semantics" / "B_ACDC.parquet"]
    inputs += [run.path / f for f in _manifest_files(tb) or {}]
    for p in predictors:
        inputs += predictor_input_files(ctx, p, split) if have_pred[p] else []
    for r in representations:
        inputs += representation_input_files(ctx, r, split) if have_rep[r] else []
    routes_file = run.path / str(tcfg["s1_selective_prediction"]["rule_routes_alongside"]["file"]).format(
        rulebook=tcfg["s1_selective_prediction"]["rule_routes_alongside"]["rulebook"], partition=split)
    if routes_file.is_file():
        inputs.append(routes_file)
    for extra in (_input_path(ctx, "d20", "predictor"),):
        if extra.is_file() and "d20" in representations:
            inputs.append(extra)
    inputs = list(dict.fromkeys(inputs))
    boot = _boot(ctx)
    min_clusters = int(tcfg["min_clusters"])
    classes = _classes(ctx)

    def compute() -> dict[str, Any]:
        sd, mean = v6._dev_reference(ctx)
        pdata = partition_data(ctx, split)
        eval_ids = pdata.eval_ids
        family = ctx.study.primary_family()
        base = {"partition": split, "evidence": info["evidence"], "label": info["label"], "freeze_order": order,
                "descriptor_bundle_sha256": frozen["bundle_sha256"], "trust_bundle": trust_freeze, "estimation_only": True}
        thresholds = json.loads((tdir / "s1_thresholds.json").read_text(encoding="utf-8"))["predictors"]
        temps = json.loads((tdir / "s4_temperatures.json").read_text(encoding="utf-8"))["predictors"]
        outputs_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]] = {}
        for p in predictors:
            if have_pred[p]:
                outputs_cache[p] = predictor_outputs(ctx, p, split, seed=_seeds(ctx)[0] if p == "f9" else None)

        def correct_vector(proba: pd.DataFrame, rows: list[str]) -> np.ndarray:
            labels = argmax_labels(proba.reindex(rows))
            truth = pdata.truth.reindex(rows)
            return np.array([np.nan if (t is None or str(t) not in classes or lab is None) else float(str(lab) == str(t))
                             for t, lab in zip(truth.to_numpy(dtype=object), labels)], dtype=float)

        # ---- S1 selective prediction
        s1: dict[str, Any] = {**base, "schema": "tmmri.s1_selective/1", "predictors": {}}
        curves: list[pd.DataFrame] = []
        for p in tcfg["predictors"]:
            if p not in outputs_cache:
                s1["predictors"][p] = {"status": "NE", "reason": NOT_RECORDED}
                continue
            proba = outputs_cache[p][0]
            conf = max_softmax(proba.reindex(eval_ids).to_numpy(dtype=float))
            corr = correct_vector(proba, eval_ids)
            ok = np.isfinite(conf) & np.isfinite(corr)
            rows_ok = np.flatnonzero(ok)
            cl = pdata.clusters.reindex(eval_ids).to_numpy(dtype=object)[ok]
            entry = {"provenance": outputs_cache[p][2], "n_evaluable": int(ok.sum()),
                     "aurc": bootstrap_statistic(lambda r, c=conf[rows_ok], y=corr[rows_ok]: aurc(c[r], y[r]), int(ok.sum()), cl, n_boot=boot["n_boot"],
                                                 seed=boot["seed"], min_valid_fraction=boot["min_valid_fraction"]) if ok.any() else
                     {"estimate": None, "status": "NE", "reason": "no_evaluable_cases"}}
            for name in tcfg["s1_selective_prediction"]["coverages"]:
                tau = (thresholds.get(p) or {}).get(name, {}).get("tau")
                entry[name] = {"development": thresholds.get(p, {}).get(name), "realised": realised_at_threshold(conf, corr, tau)}
            s1["predictors"][p] = entry
            curve = risk_coverage_curve(conf, corr)
            curve.insert(0, "predictor", p)
            curve.insert(1, "partition", split)
            curves.append(curve)
        routes = json.loads(routes_file.read_text(encoding="utf-8")) if routes_file.is_file() else None
        fields = tcfg["s1_selective_prediction"]["rule_routes_alongside"]["fields"]
        s1["rule_routes"] = ({"rulebook": tcfg["s1_selective_prediction"]["rule_routes_alongside"]["rulebook"], "source": run.portable(routes_file),
                              **{f: routes.get(f) for f in fields}} if routes is not None else {"status": "NE", "reason": f"{run.portable(routes_file)} not recorded"})
        _write_json(outputs[0], s1)
        v6._write_table(pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(columns=["predictor", "partition", "threshold", "n_accepted",
                                                                                                  "coverage", "risk", "n_errors"]), outputs[1])

        # ---- S2 label-free OOD flags
        s2: dict[str, Any] = {**base, "schema": "tmmri.s2_ood/1", "representations": {}, "min_clusters": min_clusters}
        score_frames: list[pd.DataFrame] = []
        for r in tcfg["representations"]:
            if r not in representations or not have_rep.get(r):
                s2["representations"][r] = {"status": "NE", "reason": NOT_RECORDED}
                continue
            ref_doc = json.loads((tdir / "s2_references" / f"{r}.json").read_text(encoding="utf-8"))
            ref = MahalanobisReference.from_dict(ref_doc["reference"])
            tau = float(ref_doc["flag_threshold"])
            x = _s2_final_matrix(ctx, r, split)
            rows = [s for s in pdata.ids if s in x.index]
            scores = pd.Series(ref.score(x.loc[rows].to_numpy(dtype=float)), index=rows)
            flagged = scores > tau
            finite = scores.notna()
            n, k = int(finite.sum()), int((flagged & finite).sum())
            lo, hi = wilson_interval(k, n) if n else (None, None)
            entry: dict[str, Any] = {"flag_threshold": tau, "n_scored": n, "n_flagged": k, "flag_rate": k / n if n else None, "flag_rate_wilson_95": [lo, hi]}
            vendors = pdata.vendor.reindex(rows).astype(object).where(pdata.vendor.reindex(rows).notna(), "<missing>").astype(str) \
                if pdata.vendor is not None else pd.Series("not_recorded", index=rows)
            by_vendor = {}
            for v in sorted(vendors.unique()):
                m = [s for s in rows if vendors[s] == v and finite[s]]
                kv = int(flagged.reindex(m).sum())
                vlo, vhi = wilson_interval(kv, len(m)) if m else (None, None)
                by_vendor[v] = {"n": len(m), "n_flagged": kv, "flag_rate": kv / len(m) if m else None, "wilson_95": [vlo, vhi],
                                "n_clusters": int(pdata.clusters.reindex(m).nunique()),
                                "descriptive_only": int(pdata.clusters.reindex(m).nunique()) < min_clusters}
            entry["by_vendor"] = by_vendor
            predictor = str(pairing[r])
            bhat = representation_bhat(ctx, r, split, seed=_seeds(ctx)[0] if r == "f9b" else None)
            nre = _normalized_reconstruction_error(pdata.b, bhat, family, sd).reindex(rows)
            if predictor in outputs_cache:
                corr = pd.Series(correct_vector(outputs_cache[predictor][0], rows), index=rows)
            else:
                corr = pd.Series(np.nan, index=rows)
            ev = [s for s in rows if s in eval_ids and finite[s] and np.isfinite(corr[s])]
            cl_ev = pdata.clusters.reindex(ev).to_numpy(dtype=object)
            mis = (1.0 - corr.reindex(ev)).to_numpy(dtype=float)
            sc = scores.reindex(ev).to_numpy(dtype=float)
            entry["misclassification_auroc"] = ({"predictor": predictor, "n": len(ev), "n_misclassified": int(mis.sum()),
                                                 **bootstrap_statistic(lambda r_, m=mis, s_=sc: _auroc_from_scores(m[r_].astype(int), s_[r_]), len(ev), cl_ev,
                                                                       n_boot=boot["n_boot"], seed=boot["seed"], min_valid_fraction=boot["min_valid_fraction"])}
                                                if ev else {"predictor": predictor, "status": "NE", "reason": "no_evaluable_cases"})
            sp = [s for s in rows if finite[s] and np.isfinite(nre[s])]
            entry["spearman_reconstruction_error"] = ({"n": len(sp), "error": "mean over primary-family targets of |Bhat - B| / acdc_dev fit SD",
                                                       **bootstrap_statistic(lambda r_, a=scores.reindex(sp).to_numpy(float), b_=nre.reindex(sp).to_numpy(float):
                                                                             _spearman(a[r_], b_[r_]), len(sp), pdata.clusters.reindex(sp).to_numpy(dtype=object),
                                                                             n_boot=boot["n_boot"], seed=boot["seed"], min_valid_fraction=boot["min_valid_fraction"])}
                                                      if len(sp) > 2 else {"status": "NE", "reason": "fewer_than_three_cases"})
            rho_rows = _rho_rows(ctx, pdata)
            groups = {}
            for name, members in (("flagged", [s for s in rows if finite[s] and flagged[s]]), ("unflagged", [s for s in rows if finite[s] and not flagged[s]])):
                acc_rows = [s for s in members if s in ev]
                kk = int(corr.reindex(acc_rows).sum()) if acc_rows else 0
                alo, ahi = wilson_interval(kk, len(acc_rows)) if acc_rows else (None, None)
                groups[name] = {"n": len(members), "n_clusters": int(pdata.clusters.reindex(members).nunique()),
                                "accuracy": {"n": len(acc_rows), "n_correct": kk, "estimate": kk / len(acc_rows) if acc_rows else None, "wilson_95": [alo, ahi],
                                             "descriptive_only": int(pdata.clusters.reindex(acc_rows).nunique()) < min_clusters},
                                "rho": rho_block(ctx, pdata, bhat, [s for s in members if s in rho_rows], mean, sd, min_cases=min_clusters)}
            entry["flagged_vs_unflagged"] = groups
            s2["representations"][r] = entry
            score_frames.append(pd.DataFrame({"sample_id": rows, "representation": r, "score": scores.to_numpy(dtype=float),
                                              "flagged": flagged.to_numpy(dtype=bool), "vendor": vendors.reindex(rows).to_numpy(dtype=object),
                                              "paired_predictor": predictor, "correct": corr.reindex(rows).to_numpy(dtype=float),
                                              "normalized_reconstruction_error": nre.to_numpy(dtype=float)}))
        _write_json(outputs[2], s2)
        v6._write_table(pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame(columns=["sample_id", "representation", "score"]),
                        outputs[3])

        # ---- S3 label-free harmonization (transductive)
        s3cfg = tcfg["s3_harmonization"]
        forbidden = sorted({*map(str, pdata.b.columns), *LABEL_COLUMNS})
        s3: dict[str, Any] = {**base, "schema": "tmmri.s3_harmonization/1", "reporting": s3cfg["reporting"],
                              "frozen": list(s3cfg["frozen"]), "forbidden_inputs": list(s3cfg["forbidden_inputs"]), "representations": {}}
        for r in tcfg["representations"]:
            if r not in representations or not have_rep.get(r):
                s3["representations"][r] = {"status": "NE", "reason": NOT_RECORDED}
                continue
            seed = _seeds(ctx)[0] if r == "f9b" else None
            x = representation_matrix(ctx, r, split, seed=seed)
            rows = [s for s in pdata.ids if s in x.index]
            a_frame = x.loc[rows]
            cols = list(a_frame.columns)
            model = _transition(representation_transition_path(ctx, r, seed=seed))
            predictor = _harmonizable_predictor(ctx, r)
            frozen_bhat = model.predict(a_frame.to_numpy(dtype=np.float64))
            frozen_bhat.index = rows
            pred_name = str(pairing[r])
            frozen_proba = None
            if predictor is not None:
                frozen_proba = predictor.predict_proba(a_frame.to_numpy(dtype=np.float64)).reindex(columns=classes, fill_value=0.0)
                frozen_proba.index = rows
                frozen_proba.loc[~np.isfinite(a_frame.to_numpy(dtype=float)).all(axis=1)] = np.nan
            rho_rows = [s for s in _rho_rows(ctx, pdata) if s in rows] if r == "descriptors" else rows
            entry = {"frozen": {"rho": rho_block(ctx, pdata, frozen_bhat, rho_rows, mean, sd)}, "predictor": pred_name, "variants": {}}
            if predictor is None:
                entry["predictor_status"] = {"status": "NE", "reason": NO_A_STANDARDIZATION if r == "f9b" else NOT_RECORDED}
            for variant in VARIANTS:
                vendor = pdata.vendor.reindex(rows) if (variant == "H-b" and pdata.vendor is not None) else None
                if variant == "H-b" and vendor is None:
                    entry["variants"][variant] = {"status": "NE", "reason": "VENDOR_METADATA_NOT_RECORDED"}
                    continue
                harm = fit_harmonization(a_frame, cols, variant=variant, vendor=vendor, min_rows_per_vendor=int(s3cfg["min_rows_per_vendor"]),
                                         forbidden=forbidden)
                bhat_h, record = apply_harmonized(model, a_frame.to_numpy(dtype=np.float64), harm, pd.Index(rows))
                v_entry: dict[str, Any] = {"harmonization": record, "label_free_assertion": "passed (representation columns only; B and labels read only to evaluate)",
                                           "rho": rho_block(ctx, pdata, bhat_h, rho_rows, mean, sd),
                                           "paired_rho_vs_frozen": paired_rho_block(ctx, pdata, frozen_bhat, bhat_h, rho_rows, mean, sd)}
                if predictor is not None:
                    proba_h, _ = apply_harmonized(predictor, a_frame.to_numpy(dtype=np.float64), harm, pd.Index(rows), predict="predict_proba")
                    proba_h = proba_h.reindex(columns=classes, fill_value=0.0)
                    proba_h.loc[~np.isfinite(a_frame.to_numpy(dtype=float)).all(axis=1)] = np.nan
                    ev = [s for s in eval_ids if s in rows]
                    v_entry["predictor_paired_vs_frozen"] = paired_predictor_block(ctx, pdata, frozen_proba, proba_h, ev)
                entry["variants"][variant] = v_entry
            s3["representations"][r] = entry
        _write_json(outputs[4], s3)

        # ---- S4 calibration transfer
        s4: dict[str, Any] = {**base, "schema": "tmmri.s4_calibration/1", "predictors": {}}
        rel_rows: list[pd.DataFrame] = []
        s4cfg = tcfg["s4_calibration"]
        for p in tcfg["predictors"]:
            temp = (temps.get(p) or {}).get("temperature")
            if p not in outputs_cache or temp is None:
                s4["predictors"][p] = {"status": "NE", "reason": NOT_RECORDED if p not in outputs_cache else "temperature_not_fitted"}
                continue
            proba, logits, prov = outputs_cache[p]
            rows = [s for s in eval_ids if s in proba.index]
            after = pd.DataFrame(scaled_probabilities(logits.reindex(rows).to_numpy(dtype=float), float(temp)), index=rows, columns=classes)
            before = proba.reindex(rows)
            paired = paired_predictor_block(ctx, pdata, before, after, rows)
            keep = evaluable_rows(pdata.truth.reindex(rows).to_numpy(dtype=object), before, classes) & evaluable_rows(
                pdata.truth.reindex(rows).to_numpy(dtype=object), after, classes)
            kept = [s for s, k in zip(rows, keep) if k]
            y = np.array([classes.index(str(pdata.truth[s])) for s in kept], dtype=int)
            entry = {"temperature": float(temp), "logits": prov.get("logits"), "provenance": prov, "before_after": paired, "n_evaluable": len(kept),
                     "ece_equal_width": {}}
            for state, frame in (("before", before), ("after", after)):
                pm = frame.loc[kept].to_numpy(dtype=float)
                if len(kept):
                    entry["ece_equal_width"][state] = ece_equal_width(pm, y, n_bins=int(s4cfg["ece_bins"]))
                    for binning in (str(s4cfg["primary_binning"]), str(s4cfg["companion_binning"])):
                        table = reliability_table(pm, y, n_bins=int(s4cfg["ece_bins"]), binning=binning)
                        table.insert(0, "predictor", p)
                        table.insert(1, "partition", split)
                        table.insert(2, "state", state)
                        table.insert(3, "binning", binning)
                        rel_rows.append(table)
            s4["predictors"][p] = entry
        _write_json(outputs[5], s4)
        v6._write_table(pd.concat(rel_rows, ignore_index=True) if rel_rows else pd.DataFrame(columns=["predictor", "partition", "state", "binning", "bin_id"]),
                        outputs[6])
        return {"partition": split, "evidence": info["evidence"], "not_executed": missing}

    return _lane_stage(ctx, "evaluate-trust", f"evaluate-trust:{split}", compute, inputs, outputs,
                       {"split": split, "allow_missing": bool(allow_missing), "not_executed": missing, "evidence": info["evidence"]},
                       status="EVALUATED")


def _harmonizable_predictor(ctx: v6.StageContext, representation: str):
    """The frozen predictor whose A standardization S3 replaces (None when it has none or is not recorded)."""

    import joblib

    if representation == "descriptors":
        return joblib.load(_input_path(ctx, "descriptors", "predictor"))
    if representation == "d20":
        path = _input_path(ctx, "d20", "predictor")
        return joblib.load(path) if path.is_file() else None
    return None


def _normalized_reconstruction_error(b: pd.DataFrame, bhat: pd.DataFrame, family: list[str], sd: pd.Series) -> pd.Series:
    rows = [s for s in bhat.index if s in b.index]
    t = b.loc[rows, family].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    p = bhat.loc[rows, family].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    s = np.array([float(sd.get(f, np.nan)) for f in family])
    err = np.abs(p - t) / np.where(np.isfinite(s) & (s > 0), s, np.nan)
    with np.errstate(all="ignore"):
        counts = np.isfinite(err).sum(axis=1)
        values = np.where(counts > 0, np.nansum(err, axis=1) / np.maximum(counts, 1), np.nan)
    return pd.Series(values, index=rows)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if a.size < 3 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return float("nan")
    return float(spearmanr(a, b).statistic)


# ============================================================================= build-comparison-tables (Table 4, Figure 5)


def build_comparison_tables(ctx: v6.StageContext):
    """Table 4 (representation x partition) and the tidy Figure-5 parquet from the recorded evaluations (read-only)."""

    from .comparison_tables import FIGURE5_NAME, TABLE4_NAME, build_comparison

    run = ctx.require_run("build-comparison-tables")
    pcfg = _probes(ctx)
    out_dir = run.path / str(pcfg["table4"]["outputs"])
    sources = [evaluation_path(ctx, b, p) for b in pcfg["bundles"] for p in pcfg["table4"]["partitions"]]
    present = [s for s in sources if s.is_file()]
    outputs = [out_dir / f"{TABLE4_NAME}.csv", out_dir / f"{TABLE4_NAME}.tex", out_dir / FIGURE5_NAME, out_dir / "provenance.json"]
    inputs = [run.path / "contracts" / "split_manifest.parquet", *present]

    def compute() -> dict[str, Any]:
        return build_comparison(run, pcfg, sources, out_dir, min_clusters=int(pcfg["table4"]["min_clusters"]))

    # Read-only report: content-addressed, never once-only locked (it is rebuilt as evaluations are added).
    return _lane_stage(ctx, "build-comparison-tables", "build-comparison-tables", compute, inputs, outputs,
                       {"sources_present": [run.portable(p) for p in present]}, once_only=False)
