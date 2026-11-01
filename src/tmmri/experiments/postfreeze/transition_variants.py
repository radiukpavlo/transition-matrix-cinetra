"""Typed-transition variants of the post-freeze block (B11 descriptor families, B12(b) geometry covariates).

Each variant is fitted exactly as the primary transition (``configs/transition.yaml``; the template of
``tmmri.stages.fit_transition`` and ``run_ablations``): nested out-of-fold B-hat on the recorded acdc_dev outer and inner
folds (``contracts/split_manifest.parquet``, ``contracts/inner_folds.parquet``), the final model with lambda by CV over
the outer folds, typed transforms, winsorization and standardization fitted on the fitting rows and zero-variance
columns dropped by :func:`tmmri.transition_fit.fit_transition_v6`.  Only the input columns differ.  Every fitted object
is written and hashed in the new run root (``FROZEN.json``) before any evaluated partition is processed, and the
evaluation reads it back from the hashed file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ...runs import hash_path
from .common import PostfreezeContext, load_partition, write_json, write_table

PRIMARY = "primary_ridge"            # the recorded primary typed ridge of the source run (B-hat as evaluated; acdc_dev OOF)


def dev_design(pctx: PostfreezeContext) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, Callable[..., np.ndarray]]:
    """(ids, formal A in ``formal_descriptor_names`` order, B, outer folds, inner-fold function) of the acdc_dev fit rows
    (critical QC removed), read from the verified source run with the v6 helpers."""

    from ...stages import _dev_matrices, _inner_fold_fn

    for rel in ("formal/A_ACDC.parquet", "semantics/B_ACDC.parquet", "contracts/split_manifest.parquet", "contracts/inner_folds.parquet"):
        pctx.source_file(rel)
    ids, a, b, folds = _dev_matrices(pctx.stage_ctx)
    return ids, a, b, folds, _inner_fold_fn(pctx.stage_ctx, ids)


def fit_and_freeze(pctx: PostfreezeContext, out: Path, estimator_id: str, x: np.ndarray, b: pd.DataFrame, ids: pd.DataFrame, folds: np.ndarray,
                   inner_fn: Callable[..., np.ndarray], columns: list[str]) -> dict[str, Any]:
    """Nested OOF and final fit of one variant on acdc_dev; writes ``models/<id>/`` and its ``FROZEN.json``."""

    from ...transition_fit import fit_transition_v6, nested_oof_transition

    study = pctx.study
    targets = study.transition_targets()
    if x.shape != (len(ids), len(columns)):
        raise ValueError(f"{estimator_id}: design matrix {x.shape} does not match {len(ids)} rows x {len(columns)} columns")
    oof, details = nested_oof_transition(x, b, targets, study.transition, study.registry, folds, inner_fn, return_details=True)
    oof.index = ids["sample_id"].astype(str).to_numpy()
    model = fit_transition_v6(x, b, targets, study.transition, study.registry, folds, representation=estimator_id)
    mdir = out / "models" / estimator_id
    model_path, oof_path, lam_path = mdir / "transition_model.json", mdir / "oof_acdc_dev.parquet", mdir / "lambda_selection.parquet"
    write_json(model.to_dict(), model_path, pctx)
    write_table(oof.reset_index(names="sample_id"), oof_path, pctx)
    final = pd.DataFrame([{"outer_fold": pd.NA, "target": t, "lambda": model.lambdas.get(t, np.nan), "lambda_at_grid_edge": bool(model.edge_flags.get(t, False)),
                           "eligible": t in model.fitted_targets, "exclusion_reason": model.excluded.get(t, ""), "n_fit_rows": model.n_fit_rows}
                          for t in targets])
    ledger = pd.concat([details["lambda_selection"].assign(scope="outer_fold"), final.assign(scope="final")], ignore_index=True)
    ledger["outer_fold"] = ledger["outer_fold"].astype("Int64")
    write_table(ledger, lam_path, pctx)
    frozen = {"estimator_id": estimator_id, "fit_partition": "acdc_dev", "folds": "recorded outer and inner folds of the source run",
              "columns": list(columns), "n_columns": len(columns), "n_rows": int(len(ids)), "n_fit_rows": int(model.n_fit_rows),
              "n_nonfinite_rows": int((~np.isfinite(x).all(axis=1)).sum()), "fitted_targets": list(model.fitted_targets),
              "excluded_targets": dict(model.excluded), "dropped_zero_variance_columns": [c for c, k in zip(columns, model.standardizer.keep) if not k]
              if hasattr(model.standardizer, "keep") and len(model.standardizer.keep) == len(columns) else None,
              "files": {p.name: hash_path(p) for p in (model_path, oof_path, lam_path)}}
    frozen_path = write_json(frozen, mdir / "FROZEN.json", pctx)
    return {"estimator_id": estimator_id, "model_path": model_path, "oof_path": oof_path, "frozen_path": frozen_path, "frozen": frozen,
            "outputs": [model_path, oof_path, lam_path, frozen_path]}


def load_frozen(entry: dict[str, Any]) -> tuple[Any, pd.DataFrame]:
    """(model, acdc_dev OOF) read back from the frozen files after checking their hashes."""

    from ...stages import BundleError
    from ...transition_fit import TransitionModel

    for name, sha in entry["frozen"]["files"].items():
        path = entry["model_path"].parent / name
        if hash_path(path) != sha:
            raise BundleError(f"{entry['estimator_id']}: {name} changed after its freeze")
    model = TransitionModel.from_dict(json.loads(entry["model_path"].read_text(encoding="utf-8")))
    oof = pd.read_parquet(entry["oof_path"]).set_index("sample_id")
    return model, oof


def evaluate_variants(pctx: PostfreezeContext, splits: list[str], variants: dict[str, dict[str, Any]],
                      design: Callable[[str, str, list[str], dict[str, Any]], np.ndarray], *, extra_arms: Callable[..., dict[str, pd.DataFrame]] | None = None,
                      reference: str = "primary_ridge") -> dict[str, Any]:
    """Every variant once per partition (acdc_dev: its nested OOF B-hat; evaluated partitions: the frozen final model on
    ``design(split, dataset, ids, entry)``) beside the recorded primary ridge (arm ``primary_ridge``), with the dev-mean
    null and the paired cluster bootstrap of :func:`tmmri.experiments.postfreeze.estimates.paired_rho`; differences are
    taken against the arm ``reference``."""

    from .common import dev_reference
    from .estimates import families, paired_rho, rho_rows, target_rows

    sd, mean = dev_reference(pctx)
    fams = families(pctx.lane_cfg)
    targets = fams["rho"]
    loaded = {name: load_frozen(entry) for name, entry in variants.items()}
    results, rho_frames, target_frames, preds_by_split = {}, [], [], {}
    for split in splits:
        part = load_partition(pctx, split)
        y = part.observed(targets)
        null = pd.DataFrame([mean[targets]] * len(part.ids), index=part.ids)
        preds = {PRIMARY: part.bhat}
        for name, (model, oof) in loaded.items():
            if split == "acdc_dev":
                preds[name] = oof.reindex(part.ids)
            else:
                x = design(split, part.dataset, part.ids, variants[name])
                preds[name] = model.predict(x, index=pd.Index(part.ids))
        if extra_arms is not None:
            preds.update(extra_arms(split, part))
        result = paired_rho(y, {k: (v, null) for k, v in preds.items()}, sd, fams, units=part.units, clusters=part.clusters, boot=pctx.boot,
                            study=pctx.study, min_cases=int(pctx.lane_cfg["statistics"]["min_cases_per_stratum"]), reference=reference)
        role = "development (nested out-of-fold)" if split == "acdc_dev" else "evaluation (frozen objects, once)"
        label = "acdc_dev_oof" if split == "acdc_dev" else split
        result.update({"partition": label, "role": role, "n_excluded_critical_qc": part.n_excluded_critical_qc,
                       "n_nonfinite_predictions": {k: int((~np.isfinite(v.reindex(index=part.ids, columns=targets).to_numpy(dtype=float))).all(axis=1).sum())
                                                   for k, v in preds.items()}})
        results[label] = result
        rho_frames.append(rho_rows(result, partition=label, role=role))
        for name, pred in preds.items():
            target_frames.append(target_rows(y, pred, sd, targets, result["arms"][name], units=part.units, partition=label, arm=name))
        preds_by_split[label] = (part, y, preds)
    return {"results": results, "rho": pd.concat(rho_frames, ignore_index=True), "per_target": pd.concat(target_frames, ignore_index=True),
            "predictions": preds_by_split, "sd": sd, "mean": mean}


def difference_rows(results: dict[str, Any]) -> pd.DataFrame:
    """Paired differences (arm minus reference) of rho per family, rho_j and nMAE per target, by partition."""

    rows = []
    for partition, result in results.items():
        for arm, diff in result.get("differences", {}).items():
            for fam, r in diff["rho"].items():
                rows.append({"partition": partition, "arm": arm, "versus": diff["versus"], "quantity": fam, "target": None,
                             **{k: r.get(k) for k in ("estimate", "ci_lower", "ci_upper", "status", "reason")}})
            for key in ("rho_j", "nmae"):
                for target, r in diff[key].items():
                    rows.append({"partition": partition, "arm": arm, "versus": diff["versus"], "quantity": key, "target": target,
                                 **{k: r.get(k) for k in ("estimate", "ci_lower", "ci_upper", "status", "reason")}})
    return pd.DataFrame(rows)
