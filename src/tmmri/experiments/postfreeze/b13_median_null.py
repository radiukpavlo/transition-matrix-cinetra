"""B13: the development-median baseline (EPC-6; ``configs/exploratory_postfreeze.yaml`` analyses.B13).

Under absolute loss the optimal constant is the median.  rho, rho_EF and rho_j of the recorded primary ridge against
the acdc_dev median of each target (physical units, non-critical-QC fit rows, OBSERVED cells) are reported beside the
mean-null results with one paired cluster bootstrap (median minus mean).  On acdc_dev (nested out-of-fold) both nulls are
fold-specific: the median and the mean of the outer-training rows of each fold (the recorded out-of-fold mean,
``transitions/comparator_mean.parquet``), with the full-development mean of the recorded development rho beside them.
The recorded rho of each partition is quoted unchanged; nothing replaces it and no recorded decision changes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .common import (
    EVALUATED,
    clean,
    dev_reference,
    included_partitions,
    load_partition,
    write_json,
    write_table,
)
from .estimates import families, paired_rho, rho_rows, target_rows
from .transition_variants import difference_rows

ANALYSIS = "B13"
RECORDED = {"acdc_test": "metrics/primary_rho_acdc_test.json", "mms_external": "metrics/primary_rho_mms_external.json",
            "mms2_external": "metrics/primary_rho_mms2_external.json"}


def dev_fit_rows(pctx: Any) -> pd.DataFrame:
    """Measured B (OBSERVED cells) of the acdc_dev fit rows with their outer fold (critical QC removed)."""

    from ...stages import critical_qc_mask
    from ..exploratory.common import observed

    split = pctx.read_parquet("contracts/split_manifest.parquet")
    dev = split[split["partition"] == "acdc_dev"].set_index("sample_id")
    b = pctx.read_parquet("semantics/B_ACDC.parquet").set_index("sample_id")
    b = b[b.index.isin(dev.index) & ~critical_qc_mask(b, pctx.study).to_numpy(dtype=bool)]
    targets = pctx.study.transition_targets()
    frame = pd.DataFrame({t: observed(b, t) for t in targets}, index=b.index)
    frame["outer_fold"] = dev.loc[frame.index, "outer_fold"].astype(int)
    return frame


def fold_null(frame: pd.DataFrame, ids: list[str], folds: pd.Series, targets: list[str], how: str) -> pd.DataFrame:
    """Per held-out row: the median or mean of the outer-training rows of its fold."""

    out = pd.DataFrame(np.nan, index=ids, columns=targets)
    for f in sorted(set(folds.dropna().astype(int))):
        train = frame[frame["outer_fold"] != f][targets]
        stat = train.median() if how == "median" else train.mean()
        rows = [s for s in ids if int(folds.loc[s]) == f]
        out.loc[rows, targets] = np.tile(stat[targets].to_numpy(dtype=float), (len(rows), 1))
    return out


def run(pctx: Any) -> dict[str, Any]:
    acfg = pctx.analysis_cfg(ANALYSIS)
    out = pctx.out_dir(ANALYSIS)
    fams = families(pctx.lane_cfg)
    targets = fams["rho"]
    min_cases = int(pctx.lane_cfg["statistics"]["min_cases_per_stratum"])
    sd, mean = dev_reference(pctx)
    fit = dev_fit_rows(pctx)
    median = fit[targets].median()
    splits, skipped = included_partitions(pctx, ["acdc_dev", *EVALUATED], f"run-postfreeze {ANALYSIS}")
    results, rho_frames, target_frames, side = {}, [], [], []
    for split in splits:
        part = load_partition(pctx, split)
        label = "acdc_dev_oof" if split == "acdc_dev" else split
        y = part.observed(targets)
        full_mean = pd.DataFrame([mean[targets]] * len(part.ids), index=part.ids)
        if split == "acdc_dev":
            folds = part.bhat["outer_fold"].astype(float)
            oof_mean = pctx.read_parquet("transitions/comparator_mean.parquet").set_index("sample_id").reindex(part.ids)[targets]
            arms = {"mean_null_outer_training": (part.bhat, oof_mean),
                    "median_null_outer_training": (part.bhat, fold_null(fit, part.ids, folds, targets, "median")),
                    "mean_null_full_development": (part.bhat, full_mean)}
            reference = "mean_null_outer_training"
            rel = "metrics/ablations/transition_variants.json"
            recorded = pctx.read_json(rel).get("acdc_dev_oof", {}).get("ridge_typed_formal57") if pctx.exists(rel) else None
            recorded_source = f"{rel} acdc_dev_oof.ridge_typed_formal57" if recorded is not None else None
        else:
            arms = {"mean_null": (part.bhat, full_mean), "median_null": (part.bhat, pd.DataFrame([median[targets]] * len(part.ids), index=part.ids))}
            reference = "mean_null"
            recorded = pctx.read_json(RECORDED[split])
            recorded_source = RECORDED[split]
        result = paired_rho(y, arms, sd, fams, units=part.units, clusters=part.clusters, boot=pctx.boot, study=pctx.study, min_cases=min_cases,
                            reference=reference)
        result.update({"partition": label, "role": part.role, "n_excluded_critical_qc": part.n_excluded_critical_qc})
        results[label] = result
        rho_frames.append(rho_rows(result, partition=label))
        for name, (pred, null) in arms.items():
            target_frames.append(target_rows(y, pred, sd, targets, result["arms"][name], units=part.units, partition=label, null=name))
        mean_arm = "mean_null_full_development" if split == "acdc_dev" else "mean_null"
        median_arm = "median_null_outer_training" if split == "acdc_dev" else "median_null"
        for fam in fams:
            rec = (recorded or {}) if fam == "rho" else {}
            side.append({"partition": label, "quantity": fam, "target": None, "recorded_source": recorded_source if fam == "rho" else None,
                         "recorded_mean_null": rec.get("rho"), "recorded_ci_lower": rec.get("ci_lower"), "recorded_ci_upper": rec.get("ci_upper"),
                         "recorded_n_cases": rec.get("n_cases"), "recomputed_mean_null": result["arms"][mean_arm]["rho"][fam]["estimate"],
                         "median_null": result["arms"][median_arm]["rho"][fam]["estimate"],
                         "median_null_ci_lower": result["arms"][median_arm]["rho"][fam]["ci_lower"],
                         "median_null_ci_upper": result["arms"][median_arm]["rho"][fam]["ci_upper"], "n_cases": result["n_cases"]})
        for t in targets:
            rec_j = ((recorded or {}).get("rho_j") or {}).get(t)
            side.append({"partition": label, "quantity": "rho_j", "target": t, "recorded_source": recorded_source, "recorded_mean_null": rec_j,
                         "recorded_ci_lower": (((recorded or {}).get("rho_j_ci") or {}).get(t) or [None, None])[0],
                         "recorded_ci_upper": (((recorded or {}).get("rho_j_ci") or {}).get(t) or [None, None])[1], "recorded_n_cases": (recorded or {}).get("n_cases"),
                         "recomputed_mean_null": result["arms"][mean_arm]["rho_j"][t]["estimate"], "median_null": result["arms"][median_arm]["rho_j"][t]["estimate"],
                         "median_null_ci_lower": result["arms"][median_arm]["rho_j"][t]["ci_lower"],
                         "median_null_ci_upper": result["arms"][median_arm]["rho_j"][t]["ci_upper"], "n_cases": result["n_cases"]})
    paths = {"rho": out / "rho_median.parquet", "per_target": out / "rho_j_median.parquet", "side": out / "side_by_side_with_mean_null.parquet",
             "diff": out / "paired_difference_median_minus_mean.parquet", "summary": out / "summary.json"}
    write_table(pd.concat(rho_frames, ignore_index=True), paths["rho"], pctx)
    write_table(pd.concat(target_frames, ignore_index=True), paths["per_target"], pctx)
    write_table(pd.DataFrame(side), paths["side"], pctx)
    write_table(difference_rows(results), paths["diff"], pctx)
    labels = list(results)
    summary = clean({"analysis": ANALYSIS, "null": acfg["null"], "rule": acfg["rule"], "development_median": {t: median.get(t) for t in targets},
                     "development_mean": {t: mean.get(t) for t in targets}, "included": labels, "not_executed": skipped, "results": results})
    write_json(summary, paths["summary"], pctx)
    pctx.write_provenance(ANALYSIS, dict(pctx.verified_log), list(paths.values()), {"included": labels, "not_executed": skipped})
    return {"included": labels, "not_executed": skipped}
