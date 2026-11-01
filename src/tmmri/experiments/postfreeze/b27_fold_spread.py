"""B27: fold-spread concept uncertainty (EPC-7, readiness review R7; ``configs/exploratory_postfreeze_v603.yaml``
analyses.B27).

* **Signal.**  Per evaluated case and primary target, the standard deviation (ddof = 1) of the reconstructions of the five
  recorded outer-fold transitions (``transitions/fold_models/transition_fold1-5.json``) on the decoded physical scale
  (the fold mappers decoded as for per-object stability, Section 11.4), divided by s_j (the acdc_dev fit-row SD).  A
  cell needs all five fold values; otherwise it has no signal.  The case-level signal is the maximum over the targets
  with a signal.
* **Large concept errors.**  |B-hat - B| / s_j of the recorded raw reconstruction on the OBSERVED cells (as B17), large
  (a) above the B17 development threshold (``postfreeze/B17/summary.json`` development_thresholds of the second source
  run; the acdc_dev out-of-fold 90th percentile per target, recomputed from the recorded development out-of-fold B-hat;
  a difference above 1e-12 refuses before any output) or (b) in the top decile within each partition (above the
  partition's 90th percentile, numpy linear interpolation; an outcome definition, not a setting of any method);
  any-large when at least one target is large.
* **Metrics.**  AUROC of the signal for each indicator (per target with the target's signal, any-large with the
  case-level signal), with the cluster bootstrap of the lane; the Spearman correlation of the signal with the normalized
  error per target, with the same bootstrap; and the B17 continuous signals (uncertainty, OOD score, count of unstable
  bins; :func:`b17_concept_reliability.signals`) re-read under definition (b).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .b17_concept_reliability import CONTINUOUS, large_error_labels, normalized_abs_error, signals
from .common import COMMAND, clean, formal_a, included_partitions, load_partition, write_json, write_table
from .estimates import BELOW_MIN_CASES, auroc_ci

ANALYSIS = "B27"
N_FOLDS = 5
TOLERANCE = 1e-12


def fold_spread(fold_values: list[pd.DataFrame], sd: pd.Series, targets: list[str]) -> pd.DataFrame:
    """SD (ddof = 1) over the fold reconstructions per case and target divided by s_j (NaN unless every fold value is
    finite), with the case-level maximum in ``any_large``."""

    stack = np.stack([fv.reindex(columns=targets).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float) for fv in fold_values])
    complete = np.isfinite(stack).all(axis=0)
    with np.errstate(invalid="ignore"):
        spread = np.where(complete, np.std(stack, axis=0, ddof=1), np.nan) / sd.reindex(targets).to_numpy(dtype=float)
    out = pd.DataFrame(spread, index=fold_values[0].index, columns=targets)
    out["any_large"] = out[targets].max(axis=1, skipna=True)
    return out


def within_partition_labels(err: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Definition (b): large above the partition's 90th percentile of the normalized error per target."""

    thresholds = err.quantile(0.90, interpolation="linear")
    return large_error_labels(err, thresholds), thresholds


def spearman_ci(signal: pd.Series, error: pd.Series, clusters: pd.Series, boot: dict[str, Any], min_cases: int) -> dict[str, Any]:
    from scipy.stats import spearmanr

    from ..alt_bundles import bootstrap_statistic

    both = signal.notna() & error.notna()
    s, e = signal[both].to_numpy(dtype=float), error[both].to_numpy(dtype=float)
    if both.sum() < 3:
        return {"n": int(both.sum()), "estimate": None, "ci_lower": None, "ci_upper": None, "status": "NE", "reason": "FEWER_THAN_3_CASES"}

    def rho(rows: np.ndarray) -> float:
        if np.unique(s[rows]).size < 2 or np.unique(e[rows]).size < 2:
            return float("nan")
        return float(spearmanr(s[rows], e[rows])[0])

    below = int(both.sum()) < int(min_cases)
    r = bootstrap_statistic(rho, int(both.sum()), clusters.reindex(signal.index)[both].astype(str).to_numpy(dtype=object),
                            n_boot=0 if below else int(boot["replicates"]), seed=int(boot["seed"]), level=float(boot.get("level", 0.95)),
                            min_valid_fraction=0.0)
    if below and r.get("estimate") is not None:
        r.update({"ci_lower": None, "ci_upper": None, "status": "NE", "reason": BELOW_MIN_CASES})
    return {"n": int(both.sum()), **{k: r.get(k) for k in ("estimate", "ci_lower", "ci_upper", "status", "reason")}}


def auroc_row(score: pd.Series, label: pd.Series, clusters: pd.Series, boot: dict[str, Any], min_cases: int, **labels: Any) -> dict[str, Any]:
    r = auroc_ci(score.to_numpy(dtype=float), label.reindex(score.index).to_numpy(dtype=float), clusters.reindex(score.index).to_numpy(dtype=object),
                 boot, min_cases=min_cases)
    return {**labels, "auroc": r["estimate"], "ci_lower": r["ci_lower"], "ci_upper": r["ci_upper"], "n": r["n"], "n_positive": r["n_positive"],
            "n_negative": r["n_negative"], "status": r["status"], "reason": r.get("reason")}


def run(pctx: Any) -> dict[str, Any]:
    from ...formal_features import formal_descriptor_names
    from ...stages import fold_model_paths, load_fold_models
    from ...wedd import CutSet
    from .common import dev_reference

    acfg = pctx.analysis_cfg(ANALYSIS)
    out = pctx.out_dir(ANALYSIS)
    study = pctx.study
    stats = pctx.lane_cfg["statistics"]
    targets = [str(t) for t in stats["primary_targets"]]
    min_cases = int(stats["min_cases_per_stratum"])
    names = formal_descriptor_names()
    sd, _ = dev_reference(pctx)
    paths_rel = [p.relative_to(pctx.source_root).as_posix() for p in fold_model_paths(pctx.stage_ctx, pctx.stage_ctx.run)]
    if len(paths_rel) != N_FOLDS:
        raise ValueError(f"{ANALYSIS}: {len(paths_rel)} recorded outer-fold transitions, expected {N_FOLDS}")
    for rel in paths_rel:
        pctx.source_file(rel)
    fold_models = load_fold_models(pctx.stage_ctx, pctx.stage_ctx.run)
    b17 = pctx.read_second_json("postfreeze/B17/summary.json")
    dev_thresholds = pd.Series({t: float(b17["development_thresholds"][t]) for t in targets})
    # The B17 thresholds recomputed from the recorded development out-of-fold B-hat (a reproduction check only).
    dev = load_partition(pctx, "acdc_dev")
    recomputed = normalized_abs_error(dev.observed(targets), dev.bhat, sd, targets).quantile(0.90, interpolation="linear")
    reproduced = bool(np.allclose(recomputed.reindex(targets).to_numpy(dtype=float), dev_thresholds.to_numpy(dtype=float), rtol=0, atol=TOLERANCE))
    if not reproduced:                  # analyses.B27.large_error.development: a mismatch refuses, before any output
        from ...stages import BundleError

        raise BundleError(f"{COMMAND} {ANALYSIS}: the B17 development thresholds recomputed from the recorded development out-of-fold B-hat "
                          f"differ from postfreeze/B17/summary.json of the second source run by more than {TOLERANCE:g}; refused")
    profile = str(study.wedd["primary_profile"])
    cuts = CutSet.from_dict(pctx.read_json(f"thresholds/d_model_{profile}.json"))
    candidates = [c for c in study.rule_candidates() if c in cuts.features]
    splits, skipped = included_partitions(pctx, [str(p) for p in acfg["partitions"]], f"{COMMAND} {ANALYSIS}")
    auroc_rows, spearman_rows, reread_rows, partition_thresholds = [], [], [], {}
    for split in splits:
        part = load_partition(pctx, split)
        a_p = formal_a(pctx, part.dataset).reindex(part.ids)[names].to_numpy(dtype=float)
        index = pd.Index(part.ids, name="sample_id")
        folds = [m.decode(a_p, dict(study.registry), dict(study.transition), index=index)[0] for m in fold_models]
        signal = fold_spread(folds, sd, targets)
        err = normalized_abs_error(part.observed(targets), part.bhat, sd, targets)
        labels = {"development_b17": large_error_labels(err, dev_thresholds)}
        labels["within_partition_top_decile"], thresholds = within_partition_labels(err)
        partition_thresholds[split] = thresholds.to_dict()
        clusters = part.clusters.reindex(part.ids)
        for definition, lab in labels.items():
            for indicator in [*targets, "any_large"]:
                auroc_rows.append(auroc_row(signal[indicator], lab[indicator], clusters, pctx.boot, min_cases, partition=split, definition=definition,
                                            indicator=indicator, signal="fold_spread",
                                            n_signal_missing=int(signal[indicator].isna().sum())))
        for t in targets:
            spearman_rows.append({"partition": split, "target": t, "signal": "fold_spread", "error": "|B-hat - B| / s_j",
                                  **spearman_ci(signal[t], err[t], clusters, pctx.boot, min_cases)})
        sig, missing = signals(pctx, split, part.ids, candidates)
        for s in CONTINUOUS:
            for indicator in [*targets, "any_large"]:
                if s in missing:
                    reread_rows.append({"partition": split, "definition": "within_partition_top_decile", "indicator": indicator, "signal": s,
                                        "status": "NE", "reason": missing[s]})
                    continue
                reread_rows.append(auroc_row(sig[s], labels["within_partition_top_decile"][indicator], clusters, pctx.boot, min_cases, partition=split,
                                             definition="within_partition_top_decile", indicator=indicator, signal=s))
    paths = {"auroc": out / "spread_auroc_table.parquet", "spearman": out / "spread_spearman_table.parquet", "reread": out / "b17_reread_table.parquet",
             "summary": out / "summary.json"}
    write_table(pd.DataFrame(auroc_rows), paths["auroc"], pctx)
    write_table(pd.DataFrame(spearman_rows), paths["spearman"], pctx)
    write_table(pd.DataFrame(reread_rows), paths["reread"], pctx)
    summary = clean({"analysis": ANALYSIS, "signal": acfg["signal"], "large_error": acfg["large_error"], "metrics": acfg["metrics"], "reread": acfg["reread"],
                     "fold_models": paths_rel, "development_thresholds": dev_thresholds.to_dict(), "development_thresholds_source":
                     "postfreeze/B17/summary.json (second source run)", "development_thresholds_reproduced": reproduced,
                     "within_partition_thresholds": partition_thresholds, "included": splits, "not_executed": skipped,
                     "auroc": json.loads(pd.DataFrame(auroc_rows).to_json(orient="records")) if auroc_rows else []})
    write_json(summary, paths["summary"], pctx)
    pctx.write_provenance(ANALYSIS, dict(pctx.verified_log), list(paths.values()),
                          {"included": splits, "not_executed": skipped, "development_thresholds_reproduced": reproduced})
    return {"included": splits, "not_executed": skipped, "development_thresholds_reproduced": reproduced}
