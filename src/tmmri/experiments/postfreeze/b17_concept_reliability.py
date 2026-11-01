"""B17: reliability of the reconstructed concepts (EPC-7; ``configs/exploratory_postfreeze.yaml`` analyses.B17).

Error: per case and primary target |B-hat - B| / s_j (raw recorded reconstruction, OBSERVED cells, non-critical-QC rows).
A target error is large above the 90th percentile of the same quantity on the acdc_dev nested out-of-fold
reconstructions (fixed on development data); a case is any-large when at least one primary target is large.
Signals, all recorded: 1 - maximum softmax probability of the audited predictor, not accepted at tau_50 / tau_80
(``trust/s1_thresholds.json``; accepted when the confidence is at least tau, the S1 rule), the S2 Mahalanobis score of
the descriptors and its flag (``trust/<p>/s2_scores.parquet``) and the number of rule concepts with UNSTABLE_ESTIMATE
(``reconstructed/stability_d_model_<p>.parquet``).  AUROC of each continuous signal (cluster bootstrap) and sensitivity,
specificity and PPV of each binary signal (Wilson).  Symbol errors: for each concept with at least one retained cut, the
bin of the decoded B-hat (ESTIMATED cells) differs from the bin of the measured B (OBSERVED cells) under the frozen
``d_model`` cuts; UNSTABLE_ESTIMATE against symbol error as 2 x 2 counts with sensitivity, specificity (Wilson) and the
risk ratio (Katz), per concept and pooled over concept cells (descriptive; cells of one case are not independent).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .common import clean, included_partitions, load_partition, write_json, write_table
from .estimates import auroc_ci, binary_signal, risk_ratio, wilson

ANALYSIS = "B17"
CONTINUOUS = ("uncertainty", "ood_score", "unstable_bins")
BINARY = ("not_accepted_tau50", "not_accepted_tau80", "ood_flag")
UNSTABLE = "UNSTABLE_ESTIMATE"


def normalized_abs_error(y: pd.DataFrame, bhat: pd.DataFrame, sd: pd.Series, targets: list[str]) -> pd.DataFrame:
    """|B-hat - B| / s_j per case and target (NaN where B is not OBSERVED or B-hat is not finite)."""

    pred = bhat.reindex(index=y.index, columns=targets).apply(pd.to_numeric, errors="coerce")
    return (pred - y[targets]).abs() / sd.reindex(targets)


def large_error_labels(err: pd.DataFrame, thresholds: pd.Series) -> pd.DataFrame:
    """1/0 per target (NaN without a finite error) and ``any_large`` (NaN when no target is finite)."""

    out = pd.DataFrame({t: np.where(err[t].notna(), (err[t] > thresholds[t]).astype(float), np.nan) for t in err.columns}, index=err.index)
    finite = out.notna().any(axis=1)
    out["any_large"] = np.where(finite, (out.fillna(0.0) > 0.5).any(axis=1).astype(float), np.nan)
    return out


def signals(pctx: Any, split: str, ids: list[str], candidates: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    """The recorded signals of ``split`` on ``ids`` (NaN where not recorded) and the reason of every signal not recorded."""

    from ..trust.selective import max_softmax

    frame = pd.DataFrame(index=pd.Index(ids, name="sample_id"), columns=[*CONTINUOUS, *BINARY], dtype=float)
    missing: dict[str, str] = {}
    rel = f"inference/audited_predictor/{split}/predictions.parquet"
    if pctx.exists(rel):
        pred = pctx.read_parquet(rel).set_index("sample_id")
        probs = pred[[c for c in pred.columns if str(c).startswith("prob_")]].reindex(ids)
        conf = pd.Series(max_softmax(probs.to_numpy(dtype=float)), index=ids)
        frame["uncertainty"] = 1.0 - conf
        thresholds = (pctx.read_json("trust/s1_thresholds.json").get("predictors") or {}).get("audited_predictor") or {} \
            if pctx.exists("trust/s1_thresholds.json") else {}
        for name in ("tau50", "tau80"):
            tau = (thresholds.get(name) or {}).get("tau")
            if tau is None:
                missing[f"not_accepted_{name}"] = "trust/s1_thresholds.json audited_predictor threshold not recorded"
            else:
                frame[f"not_accepted_{name}"] = np.where(conf.notna(), (conf < float(tau)).astype(float), np.nan)
    else:
        for s in ("uncertainty", "not_accepted_tau50", "not_accepted_tau80"):
            missing[s] = f"{rel} not recorded"
    rel = f"trust/{split}/s2_scores.parquet"
    scores = pctx.read_parquet(rel) if pctx.exists(rel) else pd.DataFrame(columns=["sample_id", "representation"])
    scores = scores[scores["representation"].astype(str) == "descriptors"].set_index("sample_id")
    if len(scores):
        frame["ood_score"] = pd.to_numeric(scores["score"], errors="coerce").reindex(ids)
        flag = scores["flagged"].reindex(ids)
        frame["ood_flag"] = np.where(flag.notna() & frame["ood_score"].notna(), flag.fillna(False).astype(bool).astype(float), np.nan)
    else:
        missing["ood_score"] = missing["ood_flag"] = f"{rel} (descriptors) not recorded"
    rel = f"reconstructed/stability_d_model_{split}.parquet"
    if pctx.exists(rel):
        stab = pctx.read_parquet(rel).set_index("sample_id").reindex(ids)
        cols = [c for c in candidates if c in stab.columns]
        frame["unstable_bins"] = (stab[cols].astype(object) == UNSTABLE).sum(axis=1).astype(float)
    else:
        missing["unstable_bins"] = f"{rel} not recorded"
    return frame, missing


def symbol_errors(cuts: Any, concepts: list[str], b_obs: pd.DataFrame, decoded: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    """Per case and concept: symbol_error and unstable (rows with an OBSERVED B and an ESTIMATED decoded B-hat)."""

    bins_b = cuts.bin_indices(b_obs.reindex(columns=cuts.order))
    bins_h = cuts.bin_indices(decoded.reindex(index=b_obs.index, columns=cuts.order))
    rows = []
    for c in concepts:
        state = decoded.reindex(b_obs.index)[f"{c}__state"].astype(object) if f"{c}__state" in decoded.columns else pd.Series("ESTIMATED", index=b_obs.index)
        ok = (bins_b[c] >= 0) & (bins_h[c] >= 0) & (state == "ESTIMATED")
        unstable = stability.reindex(b_obs.index)[c].astype(object) == UNSTABLE if c in stability.columns else pd.Series(False, index=b_obs.index)
        rows.append(pd.DataFrame({"concept": c, "symbol_error": (bins_b[c] != bins_h[c])[ok].to_numpy(), "unstable": unstable[ok].to_numpy(dtype=bool)}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["concept", "symbol_error", "unstable"])


def two_by_two(cells: pd.DataFrame) -> dict[str, Any]:
    err, uns = cells["symbol_error"].astype(bool).to_numpy(), cells["unstable"].astype(bool).to_numpy()
    a, b, c, d = int((uns & err).sum()), int((uns & ~err).sum()), int((~uns & err).sum()), int((~uns & ~err).sum())
    sens, spec = wilson(a, a + c), wilson(d, b + d)
    rr = risk_ratio(a, b, c, d)
    return {"n_cells": int(len(cells)), "unstable_error": a, "unstable_no_error": b, "stable_error": c, "stable_no_error": d,
            "sensitivity": sens["estimate"], "sensitivity_ci_lower": sens["ci_lower"], "sensitivity_ci_upper": sens["ci_upper"],
            "specificity": spec["estimate"], "specificity_ci_lower": spec["ci_lower"], "specificity_ci_upper": spec["ci_upper"],
            "risk_ratio": rr["estimate"], "risk_ratio_ci_lower": rr["ci_lower"], "risk_ratio_ci_upper": rr["ci_upper"], "risk_ratio_status": rr["status"],
            "risk_ratio_reason": rr["reason"]}


def run(pctx: Any) -> dict[str, Any]:
    from ...wedd import CutSet
    from .common import dev_reference

    acfg = pctx.analysis_cfg(ANALYSIS)
    out = pctx.out_dir(ANALYSIS)
    study = pctx.study
    targets = [str(t) for t in pctx.lane_cfg["statistics"]["primary_targets"]]
    min_cases = int(pctx.lane_cfg["statistics"]["min_cases_per_stratum"])
    sd, _ = dev_reference(pctx)
    dev = load_partition(pctx, "acdc_dev")
    dev_err = normalized_abs_error(dev.observed(targets), dev.bhat, sd, targets)
    thresholds = dev_err.quantile(0.90, interpolation="linear")
    profile = str(study.wedd["primary_profile"])
    cuts = CutSet.from_dict(pctx.read_json(f"thresholds/d_model_{profile}.json"))
    retained = cuts.retain(pctx.read_parquet(f"thresholds/d_model_{profile}_stability.parquet"))
    candidates = [c for c in study.rule_candidates() if c in cuts.features]
    concepts = [c for c in candidates if c in retained.features and len(retained.features[c].cuts)]
    splits, skipped = included_partitions(pctx, [str(p) for p in acfg["partitions"]], f"run-postfreeze {ANALYSIS}")
    auroc_rows, binary_rows, symbol_rows, not_recorded = [], [], [], {}
    for split in splits:
        part = load_partition(pctx, split)
        labels = large_error_labels(normalized_abs_error(part.observed(targets), part.bhat, sd, targets), thresholds)
        sig, missing = signals(pctx, split, part.ids, candidates)
        not_recorded[split] = missing
        clusters = part.clusters.reindex(part.ids)
        for indicator in [*targets, "any_large"]:
            y = labels[indicator]
            for s in CONTINUOUS:
                if s in missing:
                    auroc_rows.append({"partition": split, "signal": s, "indicator": indicator, "status": "NE", "reason": missing[s]})
                    continue
                r = auroc_ci(sig[s], y, clusters, pctx.boot, min_cases=min_cases)
                auroc_rows.append({"partition": split, "signal": s, "indicator": indicator, "auroc": r["estimate"], "ci_lower": r["ci_lower"],
                                   "ci_upper": r["ci_upper"], "n": r["n"], "n_positive": r["n_positive"], "n_negative": r["n_negative"],
                                   "status": r["status"], "reason": r.get("reason")})
            for s in BINARY:
                if s in missing:
                    binary_rows.append({"partition": split, "signal": s, "indicator": indicator, "status": "NE", "reason": missing[s]})
                    continue
                r = binary_signal(sig[s], y)
                row = {"partition": split, "signal": s, "indicator": indicator, **{k: r[k] for k in ("n", "tp", "fp", "fn", "tn")}, "status": "EVALUATED"}
                for m in ("sensitivity", "specificity", "ppv"):
                    row.update({m: r[m]["estimate"], f"{m}_ci_lower": r[m]["ci_lower"], f"{m}_ci_upper": r[m]["ci_upper"]})
                binary_rows.append(row)
        dec_rel, stab_rel = f"reconstructed/bhat_{split}_decoded.parquet", f"reconstructed/stability_d_model_{split}.parquet"
        if not (pctx.exists(dec_rel) and pctx.exists(stab_rel)):
            symbol_rows.append({"partition": split, "concept": "pooled", "status": "NE", "reason": f"{dec_rel} or {stab_rel} not recorded"})
            continue
        decoded = pctx.read_parquet(dec_rel).set_index("sample_id")
        stability = pctx.read_parquet(stab_rel).set_index("sample_id")
        cells = symbol_errors(cuts, concepts, part.observed(concepts), decoded, stability)
        for c in [*concepts, "pooled"]:
            sub = cells if c == "pooled" else cells[cells["concept"] == c]
            symbol_rows.append({"partition": split, "concept": c, **two_by_two(sub), "status": "EVALUATED"})
    paths = {"auroc": out / "signal_auroc_table.parquet", "binary": out / "binary_signal_table.parquet", "symbol": out / "symbol_error_table.parquet",
             "summary": out / "summary.json"}
    write_table(pd.DataFrame(auroc_rows), paths["auroc"], pctx)
    write_table(pd.DataFrame(binary_rows), paths["binary"], pctx)
    write_table(pd.DataFrame(symbol_rows), paths["symbol"], pctx)
    summary = clean({"analysis": ANALYSIS, "error": acfg["error"], "large_error_threshold": acfg["large_error_threshold"], "any_large": acfg["any_large"],
                     "signals": acfg["signals"], "symbol_errors": acfg["symbol_errors"], "development_thresholds": thresholds.to_dict(),
                     "development_n_cases": int(len(dev.ids)), "symbol_concepts": concepts, "cuts": f"thresholds/d_model_{profile}.json",
                     "included": splits, "not_executed": skipped, "signals_not_recorded": not_recorded,
                     "symbol_error_table": json.loads(pd.DataFrame(symbol_rows).to_json(orient="records"))})
    write_json(summary, paths["summary"], pctx)
    pctx.write_provenance(ANALYSIS, dict(pctx.verified_log), list(paths.values()),
                          {"included": splits, "not_executed": skipped, "symbol_concepts": concepts, "signals_not_recorded": not_recorded})
    return {"included": splits, "not_executed": skipped, "symbol_concepts": len(concepts)}
