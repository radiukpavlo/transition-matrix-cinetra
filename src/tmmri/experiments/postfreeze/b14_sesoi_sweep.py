"""B14: the smallest effect of interest in physical units and a decision sweep (EPC-6; ``configs/exploratory_postfreeze.yaml``
analyses.B14).

Translation: 0.10 * MAE_0,j, with MAE_0,j the patient-weighted mean-null MAE of each primary target on each partition
(physical units; the rows and cells of the rho estimate: non-critical-QC rows, OBSERVED cells, finite B-hat), beside
the ridge MAE and s_j.  Sweep: for each recorded hypothesis estimate (H-primary-1, H-primary-2, its 220-case sensitivity,
H-primary-3 and H4a on every partition where it is recorded) whether the recorded 95 % lower bound exceeds each
threshold - arithmetic on recorded intervals, no resampling.  The recorded decisions are quoted beside the sweep and
never changed.
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

ANALYSIS = "B14"


def null_mae(y: pd.DataFrame, pred: pd.DataFrame, mean: pd.Series, sd: pd.Series, units: pd.Series, targets: list[str]) -> list[dict[str, Any]]:
    """Per target: patient-weighted MAE of the mean null and of the ridge (rows where both errors are finite), s_j, rho_j."""

    from ...metrics import _cluster_codes
    from .estimates import _terms

    t = y.reindex(columns=targets).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    p = pred.reindex(index=y.index, columns=targets).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    z = np.tile(mean.reindex(targets).to_numpy(dtype=float), (len(y), 1))
    sdv = sd.reindex(targets).to_numpy(dtype=float)
    codes, n_units = _cluster_codes(units.reindex(y.index).to_numpy(dtype=object), len(y))
    m, zn, rho = _terms(np.abs(p - t), np.abs(z - t), sdv, codes, n_units, None)
    both = np.isfinite(np.abs(p - t)) & np.isfinite(t)
    return [{"target": c, "n": int(both[:, j].sum()), "reference_sd": float(sdv[j]), "mae_null": float(zn[j] * sdv[j]), "mae_ridge": float(m[j] * sdv[j]),
             "nmae_null": float(zn[j]), "nmae_ridge": float(m[j]), "rho_j": float(rho[j])} for j, c in enumerate(targets)]


def hypotheses(pctx: Any, included: list[str]) -> list[dict[str, Any]]:
    """The recorded hypothesis estimates with their recorded decisions (only partitions whose lock passed)."""

    out = []
    sequence = pctx.read_json("metrics/fixed_sequence.json") if "mms_external" in included and pctx.exists("metrics/fixed_sequence.json") else {}
    decided = {str(r.get("hypothesis_id")): r for r in sequence.get("results", [])}

    def add(hid: str, partition: str, rel: str, block: dict[str, Any], estimand: str, passed: Any, status: Any) -> None:
        out.append({"hypothesis_id": hid, "partition": partition, "estimand": estimand, "source": rel, "estimate": block.get("rho"),
                    "ci_lower": block.get("ci_lower"), "ci_upper": block.get("ci_upper"), "n_cases": block.get("n_cases"),
                    "recorded_status": status, "recorded_passed": passed})

    if "acdc_test" in included:
        rho = pctx.read_json("metrics/primary_rho_acdc_test.json")
        hid = str(rho.get("hypothesis_id") or "H-primary-1")
        add(hid, "acdc_test", "metrics/primary_rho_acdc_test.json", rho, "rho", decided.get(hid, {}).get("passed"), decided.get(hid, {}).get("status"))
    if "mms_external" in included:
        for rel, default in (("metrics/primary_rho_mms_external.json", "H-primary-2"), ("metrics/primary_rho_mms_external_never_analysed.json",
                                                                                         "H-primary-2-never-analysed")):
            if pctx.exists(rel):
                rho = pctx.read_json(rel)
                hid = str(rho.get("hypothesis_id") or default)
                add(hid, str(rho.get("partition") or "mms_external"), rel, rho, "rho", decided.get(hid, {}).get("passed"), decided.get(hid, {}).get("status"))
    if "mms2_external" in included and pctx.exists("metrics/h_primary_3.json"):
        h3 = pctx.read_json("metrics/h_primary_3.json")
        result = h3.get("result") or {}
        add(str(result.get("hypothesis_id") or "H-primary-3"), "mms2_external", "metrics/h_primary_3.json", result, "rho", result.get("passed"), result.get("status"))
    for split in EVALUATED:
        rel = f"metrics/confirmatory_h4/h4_{split}.json"
        if split in included and pctx.exists(rel):
            h4 = pctx.read_json(rel)
            decision = h4.get("family_decision") or {}
            passed = True if "H4a" in (decision.get("passed") or []) else (False if "H4a" in (decision.get("failed") or []) else None)
            add("H4a", split, rel, h4.get("h4a") or {}, "rho_EF", passed, (h4.get("h4a") or {}).get("status"))
    return out


def run(pctx: Any) -> dict[str, Any]:
    acfg = pctx.analysis_cfg(ANALYSIS)
    out = pctx.out_dir(ANALYSIS)
    sesoi = float(pctx.study.bootstrap["primary_test"]["smallest_effect_of_interest"])
    if abs(sesoi - 0.10) > 1e-12 or "0.10" not in str(acfg["translation"]):
        raise ValueError(f"{ANALYSIS}: the translated effect must be the registered smallest effect of interest 0.10")
    targets = [str(t) for t in pctx.lane_cfg["statistics"]["primary_targets"]]
    thresholds = [float(v) for v in acfg["sweep_thresholds"]]
    splits, skipped = included_partitions(pctx, ["acdc_dev", *EVALUATED], f"run-postfreeze {ANALYSIS}")
    sd, mean = dev_reference(pctx)
    translation = []
    for split in splits:
        part = load_partition(pctx, split)
        label = "acdc_dev_oof" if split == "acdc_dev" else split
        for row in null_mae(part.observed(targets), part.bhat, mean, sd, part.units, targets):
            translation.append({"partition": label, **row, "sesoi": sesoi, "threshold_physical_units": sesoi * row["mae_null"],
                                "threshold_normalized": sesoi * row["nmae_null"], "n_excluded_critical_qc": part.n_excluded_critical_qc})
    sweep = []
    recorded = hypotheses(pctx, splits)
    for h in recorded:
        lower = h["ci_lower"]
        largest = max((t for t in thresholds if lower is not None and float(lower) > t), default=None)
        for t in thresholds:
            sweep.append({**h, "threshold": t, "lower_bound_exceeds_threshold": None if lower is None else bool(float(lower) > t),
                          "largest_threshold_exceeded": largest})
    paths = {"translation": out / "translation_table.parquet", "sweep": out / "sweep_table.parquet", "summary": out / "summary.json"}
    write_table(pd.DataFrame(translation), paths["translation"], pctx)
    write_table(pd.DataFrame(sweep), paths["sweep"], pctx)
    labels = [("acdc_dev_oof" if s == "acdc_dev" else s) for s in splits]
    summary = clean({"analysis": ANALYSIS, "translation": acfg["translation"], "sweep_rule": acfg["sweep_rule"], "thresholds": thresholds, "sesoi": sesoi,
                     "decision_rule": "recorded 95 % lower bound > threshold (as the registered success rule)", "included": labels, "not_executed": skipped,
                     "hypotheses": recorded})
    write_json(summary, paths["summary"], pctx)
    pctx.write_provenance(ANALYSIS, dict(pctx.verified_log), list(paths.values()),
                          {"included": labels, "not_executed": skipped, "hypotheses": [f"{h['hypothesis_id']}@{h['partition']}" for h in recorded]})
    return {"included": labels, "not_executed": skipped, "hypotheses": len(recorded)}
