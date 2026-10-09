"""B2 external volume-error decomposition (exploratory, post hoc).

The external reconstruction error of each volume target (B-hat minus measured B, divided by the acdc_dev SD; critical-QC
rows excluded) is regressed by ordinary least squares on z-scored acquisition covariates (in-plane spacing, slice
thickness, slice count, matrix and FOV) per partition: a descriptive association with cluster-bootstrap percentile
intervals, never a predictor and never fed back.  Per vendor and target: bias (mean normalized error) against spread
(its SD), each with a cluster-bootstrap interval.  Strata below the reporting threshold keep point estimates (NE).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .common import cluster_bootstrap, mean_with_ci, weighted_sd

OUTPUTS = ("error_regression.parquet", "error_regression.csv", "bias_spread.parquet", "bias_spread.csv", "summary.json")


def ols_terms(y: np.ndarray, x: pd.DataFrame, clusters: np.ndarray, *, n_boot: int, seed: int, level: float, min_cases: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """OLS of ``y`` on z-scored columns of ``x`` (rows finite in both); coefficients per covariate SD with cluster-bootstrap
    percentile intervals (weighted least squares on the bootstrap multiplicities)."""

    from ...metrics import _cluster_codes, _cluster_draw_weights

    ok = np.isfinite(y) & np.isfinite(x.to_numpy(dtype=float)).all(axis=1)
    y, x, clusters = y[ok], x.loc[ok], np.asarray(clusters, dtype=object)[ok]
    sd = x.std(ddof=1)
    usable = [c for c in x.columns if np.isfinite(sd[c]) and sd[c] > 0]
    dropped = [c for c in x.columns if c not in usable]
    info: dict[str, Any] = {"n": int(len(y)), "n_clusters": int(pd.Series(clusters).astype(str).nunique()) if len(y) else 0,
                            "covariates": usable, "dropped_covariates": dropped}
    if len(y) <= len(usable) + 1 or not usable:
        reason = "NO_VARYING_COVARIATE" if not usable else "TOO_FEW_ROWS_FOR_REGRESSION"
        info.update({"status": "NE", "reason": reason, "r_squared": None, "condition_number": None})
        return pd.DataFrame(columns=["term", "coef", "ci_lower", "ci_upper"]), info
    z = (x[usable] - x[usable].mean()) / sd[usable]
    design = np.column_stack([np.ones(len(y)), z.to_numpy(dtype=float)])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    resid = y - design @ beta
    tss = float(np.sum((y - y.mean()) ** 2))
    info.update({"r_squared": 1.0 - float(np.sum(resid ** 2)) / tss if tss > 0 else None, "condition_number": float(np.linalg.cond(design)),
                 "status": "NE" if len(y) < min_cases else "EVALUATED", "reason": "STRATUM_BELOW_MIN_CASES" if len(y) < min_cases else None})
    terms = ["intercept", *usable]
    lo = hi = [None] * len(terms)
    if len(y) >= min_cases and n_boot > 0:
        codes, n_c = _cluster_codes(clusters, len(y))
        boot = []
        for ss in np.random.SeedSequence(int(seed)).spawn(int(n_boot)):
            w = np.sqrt(_cluster_draw_weights(np.random.default_rng(ss), codes, n_c))
            boot.append(np.linalg.lstsq(design * w[:, None], y * w, rcond=None)[0])
        arr = np.asarray(boot)
        alpha = (1.0 - level) / 2.0
        lo, hi = list(np.quantile(arr, alpha, axis=0)), list(np.quantile(arr, 1.0 - alpha, axis=0))
    table = pd.DataFrame({"term": terms, "coef": beta, "ci_lower": [None if v is None else float(v) for v in lo],
                          "ci_upper": [None if v is None else float(v) for v in hi]})
    return table, info


def plan(ctx: Any, cfg: Mapping[str, Any], acfg: Mapping[str, Any]) -> dict[str, Any]:
    from ...stages import _contracts
    from .common import external_partitions, partition_inputs

    run = ctx.require_run("run-exploratory")
    included, skipped = external_partitions(ctx, cfg, "run-exploratory B2")
    inputs = {"B_ACDC": run.path / "semantics" / "B_ACDC.parquet", "split_manifest": _contracts(ctx) / "split_manifest.parquet"}
    for split, ds in included.items():
        inputs.update(partition_inputs(ctx, split, ds))
    return {"inputs": inputs, "included": included, "not_executed": skipped}


def compute(ctx: Any, cfg: Mapping[str, Any], acfg: Mapping[str, Any], plan: Mapping[str, Any], out) -> dict[str, Any]:
    from ...runs import atomic_write_json
    from ...stages import _write_table
    from .common import partition_errors

    boot = cfg["bootstrap"]
    min_cases = int(cfg["min_cases_per_stratum"])
    targets = [t for t in cfg["volume_targets"] if t in ctx.study.transition_targets()]
    covariates = list(acfg["covariates"])
    reg_rows, bias_rows, summary = [], [], {}
    for split, ds in plan["included"].items():
        part = partition_errors(ctx, cfg, split, ds, targets)
        errors, geometry, clusters, vendor = part["errors"], part["geometry"], part["clusters"], part["vendor"]
        summary[split] = {"n_rows": len(part["ids"]), "n_excluded_critical_qc": part["n_excluded_critical_qc"],
                          "covariates_available": [c for c in covariates if geometry[c].notna().any()], "targets": {}}
        for t in targets:
            table, info = ols_terms(errors[t].to_numpy(dtype=float), geometry[covariates], clusters.to_numpy(dtype=object),
                                    n_boot=int(boot["replicates"]), seed=int(boot["seed"]), level=float(boot["level"]), min_cases=min_cases)
            summary[split]["targets"][t] = info
            for r in table.itertuples(index=False):
                reg_rows.append({"partition": split, "target": t, "term": r.term, "coef_per_covariate_sd": float(r.coef), "ci_lower": r.ci_lower,
                                 "ci_upper": r.ci_upper, "n": info["n"], "n_clusters": info["n_clusters"], "r_squared": info["r_squared"],
                                 "condition_number": info["condition_number"], "status": info["status"]})
            for v in ["all", *sorted(set(vendor))]:
                rows = errors.index if v == "all" else errors.index[vendor.reindex(errors.index).to_numpy() == v]
                e = errors.loc[rows, t]
                bias = mean_with_ci(e, clusters, boot, min_cases)
                ok = np.isfinite(e.to_numpy(dtype=float))
                below = int(ok.sum()) < min_cases
                spread, s_lo, s_hi = cluster_bootstrap(e.to_numpy(dtype=float)[ok], clusters.reindex(rows).to_numpy(dtype=object)[ok], weighted_sd,
                                                       n_boot=0 if below else int(boot["replicates"]), seed=int(boot["seed"]), level=float(boot["level"]))
                bias_rows.append({"partition": split, "vendor": v, "target": t, "n": bias["n"], "n_clusters": bias["n_clusters"],
                                  "bias": bias["estimate"], "bias_ci_lower": bias["ci_lower"], "bias_ci_upper": bias["ci_upper"],
                                  "spread_sd": spread if np.isfinite(spread) else None, "spread_ci_lower": s_lo, "spread_ci_upper": s_hi,
                                  "abs_bias_over_spread": abs(bias["estimate"]) / spread if bias["estimate"] is not None and np.isfinite(spread) and spread > 0 else None,
                                  "status": bias["status"], "reason": bias["reason"]})
    reg = pd.DataFrame(reg_rows, columns=["partition", "target", "term", "coef_per_covariate_sd", "ci_lower", "ci_upper", "n", "n_clusters",
                                          "r_squared", "condition_number", "status"])
    bias = pd.DataFrame(bias_rows, columns=["partition", "vendor", "target", "n", "n_clusters", "bias", "bias_ci_lower", "bias_ci_upper", "spread_sd",
                                            "spread_ci_lower", "spread_ci_upper", "abs_bias_over_spread", "status", "reason"])
    for frame, cols in ((reg, ["coef_per_covariate_sd", "ci_lower", "ci_upper", "r_squared", "condition_number"]),
                        (bias, ["bias", "bias_ci_lower", "bias_ci_upper", "spread_sd", "spread_ci_lower", "spread_ci_upper", "abs_bias_over_spread"])):
        for c in cols:
            frame[c] = pd.to_numeric(frame[c], errors="coerce")
    _write_table(reg, out / "error_regression.parquet", csv=True)
    _write_table(bias, out / "bias_spread.parquet", csv=True)
    atomic_write_json(out / "summary.json", {"analysis": "B2", "included": plan["included"], "not_executed": plan["not_executed"],
                                             "error_definition": cfg["error_definition"], "covariates": covariates, "partitions": summary,
                                             "model": acfg["model"]}, allow_nan=True)
    return {"included": list(plan["included"]), "not_executed": plan["not_executed"]}
