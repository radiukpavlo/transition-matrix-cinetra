"""B4 cut stability (exploratory, post hoc; acdc_dev only).

The WEDD cuts of the rulebook substrate (nested OOF B-hat against ``d_model``; ``stages._rule_substrate``) are refitted
on the cluster bootstrap of the threshold stability gate (:func:`tmmri.wedd.stability_gate`, 500 replicates, seed
20260928) for each ``max_bins`` of the WEDD sensitivity grid, and the recurrence of every cut is related to the class
separation of its feature (eta squared and the Kruskal-Wallis epsilon squared of the values by ``d_model`` class).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr

OUTPUTS = ("recurrence.parquet", "recurrence.csv", "feature_summary.parquet", "feature_summary.csv", "summary.json")


def class_separation(values: pd.DataFrame, decision: pd.Series) -> pd.DataFrame:
    """Per feature: eta squared (between-class share of the sum of squares) and the Kruskal-Wallis epsilon squared."""

    rows = []
    d = pd.Series(decision).astype(str).reindex(values.index)
    for col in [c for c in values.columns if not str(c).endswith(("__state", "__evidence"))]:
        x = pd.to_numeric(values[col], errors="coerce")
        ok = x.notna() & d.notna()
        x, y = x[ok].to_numpy(dtype=float), d[ok].to_numpy()
        groups = [x[y == c] for c in sorted(set(y))]
        groups = [g for g in groups if g.size]
        if x.size < 3 or len(groups) < 2 or np.ptp(x) == 0:
            rows.append({"feature": col, "n": int(x.size), "n_classes": len(groups), "eta_squared": np.nan, "kruskal_epsilon_squared": np.nan})
            continue
        sst = float(np.sum((x - x.mean()) ** 2))
        ssb = float(sum(g.size * (g.mean() - x.mean()) ** 2 for g in groups))
        h = float(kruskal(*groups).statistic)
        rows.append({"feature": col, "n": int(x.size), "n_classes": len(groups), "eta_squared": ssb / sst if sst > 0 else np.nan,
                     "kruskal_epsilon_squared": h / (x.size - 1)})
    return pd.DataFrame(rows)


def plan(ctx: Any, cfg: Mapping[str, Any], acfg: Mapping[str, Any]) -> dict[str, Any]:
    from ...stages import _contracts
    from .common import check_recorded

    run = ctx.require_run("run-exploratory")
    p = run.path
    oof, dm = p / "oof" / "bhat_oof_ACDC_dev.parquet", p / "formal" / "d_model_ACDC_dev.parquet"
    check_recorded(run, "fit-transition", [oof], "run-exploratory B4")
    check_recorded(run, "fit-predictor", [dm], "run-exploratory B4")
    return {"inputs": {"oof_bhat": oof, "d_model": dm, "B_ACDC": p / "semantics" / "B_ACDC.parquet", "cohort_ACDC": _contracts(ctx) / "cohort_ACDC.parquet",
                       "split_manifest": _contracts(ctx) / "split_manifest.parquet"}, "included": {"acdc_dev": "ACDC"}, "not_executed": {}}


def compute(ctx: Any, cfg: Mapping[str, Any], acfg: Mapping[str, Any], plan: Mapping[str, Any], out) -> dict[str, Any]:
    from ...runs import atomic_write_json
    from ...stages import _cluster_ids, _rule_substrate, _rulebooks, _wedd_cfg_variant, _write_table
    from ...wedd import stability_gate

    study = ctx.study
    rb = str(acfg.get("rulebook", "d_model"))
    values, decision = _rule_substrate(ctx, _rulebooks(ctx)[rb])
    profile = str(study.wedd["primary_profile"])
    seed = int(study.bootstrap["threshold_bootstrap"]["seed"])
    replicates = int(acfg["replicates"])
    groups = _cluster_ids(ctx, values.index).to_numpy(dtype=object)
    separation = class_separation(values, decision)
    tables = []
    for nb in [int(v) for v in acfg["max_bins"]]:
        wedd_cfg = _wedd_cfg_variant(study.wedd, profile, {"max_bins": nb})
        gate = stability_gate(values, decision, profile, wedd_cfg, seed=seed, n_replicates=replicates, n_jobs=ctx.jobs, groups=groups)
        tables.append(gate.assign(max_bins=nb, rulebook=rb))
    rec = pd.concat(tables, ignore_index=True)
    feats = []
    for (nb, f), g in rec.groupby(["max_bins", "feature"], sort=True):
        cut_rows = g[g["cut"].notna()]
        feats.append({"max_bins": int(nb), "feature": f, "n_cuts": int(len(cut_rows)),
                      "mean_recurrence": float(cut_rows["recurrence"].mean()) if len(cut_rows) else np.nan,
                      "min_recurrence": float(cut_rows["recurrence"].min()) if len(cut_rows) else np.nan,
                      "retained_fraction": float(cut_rows["retained"].astype(bool).mean()) if len(cut_rows) else np.nan})
    summary_table = pd.DataFrame(feats).merge(separation, on="feature", how="left")
    correlations = {}
    for nb, g in summary_table.groupby("max_bins", sort=True):
        ok = g["mean_recurrence"].notna() & g["eta_squared"].notna()
        rho = float(spearmanr(g.loc[ok, "mean_recurrence"], g.loc[ok, "eta_squared"]).statistic) if ok.sum() >= 3 else None
        correlations[str(int(nb))] = {"n_features_with_cuts": int(ok.sum()), "spearman_mean_recurrence_vs_eta_squared": rho,
                                      "mean_recurrence": float(g["mean_recurrence"].mean()) if g["mean_recurrence"].notna().any() else None}
    _write_table(rec, out / "recurrence.parquet", csv=True)
    _write_table(summary_table, out / "feature_summary.parquet", csv=True)
    atomic_write_json(out / "summary.json", {"analysis": "B4", "rulebook": rb, "profile": profile, "replicates": replicates, "seed": seed,
                                             "n_fit_rows": int(len(values)), "by_max_bins": correlations,
                                             "recurrence_min_of_gate": float(study.wedd["stability_gate"]["recurrence_min"])}, allow_nan=True)
    return {"included": ["acdc_dev"], "max_bins": [int(v) for v in acfg["max_bins"]]}
