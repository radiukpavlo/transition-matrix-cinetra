"""B1 shift attribution (exploratory, post hoc): which formal descriptors carry the acdc_dev -> external shift.

Per descriptor of the frozen-standardized formal A (the input of ``audit-shift``): the AUROC of the descriptor value for
external against acdc_dev membership, its distance from 0.5 and the two-sample Kolmogorov-Smirnov distance, pooled and
per vendor.  Group drop: the label-free domain classifier of ``audit-shift`` (``configs/bootstrap.yaml`` ``shift_audit``,
:func:`tmmri.classification.domain_shift_audit`) refitted without each descriptor group; the AUROC drop attributes the
shift to that group.  Domain membership only: no B, no Y; never a selection criterion.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from ...classification import _auroc_from_scores, domain_shift_audit


def descriptor_shift(z_ref: pd.DataFrame, z_cmp: pd.DataFrame, *, min_cases: int) -> pd.DataFrame:
    """Per descriptor: AUROC (comparison scored positive), |AUROC - 0.5| and the KS distance; NE flag below ``min_cases``."""

    rows = []
    y = np.r_[np.zeros(len(z_ref), dtype=int), np.ones(len(z_cmp), dtype=int)]
    for col in z_ref.columns:
        a, b = z_ref[col].to_numpy(dtype=float), z_cmp[col].to_numpy(dtype=float)
        ok_a, ok_b = a[np.isfinite(a)], b[np.isfinite(b)]
        auc = _auroc_from_scores(y, np.r_[a, b]) if ok_a.size == a.size and ok_b.size == b.size else float("nan")
        ks = float(ks_2samp(ok_a, ok_b).statistic) if ok_a.size and ok_b.size else float("nan")
        rows.append({"descriptor": col, "n_reference": int(ok_a.size), "n_comparison": int(ok_b.size), "auroc": auc,
                     "abs_auroc_minus_half": abs(auc - 0.5) if np.isfinite(auc) else np.nan, "ks_distance": ks,
                     "descriptive_only": bool(len(z_cmp) < int(min_cases)),
                     "status": "NE" if len(z_cmp) < int(min_cases) else "EVALUATED"})
    out = pd.DataFrame(rows)
    out["rank_abs_auroc"] = out["abs_auroc_minus_half"].rank(ascending=False, method="min")
    return out


def group_columns(columns: list[str], groups: Mapping[str, str]) -> dict[str, list[str]]:
    return {name: [c for c in columns if re.search(pattern, c)] for name, pattern in groups.items()}


def group_drop(z_ref: pd.DataFrame, z_cmp: pd.DataFrame, ref_ids: list[str], cmp_ids: list[str], groups: Mapping[str, str],
               shift_cfg: Mapping[str, Any], *, n_boot: int, seed: int, min_cases: int, ref_clusters: list[str],
               cmp_clusters: list[str]) -> pd.DataFrame:
    """Domain-classifier AUROC with every descriptor and without each group (same folds, seed and bootstrap)."""

    def audit(cols: list[str]) -> dict[str, Any]:
        record, _ = domain_shift_audit(z_ref[cols].to_numpy(dtype=float), z_cmp[cols].to_numpy(dtype=float), ref_ids, cmp_ids, dict(shift_cfg),
                                       n_boot=n_boot, seed=seed, min_cases=min_cases, reference_clusters=ref_clusters, comparison_clusters=cmp_clusters)
        return record

    columns = list(z_ref.columns)
    full = audit(columns)
    rows = [{"group": "none_dropped", "n_dropped": 0, "n_kept": len(columns), "auroc": full.get("auroc"), "ci_lower": full.get("ci_lower"),
             "ci_upper": full.get("ci_upper"), "auroc_drop": 0.0, "status": full.get("status")}]
    for name, cols in group_columns(columns, groups).items():
        kept = [c for c in columns if c not in set(cols)]
        if not cols or not kept:
            rows.append({"group": name, "n_dropped": len(cols), "n_kept": len(kept), "auroc": None, "ci_lower": None, "ci_upper": None,
                         "auroc_drop": None, "status": "NE", "reason": "EMPTY_GROUP" if not cols else "NO_DESCRIPTOR_LEFT"})
            continue
        rec = audit(kept)
        drop = None if rec.get("auroc") is None or full.get("auroc") is None else float(full["auroc"]) - float(rec["auroc"])
        rows.append({"group": name, "n_dropped": len(cols), "n_kept": len(kept), "auroc": rec.get("auroc"), "ci_lower": rec.get("ci_lower"),
                     "ci_upper": rec.get("ci_upper"), "auroc_drop": drop, "status": rec.get("status")})
    return pd.DataFrame(rows)


OUTPUTS = ("descriptor_shift.parquet", "descriptor_shift.csv", "group_drop.parquet", "group_drop.csv", "summary.json")


def plan(ctx: Any, cfg: Mapping[str, Any], acfg: Mapping[str, Any]) -> dict[str, Any]:
    from ...stages import _contracts
    from .common import external_partitions

    run = ctx.require_run("run-exploratory")
    included, skipped = external_partitions(ctx, cfg, "run-exploratory B1")
    p = run.path
    inputs = {"A_ACDC": p / "formal" / "A_ACDC.parquet", "B_ACDC": p / "semantics" / "B_ACDC.parquet",
              "transition_model": p / "transitions" / "transition_model.json", "split_manifest": _contracts(ctx) / "split_manifest.parquet"}
    for split, ds in included.items():
        inputs.update({f"A_{ds}": p / "formal" / f"A_{ds}.parquet", f"B_{ds}": p / "semantics" / f"B_{ds}.parquet",
                       f"cohort_{ds}": _contracts(ctx) / f"cohort_{ds}.parquet", f"evaluate_{split}_record": run.record_path(f"evaluate:{split}")})
    return {"inputs": inputs, "included": included, "not_executed": skipped}


def compute(ctx: Any, cfg: Mapping[str, Any], acfg: Mapping[str, Any], plan: Mapping[str, Any], out) -> dict[str, Any]:
    """Stage body of B1 (see :func:`tmmri.experiments.exploratory.run_exploratory`)."""

    import json

    from ...formal_features import formal_descriptor_names
    from ...runs import atomic_write_json
    from ...stages import _cluster_ids, _cohort, _split_manifest, _write_table, critical_qc_mask, methods_api

    run = ctx.require_run("run-exploratory")
    study = ctx.study
    included, skipped = dict(plan["included"]), dict(plan["not_executed"])
    model = methods_api("tmmri.transition_fit", "TransitionModel").from_dict(
        json.loads((run.path / "transitions" / "transition_model.json").read_text(encoding="utf-8")))
    names = formal_descriptor_names()
    kept_names = [n for n, k in zip(names, model.standardizer.keep) if k]
    split = _split_manifest(ctx)
    boot = cfg["bootstrap"]
    min_cases = int(cfg["min_cases_per_stratum"])

    def usable(ds: str, partition: str) -> tuple[list[str], pd.DataFrame]:
        a = pd.read_parquet(run.path / "formal" / f"A_{ds}.parquet").set_index("sample_id")
        b = pd.read_parquet(run.path / "semantics" / f"B_{ds}.parquet").set_index("sample_id")
        ids = [s for s in sorted(split.loc[split["partition"] == partition, "sample_id"].astype(str)) if s in a.index and s in b.index]
        if not ids and partition != "acdc_dev":    # partitions outside the v6 split manifest: the recorded evaluation rows
            ids = sorted(pd.read_parquet(run.path / "reconstructed" / f"bhat_{partition}.parquet")["sample_id"].astype(str))
            ids = [s for s in ids if s in a.index and s in b.index]
        x = a.loc[ids, names].to_numpy(dtype=float)
        ok = np.isfinite(x).all(axis=1) & ~critical_qc_mask(b.loc[ids], study).to_numpy(dtype=bool)
        ids = [ids[i] for i in np.flatnonzero(ok)]
        return ids, pd.DataFrame(model.standardizer.transform(x[ok]), index=ids, columns=kept_names)

    ref_ids, z_ref = usable("ACDC", str(acfg.get("reference_partition", "acdc_dev")))
    shift_cfg = study.bootstrap["shift_audit"]
    tables, drops, summary = [], [], {}
    for part, ds in included.items():
        cmp_ids, z_cmp = usable(ds, part)
        vendor_col = str(cfg["external_partitions"][part].get("vendor_column", "vendor"))
        cohort = _cohort(ctx, ds).set_index("sample_id")
        vendor = cohort.reindex(cmp_ids)[vendor_col].astype(str) if vendor_col in cohort.columns else pd.Series("<missing>", index=cmp_ids)
        strata = [("pooled", "all", cmp_ids)] + [("vendor", v, [s for s in cmp_ids if vendor[s] == v]) for v in sorted(set(vendor))]
        summary[part] = {"n_reference": len(ref_ids), "n_comparison": len(cmp_ids), "strata": {}}
        for kind, group, ids in strata:
            t = descriptor_shift(z_ref, z_cmp.loc[ids], min_cases=min_cases).assign(partition=part, stratum_variable=kind, stratum=group)
            tables.append(t)
            d = group_drop(z_ref, z_cmp.loc[ids], ref_ids, ids, acfg["group_drop"]["groups"], shift_cfg, n_boot=int(boot["replicates"]),
                           seed=int(boot["seed"]), min_cases=min_cases, ref_clusters=list(_cluster_ids(split, ref_ids)),
                           cmp_clusters=list(_cluster_ids(split, ids))).assign(partition=part, stratum_variable=kind, stratum=group)
            drops.append(d)
            top = t.sort_values(["abs_auroc_minus_half", "descriptor"], ascending=[False, True]).head(5)
            dropped = d.loc[d["group"] != "none_dropped"].assign(_drop=lambda f: pd.to_numeric(f["auroc_drop"], errors="coerce"))
            summary[part]["strata"][f"{kind}:{group}"] = {"n": len(ids), "top_descriptors_by_abs_auroc": top["descriptor"].tolist(),
                                                          "largest_group_drop": dropped.dropna(subset=["_drop"]).sort_values(["_drop", "group"], ascending=[False, True])["group"].head(1).tolist()}
    columns = ["descriptor", "n_reference", "n_comparison", "auroc", "abs_auroc_minus_half", "ks_distance", "descriptive_only", "status", "rank_abs_auroc",
               "partition", "stratum_variable", "stratum"]
    per = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=columns)
    grp = pd.concat(drops, ignore_index=True) if drops else pd.DataFrame(columns=["group", "partition", "stratum_variable", "stratum"])
    _write_table(per, out / "descriptor_shift.parquet", csv=True)
    for col in ("auroc", "ci_lower", "ci_upper", "auroc_drop"):
        if col in grp.columns:
            grp[col] = pd.to_numeric(grp[col], errors="coerce")
    _write_table(grp, out / "group_drop.parquet", csv=True)
    atomic_write_json(out / "summary.json", {"analysis": "B1", "included": included, "not_executed": skipped, "partitions": summary,
                                             "standardization": "frozen transition standardizer (kept descriptors)", "label_free": True}, allow_nan=True)
    return {"included": list(included), "not_executed": skipped}
