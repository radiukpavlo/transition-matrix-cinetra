"""B3 vendor-A profile (exploratory, post hoc): what distinguishes one M&Ms-1 vendor.

For the configured vendor of ``mms_external`` against the other vendors: the pathology mix (counts and proportions),
the per-target bias of the frozen reconstruction (mean normalized error with a cluster-bootstrap interval; primary
family) and the acquisition geometry against ACDC (in-plane spacing, slice thickness and increment, slice count, matrix,
FOV: median, quartiles and range).  Descriptive only.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from .common import describe, geometry_frame, manifest_path, mean_with_ci

OUTPUTS = ("pathology_mix.parquet", "pathology_mix.csv", "target_bias.parquet", "target_bias.csv", "geometry.parquet", "geometry.csv", "summary.json")


def plan(ctx: Any, cfg: Mapping[str, Any], acfg: Mapping[str, Any]) -> dict[str, Any]:
    from ...stages import _contracts
    from .common import external_partitions, partition_inputs

    run = ctx.require_run("run-exploratory")
    split = str(acfg["partition"])
    included, skipped = external_partitions(ctx, cfg, "run-exploratory B3", wanted=[split])
    inputs = {"B_ACDC": run.path / "semantics" / "B_ACDC.parquet", "split_manifest": _contracts(ctx) / "split_manifest.parquet"}
    acdc_manifest = manifest_path(ctx, "ACDC")
    if acdc_manifest is not None:
        inputs["staging_manifest_ACDC"] = acdc_manifest
    for s, ds in included.items():
        inputs.update(partition_inputs(ctx, s, ds))
    return {"inputs": inputs, "included": included, "not_executed": skipped}


def compute(ctx: Any, cfg: Mapping[str, Any], acfg: Mapping[str, Any], plan: Mapping[str, Any], out) -> dict[str, Any]:
    import json

    from ...runs import atomic_write_json
    from ...stages import _split_manifest, _write_table
    from .common import partition_errors

    run = ctx.require_run("run-exploratory")
    boot, min_cases = cfg["bootstrap"], int(cfg["min_cases_per_stratum"])
    vendor_a = str(acfg["vendor"])
    targets = ctx.study.primary_family()
    mix_rows, bias_rows, geo_rows = [], [], []
    summary: dict[str, Any] = {"analysis": "B3", "vendor": vendor_a, "included": plan["included"], "not_executed": plan["not_executed"]}
    geometry_cols = list(acfg["geometry"])
    split_manifest = _split_manifest(ctx)
    b_acdc = pd.read_parquet(run.path / "semantics" / "B_ACDC.parquet").set_index("sample_id")
    acdc_manifest = manifest_path(ctx, "ACDC")
    acdc_geom = geometry_frame(b_acdc, json.loads(acdc_manifest.read_text(encoding="utf-8")) if acdc_manifest is not None else None)
    for part in acfg.get("acdc_partitions", ["acdc_dev", "acdc_test"]):
        ids = [s for s in split_manifest.loc[split_manifest["partition"] == part, "sample_id"].astype(str) if s in acdc_geom.index]
        for col in geometry_cols:
            geo_rows.append({"group": f"ACDC:{part}", "variable": col, **describe(acdc_geom.loc[ids, col])})
    for split, ds in plan["included"].items():
        part = partition_errors(ctx, cfg, split, ds, targets)
        vendor, cohort, errors, clusters, geometry = part["vendor"], part["cohort"], part["errors"], part["clusters"], part["geometry"]
        groups = {vendor_a: [s for s in part["ids"] if vendor[s] == vendor_a], "other_vendors": [s for s in part["ids"] if vendor[s] != vendor_a]}
        pathology = cohort["pathology"].astype(object).where(cohort["pathology"].notna(), "<missing>") if "pathology" in cohort.columns \
            else pd.Series("<missing>", index=part["ids"])
        for g, ids in groups.items():
            counts = pathology.loc[ids].astype(str).value_counts().sort_index()
            for value, n in counts.items():
                mix_rows.append({"partition": split, "group": g, "pathology": value, "n": int(n), "proportion": n / len(ids) if ids else None})
            for t in targets:
                stat = mean_with_ci(errors.loc[ids, t], clusters, boot, min_cases)
                bias_rows.append({"partition": split, "group": g, "target": t, "n": stat["n"], "n_clusters": stat["n_clusters"], "bias": stat["estimate"],
                                  "ci_lower": stat["ci_lower"], "ci_upper": stat["ci_upper"], "status": stat["status"], "reason": stat["reason"]})
            for col in geometry_cols:
                geo_rows.append({"group": f"{split}:{g}", "variable": col, **describe(geometry.loc[ids, col])})
        summary[split] = {"n": {g: len(v) for g, v in groups.items()}, "n_excluded_critical_qc": part["n_excluded_critical_qc"]}
    mix = pd.DataFrame(mix_rows, columns=["partition", "group", "pathology", "n", "proportion"])
    bias = pd.DataFrame(bias_rows, columns=["partition", "group", "target", "n", "n_clusters", "bias", "ci_lower", "ci_upper", "status", "reason"])
    for c in ("proportion",):
        mix[c] = pd.to_numeric(mix[c], errors="coerce")
    for c in ("bias", "ci_lower", "ci_upper"):
        bias[c] = pd.to_numeric(bias[c], errors="coerce")
    geo = pd.DataFrame(geo_rows, columns=["group", "variable", "n", "median", "q25", "q75", "min", "max", "mean"])
    for c in ("median", "q25", "q75", "min", "max", "mean"):
        geo[c] = pd.to_numeric(geo[c], errors="coerce")
    _write_table(mix, out / "pathology_mix.parquet", csv=True)
    _write_table(bias, out / "target_bias.parquet", csv=True)
    _write_table(geo, out / "geometry.parquet", csv=True)
    summary["note"] = "descriptive; vendor and centre are confounded in M&Ms-1"
    atomic_write_json(out / "summary.json", summary, allow_nan=True)
    return {"included": list(plan["included"]), "not_executed": plan["not_executed"]}
