"""B11: descriptor-family ablations of the typed transition (EPC-1; ``configs/exploratory_postfreeze.yaml`` analyses.B11).

The primary typed ridge refitted on the 17 fraction descriptors alone (``ridge_typed_fractions17``) and on the 40
intensity descriptors alone (``ridge_typed_intensities40``): the families are regular expressions on the formal
descriptor names, and everything except the input columns is the primary transition
(:mod:`tmmri.experiments.postfreeze.transition_variants`).  Fitted on acdc_dev only, frozen in the new run root, then
evaluated once on every included partition beside the recorded primary ridge: rho, rho_EF, rho of the volumes, rho_j,
per-target MAE, bias and limits of agreement, and paired differences against the primary ridge (same cases, same
resampled clusters).
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from .common import (
    EVALUATED,
    clean,
    formal_a,
    included_partitions,
    write_json,
    write_table,
)
from .transition_variants import (
    dev_design,
    difference_rows,
    evaluate_variants,
    fit_and_freeze,
)

ANALYSIS = "B11"


def family_columns(families: dict[str, str]) -> dict[str, list[str]]:
    """Descriptor columns of each family (``formal_descriptor_names`` order); the trailing count of the family name
    (``fractions17``) must match, else the analysis refuses."""

    from ...formal_features import formal_descriptor_names

    names = formal_descriptor_names()
    out = {}
    for fam, pattern in families.items():
        cols = [c for c in names if re.search(str(pattern), c)]
        declared = re.search(r"(\d+)$", fam)
        if declared and int(declared.group(1)) != len(cols):
            raise ValueError(f"{ANALYSIS}: family {fam} matches {len(cols)} descriptors, not {declared.group(1)}")
        out[fam] = cols
    return out


def run(pctx: Any) -> dict[str, Any]:
    from ...formal_features import formal_descriptor_names

    acfg = pctx.analysis_cfg(ANALYSIS)
    out = pctx.out_dir(ANALYSIS)
    columns = family_columns(dict(acfg["families"]))
    registry = {str(e["id"]): e for e in pctx.lane_cfg.get("estimators", [])}
    ids, a, b, folds, inner_fn = dev_design(pctx)
    names = formal_descriptor_names()
    variants = {}
    for fam, cols in columns.items():
        estimator_id = f"ridge_typed_{fam}"
        if registry.get(estimator_id, {}).get("analysis") != ANALYSIS:
            raise ValueError(f"{ANALYSIS}: estimator {estimator_id} is not registered for {ANALYSIS} in configs/exploratory_postfreeze.yaml estimators")
        x = a[:, [names.index(c) for c in cols]]
        variants[estimator_id] = {**fit_and_freeze(pctx, out, estimator_id, x, b, ids, folds, inner_fn, cols), "columns": cols}
    splits, skipped = included_partitions(pctx, ["acdc_dev", *EVALUATED], f"run-postfreeze {ANALYSIS}")
    cache: dict[str, pd.DataFrame] = {}

    def design(split: str, ds: str, ids_p: list[str], entry: dict[str, Any]) -> np.ndarray:
        if ds not in cache:
            cache[ds] = formal_a(pctx, ds)
        return cache[ds].reindex(ids_p)[entry["columns"]].to_numpy(dtype=float)

    ev = evaluate_variants(pctx, splits, variants, design)
    paths = {"rho": out / "rho.parquet", "per_target": out / "per_target_mae_bias_loa.parquet",
             "diff": out / "paired_difference_vs_primary_ridge.parquet", "summary": out / "summary.json"}
    write_table(ev["rho"], paths["rho"], pctx)
    write_table(ev["per_target"], paths["per_target"], pctx)
    write_table(difference_rows(ev["results"]), paths["diff"], pctx)
    labels = [("acdc_dev_oof" if s == "acdc_dev" else s) for s in splits]
    summary = clean({"analysis": ANALYSIS, "families": {f: {"pattern": acfg["families"][f], "n_columns": len(c), "columns": c} for f, c in columns.items()},
                     "fitting": acfg["fitting"], "frozen": {k: v["frozen"] for k, v in variants.items()}, "included": labels, "not_executed": skipped,
                     "null": "acdc_dev mean of the non-critical fit rows (as the recorded rho)", "results": ev["results"]})
    write_json(summary, paths["summary"], pctx)
    outputs = [p for v in variants.values() for p in v["outputs"]] + list(paths.values())
    pctx.write_provenance(ANALYSIS, dict(pctx.verified_log), outputs,
                          {"included": labels, "not_executed": skipped, "estimators": sorted(variants),
                           "frozen_sha256": {k: v["frozen"]["files"] for k, v in variants.items()}})
    return {"included": labels, "not_executed": skipped, "estimators": sorted(variants)}
