"""B26: robust agreement and outlier description (EPC-6, readiness review R5; ``configs/exploratory_postfreeze_v603.yaml``
analyses.B26).

* **Differences.**  Per primary target and evaluated partition, d = B-hat - B in physical units: the recorded raw
  frozen-bundle reconstruction against the OBSERVED cells of measured B on the non-critical-QC rows.
* **Robust agreement.**  The median of d and its 2.5th and 97.5th percentiles (numpy linear interpolation), each with
  the cluster bootstrap of the lane (2,000 replicates, seed 20260928, unit cluster_id; below 20 cases the interval is
  NE), beside the recorded Bland-Altman bias and limits (``metrics/per_target_<p>.parquet`` of the frozen run).
* **Outliers.**  |d - median(d)| > 3 x 1.4826 x MAD(d), per target and partition; counts.
* **Description.**  Outliers against the other scored cases of the same target and partition by vendor and center
  (M&Ms partitions; the cohort tables of the frozen run; M&Ms-2 records no center), slice count, voxel volume and imaged
  volume (staging manifests, hash-checked against ``build-semantics:<DS>``, as B12).  Only cells of at least five cases
  are described: a category with 1-4 cases in either group is merged into ``other`` in both groups (one mapping for the
  outliers and the comparison group), and when a merged ``other`` cell is still 1-4 in either group the attribute is
  suppressed for both groups, so no outlier cell can be recovered by subtraction; a continuous attribute is described (median and
  interquartile range) only for a group of at least five cases with a value.  No output carries a case or subject
  identifier.
* **Geometry check.**  For each outlier, whether it is still an outlier under the geometry-aware transition of B12
  (``postfreeze/B12/models/ridge_typed_formal57_plus_geometry3`` of the second source run, applied as frozen to the 57
  descriptors and the three log geometry covariates) by the same rule on its differences; counts.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .b12_geometry import COVARIATES, ESTIMATOR as B12_ESTIMATOR, geometry, staging_geometry
from .common import COMMAND, clean, formal_a, included_partitions, load_partition, write_json, write_table
from .estimates import BELOW_MIN_CASES

ANALYSIS = "B26"
MAD_SCALE = 1.4826
MAD_MULTIPLIER = 3.0
OTHER = "other"
SUPPRESSED = "SUPPRESSED_BELOW_MIN_CELL"
CATEGORICAL = {"vendor": "vendor", "centre": "centre"}            # cohort-table columns (M&Ms partitions)
CONTINUOUS = ("slice_count", "voxel_volume_ml", "imaged_volume_ml")
IDENTIFIER_COLUMNS = ("sample_id", "case_id", "patient_id", "subject_id", "cluster_id", "duplicate_pair_id", "anchor")


def outliers(d: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    """The outlier flag of every finite difference (|d - median| > 3 x 1.4826 x MAD) and the rule's quantities."""

    v = pd.to_numeric(d, errors="coerce").dropna()
    if v.empty:
        return pd.Series(False, index=v.index, dtype=bool), {"median": None, "mad": None, "threshold": None}
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    threshold = MAD_MULTIPLIER * MAD_SCALE * mad
    return (v - med).abs() > threshold, {"median": med, "mad": mad, "threshold": threshold}


def describe_categorical(values: pd.Series, flag: pd.Series, min_cell: int, **labels: Any) -> list[dict[str, Any]]:
    """Counts per category for outliers and the other scored cases, with one category mapping for both groups: a
    category is kept when its count is 0 or at least ``min_cell`` in each group, every other category is merged into
    ``other``; when ``other`` holds 1 to ``min_cell - 1`` cases in either group the attribute is suppressed for both
    groups, so no outlier cell can be recovered by subtraction from the category totals.  Zero cells are not listed."""

    groups = {"outlier": flag[flag].index, "other_cases": flag[~flag].index}
    vals = {g: values.reindex(m).dropna().astype(str) for g, m in groups.items()}
    counts = pd.DataFrame({g: v.value_counts() for g, v in vals.items()}).reindex(columns=list(groups)).fillna(0).astype(int).sort_index()
    small = ((counts > 0) & (counts < min_cell)).any(axis=1)
    kept, merged = counts[~small], counts[small].sum(axis=0)
    suppressed = bool(((merged > 0) & (merged < min_cell)).any())
    rows = []
    for group, members in groups.items():
        base = {**labels, "group": group, "n_group": int(len(members)), "n_with_value": int(len(vals[group]))}
        if suppressed:
            rows.append({**base, "category": None, "n": None, "status": SUPPRESSED})
            continue
        rows += [{**base, "category": str(cat), "n": int(n), "status": "DESCRIBED"} for cat, n in kept[group].items() if n > 0]
        if int(merged.get(group, 0)):
            rows.append({**base, "category": OTHER, "n": int(merged[group]), "status": "DESCRIBED"})
    return rows


def describe_continuous(values: pd.Series, flag: pd.Series, min_cell: int, **labels: Any) -> list[dict[str, Any]]:
    """Median and interquartile range per group with at least ``min_cell`` finite values, else suppressed."""

    rows = []
    for group, members in (("outlier", flag[flag].index), ("other_cases", flag[~flag].index)):
        vals = pd.to_numeric(values.reindex(members), errors="coerce").dropna()
        base = {**labels, "group": group, "n_group": int(len(members)), "n_with_value": int(len(vals))}
        if len(vals) < min_cell:
            rows.append({**base, "median": None, "q25": None, "q75": None, "status": SUPPRESSED})
        else:
            rows.append({**base, "median": float(np.median(vals)), "q25": float(np.quantile(vals, 0.25)), "q75": float(np.quantile(vals, 0.75)),
                         "status": "DESCRIBED"})
    return rows


def quantile_ci(d: np.ndarray, clusters: np.ndarray, q: float, boot: dict[str, Any], min_cases: int) -> dict[str, Any]:
    from ..alt_bundles import bootstrap_statistic

    if not d.size:
        return {"estimate": None, "ci_lower": None, "ci_upper": None, "status": "NE", "reason": "NO_CASES"}
    below = d.size < int(min_cases)
    r = bootstrap_statistic(lambda rows: float(np.quantile(d[rows], q)), d.size, clusters, n_boot=0 if below else int(boot["replicates"]),
                            seed=int(boot["seed"]), level=float(boot.get("level", 0.95)), min_valid_fraction=0.0)
    if below:
        r.update({"ci_lower": None, "ci_upper": None, "status": "NE", "reason": BELOW_MIN_CASES})
    return {k: r.get(k) for k in ("estimate", "ci_lower", "ci_upper", "status", "reason")}


def assert_no_identifiers(frame: pd.DataFrame, ids: list[str]) -> None:
    bad = [c for c in frame.columns if str(c) in IDENTIFIER_COLUMNS]
    if bad:
        raise RuntimeError(f"{ANALYSIS}: identifier columns {bad} would leave the run root")
    text = frame.astype(str).to_numpy().ravel()
    tokens = {s for i in ids for s in (i, i.split(":", 1)[-1])}
    if any(t in tokens for t in text):
        raise RuntimeError(f"{ANALYSIS}: a case identifier would leave the run root")


def run(pctx: Any) -> dict[str, Any]:
    from ...formal_features import formal_descriptor_names
    from ...transition_fit import TransitionModel
    from .common import dev_reference

    acfg = pctx.analysis_cfg(ANALYSIS)
    out = pctx.out_dir(ANALYSIS)
    stats = pctx.lane_cfg["statistics"]
    targets = [str(t) for t in stats["primary_targets"]]
    min_cases = int(stats["min_cases_per_stratum"])
    min_cell = int(stats["b26_min_cell"])
    names = formal_descriptor_names()
    sd, _ = dev_reference(pctx)
    # The frozen geometry-aware transition of B12 (second source run).
    gdir = f"postfreeze/B12/models/{B12_ESTIMATOR}"
    b12_frozen = pctx.read_second_json(f"{gdir}/FROZEN.json")
    if list(b12_frozen.get("columns") or []) != [*names, *COVARIATES]:
        raise ValueError(f"{ANALYSIS}: the B12 geometry-aware transition was fitted on other columns than the 57 descriptors and {list(COVARIATES)}")
    geo_model = TransitionModel.from_dict(pctx.read_second_json(f"{gdir}/transition_model.json"))
    splits, skipped = included_partitions(pctx, [str(p) for p in acfg["partitions"]], f"{COMMAND} {ANALYSIS}")
    agreement_rows, count_rows, describe_rows, check_rows, all_ids = [], [], [], [], []
    cases: dict[str, dict[str, dict[str, Any]]] = {}
    for split in splits:
        part = load_partition(pctx, split)
        all_ids += part.ids
        y = part.observed(targets)
        pred = part.bhat.reindex(columns=targets).apply(pd.to_numeric, errors="coerce")
        if part.dataset not in cases:
            cases[part.dataset] = staging_geometry(pctx, part.dataset)
        geo = geometry(part.b, cases[part.dataset], part.ids)
        attrs = pd.DataFrame({"slice_count": geo["n_slices"], "voxel_volume_ml": geo["voxel_volume_ml"],
                              "imaged_volume_ml": geo["n_voxels"] * geo["voxel_volume_ml"]}, index=part.ids)
        categorical = []
        if part.dataset != "ACDC":
            cohort = pctx.read_parquet(f"contracts/cohort_{part.dataset}.parquet").set_index("sample_id")
            for attr, column in CATEGORICAL.items():
                if column in cohort.columns:
                    attrs[attr] = cohort[column].reindex(part.ids).astype(object)
                    categorical.append(attr)
        a_p = formal_a(pctx, part.dataset).reindex(part.ids)[names].to_numpy(dtype=float)
        pred_geo = geo_model.predict(np.column_stack([a_p, geo.loc[part.ids, list(COVARIATES)].to_numpy(dtype=float)]), index=pd.Index(part.ids))
        recorded = pctx.read_parquet(f"metrics/per_target_{split}.parquet").set_index("target") if pctx.exists(f"metrics/per_target_{split}.parquet") else pd.DataFrame()
        for t in targets:
            d = (pred[t] - y[t]).dropna()
            clusters = part.clusters.reindex(d.index).astype(str).to_numpy(dtype=object)
            flag, rule = outliers(d)
            row = {"partition": split, "target": t, "unit": "physical", "n": int(len(d)), "reference_sd": float(sd.get(t, np.nan))}
            for name, q in (("median_bias", 0.5), ("p2_5", 0.025), ("p97_5", 0.975)):
                r = quantile_ci(d.to_numpy(dtype=float), clusters, q, pctx.boot, min_cases)
                row.update({name: r["estimate"], f"{name}_ci_lower": r["ci_lower"], f"{name}_ci_upper": r["ci_upper"], f"{name}_status": r["status"]})
            row.update({"mad": rule["mad"], "outlier_threshold": rule["threshold"], "n_outliers": int(flag.sum())})
            if t in recorded.index:
                rec = recorded.loc[t]
                row.update({"recorded_ba_n": int(rec["n"]) if pd.notna(rec.get("n")) else None, "recorded_ba_bias": float(rec["bias"]),
                            "recorded_loa_lower": float(rec["loa_lower"]), "recorded_loa_upper": float(rec["loa_upper"]),
                            "recorded_source": f"metrics/per_target_{split}.parquet"})
            agreement_rows.append(row)
            count_rows.append({"partition": split, "target": t, "n_scored": int(len(d)), "n_outliers": int(flag.sum()), "median": rule["median"],
                               "mad": rule["mad"], "threshold": rule["threshold"], "rule": stats["b26_outlier_rule"]})
            labels = {"partition": split, "target": t}
            for attr in categorical:
                describe_rows += [{**r, "attribute": attr, "kind": "categorical"} for r in describe_categorical(attrs[attr], flag, min_cell, **labels)]
            for attr in CONTINUOUS:
                describe_rows += [{**r, "attribute": attr, "kind": "continuous"} for r in describe_continuous(attrs[attr], flag, min_cell, **labels)]
            d_geo = (pd.to_numeric(pred_geo[t], errors="coerce") - y[t]).dropna() if t in pred_geo.columns else pd.Series(dtype=float)
            flag_geo, rule_geo = outliers(d_geo)
            out_ids = flag[flag].index
            scored = [i for i in out_ids if i in flag_geo.index]
            still = int(flag_geo.reindex(scored).fillna(False).astype(bool).sum())
            check_rows.append({"partition": split, "target": t, "n_outliers": int(len(out_ids)), "n_still_outlier_under_geometry": still,
                               "n_no_longer_outlier": int(len(scored) - still), "n_not_scored_under_geometry": int(len(out_ids) - len(scored)),
                               "n_outliers_under_geometry": int(flag_geo.sum()), "geometry_threshold": rule_geo["threshold"],
                               "geometry_transition": f"{gdir} (second source run)"})
    frames = {"agreement": pd.DataFrame(agreement_rows), "counts": pd.DataFrame(count_rows), "description": pd.DataFrame(describe_rows),
              "geometry": pd.DataFrame(check_rows)}
    for frame in frames.values():
        assert_no_identifiers(frame, all_ids)
    paths = {"agreement": out / "robust_agreement_table.parquet", "counts": out / "outlier_counts.parquet",
             "description": out / "outlier_description.parquet", "geometry": out / "geometry_check.parquet", "summary": out / "summary.json"}
    for key, frame in frames.items():
        write_table(frame, paths[key], pctx)
    summary = clean({"analysis": ANALYSIS, "differences": acfg["differences"], "agreement": acfg["agreement"], "outliers": acfg["outliers"],
                     "description": acfg["description"], "geometry_check": acfg["geometry_check"], "reference": acfg["reference"],
                     "min_cell": min_cell, "quantile_method": "numpy_linear", "included": splits, "not_executed": skipped,
                     "attributes_not_recorded": {"mms2_external": "centre"} if "mms2_external" in splits else {},
                     "agreement_rows": json.loads(frames["agreement"].to_json(orient="records")) if agreement_rows else [],
                     "geometry_rows": check_rows})
    write_json(summary, paths["summary"], pctx)
    text = paths["summary"].read_text(encoding="utf-8")
    if any(json.dumps(s) in text for i in all_ids for s in (i, i.split(":", 1)[-1])):
        raise RuntimeError(f"{ANALYSIS}: a case identifier would leave the run root")
    pctx.write_provenance(ANALYSIS, dict(pctx.verified_log), list(paths.values()),
                          {"included": splits, "not_executed": skipped,
                           "n_outliers": {f"{r['partition']}:{r['target']}": r["n_outliers"] for r in count_rows}})
    return {"included": splits, "not_executed": skipped}
