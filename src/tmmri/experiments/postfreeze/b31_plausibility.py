"""B31: reconstruction plausibility (``configs/exploratory_postfreeze_v604.yaml`` analyses.B31;
SPECIFICATION.md 6.0.4 Section 13.9; no reading, D48).

On the recorded raw reconstructions (frozen-bundle B-hat on the physical scale of the recorded per-target errors;
``acdc_dev_oof``: the recorded nested out-of-fold B-hat) of the non-critical-QC rows of each partition:

* **Counts** (Wilson): (i) |EF reconstructed - EF implied by the reconstructed volumes, 100 x (EDV - ESV) / EDV| >
  ``statistics.ef_inconsistency_points``, LV and RV, on the cases with finite reconstructed EF, EDV and ESV and EDV > 0
  (the implied EF is undefined otherwise; such cases are counted); (ii) reconstructed ESV >= reconstructed EDV, LV and
  RV, on the cases with both finite; (iii) a primary target outside ``statistics.plausibility_bounds`` (inclusive
  bounds), per target on its finite cells and in any target on the cases with at least one finite primary target.
* **Distribution**: median, 2.5th and 97.5th percentiles (numpy linear interpolation) of EF implied - EF reconstructed,
  LV and RV, on the cases of (i).
* **Clipped skill**: rho_j and rho (the v6 estimand, 1 - MAE_model,j / MAE_mean,j on the OBSERVED cells, patient
  weighted, normalized by the acdc_dev fit-row SD; null = the acdc_dev fit-row mean) with every reconstruction clipped
  to the plausibility bounds, beside the unclipped values, with the paired cluster bootstrap of the lane (the same
  cluster draw for both; :func:`tmmri.experiments.alt_bundles.paired_rho_difference`, clipped minus unclipped).

Values derived from exact counts carry a half-up companion (BLD-1); no output carries a case or subject identifier.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .common import COMMAND, clean, dev_reference, load_partition
from .lane_v604 import DEVELOPMENT, block_partitions, identifier_tokens, proportion, quantiles, write_checked_json, write_checked_table

ANALYSIS = "B31"
SIDES = {"lv": ("lv_edv_ml", "lv_esv_ml", "lv_ef_percent"), "rv": ("rv_edv_ml", "rv_esv_ml", "rv_ef_percent")}
DISTRIBUTION = {"median": 0.5, "p2_5": 0.025, "p97_5": 0.975}


def implied_ef(edv: pd.Series, esv: pd.Series) -> pd.Series:
    """100 x (EDV - ESV) / EDV where EDV > 0 and both are finite; NaN otherwise."""

    edv, esv = pd.to_numeric(edv, errors="coerce"), pd.to_numeric(esv, errors="coerce")
    ok = np.isfinite(edv) & np.isfinite(esv) & (edv > 0)
    return (100.0 * (edv - esv) / edv.where(ok)).where(ok)


def side_rows(bhat: pd.DataFrame, points: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Counts (i) and (ii) and the distribution of EF implied - EF reconstructed, per side."""

    counts, dist = [], []
    for side, (edv_c, esv_c, ef_c) in SIDES.items():
        edv, esv, ef = (pd.to_numeric(bhat[c], errors="coerce") for c in (edv_c, esv_c, ef_c))
        implied = implied_ef(edv, esv)
        finite = np.isfinite(edv) & np.isfinite(esv) & np.isfinite(ef)
        nonpositive = finite & ~(edv > 0)                     # exclusive counts: non-finite first, then EDV <= 0
        defined = finite & (edv > 0)
        if int(defined.sum()) + int((~finite).sum()) + int(nonpositive.sum()) != len(bhat) or not (implied.notna() == defined).where(finite, True).all():
            raise RuntimeError(f"{ANALYSIS}: the scored, non-finite and EDV <= 0 cases of {side} do not partition the cases")
        diff = (implied - ef)[defined]
        volumes = np.isfinite(edv) & np.isfinite(esv)
        counts.append({"check": "ef_inconsistent", "side": side, "target": ef_c, "threshold_points": float(points),
                       "n_not_finite": int((~finite).sum()), "n_edv_nonpositive": int(nonpositive.sum()),
                       **proportion("proportion", int((diff.abs() > float(points)).sum()), int(defined.sum()))})
        counts.append({"check": "esv_ge_edv", "side": side, "target": f"{esv_c} >= {edv_c}", "n_not_finite": int((~volumes).sum()),
                       **proportion("proportion", int((esv[volumes] >= edv[volumes]).sum()), int(volumes.sum()))})
        dist.append({"side": side, "difference": f"implied {ef_c} - reconstructed {ef_c}", "unit": "percentage points",
                     **quantiles(diff.to_numpy(dtype=float), DISTRIBUTION)})
    return counts, dist


def bounds_rows(bhat: pd.DataFrame, bounds: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Count (iii): per target on its finite cells, and in any target on the cases with a finite primary target."""

    rows, outside_any, finite_any = [], pd.Series(False, index=bhat.index), pd.Series(False, index=bhat.index)
    for target, (lo, hi) in bounds.items():
        v = pd.to_numeric(bhat[target], errors="coerce") if target in bhat.columns else pd.Series(np.nan, index=bhat.index)
        finite = np.isfinite(v)
        outside = finite & ((v < float(lo)) | (v > float(hi)))
        outside_any |= outside
        finite_any |= finite
        rows.append({"check": "outside_bounds", "side": None, "target": target, "lower": float(lo), "upper": float(hi), "n_not_finite": int((~finite).sum()),
                     "n_below": int((finite & (v < float(lo))).sum()), "n_above": int((finite & (v > float(hi))).sum()),
                     **proportion("proportion", int(outside.sum()), int(finite.sum()))})
    rows.append({"check": "outside_bounds", "side": None, "target": "any primary target", "n_not_finite": int((~finite_any).sum()),
                 **proportion("proportion", int(outside_any.sum()), int(finite_any.sum()))})
    return rows


def clip(bhat: pd.DataFrame, bounds: Mapping[str, Any]) -> pd.DataFrame:
    out = bhat.copy()
    for target, (lo, hi) in bounds.items():
        out[target] = pd.to_numeric(out[target], errors="coerce").clip(lower=float(lo), upper=float(hi))
    return out


def run(pctx: Any) -> dict[str, Any]:
    from ..alt_bundles import paired_rho_difference

    acfg = pctx.analysis_cfg(ANALYSIS)
    out = pctx.out_dir(ANALYSIS)
    study = pctx.study
    stats = pctx.lane_cfg["statistics"]
    primary = [str(t) for t in stats["primary_targets"]]
    bounds = {str(t): (float(v[0]), float(v[1])) for t, v in stats["plausibility_bounds"].items()}
    family = [str(t) for t in study.primary_family()]
    if sorted(family) != sorted(primary) or sorted(bounds) != sorted(primary):
        raise ValueError(f"{ANALYSIS}: statistics.primary_targets, statistics.plausibility_bounds and the primary family of the study must name "
                         "the same seven targets")
    if any(lo >= hi for lo, hi in bounds.values()):
        raise ValueError(f"{ANALYSIS}: every plausibility bound needs lower < upper")
    points = float(stats["ef_inconsistency_points"])
    boot = pctx.boot
    bcfg = study.bootstrap
    min_estimable = int(bcfg["primary_test"]["min_estimable_targets"])
    min_valid = float(bcfg["fixed_model_bootstrap"]["min_valid_fraction"])
    sd, mean = dev_reference(pctx)
    included, skipped = block_partitions(pctx, [str(p) for p in acfg["partitions"]], f"{COMMAND} {ANALYSIS}")
    count_rows, dist_rows, skill_rows, tokens = [], [], [], set()
    for label, split in included:
        part = load_partition(pctx, split)
        tokens |= identifier_tokens(part.ids)
        bhat = part.bhat.reindex(columns=primary).apply(pd.to_numeric, errors="coerce")
        counts, dist = side_rows(bhat, points)
        count_rows.extend({"partition": label, "n_cases": len(part.ids), **r} for r in [*counts, *bounds_rows(bhat, bounds)])
        dist_rows.extend({"partition": label, **r} for r in dist)
        clipped = clip(bhat, bounds)
        n_clipped = {t: int((np.isfinite(bhat[t]) & (bhat[t] != clipped[t])).sum()) for t in family}
        y_true = part.observed(family)
        null = pd.DataFrame([mean[family]] * len(part.ids), index=part.ids)
        r = paired_rho_difference(y_true, bhat[family], clipped[family], null, sd, family, cluster=part.units.reindex(part.ids).to_numpy(dtype=object),
                                  resample_cluster=part.clusters.reindex(part.ids).to_numpy(dtype=object), n_boot=int(boot["replicates"]),
                                  seed=int(boot["seed"]), min_estimable=min_estimable, level=float(boot.get("level", 0.95)), min_valid_fraction=min_valid)
        in_sample = label == DEVELOPMENT                     # reference mean and SD of the acdc_dev fit rows (Section 13.9 B31)
        skill_rows.append({"partition": label, "target": "rho", "in_sample": in_sample, "n_cases": len(part.ids), "n_cells_clipped": int(sum(n_clipped.values())),
                           "rho_unclipped": r.get("rho_reference"), "rho_clipped": r.get("rho_comparator"), "difference_clipped_minus_unclipped": r.get("difference"),
                           "ci_lower": r.get("ci_lower"), "ci_upper": r.get("ci_upper"), "status": r.get("status", "EVALUATED"), "reason": r.get("reason"),
                           "n_estimable_targets": len(r.get("estimable_targets", []))})
        for t in family:
            j = (r.get("rho_j") or {}).get(t) or {}
            skill_rows.append({"partition": label, "target": t, "in_sample": in_sample, "n_cases": len(part.ids), "n_cells_clipped": n_clipped[t],
                               "rho_unclipped": j.get("reference"), "rho_clipped": j.get("comparator"), "difference_clipped_minus_unclipped": j.get("difference"),
                               "ci_lower": j.get("ci_lower"), "ci_upper": j.get("ci_upper"), "status": j.get("status", "EVALUATED" if j else "NE"),
                               "reason": j.get("reason"), "n_estimable_targets": None})
    paths = {"counts": out / "plausibility_counts.parquet", "distribution": out / "ef_difference_distribution.parquet",
             "skill": out / "clipped_skill_table.parquet", "summary": out / "summary.json"}
    write_checked_table(pd.DataFrame(count_rows), paths["counts"], pctx, tokens)
    write_checked_table(pd.DataFrame(dist_rows), paths["distribution"], pctx, tokens)
    write_checked_table(pd.DataFrame(skill_rows), paths["skill"], pctx, tokens)
    summary = clean({"analysis": ANALYSIS, "label": "EXPLORATORY_POST_HOC", "reading": None, "counts": acfg["counts"], "distribution": acfg["distribution"],
                     "clipped_skill": acfg["clipped_skill"], "ef_inconsistency_points": points, "plausibility_bounds": bounds,
                     "bounds_rule": "inclusive; a value below the lower or above the upper bound is outside (D49)",
                     "rho": {"estimand": "v6 rho_j = 1 - MAE_model,j / MAE_mean,j (OBSERVED cells, patient weighted, acdc_dev fit-row SD and mean)",
                             "bootstrap": {"replicates": int(boot["replicates"]), "seed": int(boot["seed"]), "unit": "cluster_id", "paired": True},
                             "min_estimable_targets": min_estimable,
                             "in_sample": {DEVELOPMENT: "in-sample and descriptive only: the reference mean and SD are those of the acdc_dev fit "
                                                        "rows (Section 13.9 B31)"}},
                     "included": [lab for lab, _ in included], "not_executed": skipped, "rows": skill_rows})
    write_checked_json(summary, paths["summary"], pctx, tokens)
    pctx.write_provenance(ANALYSIS, dict(pctx.verified_log), list(paths.values()),
                          {"included": [lab for lab, _ in included], "not_executed": skipped, "reading": None})
    return {"included": [lab for lab, _ in included], "not_executed": skipped}
