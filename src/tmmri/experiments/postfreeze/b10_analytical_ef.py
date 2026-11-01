"""B10: analytical ejection fraction from the voxel fractions of formal A, without learning (EPC-1;
``configs/exploratory_postfreeze.yaml`` analyses.B10).

EF_k = 100 (f_ED,k - f_ES,k) / f_ED,k for k = LV (``fd_l1_fraction_ed``, ``fd_l1_fraction_es``) and RV (label 3).  As
the registry: NOT_APPLICABLE when the measured EDV is below 1 mL; f_ED,k = 0 gives NE.  The exactness check counts,
per partition and target, the cases with |EF_analytical - EF_B| > 1e-6 percentage points (unequal ED and ES voxel grids)
and the cases whose B state is not OBSERVED; they are reported, never corrected.  rho_EF and rho_j of the analytical EF
and of the recorded primary ridge (acdc_dev: nested out-of-fold) share one paired cluster bootstrap against the
development-mean null.  When inexact cases exist, the same estimates are repeated on the exact cases (fallback of the
revision plan).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .common import (
    clean,
    dev_reference,
    formal_a,
    included_partitions,
    load_partition,
    write_json,
    write_table,
)
from .estimates import paired_rho, rho_rows, target_rows
from .transition_variants import difference_rows

ANALYSIS = "B10"
TOLERANCE_PP = 1e-6
MIN_EDV_ML = 1.0
EF = {"lv_ef_percent": ("fd_l1_fraction_ed", "fd_l1_fraction_es", "lv_edv_ml"),
      "rv_ef_percent": ("fd_l3_fraction_ed", "fd_l3_fraction_es", "rv_edv_ml")}


def analytical_ef(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(EF frame, state frame): ``ANALYTICAL``, ``NE_ZERO_ED_FRACTION``, ``NE_NONFINITE_FRACTION`` or ``NOT_APPLICABLE``
    (measured EDV below 1 mL)."""

    from ..exploratory.common import observed

    values, states = {}, {}
    for target, (ed, es, edv) in EF.items():
        f_ed = pd.to_numeric(a[ed], errors="coerce").to_numpy(dtype=float)
        f_es = pd.to_numeric(a[es], errors="coerce").to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ef = 100.0 * (f_ed - f_es) / f_ed
        state = np.where(~(np.isfinite(f_ed) & np.isfinite(f_es)), "NE_NONFINITE_FRACTION",
                         np.where(f_ed == 0, "NE_ZERO_ED_FRACTION", "ANALYTICAL")).astype(object)
        measured_edv = observed(b, edv).reindex(a.index).to_numpy(dtype=float)
        state = np.where((state == "ANALYTICAL") & np.isfinite(measured_edv) & (measured_edv < MIN_EDV_ML), "NOT_APPLICABLE", state)
        values[target] = np.where(state == "ANALYTICAL", ef, np.nan)
        states[target] = state
    return pd.DataFrame(values, index=a.index), pd.DataFrame(states, index=a.index)


def exactness(ef: pd.DataFrame, states: pd.DataFrame, b: pd.DataFrame) -> tuple[list[dict[str, Any]], pd.Series]:
    """Per target: counts and the largest |EF_analytical - EF_B|; and the per-row flag 'every finite pair exact'."""

    rows, inexact_any = [], pd.Series(False, index=ef.index)
    for target in EF:
        b_raw = pd.to_numeric(b[target], errors="coerce").reindex(ef.index) if target in b.columns else pd.Series(np.nan, index=ef.index)
        state_b = b[f"{target}__state"].astype("string").fillna("").reindex(ef.index) if f"{target}__state" in b.columns else pd.Series("", index=ef.index)
        both = ef[target].notna() & b_raw.notna()
        diff = (ef[target] - b_raw).abs()
        inexact = both & (diff > TOLERANCE_PP)
        inexact_any |= inexact.fillna(False)
        rows.append({"target": target, "n_rows": int(len(ef)), "n_analytical": int((states[target] == "ANALYTICAL").sum()),
                     "n_ne_zero_ed_fraction": int((states[target] == "NE_ZERO_ED_FRACTION").sum()),
                     "n_ne_nonfinite_fraction": int((states[target] == "NE_NONFINITE_FRACTION").sum()),
                     "n_not_applicable_edv_below_1ml": int((states[target] == "NOT_APPLICABLE").sum()),
                     "n_b_state_not_observed": int((state_b != "OBSERVED").sum()), "n_both_finite": int(both.sum()),
                     "n_exact": int((both & (diff <= TOLERANCE_PP)).sum()), "n_inexact": int(inexact.sum()), "tolerance_pp": TOLERANCE_PP,
                     "max_abs_difference_pp": float(diff[both].max()) if both.any() else None})
    return rows, inexact_any


def run(pctx: Any) -> dict[str, Any]:
    acfg = pctx.analysis_cfg(ANALYSIS)
    out = pctx.out_dir(ANALYSIS)
    targets = [str(t) for t in pctx.lane_cfg["statistics"]["ef_targets"]]
    if sorted(targets) != sorted(EF):
        raise ValueError(f"{ANALYSIS}: statistics.ef_targets must be {sorted(EF)}")
    fams = {"rho_ef": targets}
    min_cases = int(pctx.lane_cfg["statistics"]["min_cases_per_stratum"])
    splits, skipped = included_partitions(pctx, acfg["partitions"], f"run-postfreeze {ANALYSIS}")
    sd, mean = dev_reference(pctx)
    results, exact_rows, rho_frames, target_frames, exact_results = {}, [], [], [], {}
    for split in splits:
        part = load_partition(pctx, split)
        label = "acdc_dev_oof" if split == "acdc_dev" else split
        a = formal_a(pctx, part.dataset).reindex(part.ids)
        ef, states = analytical_ef(a, part.b)
        counts, inexact = exactness(ef, states, part.b)
        exact_rows += [{"partition": label, **row} for row in counts]
        y = part.observed(targets)
        null = pd.DataFrame([mean[targets]] * len(part.ids), index=part.ids)
        arms = {"analytical": (ef, null), "primary_ridge": (part.bhat, null)}
        kw = dict(units=part.units, clusters=part.clusters, boot=pctx.boot, study=pctx.study, min_cases=min_cases, reference="primary_ridge")
        result = paired_rho(y, arms, sd, fams, **kw)
        result.update({"partition": label, "role": part.role, "n_excluded_critical_qc": part.n_excluded_critical_qc})
        results[label] = result
        rho_frames.append(rho_rows(result, partition=label, subset="all"))
        for name, (pred, _) in arms.items():
            target_frames.append(target_rows(y, pred, sd, targets, result["arms"][name], units=part.units, partition=label, subset="all", arm=name))
        if inexact.any():   # fallback: the exact (equal-grid) cases only
            keep = [s for s in part.ids if not inexact.loc[s]]
            sub = paired_rho(y.loc[keep], {k: (p.reindex(keep), n.reindex(keep)) for k, (p, n) in arms.items()}, sd, fams,
                             **{**kw, "units": part.units.loc[keep], "clusters": part.clusters.loc[keep]})
            sub.update({"partition": label, "subset": "exact_cases", "n_inexact_cases_removed": int(inexact.sum())})
            exact_results[label] = sub
            rho_frames.append(rho_rows(sub, partition=label, subset="exact_cases"))
    paths = {"per_target": out / "per_target_mae_bias_loa.parquet", "rho": out / "rho.parquet", "exact": out / "exactness_counts.parquet",
             "diff": out / "paired_difference_vs_primary_ridge.parquet", "summary": out / "summary.json"}
    write_table(pd.concat(target_frames, ignore_index=True), paths["per_target"], pctx)
    write_table(pd.concat(rho_frames, ignore_index=True), paths["rho"], pctx)
    write_table(pd.DataFrame(exact_rows), paths["exact"], pctx)
    diffs = difference_rows(results)
    if exact_results:
        diffs = pd.concat([diffs.assign(subset="all"), difference_rows(exact_results).assign(subset="exact_cases")], ignore_index=True)
    write_table(diffs, paths["diff"], pctx)
    summary = clean({"analysis": ANALYSIS, "formula": acfg["formula"], "tolerance_pp": TOLERANCE_PP, "included": [("acdc_dev_oof" if s == "acdc_dev" else s) for s in splits],
                     "not_executed": skipped, "null": "acdc_dev mean of the non-critical fit rows (as the recorded rho)",
                     "exactness": exact_rows, "results": results, "exact_case_results": exact_results})
    write_json(summary, paths["summary"], pctx)
    pctx.write_provenance(ANALYSIS, dict(pctx.verified_log), list(paths.values()),
                          {"included": summary["included"], "not_executed": skipped,
                           "n_inexact": {f"{r['partition']}:{r['target']}": r["n_inexact"] for r in exact_rows}})
    return {"included": summary["included"], "not_executed": skipped, "outputs": len(paths)}
