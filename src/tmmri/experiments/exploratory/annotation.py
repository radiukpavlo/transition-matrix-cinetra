"""B8 annotation convention (exploratory, post hoc; descriptive and confounded by centre).

Per vendor and target, the bias of the frozen reconstruction (mean normalized error, cluster-bootstrap interval) on
M&Ms-1 (``mms_external``) and on M&Ms-2 (``mms2_external``, contoured with the ACDC procedure), and their difference
(M&Ms-2 minus M&Ms-1; independent cluster bootstraps of the two partitions).  A partition that is not recorded (or
whose lock refuses) is reported as not executed.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .common import mean_with_ci, weighted_mean

OUTPUTS = ("vendor_bias.parquet", "vendor_bias.csv", "difference.parquet", "difference.csv", "summary.json")


def difference_ci(v1: np.ndarray, c1: np.ndarray, v2: np.ndarray, c2: np.ndarray, *, n_boot: int, seed: int, level: float) -> tuple[float, float | None, float | None]:
    """mean(v2) - mean(v1) with a percentile interval from independent cluster draws of both samples per replicate."""

    from ...metrics import _cluster_codes, _cluster_draw_weights

    est = float(np.mean(v2) - np.mean(v1)) if v1.size and v2.size else float("nan")
    if not (v1.size and v2.size) or n_boot <= 0:
        return est, None, None
    k1, n1 = _cluster_codes(c1, v1.size)
    k2, n2 = _cluster_codes(c2, v2.size)
    boot = []
    for ss in np.random.SeedSequence(int(seed)).spawn(int(n_boot)):
        rng = np.random.default_rng(ss)
        boot.append(weighted_mean(v2, _cluster_draw_weights(rng, k2, n2)) - weighted_mean(v1, _cluster_draw_weights(rng, k1, n1)))
    arr = np.asarray(boot, dtype=float)
    arr = arr[np.isfinite(arr)]
    alpha = (1.0 - level) / 2.0
    return est, float(np.quantile(arr, alpha)), float(np.quantile(arr, 1.0 - alpha))


def plan(ctx: Any, cfg: Mapping[str, Any], acfg: Mapping[str, Any]) -> dict[str, Any]:
    from ...stages import _contracts
    from .common import external_partitions, partition_inputs

    run = ctx.require_run("run-exploratory")
    included, skipped = external_partitions(ctx, cfg, "run-exploratory B8", wanted=list(acfg["partitions"]))
    inputs = {"B_ACDC": run.path / "semantics" / "B_ACDC.parquet", "split_manifest": _contracts(ctx) / "split_manifest.parquet"}
    for split, ds in included.items():
        inputs.update(partition_inputs(ctx, split, ds))
    return {"inputs": inputs, "included": included, "not_executed": skipped}


def compute(ctx: Any, cfg: Mapping[str, Any], acfg: Mapping[str, Any], plan: Mapping[str, Any], out) -> dict[str, Any]:
    from ...runs import atomic_write_json
    from ...stages import _write_table
    from .common import partition_errors

    boot, min_cases = cfg["bootstrap"], int(cfg["min_cases_per_stratum"])
    targets = [t for t in acfg["targets"] if t in ctx.study.transition_targets()]
    rows, diffs, parts = [], [], {}
    for split, ds in plan["included"].items():
        part = partition_errors(ctx, cfg, split, ds, targets)
        parts[split] = part
        for v in sorted(set(part["vendor"])):
            ids = [s for s in part["ids"] if part["vendor"][s] == v]
            for t in targets:
                stat = mean_with_ci(part["errors"].loc[ids, t], part["clusters"], boot, min_cases)
                rows.append({"partition": split, "vendor": v, "target": t, "n": stat["n"], "n_clusters": stat["n_clusters"], "bias": stat["estimate"],
                             "ci_lower": stat["ci_lower"], "ci_upper": stat["ci_upper"], "status": stat["status"], "reason": stat["reason"]})
    names = list(acfg["partitions"])
    if len(names) == 2 and all(n in parts for n in names):
        first, second = parts[names[0]], parts[names[1]]
        for v in sorted(set(first["vendor"]) & set(second["vendor"])):
            for t in targets:
                e1 = first["errors"].loc[[s for s in first["ids"] if first["vendor"][s] == v], t]
                e2 = second["errors"].loc[[s for s in second["ids"] if second["vendor"][s] == v], t]
                m1, m2 = np.isfinite(e1.to_numpy(dtype=float)), np.isfinite(e2.to_numpy(dtype=float))
                below = min(int(m1.sum()), int(m2.sum())) < min_cases
                est, lo, hi = difference_ci(e1.to_numpy(dtype=float)[m1], first["clusters"].reindex(e1.index).to_numpy(dtype=object)[m1],
                                            e2.to_numpy(dtype=float)[m2], second["clusters"].reindex(e2.index).to_numpy(dtype=object)[m2],
                                            n_boot=0 if below else int(boot["replicates"]), seed=int(boot["seed"]), level=float(boot["level"]))
                diffs.append({"vendor": v, "target": t, "minuend": names[1], "subtrahend": names[0], "n_minuend": int(m2.sum()), "n_subtrahend": int(m1.sum()),
                              "difference": est if np.isfinite(est) else None, "ci_lower": lo, "ci_upper": hi,
                              "status": "NE" if below else "EVALUATED", "reason": "STRATUM_BELOW_MIN_CASES" if below else None})
    bias = pd.DataFrame(rows, columns=["partition", "vendor", "target", "n", "n_clusters", "bias", "ci_lower", "ci_upper", "status", "reason"])
    diff = pd.DataFrame(diffs, columns=["vendor", "target", "minuend", "subtrahend", "n_minuend", "n_subtrahend", "difference", "ci_lower", "ci_upper",
                                        "status", "reason"])
    for frame, cols in ((bias, ("bias", "ci_lower", "ci_upper")), (diff, ("difference", "ci_lower", "ci_upper"))):
        for c in cols:
            frame[c] = pd.to_numeric(frame[c], errors="coerce")
    _write_table(bias, out / "vendor_bias.parquet", csv=True)
    _write_table(diff, out / "difference.parquet", csv=True)
    atomic_write_json(out / "summary.json", {"analysis": "B8", "included": plan["included"], "not_executed": plan["not_executed"],
                                             "difference_status": "EVALUATED" if len(diff) else "NOT_EXECUTED", "targets": targets,
                                             "note": "descriptive; annotation convention, vendor and centre are confounded"}, allow_nan=True)
    return {"included": list(plan["included"]), "not_executed": plan["not_executed"]}
