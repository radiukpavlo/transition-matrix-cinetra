"""B29: ejection-fraction categories, measured against reconstructed
(``configs/exploratory_postfreeze_v604.yaml`` analyses.B29; SPECIFICATION.md 6.0.4 Section 13.9; no reading, D48).

The categories of D49 (``statistics.ef_categories``; lower bound inclusive, upper bound exclusive, in percent: LVEF
< 40, 40 to < 50, >= 50; RVEF < 40, >= 40) are applied to the OBSERVED cells of measured B and to the recorded raw
reconstructions (frozen-bundle B-hat; ``acdc_dev_oof``: the recorded nested out-of-fold B-hat), on the
non-critical-QC rows of each partition.  Per partition and target: the cross-tabulation (every pair of categories,
zeros included), the proportion of cases in the same category (Wilson) and Cohen's kappa (unweighted) with the cluster
bootstrap of the lane.  A cell that is not OBSERVED is not scored; an OBSERVED cell without a finite reconstruction is
counted, not scored.  Values derived from exact counts carry a half-up companion (BLD-1); no output carries a case or
subject identifier.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .common import COMMAND, clean, load_partition
from .lane_v604 import block_partitions, identifier_tokens, kappa_row, proportion, write_checked_json, write_checked_table

ANALYSIS = "B29"


def _number(v: Any) -> str:
    return str(int(v)) if float(v).is_integer() else str(v)


def category_labels(bounds: Sequence[Sequence[Any]]) -> list[str]:
    """Labels of half-open intervals [lower, upper): ``<40``, ``40-<50``, ``>=50`` (None is unbounded)."""

    out = []
    for lower, upper in bounds:
        if lower is None and upper is None:
            raise ValueError(f"{ANALYSIS}: a category needs a lower or an upper bound")
        out.append(f"<{_number(upper)}" if lower is None else f">={_number(lower)}" if upper is None else f"{_number(lower)}-<{_number(upper)}")
    return out


def check_categories(bounds: Sequence[Sequence[Any]]) -> None:
    """The categories must tile the real line in order: unbounded at both ends, each upper bound the next lower bound."""

    edges = [(None if lo is None else float(lo), None if hi is None else float(hi)) for lo, hi in bounds]
    ok = bool(edges) and edges[0][0] is None and edges[-1][1] is None and all(a[1] is not None and a[1] == b[0] for a, b in zip(edges, edges[1:]))
    ok = ok and all(lo is None or hi is None or lo < hi for lo, hi in edges)
    if not ok:
        raise ValueError(f"{ANALYSIS}: statistics.ef_categories must be contiguous half-open intervals covering every value, got {list(bounds)}")


def categorize(values: pd.Series, bounds: Sequence[Sequence[Any]]) -> pd.Series:
    """The category label of every finite value (lower bound inclusive, upper bound exclusive); None otherwise."""

    check_categories(bounds)
    labels = category_labels(bounds)
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    out = np.full(v.shape, None, dtype=object)
    for label, (lower, upper) in zip(labels, bounds):
        hit = np.isfinite(v)
        if lower is not None:
            hit &= v >= float(lower)
        if upper is not None:
            hit &= v < float(upper)
        out[hit] = label
    return pd.Series(out, index=values.index, dtype=object)


def crosstab(measured: pd.Series, reconstructed: pd.Series, labels: Sequence[str]) -> list[dict[str, Any]]:
    both = measured.notna() & reconstructed.notna()
    m, r = measured[both].astype(str), reconstructed[both].astype(str)
    return [{"measured_category": a, "reconstructed_category": b, "n": int(((m == a) & (r == b)).sum())} for a in labels for b in labels]


def agreement_row(measured: pd.Series, reconstructed: pd.Series, clusters: pd.Series, boot: Mapping[str, Any]) -> dict[str, Any]:
    both = measured.notna() & reconstructed.notna()
    k = int((measured[both].astype(str) == reconstructed[both].astype(str)).sum())
    return {**proportion("same_category", k, int(both.sum())), **kappa_row(reconstructed.where(both), measured.where(both), clusters, boot)}


def run(pctx: Any) -> dict[str, Any]:
    acfg = pctx.analysis_cfg(ANALYSIS)
    out = pctx.out_dir(ANALYSIS)
    stats = pctx.lane_cfg["statistics"]
    categories = {str(t): [list(b) for b in v] for t, v in stats["ef_categories"].items()}
    for bounds in categories.values():
        check_categories(bounds)
    included, skipped = block_partitions(pctx, [str(p) for p in acfg["partitions"]], f"{COMMAND} {ANALYSIS}")
    cross_rows, agree_rows, tokens = [], [], set()
    for label, split in included:
        part = load_partition(pctx, split)
        tokens |= identifier_tokens(part.ids)
        observed = part.observed(list(categories))
        for target, bounds in categories.items():
            labels = category_labels(bounds)
            recon = pd.to_numeric(part.bhat[target], errors="coerce") if target in part.bhat.columns else pd.Series(np.nan, index=part.ids)
            m_cat, r_cat = categorize(observed[target], bounds), categorize(recon, bounds)
            counted = {"n_cases": len(part.ids), "n_not_observed": int(m_cat.isna().sum()),
                       "n_observed_without_reconstruction": int((m_cat.notna() & r_cat.isna()).sum())}
            cross_rows.extend({"partition": label, "target": target, **row} for row in crosstab(m_cat, r_cat, labels))
            agree_rows.append({"partition": label, "target": target, "categories": " | ".join(labels), **counted,
                               **agreement_row(m_cat, r_cat, part.clusters, pctx.boot)})
    paths = {"crosstab": out / "ef_category_crosstab.parquet", "agreement": out / "ef_category_agreement.parquet", "summary": out / "summary.json"}
    write_checked_table(pd.DataFrame(cross_rows, columns=["partition", "target", "measured_category", "reconstructed_category", "n"]),
                        paths["crosstab"], pctx, tokens)
    write_checked_table(pd.DataFrame(agree_rows), paths["agreement"], pctx, tokens)
    summary = clean({"analysis": ANALYSIS, "label": "EXPLORATORY_POST_HOC", "reading": None, "method": acfg["method"],
                     "categories": {t: dict(zip(category_labels(b), b)) for t, b in categories.items()},
                     "bounds": "lower bound inclusive, upper bound exclusive (D49)", "included": [lab for lab, _ in included], "not_executed": skipped,
                     "rows": agree_rows})
    write_checked_json(summary, paths["summary"], pctx, tokens)
    pctx.write_provenance(ANALYSIS, dict(pctx.verified_log), list(paths.values()),
                          {"included": [lab for lab, _ in included], "not_executed": skipped, "reading": None})
    return {"included": [lab for lab, _ in included], "not_executed": skipped}
