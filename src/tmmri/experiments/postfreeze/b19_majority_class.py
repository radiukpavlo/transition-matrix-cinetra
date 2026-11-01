"""B19: majority-class references (EPC-A2; ``configs/exploratory_postfreeze.yaml`` analyses.B19).

From the recorded class counts only: the reference-class row totals of the confusion-matrix companion of
``metrics/predictor_<p>.json`` (the evaluable same-definition cases of ``summarize-predictor``; the M&Ms-2 file of
``evaluate-mms2``), cross-checked against ``counts.n_evaluable`` and, where present, ``metrics/predictor_confusion.parquet``.
The majority-class rate is the largest reference-class count divided by the evaluable cases, with the class named (ties
by class order) and its Wilson interval, beside the uniform chance rate.  Nothing is recomputed from case rows.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .common import clean, included_partitions, write_json, write_table
from .estimates import wilson

ANALYSIS = "B19"


def majority(classes: list[str], counts: list[list[int]]) -> dict[str, Any]:
    totals = {str(c): int(sum(row)) for c, row in zip(classes, counts)}
    n = int(sum(totals.values()))
    top = max(totals.values()) if totals else 0
    name = next((c for c in classes if totals[str(c)] == top), None) if n else None
    return {"class_counts": totals, "n_evaluable": n, "majority_class": name, "majority_rate": wilson(top, n),
            "uniform_chance_rate": 1.0 / len(classes) if classes else None}


def run(pctx: Any) -> dict[str, Any]:
    acfg = pctx.analysis_cfg(ANALYSIS)
    out = pctx.out_dir(ANALYSIS)
    splits, skipped = included_partitions(pctx, [str(p) for p in acfg["partitions"]], f"run-postfreeze {ANALYSIS}")
    confusion = pctx.read_parquet("metrics/predictor_confusion.parquet") if pctx.exists("metrics/predictor_confusion.parquet") else None
    rows, notes = [], {}
    for split in splits:
        rel = f"metrics/predictor_{split}.json"
        if not pctx.exists(rel):
            rows.append({"partition": split, "status": "NE", "reason": f"{rel} not recorded"})
            continue
        doc = pctx.read_json(rel)
        cm = (doc.get("companions") or {}).get("confusion_matrix")
        if not cm:
            rows.append({"partition": split, "status": "NE", "reason": f"{rel} has no confusion-matrix companion (status {doc.get('status')})"})
            continue
        m = majority([str(c) for c in cm["rows_reference"]], cm["counts"])
        recorded_n = (doc.get("counts") or {}).get("n_evaluable")
        check = {"n_evaluable_recorded": recorded_n, "n_evaluable_matches": recorded_n is None or int(recorded_n) == m["n_evaluable"]}
        if confusion is not None and (confusion["partition"] == split).any():
            sub = confusion[confusion["partition"] == split].groupby("reference_class")["n"].sum()
            check["confusion_parquet_matches"] = {str(k): int(v) for k, v in sub.items()} == m["class_counts"]
        notes[split] = check
        if not check["n_evaluable_matches"] or check.get("confusion_parquet_matches") is False:
            raise ValueError(f"{ANALYSIS}: the recorded class counts of {split} disagree between {rel} and its companions")
        rate = m["majority_rate"]
        rows.append({"partition": split, "source": rel, "n_evaluable": m["n_evaluable"], "majority_class": m["majority_class"],
                     "majority_count": rate["k"], "majority_rate": rate["estimate"], "ci_lower": rate["ci_lower"], "ci_upper": rate["ci_upper"],
                     "uniform_chance_rate": m["uniform_chance_rate"], **{f"n_{c}": v for c, v in m["class_counts"].items()}, "status": "EVALUATED"})
    paths = {"table": out / "majority_class_table.parquet", "summary": out / "summary.json"}
    write_table(pd.DataFrame(rows), paths["table"], pctx)
    summary = clean({"analysis": ANALYSIS, "source": acfg["source"], "rate": acfg["rate"], "included": splits, "not_executed": skipped,
                     "cross_checks": notes, "rows": rows})
    write_json(summary, paths["summary"], pctx)
    pctx.write_provenance(ANALYSIS, dict(pctx.verified_log), list(paths.values()), {"included": splits, "not_executed": skipped, "cross_checks": notes})
    return {"included": splits, "not_executed": skipped}
