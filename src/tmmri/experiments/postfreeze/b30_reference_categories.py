"""B30: audited decisions by reference category (``configs/exploratory_postfreeze_v604.yaml``
analyses.B30; SPECIFICATION.md 6.0.4 Section 13.9; no reading, D48).

* **(a) Recorded confusion.**  ``metrics/predictor_confusion.parquet`` of the frozen run for acdc_test and mms_external,
  checked against the recorded decisions (``inference/d_model_<p>.parquet``) and reference labels on the rows of
  ``summarize-predictor`` (partition members with a recorded decision, critical-QC rows removed; a known reference: the
  ACDC class, or the M&Ms crosswalk class of the same-definition cases); any differing cell refuses (BundleError).
  Reported: the recorded cells and per-class recall (Wilson).
* **(b) M&Ms-2 categories.**  On the non-critical-QC rows of mms2_external with a recorded reconstruction: the ``d_model``
  decisions per released pathology category (``contracts/cohort_MMS2.parquet``, ``pathology``; cases without a
  decision counted), per-class recall on the same-definition subset (Wilson), and per category the number of cases whose
  OBSERVED measured LVEF or RVEF is below 40 %, beside the number with at least one OBSERVED EF.
* **(c) Cases covered by the B25 rulebook.**  The frozen B25 rulebook of the third source run
  (``postfreeze/B25/models/rulebook_pooled_d_model``, hash-checked) re-applied to mms2_external exactly as B25 applied it:
  the frozen cuts of the frozen run (their SHA-256 must equal the one B25 recorded), the decoded reconstructions and their
  stability, and the routing of Section 11.7.  The accepted cases and their agreements with ``d_model`` must reproduce
  the recorded ``coverage_fidelity_mms2.parquet`` (numerators and denominators) and, under the signed tag, the 12
  accepted cases and 9 agreements of Section 13.9; otherwise the analysis refuses.  Reported: the released categories of
  the accepted cases only in cells of at least ``statistics.min_cell`` cases, smaller categories pooled as ``other`` and
  ``other`` suppressed below ``min_cell`` (D51), and the counts of the ``d_model`` decisions of the cases where the rule
  and ``d_model`` disagree (by rule label).

Values derived from exact counts carry a half-up companion (BLD-1); no output carries a case or subject identifier.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .b16_coverage_fidelity import route_labels, system_metrics
from .common import COMMAND, DATASET, OUTPUT_ROOT, clean, formal_a, load_partition, reference_labels, split_manifest
from .lane_v604 import block_partitions, identifier_tokens, proportion, write_checked_json, write_checked_table

ANALYSIS = "B30"
CONFUSION = ("acdc_test", "mms_external")
MMS2 = "mms2_external"
B25 = "B25"
B25_ESTIMATOR = "rulebook_pooled_d_model"
RULEBOOK = "d_model"
DECLARED_B25 = {"n_accepted": 12, "fidelity_k": 9}    # Section 13.9 B30 (c): the recorded 12 accepted cases and 9 agreements
OTHER = "other"
MISSING = "missing"                             # an accepted case without a released category (counted, never a category)
SUPPRESSED = "SUPPRESSED_BELOW_MIN_CELL"
EF_THRESHOLD = 40.0
EF_TARGETS = ("lv_ef_percent", "rv_ef_percent")


def b25_rel(*parts: str) -> str:
    return "/".join([OUTPUT_ROOT, B25, *parts])


def _known(values: pd.Series) -> pd.Series:
    return values.notna() & values.map(lambda v: v is not None and str(v) not in ("", "nan", "None"))


# ----------------------------------------------------------------------------------------------- (a) recorded confusion


def summary_rows(pctx: Any, split: str, decisions: pd.Series) -> list[str]:
    """The rows of ``summarize-predictor``: partition members with a recorded decision, critical-QC rows removed."""

    from ...stages import critical_qc_mask

    manifest = split_manifest(pctx, split)
    members = sorted(manifest.loc[manifest["partition"].astype(str) == split, "sample_id"].astype(str))
    ids = [s for s in members if s in decisions.index]
    b = pctx.read_parquet(f"semantics/B_{DATASET[split]}.parquet").set_index("sample_id")
    critical = set(b.index[critical_qc_mask(b, pctx.study).to_numpy(dtype=bool)].astype(str))
    return [s for s in ids if s not in critical]


def confusion_cells(reference: pd.Series, decision: pd.Series) -> dict[tuple[str, str], int]:
    """Nonzero cells (reference class, decision) on the rows with a known reference and a decision."""

    rows = _known(reference) & _known(decision.reindex(reference.index))
    pairs = pd.DataFrame({"r": reference[rows].astype(str), "d": decision.reindex(reference.index)[rows].astype(str)})
    return {(str(r), str(d)): int(n) for (r, d), n in pairs.value_counts().items() if int(n)}


def check_confusion(recorded: pd.DataFrame, recomputed: Mapping[tuple[str, str], int], split: str) -> None:
    """The recorded cells of ``split`` must equal the recomputed ones (zero cells ignored); BundleError otherwise."""

    from ...stages import BundleError

    rec = {(str(r), str(d)): int(n) for r, d, n in recorded[["reference_class", "d_model", "n"]].itertuples(index=False) if int(n)}
    if rec != dict(recomputed):
        cells = sorted(set(rec) ^ set(recomputed) | {c for c in set(rec) & set(recomputed) if rec[c] != recomputed[c]})
        raise BundleError(f"{COMMAND} {ANALYSIS}: metrics/predictor_confusion.parquet of {split} differs from the recorded decisions and reference "
                          f"labels in {len(cells)} cells; refused")


def recall_rows(cells: Mapping[tuple[str, str], int], classes: list[str], **labels: Any) -> list[dict[str, Any]]:
    rows = []
    for c in classes:
        n = int(sum(v for (r, _), v in cells.items() if r == c))
        rows.append({**labels, "reference_class": c, **proportion("recall", int(cells.get((c, c), 0)), n)})
    return rows


# ----------------------------------------------------------------------------------------------- (c) small cells (D51)


def small_cells(categories: pd.Series, min_cell: int) -> list[dict[str, Any]]:
    """Counts per category in cells of at least ``min_cell`` cases; smaller categories pooled as ``other``, which is
    suppressed (count withheld) when it has fewer than ``min_cell`` cases.  The pooled categories are never named.  Cases
    without a released category are counted as ``missing``, so the cells, ``other`` and ``missing`` add up to the cases."""

    missing = int(categories.isna().sum())
    counts = categories.dropna().astype(str).value_counts()
    shown = sorted(c for c, n in counts.items() if int(n) >= int(min_cell))
    pooled = int(counts[[c for c in counts.index if c not in shown]].sum())
    rows = [{"category": c, "n": int(counts[c]), "status": "REPORTED"} for c in shown]
    if pooled:
        rows.append({"category": OTHER, "n": pooled if pooled >= int(min_cell) else None, "status": "REPORTED" if pooled >= int(min_cell) else SUPPRESSED})
    if missing:
        rows.append({"category": MISSING, "n": missing, "status": "CATEGORY_MISSING"})
    return rows


def check_b25_reproduction(recorded: pd.Series, metrics: Mapping[str, Any], declared: Mapping[str, int] | None) -> list[dict[str, Any]]:
    """The re-applied rulebook against the recorded mms2_external row (n_accepted, coverage and fidelity numerators and
    denominators) and, when given, the declared values of Section 13.9; BundleError otherwise."""

    from ...stages import BundleError

    new = {"n_attempted": metrics["n_attempted"], "n_processed": metrics["n_processed"], "n_accepted": metrics["n_accepted"],
           "coverage_k": metrics["coverage"]["k"], "coverage_n": metrics["coverage"]["n"], "fidelity_k": metrics["fidelity"]["k"],
           "fidelity_n": metrics["fidelity"]["n"]}
    rows, failures = [], []
    for key, value in new.items():
        rec = recorded.get(key)
        same = rec is not None and not pd.isna(rec) and int(rec) == int(value)
        rows.append({"quantity": key, "recomputed": int(value), "recorded": None if rec is None or pd.isna(rec) else int(rec), "reproduced": bool(same)})
        if not same:
            failures.append(f"{key} differs from the recorded B25 row")
    for key, value in (declared or {}).items():
        same = int(new[key]) == int(value)
        rows.append({"quantity": f"{key} (Section 13.9)", "recomputed": int(new[key]), "recorded": int(value), "reproduced": bool(same)})
        if not same:
            failures.append(f"{key} differs from Section 13.9")
    if failures:
        raise BundleError(f"{COMMAND} {ANALYSIS}: the frozen B25 rulebook re-applied to {MMS2} does not reproduce the recorded coverage and "
                          f"fidelity ({'; '.join(failures)}); refused")
    return rows


def reapply_b25(pctx: Any, records: list[dict[str, Any]], cuts: Any) -> tuple[pd.Series, pd.Series, pd.Series, dict[str, Any]]:
    """(rule label, processed, d_model, system metrics) of the B25 rulebook on mms2_external, as B25 evaluated it."""

    from ...formal_features import formal_descriptor_names
    from ...rule_pipeline import decode_reconstructions, qc_route_frame, route_records, routing_substrate
    from ...stages import fold_model_paths, load_fold_models
    from ...transition_fit import TransitionModel

    study = pctx.study
    names = formal_descriptor_names()
    model = TransitionModel.from_dict(pctx.read_json("transitions/transition_model.json"))
    for rel in [p.relative_to(pctx.source_root).as_posix() for p in fold_model_paths(pctx.stage_ctx, pctx.stage_ctx.run)]:
        pctx.source_file(rel)
    fold_models = load_fold_models(pctx.stage_ctx, pctx.stage_ctx.run)
    keep = [str(s) for s in pctx.read_parquet(f"reconstructed/bhat_{MMS2}.parquet")["sample_id"]]
    ds = DATASET[MMS2]
    a_p = formal_a(pctx, ds).reindex(keep)[names].to_numpy(dtype=float)
    b_p = pctx.read_parquet(f"semantics/B_{ds}.parquet").set_index("sample_id").reindex(keep)
    d_model = pctx.read_parquet(f"inference/d_model_{MMS2}.parquet").set_index("sample_id")["d_model"].reindex(keep)
    qc = qc_route_frame(b_p, np.isfinite(a_p).all(axis=1), study)
    dec, st, fold_dec = decode_reconstructions(model, fold_models, a_p, pd.Index(keep, name="sample_id"), study.registry, study.transition)
    substrate, _ = routing_substrate(dec, st, fold_dec, cuts, study.wedd, study.rule_candidates())
    routes = route_records(records, RULEBOOK, cuts, substrate, qc, study)
    label, processed = route_labels(routes)
    return label, processed, d_model, system_metrics(label, processed, d_model, reference_labels(pctx, MMS2).reindex(keep))


def mms2_categories(pctx: Any, split: str | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], pd.Series, list[str]]:
    """(b) on ``split`` (mms2_external): rows per released pathology category, per-class recall on the same-definition
    subset, counts, the released category of every row (index sample_id) and the row ids (kept in the run root)."""

    split = split or MMS2
    part = load_partition(pctx, split)
    cohort = pctx.read_parquet(f"contracts/cohort_{DATASET[split]}.parquet").set_index("sample_id")
    released = cohort["pathology"].astype(object)
    pathology = released.reindex(part.ids)
    decisions = pctx.read_parquet(f"inference/d_model_{split}.parquet").set_index("sample_id")["d_model"].reindex(part.ids).astype(object)
    observed = part.observed(list(EF_TARGETS))
    low = pd.concat([(observed[t] < EF_THRESHOLD) for t in EF_TARGETS], axis=1).any(axis=1)
    any_ef = observed.notna().any(axis=1)
    d_classes = sorted(decisions.dropna().astype(str).unique())
    rows = []
    for category in sorted(pathology.dropna().astype(str).unique()) + (["<missing>"] if pathology.isna().any() else []):
        member = (pathology.astype(str) == category) if category != "<missing>" else pathology.isna()
        d_c = decisions[member]
        rows.append({"partition": split, "pathology": category, "n_cases": int(member.sum()), "n_no_decision": int(d_c.isna().sum()),
                     **{f"n_d_model_{c}": int((d_c.astype(str) == c).sum()) for c in d_classes},
                     "n_with_observed_ef": int(any_ef[member].sum()), "n_lvef_or_rvef_below_40": int(low[member].sum())})
    reference = reference_labels(pctx, split).reindex(part.ids)
    recall = recall_rows(confusion_cells(reference, decisions), sorted(reference.dropna().astype(str).unique()), partition=split, part="b",
                         reference="same-definition crosswalk subset")
    note = {"n_rows": len(part.ids), "n_no_decision": int(decisions.isna().sum()), "n_same_definition": int(reference.notna().sum()),
            "n_same_definition_without_decision": int((reference.notna() & decisions.isna()).sum())}
    return rows, recall, note, released, list(part.ids)


def b25_covered(pctx: Any, released: pd.Series, min_cell: int) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    """(c): the frozen B25 rulebook re-applied to mms2_external after the checks of its cuts and its recorded counts;
    rows (released categories of the accepted cases under D51; d_model decisions of the disagreeing cases), a note and
    the row ids (kept in the run root)."""

    from ...runs import hash_path
    from ...stages import BundleError
    from ...wedd import CutSet

    recorded = pctx.read_third_parquet(b25_rel("coverage_fidelity_mms2.parquet"))
    rec = recorded[recorded["partition"].astype(str) == MMS2]
    if len(rec) != 1:
        raise ValueError(f"{ANALYSIS}: coverage_fidelity_mms2.parquet of B25 holds {len(rec)} {MMS2} rows")
    rec = rec.iloc[0]
    status = str(rec.get("status"))
    if pctx.tag_state == "protocol_frozen" and status != "EVALUATED":
        raise BundleError(f"{COMMAND} {ANALYSIS}: the recorded B25 row of {MMS2} has status {status}; under the signed tag it must be EVALUATED "
                          f"and reproduce {DECLARED_B25['n_accepted']} accepted cases and {DECLARED_B25['fidelity_k']} agreements (Section 13.9); refused")
    if status not in ("EVALUATED", "EMPTY_RULEBOOK"):
        return [], {"status": "NOT_EXECUTED", "reason": f"B25 recorded no evaluation of {MMS2} (status {status})"}, []
    frozen = pctx.read_third_json(b25_rel("models", B25_ESTIMATOR, "FROZEN.json"))
    records = list(pctx.read_third_json(b25_rel("models", B25_ESTIMATOR, str(frozen["rulebook_file"]))).get("records", []))
    cuts_rel = f"thresholds/d_model_{pctx.study.wedd['primary_profile']}.json"
    cuts_path = pctx.source_file(cuts_rel)
    if cuts_rel != frozen.get("cuts_file") or hash_path(cuts_path) != frozen.get("cuts_sha256"):
        raise BundleError(f"{COMMAND} {ANALYSIS}: the frozen cuts {cuts_rel} differ from those B25 recorded in its FROZEN.json; refused")
    label, processed, d_model, metrics = reapply_b25(pctx, records, CutSet.from_dict(pctx.read_json(cuts_rel)))
    declared = DECLARED_B25 if pctx.tag_state == "protocol_frozen" else None
    reproduction = check_b25_reproduction(rec, metrics, declared)
    accepted = processed.reindex(label.index).fillna(False).astype(bool) & label.notna()
    categories = released.reindex(label.index[accepted.to_numpy()]).astype(object)
    rows = [{"part": "accepted_by_released_category", "min_cell": int(min_cell), **r} for r in small_cells(categories, min_cell)]
    dm = d_model.reindex(label.index)
    disagree = accepted & dm.notna() & (label.astype(str) != dm.astype(str))
    pairs = pd.DataFrame({"rule_label": label[disagree].astype(str), "d_model": dm[disagree].astype(str)}).value_counts()
    rows.extend({"part": "disagreement_d_model_decision", "rule_label": str(r), "d_model": str(d), "n": int(n), "status": "REPORTED"}
                for (r, d), n in sorted(pairs.items()))
    note = {"status": "EVALUATED" if records else "EMPTY_RULEBOOK", "n_rules": len(records), "reproduction": reproduction,
            "declared_checked": declared is not None, "n_accepted": int(accepted.sum()), "n_category_missing": int(categories.isna().sum()),
            "n_disagreeing": int(disagree.sum()),
            "routing": "Section 11.7 (route_records with the frozen cuts and the decoded stability)"}
    return rows, note, [str(s) for s in label.index]


def run(pctx: Any) -> dict[str, Any]:
    acfg = pctx.analysis_cfg(ANALYSIS)
    out = pctx.out_dir(ANALYSIS)
    min_cell = int(pctx.lane_cfg["statistics"]["min_cell"])
    if [str(p) for p in acfg["partitions"]] != [*CONFUSION, MMS2]:
        raise ValueError(f"{ANALYSIS}: the partitions are {[*CONFUSION, MMS2]} (Section 13.9), got {acfg['partitions']}")
    included, skipped = block_partitions(pctx, [str(p) for p in acfg["partitions"]], f"{COMMAND} {ANALYSIS}")
    splits = [s for _, s in included]
    confusion_rows, recall, category_rows, covered_rows, tokens, notes = [], [], [], [], set(), {}
    # (a) the recorded confusion of acdc_test and mms_external.
    recorded_confusion = pctx.read_parquet("metrics/predictor_confusion.parquet") if any(s in CONFUSION for s in splits) else None
    for split in [s for s in CONFUSION if s in splits]:
        decisions = pctx.read_parquet(f"inference/d_model_{split}.parquet").set_index("sample_id")["d_model"].astype(object)
        ids = summary_rows(pctx, split, decisions)
        tokens |= identifier_tokens(ids)
        cells = confusion_cells(reference_labels(pctx, split).reindex(ids), decisions.reindex(ids))
        recorded = recorded_confusion[recorded_confusion["partition"].astype(str) == split]
        if recorded.empty:
            raise ValueError(f"{ANALYSIS}: metrics/predictor_confusion.parquet records no {split} cells")
        check_confusion(recorded, cells, split)
        classes = list(dict.fromkeys(recorded["reference_class"].astype(str)))
        confusion_rows.extend({"partition": split, "reference_class": str(r), "d_model": str(d), "n": int(n), "checked_against_recorded_decisions": True}
                              for r, d, n in recorded[["reference_class", "d_model", "n"]].itertuples(index=False))
        recall.extend(recall_rows(cells, classes, partition=split, part="a", reference="recorded confusion (summarize-predictor rows)"))
        notes[split] = {"n_rows": len(ids), "n_cells_nonzero": len(cells), "confusion_reproduced": True}
    # (b) and (c) on mms2_external.
    if MMS2 in splits:
        category_rows, rows_b, notes[MMS2], released, ids_b = mms2_categories(pctx)
        recall.extend(rows_b)
        covered_rows, notes["b25_covered"], ids_c = b25_covered(pctx, released, min_cell)
        tokens |= identifier_tokens([*ids_b, *ids_c])
    elif pctx.tag_state == "protocol_frozen":
        from ...stages import BundleError

        raise BundleError(f"{COMMAND} {ANALYSIS}: (c) must reproduce the recorded B25 counts of {MMS2} (Section 13.9), but {MMS2} is not "
                          f"executed ({skipped.get(MMS2)}); refused")
    else:
        notes["b25_covered"] = {"status": "NOT_EXECUTED", "reason": skipped.get(MMS2)}
    paths = {"confusion": out / "confusion_table.parquet", "recall": out / "recall_table.parquet", "categories": out / "mms2_category_table.parquet",
             "covered": out / "b25_covered_table.parquet", "summary": out / "summary.json"}
    write_checked_table(pd.DataFrame(confusion_rows, columns=["partition", "reference_class", "d_model", "n", "checked_against_recorded_decisions"]),
                        paths["confusion"], pctx, tokens)
    write_checked_table(pd.DataFrame(recall), paths["recall"], pctx, tokens)
    write_checked_table(pd.DataFrame(category_rows), paths["categories"], pctx, tokens)
    write_checked_table(pd.DataFrame(covered_rows, columns=["part", "min_cell", "category", "rule_label", "d_model", "n", "status"]), paths["covered"],
                        pctx, tokens)
    summary = clean({"analysis": ANALYSIS, "label": "EXPLORATORY_POST_HOC", "reading": None, "recorded_confusion": acfg["recorded_confusion"],
                     "mms2_categories": acfg["mms2_categories"], "b25_covered": acfg["b25_covered"], "min_cell": min_cell,
                     "small_cells": "D51: categories below min_cell pooled as other; other suppressed below min_cell", "checks": notes,
                     "included": [lab for lab, _ in included], "not_executed": skipped})
    write_checked_json(summary, paths["summary"], pctx, tokens)
    pctx.write_provenance(ANALYSIS, dict(pctx.verified_log), list(paths.values()),
                          {"included": [lab for lab, _ in included], "not_executed": skipped, "reading": None,
                           "b25_covered": (notes.get("b25_covered") or {}).get("status")})
    return {"included": [lab for lab, _ in included], "not_executed": skipped}
