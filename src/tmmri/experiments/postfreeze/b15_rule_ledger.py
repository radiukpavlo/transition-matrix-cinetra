"""B15: candidate-level rule ledger of the three rulebooks (EPC-4; ``configs/exploratory_postfreeze.yaml`` analyses.B15).

* Concept table: the nine rule concepts with their recorded eligibility (``rules/semantic_eligibility.parquet``).
* Candidates: every conjunction of at most ``max_length`` feature-state literals on the recorded eligible concepts that
  discovery enumerates (attributes with at least two available states), with at least one supporting development
  patient, from the recorded granules (``granules/<rb>_<profile>_states_ACDC_dev.parquet``) and frozen cuts: the
  majority consequent (ties by name, as discovery), support (matching patients), joint support, confidence and the
  smallest recurrence of the cuts bounding its literals.
* Pipeline: the first failed condition of ``pipeline_order`` (concept eligibility, granule support, granule
  confidence, cross-fit reproduction, pooled held-out matches, Wilson lower bound).  A candidate passing support and
  confidence reaches the cross-fitted gate as the rule discovery makes of it (greedy backward deletion; a redundant
  superset is represented by the recorded rule it contains); its gate statistics are the recorded ``gate.parquet`` row
  where present, otherwise recomputed with :func:`tmmri.rule_pipeline.gate_and_apply` and the recorded nested substrate
  (:class:`tmmri.rule_pipeline.NestedGateSubstrate` with the recorded inner out-of-fold reconstructions).
* Agreement: against the fold-specific audited predictor refitted on each outer-training set (as the recorded gate),
  and, labelled in-sample, against the final frozen predictor on the same held-out matches (``d_model`` consequents).
"""

from __future__ import annotations

import itertools
import re
from typing import Any

import numpy as np
import pandas as pd

from .common import clean, write_json, write_table
from .estimates import wilson

ANALYSIS = "B15"
CONDITIONS = ("concept_eligibility", "granule_support", "granule_confidence", "crossfit_reproduction", "pooled_matches", "wilson_lower")
PASSED = "PASSED_ALL"


def pipeline_thresholds(order: list[str], policy: dict[str, Any]) -> dict[str, float]:
    """The thresholds named in ``pipeline_order``; each must equal the frozen ``configs/rule_policy.yaml`` value."""

    names = [re.sub(r"_min_[0-9.]+$", "", str(n)) for n in order]
    if tuple(names) != CONDITIONS:
        raise ValueError(f"{ANALYSIS}: pipeline_order must be {list(CONDITIONS)} (with their minima), got {order}")
    named = {re.sub(r"_min_[0-9.]+$", "", str(n)): float(m.group(1)) for n in order if (m := re.search(r"_min_([0-9.]+)$", str(n)))}
    frozen = {"granule_support": float(policy["discovery"]["min_antecedent_n_patients"]),
              "granule_confidence": float(policy["discovery"]["tiers"]["primary"]["min_confidence"]),
              "pooled_matches": float(policy["validation_gate"]["min_pooled_oof_matched_patients"]),
              "wilson_lower": float(policy["validation_gate"]["agreement_wilson_lower_min"])}
    for key, value in frozen.items():
        if abs(named.get(key, np.nan) - value) > 1e-12:
            raise ValueError(f"{ANALYSIS}: pipeline_order {key} minimum {named.get(key)} differs from configs/rule_policy.yaml ({value})")
    return frozen


def literal_recurrence(cuts: Any, stability: pd.DataFrame, feature: str, token: str) -> float | None:
    """Smallest bootstrap recurrence of the frozen cuts bounding the bin ``token`` of ``feature`` (None: no cut)."""

    from ...wedd import parse_state_token

    parsed = parse_state_token(token)
    if parsed is None or feature not in cuts.features:
        return None
    fc = cuts.features[feature]
    rows = stability[(stability["feature"] == feature) & stability["cut"].notna()]
    values = []
    for pos in (parsed[0] - 1, parsed[0]):
        if 0 <= pos < len(fc.cuts):
            match = rows[(rows["node_path"] == fc.node_paths[pos]) & np.isclose(rows["cut"].astype(float), float(fc.cuts[pos]), rtol=0, atol=1e-12)]
            values.append(float(match["recurrence"].iloc[0]) if len(match) else 0.0)
    return min(values) if values else None


def enumerate_candidates(states: pd.DataFrame, decision: np.ndarray, attributes: list[str], max_length: int) -> list[dict[str, Any]]:
    """Every indiscernibility class of every attribute subset of size 1..max_length (registry order) with its majority
    consequent (ties by name), size, joint support and confidence."""

    from ...rough_sets import indiscernibility_classes

    lookup = dict(zip(states.index, decision))
    out = []
    for size in range(1, max_length + 1):
        for subset in itertools.combinations(attributes, size):
            classes, _ = indiscernibility_classes(states, list(subset))
            for sig, members in sorted(classes.items()):
                counts: dict[str, int] = {}
                for m in members:
                    counts[lookup[m]] = counts.get(lookup[m], 0) + 1
                majority, joint = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
                out.append({"antecedent": tuple(zip(subset, sig)), "consequent": str(majority), "antecedent_n": len(members), "support": int(joint),
                            "confidence": joint / len(members), "class_counts": dict(sorted(counts.items()))})
    return out


def gate_tables(*, values: pd.DataFrame, gate_ids: np.ndarray, subset: np.ndarray, folds: np.ndarray, book: Any, rb: str, spec: dict[str, Any],
                study: Any, profile: str, attributes: list[str], decision_fn: Any, final_decision: pd.Series | None, substrate_fn: Any) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Cross-fitted gate tables of ``book``: against the fold-specific references (as recorded) and, for ``d_model``
    consequents, with the final frozen predictor's decisions as the held-out reference (same folds, same discovery)."""

    from ...rule_pipeline import gate_and_apply

    kw = dict(values=values.loc[gate_ids], folds=folds[subset], rule_policy=study.rule_policy, book=book, profile=profile, wedd_cfg=study.wedd,
              anchors=np.array([s.split(":", 1)[1] for s in gate_ids], dtype=object), substrate_fn=substrate_fn, registry=study.registry,
              rulebook_id=rb, attributes=attributes)
    fold_table, _ = gate_and_apply(decision_fn=decision_fn, **kw)
    if final_decision is None:
        return fold_table, None

    def final_fn(train: np.ndarray, target: np.ndarray) -> np.ndarray:
        if target is train or np.array_equal(np.asarray(train), np.asarray(target)):
            return decision_fn(train, target)          # discovery on the training folds stays exactly as the gate's
        return final_decision.reindex(gate_ids[np.asarray(target, dtype=int)]).to_numpy(dtype=object)

    final_table, _ = gate_and_apply(decision_fn=final_fn, **kw)
    return fold_table, final_table


def run(pctx: Any) -> dict[str, Any]:
    from ...classification import fit_formal_predictor
    from ...rough_sets import (
        Rule,
        Rulebook,
        _discovery_params,
        _reduce,
        _universe,
        antecedent_policy,
        make_rule_id,
        rule_identity,
        violates_exclusion,
    )
    from ...rule_pipeline import NestedGateSubstrate, acdc_inner_labels
    from ...stages import (
        _cohort,
        _dev_matrices,
        _rule_substrate,
        _rulebooks,
        predict_labels,
    )
    from ...wedd import CutSet

    acfg = pctx.analysis_cfg(ANALYSIS)
    out = pctx.out_dir(ANALYSIS)
    study = pctx.study
    policy = study.rule_policy
    limits = pipeline_thresholds(list(acfg["pipeline_order"]), policy)
    params = _discovery_params(policy)
    gate_cfg = policy["validation_gate"]
    sensitivities = [float(s) for s in gate_cfg.get("gate_sensitivities", [])]
    profile = str(study.wedd["primary_profile"])
    books = _rulebooks(pctx.stage_ctx)
    if sorted(books) != sorted(acfg["rulebooks"]):
        raise ValueError(f"{ANALYSIS}: rulebooks {sorted(books)} differ from configs/exploratory_postfreeze.yaml {acfg['rulebooks']}")
    # Every source file the v6 helpers below read, verified first.
    for rel in ("formal/A_ACDC.parquet", "semantics/B_ACDC.parquet", "contracts/split_manifest.parquet", "contracts/cohort_ACDC.parquet",
                "contracts/inner_folds.parquet", "oof/bhat_oof_ACDC_dev.parquet", "formal/d_model_ACDC_dev.parquet",
                "rules/semantic_eligibility.parquet", "rules/semantic_eligibility.json"):
        pctx.source_file(rel)
    eligibility = pd.read_parquet(pctx.source_root / "rules" / "semantic_eligibility.parquet")
    ids, a, b, folds_all = _dev_matrices(pctx.stage_ctx)
    row_ids = ids["sample_id"].astype(str).to_numpy()
    labels_all = _cohort(pctx.stage_ctx, "ACDC").set_index("sample_id").loc[row_ids, "acdc_class"].astype(str).to_numpy()
    pcfg = study.formal["audited_predictor"]
    oof = pd.read_parquet(pctx.source_root / "oof" / "bhat_oof_ACDC_dev.parquet").set_index("sample_id")
    inner = pd.read_parquet(pctx.source_root / "contracts" / "inner_folds.parquet")
    targets = study.transition_targets()
    candidates_cols = [c for c in study.rule_candidates() if c in oof.columns]
    shared = NestedGateSubstrate(a, b, targets, study.transition, study.registry, folds_all, acdc_inner_labels(inner, row_ids), oof, candidates_cols, row_ids)
    n_folds = int(gate_cfg["folds"])
    shared.preload({f: pd.read_parquet(pctx.source_file(f"oof/inner_oof_ACDC_dev_fold{f}.parquet")).set_index("sample_id") for f in range(1, n_folds + 1)
                    if (pctx.source_root / "oof" / f"inner_oof_ACDC_dev_fold{f}.parquet").is_file()})
    final_d = pctx.read_parquet("formal/d_model_ACDC_dev.parquet").set_index("sample_id")["d_model"]
    _, groups = antecedent_policy(study.registry)
    concept_rows, ledger_rows, rulebook_summary = [], [], {}
    for rb, spec in books.items():
        induced = pctx.read_json(f"rules/{rb}/rulebook_induced.json")
        book = Rulebook.from_records(induced["records"], rulebook_id=rb, consequent_kind=rb)
        attributes = [str(x) for x in induced.get("attributes", [])]
        states = pctx.read_parquet(f"granules/{rb}_{profile}_states_ACDC_dev.parquet").set_index("sample_id")
        cuts = CutSet.from_dict(pctx.read_json(f"thresholds/{rb}_{profile}.json"))
        stability = pctx.read_parquet(f"thresholds/{rb}_{profile}_stability.parquet")
        recorded_gate = pctx.read_parquet(f"rules/{rb}/gate.parquet") if pctx.exists(f"rules/{rb}/gate.parquet") else pd.DataFrame()
        values, decision = _rule_substrate(pctx.stage_ctx, spec)
        states = states.loc[[i for i in states.index if i in decision.index]]
        st, d = _universe(states, decision.loc[states.index])
        order = {a_: i for i, a_ in enumerate(attributes)}
        usable = [x for x in attributes if len({str(v) for v in st[x] if _available_token(v)}) >= 2]
        class_records = dict(induced.get("class_records") or {})
        eligible_classes = {c for c, r in class_records.items() if r.get("eligible")}
        # Concept table (recorded eligibility of the nine concepts).
        for row in eligibility[eligibility["rulebook"] == rb].to_dict("records"):
            concept_rows.append({**row, "in_discovery_attributes": row["feature"] in attributes, "usable_in_discovery": row["feature"] in usable})
        # Candidates and their discovery form.
        cands = enumerate_candidates(st, d, usable, int(params["max_length"]))
        by_identity = {r.identity: r for r in book.rules}
        extra: dict[str, Any] = {}
        for c in cands:
            ant = tuple(sorted(c["antecedent"], key=lambda lit: order[lit[0]]))
            c["antecedent"] = ant
            c["exclusion_group_violation"] = bool(violates_exclusion([f for f, _ in ant], groups))
            c["class_eligible"] = c["consequent"] in eligible_classes
            c["concept_eligible"] = all(f in attributes for f, _ in ant)
            c["passes"] = {"concept_eligibility": c["concept_eligible"] and c["class_eligible"] and not c["exclusion_group_violation"],
                           "granule_support": c["antecedent_n"] >= limits["granule_support"],
                           "granule_confidence": c["confidence"] >= limits["granule_confidence"] - 1e-12}
            c["mapping"], c["gate_rule"] = None, None
            if all(c["passes"].values()):
                reduced = _reduce(st, d, ant, c["consequent"], params, order)
                ident = rule_identity(reduced, c["consequent"])
                if ident in by_identity:
                    c["gate_rule"], c["mapping"] = by_identity[ident], ("IRREDUCIBLE" if len(reduced) == len(ant) else "REDUCED")
                else:
                    covering = [r for r in book.rules if r.consequent == c["consequent"] and set(r.antecedent) < set(reduced)]
                    if covering:
                        c["gate_rule"], c["mapping"] = covering[0], "REDUNDANT_SUPERSET"
                    else:   # not in the recorded discovery: gated as it stands (recomputed)
                        an_n, joint = int(c["antecedent_n"]), int(c["support"])
                        rule = Rule(rule_id=make_rule_id(rb, reduced, c["consequent"]), antecedent=tuple(reduced), consequent=c["consequent"], support=joint,
                                    confidence=c["confidence"], strict=joint == an_n, anchors=(), antecedent_n=an_n, consequent_kind=rb)
                        extra.setdefault(rule.identity, rule)
                        c["gate_rule"], c["mapping"] = extra[rule.identity], "NOT_IN_RECORDED_DISCOVERY"
                c["reduced_antecedent"] = reduced
        derived = {rule_identity(c["reduced_antecedent"], c["consequent"]) for c in cands if c.get("mapping") in ("IRREDUCIBLE", "REDUCED")}
        # Cross-fitted gate: fold-specific references (as recorded) and the final predictor (held-out reference only).
        subset = np.flatnonzero(pd.Index(row_ids).isin(values.index))
        gate_ids = row_ids[subset]
        cache: dict[tuple[int, ...], Any] = {}

        def d_model_fn(train: np.ndarray, target: np.ndarray, _subset: np.ndarray = subset, _cache: dict = cache) -> np.ndarray:
            key = tuple(int(i) for i in train)
            if key not in _cache:
                _cache[key] = fit_formal_predictor(a[_subset[train]], pd.Series(labels_all[_subset[train]]), pcfg)
            return predict_labels(_cache[key], a[_subset[target]])

        def label_fn(train: np.ndarray, target: np.ndarray, _subset: np.ndarray = subset) -> np.ndarray:
            return labels_all[_subset[target]]

        is_label = spec["consequent"] == "Y_acdc_class"
        nested = spec["substrate"] != "measured_B"
        gated_book = Rulebook([*book.rules, *extra.values()], rb, rb, class_records, [], {})
        fold_table, final_table = gate_tables(values=values, gate_ids=gate_ids, subset=subset, folds=folds_all, book=gated_book, rb=rb, spec=spec,
                                              study=study, profile=profile, attributes=attributes, decision_fn=label_fn if is_label else d_model_fn,
                                              final_decision=None if is_label else final_d, substrate_fn=shared.with_subset(subset) if nested else None)
        fold_by_id = fold_table.set_index("rule_id") if len(fold_table) else pd.DataFrame()
        final_by_id = final_table.set_index("rule_id") if final_table is not None and len(final_table) else pd.DataFrame()
        recorded_by_id = recorded_gate.set_index("rule_id") if len(recorded_gate) else pd.DataFrame()
        reproduces = {}
        for rule in book.rules:
            if rule.rule_id in recorded_by_id.index and rule.rule_id in fold_by_id.index:
                rec, new = recorded_by_id.loc[rule.rule_id], fold_by_id.loc[rule.rule_id]
                reproduces[rule.rule_id] = all(int(rec[k]) == int(new[k]) for k in ("oof_matched", "oof_agree", "folds_reproduced"))
        for c in cands:
            ant_text = " & ".join(f"{f}={s}" for f, s in c["antecedent"])
            recs = [literal_recurrence(cuts, stability, f, s) for f, s in c["antecedent"]]
            recs = [r for r in recs if r is not None]
            row = {"rulebook": rb, "candidate": f"{ant_text} -> {c['consequent']}", "length": len(c["antecedent"]),
                   "features": ";".join(f for f, _ in c["antecedent"]), "consequent": c["consequent"], "antecedent_n_patients": c["antecedent_n"],
                   "joint_support_patients": c["support"], "confidence": c["confidence"],
                   "class_counts": ";".join(f"{k}:{v}" for k, v in c["class_counts"].items()),
                   "min_literal_cut_recurrence": min(recs) if recs else None, "concept_eligible": c["concept_eligible"], "class_eligible": c["class_eligible"],
                   "exclusion_group_violation": c["exclusion_group_violation"], "discovery_mapping": c["mapping"],
                   "gate_rule_id": c["gate_rule"].rule_id if c["gate_rule"] is not None else None,
                   "gate_rule": (" & ".join(f"{f}={s}" for f, s in c["gate_rule"].antecedent) + f" -> {c['gate_rule'].consequent}") if c["gate_rule"] is not None else None}
            passes = dict(c["passes"])
            gate = None
            if c["gate_rule"] is not None:
                rid = c["gate_rule"].rule_id
                if rid in recorded_by_id.index:
                    gate, source = recorded_by_id.loc[rid], "recorded gate.parquet"
                else:
                    gate, source = fold_by_id.loc[rid], "recomputed (recorded nested substrate)"
                row.update({"gate_source": source, "recomputation_matches_recorded": reproduces.get(rid),
                            "folds_reproduced": int(gate["folds_reproduced"]), "reproduced_in_folds": str(gate["reproduced_in_folds"]),
                            "oof_matched": int(gate["oof_matched"]), "oof_agree": int(gate["oof_agree"]),
                            "agreement": None if pd.isna(gate["agreement"]) else float(gate["agreement"]),
                            "wilson_lb": None if pd.isna(gate["wilson_lb"]) else float(gate["wilson_lb"]),
                            "wilson_ub": None if pd.isna(gate["wilson_ub"]) else float(gate["wilson_ub"]),
                            "agreement_reference": "dataset label (Y_acdc_class)" if is_label else "fold-specific audited predictor (refitted without the held-out fold)"})
                for s in sensitivities:
                    col = f"eligible_at_{s:.2f}"
                    row[col] = bool(gate[col]) if col in gate.index else None
                if not is_label and rid in final_by_id.index:
                    fin = final_by_id.loc[rid]
                    w = wilson(int(fin["oof_agree"]), int(fin["oof_matched"]))
                    row.update({"final_predictor_matched": int(fin["oof_matched"]), "final_predictor_agree": int(fin["oof_agree"]),
                                "final_predictor_agreement_in_sample": w["estimate"], "final_predictor_wilson_lb_in_sample": w["ci_lower"],
                                "final_predictor_wilson_ub_in_sample": w["ci_upper"]})
                passes["crossfit_reproduction"] = int(gate["folds_reproduced"]) >= 1
                passes["pooled_matches"] = int(gate["oof_matched"]) >= limits["pooled_matches"]
                passes["wilson_lower"] = bool(not pd.isna(gate["wilson_lb"]) and float(gate["wilson_lb"]) >= limits["wilson_lower"])
            first = next((cond for cond in CONDITIONS if not passes.get(cond, False)), PASSED)
            row.update({f"passes_{k}": passes.get(k) for k in CONDITIONS})
            row["first_failed_condition"] = first if first == PASSED else str(acfg["pipeline_order"][CONDITIONS.index(first)])
            ledger_rows.append(row)
        rulebook_summary[rb] = {"attributes": attributes, "usable_attributes": usable, "n_candidates": len(cands), "n_recorded_rules": len(book.rules),
                                "n_recomputed_rules": len(extra), "discovery_reproduced": derived == set(by_identity),
                                "gate_recomputation_matches_recorded": reproduces, "agreement_reference": "dataset label" if is_label else
                                "fold-specific audited predictor; final frozen predictor beside it (in-sample)",
                                "gate_substrate": "nested_inner_oof" if nested else "measured_B"}
    ledger = pd.DataFrame(ledger_rows)
    counts = (ledger.groupby(["rulebook", "first_failed_condition"]).size().rename("n_candidates").reset_index() if len(ledger)
              else pd.DataFrame(columns=["rulebook", "first_failed_condition", "n_candidates"]))
    paths = {"concepts": out / "concept_table.parquet", "ledger": out / "candidate_ledger.parquet", "counts": out / "rejection_counts_by_reason.parquet",
             "summary": out / "summary.json"}
    write_table(pd.DataFrame(concept_rows), paths["concepts"], pctx)
    write_table(ledger, paths["ledger"], pctx)
    write_table(counts, paths["counts"], pctx)
    summary = clean({"analysis": ANALYSIS, "pipeline_order": acfg["pipeline_order"], "limits": limits, "max_length": params["max_length"],
                     "candidates": acfg["candidates"], "agreement_reference": acfg["agreement_reference"], "rulebooks": rulebook_summary,
                     "rejection_counts": counts.to_dict("records")})
    write_json(summary, paths["summary"], pctx)
    pctx.write_provenance(ANALYSIS, dict(pctx.verified_log), list(paths.values()),
                          {"n_candidates": {rb: v["n_candidates"] for rb, v in rulebook_summary.items()},
                           "discovery_reproduced": {rb: v["discovery_reproduced"] for rb, v in rulebook_summary.items()},
                           "passed_all": int((ledger["first_failed_condition"] == PASSED).sum()) if len(ledger) else 0})
    return {"rulebooks": sorted(rulebook_summary), "n_candidates": int(len(ledger))}


def _available_token(value: Any) -> bool:
    from ...wedd import is_available_state

    return is_available_state(value)
