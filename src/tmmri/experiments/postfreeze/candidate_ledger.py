"""Candidate ledger of a rulebook fitted inside the post-freeze block (B24, B25; "as B15").

The rules of B15 (:mod:`b15_rule_ledger`) applied to a rulebook that the block fits itself rather than to a recorded
one: every conjunction of at most ``max_length`` feature-state literals over the eligible attributes with at least two
available states (:func:`b15_rule_ledger.enumerate_candidates`), its majority consequent, support and confidence; for a
candidate passing concept eligibility, granule support and granule confidence, the discovery form (greedy backward
deletion; IRREDUCIBLE, REDUCED, REDUNDANT_SUPERSET of an induced rule, or NOT_IN_DISCOVERY, gated as it stands) and its
cross-fitted gate statistics; and the first failed condition of the B15 pipeline order.  The minima are those of the
frozen ``configs/rule_policy.yaml`` (checked with :func:`b15_rule_ledger.pipeline_thresholds`).
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from .b15_rule_ledger import CONDITIONS, PASSED, enumerate_candidates, literal_recurrence, pipeline_thresholds


def pipeline_order(policy: dict[str, Any]) -> list[str]:
    """The B15 pipeline order with the minima of the frozen rule policy (``granule_support_min_5`` ...)."""

    d, g = policy["discovery"], policy["validation_gate"]
    order = ["concept_eligibility", f"granule_support_min_{float(d['min_antecedent_n_patients']):g}",
             f"granule_confidence_min_{float(d['tiers']['primary']['min_confidence']):.2f}", "crossfit_reproduction",
             f"pooled_matches_min_{float(g['min_pooled_oof_matched_patients']):g}", f"wilson_lower_min_{float(g['agreement_wilson_lower_min']):.2f}"]
    pipeline_thresholds(order, policy)
    return order


def ledger(*, rulebook: str, states: pd.DataFrame, decision: pd.Series, attributes: list[str], book: Any, gate_table: pd.DataFrame,
           gate_fn: Callable[[Any], pd.DataFrame], policy: dict[str, Any], registry: dict[str, Any], cuts: Any = None,
           stability: pd.DataFrame | None = None, agreement_reference: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """(candidate ledger, summary) of ``book`` (induced on ``states`` and ``decision``) with its gate table; candidates
    outside the discovery are gated by ``gate_fn(Rulebook)``."""

    from ...rough_sets import Rule, Rulebook, _discovery_params, _reduce, _universe, antecedent_policy, make_rule_id, rule_identity, violates_exclusion
    from ...wedd import is_available_state

    order_names = pipeline_order(policy)
    limits = pipeline_thresholds(order_names, policy)
    params = _discovery_params(policy)
    st, d = _universe(states, decision.reindex(states.index))
    order = {a: i for i, a in enumerate(attributes)}
    usable = [x for x in attributes if x in st.columns and len({str(v) for v in st[x] if is_available_state(v)}) >= 2]
    _, groups = antecedent_policy(registry)
    eligible_classes = {c for c, r in (book.class_records or {}).items() if r.get("eligible")}
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
            c["reduced_antecedent"] = reduced
            ident = rule_identity(reduced, c["consequent"])
            if ident in by_identity:
                c["gate_rule"], c["mapping"] = by_identity[ident], ("IRREDUCIBLE" if len(reduced) == len(ant) else "REDUCED")
            else:
                covering = [r for r in book.rules if r.consequent == c["consequent"] and set(r.antecedent) < set(reduced)]
                if covering:
                    c["gate_rule"], c["mapping"] = covering[0], "REDUNDANT_SUPERSET"
                else:
                    an_n, joint = int(c["antecedent_n"]), int(c["support"])
                    rule = Rule(rule_id=make_rule_id(book.rulebook_id, reduced, c["consequent"]), antecedent=tuple(reduced), consequent=c["consequent"],
                                support=joint, confidence=c["confidence"], strict=joint == an_n, anchors=(), antecedent_n=an_n,
                                consequent_kind=book.consequent_kind)
                    extra.setdefault(rule.identity, rule)
                    c["gate_rule"], c["mapping"] = extra[rule.identity], "NOT_IN_DISCOVERY"
    tables = [gate_table]
    if extra:
        tables.append(gate_fn(Rulebook(list(extra.values()), book.rulebook_id, book.consequent_kind, dict(book.class_records or {}), [], {})))
    gates = pd.concat([t for t in tables if len(t)], ignore_index=True).set_index("rule_id") if any(len(t) for t in tables) else pd.DataFrame()
    sensitivities = [float(s) for s in policy["validation_gate"].get("gate_sensitivities", [])]
    rows = []
    for c in cands:
        recs = [literal_recurrence(cuts, stability, f, s) for f, s in c["antecedent"]] if cuts is not None and stability is not None else []
        recs = [r for r in recs if r is not None]
        row = {"rulebook": rulebook, "candidate": " & ".join(f"{f}={s}" for f, s in c["antecedent"]) + f" -> {c['consequent']}",
               "length": len(c["antecedent"]), "features": ";".join(f for f, _ in c["antecedent"]), "consequent": c["consequent"],
               "antecedent_n_patients": c["antecedent_n"], "joint_support_patients": c["support"], "confidence": c["confidence"],
               "class_counts": ";".join(f"{k}:{v}" for k, v in c["class_counts"].items()), "min_literal_cut_recurrence": min(recs) if recs else None,
               "concept_eligible": c["concept_eligible"], "class_eligible": c["class_eligible"], "exclusion_group_violation": c["exclusion_group_violation"],
               "discovery_mapping": c["mapping"], "gate_rule_id": c["gate_rule"].rule_id if c["gate_rule"] is not None else None,
               "gate_rule": (" & ".join(f"{f}={s}" for f, s in c["gate_rule"].antecedent) + f" -> {c['gate_rule'].consequent}") if c["gate_rule"] is not None else None,
               "agreement_reference": agreement_reference}
        passes = dict(c["passes"])
        if c["gate_rule"] is not None and c["gate_rule"].rule_id in gates.index:
            gate = gates.loc[c["gate_rule"].rule_id]
            row.update({"folds_reproduced": int(gate["folds_reproduced"]), "reproduced_in_folds": str(gate["reproduced_in_folds"]),
                        "oof_matched": int(gate["oof_matched"]), "oof_agree": int(gate["oof_agree"]),
                        "agreement": None if pd.isna(gate["agreement"]) else float(gate["agreement"]),
                        "wilson_lb": None if pd.isna(gate["wilson_lb"]) else float(gate["wilson_lb"]),
                        "wilson_ub": None if pd.isna(gate["wilson_ub"]) else float(gate["wilson_ub"])})
            for s in sensitivities:
                col = f"eligible_at_{s:.2f}"
                row[col] = bool(gate[col]) if col in gate.index else None
            passes["crossfit_reproduction"] = int(gate["folds_reproduced"]) >= 1
            passes["pooled_matches"] = int(gate["oof_matched"]) >= limits["pooled_matches"]
            passes["wilson_lower"] = bool(not pd.isna(gate["wilson_lb"]) and float(gate["wilson_lb"]) >= limits["wilson_lower"])
        first = next((cond for cond in CONDITIONS if not passes.get(cond, False)), PASSED)
        row.update({f"passes_{k}": passes.get(k) for k in CONDITIONS})
        row["first_failed_condition"] = first if first == PASSED else order_names[CONDITIONS.index(first)]
        rows.append(row)
    frame = pd.DataFrame(rows)
    counts = (frame.groupby(["rulebook", "first_failed_condition"]).size().rename("n_candidates").reset_index() if len(frame)
              else pd.DataFrame(columns=["rulebook", "first_failed_condition", "n_candidates"]))
    derived = {rule_identity(c["reduced_antecedent"], c["consequent"]) for c in cands if c.get("mapping") in ("IRREDUCIBLE", "REDUCED")}
    summary = {"pipeline_order": order_names, "limits": limits, "attributes": list(attributes), "usable_attributes": usable,
               "n_candidates": int(len(frame)), "n_induced_rules": len(book.rules), "n_rules_gated_outside_discovery": len(extra),
               "discovery_reproduced": derived == set(by_identity), "passed_all": int((frame["first_failed_condition"] == PASSED).sum()) if len(frame) else 0,
               "rejection_counts": counts.to_dict("records")}
    return frame, summary

