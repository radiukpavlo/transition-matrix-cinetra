"""Inference routes, soft matching and reason codes (SPECIFICATION.md 6.0.0, Sections 11.7-11.8; Decision D14;
configs/rule_policy.yaml; registry/reason_codes.yaml; registry/pattern_templates.yaml).

Routes partition every attempted input: QC_ABSTAIN, EMPTY_RULEBOOK, EXACT_ACCEPT, CONFLICT_ABSTAIN,
SOFT_CANDIDATES_ONLY, UNAVAILABLE_FEATURE_ABSTAIN, NO_MATCH_ABSTAIN, in the configured precedence. Only EXACT_ACCEPT
emits a label. There is no conflict resolution and the base decision is never a tie-breaker; references (measured B, Y)
are never read here. Pattern templates are descriptive only.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .rough_sets import ELIGIBLE, Rule, Rulebook
from .wedd import UNAVAILABLE_PREFIX, is_available_state, parse_state_token

LABEL_ROUTE = "EXACT_ACCEPT"


def _codes_by_name(reason_codes: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(c["code"]): c for c in reason_codes.get("codes", [])}


def _validate_policy(cfg: dict[str, Any], reason_codes: dict[str, Any]) -> None:
    if str(cfg.get("conflict_resolution", "none")) != "none":
        raise ValueError("configs/rule_policy.yaml conflict_resolution must be none")
    if str(cfg.get("base_decision_as_tiebreaker", "forbidden")) != "forbidden":
        raise ValueError("configs/rule_policy.yaml base_decision_as_tiebreaker must be forbidden")
    routes = list(cfg.get("routes") or reason_codes.get("routes") or [])
    if set(routes) != set(reason_codes.get("routes", routes)):
        raise ValueError("Routes in configs/rule_policy.yaml and registry/reason_codes.yaml disagree")
    if list(reason_codes.get("label_emitting_routes", [LABEL_ROUTE])) != [LABEL_ROUTE]:
        raise ValueError("Only EXACT_ACCEPT may emit a label")
    if soft_emits_label(cfg):
        raise ValueError("configs/rule_policy.yaml soft_match.emits_label must be false")


def soft_emits_label(cfg: dict[str, Any]) -> bool:
    return bool((cfg.get("soft_match") or {}).get("emits_label", False))


def _emit(code: str, route: str, codes: dict[str, dict[str, Any]]) -> str:
    entry = codes.get(code)
    if entry is None:
        raise KeyError(f"Reason code {code} is not registered in registry/reason_codes.yaml")
    if entry.get("route") != route:
        raise ValueError(f"Reason code {code} belongs to route {entry.get('route')}, not {route}")
    return code


def _qc_decisions(qc: pd.DataFrame | None, sample_ids: list[str], qc_policy: dict[str, Any] | None, reason_codes: dict[str, Any]) -> dict[str, list[str]]:
    """Map sample_id -> QC_ABSTAIN reason codes (empty when the case may be routed)."""
    out = {sid: [] for sid in sample_ids}
    if qc is None or len(qc) == 0:
        return out
    frame = qc.copy()
    if "sample_id" not in frame.columns:
        frame = frame.reset_index().rename(columns={frame.index.name or "index": "sample_id"})
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame = frame.drop_duplicates("sample_id", keep="last").set_index("sample_id")
    for sid in sample_ids:
        if sid not in frame.index:
            continue
        row = frame.loc[sid]
        if "qc_abstain" in frame.columns:
            if bool(row["qc_abstain"]):
                code = row.get("qc_reason_code") if "qc_reason_code" in frame.columns else None
                out[sid] = [str(c) for c in str(code).split(";") if c] if isinstance(code, str) and code else ["QC_UNREGISTERED_REASON"]
            continue
        if "qc_failure_reason" in frame.columns:
            if qc_policy is None:
                raise ValueError("qc_policy (configs/cohorts.yaml) is required to classify qc_failure_reason tokens")
            from .semantics.qc import assess_qc_v6, parse_reason_tokens

            assessment = assess_qc_v6(parse_reason_tokens(row["qc_failure_reason"]), qc_policy, reason_codes)
            if assessment.route == "QC_ABSTAIN":
                out[sid] = list(assessment.abstain_codes)
    return out


def _unavailable_code(value: Any) -> str | None:
    text = "" if value is None else str(value)
    return text[len(UNAVAILABLE_PREFIX):] if text.startswith(UNAVAILABLE_PREFIX) else None


def _row_dict(states: pd.DataFrame, i: int) -> dict[str, Any]:
    return {c: states.iat[i, j] for j, c in enumerate(states.columns)}


def route(
    rulebook: Rulebook,
    states: pd.DataFrame,
    qc: pd.DataFrame | None,
    cfg: dict[str, Any],
    reason_codes: dict[str, Any],
    *,
    qc_policy: dict[str, Any] | None = None,
    eligible_only: bool = True,
) -> pd.DataFrame:
    """Route every input. ``states`` holds one row per input with a ``sample_id`` column (else the index is used) and
    one state-token column per antecedent feature (``missing`` or ``UNAVAILABLE:<code>`` when unavailable).

    ``qc`` optionally holds ``sample_id`` with either ``qc_abstain``/``qc_reason_code`` or ``qc_failure_reason`` tokens
    (classified with the v6 severities of ``qc_policy``). Columns of the result: sample_id, route, reason_code,
    reason_codes, label (EXACT_ACCEPT only), fired_rule_ids, candidate_labels, candidate_rule_ids, n_fired,
    conflict_encountered.
    """
    _validate_policy(cfg, reason_codes)
    codes = _codes_by_name(reason_codes)
    soft = cfg.get("soft_match") or {}
    min_ant = int(soft["min_antecedents"])
    min_sim = float(soft["min_similarity"])
    max_bad = int(soft["max_mismatch_or_unavailable"])
    max_cand = int(soft["max_candidates"])
    frame = states.copy()
    if "sample_id" in frame.columns:
        sample_ids = frame["sample_id"].astype(str).tolist()
        frame = frame.drop(columns=["sample_id"])
    else:
        sample_ids = [str(i) for i in frame.index]
    rules: list[Rule] = [r for r in rulebook.rules if (r.eligibility == ELIGIBLE or not eligible_only)]
    qc_codes = _qc_decisions(qc, sample_ids, qc_policy, reason_codes)
    rows = []
    for i, sid in enumerate(sample_ids):
        base = {"sample_id": sid, "label": None, "fired_rule_ids": "", "candidate_labels": "", "candidate_rule_ids": "", "n_fired": 0, "conflict_encountered": False}
        if qc_codes.get(sid):
            primary = qc_codes[sid][0]
            rows.append({**base, "route": "QC_ABSTAIN", "reason_code": _emit(primary, "QC_ABSTAIN", codes), "reason_codes": ";".join(qc_codes[sid])})
            continue
        if not rules:
            rows.append({**base, "route": "EMPTY_RULEBOOK", "reason_code": _emit("RULEBOOK_EMPTY", "EMPTY_RULEBOOK", codes), "reason_codes": "RULEBOOK_EMPTY"})
            continue
        row = _row_dict(frame, i)
        fired: list[Rule] = []
        undetermined: list[Rule] = []
        soft_cands: list[tuple[float, int, str, Rule]] = []
        for rule in rules:
            fires, matched, unavailable = rule.evaluate(row)
            if fires is True:
                fired.append(rule)
                continue
            if fires is None:
                undetermined.append(rule)
            length = rule.length
            bad = length - matched
            sim = matched / length if length else 0.0
            if length >= min_ant and bad <= max_bad and sim >= min_sim - 1e-12:
                soft_cands.append((sim, rule.support, rule.rule_id, rule))
        consequents = sorted({r.consequent for r in fired})
        base["fired_rule_ids"] = ";".join(sorted(r.rule_id for r in fired))
        base["n_fired"] = len(fired)
        base["conflict_encountered"] = len(consequents) > 1
        if len(consequents) == 1:
            rows.append({**base, "route": LABEL_ROUTE, "reason_code": None, "reason_codes": "", "label": consequents[0]})
            continue
        if len(consequents) > 1:
            code = _emit("MATCH_CONFLICTING_CONSEQUENTS", "CONFLICT_ABSTAIN", codes)
            rows.append({**base, "route": "CONFLICT_ABSTAIN", "reason_code": code, "reason_codes": code, "candidate_labels": ";".join(consequents)})
            continue
        if soft_cands:
            soft_cands.sort(key=lambda t: (-t[0], -t[1], t[2]))
            top = [t[3] for t in soft_cands[:max_cand]]
            labels = []
            for r in top:
                if r.consequent not in labels:
                    labels.append(r.consequent)
            code = _emit("MATCH_SOFT_ONLY", "SOFT_CANDIDATES_ONLY", codes)
            rows.append({**base, "route": "SOFT_CANDIDATES_ONLY", "reason_code": code, "reason_codes": code,
                         "candidate_labels": ";".join(labels), "candidate_rule_ids": ";".join(r.rule_id for r in top)})
            continue
        if undetermined:
            feature_codes = sorted({c for r in undetermined for f, _ in r.antecedent if (c := _unavailable_code(row.get(f))) is not None})
            primary = feature_codes[0] if len(feature_codes) == 1 else "MATCH_MISSING_REQUIRED_SYMBOL"
            all_codes = ["MATCH_MISSING_REQUIRED_SYMBOL", *[c for c in feature_codes if c != "MATCH_MISSING_REQUIRED_SYMBOL"]]
            for c in all_codes:
                _emit(c, "UNAVAILABLE_FEATURE_ABSTAIN", codes)
            rows.append({**base, "route": "UNAVAILABLE_FEATURE_ABSTAIN", "reason_code": primary, "reason_codes": ";".join(all_codes)})
            continue
        code = _emit("MATCH_NONE", "NO_MATCH_ABSTAIN", codes)
        rows.append({**base, "route": "NO_MATCH_ABSTAIN", "reason_code": code, "reason_codes": code})
    cols = ["sample_id", "route", "reason_code", "reason_codes", "label", "fired_rule_ids", "candidate_labels", "candidate_rule_ids", "n_fired", "conflict_encountered"]
    out = pd.DataFrame(rows, columns=cols)
    out["label"] = out["label"].astype(object)
    return out


def route_counts(routes: pd.DataFrame, reason_codes: dict[str, Any] | None = None) -> dict[str, int]:
    """Count of each route (all seven listed); the counts sum to the number of attempted inputs."""
    names = list((reason_codes or {}).get("routes") or ["QC_ABSTAIN", "EMPTY_RULEBOOK", "EXACT_ACCEPT", "CONFLICT_ABSTAIN", "SOFT_CANDIDATES_ONLY", "UNAVAILABLE_FEATURE_ABSTAIN", "NO_MATCH_ABSTAIN"])
    counts = routes["route"].value_counts().to_dict()
    unknown = set(counts) - set(names)
    if unknown:
        raise ValueError(f"Unregistered routes: {sorted(unknown)}")
    return {name: int(counts.get(name, 0)) for name in names}


# ----------------------------------------------------------------------------------------------------------------
# Pattern templates (descriptive only)
# ----------------------------------------------------------------------------------------------------------------
def _token_satisfies(state: Any, token: str) -> bool:
    parsed = parse_state_token(state)
    if parsed is None:
        return False
    index, n_bins = parsed
    if token == "lowest_bin":
        return index == 0
    if token == "highest_bin":
        return index == n_bins - 1
    if token == "not_lowest_bin":
        return index != 0
    if token == "not_highest_bin":
        return index != n_bins - 1
    raise ValueError(f"Unknown template state token {token!r}")


def match_templates(states: pd.DataFrame, templates: dict[str, Any]) -> pd.DataFrame:
    """Per input and supported template: whether every template condition holds on the input's symbolic states.
    Descriptive only (registry/pattern_templates.yaml); unsupported templates are listed and never matched."""
    frame = states.copy()
    sample_ids = frame.pop("sample_id").astype(str).tolist() if "sample_id" in frame.columns else [str(i) for i in frame.index]
    rows = []
    for tpl in templates.get("templates", []):
        supported = str(tpl.get("support_status", "")).startswith("supported") and bool(tpl.get("conditions"))
        conds = dict(tpl.get("conditions") or {})
        for i, sid in enumerate(sample_ids):
            if supported and all(f in frame.columns for f in conds):
                matched = all(_token_satisfies(frame.iloc[i][f], tok) for f, tok in conds.items())
                evaluable = all(is_available_state(frame.iloc[i][f]) for f in conds)
            else:
                matched, evaluable = False, False
            rows.append({"sample_id": sid, "template_id": tpl["id"], "support_status": tpl.get("support_status"), "evaluable": evaluable, "matched": bool(matched), "descriptive_only": True})
    return pd.DataFrame(rows, columns=["sample_id", "template_id", "support_status", "evaluable", "matched", "descriptive_only"])


def template_rule_matches(rulebook: Rulebook, templates: dict[str, Any], *, eligible_only: bool = True) -> pd.DataFrame:
    """Per template: the eligible rules whose antecedent set structurally implies the template (exact inclusion of
    every listed condition), their count and consequents. Descriptive only; a template is never a diagnosis."""
    rules = [r for r in rulebook.rules if r.eligibility == ELIGIBLE or not eligible_only]
    rows = []
    for tpl in templates.get("templates", []):
        applies = rulebook.consequent_kind in (tpl.get("rulebooks") or []) or not tpl.get("rulebooks")
        supported = str(tpl.get("support_status", "")).startswith("supported") and bool(tpl.get("conditions")) and applies
        implying = []
        if supported:
            conds = dict(tpl["conditions"])
            for r in rules:
                ant = dict(r.antecedent)
                if all(f in ant and _token_satisfies(ant[f], tok) for f, tok in conds.items()):
                    implying.append(r)
        rows.append({"template_id": tpl["id"], "rulebook": rulebook.rulebook_id, "support_status": tpl.get("support_status"), "applies_to_rulebook": applies,
                     "n_implying_rules": len(implying), "implying_rule_ids": ";".join(r.rule_id for r in implying),
                     "consequents": ";".join(sorted({r.consequent for r in implying})), "descriptive_only": True})
    return pd.DataFrame(rows)
