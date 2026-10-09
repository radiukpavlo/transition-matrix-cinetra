"""Rough-set decision tables, variable-precision rule discovery and the cross-fitted validation gate
(SPECIFICATION.md 6.0.0, Sections 11.5-11.6; Decisions D12, D13; configs/rule_policy.yaml).

The decision system is ``(U, C, d)`` with ``U`` the development patients, ``C`` the symbolic antecedent columns (state
tokens ``b<i>of<n>`` from :mod:`tmmri.wedd`) and ``d`` the rulebook consequent. Missing or unavailable states never act
as literals: for an attribute subset ``P`` the indiscernibility relation is computed on the complete-symbol
sub-universe of ``P`` and the excluded objects are counted separately. A granule with majority class ``y`` is admitted
as VARIABLE_PRECISION when at most ``beta`` of its members disagree; it is STRICT when none disagree. Discovery is
exhaustive over attribute subsets of size 1..max_length (independent of column order), followed by greedy backward
deletion, de-duplication and removal of redundant supersets; there is no rule cap and the ranking is deterministic.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from .wedd import MISSING_STATE, is_available_state

VARIABLE_PRECISION = "VARIABLE_PRECISION"
STRICT = "STRICT"
ELIGIBLE = "ELIGIBLE"
CANDIDATE = "CANDIDATE"
CONSEQUENT_KINDS = ("d_model", "dataset_label", "measured_oracle")
CAVEATS = {
    "d_model": "model decision, not diagnosis",
    "dataset_label": "dataset label, not diagnosis",
    "measured_oracle": "model decision, not diagnosis (measured-B oracle)",
}
_EPS = 1e-12


class TautologyError(ValueError):
    """An antecedent set encodes the consequent or violates the registry antecedent exclusion rule."""


# ----------------------------------------------------------------------------------------------------------------
# Antecedent policy from the registry
# ----------------------------------------------------------------------------------------------------------------
def antecedent_policy(registry: dict[str, Any]) -> tuple[list[str], list[frozenset[str]]]:
    """Allowed antecedents (``primary_rulebook_antecedent_candidates``, registry order) and exclusion groups.

    An exclusion group is a candidate feature together with all of its registry ``dependencies`` when every dependency
    is itself a candidate (for the v6 registry: each EF with its two volumes). A rule may never contain a whole group.
    """
    allowed = [str(c) for c in registry.get("primary_rulebook_antecedent_candidates", [])]
    features = {str(e["name"]): e for e in registry.get("features", [])}
    groups = []
    for name in allowed:
        deps = [str(d) for d in (features.get(name, {}).get("dependencies") or [])]
        if len(deps) >= 2 and all(d in allowed for d in deps):
            groups.append(frozenset([name, *deps]))
    return allowed, groups


def violates_exclusion(attributes: Iterable[str], exclusion_groups: Iterable[frozenset[str]]) -> bool:
    attrs = set(attributes)
    return any(group <= attrs for group in exclusion_groups)


def check_no_tautology(
    attributes: Iterable[str],
    registry: dict[str, Any] | None = None,
    *,
    allowed: Iterable[str] | None = None,
    exclusion_groups: Iterable[frozenset[str]] | None = None,
) -> None:
    """Raise :class:`TautologyError` when an attribute is not an allowed antecedent (metadata, labels, QC columns and
    reconstruction-only features never are) or when the set contains a whole exclusion group."""
    attrs = list(attributes)
    if registry is not None:
        reg_allowed, reg_groups = antecedent_policy(registry)
        allowed = reg_allowed if allowed is None else allowed
        exclusion_groups = reg_groups if exclusion_groups is None else exclusion_groups
    if allowed is not None:
        bad = [a for a in attrs if a not in set(allowed)]
        if bad:
            raise TautologyError(f"Not permitted as antecedents (tautological or not a rule candidate): {bad}")
    if exclusion_groups is not None and violates_exclusion(attrs, exclusion_groups):
        raise TautologyError(f"Antecedent set {sorted(attrs)} contains a deterministic function together with all of its inputs")


# ----------------------------------------------------------------------------------------------------------------
# Decision-table primitives
# ----------------------------------------------------------------------------------------------------------------
def _available(column: pd.Series) -> np.ndarray:
    return np.array([is_available_state(v) for v in column], dtype=bool)


def _universe(states: pd.DataFrame, decision: pd.Series | np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    d = pd.Series(np.asarray(decision, dtype=object), index=states.index)
    known = d.notna().to_numpy()
    return states.loc[known], d[known].astype(str).to_numpy()


def indiscernibility_classes(states: pd.DataFrame, attributes: list[str]) -> tuple[dict[tuple[str, ...], list[Any]], list[Any]]:
    """Equivalence classes of IND(P) on the complete-symbol sub-universe of P; returns (classes, excluded labels)."""
    if not attributes:
        return {(): list(states.index)}, []
    avail = np.ones(len(states), dtype=bool)
    for a in attributes:
        avail &= _available(states[a])
    classes: dict[tuple[str, ...], list[Any]] = {}
    sub = states.loc[avail, attributes]
    for label, row in zip(sub.index, sub.itertuples(index=False, name=None)):
        classes.setdefault(tuple(str(v) for v in row), []).append(label)
    return classes, list(states.index[~avail])


def lower_upper_boundary(states: pd.DataFrame, decision: pd.Series | np.ndarray, attributes: list[str]) -> dict[str, Any]:
    """Lower and upper approximations and boundary region of every decision class (E15), by object label."""
    st, d = _universe(states, decision)
    lookup = dict(zip(st.index, d))
    classes, excluded = indiscernibility_classes(st, list(attributes))
    out: dict[str, Any] = {"attributes": list(attributes), "universe": [i for g in classes.values() for i in g], "excluded": excluded, "classes": {}}
    for y in sorted(set(d)):
        lower: set[Any] = set()
        upper: set[Any] = set()
        for members in classes.values():
            labels = {lookup[m] for m in members}
            if labels == {y}:
                lower.update(members)
            if y in labels:
                upper.update(members)
        out["classes"][y] = {"lower": lower, "upper": upper, "boundary": upper - lower}
    return out


@dataclass(frozen=True)
class Granule:
    signature: tuple[tuple[str, str], ...]
    members: tuple[Any, ...]
    majority: str
    antecedent_n: int
    joint_n: int
    confidence: float
    tier: str


def granules(states: pd.DataFrame, decision: pd.Series | np.ndarray, attributes: list[str], beta: float, *, min_size: int = 1) -> list[Granule]:
    """Granules of IND(attributes) admitted at precision ``beta`` (at most ``beta`` of the members disagree with the
    majority). ``beta = 0`` admits only pure (STRICT) granules."""
    st, d = _universe(states, decision)
    lookup = dict(zip(st.index, d))
    classes, _ = indiscernibility_classes(st, list(attributes))
    out = []
    for sig, members in sorted(classes.items()):
        counts: dict[str, int] = {}
        for m in members:
            counts[lookup[m]] = counts.get(lookup[m], 0) + 1
        majority = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        n = len(members)
        conf = majority[1] / n
        if n < min_size or (1.0 - conf) > beta + _EPS:
            continue
        out.append(Granule(tuple(zip(attributes, sig)), tuple(members), majority[0], n, majority[1], conf, STRICT if majority[1] == n else VARIABLE_PRECISION))
    return out


@dataclass(frozen=True)
class Approximation:
    lower: set
    upper: set
    boundary: set


def approximations(signatures: dict[Any, str], decisions: dict[Any, Any], decision_value: Any) -> Approximation:
    """Convenience form of :func:`lower_upper_boundary` for one attribute given as ``{object: signature}``."""
    frame = pd.DataFrame({"sig": pd.Series(signatures, dtype=object)})
    res = lower_upper_boundary(frame, pd.Series(decisions, dtype=object).reindex(frame.index), ["sig"])
    cls = res["classes"].get(str(decision_value), {"lower": set(), "upper": set(), "boundary": set()})
    return Approximation(cls["lower"], cls["upper"], cls["boundary"])


# ----------------------------------------------------------------------------------------------------------------
# Rules and rulebooks
# ----------------------------------------------------------------------------------------------------------------
def rule_identity(antecedent: Iterable[tuple[str, str]], consequent: str) -> str:
    """Rule identity of configs/rule_policy.yaml: same feature set, same bin index and bin count, same consequent."""
    return json.dumps({"antecedent": sorted([list(a) for a in antecedent]), "consequent": str(consequent)}, sort_keys=True)


def make_rule_id(consequent_kind: str, antecedent: Iterable[tuple[str, str]], consequent: str) -> str:
    digest = hashlib.sha256(f"{consequent_kind}|{rule_identity(antecedent, consequent)}".encode()).hexdigest()
    return f"R_{digest[:16]}"


@dataclass(frozen=True)
class Rule:
    rule_id: str
    antecedent: tuple[tuple[str, str], ...]
    consequent: str
    support: int                      # fit joint count (anchors matching antecedent and consequent)
    confidence: float
    strict: bool
    anchors: tuple[str, ...]
    antecedent_n: int = 0
    tier: str = VARIABLE_PRECISION
    eligibility: str = CANDIDATE
    consequent_kind: str = "d_model"
    crossfit: dict[str, Any] = field(default_factory=dict)
    bootstrap_recurrence: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.antecedent)

    @property
    def identity(self) -> str:
        return rule_identity(self.antecedent, self.consequent)

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(a for a, _ in self.antecedent)

    def evaluate(self, row: dict[str, Any] | pd.Series) -> tuple[bool | None, int, int]:
        """(fires, matched_available, unavailable): fires is True (all equal), False (an available antecedent
        mismatches) or None (undetermined: no mismatch but at least one antecedent unavailable)."""
        matched = unavailable = 0
        mismatch = False
        for feat, state in self.antecedent:
            value = row.get(feat, MISSING_STATE) if hasattr(row, "get") else MISSING_STATE
            if not is_available_state(value):
                unavailable += 1
            elif str(value) == str(state):
                matched += 1
            else:
                mismatch = True
        if mismatch:
            return False, matched, unavailable
        return (True if unavailable == 0 else None), matched, unavailable

    def to_record(self, rulebook_id: str) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rulebook": rulebook_id,
            "tier": self.tier,
            "strict": bool(self.strict),
            "consequent_endpoint": self.consequent_kind,
            "antecedents": [{"feature": f, "state": s} for f, s in self.antecedent],
            "consequent": self.consequent,
            "fit_antecedent_n_patients": int(self.antecedent_n),
            "fit_joint_n_patients": int(self.support),
            "fit_confidence": float(self.confidence),
            "crossfit_matched_n_patients": self.crossfit.get("oof_matched"),
            "crossfit_agreement": self.crossfit.get("agreement"),
            "crossfit_wilson_lower": self.crossfit.get("wilson_lb"),
            "crossfit_folds_reproduced": self.crossfit.get("folds_reproduced"),
            "length": self.length,
            "bootstrap_recurrence": dict(self.bootstrap_recurrence),
            "eligibility": self.eligibility,
            "caveat": CAVEATS.get(self.consequent_kind, "not diagnosis"),
            "anchors": list(self.anchors),
        }

    @classmethod
    def from_record(cls, rec: dict[str, Any]) -> "Rule":
        return cls(
            rule_id=str(rec["rule_id"]),
            antecedent=tuple((str(a["feature"]), str(a["state"])) for a in rec["antecedents"]),
            consequent=str(rec["consequent"]),
            support=int(rec.get("fit_joint_n_patients", 0)),
            confidence=float(rec.get("fit_confidence", 0.0)),
            strict=bool(rec.get("strict", rec.get("tier") == STRICT)),
            anchors=tuple(str(a) for a in rec.get("anchors", [])),
            antecedent_n=int(rec.get("fit_antecedent_n_patients", 0)),
            tier=str(rec.get("tier", VARIABLE_PRECISION)),
            eligibility=str(rec.get("eligibility", CANDIDATE)),
            consequent_kind=str(rec.get("consequent_endpoint", "d_model")),
            crossfit={k: rec.get(f"crossfit_{v}") for k, v in (("oof_matched", "matched_n_patients"), ("agreement", "agreement"), ("wilson_lb", "wilson_lower"), ("folds_reproduced", "folds_reproduced")) if rec.get(f"crossfit_{v}") is not None},
            bootstrap_recurrence=dict(rec.get("bootstrap_recurrence") or {}),
        )


@dataclass
class Rulebook:
    rules: list[Rule]
    rulebook_id: str
    consequent_kind: str
    class_records: dict[str, dict[str, Any]] = field(default_factory=dict)
    failing_gates: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return len(self.rules) == 0

    def eligible(self) -> "Rulebook":
        return Rulebook([r for r in self.rules if r.eligibility == ELIGIBLE], self.rulebook_id, self.consequent_kind, dict(self.class_records), list(self.failing_gates), dict(self.provenance))

    def to_records(self) -> list[dict[str, Any]]:
        return [r.to_record(self.rulebook_id) for r in self.rules]

    @classmethod
    def from_records(cls, records: list[dict[str, Any]], *, rulebook_id: str | None = None, consequent_kind: str | None = None) -> "Rulebook":
        rules = [Rule.from_record(r) for r in records]
        rb_id = rulebook_id or (str(records[0]["rulebook"]) if records else "rulebook")
        kind = consequent_kind or (rules[0].consequent_kind if rules else "d_model")
        return cls(rules, rb_id, kind)

    def empty_record(self) -> dict[str, Any]:
        """Content of rules/<rulebook>/EMPTY.json (configs/rule_policy.yaml empty_rulebook_record)."""
        return {"rulebook": self.rulebook_id, "consequent_endpoint": self.consequent_kind, "reason_code": "RULEBOOK_EMPTY",
                "failing_gates": list(self.failing_gates), "class_records": dict(self.class_records), "provenance": dict(self.provenance)}


def _match_mask(states: pd.DataFrame, antecedent: Iterable[tuple[str, str]]) -> np.ndarray:
    mask = np.ones(len(states), dtype=bool)
    for feat, state in antecedent:
        mask &= states[feat].astype(str).to_numpy() == str(state)
    return mask


def _stats(states: pd.DataFrame, d: np.ndarray, antecedent: tuple[tuple[str, str], ...], consequent: str) -> tuple[int, int, np.ndarray]:
    m = _match_mask(states, antecedent)
    joint = m & (d == consequent)
    return int(m.sum()), int(joint.sum()), joint


def _discovery_params(cfg: dict[str, Any]) -> dict[str, Any]:
    disc = cfg["discovery"]
    primary = disc["tiers"]["primary"]
    beta = float(primary["beta"])
    min_conf = float(primary["min_confidence"])
    if not math.isclose(1.0 - beta, min_conf, rel_tol=0, abs_tol=1e-12):
        raise ValueError("configs/rule_policy.yaml: min_confidence must equal 1 - beta")
    if disc.get("missing_states", "never_literals") != "never_literals":
        raise ValueError("Only missing_states: never_literals is implemented")
    if str(disc.get("rule_cap", "none")) != "none":
        raise ValueError("A rule cap is not part of protocol v6 (configs/rule_policy.yaml rule_cap: none)")
    min_class = int(((cfg.get("eligibility_gates") or {}).get("consequent_support") or {}).get("min_fit_patients_per_class", 0))
    return {"beta": beta, "min_confidence": min_conf, "min_n": int(disc["min_antecedent_n_patients"]), "max_length": int(disc["max_length"]), "min_class": min_class,
            "remove_supersets": bool(disc.get("remove_redundant_supersets_same_consequent", True))}


def induce_vprs_rules(
    states: pd.DataFrame,
    decision: pd.Series | np.ndarray,
    anchors: pd.Series | np.ndarray,
    cfg: dict[str, Any],
    *,
    registry: dict[str, Any] | None = None,
    attributes: list[str] | None = None,
    allowed: list[str] | None = None,
    exclusion_groups: list[frozenset[str]] | None = None,
    consequent_kind: str = "d_model",
    rulebook_id: str | None = None,
) -> Rulebook:
    """Variable-precision rough-set discovery per configs/rule_policy.yaml ``discovery`` (beta, minimum anchors,
    maximum length, exhaustive enumeration, greedy backward deletion, de-duplication, no cap, deterministic ranking).

    ``attributes`` default to the registry antecedent candidates present in ``states`` (registry order). The tautology
    guard rejects any attribute outside the allowed antecedents and any subset containing a whole exclusion group.
    """
    if consequent_kind not in CONSEQUENT_KINDS:
        raise ValueError(f"consequent_kind must be one of {CONSEQUENT_KINDS}")
    params = _discovery_params(cfg)
    if registry is not None:
        reg_allowed, reg_groups = antecedent_policy(registry)
        allowed = reg_allowed if allowed is None else allowed
        exclusion_groups = reg_groups if exclusion_groups is None else exclusion_groups
    groups = list(exclusion_groups or [])
    if attributes is None:
        attributes = [a for a in (allowed or list(states.columns)) if a in states.columns]
    check_no_tautology(attributes, allowed=allowed, exclusion_groups=None)
    order = {a: i for i, a in enumerate(attributes)}
    anchor_ids = pd.Series(np.asarray(anchors, dtype=object), index=states.index).astype(str)
    st, d = _universe(states, decision)
    anchor_ids = anchor_ids.loc[st.index].to_numpy()
    rb_id = rulebook_id or consequent_kind
    class_counts = pd.Series(d).value_counts().to_dict() if len(d) else {}
    eligible_classes = {c for c, n in class_counts.items() if n >= params["min_class"]}
    class_records = {str(c): {"fit_patients": int(n), "eligible": c in eligible_classes, "reason_code": None if c in eligible_classes else "CLASS_INELIGIBLE"} for c, n in sorted(class_counts.items())}
    usable = [a for a in attributes if len({str(v) for v in st[a] if is_available_state(v)}) >= 2]
    candidates: dict[str, tuple[tuple[tuple[str, str], ...], str]] = {}
    for size in range(1, params["max_length"] + 1):
        for subset in itertools.combinations(usable, size):
            if violates_exclusion(subset, groups):
                continue
            for g in granules(st, d, list(subset), params["beta"], min_size=params["min_n"]):
                if g.majority not in eligible_classes:
                    continue
                reduced = _reduce(st, d, g.signature, g.majority, params, order)
                candidates.setdefault(rule_identity(reduced, g.majority), (reduced, g.majority))
    items = list(candidates.values())
    if params["remove_supersets"]:
        items = [(ant, y) for ant, y in items if not any(y2 == y and set(a2) < set(ant) for a2, y2 in items)]
    rules = []
    for ant, y in items:
        ant = tuple(sorted(ant, key=lambda lit: order[lit[0]]))
        an_n, joint_n, joint = _stats(st, d, ant, y)
        conf = joint_n / an_n if an_n else 0.0
        rules.append(
            Rule(
                rule_id=make_rule_id(consequent_kind, ant, y),
                antecedent=ant,
                consequent=y,
                support=joint_n,
                confidence=conf,
                strict=joint_n == an_n,
                anchors=tuple(sorted(anchor_ids[joint].tolist())),
                antecedent_n=an_n,
                tier=STRICT if joint_n == an_n else VARIABLE_PRECISION,
                consequent_kind=consequent_kind,
            )
        )
    rules.sort(key=lambda r: (-r.confidence, -r.antecedent_n, r.length, tuple(order[f] for f in r.features), tuple(s for _, s in r.antecedent), r.rule_id))
    failing = [] if rules else [f"discovery: no granule with >= {params['min_n']} anchors and confidence >= {params['min_confidence']} over attribute subsets of size <= {params['max_length']}"]
    return Rulebook(rules, rb_id, consequent_kind, class_records, failing, {"attributes": list(attributes), "n_universe": int(len(d)), "params": params})


def _reduce(states: pd.DataFrame, d: np.ndarray, antecedent: tuple[tuple[str, str], ...], consequent: str, params: dict[str, Any], order: dict[str, int]) -> tuple[tuple[str, str], ...]:
    """Greedy backward deletion: repeatedly delete the literal whose removal keeps the rule valid (anchors >= min,
    confidence >= min_confidence, same consequent) with the largest resulting joint support; ties delete the literal
    latest in registry order (so earlier registry features are retained)."""
    current = tuple(sorted(antecedent, key=lambda lit: order[lit[0]]))
    while len(current) > 1:
        best = None
        for lit in current:
            reduced = tuple(x for x in current if x != lit)
            an_n, joint_n, _ = _stats(states, d, reduced, consequent)
            if an_n < params["min_n"] or joint_n / an_n < params["min_confidence"] - _EPS:
                continue
            key = (joint_n, order[lit[0]])
            if best is None or key > best[0]:
                best = (key, reduced)
        if best is None:
            break
        current = best[1]
    return current


# ----------------------------------------------------------------------------------------------------------------
# Cross-fitted validation gate
# ----------------------------------------------------------------------------------------------------------------
def cross_fitted_gate(
    values: pd.DataFrame,
    decision_fn: Callable[[np.ndarray, np.ndarray], Any],
    outer_folds: np.ndarray,
    cfg: dict[str, Any],
    *,
    final_rulebook: Rulebook,
    cutset_fn: Callable[[pd.DataFrame, np.ndarray, int], Any],
    anchors: pd.Series | np.ndarray,
    substrate_fn: Callable[[int, np.ndarray, np.ndarray], tuple[pd.DataFrame, pd.DataFrame]] | None = None,
    states_fn: Callable[[Any, pd.DataFrame], pd.DataFrame] | None = None,
    induce_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Cross-fitted rule validation over the outer folds (configs/rule_policy.yaml ``validation_gate``).

    For each outer fold f: the consequent of the training folds is ``decision_fn(train, train)`` and the held-out
    consequent ``decision_fn(train, heldout)`` (the predictor refitted without fold f; the dataset label for the
    companion rulebook); thresholds are refitted on the training folds (``cutset_fn(train_values, train_decision,
    n_train)``) and discovery is re-run there. A final rule is reproduced in fold f when a rule with the same identity
    (same feature set, bin index and bin count, consequent) is re-discovered; a bin-count mismatch therefore means not
    reproduced. Its held-out matches and agreements are counted only in folds where it was reproduced. Pooled over
    folds, a rule is eligible if matched >= ``min_pooled_oof_matched_patients`` and the Wilson lower bound of the
    agreement is >= ``agreement_wilson_lower_min``; the gate sensitivities are reported beside it.

    ``substrate_fn(f, train_positions, heldout_positions) -> (train_values, heldout_values)`` supplies fold-specific
    substrates (for example inner nested OOF reconstructions for the training folds and the outer OOF prediction for
    the held-out fold); by default the rows of ``values`` are used for both.
    """
    from .metrics import wilson_interval

    gate = cfg["validation_gate"]
    if gate.get("method", "cross_fitted_outer_folds") != "cross_fitted_outer_folds":
        raise ValueError("Only the cross_fitted_outer_folds validation gate is implemented")
    min_matched = int(gate["min_pooled_oof_matched_patients"])
    lb_min = float(gate["agreement_wilson_lower_min"])
    sensitivities = [float(s) for s in gate.get("gate_sensitivities", [])]
    outer = pd.to_numeric(pd.Series(np.asarray(outer_folds)), errors="coerce").to_numpy(dtype=float)
    dev = np.isfinite(outer)
    anchor_arr = np.asarray(anchors, dtype=object)
    fold_ids = sorted({int(v) for v in outer[dev]})
    if len(fold_ids) != int(gate["folds"]):
        raise ValueError(f"Expected {gate['folds']} outer folds, found {len(fold_ids)}")
    kwargs = dict(induce_kwargs or {})
    kwargs.setdefault("consequent_kind", final_rulebook.consequent_kind)
    tally = {r.rule_id: {"matched": 0, "agree": 0, "folds": []} for r in final_rulebook.rules}
    for f in fold_ids:
        train = np.flatnonzero(dev & (outer != f))
        held = np.flatnonzero(outer == f)
        if substrate_fn is None:
            train_vals, held_vals = values.iloc[train], values.iloc[held]
        else:
            train_vals, held_vals = substrate_fn(f, train, held)
        d_train = np.asarray(decision_fn(train, train), dtype=object)
        d_held = np.asarray(decision_fn(train, held), dtype=object)
        cutset = cutset_fn(train_vals, d_train, int(train.size))
        s_train = states_fn(cutset, train_vals) if states_fn else cutset.transform(train_vals)
        s_held = states_fn(cutset, held_vals) if states_fn else cutset.transform(held_vals)
        fold_rb = induce_vprs_rules(s_train, d_train, anchor_arr[train], cfg, **kwargs)
        fold_ids_set = {r.identity for r in fold_rb.rules}
        for rule in final_rulebook.rules:
            if rule.identity not in fold_ids_set:
                continue
            m = np.ones(len(s_held), dtype=bool)
            for feat, state in rule.antecedent:
                m &= s_held[feat].astype(str).to_numpy() == str(state) if feat in s_held.columns else False
            agree = m & (pd.Series(d_held).astype("string").fillna("").to_numpy() == str(rule.consequent))
            tally[rule.rule_id]["matched"] += int(m.sum())
            tally[rule.rule_id]["agree"] += int(agree.sum())
            tally[rule.rule_id]["folds"].append(f)
    rows = []
    for rule in final_rulebook.rules:
        t = tally[rule.rule_id]
        lb, ub = wilson_interval(t["agree"], t["matched"])
        agreement = t["agree"] / t["matched"] if t["matched"] else float("nan")
        support_ok = t["matched"] >= min_matched
        eligible = bool(support_ok and np.isfinite(lb) and lb >= lb_min)
        row = {
            "rule_id": rule.rule_id,
            "consequent": rule.consequent,
            "length": rule.length,
            "oof_matched": int(t["matched"]),
            "oof_agree": int(t["agree"]),
            "agreement": agreement,
            "wilson_lb": lb,
            "wilson_ub": ub,
            "folds_reproduced": int(len(t["folds"])),
            "reproduced_in_folds": ";".join(str(x) for x in t["folds"]),
            "eligible": eligible,
            "eligibility": ELIGIBLE if eligible else ("INELIGIBLE_CROSSFIT_SUPPORT" if not support_ok else "INELIGIBLE_CROSSFIT_AGREEMENT"),
        }
        for s in sensitivities:
            row[f"eligible_at_{s:.2f}"] = bool(support_ok and np.isfinite(lb) and lb >= s)
        rows.append(row)
    return pd.DataFrame(rows, columns=None if rows else ["rule_id", "consequent", "length", "oof_matched", "oof_agree", "agreement", "wilson_lb", "wilson_ub", "folds_reproduced", "reproduced_in_folds", "eligible", "eligibility"])


def apply_gate(rulebook: Rulebook, gate: pd.DataFrame) -> Rulebook:
    """Set each rule's eligibility and cross-fitting statistics; a rulebook left without eligible rules records the
    failing gate for EMPTY.json."""
    by_id = {str(r["rule_id"]): r for r in gate.to_dict("records")}
    rules = []
    for rule in rulebook.rules:
        g = by_id.get(rule.rule_id)
        if g is None:
            rules.append(replace(rule, eligibility="INELIGIBLE_NOT_GATED"))
            continue
        crossfit = {"oof_matched": int(g["oof_matched"]), "agreement": None if pd.isna(g["agreement"]) else float(g["agreement"]),
                    "wilson_lb": None if pd.isna(g["wilson_lb"]) else float(g["wilson_lb"]), "folds_reproduced": int(g["folds_reproduced"])}
        rules.append(replace(rule, eligibility=str(g["eligibility"]), crossfit=crossfit))
    out = Rulebook(rules, rulebook.rulebook_id, rulebook.consequent_kind, dict(rulebook.class_records), list(rulebook.failing_gates), dict(rulebook.provenance))
    if not any(r.eligibility == ELIGIBLE for r in rules):
        out.failing_gates.append("validation_gate: no rule met the pooled cross-fitted support and Wilson lower-bound conditions" if rules else "discovery: no candidate rule")
    return out
