"""Shared protocol-6.0.0 rule-pipeline steps (SPECIFICATION.md 6.0.0, Sections 11.2, 11.6, 11.7 and 13.3).

The CLI stages of :mod:`tmmri.stages` call these helpers; the refit arm (Decision D17), the ablation matrix and the
downstream refit bootstrap reuse them so that every arm runs the same code path.

* :func:`semantic_eligibility` -- the M7 gates of ``configs/rule_policy.yaml`` ``eligibility_gates`` per rulebook and
  candidate feature (reference availability, variability, reconstruction, post-discretization stability, no
  tautology); consequent support is enforced per class by :func:`tmmri.rough_sets.induce_vprs_rules`.
* :class:`NestedGateSubstrate` -- the ``substrate_fn`` of :func:`tmmri.rough_sets.cross_fitted_gate`: for outer fold
  ``f`` the training rows carry *inner* nested out-of-fold reconstructions from transitions refitted on the other
  folds only (per inner fold ``g`` the transition is fitted on the remaining inner folds, lambda by CV over them),
  and the held-out rows carry the outer out-of-fold reconstruction (the fold-``f`` transition never saw fold ``f``).
* :func:`fit_rulebook` -- cuts, stability gate, eligibility, discovery and the cross-fitted gate for one rulebook.
* :func:`qc_route_frame` and :func:`route_records` -- inference inputs: critical QC and non-finite formal A rows are
  ``QC_ABSTAIN`` (``INPUT_NONFINITE`` first) before routing.
* :func:`downstream_refit_bootstrap` -- rulebook stability under refitting of target transforms, transition,
  thresholds and discovery on cluster resamples (``configs/bootstrap.yaml`` ``downstream_refit_bootstrap``).
"""

from __future__ import annotations

import functools
import math
import re
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

_K_OF_N = re.compile(r"^\s*(\d+)_of_(\d+)\s*$")
REFIT_RULEBOOK_KINDS = ("d_model", "dataset_label", "measured_oracle")


def k_of_n(text: object) -> tuple[int, int]:
    match = _K_OF_N.match(str(text))
    if not match:
        raise ValueError(f"expected '<k>_of_<n>', got {text!r}")
    return int(match.group(1)), int(match.group(2))


# ----------------------------------------------------------------------------- eligibility gates (Section 11.2)


def reconstruction_gate(measured: pd.DataFrame, oof: pd.DataFrame, mean_oof: pd.DataFrame, folds: np.ndarray,
                        features: Iterable[str], gate: Mapping[str, Any]) -> pd.DataFrame:
    """Pooled OOF MAE improvement over the out-of-fold mean predictor and the number of folds that improve."""

    from .target_transforms import observed_values

    need, _ = k_of_n(gate["folds_required"])
    threshold = float(gate["min_relative_mae_improvement_over_mean"])
    folds = np.asarray(folds, dtype=float)
    rows = []
    for f in features:
        y = observed_values(measured, f)
        p = pd.to_numeric(oof[f], errors="coerce").to_numpy(dtype=float) if f in oof.columns else np.full(len(y), np.nan)
        m = pd.to_numeric(mean_oof[f], errors="coerce").to_numpy(dtype=float) if f in mean_oof.columns else np.full(len(y), np.nan)
        ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(m)
        mae_r = float(np.mean(np.abs(p[ok] - y[ok]))) if ok.any() else float("nan")
        mae_m = float(np.mean(np.abs(m[ok] - y[ok]))) if ok.any() else float("nan")
        rel = 1.0 - mae_r / mae_m if np.isfinite(mae_r) and np.isfinite(mae_m) and mae_m > 0 else float("nan")
        improved = evaluated = 0
        for fold in sorted({int(v) for v in folds[np.isfinite(folds)]}):
            sel = ok & (folds == fold)
            if not sel.any():
                continue
            evaluated += 1
            improved += int(np.mean(np.abs(p[sel] - y[sel])) < np.mean(np.abs(m[sel] - y[sel])))
        rows.append({"feature": f, "oof_mae": mae_r, "mean_mae": mae_m, "oof_relative_improvement": rel,
                     "folds_improved": improved, "folds_evaluated": evaluated,
                     "reconstruction_pass": bool(np.isfinite(rel) and rel >= threshold and improved >= need)})
    return pd.DataFrame(rows)


def semantic_eligibility(*, rulebook: str, reference: pd.DataFrame, fit_rows: pd.DataFrame, oof: pd.DataFrame | None,
                         mean_oof: pd.DataFrame | None, folds: np.ndarray | None, stability: pd.DataFrame | None,
                         candidates: list[str], rule_policy: Mapping[str, Any], registry: Mapping[str, Any]) -> pd.DataFrame:
    """Eligibility of every candidate antecedent for one rulebook (``configs/rule_policy.yaml`` ``eligibility_gates``).

    ``reference`` holds the measured B of every development patient (availability denominator), ``fit_rows`` the
    measured B of the fit rows (critical QC removed), ``oof``/``mean_oof`` the out-of-fold ridge and mean
    reconstructions aligned with ``fit_rows`` and ``folds`` their outer folds; ``stability`` is the rulebook's
    bootstrap stability table (``tmmri.wedd.stability_gate``).  A feature is eligible only if every gate passes; a
    feature without any cut, or with a cut below the recurrence minimum, fails the stability gate.
    """

    from .rough_sets import antecedent_policy
    from .target_transforms import observed_values

    gates = rule_policy["eligibility_gates"]
    min_frac = float(gates["reference_availability"]["min_observed_fraction_acdc_dev"])
    min_distinct = int(gates["variability"]["min_distinct_values"])
    need_sd = bool(gates["variability"].get("fit_sd_gt_zero", True))
    recon = gates["reconstruction"]
    applies = rulebook in {str(r) for r in recon.get("applies_to", [])}
    min_recurrence = float(gates["post_discretization_stability"]["min_cut_recurrence"])
    allowed, _ = antecedent_policy(dict(registry))
    recon_table = (reconstruction_gate(fit_rows, oof, mean_oof, folds, candidates, recon).set_index("feature")
                   if applies and oof is not None and mean_oof is not None and folds is not None else None)
    rows = []
    for f in candidates:
        ref = observed_values(reference, f)
        fit = observed_values(fit_rows, f)
        fit = fit[np.isfinite(fit)]
        row: dict[str, Any] = {"rulebook": rulebook, "feature": f,
                               "observed_fraction": float(np.isfinite(ref).mean()) if ref.size else 0.0,
                               "n_distinct": int(np.unique(fit).size), "fit_sd": float(np.std(fit, ddof=1)) if fit.size > 1 else float("nan")}
        row["reference_availability_pass"] = bool(row["observed_fraction"] >= min_frac)
        row["variability_pass"] = bool(row["n_distinct"] >= min_distinct and (not need_sd or (np.isfinite(row["fit_sd"]) and row["fit_sd"] > 0)))
        if recon_table is not None:
            rec = recon_table.loc[f]
            row.update({"oof_relative_improvement": float(rec["oof_relative_improvement"]), "folds_improved": int(rec["folds_improved"]),
                        "reconstruction_pass": bool(rec["reconstruction_pass"])})
        else:
            row.update({"oof_relative_improvement": float("nan"), "folds_improved": None, "reconstruction_pass": None})
        cuts = stability[(stability["feature"] == f) & stability["cut"].notna()] if stability is not None and len(stability) else pd.DataFrame()
        row["n_cuts"] = int(len(cuts))
        row["min_cut_recurrence"] = float(cuts["recurrence"].min()) if len(cuts) else float("nan")
        row["stability_pass"] = bool(len(cuts) and (cuts["recurrence"].astype(float) >= min_recurrence).all()) if stability is not None else None
        row["no_tautology_pass"] = f in allowed
        order = [("reference_availability_pass", "INELIGIBLE_REFERENCE_AVAILABILITY"), ("variability_pass", "INELIGIBLE_VARIABILITY"),
                 ("reconstruction_pass", "INELIGIBLE_RECONSTRUCTION"),
                 ("stability_pass", "INELIGIBLE_NO_CUT" if stability is not None and not len(cuts) else "INELIGIBLE_UNSTABLE_CUT"),
                 ("no_tautology_pass", "INELIGIBLE_TAUTOLOGY")]
        failing = [code for key, code in order if row[key] is False]
        row["eligible"] = not failing
        row["ineligibility"] = ";".join(failing) if failing else None
        rows.append(row)
    return pd.DataFrame(rows)


def eligibility_summary(table: pd.DataFrame, class_counts: Mapping[str, Mapping[str, int]], min_class: int) -> dict[str, Any]:
    out: dict[str, Any] = {"schema": "tmmri.semantic_eligibility/1", "rulebooks": {}}
    for rb, sub in table.groupby("rulebook", sort=True):
        counts = {str(k): int(v) for k, v in class_counts.get(rb, {}).items()}
        out["rulebooks"][str(rb)] = {
            "eligible_features": [str(f) for f in sub.loc[sub["eligible"].astype(bool), "feature"]],
            "ineligible": {str(r.feature): str(r.ineligibility) for r in sub.itertuples(index=False) if not bool(r.eligible)},
            "consequent_classes": {c: {"fit_patients": n, "eligible": n >= min_class,
                                       "reason_code": None if n >= min_class else "CLASS_INELIGIBLE"} for c, n in sorted(counts.items())},
        }
    return out


# ----------------------------------------------------------------------------- nested gate substrate (Section 11.6)


class NestedGateSubstrate:
    """``substrate_fn(f, train, held)`` for :func:`tmmri.rough_sets.cross_fitted_gate` (see the module docstring).

    ``A``, ``B``, ``folds`` and ``row_ids`` are aligned with the fit rows; ``inner_labels(f)`` returns full-length
    inner-fold labels for the rows of outer-train(f) (NaN elsewhere); ``heldout`` is the outer OOF reconstruction
    indexed by ``row_ids``.  ``subset`` maps the gate's row positions to fit-row positions (default identity), so a
    rulebook whose universe is a subset (labelled cases only) still uses transitions fitted on every fit row.
    """

    def __init__(self, A: np.ndarray, B: pd.DataFrame, targets: list[str], transition_cfg: Mapping[str, Any],
                 registry: Mapping[str, Any], folds: np.ndarray, inner_labels: Callable[[int], np.ndarray],
                 heldout: pd.DataFrame, columns: list[str], row_ids: Iterable[str], subset: np.ndarray | None = None) -> None:
        self.A = np.asarray(A, dtype=float)
        self.B = B.reset_index(drop=True)
        self.targets = list(targets)
        self.cfg = dict(transition_cfg)
        self.registry = dict(registry)
        self.folds = np.asarray(pd.to_numeric(pd.Series(np.asarray(folds)), errors="coerce"), dtype=float)
        self.inner_labels = inner_labels
        self.heldout = heldout
        self.columns = list(columns)
        self.row_ids = np.asarray(list(row_ids), dtype=object)
        self.subset = np.arange(len(self.row_ids)) if subset is None else np.asarray(subset, dtype=int)
        self._cache: dict[int, pd.DataFrame] = {}

    def frames(self) -> dict[int, pd.DataFrame]:
        """The inner nested OOF reconstructions computed so far, by outer fold (``oof/inner_oof_*_fold<f>.parquet``)."""

        return dict(sorted(self._cache.items()))

    def preload(self, frames: Mapping[int, pd.DataFrame]) -> "NestedGateSubstrate":
        """Reuse inner OOF reconstructions written by ``gate-rules`` (identical inputs; saves the refits)."""

        self._cache.update({int(f): frame for f, frame in frames.items()})
        return self

    def with_subset(self, subset: np.ndarray) -> "NestedGateSubstrate":
        """The same substrate for a sub-universe of the fit rows; the inner OOF reconstructions are shared."""

        import copy

        other = copy.copy(self)
        other.subset = np.asarray(subset, dtype=int)
        return other

    def inner_oof(self, f: int) -> pd.DataFrame:
        from .transition_fit import nested_oof_transition

        if f not in self._cache:
            train = np.flatnonzero(np.isfinite(self.folds) & (self.folds != f))
            inner = np.asarray(self.inner_labels(int(f)), dtype=float)[train]
            if not np.isfinite(inner).all():
                raise ValueError(f"inner fold labels missing for outer-train({f})")
            oof = nested_oof_transition(self.A[train], self.B.iloc[train].reset_index(drop=True), self.targets, self.cfg,
                                        self.registry, inner, lambda g, pos, _inner=inner: _inner[np.asarray(pos, dtype=int)])
            oof.index = self.row_ids[train]
            self._cache[f] = oof
        return self._cache[f]

    def __call__(self, f: int, train: np.ndarray, held: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
        oof = self.inner_oof(int(f))
        train_ids = self.row_ids[self.subset[np.asarray(train, dtype=int)]]
        held_ids = self.row_ids[self.subset[np.asarray(held, dtype=int)]]
        return oof.loc[train_ids, self.columns], self.heldout.loc[held_ids, self.columns]


def acdc_inner_labels(inner_table: pd.DataFrame, row_ids: Iterable[str]) -> Callable[[int], np.ndarray]:
    """Inner-fold labels from ``contracts/inner_folds.parquet`` (hash rule of ``configs/splits.yaml``)."""

    ids = list(row_ids)

    def fn(f: int) -> np.ndarray:
        rows = inner_table[inner_table["outer_fold_heldout"] == int(f)].set_index("sample_id")["inner_fold"]
        return rows.reindex(ids).to_numpy(dtype=float)

    return fn


def remaining_fold_labels(folds: np.ndarray) -> Callable[[int], np.ndarray]:
    """Refit-arm inner labels: inside outer-train(f) the remaining refit folds are the inner folds."""

    base = np.asarray(pd.to_numeric(pd.Series(np.asarray(folds)), errors="coerce"), dtype=float)

    def fn(f: int) -> np.ndarray:
        return np.where(base == float(f), np.nan, base)

    return fn


# ----------------------------------------------------------------------------- one rulebook end to end


def cutset_fn_for(profile: str, wedd_cfg: Mapping[str, Any]) -> Callable[[pd.DataFrame, np.ndarray, int], Any]:
    from .wedd import fit_cuts

    def fn(values: pd.DataFrame, decision: np.ndarray, n_train: int):
        return fit_cuts(values, pd.Series(np.asarray(decision, dtype=object), index=values.index), profile, dict(wedd_cfg), n_fit_patients=int(n_train))

    return fn


def gate_and_apply(*, values: pd.DataFrame, decision_fn: Callable[[np.ndarray, np.ndarray], Any], folds: np.ndarray,
                   rule_policy: Mapping[str, Any], book: Any, profile: str, wedd_cfg: Mapping[str, Any], anchors: np.ndarray,
                   substrate_fn: Callable[..., Any] | None, registry: Mapping[str, Any], rulebook_id: str,
                   attributes: list[str]) -> tuple[pd.DataFrame, Any]:
    """Cross-fitted gate (Section 11.6) followed by :func:`tmmri.rough_sets.apply_gate`."""

    from .rough_sets import apply_gate, cross_fitted_gate

    table = cross_fitted_gate(values[attributes] if attributes else values, decision_fn, folds, dict(rule_policy), final_rulebook=book,
                              cutset_fn=cutset_fn_for(profile, wedd_cfg), anchors=anchors, substrate_fn=substrate_fn,
                              induce_kwargs={"registry": dict(registry), "rulebook_id": rulebook_id, "attributes": list(attributes)})
    return table, apply_gate(book, table)


def rulebook_payload(rulebook_id: str, gated: Any, gate: pd.DataFrame, provenance: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """(file name, JSON payload): ``rulebook.json`` with the eligible rules, or ``EMPTY.json`` with every failing gate."""

    from .rough_sets import ELIGIBLE

    eligible = [r for r in gated.rules if r.eligibility == ELIGIBLE]
    if eligible:
        return "rulebook.json", {"rulebook": rulebook_id, "consequent_endpoint": gated.consequent_kind,
                                 "records": [dict(r.to_record(rulebook_id), provenance=dict(provenance)) for r in eligible],
                                 "provenance": dict(provenance)}
    record = gated.empty_record()
    record["gate_table"] = gate.to_dict("records")
    record["provenance"] = dict(provenance)
    return "EMPTY.json", record


def fit_rulebook(*, rulebook_id: str, values: pd.DataFrame, decision: pd.Series, anchors: pd.Series, folds: np.ndarray,
                 profile: str, study: Any, decision_fn: Callable[[np.ndarray, np.ndarray], Any],
                 substrate_fn: Callable[..., Any] | None, reference: pd.DataFrame, oof: pd.DataFrame | None,
                 mean_oof: pd.DataFrame | None, rule_policy: Mapping[str, Any] | None = None,
                 stability_replicates: int | None = None, n_jobs: int = 1,
                 availability: pd.DataFrame | None = None, wedd_cfg: Mapping[str, Any] | None = None,
                 clusters: pd.Series | None = None) -> dict[str, Any]:
    """Cuts, stability gate, eligibility, VPRS discovery and the cross-fitted gate for one rulebook.

    ``clusters`` (indexed like ``values``; the split-manifest ``cluster_id``) are the bootstrap clusters of the
    threshold stability gate (default: every row its own cluster).

    ``values`` (index = row ids, candidate columns) is the rulebook substrate of the fit rows, ``decision`` its
    consequent (``None`` = no label; excluded by discovery), ``reference`` the measured B of the same rows and
    ``availability`` the measured B of every partition patient (reference-availability denominator; default
    ``reference``).  ``wedd_cfg`` replaces ``study.wedd`` (the WEDD settings sensitivity grid of ``run-ablations``).
    """

    from .rough_sets import induce_vprs_rules
    from .wedd import fit_cuts, stability_gate

    policy = dict(rule_policy or study.rule_policy)
    wedd = dict(wedd_cfg if wedd_cfg is not None else study.wedd)
    boot = study.bootstrap["threshold_bootstrap"]
    replicates = int(boot["replicates"] if stability_replicates is None else stability_replicates)
    candidates = [c for c in study.rule_candidates() if c in values.columns]
    cuts = fit_cuts(values[candidates], decision, profile, wedd)
    stability = stability_gate(values[candidates], decision, profile, wedd, seed=int(boot["seed"]), n_replicates=replicates,
                               n_jobs=n_jobs, reference=cuts, groups=None if clusters is None else clusters.loc[values.index].to_numpy(dtype=object))
    eligibility = semantic_eligibility(rulebook=rulebook_id, reference=reference if availability is None else availability,
                                       fit_rows=reference, oof=oof, mean_oof=mean_oof,
                                       folds=folds, stability=stability, candidates=candidates, rule_policy=policy, registry=study.registry)
    attributes = [str(f) for f in eligibility.loc[eligibility["eligible"].astype(bool), "feature"]]
    states = cuts.transform(values[candidates])
    book = induce_vprs_rules(states, decision.loc[states.index], anchors.loc[states.index], policy, registry=study.registry,
                             attributes=attributes, consequent_kind=rulebook_id, rulebook_id=rulebook_id)
    gate, gated = gate_and_apply(values=values[candidates], decision_fn=decision_fn, folds=folds, rule_policy=policy, book=book,
                                 profile=profile, wedd_cfg=wedd, anchors=anchors.loc[values.index].to_numpy(dtype=object),
                                 substrate_fn=substrate_fn, registry=study.registry, rulebook_id=rulebook_id, attributes=attributes)
    return {"cuts": cuts, "stability": stability, "eligibility": eligibility, "attributes": attributes, "states": states,
            "induced": book, "gate": gate, "gated": gated}


# ----------------------------------------------------------------------------- inference inputs (Section 11.7)


def qc_route_frame(b_rows: pd.DataFrame, formal_finite: np.ndarray, study: Any) -> pd.DataFrame:
    """``sample_id``, ``qc_abstain``, ``qc_reason_code`` for :func:`tmmri.inference.route`.

    Critical QC reasons (``configs/cohorts.yaml`` severities, fail closed) and non-finite formal A rows
    (``INPUT_NONFINITE``, registry/reason_codes.yaml; configs/formal.yaml ``finiteness``) route ``QC_ABSTAIN``.
    """

    from .semantics.qc import assess_qc_v6, parse_reason_tokens

    finite = np.asarray(formal_finite, dtype=bool)
    reasons = b_rows["qc_failure_reason"] if "qc_failure_reason" in b_rows.columns else pd.Series(None, index=b_rows.index)
    rows = []
    for i, (sid, value) in enumerate(zip(b_rows.index, reasons.tolist())):
        codes = [] if finite[i] else ["INPUT_NONFINITE"]
        assessment = assess_qc_v6(parse_reason_tokens(value), study.cohorts, study.reason_codes)
        if assessment.route == "QC_ABSTAIN":
            codes += [c for c in assessment.abstain_codes if c not in codes]
        rows.append({"sample_id": str(sid), "qc_abstain": bool(codes), "qc_reason_code": ";".join(codes) if codes else None})
    return pd.DataFrame(rows, columns=["sample_id", "qc_abstain", "qc_reason_code"])


def decode_reconstructions(model: Any, fold_models: list[Any], x: np.ndarray, index: pd.Index, registry: Mapping[str, Any],
                           transition_cfg: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.DataFrame]]:
    """(decoded values, reconstructed states, decoded fold-mapper values) of the frozen transition on ``x``.

    Decoders of SPECIFICATION.md 9.5: the registry-domain guard (``configs/transition.yaml`` ``decoders.domain_guard``)
    blocks a value outside the domain by more than the configured fraction of the fit range (``NUMERIC_DOMAIN_BLOCKED``)
    and clips values inside the guard; the outer-fold transitions are decoded the same way for per-object stability
    (Section 11.4).  Raw (unclipped) reconstructions stay the substrate of every error metric.
    """

    values, states = model.decode(x, dict(registry), dict(transition_cfg), index=index)
    folds = [m.decode(x, dict(registry), dict(transition_cfg), index=index)[0] for m in fold_models]
    return values, states, folds


def routing_substrate(values: pd.DataFrame, states: pd.DataFrame, fold_values: list[pd.DataFrame], cuts: Any,
                      wedd_cfg: Mapping[str, Any], columns: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Antecedent values with ``<feature>__state`` companions for :func:`route_records`, and the per-object states.

    A cell keeps its decoder state (``ESTIMATED``, ``NUMERIC_DOMAIN_BLOCKED``, ``TARGET_INELIGIBLE``,
    ``INPUT_QC_BLOCKED``); an ``ESTIMATED`` cell whose symbol under the frozen cuts of the rulebook is not reproduced
    by the required number of outer-fold transitions (``configs/wedd.yaml`` ``per_object_stability``) becomes
    ``UNSTABLE_ESTIMATE``.  Blocked and unstable cells route with FEATURE_DOMAIN_BLOCKED, FEATURE_TARGET_INELIGIBLE or
    FEATURE_UNSTABLE when a rule needs them (``tmmri.wedd.STATE_TO_UNAVAILABLE_CODE``).
    """

    from .wedd import CutSet, per_object_stability

    cols = [c for c in columns if c in values.columns]
    features = [f for f in cuts.order if f in cols]
    stab = pd.DataFrame(index=values.index)
    if features:
        sub = CutSet({f: cuts.features[f] for f in features}, cuts.profile_name, features)
        stab = per_object_stability([fv[features] for fv in fold_values], values[features], sub, dict(wedd_cfg))
    failure = str(dict(wedd_cfg)["per_object_stability"]["failure_state"])
    out = values[cols].copy()
    for c in cols:
        st = states[c].astype(object).to_numpy() if c in states.columns else np.full(len(out), "ESTIMATED", dtype=object)
        if c in stab.columns:
            st = np.where((st == "ESTIMATED") & (stab[c].to_numpy() == failure), failure, st)
        out[f"{c}__state"] = st
    return out, stab


def route_records(records: list[dict[str, Any]], rulebook_id: str, cuts: Any, substrate: pd.DataFrame, qc: pd.DataFrame,
                  study: Any, *, rule_policy: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Route every row of ``substrate`` (index = sample_id; ``<feature>__state`` companions honoured)."""

    from .inference import route
    from .rough_sets import Rulebook

    candidates = [c for c in study.rule_candidates() if c in substrate.columns]
    frame = substrate[candidates + [f"{c}__state" for c in candidates if f"{c}__state" in substrate.columns]]
    states = cuts.transform(frame)
    states = states.rename_axis("sample_id").reset_index()
    book = Rulebook.from_records(records, rulebook_id=rulebook_id, consequent_kind=rulebook_id)
    return route(book, states, qc, dict(rule_policy or study.rule_policy), study.reason_codes, qc_policy=study.cohorts)


# ----------------------------------------------------------------------------- downstream refit bootstrap (Section 13.3)


def rule_intervals(rule: Any, cuts: Any) -> dict[str, tuple[float, float]]:
    """Physical interval ``(low, high]`` of every antecedent literal under ``cuts`` (state ``b<i>of<n>``)."""

    from .wedd import parse_state_token

    out = {}
    for feature, token in rule.antecedent:
        parsed = parse_state_token(token)
        edges = list(cuts.features[feature].cuts) if feature in cuts.features else []
        if parsed is None or parsed[1] != len(edges) + 1:
            out[feature] = (float("nan"), float("nan"))
            continue
        i, n = parsed
        out[feature] = (-math.inf if i == 0 else float(edges[i - 1]), math.inf if i == n - 1 else float(edges[i]))
    return out


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Right-closed intervals ``(low, high]`` overlap (a NaN bound, i.e. an unparseable literal, never overlaps)."""

    if any(math.isnan(x) for x in (*a, *b)):
        return False
    return max(a[0], b[0]) < min(a[1], b[1])


def _refit_replicate(item: tuple[int, np.random.SeedSequence], payload: Mapping[str, Any]) -> dict[str, Any]:
    from .rough_sets import induce_vprs_rules
    from .transition_fit import nested_oof_transition
    from .wedd import fit_cuts

    index, seed_seq = item
    rng = np.random.default_rng(seed_seq)
    ids = payload["row_ids"]
    groups = payload["cluster_rows"]
    picks = rng.integers(0, len(groups), size=len(groups))
    draw = np.concatenate([groups[g] for g in picks]) if len(groups) else np.zeros(0, dtype=int)   # a drawn cluster brings all its rows
    n = int(draw.size)
    a = payload["A"][draw]
    b = payload["B"].iloc[draw].reset_index(drop=True)
    folds = payload["folds"][draw]
    inner = {f: labels[draw] for f, labels in payload["inner_by_fold"].items()}
    out: dict[str, Any] = {"replicate": int(index), "rulebooks": {}, "error": None}
    try:
        oof = None
        if any(spec["substrate"] == "reconstructed" for spec in payload["rulebooks"].values()):
            oof = nested_oof_transition(a, b, payload["targets"], payload["transition_cfg"], payload["registry"], folds,
                                        lambda f, pos: inner[int(f)][np.asarray(pos, dtype=int)])
        for rb, spec in payload["rulebooks"].items():
            attrs = list(spec["attributes"])
            if not attrs:
                out["rulebooks"][rb] = []
                continue
            source = oof if spec["substrate"] == "reconstructed" else b
            cols = attrs + ([f"{c}__state" for c in attrs if f"{c}__state" in source.columns] if spec["substrate"] == "measured" else [])
            values = source[cols].copy()
            values.index = [f"r{i}" for i in range(n)]
            decision = pd.Series(np.asarray(spec["decision"], dtype=object)[draw], index=values.index)
            cuts = fit_cuts(values, decision, payload["profile"], payload["wedd_cfg"], features=attrs, n_fit_patients=n, record_candidates=False)
            states = cuts.transform(values)
            anchors = pd.Series(np.asarray(ids, dtype=object)[draw], index=values.index)
            book = induce_vprs_rules(states, decision, anchors, payload["rule_policy"], registry=payload["registry"], attributes=attrs,
                                     consequent_kind=rb, rulebook_id=rb)
            out["rulebooks"][rb] = [(r.identity, str(r.consequent), tuple(sorted(r.features)), rule_intervals(r, cuts)) for r in book.rules]
    except Exception as exc:  # a failed replicate is invalid, never silently a non-recurrence
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def downstream_refit_bootstrap(*, A: np.ndarray, B: pd.DataFrame, folds: np.ndarray, inner_labels: Callable[[int], np.ndarray],
                               row_ids: Iterable[str], study: Any, rulebooks: Mapping[str, Mapping[str, Any]], profile: str,
                               n_replicates: int, seed: int, n_jobs: int = 1, clusters: Iterable[str] | None = None) -> pd.DataFrame:
    """Recurrence of every reference rule over ``n_replicates`` cluster resamples with the transition, target
    transforms, thresholds and discovery refitted (``configs/bootstrap.yaml`` ``downstream_refit_bootstrap``).

    ``clusters`` are the row-aligned bootstrap clusters (``cluster_id``; default: every row its own cluster); a drawn
    cluster brings all its rows, and clusters are ordered by first appearance, so singleton clusters reproduce the
    patient bootstrap draw for draw.

    ``rulebooks[name]`` = ``{"substrate": "reconstructed" | "measured", "decision": array aligned with the rows,
    "attributes": eligible features, "reference": Rulebook, "cuts": CutSet}``.  *Exact* recurrence = a rule with the
    same identity (features, bin indices and counts, consequent); *relaxed* = same consequent and feature set with
    overlapping physical intervals on every antecedent.  Replicate ``r`` uses ``SeedSequence(seed).spawn(n)[r]``;
    resampled rows keep their folds, so results do not depend on ``n_jobs``.
    """

    from .runs import parallel_map

    fold_arr = np.asarray(pd.to_numeric(pd.Series(np.asarray(folds)), errors="coerce"), dtype=float)
    fold_ids = sorted({int(v) for v in fold_arr[np.isfinite(fold_arr)]})
    row_ids = list(row_ids)
    if clusters is None:
        cluster_rows = [np.array([i]) for i in range(len(row_ids))]
    else:
        codes, uniques = pd.factorize(pd.Series(list(clusters), dtype=object).astype(str))
        if len(codes) != len(row_ids):
            raise ValueError("clusters must be row-aligned with row_ids")
        cluster_rows = [np.flatnonzero(codes == k) for k in range(len(uniques))]
    payload = {
        "A": np.asarray(A, dtype=float), "B": B.reset_index(drop=True), "folds": fold_arr, "row_ids": row_ids, "cluster_rows": cluster_rows,
        "inner_by_fold": {f: np.asarray(inner_labels(f), dtype=float) for f in fold_ids},
        "targets": study.transition_targets(), "transition_cfg": study.transition, "registry": study.registry,
        "wedd_cfg": study.wedd, "rule_policy": study.rule_policy, "profile": profile,
        "rulebooks": {rb: {"substrate": spec["substrate"], "decision": list(spec["decision"]), "attributes": list(spec["attributes"])}
                      for rb, spec in rulebooks.items()},
    }
    children = np.random.SeedSequence(int(seed)).spawn(int(n_replicates))
    results = parallel_map(functools.partial(_refit_replicate, payload=payload), list(enumerate(children)), n_jobs=n_jobs)
    valid = [r for r in results if r["error"] is None]
    rows = []
    for rb, spec in rulebooks.items():
        for rule in spec["reference"].rules:
            ref_iv = rule_intervals(rule, spec["cuts"])
            exact = relaxed = 0
            for rep in valid:
                found = rep["rulebooks"].get(rb, [])
                if any(identity == rule.identity for identity, *_ in found):
                    exact += 1
                if any(cons == str(rule.consequent) and feats == tuple(sorted(rule.features))
                       and all(_overlap(ref_iv[f], iv.get(f, (math.nan, math.nan))) for f in feats) for _, cons, feats, iv in found):
                    relaxed += 1
            n_valid = len(valid)
            rows.append({"rulebook": rb, "rule_id": rule.rule_id, "consequent": str(rule.consequent), "length": rule.length,
                         "eligibility": rule.eligibility, "n_replicates": int(n_replicates), "n_valid_replicates": n_valid,
                         "exact_recurrence": exact / n_valid if n_valid else float("nan"),
                         "relaxed_recurrence": relaxed / n_valid if n_valid else float("nan"), "seed": int(seed)})
    table = pd.DataFrame(rows, columns=["rulebook", "rule_id", "consequent", "length", "eligibility", "n_replicates", "n_valid_replicates",
                                        "exact_recurrence", "relaxed_recurrence", "seed"])
    table.attrs["failed_replicates"] = [{"replicate": r["replicate"], "error": r["error"]} for r in results if r["error"] is not None]
    return table
