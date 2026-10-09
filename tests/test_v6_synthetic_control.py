"""End-to-end synthetic implementation control (SPECIFICATION.md 6.0.0, Section 15.2): formal A -> nested OOF B-hat ->
WEDD (primary profile) -> VPRS -> cross-fitted gate -> routes on an independent test lattice.

Semantic targets lie on a 9 x 9 lattice with planted cuts 2.5 and 5.5 on both antecedents; the nine classes are the
cells of the fully crossed 3 x 3 bin grid (nine two-feature rules). The crossed design matters: when a class spans a
feature's gap, sub-1e-7 ridge perturbations that correlate with the class let a within-level micro-cut compete with the
gap, which is a property of the fixture, not of WEDD. A is a 10-d formal space with an exact affine map to B (plus
null-space variation); noise is added to both A and B in units of their SD. Thresholds and rules are read from the fitted CutSet and Rulebook objects, never from
proxies. This is an algorithm test, not simulated clinical evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import REPO, SEED_SYNTHETIC

from tmmri import inference as inf
from tmmri import rough_sets as rs
from tmmri import transition_fit as tf
from tmmri import wedd

PLANTED = {"sem_0": [2.5, 5.5], "sem_1": [2.5, 5.5]}
PLANTED_RULES = {((("sem_0", f"b{a}of3"), ("sem_1", f"b{b}of3")), f"C{a}{b}") for a in range(3) for b in range(3)}
SYNTHETIC_REGISTRY = {
    "features": [
        {"name": "sem_0", "target_transform": "identity", "transition_role": "candidate", "rule_role": "candidate", "domain": [None, None], "dependencies": []},
        {"name": "sem_1", "target_transform": "identity", "transition_role": "candidate", "rule_role": "candidate", "domain": [None, None], "dependencies": []},
        {"name": "sem_2", "target_transform": "identity", "transition_role": "candidate_secondary", "rule_role": "reconstruction_only", "domain": [None, None], "dependencies": []},
    ],
    "primary_rulebook_antecedent_candidates": ["sem_0", "sem_1"],
}


def _load(rel):
    with (REPO / rel).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _labels(b0, b1):
    bins0 = np.searchsorted(PLANTED["sem_0"], b0, side="left")
    bins1 = np.searchsorted(PLANTED["sem_1"], b1, side="left")
    return np.array([f"C{a}{b}" for a, b in zip(bins0, bins1)])


def _lattice(rng, T, shift, reps):
    lv = np.arange(9.0)
    b0, b1 = (g.ravel() for g in np.meshgrid(lv, lv, indexing="ij"))
    b0, b1 = np.tile(b0, reps), np.tile(b1, reps)
    B = np.column_stack([b0, b1, rng.normal(size=b0.size)])
    P = np.linalg.pinv(T)
    Q = np.eye(T.shape[0]) - T @ P  # projector onto the complement of range(T): (Z Q) T = 0
    A = (B - shift) @ P + rng.normal(size=(b0.size, T.shape[0])) @ Q
    return A, B, _labels(b0, b1)


def _macro_f1(y, pred, classes):
    f1 = []
    for c in classes:
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        f1.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0)
    return float(np.mean(f1))


def run_control(seed: int, noise_sd: float) -> dict:
    tcfg, wcfg, rcfg, codes = _load("configs/transition.yaml"), _load("configs/wedd.yaml"), _load("configs/rule_policy.yaml"), _load("registry/reason_codes.yaml")
    rng = np.random.default_rng(seed)
    T = rng.normal(size=(10, 3))
    shift = np.array([1.0, -2.0, 0.5])
    A_clean, B_clean, y = _lattice(rng, T, shift, reps=5)
    A = A_clean + noise_sd * A_clean.std(axis=0) * rng.normal(size=A_clean.shape)
    B = B_clean + noise_sd * B_clean.std(axis=0) * rng.normal(size=B_clean.shape)
    Bdf = pd.DataFrame(B, columns=["sem_0", "sem_1", "sem_2"])
    targets = list(Bdf.columns)
    folds = np.zeros(len(y), dtype=int)
    for c in sorted(set(y)):
        idx = np.flatnonzero(y == c)
        folds[idx] = np.arange(idx.size) % 5 + 1
    inner = tf.positional_inner_folds(5)
    oof = tf.nested_oof_transition(A, Bdf, targets, tcfg, SYNTHETIC_REGISTRY, folds, inner)
    antecedents = SYNTHETIC_REGISTRY["primary_rulebook_antecedent_candidates"]
    substrate = oof[antecedents]
    profile = wcfg["primary_profile"]
    cuts = wedd.fit_cuts(substrate, y, profile, wcfg)
    states = cuts.transform(substrate)
    anchors = pd.Series([f"syn{i:04d}" for i in range(len(y))])
    kw = dict(registry=SYNTHETIC_REGISTRY, consequent_kind="dataset_label")
    candidates = rs.induce_vprs_rules(states, y, anchors, rcfg, **kw)

    def cutset_fn(train_values, train_decision, n_train):
        return wedd.fit_cuts(train_values, train_decision, profile, wcfg, n_fit_patients=n_train)

    gate = rs.cross_fitted_gate(substrate, lambda tr, ap: y[ap], folds, rcfg, final_rulebook=candidates, cutset_fn=cutset_fn, anchors=anchors, induce_kwargs=kw)
    rulebook = rs.apply_gate(candidates, gate).eligible()
    # frozen bundle: final transition on all development rows, applied to an independent test lattice
    final = tf.fit_transition_v6(A, Bdf, targets, tcfg, SYNTHETIC_REGISTRY, folds)
    test_rng = np.random.default_rng(seed + 1)
    A_test, B_test, y_test = _lattice(test_rng, T, shift, reps=2)
    A_test = A_test + noise_sd * A_clean.std(axis=0) * test_rng.normal(size=A_test.shape)
    test_states = cuts.transform(final.predict(A_test)[antecedents])
    test_states.insert(0, "sample_id", [f"test{i:04d}" for i in range(len(y_test))])
    routes = inf.route(rulebook, test_states, None, rcfg, codes)
    accepted = routes["route"].eq("EXACT_ACCEPT").to_numpy()
    pred = routes["label"].fillna("ABSTAIN").astype(str).to_numpy()
    recovered = {(r.antecedent, r.consequent) for r in rulebook.rules}
    fitted_cuts = cuts.cuts()
    errors = [min(abs(c - t) for c in fitted_cuts[f]) if fitted_cuts[f] else np.inf for f, ts in PLANTED.items() for t in ts]
    return {
        "noise_sd": noise_sd,
        "cuts": fitted_cuts,
        "max_threshold_error": float(max(errors)),
        "n_rules": len(rulebook.rules),
        "rules_recovered_exactly": len(recovered & PLANTED_RULES),
        "spurious_rules": len(recovered - PLANTED_RULES),
        "accepted_coverage": float(accepted.mean()),
        "macro_f1": _macro_f1(y_test, pred, sorted(set(y_test))),
        "route_counts": inf.route_counts(routes, codes),
        "thresholds_from_fitted_objects": isinstance(cuts, wedd.CutSet) and isinstance(rulebook, rs.Rulebook),
        "lambda_edge_flags": dict(final.edge_flags),
    }


def test_exact_recovery_at_zero_noise():
    result = run_control(SEED_SYNTHETIC, 0.0)
    assert result["rules_recovered_exactly"] == 9 and result["spurious_rules"] == 0
    assert result["max_threshold_error"] < 0.02
    assert result["accepted_coverage"] == 1.0 and result["macro_f1"] > 0.99
    assert result["route_counts"]["EXACT_ACCEPT"] == sum(result["route_counts"].values())


@pytest.mark.parametrize("noise_sd", [0.05, 0.10, 0.15, 0.20])
def test_degradation_is_reported_not_hidden(noise_sd):
    result = run_control(SEED_SYNTHETIC, noise_sd)
    print({k: result[k] for k in ("noise_sd", "max_threshold_error", "n_rules", "rules_recovered_exactly", "accepted_coverage", "macro_f1")})
    assert 0.0 <= result["macro_f1"] <= 1.0 and 0.0 <= result["accepted_coverage"] <= 1.0
    assert result["thresholds_from_fitted_objects"] is True
    assert sum(result["route_counts"].values()) == 162
