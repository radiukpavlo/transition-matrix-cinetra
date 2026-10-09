"""B5 gate power (exploratory design property of the v6 gate; synthetic data only, no cohort data).

A rule "planted feature in the high region -> consequent" is planted in synthetic acdc_dev-sized datasets: ``support``
of ``n_patients`` patients lie in the region and carry the consequent with probability ``agreement`` (otherwise another
class); outside the region the consequent occurs at the background rate.  Per cell (agreement x support, plus null
cells whose region agreement is a value a rule should fail at) ``n_datasets`` datasets are drawn and the share in which
the planted rule passes each gate is reported with a Wilson interval: P(a true rule passes) for the planted cells, the
false-pass rate for the null cells.

* ``pipeline`` mode runs the v6 code on every dataset: WEDD cuts (:func:`tmmri.wedd.fit_cuts`), VPRS discovery
  (:func:`tmmri.rough_sets.induce_vprs_rules`) and the cross-fitted gate over five folds
  (:func:`tmmri.rough_sets.cross_fitted_gate`; cuts and discovery refitted per fold, matches counted only where the
  rule is reproduced).  The eligibility gates (stability bootstrap, reconstruction) are not simulated: the planted
  feature is taken as eligible.  A rule counts as the planted one when its consequent is the planted consequent and its
  antecedent uses the planted feature; any other rule passing the v6 gate is a spurious pass.
* ``counts`` mode applies the same discovery and gate arithmetic to the planted counts directly (the region is
  discretized exactly): discovered when the region holds >= ``min_antecedent_n_patients`` patients with confidence
  >= ``1 - beta``; reproduced in a fold when the training folds satisfy the same; pooled held-out matches and
  agreements over the reproduced folds.

:func:`gate_power_cell` is a pure function of one cell (plain-data arguments, seeded by the cell), so the grid can be
fanned out (for example on Modal CPU) with results identical to the local loop of :func:`run_grid`.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

OUTPUTS = ("cells.parquet", "cells.csv", "summary.json")
KIND_CODES = {"planted": 1, "null": 2}
MODE_CODES = {"counts": 1, "pipeline": 2}


def gate_passes(matched: int, agree: int, gate: Mapping[str, Any]) -> bool:
    """One gate on pooled held-out counts: ``min_matched``, optional ``min_agreement`` and ``wilson_lower_min``."""

    from ...metrics import wilson_interval

    if matched < int(gate.get("min_matched", 0)) or matched <= 0:
        return False
    if "min_agreement" in gate and agree / matched < float(gate["min_agreement"]) - 1e-12:
        return False
    if "wilson_lower_min" in gate:
        lo, _ = wilson_interval(agree, matched)
        if not np.isfinite(lo) or lo < float(gate["wilson_lower_min"]) - 1e-12:
            return False
    return True


def cell_seed_sequence(seed: int, cell: Mapping[str, Any]) -> np.random.SeedSequence:
    """Seed of one cell from (seed, mode, kind, agreement in thousandths, support): independent of the grid layout."""

    return np.random.SeedSequence([int(seed), MODE_CODES[str(cell["mode"])], KIND_CODES[str(cell["kind"])],
                                   int(round(float(cell["agreement"]) * 1000)), int(cell["support"])])


def _balanced_folds(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    folds = (np.arange(n) % k) + 1
    return folds[rng.permutation(n)]


def synthetic_dataset(rng: np.random.Generator, *, agreement: float, support: int, settings: Mapping[str, Any]) -> dict[str, Any]:
    """One dataset: planted region, decisions, folds and feature values (planted and distractor features)."""

    n, k = int(settings["n_patients"]), int(settings["n_folds"])
    classes = [str(c) for c in settings["classes"]]
    consequent = str(settings["consequent"])
    others = [c for c in classes if c != consequent]
    region = np.zeros(n, dtype=bool)
    region[rng.choice(n, size=int(support), replace=False)] = True
    hit = rng.random(n) < np.where(region, float(agreement), float(settings["background_consequent_rate"]))
    decision = np.where(hit, consequent, np.asarray(others, dtype=object)[rng.integers(0, len(others), size=n)]).astype(object)
    folds = _balanced_folds(rng, n, k)
    planted = np.where(region, rng.uniform(150.0, 200.0, size=n), rng.uniform(50.0, 140.0, size=n))
    distractor = rng.uniform(50.0, 200.0, size=n)
    return {"region": region, "decision": decision, "folds": folds, "planted": planted, "distractor": distractor}


def _counts_replicate(data: Mapping[str, Any], params: Mapping[str, Any], consequent: str) -> tuple[bool, int, int]:
    region, decision, folds = data["region"], data["decision"], data["folds"]
    hit = decision == consequent

    def valid(rows: np.ndarray) -> bool:
        n = int(rows.sum())
        return n >= params["min_n"] and n > 0 and hit[rows].sum() / n >= params["min_confidence"] - 1e-12

    if not valid(region):
        return False, 0, 0
    matched = agree = 0
    for f in sorted(set(folds.tolist())):
        if valid(region & (folds != f)):
            held = region & (folds == f)
            matched += int(held.sum())
            agree += int((held & hit).sum())
    return True, matched, agree


def _pipeline_replicate(data: Mapping[str, Any], settings: Mapping[str, Any], rule_policy: Mapping[str, Any], wedd_cfg: Mapping[str, Any],
                        registry: Mapping[str, Any], profile: str) -> tuple[bool, list[tuple[bool, int, int]], int]:
    """(planted rule discovered, [(is planted, matched, agree) per final rule], n final rules) from the v6 code."""

    from ...rough_sets import cross_fitted_gate, induce_vprs_rules
    from ...rule_pipeline import cutset_fn_for

    f1, f2 = str(settings["planted_feature"]), str(settings["distractor_feature"])
    consequent = str(settings["consequent"])
    n = len(data["decision"])
    ids = [f"SYN:p{i:03d}" for i in range(n)]
    values = pd.DataFrame({f1: data["planted"], f2: data["distractor"]}, index=ids)
    decision = pd.Series(data["decision"], index=ids, dtype=object)
    cutset_fn = cutset_fn_for(profile, wedd_cfg)
    cuts = cutset_fn(values, decision.to_numpy(dtype=object), n)
    attributes = [f1, f2]
    book = induce_vprs_rules(cuts.transform(values), decision, pd.Series(ids, index=ids), dict(rule_policy), registry=dict(registry),
                             attributes=attributes, consequent_kind="d_model", rulebook_id="d_model")
    planted = [consequent == r.consequent and f1 in r.features for r in book.rules]
    if not book.rules:
        return False, [], 0
    labels = decision.to_numpy(dtype=object)
    table = cross_fitted_gate(values, lambda train, target: labels[target], data["folds"], dict(rule_policy), final_rulebook=book,
                              cutset_fn=cutset_fn, anchors=np.asarray(ids, dtype=object),
                              induce_kwargs={"registry": dict(registry), "rulebook_id": "d_model", "attributes": attributes}).set_index("rule_id")
    stats = [(is_planted, int(table.at[r.rule_id, "oof_matched"]), int(table.at[r.rule_id, "oof_agree"])) for r, is_planted in zip(book.rules, planted)]
    return any(planted), stats, len(book.rules)


def gate_power_cell(cell: Mapping[str, Any], *, settings: Mapping[str, Any], rule_policy: Mapping[str, Any], wedd_cfg: Mapping[str, Any],
                    registry: Mapping[str, Any], profile: str) -> dict[str, Any]:
    """Pass counts of every gate for one cell ``{mode, kind, agreement, support, n_datasets}`` (pure; seeded by the cell)."""

    from ...metrics import wilson_interval
    from ...rough_sets import _discovery_params

    params = _discovery_params(dict(rule_policy))
    gates = dict(settings["gates"])
    n_datasets = int(cell["n_datasets"])
    children = cell_seed_sequence(int(settings["seed"]), cell).spawn(n_datasets)
    passes = {g: 0 for g in gates}
    spurious = discovered = 0
    matched_sum = 0
    for ss in children:
        rng = np.random.default_rng(ss)
        data = synthetic_dataset(rng, agreement=float(cell["agreement"]), support=int(cell["support"]), settings=settings)
        if str(cell["mode"]) == "counts":
            found, matched, agree = _counts_replicate(data, params, str(settings["consequent"]))
            stats = [(True, matched, agree)] if found else []
        else:
            found, stats, _ = _pipeline_replicate(data, settings, rule_policy, wedd_cfg, registry, profile)
        discovered += int(found)
        planted_stats = [(m, a) for is_planted, m, a in stats if is_planted]
        matched_sum += max([m for m, _ in planted_stats], default=0)
        for g, spec in gates.items():
            passes[g] += int(any(gate_passes(m, a, spec) for m, a in planted_stats))
        spurious += int(any(gate_passes(m, a, gates["v6"]) for is_planted, m, a in stats if not is_planted)) if "v6" in gates else 0
    out = {"mode": str(cell["mode"]), "kind": str(cell["kind"]), "agreement": float(cell["agreement"]), "support": int(cell["support"]),
           "n_datasets": n_datasets, "discovery_rate": discovered / n_datasets if n_datasets else None,
           "mean_pooled_oof_matched": matched_sum / n_datasets if n_datasets else None,
           "spurious_v6_pass_rate": spurious / n_datasets if n_datasets else None}
    for g in gates:
        lo, hi = wilson_interval(passes[g], n_datasets) if n_datasets else (None, None)
        out[f"pass_{g}"] = passes[g] / n_datasets if n_datasets else None
        out[f"pass_{g}_wilson_lower"], out[f"pass_{g}_wilson_upper"] = lo, hi
        out[f"n_pass_{g}"] = int(passes[g])
    return out


def grid_cells(settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The cells of the grid: planted agreements x supports, then null agreements x supports, per mode."""

    cells = []
    for mode in settings["modes"]:
        for kind, agreements in (("planted", settings["agreements"]), ("null", settings.get("null_agreements", []))):
            for a in agreements:
                for s in settings["supports"]:
                    cells.append({"mode": str(mode), "kind": kind, "agreement": float(a), "support": int(s), "n_datasets": int(settings["n_datasets"])})
    return cells


def _cell_worker(cell: Mapping[str, Any], settings: Mapping[str, Any], rule_policy: Mapping[str, Any], wedd_cfg: Mapping[str, Any],
                 registry: Mapping[str, Any], profile: str) -> dict[str, Any]:
    return gate_power_cell(cell, settings=settings, rule_policy=rule_policy, wedd_cfg=wedd_cfg, registry=registry, profile=profile)


def run_grid(settings: Mapping[str, Any], *, rule_policy: Mapping[str, Any], wedd_cfg: Mapping[str, Any], registry: Mapping[str, Any], profile: str,
             n_jobs: int = 1) -> pd.DataFrame:
    """The local path: every cell of :func:`grid_cells` (results independent of ``n_jobs``)."""

    import functools

    from ...runs import parallel_map

    worker = functools.partial(_cell_worker, settings=dict(settings), rule_policy=dict(rule_policy), wedd_cfg=dict(wedd_cfg),
                               registry=dict(registry), profile=str(profile))
    return pd.DataFrame(parallel_map(worker, grid_cells(settings), n_jobs=n_jobs))


def plan(ctx: Any, cfg: Mapping[str, Any], acfg: Mapping[str, Any]) -> dict[str, Any]:
    return {"inputs": {}, "included": {"synthetic": "none"}, "not_executed": {}}


def compute(ctx: Any, cfg: Mapping[str, Any], acfg: Mapping[str, Any], plan: Mapping[str, Any], out) -> dict[str, Any]:
    from ...runs import atomic_write_json
    from ...stages import _write_table

    study = ctx.study
    profile = str(study.wedd["primary_profile"])
    gate = study.rule_policy["validation_gate"]
    # The "v6" gate is always the frozen one of configs/rule_policy.yaml (never a copy that could drift).
    settings = {**dict(acfg), "gates": {**dict(acfg["gates"]), "v6": {"min_matched": int(gate["min_pooled_oof_matched_patients"]),
                                                                     "wilson_lower_min": float(gate["agreement_wilson_lower_min"])}}}
    table = run_grid(settings, rule_policy=study.rule_policy, wedd_cfg=study.wedd, registry=study.registry, profile=profile, n_jobs=ctx.jobs)
    _write_table(table, out / "cells.parquet", csv=True)
    atomic_write_json(out / "summary.json", {
        "analysis": "B5", "data": "synthetic only (no cohort data)", "n_cells": int(len(table)), "n_datasets_per_cell": int(acfg["n_datasets"]),
        "seed": int(acfg["seed"]), "v6_gate": {"min_pooled_oof_matched_patients": int(gate["min_pooled_oof_matched_patients"]),
                                              "agreement_wilson_lower_min": float(gate["agreement_wilson_lower_min"])},
        "gates": settings["gates"], "modes": list(acfg["modes"]), "profile": profile,
        "outputs": {"P(true rule passes)": "rows kind == planted, columns pass_<gate>", "false-pass rate": "rows kind == null, columns pass_<gate>"},
        "not_simulated": "eligibility gates (cut stability bootstrap, reconstruction gate): the planted feature is taken as eligible"}, allow_nan=True)
    return {"cells": int(len(table))}
