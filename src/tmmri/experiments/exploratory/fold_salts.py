"""B6 fold variance (exploratory, post hoc; acdc_dev only, never a test or external row).

The v6 hash rule of ``configs/splits.yaml`` is re-applied with ``n_salts`` alternative salts (outer and inner folds);
for each, the development pipeline of the primary ``d_model`` rulebook is refitted exactly as frozen: the transition
(lambda per target by CV over the alternative outer folds), the nested OOF B-hat (inner folds of the alternative salt),
WEDD cuts and their stability gate, the eligibility gates, VPRS discovery and the cross-fitted gate over the
alternative outer folds with the nested inner OOF substrate (``d_model`` refitted without each fold).  The output is the
spread of lambda, cuts, eligible features and gated rules across salts, beside the recorded primary values.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd

OUTPUTS = ("lambdas.parquet", "lambdas.csv", "cuts.parquet", "cuts.csv", "eligible_features.parquet", "eligible_features.csv", "rules.parquet",
           "rules.csv", "summary.json")


def salt_splits(splits_cfg: Mapping[str, Any], acfg: Mapping[str, Any], k: int) -> dict[str, Any]:
    cfg = copy.deepcopy(dict(splits_cfg))
    dev = cfg["acdc"]["acdc_dev"]
    dev["outer_folds"]["hash_salt"] = str(acfg["outer_salt_template"]).format(k=k)
    dev["inner_folds"]["hash_salt"] = str(acfg["inner_salt_template"]).format(k=k)
    return cfg


def refit_salt(ctx: Any, k: int, acfg: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, Any]:
    """The d_model development pipeline under alternative salt ``k`` (inputs in ``base``; acdc_dev rows only)."""

    from ...classification import fit_formal_predictor
    from ...rough_sets import ELIGIBLE
    from ...rule_pipeline import NestedGateSubstrate, acdc_inner_labels, fit_rulebook
    from ...splits import acdc_all_inner_folds_v6, acdc_outer_folds_v6
    from ...stages import _cluster_ids, _mean_oof, predict_labels
    from ...transition_fit import fit_transition_v6, nested_oof_transition

    study = ctx.study
    splits_cfg = salt_splits(study.splits, acfg, k)
    outer = acdc_outer_folds_v6(base["cohort"], splits_cfg).set_index("sample_id")
    inner = acdc_all_inner_folds_v6(outer.reset_index(), splits_cfg)
    row_ids, a, b = base["row_ids"], base["A"], base["B"]
    folds = outer.loc[row_ids, "outer_fold"].to_numpy(dtype=int)
    targets = study.transition_targets()
    labels_inner = inner.set_index(["outer_fold_heldout", "sample_id"])["inner_fold"]

    def inner_fn(f: int, train_positions: np.ndarray) -> np.ndarray:
        train = row_ids[np.asarray(train_positions, dtype=int)]
        return labels_inner.loc[int(f)].reindex(train).to_numpy(dtype=int)

    model = fit_transition_v6(a, b, targets, study.transition, study.registry, folds)
    oof = nested_oof_transition(a, b, targets, study.transition, study.registry, folds, inner_fn)
    oof.index = row_ids
    b_idx = b.set_index(pd.Index(row_ids))
    mean_oof = _mean_oof(b_idx, targets, folds)
    candidates = [c for c in study.rule_candidates() if c in oof.columns]
    decision = base["d_model"]
    gate_ids = np.array([s for s in row_ids if s in decision.index], dtype=object)
    subset = np.flatnonzero(pd.Index(row_ids).isin(gate_ids))
    labels_all = base["labels"]
    pcfg = study.formal["audited_predictor"]
    cache: dict[tuple[int, ...], Any] = {}

    def d_model_fn(train: np.ndarray, target: np.ndarray) -> np.ndarray:
        key = tuple(int(i) for i in train)
        if key not in cache:
            cache[key] = fit_formal_predictor(a[subset[train]], pd.Series(labels_all[subset[train]]), pcfg)
        return predict_labels(cache[key], a[subset[target]])

    substrate_fn = NestedGateSubstrate(a, b, targets, study.transition, study.registry, folds, acdc_inner_labels(inner, row_ids), oof, candidates,
                                       row_ids, subset=subset)
    anchors = pd.Series([s.split(":", 1)[1] for s in gate_ids], index=gate_ids)
    fit = fit_rulebook(rulebook_id="d_model", values=oof.loc[gate_ids, candidates], decision=decision.loc[gate_ids], anchors=anchors,
                       folds=folds[subset], profile=str(study.wedd["primary_profile"]), study=study, decision_fn=d_model_fn, substrate_fn=substrate_fn,
                       reference=base["B_all"].loc[gate_ids], oof=oof.loc[gate_ids], mean_oof=mean_oof.loc[gate_ids],
                       stability_replicates=int(acfg["stability_replicates"]), n_jobs=ctx.jobs, availability=base["availability"],
                       clusters=_cluster_ids(base["split"], gate_ids))
    rules = [{"rule_id": r.rule_id, "identity": r.identity, "signature": f"{'+'.join(sorted(r.features))}->{r.consequent}",
              "features": "+".join(r.features), "consequent": r.consequent, "eligible": r.eligibility == ELIGIBLE} for r in fit["gated"].rules]
    return {"lambdas": dict(model.lambdas), "edge_flags": dict(model.edge_flags), "cuts": {f: list(v) for f, v in fit["cuts"].cuts().items()},
            "eligible_features": list(fit["attributes"]), "rules": rules, "n_fit_rows": int(len(gate_ids))}


def _primary(run: Any, study: Any) -> dict[str, Any]:
    """The recorded primary values (salt of configs/splits.yaml) for reference."""

    from ...rough_sets import Rule
    from ...wedd import CutSet

    profile = str(study.wedd["primary_profile"])
    model = json.loads((run.path / "transitions" / "transition_model.json").read_text(encoding="utf-8"))
    cuts = CutSet.from_dict(json.loads((run.path / "thresholds" / f"d_model_{profile}.json").read_text(encoding="utf-8"))).cuts()
    eligibility = json.loads((run.path / "rules" / "semantic_eligibility.json").read_text(encoding="utf-8"))
    book = run.path / "rules" / "d_model" / "rulebook.json"
    records = json.loads(book.read_text(encoding="utf-8")).get("records", []) if book.is_file() else []
    rules = []
    for rec in records:
        r = Rule.from_record(rec)
        rules.append({"rule_id": r.rule_id, "identity": r.identity, "signature": f"{'+'.join(sorted(r.features))}->{r.consequent}",
                      "features": "+".join(r.features), "consequent": r.consequent, "eligible": True})
    return {"lambdas": {k: float(v) for k, v in model["lambdas"].items()}, "edge_flags": dict(model.get("edge_flags", {})),
            "cuts": {f: list(v) for f, v in cuts.items()}, "eligible_features": list(eligibility["rulebooks"]["d_model"]["eligible_features"]),
            "rules": rules}


def plan(ctx: Any, cfg: Mapping[str, Any], acfg: Mapping[str, Any]) -> dict[str, Any]:
    from ...stages import _contracts
    from .common import check_recorded

    run = ctx.require_run("run-exploratory")
    p = run.path
    profile = str(ctx.study.wedd["primary_profile"])
    check_recorded(run, "fit-predictor", [p / "formal" / "d_model_ACDC_dev.parquet"], "run-exploratory B6")
    inputs = {"A_ACDC": p / "formal" / "A_ACDC.parquet", "B_ACDC": p / "semantics" / "B_ACDC.parquet", "d_model": p / "formal" / "d_model_ACDC_dev.parquet",
              "cohort_ACDC": _contracts(ctx) / "cohort_ACDC.parquet", "split_manifest": _contracts(ctx) / "split_manifest.parquet",
              "transition_model": p / "transitions" / "transition_model.json", "thresholds_d_model": p / "thresholds" / f"d_model_{profile}.json",
              "semantic_eligibility": p / "rules" / "semantic_eligibility.json", "rulebook_or_empty_d_model": p / "rules" / "d_model" / "rulebook_or_empty.json"}
    return {"inputs": inputs, "included": {"acdc_dev": "ACDC"}, "not_executed": {}}


def _spread(values: list[float]) -> dict[str, Any]:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if not v.size:
        return {"n": 0}
    return {"n": int(v.size), "min": float(v.min()), "q25": float(np.quantile(v, 0.25)), "median": float(np.median(v)),
            "q75": float(np.quantile(v, 0.75)), "max": float(v.max())}


def compute(ctx: Any, cfg: Mapping[str, Any], acfg: Mapping[str, Any], plan: Mapping[str, Any], out) -> dict[str, Any]:
    from ...runs import atomic_write_json
    from ...stages import _cohort, _dev_matrices, _split_manifest, _write_table

    run = ctx.require_run("run-exploratory")
    study = ctx.study
    ids, a, b, _ = _dev_matrices(ctx)
    row_ids = ids["sample_id"].astype(str).to_numpy()
    cohort = _cohort(ctx, "ACDC")
    split = _split_manifest(ctx)
    b_all = pd.read_parquet(run.path / "semantics" / "B_ACDC.parquet").set_index("sample_id")
    dev_ids = [s for s in split.loc[split["partition"] == "acdc_dev", "sample_id"] if s in b_all.index]
    base = {"cohort": cohort, "row_ids": row_ids, "A": a, "B": b, "B_all": b_all, "availability": b_all.loc[dev_ids], "split": split,
            "labels": cohort.set_index("sample_id").loc[row_ids, "acdc_class"].astype(str).to_numpy(),
            "d_model": pd.read_parquet(run.path / "formal" / "d_model_ACDC_dev.parquet").set_index("sample_id")["d_model"].dropna().astype(str)}
    results = {"primary": _primary(run, study)}
    for k in range(1, int(acfg["n_salts"]) + 1):
        results[f"salt{k:02d}"] = refit_salt(ctx, k, acfg, base)
    lam_rows, cut_rows, elig_rows, rule_rows = [], [], [], []
    candidates = study.rule_candidates()
    for salt, r in results.items():
        for t, v in sorted(r["lambdas"].items()):
            lam_rows.append({"salt": salt, "target": t, "lambda": float(v), "log10_lambda": float(np.log10(v)) if v > 0 else np.nan,
                             "lambda_at_grid_edge": bool(r["edge_flags"].get(t, False))})
        for f, cuts in sorted(r["cuts"].items()):
            for i, c in enumerate(cuts):
                cut_rows.append({"salt": salt, "feature": f, "n_cuts": len(cuts), "cut_index": i, "cut": float(c)})
        for f in candidates:
            elig_rows.append({"salt": salt, "feature": f, "eligible": f in set(r["eligible_features"])})
        for rule in r["rules"]:
            rule_rows.append({"salt": salt, **rule})
    lambdas = pd.DataFrame(lam_rows, columns=["salt", "target", "lambda", "log10_lambda", "lambda_at_grid_edge"])
    cuts = pd.DataFrame(cut_rows, columns=["salt", "feature", "n_cuts", "cut_index", "cut"])
    elig = pd.DataFrame(elig_rows, columns=["salt", "feature", "eligible"])
    rules = pd.DataFrame(rule_rows, columns=["salt", "rule_id", "identity", "signature", "features", "consequent", "eligible"])
    alt = [s for s in results if s != "primary"]
    alt_rules = rules[rules["salt"].isin(alt) & rules["eligible"].astype(bool)]
    primary_signatures = {r["signature"] for r in results["primary"]["rules"]}
    summary = {
        "analysis": "B6", "n_salts": len(alt), "outer_salt_template": acfg["outer_salt_template"], "inner_salt_template": acfg["inner_salt_template"],
        "stability_replicates": int(acfg["stability_replicates"]),
        "lambda_log10_spread": {t: _spread(g.loc[g["salt"].isin(alt), "log10_lambda"].tolist()) for t, g in lambdas.groupby("target", sort=True)},
        "lambda_primary": results["primary"]["lambdas"],
        "cuts": {f: {"n_cuts_by_salt": {str(n): int(c) for n, c in g[g["salt"].isin(alt)].drop_duplicates("salt")["n_cuts"].value_counts().sort_index().items()},
                     "cut_spread_by_index": {str(i): _spread(h.loc[h["salt"].isin(alt), "cut"].tolist()) for i, h in g.groupby("cut_index", sort=True)},
                     "primary": results["primary"]["cuts"].get(f, [])} for f, g in cuts.groupby("feature", sort=True)},
        "eligible_feature_frequency": {f: float(g.loc[g["salt"].isin(alt), "eligible"].mean()) if len(alt) else None for f, g in elig.groupby("feature", sort=True)},
        "eligible_features_primary": results["primary"]["eligible_features"],
        "eligible_rules_per_salt": {s: int(((rules["salt"] == s) & rules["eligible"].astype(bool)).sum()) for s in alt},
        "rule_signature_recurrence": {sig: int(g["salt"].nunique()) / len(alt) for sig, g in alt_rules.groupby("signature", sort=True)} if len(alt) else {},
        "primary_rule_signatures": sorted(primary_signatures),
        "primary_rule_signature_recurrence": {sig: (int(alt_rules.loc[alt_rules["signature"] == sig, "salt"].nunique()) / len(alt) if len(alt) else None)
                                              for sig in sorted(primary_signatures)}}
    _write_table(lambdas, out / "lambdas.parquet", csv=True)
    _write_table(cuts, out / "cuts.parquet", csv=True)
    _write_table(elig, out / "eligible_features.parquet", csv=True)
    _write_table(rules, out / "rules.parquet", csv=True)
    atomic_write_json(out / "summary.json", summary, allow_nan=True)
    return {"salts": len(alt)}
