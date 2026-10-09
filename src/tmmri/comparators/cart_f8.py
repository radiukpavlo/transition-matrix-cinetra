"""Registered comparator F8: a CART surrogate of the audited predictor ``d_model`` (protocol-v6.0.1 amendment item 7).

``fit-cart-surrogate`` (after the v6 freeze, before any M&Ms-2 staging) fits ``DecisionTreeClassifier`` with the
settings of ``configs/cart_f8.yaml`` (depth 3, minimum leaf 5, balanced class weights, seed 20260928) on the acdc_dev
rows: features are the registry rule antecedents of the nested out-of-fold B-hat (``oof/bhat_oof_ACDC_dev.parquet``),
the consequent is the out-of-fold (cross-fitted) ``d_model`` (``formal/d_model_crossfit_ACDC_dev.parquet``); critical-QC
rows, rows with a non-finite feature and rows without a decision are excluded and counted.

Two versions share the tree.  The *full-coverage* surrogate labels every row with available features by its leaf.  The
*selective* surrogate keeps only the leaves that pass the v6 gate on a nested out-of-fold leaf fidelity: within each
v6 outer fold f the tree is refitted (same settings) on the inner nested OOF B-hat of outer-train(f)
(``oof/inner_oof_ACDC_dev_fold<f>.parquet``, the substrate of the v6 gate) and the held-out rows (outer OOF B-hat) are
routed by the final tree to a leaf L and by the fold tree to a label.  A held-out row is a match of L when the fold tree
reproduces L's label for it, and it agrees when L's label equals its out-of-fold ``d_model``; pooled over the folds a
leaf is selected when matches >= 5, fidelity >= 0.90 and the Wilson 95 % lower bound >= 0.60.

The fitted tree, the five fold trees, the per-leaf table, the rule text in physical units and the nested ledger form a
supplementary bundle whose hashes are frozen in ``comparators/cart_f8/FROZEN_F8.json``; it never enters the v6 bundle.
``evaluate-cart-surrogate --split`` applies the frozen tree once per partition (once-only stage) to the recorded raw
reconstruction of the frozen transition, after the recorded ``evaluate:<split>``; it reports coverage, fidelity to
``d_model`` and accuracy against the reference with Wilson intervals (numerator and denominator) for both versions.
Routing mirrors sklearn exactly (float32 feature values, ``x <= threshold`` goes left); a non-finite feature on the path
abstains (``FEATURE_UNAVAILABLE_ABSTAIN``) instead of being imputed, as missing states are never rule literals in v6.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..metrics import wilson_interval
from ..runs import atomic_write_json, code_version, hash_path, sha256_json
from ..secondary_stages import (
    input_hashes,
    lane_config,
    lane_config_digest,
    require_partition_unlocked,
    run_secondary_stage,
)

CONFIG_KEY = "cart_f8"
FIT_STAGE = "fit-cart-surrogate"
EVAL_STAGE = "evaluate-cart-surrogate"
FROZEN_NAME = "FROZEN_F8.json"
TREE_SCHEMA = "tmmri.cart_f8_tree/1"
BUNDLE_SCHEMA = "tmmri.supplementary_bundle/1"
FIT_SECTIONS = ("registry.primary_rulebook_antecedent_candidates", "registry.features", "cohorts.qc_severity", "splits.acdc.acdc_dev.outer_folds")
EVAL_SECTIONS = ("registry.primary_rulebook_antecedent_candidates", "registry.label_tables", "cohorts.qc_severity", "reason_codes")
PREDICTED = "PREDICTED"
QC_ABSTAIN = "QC_ABSTAIN"
UNAVAILABLE = "FEATURE_UNAVAILABLE_ABSTAIN"
NOT_SELECTED = "LEAF_NOT_SELECTED_ABSTAIN"
SELECTED = "SELECTED"


# ----------------------------------------------------------------------------- the tree as data


def estimator_kwargs(cfg: Mapping[str, Any]) -> dict[str, Any]:
    est = dict(cfg["estimator"])
    if str(est.get("kind")) != "sklearn.tree.DecisionTreeClassifier":
        raise ValueError("configs/cart_f8.yaml estimator.kind must be sklearn.tree.DecisionTreeClassifier")
    return {"max_depth": int(est["max_depth"]), "min_samples_leaf": int(est["min_samples_leaf"]), "class_weight": est["class_weight"],
            "random_state": int(est["random_state"]), "criterion": str(est.get("criterion", "gini"))}


def fit_tree(values: pd.DataFrame, decision: pd.Series, cfg: Mapping[str, Any]) -> Any:
    """``DecisionTreeClassifier`` of ``configs/cart_f8.yaml`` on finite ``values`` (rows) against ``decision``."""

    from sklearn.tree import DecisionTreeClassifier

    x = values.to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("fit_tree: every fitted feature value must be finite (rows are filtered before the fit)")
    y = pd.Series(decision).astype(str).to_numpy()
    return DecisionTreeClassifier(**estimator_kwargs(cfg)).fit(x, y)


def tree_payload(clf: Any, features: list[str], units: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The fitted tree as JSON-able data: nodes (feature, threshold, children, counts, class weights, label) and leaves."""

    import sklearn

    t = clf.tree_
    classes = [str(c) for c in clf.classes_]
    depth = {0: 0}
    nodes = []
    for node in range(int(t.node_count)):
        left, right = int(t.children_left[node]), int(t.children_right[node])
        leaf = left == -1
        for child in (left, right):
            if child != -1:
                depth[child] = depth[node] + 1
        value = np.asarray(t.value[node][0], dtype=float)
        nodes.append({"node": node, "depth": depth[node], "is_leaf": leaf,
                      "feature": None if leaf else features[int(t.feature[node])],
                      "threshold": None if leaf else float(t.threshold[node]), "left": left, "right": right,
                      "n_fit": int(t.n_node_samples[node]), "weighted_n_fit": float(t.weighted_n_node_samples[node]),
                      "class_value": [float(v) for v in value], "label": classes[int(np.argmax(value))]})
    payload = {"schema": TREE_SCHEMA, "features": list(features), "classes": classes, "nodes": nodes,
               "leaves": [n["node"] for n in nodes if n["is_leaf"]], "sklearn_version": sklearn.__version__,
               "params": {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in clf.get_params().items()}}
    payload["rules"] = leaf_rules(payload, units or {})
    return payload


def _paths(tree: Mapping[str, Any]) -> dict[int, list[tuple[str, str, float]]]:
    """Root-to-leaf conditions ``(feature, '<=' | '>', threshold)`` of every leaf."""

    nodes = {int(n["node"]): n for n in tree["nodes"]}
    out: dict[int, list[tuple[str, str, float]]] = {}
    stack: list[tuple[int, list[tuple[str, str, float]]]] = [(0, [])]
    while stack:
        node, conds = stack.pop()
        n = nodes[node]
        if n["is_leaf"]:
            out[node] = conds
            continue
        stack.append((int(n["right"]), conds + [(str(n["feature"]), ">", float(n["threshold"]))]))
        stack.append((int(n["left"]), conds + [(str(n["feature"]), "<=", float(n["threshold"]))]))
    return dict(sorted(out.items()))


def _number(value: float) -> str:
    return f"{value:.6g}"


def leaf_rules(tree: Mapping[str, Any], units: Mapping[str, str]) -> list[dict[str, Any]]:
    """Per leaf: the interval ``(low, high]`` of each feature on its path and the rule text in physical units."""

    nodes = {int(n["node"]): n for n in tree["nodes"]}
    rules = []
    for leaf, conds in _paths(tree).items():
        intervals: dict[str, list[float]] = {}
        for feature, op, threshold in conds:
            lo, hi = intervals.setdefault(feature, [-math.inf, math.inf])
            intervals[feature] = [lo, min(hi, threshold)] if op == "<=" else [max(lo, threshold), hi]
        parts = []
        for feature in [f for f in tree["features"] if f in intervals]:
            lo, hi = intervals[feature]
            unit = f" {units[feature]}" if units.get(feature) else ""
            if math.isinf(lo):
                parts.append(f"{feature} <= {_number(hi)}{unit}")
            elif math.isinf(hi):
                parts.append(f"{feature} > {_number(lo)}{unit}")
            else:
                parts.append(f"{_number(lo)}{unit} < {feature} <= {_number(hi)}{unit}")
        label = nodes[leaf]["label"]
        text = f"IF {' AND '.join(parts) if parts else 'TRUE'} THEN d_model = {label}"
        rules.append({"leaf": leaf, "label": label, "conditions": [{"feature": f, "op": op, "threshold": t} for f, op, t in conds],
                      "intervals": {f: {"low": None if math.isinf(v[0]) else v[0], "high": None if math.isinf(v[1]) else v[1],
                                        "closed": "(low, high]", "unit": units.get(f)} for f, v in intervals.items()},
                      "text": text})
    return rules


def route_tree(tree: Mapping[str, Any], values: pd.DataFrame) -> pd.DataFrame:
    """Leaf and label of every row (index kept); a non-finite feature on the path abstains with that feature named.

    Feature values are cast to float32 and ``x <= threshold`` (in float64) goes left, exactly as sklearn's ``apply``."""

    nodes = {int(n["node"]): n for n in tree["nodes"]}
    cols = {f: pd.to_numeric(values[f], errors="coerce").to_numpy(dtype=np.float64).astype(np.float32) if f in values.columns
            else np.full(len(values), np.nan, dtype=np.float32) for f in tree["features"]}
    leaf = np.full(len(values), -1, dtype=int)
    label = np.full(len(values), None, dtype=object)
    missing = np.full(len(values), None, dtype=object)
    for i in range(len(values)):
        node = 0
        while not nodes[node]["is_leaf"]:
            f = str(nodes[node]["feature"])
            v = cols[f][i]
            if not np.isfinite(v):
                missing[i] = f
                break
            # float32 value, compared in float64 against the float64 threshold (sklearn's tree traversal)
            node = int(nodes[node]["left"]) if float(v) <= float(nodes[node]["threshold"]) else int(nodes[node]["right"])
        else:
            leaf[i] = node
            label[i] = nodes[node]["label"]
    return pd.DataFrame({"leaf": pd.array([int(v) if v >= 0 else pd.NA for v in leaf], dtype="Int64"), "label": label,
                         "unavailable_feature": missing}, index=values.index)


# ----------------------------------------------------------------------------- nested leaf fidelity and selection


def proportion(k: int, n: int, level: float = 0.95) -> dict[str, Any]:
    """``k`` of ``n`` with its Wilson interval (NE when ``n`` is 0)."""

    k, n = int(k), int(n)
    if n <= 0:
        return {"k": k, "n": n, "estimate": None, "wilson_lower": None, "wilson_upper": None, "status": "NE", "reason": "EMPTY_DENOMINATOR"}
    lo, hi = wilson_interval(k, n, level)
    return {"k": k, "n": n, "estimate": k / n, "wilson_lower": lo, "wilson_upper": hi, "status": "EVALUATED", "reason": None}


def nested_ledger(final_tree: Mapping[str, Any], fold_trees: Mapping[int, Mapping[str, Any]], heldout: pd.DataFrame,
                  folds: pd.Series, decision: pd.Series) -> pd.DataFrame:
    """One row per held-out fit row: its fold, final leaf and label, the fold tree's label, reproduction and agreement."""

    final = route_tree(final_tree, heldout)
    rows = []
    for f in sorted({int(v) for v in folds}):
        ids = [s for s in heldout.index if int(folds[s]) == f]
        fold_route = route_tree(fold_trees[f], heldout.loc[ids])
        for s in ids:
            leaf, label = final.at[s, "leaf"], final.at[s, "label"]
            fold_label = fold_route.at[s, "label"]
            reproduced = label is not None and fold_label is not None and str(fold_label) == str(label)
            rows.append({"sample_id": s, "outer_fold": f, "final_leaf": leaf, "final_label": label, "fold_label": fold_label,
                         "d_model_oof": str(decision[s]), "reproduced": bool(reproduced),
                         "agree": bool(reproduced and str(label) == str(decision[s]))})
    ledger = pd.DataFrame(rows, columns=["sample_id", "outer_fold", "final_leaf", "final_label", "fold_label", "d_model_oof", "reproduced", "agree"])
    ledger["final_leaf"] = ledger["final_leaf"].astype("Int64")
    return ledger.sort_values("sample_id").reset_index(drop=True)


def leaf_table(final_tree: Mapping[str, Any], fit_values: pd.DataFrame, fit_decision: pd.Series, ledger: pd.DataFrame,
               cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Per leaf: rule text, fit counts, pooled nested OOF matches, agreement, Wilson interval and the selection."""

    sel = cfg["selective"]
    min_n, min_fid, lb_min = int(sel["min_matches"]), float(sel["min_oof_fidelity"]), float(sel["wilson_lower_min"])
    level = float(sel.get("wilson_level", 0.95))
    fit_route = route_tree(final_tree, fit_values)
    rules = {int(r["leaf"]): r for r in final_tree["rules"]}
    rows = []
    for leaf in final_tree["leaves"]:
        rule = rules[int(leaf)]
        in_leaf = fit_route["leaf"].to_numpy(dtype=float) == leaf
        fit_agree = int((fit_decision.astype(str).to_numpy()[in_leaf] == rule["label"]).sum())
        routed = ledger[ledger["final_leaf"].to_numpy(dtype=float) == leaf]
        matched = routed[routed["reproduced"].astype(bool)]
        p = proportion(int(matched["agree"].sum()), len(matched), level)
        reasons = []
        if len(matched) < min_n:
            reasons.append("INELIGIBLE_SUPPORT")
        if p["estimate"] is None or p["estimate"] < min_fid:
            reasons.append("INELIGIBLE_FIDELITY")
        if p["wilson_lower"] is None or p["wilson_lower"] < lb_min:
            reasons.append("INELIGIBLE_WILSON_LOWER")
        rows.append({"leaf": int(leaf), "label": rule["label"], "rule": rule["text"], "n_fit": int(in_leaf.sum()), "fit_agree": fit_agree,
                     "fit_fidelity": fit_agree / int(in_leaf.sum()) if in_leaf.any() else np.nan,
                     "n_heldout_routed": int(len(routed)), "oof_matched": int(len(matched)), "oof_agree": int(matched["agree"].sum()),
                     "oof_fidelity": np.nan if p["estimate"] is None else float(p["estimate"]),
                     "wilson_lower": np.nan if p["wilson_lower"] is None else float(p["wilson_lower"]),
                     "wilson_upper": np.nan if p["wilson_upper"] is None else float(p["wilson_upper"]),
                     "folds_reproduced": int(matched["outer_fold"].nunique()), "selected": not reasons,
                     "selection": SELECTED if not reasons else ";".join(reasons)})
    return pd.DataFrame(rows)


def fit_cart_f8(values: pd.DataFrame, decision: pd.Series, folds: pd.Series, fold_substrates: Mapping[int, pd.DataFrame],
                cfg: Mapping[str, Any], units: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Final tree on ``values`` (finite fit rows, index sample_id) against ``decision``, the fold trees on each fold's
    training substrate (``fold_substrates[f]``, rows of outer-train(f); non-finite rows dropped), the nested ledger on the
    held-out rows of ``values`` and the leaf table with the selective version."""

    features = list(values.columns)
    decision = decision.loc[values.index].astype(str)
    folds = pd.Series(folds).loc[values.index].astype(int)
    final = tree_payload(fit_tree(values, decision, cfg), features, units)
    fold_trees: dict[int, dict[str, Any]] = {}
    fold_rows: dict[int, int] = {}
    for f in sorted(set(folds.tolist())):
        train_ids = [s for s in values.index if int(folds[s]) != f]
        sub = fold_substrates[f].reindex(train_ids)[features].apply(pd.to_numeric, errors="coerce")
        ok = np.isfinite(sub.to_numpy(dtype=np.float64)).all(axis=1)
        fold_trees[f] = tree_payload(fit_tree(sub.loc[ok], decision.loc[sub.index[ok]], cfg), features, units)
        fold_rows[f] = int(ok.sum())
    ledger = nested_ledger(final, fold_trees, values, folds, decision)
    table = leaf_table(final, values, decision, ledger, cfg)
    return {"tree": final, "fold_trees": fold_trees, "fold_fit_rows": fold_rows, "ledger": ledger, "leaves": table}


def version_metrics(label: pd.Series, d_model: pd.Series, reference: pd.Series, route: pd.Series, level: float = 0.95) -> dict[str, Any]:
    """Coverage, fidelity to d_model and accuracy against the reference (Wilson) of one surrogate version."""

    covered = label.notna().to_numpy()
    lab = label.astype(object).to_numpy()
    dm = d_model.astype(object).where(d_model.notna(), None).to_numpy()
    ref = reference.astype(object).where(reference.notna(), None).to_numpy()
    fid = covered & np.array([v is not None for v in dm])
    acc = covered & np.array([v is not None for v in ref])
    both = acc & np.array([v is not None for v in dm])
    return {"n_attempted": int(len(label)), "routes": {str(k): int(v) for k, v in route.value_counts().sort_index().items()},
            "coverage": proportion(int(covered.sum()), len(label), level),
            "fidelity_to_d_model": proportion(int(sum(lab[i] == dm[i] for i in np.flatnonzero(fid))), int(fid.sum()), level),
            "accuracy_vs_reference": proportion(int(sum(lab[i] == ref[i] for i in np.flatnonzero(acc))), int(acc.sum()), level),
            "d_model_accuracy_same_rows": proportion(int(sum(dm[i] == ref[i] for i in np.flatnonzero(both))), int(both.sum()), level)}


# ----------------------------------------------------------------------------- stages


def _units(study: Any) -> dict[str, str]:
    return {str(f["name"]): str(f["unit"]) for f in study.registry.get("features", []) if f.get("unit")}


def _out_dir(run: Any, cfg: Mapping[str, Any]) -> Path:
    return run.sub(str(cfg.get("output_dir", "comparators/cart_f8")))


def _bundle_files(out: Path) -> list[Path]:
    return [out / "tree.json", out / "fold_trees.json", out / "rules.json", out / "leaves.parquet", out / "leaves.csv",
            out / "nested_oof_ledger.parquet", out / "fit_rows.parquet", out / "model.joblib"]


def _mms2_staged(ctx: Any) -> bool:
    """Whether M&Ms-2 has been extracted or staged: an extraction record (``extraction_MMS2.json`` or the receipt of the
    source area; SPECIFICATION.md 14.6 dates every freeze against the extraction) or a staging manifest.  F8 must be
    frozen before either."""

    from ..ingest.staging import manifest_path_for
    from ..stages import mms2_extraction_times

    try:
        path = manifest_path_for(ctx.manifests_dir, "mms2")
    except (KeyError, ValueError):
        path = Path(ctx.manifests_dir) / "staging_MMS2.json"
    return path.is_file() or any(Path(ctx.manifests_dir).glob("staging_MMS2*.json")) or bool(mms2_extraction_times(ctx)["exists"])


def fit_cart_surrogate(ctx: Any):
    """``fit-cart-surrogate``: fit F8 on acdc_dev and freeze the supplementary bundle (``FROZEN_F8.json``)."""

    import joblib

    from ..stages import (BundleError, _check_recorded_output, _contracts, _split_manifest, _write_table, critical_qc_mask, inner_oof_path,
                          verify_frozen_bundle)

    run = ctx.require_run(FIT_STAGE)
    if not run.frozen_path.is_file():
        raise BundleError(f"{FIT_STAGE}: run {run.run_id} has no frozen v6 analysis bundle; F8 is fitted after the freeze "
                          "(run freeze-analysis-bundle first)")
    frozen = verify_frozen_bundle(ctx)
    study = ctx.study
    cfg = lane_config(study, CONFIG_KEY)
    out = _out_dir(run, cfg)
    p = run.path
    oof_path, d_path = p / "oof" / "bhat_oof_ACDC_dev.parquet", p / "formal" / "d_model_crossfit_ACDC_dev.parquet"
    b_path, split_path = p / "semantics" / "B_ACDC.parquet", _contracts(ctx) / "split_manifest.parquet"
    n_folds = int(study.splits["acdc"]["acdc_dev"]["outer_folds"]["n_folds"])
    inner_paths = {f: inner_oof_path(run, f) for f in range(1, n_folds + 1)}
    for path, stage in [(oof_path, "fit-transition"), (d_path, "fit-predictor"), *[(q, "gate-rules") for q in inner_paths.values()]]:
        if path.is_file():
            _check_recorded_output(run, run.read_record(stage), path, FIT_STAGE)
    files = _bundle_files(out)
    outputs = files + [out / FROZEN_NAME]

    def compute() -> dict[str, Any]:
        if _mms2_staged(ctx):
            raise BundleError(f"{FIT_STAGE}: an M&Ms-2 extraction record or staging manifest exists; F8 must be frozen before the "
                              "M&Ms-2 extraction and before any M&Ms-2 staging (protocol-v6.0.1 amendment item 7; SPECIFICATION.md 14.6); "
                              "a new run id is required")
        oof = pd.read_parquet(oof_path).set_index("sample_id")
        split = _split_manifest(ctx)
        dev = split[split["partition"] == "acdc_dev"].set_index("sample_id")["outer_fold"]
        b = pd.read_parquet(b_path)
        critical = set(b.loc[critical_qc_mask(b, study).to_numpy(dtype=bool), "sample_id"].astype(str))
        decision = pd.read_parquet(d_path).set_index("sample_id")["d_model_crossfit"]
        candidates = [c for c in study.rule_candidates() if c in oof.columns]
        ids = sorted(set(dev.index.astype(str)) & set(oof.index.astype(str)))
        counts = {"n_acdc_dev_with_oof": len(ids), "n_excluded_critical_qc": sum(s in critical for s in ids)}
        ids = [s for s in ids if s not in critical]
        labelled = [s for s in ids if s in decision.index and decision[s] is not None and pd.notna(decision[s])]
        counts["n_excluded_no_d_model"] = len(ids) - len(labelled)
        x = oof.loc[labelled, candidates].apply(pd.to_numeric, errors="coerce")
        drop = bool(cfg["fit"].get("drop_features_without_finite_fit_values", True))
        dropped = [c for c in candidates if drop and not np.isfinite(x[c].to_numpy(dtype=np.float64)).any()]
        features = [c for c in candidates if c not in dropped]
        if not features:
            raise ValueError(f"{FIT_STAGE}: no rule-antecedent column of the OOF B-hat has a finite value")
        finite = np.isfinite(x[features].to_numpy(dtype=np.float64)).all(axis=1)
        counts["n_excluded_nonfinite_feature"] = int((~finite).sum())
        values = x.loc[finite, features]
        counts["n_fit_rows"] = int(len(values))
        substrates = {f: pd.read_parquet(q).set_index("sample_id") for f, q in inner_paths.items()}
        result = fit_cart_f8(values, decision, dev.astype(int), substrates, cfg, _units(study))
        tree, leaves = result["tree"], result["leaves"]
        atomic_write_json(files[0], tree, allow_nan=True)
        atomic_write_json(files[1], {"schema": "tmmri.cart_f8_fold_trees/1", "fold_fit_rows": {str(k): v for k, v in result["fold_fit_rows"].items()},
                                     "fold_trees": {str(f): t for f, t in result["fold_trees"].items()}}, allow_nan=True)
        selected = set(leaves.loc[leaves["selected"].astype(bool), "leaf"].astype(int))
        stats = leaves.set_index("leaf")
        atomic_write_json(files[2], {
            "schema": "tmmri.cart_f8_rules/1", "comparator_id": cfg["comparator_id"], "units": "physical units of the registry (B-hat units)",
            "features": features, "dropped_features": dropped, "selective_gate": cfg["selective"],
            "rules": [{**r, "selected": int(r["leaf"]) in selected, "n_fit": int(stats.at[r["leaf"], "n_fit"]),
                       "oof_matched": int(stats.at[r["leaf"], "oof_matched"]), "oof_agree": int(stats.at[r["leaf"], "oof_agree"]),
                       "oof_fidelity": stats.at[r["leaf"], "oof_fidelity"], "wilson_lower": stats.at[r["leaf"], "wilson_lower"],
                       "selection": stats.at[r["leaf"], "selection"]} for r in tree["rules"]]}, allow_nan=True)
        _write_table(leaves, files[3], csv=True)
        _write_table(result["ledger"], files[5])
        fit_rows = values.reset_index(names="sample_id").assign(outer_fold=lambda d: dev.loc[d["sample_id"]].to_numpy(dtype=int),
                                                                d_model_oof=lambda d: decision.loc[d["sample_id"]].astype(str).to_numpy())
        _write_table(fit_rows, files[6])
        joblib.dump(fit_tree(values, decision.loc[values.index], cfg), files[7])
        bundle = {f.relative_to(run.path).as_posix(): hash_path(f) for f in files}   # run-root relative, as FROZEN.json
        atomic_write_json(out / FROZEN_NAME, {
            "schema": BUNDLE_SCHEMA, "comparator_id": cfg["comparator_id"], "role": cfg["role"], "run_id": run.run_id,
            "bundle": bundle, "bundle_sha256": sha256_json(bundle),
            "config": {"file": f"configs/{CONFIG_KEY}.yaml", "sha256": lane_config_digest(study, CONFIG_KEY)},
            "v6_bundle_sha256": frozen["bundle_sha256"], "inputs": input_hashes(run, {"oof_bhat": oof_path, "d_model_oof": d_path, "B": b_path,
                                                                                          "split_manifest": split_path,
                                                                                          **{f"inner_oof_fold{f}": q for f, q in inner_paths.items()}}),
            "counts": counts, "features": features, "dropped_features": dropped, "n_leaves": len(tree["leaves"]),
            "n_selected_leaves": len(selected), "mms2_staged_at_freeze": False,
            "evaluation_status": {k: v["status"] for k, v in cfg["evaluation"]["partitions"].items()},
            "frozen_before": "any M&Ms-2 staging (protocol-v6.0.1 amendment item 7); never part of the v6 analysis bundle",
            "code": code_version()}, allow_nan=True)
        return {**counts, "features": len(features), "leaves": len(tree["leaves"]), "selected_leaves": len(selected)}

    inputs = [run.frozen_path, oof_path, d_path, b_path, split_path, *inner_paths.values()]
    return run_secondary_stage(ctx, FIT_STAGE, compute, inputs, outputs, sections=FIT_SECTIONS, lane_keys=[CONFIG_KEY],
                               params={"comparator": "F8"})


def verify_f8_bundle(run: Any, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The frozen F8 supplementary bundle, refusing when it is absent or a file changed after the freeze."""

    from ..stages import BundleError

    path = _out_dir(run, cfg) / FROZEN_NAME
    if not path.is_file():
        raise BundleError(f"{EVAL_STAGE}: {run.portable(path)} missing; run {FIT_STAGE} first")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    changed = [rel for rel, digest in frozen["bundle"].items() if hash_path(run.path / rel) != digest]
    if changed or sha256_json(frozen["bundle"]) != frozen["bundle_sha256"]:
        raise BundleError(f"{EVAL_STAGE}: the frozen F8 bundle changed after its freeze: {changed[:5]}; a new run id is required")
    return frozen


def evaluate_cart_surrogate(ctx: Any, split: str):
    """``evaluate-cart-surrogate --split``: the frozen F8 on one recorded evaluation partition (once-only stage)."""

    from ..formal_features import formal_descriptor_names
    from ..rule_pipeline import qc_route_frame
    from ..stages import (_check_recorded_output, _cluster_ids, _contracts, _require_evaluated, _write_table, mms2_freeze_order, reference_labels,
                          split_manifest_for, verify_frozen_bundle)

    cfg = lane_config(ctx.study, CONFIG_KEY)
    partitions = dict(cfg["evaluation"]["partitions"])
    if split not in partitions:
        raise ValueError(f"{EVAL_STAGE} --split must be one of {sorted(partitions)} (configs/cart_f8.yaml evaluation.partitions)")
    ds = str(partitions[split]["dataset"])
    require_partition_unlocked(ctx, split, f"{EVAL_STAGE} --split {split}")
    run = ctx.require_run(EVAL_STAGE)
    frozen = verify_frozen_bundle(ctx)
    f8 = verify_f8_bundle(run, cfg)
    record = _require_evaluated(run, f"evaluate:{split}", EVAL_STAGE)
    p = run.path
    out = _out_dir(run, cfg)
    bhat_path, d_path = p / "reconstructed" / f"bhat_{split}.parquet", p / "inference" / f"d_model_{split}.parquet"
    a_path, b_path, cohort_path = p / "formal" / f"A_{ds}.parquet", p / "semantics" / f"B_{ds}.parquet", _contracts(ctx) / f"cohort_{ds}.parquet"
    split_path = _contracts(ctx) / ("split_manifest_MMS2.parquet" if ds == "MMS2" else "split_manifest.parquet")
    for path in (bhat_path, d_path):
        _check_recorded_output(run, record, path, EVAL_STAGE)
    edir = run.sub(f"{cfg.get('output_dir', 'comparators/cart_f8')}/evaluation")
    outputs = [edir / f"predictions_{split}.parquet", edir / f"leaves_{split}.parquet", edir / f"metrics_{split}.json"]
    status = str(partitions[split]["status"])
    order = mms2_freeze_order(ctx, ["F8"]) if ds == "MMS2" else None       # SPECIFICATION.md 14.6: late freeze -> post hoc
    if order is not None and not order["before_extraction"]:
        status = "post_hoc"
    level = float(cfg["selective"].get("wilson_level", 0.95))

    def compute() -> dict[str, Any]:
        tree = json.loads((out / "tree.json").read_text(encoding="utf-8"))
        leaves = pd.read_parquet(out / "leaves.parquet")
        selected = set(leaves.loc[leaves["selected"].astype(bool), "leaf"].astype(int))
        bhat = pd.read_parquet(bhat_path).set_index("sample_id")
        ids = sorted(bhat.index.astype(str))
        a = pd.read_parquet(a_path).set_index("sample_id").reindex(ids)
        finite = np.isfinite(a[formal_descriptor_names()].to_numpy(dtype=np.float64)).all(axis=1)
        b = pd.read_parquet(b_path).set_index("sample_id").reindex(ids)
        qc = qc_route_frame(b, finite, ctx.study).set_index("sample_id")
        routed = route_tree(tree, bhat.loc[ids])
        d_model = pd.read_parquet(d_path).set_index("sample_id")["d_model"].reindex(ids).astype(object)
        cohort = pd.read_parquet(cohort_path).set_index("sample_id")
        reference = reference_labels(cohort, "ACDC" if ds == "ACDC" else "MMS").reindex(ids)
        qc_abstain = qc["qc_abstain"].reindex(ids).fillna(True).astype(bool).to_numpy()
        leaf = routed["leaf"]
        route_full = np.where(qc_abstain, QC_ABSTAIN, np.where(leaf.isna().to_numpy(), UNAVAILABLE, PREDICTED))
        label_full = pd.Series(np.where(route_full == PREDICTED, routed["label"].to_numpy(dtype=object), None), index=ids, dtype=object)
        in_selected = np.array([pd.notna(v) and int(v) in selected for v in leaf], dtype=bool)
        route_sel = np.where((route_full == PREDICTED) & ~in_selected, NOT_SELECTED, route_full)
        label_sel = pd.Series(np.where(route_sel == PREDICTED, label_full.to_numpy(dtype=object), None), index=ids, dtype=object)
        frame = pd.DataFrame({"sample_id": ids, "leaf": leaf.to_numpy(), "route_full": route_full, "label_full": label_full.to_numpy(),
                              "route_selective": route_sel, "label_selective": label_sel.to_numpy(), "leaf_selected": in_selected,
                              "unavailable_feature": routed["unavailable_feature"].to_numpy(), "qc_reason_code": qc["qc_reason_code"].reindex(ids).to_numpy(),
                              "d_model": d_model.to_numpy(), "reference": reference.to_numpy(dtype=object)})
        _write_table(frame, outputs[0])
        per_leaf = []
        for r in tree["rules"]:
            rows = (frame["route_full"] == PREDICTED).to_numpy() & (frame["leaf"].to_numpy(dtype=float) == int(r["leaf"]))
            sub = frame.loc[rows]
            per_leaf.append({"leaf": int(r["leaf"]), "label": r["label"], "rule": r["text"], "selected": int(r["leaf"]) in selected, "n": int(rows.sum()),
                             "fidelity_agree": int((sub["label_full"] == sub["d_model"]).sum()), "n_d_model": int(sub["d_model"].notna().sum()),
                             "reference_agree": int((sub["label_full"] == sub["reference"]).sum()), "n_reference": int(sub["reference"].notna().sum())})
        _write_table(pd.DataFrame(per_leaf), outputs[1])
        clusters = _cluster_ids(split_manifest_for(ctx, split), ids)
        mms2_ok = not bool(f8.get("mms2_staged_at_freeze", False)) and (order is None or order["before_extraction"])
        atomic_write_json(outputs[2], {
            "schema": "tmmri.cart_f8_evaluation/1", "comparator_id": cfg["comparator_id"], "split": split, "dataset": ds,
            "status": status, "prospective": status == "prospective" and mms2_ok, "freeze_order": order,
            "n_cases": len(ids), "n_clusters": int(clusters.nunique()),
            "reference": "acdc_class" if ds == "ACDC" else "crosswalk_same_definition_only",
            "substrate": f"reconstructed/bhat_{split}.parquet (raw reconstruction of the frozen transition; evaluate:{split})",
            "f8_bundle_sha256": f8["bundle_sha256"], "v6_bundle_sha256": frozen["bundle_sha256"], "rules": "comparators/cart_f8/rules.json",
            "selected_leaves": [{"leaf": r["leaf"], "rule": r["text"]} for r in tree["rules"] if int(r["leaf"]) in selected],
            "interval": f"wilson_{int(round(level * 100))}",
            "versions": {"full_coverage": version_metrics(label_full, d_model, reference, pd.Series(route_full), level),
                         "selective": version_metrics(label_sel, d_model, reference, pd.Series(route_sel), level)}}, allow_nan=True)
        return {"rows": len(ids), "split": split, "status": status}

    inputs = [run.frozen_path, out / FROZEN_NAME, run.record_path(f"evaluate:{split}"), bhat_path, d_path, a_path, b_path, cohort_path, split_path]
    return run_secondary_stage(ctx, f"{EVAL_STAGE}:{split}", compute, inputs, outputs, sections=EVAL_SECTIONS, lane_keys=[CONFIG_KEY],
                               params={"split": split, "status": status}, status="EVALUATED")
