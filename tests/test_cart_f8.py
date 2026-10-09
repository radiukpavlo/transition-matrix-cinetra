"""Registered comparator F8, the CART surrogate (protocol-v6.0.1 amendment item 7): tree routing against sklearn, rule
text in physical units, nested leaf fidelity and the selective gate, Wilson summaries.  Synthetic inputs only
(algorithm controls); the stages are exercised on a synthetic run in test_exploratory.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tmmri.comparators.cart_f8 import (
    NOT_SELECTED,
    estimator_kwargs,
    fit_cart_f8,
    fit_tree,
    leaf_table,
    nested_ledger,
    proportion,
    route_tree,
    tree_payload,
    version_metrics,
)
from tmmri.metrics import wilson_interval
from tmmri.secondary_stages import lane_config

FEATURES = ["lv_edv_ml", "rv_edv_ml", "lv_ef_percent"]
UNITS = {"lv_edv_ml": "mL", "rv_edv_ml": "mL", "lv_ef_percent": "%"}


@pytest.fixture(scope="module")
def cfg():
    from tmmri.config import load_study

    return lane_config(load_study(), "cart_f8")


def _data(n: int = 120, seed: int = 20260613) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({"lv_edv_ml": rng.uniform(60, 260, n), "rv_edv_ml": rng.uniform(60, 260, n), "lv_ef_percent": rng.uniform(10, 75, n)},
                     index=[f"ACDC:patient{i:03d}" for i in range(n)])
    y = np.where(x["lv_edv_ml"] > 200, "DCM", np.where(x["lv_ef_percent"] < 30, "MINF", np.where(x["rv_edv_ml"] > 220, "RV", "NOR")))
    flip = rng.random(n) < 0.08
    y = np.where(flip, "HCM", y)
    folds = pd.Series((np.arange(n) % 5) + 1, index=x.index)
    return x, pd.Series(y, index=x.index), folds


def test_settings_are_the_registered_ones(cfg):
    assert estimator_kwargs(cfg) == {"max_depth": 3, "min_samples_leaf": 5, "class_weight": "balanced", "random_state": 20260928, "criterion": "gini"}
    sel = cfg["selective"]
    assert (sel["min_oof_fidelity"], sel["min_matches"], sel["wilson_lower_min"]) == (0.90, 5, 0.60)
    assert cfg["evaluation"]["partitions"]["mms2_external"]["status"] == "prospective"
    assert {cfg["evaluation"]["partitions"][s]["status"] for s in ("acdc_test", "mms_external")} == {"post_hoc"}


def test_route_tree_equals_sklearn_and_rules_are_physical_intervals(cfg):
    x, y, _ = _data()
    clf = fit_tree(x, y, cfg)
    tree = tree_payload(clf, FEATURES, UNITS)
    assert max(n["depth"] for n in tree["nodes"]) <= 3 and all(n["n_fit"] >= 5 for n in tree["nodes"] if n["is_leaf"])
    rng = np.random.default_rng(1)
    probe = pd.DataFrame({"lv_edv_ml": rng.uniform(40, 300, 400), "rv_edv_ml": rng.uniform(40, 300, 400), "lv_ef_percent": rng.uniform(0, 90, 400)})
    thresholds = [(n["feature"], n["threshold"]) for n in tree["nodes"] if not n["is_leaf"]]
    for i, (f, t) in enumerate(thresholds):          # values on and next to every threshold
        probe.loc[i, f] = t
        probe.loc[len(thresholds) + i, f] = np.nextafter(np.float32(t), np.float32(np.inf))
    routed = route_tree(tree, probe)
    assert routed["leaf"].astype(int).tolist() == clf.apply(probe.to_numpy()).tolist()
    assert routed["label"].tolist() == clf.predict(probe.to_numpy()).tolist()
    # Every fit row satisfies the physical interval of its leaf; the text names the unit.
    rules = {r["leaf"]: r for r in tree["rules"]}
    fit_route = route_tree(tree, x)
    for sid, leaf in fit_route["leaf"].items():
        for f, iv in rules[int(leaf)]["intervals"].items():
            assert (iv["low"] is None or x.at[sid, f] > iv["low"]) and (iv["high"] is None or x.at[sid, f] <= iv["high"])
    assert all(r["text"].startswith("IF ") and "THEN d_model = " in r["text"] for r in tree["rules"])
    assert any(" mL" in r["text"] for r in tree["rules"])
    # A non-finite feature on the path abstains with the feature named; a feature off the path does not matter.
    root = tree["nodes"][0]["feature"]
    gap = x.iloc[[0]].copy()
    gap[root] = np.nan
    assert route_tree(tree, gap)["leaf"].isna().all() and route_tree(tree, gap)["unavailable_feature"].iloc[0] == root


def test_nested_ledger_and_selective_gate_by_hand(cfg):
    x, y, folds = _data()
    rng = np.random.default_rng(3)
    substrates = {f: x.mul(1.0 + rng.normal(0, 0.01, x.shape)) for f in range(1, 6)}     # inner nested OOF stand-ins
    out = fit_cart_f8(x, y, folds, substrates, cfg, UNITS)
    ledger, leaves = out["ledger"], out["leaves"]
    assert len(ledger) == len(x) and set(ledger["outer_fold"]) == {1, 2, 3, 4, 5} and out["fold_fit_rows"] == {f: int((folds != f).sum()) for f in range(1, 6)}
    # The ledger equals routing by hand: final leaf by the final tree, fold label by the fold tree fitted without the fold.
    final = route_tree(out["tree"], x)
    for f in range(1, 6):
        held = folds.index[folds == f]
        fold_route = route_tree(out["fold_trees"][f], x.loc[held])
        sub = ledger.set_index("sample_id").loc[held]
        assert (sub["final_leaf"].astype(int) == final.loc[held, "leaf"].astype(int)).all()
        assert (sub["fold_label"] == fold_route["label"]).all()
        assert (sub["reproduced"] == (sub["fold_label"] == sub["final_label"])).all()
        assert (sub["agree"] == (sub["reproduced"] & (sub["final_label"] == y.loc[held]))).all()
    # Leaf statistics and the gate (matches >= 5, fidelity >= 0.90, Wilson lower >= 0.60) recomputed from the ledger.
    for row in leaves.itertuples(index=False):
        m = ledger[(ledger["final_leaf"].astype(int) == row.leaf) & ledger["reproduced"]]
        k, n = int(m["agree"].sum()), len(m)
        assert (row.oof_matched, row.oof_agree) == (n, k)
        lo = wilson_interval(k, n)[0] if n else np.nan
        expected = n >= 5 and k / n >= 0.90 and lo >= 0.60 if n else False
        assert bool(row.selected) == expected and (row.selection == "SELECTED") == expected
        if n:
            assert row.wilson_lower == pytest.approx(lo)
    assert leaves["selected"].any() and (~leaves["selected"]).any()
    assert int(leaves["n_fit"].sum()) == len(x)


def test_selection_reasons_and_version_metrics():
    tree = {"features": ["lv_edv_ml"], "classes": ["DCM", "NOR"], "leaves": [1, 2],
            "nodes": [{"node": 0, "depth": 0, "is_leaf": False, "feature": "lv_edv_ml", "threshold": 150.0, "left": 1, "right": 2, "label": "NOR"},
                      {"node": 1, "depth": 1, "is_leaf": True, "feature": None, "threshold": None, "left": -1, "right": -1, "label": "NOR"},
                      {"node": 2, "depth": 1, "is_leaf": True, "feature": None, "threshold": None, "left": -1, "right": -1, "label": "DCM"}]}
    from tmmri.comparators.cart_f8 import leaf_rules

    tree["rules"] = leaf_rules(tree, {"lv_edv_ml": "mL"})
    assert [r["text"] for r in tree["rules"]] == ["IF lv_edv_ml <= 150 mL THEN d_model = NOR", "IF lv_edv_ml > 150 mL THEN d_model = DCM"]
    ids = [f"ACDC:p{i:02d}" for i in range(16)]
    x = pd.DataFrame({"lv_edv_ml": [100.0] * 10 + [200.0] * 6}, index=ids)
    d = pd.Series(["NOR"] * 10 + ["DCM"] * 5 + ["NOR"], index=ids)
    folds = pd.Series([1, 2, 3, 4, 5] * 3 + [1], index=ids)
    ledger = nested_ledger(tree, {f: tree for f in range(1, 6)}, x, folds, d)
    table = leaf_table(tree, x, d, ledger, {"selective": {"min_matches": 5, "min_oof_fidelity": 0.90, "wilson_lower_min": 0.60}}).set_index("leaf")
    assert (table.at[1, "oof_matched"], table.at[1, "oof_agree"], table.at[1, "selection"]) == (10, 10, "SELECTED")
    assert (table.at[2, "oof_matched"], table.at[2, "oof_agree"]) == (6, 5)
    assert table.at[2, "selection"] == "INELIGIBLE_FIDELITY;INELIGIBLE_WILSON_LOWER"          # 5/6 = 0.83; Wilson lower 0.44
    label = pd.Series(["NOR", "NOR", None, "DCM"], dtype=object)
    metrics = version_metrics(label, pd.Series(["NOR", "DCM", "NOR", None], dtype=object), pd.Series(["NOR", "NOR", "NOR", "DCM"], dtype=object),
                              pd.Series(["PREDICTED", "PREDICTED", NOT_SELECTED, "PREDICTED"]))
    assert metrics["coverage"]["k"] == 3 and metrics["coverage"]["n"] == 4
    assert (metrics["fidelity_to_d_model"]["k"], metrics["fidelity_to_d_model"]["n"]) == (1, 2)
    assert (metrics["accuracy_vs_reference"]["k"], metrics["accuracy_vs_reference"]["n"]) == (3, 3)
    assert metrics["coverage"]["wilson_lower"] == pytest.approx(wilson_interval(3, 4)[0])
    assert proportion(0, 0)["status"] == "NE" and metrics["routes"] == {NOT_SELECTED: 1, "PREDICTED": 3}
