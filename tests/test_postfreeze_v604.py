"""Third post-freeze exploratory block B28-B31 (protocol-v6.0.4) on a synthetic frozen v6 run, a synthetic B10-B22 run and
a synthetic B23-B27 run (algorithm controls only; never cohort results): the protocol-v6.0.4 lock with the D47
specification check, the third-source hash check, the B23 and B25 reproduction refusals, the D49 category bounds, the
D51 small-cell rule, the identifier policy and that nothing is written outside the new run root; every analysis
through the CLI in the run order of D50.

The synthetic source runs are those of ``test_postfreeze_v603.py`` (its ``lane`` fixture); the third source run is made
here by running B23 and B25 of the v6.0.3 lane under the run id the v6.0.4 lane names."""
from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from test_postfreeze_v603 import SECOND, SOURCE, _main, _postfreeze, _tree, lane  # noqa: F401  (lane is the module fixture)
from test_v6_cli import _edit_yaml
from tmmri import cli, stages
from tmmri import manuscript_tables as mt
from tmmri.experiments.postfreeze import b28_kappa_ceilings as b28
from tmmri.experiments.postfreeze import b29_ef_categories as b29
from tmmri.experiments.postfreeze import b30_reference_categories as b30
from tmmri.experiments.postfreeze import b31_plausibility as b31
from tmmri.experiments.postfreeze import common as pf
from tmmri.experiments.postfreeze import lane_v604 as v604
from tmmri.runs import hash_path

THIRD = "v6-postfreeze-20261028-2135"           # the lane's third source run (B23-B27)
RUN = "v6-postfreeze-20261030-1300"
ORDER = ("B28", "B29", "B31", "B30")             # D50 run order
OUTPUTS = {"B28": ("kappa_table", "kappa_null_summary", "majority_constant_table", "ceiling_table"),
           "B29": ("ef_category_crosstab", "ef_category_agreement"),
           "B30": ("confusion_table", "recall_table", "mms2_category_table", "b25_covered_table"),
           "B31": ("plausibility_counts", "ef_difference_distribution", "clipped_skill_table")}
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "exploratory_postfreeze_v604.yaml"


@pytest.fixture(scope="module")
def block(lane):  # noqa: F811
    """The synthetic runs of the v6.0.3 tests plus a third source run (B23 and B25 under the v6.0.3 lane)."""

    _edit_yaml(lane.study.parent / "exploratory_postfreeze_v604.yaml", lambda d: d["statistics"]["bootstrap"].update(replicates=40))
    for analysis in ("B23", "B25"):
        _main(["run-postfreeze", "--analysis", analysis, *lane.args, "--run-id", THIRD, "--allow-unfrozen-synthetic"])
    return SimpleNamespace(**vars(lane), third=lane.artifacts / THIRD)


@pytest.fixture(scope="module")
def outputs(block):
    """B28, B29, B31 and B30 through the CLI in the run order of D50, with the trees of every source run before and after."""

    roots = (SOURCE, SECOND, THIRD)
    before = {name: _tree(block.artifacts / name) for name in roots}
    data_before = _tree(block.data)
    for analysis in ORDER:
        _main(["run-postfreeze", "--analysis", analysis, *block.args, "--run-id", RUN, "--allow-unfrozen-synthetic"])
    return SimpleNamespace(before=before, after={name: _tree(block.artifacts / name) for name in roots}, data_before=data_before,
                           data_after=_tree(block.data), root=block.artifacts / RUN / "postfreeze")


def _ctx(block, artifacts=None, run_id: str = "v6-postfreeze-20261030-1900", synthetic: bool = True):
    return v604.make_v604_context(block.study, SOURCE, run_id, artifact_root=artifacts or block.artifacts, data_root=block.data,
                                  allow_unfrozen_synthetic=synthetic)


def _tampered(block, name: str):
    """A copy of the three source runs under ``name`` and the CLI arguments that point at it."""

    tampered = block.tmp / name
    for run_id in (SOURCE, SECOND, THIRD):
        shutil.copytree(block.artifacts / run_id, tampered / run_id)
    return tampered, [a if a != str(block.artifacts) else str(tampered) for a in block.args]


def _relax(d: dict) -> None:
    """Discovery and gate bounds low enough for the synthetic pooled sample to admit rules (algorithm control only)."""

    d["discovery"]["tiers"]["primary"].update(beta=0.6, min_confidence=0.4)
    d["discovery"]["min_antecedent_n_patients"] = 2
    d["validation_gate"].update(min_pooled_oof_matched_patients=1, agreement_wilson_lower_min=0.0)


@contextlib.contextmanager
def _relaxed_policy(block):
    path = block.study.parent / "rule_policy.yaml"
    original = path.read_bytes()
    _edit_yaml(path, _relax)
    try:
        yield
    finally:
        path.write_bytes(original)


def _rehash_third(tampered, analysis: str, rel: str) -> None:
    prov_path = tampered / THIRD / "postfreeze" / analysis / "provenance.json"
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    prov["outputs"][f"run:{rel}"] = hash_path(tampered / THIRD / rel)
    prov_path.write_text(json.dumps(prov), encoding="utf-8")


# ----------------------------------------------------------------------------------------------- unit checks


def test_half_up_rounding_and_kappa_on_known_examples():
    assert v604.half_up(7, 80) == "0.088" and v604.half_up(1, 8, 2) == "0.13" and v604.half_up(1, 3) == "0.333"
    assert v604.half_up(-1, 8, 2) == "-0.13" and v604.half_up(0, 5) == "0.000" and v604.half_up(3, 0) is None
    assert v604.cohen_kappa(list("AABB"), list("ABBB")) == pytest.approx(0.5)
    assert v604.kappa_counts(list("AABB"), list("ABBB")) == (4 * 3 - (2 * 1 + 2 * 3), 16 - (2 * 1 + 2 * 3))
    assert v604.cohen_kappa(list("AAA"), list("AAA")) is None and v604.cohen_kappa([], []) is None
    assert v604.cohen_kappa(list("AB"), list("BA")) == pytest.approx(-1.0)
    row = v604.proportion("p", 7, 80)
    assert row["p_k"] == 7 and row["p_n"] == 80 and row["p_half_up"] == "0.088" and row["p_ci_lower"] < 0.0875 < row["p_ci_upper"]
    pred = pd.Series(list("AABBAB") * 4, index=[f"S:{i}" for i in range(24)], dtype=object)
    ref = pd.Series(list("AABBBB") * 4, index=pred.index, dtype=object)
    k = v604.kappa_row(pred, ref, pd.Series(pred.index, index=pred.index), {"replicates": 50, "seed": 20260928})
    assert k["kappa_n"] == 24 and k["kappa"] == pytest.approx(v604.cohen_kappa(pred, ref)) and k["kappa_ci_lower"] <= k["kappa"] <= k["kappa_ci_upper"]
    assert k["kappa_half_up"] == v604.half_up(*v604.kappa_counts(pred, ref)) and 0 < k["kappa_n_boot_valid"] <= 50
    none = v604.kappa_row(pd.Series([None, None], index=["a", "b"], dtype=object), ref, pd.Series(dtype=object), {"replicates": 5, "seed": 1})
    assert none["kappa"] is None and none["kappa_status"] == "NE" and none["kappa_n"] == 0


def test_ef_category_bounds_of_the_lane():
    """D49: lower bound inclusive, upper bound exclusive; the categories of the lane file tile the line."""

    categories = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["statistics"]["ef_categories"]
    lv, rv = categories["lv_ef_percent"], categories["rv_ef_percent"]
    assert b29.category_labels(lv) == ["<40", "40-<50", ">=50"] and b29.category_labels(rv) == ["<40", ">=40"]
    values = pd.Series([39.999, 40.0, 49.999, 50.0, 90.0, np.nan, -5.0], index=list("abcdefg"))
    assert b29.categorize(values, lv).tolist() == ["<40", "40-<50", "40-<50", ">=50", ">=50", None, "<40"]
    assert b29.categorize(values, rv).tolist() == ["<40", ">=40", ">=40", ">=40", ">=40", None, "<40"]
    for bad in ([[None, 40], [45, None]], [[None, 40], [30, None]], [[0, 40], [40, None]], [[None, 40], [40, 50]], [[None, 50], [50, 40], [40, None]]):
        with pytest.raises(ValueError, match="contiguous"):
            b29.check_categories(bad)
    measured = b29.categorize(pd.Series([30.0, 45.0, 55.0, np.nan, 41.0]), lv)
    recon = b29.categorize(pd.Series([35.0, 52.0, 58.0, 45.0, np.nan]), lv)
    table = pd.DataFrame(b29.crosstab(measured, recon, b29.category_labels(lv)))
    assert len(table) == 9 and table["n"].sum() == 3 and table.set_index(["measured_category", "reconstructed_category"])["n"].loc[("40-<50", ">=50")] == 1
    row = b29.agreement_row(measured, recon, pd.Series(range(5), dtype=object).astype(str), {"replicates": 10, "seed": 1})
    assert row["same_category_k"] == 2 and row["same_category_n"] == 3 and row["kappa_n"] == 3


def test_small_cells_pool_and_suppress():
    """D51: categories below min_cell are pooled as other, which is suppressed below min_cell; pooled names never appear."""

    def cells(counts):
        return b30.small_cells(pd.Series([c for c, n in counts.items() for _ in range(n)], dtype=object), 5)

    assert cells({"HCM": 6, "DLV": 3, "ARR": 2}) == [{"category": "HCM", "n": 6, "status": "REPORTED"},
                                                      {"category": "other", "n": 5, "status": "REPORTED"}]
    assert cells({"HCM": 7, "DLV": 3, "ARR": 1}) == [{"category": "HCM", "n": 7, "status": "REPORTED"},
                                                      {"category": "other", "n": None, "status": b30.SUPPRESSED}]
    assert cells({"DLV": 3}) == [{"category": "other", "n": None, "status": b30.SUPPRESSED}]
    assert cells({"HCM": 5, "NOR": 5}) == [{"category": "HCM", "n": 5, "status": "REPORTED"}, {"category": "NOR", "n": 5, "status": "REPORTED"}]
    assert cells({}) == []
    with_missing = pd.Series(["HCM"] * 7 + ["DLV"] * 2 + [None] * 2, dtype=object)
    rows = b30.small_cells(with_missing, 5)
    assert rows == [{"category": "HCM", "n": 7, "status": "REPORTED"}, {"category": "other", "n": None, "status": b30.SUPPRESSED},
                    {"category": "missing", "n": 2, "status": "CATEGORY_MISSING"}]
    assert not {"DLV", "ARR"} & {r["category"] for r in cells({"HCM": 7, "DLV": 3, "ARR": 1})}


def test_b23_reproduction_refuses_any_difference():
    single = pd.DataFrame({"partition": ["acdc_test", "acdc_test"], "input": ["bhat7", "constant"], "fidelity_k": [8, 5], "fidelity_n": [10, 10]})
    draws = pd.DataFrame({"partition": "acdc_test", "null": "random7", "draw": [2, 0, 1], "fidelity_k": [7, 5, 6], "fidelity_n": [10, 10, 10]})
    good = {"acdc_test": {"bhat7": (8, 10), "constant": (5, 10)}}
    good_draws = {"acdc_test": {"random7": ([5, 6, 7], [10, 10, 10])}}
    rows = b28.check_b23_reproduction(single, draws, good, good_draws)
    assert len(rows) == 3 and all(r["reproduced"] for r in rows)
    for bad_single, bad_draws, message in (({"acdc_test": {"bhat7": (8, 11), "constant": (5, 10)}}, good_draws, "bhat7: numerator or denominator"),
                                           ({"acdc_test": {"bhat7": (9, 10), "constant": (5, 10)}}, good_draws, "bhat7: numerator or denominator"),
                                           ({"acdc_test": {"pca7": (5, 10)}}, good_draws, "pca7: 0 recorded rows"),
                                           (good, {"acdc_test": {"random7": ([5, 6], [10, 10])}}, "number or order"),
                                           (good, {"acdc_test": {"random7": ([5, 6, 8], [10, 10, 10])}}, "1 draws differ"),
                                           (good, {"acdc_test": {"permuted_bhat7": ([5], [10])}}, "number or order")):
        with pytest.raises(stages.BundleError, match=message):
            b28.check_b23_reproduction(single, draws, bad_single, bad_draws)


def test_b25_reproduction_refuses_a_different_coverage_or_the_declared_counts():
    from tmmri.experiments.postfreeze.b16_coverage_fidelity import system_metrics

    idx = ["S:1", "S:2", "S:3", "S:4"]
    metrics = system_metrics(pd.Series(["A", None, "B", "A"], index=idx, dtype=object), pd.Series([True, True, True, False], index=idx),
                             pd.Series(["A", "B", "A", "A"], index=idx, dtype=object), pd.Series([None] * 4, index=idx, dtype=object))
    recorded = pd.Series({"n_attempted": 4, "n_processed": 3, "n_accepted": 2, "coverage_k": 2, "coverage_n": 3, "fidelity_k": 1, "fidelity_n": 2})
    rows = b30.check_b25_reproduction(recorded, metrics, {"n_accepted": 2, "fidelity_k": 1})
    assert all(r["reproduced"] for r in rows) and len(rows) == 9
    with pytest.raises(stages.BundleError, match="n_accepted differs from Section 13.9"):
        b30.check_b25_reproduction(recorded, metrics, b30.DECLARED_B25)
    assert b30.DECLARED_B25 == {"n_accepted": 12, "fidelity_k": 9}
    with pytest.raises(stages.BundleError, match="fidelity_k differs from the recorded B25 row"):
        b30.check_b25_reproduction(pd.concat([recorded.drop("fidelity_k"), pd.Series({"fidelity_k": 2})]), metrics, None)
    with pytest.raises(stages.BundleError, match="coverage_n differs"):
        b30.check_b25_reproduction(recorded.drop("coverage_n"), metrics, None)


def test_recorded_confusion_check_and_recall():
    idx = ["S:1", "S:2", "S:3", "S:4", "S:5"]
    reference = pd.Series(["A", "A", "B", None, "B"], index=idx, dtype=object)
    decision = pd.Series(["A", "B", "B", "A", None], index=idx, dtype=object)
    cells = b30.confusion_cells(reference, decision)
    assert cells == {("A", "A"): 1, ("A", "B"): 1, ("B", "B"): 1}
    recorded = pd.DataFrame({"reference_class": ["A", "A", "B", "B"], "d_model": ["A", "B", "A", "B"], "n": [1, 1, 0, 1]})
    b30.check_confusion(recorded, cells, "acdc_test")
    with pytest.raises(stages.BundleError, match="differs from the recorded decisions"):
        b30.check_confusion(recorded.assign(n=[2, 1, 0, 1]), cells, "acdc_test")
    rows = b30.recall_rows(cells, ["A", "B"], partition="acdc_test")
    assert [(r["recall_k"], r["recall_n"], r["recall_half_up"]) for r in rows] == [(1, 2, "0.500"), (1, 1, "1.000")]


def test_identifier_scan():
    tokens = v604.identifier_tokens(["ACDC:patient001", "MMS:A1B2C3"])
    assert tokens == {"ACDC:patient001", "patient001", "MMS:A1B2C3", "A1B2C3"}
    assert v604.find_identifiers(pd.DataFrame({"sample_id": ["x"]}), tokens) == ["column sample_id"]
    assert v604.find_identifiers(pd.DataFrame({"category": ["HCM", "patient001"]}), tokens) == ["cells of category"]
    assert v604.find_identifiers(pd.DataFrame({"n": [1, 2]}, index=["MMS:A1B2C3", "y"]), tokens) == ["index"]
    assert v604.find_identifiers(pd.DataFrame({"n": [1, 1]}), {"1"}) == []                  # counts are never identifiers
    assert v604.find_identifiers({"rows": [{"x": "A1B2C3"}], "patient001": 1, "anchor": 2}, tokens) == ["a text value", "key patient001", "key anchor"]
    assert v604.find_identifiers({"rows": [{"category": "other", "n": 5}]}, tokens) == []
    with pytest.raises(RuntimeError, match="identifier"):
        v604.require_no_identifiers(pd.DataFrame({"category": ["A1B2C3"]}), tokens, "a table")


def test_b31_counts_distribution_and_clipping_on_a_known_example():
    bhat = pd.DataFrame({"lv_edv_ml": [100.0, 100.0, 0.0, 100.0, -10.0], "lv_esv_ml": [40.0, 40.0, 10.0, 100.0, 5.0],
                         "lv_ef_percent": [60.0, 50.0, 50.0, 0.0, np.nan], "rv_edv_ml": 150.0, "rv_esv_ml": 75.0, "rv_ef_percent": 50.0,
                         "lv_myocardial_mass_ed_g": 100.0}, index=list("abcde"))
    counts, dist = b31.side_rows(bhat, 5)
    lv = {r["check"]: r for r in counts if r["side"] == "lv"}
    ef = lv["ef_inconsistent"]                                   # e: EF not finite (counted there, not as EDV <= 0); c: EDV <= 0
    assert (ef["proportion_k"], ef["proportion_n"], ef["n_not_finite"], ef["n_edv_nonpositive"]) == (1, 3, 1, 1)
    assert ef["proportion_n"] + ef["n_not_finite"] + ef["n_edv_nonpositive"] == len(bhat)
    assert (lv["esv_ge_edv"]["proportion_k"], lv["esv_ge_edv"]["proportion_n"], lv["esv_ge_edv"]["proportion_half_up"]) == (3, 5, "0.600")
    rv = {r["check"]: r for r in counts if r["side"] == "rv"}
    assert rv["ef_inconsistent"]["proportion_k"] == 0 and rv["esv_ge_edv"]["proportion_k"] == 0
    d = {r["side"]: r for r in dist}
    assert d["lv"]["n"] == 3 and d["lv"]["median"] == 0.0 and d["lv"]["p97_5"] == pytest.approx(np.quantile([0.0, 10.0, 0.0], 0.975))
    bounds = {t: tuple(v) for t, v in yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["statistics"]["plausibility_bounds"].items()}
    rows = {r["target"]: r for r in b31.bounds_rows(bhat, bounds)}
    assert rows["lv_edv_ml"]["proportion_k"] == 2 and rows["lv_ef_percent"]["proportion_k"] == 1 and rows["lv_esv_ml"]["proportion_k"] == 0
    assert (rows["lv_ef_percent"]["proportion_n"], rows["lv_ef_percent"]["n_not_finite"]) == (4, 1)
    assert (rows["any primary target"]["proportion_k"], rows["any primary target"]["proportion_n"]) == (3, 5)
    edge = pd.DataFrame({t: [lo, hi] for t, (lo, hi) in bounds.items()})                            # inclusive bounds
    assert all(r["proportion_k"] == 0 for r in b31.bounds_rows(edge, bounds))
    clipped = b31.clip(bhat, bounds)
    assert clipped.loc["c", "lv_edv_ml"] == 20 and clipped.loc["d", "lv_ef_percent"] == 5 and clipped.loc["a", "lv_ef_percent"] == 60


def test_supervised_ceiling_lda_reproduces_sklearn():
    rng = np.random.default_rng(3)
    y = np.repeat(np.array(["DCM", "HCM", "MINF", "NOR", "RV"], dtype=object), 20)
    z = rng.standard_normal((100, 8)) + np.repeat(np.eye(5, 8) * 3, 20, axis=0)
    z[0, 2] = np.nan
    doc = b28.fit_lda(z, y, 4)
    assert doc["n_rows"] == 99 and np.asarray(doc["scalings"]).shape == (8, 4) and doc["classes"] == ["DCM", "HCM", "MINF", "NOR", "RV"]
    scores = b28.lda_scores(z, doc)
    assert scores.shape == (100, 4) and not np.isfinite(scores[0]).any() and np.isfinite(scores[1:]).all()
    with pytest.raises(ValueError, match="at least 5 classes"):
        b28.fit_lda(z[y != "RV"], y[y != "RV"], 4)


def test_working_tree_check_against_a_tag(tmp_path):
    """The lane file and src/ must equal their copies at the tag, with no untracked file below them."""

    def git(*args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "core.autocrlf=false", "-c", "commit.gpgsign=false",
                        "-c", "tag.gpgsign=false", *args], cwd=tmp_path,
                       check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "configs").mkdir()
    (tmp_path / "src").mkdir()
    lane_file, module = tmp_path / "configs" / f"{v604.CONFIG_KEY}.yaml", tmp_path / "src" / "a.py"
    lane_file.write_text("label: x\n", encoding="utf-8")
    module.write_text("A = 1\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("r\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "c")
    git("tag", "protocol-v6.0.4")
    assert v604.working_tree_check("protocol-v6.0.4", tmp_path)["equal"] is True
    (tmp_path / "README.md").write_text("changed\n", encoding="utf-8")                  # outside the pinned paths
    assert v604.working_tree_check("protocol-v6.0.4", tmp_path)["equal"] is True
    for path, text in ((module, "A = 2\n"), (lane_file, "label: y\n")):
        original = path.read_text(encoding="utf-8")
        path.write_text(text, encoding="utf-8")
        check = v604.working_tree_check("protocol-v6.0.4", tmp_path)
        assert check["equal"] is False and check["differs_from_tag"] is True
        path.write_text(original, encoding="utf-8")
    (tmp_path / "src" / "b.py").write_text("B = 1\n", encoding="utf-8")
    check = v604.working_tree_check("protocol-v6.0.4", tmp_path)
    assert check["equal"] is False and check["differs_from_tag"] is False and check["n_untracked"] == 1
    (tmp_path / "src" / "b.py").unlink()
    assert v604.working_tree_check("protocol-v6.0.4", tmp_path)["equal"] is True
    assert v604.working_tree_check("protocol-v6.0.9", tmp_path)["equal"] is None


# ----------------------------------------------------------------------------------------------- lock and sources


def test_v604_lock_tag_and_specification_check(block, capsys, monkeypatch):
    """The fixed source runs, the signed protocol-v6.0.4 tag and the D47 specification check against it (D52)."""

    code, out = _postfreeze(block, "B29", "v6-other-20261030-1300", capsys)
    assert code == cli.EXIT_USAGE_OR_INPUT and "must match" in out
    code, out = _postfreeze(block, "B29", THIRD, capsys)
    assert code == cli.EXIT_USAGE_OR_INPUT and "new run root" in out
    code, out = _postfreeze(block, "B29", RUN.replace("1300", "1310"), capsys, synthetic=False)
    assert code == cli.EXIT_PROTOCOL_NOT_FROZEN and "source run is fixed" in out
    assert not (block.artifacts / RUN.replace("1300", "1310")).exists()
    pctx = _ctx(block, synthetic=False)
    pctx.lane_cfg["source_runs"]["frozen"]["run_id"] = SOURCE
    with pytest.raises(stages.ProtocolFreezeError, match="before the owner signs protocol-v6.0.4"):
        v604.require_v604_unlocked(pctx)
    monkeypatch.setattr(stages, "protocol_frozen", lambda tag, *a, **k: tag == "protocol-v6.0.4")
    for newest, same, message in (("protocol-v6.0.4", False, "specification check of D47"), ("protocol-v6.0.5", True, "tag protocol-v6.0.5"),
                                  (None, True, "specification check of D47"), ("protocol-v6.0.4", None, "specification check of D47")):
        monkeypatch.setattr(mt, "specification_tag", lambda root, newest=newest: newest)
        monkeypatch.setattr(mt, "spec_matches_tag", lambda path, tag, same=same: same)
        with pytest.raises(stages.ProtocolFreezeError, match=message):
            v604.require_v604_unlocked(pctx)
        assert pctx.specification_check is None
    monkeypatch.setattr(mt, "specification_tag", lambda root: "protocol-v6.0.4")
    monkeypatch.setattr(mt, "spec_matches_tag", lambda path, tag: tag == "protocol-v6.0.4")
    with pytest.raises(stages.ProtocolFreezeError, match="must equal their copies at protocol-v6.0.4"):   # no such tag in this repository
        v604.require_v604_unlocked(pctx)
    for tree in ({"equal": False, "differs_from_tag": True, "n_untracked": 0}, {"equal": False, "differs_from_tag": False, "n_untracked": 2}):
        monkeypatch.setattr(v604, "working_tree_check", lambda tag, tree=tree: {"tag": tag, "paths": list(v604.TAG_PINNED), **tree})
        with pytest.raises(stages.ProtocolFreezeError, match="src/ must equal their copies"):
            v604.require_v604_unlocked(pctx)
    monkeypatch.setattr(v604, "working_tree_check", lambda tag: {"tag": tag, "paths": list(v604.TAG_PINNED), "equal": True, "differs_from_tag": False,
                                                                  "n_untracked": 0})
    with pytest.raises(stages.ProtocolFreezeError, match="--study must lie inside the project root"):     # the study copy of the fixture
        v604.require_v604_unlocked(pctx)
    assert pctx.specification_check is None
    monkeypatch.setattr(v604, "PROJECT_ROOT", block.study.parent.parent)                 # the copy's root holds configs/ itself
    assert v604.require_v604_unlocked(pctx) == "protocol_frozen"
    check = pctx.specification_check
    assert check["passed"] and check["compared_tag"] == "protocol-v6.0.4" and check["working_tree"]["equal"] and check["study"]["passed"]
    pctx.third_root = block.artifacts / SECOND
    with pytest.raises(stages.ProtocolFreezeError, match="third source run is fixed"):
        v604.require_v604_unlocked(pctx)
    good = _ctx(block)
    for target in (block.third / "postfreeze" / "B99" / "x.json", block.second / "x.json", block.source / "x.json", block.artifacts / "x" / "y.json"):
        with pytest.raises(PermissionError):
            good.require_new_output(target)
    assert good.portable(block.third / "postfreeze" / "B23" / "summary.json") == "postfreeze_v603:postfreeze/B23/summary.json"


def test_changed_or_foreign_third_source_refuses(block, capsys):
    tampered, args = _tampered(block, "tampered_third")
    path = tampered / THIRD / "postfreeze" / "B23" / "null_fidelity_table.parquet"
    path.write_bytes(path.read_bytes() + b" ")
    code, out = _postfreeze(block, "B28", RUN, capsys, args=args)
    assert code == cli.EXIT_BUNDLE and "differs from the SHA-256" in out and "postfreeze_v603" in out
    assert not (tampered / RUN / "postfreeze" / "B28" / "kappa_table.parquet").exists()
    good = _ctx(block)
    projections = "postfreeze/B23/models/lr_decision_link_random7_x1000/projections.npy"
    assert good.third_file(projections) == block.third / projections
    with pytest.raises(stages.BundleError, match="recorded neither"):
        good.third_file("postfreeze/B23/unlisted.json")
    for key, value, message in (("specification_version", "6.0.2", "specification_version"), ("source_runs", {}, "postfreeze_v602"),
                                ("run_id", SECOND, "run_id")):
        prov_path = tampered / THIRD / "postfreeze" / "B25" / "provenance.json"
        original = prov_path.read_text(encoding="utf-8")
        prov_path.write_text(json.dumps({**json.loads(original), key: value}), encoding="utf-8")
        try:
            with pytest.raises(stages.BundleError, match=message):
                _ctx(block, artifacts=tampered).third_record("B25")
        finally:
            prov_path.write_text(original, encoding="utf-8")


@pytest.mark.parametrize("record", ["single", "draw"])
def test_b28_refuses_when_b23_is_not_reproduced(block, capsys, record):
    """A recorded B23 numerator off by one (its hash re-recorded, so only the reproduction check can catch it) refuses
    before any B28 table is written."""

    tampered, args = _tampered(block, f"tampered_b23_{record}")
    name = "null_fidelity_table.parquet" if record == "single" else "null_distributions.parquet"
    rel = f"postfreeze/B23/{name}"
    table = pd.read_parquet(tampered / THIRD / rel)
    row = table.index[(table["partition"] == "acdc_test") & ((table["input"] == "bhat7") if record == "single" else (table["draw"] == 0))][0]
    table.loc[row, "fidelity_k"] = int(table.loc[row, "fidelity_k"]) + 1
    table.to_parquet(tampered / THIRD / rel, index=False)
    _rehash_third(tampered, "B23", rel)
    code, out = _postfreeze(block, "B28", RUN, capsys, args=args)
    assert code == cli.EXIT_BUNDLE and "do not reproduce the recorded B23 fidelities" in out
    assert ("acdc_test/bhat7" if record == "single" else "acdc_test/random7") in out
    run = tampered / RUN / "postfreeze" / "B28"
    assert not any((run / f"{name}.parquet").exists() for name in OUTPUTS["B28"])
    assert (run / "models" / b28.LDA_ESTIMATOR / "FROZEN.json").is_file()        # the ceiling is frozen before any evaluated partition


def test_b28_freezes_the_ceiling_before_any_evaluated_partition(block, capsys, monkeypatch):
    run_id = "v6-postfreeze-20261030-1400"
    seen = []
    original = b28.load_partition

    def spy(pctx, split):
        mdir = block.artifacts / run_id / "postfreeze" / "B28" / "models"
        seen.append((split, all((mdir / e / "FROZEN.json").is_file() for e in (b28.LDA_ESTIMATOR, b28.LINK_ESTIMATOR))))
        return original(pctx, split)

    monkeypatch.setattr(b28, "load_partition", spy)
    code, res = _postfreeze(block, "B28", run_id, capsys)
    assert code == 0, res
    assert seen[0] == ("acdc_dev", False) and [s for s, _ in seen[1:]] == ["acdc_test", "mms_external"] and all(f for _, f in seen[1:])


@pytest.fixture(scope="module")
def standin(block):
    """M&Ms-2 is locked in the synthetic run, so a third source run whose B25 evaluated acdc_test (as in the v6.0.3
    tests) stands in for the recorded mms2_external row; the rule policy is relaxed for that run so that the synthetic
    pooled sample admits rules (algorithm control only)."""

    from test_postfreeze_v603 import b25

    root = block.tmp / "standin_b25"
    for run_id in (SOURCE, SECOND):
        shutil.copytree(block.artifacts / run_id, root / run_id)
    args = [a if a != str(block.artifacts) else str(root) for a in block.args]
    path = block.study.parent / "exploratory_postfreeze_v603.yaml"
    mp = pytest.MonkeyPatch()
    mp.setattr(b25, "EVALUATION", ("acdc_test",))
    _edit_yaml(path, lambda d: d["analyses"]["B25"].update(partitions=["acdc_test"]))
    try:
        with _relaxed_policy(block):
            _main(["run-postfreeze", "--analysis", "B25", *args, "--run-id", THIRD, "--allow-unfrozen-synthetic"])
    finally:
        _edit_yaml(path, lambda d: d["analyses"]["B25"].update(partitions=["mms2_external"]))
        mp.undo()
    return root


def test_b30_b25_reapplication_on_a_stand_in_partition(block, standin, monkeypatch):
    """B30 (c) re-applies the stand-in B25 rulebook (at least one admitted rule) on acdc_test with real routing: it must
    reproduce the recorded row with non-zero accepted and agreement counts, and the declared counts refuse under the signed
    tag when they differ."""

    from tmmri.wedd import CutSet

    monkeypatch.setattr(b30, "MMS2", "acdc_test")
    with _relaxed_policy(block):
        pctx = _ctx(block, artifacts=standin)                       # the study as B25 used it
    pctx.tag_state = "synthetic_override"
    frozen = pctx.read_third_json(b30.b25_rel("models", b30.B25_ESTIMATOR, "FROZEN.json"))
    records = list(pctx.read_third_json(b30.b25_rel("models", b30.B25_ESTIMATOR, frozen["rulebook_file"])).get("records", []))
    cuts = CutSet.from_dict(pctx.read_json(frozen["cuts_file"]))
    *_, metrics = b30.reapply_b25(pctx, records, cuts)
    recorded = pctx.read_third_parquet(b30.b25_rel("coverage_fidelity_mms2.parquet")).iloc[0]
    assert recorded["partition"] == "acdc_test" and recorded["status"] == "EVALUATED" and len(records) >= 1
    assert metrics["n_processed"] > 0 and metrics["n_accepted"] > 0 and metrics["fidelity"]["k"] > 0
    rows = b30.check_b25_reproduction(recorded, metrics, None)
    assert all(r["reproduced"] for r in rows)
    with pytest.raises(stages.BundleError, match="Section 13.9"):
        b30.check_b25_reproduction(recorded, metrics, {"n_accepted": metrics["n_accepted"] + 1, "fidelity_k": metrics["fidelity"]["k"]})
    released = pd.read_parquet(standin / SOURCE / "contracts" / "cohort_ACDC.parquet").set_index("sample_id")["acdc_class"]  # stand-in category
    rows, note, ids = b30.b25_covered(pctx, released, 5)
    assert note["status"] == "EVALUATED" and all(r["reproduced"] for r in note["reproduction"]) and not note["declared_checked"]
    assert note["n_accepted"] == metrics["n_accepted"] and len(ids) == metrics["n_attempted"]
    shown = [r for r in rows if r["part"] == "accepted_by_released_category"]
    assert all(r["n"] is None or r["n"] >= 5 or r["status"] == "CATEGORY_MISSING" for r in shown)
    assert sum(r["n"] or 0 for r in shown) <= note["n_accepted"] and note["n_category_missing"] == 0
    disagree = [r for r in rows if r["part"] == "disagreement_d_model_decision"]
    assert sum(r["n"] for r in disagree) == note["n_disagreeing"] == metrics["fidelity"]["n"] - metrics["fidelity"]["k"]
    assert v604.find_identifiers(pd.DataFrame(rows), v604.identifier_tokens(ids)) == []
    pctx.tag_state = "protocol_frozen"                                    # under the signed tag the declared counts apply
    reproduced = {"n_accepted": metrics["n_accepted"], "fidelity_k": metrics["fidelity"]["k"]}
    monkeypatch.setattr(b30, "DECLARED_B25", reproduced)
    assert b30.b25_covered(pctx, released, 5)[1]["declared_checked"]
    monkeypatch.setattr(b30, "DECLARED_B25", {**reproduced, "fidelity_k": reproduced["fidelity_k"] + 1})
    with pytest.raises(stages.BundleError, match="fidelity_k differs from Section 13.9"):
        b30.b25_covered(pctx, released, 5)


def test_b30_refuses_under_the_signed_tag_without_an_evaluated_b25_row(block):
    """Under the signed tag (c) refuses when the recorded B25 row is not EVALUATED or mms2_external is not executed;
    NOT_EXECUTED is kept for the synthetic override only."""

    pctx = _ctx(block)
    pctx.third_record("B25")                                  # the synthetic record (mms2_external not executed) is cached first
    assert b30.b25_covered(pctx, pd.Series(dtype=object), 5)[1]["status"] == "NOT_EXECUTED"
    pctx.tag_state = "protocol_frozen"
    with pytest.raises(stages.BundleError, match="must be EVALUATED"):
        b30.b25_covered(pctx, pd.Series(dtype=object), 5)
    run_id = "v6-postfreeze-20261030-1500"
    whole = _ctx(block, run_id=run_id)
    whole.tag_state = "protocol_frozen"
    with pytest.raises(stages.BundleError, match="mms2_external is not executed"):
        b30.run(whole)
    assert not any((block.artifacts / run_id / "postfreeze" / "B30").glob("*.parquet"))


def test_provenance_is_checked_for_identifiers(block):
    pctx = _ctx(block, run_id="v6-postfreeze-20261030-1600")
    pctx.tag_state = "synthetic_override"
    pctx.identifier_tokens |= v604.identifier_tokens(["ACDC:patient999"])
    out = pctx.out_dir("B29") / "x.json"
    out.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="identifier"):
        pctx.write_provenance("B29", {}, [out], {"rows": ["patient999"]})
    assert not (out.parent / "provenance.json").exists()
    pctx.write_provenance("B29", {}, [out], {"rows": ["other"]})
    assert (out.parent / "provenance.json").is_file()


def test_b30_covered_rows_under_the_declared_counts(block, standin, monkeypatch):
    """The reporting of (c) on 12 accepted cases with 9 agreements (the counts of Section 13.9, given as routed labels):
    released categories under D51 with one accepted case without a category (7 shown + 4 suppressed in other + 1
    missing = 12), and the d_model decisions of the 3 disagreements; one agreement more refuses under the signed tag."""

    from tmmri.experiments.postfreeze.b16_coverage_fidelity import system_metrics

    ids = [f"ACDC:case{i:02d}" for i in range(14)]
    label = pd.Series(["NOR"] * 8 + ["DCM"] * 4 + [None] * 2, index=ids, dtype=object)
    processed = pd.Series([True] * 13 + [False], index=ids)
    released = pd.Series(["HCM"] * 7 + ["DLV"] * 3 + ["ARR"] + [None] + ["NOR"] * 2, index=ids, dtype=object)
    original = b30.check_b25_reproduction

    def as_recorded(_recorded, metrics, declared):          # the stand-in row is not this example; the declared counts still apply
        row = {"n_attempted": metrics["n_attempted"], "n_processed": metrics["n_processed"], "n_accepted": metrics["n_accepted"],
               **{f"{m}_{x}": metrics[m][x] for m in ("coverage", "fidelity") for x in ("k", "n")}}
        return original(pd.Series(row), metrics, declared)

    def routed(d_model):
        return lambda pctx, records, cuts: (label, processed, d_model, system_metrics(label, processed, d_model, pd.Series(None, index=ids, dtype=object)))

    monkeypatch.setattr(b30, "MMS2", "acdc_test")
    monkeypatch.setattr(b30, "check_b25_reproduction", as_recorded)
    pctx = _ctx(block, artifacts=standin)
    pctx.tag_state = "synthetic_override"
    pctx.third_record("B25")                                  # the synthetic record is read (and cached) under the synthetic override
    pctx.tag_state = "protocol_frozen"
    d_model = pd.Series(["NOR"] * 6 + ["HCM"] * 2 + ["DCM"] * 3 + ["MINF"] + ["NOR"] * 2, index=ids, dtype=object)
    monkeypatch.setattr(b30, "reapply_b25", routed(d_model))
    rows, note, _ = b30.b25_covered(pctx, released, 5)
    assert note["n_accepted"] == 12 and note["n_disagreeing"] == 3 and note["declared_checked"] and note["n_category_missing"] == 1
    assert [r for r in rows if r["part"] == "accepted_by_released_category"] == [
        {"part": "accepted_by_released_category", "min_cell": 5, "category": "HCM", "n": 7, "status": "REPORTED"},
        {"part": "accepted_by_released_category", "min_cell": 5, "category": "other", "n": None, "status": b30.SUPPRESSED},
        {"part": "accepted_by_released_category", "min_cell": 5, "category": "missing", "n": 1, "status": "CATEGORY_MISSING"}]
    assert sorted((r["rule_label"], r["d_model"], r["n"]) for r in rows if r["part"] == "disagreement_d_model_decision") == [
        ("DCM", "MINF", 1), ("NOR", "HCM", 2)]
    monkeypatch.setattr(b30, "reapply_b25", routed(d_model.where(d_model.index != "ACDC:case11", "DCM")))
    with pytest.raises(stages.BundleError, match="fidelity_k differs from Section 13.9"):
        b30.b25_covered(pctx, released, 5)


def test_b30_mms2_categories_on_a_stand_in_partition(block):
    """(b) runs on mms_external as a stand-in for the locked mms2_external (its cohort table also releases pathology)."""

    rows, recall, note, released, ids = b30.mms2_categories(_ctx(block), "mms_external")
    table = pd.DataFrame(rows)
    assert table["n_cases"].sum() == note["n_rows"] == len(ids) and set(released.index) >= set(ids)
    decided = table[[c for c in table.columns if c.startswith("n_d_model_")]].sum(axis=1)
    assert (decided + table["n_no_decision"] == table["n_cases"]).all()
    assert ((table["n_lvef_or_rvef_below_40"] <= table["n_with_observed_ef"]) & (table["n_with_observed_ef"] <= table["n_cases"])).all()
    assert sum(r["recall_n"] for r in recall) == note["n_same_definition"] - note["n_same_definition_without_decision"]
    assert all(r["part"] == "b" and r["recall_k"] <= r["recall_n"] for r in recall)
    tokens = v604.identifier_tokens(ids)
    assert v604.find_identifiers(table, tokens) == [] and v604.find_identifiers(pd.DataFrame(recall), tokens) == []


# ----------------------------------------------------------------------------------------------- every analysis


def _read(outputs, analysis: str, name: str):
    path = outputs.root / analysis / name
    return pd.read_parquet(path) if name.endswith(".parquet") else json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("analysis", ORDER)
def test_provenance_outputs_and_read_only_sources(block, outputs, analysis):
    assert outputs.after == outputs.before and outputs.data_after == outputs.data_before             # every source run untouched
    prov = _read(outputs, analysis, "provenance.json")
    assert prov["label"] == pf.LABEL and prov["analysis"] == analysis and prov["specification_version"] == "6.0.4"
    assert prov["protocol_tag"] == "protocol-v6.0.4" and prov["config"]["file"] == "configs/exploratory_postfreeze_v604.yaml"
    assert prov["protocol_tag_state"] == "synthetic_override" and prov["specification_check"] is None
    assert prov["source_run"]["run_id"] == SOURCE and prov["source_runs"]["postfreeze_v602"]["run_id"] == SECOND
    assert prov["source_runs"]["postfreeze_v602"]["read"] is False
    assert prov["source_runs"]["postfreeze_v603"]["run_id"] == THIRD
    assert not any(k.startswith("postfreeze_v602:") for k in prov["inputs"])
    for out_path, sha in prov["outputs"].items():
        assert out_path.startswith(f"run:postfreeze/{analysis}/")
        assert hashlib.sha256((outputs.root.parent / out_path.split(":", 1)[1]).read_bytes()).hexdigest() == sha
    names = {Path(k).name for k in prov["outputs"]}
    assert {f"{n}.parquet" for n in OUTPUTS[analysis]} | {"summary.json"} <= names
    assert _read(outputs, analysis, "summary.json")["label"] == pf.LABEL


def test_b28_outputs(block, outputs):
    prov = _read(outputs, "B28", "provenance.json")
    assert {"postfreeze_v603:postfreeze/B23/null_fidelity_table.parquet", "postfreeze_v603:postfreeze/B23/null_distributions.parquet"} <= set(prov["inputs"])
    assert prov["source_runs"]["postfreeze_v603"]["provenance_sha256"].keys() == {"B23"}
    recorded = pd.read_parquet(block.third / "postfreeze" / "B23" / "null_fidelity_table.parquet")
    kappa = _read(outputs, "B28", "kappa_table.parquet")
    assert sorted(kappa["partition"].unique()) == ["acdc_test", "mms_external"] and set(kappa["input"]) == set(b28.SINGLE)
    merged = kappa.merge(recorded, on=["partition", "input"], suffixes=("", "_b23"))
    assert len(merged) == len(kappa) and (merged["fidelity_k"] == merged["fidelity_k_b23"]).all() and (merged["fidelity_n"] == merged["fidelity_n_b23"]).all()
    assert (merged["recorded_b23_fidelity"] == merged["fidelity_b23"]).all()
    finite = kappa["kappa"].dropna()
    assert ((finite >= -1) & (finite <= 1)).all() and (kappa["kappa_n"] == kappa["fidelity_n"]).all()
    assert kappa.loc[kappa["kappa"].notna(), "kappa_n_boot_valid"].between(1, 40).all()
    assert (kappa.loc[kappa["input"] == "constant", "n_excluded_nonfinite"] == 0).all()
    assert (kappa["fidelity_n"] + kappa["n_excluded_nonfinite"] <= kappa["n_cases"]).all() and (kappa["n_no_decision"] >= 0).all()
    nulls = _read(outputs, "B28", "kappa_null_summary.parquet")
    assert set(nulls["null"]) == set(b28.NULLS) and (nulls["n_draws"] == 25).all() and (nulls["kappa_median"] <= nulls["kappa_p95"]).all()
    assert (nulls["fidelity_null_p95"] == nulls["recorded_b23_null_p95"]).all()
    assert (nulls["fidelity_bhat7_common"] == nulls["recorded_b23_fidelity_bhat7_common"]).all()
    assert (nulls["fidelity_bhat7_common_n"] == nulls["n_common_with_bhat7"]).all() and nulls["fidelity_bhat7_common_half_up"].notna().all()
    assert nulls["kappa_bhat7_common_half_up"].notna().all()
    reproduction = _read(outputs, "B28", "b23_reproduction.parquet")
    assert reproduction["reproduced"].all() and len(reproduction) == 2 * (len(b28.SINGLE) + len(b28.NULLS))
    majority = _read(outputs, "B28", "majority_constant_table.parquet")
    for split, rows in majority.groupby("partition"):
        d = pd.read_parquet(block.source / "inference" / f"d_model_{split}.parquet")["d_model"].dropna().astype(str).value_counts()
        cohort = rows.set_index("constant").loc["cohort_majority"]
        assert cohort["decision"] == sorted(d.index[d == d.max()])[0] and cohort["fidelity_k"] >= rows.set_index("constant").loc["development_majority_b23", "fidelity_k"]
        assert pd.isna(cohort["kappa"]) or abs(cohort["kappa"]) < 1e-12
    ceiling = _read(outputs, "B28", "ceiling_table.parquet")
    assert (ceiling["fidelity_n"] > 0).all() and (ceiling["n_predicted"] + ceiling["n_excluded_nonfinite"] == ceiling["n_cases"]).all()
    for estimator in (b28.LDA_ESTIMATOR, b28.LINK_ESTIMATOR):
        mdir = outputs.root / "B28" / "models" / estimator
        frozen = json.loads((mdir / "FROZEN.json").read_text(encoding="utf-8"))
        assert frozen["fit_partition"] == "acdc_dev" and all(hash_path(mdir / f) == sha for f, sha in frozen["files"].items())
    assert json.loads((outputs.root / "B28" / "models" / b28.LDA_ESTIMATOR / "lda.json").read_text(encoding="utf-8"))["lda"]["n_components"] == 4


def test_b29_outputs(outputs):
    cross = _read(outputs, "B29", "ef_category_crosstab.parquet")
    agree = _read(outputs, "B29", "ef_category_agreement.parquet")
    assert sorted(agree["partition"].unique()) == ["acdc_dev_oof", "acdc_test", "mms_external"]
    assert set(agree["target"]) == {"lv_ef_percent", "rv_ef_percent"}
    totals = cross.groupby(["partition", "target"])["n"].sum()
    assert (agree.set_index(["partition", "target"])["same_category_n"] == totals.reindex(agree.set_index(["partition", "target"]).index)).all()
    assert set(cross.loc[cross["target"] == "lv_ef_percent", "measured_category"]) == {"<40", "40-<50", ">=50"}
    assert set(cross.loc[cross["target"] == "rv_ef_percent", "measured_category"]) == {"<40", ">=40"}
    assert (agree["same_category_n"] + agree["n_not_observed"] + agree["n_observed_without_reconstruction"] == agree["n_cases"]).all()


def test_b30_outputs(outputs):
    confusion = _read(outputs, "B30", "confusion_table.parquet")
    assert sorted(confusion["partition"].unique()) == ["acdc_test", "mms_external"] and confusion["checked_against_recorded_decisions"].all()
    recall = _read(outputs, "B30", "recall_table.parquet")
    for split, rows in confusion.groupby("partition"):
        sub = recall[recall["partition"] == split].set_index("reference_class")
        diag = rows[rows["reference_class"] == rows["d_model"]].set_index("reference_class")["n"]
        assert (sub["recall_k"] == diag.reindex(sub.index).fillna(0).astype(int)).all()
        assert (sub["recall_n"] == rows.groupby("reference_class")["n"].sum().reindex(sub.index)).all()
    summary = _read(outputs, "B30", "summary.json")
    assert summary["not_executed"].keys() == {"mms2_external"} and summary["checks"]["b25_covered"]["status"] == "NOT_EXECUTED"
    assert summary["checks"]["acdc_test"]["confusion_reproduced"] and _read(outputs, "B30", "b25_covered_table.parquet").empty


def test_b31_outputs(outputs):
    counts = _read(outputs, "B31", "plausibility_counts.parquet")
    assert sorted(counts["partition"].unique()) == ["acdc_dev_oof", "acdc_test", "mms_external"]
    per = counts.groupby("partition")["check"].value_counts().unstack()
    assert (per["ef_inconsistent"] == 2).all() and (per["esv_ge_edv"] == 2).all() and (per["outside_bounds"] == 8).all()
    ef = counts[counts["check"] == "ef_inconsistent"]
    assert (ef["proportion_n"] + ef["n_not_finite"] + ef["n_edv_nonpositive"] == ef["n_cases"]).all()
    assert ((counts["proportion_k"] <= counts["proportion_n"]) & (counts["proportion_n"] <= counts["n_cases"])).all()
    dist = _read(outputs, "B31", "ef_difference_distribution.parquet")
    assert set(dist["side"]) == {"lv", "rv"} and (dist["p2_5"] <= dist["median"]).all() and (dist["median"] <= dist["p97_5"]).all()
    skill = _read(outputs, "B31", "clipped_skill_table.parquet")
    assert (skill["in_sample"] == (skill["partition"] == "acdc_dev_oof")).all()
    assert "acdc_dev_oof" in _read(outputs, "B31", "summary.json")["rho"]["in_sample"]
    assert skill.groupby("partition").size().eq(8).all() and (skill.groupby("partition")["target"].apply(set) == {"rho", *b31_family()}).all()
    assert (skill["status"] != "NE").all() and skill[["rho_unclipped", "rho_clipped", "ci_lower", "ci_upper"]].notna().all().all()
    assert np.allclose(skill["difference_clipped_minus_unclipped"], skill["rho_clipped"] - skill["rho_unclipped"])
    unclipped = skill[skill["n_cells_clipped"] == 0]
    assert len(unclipped) and np.allclose(unclipped["difference_clipped_minus_unclipped"], 0.0)       # no clipped cell, no difference


def b31_family() -> set[str]:
    return set(yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["statistics"]["primary_targets"])


def test_no_identifier_leaves_the_run_root(block, outputs):
    ids = set()
    for path in block.source.glob("contracts/cohort_*.parquet"):
        cohort = pd.read_parquet(path)
        for column in ("sample_id", "case_id", "patient_id"):
            if column in cohort.columns:
                ids |= {str(v) for v in cohort[column].dropna()}
    tokens = v604.identifier_tokens(ids)
    assert tokens
    checked = 0
    for path in outputs.root.rglob("*"):
        if path.suffix == ".parquet":
            assert v604.find_identifiers(pd.read_parquet(path), tokens) == [], path
            checked += 1
        elif path.suffix == ".json":                                                     # provenance records included
            assert v604.find_identifiers(json.loads(path.read_text(encoding="utf-8")), tokens) == [], path
            checked += 1
    assert checked >= 19
