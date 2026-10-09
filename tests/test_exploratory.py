"""Exploratory register B1-B9 and the F8 stages (protocol-v6.0.1 amendment items 7-8): hand-checked pure functions, the
B5 gate-power grid on a tiny synthetic grid, and every lane-L3 stage on a synthetic v6 run (algorithm controls only;
never cohort results)."""
from __future__ import annotations

import json
import math
import shutil

import numpy as np
import pandas as pd
import pytest

import fixtures_v6 as fx
from test_v6_cli import METHODS_MODULES, _edit_yaml, _run, _study_copy
from tmmri import cli, stages
from tmmri.config import load_study
from tmmri.experiments.exploratory import LABEL, run_exploratory
from tmmri.experiments.exploratory.annotation import difference_ci
from tmmri.experiments.exploratory.cut_stability import class_separation
from tmmri.experiments.exploratory.errors import ols_terms
from tmmri.experiments.exploratory.gate_power import gate_passes, gate_power_cell, grid_cells, run_grid
from tmmri.experiments.exploratory.orientation import case_orientation, lv_rv_direction, mid_ventricular_slice, strata_table
from tmmri.neural import preprocess as pp
from tmmri.experiments.exploratory.segmentation import basal_slice, dilate, erode, perturb_masks
from tmmri.experiments.exploratory.shift import descriptor_shift, group_drop
from tmmri.runs import RunRoot


@pytest.fixture(scope="module")
def study():
    return load_study()


# ----------------------------------------------------------------------------- B5 gate power


V6 = {"min_matched": 5, "wilson_lower_min": 0.60}


def test_gate_arithmetic_by_hand():
    assert not gate_passes(5, 5, V6)                    # Wilson lower of 5/5 is 0.566: a perfect 5-patient rule cannot pass
    assert gate_passes(6, 6, V6)                        # 6/6: 0.610
    assert not gate_passes(4, 4, {"min_matched": 5, "min_agreement": 0.90})
    assert gate_passes(5, 5, {"min_matched": 5, "min_agreement": 0.90})
    assert not gate_passes(20, 17, {"min_matched": 5, "min_agreement": 0.90})   # 0.85
    assert not gate_passes(0, 0, {"min_matched": 0})


def _settings(**over):
    base = {"agreements": [0.95], "supports": [5, 20], "null_agreements": [0.50], "n_datasets": 40, "seed": 20260928, "n_patients": 98, "n_folds": 5,
            "modes": ["counts"], "classes": ["DCM", "HCM", "MINF", "NOR", "RV"], "consequent": "NOR", "planted_feature": "lv_edv_ml",
            "distractor_feature": "rv_edv_ml", "background_consequent_rate": 0.10,
            "gates": {"v6": V6, "point_fidelity_0.90": {"min_matched": 5, "min_agreement": 0.90}}}
    base.update(over)
    return base


def test_gate_power_counts_grid_is_deterministic_and_plausible(study):
    kw = dict(rule_policy=study.rule_policy, wedd_cfg=study.wedd, registry=study.registry, profile=str(study.wedd["primary_profile"]))
    settings = _settings()
    assert [(c["kind"], c["support"]) for c in grid_cells(settings)] == [("planted", 5), ("planted", 20), ("null", 5), ("null", 20)]
    one = run_grid(settings, n_jobs=1, **kw)
    two = run_grid(settings, n_jobs=2, **kw)
    pd.testing.assert_frame_equal(one, two)
    rows = one.set_index(["kind", "support"])
    assert rows.at[("planted", 5), "pass_v6"] == 0.0                       # at most 5 matches: never passes the v6 gate
    assert rows.at[("planted", 20), "pass_v6"] > 0.5
    assert rows.at[("null", 20), "pass_v6"] == 0.0 and rows.at[("null", 20), "discovery_rate"] == 0.0
    assert set(one["n_datasets"]) == {40} and (one["pass_v6_wilson_lower"] <= one["pass_v6"]).all()
    # The cell result depends on the cell only (fan-out safe): a single cell reproduces its grid row.
    cell = {"mode": "counts", "kind": "planted", "agreement": 0.95, "support": 20, "n_datasets": 40}
    alone = gate_power_cell(cell, settings=settings, **kw)
    assert alone["n_pass_v6"] == int(rows.at[("planted", 20), "n_pass_v6"])


def test_gate_power_pipeline_cell_runs_v6_code(study):
    kw = dict(rule_policy=study.rule_policy, wedd_cfg=study.wedd, registry=study.registry, profile=str(study.wedd["primary_profile"]))
    settings = _settings(modes=["pipeline"], n_datasets=3)
    cell = {"mode": "pipeline", "kind": "planted", "agreement": 0.95, "support": 20, "n_datasets": 3}
    first = gate_power_cell(cell, settings=settings, **kw)
    assert first == gate_power_cell(cell, settings=settings, **kw)
    assert first["discovery_rate"] > 0 and 0.0 <= first["pass_v6"] <= 1.0 and first["mean_pooled_oof_matched"] > 0


# ----------------------------------------------------------------------------- B1, B2, B4, B7, B8 pure parts


def test_descriptor_shift_and_group_drop_find_the_planted_shift(study):
    rng = np.random.default_rng(20260613)
    cols = ["fd_ed_l1_mean", "fd_ed_l2_mean", "fd_es_l3_mean", "fd_l1_fraction_ed"]
    ref = pd.DataFrame(rng.normal(size=(60, 4)), columns=cols, index=[f"ACDC:p{i}" for i in range(60)])
    cmp_ = pd.DataFrame(rng.normal(size=(50, 4)), columns=cols, index=[f"MMS:c{i}" for i in range(50)])
    cmp_["fd_ed_l2_mean"] += 2.0
    table = descriptor_shift(ref, cmp_, min_cases=20).set_index("descriptor")
    assert table["abs_auroc_minus_half"].idxmax() == "fd_ed_l2_mean" and table["ks_distance"].idxmax() == "fd_ed_l2_mean"
    assert table.at["fd_ed_l2_mean", "auroc"] > 0.85 and (table["status"] == "EVALUATED").all()
    drops = group_drop(ref, cmp_, list(ref.index), list(cmp_.index), {"myo": "_l2_", "rv": "_l3_", "none": "_l9_"}, study.bootstrap["shift_audit"],
                       n_boot=20, seed=20260928, min_cases=20, ref_clusters=list(ref.index), cmp_clusters=list(cmp_.index)).set_index("group")
    assert drops.at["myo", "auroc_drop"] > 0.2 and abs(drops.at["rv", "auroc_drop"]) < 0.1 and drops.at["none", "status"] == "NE"


def test_ols_terms_recover_a_planted_slope():
    rng = np.random.default_rng(5)
    x = pd.DataFrame({"inplane_spacing_mm": rng.uniform(1.0, 2.0, 80), "slice_thickness_mm": rng.uniform(5, 10, 80), "num_slices": np.full(80, 10.0)})
    y = 0.5 * (x["slice_thickness_mm"] - x["slice_thickness_mm"].mean()) / x["slice_thickness_mm"].std(ddof=1) + rng.normal(0, 0.05, 80)
    table, info = ols_terms(y.to_numpy(), x, np.array([f"c{i}" for i in range(80)], dtype=object), n_boot=100, seed=1, level=0.95, min_cases=20)
    terms = table.set_index("term")
    assert info["dropped_covariates"] == ["num_slices"] and info["status"] == "EVALUATED" and info["r_squared"] > 0.9
    assert terms.at["slice_thickness_mm", "coef"] == pytest.approx(0.5, abs=0.03)
    assert terms.at["slice_thickness_mm", "ci_lower"] < 0.5 < terms.at["slice_thickness_mm", "ci_upper"]


def test_class_separation_by_hand():
    values = pd.DataFrame({"lv_edv_ml": [1.0, 2.0, 3.0, 7.0, 8.0, 9.0], "rv_edv_ml": [5.0] * 6})
    sep = class_separation(values, pd.Series(["A", "A", "A", "B", "B", "B"])).set_index("feature")
    # Grand mean 5; SS_between = 3 * 9 + 3 * 9 = 54; SS_total = 16 + 9 + 4 + 4 + 9 + 16 = 58.
    assert sep.at["lv_edv_ml", "eta_squared"] == pytest.approx(54 / 58) and np.isnan(sep.at["rv_edv_ml", "eta_squared"])


def test_difference_ci_and_mask_perturbations():
    est, lo, hi = difference_ci(np.array([0.0, 0.1, -0.1, 0.05]), np.array(list("abcd"), dtype=object), np.array([1.0, 1.1, 0.9, 1.05]),
                                np.array(list("efgh"), dtype=object), n_boot=200, seed=3, level=0.95)
    assert est == pytest.approx(1.0) and lo < 1.0 < hi
    m = np.zeros((7, 7, 3), dtype=np.int16)
    m[1:6, 1:6, 1] = 2                                   # myocardium square
    m[2:5, 2:5, 1] = 1                                   # LV inside
    m[2:5, 2:5, 2] = 1                                   # a smaller apical slice: LV only
    eroded = erode(m, 1)
    assert int((eroded[..., 1] == 1).sum()) == 1 and int((eroded[..., 1] == 2).sum()) == 24   # LV ring goes to the myocardium
    assert int((eroded[..., 2] == 1).sum()) == 1 and int((eroded[..., 2] == 0).sum()) == 48   # without neighbours: background
    dilated = dilate(m, 1)
    assert int((dilated[..., 1] == 1).sum()) == 21 and int((dilated[..., 1] == 2).sum()) == 4   # cross-shaped growth into the myocardium
    assert basal_slice(m) == 1                           # the outer labelled slice with the larger LV area
    ed, es = perturb_masks(m, m, "drop_basal_slice")
    assert not ed[..., 1].any() and not es[..., 1].any() and ed[..., 2].any()
    with pytest.raises(ValueError):
        perturb_masks(m, m, "erode_aorta")


def test_b9_orientation_measurement_by_hand():
    s = np.zeros((10, 10), dtype=np.int16)
    s[4:6, 6:8] = 1                                      # LV centroid (row 4.5, col 6.5)
    s[2:4, 1:3] = 3                                      # RV centroid (row 2.5, col 1.5): 2 rows up, 5 columns left
    d = lv_rv_direction(s, (1.0, 2.0), lv=1, rv=3)       # row spacing 1 mm, column spacing 2 mm
    assert (d["d_row_mm"], d["d_col_mm"]) == (-2.0, -10.0) and d["rv_left"] and d["rv_above"]
    assert d["angle_deg"] == pytest.approx(math.degrees(math.atan2(2.0, -10.0)))          # y up: about +168.7 degrees
    flat = np.zeros((4, 4), dtype=np.int16)
    flat[1, 3], flat[1, 0] = 1, 3                        # RV straight left: +180, never -180
    assert lv_rv_direction(flat, (1.0, 1.0), lv=1, rv=3)["angle_deg"] == 180.0
    stack = np.zeros((5, 10, 10), dtype=np.int16)
    stack[1:4] = s
    stack[4][s == 1] = 1                                 # an apical slice with LV only
    assert mid_ventricular_slice(stack, lv=1, rv=3) == (2, 3) and mid_ventricular_slice(stack[:1], lv=1, rv=3) == (None, 0)
    # The same voxels under a header whose second in-plane axis is reversed: the canonical frame mirrors them vertically.
    volume = np.moveaxis(stack, 0, 2)
    plain = case_orientation(volume, pp.geometry_from_parameters(volume.shape, (1.0, 1.0), 8.0, np.eye(3)), lv=1, rv=3)
    mirrored = case_orientation(volume, pp.geometry_from_parameters(volume.shape, (1.0, 1.0), 8.0, np.diag([1.0, -1.0, -1.0])), lv=1, rv=3)
    assert plain["status"] == mirrored["status"] == "MEASURED" and plain["slice_index"] == 2 and plain["orientation_code"] != mirrored["orientation_code"]
    assert mirrored["d_row_mm"] == pytest.approx(-plain["d_row_mm"]) and mirrored["d_col_mm"] == pytest.approx(plain["d_col_mm"])
    assert mirrored["angle_deg"] == pytest.approx(-plain["angle_deg"]) and mirrored["rv_above"] != plain["rv_above"]
    assert case_orientation(np.zeros((6, 6, 2), dtype=np.int16), pp.geometry_from_parameters((6, 6, 2), (1.0, 1.0), 8.0, np.eye(3)),
                            lv=1, rv=3)["status"] == "NO_LV_RV_SLICE"
    rows = pd.DataFrame([
        {"sample_id": "MMS:a", "dataset": "MMS", "partition": "mms_external", "vendor": "A", "orientation_code": "rows=+1,cols=+0",
         "status": "MEASURED", "angle_deg": -150.0, "rv_left": True, "rv_above": False},
        {"sample_id": "MMS:b", "dataset": "MMS", "partition": "mms_external", "vendor": "B", "orientation_code": "rows=-1,cols=+0",
         "status": "MEASURED", "angle_deg": 150.0, "rv_left": True, "rv_above": True},
        {"sample_id": "MMS:c", "dataset": "MMS", "partition": "mms_external", "vendor": "B", "orientation_code": "rows=+1,cols=+0",
         "status": "MEASURED", "angle_deg": -140.0, "rv_left": True, "rv_above": False},
        {"sample_id": "MMS:d", "dataset": "MMS", "partition": "mms_external", "vendor": "B", "orientation_code": "rows=+1,cols=+0",
         "status": "NO_LV_RV_SLICE", "angle_deg": np.nan, "rv_left": None, "rv_above": None}])
    table = strata_table(rows).set_index(["vendor", "orientation_code"])
    assert table.at[("all", "all"), "n"] == 3 and table.at[("all", "all"), "angle_median_deg"] == -140.0
    assert table.at[("all", "all"), "fraction_rv_left"] == 1.0 and table.at[("all", "all"), "fraction_rv_above"] == pytest.approx(1 / 3)
    assert table.at[("B", "rows=-1,cols=+0"), "n"] == 1 and table.at[("all", "rows=+1,cols=+0"), "n"] == 2
    assert table.at[("B", "all"), "angle_q10_deg"] == pytest.approx(-140.0 + 0.1 * 290.0)          # linear: 10 % of the way


def test_perturbed_phantom_volumes_move_in_the_expected_direction(registry, study):
    from tmmri.semantics import features as feat
    import test_v6_semantics as sem

    case = sem._case()
    base = feat.case_semantics_v6(case, registry, cohorts=study.cohorts, reason_codes=study.reason_codes)
    from tmmri.experiments.exploratory.segmentation import perturbed_case

    for name, sign in (("erode_lv", -1), ("dilate_lv", 1)):
        row = feat.case_semantics_v6(perturbed_case(case, name), registry, cohorts=study.cohorts, reason_codes=study.reason_codes)
        assert np.sign(row["lv_edv_ml"] - base["lv_edv_ml"]) == sign, name
    row = feat.case_semantics_v6(perturbed_case(case, "drop_basal_slice"), registry, cohorts=study.cohorts, reason_codes=study.reason_codes)
    assert row["lv_edv_ml"] == pytest.approx(base["lv_edv_ml"] * 5 / 6)          # one of six identical heart slices removed


# ----------------------------------------------------------------------------- B5 as a stage (no cohort data)


def test_b5_stage_provenance_and_once_only_guard(tmp_path):
    study_path = _study_copy(tmp_path)
    _edit_yaml(study_path.parent / "exploratory.yaml", lambda d: d["analyses"]["B5"].update(
        agreements=[0.9], supports=[10], null_agreements=[0.5], n_datasets=4, modes=["counts", "pipeline"]))
    study = load_study(study_path)
    run = RunRoot("v6-exploratory-20261003-0900", tmp_path / "artifacts", subfolders=study.study["artifact_subfolders"])
    ctx = stages.StageContext(study=study, run=run, data_root=tmp_path / "data", n_jobs=1)
    first = run_exploratory(ctx, "B5")
    assert first.action == "RAN" and first.record.params["label"] == LABEL
    out = run.path / "exploratory" / "B5"
    prov = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    assert prov["label"] == LABEL and prov["confirmatory"] is False and prov["analysis"] == "B5"
    cells = pd.read_parquet(out / "cells.parquet")
    assert len(cells) == 4 and set(cells["mode"]) == {"counts", "pipeline"}
    assert json.loads((out / "summary.json").read_text(encoding="utf-8"))["gates"]["v6"] == V6
    assert run_exploratory(ctx, "B5").action == "SKIPPED"
    run.frozen_path.write_text("{}", encoding="utf-8")                 # a frozen run: a recorded stage may only skip
    _edit_yaml(study_path.parent / "exploratory.yaml", lambda d: d["analyses"]["B5"].update(n_datasets=5))
    with pytest.raises(stages.StageRecordLockedError, match="config hash"):
        run_exploratory(stages.StageContext(study=load_study(study_path), run=run, data_root=tmp_path / "data", n_jobs=1), "B5")


# ----------------------------------------------------------------------------- every L3 stage on a synthetic v6 run


def _fast(d: dict) -> None:
    d["bootstrap"]["replicates"] = 40
    d["analyses"]["B4"]["replicates"] = 10
    d["analyses"]["B6"].update(n_salts=2, stability_replicates=3)
    d["analyses"]["B7"]["rho_bootstrap_replicates"] = 40


def test_lane_l3_stages_on_a_synthetic_run(tmp_path, monkeypatch, capsys):
    """fit-cart-surrogate, evaluate-cart-surrogate and run-exploratory B1-B4, B6-B8 through the CLI on a frozen synthetic run."""

    from conftest import require_module

    for module in METHODS_MODULES:
        require_module(module)
    monkeypatch.setattr(stages, "protocol_frozen", lambda *a, **k: False)
    monkeypatch.delattr(stages, "require_mms2_unlocked", raising=False)   # the M&Ms-2 lock is lane L1's; without it M&Ms-2 refuses
    acdc = fx.make_acdc_source(tmp_path / "src" / "acdc", fx.acdc_patients(5, testing_per_class=2))
    fx.write_fixture_manifests(acdc, fx.acdc_manifest(acdc))
    cases = fx.mms_refit_cases()
    mms = fx.make_mms_source(tmp_path / "src" / "mms", cases, shape=(26, 24, 3))
    fx.write_fixture_manifests(mms, fx.mms_manifest(mms, cases), fx.mms_metadata(mms))
    study_path = _study_copy(tmp_path)
    _edit_yaml(study_path.parent / "exploratory.yaml", _fast)
    run_id = "v6-l3-20261003-1000"
    common = ["--study", str(study_path), "--run-id", run_id, "--artifact-root", str(tmp_path / "artifacts"), "--data-root", str(tmp_path / "data"),
              "--acdc-source", str(acdc), "--mms-source", str(mms), "--n-jobs", "1", "--allow-unfrozen-synthetic"]
    code, out = _run(["run", *common, "--until", "audit-shift"], capsys)
    assert code == 0, out
    run = tmp_path / "artifacts" / run_id
    # F8: fit and freeze, then each recorded partition once.
    code, fit = _run(["fit-cart-surrogate", *common], capsys)
    assert code == 0 and fit["action"] == "RAN", fit
    f8 = run / "comparators" / "cart_f8"
    frozen = json.loads((f8 / "FROZEN_F8.json").read_text(encoding="utf-8"))
    assert frozen["v6_bundle_sha256"] == json.loads((run / ".stages" / "FROZEN.json").read_text(encoding="utf-8"))["bundle_sha256"]
    assert frozen["mms2_staged_at_freeze"] is False and frozen["counts"]["n_fit_rows"] > 0
    assert "comparators/cart_f8/tree.json" in frozen["bundle"] and not any(k.startswith("comparators") for k in
                                                                          json.loads((run / ".stages" / "FROZEN.json").read_text(encoding="utf-8"))["bundle"])
    rules = json.loads((f8 / "rules.json").read_text(encoding="utf-8"))
    assert rules["rules"] and all(r["text"].startswith("IF ") for r in rules["rules"])
    for split in ("acdc_test", "mms_external"):
        code, res = _run(["evaluate-cart-surrogate", *common, "--split", split], capsys)
        assert code == 0 and res["action"] == "RAN", (split, res)
        metrics = json.loads((f8 / "evaluation" / f"metrics_{split}.json").read_text(encoding="utf-8"))
        pred = pd.read_parquet(f8 / "evaluation" / f"predictions_{split}.parquet")
        full = metrics["versions"]["full_coverage"]
        assert metrics["status"] == "post_hoc" and metrics["prospective"] is False and full["n_attempted"] == len(pred)
        covered = pred["label_full"].notna()
        assert full["coverage"]["k"] == int(covered.sum())
        fid = covered & pred["d_model"].notna()
        assert full["fidelity_to_d_model"]["k"] == int((pred.loc[fid, "label_full"] == pred.loc[fid, "d_model"]).sum())
        sel = metrics["versions"]["selective"]
        assert sel["coverage"]["k"] <= full["coverage"]["k"] and set(pred.loc[pred["label_selective"].notna(), "leaf_selected"]) <= {True}
    assert len(pd.read_parquet(f8 / "evaluation" / "predictions_acdc_test.parquet")) == 10
    code, res = _run(["evaluate-cart-surrogate", *common, "--split", "acdc_test"], capsys)
    assert code == 0 and res["action"] == "SKIPPED"
    code, res = _run(["evaluate-cart-surrogate", *common, "--split", "acdc_test", "--force"], capsys)
    assert code == cli.EXIT_BUNDLE and "once-only guard" in res, res                    # once-only per partition
    code, res = _run(["evaluate-cart-surrogate", *common, "--split", "mms2_external"], capsys)
    assert code == cli.EXIT_PROTOCOL_NOT_FROZEN, res
    # Exploratory analyses on the recorded artifacts.
    for analysis in ("B1", "B2", "B3", "B4", "B6", "B7", "B8"):
        code, res = _run(["run-exploratory", *common, "--analysis", analysis], capsys)
        assert code == 0 and res["action"] == "RAN", (analysis, res)
        prov = json.loads((run / "exploratory" / analysis / "provenance.json").read_text(encoding="utf-8"))
        assert prov["label"] == LABEL and prov["confirmatory"] is False and prov["inputs"], analysis
        record = json.loads((run / ".stages" / f"run-exploratory__{analysis}.json").read_text(encoding="utf-8"))
        assert record["payload"]["params"]["label"] == LABEL and record["payload"]["params"]["lane_configs"] == ["configs/exploratory.yaml"]
    # B9: the label-free orientation audit on the staged masks (no frozen object read; the geometry settings of neural_input).
    code, res = _run(["run-exploratory", *common, "--analysis", "B9"], capsys)
    assert code == 0 and res["action"] == "RAN", res
    record = json.loads((run / ".stages" / "run-exploratory__B9.json").read_text(encoding="utf-8"))
    assert record["payload"]["params"]["lane_configs"] == ["configs/exploratory.yaml", "configs/neural_input.yaml"]
    prov = json.loads((run / "exploratory" / "B9" / "provenance.json").read_text(encoding="utf-8"))
    assert prov["label"] == LABEL and "B9 added" in prov["declared"] and "staging_manifest_MMS" in prov["inputs"]
    b9 = json.loads((run / "exploratory" / "B9" / "summary.json").read_text(encoding="utf-8"))
    assert set(b9["included"]) == {"acdc_dev", "acdc_test", "mms_external"} and "mms2_external" in b9["not_executed"]
    cases_b9 = pd.read_parquet(run / "exploratory" / "B9" / "per_case.parquet")
    assert (cases_b9["status"] == "MEASURED").all() and not ({"acdc_class", "pathology", "label", "y_true"} & set(cases_b9.columns))
    strata = pd.read_parquet(run / "exploratory" / "B9" / "strata.parquet")
    pooled = strata[(strata["vendor"] == "all") & (strata["orientation_code"] == "all")].set_index("partition")
    assert pooled.loc["acdc_dev", "n"] == int((cases_b9["partition"] == "acdc_dev").sum()) > 0
    # Phantom RV along +voxel axis 0: ACDC (affine diag(-1,-1,1)) keeps the axis, the identity-affine M&Ms fixture flips it.
    assert set(cases_b9.loc[cases_b9["dataset"] == "ACDC", "orientation_code"]) == {"rows=+1,cols=+0"}
    assert pooled.loc["acdc_test", "fraction_rv_left"] == 0.0 and pooled.loc["mms_external", "fraction_rv_left"] == 1.0
    assert set(strata.loc[strata["partition"] == "mms_external", "vendor"]) >= {"all", "A"}
    b1 = pd.read_parquet(run / "exploratory" / "B1" / "descriptor_shift.parquet")
    assert set(b1["stratum"]) >= {"all", "A", "B", "C", "D"} and b1["auroc"].between(0, 1).all()
    assert json.loads((run / "exploratory" / "B1" / "summary.json").read_text(encoding="utf-8"))["not_executed"] == {"mms2_external": "evaluate:mms2_external not recorded"}
    b2 = pd.read_parquet(run / "exploratory" / "B2" / "bias_spread.parquet")
    assert set(b2["vendor"]) >= {"all", "A"} and (b2.loc[b2["n"] < 20, "status"] == "NE").all()
    b3 = json.loads((run / "exploratory" / "B3" / "summary.json").read_text(encoding="utf-8"))
    assert b3["vendor"] == "A" and b3["mms_external"]["n"]["A"] > 0
    geo = pd.read_parquet(run / "exploratory" / "B3" / "geometry.parquet")
    assert set(geo["group"]) >= {"ACDC:acdc_dev", "ACDC:acdc_test"} and geo.loc[geo["variable"] == "num_slices", "median"].notna().all()
    b4 = pd.read_parquet(run / "exploratory" / "B4" / "feature_summary.parquet")
    assert set(b4["max_bins"]) == {2, 3, 4}
    b6 = json.loads((run / "exploratory" / "B6" / "summary.json").read_text(encoding="utf-8"))
    assert b6["n_salts"] == 2 and set(b6["eligible_rules_per_salt"]) == {"salt01", "salt02"}
    lam = pd.read_parquet(run / "exploratory" / "B6" / "lambdas.parquet")
    assert set(lam["salt"]) == {"primary", "salt01", "salt02"}
    b7 = json.loads((run / "exploratory" / "B7" / "summary.json").read_text(encoding="utf-8"))
    assert b7["baseline_recompute_max_abs_diff_formal_A_vs_recorded"] == pytest.approx(0.0, abs=1e-9)
    assert set(b7["perturbations"]) == {"erode_lv", "dilate_lv", "erode_myo", "dilate_myo", "erode_rv", "dilate_rv", "drop_basal_slice"}
    assert all(0 <= p["d_model_flip"]["k"] <= p["d_model_flip"]["n"] for p in b7["perturbations"].values())
    b8 = json.loads((run / "exploratory" / "B8" / "summary.json").read_text(encoding="utf-8"))
    assert b8["difference_status"] == "NOT_EXECUTED" and "mms2_external" in b8["not_executed"]
    code, res = _run(["run-exploratory", *common, "--analysis", "B1"], capsys)
    assert code == 0 and res["action"] == "SKIPPED"
    # A copy of the run: a tampered F8 bundle refuses; F8 is never fitted after M&Ms-2 staging.
    copy_id = "v6-l3-20261003-1001"
    shutil.copytree(run, tmp_path / "artifacts" / copy_id)
    other = [copy_id if a == run_id else a for a in common]
    tree = tmp_path / "artifacts" / copy_id / "comparators" / "cart_f8" / "tree.json"
    tree.write_text(tree.read_text(encoding="utf-8").replace('"label"', '"label" ', 1), encoding="utf-8")
    (tmp_path / "artifacts" / copy_id / ".stages" / "evaluate-cart-surrogate__mms_external.json").unlink()
    code, res = _run(["evaluate-cart-surrogate", *other, "--split", "mms_external"], capsys)
    assert code == cli.EXIT_BUNDLE and "changed after its freeze" in res, res
    (tmp_path / "artifacts" / copy_id / ".stages" / "fit-cart-surrogate.json").unlink()
    (tmp_path / "data" / "manifests" / "staging_MMS2.json").write_text("{}", encoding="utf-8")
    code, res = _run(["fit-cart-surrogate", *other], capsys)
    assert code == cli.EXIT_BUNDLE and "before any M&Ms-2 staging" in res, res
    (tmp_path / "data" / "manifests" / "staging_MMS2.json").unlink()
    (tmp_path / "data" / "manifests" / "extraction_MMS2.json").write_text(json.dumps({"extraction_started_utc": "2026-10-04T09:00:00+00:00"}),
                                                                           encoding="utf-8")
    code, res = _run(["fit-cart-surrogate", *other], capsys)
    assert code == cli.EXIT_BUNDLE and "extraction record" in res, res          # SPECIFICATION 14.6: dated against the extraction
