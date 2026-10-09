"""v6 transition math: ridge core, typed transforms, eligibility, lambda selection, nested OOF, comparators
(SPECIFICATION.md 6.0.0, Section 9; Decision D10). Synthetic algorithm controls only."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import REPO, SEED_SYNTHETIC

from tmmri import target_transforms as tt
from tmmri import transition as tr
from tmmri import transition_fit as tf


@pytest.fixture(scope="module")
def tcfg():
    with (REPO / "configs" / "transition.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _no_clip(cfg):
    out = copy.deepcopy(cfg)
    out["target_transforms"]["clipping"] = None
    return out


def _typed_fixture(n=100, noise=0.0, seed=SEED_SYNTHETIC):
    """Registry targets whose typed transform is exactly affine in A: log volume, logit EF, identity thickness."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, 12))
    z = A @ rng.normal(size=(12, 3)) / np.sqrt(12)
    B = pd.DataFrame(
        {
            "lv_edv_ml": np.exp(4.8 + 0.3 * z[:, 0] + noise * rng.normal(size=n)),
            "lv_ef_percent": 100.0 / (1.0 + np.exp(-(0.2 + 0.6 * z[:, 1] + noise * rng.normal(size=n)))),
            "global_wall_thickness_ed_mm": 8.0 + 1.5 * z[:, 2] + noise * rng.normal(size=n),
        }
    )
    return A, B


def test_affine_recovery_with_masks():
    rng = np.random.default_rng(SEED_SYNTHETIC)
    n, k, l = 400, 12, 5
    A = rng.normal(size=(n, k))
    T_true = rng.normal(size=(k, l))
    b_true = rng.normal(size=l)
    B = A @ T_true + b_true
    M = rng.random((n, l)) > 0.3
    fit = tr.fit_masked_ridge(A, B, M, lam=0.0)
    B_hat = tr.predict_masked_ridge(fit, A)
    rel = np.linalg.norm((B_hat - B)[M]) / np.linalg.norm(B[M])
    assert rel < 1e-10
    assert np.allclose(fit.coef, T_true, atol=1e-8)
    assert np.allclose(fit.intercept_raw, b_true, atol=1e-8)  # no drift from global versus observed-subset centring


@pytest.mark.parametrize("convention", ["sum_loss", "mean_loss"])
def test_svd_ridge_matches_reference_solver(convention):
    rng = np.random.default_rng(SEED_SYNTHETIC + 1)
    n, r = 300, 20
    Z = rng.normal(size=(n, r))
    y = Z @ rng.normal(size=r) + 0.1 * rng.normal(size=n)
    lam = 0.5
    w, c = tr.ridge_svd(Z, y, lam, convention=convention)
    Zc, yc = Z - Z.mean(0), y - y.mean()
    penalty = lam * (n if convention == "mean_loss" else 1.0)
    w_ref = np.linalg.solve(Zc.T @ Zc + penalty * np.eye(r), Zc.T @ yc)
    assert np.allclose(w, w_ref, rtol=1e-8, atol=1e-10)
    assert np.isclose(c, y.mean() - Z.mean(0) @ w_ref, rtol=1e-8)


def test_pseudoinverse_baseline_matches_numpy():
    rng = np.random.default_rng(SEED_SYNTHETIC + 2)
    A = rng.normal(size=(50, 8))
    B = rng.normal(size=(50, 3))
    assert np.allclose(tr.pseudoinverse_transition(A, B), np.linalg.pinv(A - A.mean(0)) @ (B - B.mean(0)), atol=1e-10)
    # rank-deficient design (duplicated column): lambda = 0 ridge is the minimum-norm solution
    A2 = np.column_stack([A, A[:, 0]])
    w, _ = tr.ridge_svd(A2, B, 0.0)
    assert np.allclose(w, np.linalg.pinv(A2 - A2.mean(0)) @ (B - B.mean(0)), atol=1e-10)


def test_typed_transforms_round_trip(tcfg, registry):
    lo, hi = tt.logit_clip_from_config(tcfg)
    x = np.array([0.0, 0.05, 20.0, 55.0, 99.95, 100.0])
    g = tt.get_transform("logit_percent", tcfg)
    assert np.allclose(g.inverse(g.forward(x)), 100.0 * np.clip(x / 100.0, lo, hi))
    v = np.array([0.5, 12.0, 160.0])
    assert np.allclose(tt.get_transform("log").inverse(tt.get_transform("log").forward(v)), v)
    assert np.isnan(tt.get_transform("log").forward(np.array([0.0, -1.0]))).all()  # outside the domain
    w = np.array([-30.0, 0.0, 4.2])
    assert np.allclose(tt.get_transform("identity").inverse(tt.get_transform("identity").forward(w)), w)
    _, B = _typed_fixture()
    tts = tt.TargetTransformSet.fit(B, registry, list(B.columns), cfg=tcfg)
    assert [tts.fits[c].kind for c in B.columns] == ["log", "logit_percent", "identity"]  # registry-driven
    back = tts.inverse(tts.forward(B, clip=False))
    assert np.allclose(back.to_numpy(), B.to_numpy(), rtol=1e-10)
    again = tt.TargetTransformSet.from_dict(tts.to_dict())
    assert np.allclose(again.forward(B).to_numpy(), tts.forward(B).to_numpy())


def test_targets_with_fewer_than_ten_rows_are_excluded(tcfg, registry):
    rng = np.random.default_rng(SEED_SYNTHETIC + 3)
    A = rng.normal(size=(60, 6))
    fit = tr.fit_masked_ridge(A, rng.normal(size=(60, 2)), np.column_stack([np.ones(60, bool), np.arange(60) < 9]), lam=0.0)
    assert fit.excluded_targets == {1: "fewer_than_10_observed_rows"}
    A, B = _typed_fixture(n=60)
    B.loc[9:, "lv_ef_percent"] = np.nan  # 9 observed fit rows
    B["lv_edv_indexed_ml_m2"] = 80.0     # registry transition_role excluded (never chosen by substring)
    model = tf.fit_transition_v6(A, B, ["lv_edv_ml", "lv_ef_percent", "global_wall_thickness_ed_mm", "lv_edv_indexed_ml_m2"], tcfg, registry, np.arange(60) % 5 + 1)
    assert model.excluded == {"lv_ef_percent": "TARGET_INELIGIBLE:fewer_than_10_observed_fit_rows", "lv_edv_indexed_ml_m2": "TARGET_INELIGIBLE:not_a_transition_candidate"}
    assert model.predict(A)["lv_ef_percent"].isna().all()


def test_target_eligibility_comes_from_registry_roles(registry):
    cands = tf.transition_candidates_v6(registry)
    assert len(cands) == registry["counts"]["transition_candidates"] == 25
    roles = {e["name"]: e["transition_role"] for e in registry["features"]}
    assert all(roles[c] in ("candidate", "candidate_secondary") for c in cands)
    assert "lv_myo_volume_ed_ml" not in cands and "bsa_m2" not in cands


def test_lambda_grid_is_the_configured_rule(tcfg):
    grid = tf.lambda_grid_v6(tcfg)
    assert grid.size == 21 and np.isclose(grid[0], 1e-6) and np.isclose(grid[-1], 1e4)


def test_lambda_grid_edge_is_flagged(tcfg, registry):
    A, B = _typed_fixture(n=100)
    rng = np.random.default_rng(SEED_SYNTHETIC + 4)
    B["global_wall_thickness_ed_mm"] = 8.0 + rng.normal(size=100)  # pure noise: heavy shrinkage
    model = tf.fit_transition_v6(A, B, list(B.columns), _no_clip(tcfg), registry, np.arange(100) % 5 + 1)
    assert model.lambdas["lv_edv_ml"] == pytest.approx(1e-6) and model.edge_flags["lv_edv_ml"]  # exact signal: lower edge
    assert model.lambdas["global_wall_thickness_ed_mm"] >= 10.0
    grid = tf.lambda_grid_v6(tcfg)
    flags = {row["target"]: row["flag"] for row in model.lambda_selection}
    for target, lam in model.lambdas.items():
        at_edge = bool(np.isclose(lam, grid[0]) or np.isclose(lam, grid[-1]))
        assert model.edge_flags[target] == at_edge
        assert flags[target] == (tf.LAMBDA_AT_GRID_EDGE if at_edge else "")


def test_nested_oof_recovers_typed_affine_signal(tcfg, registry):
    A, B = _typed_fixture(n=100)
    folds = np.arange(100) % 5 + 1
    oof, details = tf.nested_oof_transition(A, B, list(B.columns), _no_clip(tcfg), registry, folds, tf.positional_inner_folds(5), return_details=True)
    assert oof[list(B.columns)].notna().all().all()
    assert np.allclose(oof[list(B.columns)].to_numpy(), B.to_numpy(), rtol=1e-5)
    assert set(details["lambda_selection"]["outer_fold"]) == {1, 2, 3, 4, 5}
    # held-out rows never enter their own fold's fit: perturbing fold 1 targets leaves fold-1 predictions unchanged
    B2 = B.copy()
    B2.loc[folds == 1, "global_wall_thickness_ed_mm"] += 50.0
    oof2 = tf.nested_oof_transition(A, B2, list(B.columns), _no_clip(tcfg), registry, folds, tf.positional_inner_folds(5))
    assert np.allclose(oof2.loc[folds == 1, "global_wall_thickness_ed_mm"], oof.loc[folds == 1, "global_wall_thickness_ed_mm"])


def test_clipping_is_fitted_on_fit_rows_and_model_round_trips(tcfg, registry):
    A, B = _typed_fixture(n=100, noise=0.05)
    model = tf.fit_transition_v6(A, B, list(B.columns), tcfg, registry, np.arange(100) % 5 + 1)
    fit = model.transforms.fits["lv_edv_ml"]
    z = np.log(B["lv_edv_ml"].to_numpy())
    assert (fit.lo, fit.hi) == pytest.approx(tuple(np.quantile(z, tcfg["target_transforms"]["clipping"]["quantiles"])))
    again = tf.TransitionModel.from_dict(model.to_dict())
    assert np.array_equal(again.predict(A).to_numpy(), model.predict(A).to_numpy())


def test_comparators_share_rows_and_folds(tcfg, registry):
    A, B = _typed_fixture(n=100, noise=0.1)
    folds = np.arange(100) % 5 + 1
    comp = tf.comparators(A, B, list(B.columns), tcfg, registry, folds, inner_fold_fn=tf.positional_inner_folds(5))
    assert set(comp) == {"mean", "pseudoinverse_p1", "identity_transform_ridge"}
    for frame in comp.values():
        assert frame.shape[0] == 100 and frame[list(B.columns)].notna().all().all()
    held = folds == 3
    assert np.allclose(comp["mean"].loc[held, "lv_edv_ml"], B.loc[~held, "lv_edv_ml"].mean())


def test_domain_guard_blocks_far_out_of_domain_values(tcfg, registry):
    pred = pd.DataFrame({"global_wall_thickness_ed_mm": [-0.1, -5.0, 7.0], "lv_eccentricity_mean_ed": [1.02, 1.5, 0.4]})
    values, states = tf.decode_with_domain_guard(pred, registry, {"global_wall_thickness_ed_mm": (4.0, 14.0), "lv_eccentricity_mean_ed": (0.2, 0.9)}, tcfg)
    assert states["global_wall_thickness_ed_mm"].tolist() == ["ESTIMATED", "NUMERIC_DOMAIN_BLOCKED", "ESTIMATED"]
    assert values["global_wall_thickness_ed_mm"].iloc[0] == 0.0 and np.isnan(values["global_wall_thickness_ed_mm"].iloc[1])
    assert states["lv_eccentricity_mean_ed"].tolist() == ["ESTIMATED", "NUMERIC_DOMAIN_BLOCKED", "ESTIMATED"]


def test_global_lambda_and_rbf_kernel_ridge_comparators(tcfg, registry):
    """SPECIFICATION.md 9.4 (f) and (g): one lambda for every target, and the RBF kernel ridge of the linearity criterion
    (median pairwise-distance bandwidth, per-target inner-CV penalty on the primary grid), on the primary rows and folds."""
    A, B = _typed_fixture(n=100, noise=0.1)
    folds = np.arange(100) % 5 + 1
    glob = tf.fit_transition_global_lambda_v6(A, B, list(B.columns), tcfg, registry, folds)
    assert len(set(glob.lambdas.values())) == 1 and glob.lambda_selection[0]["scope"] == "global"
    assert glob.lambda_selection[0]["n_targets_pooled"] == 3 and np.isfinite(glob.predict(A).to_numpy()).all()
    rng = np.random.default_rng(SEED_SYNTHETIC + 7)
    A2 = rng.uniform(-2.0, 2.0, size=(150, 2))
    B2 = pd.DataFrame({"global_wall_thickness_ed_mm": 8.0 + 3.0 * np.sin(2.0 * A2[:, 0]) * np.cos(A2[:, 1])})
    cols, folds2, inner = list(B2.columns), np.arange(150) % 5 + 1, tf.positional_inner_folds(5)
    ridge = tf.nested_oof_transition(A2, B2, cols, tcfg, registry, folds2, inner)
    krr = tf.nested_oof_transition(A2, B2, cols, tcfg, registry, folds2, inner, fitter=tf.fit_rbf_kernel_ridge_v6)
    y = B2[cols[0]].to_numpy()
    assert np.mean(np.abs(krr[cols[0]] - y)) < 0.5 * np.mean(np.abs(ridge[cols[0]] - y))   # the nonlinear comparator sees curvature
    model = tf.fit_rbf_kernel_ridge_v6(A2, B2, cols, tcfg, registry, folds2)
    assert model.sigma == pytest.approx(tf.median_pairwise_bandwidth(model.z_fit)) and model.lambdas[cols[0]] in set(tf.lambda_grid_v6(tcfg))
    comp = tf.comparators(A, B, list(B.columns), tcfg, registry, folds, inner_fold_fn=inner, global_lambda=True, rbf=True)
    assert {"ridge_global_lambda", "rbf_kernel_ridge"} <= set(comp) and all(frame.shape[0] == 100 for frame in comp.values())
