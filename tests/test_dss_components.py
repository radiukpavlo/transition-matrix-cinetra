from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tmmri.formal_features import build_a_matrix
from tmmri.classification import expected_calibration_error_equal_mass
from tmmri.equivariance import _native_vector_from_arrays, _project_native_vector, _shift_zero
from tmmri.metrics import bca_bootstrap_ci, benjamini_hochberg, evaluate_residuals, metric_summary
from tmmri.semantics.manual_review import build_manual_review_table, parse_qc_reasons, severity_for_reasons
from tmmri.rules import evaluate_rulebook, induce_rules, infer_with_rulebook
from tmmri.transition_fit import cross_validate_lambdas, fit_final_transition, selected_lambda


def _small_b() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(12)],
            "patient_id": [f"p{i}" for i in range(12)],
            "dataset_source": ["ACDC"] * 12,
            "split": ["train"] * 8 + ["val"] * 2 + ["test"] * 2,
            "disease_label": ["A", "A", "B", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            "ed_frame_index": list(range(12)),
            "es_frame_index": list(range(1, 13)),
            "spacing_x_mm": np.linspace(1.0, 2.0, 12),
            "spacing_y_mm": np.linspace(1.0, 2.0, 12),
            "lv_edv_ml": np.linspace(50, 100, 12),
            "lv_esv_ml": np.linspace(20, 50, 12),
            "lv_sv_ml": np.linspace(30, 50, 12),
            "lv_ef_percent": np.linspace(40, 60, 12),
            "rv_edv_ml": np.linspace(45, 95, 12),
            "rv_esv_ml": np.linspace(20, 45, 12),
            "rv_sv_ml": np.linspace(25, 50, 12),
            "rv_ef_percent": np.linspace(38, 65, 12),
            "global_wall_thickness_ed_mm": np.linspace(6, 14, 12),
            "lv_myocardial_mass_ed_g": np.linspace(60, 120, 12),
            "qc_manual_review_required": [False, True] + [False] * 10,
            "qc_failure_reason": [None, "lv_rv_sv_discrepancy_gt_15_ml;extreme_robust_outlier"] + [None] * 10,
        }
    )


def test_manual_review_reason_parsing_and_severity() -> None:
    reasons = parse_qc_reasons("lv_rv_sv_discrepancy_gt_15_ml; extreme_robust_outlier")
    assert reasons == ["lv_rv_sv_discrepancy_gt_15_ml", "extreme_robust_outlier"]
    assert severity_for_reasons(reasons) == "high"
    table = build_manual_review_table([_small_b()])
    assert len(table) == 1
    assert table.loc[0, "adjudication_status"] == "pending_review"


def test_a_extraction_semantic_proxy_is_deterministic_and_aligned() -> None:
    b = _small_b()
    a1, meta1 = build_a_matrix(b, latent_dim=16, seed=7, allow_semantic_proxy_without_native=True)
    a2, meta2 = build_a_matrix(b, latent_dim=16, seed=7, allow_semantic_proxy_without_native=True)
    assert list(a1["sample_id"]) == list(b["sample_id"])
    assert [c for c in a1.columns if c.startswith("a_")] == [f"a_{i:04d}" for i in range(16)]
    np.testing.assert_allclose(a1.filter(like="a_").to_numpy(), a2.filter(like="a_").to_numpy())
    assert meta1["projection_checksum"] == meta2["projection_checksum"]


def test_transition_cv_and_fit_shape_contract() -> None:
    b = _small_b()
    a, _ = build_a_matrix(b, latent_dim=8, seed=11, allow_semantic_proxy_without_native=True)
    a_cols = [c for c in a.columns if c.startswith("a_")]
    b_cols = ["lv_edv_ml", "lv_esv_ml", "lv_ef_percent"]
    train_mask = b["split"].eq("train")
    cv = cross_validate_lambdas(a, b, a_cols, b_cols, train_mask, lambda_grid=[0.1, 1.0], cv_folds=2, seed=3)
    lam = selected_lambda(cv)
    t, state = fit_final_transition(a, b, a_cols, b_cols, train_mask, lam)
    assert t.shape == (len(a_cols), len(b_cols))
    assert state.b_columns == b_cols


def test_evaluation_bh_and_metric_summary() -> None:
    bh = benjamini_hochberg([0.001, 0.02, 0.9], alpha=0.05)
    assert bh["rejected"].tolist()[:2] == [True, True]
    m = metric_summary(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.5, 4.0]))
    assert m["mae"] == pytest.approx(0.5)
    ci_low, ci_high = bca_bootstrap_ci(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), np.nanmean, iterations=100, seed=1)
    assert ci_low <= ci_high
    residuals = pd.DataFrame(
        {
            "dataset_source": ["ACDC"] * 8,
            "split": ["test"] * 8,
            "feature_name": ["lv_edv_ml"] * 8,
            "observed_value": np.linspace(10.0, 20.0, 8),
            "predicted_value": np.linspace(11.0, 21.5, 8),
            "manual_review_flag": [False] * 8,
        }
    )
    metrics, failures = evaluate_residuals(residuals, "exp", bootstrap_iterations=50, bootstrap_seed=2)
    assert {"rmse_ci_lower", "rmse_ci_upper", "mae_ci_lower", "mae_ci_upper", "bias_ci_lower", "bias_ci_upper"}.issubset(metrics.columns)


def test_ece_uses_exactly_10_equal_mass_bins() -> None:
    y = np.array([0, 1] * 10)
    p = np.vstack([np.linspace(0.51, 0.99, 20), 1 - np.linspace(0.51, 0.99, 20)]).T
    ece, bins = expected_calibration_error_equal_mass(y, p, n_bins=10)
    assert len(bins) == 10
    assert bins["count"].sum() == 20
    assert ece >= 0


def test_rulebook_abstains_when_no_rule_matches() -> None:
    b = _small_b()
    rulebook, _, cutpoints = induce_rules(b[b["split"].eq("train")], feature_cols=["lv_ef_percent", "global_wall_thickness_ed_mm"], min_support=2, min_confidence=0.5)
    assert rulebook["default_action"] == "abstain"
    pred, rule_id, conf, status = infer_with_rulebook({"rules": []}, pd.Series({"lv_ef_percent": "low"}))
    assert pred == "abstain"
    metrics, preds = evaluate_rulebook(rulebook, b[b["split"].eq("test")], cutpoints, "exp", "test")
    assert "abstention_rate" in metrics.columns


def test_equivariance_native_vector_helpers_are_finite() -> None:
    image_ed = np.zeros((8, 8, 2), dtype=float)
    image_es = np.ones((8, 8, 2), dtype=float)
    mask_ed = np.zeros((8, 8, 2), dtype=int)
    mask_es = np.zeros((8, 8, 2), dtype=int)
    mask_ed[2:5, 2:5, :] = 1
    mask_es[3:5, 3:5, :] = 1
    shifted = _shift_zero(mask_ed, (1, 0))
    assert shifted.shape == mask_ed.shape
    vec = _native_vector_from_arrays(image_ed, image_es, mask_ed, mask_es)
    latent = _project_native_vector(vec, latent_dim=8, seed=3)
    assert vec.shape[0] > 0
    assert latent.shape == (8,)
    assert np.isfinite(latent).all()

from tmmri.formal_features import _numeric_source_columns
from tmmri.for_rules_import import validate_for_rules_frames


def test_strict_semantic_proxy_excludes_clinical_target_families() -> None:
    b = _small_b().copy()
    b["lv_edv_indexed_ml_m2"] = np.linspace(10, 20, len(b))
    b["global_wall_thickness_es_mm"] = np.linspace(8, 15, len(b))
    b["spacing_x_mm"] = np.linspace(1.2, 1.8, len(b))
    cols = _numeric_source_columns(b, include_transition_targets=False, proxy_policy="strict_metadata_qc_no_targets")
    assert "spacing_x_mm" in cols
    assert "lv_edv_ml" not in cols
    assert "lv_edv_indexed_ml_m2" not in cols
    assert "global_wall_thickness_es_mm" not in cols


def test_for_rules_frame_validator_checks_required_schema_without_strict_rows() -> None:
    base = _small_b().copy()
    base["schema_version"] = "test"
    base["case_id"] = base["patient_id"]
    # Create the exact same schema across all frames, with expected split values.
    frames = {
        "acdc_train": base.iloc[[0, 1]].assign(split="train"),
        "acdc_val": base.iloc[[2]].assign(split="val"),
        "acdc_test": base.iloc[[3]].assign(split="test"),
        "mms_100": base.iloc[[4]].assign(sample_id=["mms_s0"], patient_id=["mms_p0"], dataset_source="MMS_100", split="mms_100_external"),
    }
    assert validate_for_rules_frames(frames, strict_rows=False) == []
    broken = {k: v.copy() for k, v in frames.items()}
    broken["acdc_train"] = broken["acdc_train"].drop(columns=["lv_edv_ml"])
    failures = validate_for_rules_frames(broken, strict_rows=False)
    assert any("missing_required_columns" in f for f in failures)
