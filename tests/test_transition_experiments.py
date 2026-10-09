from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tmmri.formal_features import _matrix_checksum, _projection_matrix
from tmmri.experiments.transition_grid import (
    build_formal_branches,
    normalized_cv_lambda_grid,
    recover_native_descriptors,
    run_synthetic_recovery_benchmark,
)


def _frozen_projection(
    *,
    rows: int = 24,
    input_dim: int = 57,
    latent_dim: int = 128,
    seed: int = 20260611,
) -> tuple[pd.DataFrame, dict[str, object], np.ndarray]:
    rng = np.random.default_rng(314159)
    native = rng.normal(size=(rows, input_dim))
    basis = _projection_matrix(input_dim, latent_dim, seed)
    projected = native @ basis
    direct_n = min(32, input_dim, latent_dim)
    projected[:, :direct_n] = native[:, :direct_n]
    frame = pd.DataFrame(projected, columns=[f"a_{index:04d}" for index in range(latent_dim)])
    frame.insert(0, "split", ["train"] * 16 + ["val"] * 4 + ["test"] * 4)
    frame.insert(0, "dataset_source", "synthetic")
    frame.insert(0, "patient_id", [f"patient_{index:03d}" for index in range(rows)])
    frame.insert(0, "sample_id", [f"sample_{index:03d}" for index in range(rows)])
    metadata = {
        "latent_dim": latent_dim,
        "projection_input_dimension": input_dim,
        "seed": seed,
        "projection_checksum": _matrix_checksum(basis),
    }
    return frame, metadata, native


def test_recover_native_descriptors_is_exact_and_checksum_guarded() -> None:
    frozen, metadata, native = _frozen_projection()

    recovered, audit = recover_native_descriptors(frozen, metadata)

    observed = recovered.filter(regex=r"^a0_").to_numpy(dtype=float)
    assert observed == pytest.approx(native, abs=1e-11)
    assert audit["untouched_projection_rank"] == 57
    assert audit["roundtrip_max_absolute_error"] < 1e-11
    assert audit["recovery_status"] == "verified_exact_within_tolerance"

    corrupt = dict(metadata)
    corrupt["projection_checksum"] = "0" * 64
    with pytest.raises(ValueError, match="checksum mismatch"):
        recover_native_descriptors(frozen, corrupt)


def test_formal_branches_are_nested_and_keep_all_patients() -> None:
    acdc, metadata, _ = _frozen_projection()
    mms = acdc.copy()
    mms["sample_id"] = [f"mms_{index:03d}" for index in range(len(mms))]
    mms["patient_id"] = [f"mms_patient_{index:03d}" for index in range(len(mms))]

    branches, dimension_audit, recovery_audit = build_formal_branches(
        acdc, mms, metadata, metadata
    )

    expected = {
        "A0_native_unprojected_57": 57,
        "A2_nested_32": 32,
        "A2_nested_64": 64,
        "A2_nested_96": 96,
        "A1_primary_128": 128,
        "A3_nested_expansion_256": 256,
    }
    assert set(branches) == set(expected)
    for name, dimension in expected.items():
        acdc_branch, mms_branch = branches[name]
        formal = [column for column in acdc_branch if column.startswith(("a_", "a0_"))]
        assert len(formal) == dimension
        assert len(acdc_branch) == len(acdc)
        assert len(mms_branch) == len(mms)
    assert dict(zip(dimension_audit["a_branch"], dimension_audit["declared_dimension"])) == expected
    assert set(recovery_audit["recovery_status"]) == {"verified_exact_within_tolerance"}


def test_normalized_cv_uses_training_only_objective_and_is_deterministic() -> None:
    rng = np.random.default_rng(2718)
    rows = 60
    a = pd.DataFrame(rng.normal(size=(rows, 6)), columns=[f"a_{index:04d}" for index in range(6)])
    b = pd.DataFrame(
        {
            "target_small": 0.8 * a["a_0000"] - 0.3 * a["a_0001"] + rng.normal(0, 0.05, rows),
            "target_large": 1000 * a["a_0002"] + rng.normal(0, 80, rows),
            "disease_label": np.repeat(["NOR", "DCM", "HCM"], rows // 3),
        }
    )
    train_mask = pd.Series([True] * 45 + [False] * 15)

    first, folds, selected = normalized_cv_lambda_grid(
        a,
        b,
        list(a.columns),
        ["target_small", "target_large"],
        train_mask,
        lambda_grid=(0.01, 0.1, 1.0, 10.0),
        cv_folds=3,
        seed=42,
    )
    second, _, selected_again = normalized_cv_lambda_grid(
        a,
        b,
        list(a.columns),
        ["target_small", "target_large"],
        train_mask,
        lambda_grid=(0.01, 0.1, 1.0, 10.0),
        cv_folds=3,
        seed=42,
    )

    assert selected == selected_again
    pd.testing.assert_frame_equal(first, second)
    assert first["selected"].sum() == 1
    assert set(first["selection_objective"]) == {
        "mean_per_target_RMSE_divided_by_training_fold_target_SD"
    }
    assert set(folds["selection_partition"]) == {"ACDC_train_internal_CV"}
    assert folds["fit_patients"].max() + folds["held_out_patients"].max() <= 45


def test_synthetic_recovery_covers_every_required_stress_axis() -> None:
    detail, summary = run_synthetic_recovery_benchmark(
        seed=123,
        replicates=1,
        n_train=80,
        n_test=40,
        lambda_value=0.1,
    )

    assert len(detail) == 12
    assert {"baseline", "noise", "collinearity", "missingness", "outliers", "combined"}.issubset(
        set(detail["scenario_family"])
    )
    assert detail["clinical_validity_claim_allowed"].eq(False).all()
    assert detail["semantic_macro_train_scale_normalized_rmse"].notna().all()
    assert detail["transition_relative_frobenius_error"].ge(0).all()
    assert len(summary) == 12
