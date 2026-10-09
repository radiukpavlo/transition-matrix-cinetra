"""Protocol-v6.0.1 trust analyses S1-S4 and the comparative estimands (lane L57) on synthetic arrays only."""

from __future__ import annotations

import inspect
import math

import numpy as np
import pandas as pd
import pytest

from tmmri.classification import _auroc_from_scores, fit_formal_predictor
from tmmri.config import load_study
from tmmri.experiments.alt_bundles import paired_predictor_differences, paired_rho_difference, predictor_metrics
from tmmri.experiments.trust.calibration import (brier, ece_equal_width, fit_temperature, logits_from_probabilities, reliability_table,
                                                 scaled_probabilities, softmax)
from tmmri.experiments.trust.harmonize import LabelLeakError, apply_harmonized, assert_label_free, fit_harmonization
from tmmri.experiments.trust.ood import MahalanobisReference, fit_mahalanobis_reference, flag_threshold
from tmmri.experiments.trust.selective import aurc, coverage_threshold, max_softmax, realised_at_threshold, risk_coverage_curve
from tmmri.metrics import primary_rho
from tmmri.transition_fit import fit_transition_v6

CLASSES = ["DCM", "HCM", "MINF", "NOR", "RV"]


# ----------------------------------------------------------------------------- S1


@pytest.mark.parametrize("n", [98, 50, 7])
def test_thresholds_give_the_planned_development_coverage(n):
    rng = np.random.default_rng(n)
    proba = softmax(rng.normal(scale=2.0, size=(n, 5)))
    conf = max_softmax(proba)
    for c in (0.5, 0.8):
        fit = coverage_threshold(conf, c)
        k = math.ceil(c * n)
        assert fit["k"] == k and fit["n_accepted_development"] == k
        assert fit["development_coverage"] == pytest.approx(k / n) and fit["development_coverage"] >= c
        accepted = conf >= fit["tau"]
        assert accepted.sum() == k and conf[~accepted].max(initial=-1) < fit["tau"]
        realised = realised_at_threshold(conf, np.ones(n), fit["tau"])
        assert realised["coverage"] == pytest.approx(k / n) and realised["n_accepted"] == k


def test_risk_coverage_curve_and_aurc_by_hand():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([1.0, 0.0, 1.0, 0.0])
    curve = risk_coverage_curve(conf, correct)
    assert curve["coverage"].tolist() == [0.25, 0.5, 0.75, 1.0]
    assert curve["risk"].tolist() == pytest.approx([0.0, 0.5, 1 / 3, 0.5])
    assert aurc(conf, correct) == pytest.approx(np.mean([0.0, 0.5, 1 / 3, 0.5]))
    tied = risk_coverage_curve(np.array([0.9, 0.9, 0.5]), np.array([1.0, 0.0, 1.0]))   # ties enter together
    assert tied["n_accepted"].tolist() == [2, 3] and tied["risk"].tolist() == pytest.approx([0.5, 1 / 3])
    r = realised_at_threshold(conf, np.array([1.0, 0.0, np.nan, 0.0]), 0.75)            # unknown reference: not evaluable
    assert r["n"] == 3 and r["n_accepted"] == 2 and r["accuracy_accepted"] == pytest.approx(0.5)


# ----------------------------------------------------------------------------- S2


def test_mahalanobis_flags_and_auroc_detect_a_planted_shift():
    rng = np.random.default_rng(7)
    cov = np.eye(12) + 0.4
    dev = rng.multivariate_normal(np.zeros(12), cov, size=98)
    folds = np.arange(98) % 5
    oof = np.empty(98)
    for k in range(5):
        oof[folds == k] = fit_mahalanobis_reference(dev[folds != k]).score(dev[folds == k])
    tau = flag_threshold(oof, 0.95)
    assert np.mean(oof > tau) == pytest.approx(0.05, abs=0.011)
    final = fit_mahalanobis_reference(dev)
    same = rng.multivariate_normal(np.zeros(12), cov, size=60)
    shifted = rng.multivariate_normal(np.tile([2.5, -2.5], 6), cov, size=60)
    scores = final.score(np.vstack([same, shifted]))
    is_shifted = np.r_[np.zeros(60), np.ones(60)].astype(int)
    assert _auroc_from_scores(is_shifted, scores) > 0.95
    assert np.mean(scores[60:] > tau) > 0.9 and np.mean(scores[:60] > tau) < 0.25
    again = MahalanobisReference.from_dict(final.to_dict())
    assert np.array_equal(again.score(shifted), final.score(shifted))


def test_pca_reference_for_wide_features_and_nonfinite_rows():
    rng = np.random.default_rng(11)
    dev = rng.normal(size=(40, 300))
    ref = fit_mahalanobis_reference(dev, pca_components=50)
    assert ref.dimension == 39 and ref.pca_components.shape == (39, 300)      # min(50, n - 1)
    x = rng.normal(size=(5, 300))
    x[2, 0] = np.nan
    s = ref.score(x)
    assert np.isnan(s[2]) and np.isfinite(np.delete(s, 2)).all()
    far = ref.score(rng.normal(loc=3.0, size=(5, 300)))
    assert np.median(far) > np.median(ref.score(rng.normal(size=(20, 300))))


# ----------------------------------------------------------------------------- S3


def _synthetic_b(a: np.ndarray, targets: list[str], rng: np.random.Generator) -> pd.DataFrame:
    cols = {t: 50.0 + 15.0 * np.tanh(0.5 * (a[:, j % a.shape[1]] + a[:, (j + 3) % a.shape[1]]) + 0.1 * rng.normal(size=a.shape[0]))
            for j, t in enumerate(targets)}
    return pd.DataFrame(cols)


def test_h_a_exactly_undoes_a_pure_location_scale_shift():
    study = load_study()
    rng = np.random.default_rng(3)
    a = rng.normal(size=(40, 8))
    targets = study.transition_targets()
    b = _synthetic_b(a, targets, rng)
    model = fit_transition_v6(a, b, targets, study.transition, study.registry, np.arange(40) % 5 + 1)
    y = np.array(CLASSES * 8, dtype=object)
    a_cls = a.copy()
    a_cls[np.arange(40), np.arange(40) % 5] += 2.0
    predictor = fit_formal_predictor(a_cls, y, study.formal["audited_predictor"])
    cols = [f"x{j}" for j in range(8)]
    for frozen_model, rows, method in ((model, a, "predict"), (predictor, a_cls, "predict_proba")):
        shift, scale = rng.normal(size=8) * 5.0, np.exp(rng.normal(size=8))
        target = pd.DataFrame(shift + scale * rows, columns=cols, index=[f"T:{i}" for i in range(40)])
        harm = fit_harmonization(target, cols, variant="H-a")
        out, record = apply_harmonized(frozen_model, target.to_numpy(), harm, target.index, predict=method)
        reference = getattr(frozen_model, method)(rows)
        assert np.allclose(out.to_numpy(dtype=float), np.asarray(reference, dtype=float), rtol=1e-9, atol=1e-9, equal_nan=True)
        assert record["groups"]["all"]["degenerate_kept_columns"] == []
        unadjusted = np.asarray(getattr(frozen_model, method)(target.to_numpy()), dtype=float)
        assert not np.allclose(unadjusted, np.asarray(reference, dtype=float), equal_nan=True)


def test_h_b_uses_vendor_groups_and_falls_back_for_small_vendors():
    rng = np.random.default_rng(5)
    cols = ["x0", "x1"]
    frame = pd.DataFrame(rng.normal(size=(30, 2)), columns=cols, index=[f"S:{i}" for i in range(30)])
    vendor = pd.Series(["A"] * 20 + ["B"] * 6 + [None] * 4, index=frame.index, name="vendor")
    harm = fit_harmonization(frame, cols, variant="H-b", vendor=vendor, min_rows_per_vendor=10)
    assert set(harm["groups"]) == {"all", "vendor:A"}
    assert {f["vendor"] for f in harm["fallback_groups"]} == {"B", "<missing>"}
    assert (harm["row_group"].iloc[:20] == "vendor:A").all() and (harm["row_group"].iloc[20:] == "all").all()


def test_harmonization_cannot_access_labels_or_b():
    cols = ["x0", "x1"]
    frame = pd.DataFrame(np.arange(12, dtype=float).reshape(6, 2), columns=cols)
    for leaked in ("acdc_class", "lv_ef_percent", "y_true"):
        with pytest.raises(LabelLeakError):
            fit_harmonization(frame.assign(**{leaked: 1.0}), cols, variant="H-a", forbidden=["lv_ef_percent"])
    with pytest.raises(LabelLeakError):           # a forbidden name never becomes an allowed representation column
        assert_label_free(["x0", "lv_ef_percent"], ["x0", "lv_ef_percent"], forbidden=["lv_ef_percent"])
    with pytest.raises(LabelLeakError):           # H-b reads vendor metadata only
        fit_harmonization(frame, cols, variant="H-b", vendor=pd.Series(["A"] * 6, name="pathology"))
    params = set(inspect.signature(fit_harmonization).parameters)
    assert params == {"a_frame", "columns", "variant", "vendor", "min_rows_per_vendor", "forbidden"}


# ----------------------------------------------------------------------------- S4


def test_temperature_recovers_a_planted_temperature_and_log_probabilities_agree():
    rng = np.random.default_rng(17)
    z = rng.normal(scale=3.0, size=(20000, 5))
    planted = 2.5
    p = softmax(z / planted)
    y = np.minimum((p.cumsum(axis=1) < rng.random(len(p))[:, None]).sum(axis=1), 4)
    fit = fit_temperature(z, y)
    assert fit["temperature"] == pytest.approx(planted, rel=0.05) and fit["nll_after"] < fit["nll_before"] and not fit["at_bound"]
    via_logp = fit_temperature(logits_from_probabilities(softmax(z)), y)
    assert via_logp["temperature"] == pytest.approx(fit["temperature"], rel=1e-6)
    after = scaled_probabilities(z, fit["temperature"])
    assert np.array_equal(after.argmax(axis=1), z.argmax(axis=1))            # accuracy unchanged by scaling


def test_brier_equal_width_ece_and_reliability_by_hand():
    p = np.array([[0.72, 0.28], [0.17, 0.83], [0.55, 0.45]])
    y = np.array([0, 0, 1])
    assert brier(p, y) == pytest.approx(np.mean([2 * 0.28 ** 2, 2 * 0.83 ** 2, 2 * 0.55 ** 2]))
    table = reliability_table(p, y, n_bins=10, binning="equal_width")
    assert table["count"].sum() == 3 and table.loc[table["count"] > 0, "bin_id"].tolist() == [5, 7, 8]
    assert ece_equal_width(p, y, n_bins=10) == pytest.approx((abs(0.55 - 0) + abs(0.72 - 1) + abs(0.83 - 0)) / 3)


# ----------------------------------------------------------------------------- paired comparisons


def test_paired_rho_difference_is_zero_for_identical_pipelines_and_matches_primary_rho():
    rng = np.random.default_rng(23)
    family = ["t1", "t2", "t3"]
    truth = pd.DataFrame(rng.normal(size=(30, 3)), columns=family)
    pred = truth + rng.normal(scale=0.5, size=(30, 3))
    null = pd.DataFrame(np.zeros((30, 3)), columns=family)
    sd = pd.Series(1.0, index=family)
    cluster = [f"p{i}" for i in range(30)]
    out = paired_rho_difference(truth, pred, pred, null, sd, family, cluster=cluster, resample_cluster=cluster, n_boot=64, seed=20260928, min_estimable=2)
    assert out["difference"] == 0.0 and out["ci_lower"] == 0.0 and out["ci_upper"] == 0.0
    ref = primary_rho(truth, pred, null, sd, family, 64, 20260928, cluster, 2, resample_cluster=cluster)
    assert out["rho_reference"] == pytest.approx(ref["rho"])
    better = paired_rho_difference(truth, pred, truth + 0.1 * (pred - truth), null, sd, family, cluster=cluster, resample_cluster=cluster,
                                   n_boot=64, seed=20260928, min_estimable=2)
    assert better["difference"] > 0 and better["ci_lower"] > 0


def test_predictor_metrics_add_brier_and_paired_differences_vanish_for_identical_predictors():
    rng = np.random.default_rng(29)
    truth = np.array(CLASSES * 8, dtype=object)
    z = rng.normal(size=(40, 5))
    z[np.arange(40), np.arange(40) % 5] += 2.0
    proba = pd.DataFrame(softmax(z), columns=CLASSES)
    clusters = np.array([f"c{i}" for i in range(40)], dtype=object)
    summary = predictor_metrics(truth, proba, CLASSES, clusters, n_boot=64, seed=20260928, n_bins=10, min_valid_fraction=0.5)
    t = np.arange(40) % 5
    assert summary["metrics"]["brier"]["estimate"] == pytest.approx(brier(proba.to_numpy(), t))
    assert summary["metrics"]["accuracy"]["estimate"] == pytest.approx(np.mean(proba.to_numpy().argmax(axis=1) == t))
    same = paired_predictor_differences(truth, proba, proba, CLASSES, clusters, n_boot=64, seed=20260928, n_bins=10, min_valid_fraction=0.5)
    assert all(m["difference"] == 0.0 for m in same["metrics"].values())
