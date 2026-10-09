"""Decision D36: the H4 family (H4a rho_EF, H4b H-a delta-rho) on planted synthetic data only (never cohort data).

The estimands reuse the H-primary-3 code (:func:`tmmri.metrics.primary_rho`), the S3 harmonization and the paired
cluster bootstrap; each test is judged by its two-sided 97.5 % interval (Bonferroni within the family).  The stage
``evaluate-h4`` runs end to end in ``tests/test_mms2_pipeline.py``; Table 3 rows in ``tests/test_v601_reporting.py``.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from test_l57_trust import _synthetic_b
from tmmri import confirmatory_h4 as h4
from tmmri.config import load_study
from tmmri.metrics import primary_rho
from tmmri.stages import BundleError

N = 180                     # about the size of mms2_external, every case its own cluster
EF = ["lv_ef_percent", "rv_ef_percent"]


@pytest.fixture(scope="module")
def study():
    return load_study()


@pytest.fixture(scope="module")
def cfg(study):
    return h4.h4_config(study)


def _boot(cfg, hypothesis: str, n_boot: int = 400) -> dict:
    return {"n_boot": n_boot, "seed": h4.sub_seed(int(cfg["bootstrap"]["seed"]), hypothesis), "level": h4.interval_level(cfg),
            "min_valid_fraction": 0.95}


def _clusters(n: int = N) -> dict:
    ids = [f"S{i:03d}" for i in range(n)]
    return {"cluster": ids, "resample_cluster": ids}


def _planted(study, *, good: list[str], seed: int = 11) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Truth around the development mean; targets in ``good`` predicted with small error, the others with large error."""

    family = study.primary_family()
    rng = np.random.default_rng(seed)
    mean = pd.Series({t: 50.0 + 5.0 * j for j, t in enumerate(family)})
    sd = pd.Series({t: 10.0 for t in family})
    truth = pd.DataFrame({t: mean[t] + sd[t] * rng.normal(size=N) for t in family}, index=[f"X:S{i:03d}" for i in range(N)])
    pred = pd.DataFrame({t: truth[t] + (0.15 if t in good else 2.5) * sd[t] * rng.normal(size=N) for t in family}, index=truth.index)
    return truth, pred, mean, sd


def test_registered_values_sub_seeds_and_bonferroni_level(study, cfg):
    hyp = cfg["hypotheses"]
    assert cfg["partition"] == "mms2_external" and cfg["descriptive_partitions"] == ["acdc_test", "mms_external"]
    assert hyp["h4a"]["targets"] == EF and set(EF) <= set(study.primary_family()) and hyp["h4a"]["threshold"] == 0.10
    assert hyp["h4b"]["harmonization"] == "H-a" and hyp["h4b"]["threshold"] == 0.0
    assert (cfg["bootstrap"]["replicates"], cfg["bootstrap"]["seed"], cfg["bootstrap"]["unit"]) == (2000, 20260928, "cluster_id")
    assert h4.interval_level(cfg) == pytest.approx(0.975)                         # Bonferroni: alpha 0.05 / 2 tests, two-sided
    seeds = {k: h4.sub_seed(20260928, k) for k in ("H4a", "H4b")}
    assert seeds == {"H4a": 1958159060, "H4b": 1591327988} and len(set(seeds.values()) | {20260928}) == 3
    assert seeds["H4a"] == int(hashlib.sha256(b"20260928|H4a").hexdigest()[:8], 16)
    assert all(f"{k} {v}" in cfg["bootstrap"]["sub_seed_rule"] for k, v in seeds.items())


def test_h4a_passes_when_the_ef_targets_transfer_although_the_family_fails(study, cfg):
    truth, pred, mean, sd = _planted(study, good=EF)
    family = study.primary_family()
    out = h4.rho_ef(truth, pred, mean, sd, cfg["hypotheses"]["h4a"]["targets"], family=family, **_clusters(), **_boot(cfg, "H4a"))
    assert out["status"] == "EVALUATED" and out["family"] == EF and out["min_estimable"] == 2 and out["level"] == pytest.approx(0.975)
    assert out["rho"] > 0.7 and out["ci_lower"] > 0.10 and out["interval"] == "percentile_two_sided_97.5"
    decision = h4.decide(cfg["hypotheses"]["h4a"], out, "rho", confirmatory=True)
    assert decision["status"] == "PASSED" and decision["passed"] is True and decision["tested"] is True
    # the same truth through the H-primary family estimand (5 volume/mass targets badly predicted) does not pass
    null = pd.DataFrame([mean[family]] * N, index=truth.index)
    whole = primary_rho(truth, pred, null, sd, family, 400, 20260928, _clusters()["cluster"], 5)
    assert whole["ci_lower"] <= 0.10
    # 97.5 % interval contains the 95 % interval of the same code and sub-seed
    narrow = primary_rho(truth[EF], pred[EF], null[EF], sd, EF, 400, out["seed"], _clusters()["cluster"], 2, resample_cluster=_clusters()["resample_cluster"])
    assert narrow["rho"] == pytest.approx(out["rho"]) and out["ci_lower"] <= narrow["ci_lower"] and out["ci_upper"] >= narrow["ci_upper"]
    descriptive = h4.decide(cfg["hypotheses"]["h4a"], out, "rho", confirmatory=False)
    assert descriptive["status"] == "DESCRIPTIVE" and descriptive["criterion_met"] is True and descriptive["passed"] is None


def test_h4a_fails_when_only_the_volumes_transfer(study, cfg):
    volumes = [t for t in study.primary_family() if t not in EF]
    truth, pred, mean, sd = _planted(study, good=volumes)
    out = h4.rho_ef(truth, pred, mean, sd, EF, family=study.primary_family(), **_clusters(), **_boot(cfg, "H4a"))
    assert out["status"] == "EVALUATED" and out["rho"] < 0 and out["ci_lower"] <= 0.10
    assert h4.decide(cfg["hypotheses"]["h4a"], out, "rho", confirmatory=True)["status"] == "FAILED"


def test_h4a_is_not_estimable_without_both_ef_targets(study, cfg):
    family = study.primary_family()
    truth, pred, mean, sd = _planted(study, good=EF)
    missing = truth.assign(rv_ef_percent=np.nan)
    out = h4.rho_ef(missing, pred, mean, sd, EF, family=family, **_clusters(), **_boot(cfg, "H4a"))
    assert out["status"] == "NE" and out["reason"] == "fewer_than_min_estimable_targets" and out["rho"] is None
    decision = h4.decide(cfg["hypotheses"]["h4a"], out, "rho", confirmatory=True)
    assert decision["status"] == "NE" and decision["passed"] is False and decision["criterion_met"] is None
    assert h4.decide(cfg["hypotheses"]["h4a"], out, "rho", confirmatory=False)["status"] == "NE"
    absent = h4.rho_ef(truth.drop(columns=["lv_ef_percent"]), pred, mean, sd, EF, family=family, **_clusters(), **_boot(cfg, "H4a"))
    assert absent["status"] == "NE" and absent["reason"] == h4.EF_NOT_RECORDED
    outside = h4.rho_ef(truth, pred, mean, sd, EF, family=[t for t in family if t != "rv_ef_percent"], **_clusters(), **_boot(cfg, "H4a"))
    assert outside["status"] == "NE" and outside["reason"] == h4.EF_NOT_IN_FAMILY


def test_h4b_passes_on_a_pure_location_scale_shift_and_not_without_one(study, cfg):
    from tmmri.transition_fit import fit_transition_v6

    rng = np.random.default_rng(29)
    targets, family = study.transition_targets(), study.primary_family()
    a_dev = rng.normal(size=(80, 8))
    b_dev = _synthetic_b(a_dev, targets, rng)
    model = fit_transition_v6(a_dev, b_dev, targets, study.transition, study.registry, np.arange(80) % 5 + 1)
    mean, sd = b_dev[family].mean(), b_dev[family].std(ddof=1)
    a_target = rng.normal(size=(N, 8))
    index = [f"X:S{i:03d}" for i in range(N)]
    truth = _synthetic_b(a_target, targets, rng).set_axis(index)
    cols = [f"x{j}" for j in range(8)]
    results = {}
    for name, shift, scale in (("shifted", rng.normal(size=8) * 3.0, np.exp(rng.normal(size=8))), ("unshifted", np.zeros(8), np.ones(8))):
        observed = pd.DataFrame(shift + scale * a_target, columns=cols, index=index)
        frozen = model.predict(observed.to_numpy())
        frozen.index = index
        harmonized, record = h4.harmonize_h_a(model, observed, forbidden=[*targets, "acdc_class"])
        assert record["variant"] == "H-a" and record["groups"]["all"]["n_rows"] == N and "statistics" in record
        results[name] = h4.delta_rho(truth, frozen, harmonized, mean, sd, family, **_clusters(), min_estimable=5, **_boot(cfg, "H4b"))
    shifted, unshifted = results["shifted"], results["unshifted"]
    assert shifted["status"] == "EVALUATED" and shifted["difference"] > 0.2 and shifted["ci_lower"] > 0
    assert shifted["direction"] == "comparator_minus_reference" and shifted["comparator"] == "H-a" and shifted["level"] == pytest.approx(0.975)
    assert h4.decide(cfg["hypotheses"]["h4b"], shifted, "difference", confirmatory=True)["status"] == "PASSED"
    assert unshifted["difference"] < 0.05 and unshifted["ci_lower"] <= 0                     # no shift: nothing to recover (sampling noise only)
    assert h4.decide(cfg["hypotheses"]["h4b"], unshifted, "difference", confirmatory=True)["status"] == "FAILED"
    empty, record = h4.harmonize_h_a(model, pd.DataFrame(np.nan, index=index, columns=cols))
    assert empty is None and record["status"] == "NE" and record["reason"] == h4.HARMONIZATION_NOT_ESTIMABLE


def test_family_summary_and_the_once_only_decision_record(cfg, tmp_path):
    hyp = cfg["hypotheses"]
    results = [h4.decide(hyp["h4a"], {"rho": 0.4, "ci_lower": 0.2, "ci_upper": 0.5, "interval": "percentile_two_sided_97.5"}, "rho", confirmatory=True),
               h4.decide(hyp["h4b"], {"difference": 0.01, "ci_lower": -0.02, "ci_upper": 0.04}, "difference", confirmatory=True)]
    summary = h4.family_summary(results, cfg, confirmatory=True)
    assert summary["passed"] == ["H4a"] and summary["failed"] == ["H4b"] and summary["multiplicity"]["per_test_alpha"] == pytest.approx(0.025)
    path = tmp_path / "metrics" / h4.RECORD_DIR / "h4_mms2_external.json"
    h4.write_decision_record(path, {"results": results})
    before = path.read_bytes()
    with pytest.raises(h4.H4RecordExistsError, match="never overwritten"):
        h4.write_decision_record(path, {"results": []})
    assert path.read_bytes() == before and issubclass(h4.H4RecordExistsError, BundleError)   # exit 7 through the CLI
