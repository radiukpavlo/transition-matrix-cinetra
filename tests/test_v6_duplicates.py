"""Declared duplicate images and cluster bootstraps (PI decision of September 28, 2026; protocol/amendments.md;
SPECIFICATION.md 4.5, 12.2, 13.1 and 15.2 acceptance test 17).  Synthetic data only."""
from __future__ import annotations

import copy

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

import fixtures_v6 as fx
from conftest import SEED_SYNTHETIC
from tmmri import stages
from tmmri.classification import domain_shift_audit
from tmmri.config import load_study
from tmmri.metrics import _cluster_codes, _cluster_draw_weights, _resampling_clusters, paired_error_tests, paired_nmae_difference, primary_rho
from tmmri.rule_pipeline import downstream_refit_bootstrap
from tmmri.splits import (apply_duplicate_declarations, build_split_manifest_v6, declared_duplicate_findings, detect_image_duplicates,
                          pair_cluster_id, partition_cluster_report)
from tmmri.stages import CohortValidationError, StageContext, stage_sources, validate_cohort
from tmmri.wedd import fit_cuts, stability_gate

FAMILY = ["lv_edv_ml", "lv_esv_ml", "lv_ef_percent", "rv_edv_ml", "rv_esv_ml", "rv_ef_percent", "lv_myocardial_mass_ed_g"]


@pytest.fixture(scope="module")
def study():
    return load_study()


# ----------------------------------------------------------------------------- declarations and partitions


def _acdc_release() -> pd.DataFrame:
    rows = []
    for i in range(1, 151):
        cls = ("DCM", "HCM", "MINF", "NOR", "RV")[((i - 1) % 100) // 20] if i <= 100 else ("DCM", "HCM", "MINF", "NOR", "RV")[(i - 101) // 10]
        pid = f"patient{i:03d}"
        rows.append({"sample_id": f"ACDC:{pid}", "dataset": "ACDC", "case_id": pid, "patient_id": pid, "release_split": "training" if i <= 100 else "testing",
                     "partition": "acdc_dev" if i <= 100 else "acdc_test", "acdc_class": cls, "eligible_v6": True, "exclusion_reason": None})
    return pd.DataFrame(rows)


def _mms_cases() -> pd.DataFrame:
    cases = [("A7G0P5", "Training/Labeled", "B", "mms_refit_train", False), ("K3R0Y7", "Validation", "C", "mms_refit_eval", True),
             ("C8J7L5", "Validation", "C", "mms_refit_eval", True), ("C8O0P2", "Testing", "C", "mms_refit_eval", True),
             ("X1X1X1", "Testing", "D", "mms_refit_eval", False)]
    return pd.DataFrame([{"sample_id": f"MMS:{c}", "dataset": "MMS", "case_id": c, "patient_id": c, "release_split": split, "vendor": vendor,
                          "centre": 2 if vendor == "B" else (4 if vendor == "C" else 5), "eligible_v6": True, "exclusion_reason": None,
                          "mms100_member": member, "never_analysed": not member, "mms_external": True, "refit_partition": refit}
                         for c, split, vendor, refit, member in cases])


def test_declared_duplicates_are_excluded_or_clustered_as_configured(study):
    acdc = apply_duplicate_declarations(_acdc_release(), "ACDC", study.cohorts)
    mms = apply_duplicate_declarations(_mms_cases(), "MMS", study.cohorts)
    a = acdc.set_index("case_id")
    assert a.loc[~a["eligible_v6"], "exclusion_reason"].to_dict() == {"patient055": "DUPLICATE_IMAGE_OF_TEST_CASE",
                                                                         "patient076": "DUPLICATE_IMAGE_WITHIN_PARTITION"}
    assert bool(a.at["patient118", "eligible_v6"]) and bool(a.at["patient074", "eligible_v6"])
    assert a.at["patient118", "cluster_id"] == a.at["patient055", "cluster_id"] == pair_cluster_id("ACDC", ["patient118", "patient055"])
    singles = a.drop(index=["patient055", "patient118", "patient074", "patient076"])
    assert (singles["cluster_id"] == singles["sample_id"]).all()
    m = mms.set_index("case_id")
    assert bool(m.at["K3R0Y7", "mms_external"]) and pd.isna(m.at["K3R0Y7", "refit_partition"])
    assert m.at["K3R0Y7", "duplicate_reason_code"] == "DUPLICATE_IMAGE_OF_REFIT_TRAIN_CASE" and m.at["A7G0P5", "refit_partition"] == "mms_refit_train"
    assert m.at["A7G0P5", "cluster_id"] == m.at["K3R0Y7", "cluster_id"] == "MMS:A7G0P5+K3R0Y7"
    assert m.at["C8J7L5", "cluster_id"] == m.at["C8O0P2", "cluster_id"] == "MMS:C8J7L5+C8O0P2"
    assert m.at["C8J7L5", "refit_partition"] == m.at["C8O0P2", "refit_partition"] == "mms_refit_eval"
    pd.testing.assert_frame_equal(apply_duplicate_declarations(acdc, "ACDC", study.cohorts), acdc)   # idempotent
    split = build_split_manifest_v6(acdc, mms, study.splits)
    s = split.set_index("sample_id")
    assert s.at["ACDC:patient055", "partition"] == "excluded" and pd.isna(s.at["ACDC:patient055", "outer_fold"])
    assert int((s["partition"] == "acdc_dev").sum()) == 98 and s.loc[s["partition"] == "acdc_dev", "outer_fold"].notna().all()
    report = partition_cluster_report(split, mms, ["vendor", "release_split", "mms100_member"], study.splits["refit_arm"]["eval_strata"])
    parts = report["partitions"]
    assert parts["acdc_dev"] == {"cases": 98, "clusters": 98} and parts["acdc_test"] == {"cases": 50, "clusters": 50}
    assert parts["mms_external"] == {"cases": 5, "clusters": 3} and parts["mms_refit_eval"] == {"cases": 3, "clusters": 2}
    assert parts["mms100_member"] == {"cases": 3, "clusters": 2} and parts["mms_external_never_analysed"] == {"cases": 2, "clusters": 2}
    assert report["mms_external_strata"]["vendor"]["C"] == {"cases": 3, "clusters": 2}
    assert sum(f["cases"] for f in report["acdc_dev_outer_folds"].values()) == 98


def test_detection_compares_detected_and_declared_frames(study):
    def rec(sid, ed, es):
        return {"sample_id": sid, "staged": {"ed_image": {"voxel_sha256": ed}, "es_image": {"voxel_sha256": es}, "ed_mask": {"voxel_sha256": sid}}}

    records = [rec("MMS:C8J7L5", "e1", "s1"), rec("MMS:C8O0P2", "e1", "s2"), rec("MMS:A7G0P5", "e2", "s3"), rec("MMS:K3R0Y7", "e2", "s3"),
               rec("ACDC:patient001", "e9", "s9"), rec("ACDC:patient002", "s9", "x")]
    detected = detect_image_duplicates(records)
    assert detected == {("ACDC:patient001", "ACDC:patient002"): ["ES/ED"], ("MMS:A7G0P5", "MMS:K3R0Y7"): ["ED", "ES"], ("MMS:C8J7L5", "MMS:C8O0P2"): ["ED"]}
    staged = {r["sample_id"] for r in records}
    findings = declared_duplicate_findings(detected, staged, study.cohorts)
    assert findings["undeclared"] == [["ACDC:patient001", "ACDC:patient002", ["ES/ED"]]] and findings["frame_mismatch"] == []
    # A declared pair whose staged members no longer match on the declared frames is a finding as well.
    lost = declared_duplicate_findings({k: v for k, v in detected.items() if "MMS:C8J7L5" not in k}, staged, study.cohorts)
    assert lost["frame_mismatch"] == [["MMS:C8J7L5", "MMS:C8O0P2", {"declared": ["ED"], "detected": []}]]


def _duplicate_source(tmp_path):
    """Synthetic ACDC release in which patient002's ED and ES images are voxel copies of patient001's (new files,
    different source hashes, own masks) plus the synthetic M&Ms release."""
    acdc = fx.make_acdc_source(tmp_path / "src" / "acdc", fx.acdc_patients(1))
    for frame in (1, 3):
        src = nib.load(str(acdc / "training" / "patient001" / f"patient001_frame{frame:02d}.nii.gz"))
        header = src.header.copy()
        header["descrip"] = b"voxel copy"
        nib.save(nib.Nifti1Image(np.asanyarray(src.dataobj), src.affine, header), str(acdc / "training" / "patient002" / f"patient002_frame{frame:02d}.nii.gz"))
    mms = fx.make_mms_source(tmp_path / "src" / "mms")
    return acdc, mms


def _context(study, acdc, mms, data_root):
    return StageContext(study=study, run=None, data_root=data_root, n_jobs=1, sources={"acdc": acdc, "mms": mms}, allow_unfrozen_synthetic=True,
                        source_manifests={"ACDC": fx.acdc_manifest(acdc), "MMS": fx.mms_manifest(mms)}, mms_metadata=fx.mms_metadata(mms))


def test_undeclared_duplicate_fails_strict_validation_and_a_declared_one_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "protocol_frozen", lambda *a, **k: False)
    acdc, mms = _duplicate_source(tmp_path)
    base = load_study()
    ctx = _context(base, acdc, mms, tmp_path / "data")
    assert stage_sources(ctx, "acdc").ok and stage_sources(ctx, "mms").ok
    with pytest.raises(CohortValidationError) as info:
        validate_cohort(ctx, strict=True)
    failed = {c["check"]: c for c in info.value.report["checks"] if not (c["passed"] or c.get("waived"))}
    assert set(failed) == {"decoded_voxel_duplicates_within_and_across_cohorts"}
    assert failed["decoded_voxel_duplicates_within_and_across_cohorts"]["observed"] == [["ACDC:patient001", "ACDC:patient002", ["ED", "ES"]]]

    declared = copy.deepcopy(base)
    declared.configs["cohorts"]["duplicate_image_pairs"]["pairs"].append(
        {"id": "SYN_DUP", "cohort": "acdc", "ids": ["patient001", "patient002"], "matched_frames": ["ED", "ES"], "action": "exclude",
         "exclude": {"case": "patient002", "from": "acdc_dev"}, "reason_code": "DUPLICATE_IMAGE_WITHIN_PARTITION"})
    report = validate_cohort(_context(declared, acdc, mms, tmp_path / "data"), strict=True)
    assert report["passed"] and not [c for c in report["checks"] if not (c["passed"] or c.get("waived"))]
    split = pd.read_parquet(tmp_path / "data" / "manifests" / "cohort" / "split_manifest.parquet").set_index("sample_id")
    assert split.at["ACDC:patient002", "partition"] == "excluded" and pd.isna(split.at["ACDC:patient002", "outer_fold"])
    assert split.at["ACDC:patient002", "duplicate_reason_code"] == "DUPLICATE_IMAGE_WITHIN_PARTITION"
    assert report["cluster_counts"]["partitions"]["acdc_dev"] == {"cases": 4, "clusters": 4}
    assert [r["detected_frames"] for r in report["duplicate_image_pairs"] if r["pair_id"] == "SYN_DUP"] == [["ED", "ES"]]

    wrong = copy.deepcopy(declared)
    wrong.configs["cohorts"]["duplicate_image_pairs"]["pairs"][-1]["matched_frames"] = ["ED"]
    with pytest.raises(CohortValidationError) as info:
        validate_cohort(_context(wrong, acdc, mms, tmp_path / "data"), strict=True)
    assert {c["check"] for c in info.value.report["checks"] if not (c["passed"] or c.get("waived"))} == {"declared_duplicate_image_pairs_detected"}


# ----------------------------------------------------------------------------- cluster bootstraps


def _rho_data(n: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = [f"c{i:03d}" for i in range(n)]
    truth = pd.DataFrame(rng.normal(100.0, 10.0, (n, len(FAMILY))), columns=FAMILY, index=idx)
    ridge = truth + rng.normal(0.0, 4.0, truth.shape)
    mean = pd.DataFrame(np.tile(truth.mean().to_numpy(), (n, 1)), columns=FAMILY, index=idx)
    return truth, ridge, mean, truth.std(ddof=1)


def _doubled(frame: pd.DataFrame) -> pd.DataFrame:
    """Every row twice (``<id>a``, ``<id>b``): 20 duplicate pairs."""
    out = pd.concat([frame.set_axis([f"{i}a" for i in frame.index]), frame.set_axis([f"{i}b" for i in frame.index])])
    return out.loc[sorted(out.index)]


def test_cluster_bootstrap_equals_case_bootstrap_for_singletons_and_resamples_pairs_together():
    truth, ridge, mean, sd = _rho_data(20, SEED_SYNTHETIC)
    kw = dict(family=FAMILY, n_boot=300, seed=20260928, min_estimable=5)
    case = primary_rho(truth, ridge, mean, sd, cluster=truth.index.to_numpy(), **kw)
    single = primary_rho(truth, ridge, mean, sd, cluster=truth.index.to_numpy(), resample_cluster=[f"MMS:{i}" for i in truth.index], **kw)
    assert {k: v for k, v in single.items() if k != "resampling_unit"} == {k: v for k, v in case.items() if k != "resampling_unit"}
    # 40 rows = 20 duplicated pairs: the pair bootstrap reproduces the 20-case bootstrap exactly; the naive case bootstrap
    # of the 40 rows treats the copies as independent and gives a narrower interval.
    t2, r2, m2 = _doubled(truth), _doubled(ridge), _doubled(mean)
    pairs = primary_rho(t2, r2, m2, sd, cluster=t2.index.to_numpy(), resample_cluster=[i[:-1] for i in t2.index], **kw)
    naive = primary_rho(t2, r2, m2, sd, cluster=t2.index.to_numpy(), **kw)
    assert pairs["n_units"] == 40 and pairs["n_clusters"] == 20 and naive["n_clusters"] == 40
    assert pairs["rho"] == pytest.approx(case["rho"]) and (pairs["ci_lower"], pairs["ci_upper"]) == pytest.approx((case["ci_lower"], case["ci_upper"]))
    assert naive["ci_upper"] - naive["ci_lower"] < case["ci_upper"] - case["ci_lower"]
    codes, n_units = _cluster_codes(np.array(["a1", "a2", "b", "c"]), 4)
    unit_cluster, n_clusters = _resampling_clusters(codes, n_units, ["A", "A", "b", "c"])
    assert n_clusters == 3
    for s in range(50):
        w = _cluster_draw_weights(np.random.default_rng(s), unit_cluster, n_clusters)
        assert w[0] == w[1]                                       # the pair is always drawn together
    with pytest.raises(ValueError, match="spans two"):
        _resampling_clusters(*_cluster_codes(np.array(["u", "u"]), 2), ["A", "B"])
    diff_case = paired_nmae_difference(truth, ridge, mean, FAMILY, sd, n_boot=200, seed=7)
    diff_pairs = paired_nmae_difference(t2, r2, m2, FAMILY, sd, n_boot=200, seed=7, cluster=t2.index.to_numpy(), resample_cluster=[i[:-1] for i in t2.index])
    pd.testing.assert_frame_equal(diff_case.drop(columns=["n", "n_clusters"]), diff_pairs.drop(columns=["n", "n_clusters"]))


def test_cluster_level_wilcoxon_counts_a_pair_once():
    truth, ridge, mean, sd = _rho_data(20, SEED_SYNTHETIC + 1)
    kw = dict(family="F3_external_transport_mms_external", partition="mms_external", comparison="mean_vs_ridge_per_target")
    case = paired_error_tests(truth, ridge, mean, FAMILY, sd, **kw)
    single = paired_error_tests(truth, ridge, mean, FAMILY, sd, cluster=list(truth.index), **kw)
    pd.testing.assert_frame_equal(case, single)
    t2, r2, m2 = _doubled(truth), _doubled(ridge), _doubled(mean)
    pairs = paired_error_tests(t2, r2, m2, FAMILY, sd, cluster=[i[:-1] for i in t2.index], **kw)
    assert (pairs["n"] == 40).all() and (pairs["n_clusters"] == 20).all()
    assert pairs["p_value"].to_numpy() == pytest.approx(case["p_value"].to_numpy()) and pairs["effect"].to_numpy() == pytest.approx(case["effect"].to_numpy())


def test_shift_audit_and_threshold_bootstraps_resample_clusters(study):
    rng = np.random.default_rng(SEED_SYNTHETIC + 2)
    ref, cmp_ = rng.normal(size=(40, 4)), rng.normal(0.8, 1.0, size=(30, 4))
    ids_ref, ids_cmp = [f"ACDC:r{i:02d}" for i in range(40)], [f"MMS:c{i:02d}" for i in range(30)]
    cfg = study.bootstrap["shift_audit"]
    plain, _ = domain_shift_audit(ref, cmp_, ids_ref, ids_cmp, cfg, n_boot=100, seed=11, min_cases=20)
    single, _ = domain_shift_audit(ref, cmp_, ids_ref, ids_cmp, cfg, n_boot=100, seed=11, min_cases=20, reference_clusters=ids_ref, comparison_clusters=ids_cmp)
    assert (single["ci_lower"], single["ci_upper"], single["auroc"]) == (plain["ci_lower"], plain["ci_upper"], plain["auroc"])
    paired = ["MMS:pair" if i in (0, 1) else s for i, s in enumerate(ids_cmp)]
    clustered, table = domain_shift_audit(ref, cmp_, ids_ref, ids_cmp, cfg, n_boot=100, seed=11, min_cases=20, comparison_clusters=paired)
    assert clustered["n_comparison_clusters"] == 29 and clustered["n_reference_clusters"] == 40 and clustered["fold_unit"] == "cluster_id"
    pair_folds = table.set_index("sample_id").loc[["MMS:c00", "MMS:c01"], "fold"]
    assert pair_folds.nunique() == 1 and np.isfinite(clustered["auroc"])
    x = np.r_[rng.normal(80.0, 3.0, 30), rng.normal(160.0, 3.0, 30)]
    values = pd.DataFrame({"lv_edv_ml": x, "lv_edv_ml__state": "OBSERVED"}, index=[f"ACDC:p{i:02d}" for i in range(60)])
    decision = pd.Series(["NOR"] * 30 + ["DCM"] * 30, index=values.index)
    kw = dict(seed=20260928, n_replicates=30, features=["lv_edv_ml"])
    base = stability_gate(values, decision, "wedd_gaussian_mm_v1", study.wedd, **kw)
    single = stability_gate(values, decision, "wedd_gaussian_mm_v1", study.wedd, groups=list(values.index), **kw)
    pd.testing.assert_frame_equal(base, single)


def test_downstream_refit_bootstrap_with_singleton_clusters_matches_patient_bootstrap(study):
    rng = np.random.default_rng(SEED_SYNTHETIC + 3)
    n = 40
    targets = study.transition_targets()
    b = pd.DataFrame({t: rng.normal(50.0, 5.0, n) for t in targets})
    b["lv_edv_ml"] = np.r_[rng.normal(80.0, 3.0, n // 2), rng.normal(160.0, 3.0, n // 2)]
    for t in targets:
        b[f"{t}__state"] = "OBSERVED"
    decision = np.array(["NOR"] * (n // 2) + ["DCM"] * (n // 2), dtype=object)
    ids = [f"ACDC:p{i:03d}" for i in range(n)]
    values = b[["lv_edv_ml", "lv_edv_ml__state"]].set_axis(ids)
    cuts = fit_cuts(values, pd.Series(decision, index=ids), "wedd_gaussian_mm_v1", study.wedd)
    from tmmri.rough_sets import induce_vprs_rules

    reference = induce_vprs_rules(cuts.transform(values), pd.Series(decision, index=ids), pd.Series(ids, index=ids), study.rule_policy,
                                  registry=study.registry, attributes=["lv_edv_ml"], consequent_kind="measured_oracle", rulebook_id="measured_oracle")
    folds = np.tile(np.arange(1, 6), n // 5).astype(float)
    kw = dict(A=rng.normal(size=(n, 57)), B=b, folds=folds, inner_labels=lambda f: np.where(folds == f, np.nan, folds), row_ids=ids, study=study,
              rulebooks={"measured_oracle": {"substrate": "measured", "decision": decision, "attributes": ["lv_edv_ml"], "reference": reference, "cuts": cuts}},
              profile="wedd_gaussian_mm_v1", n_replicates=3, seed=20260928, n_jobs=1)
    pd.testing.assert_frame_equal(downstream_refit_bootstrap(**kw), downstream_refit_bootstrap(**kw, clusters=ids))
    pairs = [f"pair{i // 2}" for i in range(n)]
    table = downstream_refit_bootstrap(**kw, clusters=pairs)
    assert (table["n_valid_replicates"] == 3).all()


def test_cluster_bootstraps_do_not_depend_on_n_jobs(study):
    truth, ridge, mean, sd = _rho_data(24, SEED_SYNTHETIC + 4)
    clusters = [f"k{i // 2}" if i < 8 else f"s{i}" for i in range(24)]
    kw = dict(family=FAMILY, n_boot=120, seed=20260928, min_estimable=5, cluster=truth.index.to_numpy(), resample_cluster=clusters)
    assert primary_rho(truth, ridge, mean, sd, n_jobs=1, **kw) == primary_rho(truth, ridge, mean, sd, n_jobs=3, **kw)
    x = np.r_[np.random.default_rng(5).normal(80.0, 3.0, 24), np.random.default_rng(6).normal(160.0, 3.0, 24)]
    values = pd.DataFrame({"lv_edv_ml": x}, index=[f"ACDC:p{i:02d}" for i in range(48)])
    decision = pd.Series(["NOR"] * 24 + ["DCM"] * 24, index=values.index)
    groups = [f"g{i // 2}" for i in range(48)]
    kw = dict(seed=20260928, n_replicates=24, features=["lv_edv_ml"], groups=groups)
    pd.testing.assert_frame_equal(stability_gate(values, decision, "wedd_gaussian_mm_v1", study.wedd, n_jobs=1, **kw),
                                  stability_gate(values, decision, "wedd_gaussian_mm_v1", study.wedd, n_jobs=3, **kw))


def test_shift_audit_folds_keep_declared_pairs_together_and_equal_case_folds_for_singletons():
    from tmmri.classification import hash_rule_folds

    ids = [f"MMS:C{i:03d}" for i in range(40)]
    domains = np.ones(len(ids), dtype=int)
    per_case = hash_rule_folds(ids, domains, "shift-audit-v6|", 5)
    assert np.array_equal(hash_rule_folds(ids, domains, "shift-audit-v6|", 5, cluster_keys=ids), per_case)
    keys = list(ids)
    keys[7] = keys[31] = "MMS:pair:C007+C031"
    clustered = hash_rule_folds(ids, domains, "shift-audit-v6|", 5, cluster_keys=keys)
    assert clustered[7] == clustered[31]
    assert np.array_equal(hash_rule_folds(ids[::-1], domains, "shift-audit-v6|", 5, cluster_keys=keys[::-1]), clustered[::-1])
