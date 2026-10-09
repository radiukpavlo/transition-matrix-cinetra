"""Protocol 6.0.0 ingest: cohort tables, tracked manifests, frame convention, labels and the v6 fold rule."""
from __future__ import annotations

import hashlib
import json

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

import fixtures_v6 as fx
from tmmri.config import load_study
from tmmri.ingest.acdc import acdc_cohort_table
from tmmri.ingest.labels import DATASET_LABEL_MAPPINGS
from tmmri.ingest.mms import mms_cohort_table
from tmmri.ingest.sources import canonical_dataset, load_mms_manifest, load_mms_metadata_manifest
from tmmri.ingest.staging import load_staged_case, stage_dataset
from tmmri.splits import acdc_inner_folds_v6, acdc_outer_folds_v6, fold_balance, fold_digest, mms_refit_folds_v6


@pytest.fixture(scope="module")
def study():
    return load_study()


@pytest.fixture()
def acdc(tmp_path):
    root = fx.make_acdc_source(tmp_path / "acdc", fx.acdc_patients(1))
    return root, fx.acdc_manifest(root)


@pytest.fixture()
def mms(tmp_path):
    root = fx.make_mms_source(tmp_path / "mms")
    return root, fx.mms_manifest(root), fx.mms_metadata(root)


def test_dataset_ids_and_legacy_alias():
    assert canonical_dataset("acdc") == "ACDC" and canonical_dataset("MMS") == "MMS"
    assert canonical_dataset("MMS_100") == "MMS"
    assert DATASET_LABEL_MAPPINGS["MMS"] == DATASET_LABEL_MAPPINGS["MMS_100"]


def test_label_maps_match_cohort_config(study):
    assert DATASET_LABEL_MAPPINGS["ACDC"] == study.label_map("acdc")
    assert DATASET_LABEL_MAPPINGS["MMS"] == study.label_map("mms")


def test_tracked_full_release_manifest(study):
    manifest = load_mms_manifest()
    binding = study.source_binding("mms")["release_files_observed"]
    assert len(manifest) == binding["cases"] == 345
    assert int(manifest["eligible_v6"].sum()) == study.cohorts["mms"]["eligibility"]["expected_count"] == 320
    assert manifest.loc[~manifest["eligible_v6"], "exclusion_reason"].eq("MMS_RELEASE_UNLABELED").sum() == 25
    assert manifest.loc[~manifest["eligible_v6"], "release_split"].eq("Training/Unlabeled").all()
    dup = manifest[manifest["duplicate_of"] != ""].set_index("case_id")["duplicate_of"].to_dict()
    assert {tuple(sorted(p)) for p in study.splits["duplicate_control"]["known_duplicates"]} == {tuple(sorted(kv)) for kv in dup.items()}
    assert all(dup[dup[c]] == c for c in dup)
    assert int(manifest["mms100_member"].sum()) == 100
    assert int((manifest["eligible_v6"] & ~manifest["mms100_member"]).sum()) == 220
    total = int(manifest["image_bytes"].sum() + manifest["mask_bytes"].sum()) + sum(e.bytes for e in load_mms_metadata_manifest())
    assert total == binding["bytes_total"] == 15_380_121_327
    forbidden = {"pathology", "vendor", "centre", "center", "age", "sex", "height", "weight", "Pathology", "Vendor"}
    assert not forbidden & set(manifest.columns)
    assert manifest["image_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()


def test_acdc_cohort_table_zero_based(acdc):
    root, manifest = acdc
    table = acdc_cohort_table(root, manifest)
    assert list(table["sample_id"]) == sorted(table["sample_id"])
    assert table["eligible_v6"].all()
    assert (table["ed_frame_index"] == table["ed_frame_number"] - 1).all()
    assert (table["es_frame_index"] == 2).all() and (table["ed_frame_index"] == 0).all()
    assert set(table["partition"]) == {"acdc_dev"}
    assert table.loc[0, "ed_image_relpath"].endswith("patient001_frame01.nii.gz")


def test_acdc_info_cfg_corruption_is_recorded(acdc):
    root, manifest = acdc
    (root / "training" / "patient002" / "Info.cfg").write_text("ED: 1\nES: 3\nGroup: XXX\n", encoding="utf-8")
    table = acdc_cohort_table(root, manifest).set_index("case_id")
    assert table.loc["patient002", "exclusion_reason"] == "SOURCE_HASH_MISMATCH"
    assert not table.loc["patient002", "eligible_v6"] and table.drop("patient002")["eligible_v6"].all()


def test_mms_cohort_table(mms, registry):
    root, manifest, meta = mms
    table = mms_cohort_table(root, manifest, registry=registry, metadata=meta).set_index("case_id")
    assert not table.loc["U5U5U5", "eligible_v6"] and table.loc["U5U5U5", "exclusion_reason"] == "MMS_RELEASE_UNLABELED"
    assert table.loc["U5U5U5", "duplicate_of"] == "U6U6U6" and table.loc["U6U6U6", "duplicate_of"] == "U5U5U5"
    assert table.loc["A1A1A1", "mms100_member"] and not table.loc["A1A1A1", "never_analysed"]
    assert table.loc["B2B2B2", "never_analysed"] and table.loc["B2B2B2", "refit_partition"] == "mms_refit_eval"
    assert table.loc["A1A1A1", "refit_partition"] == "mms_refit_train"
    assert (table.loc["D4D4D4", "ed_frame_index"], table.loc["D4D4D4", "es_frame_index"]) == (4, 1)
    assert table.loc["C3C3C3", "crosswalk_compatibility"] == "OFF_ONTOLOGY"
    assert table.loc["B2B2B2", "crosswalk_acdc_class"] == "DCM"


def test_mms_info_csv_hash_is_enforced(mms):
    root, manifest, meta = mms
    fx.corrupt_one_byte(root / fx.MMS_INFO_CSV, offset=5)
    with pytest.raises(Exception, match="sha256 mismatch"):
        mms_cohort_table(root, manifest, metadata=meta)


def test_manifest_verification_catches_a_corrupted_byte(mms, tmp_path):
    root, manifest, meta = mms
    fx.corrupt_one_byte(root / "Validation" / "B2B2B2" / "B2B2B2_sa_gt.nii.gz")
    report = stage_dataset("mms", root, tmp_path / "staged", manifest, n_jobs=1, verify_only=True, metadata=meta)
    assert [(f["case_id"], f["reason_code"]) for f in report.failures] == [("B2B2B2", "SOURCE_HASH_MISMATCH")]
    assert report.actions["verified"] == len(manifest) + 1          # every case plus the information CSV
    assert not (tmp_path / "staged").exists()                       # verify-only writes no staged files


def test_frame_selection_is_zero_based(mms, acdc, tmp_path):
    root, manifest, meta = mms
    stage_dataset("mms", root, tmp_path / "staged", manifest, n_jobs=1, metadata=meta)
    case = load_staged_case("MMS", "D4D4D4", tmp_path / "staged")
    image = np.asanyarray(nib.load(str(root / "Testing" / "D4D4D4" / "D4D4D4_sa.nii.gz")).dataobj)
    assert (case.ed_frame_index, case.es_frame_index) == (4, 1)
    assert np.array_equal(case.ed_image, image[..., 4].astype(np.float32))
    assert np.array_equal(case.es_image, image[..., 1].astype(np.float32))
    aroot, amanifest = acdc
    stage_dataset("acdc", aroot, tmp_path / "staged", amanifest, n_jobs=1)
    acase = load_staged_case("ACDC", "patient001", tmp_path / "staged")
    frame01 = np.asanyarray(nib.load(str(aroot / "training" / "patient001" / "patient001_frame01.nii.gz")).dataobj)
    assert acase.ed_frame_index == 0 and acase.meta["frame_numbers_1based"] == {"ed": 1, "es": 3}
    assert np.array_equal(acase.ed_image, frame01.astype(np.float32))


def test_labels_are_canonicalized(mms, acdc, tmp_path):
    aroot, amanifest = acdc
    stage_dataset("acdc", aroot, tmp_path / "staged", amanifest, n_jobs=1)
    acase = load_staged_case("ACDC", "patient003", tmp_path / "staged")
    released = np.rint(np.asanyarray(nib.load(str(aroot / "training" / "patient003" / "patient003_frame01_gt.nii.gz")).dataobj)).astype(int)
    assert acase.ed_mask.dtype == np.int16
    assert np.array_equal(acase.ed_mask == 1, released == 3) and np.array_equal(acase.ed_mask == 3, released == 1)
    assert np.array_equal(acase.ed_mask == 2, released == 2)
    root, manifest, meta = mms
    stage_dataset("mms", root, tmp_path / "staged", manifest, n_jobs=1, metadata=meta)
    case = load_staged_case("MMS", "A1A1A1", tmp_path / "staged")          # uint8 mask with scl_slope 3/255
    scaled = np.asanyarray(nib.load(str(root / "Training" / "Labeled" / "A1A1A1" / "A1A1A1_sa_gt.nii.gz")).dataobj)[..., 0]
    assert not np.array_equal(scaled, np.rint(scaled))                    # reads as 1.00000002 etc.
    assert set(np.unique(case.ed_mask)) == {0, 1, 2, 3}
    assert np.array_equal(case.ed_mask, np.rint(scaled).astype(np.int16))
    assert case.meta["staged"]["ed_mask"]["dtype"] == "uint8"


def test_staged_geometry_uses_the_header_pixdim_fallback(acdc, tmp_path):
    root, manifest = acdc
    stage_dataset("acdc", root, tmp_path / "staged", manifest, n_jobs=1)
    case = load_staged_case("ACDC", "patient001", tmp_path / "staged")
    assert case.spacing_mm == pytest.approx((1.5625, 1.5625, 10.0))
    assert case.geometry_method == "nifti_header_pixdim_unscaled_affine"
    source = nib.load(str(root / "training" / "patient001" / "patient001_frame01_gt.nii.gz"))
    assert np.allclose(case.affine, source.affine)


def _acdc_dev_frame(n_per_class: int = 20) -> pd.DataFrame:
    rows, number = [], 1
    for cls in ("DCM", "HCM", "MINF", "NOR", "RV"):
        for _ in range(n_per_class):
            rows.append({"sample_id": f"ACDC:patient{number:03d}", "patient_id": f"patient{number:03d}", "acdc_class": cls, "partition": "acdc_dev"})
            number += 1
    return pd.DataFrame(rows)


def test_v6_fold_rule_is_deterministic_and_stratified(study):
    frame = _acdc_dev_frame()
    outer = acdc_outer_folds_v6(frame, study.splits)
    again = acdc_outer_folds_v6(frame.sample(frac=1.0, random_state=3), study.splits)
    pd.testing.assert_frame_equal(outer, again)
    balance = fold_balance(outer, "outer_fold", "acdc_class")
    assert set(balance.sum(axis=1)) == {20} and set(balance.to_numpy().ravel()) == {4}   # 100 synthetic patients, 20 per class
    dcm = sorted(frame.loc[frame["acdc_class"] == "DCM", "patient_id"], key=lambda p: hashlib.sha256(("acdc-fold-v6|" + p).encode()).hexdigest())
    by_patient = outer.set_index("patient_id")["outer_fold"]
    assert [int(by_patient[p]) for p in dcm[:6]] == [1, 2, 3, 4, 5, 1]
    assert fold_digest("acdc-fold-v6|", "patient001") == hashlib.sha256(b"acdc-fold-v6|patient001").hexdigest()


def test_v6_folds_are_assigned_on_the_eligible_patients(study):
    """The declared duplicate-image exclusions (patient055 MINF, patient076 NOR; amendments 2026-09-28) take no fold rank:
    the unchanged hash rule runs on the 98 eligible patients and gives the configured fold sizes and class counts."""
    from tmmri.splits import apply_duplicate_declarations

    frame = _acdc_dev_frame().assign(dataset="ACDC", case_id=lambda f: f["patient_id"], eligible_v6=True, exclusion_reason=None)
    assert frame.set_index("patient_id").loc[["patient055", "patient076"], "acdc_class"].tolist() == ["MINF", "NOR"]
    declared = apply_duplicate_declarations(frame, "ACDC", study.cohorts)
    assert declared.loc[~declared["eligible_v6"]].set_index("case_id")["exclusion_reason"].to_dict() == {
        "patient055": "DUPLICATE_IMAGE_OF_TEST_CASE", "patient076": "DUPLICATE_IMAGE_WITHIN_PARTITION"}
    outer = acdc_outer_folds_v6(declared, study.splits)
    assert len(outer) == 98 and not {"patient055", "patient076"} & set(outer["patient_id"])
    balance = fold_balance(outer, "outer_fold", "acdc_class")
    expected = study.splits["acdc"]["acdc_dev"]["outer_folds"]["expected"]
    assert {int(f): int(n) for f, n in balance.sum(axis=1).items()} == {int(k): int(v) for k, v in expected["patients_per_fold"].items()}
    assert {str(c): [int(v) for v in balance[c]] for c in balance.columns} == {str(c): list(v) for c, v in expected["patients_per_class_per_fold"].items()}
    by_patient = outer.set_index("patient_id")["outer_fold"]
    minf = sorted((p for p in outer.loc[outer["acdc_class"] == "MINF", "patient_id"]), key=lambda p: hashlib.sha256(("acdc-fold-v6|" + p).encode()).hexdigest())
    assert [int(by_patient[p]) for p in minf] == [r % 5 + 1 for r in range(19)]
    inner = acdc_inner_folds_v6(outer, 5, study.splits)
    assert len(inner) == 80 and len(acdc_inner_folds_v6(outer, 1, study.splits)) == 78


def test_v6_inner_and_refit_folds(study):
    outer = acdc_outer_folds_v6(_acdc_dev_frame(), study.splits)
    inner = acdc_inner_folds_v6(outer, 2, study.splits)
    assert len(inner) == 80 and not set(inner["patient_id"]) & set(outer.loc[outer["outer_fold"] == 2, "patient_id"])
    assert set(inner["inner_fold"]) == {1, 2, 3, 4, 5}
    counts = fold_balance(inner, "inner_fold", "acdc_class")
    assert (counts.max() - counts.min()).max() <= 1
    salt = "acdc-inner-fold-v6|2|"
    nor = sorted(inner.loc[inner["acdc_class"] == "NOR", "patient_id"], key=lambda p: hashlib.sha256((salt + p).encode()).hexdigest())
    assert int(inner.set_index("patient_id").loc[nor[0], "inner_fold"]) == 1
    cohort = pd.DataFrame({"sample_id": [f"MMS:C{i:05d}" for i in range(20)], "case_id": [f"C{i:05d}" for i in range(20)],
                           "vendor": ["A"] * 10 + ["B"] * 10, "refit_partition": ["mms_refit_train"] * 20})
    refit = mms_refit_folds_v6(cohort, study.splits)
    assert fold_balance(refit, "refit_fold", "vendor").to_numpy().tolist() == [[2, 2]] * 5
