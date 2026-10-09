"""Protocol 6.0.0 verify-and-stage and formal A: idempotence, content addressing, n_jobs invariance, legacy parity."""
from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

import fixtures_v6 as fx
from tmmri.formal_features import (
    _native_a_features,
    _native_features_for_row,
    extract_formal_table,
    formal_descriptor_names,
    formal_descriptors_v3,
    p6_projection_128,
    p6_settings,
)
from tmmri.config import load_study
from tmmri.ingest.mms import mms_cohort_table
from tmmri.ingest.staging import load_staged_case, load_staging_manifest, stage_dataset


@pytest.fixture()
def sources(tmp_path):
    acdc = fx.make_acdc_source(tmp_path / "acdc", fx.acdc_patients(1))
    mms = fx.make_mms_source(tmp_path / "mms")
    return {"acdc": (acdc, fx.acdc_manifest(acdc)), "mms": (mms, fx.mms_manifest(mms), fx.mms_metadata(mms))}


def _stage_mms(sources, out, **kwargs):
    root, manifest, meta = sources["mms"]
    return stage_dataset("mms", root, out, manifest, metadata=meta, **kwargs)


def _files(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_staging_is_idempotent_and_content_addressed(sources, tmp_path):
    out = tmp_path / "data" / "staged"
    first = _stage_mms(sources, out, n_jobs=1)
    assert first.actions == {"written": 4} and first.ok
    assert [e["reason_code"] for e in first.excluded] == ["MMS_RELEASE_UNLABELED", "MMS_RELEASE_UNLABELED"]
    before = _files(out)
    manifest_before = (tmp_path / "data" / "manifests" / "staging_MMS.json").read_bytes()
    second = _stage_mms(sources, out, n_jobs=1)
    assert second.actions == {"skipped_identical": 4} and second.source_bytes_read == 0
    assert _files(out) == before
    assert (tmp_path / "data" / "manifests" / "staging_MMS.json").read_bytes() == manifest_before
    staged = out / "MMS" / "B2B2B2" / "es_image.nii"
    staged.write_bytes(staged.read_bytes()[:-8] + b"\x00" * 8)
    third = _stage_mms(sources, out, n_jobs=1)
    assert third.actions == {"skipped_identical": 3, "written": 1}
    assert _files(out) == before


def test_staging_manifest_carries_hashes_and_no_timestamps(sources, tmp_path):
    _stage_mms(sources, tmp_path / "staged", n_jobs=1)
    manifest = load_staging_manifest(tmp_path / "manifests", "MMS")
    text = json.dumps(manifest)
    assert "utc" not in text and str(tmp_path).replace("\\", "/") not in text.replace("\\\\", "/")
    record = manifest["cases"]["A1A1A1"]
    root, tracked, _ = sources["mms"]
    row = tracked.set_index("case_id").loc["A1A1A1"]
    assert record["sources"]["image"]["sha256"] == row["image_sha256"] and record["sources"]["mask"]["sha256"] == row["mask_sha256"]
    assert set(record["staged"]) == {"ed_image", "ed_mask", "es_image", "es_mask"}
    assert record["mask_annotated_frames"] == [0, 2] and record["annotation_matches_metadata"] is True
    assert manifest["synthetic_fixture"] is True
    assert manifest["by_source_sha256"][row["image_sha256"]] == "A1A1A1/image"


def test_staged_frames_preserve_values_and_geometry(sources, tmp_path):
    _stage_mms(sources, tmp_path / "staged", n_jobs=1)
    root = sources["mms"][0]
    source_mask = nib.load(str(root / "Testing" / "C3C3C3" / "C3C3C3_sa_gt.nii.gz"))
    staged_mask = nib.load(str(tmp_path / "staged" / "MMS" / "C3C3C3" / "es_mask.nii"))
    assert staged_mask.get_data_dtype() == np.uint8 and staged_mask.ndim == 3
    assert np.allclose(staged_mask.affine, source_mask.affine)
    assert staged_mask.header.get_zooms() == pytest.approx(source_mask.header.get_zooms()[:3])
    assert staged_mask.header["sform_code"] == source_mask.header["sform_code"]
    assert np.array_equal(np.asanyarray(staged_mask.dataobj), np.rint(np.asanyarray(source_mask.dataobj)[..., 3]).astype(np.uint8))


def test_staging_is_atomic_and_leaves_no_temporary_files(sources, tmp_path):
    _stage_mms(sources, tmp_path / "staged", n_jobs=1)
    leftovers = [p.name for p in tmp_path.rglob("*") if ".tmp" in p.name]
    assert leftovers == []


def test_one_failure_never_aborts_other_cases(sources, tmp_path):
    root = sources["mms"][0]
    fx.corrupt_one_byte(root / "Testing" / "D4D4D4" / "D4D4D4_sa.nii.gz", offset=100)
    report = _stage_mms(sources, tmp_path / "staged", n_jobs=1)
    assert [(f["case_id"], f["reason_code"]) for f in report.failures] == [("D4D4D4", "SOURCE_HASH_MISMATCH")]
    assert report.actions == {"written": 3, "failed": 1}
    manifest = load_staging_manifest(tmp_path / "manifests", "MMS")
    assert set(manifest["cases"]) == {"A1A1A1", "B2B2B2", "C3C3C3"}
    assert manifest["failures"]["D4D4D4"]["reason_code"] == "SOURCE_HASH_MISMATCH"
    assert not (tmp_path / "staged" / "MMS" / "D4D4D4" / "case.json").exists()


def test_unannotated_es_frame_is_refused(sources, tmp_path):
    root, manifest, meta = sources["mms"]
    cohort = mms_cohort_table(root, manifest, metadata=meta)
    cohort.loc[cohort["case_id"] == "B2B2B2", "es_frame_index"] = 4        # the mask carries labels on frames 1 and 3 only
    report = stage_dataset("mms", root, tmp_path / "staged", manifest, n_jobs=1, metadata=meta, cohort=cohort)
    assert [(f["case_id"], f["reason_code"]) for f in report.failures] == [("B2B2B2", "INPUT_MISSING_ED_ES_FRAME")]


def test_staging_and_formal_extraction_are_identical_for_n_jobs_1_and_3(sources, tmp_path):
    for dataset in ("acdc", "mms"):
        for n in (1, 3):
            out = tmp_path / f"n{n}" / "staged"
            if dataset == "acdc":
                root, manifest = sources["acdc"]
                report = stage_dataset("acdc", root, out, manifest, n_jobs=n, source_readers=3)
            else:
                report = _stage_mms(sources, out, n_jobs=n, source_readers=3)
            assert report.ok and report.n_workers == n
    for ds in ("ACDC", "MMS"):
        one, three = _files(tmp_path / "n1" / "staged" / ds), _files(tmp_path / "n3" / "staged" / ds)
        assert one == three and one
        assert load_staging_manifest(tmp_path / "n1" / "manifests", ds)["content_sha256"] == \
            load_staging_manifest(tmp_path / "n3" / "manifests", ds)["content_sha256"]
    ids = ["patient001", "patient002", "patient003", "patient004", "patient005"]
    t1, f1 = extract_formal_table("ACDC", ids, tmp_path / "n1" / "staged", n_jobs=1)
    t3, f3 = extract_formal_table("ACDC", ids, tmp_path / "n1" / "staged", n_jobs=3)
    pd.testing.assert_frame_equal(t1, t3)
    assert t1.shape == (5, 3 + 57) and f1.empty and f3.empty
    assert list(t1.columns[3:]) == formal_descriptor_names()


def _legacy_row(sample_id, dataset, image_ed, image_es, mask_ed, mask_es, ed, es):
    return pd.Series({"sample_id": sample_id, "dataset_source": dataset, "raw_ed_image_path": str(image_ed),
                      "raw_es_image_path": str(image_es), "raw_ed_mask_path": str(mask_ed), "raw_es_mask_path": str(mask_es),
                      "ed_frame_index": ed, "es_frame_index": es})


def test_v6_formal_vector_equals_legacy_first_stage_descriptors(sources, tmp_path):
    aroot, amanifest = sources["acdc"]
    stage_dataset("acdc", aroot, tmp_path / "staged", amanifest, n_jobs=1)
    folder = aroot / "training" / "patient002"
    row = _legacy_row("ACDC:patient002", "ACDC", folder / "patient002_frame01.nii.gz", folder / "patient002_frame03.nii.gz",
                      folder / "patient002_frame01_gt.nii.gz", folder / "patient002_frame03_gt.nii.gz", 0, 2)
    v6 = formal_descriptors_v3(load_staged_case("ACDC", "patient002", tmp_path / "staged"))
    assert v6.shape == (57,)
    assert np.array_equal(v6, np.asarray(_native_features_for_row(row)))
    _stage_mms(sources, tmp_path / "staged", n_jobs=1)
    mfolder = sources["mms"][0] / "Training" / "Labeled" / "A1A1A1"
    mrow = _legacy_row("MMS:A1A1A1", "MMS_100", mfolder / "A1A1A1_sa.nii.gz", mfolder / "A1A1A1_sa.nii.gz",
                       mfolder / "A1A1A1_sa_gt.nii.gz", mfolder / "A1A1A1_sa_gt.nii.gz", 0, 2)
    mv6 = formal_descriptors_v3(load_staged_case("MMS", "A1A1A1", tmp_path / "staged"))
    assert np.array_equal(mv6, np.asarray(_native_features_for_row(mrow)))


def test_p6_projection_equals_the_legacy_128d_matrix(sources, tmp_path):
    aroot, amanifest = sources["acdc"]
    stage_dataset("acdc", aroot, tmp_path / "staged", amanifest, n_jobs=1)
    rows = []
    for pid in ("patient001", "patient004"):
        folder = aroot / "training" / pid
        rows.append(_legacy_row(f"ACDC:{pid}", "ACDC", folder / f"{pid}_frame01.nii.gz", folder / f"{pid}_frame03.nii.gz",
                                folder / f"{pid}_frame01_gt.nii.gz", folder / f"{pid}_frame03_gt.nii.gz", 0, 2))
    settings = p6_settings(load_study().formal)
    assert settings == {"seed": 20260611, "output_dimension": 128, "raw_retained": 32}
    legacy, _ = _native_a_features(pd.DataFrame(rows), latent_dim=128, seed=settings["seed"])
    table, _ = extract_formal_table("ACDC", ["patient001", "patient004"], tmp_path / "staged")
    v6 = p6_projection_128(table[formal_descriptor_names()].to_numpy(), **settings)
    assert np.allclose(v6, legacy[[c for c in legacy.columns if c.startswith("a_")]].to_numpy(), rtol=0, atol=1e-12)
