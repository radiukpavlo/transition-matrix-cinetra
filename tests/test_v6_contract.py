"""Protocol 6.0.0 typed-B contract (registry validation of B_v6.0)."""
from __future__ import annotations

import pandas as pd
import pytest

from tmmri.config import load_study
from tmmri.semantics.contract import ContractViolation, registry_features, typed_columns, validate_typed_b

CASES = (("ACDC", "patient001", "acdc", "training", "acdc_dev"), ("MMS", "A1A1A1", "mms", "Training/Labeled", "mms_external"))


def _value(spec):
    domain = spec.get("domain")
    if isinstance(domain, list):
        low, high = domain
        if low is not None and high is not None:
            return (low + high) / 2.0
        return (low or 0.0) + 10.0
    return 1.0


def valid_frame(registry) -> pd.DataFrame:
    rows = []
    for dataset, case, cohort, subset, split in CASES:
        row: dict[str, object] = {name: None for name in typed_columns(registry)}
        row.update({"schema_version": "B_v6.0", "sample_id": f"{dataset}:{case}", "case_id": case, "patient_id": case,
                    "dataset_source": dataset, "subset_name": subset, "split": split, "ed_frame_index": 0, "es_frame_index": 9,
                    "spacing_x_mm": 1.25, "spacing_y_mm": 1.25, "slice_increment_mm": 10.0, "ed_es_source": "metadata",
                    "z_spacing_method": "nifti_affine_projection", "z_spacing_estimated_flag": False,
                    "papillary_inclusion_status": "unknown", "lv_rv_sv_discrepancy_ml": 3.0, "qc_failure_reason": None})
        for name, spec in registry["qc_columns"].items():
            if spec.get("dtype") == "bool":
                row[name] = True
            elif spec.get("dtype") == "int64":
                row[name] = 0
        for name, spec in registry_features(registry).items():
            sources = spec.get("evidence_sources", {}).get(cohort)
            if isinstance(sources, list) and spec.get("domain") != "null_only":
                row.update({name: _value(spec), f"{name}__state": "OBSERVED", f"{name}__evidence": ";".join(sources)})
            else:
                row.update({name: None, f"{name}__state": "UNSUPPORTED", f"{name}__evidence": None})
        rows.append(row)
    frame = pd.DataFrame(rows)
    for name, spec in {**registry["metadata_columns"], **registry["qc_columns"], **registry_features(registry)}.items():
        dtype = spec.get("dtype")
        if dtype == "float64":
            frame[name] = pd.to_numeric(frame[name]).astype("float64")
        elif dtype == "int64" and frame[name].notna().all():
            frame[name] = frame[name].astype("int64")
        elif dtype == "bool" and frame[name].notna().all():
            frame[name] = frame[name].astype(bool)
    return frame.copy()


@pytest.fixture(scope="module")
def study():
    return load_study()


def test_valid_typed_table_passes(registry, study):
    frame = valid_frame(registry)
    assert frame.shape[1] == registry["counts"]["total_typed_B_columns"] == 323
    summary = validate_typed_b(frame, registry, cohorts_cfg=study.cohorts)
    assert summary["rows"] == 2 and summary["features"] == 83


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda f: f.drop(columns=["lv_edv_ml__evidence"]), "missing columns"),
        (lambda f: f.assign(**{"lv_ef_percent__state": ["QC_BLOCKED", "OBSERVED"]}), "non-null value with state != OBSERVED"),
        (lambda f: f.assign(lv_esv_ml=[None, 40.0]), "OBSERVED with a null value"),
        (lambda f: f.assign(lv_ef_percent=[150.0, 50.0]), "outside domain"),
        # no critical-QC exemption: the builder marks such values QC_BLOCKED, so OBSERVED out of domain always fails
        (lambda f: f.assign(lv_ef_percent=[150.0, 50.0], qc_failure_reason=["lv_ef_outside_0_100", None]), "outside domain"),
        (lambda f: f.iloc[::-1].reset_index(drop=True), "not sorted"),
        (lambda f: pd.concat([f, f.iloc[[1]]], ignore_index=True), "not unique"),
        (lambda f: f.assign(**{"rv_edv_ml__state": ["OBSERVED", "MEASURED"]}), "state outside"),
        (lambda f: f.assign(lv_edv_ml=["big", "small"]), "dtype"),
        (lambda f: f.assign(extra_feature=[1.0, 2.0]), "not in the registry"),
        (lambda f: f.assign(qc_failure_reason=["lv_edv_le_lvesv", "made_up_reason:[4]"]), "QC_UNREGISTERED_REASON"),
        (lambda f: f.assign(aha16_seg01_wall_thickness_ed_mm=[2.0, None], **{"aha16_seg01_wall_thickness_ed_mm__state": ["OBSERVED", "UNSUPPORTED"],
                                                                           "aha16_seg01_wall_thickness_ed_mm__evidence": ["E1_EXPERT_MASK_GEOMETRY", None]}), "null_only"),
        (lambda f: f.assign(**{"lv_edv_ml__evidence": ["E2_RELEASE_METADATA", "E1_EXPERT_MASK_GEOMETRY"]}), "not in evidence_sources"),
        (lambda f: f.assign(sample_id=["ACDC:patient009", "MMS:A1A1A1"]), "sample_id != dataset_source:case_id"),
        (lambda f: f.assign(schema_version=["B_v2.0", "B_v6.0"]), "schema_version"),
        (lambda f: f.assign(dataset_source=["ACDC", "MMS_100"], sample_id=["ACDC:patient001", "MMS_100:A1A1A1"]), "enum"),
        (lambda f: f.drop(columns=["spacing_x_mm"]), "missing required metadata column spacing_x_mm"),
    ],
)
def test_contract_catches_violations(registry, study, mutate, expected):
    frame = mutate(valid_frame(registry))
    with pytest.raises(ContractViolation) as info:
        validate_typed_b(frame, registry, cohorts_cfg=study.cohorts)
    assert any(expected in v for v in info.value.violations), info.value.violations


def test_contract_lists_every_violation(registry, study):
    frame = valid_frame(registry).assign(lv_ef_percent=[150.0, 50.0]).drop(columns=["rv_esv_ml__state"])
    with pytest.raises(ContractViolation) as info:
        validate_typed_b(frame, registry, cohorts_cfg=study.cohorts)
    assert len(info.value.violations) >= 2
