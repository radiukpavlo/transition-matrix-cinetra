"""Registry and configuration integrity (SPECIFICATION.md 6.0.0, Section 15.2). These tests run against the delivered files."""
from __future__ import annotations

import collections
import csv
import json
import pathlib
import re

import pytest

EXPECTED_FEATURES = 83
EXPECTED_METADATA = 42
EXPECTED_QC = 32
EXPECTED_B_V2_COLUMNS = 157
REQUIRED_FIELDS = [
    "name", "family", "meaning", "mental_model_tier", "dtype", "unit", "domain", "evidence_sources", "algorithm_id",
    "source_requirement", "target_transform", "discretization", "transition_role", "rule_role",
    "primary_family", "key_secondary", "harmonization", "dependencies", "standard_ref", "notes",
]
PRIMARY_FAMILY = {
    "lv_edv_ml", "lv_esv_ml", "lv_ef_percent", "rv_edv_ml", "rv_esv_ml", "rv_ef_percent", "lv_myocardial_mass_ed_g",
}
KEY_SECONDARY = {"global_wall_thickness_ed_mm", "global_wall_thickening_percent"}
STATES = {"OBSERVED", "NOT_APPLICABLE", "UNSUPPORTED", "QC_BLOCKED", "MISSING_SOURCE", "CONFLICTING"}
ROUTES = {
    "EXACT_ACCEPT", "CONFLICT_ABSTAIN", "SOFT_CANDIDATES_ONLY", "UNAVAILABLE_FEATURE_ABSTAIN",
    "NO_MATCH_ABSTAIN", "QC_ABSTAIN", "EMPTY_RULEBOOK",
}
HARMONIZATION = {"UNSUPPORTED", "HARMONIZABLE_NOT_EXACT", "HARMONIZABLE_SAME_ALGORITHM"}
FORBIDDEN_FIT_OR_SELECT = {"acdc_test", "mms_external", "mms_external_never_analysed", "mms_refit_eval"}
# Label-free diagnostics (SPECIFICATION.md 12.3): fitted on domain membership only, never on B or Y, never part of the
# analysis bundle and never a selection criterion.  They are the only objects whose fitting cell may name an external
# partition, and they must say so explicitly with the ``label_free_diagnostic(`` prefix.
LABEL_FREE_DIAGNOSTICS = {"shift_audit_domain_classifier_lr", "shift_audit_domain_classifier_lr_mms2"}   # _mms2: protocol-v6.0.1
EXTERNAL_OR_TEST_TOKEN = re.compile(r"\b(acdc_test|mms_external\w*|mms_refit_eval)\b")
CONFIG_FILES = [
    "study_v6.yaml", "cohorts.yaml", "splits.yaml", "formal.yaml", "transition.yaml", "wedd.yaml",
    "rule_policy.yaml", "equivariance.yaml", "deep_encoder.yaml", "bootstrap.yaml",
]
SNAKE = re.compile(r"^[a-z][a-z0-9_]*$")


def _yaml():
    return pytest.importorskip("yaml")


def _all_columns(registry):
    return (
        [f["name"] for f in registry["features"]]
        + list(registry["metadata_columns"])
        + list(registry["qc_columns"])
    )


def test_counts_match_specification(registry):
    counts = registry["counts"]
    assert len(registry["features"]) == EXPECTED_FEATURES == counts["features"]
    assert len(registry["metadata_columns"]) == EXPECTED_METADATA == counts["metadata_context_columns"]
    assert len(registry["qc_columns"]) == EXPECTED_QC == counts["qc_columns"]
    assert EXPECTED_FEATURES + EXPECTED_METADATA + EXPECTED_QC == EXPECTED_B_V2_COLUMNS == counts["total_b_v2_columns"]
    assert EXPECTED_METADATA + EXPECTED_QC + 3 * EXPECTED_FEATURES == counts["total_typed_B_columns"]
    by_family = collections.Counter(f["family"] for f in registry["features"])
    assert dict(by_family) == counts["features_by_family"]
    roles = collections.Counter(f["transition_role"] for f in registry["features"])
    assert roles["candidate"] + roles["candidate_secondary"] == counts["transition_candidates"]
    assert len(registry["primary_rulebook_antecedent_candidates"]) == counts["rule_antecedent_candidates"]


def test_names_unique_and_snake_case(registry):
    names = _all_columns(registry)
    dupes = [n for n, c in collections.Counter(names).items() if c > 1]
    assert not dupes, dupes
    assert all(SNAKE.match(n) for n in names), [n for n in names if not SNAKE.match(n)]


def test_every_feature_has_required_fields(registry):
    missing = {f["name"]: [k for k in REQUIRED_FIELDS if k not in f] for f in registry["features"]}
    missing = {k: v for k, v in missing.items() if v}
    assert not missing, missing
    for f in registry["features"]:
        assert f["target_transform"] in registry["target_transforms"], f["name"]
        assert f["discretization"] in registry["discretization_profiles"], f["name"]
        assert set(f["harmonization"].values()) <= HARMONIZATION, f["name"]
        if f["transition_role"] == "excluded":
            assert f["target_transform"] == "none" or f.get("exclusion_reason"), f["name"]


def test_primary_and_secondary_sets_resolve(registry):
    features = {f["name"]: f for f in registry["features"]}
    for group in ("primary_target_set", "key_secondary_targets", "primary_rulebook_antecedent_candidates"):
        unknown = [n for n in registry[group] if n not in features]
        assert not unknown, (group, unknown)
    assert set(registry["primary_target_set"]) == PRIMARY_FAMILY
    assert set(registry["primary_families"]["ventricular_function_size"]) == PRIMARY_FAMILY
    assert {n for n, f in features.items() if f.get("primary_family")} == PRIMARY_FAMILY
    assert set(registry["key_secondary_targets"]) == KEY_SECONDARY
    assert {n for n, f in features.items() if f.get("key_secondary")} == KEY_SECONDARY
    candidates = set(registry["primary_rulebook_antecedent_candidates"])
    assert candidates == {n for n, f in features.items() if f["rule_role"] == "candidate"}
    for name in PRIMARY_FAMILY | KEY_SECONDARY:
        assert features[name]["transition_role"] == "candidate", name
    for name in candidates:
        assert features[name]["discretization"] == "wedd_gaussian_mm_v1", name


def test_target_transforms_follow_decision_d10(registry):
    features = {f["name"]: f for f in registry["features"]}
    for name in ("lv_edv_ml", "lv_esv_ml", "rv_edv_ml", "rv_esv_ml", "lv_myocardial_mass_ed_g"):
        assert features[name]["target_transform"] == "log", name
    for name in ("lv_ef_percent", "rv_ef_percent"):
        assert features[name]["target_transform"] == "logit_percent", name
        assert features[name]["domain"] == [0.0, 100.0], name
    assert features["global_wall_thickening_percent"]["target_transform"] == "identity"


def test_qc_and_metadata_are_never_features_or_antecedents(registry):
    candidates = set(registry["primary_rulebook_antecedent_candidates"])
    assert "lv_rv_sv_discrepancy_ml" in registry["qc_columns"]
    assert "lv_rv_sv_discrepancy_ml" not in candidates
    for group in ("metadata_columns", "qc_columns"):
        assert not candidates & set(registry[group]), group
        assert not set(registry["primary_target_set"]) & set(registry[group]), group
    for name in ("disease_label", "split", "scanner_vendor", "center_id", "ed_frame_index", "slice_increment_mm"):
        assert name in registry["metadata_columns"], name


def test_reserved_and_unsupported_columns_are_excluded_everywhere(registry):
    reserved = [f for f in registry["features"] if f["family"] == "reserved_unsupported"]
    assert len(reserved) == 48
    for f in reserved:
        assert f["name"].startswith("aha16_seg")
        assert f["transition_role"] == "excluded" and f["rule_role"] == "excluded"
        assert f["evidence_sources"] == {"acdc": "UNSUPPORTED", "mms": "UNSUPPORTED"}
    features = {f["name"]: f for f in registry["features"]}
    indexed = [n for n in features if "indexed" in n] + ["bsa_m2", "bmi_kg_m2"]
    assert len(indexed) == 7
    for name in indexed:
        f = features[name]
        assert f["transition_role"] == "excluded" and f["rule_role"] == "excluded", name
        assert f["harmonization"]["mms"] == "UNSUPPORTED", name
        assert "Height" in f["exclusion_reason"] and "39.1 %" in f["exclusion_reason"], name
    for name in ("lv_long_axis_length_mm", "lv_sphericity_index_3d", "lv_myo_volume_ed_ml"):
        assert features[name]["transition_role"] == "excluded", name
        assert features[name].get("exclusion_reason"), name


def test_states_and_tiers_declared(registry):
    assert set(registry["states"]) == STATES
    assert set(registry["state_definitions"]) == STATES
    tiers = set(registry["evidence_tiers"])
    for f in registry["features"]:
        for cohort, value in f["evidence_sources"].items():
            assert cohort in {"acdc", "mms"}
            if value != "UNSUPPORTED":
                assert set(value) <= tiers, (f["name"], cohort, value)
    for group in ("metadata_columns", "qc_columns"):
        for name, spec in registry[group].items():
            assert spec["evidence_tier"] in tiers, (name, spec["evidence_tier"])


def test_registry_columns_match_b_v2_dictionary(registry, repo_root: pathlib.Path):
    candidates = [
        repo_root / "legacy" / "v5" / "data" / "features" / "feature_dictionary_v5.csv",
        repo_root / "data" / "features" / "feature_dictionary_v5.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        pytest.skip("v5 feature dictionary not present")
    with path.open(encoding="utf-8", newline="") as fh:
        b_columns = [row["column_name"] for row in csv.DictReader(fh)]
    assert len(b_columns) == EXPECTED_B_V2_COLUMNS
    registered = _all_columns(registry)
    assert sorted(registered) == sorted(b_columns), sorted(set(registered) ^ set(b_columns))


def test_reason_codes_cover_all_routes(reason_codes):
    assert set(reason_codes["routes"]) == ROUTES
    assert set(reason_codes["route_precedence"]) == ROUTES
    abstention_routes = ROUTES - {"EXACT_ACCEPT"}  # accepted outputs carry no reason code
    coded_routes = {c["route"] for c in reason_codes["codes"] if c["route"]}
    assert coded_routes == abstention_routes
    codes = [c["code"] for c in reason_codes["codes"]]
    assert len(set(codes)) == len(codes)
    by_code = {c["code"]: c for c in reason_codes["codes"]}
    for code in ("REFERENCE_OFF_ONTOLOGY", "REFERENCE_NOT_EXACT_IHD_MINF", "MMS_RELEASE_UNLABELED", "MMS_DUPLICATE_CASE",
                 "DUPLICATE_IMAGE_OF_TEST_CASE", "DUPLICATE_IMAGE_WITHIN_PARTITION", "DUPLICATE_IMAGE_OF_REFIT_TRAIN_CASE"):
        assert by_code[code]["route"] is None, code
    for code in ("INPUT_DECODE_FAILURE", "INPUT_NONFINITE", "INPUT_GEOMETRY_INVALID", "INPUT_MISSING_ED_ES_FRAME",
                 "QC_MISSING_LV_BLOOD_POOL", "QC_MISSING_RV_BLOOD_POOL", "QC_MISSING_LV_MYOCARDIUM"):
        assert by_code[code]["route"] == "QC_ABSTAIN", code
    assert all(c["counted_in"] for c in reason_codes["codes"])


def test_declared_duplicate_images_are_registered_and_sizes_consistent(reason_codes, registry, repo_root: pathlib.Path):
    """configs/cohorts.yaml duplicate_image_pairs (PI decision, amendments 2026-09-28) against the registry and sizes."""
    yaml = _yaml()
    load = lambda name: yaml.safe_load((repo_root / "configs" / name).read_text(encoding="utf-8"))  # noqa: E731
    cohorts, splits, study = load("cohorts.yaml"), load("splits.yaml"), load("study_v6.yaml")
    codes = {c["code"]: c for c in reason_codes["codes"]}
    pairs = cohorts["duplicate_image_pairs"]["pairs"]
    assert {(p["cohort"], tuple(sorted(p["ids"]))) for p in pairs} == {
        ("acdc", ("patient055", "patient118")), ("acdc", ("patient074", "patient076")),
        ("mms", ("A7G0P5", "K3R0Y7")), ("mms", ("C8J7L5", "C8O0P2"))}
    excluded = {p["exclude"]["case"]: (p["exclude"]["from"], p["reason_code"]) for p in pairs if p.get("exclude")}
    assert excluded == {"patient055": ("acdc_dev", "DUPLICATE_IMAGE_OF_TEST_CASE"), "patient076": ("acdc_dev", "DUPLICATE_IMAGE_WITHIN_PARTITION"),
                        "K3R0Y7": ("mms_refit_eval", "DUPLICATE_IMAGE_OF_REFIT_TRAIN_CASE")}
    for _, code in excluded.values():
        assert codes[code]["route"] is None and "flow_diagram" in codes[code]["counted_in"], code
    assert all(set(p["matched_frames"]) <= {"ED", "ES"} and p["matched_frames"] for p in pairs)
    sizes = study["partition_sizes"]
    folds = splits["acdc"]["acdc_dev"]["outer_folds"]["expected"]
    y_dev = registry["label_tables"]["Y_acdc"]["counts"]["acdc_dev"]
    assert sizes["acdc_dev"] == sum(y_dev.values()) == sum(folds["patients_per_fold"].values()) == 98
    assert {c: sum(v) for c, v in folds["patients_per_class_per_fold"].items()} == y_dev
    assert [sum(v[i] for v in folds["patients_per_class_per_fold"].values()) for i in range(5)] == [folds["patients_per_fold"][f] for f in range(1, 6)]
    refit_eval = splits["refit_arm"]["mms_refit_eval"]["expected"]
    assert sizes["mms_refit_eval"] == refit_eval["total"] == sum(refit_eval["by_vendor"].values()) == 169
    clusters = splits["duplicate_control"]["expected_clusters"]
    assert all(clusters[p] <= sizes[p] for p in clusters if p in sizes)
    assert sizes["mms_external"] - clusters["mms_external"] == 2   # two declared M&Ms pairs stay in mms_external


def test_every_critical_qc_reason_routes_to_qc_abstain(reason_codes, repo_root: pathlib.Path):
    yaml = _yaml()
    cohorts = yaml.safe_load((repo_root / "configs" / "cohorts.yaml").read_text(encoding="utf-8"))
    critical = set(cohorts["qc_severity"]["critical"])
    routed = {r for c in reason_codes["codes"] if c["route"] == "QC_ABSTAIN" for r in c.get("qc_reasons", [])}
    assert critical <= routed, sorted(critical - routed)
    assert {r for r in critical if r.startswith("missing_")} >= {
        "missing_lv_ed_present", "missing_lv_es_present", "missing_rv_ed_present",
        "missing_rv_es_present", "missing_myo_ed_present", "missing_myo_es_present",
    }
    assert cohorts["qc_severity"]["unknown_reason_policy"] == "fail_closed"


def test_estimator_partition_registry_has_no_forbidden_fitting(repo_root: pathlib.Path):
    path = repo_root / "registry" / "estimator_partition_registry.csv"
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == ["object", "fitted_on", "selected_on", "evaluated_on", "forbidden_partitions", "code_path", "hash"]
        rows = list(reader)
    assert rows
    objects = [r["object"] for r in rows]
    assert len(set(objects)) == len(objects)

    def partitions(cell: str) -> set[str]:
        return {m.group(0) for part in cell.split(";") if (m := re.match(r"[a-z_]+", part.strip()))}

    diagnostics = {r["object"] for r in rows if r["fitted_on"].startswith("label_free_diagnostic(")}
    assert diagnostics == LABEL_FREE_DIAGNOSTICS
    for row in rows:
        if row["object"] in LABEL_FREE_DIAGNOSTICS:
            assert row["selected_on"] == "none" and "acdc_test" not in row["fitted_on"], row["object"]
            assert "acdc_test" in partitions(row["forbidden_partitions"]) and (repo_root / row["code_path"]).exists(), row["object"]
            assert "no B or Y" in row["fitted_on"] and "never in the bundle" in row["fitted_on"], row["object"]
            continue
        assert not EXTERNAL_OR_TEST_TOKEN.search(row["fitted_on"]), row["object"]    # anywhere in the cell, not only its head
        assert not (partitions(row["fitted_on"]) & FORBIDDEN_FIT_OR_SELECT), row["object"]
        assert not (partitions(row["selected_on"]) & FORBIDDEN_FIT_OR_SELECT), row["object"]
        assert row["code_path"], row["object"]
        # every implementing path exists (the D20 module src/tmmri/deep_encoder.py is implemented since protocol-v6.0.1)
        assert (repo_root / row["code_path"]).exists(), row["object"]
        assert row["hash"] == "", row["object"]  # filled at M0
        if row["fitted_on"].startswith("acdc_dev"):
            forbidden = partitions(row["forbidden_partitions"])
            assert {"acdc_test", "mms_external", "mms_external_never_analysed"} <= forbidden, row["object"]


def test_pattern_templates_use_states_only(registry, repo_root: pathlib.Path):
    yaml = _yaml()
    templates = yaml.safe_load((repo_root / "registry" / "pattern_templates.yaml").read_text(encoding="utf-8"))
    candidates = set(registry["primary_rulebook_antecedent_candidates"])
    tokens = set(templates["state_tokens"])
    for t in templates["templates"]:
        assert t["id"].startswith("PT_")
        if t["support_status"] == "unsupported":
            assert t["conditions"] == {}, t["id"]
        else:
            assert set(t["conditions"]) <= candidates, t["id"]
            assert set(t["conditions"].values()) <= tokens, t["id"]


def test_configs_parse_and_carry_specification_version(repo_root: pathlib.Path):
    yaml = _yaml()
    for name in CONFIG_FILES:
        data = yaml.safe_load((repo_root / "configs" / name).read_text(encoding="utf-8"))
        assert data["specification_version"] == "6.0.0", name
    for name in ("acdc.json", "mms.json"):
        data = json.loads((repo_root / "configs" / "source_bindings" / name).read_text(encoding="utf-8"))
        assert data["status"] == "bound_on_mount" and data["specification_version"] == "6.0.0", name
        assert all({"expected", "bound", "note"} <= set(item) for item in data["bindings"].values()), name
    study = yaml.safe_load((repo_root / "configs" / "study_v6.yaml").read_text(encoding="utf-8"))
    assert study["freeze"]["tag"] == "protocol-v6.0.0"
    assert study["seeds"]["synthetic_controls"] == 20260613
    assert study["partitions"] == ["acdc_dev", "acdc_test", "mms_external", "mms_external_never_analysed", "mms_refit_train", "mms_refit_eval"]
    for name in ("reason_codes.yaml", "pattern_templates.yaml", "semantic_registry.yaml"):
        data = yaml.safe_load((repo_root / "registry" / name).read_text(encoding="utf-8"))
        assert data["specification_version"] == "6.0.0", name
