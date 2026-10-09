from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

from .io import resolve_path, utc_now_iso, write_json, write_table_pair, write_yaml

CLINICAL_PREDICATE_LIBRARY = [
    {
        "predicate_id": "P_DCM_CINE_PROXY",
        "pathology_name": "dilated_cardiomyopathy",
        "clinical_concept": "LV dilation with reduced systolic function; cine-only proxy.",
        "required_features": ["lv_edv_ml", "lv_ef_percent"],
        "optional_features": ["rv_ef_percent", "lv_myocardial_mass_ed_g"],
        "threshold_source": "Research approximation; use local normal ranges or sex/body-size indexed references for formal studies.",
        "directionality": {"lv_edv_ml": "high", "lv_ef_percent": "low"},
        "uncertainty_band": "dataset-derived cutpoints in current prototype",
        "unsupported_prerequisites": [],
        "dataset_support_status": "partially_supported_cine_mask_proxy",
        "expected_B_columns": ["lv_edv_ml", "lv_ef_percent", "lv_edv_indexed_ml_m2"],
        "symbolic_states": ["lv_volume_high", "lv_ef_low"],
        "rulebook_compatibility": True,
    },
    {
        "predicate_id": "P_HCM_CINE_PROXY",
        "pathology_name": "hypertrophic_cardiomyopathy",
        "clinical_concept": "Increased myocardial mass or wall thickness with preserved or variable function; cine-only proxy.",
        "required_features": ["global_wall_thickness_ed_mm", "lv_myocardial_mass_ed_g"],
        "optional_features": ["lv_ef_percent"],
        "threshold_source": "Research approximation; confirmatory HCM requires validated wall-thickness measurement convention and clinical context.",
        "directionality": {"global_wall_thickness_ed_mm": "high", "lv_myocardial_mass_ed_g": "high"},
        "uncertainty_band": "dataset-derived cutpoints in current prototype",
        "unsupported_prerequisites": [],
        "dataset_support_status": "partially_supported_cine_mask_proxy",
        "expected_B_columns": ["global_wall_thickness_ed_mm", "lv_myocardial_mass_ed_g"],
        "symbolic_states": ["wall_thickness_high", "lv_mass_high"],
        "rulebook_compatibility": True,
    },
    {
        "predicate_id": "P_ARV_CINE_PROXY",
        "pathology_name": "abnormal_right_ventricle_or_arvc_proxy",
        "clinical_concept": "RV enlargement or reduced RVEF; not a complete ARVC diagnosis without regional motion and tissue criteria.",
        "required_features": ["rv_edv_ml", "rv_ef_percent"],
        "optional_features": ["lv_rv_sv_discrepancy_ml"],
        "threshold_source": "Research approximation; clinical ARVC criteria require additional expert features.",
        "directionality": {"rv_edv_ml": "high", "rv_ef_percent": "low"},
        "uncertainty_band": "dataset-derived cutpoints",
        "unsupported_prerequisites": ["regional_wall_motion", "fibrofatty_replacement", "family_history"],
        "dataset_support_status": "partially_supported_cine_mask_proxy",
        "expected_B_columns": ["rv_edv_ml", "rv_ef_percent"],
        "symbolic_states": ["rv_volume_high", "rv_ef_low"],
        "rulebook_compatibility": True,
    },
    {
        "predicate_id": "P_MINF_CINE_PROXY",
        "pathology_name": "myocardial_infarction_or_tissue_viability",
        "clinical_concept": "Cine-derived wall thinning/remodeling proxy only; LGE/edema/T1/T2/ECV are unsupported in this cine-mask pipeline.",
        "required_features": ["lv_ef_percent", "global_wall_thickness_ed_mm"],
        "optional_features": ["lv_edv_ml"],
        "threshold_source": "Research approximation; infarct/viability claims require tissue characterization sequences.",
        "directionality": {"lv_ef_percent": "low", "global_wall_thickness_ed_mm": "low"},
        "uncertainty_band": "dataset-derived cutpoints",
        "unsupported_prerequisites": ["late_gadolinium_enhancement", "edema", "t1_mapping", "t2_mapping", "ecv"],
        "dataset_support_status": "partially_supported_cine_proxy_only",
        "expected_B_columns": ["lv_ef_percent", "global_wall_thickness_ed_mm"],
        "symbolic_states": ["lv_ef_low", "wall_thickness_low"],
        "rulebook_compatibility": True,
    },
    {
        "predicate_id": "P_RESTRICTIVE_UNSUPPORTED",
        "pathology_name": "restrictive_cardiomyopathy",
        "clinical_concept": "Restrictive physiology cannot be established from current cine segmentation features alone.",
        "required_features": ["atrial_size", "diastolic_filling", "tissue_characterization"],
        "optional_features": [],
        "threshold_source": "unsupported",
        "directionality": {},
        "uncertainty_band": "not_applicable",
        "unsupported_prerequisites": ["atrial_measurements", "diastolic_function", "LGE_or_mapping"],
        "dataset_support_status": "unsupported",
        "expected_B_columns": [],
        "symbolic_states": [],
        "rulebook_compatibility": False,
    },
    {
        "predicate_id": "P_NONCOMPACTION_UNSUPPORTED",
        "pathology_name": "left_ventricular_noncompaction",
        "clinical_concept": "Requires trabeculation/noncompacted-to-compacted metrics not present in Matrix B.",
        "required_features": ["trabeculation_metric"],
        "optional_features": [],
        "threshold_source": "unsupported",
        "directionality": {},
        "uncertainty_band": "not_applicable",
        "unsupported_prerequisites": ["trabeculation_segmentation"],
        "dataset_support_status": "unsupported",
        "expected_B_columns": [],
        "symbolic_states": [],
        "rulebook_compatibility": False,
    },
    {
        "predicate_id": "P_VALVULAR_UNSUPPORTED_OR_VOLUME_PROXY",
        "pathology_name": "valvular_heart_disease",
        "clinical_concept": "Volume overload patterns are candidate review signals only; valve stenosis/regurgitation labels require flow or valve morphology.",
        "required_features": ["regurgitant_fraction_or_valve_morphology"],
        "optional_features": ["lv_edv_ml", "rv_edv_ml", "lv_rv_sv_discrepancy_ml"],
        "threshold_source": "unsupported for diagnosis; volume imbalance can trigger review.",
        "directionality": {"lv_rv_sv_discrepancy_ml": "high"},
        "uncertainty_band": "manual_review",
        "unsupported_prerequisites": ["phase_contrast_flow", "valve_morphology"],
        "dataset_support_status": "unsupported_except_review_proxy",
        "expected_B_columns": ["lv_rv_sv_discrepancy_ml"],
        "symbolic_states": ["stroke_volume_discrepancy_high"],
        "rulebook_compatibility": True,
    },
    {
        "predicate_id": "P_CONGENITAL_REVIEW_PROXY",
        "pathology_name": "congenital_heart_disease_candidate_review",
        "clinical_concept": "Chamber disproportion or shunt-like volume pattern can route to review, not diagnose congenital anatomy.",
        "required_features": ["rv_edv_ml", "lv_edv_ml", "lv_rv_sv_discrepancy_ml"],
        "optional_features": [],
        "threshold_source": "research-only approximation",
        "directionality": {"rv_edv_ml": "high", "lv_rv_sv_discrepancy_ml": "high"},
        "uncertainty_band": "manual_review",
        "unsupported_prerequisites": ["anatomy_labels", "flow_measures", "expert_annotation"],
        "dataset_support_status": "research_only_approximation",
        "expected_B_columns": ["rv_edv_ml", "lv_edv_ml", "lv_rv_sv_discrepancy_ml"],
        "symbolic_states": ["chamber_disproportion", "sv_discrepancy_high"],
        "rulebook_compatibility": True,
    },
    {
        "predicate_id": "P_INFLAM_INFILTRATIVE_UNSUPPORTED",
        "pathology_name": "myocarditis_sarcoidosis_amyloidosis_fabry_like_patterns",
        "clinical_concept": "Inflammatory/infiltrative tissue characterization is unsupported without T1/T2/LGE/ECV or validated metadata.",
        "required_features": ["T1", "T2", "LGE", "ECV"],
        "optional_features": [],
        "threshold_source": "unsupported",
        "directionality": {},
        "uncertainty_band": "not_applicable",
        "unsupported_prerequisites": ["t1_mapping", "t2_mapping", "late_gadolinium_enhancement", "ecv"],
        "dataset_support_status": "unsupported",
        "expected_B_columns": [],
        "symbolic_states": [],
        "rulebook_compatibility": False,
    },
    {
        "predicate_id": "P_NORMAL_CONSERVATIVE",
        "pathology_name": "normal_state_conservative",
        "clinical_concept": "No major abnormality by available cine-mask features; absence of evidence is not clinical normality.",
        "required_features": ["lv_ef_percent", "rv_ef_percent", "lv_edv_ml", "rv_edv_ml", "global_wall_thickness_ed_mm"],
        "optional_features": [],
        "threshold_source": "dataset-derived normal-like bands in prototype",
        "directionality": {"lv_ef_percent": "normal", "rv_ef_percent": "normal"},
        "uncertainty_band": "manual_review_near_boundaries",
        "unsupported_prerequisites": [],
        "dataset_support_status": "partially_supported_as_research_filter",
        "expected_B_columns": ["lv_ef_percent", "rv_ef_percent", "lv_edv_ml", "rv_edv_ml", "global_wall_thickness_ed_mm"],
        "symbolic_states": ["major_features_normal_range"],
        "rulebook_compatibility": True,
    },
]

FEATURE_WEIGHTS = {
    "lv_ef_percent": 3.0,
    "rv_ef_percent": 3.0,
    "lv_edv_ml": 2.5,
    "rv_edv_ml": 2.5,
    "lv_esv_ml": 2.0,
    "rv_esv_ml": 2.0,
    "lv_myocardial_mass_ed_g": 2.0,
    "global_wall_thickness_ed_mm": 2.0,
    "global_wall_thickening_percent": 1.5,
    "lv_rv_sv_discrepancy_ml": 1.5,
}


def write_predicate_library(experiment_id: str) -> list[Path]:
    payload = {
        "schema_version": "clinical_predicate_library_v3.0",
        "experiment_id": experiment_id,
        "clinical_status": "research_use_only",
        "generated_at_utc": utc_now_iso(),
        "predicates": CLINICAL_PREDICATE_LIBRARY,
    }
    artifacts = [write_yaml(payload, f"artifacts/rules/clinical_predicate_library_{experiment_id}.yaml")]
    json_path = resolve_path(f"artifacts/rules/clinical_predicate_library_{experiment_id}.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts.append(json_path)
    md = resolve_path(f"artifacts/reports/rules/clinical_predicate_library_{experiment_id}.md")
    md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Clinical Predicate Library - {experiment_id}",
        "",
        "Clinical status: research use only. These predicates are not clinically validated for diagnosis, triage, treatment selection, or patient management.",
        "",
        "The library deliberately marks unsupported tissue-characterization, flow, valve, congenital anatomy, and trabeculation predicates as unsupported when Matrix B lacks the required prerequisites.",
        "",
    ]
    for pred in CLINICAL_PREDICATE_LIBRARY:
        lines.extend(
            [
                f"## {pred['predicate_id']} - {pred['pathology_name']}",
                "",
                f"Support status: `{pred['dataset_support_status']}`",
                f"Concept: {pred['clinical_concept']}",
                f"Required features: {', '.join(pred['required_features']) if pred['required_features'] else 'None'}",
                f"Unsupported prerequisites: {', '.join(pred['unsupported_prerequisites']) if pred['unsupported_prerequisites'] else 'None'}",
                "",
            ]
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifacts.append(md)
    return artifacts


def direction_score(pred_direction: str, rule_state: str) -> float:
    pred_direction = str(pred_direction).lower()
    rule_state = str(rule_state).lower()
    if pred_direction in rule_state:
        return 1.0
    if pred_direction == "low" and ("reduced" in rule_state or "small" in rule_state):
        return 1.0
    if pred_direction == "high" and ("large" in rule_state or "increased" in rule_state):
        return 1.0
    if pred_direction == "normal" and "normal" in rule_state:
        return 1.0
    return 0.0


def predicate_rule_similarity(predicate: dict[str, object], rule: dict[str, object]) -> float:
    antecedents = rule.get("antecedents", []) or []
    rule_features = {a.get("feature") for a in antecedents}
    pred_features = set(predicate.get("expected_B_columns") or predicate.get("required_features") or [])
    union = pred_features | rule_features
    if not union:
        feature_overlap = 0.0
    else:
        feature_overlap = len(pred_features & rule_features) / len(union)
    weighted_scores = []
    weights = []
    directions = predicate.get("directionality", {}) or {}
    for feat in pred_features:
        w = FEATURE_WEIGHTS.get(feat, 1.0)
        matching = [a for a in antecedents if a.get("feature") == feat]
        if not matching:
            s = 0.0
        else:
            if feat in directions:
                s = max(direction_score(directions[feat], a.get("state", a.get("discretized_bin", ""))) for a in matching)
            else:
                s = 0.7
        weighted_scores.append(w * s)
        weights.append(w)
    feature_direction = sum(weighted_scores) / sum(weights) if weights else 0.0
    support = float(rule.get("support", 0) or 0)
    confidence = float(rule.get("confidence", 0) or 0)
    conflict = float(rule.get("conflict_rate", 0) or 0)
    unsupported = predicate.get("dataset_support_status") == "unsupported"
    missing_penalty = 0.25 if unsupported else 0.0
    support_bonus = min(support / 20.0, 1.0) * 0.05
    sim = 0.35 * feature_overlap + 0.45 * feature_direction + 0.20 * confidence + support_bonus - 0.30 * conflict - missing_penalty
    return float(max(-1.0, min(1.0, sim)))


def compare_predicates_to_rules(rulebook: dict[str, object], experiment_id: str, min_support: int = 3, min_confidence: float = 0.5) -> tuple[pd.DataFrame, Path]:
    rules = [r for r in rulebook.get("rules", []) if int(r.get("support", 0) or 0) >= min_support and float(r.get("confidence", 0) or 0) >= min_confidence]
    rows = []
    for pred in CLINICAL_PREDICATE_LIBRARY:
        best_rule = None
        best_score = -999.0
        for rule in rules:
            if pred.get("dataset_support_status") == "unsupported":
                continue
            score = predicate_rule_similarity(pred, rule)
            if score > best_score:
                best_score = score
                best_rule = rule
        rows.append(
            {
                "predicate_id": pred["predicate_id"],
                "pathology_name": pred["pathology_name"],
                "dataset_support_status": pred["dataset_support_status"],
                "best_rule_id": best_rule.get("rule_id") if best_rule else None,
                "best_similarity": best_score if best_rule else np.nan,
                "minimum_support": min_support,
                "minimum_confidence": min_confidence,
                "missing_feature_or_unsupported_penalty_applied": pred.get("dataset_support_status") == "unsupported",
                "experiment_id": experiment_id,
            }
        )
    df = pd.DataFrame(rows)
    artifacts = write_table_pair(df, f"artifacts/reports/rules/predicate_similarity_{experiment_id}.parquet", f"artifacts/reports/rules/predicate_similarity_{experiment_id}.csv")
    md = resolve_path(f"artifacts/reports/rules/predicate_similarity_{experiment_id}.md")
    md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Predicate Similarity Report - {experiment_id}",
        "",
        "Clinical status: research use only. Similarity compares symbolic rule alignment with physician-style predicates; it is not clinical validation.",
        "",
        "Similarity formula: weighted feature/direction agreement plus confidence/support bonuses minus conflict and missing-prerequisite penalties. Unsupported predicates are not forced into matches.",
        "",
        df.to_markdown(index=False),
    ]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifacts.append(md)
    return df, md
