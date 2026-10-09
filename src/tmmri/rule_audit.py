from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .rules import discretize_frame, infer_with_rulebook


SAFETY_SENSITIVE_LABELS = frozenset({"MINF", "IHD", "RV", "ARV"})


def sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_rulebook(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload.get("rules"), list):
        raise ValueError("Rulebook must contain a list-valued 'rules' field")
    return payload


def load_cutpoints(path: str | Path) -> dict[str, dict[str, float]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    cutpoints = payload.get("cutpoints", payload)
    if not isinstance(cutpoints, dict):
        raise ValueError("Cutpoint file does not contain a mapping")
    return cutpoints


def antecedent_pairs(rule: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    pairs = []
    for ant in rule.get("antecedents", []) or []:
        feature = str(ant.get("feature", ""))
        state = str(ant.get("state", ant.get("discretized_bin", "")))
        if feature:
            pairs.append((feature, state))
    return tuple(sorted(pairs))


def antecedent_text(rule: dict[str, Any]) -> str:
    pairs = antecedent_pairs(rule)
    return " AND ".join(f"{feature} is {state}" for feature, state in pairs) if pairs else "<empty antecedent>"


def build_transition_fidelity_gate(metrics: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    required = {"dataset_source", "split", "feature_name", "pearson_r", "normalized_rmse_by_split_sd", "bias", "rmse", "n"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Transition metrics missing fields required for fidelity gating: {sorted(missing)}")
    split_name = str(cfg.get("transition_fidelity", {}).get("validation_split", "val"))
    validation = metrics.loc[
        metrics["dataset_source"].astype(str).eq("ACDC") & metrics["split"].astype(str).eq(split_name)
    ].copy()
    green = cfg.get("transition_fidelity", {}).get("green", {})
    amber = cfg.get("transition_fidelity", {}).get("amber", {})
    rows: list[dict[str, object]] = []
    for feature, group in validation.groupby("feature_name", sort=True):
        row = group.iloc[0]
        r = float(row["pearson_r"]) if pd.notna(row["pearson_r"]) else np.nan
        nrmse = float(row["normalized_rmse_by_split_sd"]) if pd.notna(row["normalized_rmse_by_split_sd"]) else np.nan
        rmse = float(row["rmse"]) if pd.notna(row["rmse"]) else np.nan
        bias = float(row["bias"]) if pd.notna(row["bias"]) else np.nan
        relative_bias = abs(bias) / rmse if np.isfinite(rmse) and rmse > 0 and np.isfinite(bias) else np.inf
        if (
            np.isfinite(r)
            and np.isfinite(nrmse)
            and r >= float(green.get("pearson_r_minimum", 0.70))
            and nrmse <= float(green.get("normalized_rmse_maximum", 1.0))
            and relative_bias <= float(green.get("absolute_bias_sd_maximum", 0.50))
        ):
            status = "green_reliable_for_explanation"
            rule_eligible = True
            display_caveat = ""
        elif (
            np.isfinite(r)
            and np.isfinite(nrmse)
            and r >= float(amber.get("pearson_r_minimum", 0.40))
            and nrmse <= float(amber.get("normalized_rmse_maximum", 1.50))
        ):
            status = "amber_display_with_residual_warning"
            rule_eligible = False
            display_caveat = "Moderate validation fidelity; do not use as the sole antecedent of an exact rule."
        else:
            status = "red_not_for_case_level_use"
            rule_eligible = False
            display_caveat = "Poor or unstable validation fidelity; suppress from case-level reconstructed evidence."
        rows.append(
            {
                "feature_name": str(feature),
                "fidelity_status": status,
                "rule_antecedent_eligible_from_bhat": rule_eligible,
                "display_caveat": display_caveat,
                "validation_pearson_r": r,
                "validation_normalized_rmse": nrmse,
                "validation_bias": bias,
                "validation_rmse": rmse,
                "validation_relative_bias_to_rmse": relative_bias,
                "validation_n": int(row["n"]),
                "threshold_source": "configs/clinical/clinical_validation_single_reader.yaml",
            }
        )
    return pd.DataFrame(rows)


def flatten_rulebook(
    rulebook: dict[str, Any],
    *,
    training_labels: pd.Series | None = None,
    feature_fidelity: pd.DataFrame | None = None,
    feature_dictionary_hash: str = "",
    transition_model_hash: str = "",
    source_code_commit: str = "",
    rulebook_version: str = "clinical_audit_v1",
) -> pd.DataFrame:
    prevalence: dict[str, float] = {}
    if training_labels is not None and len(training_labels):
        prevalence = training_labels.astype(str).value_counts(normalize=True).to_dict()
    fidelity = {}
    if feature_fidelity is not None and not feature_fidelity.empty:
        fidelity = dict(zip(feature_fidelity["feature_name"].astype(str), feature_fidelity["fidelity_status"].astype(str)))
    rows = []
    for rule in rulebook.get("rules", []):
        pairs = antecedent_pairs(rule)
        feature_states = [{"feature": feature, "state": state} for feature, state in pairs]
        predicted = str(rule.get("predicted_label", ""))
        confidence = float(rule.get("confidence", np.nan))
        base_rate = prevalence.get(predicted, np.nan)
        statuses = [fidelity.get(feature, "not_evaluated") for feature, _ in pairs]
        worst_fidelity = (
            "red_not_for_case_level_use"
            if "red_not_for_case_level_use" in statuses
            else "amber_display_with_residual_warning"
            if "amber_display_with_residual_warning" in statuses
            else "green_reliable_for_explanation"
            if statuses and all(status == "green_reliable_for_explanation" for status in statuses)
            else "not_evaluated"
        )
        row = {
            "rule_id": str(rule.get("rule_id")),
            "rulebook_version": rulebook_version,
            "antecedent_text": antecedent_text(rule),
            "antecedent_json": json.dumps(feature_states, sort_keys=True, separators=(",", ":")),
            "antecedent_length": len(pairs),
            "predicted_label": predicted,
            "support_count": int(rule.get("support", 0)),
            "training_coverage": float(rule.get("training_coverage", np.nan)),
            "confidence": confidence,
            "lift": float(confidence / base_rate) if np.isfinite(base_rate) and base_rate > 0 else np.nan,
            "conflict_rate": float(rule.get("conflict_rate", np.nan)),
            "class_distribution_json": json.dumps(rule.get("class_distribution", {}), sort_keys=True),
            "feature_source": str(rule.get("feature_source", rulebook.get("source_features", "unknown"))),
            "clinical_caveats": str(rule.get("clinical_caveats", "research_use_only")),
            "clinical_support_status": "physician_review_pending",
            "worst_antecedent_fidelity": worst_fidelity,
            "antecedent_hash": sha256_json(feature_states),
            "threshold_version": str(rulebook.get("discretization", "unknown")),
            "feature_dictionary_hash": feature_dictionary_hash,
            "transition_model_hash": transition_model_hash,
            "source_code_commit": source_code_commit,
        }
        row["rule_hash"] = sha256_json(row)
        rows.append(row)
    full = pd.DataFrame(rows)
    if not full.empty and not full["rule_id"].is_unique:
        raise ValueError("Source rule identifiers must be unique")
    return full


def consolidate_rules(full_catalog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if full_catalog.empty:
        return pd.DataFrame(), pd.DataFrame()
    parsed = {
        row.rule_id: frozenset((item["feature"], item["state"]) for item in json.loads(row.antecedent_json))
        for row in full_catalog.itertuples(index=False)
    }
    assigned: set[str] = set()
    canonical_rows: list[dict[str, object]] = []
    map_rows: list[dict[str, object]] = []
    ordered = full_catalog.sort_values(
        ["predicted_label", "antecedent_length", "confidence", "support_count", "rule_id"],
        ascending=[True, True, False, False, True],
        kind="mergesort",
    )
    canonical_index = 0
    for seed in ordered.itertuples(index=False):
        if seed.rule_id in assigned:
            continue
        seed_set = parsed[seed.rule_id]
        candidates = []
        for candidate in ordered.itertuples(index=False):
            if candidate.rule_id in assigned or candidate.predicted_label != seed.predicted_label:
                continue
            candidate_set = parsed[candidate.rule_id]
            if seed_set.issubset(candidate_set):
                candidates.append(candidate)
        if not candidates:
            candidates = [seed]
        canonical_index += 1
        canonical_id = f"CR{canonical_index:03d}"
        source_ids = [candidate.rule_id for candidate in candidates]
        for candidate in candidates:
            assigned.add(candidate.rule_id)
            map_rows.append(
                {
                    "canonical_rule_id": canonical_id,
                    "source_rule_id": candidate.rule_id,
                    "relationship": "identical" if parsed[candidate.rule_id] == seed_set else "nested_more_specific",
                    "reversible": True,
                }
            )
        source_frame = full_catalog.loc[full_catalog["rule_id"].isin(source_ids)]
        canonical_rows.append(
            {
                "canonical_rule_id": canonical_id,
                "predicted_label": seed.predicted_label,
                "canonical_antecedent_text": seed.antecedent_text,
                "canonical_antecedent_json": seed.antecedent_json,
                "source_rule_ids": ";".join(source_ids),
                "source_rule_count": len(source_ids),
                "support_count_max": int(source_frame["support_count"].max()),
                "confidence_min": float(source_frame["confidence"].min()),
                "confidence_max": float(source_frame["confidence"].max()),
                "training_coverage_max": float(source_frame["training_coverage"].max()),
                "worst_antecedent_fidelity": (
                    "red_not_for_case_level_use"
                    if source_frame["worst_antecedent_fidelity"].eq("red_not_for_case_level_use").any()
                    else "amber_display_with_residual_warning"
                    if source_frame["worst_antecedent_fidelity"].eq("amber_display_with_residual_warning").any()
                    else "green_reliable_for_explanation"
                    if source_frame["worst_antecedent_fidelity"].eq("green_reliable_for_explanation").all()
                    else "not_evaluated"
                ),
                "clinical_wording": f"Cine-derived pattern: {seed.antecedent_text}; supports review of the {seed.predicted_label} benchmark family.",
                "required_caveat": "Research-only cine/mask proxy; not a complete clinical diagnosis.",
                "physician_review_status": "pending",
                "canonical_hash": sha256_json({"predicted_label": seed.predicted_label, "antecedent": json.loads(seed.antecedent_json), "source_rule_ids": source_ids}),
            }
        )
    mapping = pd.DataFrame(map_rows)
    canonical = pd.DataFrame(canonical_rows)
    if set(mapping["source_rule_id"]) != set(full_catalog["rule_id"]):
        raise AssertionError("Canonical consolidation lost or duplicated source-rule identifiers")
    if mapping["source_rule_id"].duplicated().any():
        raise AssertionError("A source rule maps to more than one canonical rule")
    return canonical, mapping


def identify_high_risk_rules(full_catalog: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    audit_cfg = cfg.get("rule_audit", {})
    coverage_threshold = float(audit_cfg.get("high_coverage_threshold", 0.05))
    conflict_threshold = float(audit_cfg.get("high_conflict_threshold", 0.20))
    rows = []
    for row in full_catalog.itertuples(index=False):
        reasons = []
        if float(row.training_coverage) >= coverage_threshold:
            reasons.append("high_training_coverage")
        if np.isfinite(float(row.conflict_rate)) and float(row.conflict_rate) >= conflict_threshold:
            reasons.append("high_training_conflict")
        if str(row.predicted_label) in SAFETY_SENSITIVE_LABELS:
            reasons.append("safety_sensitive_family")
        if str(row.worst_antecedent_fidelity).startswith(("amber", "red")):
            reasons.append("non_green_antecedent_fidelity")
        if "MINF" in str(row.predicted_label) or "RV" in str(row.predicted_label):
            reasons.append("mandatory_cine_proxy_wording_review")
        if reasons:
            rows.append({"rule_id": row.rule_id, "high_risk_reasons": ";".join(sorted(set(reasons))), "mandatory_individual_review": True})
    return full_catalog.merge(pd.DataFrame(rows), on="rule_id", how="inner") if rows else full_catalog.iloc[0:0].assign(high_risk_reasons=pd.Series(dtype=str), mandatory_individual_review=pd.Series(dtype=bool))


def rule_feature_map(full_catalog: pd.DataFrame, feature_dictionary: pd.DataFrame, fidelity_gate: pd.DataFrame) -> pd.DataFrame:
    feature_rows = []
    for row in full_catalog.itertuples(index=False):
        for item in json.loads(row.antecedent_json):
            feature_rows.append({"rule_id": row.rule_id, "feature_name": item["feature"], "discretized_state": item["state"]})
    out = pd.DataFrame(feature_rows)
    if out.empty:
        return out
    dictionary = feature_dictionary.rename(columns={"column_name": "feature_name"}).copy()
    keep = [col for col in ["feature_name", "unit", "derivation_formula", "clinical_validity_status", "failure_modes", "transition_target_eligible", "predicate_eligible"] if col in dictionary.columns]
    out = out.merge(dictionary[keep].drop_duplicates("feature_name"), on="feature_name", how="left")
    out = out.merge(fidelity_gate, on="feature_name", how="left")
    return out


def _all_fired_rules(rulebook: dict[str, Any], discretized_row: pd.Series) -> list[dict[str, Any]]:
    fired = []
    for rule in rulebook.get("rules", []):
        if all(str(discretized_row.get(feature, "missing")) == state for feature, state in antecedent_pairs(rule)):
            fired.append(rule)
    return fired


def build_case_rule_trace(
    rulebook: dict[str, Any],
    cutpoints: dict[str, dict[str, float]],
    cases: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    disc = discretize_frame(cases, cutpoints)
    rows = []
    qc_state_by_idx = cases.get("qc_state", pd.Series("qc_pass", index=cases.index)).astype(str)
    open_set_by_idx = cases.get("open_set_outlier_flag", pd.Series(False, index=cases.index)).astype(bool)
    unsupported_by_idx = cases.get("unsupported_feature_flag", pd.Series(False, index=cases.index)).fillna(False).astype(bool)
    indexing_missing_by_idx = cases.get("indexing_missing_flag", pd.Series(False, index=cases.index)).fillna(False).astype(bool)
    boundary_sensitive_by_idx = cases.get("threshold_boundary_flag", pd.Series(False, index=cases.index)).fillna(False).astype(bool)
    for position, (idx, state_row) in enumerate(disc.iterrows()):
        source = cases.iloc[position]
        fired = _all_fired_rules(rulebook, state_row)
        max_specificity = max((len(rule.get("antecedents", []) or []) for rule in fired), default=0)
        decisive = [
            rule
            for rule in fired
            if len(rule.get("antecedents", []) or []) == max_specificity
        ]
        pred, winner, confidence, fire_status = infer_with_rulebook(rulebook, state_row, allow_soft=False)
        all_labels = sorted({str(rule.get("predicted_label")) for rule in fired})
        decisive_labels = sorted({str(rule.get("predicted_label")) for rule in decisive})
        if qc_state_by_idx.iloc[position] == "qc_exclude":
            route = "data_quality_exclusion"
        elif qc_state_by_idx.iloc[position] == "qc_manual_review":
            route = "low_fidelity_review"
        elif bool(unsupported_by_idx.iloc[position]):
            route = "unsupported_feature_review"
        elif bool(open_set_by_idx.iloc[position]) and str(source.get("dataset_source")) == "MMS_100":
            route = "off_ontology_review"
        elif len(decisive_labels) > 1 and "conflict_resolved" not in fire_status:
            route = "conflict_review"
        elif pred == "abstain":
            route = "no_rule_abstention"
        elif bool(indexing_missing_by_idx.iloc[position]) or bool(boundary_sensitive_by_idx.iloc[position]):
            route = "supported_rule_with_caveat"
        elif str(pred) in SAFETY_SENSITIVE_LABELS or str(source.get("dataset_source")) == "MMS_100":
            route = "supported_rule_with_caveat"
        else:
            route = "exact_supported_rule"
        qc_failure_reason = str(source.get("qc_failure_reason") or "").strip()
        if route == "data_quality_exclusion":
            route_reason = qc_failure_reason or "case_level_quality_control_exclusion"
        elif route == "low_fidelity_review":
            route_reason = qc_failure_reason or "case_level_quality_control_manual_review"
        elif route == "unsupported_feature_review":
            route_reason = "required_evidence_or_feature_is_unsupported"
        elif route == "off_ontology_review":
            route_reason = "external_case_outside_frozen_training_envelope"
        elif route in {"conflict_review", "no_rule_abstention"}:
            route_reason = fire_status
        else:
            route_reason = ""
        route_group = cfg.get("routes", {}).get("groups", {}).get(route, "unknown")
        ordered = sorted(fired, key=lambda r: (float(r.get("confidence", 0.0)), int(r.get("support", 0))), reverse=True)
        decisive_ordered = sorted(decisive, key=lambda r: (float(r.get("confidence", 0.0)), int(r.get("support", 0))), reverse=True)
        top_score = float(decisive_ordered[0].get("confidence", 0.0)) * max(int(decisive_ordered[0].get("support", 0)), 1) if decisive_ordered else 0.0
        second_score = float(decisive_ordered[1].get("confidence", 0.0)) * max(int(decisive_ordered[1].get("support", 0)), 1) if len(decisive_ordered) > 1 else 0.0
        margin = (top_score - second_score) / max(top_score, 1e-12) if decisive_ordered else 0.0
        rows.append(
            {
                "reader_case_id": source.get("reader_case_id"),
                "sample_id": source.get("sample_id"),
                "system_route": route,
                "route_group": route_group,
                "predicted_label_restricted": None if pred == "abstain" else pred,
                "winner_rule_id": winner,
                "winner_confidence": float(confidence),
                "fire_status": fire_status,
                "fired_rule_ids": ";".join(str(rule.get("rule_id")) for rule in ordered),
                "fired_rule_labels": ";".join(all_labels),
                "fired_rule_count": len(ordered),
                "decisive_rule_ids": ";".join(str(rule.get("rule_id")) for rule in decisive_ordered),
                "decisive_rule_labels": ";".join(decisive_labels),
                "decisive_rule_count": len(decisive_ordered),
                "conflict_margin": float(margin),
                "missing_evidence_caveat": " ".join(
                    text
                    for text in [
                        (
                            "Cine-only evidence cannot establish tissue characterization, perfusion, phase-contrast flow, shunt fraction, or complete diagnostic criteria."
                            if route in {"supported_rule_with_caveat", "unsupported_feature_review", "off_ontology_review"}
                            else ""
                        ),
                        (
                            "Height or weight is missing or invalid; indexed measurements are suppressed."
                            if bool(indexing_missing_by_idx.iloc[position])
                            else ""
                        ),
                        (
                            "At least one rule feature is adjacent to a frozen cutpoint; treat the route as boundary-sensitive."
                            if bool(boundary_sensitive_by_idx.iloc[position])
                            else ""
                        ),
                        (
                            f"Case-level quality control requires review: {qc_failure_reason}."
                            if route in {"low_fidelity_review", "data_quality_exclusion"} and qc_failure_reason
                            else ""
                        ),
                    ]
                    if text
                ),
                "abstention_reason": route_reason,
            }
        )
    return pd.DataFrame(rows)


def _distance_to_cutpoint(row: pd.Series, pairs: Iterable[tuple[str, str]], cutpoints: dict[str, dict[str, float]]) -> float:
    distances = []
    for feature, _ in pairs:
        value = pd.to_numeric(pd.Series([row.get(feature)]), errors="coerce").iloc[0]
        if not np.isfinite(value) or feature not in cutpoints:
            continue
        cp = cutpoints[feature]
        boundaries = [cp.get("low_high_boundary"), cp.get("normal_high_boundary")]
        distances.extend(abs(float(value) - float(boundary)) for boundary in boundaries if boundary is not None)
    return min(distances) if distances else np.inf


def build_rule_examples(
    rulebook: dict[str, Any],
    cutpoints: dict[str, dict[str, float]],
    development_cases: pd.DataFrame,
) -> pd.DataFrame:
    disc = discretize_frame(development_cases, cutpoints)
    rows = []
    for rule in rulebook.get("rules", []):
        pairs = antecedent_pairs(rule)
        matches = disc.apply(lambda row: all(str(row.get(feature, "missing")) == state for feature, state in pairs), axis=1)
        matched = development_cases.loc[matches.to_numpy()].copy()
        candidates: list[tuple[str, pd.Series | None]] = []
        correct = matched.loc[matched["disease_label"].astype(str).eq(str(rule.get("predicted_label")))]
        incorrect = matched.loc[~matched["disease_label"].astype(str).eq(str(rule.get("predicted_label")))]
        candidates.append(("matched_correct", correct.iloc[0] if len(correct) else None))
        candidates.append(("matched_counterexample", incorrect.iloc[0] if len(incorrect) else None))
        if len(development_cases):
            near_idx = min(development_cases.index, key=lambda idx: _distance_to_cutpoint(development_cases.loc[idx], pairs, cutpoints))
            candidates.append(("near_threshold", development_cases.loc[near_idx]))
        quality = development_cases.loc[development_cases.get("qc_manual_review_required", False).astype(bool)] if "qc_manual_review_required" in development_cases.columns else development_cases.iloc[0:0]
        candidates.append(("quality_failure", quality.iloc[0] if len(quality) else None))
        candidates.append(("limitation", matched.iloc[0] if len(matched) else None))
        for example_type, case in candidates:
            rows.append(
                {
                    "rule_id": str(rule.get("rule_id")),
                    "example_type": example_type,
                    "example_status": "available" if case is not None else "not_available_in_development_partition",
                    "sample_id": None if case is None else case.get("sample_id"),
                    "dataset_source": None if case is None else case.get("dataset_source"),
                    "split": None if case is None else case.get("split"),
                    "benchmark_label_restricted": None if case is None else case.get("disease_label"),
                    "qc_manual_review": None if case is None else bool(case.get("qc_manual_review_required", False)),
                    "rationale": (
                        "Development-partition example only; excluded from primary reader endpoints."
                        if case is not None
                        else "No qualifying development-partition case; physician review must treat this as missing evidence."
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_rule_test_bench(
    rulebook: dict[str, Any],
    cutpoints: dict[str, dict[str, float]],
    canonical: pd.DataFrame,
) -> pd.DataFrame:
    rules_by_id = {str(rule.get("rule_id")): rule for rule in rulebook.get("rules", [])}
    rows = []
    for row in canonical.itertuples(index=False):
        states = {item["feature"]: item["state"] for item in json.loads(row.canonical_antecedent_json)}
        source_ids = str(row.source_rule_ids).split(";")
        representative = next(
            (
                rules_by_id[source_id]
                for source_id in source_ids
                if source_id in rules_by_id
                and dict(antecedent_pairs(rules_by_id[source_id])) == states
                and str(rules_by_id[source_id].get("predicted_label")) == str(row.predicted_label)
            ),
            None,
        )
        if representative is None:
            raise ValueError(f"Canonical rule {row.canonical_rule_id} has no exact representative source rule")
        representative_id = str(representative.get("rule_id"))
        fired_ids = [
            str(rule.get("rule_id"))
            for rule in rulebook.get("rules", [])
            if all(str(states.get(feature, "missing")) == state for feature, state in antecedent_pairs(rule))
        ]
        pred, rule_id, confidence, status = infer_with_rulebook(rulebook, pd.Series(states), allow_soft=False)
        rows.append(
            {
                "test_id": f"{row.canonical_rule_id}-positive",
                "canonical_rule_id": row.canonical_rule_id,
                "test_type": "canonical_positive",
                "target_source_rule_id": representative_id,
                "input_state_json": json.dumps(states, sort_keys=True),
                "expected_consequent": row.predicted_label,
                "observed_consequent": pred,
                "observed_rule_id": rule_id,
                "observed_fired_rule_ids": ";".join(fired_ids),
                "observed_confidence": confidence,
                "observed_status": status,
                "passed": representative_id in fired_ids,
                "safety_critical": False,
                "rationale": "The exact source rule represented by the canonical antecedent must fire. The final conflict-aware route may still abstain.",
            }
        )
        for feature in list(states):
            counterfactual = dict(states)
            counterfactual[feature] = {"low": "mid", "mid": "high", "high": "mid"}.get(states[feature], "missing")
            cf_fired_ids = [
                str(rule.get("rule_id"))
                for rule in rulebook.get("rules", [])
                if all(str(counterfactual.get(ant_feature, "missing")) == ant_state for ant_feature, ant_state in antecedent_pairs(rule))
            ]
            cf_pred, cf_rule, cf_conf, cf_status = infer_with_rulebook(rulebook, pd.Series(counterfactual), allow_soft=False)
            rows.append(
                {
                    "test_id": f"{row.canonical_rule_id}-counterfactual-{feature}",
                    "canonical_rule_id": row.canonical_rule_id,
                    "test_type": "single_antecedent_counterfactual",
                    "target_source_rule_id": representative_id,
                    "input_state_json": json.dumps(counterfactual, sort_keys=True),
                    "expected_consequent": f"not_forced_by_{row.canonical_rule_id}",
                    "observed_consequent": cf_pred,
                    "observed_rule_id": cf_rule,
                    "observed_fired_rule_ids": ";".join(cf_fired_ids),
                    "observed_confidence": cf_conf,
                    "observed_status": cf_status,
                    "passed": representative_id not in cf_fired_ids,
                    "safety_critical": False,
                    "rationale": "Breaking one antecedent must stop the exact representative source rule from firing; another independent rule may still fire.",
                }
            )
    return pd.DataFrame(rows)


def build_scenario_registry() -> pd.DataFrame:
    scenarios = [
        ("SCN-001", "clear_dcm_like", "supported_or_caveated"),
        ("SCN-002", "clear_hcm_like", "supported_or_caveated"),
        ("SCN-003", "clear_abnormal_rv_like", "supported_or_caveated"),
        ("SCN-004", "normal_pattern", "supported_or_caveated"),
        ("SCN-005", "minf_ihd_functional_proxy", "caveat_or_review"),
        ("SCN-006", "conflicting_dilation_hypertrophy", "conflict_review"),
        ("SCN-007", "discordant_lv_rv_stroke_volume", "low_fidelity_review"),
        ("SCN-008", "severe_segmentation_outlier", "data_quality_exclusion"),
        ("SCN-009", "missing_height_weight_indexing", "supported_with_missing_index_caveat"),
        ("SCN-010", "unsupported_lvnc_label", "off_ontology_review"),
        ("SCN-011", "unsupported_myocarditis_or_infiltrative", "unsupported_feature_review"),
        ("SCN-012", "hhd_hcm_overlap", "off_ontology_or_conflict_review"),
        ("SCN-013", "athlete_heart_disease_ambiguity", "off_ontology_review"),
        ("SCN-014", "vendor_shifted_external_measurements", "review_if_fidelity_degrades"),
        ("SCN-015", "threshold_near_boundary", "boundary_sensitive_review"),
        ("SCN-016", "no_rule_or_multiple_rule", "abstain_or_conflict_review"),
    ]
    return pd.DataFrame(scenarios, columns=["scenario_id", "scenario_family", "expected_route_family"]).assign(
        observed_route_family="not_executed",
        passed=False,
        safety_critical=False,
        rationale="Requires deterministic fixture execution before physician release.",
    )


def verify_source_map(full_catalog: pd.DataFrame, canonical: pd.DataFrame, source_map: pd.DataFrame) -> list[str]:
    failures = []
    if set(source_map.get("source_rule_id", [])) != set(full_catalog.get("rule_id", [])):
        failures.append("source_rule_coverage_mismatch")
    if source_map.get("source_rule_id", pd.Series(dtype=str)).duplicated().any():
        failures.append("source_rule_mapped_multiple_times")
    if not set(source_map.get("canonical_rule_id", [])).issubset(set(canonical.get("canonical_rule_id", []))):
        failures.append("unknown_canonical_rule_id")
    if source_map.get("reversible", pd.Series(dtype=bool)).eq(False).any():
        failures.append("non_reversible_mapping")
    return failures
