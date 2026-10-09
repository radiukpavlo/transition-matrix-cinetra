from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..io import resolve_path, utc_now_iso, write_table_pair


SEVERITY_ORDER = {"none": 0, "informational": 1, "moderate": 2, "high": 3, "critical": 4}

REASON_SEVERITY = {
    "missing_ed_or_es_mask": "critical",
    "unexpected_label_value": "critical",
    "invalid_label_mapping": "critical",
    "z_spacing_unavailable_for_volume": "critical",
    "lv_edv_le_lvesv": "critical",
    "rv_edv_le_rvesv": "critical",
    "lv_ef_outside_0_100": "critical",
    "rv_ef_outside_0_100": "critical",
    "myocardial_mass_nonpositive": "critical",
    "integer_label_gaussian_smoothed": "critical",
    "high_mask_overlap_before_resolution": "critical",
    "lv_rv_sv_discrepancy_gt_15_ml": "high",
    "ed_es_masks_identical": "high",
    "wall_thickness_invalid_global": "high",
    "extreme_robust_outlier": "moderate",
    "qc_temporal_metadata_review_flag": "moderate",
    "sphericity_short_axis_approximation": "informational",
    "aha_mapping_not_feasible": "informational",
}

FEATURE_GROUP_BY_REASON = {
    "lv_edv_le_lvesv": ["lv_volume", "lv_ef"],
    "rv_edv_le_rvesv": ["rv_volume", "rv_ef"],
    "lv_ef_outside_0_100": ["lv_ef"],
    "rv_ef_outside_0_100": ["rv_ef"],
    "myocardial_mass_nonpositive": ["lv_mass"],
    "lv_rv_sv_discrepancy_gt_15_ml": ["stroke_volume_balance"],
    "wall_thickness_invalid_global": ["wall_thickness"],
    "extreme_robust_outlier": ["outlier_sensitive_features"],
}


def parse_qc_reasons(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return []
    tokens: list[str] = []
    for raw in text.replace(",", ";").split(";"):
        token = raw.strip()
        if not token:
            continue
        # Reasons such as unexpected_label_value:[4] carry payloads. Severity is keyed
        # by the prefix while the raw token is retained for audit.
        tokens.append(token)
    return tokens


def reason_key(reason: str) -> str:
    return reason.split(":", 1)[0].strip()


def severity_for_reasons(reasons: Iterable[str]) -> str:
    severities = [REASON_SEVERITY.get(reason_key(r), "moderate") for r in reasons]
    if not severities:
        return "none"
    return max(severities, key=lambda s: SEVERITY_ORDER[s])


def affected_feature_groups(reasons: Iterable[str]) -> list[str]:
    groups: set[str] = set()
    for reason in reasons:
        groups.update(FEATURE_GROUP_BY_REASON.get(reason_key(reason), []))
    return sorted(groups)


def recommended_action(max_severity: str) -> str:
    if max_severity == "critical":
        return "exclude_from_confirmatory_transition_until_reviewed"
    if max_severity == "high":
        return "include_for_sensitivity_analysis_or_feature_level_exclusion"
    if max_severity == "moderate":
        return "include_with_sensitivity_flag"
    return "include_for_transition"


def build_manual_review_table(b_frames: list[pd.DataFrame], visual_manifest: pd.DataFrame | None = None) -> pd.DataFrame:
    panel_by_sample: dict[str, str] = {}
    if visual_manifest is not None and len(visual_manifest) and "sample_id" in visual_manifest.columns:
        panel_by_sample = dict(zip(visual_manifest["sample_id"].astype(str), visual_manifest.get("panel_path", pd.Series([None] * len(visual_manifest))).astype(str)))
    rows: list[dict[str, object]] = []
    for b_df in b_frames:
        for _, row in b_df.iterrows():
            reasons = parse_qc_reasons(row.get("qc_failure_reason"))
            manual = bool(row.get("qc_manual_review_required", False)) or bool(reasons)
            if not manual:
                continue
            severity = severity_for_reasons(reasons)
            sample_id = str(row.get("sample_id"))
            rows.append(
                {
                    "sample_id": sample_id,
                    "case_id": row.get("case_id"),
                    "patient_id": row.get("patient_id"),
                    "dataset_source": row.get("dataset_source"),
                    "split": row.get("split"),
                    "disease_label": row.get("disease_label"),
                    "qc_failure_reason": row.get("qc_failure_reason"),
                    "parsed_reasons": ";".join(reasons),
                    "max_severity": severity,
                    "recommended_action": recommended_action(severity),
                    "affected_feature_groups": ";".join(affected_feature_groups(reasons)),
                    "visual_panel_path": panel_by_sample.get(sample_id),
                    "adjudication_status": "pending_review",
                    "machine_recommendation": recommended_action(severity),
                    "excluded_features": "",
                    "exclusion_reason": "",
                    "review_priority": SEVERITY_ORDER[severity],
                    "review_status": "pending_review",
                    "reviewer": "",
                    "review_timestamp_utc": "",
                    "review_decision": "",
                    "review_notes": "",
                    "generated_at_utc": utc_now_iso(),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(
            columns=[
                "sample_id",
                "case_id",
                "patient_id",
                "dataset_source",
                "split",
                "disease_label",
                "qc_failure_reason",
                "parsed_reasons",
                "max_severity",
                "recommended_action",
                "affected_feature_groups",
                "visual_panel_path",
                "adjudication_status",
                "machine_recommendation",
                "excluded_features",
                "exclusion_reason",
                "review_priority",
                "review_status",
                "reviewer",
                "review_timestamp_utc",
                "review_decision",
                "review_notes",
                "generated_at_utc",
            ]
        )
    return out.sort_values(["review_priority", "dataset_source", "sample_id"], ascending=[False, True, True]).reset_index(drop=True)


def write_manual_review_artifacts(table: pd.DataFrame, output_dir: str | Path = "artifacts/reports/qc", suffix: str = "v3") -> list[Path]:
    out_dir = resolve_path(output_dir)
    artifacts = write_table_pair(table, out_dir / f"manual_review_cases_{suffix}.parquet", out_dir / f"manual_review_cases_{suffix}.csv")
    md = out_dir / f"manual_review_summary_{suffix}.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Manual-Review Triage Summary",
        "",
        "Clinical status: research use only. This project is not clinically validated for diagnosis, triage, treatment selection, or patient management.",
        "",
        f"Rows requiring manual review: {len(table)}",
        "",
        "Human adjudication fields are initialized to `pending_review`. Machine recommendations are routing suggestions for research QA only.",
        "",
        "## Severity Counts",
        "",
    ]
    if len(table):
        for sev, count in table["max_severity"].value_counts().sort_index().items():
            lines.append(f"- {sev}: {int(count)}")
        lines.extend(["", "## Reason Counts", ""])
        counter: Counter[str] = Counter()
        for value in table["parsed_reasons"].fillna(""):
            for reason in parse_qc_reasons(value):
                counter[reason_key(reason)] += 1
        if counter:
            for reason, count in counter.most_common():
                lines.append(f"- {reason}: {int(count)}")
        else:
            lines.append("- None")
    else:
        lines.append("- none: 0")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifacts.append(md)
    return artifacts


# ===============================================================================================================
# Protocol v6 manual-review triage (configs/cohorts.yaml qc_severity and qc_panels; Decision D21). Uses the v6
# severities (missing structures critical; unregistered reasons fail closed) and carries no wall-clock timestamp, so
# the table is content-hashable. The legacy build_manual_review_table above is unchanged.
# ===============================================================================================================
V6_REVIEW_COLUMNS = [
    "sample_id", "case_id", "dataset_source", "split", "qc_failure_reason", "max_severity", "qc_route", "qc_reason_codes",
    "masked_features", "unregistered_reasons", "adjudication_status", "reviewer", "review_decision", "review_notes",
]


def build_manual_review_table_v6(table: pd.DataFrame, cohorts: dict, reason_codes: dict | None = None) -> pd.DataFrame:
    """One row per case with any reason of severity moderate or higher (or an unregistered reason), sorted by
    severity then sample_id; adjudication fields start at the configured default (``pending_review``)."""
    from .qc import V6_SEVERITY_ORDER, assess_qc_v6, parse_reason_tokens

    default = str(((cohorts.get("qc_panels") or {}).get("adjudication_default")) or "pending_review")
    rows = []
    for _, row in table.iterrows():
        a = assess_qc_v6(parse_reason_tokens(row.get("qc_failure_reason")), cohorts, reason_codes)
        if not a.manual_review:
            continue
        rows.append(
            {
                "sample_id": row.get("sample_id"),
                "case_id": row.get("case_id"),
                "dataset_source": row.get("dataset_source"),
                "split": row.get("split"),
                "qc_failure_reason": row.get("qc_failure_reason"),
                "max_severity": a.max_severity,
                "qc_route": a.route,
                "qc_reason_codes": ";".join(a.abstain_codes + a.feature_codes),
                "masked_features": ";".join(a.masked_features),
                "unregistered_reasons": ";".join(a.unregistered),
                "adjudication_status": default,
                "reviewer": "",
                "review_decision": "",
                "review_notes": "",
                "_order": V6_SEVERITY_ORDER[a.max_severity],
            }
        )
    out = pd.DataFrame(rows, columns=V6_REVIEW_COLUMNS + ["_order"])
    return out.sort_values(["_order", "sample_id"], ascending=[False, True], kind="mergesort").drop(columns="_order").reset_index(drop=True)


def build_qc_review_v6(table: pd.DataFrame, panels: pd.DataFrame, cohorts: dict, reason_codes: dict | None = None,
                       panel_files: dict[str, str] | None = None) -> pd.DataFrame:
    """``qc/manual_review.parquet`` (v6): every case with a moderate-or-higher QC reason
    (:func:`build_manual_review_table_v6`) plus every QC-panel case (``visual_qc.select_qc_panels_v6``), one row per
    case, adjudication fields at the configured default; ``panel_selection`` and ``panel_file`` name the panel."""
    from .qc import V6_SEVERITY_ORDER, assess_qc_v6, parse_reason_tokens

    base = build_manual_review_table_v6(table, cohorts, reason_codes)
    default = str(((cohorts.get("qc_panels") or {}).get("adjudication_default")) or "pending_review")
    by_id = table.set_index(table["sample_id"].astype(str))
    extra = []
    for sid in [s for s in panels["sample_id"].astype(str) if s not in set(base["sample_id"].astype(str))]:
        row = by_id.loc[sid]
        a = assess_qc_v6(parse_reason_tokens(row.get("qc_failure_reason")), cohorts, reason_codes)
        extra.append({"sample_id": sid, "case_id": row.get("case_id"), "dataset_source": row.get("dataset_source"), "split": row.get("split"),
                      "qc_failure_reason": row.get("qc_failure_reason"), "max_severity": a.max_severity, "qc_route": a.route,
                      "qc_reason_codes": ";".join(a.abstain_codes + a.feature_codes), "masked_features": ";".join(a.masked_features),
                      "unregistered_reasons": ";".join(a.unregistered), "adjudication_status": default, "reviewer": "", "review_decision": "",
                      "review_notes": ""})
    out = pd.concat([base, pd.DataFrame(extra, columns=V6_REVIEW_COLUMNS)], ignore_index=True) if extra else base.copy()
    selection = panels.set_index(panels["sample_id"].astype(str))["selection_reason"]
    out["panel_selection"] = [str(selection.get(str(s), "")) for s in out["sample_id"]]
    out["panel_file"] = [str((panel_files or {}).get(str(s), "")) for s in out["sample_id"]]
    order = out["max_severity"].map(lambda v: V6_SEVERITY_ORDER.get(str(v), 0))
    out = out.assign(_order=order).sort_values(["_order", "sample_id"], ascending=[False, True], kind="mergesort")
    return out.drop(columns="_order").reset_index(drop=True)
