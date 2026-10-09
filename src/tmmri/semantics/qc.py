from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_B_COLUMNS = [
    "schema_version",
    "sample_id",
    "case_id",
    "patient_id",
    "dataset_source",
    "subset_name",
    "ed_frame_index",
    "es_frame_index",
    "spacing_x_mm",
    "spacing_y_mm",
    "slice_increment_mm",
    "z_spacing_method",
    "heart_rate_bpm",
    "temporal_resolution_ms",
    "height_cm",
    "weight_kg",
    "bmi_kg_m2",
    "bsa_m2",
    "papillary_inclusion_status",
    "lv_edv_ml",
    "lv_esv_ml",
    "lv_sv_ml",
    "lv_ef_percent",
    "rv_edv_ml",
    "rv_esv_ml",
    "rv_sv_ml",
    "rv_ef_percent",
    "lv_myo_volume_ed_ml",
    "lv_myocardial_mass_ed_g",
    "myocardial_density_g_per_ml",
    "global_wall_thickness_ed_mm",
    "global_wall_thickness_es_mm",
    "global_wall_thickening_percent",
    "aha_mapping_feasible_flag",
    "qc_manual_review_required",
    "qc_failure_reason",
]


def validate_b_matrix(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    missing = [col for col in REQUIRED_B_COLUMNS if col not in df.columns]
    rows.append(
        {
            "check_name": "required_columns_present",
            "passed": len(missing) == 0,
            "details": ",".join(missing) if missing else None,
        }
    )
    if "sample_id" in df.columns:
        duplicated = int(df["sample_id"].duplicated().sum())
        rows.append({"check_name": "sample_id_unique", "passed": duplicated == 0, "details": str(duplicated)})
    for col in ["spacing_x_mm", "spacing_y_mm"]:
        if col in df.columns:
            invalid = int((pd.to_numeric(df[col], errors="coerce") <= 0).fillna(True).sum())
            rows.append({"check_name": f"{col}_positive", "passed": invalid == 0, "details": str(invalid)})
    for col in ["lv_ef_percent", "rv_ef_percent"]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            invalid = int(((values < 0) | (values > 100)).sum())
            rows.append({"check_name": f"{col}_range_0_100_or_null", "passed": invalid == 0, "details": str(invalid)})
    for prefix in ["lv", "rv"]:
        edv_col = f"{prefix}_edv_ml"
        esv_col = f"{prefix}_esv_ml"
        if edv_col in df.columns and esv_col in df.columns:
            edv = pd.to_numeric(df[edv_col], errors="coerce")
            esv = pd.to_numeric(df[esv_col], errors="coerce")
            valid_pairs = edv.notna() & esv.notna()
            bad = int((edv[valid_pairs] <= esv[valid_pairs]).sum())
            total = int(valid_pairs.sum())
            rows.append({"check_name": f"{prefix}_edv_gt_esv_for_valid_pairs", "passed": bad <= max(1, int(0.1 * max(total, 1))), "details": f"{bad}/{total}"})
    if "lv_myocardial_mass_ed_g" in df.columns:
        values = pd.to_numeric(df["lv_myocardial_mass_ed_g"], errors="coerce").dropna()
        bad = int((values <= 0).sum())
        rows.append({"check_name": "myocardial_mass_positive_when_present", "passed": bad == 0, "details": str(bad)})
    if "qc_mask_labels_valid" in df.columns:
        bad = int((~df["qc_mask_labels_valid"].astype(bool)).sum())
        rows.append({"check_name": "mask_labels_valid", "passed": bad == 0, "details": str(bad)})
    if "lv_rv_sv_discrepancy_ml" in df.columns and "lv_rv_sv_discrepancy_review_flag" in df.columns:
        discrepancy = pd.to_numeric(df["lv_rv_sv_discrepancy_ml"], errors="coerce")
        bad = int(((discrepancy > 15) & (~df["lv_rv_sv_discrepancy_review_flag"].astype(bool))).sum())
        rows.append({"check_name": "sv_discrepancy_review_flagged", "passed": bad == 0, "details": str(bad)})
    out = pd.DataFrame(rows)
    out["details"] = out["details"].astype("string")
    return out


def qc_case_report(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "sample_id",
        "case_id",
        "patient_id",
        "dataset_source",
        "split",
        "disease_label",
        "qc_manual_review_required",
        "qc_failure_reason",
        "qc_extreme_robust_outlier_flag",
        "qc_extreme_outlier_features",
        "lv_edv_ml",
        "lv_esv_ml",
        "lv_ef_percent",
        "rv_edv_ml",
        "rv_esv_ml",
        "rv_ef_percent",
        "lv_rv_sv_discrepancy_ml",
    ]
    present = [col for col in cols if col in df.columns]
    return df[present].copy()


def write_qc_markdown(df: pd.DataFrame, validation: pd.DataFrame, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset = df["dataset_source"].iloc[0] if len(df) and "dataset_source" in df.columns else "unknown"
    manual = int(df["qc_manual_review_required"].astype(bool).sum()) if "qc_manual_review_required" in df.columns else 0
    lines = [
        f"# {dataset} B Matrix QC Report",
        "",
        "Clinical status: research use only. This pipeline is not clinically validated for diagnosis, triage, treatment selection, or patient management.",
        "",
        f"Rows: {len(df)}",
        f"Manual-review rows: {manual}",
        "",
        "## Validation Gates",
        "",
    ]
    for row in validation.itertuples(index=False):
        status = "PASS" if bool(row.passed) else "FAIL"
        lines.append(f"- {status}: {row.check_name} ({row.details})")
    lines.extend(["", "## Manual Review Reasons", ""])
    if "qc_failure_reason" in df.columns:
        reason_counts = df["qc_failure_reason"].dropna().astype(str).str.get_dummies(sep=";").sum().sort_values(ascending=False)
        if len(reason_counts):
            for reason, count in reason_counts.items():
                if count:
                    lines.append(f"- {reason}: {int(count)}")
        else:
            lines.append("- None")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


# ===============================================================================================================
# Protocol v6 QC severity and robust-outlier references (SPECIFICATION.md 6.0.0, Section 12.4; Decision D21;
# configs/cohorts.yaml qc_severity and qc_outlier_reference). Legacy validate_b_matrix/qc_case_report are unchanged.
# ===============================================================================================================
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

V6_SEVERITY_ORDER = {"none": 0, "informational": 1, "moderate": 2, "high": 3, "critical": 4}
QC_UNREGISTERED_REASON = "QC_UNREGISTERED_REASON"
REPO_ROOT = Path(__file__).resolve().parents[3]


class UnregisteredQCReasonError(ValueError):
    """A QC token that is not registered in configs/cohorts.yaml (contract error; fail closed)."""


def repo_config(relative: str) -> dict[str, Any]:
    """Load a tracked protocol file (configs/..., registry/...) relative to the repository root."""
    import yaml

    with (REPO_ROOT / relative).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _severity_section(qc_policy: dict[str, Any]) -> dict[str, Any]:
    return qc_policy["qc_severity"] if "qc_severity" in qc_policy else qc_policy


def qc_severity_map_v6(qc_policy: dict[str, Any]) -> dict[str, str]:
    section = _severity_section(qc_policy)
    out: dict[str, str] = {}
    for level in ("critical", "high", "moderate", "informational"):
        for reason in section.get(level, []) or []:
            if reason in out:
                raise ValueError(f"QC reason {reason} registered under two severities")
            out[str(reason)] = level
    return out


def reason_code_map(reason_codes: dict[str, Any]) -> dict[str, str]:
    """qc reason token -> reason code (registry/reason_codes.yaml ``qc_reasons``)."""
    out: dict[str, str] = {}
    for entry in reason_codes.get("codes", []):
        for reason in entry.get("qc_reasons", []) or []:
            out.setdefault(str(reason), str(entry["code"]))
    return out


def parse_reason_tokens(value: object) -> list[str]:
    """Semicolon-separated v6 tokens (payloads after ':' never contain ';')."""
    if value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NA:
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return []
    return [t.strip() for t in text.split(";") if t.strip()]


def reason_prefix(token: str) -> str:
    return token.split(":", 1)[0].strip()


@dataclass(frozen=True)
class QCAssessment:
    reasons: tuple[str, ...]
    max_severity: str
    by_severity: dict[str, tuple[str, ...]]
    unregistered: tuple[str, ...]
    route: str | None
    abstain_codes: tuple[str, ...]
    masked_features: tuple[str, ...]
    feature_codes: tuple[str, ...]

    @property
    def critical(self) -> bool:
        return self.route == "QC_ABSTAIN"

    @property
    def manual_review(self) -> bool:
        return V6_SEVERITY_ORDER[self.max_severity] >= V6_SEVERITY_ORDER["moderate"]


def assess_qc_v6(
    reasons: Iterable[str],
    qc_policy: dict[str, Any],
    reason_codes: dict[str, Any] | None = None,
    *,
    strict: bool = False,
) -> QCAssessment:
    """Classify QC tokens with the v6 severities. Severity is keyed by the token prefix (payload after ':').

    critical -> row excluded from every fit, route QC_ABSTAIN with the registered reason code; high -> the configured
    feature masks (cells QC_BLOCKED, code FEATURE_QC_MASKED); moderate/informational -> reported only. An unregistered
    token fails closed: severity critical and code QC_UNREGISTERED_REASON (``strict=True`` raises instead).
    """
    section = _severity_section(qc_policy)
    if section.get("unknown_reason_policy", "fail_closed") != "fail_closed":
        raise ValueError("configs/cohorts.yaml qc_severity.unknown_reason_policy must be fail_closed")
    sev_map = qc_severity_map_v6(section)
    codes = reason_code_map(reason_codes if reason_codes is not None else repo_config("registry/reason_codes.yaml"))
    masks = section.get("feature_masks") or {}
    tokens = [str(t) for t in reasons]
    by: dict[str, list[str]] = {k: [] for k in ("critical", "high", "moderate", "informational")}
    unregistered = []
    for tok in tokens:
        level = sev_map.get(reason_prefix(tok))
        if level is None:
            unregistered.append(tok)
        else:
            by[level].append(tok)
    if unregistered and strict:
        raise UnregisteredQCReasonError(f"Unregistered QC reasons (fail closed): {unregistered}")
    if unregistered or by["critical"]:
        max_sev = "critical"
    else:
        max_sev = next((lvl for lvl in ("high", "moderate", "informational") if by[lvl]), "none")
    abstain: list[str] = []
    for tok in by["critical"]:
        code = codes.get(reason_prefix(tok), QC_UNREGISTERED_REASON)
        if code not in abstain:
            abstain.append(code)
    if unregistered and QC_UNREGISTERED_REASON not in abstain:
        abstain.append(QC_UNREGISTERED_REASON)
    masked: list[str] = []
    for tok in by["high"]:
        for feat in masks.get(reason_prefix(tok), []) or []:
            if feat not in masked:
                masked.append(str(feat))
    return QCAssessment(
        reasons=tuple(tokens),
        max_severity=max_sev,
        by_severity={k: tuple(v) for k, v in by.items()},
        unregistered=tuple(unregistered),
        route="QC_ABSTAIN" if abstain else None,
        abstain_codes=tuple(abstain),
        masked_features=tuple(masked),
        feature_codes=("FEATURE_QC_MASKED",) if masked else (),
    )


_OUTLIER_METHOD = re.compile(r"^\s*robust z = ([0-9.]+) \* \(x - median\) / MAD; flag \|z\| > ([0-9.]+)\s*$")


def _outlier_cfg(qc_policy: dict[str, Any]) -> dict[str, Any]:
    cfg = qc_policy.get("qc_outlier_reference") if "qc_outlier_reference" in qc_policy else None
    if cfg is None:
        raise KeyError("configs/cohorts.yaml qc_outlier_reference is required")
    return cfg


def fit_outlier_reference(
    table: pd.DataFrame,
    registry: dict[str, Any],
    qc_policy: dict[str, Any],
    *,
    fit_mask: np.ndarray | pd.Series | None = None,
    reason_codes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Median/MAD references on the fit partition only (acdc_dev; mms_refit_train in the refit arm), over the
    transition candidates, OBSERVED cells, excluding critical-QC rows. Applied unchanged to every other partition."""
    ocfg = _outlier_cfg(qc_policy)
    m = _OUTLIER_METHOD.match(str(ocfg["method"]))
    if not m:
        raise ValueError(f"Unrecognized qc_outlier_reference.method {ocfg['method']!r}")
    if str(ocfg.get("features")) != "transition_candidates":
        raise ValueError("Only qc_outlier_reference.features: transition_candidates is implemented")
    if str(ocfg.get("zero_mad", "no_flag")) != "no_flag":
        raise ValueError("Only zero_mad: no_flag is implemented")
    from ..target_transforms import observed_values
    from ..transition_fit import transition_candidates_v6

    rows = np.ones(len(table), dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    rc = reason_codes if reason_codes is not None else repo_config("registry/reason_codes.yaml")
    critical = np.array([assess_qc_v6(parse_reason_tokens(v), qc_policy, rc).critical for v in table.get("qc_failure_reason", pd.Series([None] * len(table), index=table.index))], dtype=bool)
    rows &= ~critical
    fit = table.loc[rows]
    min_n = int(ocfg["min_reference_values"])
    features: dict[str, dict[str, Any]] = {}
    for f in transition_candidates_v6(registry):
        if f not in table.columns:
            continue
        x = observed_values(fit, f)
        x = x[np.isfinite(x)]
        if x.size < min_n:
            features[f] = {"median": None, "mad": None, "n": int(x.size), "status": "fewer_than_min_reference_values"}
            continue
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        features[f] = {"median": med, "mad": mad, "n": int(x.size), "status": "zero_mad_no_flag" if not mad > 0 else "ok"}
    return {"schema": "qc_outlier_reference_v6", "scale": float(m.group(1)), "threshold": float(m.group(2)), "features": features,
            "reference_partition": ocfg.get("reference_partition"), "n_fit_rows": int(rows.sum()), "severity": ocfg.get("severity", "moderate")}


def apply_outlier_reference(table: pd.DataFrame, reference: dict[str, Any], qc_policy: dict[str, Any], *, reason_codes: dict[str, Any] | None = None) -> pd.DataFrame:
    """Flag |robust z| above the threshold against the frozen reference; rewrites qc_extreme_robust_outlier_flag,
    qc_extreme_outlier_features, the extreme_robust_outlier token of qc_failure_reason (idempotent) and
    qc_manual_review_required (any reason of severity moderate or higher)."""
    from ..target_transforms import observed_values

    out = table.copy()
    scale, thr = float(reference["scale"]), float(reference["threshold"])
    flagged: list[list[str]] = [[] for _ in range(len(out))]
    for f, ref in reference["features"].items():
        if f not in out.columns or not ref.get("mad") or not ref["mad"] > 0:
            continue
        x = observed_values(out, f)
        with np.errstate(invalid="ignore"):
            z = scale * (x - float(ref["median"])) / float(ref["mad"])
        for i in np.flatnonzero(np.isfinite(z) & (np.abs(z) > thr)):
            flagged[i].append(f)
    rc = reason_codes if reason_codes is not None else repo_config("registry/reason_codes.yaml")
    reasons_out, flags, feats, manual = [], [], [], []
    for i, value in enumerate(out.get("qc_failure_reason", pd.Series([None] * len(out), index=out.index)).tolist()):
        tokens = [t for t in parse_reason_tokens(value) if reason_prefix(t) != "extreme_robust_outlier"]
        if flagged[i]:
            tokens.append("extreme_robust_outlier")
        reasons_out.append(";".join(tokens) if tokens else None)
        flags.append(bool(flagged[i]))
        feats.append(",".join(flagged[i]) if flagged[i] else None)
        manual.append(assess_qc_v6(tokens, qc_policy, rc).manual_review)
    out["qc_failure_reason"] = pd.Series(reasons_out, index=out.index, dtype=object)
    out["qc_extreme_robust_outlier_flag"] = flags
    out["qc_extreme_outlier_features"] = pd.Series(feats, index=out.index, dtype=object)
    out["qc_manual_review_required"] = manual
    return out
