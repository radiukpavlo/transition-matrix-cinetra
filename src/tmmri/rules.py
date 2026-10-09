from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import balanced_accuracy_score, f1_score

from .io import resolve_path, utc_now_iso, write_table_pair, write_yaml

DEFAULT_RULE_FEATURES = [
    "lv_ef_percent",
    "rv_ef_percent",
    "lv_edv_ml",
    "rv_edv_ml",
    "lv_esv_ml",
    "rv_esv_ml",
    "lv_myocardial_mass_ed_g",
    "global_wall_thickness_ed_mm",
    "global_wall_thickening_percent",
    "lv_rv_sv_discrepancy_ml",
]


def learn_cutpoints(train_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, dict[str, float]]:
    cutpoints: dict[str, dict[str, float]] = {}
    for col in feature_cols:
        vals = pd.to_numeric(train_df[col], errors="coerce").dropna()
        if len(vals) < 5 or vals.nunique() <= 1:
            continue
        q1, q2 = vals.quantile([1 / 3, 2 / 3]).to_list()
        if q1 == q2:
            continue
        cutpoints[col] = {"low_high_boundary": float(q1), "normal_high_boundary": float(q2), "method": "training_quantile_tertiles_wedd_approx"}
    return cutpoints


def discretize_value(value: object, cp: dict[str, float]) -> str:
    try:
        val = float(value)
    except Exception:
        return "missing"
    if not np.isfinite(val):
        return "missing"
    if val <= cp["low_high_boundary"]:
        return "low"
    if val <= cp["normal_high_boundary"]:
        return "mid"
    return "high"


def discretize_frame(df: pd.DataFrame, cutpoints: dict[str, dict[str, float]]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col, cp in cutpoints.items():
        out[col] = df[col].map(lambda v: discretize_value(v, cp)) if col in df.columns else "missing"
    return out


def _candidate_antecedents(row: pd.Series, max_len: int) -> list[tuple[tuple[str, str], ...]]:
    items = [(col, str(val)) for col, val in row.items() if str(val) != "missing"]
    candidates: list[tuple[tuple[str, str], ...]] = []
    for item in items:
        candidates.append((item,))
    if max_len >= 2:
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                candidates.append((items[i], items[j]))
    if max_len >= 3:
        # Limit cubic explosion by combining only first six high-priority features.
        limited = items[:6]
        for i in range(len(limited)):
            for j in range(i + 1, len(limited)):
                for k in range(j + 1, len(limited)):
                    candidates.append((limited[i], limited[j], limited[k]))
    return candidates


def induce_rules(
    train_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    max_rule_length: int = 3,
    min_support: int = 3,
    min_confidence: float = 0.60,
    feature_source: str = "B_measured",
    clinical_caveats: str = "research_use_only_not_clinically_validated",
    max_rules: int = 200,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, dict[str, float]]]:
    feature_cols = [c for c in (feature_cols or DEFAULT_RULE_FEATURES) if c in train_df.columns]
    cutpoints = learn_cutpoints(train_df, feature_cols)
    disc = discretize_frame(train_df, cutpoints)
    label = train_df["disease_label"].astype(str).reset_index(drop=True)
    disc = disc.reset_index(drop=True)
    buckets: dict[tuple[tuple[str, str], ...], Counter[str]] = defaultdict(Counter)
    for idx, row in disc.iterrows():
        for antecedent in _candidate_antecedents(row, max_rule_length):
            buckets[antecedent][label.iloc[idx]] += 1
    rule_rows = []
    for antecedent, counts in buckets.items():
        support = int(sum(counts.values()))
        if support < min_support:
            continue
        pred_label, pred_count = counts.most_common(1)[0]
        confidence = pred_count / support
        if confidence < min_confidence:
            continue
        conflict_rate = 1.0 - confidence
        rule_rows.append(
            {
                "antecedent_key": antecedent,
                "predicted_label": pred_label,
                "support": support,
                "confidence": confidence,
                "conflict_rate": conflict_rate,
                "class_distribution": dict(counts),
            }
        )
    # Prefer shorter, higher-confidence rules.
    rule_rows = sorted(rule_rows, key=lambda r: (len(r["antecedent_key"]), -r["confidence"], -r["support"], r["predicted_label"]))
    kept = []
    seen = set()
    for row in rule_rows:
        key = row["antecedent_key"]
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
        if len(kept) >= max_rules:
            break
    rules = []
    for idx, row in enumerate(kept, start=1):
        antecedents = [
            {"feature": feat, "operator": "is", "state": state, "discretized_bin": state}
            for feat, state in row["antecedent_key"]
        ]
        rules.append(
            {
                "rule_id": f"R{idx:03d}",
                "antecedents": antecedents,
                "predicted_label": row["predicted_label"],
                "support": int(row["support"]),
                "confidence": float(row["confidence"]),
                "training_coverage": float(row["support"] / max(len(train_df), 1)),
                "conflict_rate": float(row["conflict_rate"]),
                "class_distribution": row["class_distribution"],
                "feature_source": feature_source,
                "clinical_caveats": clinical_caveats,
                "source_predicate_family": "cine_morphology_function_proxy",
            }
        )
    rulebook = {
        "rulebook_schema_version": "3.0",
        "clinical_status": "research_use_only",
        "generated_at_utc": utc_now_iso(),
        "source_features": feature_source,
        "discretization": "training_quantile_tertiles_wedd_approximation",
        "decision_table": "S=(U,C,d,V,f)",
        "default_action": "abstain",
        "rules": rules,
    }
    return rulebook, disc, cutpoints


def _rule_matches(rule: dict[str, object], row: pd.Series) -> bool:
    for ant in rule.get("antecedents", []):
        if str(row.get(ant["feature"], "missing")) != str(ant.get("state", ant.get("discretized_bin"))):
            return False
    return True


def infer_with_rulebook(rulebook: dict[str, object], disc_row: pd.Series, allow_soft: bool = False) -> tuple[str, str | None, float, str]:
    fired = [r for r in rulebook.get("rules", []) if _rule_matches(r, disc_row)]
    if not fired and allow_soft:
        best = None
        best_mismatch = 999
        for r in rulebook.get("rules", []):
            ants = r.get("antecedents", [])
            mismatch = sum(str(disc_row.get(a["feature"], "missing")) != str(a.get("state", a.get("discretized_bin"))) for a in ants)
            if mismatch < best_mismatch and mismatch <= 1:
                best = r
                best_mismatch = mismatch
        if best is not None:
            return str(best["predicted_label"]), str(best["rule_id"]), float(best.get("confidence", 0.0)), "soft_match"
    if not fired:
        return "abstain", None, 0.0, "no_rule_fired"
    max_specificity = max(len(r.get("antecedents", []) or []) for r in fired)
    fired = [r for r in fired if len(r.get("antecedents", []) or []) == max_specificity]
    labels = {str(r["predicted_label"]) for r in fired}
    if len(labels) > 1:
        # Conflict-aware DSS policy: do not force a diagnosis-like label when
        # different rules fire for different classes. Only allow a transparent
        # override when one label has a clear weighted margin and high confidence;
        # otherwise route to abstention/manual review.
        label_scores: dict[str, float] = defaultdict(float)
        label_best_rule: dict[str, dict[str, object]] = {}
        for r in fired:
            lab = str(r["predicted_label"])
            score = float(r.get("confidence", 0.0)) * max(int(r.get("support", 0) or 0), 1)
            label_scores[lab] += score
            if lab not in label_best_rule or (float(r.get("confidence", 0.0)), int(r.get("support", 0) or 0)) > (float(label_best_rule[lab].get("confidence", 0.0)), int(label_best_rule[lab].get("support", 0) or 0)):
                label_best_rule[lab] = r
        ordered_labels = sorted(label_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_label, top_score = ordered_labels[0]
        second_score = ordered_labels[1][1] if len(ordered_labels) > 1 else 0.0
        best = label_best_rule[top_label]
        best_conf = float(best.get("confidence", 0.0))
        margin = (top_score - second_score) / max(top_score, 1e-12)
        if best_conf >= 0.95 and margin >= 0.50:
            return str(best["predicted_label"]), str(best["rule_id"]), best_conf, "conflict_resolved_by_high_margin"
        return "abstain", None, 0.0, "conflicting_rules_abstain"
    best = max(fired, key=lambda r: (float(r.get("confidence", 0)), int(r.get("support", 0))))
    if float(best.get("conflict_rate", 0.0)) > 0.35:
        return "abstain", None, 0.0, "single_rule_high_conflict_abstain"
    return str(best["predicted_label"]), str(best["rule_id"]), float(best.get("confidence", 0.0)), "exact_match"


def evaluate_rulebook(rulebook: dict[str, object], df: pd.DataFrame, cutpoints: dict[str, dict[str, float]], experiment_id: str, split_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    disc = discretize_frame(df, cutpoints)
    rows = []
    for pos, (_, row) in enumerate(disc.iterrows()):
        pred, rule_id, confidence, status = infer_with_rulebook(rulebook, row, allow_soft=True)
        rows.append(
            {
                "sample_id": df.iloc[pos].get("sample_id"),
                "split": split_name,
                "true_label": df.iloc[pos].get("disease_label"),
                "predicted_label": pred,
                "rule_id": rule_id,
                "confidence": confidence,
                "fire_status": status,
                "abstained": pred == "abstain",
                "experiment_id": experiment_id,
            }
        )
    pred_df = pd.DataFrame(rows)
    covered = pred_df[~pred_df["abstained"].astype(bool)]
    if covered.empty:
        metrics = pd.DataFrame([
            {"split": split_name, "coverage": 0.0, "abstention_rate": 1.0, "non_abstained_balanced_accuracy": np.nan, "non_abstained_macro_f1": np.nan, "conflict_rate": 0.0, "conflict_encounter_rate": float(pred_df["fire_status"].astype(str).str.contains("conflict").mean()) if len(pred_df) else 0.0, "forced_conflict_decision_rate": 0.0, "n": len(pred_df), "experiment_id": experiment_id}
        ])
    else:
        metrics = pd.DataFrame([
            {
                "split": split_name,
                "coverage": float(len(covered) / max(len(pred_df), 1)),
                "abstention_rate": float(pred_df["abstained"].mean()),
                "non_abstained_balanced_accuracy": float(balanced_accuracy_score(covered["true_label"].astype(str), covered["predicted_label"].astype(str))),
                "non_abstained_macro_f1": float(f1_score(covered["true_label"].astype(str), covered["predicted_label"].astype(str), average="macro", zero_division=0)),
                "conflict_rate": float(covered["fire_status"].astype(str).str.contains("conflict_resolved").mean()),
                "conflict_encounter_rate": float(pred_df["fire_status"].astype(str).str.contains("conflict").mean()),
                "forced_conflict_decision_rate": float(covered["fire_status"].astype(str).str.contains("conflict_resolved").mean()),
                "n": int(len(pred_df)),
                "experiment_id": experiment_id,
            }
        ])
    return metrics, pred_df


def write_rulebook_artifacts(rulebook: dict[str, object], cutpoints: dict[str, dict[str, float]], metrics: pd.DataFrame, predictions: pd.DataFrame, experiment_id: str) -> list[Path]:
    artifacts: list[Path] = []
    rb_path = resolve_path(f"artifacts/rules/rulebook_ACDC_{experiment_id}.yaml")
    rb_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(rulebook)
    payload["experiment_id"] = experiment_id
    rb_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")
    artifacts.append(rb_path)
    cut_path = write_yaml({"experiment_id": experiment_id, "cutpoints": cutpoints, "clinical_status": "research_use_only"}, f"artifacts/rules/discretization_cutpoints_{experiment_id}.yaml")
    artifacts.append(cut_path)
    artifacts.extend(write_table_pair(metrics, f"artifacts/reports/rules/rulebook_metrics_{experiment_id}.parquet", f"artifacts/reports/rules/rulebook_metrics_{experiment_id}.csv"))
    artifacts.extend(write_table_pair(predictions, f"artifacts/reports/rules/rulebook_predictions_{experiment_id}.parquet", f"artifacts/reports/rules/rulebook_predictions_{experiment_id}.csv"))
    md = resolve_path(f"artifacts/reports/rules/rulebook_evaluation_{experiment_id}.md")
    md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Rulebook Evaluation - {experiment_id}",
        "",
        "Clinical status: research use only. This rulebook is not clinically validated for diagnosis, triage, treatment selection, or patient management.",
        "",
        "Rules are rough-set-style production rules over training-only discretized Matrix B features. Ambiguous or unmatched cases are routed to abstain/manual review.",
        "",
        f"Rules induced: {len(rulebook.get('rules', []))}",
        "",
        metrics.to_markdown(index=False) if not metrics.empty else "No metrics.",
    ]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifacts.append(md)
    return artifacts


# ===============================================================================================================
# Protocol v6 antecedent eligibility gates (SPECIFICATION.md 6.0.0, Section 11.2; configs/rule_policy.yaml
# eligibility_gates; estimator registry row semantic_eligibility_gates). The legacy frequency rules above are unchanged
# and remain the governed/legacy rule path; v6 discovery lives in tmmri.rough_sets and routing in tmmri.inference.
# ===============================================================================================================
import re as _re
from typing import Any as _Any


def antecedent_eligibility_v6(
    measured: pd.DataFrame,
    oof: pd.DataFrame,
    mean_oof: pd.DataFrame,
    outer_folds: np.ndarray,
    cfg: dict[str, _Any],
    registry: dict[str, _Any],
    *,
    rulebook: str = "d_model",
    fit_mask: np.ndarray | None = None,
    stability: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Development-only gates deciding which registry antecedent candidates may enter discretization and discovery.

    Reference availability (observed fraction of fit rows), variability (distinct observed values, fit SD > 0),
    reconstruction (pooled OOF MAE improvement over the mean comparator >= the configured fraction and improvement in
    at least k of the outer folds; only for the rulebooks listed in ``applies_to``), post-discretization stability (at
    least one cut retained by the bootstrap gate, when ``stability`` is given) and the no-tautology rule (the feature is
    a registry antecedent candidate). ``measured`` is the typed B of the fit partition, ``oof`` and ``mean_oof`` the
    nested OOF ridge and mean predictions aligned with it.
    """
    from .rough_sets import antecedent_policy
    from .target_transforms import observed_values

    gates = cfg["eligibility_gates"]
    min_frac = float(gates["reference_availability"]["min_observed_fraction_acdc_dev"])
    min_distinct = int(gates["variability"]["min_distinct_values"])
    rec = gates["reconstruction"]
    min_impr = float(rec["min_relative_mae_improvement_over_mean"])
    m = _re.match(r"^\s*(\d+)_of_(\d+)\s*$", str(rec["folds_required"]))
    if not m:
        raise ValueError("Unrecognized eligibility_gates.reconstruction.folds_required")
    folds_needed = int(m.group(1))
    rec_applies = rulebook in (rec.get("applies_to") or [])
    min_recurrence = float((gates.get("post_discretization_stability") or {}).get("min_cut_recurrence", 0.0))
    allowed, _ = antecedent_policy(registry)
    rows_mask = np.ones(len(measured), dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    folds = pd.to_numeric(pd.Series(np.asarray(outer_folds)), errors="coerce").to_numpy(dtype=float)
    out = []
    for feature in allowed:
        x = observed_values(measured, feature) if feature in measured.columns else np.full(len(measured), np.nan)
        x_fit = x[rows_mask]
        observed = np.isfinite(x_fit)
        frac = float(observed.mean()) if x_fit.size else 0.0
        vals = x_fit[observed]
        n_distinct = int(np.unique(vals).size)
        sd = float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan")
        row = {"feature": feature, "rulebook": rulebook, "observed_fraction": frac, "pass_reference_availability": frac >= min_frac,
               "n_distinct": n_distinct, "fit_sd": sd, "pass_variability": bool(n_distinct >= min_distinct and np.isfinite(sd) and sd > 0)}
        if rec_applies and feature in oof.columns and feature in mean_oof.columns:
            r = pd.to_numeric(oof[feature], errors="coerce").to_numpy(dtype=float)
            z = pd.to_numeric(mean_oof[feature], errors="coerce").to_numpy(dtype=float)
            ok = rows_mask & np.isfinite(x) & np.isfinite(r) & np.isfinite(z) & np.isfinite(folds)
            mae_r = float(np.mean(np.abs(r[ok] - x[ok]))) if ok.any() else float("nan")
            mae_z = float(np.mean(np.abs(z[ok] - x[ok]))) if ok.any() else float("nan")
            pooled = 1.0 - mae_r / mae_z if np.isfinite(mae_z) and mae_z > 0 else float("nan")
            improved = 0
            for f in sorted({int(v) for v in folds[ok]}):
                k = ok & (folds == f)
                fr, fz = np.mean(np.abs(r[k] - x[k])), np.mean(np.abs(z[k] - x[k]))
                improved += int(fz > 0 and fr < fz)
            row.update({"pooled_relative_mae_improvement": pooled, "folds_improved": improved,
                        "pass_reconstruction": bool(np.isfinite(pooled) and pooled >= min_impr and improved >= folds_needed)})
        else:
            row.update({"pooled_relative_mae_improvement": np.nan, "folds_improved": np.nan, "pass_reconstruction": None if not rec_applies else False})
        if stability is not None:
            s = stability[(stability["feature"] == feature) & stability["cut"].notna()]
            retained = int((s["recurrence"] >= min_recurrence).sum()) if len(s) else 0
            row.update({"n_retained_cuts": retained, "pass_stability": retained >= 1})
        else:
            row.update({"n_retained_cuts": np.nan, "pass_stability": None})
        row["pass_no_tautology"] = feature in allowed
        checks = {k: row[k] for k in ("pass_reference_availability", "pass_variability", "pass_reconstruction", "pass_stability", "pass_no_tautology")}
        failing = [k.removeprefix("pass_") for k, v in checks.items() if v is False]
        row["failing_gates"] = ";".join(failing)
        row["eligible"] = not failing
        out.append(row)
    return pd.DataFrame(out)
