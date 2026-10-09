from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .formal_features import build_a_matrix, build_aligned_a_matrices, native_inputs_available, write_a_artifacts
from .classification import run_classification_baseline
from .config import load_config
from .equivariance import run_equivariance_diagnostics
from .metrics import evaluate_residuals, write_transition_evaluation_artifacts
from .io import PROJECT_ROOT, deterministic_seed, ensure_dirs, write_checkpoint, write_table_pair, write_yaml
from .for_rules_import import import_for_rules_archive
from .semantics.manual_review import build_manual_review_table, write_manual_review_artifacts
from .ingest.native_data import validate_native_data
from .predicates import compare_predicates_to_rules, write_predicate_library
from .release import write_release_archive, write_release_manifest
from .rules import DEFAULT_RULE_FEATURES, evaluate_rulebook, induce_rules, write_rulebook_artifacts
from .transition_fit import (
    build_row_inclusion_table,
    build_transition_target_manifest,
    cross_validate_lambdas,
    fit_final_transition,
    reconstruct_bhat,
    select_transition_targets,
    selected_lambda,
    write_transition_artifacts,
)


FULL_VALIDATION_DIRS = [
    "data/features",
    "artifacts/reports/qc/panels",
    "artifacts/reports/qc",
    "artifacts/models/transition",
    "artifacts/reports/transition",
    "artifacts/models/classification",
    "artifacts/reports/classification",
    "artifacts/reports/rules",
    "artifacts/reports/equivariance",
    "artifacts/rules",
    "artifacts/transforms",
    "artifacts/checkpoints",
    "artifacts/provenance",
]


def _read_required_parquet(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required artifact missing: {p}")
    return pd.read_parquet(p)


def _cfg_get(cfg: dict[str, Any], dotted: str, default: Any) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _merge_residuals(residual_frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, list[Path]]:
    all_resid = pd.concat(residual_frames, ignore_index=True) if residual_frames else pd.DataFrame()
    artifacts = write_table_pair(all_resid, "artifacts/models/transition/transition_residuals_all.parquet", "artifacts/models/transition/transition_residuals_all.csv")
    return all_resid, artifacts


def _write_v5_b_validation_artifacts(b_acdc: pd.DataFrame, b_mms: pd.DataFrame, fd: pd.DataFrame) -> list[Path]:
    rows = []
    for dataset, df in [("ACDC", b_acdc), ("MMS_100", b_mms)]:
        rows.append({"dataset_source": dataset, "check_name": "row_count_positive", "passed": len(df) > 0, "details": str(len(df))})
        rows.append({"dataset_source": dataset, "check_name": "sample_id_unique", "passed": df["sample_id"].is_unique, "details": str(int(df["sample_id"].duplicated().sum()))})
        for col in ["lv_edv_ml", "lv_esv_ml", "rv_edv_ml", "rv_esv_ml", "lv_myocardial_mass_ed_g"]:
            vals = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)
            rows.append({"dataset_source": dataset, "check_name": f"{col}_finite", "passed": bool(vals.notna().all()), "details": f"{int(vals.notna().sum())}/{len(df)}"})
        for col in ["lv_ef_percent", "rv_ef_percent"]:
            vals = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)
            rows.append({"dataset_source": dataset, "check_name": f"{col}_physiologic_bounds", "passed": bool(((vals >= 0) & (vals <= 100)).all()), "details": f"{float(vals.min()):.3f}-{float(vals.max()):.3f}"})
        if {"raw_ed_image_path", "raw_es_image_path", "raw_ed_mask_path", "raw_es_mask_path"}.issubset(df.columns):
            rows.append({"dataset_source": dataset, "check_name": "native_ed_es_paths_present", "passed": bool(df[["raw_ed_image_path", "raw_es_image_path", "raw_ed_mask_path", "raw_es_mask_path"]].notna().all().all()), "details": "explicit ED/ES native path columns"})
    validation = pd.DataFrame(rows)
    feature_v5 = fd.copy()
    feature_v5["predicate_eligible"] = feature_v5["column_name"].astype(str).str.contains("lv_|rv_|wall|mass|ef|edv|esv|sv", regex=True)
    feature_v5["clinical_validity_status"] = feature_v5["column_name"].astype(str).map(
        lambda c: "unsupported_without_extra_modalities"
        if any(token in c for token in ["t1", "t2", "lge", "ecv", "flow", "valve", "atrial"])
        else ("research_valid_with_qc" if c in set(fd.loc[fd.get("transition_target_eligible", False).astype(bool), "column_name"].astype(str)) else "audit_or_metadata")
    )
    feature_v5["failure_modes"] = feature_v5["column_name"].astype(str).map(
        lambda c: "missing_native_geometry;invalid_mask;manual_review_flag" if any(token in c for token in ["edv", "esv", "ef", "mass", "wall", "area", "compactness", "eccentricity"]) else ""
    )
    artifacts: list[Path] = []
    artifacts.extend(write_table_pair(validation, "artifacts/reports/qc/B_matrix_validation_v5.parquet", "artifacts/reports/qc/B_matrix_validation_v5.csv"))
    artifacts.extend(write_table_pair(feature_v5, "data/features/feature_dictionary_v5.parquet", "data/features/feature_dictionary_v5.csv"))
    md = PROJECT_ROOT / "artifacts/reports/qc/B_matrix_validation_v5.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Native Matrix B Validation - v5",
        "",
        "Clinical status: research use only. This project is not clinically validated for diagnosis, triage, treatment selection, or patient management.",
        "",
        "Matrix B was regenerated from native image/mask metadata and project-normalized masks for the native v5 run.",
        "",
        validation.to_markdown(index=False),
        "",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    artifacts.append(md)
    return artifacts


def _write_visual_qc_disabled_artifacts(suffix: str) -> tuple[pd.DataFrame, list[Path]]:
    """Record an explicit no-panel branch without invoking a figure backend."""

    columns = [
        "sample_id",
        "case_id",
        "patient_id",
        "dataset_source",
        "split",
        "panel_path",
        "selection_reason",
        "panel_status",
        "qc_manual_review_required",
        "qc_failure_reason",
        "lv_edv_ml",
        "rv_edv_ml",
        "lv_ef_percent",
        "rv_ef_percent",
        "generated_at_utc",
    ]
    manifest = pd.DataFrame(columns=columns)
    artifacts = write_table_pair(
        manifest,
        f"artifacts/reports/qc/visual_qc_manifest_{suffix}.parquet",
        f"artifacts/reports/qc/visual_qc_manifest_{suffix}.csv",
    )
    summary = PROJECT_ROOT / "artifacts" / "reports" / "qc" / f"visual_qc_summary_{suffix}.md"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        "# Visual QC Panel Summary\n\n"
        "Clinical status: research use only.\n\n"
        "Panels requested/generated: 0\n\n"
        "Visual-QC generation was disabled by the frozen configuration. "
        "Reader assets are generated separately by the Nature-only clinical figure workflow.\n",
        encoding="utf-8",
    )
    artifacts.append(summary)
    return manifest, artifacts


def run_full_validation(
    config_path: str | Path = "configs/legacy/pipeline/default.yaml",
    experiment_id: str | None = None,
    run_minimal_if_needed: bool = False,
    enable_classification: bool = False,
    enable_rulebook: bool = False,
    enable_dss_predicates: bool = False,
    enable_equivariance_diagnostics: bool = False,
    allow_semantic_proxy_a: bool = False,
    native_data_required: bool = False,
    bootstrap_iterations: int | None = None,
    max_qa_panels: int | None = None,
    for_rules_archive: str | Path | None = None,
    strict_for_rules_import: bool = True,
) -> list[Path]:
    cfg = load_config(config_path)
    experiment_id = experiment_id or str(_cfg_get(cfg, "full_validation.experiment_id", "v3_dss_rules_001"))
    seed = int(_cfg_get(cfg, "run.algorithm_seed", 0))
    deterministic_seed(seed)
    ensure_dirs(FULL_VALIDATION_DIRS)
    command = f"python -m tmmri.cli run-full-validation --config {config_path} --experiment-id {experiment_id}"
    if for_rules_archive is not None:
        command += f" --for-rules-archive {for_rules_archive}"
    artifacts: list[Path] = []
    artifact_suffix = str(_cfg_get(cfg, "full_validation.artifact_suffix", "v4"))
    native_data_required = bool(native_data_required or _cfg_get(cfg, "data_inputs.native_data_required", False))
    if native_data_required:
        command += " --native-data-required"
    if enable_classification:
        command += " --enable-classification"
    if enable_rulebook:
        command += " --enable-rulebook"
    if enable_dss_predicates:
        command += " --enable-dss-predicates"
    if enable_equivariance_diagnostics:
        command += " --enable-equivariance-diagnostics"

    archive_from_config = _cfg_get(cfg, "data_inputs.for_rules_archive", None)
    archive_to_import = for_rules_archive or archive_from_config
    if native_data_required and archive_to_import:
        raise ValueError("--native-data-required prohibits for_rules/derived archive fallback.")
    if native_data_required:
        native_artifacts = validate_native_data(config_path, "all")
        artifacts.extend(native_artifacts)
        from .cli import run_minimal

        run_minimal(config_path)
        allow_semantic_proxy_a = False
    if archive_to_import:
        imported = import_for_rules_archive(
            archive_to_import,
            experiment_id=experiment_id,
            strict_rows=bool(strict_for_rules_import),
            overwrite_b_matrices=True,
        )
        artifacts.extend(imported.artifacts)

    b_acdc_path = PROJECT_ROOT / "data/features/B_ACDC.parquet"
    b_mms_path = PROJECT_ROOT / "data/features/B_MMS_100.parquet"
    if (not b_acdc_path.exists() or not b_mms_path.exists()) and run_minimal_if_needed:
        from .cli import run_minimal

        run_minimal(config_path)
    b_acdc = _read_required_parquet(b_acdc_path)
    b_mms = _read_required_parquet(b_mms_path)
    fd_path = PROJECT_ROOT / "data/features/B_ACDC_feature_dictionary.parquet"
    fd = _read_required_parquet(fd_path)
    artifacts.extend([b_acdc_path, b_mms_path, fd_path, PROJECT_ROOT / "data/features/B_MMS_100_feature_dictionary.parquet"])
    if native_data_required and (not native_inputs_available(b_acdc) or not native_inputs_available(b_mms)):
        raise FileNotFoundError("Native data required, but B matrices do not contain resolvable native ED/ES image and mask paths.")
    if native_data_required:
        v5_b_artifacts = _write_v5_b_validation_artifacts(b_acdc, b_mms, fd)
        artifacts.extend(v5_b_artifacts)
    write_checkpoint("07_dependency_b_matrices_done.json", command, [b_acdc_path, b_mms_path], [b_acdc_path, b_mms_path], len(b_acdc) + len(b_mms))

    # Visual QC.  A disabled branch must not import or invoke the legacy
    # Matplotlib panel generator; the clinical workflow uses nature-figure
    # exclusively for all reader and manuscript figures.
    visual_qc_enabled = bool(_cfg_get(cfg, "visual_qc.enabled", True))
    if max_qa_panels is not None and int(max_qa_panels) == 0:
        visual_qc_enabled = False
    if visual_qc_enabled:
        from .semantics.visual_qc import generate_visual_qc_panels

        visual_manifest, visual_artifacts = generate_visual_qc_panels(
            b_acdc,
            b_mms,
            max_total_panels=int(max_qa_panels if max_qa_panels is not None else _cfg_get(cfg, "visual_qc.max_total_panels", 120)),
            random_seed=int(_cfg_get(cfg, "visual_qc.random_seed", 20260611)),
            random_cases_per_dataset=int(_cfg_get(cfg, "visual_qc.random_cases_per_dataset", 10)),
            suffix=artifact_suffix,
        )
    else:
        visual_manifest, visual_artifacts = _write_visual_qc_disabled_artifacts(artifact_suffix)
    artifacts.extend(visual_artifacts)
    # A configured skip is not a processing failure.  Its reason is recorded
    # in the summary artifact rather than inflating the checkpoint failure count.
    write_checkpoint("07_visual_qc_done.json", command, [b_acdc_path, b_mms_path], visual_artifacts, len(visual_manifest), [])

    # Manual review triage.
    manual_review = build_manual_review_table([b_acdc, b_mms], visual_manifest)
    manual_artifacts = write_manual_review_artifacts(manual_review, suffix=artifact_suffix)
    artifacts.extend(manual_artifacts)
    write_checkpoint("08_manual_review_triage_done.json", command, [b_acdc_path, b_mms_path], manual_artifacts, len(manual_review))

    # Matrix A extraction.
    latent_dim = int(_cfg_get(cfg, "a_extraction.output_latent_dim", 128))
    a_seed = int(_cfg_get(cfg, "a_extraction.random_projection_seed", 20260611))
    allow_proxy = bool(allow_semantic_proxy_a or _cfg_get(cfg, "a_extraction.allow_semantic_proxy_without_native", False))
    if native_data_required:
        allow_proxy = False
    include_targets_proxy = bool(_cfg_get(cfg, "a_extraction.semantic_proxy_include_transition_targets", False))
    proxy_policy = str(_cfg_get(cfg, "a_extraction.semantic_proxy_policy", "strict_metadata_qc_no_targets"))
    a_acdc, a_mms, meta_acdc, meta_mms = build_aligned_a_matrices(
        b_acdc,
        b_mms,
        latent_dim=latent_dim,
        seed=a_seed,
        allow_semantic_proxy_without_native=allow_proxy,
        include_transition_targets_in_proxy=include_targets_proxy,
        proxy_policy=proxy_policy,
    )
    a_artifacts = []
    a_artifacts.extend(write_a_artifacts(a_acdc, {**meta_acdc, "experiment_id": experiment_id}, "ACDC", experiment_id))
    a_artifacts.extend(write_a_artifacts(a_mms, {**meta_mms, "experiment_id": experiment_id}, "MMS_100", experiment_id))
    artifacts.extend(a_artifacts)
    write_checkpoint("09_A_extraction_done.json", command, [b_acdc_path, b_mms_path], a_artifacts, len(a_acdc) + len(a_mms), [] if meta_acdc.get("native_images_used") else [f"explicit_semantic_proxy_A_used_because_native_images_missing_policy={proxy_policy}"])

    # Transition target selection and row inclusion.
    targets = select_transition_targets(b_acdc, fd, min_train_observed=10)
    if len(targets) < 2:
        raise RuntimeError("Fewer than two transition targets selected; cannot fit transition matrix.")
    target_manifest = build_transition_target_manifest(b_acdc, b_mms, fd, targets)
    row_inclusion = build_row_inclusion_table(b_acdc, allow_high_moderate=True)
    target_artifacts = []
    target_artifacts.extend(write_table_pair(target_manifest, f"artifacts/transforms/transition_target_manifest_{experiment_id}.parquet", f"artifacts/transforms/transition_target_manifest_{experiment_id}.csv"))
    target_artifacts.extend(write_table_pair(row_inclusion, f"artifacts/transforms/transition_row_inclusion_{experiment_id}.parquet", f"artifacts/transforms/transition_row_inclusion_{experiment_id}.csv"))
    artifacts.extend(target_artifacts)
    write_checkpoint("10_transition_target_selection_done.json", command, ["data/features/B_ACDC_feature_dictionary.parquet"], target_artifacts, len(targets))

    a_cols = [c for c in a_acdc.columns if c.startswith("a_")]
    train_mask = b_acdc["split"].astype(str).eq("train") & row_inclusion["included_for_transition"].astype(bool)
    lambda_grid = _cfg_get(cfg, "transition.lambda_grid", None) or [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10, 30, 100]
    cv = cross_validate_lambdas(a_acdc, b_acdc, a_cols, targets, train_mask, lambda_grid=lambda_grid, cv_folds=int(_cfg_get(cfg, "transition.cv_folds_default", 5)), seed=20260611)
    lam = selected_lambda(cv)
    t_matrix, state = fit_final_transition(a_acdc, b_acdc, a_cols, targets, train_mask, lam)
    t_artifacts = write_transition_artifacts(t_matrix, state, cv, target_manifest, row_inclusion, experiment_id, lam)
    artifacts.extend(t_artifacts)
    write_checkpoint("11_transition_cv_done.json", command, ["data/features/A_ACDC.parquet", "data/features/B_ACDC.parquet"], t_artifacts, int(train_mask.sum()))
    write_checkpoint("12_transition_fit_done.json", command, ["data/features/A_ACDC.parquet", "data/features/B_ACDC.parquet"], [f"artifacts/models/transition/T_{experiment_id}.parquet"], int(train_mask.sum()))

    bhat_acdc, resid_acdc, rec_artifacts_acdc = reconstruct_bhat(a_acdc, b_acdc, t_matrix, state, experiment_id, "ACDC")
    bhat_mms, resid_mms, rec_artifacts_mms = reconstruct_bhat(a_mms, b_mms, t_matrix, state, experiment_id, "MMS_100")
    residuals_all = pd.concat([resid_acdc, resid_mms], ignore_index=True)
    residual_artifacts = write_table_pair(residuals_all, f"artifacts/models/transition/transition_residuals_{experiment_id}.parquet", f"artifacts/models/transition/transition_residuals_{experiment_id}.csv")
    artifacts.extend(rec_artifacts_acdc + rec_artifacts_mms + residual_artifacts)

    # Evaluation.
    metrics, failures = evaluate_residuals(
        residuals_all,
        experiment_id=experiment_id,
        bootstrap_iterations=int(bootstrap_iterations or _cfg_get(cfg, "statistics.bootstrap_iterations", 1000)),
        bootstrap_seed=int(_cfg_get(cfg, "statistics.bootstrap_seed", 20260611)),
    )
    eval_artifacts = write_transition_evaluation_artifacts(metrics, failures, experiment_id)
    artifacts.extend(eval_artifacts)
    write_checkpoint("13_transition_evaluation_done.json", command, [f"artifacts/models/transition/transition_residuals_{experiment_id}.parquet"], eval_artifacts, len(metrics))

    # Classification baseline.
    if enable_classification or bool(_cfg_get(cfg, "classification.enabled", False)):
        cls_metrics, cls_bins, cls_artifacts = run_classification_baseline(b_acdc, targets, experiment_id)
        artifacts.extend(cls_artifacts)
        write_checkpoint("14_classification_done.json", command, [b_acdc_path], cls_artifacts, len(cls_metrics))
    else:
        write_checkpoint("14_classification_done.json", command, [], [], 0, ["classification_branch_skipped"])

    # Predicate library and rulebook.
    predicate_artifacts: list[Path] = []
    if enable_dss_predicates or enable_rulebook or bool(_cfg_get(cfg, "rulebook.enabled", False)):
        predicate_artifacts.extend(write_predicate_library(experiment_id))
        artifacts.extend(predicate_artifacts)
    if enable_rulebook or bool(_cfg_get(cfg, "rulebook.enabled", False)):
        train_df = b_acdc[b_acdc["split"].astype(str).eq("train")].copy()
        rule_features = [c for c in DEFAULT_RULE_FEATURES if c in targets or c in b_acdc.columns]
        rulebook, train_disc, cutpoints = induce_rules(train_df, feature_cols=rule_features, max_rule_length=int(_cfg_get(cfg, "rulebook.max_rule_length", 3)), min_support=int(_cfg_get(cfg, "rulebook.min_support", 3)), min_confidence=float(_cfg_get(cfg, "rulebook.min_confidence", 0.60)))
        metric_frames = []
        pred_frames = []
        for split in ["train", "val", "test"]:
            split_df = b_acdc[b_acdc["split"].astype(str).eq(split)].copy()
            if split_df.empty:
                continue
            m, p = evaluate_rulebook(rulebook, split_df, cutpoints, experiment_id, split)
            metric_frames.append(m)
            pred_frames.append(p)
        rb_metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
        rb_preds = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
        rb_artifacts = write_rulebook_artifacts(rulebook, cutpoints, rb_metrics, rb_preds, experiment_id)
        artifacts.extend(rb_artifacts)
        sim_df, sim_md = compare_predicates_to_rules(rulebook, experiment_id)
        artifacts.extend([PROJECT_ROOT / f"artifacts/reports/rules/predicate_similarity_{experiment_id}.parquet", PROJECT_ROOT / f"artifacts/reports/rules/predicate_similarity_{experiment_id}.csv", sim_md])
        write_checkpoint("15_rulebook_done.json", command, [b_acdc_path], rb_artifacts + predicate_artifacts, len(rulebook.get("rules", [])))
    else:
        write_checkpoint("15_rulebook_done.json", command, [], predicate_artifacts, len(predicate_artifacts), ["rulebook_branch_skipped"])

    # Equivariance diagnostics.
    if enable_equivariance_diagnostics or bool(_cfg_get(cfg, "equivariance.enabled", False)):
        eq_artifacts = run_equivariance_diagnostics(
            experiment_id,
            native_inputs_available(b_acdc),
            b_acdc=b_acdc,
            a_acdc=a_acdc,
            t_matrix=t_matrix,
            transition_state=state,
            seed=a_seed,
            max_cases=int(_cfg_get(cfg, "equivariance.max_cases", 12)),
        )
        artifacts.extend(eq_artifacts)
        write_checkpoint("16_equivariance_done.json", command, [], eq_artifacts, 0, [] if native_inputs_available(b_acdc) else ["native_images_missing_equivariance_skipped"])
    else:
        write_checkpoint("16_equivariance_done.json", command, [], [], 0, ["equivariance_branch_skipped"])

    limitations = [
        "research_use_only_not_clinically_validated",
        "not_for_diagnosis_triage_treatment_or_patient_management",
        "transition_weights_not_causal",
    ]
    if not meta_acdc.get("native_images_used"):
        limitations.append("packaged_run_used_explicit_semantic_proxy_A_because_native_NIfTI_inputs_were_absent")
    else:
        limitations.append("native_image_mask_matrix_A_used_no_semantic_proxy_fallback")
    release_manifest = write_release_manifest(experiment_id, command, artifacts, limitations=limitations)
    artifacts.append(release_manifest)
    archive_artifacts = write_release_archive(experiment_id, release_manifest)
    artifacts.extend(archive_artifacts)
    write_checkpoint("17_release_manifest_done.json", command, artifacts, [release_manifest, *archive_artifacts], len(artifacts))
    return artifacts
