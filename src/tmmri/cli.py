from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .config import cfg_path, load_config
from .pipeline import run_full_validation
from .semantics.features import (
    add_robust_outlier_flags,
    build_acdc_b_matrix,
    build_mms_b_matrix,
    default_transition_feature_columns,
    extract_mms_ed_es_derivatives,
    feature_dictionary,
)
from .ingest.inventory import build_acdc_inventory, build_mms_inventory, dataset_provenance
from .ingest.native_data import validate_native_data
from .io import (
    PROJECT_ROOT,
    deterministic_seed,
    ensure_dirs,
    git_commit,
    resolve_path,
    runtime_environment,
    utc_now_iso,
    write_checkpoint,
    write_table_pair,
    write_yaml,
)
from .provenance import write_provenance_manifest
from .semantics.qc import qc_case_report, validate_b_matrix, write_qc_markdown
from .splits import acdc_patient_split, select_mms_100


LAYOUT_DIRS = [
    "configs/clinical",
    "configs/legacy/labels",
    "configs/legacy/pipeline",
    "configs/legacy/segmentation",
    "data/raw/acdc",
    "data/raw/mms",
    "data/interim/acdc",
    "data/interim/mms_100",
    "data/derived/mms_100_ed_es/images",
    "data/derived/mms_100_ed_es/masks",
    "data/derived/mms_100_ed_es/metadata",
    "data/features",
    "data/for_rules",
    "data/raw_registry",
    "data/splits",
    "artifacts/checkpoints",
    "artifacts/models/classification",
    "artifacts/models/transition",
    "artifacts/provenance",
    "artifacts/releases",
    "artifacts/reports/qc",
    "artifacts/reports/segmentation",
    "artifacts/reports/classification",
    "artifacts/reports/transition",
    "artifacts/reports/rules",
    "artifacts/reports/equivariance",
    "artifacts/rules",
    "artifacts/transforms",
    "docs/methodology",
    "docs/references",
    "manuscript/article/Definitions",
    "manuscript/article/figures/clinical_validation",
    "manuscript/article/figures/generated",
    "manuscript/article/tables/generated",
    "manuscript/archive",
    "manuscript/build/article",
    "manuscript/build/semtra_awa2",
    "manuscript/dist",
    "manuscript/notes",
    "manuscript/reports/clinical_validation",
    "manuscript/scripts",
    "scripts/clinical_validation",
    "scripts/data",
    "scripts/experiments",
    "scripts/maintenance",
    "tests",
]


CLINICAL_CONFIG_DEFAULT = "configs/clinical/clinical_validation_single_reader.yaml"
CLINICAL_ROOT_DEFAULT = "clinical_validation"
CLINICAL_FIGURE_ROOT_DEFAULT = "clinical_validation/figures"
MANUSCRIPT_FIGURE_ROOT_DEFAULT = "manuscript/article/figures/clinical_validation"
MANUSCRIPT_REPORT_ROOT_DEFAULT = "manuscript/reports/clinical_validation"
CLINICAL_OUTPUT_ROOT_DEFAULT = "outputs/019fe7a9-eb98-7ca3-b23d-08c18552c607"


def _cli_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return value.as_dict()
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def _emit_json(value: Any) -> None:
    print(json.dumps(value, default=_json_value, indent=2, sort_keys=True, allow_nan=False))


def _read_json_object(path: str | Path, *, description: str) -> dict[str, Any]:
    source = _cli_path(path)
    if not source.is_file():
        raise FileNotFoundError(f"{description} is missing: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object: {source}")
    return value


def _write_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def _run_prepare_clinical_validation(args: argparse.Namespace) -> None:
    from .clinical.validation import prepare_clinical_validation

    result = prepare_clinical_validation(
        config_path=args.config,
        experiment_id=args.experiment_id,
    )
    _emit_json(result)


def _run_generate_clinical_figures(args: argparse.Namespace) -> None:
    from .clinical.figures import generate_all_possible_figures

    result = generate_all_possible_figures(
        primary_manifest_path=args.primary_manifest,
        transition_metrics_path=args.transition_metrics,
        fidelity_gate_path=args.fidelity_gate,
        transition_residuals_acdc_path=args.transition_residuals_acdc,
        transition_residuals_mms_path=args.transition_residuals_mms,
        output_root=args.output_root,
        manuscript_output_root=args.manuscript_output_root,
        manuscript_report_root=args.manuscript_report_root,
    )
    _emit_json(result)


def _run_build_clinical_workbooks(args: argparse.Namespace) -> None:
    # The workbook module owns the Python-to-artifact-tool contract.  Keeping
    # that adapter out of the CLI prevents a second, divergent workbook schema.
    from .clinical import workbook as clinical_workbook

    creator = getattr(clinical_workbook, "write_artifact_tool_build_payload", None)
    if creator is None:
        raise RuntimeError(
            "Workbook generation is blocked: clinical_workbook."
            "write_artifact_tool_build_payload is unavailable. The CLI will not "
            "invent a translation between the validated Python payload and the "
            "approved @oai/artifact-tool builder contract."
        )
    contract_path = Path(
        creator(
            args.payload,
            root=args.clinical_root,
            output_root=args.output_root,
            portable_output_root=args.portable_output_root,
        )
    )
    contract = _read_json_object(contract_path, description="Artifact-tool build payload")
    if not isinstance(contract.get("workbooks"), list) or not contract["workbooks"]:
        raise RuntimeError(
            f"Artifact-tool build payload has no non-empty 'workbooks' list: {contract_path}"
        )
    builder = _cli_path(args.builder_script)
    if not builder.is_file():
        raise FileNotFoundError(f"Approved artifact-tool builder is missing: {builder}")
    completed = subprocess.run(
        [args.node, str(builder), str(contract_path), str(_cli_path(args.preview_root))],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        encoding="utf-8",
        capture_output=True,
    )
    try:
        node_result: Any = json.loads(completed.stdout)
    except json.JSONDecodeError:
        node_result = {"stdout": completed.stdout.strip()}
    hash_writer = getattr(clinical_workbook, "write_workbook_sha256_manifest", None)
    if hash_writer is None:
        raise RuntimeError("Workbook generation completed but detached SHA-256 manifest support is unavailable")
    # The detached hash manifest sits beside the workbooks it binds, so a build
    # into a scratch --output-root never writes into the governed tree.
    hash_manifest, hash_sidecar = hash_writer(
        contract,
        _cli_path(args.output_root) / "07_excel_qa" / "03_workbook_sha256_manifest_v2.json",
    )
    _emit_json(
        {
            "status": "PASS",
            "authoring_backend": "@oai/artifact-tool",
            "payload": str(contract_path.resolve()),
            "builder": str(builder),
            "workbook_sha256_manifest": str(hash_manifest),
            "workbook_sha256_sidecar": str(hash_sidecar),
            "result": node_result,
        }
    )


def _workbook_path(root: Path, kind: str, filename: str) -> Path:
    directory = {
        "unassisted": "01_unassisted",
        "assisted": "02_assisted",
        "rule_audit": "03_rule_audit",
        "retest": "04_retest",
        "analysis": "05_analysis",
    }[kind]
    return root / "workbooks" / directory / filename


def _run_validate_clinical_workbooks(args: argparse.Namespace) -> None:
    from .clinical.workbook import (
        PHYSICIAN_SHEET_MANIFESTS,
        WORKBOOK_FILENAMES,
        WorkbookSources,
        build_workbook_payload,
        inspect_xlsx_package,
        workbook_layout_contract,
    )

    root = _cli_path(args.clinical_root)
    sources = WorkbookSources.from_directory(root)
    kinds = list(WORKBOOK_FILENAMES) if args.kind == "all" else [args.kind]
    if args.workbook is not None and len(kinds) != 1:
        raise ValueError("--workbook may be used only with one explicit --kind")
    reports: dict[str, dict[str, Any]] = {}
    for kind in kinds:
        payload = build_workbook_payload(kind, sources)
        sheets = payload.get("sheets", {})
        if not isinstance(sheets, Mapping) or not sheets:
            raise RuntimeError(f"{kind} workbook payload has no sheet contract")
        expected_sheet_names = list(PHYSICIAN_SHEET_MANIFESTS[kind])
        workbook_path = (
            _cli_path(args.workbook)
            if args.workbook is not None
            else _workbook_path(root, kind, WORKBOOK_FILENAMES[kind])
        )
        hidden: list[str] = []
        protected: list[str] = []
        reports[kind] = inspect_xlsx_package(
            workbook_path,
            workbook_kind=kind,
            restricted_values=sources.restricted_values,
            retest_link_values=sources.retest_link_values,
            expected_sheet_names=expected_sheet_names,
            required_hidden_sheets=hidden,
            required_protected_sheets=protected,
            link_base=root,
            # Approved v2 layout: default tab, headers, Row Check, Validate Before Transfer.
            expected_layout=workbook_layout_contract(kind, payload),
        )
    overall = "PASS" if all(report.get("status") == "PASS" for report in reports.values()) else "FAIL"
    result = {"status": overall, "reports": reports}
    output = _cli_path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    _emit_json({"status": overall, "report": str(output), "workbooks": list(reports)})
    if overall != "PASS":
        raise SystemExit(2)


def _expected_response_ids(root: Path, phase: str) -> list[str]:
    if phase in {"unassisted", "assisted"}:
        manifest = pd.read_csv(root / "cohort" / "primary_reader_manifest.csv", dtype=object)
        column = "reader_case_id"
    elif phase == "retest":
        manifest = pd.read_csv(root / "cohort" / "retest_workbook_manifest.csv", dtype=object)
        column = "retest_case_id"
    elif phase == "rule_audit":
        canonical = pd.read_csv(root / "rules" / "canonical_rule_catalog.csv", dtype=object)
        return canonical["canonical_rule_id"].dropna().astype(str).tolist()
    else:
        raise ValueError(f"Unsupported response phase: {phase}")
    if column not in manifest.columns:
        raise ValueError(f"Expected-ID manifest for {phase} lacks {column!r}")
    return manifest[column].dropna().astype(str).tolist()


def _load_response_table(
    path: str | Path,
    phase: str,
    expected_ids: Iterable[str],
    *,
    require_complete: bool,
    sheet_name: str | None = None,
) -> pd.DataFrame:
    from .clinical.workbook import import_response_export, validate_response_csv

    source = _cli_path(path)
    suffix = source.suffix.casefold()
    schema = {
        "unassisted": "unassisted_v2",
        "assisted": "assisted_v2",
        "rule_audit": "rule_audit_v2",
        "retest": "retest_v2",
    }[phase]
    if suffix == ".xlsx":
        return import_response_export(
            source,
            schema,
            expected_ids=expected_ids,
            sheet_name=sheet_name,
            require_complete=require_complete,
        )
    if suffix == ".csv":
        return validate_response_csv(
            source,
            schema,
            expected_ids=expected_ids,
            require_complete=require_complete,
        )
    raise ValueError(f"Reader response must be a macro-free .xlsx workbook or .csv export: {source}")


def _run_import_reader_responses(args: argparse.Namespace) -> None:
    from .clinical.release import (
        create_phase_lock,
        preserve_original_response,
        verify_detached_sha256,
        write_detached_sha256,
    )

    root = _cli_path(args.clinical_root)
    expected_ids = _expected_response_ids(root, args.phase)
    frame = _load_response_table(
        args.input,
        args.phase,
        expected_ids,
        require_complete=not args.allow_incomplete,
        sheet_name=args.sheet_name,
    )
    closing_questions: pd.DataFrame | None = None
    if args.phase == "rule_audit":
        source = _cli_path(args.input)
        if source.suffix.casefold() != ".xlsx":
            raise ValueError("The v2 rule audit must be imported from XLSX so the five closing questions are preserved")
        from .clinical.workbook import import_rule_closing_questions

        closing_questions = import_rule_closing_questions(
            source,
            sheet_name=args.sheet_name or "02_Rule_Assessment",
            require_complete=not args.allow_incomplete,
        )
    responses_root = root / "responses"
    preserved = preserve_original_response(args.input, responses_root, phase=args.phase)
    validated = responses_root / args.phase / "validated" / f"{preserved.sha256}.csv"
    if validated.exists():
        existing = pd.read_csv(validated, dtype=object, keep_default_na=False)
        if existing.astype(str).to_dict("records") != frame.astype(str).to_dict("records"):
            raise FileExistsError(f"Validated response table conflicts with existing immutable export: {validated}")
    else:
        _write_frame(frame, validated)
    validated_hash = validated.with_suffix(validated.suffix + ".sha256")
    if validated_hash.exists():
        detached = verify_detached_sha256(validated, validated_hash)
        if detached.get("status") != "PASS":
            raise ValueError(f"Validated response detached hash is invalid: {detached.get('issue')}")
    else:
        validated_hash = write_detached_sha256(validated)
    lock_path = responses_root / "locks" / f"{args.phase}.lock.json"
    artifacts = [
        preserved.preserved_path,
        preserved.preserved_path.with_suffix(preserved.preserved_path.suffix + ".sha256"),
        preserved.receipt_path,
        preserved.receipt_path.with_suffix(preserved.receipt_path.suffix + ".sha256"),
        validated,
        validated_hash,
    ]
    closing_questions_path: Path | None = None
    if closing_questions is not None:
        closing_questions_path = validated.with_name(f"{preserved.sha256}.closing_questions.csv")
        if closing_questions_path.exists():
            existing = pd.read_csv(closing_questions_path, dtype=object, keep_default_na=False)
            if existing.astype(str).to_dict("records") != closing_questions.astype(str).to_dict("records"):
                raise FileExistsError(f"Closing-question table conflicts with existing immutable export: {closing_questions_path}")
        else:
            _write_frame(closing_questions, closing_questions_path)
        closing_hash = closing_questions_path.with_suffix(closing_questions_path.suffix + ".sha256")
        if closing_hash.exists():
            detached = verify_detached_sha256(closing_questions_path, closing_hash)
            if detached.get("status") != "PASS":
                raise ValueError(f"Closing-question detached hash is invalid: {detached.get('issue')}")
        else:
            closing_hash = write_detached_sha256(closing_questions_path)
        artifacts.extend([closing_questions_path, closing_hash])
    artifacts.extend(_cli_path(value) for value in args.additional_artifact)
    created_lock = create_phase_lock(
        args.phase,
        artifacts,
        lock_path,
        root=root,
        metadata={
            "schema": args.phase,
            "row_count": int(len(frame)),
            "expected_row_count": int(len(expected_ids)),
            "require_complete": not args.allow_incomplete,
            "closing_question_count": 0 if closing_questions is None else int(len(closing_questions)),
        },
    )
    _emit_json(
        {
            "status": "PASS",
            "phase": args.phase,
            "row_count": len(frame),
            "expected_row_count": len(expected_ids),
            "preserved_original": str(preserved.preserved_path),
            "validated_table": str(validated),
            "closing_questions": str(closing_questions_path) if closing_questions_path is not None else None,
            "receipt": str(preserved.receipt_path),
            "phase_lock": str(created_lock),
        }
    )


def _tri_state(value: str) -> bool | None:
    return {"yes": True, "no": False, "unknown": None}[value]


def _require_phase_locks(root: Path, phase_inputs: Mapping[str, str | Path]) -> list[Path]:
    from .clinical.release import verify_phase_lock
    from .io import sha256_file

    locks: list[Path] = []
    for phase, response_path in phase_inputs.items():
        lock = root / "responses" / "locks" / f"{phase}.lock.json"
        report = verify_phase_lock(lock, root=root)
        if report.get("status") != "PASS":
            raise RuntimeError(
                f"Reader-study analysis is blocked: {phase!r} phase lock did not pass "
                f"({report.get('status')}; issues={report.get('issues')})."
            )
        payload = _read_json_object(lock, description=f"{phase} phase lock")
        locked_hashes = {
            str(row.get("sha256", "")).lower()
            for row in payload.get("artifacts", [])
            if isinstance(row, Mapping)
        }
        response = _cli_path(response_path)
        response_hash = sha256_file(response).lower()
        if response_hash not in locked_hashes:
            raise RuntimeError(
                f"Reader-study analysis is blocked: supplied {phase!r} response "
                "is not one of the byte-for-byte artifacts in its verified phase lock."
            )
        locks.append(lock)
    return locks


def _rule_audit_with_metadata(root: Path, responses: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    canonical = pd.read_csv(root / "rules" / "canonical_rule_catalog.csv", dtype=object)
    high_risk = pd.read_csv(root / "rules" / "high_risk_rules.csv", dtype=object)
    canonical_meta = pd.DataFrame(
        {
            "rule_id": canonical["canonical_rule_id"].astype(str),
            "expected_rule_scope": "canonical",
            "rule_family": canonical.get("predicted_label", pd.Series(index=canonical.index, dtype=object)),
            "support_status": canonical.get("worst_antecedent_fidelity", pd.Series(index=canonical.index, dtype=object)),
        }
    )
    high_risk_meta = pd.DataFrame(
        {
            "rule_id": high_risk["rule_id"].astype(str),
            "expected_rule_scope": "high-risk source",
            "rule_family": high_risk.get("predicted_label", pd.Series(index=high_risk.index, dtype=object)),
            "support_status": high_risk.get("clinical_support_status", pd.Series(index=high_risk.index, dtype=object)),
        }
    )
    metadata = pd.concat([canonical_meta, high_risk_meta], ignore_index=True)
    if metadata["rule_id"].duplicated().any():
        raise RuntimeError("Frozen canonical/high-risk rule-review identifiers are not unique")
    if "rule_scope" not in responses.columns:
        # rule_audit_v2 has no rule_scope: protocol v2 section 7 audits exactly the frozen canonical rules (source-rule
        # escalation is a separate manifest), so the frozen catalog supplies the scope and the audit denominator.
        metadata = metadata.loc[metadata["expected_rule_scope"].eq("canonical")].reset_index(drop=True)
        responses = responses.assign(rule_scope="canonical")
    merged = metadata.merge(responses, on="rule_id", how="left", validate="one_to_one")
    mismatch = merged["rule_scope"].astype(str).ne(merged["expected_rule_scope"].astype(str))
    if mismatch.any():
        ids = merged.loc[mismatch, "rule_id"].head(5).tolist()
        raise ValueError(f"Rule-review scope does not match the frozen catalog for {ids}")
    return merged, len(canonical), len(metadata)


def _analysis_case_table(root: Path, unassisted: pd.DataFrame, assisted: pd.DataFrame) -> pd.DataFrame:
    manifest = pd.read_csv(root / "cohort" / "primary_reader_manifest_restricted.csv", dtype=object)
    if len(manifest) != 150 or manifest["reader_case_id"].duplicated().any():
        raise RuntimeError("Frozen primary reader manifest must contain 150 unique reader_case_id values")
    paired = unassisted.merge(
        assisted,
        on="reader_case_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_unassisted_record", "_assisted_record"),
    )
    cases = manifest.merge(paired, on="reader_case_id", how="left", validate="one_to_one")
    if len(cases) != 150:
        raise RuntimeError("Primary-case merge changed the frozen denominator")
    cases["exact_acdc_rule_assigned"] = cases["system_route"].astype(str).isin(
        {"exact_supported_rule", "supported_rule_with_caveat"}
    )
    cases["is_off_ontology"] = cases["reference_off_ontology_restricted"]
    return cases


def _safety_findings_register(cases: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def resolution(value: object) -> str:
        token = str(value).strip().lower()
        if token == "yes":
            return "resolved"
        if token in {"not applicable", "not_applicable"}:
            return "not-applicable"
        return "open"

    case_flag = cases["explanation_misleading"].astype(str).str.strip().str.lower().eq("yes")
    for _, row in cases.loc[case_flag].iterrows():
        rows.append(
            {
                "finding_id": f"CASE-{row['reader_case_id']}",
                "source": "assisted_case_review",
                "severity": row.get("defect_severity"),
                "resolution_status": resolution(row.get("safety_event_resolved")),
                "source_record_id": row["reader_case_id"],
            }
        )
    rule_flag = ~rules["defect_severity"].astype(str).str.strip().str.lower().isin(
        {"", "none", "nan", "not applicable", "not_applicable"}
    )
    for _, row in rules.loc[rule_flag].iterrows():
        rows.append(
            {
                "finding_id": f"RULE-{row['rule_id']}",
                "source": "global_rule_audit",
                "severity": row.get("defect_severity"),
                "resolution_status": resolution(row.get("defect_resolved")),
                "source_record_id": row["rule_id"],
            }
        )
    return pd.DataFrame(
        rows,
        columns=("finding_id", "source", "severity", "resolution_status", "source_record_id"),
    )


def _run_analyze_reader_study(args: argparse.Namespace) -> None:
    from .clinical.statistics_reader_study import (
        assemble_reader_study_analysis,
        explanation_utility_endpoints,
        intra_rater_metrics,
        misleading_explanation_endpoints,
        off_ontology_metrics,
        paired_correctness_analysis,
        paired_ordinal_analysis,
        physician_benchmark_agreement,
        primary_endpoint_targets,
        review_routing_metrics,
        rule_acceptance_endpoints,
    )

    root = _cli_path(args.clinical_root)
    phase_inputs: dict[str, str | Path] = {
        "unassisted": args.unassisted,
        "assisted": args.assisted,
        "rule_audit": args.rule_audit,
    }
    if args.retest is not None:
        phase_inputs["retest"] = args.retest
    locks = _require_phase_locks(root, phase_inputs)
    unassisted = _load_response_table(
        args.unassisted,
        "unassisted",
        _expected_response_ids(root, "unassisted"),
        require_complete=True,
    )
    assisted = _load_response_table(
        args.assisted,
        "assisted",
        _expected_response_ids(root, "assisted"),
        require_complete=True,
    )
    rule_response = _load_response_table(
        args.rule_audit,
        "rule_audit",
        _expected_response_ids(root, "rule_audit"),
        require_complete=True,
    )
    # The v2 contract names the physician fields `assessment`, `confidence_rating` and `image_quality_rating` in every
    # phase, so the per-phase merges suffix them (`assessment_unassisted_record`, ...); the endpoints below read
    # those names (the v1 defaults of statistics_reader_study never exist in a v2 case table).
    cases = _analysis_case_table(root, unassisted, assisted)
    rule_ratings, canonical_rule_n, audit_rule_n = _rule_audit_with_metadata(root, rule_response)

    analysis_cfg = load_config(args.config).get("statistics", {})
    iterations = int(args.bootstrap_iterations if args.bootstrap_iterations is not None else analysis_cfg.get("bootstrap_iterations", 2000))
    seed = int(args.bootstrap_seed if args.bootstrap_seed is not None else analysis_cfg.get("bootstrap_seed", 20260815))
    confidence = float(args.confidence if args.confidence is not None else analysis_cfg.get("confidence_interval", 0.95))
    endpoint_tables: list[pd.DataFrame] = [
        rule_acceptance_endpoints(
            rule_ratings,
            expected_n=canonical_rule_n,
            expected_audit_n=audit_rule_n,
            confidence=confidence,
            bootstrap_iterations=iterations,
            bootstrap_seed=seed,
        ),
        explanation_utility_endpoints(
            cases,
            patient_col="reader_case_id",
            expected_n=150,
            confidence=confidence,
            bootstrap_iterations=iterations,
            bootstrap_seed=seed,
        ),
        misleading_explanation_endpoints(
            cases,
            patient_col="reader_case_id",
            expected_n=150,
            confidence=confidence,
        ),
        paired_ordinal_analysis(
            cases,
            first_col="confidence_rating_unassisted_record",
            second_col="confidence_rating_assisted_record",
            endpoint_prefix="physician_confidence",
            patient_col="reader_case_id",
            expected_n=150,
            confidence=confidence,
            bootstrap_iterations=iterations,
            bootstrap_seed=seed,
        ),
    ]
    agreement_results = []
    acdc = cases.loc[cases["dataset_source"].astype(str).str.upper().eq("ACDC")].copy()
    if len(acdc) != 50:
        raise RuntimeError(f"Frozen ACDC reader denominator must be 50, observed {len(acdc)}")
    agreement_results.extend(
        [
            physician_benchmark_agreement(
                acdc,
                assessment_col="assessment_unassisted_record",
                phase="unassisted",
                patient_col="reader_case_id",
                benchmark_col="benchmark_label_restricted",
                expected_n=50,
                confidence=confidence,
                bootstrap_iterations=iterations,
                bootstrap_seed=seed,
            ),
            physician_benchmark_agreement(
                acdc,
                assessment_col="assessment_assisted_record",
                phase="assisted",
                patient_col="reader_case_id",
                benchmark_col="benchmark_label_restricted",
                expected_n=50,
                confidence=confidence,
                bootstrap_iterations=iterations,
                bootstrap_seed=seed,
            ),
        ]
    )
    paired_results = [
        paired_correctness_analysis(
            acdc,
            patient_col="reader_case_id",
            benchmark_col="benchmark_label_restricted",
            unassisted_col="assessment_unassisted_record",
            assisted_col="assessment_assisted_record",
            expected_n=50,
            confidence=confidence,
            bootstrap_iterations=iterations,
            bootstrap_seed=seed,
        )
    ]
    routing_results = [
        review_routing_metrics(
            cases,
            patient_col="reader_case_id",
            expected_n=150,
            confidence=confidence,
            bootstrap_iterations=iterations,
            bootstrap_seed=seed,
            # SAP v2 estimand 4 on the unassisted v2 fields (no overlapping-phenotype field in v2).
            review_target_kwargs={
                "segmentation_component_enabled": False,
                "confidence_col": "confidence_rating_unassisted_record",
                "assessment_col": "assessment_unassisted_record",
                "image_quality_col": "image_quality_rating",
                "additional_sequences_col": "additional_sequences_needed",
                "sufficient_for_task_col": "task_sufficient",
                "overlap_unsafe_col": None,
                "review_indications_col": "review_indications_unassisted_record",
            },
        )
    ]
    mms = cases.loc[cases["dataset_source"].astype(str).str.upper().isin({"MMS", "M&MS", "MMS_100"})].copy()
    off_n = int(
        mms["is_off_ontology"].astype(str).str.strip().str.lower().isin({"1", "true", "yes"}).sum()
    )
    endpoint_tables.append(
        off_ontology_metrics(
            mms,
            patient_col="reader_case_id",
            expected_off_ontology_n=off_n,
            confidence=confidence,
            bootstrap_iterations=iterations,
            bootstrap_seed=seed,
        )
    )

    intra_result = None
    if args.retest is not None:
        retest = _load_response_table(
            args.retest,
            "retest",
            _expected_response_ids(root, "retest"),
            require_complete=True,
        )
        retest_map = pd.read_csv(root / "restricted" / "restricted_retest_map.csv", dtype=object)
        first = unassisted.merge(retest_map, on="reader_case_id", how="inner", validate="one_to_one")
        pairs = first.merge(
            retest,
            on="retest_case_id",
            how="inner",
            validate="one_to_one",
            suffixes=("_unassisted_record", "_retest_record"),
        )
        # retest_v2 mirrors unassisted_v2 and carries no physician measurements, so there are no continuous pairs.
        intra_result = intra_rater_metrics(
            pairs,
            patient_col="reader_case_id",
            first_assessment_col="assessment_unassisted_record",
            repeat_assessment_col="assessment_retest_record",
            first_confidence_col="confidence_rating_unassisted_record",
            repeat_confidence_col="confidence_rating_retest_record",
            first_image_quality_col="image_quality_rating_unassisted_record",
            repeat_image_quality_col="image_quality_rating_retest_record",
            expected_n=24,
            confidence=confidence,
            bootstrap_iterations=iterations,
            bootstrap_seed=seed,
        )
        endpoint_tables.append(intra_result.endpoints)

    gate_kwargs = {
        "off_ontology_not_systematically_forced": _tri_state(args.off_ontology_not_systematically_forced),
        "unassisted_blinding_breach": _tri_state(args.unassisted_blinding_breach),
        "serious_protocol_deviations_analyzed": _tri_state(args.serious_protocol_deviations_analyzed),
    }
    analysis = assemble_reader_study_analysis(
        endpoint_tables=endpoint_tables,
        agreement_results=agreement_results,
        paired_results=paired_results,
        routing_results=routing_results,
        gate_kwargs=gate_kwargs,
        bootstrap_iterations=iterations,
        bootstrap_seed=seed,
        confidence=confidence,
        log_additional={
            "study_id": "tm_mri_single_physician_v2",
            "clinical_status": "research_use_only",
            "phase_locks": [str(path) for path in locks],
            "inter_rater_claim_permitted": False,
        },
    )
    targets = primary_endpoint_targets(analysis.endpoints)
    output = _cli_path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    written = [
        _write_frame(analysis.endpoints, output / "reader_study_endpoints.csv"),
        _write_frame(analysis.missing_denominators, output / "missing_denominators.csv"),
        _write_frame(analysis.gates, output / "safety_gates.csv"),
        _write_frame(targets, output / "primary_endpoint_targets.csv"),
        _write_frame(analysis.confusion_matrix, output / "physician_benchmark_confusion_matrices.csv"),
        _write_frame(analysis.paired_transitions, output / "paired_assessment_transitions.csv"),
        _write_frame(analysis.routing_contingency, output / "review_routing_contingency.csv"),
        _write_frame(analysis.analysis_log, output / "analysis_log.csv"),
        _write_frame(cases, output / "restricted_case_analysis_dataset.csv"),
        _write_frame(rule_ratings, output / "restricted_rule_analysis_dataset.csv"),
    ]
    if intra_result is not None:
        written.append(_write_frame(intra_result.case_level, output / "restricted_intra_rater_case_level.csv"))
    findings = _safety_findings_register(cases, rule_ratings)
    findings_path = _write_frame(findings, output / "safety_findings.csv")
    written.append(findings_path)
    manifest = {
        "status": "ANALYSIS_COMPLETE_GATES_REPORTED",
        "clinical_status": "research_use_only",
        "primary_cases": len(cases),
        "acdc_cases": len(acdc),
        "mms_cases": len(mms),
        "canonical_rule_denominator": canonical_rule_n,
        "complete_rule_audit_denominator": audit_rule_n,
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
        "confidence": confidence,
        "safety_finding_count": len(findings),
        "outputs": [str(path) for path in written],
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path = output.parent / "endpoint_manifest.json"
    manifest_path.write_text(manifest_text, encoding="utf-8")
    (output / "analysis_manifest.json").write_text(manifest_text, encoding="utf-8")
    _emit_json({**manifest, "manifest": str(manifest_path)})


def _run_analyze_reader_addendum(args: argparse.Namespace) -> None:
    # Manuscript protocol 2.1 F.3: the SAP v2 addendum (R2-R4), frozen by the module hash before any import.
    from .clinical.reader_addendum import addendum_freeze_record, run_reader_addendum, write_addendum_freeze_record

    if args.write_freeze_record is not None:
        path = write_addendum_freeze_record(_cli_path(args.write_freeze_record))
        _emit_json({"status": "ADDENDUM_FROZEN", "freeze_record": str(path), "record": _read_json_object(path, description="Addendum freeze record")})
        return
    if args.print_freeze_record:
        _emit_json(addendum_freeze_record())
        return
    missing = [flag for flag, value in (("--predictions", args.predictions), ("--gate-evidence", args.gate_evidence),
                                        ("--addendum-freeze-record", args.addendum_freeze_record)) if not value]
    if missing:
        raise ValueError(f"analyze-reader-addendum requires {missing} (or --write-freeze-record / --print-freeze-record)")
    result = run_reader_addendum(
        analysis_root=_cli_path(args.analysis_root),
        clinical_root=_cli_path(args.clinical_root),
        predictions=[_cli_path(path) for path in args.predictions],
        gate_evidence=_cli_path(args.gate_evidence),
        freeze_record=_cli_path(args.addendum_freeze_record),
        output_root=_cli_path(args.output_root),
        config_path=_cli_path(args.config),
        mms_crosswalk=None if args.mms_crosswalk is None else _cli_path(args.mms_crosswalk),
        project_root=PROJECT_ROOT,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
        confidence=args.confidence,
    )
    _emit_json(result)


def _run_package_clinical_release(args: argparse.Namespace) -> None:
    from .clinical.release import create_clinical_release, verify_packaged_release

    source_root = _cli_path(args.source_root)
    gate_evidence = _read_json_object(args.gate_evidence, description="Project-specific gate evidence")
    if not gate_evidence:
        raise ValueError("Project-specific gate evidence must contain at least one fail-closed gate")
    required_final_gates = {
        "governance_ethics_consent",
        "gate_a_data_integrity",
        "gate_b_semantic_integrity",
        "gate_c_transition_fidelity",
        "gate_d_rule_engineering",
        "gate_e_physician_rule_safety",
        "gate_f_case_safety",
        "gate_g_reproducibility",
    }
    if not args.allow_nonfinal_audit_package:
        missing_gates = sorted(required_final_gates.difference(gate_evidence))
        if missing_gates:
            raise ValueError(
                "Final release gate evidence is incomplete; missing gate IDs: "
                f"{missing_gates}. Use --allow-nonfinal-audit-package only for a "
                "clearly non-final package."
            )
    empty_gates = sorted(
        gate_id
        for gate_id, checks in gate_evidence.items()
        if not isinstance(checks, list) or not checks
    )
    if empty_gates:
        raise ValueError(f"Gate evidence must provide a non-empty check list for: {empty_gates}")
    def source_relative(value: str | Path) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (source_root / path).resolve()

    sources = [source_relative(path) for path in args.source]
    restricted_values = args.restricted_path or (
        [] if args.restricted_release else ["restricted", "responses"]
    )
    restricted = [source_relative(path) for path in restricted_values]
    result = create_clinical_release(
        source_paths=sources,
        source_root=source_root,
        release_root=_cli_path(args.release_root),
        version=args.version,
        safety_findings_path=_cli_path(args.safety_findings),
        verification_manifests=[_cli_path(path) for path in args.verification_manifest],
        phase_lock_paths=[_cli_path(path) for path in args.phase_lock],
        additional_gate_checks=gate_evidence,
        restricted_paths=restricted,
        public_release=not args.restricted_release,
        require_all_gates=not args.allow_nonfinal_audit_package,
        study_id=args.study_id,
        release_name=args.release_name,
    )
    verification = verify_packaged_release(result.release_dir)
    if verification.get("status") != "PASS":
        raise RuntimeError(f"Packaged clinical release failed verification: {verification.get('issues')}")
    _emit_json({"status": result.status, "result": result, "verification": verification})


def v6_runtime_roots() -> list[Path]:
    """Protocol-6.0.0 runtime roots (git-ignored; Docker volumes tmmri_data and tmmri_artifacts in the container)."""

    from .paths import ARTIFACT_ROOT, DATA_ROOT

    return [DATA_ROOT / "staged", DATA_ROOT / "manifests", ARTIFACT_ROOT, ARTIFACT_ROOT / "bench"]


def init_layout() -> None:
    ensure_dirs(LAYOUT_DIRS)
    ensure_dirs(v6_runtime_roots())


def write_schema(df: pd.DataFrame, path: str | Path) -> Path:
    schema = pd.DataFrame(
        [
            {
                "column_name": col,
                "dtype": str(df[col].dtype),
                "unit": None,
                "nullable": bool(df[col].isna().any()),
                "valid_range": None,
                "derivation_formula": None,
                "transition_target_eligible": False,
            }
            for col in df.columns
        ]
    )
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    schema.to_csv(out, index=False)
    return out


def _artifact(paths: list[Path], new_paths: list[Path] | Path | None) -> None:
    if new_paths is None:
        return
    if isinstance(new_paths, Path):
        paths.append(new_paths)
    else:
        paths.extend(new_paths)


def _write_validation_artifacts(df: pd.DataFrame, stem: str, artifacts: list[Path]) -> pd.DataFrame:
    validation = validate_b_matrix(df)
    case_report = qc_case_report(df)
    _artifact(artifacts, write_table_pair(case_report, f"data/features/{stem}_qc_report.parquet", f"data/features/{stem}_qc_report.csv"))
    _artifact(artifacts, write_table_pair(validation, f"artifacts/reports/qc/{stem}_validation_gates.parquet", f"artifacts/reports/qc/{stem}_validation_gates.csv"))
    _artifact(artifacts, write_qc_markdown(df, validation, resolve_path(f"artifacts/reports/qc/{stem}_qc_report.md")))
    _artifact(artifacts, write_schema(case_report, f"data/features/{stem}_qc_report_schema.csv"))
    _artifact(artifacts, write_schema(validation, f"artifacts/reports/qc/{stem}_validation_gates_schema.csv"))
    return validation


def run_minimal(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    init_layout()
    seed = int(cfg.get("run", {}).get("algorithm_seed", 0))
    deterministic_seed(seed)
    artifacts: list[Path] = []
    commands = [f"python -m tmmri.cli run-minimal --config {config_path}"]

    env_path = write_yaml(runtime_environment(), "artifacts/provenance/run_environment.yaml")
    _artifact(artifacts, env_path)
    write_checkpoint("00_environment_done.json", "tmmri.cli run-minimal", [config_path], [env_path], 0)

    features_dir = cfg_path(cfg, "output.features_dir")
    splits_dir = cfg_path(cfg, "output.splits_dir")
    derived_dir = cfg_path(cfg, "output.derived_mms_ed_es_dir")
    acdc_root = cfg_path(cfg, "acdc.raw_root")
    mms_root = cfg_path(cfg, "mms.raw_root")
    mms_csv = cfg_path(cfg, "mms.diagnosis_csv")

    acdc_inventory = build_acdc_inventory(acdc_root)
    mms_inventory = build_mms_inventory(mms_root, mms_csv)
    _artifact(artifacts, write_table_pair(acdc_inventory, features_dir / "acdc_inventory.parquet", features_dir / "acdc_inventory.csv"))
    _artifact(artifacts, write_table_pair(mms_inventory, features_dir / "mnms_inventory.parquet", features_dir / "mnms_inventory.csv"))
    _artifact(artifacts, write_schema(acdc_inventory, features_dir / "acdc_inventory_schema.csv"))
    _artifact(artifacts, write_schema(mms_inventory, features_dir / "mnms_inventory_schema.csv"))
    write_checkpoint(
        "01_inventory_done.json",
        "tmmri.cli run-minimal",
        [acdc_root, mms_root, mms_csv],
        [features_dir / "acdc_inventory.parquet", features_dir / "mnms_inventory.parquet"],
        int(len(acdc_inventory) + len(mms_inventory)),
    )

    split_seed = int(cfg.get("run", {}).get("split_seed", 20260611))
    acdc_split = acdc_patient_split(
        acdc_inventory,
        seed=split_seed,
        validation_fraction_of_training=float(cfg.get("run", {}).get("acdc_validation_fraction_of_training", 0.15)),
    )
    selected_count = int(cfg.get("mms", {}).get("selected_case_count", 100))
    mms_manifest = select_mms_100(mms_inventory, selected_count=selected_count, seed=int(cfg.get("mms", {}).get("selection_seed", 20260611)))
    _artifact(artifacts, write_table_pair(acdc_split, splits_dir / "acdc_patient_split_v2.parquet", splits_dir / "acdc_patient_split_v2.csv"))
    _artifact(artifacts, write_schema(acdc_split, splits_dir / "acdc_patient_split_v2_schema.csv"))
    _artifact(artifacts, write_table_pair(mms_manifest, features_dir / "B_MMS_100_case_manifest.parquet", features_dir / "B_MMS_100_case_manifest.csv"))
    _artifact(artifacts, write_table_pair(mms_manifest, splits_dir / "mms_100_manifest_v2.parquet", splits_dir / "mms_100_manifest_v2.csv"))
    _artifact(artifacts, write_schema(mms_manifest, features_dir / "B_MMS_100_case_manifest_schema.csv"))
    split_manifest_path = write_yaml(
        {
            "generated_at_utc": utc_now_iso(),
            "generation_command": commands[0],
            "code_commit": git_commit(),
            "random_seed": split_seed,
            "source_inventory_artifacts": [
                "data/features/acdc_inventory.parquet",
                "data/features/mnms_inventory.parquet",
            ],
            "split_version": "v2",
        },
        splits_dir / "split_manifest_v2.yaml",
    )
    _artifact(artifacts, split_manifest_path)

    if len(mms_manifest) != selected_count or mms_manifest["selected_rank"].isna().any():
        report = write_yaml(
            {
                "status": "blocked",
                "reason": "fewer_than_100_eligible_labeled_mms_cases",
                "eligible_cases": int(mms_inventory["eligible_for_mms_100"].sum()),
            },
            "artifacts/reports/qc/B_MMS_100_not_constructible.yaml",
        )
        _artifact(artifacts, report)
        write_checkpoint("02_selection_done.json", "tmmri.cli run-minimal", [features_dir / "mnms_inventory.parquet"], [report], len(mms_manifest), ["fewer_than_100_eligible_labeled_mms_cases"])
        write_provenance_manifest(artifacts, commands)
        return

    write_checkpoint(
        "02_selection_done.json",
        "tmmri.cli run-minimal",
        [features_dir / "mnms_inventory.parquet"],
        [features_dir / "B_MMS_100_case_manifest.parquet", splits_dir / "mms_100_manifest_v2.parquet"],
        len(mms_manifest),
    )

    derived_outputs = extract_mms_ed_es_derivatives(mms_manifest, derived_dir)
    write_checkpoint("03_ed_es_extraction_done.json", "tmmri.cli run-minimal", [features_dir / "B_MMS_100_case_manifest.parquet"], derived_outputs, len(mms_manifest))
    # ROI cropping is an optional branch in the deterministic native baseline;
    # an intentional skip is not a processing failure.
    write_checkpoint("04_cropping_done.json", "tmmri.cli run-minimal", [], [], 0, [])

    b_acdc = build_acdc_b_matrix(acdc_inventory, acdc_split)
    b_mms = build_mms_b_matrix(mms_manifest)
    b_acdc = add_robust_outlier_flags(b_acdc, default_transition_feature_columns(b_acdc))
    b_mms = add_robust_outlier_flags(b_mms, default_transition_feature_columns(b_mms))
    _artifact(artifacts, write_table_pair(b_acdc, features_dir / "B_ACDC.parquet", features_dir / "B_ACDC.csv"))
    _artifact(artifacts, write_table_pair(b_mms, features_dir / "B_MMS_100.parquet", features_dir / "B_MMS_100.csv"))
    _artifact(artifacts, write_table_pair(feature_dictionary(b_acdc), features_dir / "B_ACDC_feature_dictionary.parquet", features_dir / "B_ACDC_feature_dictionary.csv"))
    _artifact(artifacts, write_table_pair(feature_dictionary(b_mms), features_dir / "B_MMS_100_feature_dictionary.parquet", features_dir / "B_MMS_100_feature_dictionary.csv"))
    write_checkpoint("05_B_matrix_done.json", "tmmri.cli run-minimal", [features_dir / "acdc_inventory.parquet", features_dir / "B_MMS_100_case_manifest.parquet"], [features_dir / "B_ACDC.parquet", features_dir / "B_MMS_100.parquet"], len(b_acdc) + len(b_mms))

    acdc_validation = _write_validation_artifacts(b_acdc, "B_ACDC", artifacts)
    mms_validation = _write_validation_artifacts(b_mms, "B_MMS_100", artifacts)
    failures = []
    if not acdc_validation["passed"].all():
        failures.append("B_ACDC_validation_gate_failure")
    if not mms_validation["passed"].all():
        failures.append("B_MMS_100_validation_gate_failure")
    write_checkpoint("06_qc_done.json", "tmmri.cli run-minimal", [features_dir / "B_ACDC.parquet", features_dir / "B_MMS_100.parquet"], [f"artifacts/reports/qc/B_ACDC_qc_report.md", f"artifacts/reports/qc/B_MMS_100_qc_report.md"], len(b_acdc) + len(b_mms), failures)

    acdc_prov = dataset_provenance(
        "ACDC",
        cfg["acdc"]["release_name"],
        cfg["acdc"]["release_url_or_access_note"],
        acdc_root,
        list(acdc_inventory["ed_image_path"].dropna()) + list(acdc_inventory["ed_mask_path"].dropna()) + list(acdc_inventory["es_image_path"].dropna()) + list(acdc_inventory["es_mask_path"].dropna()),
        len(acdc_inventory),
        int(acdc_inventory["eligible_for_b"].sum()),
        len(b_acdc),
        "metadata",
        "configs/legacy/labels/label_mapping_ACDC.yaml",
        "tmmri.cli run-minimal",
    )
    mms_prov = dataset_provenance(
        "MMS_100",
        cfg["mms"]["release_name"],
        cfg["mms"]["release_url_or_access_note"],
        mms_root,
        list(mms_manifest["image_path"].dropna()) + list(mms_manifest["mask_path"].dropna()),
        len(mms_inventory),
        int(mms_inventory["eligible_for_mms_100"].sum()),
        len(mms_manifest),
        "metadata",
        "configs/legacy/labels/label_mapping_MMS_100.yaml",
        "tmmri.cli run-minimal",
    )
    _artifact(artifacts, write_yaml(acdc_prov, "artifacts/provenance/dataset_provenance_ACDC.yaml"))
    _artifact(artifacts, write_yaml(mms_prov, "artifacts/provenance/dataset_provenance_MMS_100.yaml"))
    _artifact(artifacts, write_provenance_manifest(artifacts, commands))


# ----------------------------------------------------------------------------- protocol 6.0.0 subcommands

V6_COMMANDS = (
    "stage-sources", "validate-cohort", "build-semantics", "extract-formal", "fit-transition", "fit-predictor",
    "discretize", "check-semantic-eligibility", "induce-rules", "gate-rules", "freeze-analysis-bundle", "evaluate",
    "audit-shift", "fit-refit-arm", "summarize-predictor", "run-ablations", "run-equivariance", "build-manuscript-tables",
    "write-release-manifest", "run", "bench",
    "build-manuscript-figures",   # protocol v6.0.1 A3 (lane L2)
)
V6_COMMANDS += ("extract-mms2",)   # protocol-v6.0.1 (L1): selective M&Ms-2 extraction (host only)
V6_COMMANDS += ("evaluate-h4",)    # protocol-v6.0.1 (Decision D36): the H4 family on a recorded evaluation partition
# --- protocol-v6.0.1 lane L3: registered comparator F8 and the exploratory register B1-B8 (dispatched by _run_v6) ---
V601_L3_COMMANDS = ("fit-cart-surrogate", "evaluate-cart-surrogate", "run-exploratory")
# --- end lane L3 ---
# Lane L57 (protocol-v6.0.1 items 11-14: F9b probes, alternative bundles, trust S1-S4, comparison tables); run by _run_v6.
from .experiments.l57_cli import L57_COMMANDS, add_l57_parsers, run_l57_command  # noqa: E402
# L89 (protocol-v6.0.1 S5/S6, configs/data_centric.yaml): registered data-centric analyses, dispatched by _run_v6.
DATA_CENTRIC_COMMANDS = ("run-data-efficiency", "run-simulated-shift")
# Exit codes of the v6 subcommands (documented in SPECIFICATION.md Section 15.1). 1 = usage or missing input.
EXIT_USAGE_OR_INPUT = 1
EXIT_STAGING_FAILURES = 2
EXIT_PROTOCOL_NOT_FROZEN = 3
EXIT_METHODS_API_MISSING = 4
EXIT_CONTRACT_VIOLATION = 5
EXIT_COHORT_VALIDATION = 6
EXIT_BUNDLE = 7
EXIT_HUMAN_GATE = 8                # protocol-v6.0.1: a release fact differs from the declared M&Ms-2 assumption (fail closed)


def _v6_common(parser: argparse.ArgumentParser, *, run_id: str = "required") -> None:
    parser.add_argument("--study", default="configs/study_v6.yaml", help="protocol study configuration")
    if run_id == "required":
        parser.add_argument("--run-id", required=True, help="run id, pattern v6-<purpose>-YYYYMMDD-HHMM")
    elif run_id == "optional":
        parser.add_argument("--run-id", default=None, help="run id (optional for this stage)")
    parser.add_argument("--n-jobs", type=int, default=None, help="CPU workers (default TMMRI_N_JOBS, then 12)")
    parser.add_argument("--source-readers", type=int, default=None, help="concurrent source readers (default TMMRI_SOURCE_READERS, then 4)")
    parser.add_argument("--data-root", default=None, help="runtime data root (default TMMRI_DATA_ROOT or data/)")
    parser.add_argument("--artifact-root", default=None, help="run-root parent (default TMMRI_ARTIFACT_ROOT or artifacts/)")
    parser.add_argument("--acdc-source", default=None, help="ACDC source root (default /sources/acdc, TMMRI_ACDC_SOURCE, binding host default)")
    parser.add_argument("--mms-source", default=None, help="M&Ms release root (default /sources/mms, TMMRI_MMS_SOURCE, binding host default)")
    parser.add_argument("--mms2-source", default=None, help="M&Ms-2 source area (default TMMRI_MMS2_SOURCE, then the binding host default ./data/raw/mms2)")
    parser.add_argument("--force", action="store_true", help="rerun even when the stage record matches")
    parser.add_argument("--allow-unfrozen-synthetic", action="store_true",
                        help="permit M&Ms endpoint stages before the freeze only for a synthetic fixture (SYNTHETIC_FIXTURE)")


def _v6_context(args: argparse.Namespace):
    from .stages import make_context

    sources = {k: _cli_path(v) for k, v in (("acdc", args.acdc_source), ("mms", args.mms_source),
                                            ("mms2", getattr(args, "mms2_source", None))) if v}
    ctx = make_context(args.study, getattr(args, "run_id", None),
                       artifact_root=_cli_path(args.artifact_root) if args.artifact_root else None,
                       data_root=_cli_path(args.data_root) if args.data_root else None,
                       n_jobs=args.n_jobs, source_readers=args.source_readers, force=args.force,
                       allow_unfrozen_synthetic=args.allow_unfrozen_synthetic, sources=sources)
    if getattr(args, "out", None):
        ctx.staged_root_override = _cli_path(args.out)
    return ctx


def _stage_summary(result: Any) -> Any:
    if hasattr(result, "record") and hasattr(result, "action"):
        return {"stage": result.stage, "action": result.action, "outputs": result.record.outputs,
                "status": result.record.status, "value": result.value}
    return result


def _run_v6(args: argparse.Namespace) -> int:
    from . import stages
    from .ingest.mms2 import HumanGateError
    from .ingest.sources import SourceError
    from .semantics.contract import ContractViolation

    command = args.command
    try:
        if command == "bench":
            return _run_bench(args)
        ctx = _v6_context(args)
        # --- protocol-v6.0.1 (M&Ms-2; L1) ---
        if command == "extract-mms2":
            from .mms2_stages import extract_mms2

            result = extract_mms2(ctx, archive=_cli_path(args.archive) if args.archive else None,
                                  dest=_cli_path(args.dest) if args.dest else None, tar=args.tar)
            _emit_json({k: result.get(k) for k in ("action", "archive", "list_file", "byte_check", "file_manifest", "lock", "synthetic_fixture")})
            return 0
        if command == "validate-cohort" and getattr(args, "dataset", None) == "mms2":
            from .mms2_stages import validate_cohort_mms2

            report = validate_cohort_mms2(ctx)          # always strict: a failed M&Ms-2 validation exits 6 and records nothing
            _emit_json({"passed": report["passed"], "failed_checks": [c for c in report["checks"] if not c["passed"]],
                        "outputs": report["outputs"], "written_to": str(stages._contracts_dir(ctx)),
                        "duplicate_exclusions": report.get("duplicate_exclusions"), "cluster_counts": report.get("cluster_counts")})
            return 0
        if command == "summarize-predictor" and getattr(args, "split", None) == stages.MMS2_SPLIT:
            from .mms2_stages import summarize_predictor_mms2

            _emit_json(_stage_summary(summarize_predictor_mms2(ctx)))
            return 0
        if command == "evaluate-h4":                # Decision D36: the H4 family (tmmri.confirmatory_h4)
            from .confirmatory_h4 import evaluate_h4

            _emit_json(_stage_summary(evaluate_h4(ctx, args.split)))
            return 0
        # --- end protocol-v6.0.1 ---
        if command == "stage-sources":
            if args.source:
                ctx.sources[args.dataset] = _cli_path(args.source)
            report = stages.stage_sources(ctx, args.dataset, verify_only=args.verify_only, cases=args.case or None)
            _emit_json(report.as_dict())
            return 0 if report.ok else EXIT_STAGING_FAILURES
        if command == "validate-cohort":
            report = stages.validate_cohort(ctx, strict=args.strict)
            _emit_json({"passed": report["passed"], "failed_checks": [c for c in report["checks"] if not c["passed"]],
                        "outputs": report["outputs"], "written_to": str(stages._contracts_dir(ctx)),
                        "duplicate_image_pairs": report.get("duplicate_image_pairs"), "cluster_counts": report.get("cluster_counts")})
            return 0
        if command == "build-semantics":
            result = stages.build_semantics(ctx, args.dataset)
        elif command == "extract-formal":
            result = stages.extract_formal(ctx, args.dataset)
        elif command == "fit-transition":
            result = stages.fit_transition(ctx, profile=args.profile, nested_oof=args.nested_oof)
        elif command == "fit-predictor":
            result = stages.fit_predictor(ctx)
        elif command == "discretize":
            result = stages.discretize(ctx, profile=args.profile)
        elif command == "check-semantic-eligibility":
            result = stages.check_semantic_eligibility(ctx)
        elif command == "induce-rules":
            result = stages.induce_rules(ctx, tier=args.tier)
        elif command == "gate-rules":
            result = stages.gate_rules(ctx, substrate=args.substrate)
        elif command == "freeze-analysis-bundle":
            frozen = stages.freeze_analysis_bundle(ctx)
            result = {"bundle_sha256": frozen["bundle_sha256"], "files": len(frozen["bundle"]),
                      "frozen_record": str(ctx.run.frozen_path)}
        elif command == "evaluate":
            result = stages.evaluate(ctx, args.split, no_fit=args.no_fit)
        elif command == "audit-shift":
            result = stages.audit_shift(ctx, getattr(args, "dataset", "mms"))
        elif command in L57_COMMANDS:   # lane L57 (src/tmmri/experiments/l57_cli.py)
            result = run_l57_command(ctx, args)
        elif command == "fit-refit-arm":
            result = stages.fit_refit_arm(ctx)
        elif command == "summarize-predictor":
            # protocol v6.0.1 (A2, A4, A5): the v6.0.0 summary, then one extended record per partition (tmmri.predictor_reporting)
            from .predictor_reporting import summarize_predictor_extended

            base = None if args.extended_only else stages.summarize_predictor(ctx)
            extended = summarize_predictor_extended(ctx, args.partition or None)
            result = {**(_stage_summary(base) if base is not None else {}), "extended": [_stage_summary(r) for r in extended]}
        elif command == "run-ablations":
            result = stages.run_ablations(ctx)
        elif command == "run-equivariance":
            result = stages.run_equivariance(ctx, args.dataset)
        elif command == "build-manuscript-tables":
            result = stages.build_manuscript_tables(ctx)
        elif command == "build-manuscript-figures":   # protocol v6.0.1 A3
            from .manuscript_figures import build_manuscript_figures

            result = build_manuscript_figures(ctx)
        elif command == "write-release-manifest":
            result = stages.write_release_manifest(ctx)
        elif command == "run-data-efficiency":                   # L89 (S5)
            from .experiments.data_efficiency import run_data_efficiency
            result = run_data_efficiency(ctx, args.step, device=getattr(args, "device", None))
        elif command == "run-simulated-shift":                   # L89 (S6)
            from .experiments.simulated_shift import run_simulated_shift
            result = run_simulated_shift(ctx, args.vendor, args.step, device=getattr(args, "device", None))
        elif command == "run":
            result = {"steps": stages.run_pipeline(ctx, until=args.until, start=args.from_stage)}
        # --- protocol-v6.0.1 lane L3 ---
        elif command == "fit-cart-surrogate":
            from .comparators.cart_f8 import fit_cart_surrogate
            result = fit_cart_surrogate(ctx)
        elif command == "evaluate-cart-surrogate":
            from .comparators.cart_f8 import evaluate_cart_surrogate
            result = evaluate_cart_surrogate(ctx, args.split)
        elif command == "run-exploratory":
            from .experiments.exploratory import run_exploratory
            result = run_exploratory(ctx, args.analysis)
        # --- end lane L3 ---
        else:
            raise SystemExit(f"Unknown command: {command}")
        _emit_json(_stage_summary(result))
        return 0
    except stages.ProtocolFreezeError as exc:
        print(f"REFUSED: {exc}")
        return EXIT_PROTOCOL_NOT_FROZEN
    except HumanGateError as exc:
        print(f"HUMAN_GATE: {exc}")
        return EXIT_HUMAN_GATE
    except SourceError as exc:
        print(f"SOURCE: {exc}")
        return EXIT_USAGE_OR_INPUT
    except NotImplementedError as exc:
        print(f"NOT IMPLEMENTED: {exc}")
        return EXIT_METHODS_API_MISSING
    except ContractViolation as exc:
        print(f"CONTRACT VIOLATION: {exc}")
        return EXIT_CONTRACT_VIOLATION
    except stages.CohortValidationError as exc:
        _emit_json({"passed": False, "failed_checks": [c for c in exc.report["checks"] if not c["passed"]]})
        return EXIT_COHORT_VALIDATION
    except stages.BundleError as exc:
        print(f"BUNDLE: {exc}")
        return EXIT_BUNDLE
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return EXIT_USAGE_OR_INPUT


def _run_bench(args: argparse.Namespace) -> int:
    import importlib.util

    script = PROJECT_ROOT / "scripts" / "bench" / "bench_v6.py"
    if not script.is_file():
        raise FileNotFoundError(f"benchmark script missing: {script}")
    spec = importlib.util.spec_from_file_location("tmmri_bench_v6", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    forwarded = ["--cases", str(args.cases), "--out", str(args.out), "--dataset", args.dataset]
    if args.n_jobs_list:
        forwarded += ["--n-jobs-list", args.n_jobs_list]
    for flag, value in (("--acdc-source", args.acdc_source), ("--mms-source", args.mms_source),
                        ("--source-readers", args.source_readers), ("--work-dir", args.work_dir)):
        if value:
            forwarded += [flag, str(value)]
    return int(module.main(forwarded) or 0)


def _add_v6_parsers(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("stage-sources", help="v6: verify-and-stage the ED/ES frames of one dataset (allowed for M&Ms before M0)")
    p.add_argument("--dataset", required=True, choices=["acdc", "mms", "mms2"])
    p.add_argument("--verify-only", action="store_true", help="hash-verify the sources against the tracked manifest; write no staged files")
    p.add_argument("--source", default=None, help="explicit source root (host runs)")
    p.add_argument("--out", default=None, help="staging root (default <data-root>/staged; manifests beside it in ../manifests)")
    p.add_argument("--case", action="append", default=[], help="restrict to one case id; repeat as needed")
    _v6_common(p, run_id="optional")

    p = sub.add_parser("validate-cohort", help="v6: cohort tables, expected counts, staging cross-checks, split manifest")
    p.add_argument("--strict", action="store_true", help="exit non-zero when any check fails (always on for --dataset mms2)")
    p.add_argument("--dataset", default=None, choices=["mms2"],
                   help="protocol-v6.0.1: validate the M&Ms-2 cohort alone (default: ACDC and M&Ms-1 together)")
    _v6_common(p, run_id="optional")

    for name, text in (("build-semantics", "v6: typed B (B_v6.0) from staged cases; M&Ms only after M0, M&Ms-2 only after protocol-v6.0.1"),
                       ("extract-formal", "v6: formal A (57 descriptors) + P6 comparator; M&Ms only after M0, M&Ms-2 only after protocol-v6.0.1")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--dataset", required=True, choices=["acdc", "mms", "mms2"])
        _v6_common(p)

    p = sub.add_parser("fit-transition", help="v6: nested OOF ridge transition, final model and comparators (acdc_dev)")
    p.add_argument("--profile", default="ridge", choices=["ridge"])
    p.add_argument("--nested-oof", action="store_true", help="also write the nested out-of-fold B-hat and comparators")
    _v6_common(p)

    p = sub.add_parser("fit-predictor", help="v6: audited predictor d_model on formal A (acdc_dev) + cross-fitted d_model")
    _v6_common(p)

    p = sub.add_parser("discretize", help="v6: WEDD cuts and the bootstrap stability gate per rulebook")
    p.add_argument("--profile", default="wedd_gaussian_mm_v1", choices=["wedd_gaussian_mm_v1", "wedd_uniform_mm_v1", "three_quantile_v1"])
    _v6_common(p)

    p = sub.add_parser("check-semantic-eligibility", help="v6: M7 eligibility gates of configs/rule_policy.yaml per rulebook and antecedent")
    _v6_common(p)

    p = sub.add_parser("induce-rules", help="v6: variable-precision rough-set rule induction per rulebook")
    p.add_argument("--tier", default="variable_precision", choices=["variable_precision"])
    _v6_common(p)

    p = sub.add_parser("gate-rules", help="v6: cross-fitted validation gate over the acdc_dev outer folds")
    p.add_argument("--substrate", default="nested_inner_oof", choices=["nested_inner_oof", "outer_oof"],
                   help="training-fold substrate: inner nested OOF reconstructions (default) or outer OOF (sensitivity)")
    _v6_common(p)

    p = sub.add_parser("freeze-analysis-bundle", help="v6: hash manifest of all fitted objects, configs and code (.stages/FROZEN.json)")
    _v6_common(p)

    p = sub.add_parser("evaluate", help="v6: evaluate the frozen bundle once (mms_external refuses unless frozen and tagged)")
    p.add_argument("--split", required=True, choices=["acdc_test", "mms_external", "mms2_external"])
    p.add_argument("--no-fit", action="store_true", help="required: evaluation never fits")
    _v6_common(p)

    p = sub.add_parser("audit-shift", help="v6: label-free shift audit, domain-classifier AUROC acdc_dev vs mms_external on formal A (after M0)")
    p.add_argument("--dataset", default="mms", choices=["mms", "mms2"], help="comparison cohort: mms_external (default) or mms2_external")
    _v6_common(p)

    p = sub.add_parser("fit-refit-arm", help="v6: D17 M&Ms refit ceiling (secondary; after the primary external evaluation)")
    _v6_common(p)

    p = sub.add_parser("summarize-predictor", help="v6: audited-predictor metrics of Section 13.2 from the recorded predictions (read-only; after the evaluations)")
    p.add_argument("--split", default=None, choices=["mms2_external"],
                   help="protocol-v6.0.1: summarize mms2_external only (record summarize-predictor:mms2_external)")
    # protocol v6.0.1 (A2, A4, A5): extended per-partition summaries (tmmri.predictor_reporting)
    p.add_argument("--partition", action="append", default=[], choices=["acdc_dev_oof", "acdc_test", "mms_external", "mms2_external"],
                   help="v6.0.1 extended summary of this partition only (repeatable; default every evaluated partition)")
    p.add_argument("--extended-only", action="store_true", help="skip the v6.0.0 summary; run only the v6.0.1 extended records")
    _v6_common(p)

    p = sub.add_parser("run-ablations", help="v6: Section 13.3 ablation matrix on ACDC (secondary outputs under */ablations/)")
    _v6_common(p)

    p = sub.add_parser("run-equivariance", help="v6: D19 equivariance/ETM diagnostics on seeded acdc_dev cases (optional)")
    p.add_argument("--dataset", default="acdc", choices=["acdc"])
    _v6_common(p)

    p = sub.add_parser("build-manuscript-tables", help="v6: manuscript Tables 1-10 of Section 16 (CSV, LaTeX, provenance.json) from recorded artifacts (read-only)")
    _v6_common(p)

    p = sub.add_parser("build-manuscript-figures", help="v6.0.1: manuscript Figures 2-8 (vector PDF, 300-dpi PNG, provenance) from recorded artifacts (read-only)")
    _v6_common(p)

    p = sub.add_parser("write-release-manifest", help="v6: SHA-256 manifest of the run root, configuration and registry hashes, code version, status")
    _v6_common(p)

    # L89 (protocol-v6.0.1 S5/S6; configs/data_centric.yaml)
    p = sub.add_parser("run-data-efficiency", help="v6.0.1 S5: data-efficiency curves on acdc_dev (descriptor fits; F9 fit specs and aggregation)")
    p.add_argument("--step", default="descriptor", choices=["descriptor", "f9-wave1", "f9-wave2", "f9-infer", "f9-aggregate"])
    p.add_argument("--device", default=None, help="f9-infer: torch device (default cuda when available, else cpu)")   # integration (S5 local inference)
    _v6_common(p)

    p = sub.add_parser("run-simulated-shift", help="v6.0.1 S6: simulated acquisition shift on acdc_test per M&Ms-1 vendor geometry (once-only per vendor)")
    p.add_argument("--vendor", default="all", help="M&Ms-1 vendor code (A, B, C, D) or all")
    p.add_argument("--step", default="descriptors", choices=["descriptors", "tensors", "d20", "f9"])
    p.add_argument("--device", default=None, help="d20, f9: torch device of the frozen extractor or encoder (default cuda when available, else cpu)")
    _v6_common(p)

    p = sub.add_parser("run", help="v6: run the stage DAG in order with content-addressed skips")
    p.add_argument("--until", default=None, help="last step (stage name or step id such as extract-formal:mms)")
    p.add_argument("--from", dest="from_stage", default=None, help="first step (stage name or step id)")
    _v6_common(p)
    p.set_defaults(study="configs/study_v6.yaml")

    # --- protocol-v6.0.1 (M&Ms-2; L1) ---
    p = sub.add_parser("extract-mms2", help="v6.0.1: verify MnM2.rar, then (after the protocol-v6.0.1 tag) extract only the 720 "
                       "mms2_external short-axis ED/ES files with tar -T into the read-only M&Ms-2 source area (host only)")
    p.add_argument("--archive", default=None, help="archive path (default configs/source_bindings/mms2.json archive.reference_copy)")
    p.add_argument("--dest", default=None, help="source area (default the M&Ms-2 source root: TMMRI_MMS2_SOURCE, then ./data/raw/mms2)")
    p.add_argument("--tar", default=None, help="bsdtar executable (default C:/Windows/System32/tar.exe; GNU tar cannot read RAR)")
    _v6_common(p, run_id="optional")
    p = sub.add_parser("evaluate-h4", help="v6.0.1 D36: the H4 family (H4a rho_EF, H4b H-a delta-rho) once on a recorded evaluation partition; "
                       "confirmatory on mms2_external, registered after unblinding on acdc_test and mms_external")
    p.add_argument("--split", required=True, choices=["acdc_test", "mms_external", "mms2_external"])
    _v6_common(p)
    # --- end protocol-v6.0.1 ---
    # --- protocol-v6.0.1 lane L3: F8 CART surrogate and the exploratory register (configs/cart_f8.yaml, configs/exploratory.yaml) ---
    p = sub.add_parser("fit-cart-surrogate", help="v6.0.1 F8: CART surrogate of d_model on acdc_dev OOF B-hat; freezes comparators/cart_f8/FROZEN_F8.json")
    _v6_common(p)
    p = sub.add_parser("evaluate-cart-surrogate", help="v6.0.1 F8: the frozen CART surrogate on one recorded evaluation partition (once-only)")
    p.add_argument("--split", required=True, choices=["acdc_test", "mms_external", "mms2_external"])
    _v6_common(p)
    p = sub.add_parser("run-exploratory", help="v6.0.1 exploratory register B1-B9 (EXPLORATORY_POST_HOC; recorded artifacts only; "
                                               "B9: label-free orientation audit on staged masks)")
    p.add_argument("--analysis", required=True, choices=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"])
    _v6_common(p)
    # --- end lane L3 ---

    p = sub.add_parser("bench", help="v6: timing benchmark (per-case seconds, throughput, projections for ACDC 150 and M&Ms 320/345)")
    p.add_argument("--cases", type=int, default=10)
    p.add_argument("--out", required=True, help="output JSON, e.g. artifacts/bench/<id>.json")
    p.add_argument("--dataset", default="all", choices=["all", "acdc", "mms", "synthetic"])
    p.add_argument("--n-jobs-list", default=None, help="comma-separated worker counts (default 1,TMMRI_N_JOBS)")
    p.add_argument("--work-dir", default=None, help="scratch staging root for the benchmark (default a temporary folder)")
    p.add_argument("--acdc-source", default=None)
    p.add_argument("--mms-source", default=None)
    p.add_argument("--source-readers", type=int, default=None)

    add_l57_parsers(sub, _v6_common)   # lane L57 (protocol-v6.0.1 items 11-14)


# --- L4a: protocol-v6.0.1 image data path (tensor cache, D20, local F9 inference; src/tmmri/neural/image_stages.py) ---
NEURAL_IMAGE_COMMANDS = ("build-tensor-cache", "verify-torch-weights", "extract-deep-features", "fit-deep-encoder", "infer-f9")
NEURAL_IMAGE_SPLITS = ("acdc_dev", "acdc_test", "mms_external", "mms2_external")


def _add_neural_image_parsers(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("build-tensor-cache", help="v6.0.1: neural tensor cache (contract C3) of one partition from the staged images")
    p.add_argument("--split", required=True, choices=NEURAL_IMAGE_SPLITS)
    _v6_common(p)
    p = sub.add_parser("verify-torch-weights", help="v6.0.1: download (download.pytorch.org only) and pin the ResNet-18/50 ImageNet weights")
    p.add_argument("--no-download", action="store_true", help="verify only; fail when a weight file is absent")
    _v6_common(p, run_id="optional")
    p = sub.add_parser("extract-deep-features", help="v6.0.1: D20 frozen ResNet-50 2048-d patient vectors of one partition")
    p.add_argument("--split", required=True, choices=NEURAL_IMAGE_SPLITS)
    p.add_argument("--device", default=None, help="torch device (default cuda when available, else cpu)")
    _v6_common(p)
    p = sub.add_parser("fit-deep-encoder", help="v6.0.1: D20 standardization, ridge T and audited predictor on acdc_dev (OOF + final), frozen")
    _v6_common(p)
    p = sub.add_parser("infer-f9", help="v6.0.1: local F9 predictions and 64-d activations of one partition from the recorded checkpoints")
    p.add_argument("--split", required=True, choices=NEURAL_IMAGE_SPLITS)
    p.add_argument("--device", default=None, help="torch device (default cuda when available, else cpu)")
    p.add_argument("--seed", type=int, action="append", default=[], help="restrict to one F9 seed; repeat as needed")
    _v6_common(p)


def _run_neural_image(args: argparse.Namespace) -> int:
    from . import stages
    from .neural import image_stages
    from .neural.inference import CheckpointError
    from .neural.weights import WeightsError

    try:
        result = image_stages.dispatch(_v6_context(args), args)
        _emit_json(_stage_summary(result))
        return 0
    except stages.ProtocolFreezeError as exc:
        print(f"REFUSED: {exc}")
        return EXIT_PROTOCOL_NOT_FROZEN
    except stages.BundleError as exc:
        print(f"BUNDLE: {exc}")
        return EXIT_BUNDLE
    except (WeightsError, CheckpointError) as exc:
        print(f"CONTRACT VIOLATION: {exc}")
        return EXIT_CONTRACT_VIOLATION
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return EXIT_USAGE_OR_INPUT
# --- end L4a ---


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cardiac MRI semantic matrix, transition-matrix, and DSS research pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init-layout")
    _add_v6_parsers(sub)
    # --- wf(L4b): F9 trained-CNN comparator commands, protocol-v6.0.1 (tmmri.neural.f9_cli) ---
    from .neural.f9_cli import F9_COMMANDS, add_f9_parsers, run_f9
    add_f9_parsers(sub, _v6_common)
    # --- end wf(L4b) ---
    _add_neural_image_parsers(sub)   # L4a
    # --- protocol-v6.0.1: read-only preflight of each execution step (tmmri.preflight; docs/V601_EXECUTION_RUNBOOK.md) ---
    from .preflight import add_preflight_parser
    add_preflight_parser(sub)
    # --- end preflight ---

    run = sub.add_parser("run-minimal")
    run.add_argument("--config", default="configs/legacy/pipeline/default.yaml")

    native = sub.add_parser("validate-native-data")
    native.add_argument("--config", default="configs/legacy/pipeline/default.yaml")
    native.add_argument("--dataset", default="all", choices=["all", "acdc", "mms", "mms_100"])

    full = sub.add_parser("run-full-validation")
    full.add_argument("--config", default="configs/legacy/pipeline/default.yaml")
    full.add_argument("--experiment-id", default=None)
    full.add_argument("--run-minimal-if-needed", action="store_true")
    full.add_argument("--enable-classification", action="store_true")
    full.add_argument("--enable-rulebook", action="store_true")
    full.add_argument("--enable-dss-predicates", action="store_true")
    full.add_argument("--enable-equivariance-diagnostics", action="store_true")
    full.add_argument("--allow-semantic-proxy-a", action="store_true", help="Explicitly allow a research-only semantic-proxy A fallback when native NIfTI images/masks are absent.")
    full.add_argument("--native-data-required", action="store_true", help="Require native NIfTI image/mask validation and prohibit semantic-proxy Matrix A fallback.")
    full.add_argument("--bootstrap-iterations", type=int, default=None)
    full.add_argument("--max-qa-panels", type=int, default=None)
    full.add_argument("--for-rules-archive", default=None, help="Optional path to for_rules.zip; if supplied, it is validated and imported as the authoritative B-matrix substrate before DSS stages.")
    full.add_argument("--non-strict-for-rules-import", action="store_true", help="Allow nonstandard row counts in for_rules archive validation while still enforcing schemas and unique sample IDs.")

    prepare = sub.add_parser(
        "prepare-clinical-validation",
        help="Freeze the single-physician cohort, rule-audit, fidelity, and provenance inputs without inventing physician or ethics evidence.",
    )
    prepare.add_argument("--config", default=CLINICAL_CONFIG_DEFAULT, help="Clinical-validation YAML configuration.")
    prepare.add_argument("--experiment-id", default=None, help="Optional versioned experiment identifier; defaults to the frozen study ID in the configuration.")

    figures = sub.add_parser(
        "generate-clinical-figures",
        help="Generate only the currently supportable Python/Nature figures and label-safe reader image assets.",
    )
    figures.add_argument("--primary-manifest", default="clinical_validation/restricted/primary_reader_manifest_restricted.parquet", help="Restricted frozen 150-case manifest.")
    figures.add_argument("--transition-metrics", default="legacy/v5/artifacts/reports/transition/transition_metrics_v5_native_image_dss_100.parquet", help="Frozen transition-metric table.")
    figures.add_argument("--fidelity-gate", default="clinical_validation/rules/transition_fidelity_gate.parquet", help="Validation-only transition-fidelity gate table.")
    figures.add_argument("--transition-residuals-acdc", default="legacy/v5/artifacts/models/transition/transition_residuals_ACDC_v5_native_image_dss_100.parquet", help="Frozen ACDC long-form measured/reconstructed residual table.")
    figures.add_argument("--transition-residuals-mms", default="legacy/v5/artifacts/models/transition/transition_residuals_MMS_100_v5_native_image_dss_100.parquet", help="Frozen M&Ms long-form measured/reconstructed residual table.")
    figures.add_argument("--output-root", default=CLINICAL_FIGURE_ROOT_DEFAULT, help="Compatibility destination for label-safe reader image assets.")
    figures.add_argument("--manuscript-output-root", default=MANUSCRIPT_FIGURE_ROOT_DEFAULT, help="Destination for active manuscript figures.")
    figures.add_argument("--manuscript-report-root", default=MANUSCRIPT_REPORT_ROOT_DEFAULT, help="Destination for manuscript captions and the current generation manifest.")

    build_workbooks = sub.add_parser(
        "build-clinical-workbooks",
        help="Author the five separate macro-free workbooks through the approved @oai/artifact-tool Node builder.",
    )
    build_workbooks.add_argument("--clinical-root", default=CLINICAL_ROOT_DEFAULT, help="Prepared clinical-validation artifact root.")
    build_workbooks.add_argument("--payload", default="clinical_validation/workbooks/08_build/01_artifact_tool_build_payload_v2.json", help="Destination for the validated artifact-tool build contract.")
    build_workbooks.add_argument("--output-root", default="clinical_validation/workbooks", help="Primary workbook output root.")
    build_workbooks.add_argument("--portable-output-root", default=CLINICAL_OUTPUT_ROOT_DEFAULT, help="Portable release-copy output root.")
    build_workbooks.add_argument("--preview-root", default="clinical_validation/workbooks/06_previews", help="Rendered-sheet QA output root.")
    build_workbooks.add_argument("--builder-script", default="scripts/clinical_validation/build_clinical_workbooks.mjs", help="Approved artifact-tool Node builder.")
    build_workbooks.add_argument("--node", default="node", help="Node.js executable containing the approved artifact-tool runtime.")

    validate_workbooks = sub.add_parser(
        "validate-clinical-workbooks",
        help="Fail-closed OOXML validation for leakage, formulas, links, sheets, visibility, and protection controls.",
    )
    validate_workbooks.add_argument("--clinical-root", default=CLINICAL_ROOT_DEFAULT, help="Prepared clinical-validation artifact root.")
    validate_workbooks.add_argument("--kind", default="all", choices=["all", "unassisted", "assisted", "rule_audit", "analysis", "retest"], help="Workbook kind to inspect.")
    validate_workbooks.add_argument("--workbook", default=None, help="Explicit .xlsx path; valid only with one non-'all' kind.")
    validate_workbooks.add_argument("--report", default="clinical_validation/workbooks/07_excel_qa/04_workbook_validation_report_v2.json", help="JSON validation report destination.")

    import_responses = sub.add_parser(
        "import-reader-responses",
        help="Validate, preserve byte-for-byte, and phase-lock one genuine physician response workbook or CSV.",
    )
    import_responses.add_argument("--input", required=True, help="Completed macro-free .xlsx workbook or CSV response export.")
    import_responses.add_argument("--phase", required=True, choices=["unassisted", "assisted", "rule_audit", "retest"], help="Frozen response schema and study phase.")
    import_responses.add_argument("--clinical-root", default=CLINICAL_ROOT_DEFAULT, help="Prepared clinical-validation artifact root.")
    import_responses.add_argument("--sheet-name", default=None, help="Optional explicit worksheet name for an .xlsx response.")
    import_responses.add_argument("--additional-artifact", action="append", default=[], help="Additional contemporaneous artifact to include in the immutable phase lock; repeat as needed.")
    import_responses.add_argument("--allow-incomplete", action="store_true", help="Permit a schema-valid incomplete import for controlled recovery only; primary analysis still requires completeness.")

    analyze = sub.add_parser(
        "analyze-reader-study",
        help="Compute pre-specified single-physician endpoints from complete, phase-locked genuine responses.",
    )
    analyze.add_argument("--unassisted", required=True, help="Frozen unassisted response .xlsx or CSV.")
    analyze.add_argument("--assisted", required=True, help="Frozen assisted response .xlsx or CSV.")
    analyze.add_argument("--rule-audit", required=True, help="Frozen global rule-audit response .xlsx or CSV.")
    analyze.add_argument("--retest", default=None, help="Optional frozen 24-case retest response .xlsx or CSV.")
    analyze.add_argument("--config", default="configs/clinical/clinical_validation_single_reader.yaml", help="Frozen clinical protocol configuration.")
    analyze.add_argument("--clinical-root", default=CLINICAL_ROOT_DEFAULT, help="Prepared clinical-validation artifact root.")
    analyze.add_argument("--output-root", default="clinical_validation/analysis/tables", help="Analysis tables and safety-register destination.")
    analyze.add_argument("--bootstrap-iterations", type=int, default=None, help="Optional override of the frozen patient-level bootstrap iterations.")
    analyze.add_argument("--bootstrap-seed", type=int, default=None, help="Optional override of the frozen deterministic bootstrap seed.")
    analyze.add_argument("--confidence", type=float, default=None, help="Optional override of the frozen two-sided confidence level.")
    analyze.add_argument("--off-ontology-not-systematically-forced", choices=["yes", "no", "unknown"], default="unknown", help="Approved interpretation of off-ontology routing; unknown keeps the safety gate non-evaluable.")
    analyze.add_argument("--unassisted-blinding-breach", choices=["yes", "no", "unknown"], default="unknown", help="Documented blinding determination; unknown keeps the safety gate non-evaluable.")
    analyze.add_argument("--serious-protocol-deviations-analyzed", choices=["yes", "no", "unknown"], default="unknown", help="Documented deviation-analysis determination; unknown keeps the safety gate non-evaluable.")

    addendum = sub.add_parser(
        "analyze-reader-addendum",
        help="Manuscript addendum to SAP v2 (R2 PABAK/AC1, R3 physician-model concordance, R4 assisted vs unassisted; R4 excluded from paper 1) from governed outputs only.",
    )
    addendum.add_argument("--analysis-root", default="clinical_validation/analysis/tables", help="Output root of analyze-reader-study (analysis_manifest.json, restricted_case_analysis_dataset.csv).")
    addendum.add_argument("--clinical-root", default=CLINICAL_ROOT_DEFAULT, help="Prepared clinical-validation root holding responses/locks from import-reader-responses.")
    addendum.add_argument("--predictions", action="append", default=[], help="v6.0.1 C4 predictions.parquet (audited_predictor, d20_predictor, f9); repeat as needed.")
    addendum.add_argument("--gate-evidence", default=None, help="JSON of the external gates (gate_1_governance, gate_2_vocabulary_and_feature_approval, gate_4_measurement_conventions) in the package-clinical-release format.")
    addendum.add_argument("--addendum-freeze-record", default=None, help="Freeze record written by --write-freeze-record before any import (module SHA-256 and time).")
    addendum.add_argument("--mms-crosswalk", default=None, help="Signed M&Ms label crosswalk CSV (mms_label, acdc_comparison_class); it must be passed evidence of gate_2. Without it the M&Ms cases are not evaluated.")
    addendum.add_argument("--config", default=CLINICAL_CONFIG_DEFAULT, help="Frozen clinical protocol configuration (SAP seed, iterations, ontology).")
    addendum.add_argument("--output-root", default="clinical_validation/analysis/addendum", help="Addendum tables and provenance destination (never overwritten).")
    addendum.add_argument("--bootstrap-iterations", type=int, default=None, help="Optional override of the frozen bootstrap iterations.")
    addendum.add_argument("--bootstrap-seed", type=int, default=None, help="Optional override of the frozen SAP bootstrap seed.")
    addendum.add_argument("--confidence", type=float, default=None, help="Optional override of the frozen two-sided confidence level.")
    addendum.add_argument("--write-freeze-record", default=None, help="Write the addendum freeze record (module SHA-256) to this path and exit; reads no input.")
    addendum.add_argument("--print-freeze-record", action="store_true", help="Print the freeze record (module SHA-256) without writing anything and exit.")

    package = sub.add_parser(
        "package-clinical-release",
        help="Create and verify an immutable public clinical-validation package after evidence-based release gates pass.",
    )
    package.add_argument("--source-root", default=CLINICAL_ROOT_DEFAULT, help="Root against which all packaged source paths are resolved.")
    package.add_argument("--source", action="append", required=True, help="Artifact or directory within source-root to package; repeat as needed.")
    package.add_argument("--release-root", default="clinical_validation/releases", help="Immutable release-version parent directory.")
    package.add_argument("--version", required=True, help="New semantic release version, for example 1.0.0.")
    package.add_argument("--safety-findings", required=True, help="Genuine CSV/JSON safety-findings evidence; omission never implies zero findings.")
    package.add_argument("--gate-evidence", required=True, help="JSON mapping of every project-specific gate to evidence checks.")
    package.add_argument("--verification-manifest", action="append", default=[], help="Detached SHA-256 manifest to verify; repeat as needed.")
    package.add_argument("--phase-lock", action="append", default=[], help="Completed response phase lock to verify; repeat as needed.")
    package.add_argument("--restricted-path", action="append", default=[], help="Additional path relative to source-root to exclude; public packages always exclude restricted/ and responses/ by default.")
    package.add_argument("--restricted-release", action="store_true", help="Build a controlled restricted package instead of the public package.")
    package.add_argument("--allow-nonfinal-audit-package", action="store_true", help="Permit a clearly non-final audit package when non-safety gates remain non-PASS; safety-critical evidence is still mandatory.")
    package.add_argument("--study-id", default="tm_mri_single_physician_v2", help="Frozen study identifier.")
    package.add_argument("--release-name", default="TM_MRI_Clinical_Validation", help="Filesystem-safe release name.")

    args = parser.parse_args(argv)
    if args.command == "preflight":   # read-only; exit 0 when no check failed, else 1
        from .preflight import run_preflight
        code = run_preflight(args)
        if code:
            raise SystemExit(code)
        return
    # --- wf(L4b): F9 dispatch ---
    if args.command in F9_COMMANDS:
        code = run_f9(args, _v6_context)
        if code:
            raise SystemExit(code)
        return
    # --- end wf(L4b) ---
    if args.command in NEURAL_IMAGE_COMMANDS:   # L4a
        code = _run_neural_image(args)
        if code:
            raise SystemExit(code)
        return
    # protocol-v6.0.1: lanes L3, L57 and L89 share the v6 dispatcher and its exit codes
    if args.command in (*V6_COMMANDS, *V601_L3_COMMANDS, *L57_COMMANDS, *DATA_CENTRIC_COMMANDS):
        code = _run_v6(args)
        if code:
            raise SystemExit(code)
        return
    if args.command == "init-layout":
        init_layout()
    elif args.command == "run-minimal":
        run_minimal(args.config)
    elif args.command == "validate-native-data":
        validate_native_data(args.config, args.dataset)
    elif args.command == "run-full-validation":
        run_full_validation(
            config_path=args.config,
            experiment_id=args.experiment_id,
            run_minimal_if_needed=args.run_minimal_if_needed,
            enable_classification=args.enable_classification,
            enable_rulebook=args.enable_rulebook,
            enable_dss_predicates=args.enable_dss_predicates,
            enable_equivariance_diagnostics=args.enable_equivariance_diagnostics,
            allow_semantic_proxy_a=args.allow_semantic_proxy_a,
            native_data_required=args.native_data_required,
            bootstrap_iterations=args.bootstrap_iterations,
            max_qa_panels=args.max_qa_panels,
            for_rules_archive=args.for_rules_archive,
            strict_for_rules_import=not args.non_strict_for_rules_import,
        )
    elif args.command == "prepare-clinical-validation":
        _run_prepare_clinical_validation(args)
    elif args.command == "generate-clinical-figures":
        _run_generate_clinical_figures(args)
    elif args.command == "build-clinical-workbooks":
        _run_build_clinical_workbooks(args)
    elif args.command == "validate-clinical-workbooks":
        _run_validate_clinical_workbooks(args)
    elif args.command == "import-reader-responses":
        _run_import_reader_responses(args)
    elif args.command == "analyze-reader-study":
        _run_analyze_reader_study(args)
    elif args.command == "analyze-reader-addendum":
        _run_analyze_reader_addendum(args)
    elif args.command == "package-clinical-release":
        _run_package_clinical_release(args)
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
