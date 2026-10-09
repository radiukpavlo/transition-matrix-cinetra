"""Read-only checks for the canonical repository layout and active references."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


LEGACY_COMPONENTS = (
    "config",
    "checkpoints",
    "models",
    "reports",
    "rules",
    "splits",
    "transforms",
    "provenance",
    "releases",
)
LEGACY_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_./\\-])(?P<component>config|checkpoints|models|reports|rules|splits|transforms|provenance|releases)[/\\][A-Za-z0-9_./\\-]+"
)
# Protocol-v6 run roots (artifacts/<run_id>/, src/tmmri/runs.py; SPECIFICATION.md 14.1) have a rules/ subfolder of
# their own. These run-root-relative entries are current, not references to the retired top-level rules/ directory.
V6_RUN_ROOT_RULE_DIRECTORIES = ("ablations", "d_model", "dataset_label", "measured_oracle", "refit_arm")
V6_RUN_ROOT_RULE_FILE_PREFIXES = ("rule_anchor_manifest.", "semantic_eligibility.")
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]\n]+\]\(([^)\n]+)\)")
FORBIDDEN_PARTS = {"desktop.ini", "__pycache__", ".pytest_cache", ".venv", "node_modules"}
FORBIDDEN_SUFFIXES = {".aux", ".bbl", ".blg", ".log", ".out"}

# Repository layout after the refactor (ADR-0004; design brief section 5).
REQUIRED_DIRECTORIES = (
    "clinical_validation",
    "configs/clinical",
    "configs/legacy/labels",
    "configs/legacy/pipeline",
    "configs/legacy/segmentation",
    "docs/adr",
    "docs/history",
    "docs/history/implementation_specs",
    "docs/references",
    "legacy/v5/artifacts",
    "legacy/v5/data/derived",
    "legacy/v5/data/features",
    "legacy/v5/data/for_rules",
    "legacy/v5/data/interim",
    "legacy/v5/data/splits",
    "manuscript/scripts",
    "registry/sources",
    "scripts/clinical_validation",
    "scripts/data",
    "scripts/experiments",
    "scripts/maintenance",
    "src/tmmri/clinical",
    "src/tmmri/experiments",
    "src/tmmri/ingest",
    "src/tmmri/semantics",
    "third_party/benjamin-plus",
)
REQUIRED_FILES = (
    "legacy/v5/README.md",
    "scripts/maintenance/crlf_worktree_paths.txt",
    "docs/adr/0004-repository-refactor-and-legacy-relocation.md",
    # Live protocol-v6 documents (docs/README.md indexes them).
    "SPECIFICATION.md",
    "CONTEXT.md",
    "protocol/amendments.md",
    "docs/README.md",
    "docs/EXECUTION_ENVIRONMENT_SPEC.md",
    "docs/INSTALLATION_AGENTS_PROMPT.md",
    "docs/DATA_ACQUISITION_AGENTS_PROMPT.md",
    "docs/EXPERIMENT_AGENTS_PROMPT.md",
    "docs/EXPERIMENT_EXECUTION_PLAN.md",
    "docs/V601_EXECUTION_RUNBOOK.md",
    "docs/history/README.md",
    "docs/adr/0001-acdc-development-full-mms-external.md",
    "docs/adr/0002-v6-protocol-decisions.md",
    "docs/adr/0003-docker-desktop-linux-container-runtime.md",
    # Protocol-v6 configuration and registry (SPECIFICATION.md Section 14.1).
    "configs/study_v6.yaml",
    "configs/cohorts.yaml",
    "configs/splits.yaml",
    "configs/formal.yaml",
    "configs/transition.yaml",
    "configs/wedd.yaml",
    "configs/rule_policy.yaml",
    "configs/equivariance.yaml",
    "configs/deep_encoder.yaml",
    "configs/bootstrap.yaml",
    "configs/runtime.env",
    "configs/source_bindings/acdc.json",
    "configs/source_bindings/mms.json",
    "registry/semantic_registry.yaml",
    "registry/reason_codes.yaml",
    "registry/pattern_templates.yaml",
    "registry/estimator_partition_registry.csv",
    "registry/sources/acdc_release_file_manifest.csv",
    "registry/sources/mms_full_release_manifest.csv",
    "registry/sources/mms_full_release_metadata_manifest.csv",
    "registry/sources/mms_full_release_receipt.yaml",
    # Protocol-v6 stage DAG.
    "src/tmmri/stages.py",
    "src/tmmri/runs.py",
    "src/tmmri/rule_pipeline.py",
    "src/tmmri/ingest/staging.py",
    "src/tmmri/semantics/contract.py",
    "tests/conftest.py",
    "tests/fixtures_v6.py",
    "tests/test_registry_check.py",
    "tests/test_v6_integration.py",
)
# Superseded documents archived below docs/history/ (docs/history/README.md); the old paths must not return.
ARCHIVED_DOC_PATHS = (
    "IMPLEMENTATION_PLAN.md",
    "docs/experiment_machine",
    "docs/methodology",
    "docs/project_history",
    "docs/repository_layout.md",
    "docs/reproducibility.md",
)
# Git-ignored runtime roots: only these tracked entries may live below them.
RUNTIME_ROOT_TRACKED_ALLOWLIST = ("data/README.md", "artifacts/README.md")
RUNTIME_ROOT_TRACKED_PREFIXES = ("data/raw/",)
# Module files replaced by the package re-layout; no compatibility shims.
REMOVED_MODULES = (
    "a_extraction", "evaluation", "full_validation", "transition_experiments",
    "inventory", "native_data", "geometry", "labels",
    "measurements", "features", "qc", "manual_review", "visual_qc",
    "clinical_validation", "clinical_sampling", "clinical_stress", "clinical_model_comparisons",
    "clinical_figures", "clinical_schema", "clinical_workbook", "clinical_release",
    "statistics_reader_study",
)


def _normalise(path: Path | str) -> str:
    return Path(path).as_posix().lstrip("./").lower()


def _tracked_paths(root: Path) -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            check=False,
            capture_output=True,
        )
    except OSError:
        result = None
    if result is not None and result.returncode == 0:
        return [Path(item.decode("utf-8")) for item in result.stdout.split(b"\0") if item]

    paths: list[Path] = []
    for current, dirs, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        dirs[:] = [name for name in dirs if name != ".git" and not (current_path / name).is_symlink()]
        paths.extend(current_path.relative_to(root) / name for name in files)
    return paths


def _allowlisted_path(path: Path) -> bool:
    parts = tuple(part.lower() for part in path.parts)
    text = "/".join(parts)
    return (
        text == ".gitignore"
        or text == "src/tmmri/paths.py"
        or text == "src/tmmri/clinical/validation.py"
        or text == "src/tmmri/clinical/workbook.py"
        or text == "scripts/maintenance/crlf_worktree_paths.txt"
        or text == "tests/test_repository_layout.py"
        or text == "tests/test_clinical_workflow.py"
        or text.startswith("docs/history/")
        or text.startswith("manuscript/archive/")
        or text.startswith("manuscript/reports/")
        or text.startswith("legacy/")
        or text.startswith("clinical_validation/")
        or text.startswith("registry/sources/")
    )


def _active_markdown(path: Path) -> bool:
    text = "/".join(part.lower() for part in path.parts)
    if path.suffix.lower() != ".md":
        return False
    historical = text.startswith(("docs/history/", "legacy/v5/artifacts/", "legacy/v5/data/"))
    if text in {"readme.md", "contributing.md"} or text.endswith("/readme.md"):
        return not historical
    return (
        text.startswith("docs/")
        and not historical
    ) or text.startswith("manuscript/notes/") or text == "manuscript/readme.md" or text == "manuscript/reports/clinical_validation/readme.md" or text.startswith("scripts/") or text == "configs/readme.md" or text == "data/readme.md" or text == "artifacts/readme.md"


def _read_text(root: Path, path: Path) -> str | None:
    try:
        return (root / path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _is_v6_run_root_rule_path(matched: str) -> bool:
    component, _, remainder = matched.replace("\\", "/").partition("/")
    if component != "rules":
        return False
    head = remainder.split("/", 1)[0]
    return head in V6_RUN_ROOT_RULE_DIRECTORIES or remainder.startswith(V6_RUN_ROOT_RULE_FILE_PREFIXES)


def _check_legacy_references(root: Path, tracked: list[Path]) -> list[str]:
    violations: list[str] = []
    for path in tracked:
        if _allowlisted_path(path):
            continue
        text = _read_text(root, path)
        if text is None:
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            for match in LEGACY_PATH_RE.finditer(line):
                if _is_v6_run_root_rule_path(match.group(0)):
                    continue
                violations.append(
                    f"{path.as_posix()}:{line_number}: active legacy path {match.group(0)!r}"
                )
    return violations


def _check_markdown_links(root: Path, tracked: list[Path]) -> list[str]:
    violations: list[str] = []
    for path in tracked:
        if not _active_markdown(path):
            continue
        text = _read_text(root, path)
        if text is None:
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            for match in MARKDOWN_LINK_RE.finditer(line):
                raw_target = match.group(1).strip()
                if raw_target.startswith("<") and ">" in raw_target:
                    raw_target = raw_target[1 : raw_target.index(">")]
                else:
                    raw_target = raw_target.split(maxsplit=1)[0]
                if not raw_target or raw_target.startswith(("#", "http://", "https://", "mailto:", "ftp://")):
                    continue
                target = raw_target.split("#", 1)[0].split("?", 1)[0]
                if not target:
                    continue
                candidate = Path(target)
                if candidate.is_absolute() or not (root / path.parent / candidate).exists():
                    violations.append(
                        f"{path.as_posix()}:{line_number}: missing relative Markdown link {raw_target!r}"
                    )
    return violations


def _check_forbidden_tracked(root: Path, tracked: list[Path]) -> list[str]:
    del root
    violations: list[str] = []
    for path in tracked:
        parts = {part.lower() for part in path.parts}
        if parts.intersection(FORBIDDEN_PARTS):
            violations.append(f"{path.as_posix()}: forbidden generated/local path is tracked")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(f"{path.as_posix()}: LaTeX auxiliary file is tracked")
        if _normalise(path).startswith("manuscript/build/"):
            violations.append(f"{path.as_posix()}: manuscript build output must not be tracked")
    return violations


def _check_manuscript_dependencies(root: Path) -> list[str]:
    violations: list[str] = []
    article = root / "manuscript" / "article"
    source_path = article / "main.tex"
    if not source_path.is_file():
        return ["manuscript/article/main.tex: active source is missing"]
    source = source_path.read_text(encoding="utf-8")

    class_matches = re.findall(r"\\documentclass(?:\[[^]]*\])?\{([^}]+)\}", source)
    if not class_matches:
        violations.append("manuscript/article/main.tex: document class is missing")
    for class_name in class_matches:
        class_path = article / (class_name if class_name.endswith(".cls") else f"{class_name}.cls")
        if not class_path.is_file():
            violations.append(f"manuscript/article/main.tex: missing class {class_path.relative_to(root).as_posix()}")

    bibliography_matches = re.findall(r"\\bibliography\{([^}]+)\}", source)
    if not bibliography_matches:
        violations.append("manuscript/article/main.tex: bibliography dependency is missing")
    for names in bibliography_matches:
        for name in names.split(","):
            bib_path = article / (name.strip() if name.strip().endswith(".bib") else f"{name.strip()}.bib")
            if not bib_path.is_file():
                violations.append(f"manuscript/article/main.tex: missing bibliography {bib_path.relative_to(root).as_posix()}")

    graphics_roots = [article / "figures" / "generated", article / "figures" / "clinical_validation"]
    for graphics_root in graphics_roots:
        if not graphics_root.is_dir():
            violations.append(f"{graphics_root.relative_to(root).as_posix()}: active graphics directory is missing")
    include_names = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source)
    extensions = (".pdf", ".png", ".jpg", ".jpeg", ".svg", ".tif", ".tiff")
    for name in include_names:
        include_path = Path(name.strip())
        candidates: list[Path] = []
        bases = [article] if include_path.parts and include_path.parts[0] == "figures" else graphics_roots
        for base in bases:
            candidate = base / include_path
            candidates.append(candidate)
            if candidate.suffix.lower() not in extensions:
                candidates.extend(candidate.with_suffix(ext) for ext in extensions)
        if not any(candidate.is_file() for candidate in candidates):
            violations.append(f"manuscript/article/main.tex: missing figure dependency {name.strip()!r}")
    return violations


def archived_doc_present(root: Path, path: str) -> bool:
    """True if an archived document path holds a file again (empty folders, e.g. from init-layout, are ignored)."""

    target = root / path
    return target.is_file() or (target.is_dir() and any(item.is_file() for item in target.rglob("*")))


def check_required_layout(root: Path) -> list[str]:
    """Report missing directories and files, and re-created archived documents, of the post-refactor layout."""

    violations = [f"{path}: required directory is missing" for path in REQUIRED_DIRECTORIES if not (root / path).is_dir()]
    violations.extend(f"{path}: required file is missing" for path in REQUIRED_FILES if not (root / path).is_file())
    violations.extend(
        f"{path}: archived document path exists again (it lives in docs/history/)"
        for path in ARCHIVED_DOC_PATHS
        if archived_doc_present(root, path)
    )
    return violations


def check_runtime_roots(root: Path, tracked: list[Path] | None = None) -> list[str]:
    """Report tracked files inside the git-ignored runtime roots data/ and artifacts/."""

    tracked = _tracked_paths(root) if tracked is None else tracked
    violations: list[str] = []
    for path in tracked:
        text = path.as_posix()
        if not text.startswith(("data/", "artifacts/")):
            continue
        if text in RUNTIME_ROOT_TRACKED_ALLOWLIST or text.startswith(RUNTIME_ROOT_TRACKED_PREFIXES):
            continue
        violations.append(f"{text}: tracked file inside a runtime root (relocate it to legacy/ or registry/)")
    return violations


def check_removed_modules(root: Path) -> list[str]:
    """Report module files that the package re-layout removed (no compatibility shims)."""

    package = root / "src" / "tmmri"
    return [
        f"src/tmmri/{name}.py: removed module path exists again (no compatibility shims)"
        for name in REMOVED_MODULES
        if (package / f"{name}.py").exists() or (package / name / "__init__.py").exists()
    ]


def check_repository(root: Path) -> list[str]:
    root = root.resolve()
    tracked = _tracked_paths(root)
    violations = []
    violations.extend(check_required_layout(root))
    violations.extend(check_runtime_roots(root, tracked))
    violations.extend(check_removed_modules(root))
    violations.extend(_check_forbidden_tracked(root, tracked))
    violations.extend(_check_legacy_references(root, tracked))
    violations.extend(_check_markdown_links(root, tracked))
    violations.extend(_check_manuscript_dependencies(root))
    return sorted(set(violations))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    violations = check_repository(args.root)
    if violations:
        print("Repository layout check failed:")
        print("\n".join(f"- {item}" for item in violations))
        return 1
    print(f"Repository layout check passed for {args.root.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
