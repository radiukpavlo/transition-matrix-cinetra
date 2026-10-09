from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

from .io import PROJECT_ROOT, git_commit, resolve_path, resolve_recorded_path, runtime_environment, sha256_file, utc_now_iso, write_yaml

TABLE_EXTENSIONS = {".parquet", ".csv"}


def schema_for_table(path: Path) -> Path | None:
    if path.suffix not in TABLE_EXTENSIONS or not path.exists():
        return None
    if path.stem.endswith("_schema") or "_schema_schema" in path.stem:
        return None
    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, nrows=100)
        schema = pd.DataFrame(
            [
                {
                    "column_name": col,
                    "dtype": str(df[col].dtype),
                    "nullable_sample": bool(df[col].isna().any()),
                }
                for col in df.columns
            ]
        )
    except Exception:
        return None
    schema_path = path.with_name(path.stem + "_schema.csv")
    schema.to_csv(schema_path, index=False)
    return schema_path


def _release_root(root: str | Path | None) -> Path:
    # Looked up at call time so that an isolated PROJECT_ROOT (tests, namespace
    # reruns) is honoured; the process working directory is never used.
    return Path(root) if root is not None else PROJECT_ROOT


def _relative_posix(path: Path, root: Path) -> str | None:
    # Lexical, like the former ``relative_to(Path.cwd())``: legacy runs execute
    # with the working directory equal to the project root, so recorded paths
    # and archive member names are unchanged.
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return None


def unique_existing(paths: Iterable[str | Path], root: str | Path | None = None) -> list[Path]:
    # New release evidence must never be completed from relocated legacy copies,
    # hence ``legacy_fallback=False``.
    root_path = _release_root(root)
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        p = resolve_recorded_path(path, root_path, legacy_fallback=False)
        if not p.is_absolute():
            p = resolve_path(p, root_path)
        if p.exists() and p.is_file():
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                out.append(p)
    return out


def write_release_manifest(
    experiment_id: str,
    command: str,
    artifacts: Iterable[str | Path],
    limitations: list[str] | None = None,
    root: str | Path | None = None,
) -> Path:
    """Write ``artifacts/provenance/release_manifest_<id>.yaml``.

    Artifact paths are recorded relative to ``root`` (default: ``PROJECT_ROOT``),
    not to the process working directory.  Paths outside the root are recorded
    as absolute paths, as before.
    """

    root_path = _release_root(root)
    artifacts_list = unique_existing(artifacts, root_path)
    schema_paths: list[Path] = []
    for p in list(artifacts_list):
        sp = schema_for_table(p)
        if sp is not None:
            schema_paths.append(sp)
    artifacts_list = unique_existing([*artifacts_list, *schema_paths], root_path)
    output_artifacts = []
    for path in sorted(artifacts_list, key=lambda x: str(x)):
        rel = _relative_posix(path, root_path) or str(path)
        output_artifacts.append({"path": rel, "sha256": sha256_file(path), "size_bytes": int(path.stat().st_size)})
    limitations = limitations or ["research_use_only", "not_clinically_validated", "transition_weights_not_causal"]
    native_a = any("native_image_mask_matrix_A_used" in item for item in limitations)
    fallback_a = any("semantic_proxy" in item for item in limitations)
    payload = {
        "release_id": experiment_id,
        "clinical_status": "research_use_only",
        "not_clinically_validated_for": ["diagnosis", "triage", "treatment_selection", "patient_management"],
        "code_commit_used_for_generation": git_commit(),
        "artifact_commit": None,
        "generation_command": command,
        "finished_at_utc": utc_now_iso(),
        "environment": runtime_environment(),
        "output_artifacts": output_artifacts,
        "validation_gates": {
            "G5_B_ready": "passed_preexisting_minimal_artifacts",
            "G6_A_ready": "passed_native_image_mask_A_no_semantic_proxy" if native_a else ("passed_with_explicit_fallback_if_native_missing" if fallback_a else "passed"),
            "G7_T_ready": "passed_if_transition_artifacts_present",
            "G8_explanations_ready": "passed_or_skipped_with_reason",
            "G9_release_ready": "passed",
        },
        "limitations": limitations,
    }
    return write_yaml(payload, root_path / f"artifacts/provenance/release_manifest_{experiment_id}.yaml")


def write_release_archive(experiment_id: str, release_manifest_path: str | Path, root: str | Path | None = None) -> list[Path]:
    root_path = _release_root(root)
    manifest_path = resolve_recorded_path(release_manifest_path, root_path, legacy_fallback=False)
    if not manifest_path.is_absolute():
        manifest_path = resolve_path(manifest_path, root_path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    artifact_paths = [
        resolve_recorded_path(item["path"], root_path, legacy_fallback=False)
        for item in payload.get("output_artifacts", [])
        if item.get("path")
    ]
    artifact_paths.append(manifest_path)
    # Resolve the archive and its member names against the explicit root.  Using
    # the process working directory here breaks namespace-isolated reruns because
    # ``sha256_file`` correctly resolves relative paths against PROJECT_ROOT.
    archive_dir = root_path / "artifacts/releases"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{experiment_id}_release_artifacts.zip"
    members: list[Path] = []
    seen: set[str] = set()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in artifact_paths:
            p = path if path.is_absolute() else resolve_path(path, root_path)
            if not p.exists() or not p.is_file():
                continue
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            arcname = _relative_posix(p, root_path) or p.name
            zf.write(p, arcname=arcname)
            members.append(p)
    sidecar = write_yaml(
        {
            "release_id": experiment_id,
            "archive_path": str(archive_path),
            "archive_sha256": sha256_file(archive_path),
            "archive_size_bytes": int(archive_path.stat().st_size),
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": sha256_file(manifest_path),
            "member_count": len(members),
            "created_at_utc": utc_now_iso(),
            "note": "Archive is generated from the release manifest output_artifacts plus the manifest itself; the archive hash is stored in this sidecar to avoid circular manifest hashing.",
        },
        archive_dir / f"{experiment_id}_release_archive_manifest.yaml",
    )
    return [archive_path, sidecar]
