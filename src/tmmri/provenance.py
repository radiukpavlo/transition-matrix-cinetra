from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import PROJECT_ROOT, git_commit, resolve_path, resolve_recorded_path, runtime_environment, sha256_file, utc_now_iso, write_yaml


def build_provenance_manifest(paths: list[str | Path], commands: list[str], release_id: str = "v2") -> dict[str, Any]:
    artifacts = []
    for raw_path in paths:
        # New provenance never substitutes relocated legacy copies for missing outputs.
        path = resolve_recorded_path(raw_path, legacy_fallback=False)
        if not path.is_absolute():
            path = resolve_path(path)
        if not path.exists() or not path.is_file():
            continue
        try:
            rel = path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            rel = str(path)
        artifacts.append({"path": rel, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return {
        "release_id": release_id,
        "created_at_utc": utc_now_iso(),
        "git_commit": git_commit(),
        "clinical_status": "research_use_only",
        "commands": commands,
        "runtime_environment": runtime_environment(),
        "artifacts": sorted(artifacts, key=lambda x: x["path"]),
    }


def write_provenance_manifest(paths: list[str | Path], commands: list[str], output_path: str | Path = "artifacts/provenance/provenance_manifest_v2.yaml") -> Path:
    return write_yaml(build_provenance_manifest(paths, commands), output_path)
