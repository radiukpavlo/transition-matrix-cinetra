"""Canonical repository paths, runtime roots and narrowly scoped historical path handling.

The repository root can be overridden with ``TMMRI_PROJECT_ROOT`` for isolated
tests and reproducible local runs.  The runtime roots ``DATA_ROOT``,
``ARTIFACT_ROOT`` and ``SOURCES_ROOT`` are git-ignored (Docker volumes
``tmmri_data`` and ``tmmri_artifacts`` and the read-only source mounts) and can
be overridden with ``TMMRI_DATA_ROOT``, ``TMMRI_ARTIFACT_ROOT`` and
``TMMRI_SOURCES_ROOT``.  Tracked evidence lives in ``registry/`` (active source
manifests) and ``legacy/v5/`` (relocated v4/v5 outputs); writers never target
those trees.

``resolve_recorded_path`` is intentionally read-only compatibility support for
historical manifests, checkpoints and legacy input defaults; it never creates
compatibility directories or rewrites records.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Mapping


_SOURCE_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("TMMRI_PROJECT_ROOT", _SOURCE_PROJECT_ROOT)).resolve()


def _runtime_root(variable: str, default: Path) -> Path:
    """Return an environment-overridable runtime root (relative values are project-relative)."""

    value = os.environ.get(variable, "").strip()
    if not value:
        return default
    candidate = Path(value)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


# Tracked protocol trees (read-only for every pipeline writer).
REGISTRY_DIR = PROJECT_ROOT / "registry"
SOURCE_REGISTRY_DIR = REGISTRY_DIR / "sources"
LEGACY_DIR = PROJECT_ROOT / "legacy"
LEGACY_V5_DIR = LEGACY_DIR / "v5"

# Git-ignored runtime roots (Docker volumes in the container runtime).
DATA_ROOT = _runtime_root("TMMRI_DATA_ROOT", PROJECT_ROOT / "data")
ARTIFACT_ROOT = _runtime_root("TMMRI_ARTIFACT_ROOT", PROJECT_ROOT / "artifacts")
SOURCES_ROOT = _runtime_root("TMMRI_SOURCES_ROOT", PROJECT_ROOT / "data" / "sources")
STAGED_DIR = DATA_ROOT / "staged"

RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,80}$")


def validate_run_id(run_id: str) -> str:
    """Return ``run_id`` unchanged if it matches :data:`RUN_ID_PATTERN`, else raise ``ValueError``."""

    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            f"invalid run id {run_id!r}: expected {RUN_ID_PATTERN.pattern} "
            "(lower-case letters, digits, '.', '_' and '-'; 3 to 81 characters)"
        )
    return run_id


def run_root(run_id: str, artifact_root: Path | None = None) -> Path:
    """Return the run root ``<ARTIFACT_ROOT>/<run_id>`` after validating the run id (nothing is created)."""

    return Path(artifact_root if artifact_root is not None else ARTIFACT_ROOT) / validate_run_id(run_id)


# Legacy flat layout constants.  They keep the v4/v5 and governed clinical flows
# working; since the repository refactor (ADR-0004) they point into the
# git-ignored runtime directories below the project root.
CONFIGS_DIR = PROJECT_ROOT / "configs"
LEGACY_CONFIGS_DIR = CONFIGS_DIR / "legacy"
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
RULES_DIR = ARTIFACTS_DIR / "rules"
TRANSFORMS_DIR = ARTIFACTS_DIR / "transforms"
PROVENANCE_DIR = ARTIFACTS_DIR / "provenance"
RELEASES_DIR = ARTIFACTS_DIR / "releases"
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript"
CLINICAL_VALIDATION_DIR = PROJECT_ROOT / "clinical_validation"

DOCS_DIR = PROJECT_ROOT / "docs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# These are exact first-component mappings.  In particular, ``rules`` below
# ``clinical_validation`` is not remapped because only the first component is
# examined.
LEGACY_FIRST_COMPONENT_MAP: Mapping[str, str] = {
    "config": "configs",
    "checkpoints": "artifacts/checkpoints",
    "models": "artifacts/models",
    "provenance": "artifacts/provenance",
    "reports": "artifacts/reports",
    "releases": "artifacts/releases",
    "rules": "artifacts/rules",
    "splits": "data/splits",
    "transforms": "artifacts/transforms",
}

# Relocated tracked trees (ADR-0004).  A missing path whose leading components
# match a key falls back to the same remainder below the mapped tracked root,
# but only when that fallback exists.
RELOCATED_PREFIX_MAP: Mapping[tuple[str, ...], tuple[str, ...]] = {
    ("artifacts",): ("legacy", "v5", "artifacts"),
    ("data", "derived"): ("legacy", "v5", "data", "derived"),
    ("data", "features"): ("legacy", "v5", "data", "features"),
    ("data", "for_rules"): ("legacy", "v5", "data", "for_rules"),
    ("data", "interim"): ("legacy", "v5", "data", "interim"),
    ("data", "splits"): ("legacy", "v5", "data", "splits"),
    ("data", "raw_registry"): ("registry", "sources"),
    ("configs", "pipeline"): ("configs", "legacy", "pipeline"),
    ("configs", "labels"): ("configs", "legacy", "labels"),
    ("configs", "segmentation"): ("configs", "legacy", "segmentation"),
}
PATH_RESOLVER_VERSION = "repository-layout-v2"


def resolve_path(path: str | Path, root: Path = PROJECT_ROOT) -> Path:
    """Resolve a project-relative path without changing absolute paths."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(root) / candidate


def _relative_to_root(path: Path, root: Path) -> Path | None:
    try:
        return path.resolve(strict=False).relative_to(root.resolve())
    except ValueError:
        return None


def relocated_path(relative: str | Path) -> Path | None:
    """Return the tracked relocation target of a project-relative path, or ``None``.

    ``artifacts/...`` and ``data/{derived,features,for_rules,interim,splits}/...``
    map to ``legacy/v5/<same path>``; ``data/raw_registry/...`` maps to
    ``registry/sources/...``; ``configs/{pipeline,labels,segmentation}/...``
    map to ``configs/legacy/...``.  Matching is case-insensitive on the leading
    components; the remainder is kept verbatim.
    """

    parts = Path(relative).parts
    folded = tuple(part.casefold() for part in parts)
    for prefix, target in RELOCATED_PREFIX_MAP.items():
        if len(parts) > len(prefix) and folded[: len(prefix)] == prefix:
            return Path(*target).joinpath(*parts[len(prefix) :])
    return None


def resolve_recorded_path(path: str | Path, root: Path = PROJECT_ROOT, *, legacy_fallback: bool = True) -> Path:
    """Resolve a path stored in a historical manifest, checkpoint or legacy default.

    Existing paths always win.  Missing relative paths and missing absolute
    paths inside ``root`` may have only their exact first component mapped by
    :data:`LEGACY_FIRST_COMPONENT_MAP`.  If the (mapped) path is still missing
    and ``legacy_fallback`` is true, a relocated tracked copy
    (:func:`relocated_path`) is returned when it exists.  Absolute paths outside
    ``root`` are returned unchanged, preventing accidental remapping of
    unrelated machine paths or external cache locations.  Writers of new
    evidence (release and provenance manifests) pass ``legacy_fallback=False``
    so that a missing output is never replaced by historical evidence.
    """

    recorded = Path(path)
    if recorded.exists():
        return recorded

    root_path = Path(root).resolve()
    if recorded.is_absolute():
        relative = _relative_to_root(recorded, root_path)
        if relative is None:
            return recorded
    else:
        current_candidate = root_path / recorded
        if current_candidate.exists():
            return current_candidate
        relative = recorded

    if not relative.parts:
        return recorded
    mapped_root = LEGACY_FIRST_COMPONENT_MAP.get(relative.parts[0].casefold())
    if mapped_root is not None:
        relative = Path(mapped_root).joinpath(*relative.parts[1:])
        resolved = root_path / relative
    else:
        resolved = recorded
    if legacy_fallback and not (root_path / relative).exists():
        relocated = relocated_path(relative)
        if relocated is not None and (root_path / relocated).exists():
            return root_path / relocated
    return resolved
