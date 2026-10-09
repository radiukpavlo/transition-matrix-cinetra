"""Stage runner of the protocol-v6.0.1 supplementary comparators and exploratory analyses (lane L3).

The registered comparator F8 (``tmmri.comparators.cart_f8``) and the exploratory register B1-B8
(``tmmri.experiments.exploratory``) are stages of a v6 run root like every other stage: each call goes through
:func:`tmmri.runs.run_stage` with the once-only guard of :func:`tmmri.stages.once_only_guard`, so in a frozen run a
recorded stage may only be skipped (a changed input, configuration, code version or ``--force`` refuses with exit 7).

Their settings live in lane configuration files (``configs/cart_f8.yaml``, ``configs/exploratory.yaml``) that are
read from the study root (a copied configuration tree in tests) and, when the study loads them through
``configs/study_v6.yaml`` ``config_files``, from the loaded study.  The stage configuration hash covers the study
sections a stage reads (as :data:`tmmri.stages.STAGE_CONFIG_SECTIONS` does for the v6 stages) plus the canonical JSON of
each lane configuration it reads, so a comment-only edit of a lane file never changes it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import yaml

from .config import StudyConfig
from .paths import PROJECT_ROOT
from .runs import StageResult, hash_path, run_stage, sha256_json


def lane_config_path(study: StudyConfig, key: str) -> Path:
    """``configs/<key>.yaml`` of the study root (a copied tree in tests), else of the repository."""

    for base in (Path(study.root), PROJECT_ROOT):
        path = base / "configs" / f"{key}.yaml"
        if path.is_file():
            return path
    raise FileNotFoundError(f"configs/{key}.yaml not found below {study.root} or {PROJECT_ROOT}")


def lane_config(study: StudyConfig, key: str) -> dict[str, Any]:
    """The parsed lane configuration ``key`` (the loaded study's copy when ``config_files`` lists it)."""

    if key in study.configs:
        return dict(study.configs[key])
    with lane_config_path(study, key).open("r", encoding="utf-8") as handle:
        return dict(yaml.safe_load(handle) or {})


def lane_config_digest(study: StudyConfig, key: str) -> str:
    """SHA-256 of the canonical JSON of a lane configuration (comment-only edits leave it unchanged)."""

    return sha256_json(lane_config(study, key))


def secondary_config_hash(study: StudyConfig, sections: Iterable[str], lane_keys: Iterable[str]) -> str:
    """Stage configuration hash: the study sections read (:meth:`StudyConfig.sections_digest`) plus each lane config."""

    digest = dict(study.sections_digest(list(sections)))
    for key in sorted(set(lane_keys)):
        digest[f"lane:configs/{key}.yaml"] = lane_config_digest(study, key)
    return hashlib.sha256("\n".join(f"{k}\t{v}" for k, v in sorted(digest.items())).encode("utf-8")).hexdigest()


def run_secondary_stage(ctx: Any, name: str, fn: Callable[[], Any], inputs: Iterable[Path], outputs: Iterable[Path], *,
                        sections: Iterable[str], lane_keys: Iterable[str], params: Mapping[str, Any] | None = None,
                        status: str = "BUILT") -> StageResult:
    """Content-addressed stage call with the once-only guard (the counterpart of ``tmmri.stages._stage``)."""

    from .stages import once_only_guard

    run = ctx.require_run(name)
    sections = list(sections)
    lane_keys = list(lane_keys)
    return run_stage(run, name, fn, list(inputs), list(outputs), config_hash=secondary_config_hash(ctx.study, sections, lane_keys),
                     params={**dict(params or {}), "config_sections": sections, "lane_configs": [f"configs/{k}.yaml" for k in lane_keys]},
                     force=ctx.force, n_jobs=ctx.jobs, status=status, rerun_guard=once_only_guard(run, name))


def input_hashes(run: Any, paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    """``{label: {path (run-root portable), sha256}}`` of the inputs a provenance record names."""

    return {label: {"path": run.portable(path), "sha256": hash_path(path)} for label, path in sorted(paths.items())}


def require_partition_unlocked(ctx: Any, split: str, stage: str) -> str | None:
    """The M&Ms locks for a stage reading an external partition: ``mms_external`` obeys
    :func:`tmmri.stages.require_mms_unlocked` (tag ``protocol-v6.0.0`` or the synthetic override); an M&Ms-2 partition
    obeys the protocol-v6.0.1 lock ``tmmri.stages.require_mms2_unlocked`` (lane L1) and refuses (exit 3) while that
    lock is not part of the build.  ACDC partitions need no lock."""

    from . import stages

    if split.startswith("mms2"):
        guard = getattr(stages, "require_mms2_unlocked", None)
        if guard is None:
            raise stages.ProtocolFreezeError(f"{stage}: the M&Ms-2 lock of protocol-v6.0.1 (tmmri.stages.require_mms2_unlocked) is not part "
                                             "of this build; no M&Ms-2 artifact is read")
        return guard(ctx, stage)
    if split.startswith("mms"):
        return stages.require_mms_unlocked(ctx, stage)
    return None
