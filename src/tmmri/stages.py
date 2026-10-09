"""Protocol CMR-TRANS-AUDIT 6.0.0 stages: one function per CLI stage (SPECIFICATION.md Section 15.1), on a run root.

Stages, in the order of :data:`PIPELINE_STEPS` (``tmmri.cli run``): ``stage-sources`` (verify-and-stage,
:mod:`tmmri.ingest.staging`), ``validate-cohort`` (cohort tables, expected counts from ``configs/``, split manifest
with the v6 hash folds), ``build-semantics``, ``extract-formal``, ``fit-transition``, ``fit-predictor``,
``discretize``, ``check-semantic-eligibility`` (M7 gates), ``induce-rules``, ``gate-rules`` (cross-fitted gate on
nested inner out-of-fold substrates), ``freeze-analysis-bundle``, ``evaluate`` (``acdc_test``, ``mms_external``; external strata, fixed-sequence
decision and the within-family BH tables), ``audit-shift`` (label-free domain classifier, after M0),
``fit-refit-arm`` (Decision D17), ``summarize-predictor`` (the audited-predictor metrics of Section 13.2 from the
recorded decisions), ``run-ablations`` (Section 13.3 matrix, secondary), ``run-equivariance`` (Decision D19, ACDC
development), ``build-manuscript-tables`` (Section 16 Tables 1-10 from recorded artifacts, :mod:`tmmri.manuscript_tables`)
and ``write-release-manifest``.  The METHODS APIs of ``INTERFACES_PHASE2.md`` are imported lazily; a missing module or
function raises :class:`NotImplementedError` naming it.

Content addressing (Decision D23).  Every stage writes ``.stages/<stage>.json`` through :func:`tmmri.runs.run_stage`;
its configuration hash covers only the configuration *sections* the stage reads (:data:`STAGE_CONFIG_SECTIONS`), so a
changed evaluation-only section re-runs the evaluation stages and leaves every upstream record valid.  The frozen
bundle records the sections of the fitting stages (:data:`BUNDLE_STAGES`); a change to any of them after the freeze
makes the run refuse (a new run id is required).

Once-only guard.  In a run whose ``.stages/FROZEN.json`` exists, a stage that already has a record may only be
skipped: if its inputs, code version, configuration hash or parameters differ from the record (or ``--force`` is
given, or a recorded output changed), it refuses with :class:`StageRecordLockedError` (exit 7) instead of
re-executing, so an evaluation is never silently recomputed after a code or configuration change; a new run id is
required.  Stages without a record run normally, and ``write-release-manifest`` (which re-hashes the run root) is
not a recorded stage and always runs.

Guards.  Every stage that reads M&Ms images or masks to compute B, A, reconstructions, rules or metrics calls
:func:`require_mms_unlocked`: it refuses unless :func:`protocol_frozen` (the annotated tag ``protocol-v6.0.0``
exists and HEAD contains it).  ``--allow-unfrozen-synthetic`` is accepted only when the staged M&Ms data come from
a synthetic fixture (marker ``SYNTHETIC_FIXTURE``).  Staging and hash verification of M&Ms are always allowed.
``evaluate --split mms_external`` additionally requires a frozen, hash-verified analysis bundle, and
``fit-refit-arm`` requires the recorded primary external evaluation.  M&Ms-2 (protocol-v6.0.1; ``--dataset mms2``,
``--split mms2_external``) mirrors this with :func:`require_mms2_unlocked` (tag ``protocol-v6.0.1``); its stage
functions beyond the shared ones live in :mod:`tmmri.mms2_stages`.
"""

from __future__ import annotations

import importlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from .config import StudyConfig, load_study
from .ingest.sources import (
    ACDC_RELEASE_MANIFEST,
    MMS_FULL_MANIFEST,
    MMS_METADATA_MANIFEST,
    canonical_dataset,
    cohort_key,
    is_synthetic_fixture,
    load_acdc_manifest,
    load_mms_manifest,
    load_mms_metadata_manifest,
    resolve_source_root,
    synthetic_manifest_path,
)
from .paths import DATA_ROOT, PROJECT_ROOT
from .runs import (
    RunRoot,
    atomic_write_json,
    code_version,
    hash_path,
    parallel_map,
    resolve_n_jobs,
    resolve_source_readers,
    run_stage,
    sha256_json,
)

# Ordered steps of `tmmri.cli run` (stage name, dataset/split argument); milestone order of SPECIFICATION.md Section 15.
PIPELINE_STEPS: tuple[tuple[str, str | None], ...] = (
    ("stage-sources", "acdc"),
    ("stage-sources", "mms"),
    ("validate-cohort", None),
    ("build-semantics", "acdc"),
    ("extract-formal", "acdc"),
    ("fit-transition", None),
    ("fit-predictor", None),
    ("discretize", None),
    ("check-semantic-eligibility", None),
    ("induce-rules", None),
    ("gate-rules", None),
    ("freeze-analysis-bundle", None),
    ("evaluate", "acdc_test"),
    ("build-semantics", "mms"),
    ("extract-formal", "mms"),
    ("evaluate", "mms_external"),
    ("audit-shift", None),
    ("fit-refit-arm", None),
    ("summarize-predictor", None),
    ("run-ablations", None),
    ("run-equivariance", "acdc"),
    ("build-manuscript-tables", None),
    ("build-manuscript-figures", None),   # protocol v6.0.1 A3 (tmmri.manuscript_figures)
    ("write-release-manifest", None),
)
EVALUATION_SPLITS = ("acdc_test", "mms_external")
BUNDLE_DIRS = ("transitions", "thresholds", "rules")
BUNDLE_FILES = ("formal/d_model_predictor.joblib", "formal/A_ACDC_extractor.json", "qc/outlier_reference.json",
                "contracts/split_manifest.parquet")
SECONDARY_SUBDIRS = ("refit_arm", "ablations")   # secondary outputs; never part of (or loaded by) the primary bundle

# Configuration sections read by each stage (StudyConfig.section syntax); the stage record hashes only these.
STAGE_CONFIG_SECTIONS: dict[str, tuple[str, ...]] = {
    "validate-cohort": ("study_v6.partition_sizes", "study_v6.artifact_subfolders", "cohorts", "splits",
                        "source_bindings/acdc", "source_bindings/mms", "registry.label_tables"),
    "build-semantics": ("cohorts", "registry", "reason_codes", "transition.targets.primary_family"),
    "extract-formal": ("formal", "cohorts.acdc.label_map", "cohorts.mms.label_map"),
    "fit-transition": ("transition", "registry", "cohorts.qc_severity"),
    "fit-predictor": ("formal.audited_predictor", "cohorts.qc_severity"),
    "discretize": ("wedd", "bootstrap.threshold_bootstrap", "rule_policy.rulebooks", "registry", "cohorts.qc_severity"),
    "check-semantic-eligibility": ("rule_policy.rulebooks", "rule_policy.eligibility_gates", "registry", "cohorts.qc_severity",
                                   "wedd.primary_profile"),
    "induce-rules": ("rule_policy", "registry", "wedd.primary_profile", "cohorts.qc_severity"),
    "gate-rules": ("rule_policy", "registry", "wedd", "formal.audited_predictor", "transition", "cohorts.qc_severity"),
    "evaluate": ("rule_policy", "reason_codes", "cohorts", "registry", "wedd.primary_profile", "wedd.per_object_stability", "transition",
                 "bootstrap.fixed_model_bootstrap", "bootstrap.primary_test", "bootstrap.multiplicity",
                 "bootstrap.subgroup_reporting_thresholds", "splits.acdc.acdc_dev.outer_folds", "splits.refit_arm.eval_strata"),
    "audit-shift": ("bootstrap.shift_audit", "bootstrap.fixed_model_bootstrap", "bootstrap.subgroup_reporting_thresholds",
                    "cohorts.qc_severity", "reason_codes"),
    "fit-refit-arm": ("splits.refit_arm", "cohorts", "transition", "formal.audited_predictor", "wedd", "rule_policy",
                      "bootstrap", "registry", "reason_codes"),
    "run-ablations": ("transition", "formal", "wedd", "rule_policy", "bootstrap", "registry", "reason_codes", "cohorts"),
    "run-equivariance": ("equivariance", "formal", "registry", "cohorts", "reason_codes", "transition"),
    # Read-only summaries of recorded artifacts (SPECIFICATION.md 13.2 and 16); they never fit anything.
    "summarize-predictor": ("formal.audited_predictor", "formal.comparator_predictor", "bootstrap.fixed_model_bootstrap",
                            "bootstrap.calibration", "cohorts.qc_severity", "registry.label_tables"),
    "build-manuscript-tables": ("registry", "bootstrap.subgroup_reporting_thresholds", "bootstrap.primary_test", "cohorts.qc_severity",
                                "formal.comparator_predictor", "rule_policy.rulebooks", "wedd.primary_profile"),
    # --- protocol v6.0.1 (lane L2): read-only reporting stages (tmmri.predictor_reporting, tmmri.manuscript_figures)
    "summarize-predictor-extended": ("formal.audited_predictor", "formal.comparator_predictor", "bootstrap.fixed_model_bootstrap",
                                     "bootstrap.calibration", "bootstrap.subgroup_reporting_thresholds", "cohorts.qc_severity",
                                     "registry.label_tables", "registry.primary_families", "transition.targets"),
    "build-manuscript-figures": ("registry", "bootstrap.subgroup_reporting_thresholds", "bootstrap.primary_test", "transition.targets"),
    # --- end protocol v6.0.1
}
# --- protocol-v6.0.1 (M&Ms-2; L1) --------------------------------------------------------------------------------------
# M&Ms-2 stage records (``<stage>:MMS2``, ``<stage>:mms2_external``) hash their own sections: the v6.0.0 sections above
# plus the amendment-scoped ``cohorts.mms2`` / ``splits.mms2`` (excluded from the whole-file digests; tmmri.config
# AMENDMENT_SCOPED_KEYS), so the v6.0.0 records and the frozen bundle stay valid.  :func:`_stage` looks up the full
# record name first.
MMS2_SPLIT = "mms2_external"
STAGE_CONFIG_SECTIONS.update({
    "validate-cohort:MMS2": ("cohorts.mms2", "splits.mms2", "cohorts.duplicate_image_pairs", "registry.label_tables"),
    "build-semantics:MMS2": STAGE_CONFIG_SECTIONS["build-semantics"] + ("cohorts.mms2",),
    "extract-formal:MMS2": STAGE_CONFIG_SECTIONS["extract-formal"] + ("cohorts.mms2.label_map",),
    f"evaluate:{MMS2_SPLIT}": STAGE_CONFIG_SECTIONS["evaluate"] + ("cohorts.mms2", "splits.mms2"),
    "audit-shift:MMS2": STAGE_CONFIG_SECTIONS["audit-shift"] + ("cohorts.mms2",),
    f"summarize-predictor:{MMS2_SPLIT}": STAGE_CONFIG_SECTIONS["summarize-predictor"] + ("cohorts.mms2",),
    # Decision D36: the H4 family (tmmri.confirmatory_h4), confirmatory on mms2_external, descriptive on acdc_test and mms_external
    "evaluate-h4": ("cohorts.mms2.endpoints", "bootstrap.fixed_model_bootstrap", "bootstrap.primary_test", "registry.primary_families",
                    "transition.targets", "cohorts.qc_severity"),
})
# --- end protocol-v6.0.1 (M&Ms-2) ---------------------------------------------------------------------------------------
BUNDLE_STAGES = ("validate-cohort", "build-semantics", "extract-formal", "fit-transition", "fit-predictor", "discretize",
                 "check-semantic-eligibility", "induce-rules", "gate-rules")
COUNT_CHECK_WAIVER = "synthetic_fixture_counts"   # cohort-size checks cannot hold on a synthetic fixture


def bundle_sections() -> tuple[str, ...]:
    return tuple(sorted({s for stage in BUNDLE_STAGES for s in STAGE_CONFIG_SECTIONS[stage]}))


class ProtocolFreezeError(RuntimeError):
    """An M&Ms endpoint stage was requested before the protocol freeze (M0)."""


class CohortValidationError(RuntimeError):
    def __init__(self, report: dict[str, Any]) -> None:
        failed = [c["check"] for c in report.get("checks", []) if not c.get("passed")]
        super().__init__(f"cohort validation failed: {failed[:10]}")
        self.report = report


class BundleError(RuntimeError):
    """The analysis bundle is incomplete, already frozen with other content, or changed after the freeze."""


class StageRecordLockedError(BundleError):
    """Once-only guard: a recorded stage of a frozen run would have to re-execute (exit 7; a new run id is required)."""


# ----------------------------------------------------------------------------- context and guards


@dataclass
class StageContext:
    study: StudyConfig
    run: RunRoot | None = None
    data_root: Path = DATA_ROOT
    n_jobs: int | None = None
    source_readers: int | None = None
    force: bool = False
    allow_unfrozen_synthetic: bool = False
    sources: dict[str, Path] = field(default_factory=dict)
    staged_root_override: Path | None = None
    source_manifests: dict[str, pd.DataFrame] = field(default_factory=dict)   # overrides of the tracked manifests (tests)
    mms_metadata: list | None = None                                          # override of the metadata manifest (tests)

    @property
    def staged_root(self) -> Path:
        return Path(self.staged_root_override) if self.staged_root_override else Path(self.data_root) / "staged"

    @property
    def manifests_dir(self) -> Path:
        return self.staged_root.parent / "manifests"

    @property
    def jobs(self) -> int:
        return resolve_n_jobs(self.n_jobs)

    def require_run(self, stage: str) -> RunRoot:
        if self.run is None:
            raise ValueError(f"{stage} needs --run-id")
        return self.run

    def source_root(self, dataset: str) -> Path:
        key = cohort_key(dataset)
        return resolve_source_root(dataset, self.sources.get(key))

    def _fixture_manifest(self, dataset: str, *, metadata: bool = False) -> Path | None:
        """A synthetic fixture root's own manifest (roots with the SYNTHETIC_FIXTURE marker only)."""

        try:
            root = self.source_root(dataset)
        except (KeyError, OSError, ValueError):
            return None
        return synthetic_manifest_path(root, metadata=metadata)

    def source_manifest(self, dataset: str) -> pd.DataFrame:
        ds = canonical_dataset(dataset)
        if ds in self.source_manifests:
            return self.source_manifests[ds]
        if ds == "MMS2":     # the extraction manifest of the M&Ms-2 source area (tmmri.ingest.mms2)
            from .ingest.mms2 import load_file_manifest

            return load_file_manifest(self.source_root(ds))
        fixture = self._fixture_manifest(ds)
        if fixture is not None:
            return load_acdc_manifest(fixture) if ds == "ACDC" else load_mms_manifest(fixture)
        return load_acdc_manifest() if ds == "ACDC" else load_mms_manifest()

    def source_manifest_sha256(self, dataset: str) -> str | None:
        ds = canonical_dataset(dataset)
        if ds in self.source_manifests:
            return sha256_json(self.source_manifests[ds].astype(str).to_dict("records"))
        if ds == "MMS2":
            from .ingest.mms2 import file_manifest_path

            return hash_path(file_manifest_path(self.source_root(ds)))
        fixture = self._fixture_manifest(ds)
        return hash_path(fixture if fixture is not None else _tracked_manifest_path(ds))

    def mms_metadata_entries(self) -> list | None:
        """The M&Ms information-CSV manifest: test override, a fixture root's own manifest, else the tracked one."""

        if self.mms_metadata is not None:
            return self.mms_metadata
        fixture = self._fixture_manifest("MMS", metadata=True)
        return load_mms_metadata_manifest(fixture) if fixture is not None else None

    def synthetic_sources(self) -> bool:
        """True when every available source root carries the SYNTHETIC_FIXTURE marker (tests, benchmarks)."""

        roots = [self.source_root(d) for d in ("acdc", "mms")]
        present = [r for r in roots if r.is_dir()]
        return bool(present) and all(is_synthetic_fixture(r) for r in present)


def make_context(study_path: str | Path = "configs/study_v6.yaml", run_id: str | None = None, *,
                 artifact_root: str | Path | None = None, data_root: str | Path | None = None, **kwargs: Any) -> StageContext:
    study = load_study(study_path)
    run = RunRoot(run_id, artifact_root, subfolders=study.study.get("artifact_subfolders")) if run_id else None
    return StageContext(study=study, run=run, data_root=Path(data_root) if data_root else DATA_ROOT, **kwargs)


SIGNATURE_BLOCKS = ("SSH", "PGP")           # git tag -s: an SSH signature (gpg.format=ssh, as protocol-v6.0.0) or OpenPGP


def tag_has_signature(text: str) -> bool:
    """True when the raw tag object (``git cat-file tag``) carries a complete SSH or OpenPGP signature block."""

    return any(f"-----BEGIN {kind} SIGNATURE-----" in text and f"-----END {kind} SIGNATURE-----" in text for kind in SIGNATURE_BLOCKS)


def protocol_frozen(tag: str = "protocol-v6.0.0", root: Path = PROJECT_ROOT, *, signed: bool = False) -> bool:
    """True when ``tag`` is an annotated tag and HEAD contains it (the M0 freeze).

    ``signed=True`` (the M&Ms-2 lock, protocol-v6.0.1) also requires the tag object to carry a signature block, so an
    unsigned ``git tag -a`` never unlocks.  The signature is not verified cryptographically here (the containers hold no
    allowed-signers file); ``git verify-tag`` against the owner's key remains the owner's check at signing."""

    try:
        kind = subprocess.run(["git", "--no-optional-locks", "cat-file", "-t", tag], cwd=root, capture_output=True, text=True, timeout=30)
        if kind.returncode != 0 or kind.stdout.strip() != "tag":
            return False
        if signed:
            body = subprocess.run(["git", "--no-optional-locks", "cat-file", "tag", tag], cwd=root, capture_output=True, text=True, timeout=30)
            if body.returncode != 0 or not tag_has_signature(body.stdout):
                return False
        contained = subprocess.run(["git", "--no-optional-locks", "merge-base", "--is-ancestor", tag, "HEAD"], cwd=root,
                                   capture_output=True, timeout=30)
        return contained.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _staging_manifest(ctx: StageContext, dataset: str) -> dict[str, Any] | None:
    from .ingest.staging import load_staging_manifest

    return load_staging_manifest(ctx.manifests_dir, dataset)


def require_mms_unlocked(ctx: StageContext, stage: str) -> str:
    """Refuse an M&Ms endpoint stage before M0 unless the staged data are a synthetic fixture and the override is set."""

    tag = ctx.study.freeze_tag
    if protocol_frozen(tag):
        return "protocol_frozen"
    if ctx.allow_unfrozen_synthetic:
        manifest = _staging_manifest(ctx, "MMS") or {}
        source = ctx.sources.get("mms")
        if manifest.get("synthetic_fixture") is True and (source is None or is_synthetic_fixture(source)):
            return "synthetic_override"
        raise ProtocolFreezeError(f"{stage}: --allow-unfrozen-synthetic is accepted only for synthetic fixtures "
                                  "(staged M&Ms data from a source root with the SYNTHETIC_FIXTURE marker)")
    raise ProtocolFreezeError(f"{stage}: no endpoint computation on M&Ms before the protocol freeze M0 (annotated tag "
                              f"{tag} absent or not contained in HEAD); only stage-sources and validate-cohort are allowed")


def mms2_lock_tag(study: StudyConfig) -> str:
    """The amendment tag that unlocks M&Ms-2 (``configs/cohorts.yaml`` ``mms2.lock_tag``; ``protocol-v6.0.1``)."""

    return str((study.cohorts.get("mms2") or {}).get("lock_tag") or "protocol-v6.0.1")


def require_mms2_unlocked(ctx: StageContext, stage: str, *, synthetic: bool | None = None) -> str:
    """The M&Ms-2 lock (protocol-v6.0.1), mirroring :func:`require_mms_unlocked`: refuse (exit 3) unless the annotated
    tag ``mms2.lock_tag`` exists, carries a signature (``protocol_frozen(..., signed=True)``; the owner signs it) and
    HEAD descends from it.  ``--allow-unfrozen-synthetic`` is accepted only for staged M&Ms-2 data from a synthetic
    fixture (``synthetic`` overrides that test, for extract-mms2 on a synthetic archive)."""

    tag = mms2_lock_tag(ctx.study)
    if protocol_frozen(tag, signed=True):
        return "protocol_frozen"
    if ctx.allow_unfrozen_synthetic:
        if synthetic is None:
            manifest = _staging_manifest(ctx, "MMS2") or {}
            source = ctx.sources.get("mms2")
            synthetic = manifest.get("synthetic_fixture") is True and (source is None or is_synthetic_fixture(source))
        if synthetic:
            return "synthetic_override"
        raise ProtocolFreezeError(f"{stage}: --allow-unfrozen-synthetic is accepted only for synthetic fixtures "
                                  "(M&Ms-2 data from a source root with the SYNTHETIC_FIXTURE marker)")
    raise ProtocolFreezeError(f"{stage}: no M&Ms-2 extraction or computation before the protocol-v6.0.1 amendment (signed annotated tag "
                              f"{tag} absent, unsigned, or HEAD does not descend from it)")


MMS2_VALIDATION_STAGE = "validate-cohort:MMS2"


def require_mms2_validated(ctx: StageContext, requester: str) -> Path:
    """protocol-v6.0.1 (SPECIFICATION.md 4.6 and 14.6 step 6): every M&Ms-2 stage after ``validate-cohort --dataset mms2``
    needs its stage record, a passed report and the recorded contracts unchanged; the decoded-voxel duplicate exclusion
    against ACDC and M&Ms-1 is a leakage control, so a failed or missing validation stops the later stages.  Returns the
    record path (a stage input)."""

    run = ctx.require_run(requester)
    record = run.read_record(MMS2_VALIDATION_STAGE)
    if record is None:
        raise BundleError(f"{requester}: run validate-cohort --dataset mms2 --run-id {run.run_id} first; M&Ms-2 stages need its passed "
                          f"record (.stages/{run.record_path(MMS2_VALIDATION_STAGE).name}); a failed validation is never recorded")
    contracts = _contracts_dir(ctx)
    report_path = contracts / "cohort_validation_MMS2.json"
    for path in (report_path, contracts / "cohort_MMS2.parquet", contracts / "split_manifest_MMS2.parquet"):
        if run.portable(path) not in record.outputs or hash_path(path) is None:
            raise BundleError(f"{requester}: {run.portable(path)} is not the recorded output of {MMS2_VALIDATION_STAGE}; use a new run id")
        _check_recorded_output(run, record, path, requester)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("passed") is not True:
        raise CohortValidationError(report)
    return run.record_path(MMS2_VALIDATION_STAGE)


# --- protocol-v6.0.1 SPECIFICATION.md 14.6 (D33): freeze order against the M&Ms-2 extraction record
MMS2_EXTRACTION_MANIFEST = "extraction_MMS2.json"       # <data-root>/manifests/, written by extract-mms2
MMS2_FREEZE_STAGES = {"F8": ("fit-cart-surrogate",), "D20": ("fit-deep-encoder",), "F9": ("freeze-f9",),
                      "F9b": ("fit-probes", "freeze-f9"), "S1-S4": ("fit-trust",)}
LATE_FREEZE = {"evidence": "post_hoc", "label": "post hoc", "reason": "FROZEN_AFTER_MMS2_EXTRACTION"}


def parse_utc(value: Any) -> Any:
    """``datetime`` (UTC) of an ISO-8601 record time, or None."""

    from datetime import datetime, timezone

    try:
        stamp = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=timezone.utc)


def mms2_extraction_times(ctx: StageContext) -> dict[str, Any]:
    """The M&Ms-2 extraction records and the earliest recorded extraction start: the receipt of the source area the staged
    data come from (``MMS2_EXTRACTION_RECEIPT.json``) and ``<data-root>/manifests/extraction_MMS2.json``."""

    from .ingest.mms2 import RECEIPT_JSON

    found: dict[str, Any] = {}
    starts = []
    for label, path in (("source_area_receipt", ctx.source_root("mms2") / RECEIPT_JSON),
                        ("data_root_manifest", ctx.manifests_dir / MMS2_EXTRACTION_MANIFEST)):
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        found[label] = {"file": Path(path).name, **{k: doc.get(k) for k in ("action", "extraction_started_utc", "first_extraction_started_utc")}}
        starts += [s for s in (parse_utc(doc.get(k)) for k in ("first_extraction_started_utc", "extraction_started_utc")) if s is not None]
    first = min(starts) if starts else None
    return {"records": found, "exists": bool(found), "first_extraction_started_utc": first.isoformat() if first is not None else None}


def mms2_freeze_order(ctx: StageContext, bundles: Iterable[str]) -> dict[str, Any]:
    """SPECIFICATION.md 14.6 (D33) "late freeze": a bundle (F8, D20, F9, F9b, S1-S4; :data:`MMS2_FREEZE_STAGES`) has a
    registered ``mms2_external`` result only when each of its freeze records was finished before the first M&Ms-2
    extraction started.  A record finished at most ``started_utc + seconds + 1 s`` (second resolution); a missing freeze
    record or a missing or undated extraction record cannot show the order and counts as late.  Returns the evidence
    label of the result (``prospective``, or ``post hoc`` with :data:`LATE_FREEZE`) and the evidence."""

    from datetime import timedelta

    run = ctx.require_run("mms2-freeze-order")
    extraction = mms2_extraction_times(ctx)
    start = parse_utc(extraction["first_extraction_started_utc"])
    out: dict[str, Any] = {"rule": "SPECIFICATION.md 14.6 (D33): freeze records dated before the M&Ms-2 extraction record; a later "
                                   "freeze has no registered mms2_external result and is labelled post hoc",
                           "extraction": extraction, "bundles": {}}
    for bundle in bundles:
        entries = []
        for name in MMS2_FREEZE_STAGES[bundle]:
            record = run.read_record(name)
            log = (record.run_log if record is not None else {}) or {}
            began = parse_utc(log.get("started_utc"))
            finished = began + timedelta(seconds=float(log.get("seconds") or 0.0) + 1.001) if began is not None else None
            on_time = start is not None and finished is not None and finished <= start
            entries.append({"stage": name, "recorded": record is not None, "started_utc": log.get("started_utc"),
                            "finished_utc_upper_bound": finished.isoformat() if finished is not None else None, "before_extraction": on_time})
        out["bundles"][bundle] = {"freezes": entries, "before_extraction": bool(entries) and all(e["before_extraction"] for e in entries)}
    out["before_extraction"] = all(b["before_extraction"] for b in out["bundles"].values())
    out.update({"evidence": "prospective", "label": "prospective", "reason": None} if out["before_extraction"] else LATE_FREEZE)
    return out


def methods_api(module: str, name: str) -> Callable[..., Any]:
    """Import a METHODS function lazily; raise :class:`NotImplementedError` naming it when absent."""

    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name and module.startswith(exc.name) or exc.name == module:
            raise NotImplementedError(f"{module}.{name} is not implemented yet (METHODS API, INTERFACES_PHASE2.md)") from exc
        raise
    fn = getattr(mod, name, None)
    if fn is None:
        raise NotImplementedError(f"{module}.{name} is not implemented yet (METHODS API, INTERFACES_PHASE2.md)")
    return fn


def _require_not_frozen(run: RunRoot, stage: str) -> None:
    if run.frozen_path.is_file():
        raise BundleError(f"{stage}: the analysis bundle of run {run.run_id} is frozen; fitting stages need a new run id")


def _write_table(frame: pd.DataFrame, path: Path, csv: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(tmp, engine="pyarrow", index=False)
    tmp.replace(path)
    if csv:
        frame.to_csv(path.with_suffix(".csv"), index=False, lineterminator="\n")
    return path


def _tracked_manifest_path(dataset: str) -> Path:
    return ACDC_RELEASE_MANIFEST if canonical_dataset(dataset) == "ACDC" else MMS_FULL_MANIFEST


# ----------------------------------------------------------------------------- stage-sources


def stage_sources(ctx: StageContext, dataset: str, *, verify_only: bool = False, cases: Iterable[str] | None = None):
    """Verify-and-stage one dataset into ``<staged_root>/<DATASET>/`` (allowed for M&Ms before M0; M&Ms-2 only after
    the protocol-v6.0.1 tag, SPECIFICATION.md 4.6)."""

    from .ingest.staging import stage_dataset

    ds = canonical_dataset(dataset)
    root = ctx.source_root(dataset)
    if ds == "MMS2":                          # protocol-v6.0.1: verification and staging are locked too (exit 3)
        require_mms2_unlocked(ctx, "stage-sources --dataset mms2", synthetic=is_synthetic_fixture(root))
    cohort = None
    if ds == "MMS2" and not verify_only:     # M&Ms-2 eligibility from the external CSV copy (cohorts.yaml mms2 contract)
        from .ingest.mms2 import mms2_cohort_table

        cohort = mms2_cohort_table(root, ctx.source_manifest(ds), cohort_cfg=ctx.study.cohorts["mms2"])
    report = stage_dataset(ds, root, ctx.staged_root, ctx.source_manifest(ds), n_jobs=ctx.jobs, verify_only=verify_only,
                           cases=list(cases) if cases else None, force=ctx.force,
                           source_readers=resolve_source_readers(ctx.source_readers), manifests_dir=ctx.manifests_dir,
                           source_manifest_sha256=ctx.source_manifest_sha256(ds), metadata=ctx.mms_metadata_entries(), cohort=cohort)
    if ctx.run is not None:
        ctx.run.set_status(f"stage-sources:{ds}", "BUILT" if report.ok else "BLOCKED_DATA_ACCESS", action=staging_action(report),
                           failures=len(report.failures), attempted=report.attempted)
        ctx.run.log({"stage": f"stage-sources:{ds}", "actions": report.actions, "seconds": report.seconds})
    return report


def staging_action(report: Any) -> str:
    """``SKIPPED`` when every staged case already matched its source hashes (nothing written), else ``RAN``."""

    return "RAN" if report.verify_only or report.failures or report.actions.get("written") else "SKIPPED"


# ----------------------------------------------------------------------------- validate-cohort


COUNT_CHECKS = frozenset({
    "acdc_dev_size", "acdc_test_size", "acdc_patients_per_fold", "acdc_patients_per_class_per_fold", "mms_release_cases",
    "mms_release_splits", "mms_external_size", "mms_external_by_vendor", "mms_external_by_release_split", "mms_exclusions",
    "mms_duplicate_pairs", "mms100_members", "mms100_by_release_split", "mms100_by_vendor", "mms_never_analysed_size",
    "mms_never_analysed_by_vendor", "mms_never_analysed_by_release_split", "mms_refit_train_size", "mms_refit_train_by_vendor",
    "mms_refit_eval_size", "mms_refit_eval_by_vendor", "partition_cluster_counts",
})


def _is_count_check(name: str) -> bool:
    return name in COUNT_CHECKS or name.endswith("_class_counts")


def _check(checks: list[dict[str, Any]], name: str, expected: Any, observed: Any) -> None:
    checks.append({"check": name, "expected": expected, "observed": observed, "passed": expected == observed})


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts().sort_index().items()}


def _contracts_dir(ctx: StageContext) -> Path:
    return ctx.run.sub("contracts") if ctx.run is not None else ctx.manifests_dir / "cohort"


def build_cohorts(ctx: StageContext) -> dict[str, pd.DataFrame | None]:
    from .ingest.acdc import acdc_cohort_table
    from .ingest.mms import mms_cohort_table

    out: dict[str, pd.DataFrame | None] = {"ACDC": None, "MMS": None}
    acdc_root, mms_root = ctx.source_root("acdc"), ctx.source_root("mms")
    if acdc_root.is_dir():
        out["ACDC"] = acdc_cohort_table(acdc_root, ctx.source_manifest("ACDC"))
    if mms_root.is_dir():
        out["MMS"] = mms_cohort_table(mms_root, ctx.source_manifest("MMS"), registry=ctx.study.registry, metadata=ctx.mms_metadata_entries())
    return out


def validate_cohort(ctx: StageContext, *, strict: bool = False, cohorts: Mapping[str, pd.DataFrame | None] | None = None) -> dict[str, Any]:
    """Cohort tables, expected counts (``configs/``), staging cross-checks and the v6 split manifest.

    With a run id the outputs go to ``<run>/contracts/`` through a content-addressed stage record (inputs: the
    staging manifests and the source-manifest hashes); without one to ``<data-root>/manifests/cohort/``.  With
    ``--allow-unfrozen-synthetic`` and synthetic fixture sources only, the cohort-size checks (which cannot hold on a
    fixture) are recorded as failed but waived (``synthetic_fixture_counts``); every structural check stays strict.
    """

    waive = bool(ctx.allow_unfrozen_synthetic and ctx.synthetic_sources())
    if ctx.run is None or cohorts is not None:
        return _validate_cohort_checks(ctx, strict=strict, cohorts=cohorts, waive_counts=waive)
    from .ingest.staging import manifest_path_for

    out_dir = _contracts_dir(ctx)
    available = {ds: ctx.source_root(cohort_key(ds)).is_dir() for ds in ("ACDC", "MMS")}
    outputs = [out_dir / "cohort_validation.json"]
    if available["ACDC"]:
        outputs += [out_dir / "cohort_ACDC.parquet", out_dir / "inner_folds.parquet"]
    if available["MMS"]:
        outputs.append(out_dir / "cohort_MMS.parquet")
    if any(available.values()):
        outputs.append(out_dir / "split_manifest.parquet")
    inputs = [p for p in (manifest_path_for(ctx.manifests_dir, ds) for ds in ("ACDC", "MMS")) if p.is_file()]
    params = {"strict": bool(strict), "synthetic_count_waiver": waive,
              "source_manifest_sha256": {ds: ctx.source_manifest_sha256(ds) for ds, ok in available.items() if ok},
              "mms_metadata": [getattr(e, "sha256", None) for e in (ctx.mms_metadata_entries() or [])]}
    holder: dict[str, Any] = {}

    def compute() -> dict[str, Any]:
        holder["report"] = _validate_cohort_checks(ctx, strict=strict, cohorts=None, waive_counts=waive)
        return {"passed": holder["report"]["passed"]}

    _stage(ctx, "validate-cohort", compute, inputs, outputs, params, fitting=True)
    return holder.get("report") or json.loads((out_dir / "cohort_validation.json").read_text(encoding="utf-8"))


def _validate_cohort_checks(ctx: StageContext, *, strict: bool, cohorts: Mapping[str, pd.DataFrame | None] | None,
                            waive_counts: bool) -> dict[str, Any]:
    from .splits import (acdc_all_inner_folds_v6, acdc_outer_folds_v6, apply_duplicate_declarations, build_split_manifest_v6,
                         declared_duplicate_findings, detect_image_duplicates, duplicate_action_violations, duplicate_image_pairs,
                         fold_balance, partition_cluster_report)

    study = ctx.study
    cohorts = dict(cohorts) if cohorts is not None else build_cohorts(ctx)
    checks: list[dict[str, Any]] = []
    sizes = study.study.get("partition_sizes", {})
    # Declared duplicate images (configs/cohorts.yaml duplicate_image_pairs): exclusions and shared cluster ids.
    acdc = apply_duplicate_declarations(cohorts.get("ACDC"), "ACDC", study.cohorts)
    mms = apply_duplicate_declarations(cohorts.get("MMS"), "MMS", study.cohorts)
    out_dir = _contracts_dir(ctx)
    outputs: dict[str, str] = {}
    if acdc is None:
        _check(checks, "acdc_source_available", True, False)
    else:
        present = set(acdc["case_id"].astype(str))
        declared = {str(p["exclude"]["case"]): str(p["reason_code"]) for p in duplicate_image_pairs(study.cohorts)
                    if str(p.get("cohort", "")).lower() == "acdc" and p.get("exclude") and str(p["exclude"]["case"]) in present}
        _check(checks, "acdc_ineligible_only_declared_duplicates", dict(sorted(declared.items())),
               {str(r.case_id): str(r.exclusion_reason) for r in acdc.loc[~acdc["eligible_v6"]].sort_values("case_id").itertuples(index=False)})
        eligible = acdc[acdc["eligible_v6"]]
        _check(checks, "acdc_dev_size", sizes.get("acdc_dev"), int((eligible["partition"] == "acdc_dev").sum()))
        _check(checks, "acdc_test_size", sizes.get("acdc_test"), int((eligible["partition"] == "acdc_test").sum()))
        expected_classes = study.registry.get("label_tables", {}).get("Y_acdc", {}).get("counts", {})
        for partition, counts in expected_classes.items():
            observed = _counts(eligible.loc[eligible["partition"] == partition, "acdc_class"])
            _check(checks, f"{partition}_class_counts", {str(k): int(v) for k, v in counts.items()}, observed)
        _check(checks, "acdc_ed_es_zero_based", True, bool(((acdc["ed_frame_index"] == acdc["ed_frame_number"] - 1)
                                                           & (acdc["es_frame_index"] == acdc["es_frame_number"] - 1)).all()))
        outer = acdc_outer_folds_v6(acdc, study.splits)
        expected = study.splits["acdc"]["acdc_dev"]["outer_folds"]["expected"]
        balance = fold_balance(outer, "outer_fold", "acdc_class")
        _check(checks, "acdc_patients_per_fold", {str(k): int(v) for k, v in expected["patients_per_fold"].items()},
               {str(f): int(n) for f, n in balance.sum(axis=1).items()})
        _check(checks, "acdc_patients_per_class_per_fold",
               {str(c): [int(v) for v in counts] for c, counts in expected["patients_per_class_per_fold"].items()},
               {str(c): [int(v) for v in balance[c]] for c in balance.columns})
        _write_table(acdc, out_dir / "cohort_ACDC.parquet")
        _write_table(acdc_all_inner_folds_v6(outer, study.splits), out_dir / "inner_folds.parquet")
        outputs["cohort_ACDC"] = str(out_dir / "cohort_ACDC.parquet")
        outputs["inner_folds"] = str(out_dir / "inner_folds.parquet")
    if mms is None:
        _check(checks, "mms_source_available", True, False)
    else:
        cfg = study.cohorts["mms"]
        eligible = mms[mms["eligible_v6"]]
        _check(checks, "mms_release_cases", int(study.source_binding("mms")["release_files_observed"]["cases"]), int(len(mms)))
        _check(checks, "mms_csv_rows_match_manifest", True, bool(mms["vendor"].notna().all()))
        _check(checks, "mms_release_splits", {str(k): int(v) for k, v in cfg["release_splits"].items()}, _counts(mms["release_split"]))
        _check(checks, "mms_external_size", int(cfg["eligibility"]["expected_count"]), int(len(eligible)))
        _check(checks, "mms_external_by_vendor", {str(k): int(v) for k, v in cfg["eligibility"]["by_vendor"].items()}, _counts(eligible["vendor"]))
        _check(checks, "mms_external_by_release_split", {str(k): int(v) for k, v in cfg["eligibility"]["by_release_split"].items()},
               _counts(eligible["release_split"]))
        excluded = {str(e["code"]): int(e.get("cases", 0)) for e in cfg.get("exclusions", []) if "cases" in e}
        _check(checks, "mms_exclusions", excluded, _counts(mms.loc[~mms["eligible_v6"], "exclusion_reason"].astype(str)))
        pairs = sorted(sorted(p) for p in study.splits["duplicate_control"]["known_duplicates"])
        observed_pairs = sorted({tuple(sorted((a, b))) for a, b in mms.dropna(subset=["duplicate_of"])[["case_id", "duplicate_of"]].itertuples(index=False)})
        _check(checks, "mms_duplicate_pairs", [list(p) for p in pairs], [list(p) for p in observed_pairs])
        _check(checks, "mms_duplicates_excluded", True, bool((~mms.loc[mms["duplicate_of"].notna(), "eligible_v6"]).all()))
        member_cfg = cfg["mms100_member"]
        members = mms[mms["mms100_member"]]
        _check(checks, "mms100_members", int(member_cfg["expected_members"]), int(len(members)))
        _check(checks, "mms100_by_release_split", {str(k): int(v) for k, v in member_cfg["by_release_split"].items()}, _counts(members["release_split"]))
        _check(checks, "mms100_by_vendor", {str(k): int(v) for k, v in member_cfg["by_vendor"].items()}, _counts(members["vendor"]))
        _check(checks, "mms100_members_eligible", True, bool(members["eligible_v6"].all()))
        never = study.splits["mms"]["mms_external_never_analysed"]["expected"]
        na = mms[mms["never_analysed"]]
        _check(checks, "mms_never_analysed_size", int(never["total"]), int(len(na)))
        _check(checks, "mms_never_analysed_by_vendor", {str(k): int(v) for k, v in never["by_vendor"].items()}, _counts(na["vendor"]))
        _check(checks, "mms_never_analysed_by_release_split", {str(k): int(v) for k, v in never["by_release_split"].items()}, _counts(na["release_split"]))
        for part in ("mms_refit_train", "mms_refit_eval"):
            exp = study.splits["refit_arm"][part]["expected"]
            sub = mms[mms["refit_partition"] == part]
            _check(checks, f"{part}_size", int(exp["total"]), int(len(sub)))
            if "by_vendor" in exp:
                _check(checks, f"{part}_by_vendor", {str(k): int(v) for k, v in exp["by_vendor"].items()}, _counts(sub["vendor"]))
        vc = {str(k): [int(c) for c in v["centres"]] for k, v in cfg["vendor_centre_map"].items()}
        bad_vc = [r.case_id for r in mms.itertuples(index=False) if pd.notna(r.vendor) and int(r.centre) not in vc.get(str(r.vendor), [])]
        _check(checks, "mms_vendor_centre_map", [], bad_vc[:10])
        _write_table(mms, out_dir / "cohort_MMS.parquet")
        outputs["cohort_MMS"] = str(out_dir / "cohort_MMS.parquet")
    for ds, table in (("ACDC", acdc), ("MMS", mms)):
        manifest = _staging_manifest(ctx, ds)
        if table is None:
            continue
        expected_ids = sorted(table.loc[table["eligible_v6"], "case_id"])
        if manifest is None:
            _check(checks, f"{ds.lower()}_staged", expected_ids, "not_staged")
            continue
        staged = manifest.get("cases", {})
        _check(checks, f"{ds.lower()}_staged", expected_ids, sorted(c for c in staged if c in set(expected_ids)))
        tracked = ({f"{r.relpath}": r.sha256 for r in ctx.source_manifest("ACDC").itertuples(index=False)} if ds == "ACDC"
                   else {**dict(zip(table["image_relpath"], table["image_sha256"])), **dict(zip(table["mask_relpath"], table["mask_sha256"]))})
        mismatched = sorted(cid for cid, rec in staged.items() for src in rec.get("sources", {}).values() if tracked.get(src["relpath"]) != src["sha256"])
        _check(checks, f"{ds.lower()}_staged_sources_match_tracked_manifest", [], mismatched[:10])
        frames_ok = sorted(cid for cid, rec in staged.items() if cid in set(expected_ids)
                           and (rec.get("ed_frame_index"), rec.get("es_frame_index")) != tuple(
                               int(v) for v in table.loc[table["case_id"] == cid, ["ed_frame_index", "es_frame_index"]].iloc[0]))
        _check(checks, f"{ds.lower()}_staged_frames_match_metadata", [], frames_ok[:10])
        if ds == "MMS":
            not_exact = sorted(cid for cid, rec in staged.items() if rec.get("annotation_matches_metadata") is False)
            _check(checks, "mms_mask_annotated_exactly_on_ed_es", [], not_exact[:10])
    # Decoded-voxel duplicates of the staged ED/ES images: every detected pair must be declared (configs/cohorts.yaml
    # duplicate_image_pairs) with the detected frames, and its action must be applied (below).
    staged_all = [rec for ds in ("ACDC", "MMS") for rec in ((_staging_manifest(ctx, ds) or {}).get("cases", {}).values())]
    findings = declared_duplicate_findings(detect_image_duplicates(staged_all), {str(r["sample_id"]) for r in staged_all}, study.cohorts)
    _check(checks, "decoded_voxel_duplicates_within_and_across_cohorts", [], findings["undeclared"][:10])
    _check(checks, "declared_duplicate_image_pairs_detected", [], findings["frame_mismatch"][:10])
    cluster_report: dict[str, Any] | None = None
    if acdc is not None or mms is not None:
        split_manifest = build_split_manifest_v6(acdc, mms, study.splits)
        _write_table(split_manifest, out_dir / "split_manifest.parquet")
        outputs["split_manifest"] = str(out_dir / "split_manifest.parquet")
        overlap = set(split_manifest.loc[split_manifest["dataset"] == "ACDC", "sample_id"]) & set(split_manifest.loc[split_manifest["dataset"] == "MMS", "sample_id"])
        _check(checks, "sample_id_disjoint_across_cohorts", [], sorted(overlap))
        _check(checks, "declared_duplicate_actions_applied", [], duplicate_action_violations({"ACDC": acdc, "MMS": mms}, split_manifest, study.cohorts))
        cluster_report = partition_cluster_report(split_manifest, mms, list(study.cohorts["mms"].get("strata", [])),
                                                  study.splits["refit_arm"].get("eval_strata"))
        expected_clusters = (study.splits.get("duplicate_control") or {}).get("expected_clusters") or {}
        available = [p for p in expected_clusters if (acdc is not None or not p.startswith("acdc")) and (mms is not None or p.startswith("acdc"))]
        _check(checks, "partition_cluster_counts", {p: int(expected_clusters[p]) for p in available},
               {p: int(cluster_report["partitions"][p]["clusters"]) for p in available})
    if waive_counts:
        for c in checks:
            if not c["passed"] and _is_count_check(c["check"]):
                c["waived"] = COUNT_CHECK_WAIVER
    report = {"schema": "tmmri.cohort_validation/1", "passed": all(c["passed"] or c.get("waived") for c in checks), "checks": checks,
              "synthetic_count_waiver": bool(waive_counts), "outputs": {k: Path(v).name for k, v in outputs.items()},
              "duplicate_image_pairs": findings["declared"], "cluster_counts": cluster_report,
              "config_hash": study.sections_hash(STAGE_CONFIG_SECTIONS["validate-cohort"])}
    atomic_write_json(out_dir / "cohort_validation.json", report)
    if strict and not report["passed"]:
        raise CohortValidationError(report)
    return report


# ----------------------------------------------------------------------------- shared loaders


def _contracts(ctx: StageContext) -> Path:
    return ctx.require_run("stage").path / "contracts"


def _cohort(ctx: StageContext, ds: str) -> pd.DataFrame:
    path = _contracts(ctx) / f"cohort_{ds}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: run validate-cohort --run-id {ctx.run.run_id} first")
    return pd.read_parquet(path)


def _split_manifest(ctx: StageContext, ds: str | None = None) -> pd.DataFrame:
    """``contracts/split_manifest.parquet`` (ACDC and M&Ms-1; part of the bundle), or for ``ds="MMS2"`` the separate
    ``contracts/split_manifest_MMS2.parquet`` written by ``validate-cohort --dataset mms2`` (never in the bundle)."""

    name = "split_manifest_MMS2.parquet" if ds == "MMS2" else "split_manifest.parquet"
    path = _contracts(ctx) / name
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: run validate-cohort{' --dataset mms2' if ds == 'MMS2' else ''} first")
    return pd.read_parquet(path)


def split_manifest_for(ctx: StageContext, partition: str | None = None, *, dataset: str | None = None) -> pd.DataFrame:
    """protocol-v6.0.1 (integration): the split manifest that holds ``partition`` (or ``dataset``) rows.  M&Ms-2 rows
    live only in ``contracts/split_manifest_MMS2.parquet``; every other partition in ``contracts/split_manifest.parquet``."""

    if dataset == "MMS2" or str(partition or "").startswith("mms2"):
        return _split_manifest(ctx, "MMS2")
    return _split_manifest(ctx)


def _analysis_ids(ctx: StageContext, ds: str) -> list[str]:
    cohort = _cohort(ctx, ds)
    mask = cohort[{"ACDC": "eligible_v6", "MMS": "mms_external", "MMS2": MMS2_SPLIT}[ds]]
    return sorted(cohort.loc[mask.astype(bool), "case_id"].astype(str))


def _staging_manifest_path(ctx: StageContext, ds: str) -> Path:
    from .ingest.staging import manifest_path_for

    path = manifest_path_for(ctx.manifests_dir, ds)
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: run stage-sources --dataset {cohort_key(ds)} first")
    return path


def _require_staged(ctx: StageContext, ds: str, case_ids: list[str]) -> None:
    manifest = json.loads(_staging_manifest_path(ctx, ds).read_text(encoding="utf-8"))
    missing = sorted(set(case_ids) - set(manifest.get("cases", {})))
    if missing:
        raise FileNotFoundError(f"{len(missing)} {ds} cases are not staged (e.g. {missing[:3]}); run stage-sources first")


def _stage(ctx: StageContext, name: str, fn: Callable[[], Any], inputs: list[Path], outputs: list[Path],
           params: dict | None = None, status: str = "BUILT", fitting: bool = False):
    """Content-addressed stage call; ``fitting`` stages refuse to rerun in a run whose bundle is frozen.

    The record's configuration hash covers only the sections of :data:`STAGE_CONFIG_SECTIONS` for the stage (the
    part of ``name`` before ``:``), which are also listed in the record parameters.  Once-only guard: in a frozen run a
    stage with a record may only be skipped (:func:`once_only_guard`).
    """

    run = ctx.require_run(name)
    sections = STAGE_CONFIG_SECTIONS.get(name) or STAGE_CONFIG_SECTIONS[name.split(":", 1)[0]]

    def guarded() -> Any:
        if fitting:
            _require_not_frozen(run, name)
        return fn()

    return run_stage(run, name, guarded, inputs, outputs, config_hash=ctx.study.sections_hash(sections),
                     params={**(params or {}), "config_sections": list(sections)}, force=ctx.force, n_jobs=ctx.jobs, status=status,
                     rerun_guard=once_only_guard(run, name))


def once_only_guard(run: RunRoot, name: str) -> Callable[[str], None]:
    """The ``rerun_guard`` of :func:`tmmri.runs.run_stage` for ``name``: in a run with ``.stages/FROZEN.json`` a stage
    whose record exists may only SKIP; any difference refuses with :class:`StageRecordLockedError` (exit 7)."""

    def guard(difference: str) -> None:
        if run.frozen_path.is_file():
            run.log({"stage": name, "action": "REFUSED_ONCE_ONLY", "difference": difference})
            raise StageRecordLockedError(
                f"{name}: refused by the once-only guard: run {run.run_id} is frozen (.stages/FROZEN.json) and {name} is already "
                f"recorded (.stages/{run.record_path(name).name}); the current call differs from that record in: {difference}; "
                f"a recorded stage of a frozen run may only be skipped. "
                f"The record and its outputs are unchanged; re-executing {name} needs a new run id.")

    return guard


# ----------------------------------------------------------------------------- extract-formal


def extract_formal(ctx: StageContext, dataset: str):
    """``formal/A_<DATASET>.parquet`` (57 descriptors) + ``A_<DATASET>_p6_128.parquet`` + failures + extractor record."""

    from .formal_features import FORMAL_V3_EXTRACTOR, FORMAL_V3_NAME, _matrix_checksum, _projection_matrix, extract_formal_table, formal_descriptor_names, p6_projection_table, p6_settings

    ds = canonical_dataset(dataset)
    if ds == "MMS":
        require_mms_unlocked(ctx, "extract-formal --dataset mms")
    elif ds == "MMS2":
        require_mms2_unlocked(ctx, "extract-formal --dataset mms2")
        require_mms2_validated(ctx, "extract-formal --dataset mms2")
    run = ctx.require_run("extract-formal")
    formal_cfg = ctx.study.formal
    case_ids = _analysis_ids(ctx, ds)
    _require_staged(ctx, ds, case_ids)
    out = run.sub("formal")
    paths = [out / f"A_{ds}.parquet", out / f"A_{ds}_p6_128.parquet", out / f"A_{ds}_failures.parquet", out / f"A_{ds}_extractor.json"]

    def compute() -> dict[str, Any]:
        table, failures = extract_formal_table(ds, case_ids, ctx.staged_root, n_jobs=ctx.jobs,
                                               label_map=ctx.study.label_map(cohort_key(ds)),
                                               expected_dimension=int(formal_cfg["primary"]["dimension"]))
        settings = p6_settings(formal_cfg)
        _write_table(table, paths[0])
        _write_table(p6_projection_table(table, **settings), paths[1])
        _write_table(failures, paths[2])
        names = formal_descriptor_names()
        atomic_write_json(paths[3], {
            "representation": FORMAL_V3_NAME, "extractor_id": FORMAL_V3_EXTRACTOR, "dimension": len(names), "columns": names,
            "dataset": ds, "rows": int(len(table)), "failures": int(len(failures)),
            "p6_projection": {**settings, "matrix_sha256": _matrix_checksum(_projection_matrix(len(names), settings["output_dimension"], settings["seed"]))},
            "mask_conditioned": bool(formal_cfg["primary"].get("mask_conditioned", True)), "standardization": "downstream, fit partition only",
        })
        return {"rows": int(len(table)), "failures": int(len(failures))}

    inputs = [_staging_manifest_path(ctx, ds), _contracts(ctx) / f"cohort_{ds}.parquet"]
    return _stage(ctx, f"extract-formal:{ds}", compute, inputs, paths, fitting=ds == "ACDC")


# ----------------------------------------------------------------------------- build-semantics (METHODS)


def release_metadata(dataset: str, row: Mapping[str, Any], partition: object) -> dict[str, Any]:
    """Release metadata that ``case_semantics_v6`` reads from ``case.meta`` (paths relative to the source mount)."""

    def clean(value: Any) -> Any:
        return None if value is None or (not isinstance(value, str) and pd.isna(value)) else value

    ds = canonical_dataset(dataset)
    if ds == "ACDC":
        paths = {"raw_ed_image_path": row.get("ed_image_relpath"), "raw_es_image_path": row.get("es_image_relpath"),
                 "raw_ed_mask_path": row.get("ed_mask_relpath"), "raw_es_mask_path": row.get("es_mask_relpath"),
                 "raw_image_path": row.get("ed_image_relpath"), "raw_mask_path": row.get("ed_mask_relpath")}
        extra = {"disease_label": row.get("acdc_class"), "scanner_vendor": None, "center_id": None}
    elif ds == "MMS2":       # four separate short-axis files; no frame index (phases from the file names)
        paths = {"raw_ed_image_path": row.get("ed_image_relpath"), "raw_es_image_path": row.get("es_image_relpath"),
                 "raw_ed_mask_path": row.get("ed_mask_relpath"), "raw_es_mask_path": row.get("es_mask_relpath"),
                 "raw_image_path": row.get("ed_image_relpath"), "raw_mask_path": row.get("ed_mask_relpath")}
        extra = {"disease_label": row.get("pathology"), "scanner_vendor": row.get("vendor_name"), "center_id": None}
    else:
        paths = {"raw_ed_image_path": row.get("image_relpath"), "raw_es_image_path": row.get("image_relpath"),
                 "raw_ed_mask_path": row.get("mask_relpath"), "raw_es_mask_path": row.get("mask_relpath"),
                 "raw_image_path": row.get("image_relpath"), "raw_mask_path": row.get("mask_relpath")}
        centre = clean(row.get("centre"))
        extra = {"disease_label": row.get("pathology"), "scanner_vendor": row.get("vendor_name"),
                 "center_id": None if centre is None else int(centre)}
    meta = {"patient_id": row.get("patient_id"), "subset_name": row.get("release_split"), "split": partition,
            "ed_frame_index": row.get("ed_frame_index"), "es_frame_index": row.get("es_frame_index"),
            "height_cm": row.get("height_cm"), "weight_kg": row.get("weight_kg"), **extra, **paths}
    out = {k: clean(v) for k, v in meta.items()}
    for key in ("ed_frame_index", "es_frame_index"):
        if out[key] is not None:
            out[key] = int(out[key])
    return out


def load_case_for_semantics(case_id: str, *, dataset: str, staged_root: str, label_map: dict[int, int],
                            metadata: Mapping[str, Mapping[str, Any]]):
    """Loader for ``build_semantics_table_v6``: the staged case with its release metadata merged into ``meta``."""

    from .ingest.staging import load_staged_case

    case = load_staged_case(dataset, case_id, staged_root, label_map=label_map)
    case.meta.update(dict(metadata[case_id]))
    case.meta.setdefault("slice_thickness_mm", case.meta.get("geometry", {}).get("slice_thickness_mm"))
    return case


def build_semantics(ctx: StageContext, dataset: str):
    """Typed B (``B_v6.0``) via ``tmmri.semantics.features.build_semantics_table_v6``; contract-validated.

    Robust-outlier references are fitted on ``acdc_dev`` only (``tmmri.semantics.qc.fit_outlier_reference``,
    written to ``qc/outlier_reference.json``) and applied unchanged to every partition
    (``tmmri.semantics.qc.apply_outlier_reference``); the M&Ms table reuses the frozen ACDC reference.
    """

    import functools

    from .semantics.contract import validate_typed_b

    ds = canonical_dataset(dataset)
    if ds == "MMS":
        require_mms_unlocked(ctx, "build-semantics --dataset mms")
    elif ds == "MMS2":
        require_mms2_unlocked(ctx, "build-semantics --dataset mms2")
        require_mms2_validated(ctx, "build-semantics --dataset mms2")
    run = ctx.require_run("build-semantics")
    build_table = methods_api("tmmri.semantics.features", "build_semantics_table_v6")
    fit_ref = methods_api("tmmri.semantics.qc", "fit_outlier_reference")
    apply_ref = methods_api("tmmri.semantics.qc", "apply_outlier_reference")
    cohort = _cohort(ctx, ds)
    case_ids = _analysis_ids(ctx, ds)
    _require_staged(ctx, ds, case_ids)
    split = _split_manifest(ctx, ds if ds == "MMS2" else None).set_index("sample_id")
    out = run.sub("semantics")
    target = out / f"B_{ds}.parquet"
    qdir = run.sub("qc")
    reference_path = qdir / "outlier_reference.json"
    review_path = qdir / ("manual_review.parquet" if ds == "ACDC" else f"manual_review_{ds}.parquet")
    panels_path = qdir / f"panels_{ds}.json"

    def compute() -> dict[str, Any]:
        rows = cohort.set_index("case_id")
        metadata = {cid: release_metadata(ds, {**rows.loc[cid].to_dict(), "case_id": cid}, split.loc[f"{ds}:{cid}", "partition"])
                    for cid in case_ids}
        loader = functools.partial(load_case_for_semantics, dataset=ds, staged_root=str(ctx.staged_root),
                                   label_map=ctx.study.label_map(cohort_key(ds)), metadata=metadata)
        frame = build_table((loader, case_ids), ctx.study.registry, ctx.jobs, cohorts=ctx.study.cohorts,
                            reason_codes=ctx.study.reason_codes)
        frame = frame.sort_values("sample_id").reset_index(drop=True)
        if ds == "ACDC":
            reference = fit_ref(frame, ctx.study.registry, ctx.study.cohorts, fit_mask=(frame["split"] == "acdc_dev").to_numpy(copy=True),
                                reason_codes=ctx.study.reason_codes)
            atomic_write_json(reference_path, reference)
        else:
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
        frame = apply_ref(frame, reference, ctx.study.cohorts, reason_codes=ctx.study.reason_codes)
        registry = ctx.study.registry
        if ds == "MMS2":     # protocol-v6.0.1: the mms2.typed_b_metadata overlay of the (unchanged) registry metadata schema
            from .ingest.mms2 import typed_b_registry

            registry = typed_b_registry(registry, ctx.study.cohorts["mms2"])
        validate_typed_b(frame, registry, cohorts_cfg=ctx.study.cohorts)
        _write_table(frame, target)
        n_panels = qc_panels(ctx, ds, frame, loader, qdir, review_path, panels_path)
        return {"rows": int(len(frame)), "qc_panels": n_panels}

    inputs = [_staging_manifest_path(ctx, ds), _contracts(ctx) / f"cohort_{ds}.parquet"]
    if ds == "MMS":
        inputs.append(reference_path)
    elif ds == "MMS2":       # the frozen acdc_dev outlier reference and the M&Ms-2 split manifest
        inputs += [reference_path, _contracts(ctx) / "split_manifest_MMS2.parquet"]
    outputs = [target, review_path, panels_path] + ([reference_path] if ds == "ACDC" else [])
    return _stage(ctx, f"build-semantics:{ds}", compute, inputs, outputs, fitting=ds == "ACDC")

def qc_panels(ctx: StageContext, ds: str, frame: pd.DataFrame, loader: Callable[[str], Any], qdir: Path, review_path: Path,
              manifest_path: Path) -> int:
    """QC panels and the manual review table (SPECIFICATION.md 12.4; ``configs/cohorts.yaml`` ``qc_panels``).

    Per partition of ``qc_panels.partitions[<DATASET>]`` (ACDC before M0; M&Ms only through the guarded
    ``build-semantics --dataset mms``): every critical or high QC case plus ``random_per_partition`` further cases
    ranked by ``SHA256(random_selection_salt + sample_id)``; each is rendered to ``qc/panels/<DATASET>/<case>.png``.
    ``qc/manual_review.parquet`` (ACDC; ``manual_review_MMS.parquet`` for M&Ms) lists every moderate-or-higher QC case
    and every panel case with ``adjudication_status`` at the configured default (``pending_review``).
    """

    import shutil

    from .semantics.manual_review import build_qc_review_v6
    from .semantics.visual_qc import render_qc_panel_v6, select_qc_panels_v6

    cfg = ctx.study.cohorts["qc_panels"]
    partitions = [str(p) for p in (cfg.get("partitions") or {}).get(ds, [])]
    if ds == "MMS2":         # protocol-v6.0.1: configs/cohorts.yaml mms2.qc_panels.partitions
        partitions = [str(p) for p in ((ctx.study.cohorts.get("mms2") or {}).get("qc_panels") or {}).get("partitions", [])]
    selection = select_qc_panels_v6(frame, ctx.study.cohorts, ctx.study.reason_codes, partitions)
    pdir = qdir / "panels" / ds
    if pdir.is_dir():
        shutil.rmtree(pdir)
    rows = frame.set_index(frame["sample_id"].astype(str))
    family = ctx.study.primary_family()
    files: dict[str, str] = {}
    entries: dict[str, Any] = {}
    for r in selection.itertuples(index=False):
        sid = str(r.sample_id)
        path = render_qc_panel_v6(loader(str(r.case_id)), {**rows.loc[sid].to_dict(), "sample_id": sid}, pdir / f"{r.case_id}.png", family)
        rel = path.relative_to(ctx.require_run("qc").path).as_posix()
        files[sid] = rel
        rank = None if r.selection_rank is None or pd.isna(r.selection_rank) else int(r.selection_rank)
        entries[sid] = {"file": rel, "sha256": hash_path(path), "partition": r.partition, "selection_reason": r.selection_reason,
                        "selection_rank": rank, "max_severity": r.max_severity}
    review = build_qc_review_v6(frame, selection, ctx.study.cohorts, ctx.study.reason_codes, files)
    _write_table(review, review_path, csv=True)
    atomic_write_json(manifest_path, {
        "schema": "tmmri.qc_panels/1", "dataset": ds, "partitions": partitions, "random_per_partition": int(cfg["random_per_partition"]),
        "include_all_high_and_critical": bool(cfg.get("include_all_high_and_critical", True)), "selection_rule": cfg["random_selection_rule"],
        "selection_salt": cfg["random_selection_salt"], "render": cfg.get("render"), "n_panels": len(entries), "panels": entries,
        "manual_review_table": review_path.name, "adjudication_default": cfg.get("adjudication_default"), "note": cfg.get("note")})
    return len(entries)


# ----------------------------------------------------------------------------- transition, predictor (METHODS)


def _critical_reasons(study: StudyConfig) -> set[str]:
    return {str(r) for r in study.cohorts["qc_severity"]["critical"]}


def critical_qc_mask(frame: pd.DataFrame, study: StudyConfig) -> pd.Series:
    """True for rows with a critical QC reason (excluded from every fit, ``QC_ABSTAIN`` at inference)."""

    critical = _critical_reasons(study)
    reasons = frame.get("qc_failure_reason", pd.Series(None, index=frame.index))
    return reasons.fillna("").map(lambda v: any(t.split(":", 1)[0].strip() in critical for t in str(v).split(";") if t.strip()))


def _dev_matrices(ctx: StageContext, representation: str = "formal") -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """(ids frame, A, B, outer folds) of the acdc_dev fit rows (critical QC rows removed), aligned by sample_id."""

    from .formal_features import formal_descriptor_names

    run = ctx.require_run("fit")
    split = _split_manifest(ctx)
    dev = split[split["partition"] == "acdc_dev"][["sample_id", "patient_id", "outer_fold"]]
    b = pd.read_parquet(run.path / "semantics" / "B_ACDC.parquet")
    b = b[~critical_qc_mask(b, ctx.study)]
    if representation == "p6":
        a = pd.read_parquet(run.path / "formal" / "A_ACDC_p6_128.parquet")
        a_cols = [c for c in a.columns if c.startswith("a_")]
    else:
        a = pd.read_parquet(run.path / "formal" / "A_ACDC.parquet")
        a_cols = formal_descriptor_names()
    ids = dev.merge(a[["sample_id"]], on="sample_id").merge(b[["sample_id"]], on="sample_id").sort_values("sample_id").reset_index(drop=True)
    a_rows = a.set_index("sample_id").loc[ids["sample_id"], a_cols].to_numpy(dtype=np.float64)
    b_rows = b.set_index("sample_id").loc[ids["sample_id"]].copy().reset_index()
    return ids, a_rows, b_rows, ids["outer_fold"].to_numpy(dtype=int)


def _inner_fold_fn(ctx: StageContext, ids: pd.DataFrame) -> Callable[..., np.ndarray]:
    """``fn(outer_fold, train_positions=None)`` -> v6 inner-fold labels of those rows of ``ids`` (contracts/inner_folds)."""

    inner = pd.read_parquet(_contracts(ctx) / "inner_folds.parquet")

    def fn(outer_fold: int, train_positions: np.ndarray | None = None) -> np.ndarray:
        rows = inner[inner["outer_fold_heldout"] == int(outer_fold)].set_index("sample_id")["inner_fold"]
        if train_positions is None:
            train = ids.loc[ids["outer_fold"] != int(outer_fold), "sample_id"]
        else:
            train = ids["sample_id"].iloc[np.asarray(train_positions, dtype=int)]
        return rows.loc[train].to_numpy(dtype=int)

    return fn


def fit_transition(ctx: StageContext, *, profile: str = "ridge", nested_oof: bool = True):
    """Nested OOF B-hat on acdc_dev, the final transition model (lambda by CV over the outer folds), comparators."""

    if profile != "ridge":
        raise ValueError("fit-transition: the only v6 profile is 'ridge' (configs/transition.yaml estimator.primary)")
    run = ctx.require_run("fit-transition")
    fit_v6 = methods_api("tmmri.transition_fit", "fit_transition_v6")
    nested = methods_api("tmmri.transition_fit", "nested_oof_transition")
    comparators = methods_api("tmmri.transition_fit", "comparators")
    targets = ctx.study.transition_targets()
    tdir, odir = run.sub("transitions"), run.sub("oof")
    named = [c for c in NAMED_COMPARATORS if c in (ctx.study.transition.get("comparators") or {})]
    outputs = [tdir / "transition_model.json", tdir / "lambda_selection.json"]
    if nested_oof:
        outputs += [odir / "bhat_oof_ACDC_dev.parquet", odir / "bhat_oof_ACDC_dev_p6.parquet", tdir / "lambda_selection.parquet"]
        outputs += [tdir / f"comparator_{name}.parquet" for name in ("mean", "pseudoinverse_p1", "identity_transform_ridge", *named)]
        outputs += fold_model_paths(ctx, run)

    def compute() -> dict[str, Any]:
        ids, a, b, folds = _dev_matrices(ctx)
        cfg, registry = ctx.study.transition, ctx.study.registry
        model = fit_v6(a, b, targets, cfg, registry, folds)
        atomic_write_json(outputs[0], model.to_dict(), allow_nan=True)
        atomic_write_json(outputs[1], {"lambdas": dict(model.lambdas), "edge_flags": dict(model.edge_flags)}, allow_nan=True)
        if nested_oof:
            inner_fn = _inner_fold_fn(ctx, ids)
            oof, details = nested(a, b, targets, cfg, registry, folds, inner_fn, return_details=True)
            _write_table(oof.assign(sample_id=ids["sample_id"].to_numpy()) if "sample_id" not in oof else oof, outputs[2])
            for f, fold_model in sorted(details["models"].items()):   # outer-fold mappers of per-object stability (11.4)
                atomic_write_json(fold_model_path(run, f), fold_model.to_dict(), allow_nan=True)
            ids_p6, a_p6, b_p6, folds_p6 = _dev_matrices(ctx, "p6")
            oof_p6 = nested(a_p6, b_p6, targets, cfg, registry, folds_p6, _inner_fold_fn(ctx, ids_p6))
            _write_table(oof_p6.assign(sample_id=ids_p6["sample_id"].to_numpy()) if "sample_id" not in oof_p6 else oof_p6, outputs[3])
            # Per-fold lambda ledger (LAMBDA_AT_GRID_EDGE flags) plus the final bundle's selection (configs/transition.yaml).
            final = pd.DataFrame([{"outer_fold": pd.NA, "target": t, "lambda": model.lambdas.get(t, np.nan),
                                   "lambda_at_grid_edge": bool(model.edge_flags.get(t, False)),
                                   "flag": "LAMBDA_AT_GRID_EDGE" if model.edge_flags.get(t, False) else "",
                                   "eligible": t in model.fitted_targets, "exclusion_reason": model.excluded.get(t, ""),
                                   "n_fit_rows": model.n_fit_rows} for t in targets])
            ledger = pd.concat([details["lambda_selection"].assign(scope="outer_fold"), final.assign(scope="final_bundle")], ignore_index=True)
            ledger["outer_fold"] = ledger["outer_fold"].astype("Int64")
            _write_table(ledger, outputs[4])
            for name, frame in comparators(a, b, targets, cfg, registry, folds, inner_fold_fn=inner_fn,
                                           global_lambda="ridge_global_lambda" in named, rbf="rbf_kernel_ridge" in named).items():
                _write_table(frame.assign(sample_id=ids["sample_id"].to_numpy()) if "sample_id" not in frame else frame,
                             tdir / f"comparator_{name}.parquet")
        return {"targets": len(targets)}

    inputs = [run.path / "semantics" / "B_ACDC.parquet", run.path / "formal" / "A_ACDC.parquet",
              run.path / "formal" / "A_ACDC_p6_128.parquet", _contracts(ctx) / "split_manifest.parquet", _contracts(ctx) / "inner_folds.parquet"]
    return _stage(ctx, "fit-transition", compute, inputs, outputs, {"profile": profile, "nested_oof": nested_oof}, fitting=True)


NAMED_COMPARATORS = ("ridge_global_lambda", "rbf_kernel_ridge")   # configs/transition.yaml comparators (SPECIFICATION.md 9.4 f, g)


def fold_model_path(run: RunRoot, fold: int) -> Path:
    return run.path / "transitions" / "fold_models" / f"transition_fold{int(fold)}.json"


def fold_model_paths(ctx: StageContext, run: RunRoot) -> list[Path]:
    n = int(ctx.study.splits["acdc"]["acdc_dev"]["outer_folds"]["n_folds"])
    return [fold_model_path(run, f) for f in range(1, n + 1)]


def load_fold_models(ctx: StageContext, run: RunRoot) -> list[Any]:
    """The frozen outer-fold transitions (per-object stability mappers, SPECIFICATION.md 11.4)."""

    model_cls = methods_api("tmmri.transition_fit", "TransitionModel")
    paths = fold_model_paths(ctx, run)
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{missing[0]} missing: run fit-transition --nested-oof first")
    return [model_cls.from_dict(json.loads(p.read_text(encoding="utf-8"))) for p in paths]


def fit_predictor(ctx: StageContext, *, extended: bool = True):
    """Audited predictor d_model on standardized formal A (acdc_dev) + cross-fitted d_model per outer fold.

    ``extended`` (protocol v6.0.1 A2/A4; :mod:`tmmri.predictor_reporting`) also stores the outer-fold predictors with
    their out-of-fold probabilities and the measured-B comparator; the three v6.0.0 outputs are byte-identical either way.
    """

    import joblib

    run = ctx.require_run("fit-predictor")
    fit = methods_api("tmmri.classification", "fit_formal_predictor")
    fdir = run.sub("formal")
    outputs = [fdir / "d_model_predictor.joblib", fdir / "d_model_ACDC_dev.parquet", fdir / "d_model_crossfit_ACDC_dev.parquet"]
    # --- protocol v6.0.1 (A2, A4): extra declared outputs; None/[] reproduce the v6.0.0 stage record exactly
    if extended:
        from .predictor_reporting import fit_predictor_extension_params, fit_predictor_extension_paths

        extra_outputs, extra_params = fit_predictor_extension_paths(run, ctx.study), fit_predictor_extension_params(ctx.study)
    else:
        extra_outputs, extra_params = [], None
    # --- end protocol v6.0.1

    def compute() -> dict[str, Any]:
        from .formal_features import formal_descriptor_names

        split = _split_manifest(ctx)
        dev = split[split["partition"] == "acdc_dev"][["sample_id", "outer_fold"]]
        a = pd.read_parquet(run.path / "formal" / "A_ACDC.parquet").merge(dev, on="sample_id").sort_values("sample_id").reset_index(drop=True)
        cohort = _cohort(ctx, "ACDC").set_index("sample_id")
        y = cohort.loc[a["sample_id"], "acdc_class"].astype(str).reset_index(drop=True)
        x = a[formal_descriptor_names()].to_numpy(dtype=np.float64)
        # Critical-QC rows are excluded from every fit (D21); non-finite A rows are INPUT_NONFINITE (no prediction).
        b = pd.read_parquet(run.path / "semantics" / "B_ACDC.parquet").set_index("sample_id")
        critical = critical_qc_mask(b, ctx.study).to_numpy()
        critical_ids = set(b.index[critical])
        finite = np.isfinite(x).all(axis=1)
        fit_rows = finite & ~a["sample_id"].isin(critical_ids).to_numpy()
        cfg = ctx.study.formal["audited_predictor"]
        predictor = fit(x, y, cfg, fit_mask=fit_rows)
        joblib.dump(predictor, outputs[0])
        _write_table(pd.DataFrame({"sample_id": a["sample_id"].to_numpy(), "d_model": predict_labels(predictor, x)}), outputs[1])
        folds = a["outer_fold"].to_numpy(dtype=int)
        crossfit = np.empty(len(a), dtype=object)
        fold_predictors: dict[int, Any] = {}
        for fold in sorted(set(folds)):
            held = folds == fold
            fold_predictors[int(fold)] = fit(x, y, cfg, fit_mask=fit_rows & ~held)
            crossfit[held] = predict_labels(fold_predictors[int(fold)], x[held])
        _write_table(pd.DataFrame({"sample_id": a["sample_id"].to_numpy(), "outer_fold": folds, "d_model_crossfit": crossfit}), outputs[2])
        # --- protocol v6.0.1 (A2, A4): written after, and independent of, the v6.0.0 outputs above
        if extended:
            from .predictor_reporting import store_fit_predictor_extensions

            ids = [str(s) for s in a["sample_id"]]
            store_fit_predictor_extensions(run, ctx.study, ids=ids, x=x, y=y, folds=folds, critical=a["sample_id"].isin(critical_ids).to_numpy(),
                                           fold_predictors=fold_predictors, b_dev=b.reindex(ids), clusters=_cluster_ids(split, ids).to_numpy(dtype=object))
        # --- end protocol v6.0.1
        return {"rows": int(len(a)), "fit_rows": int(fit_rows.sum())}

    inputs = [run.path / "formal" / "A_ACDC.parquet", run.path / "semantics" / "B_ACDC.parquet", _contracts(ctx) / "split_manifest.parquet",
              _contracts(ctx) / "cohort_ACDC.parquet"]
    return _stage(ctx, "fit-predictor", compute, inputs, outputs + extra_outputs, extra_params, fitting=True)


# ----------------------------------------------------------------------------- discretize, rules (METHODS)


def _rulebooks(ctx: StageContext) -> dict[str, dict[str, Any]]:
    return {str(v["name"]): dict(v) for v in ctx.study.rule_policy["rulebooks"].values()}


def predict_labels(predictor: Any, x: np.ndarray) -> np.ndarray:
    """Predictor labels for the finite rows of ``x``; ``None`` for non-finite rows (INPUT_NONFINITE, no decision)."""

    x = np.asarray(x, dtype=float)
    out = np.full(x.shape[0], None, dtype=object)
    finite = np.isfinite(x).all(axis=1)
    if finite.any():
        out[finite] = np.asarray(predictor.predict(x[finite]), dtype=object)
    return out


def _rule_substrate(ctx: StageContext, rulebook: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """(antecedent values by sample_id, consequent) of a rulebook on the acdc_dev fit rows (critical QC removed)."""

    run = ctx.require_run("rules")
    candidates = ctx.study.rule_candidates()
    if rulebook["substrate"] == "measured_B":
        values = pd.read_parquet(run.path / "semantics" / "B_ACDC.parquet")
        values = values[~critical_qc_mask(values, ctx.study).to_numpy()]
    else:
        values = pd.read_parquet(run.path / "oof" / "bhat_oof_ACDC_dev.parquet")
    values = values.set_index("sample_id")
    if rulebook["consequent"] == "Y_acdc_class":
        decision = _cohort(ctx, "ACDC").set_index("sample_id")["acdc_class"]
    else:
        decision = pd.read_parquet(run.path / "formal" / "d_model_ACDC_dev.parquet").set_index("sample_id")["d_model"]
    decision = decision.dropna()
    ids = sorted(set(values.index) & set(decision.index))
    cols = [c for c in candidates if c in values.columns]
    cols += [f"{c}__state" for c in cols if f"{c}__state" in values.columns]
    return values.loc[ids, cols], decision.loc[ids].astype(str)


def discretize(ctx: StageContext, *, profile: str = "wedd_gaussian_mm_v1"):
    run = ctx.require_run("discretize")
    if profile not in ctx.study.wedd["profiles"]:
        raise ValueError(f"unknown discretization profile {profile!r} (configs/wedd.yaml profiles)")
    fit_cuts = methods_api("tmmri.wedd", "fit_cuts")
    stability = methods_api("tmmri.wedd", "stability_gate")
    boot = ctx.study.bootstrap["threshold_bootstrap"]
    books = _rulebooks(ctx)
    tdir, gdir = run.sub("thresholds"), run.sub("granules")
    outputs = [p for rb in books for p in (tdir / f"{rb}_{profile}.json", tdir / f"{rb}_{profile}_stability.parquet", gdir / f"{rb}_{profile}_states_ACDC_dev.parquet")]
    outputs.append(tdir / "wedd_candidates.parquet")

    def compute() -> dict[str, Any]:
        from .wedd import CANDIDATE_COLUMNS

        candidates = []
        for rb, spec in books.items():
            values, decision = _rule_substrate(ctx, spec)
            cuts = fit_cuts(values, decision, profile, ctx.study.wedd)
            atomic_write_json(tdir / f"{rb}_{profile}.json", cuts.to_dict(), allow_nan=True)
            candidates.append(cuts.candidates_frame().assign(rulebook=rb))
            gate = stability(values, decision, profile, ctx.study.wedd, seed=int(boot["seed"]), n_replicates=int(boot["replicates"]), n_jobs=ctx.jobs,
                             groups=_cluster_ids(ctx, values.index).to_numpy(dtype=object))
            _write_table(gate, tdir / f"{rb}_{profile}_stability.parquet")
            _write_table(cuts.transform(values).reset_index(), gdir / f"{rb}_{profile}_states_ACDC_dev.parquet")
        # Every valid candidate cut of every split node with its objective terms (SPECIFICATION.md 11.3.1).
        table = pd.concat([c for c in candidates if len(c)], ignore_index=True) if any(len(c) for c in candidates) else \
            pd.DataFrame(columns=CANDIDATE_COLUMNS + ["rulebook"])
        _write_table(table[["rulebook", *CANDIDATE_COLUMNS]], outputs[-1])
        return {"rulebooks": list(books), "wedd_candidates": int(len(table))}

    inputs = [run.path / "oof" / "bhat_oof_ACDC_dev.parquet", run.path / "semantics" / "B_ACDC.parquet", run.path / "formal" / "d_model_ACDC_dev.parquet"]
    return _stage(ctx, "discretize", compute, inputs, outputs, {"profile": profile}, fitting=True)


def _eligibility_path(run: RunRoot) -> Path:
    return run.path / "rules" / "semantic_eligibility.json"


def eligible_attributes(ctx: StageContext, rulebook: str) -> list[str]:
    """Eligible antecedents of ``rulebook`` from ``rules/semantic_eligibility.json`` (check-semantic-eligibility)."""

    path = _eligibility_path(ctx.require_run("rules"))
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: run check-semantic-eligibility first")
    return list(json.loads(path.read_text(encoding="utf-8"))["rulebooks"][rulebook]["eligible_features"])


def check_semantic_eligibility(ctx: StageContext, *, profile: str | None = None):
    """M7 eligibility gates (``configs/rule_policy.yaml`` ``eligibility_gates``; SPECIFICATION.md Section 11.2).

    Per rulebook and candidate antecedent: reference availability on the acdc_dev patients, variability on the fit
    rows, reconstruction (OOF ridge vs OOF mean, pooled improvement and folds improved; ``d_model`` and
    ``dataset_label`` rulebooks), post-discretization stability of every cut (the rulebook's bootstrap table from
    ``discretize``) and the no-tautology policy.  Also writes the anchor manifest ``rules/rule_anchor_manifest.parquet``.
    Outputs ``rules/semantic_eligibility.parquet`` and ``.json``; ``induce-rules`` and ``gate-rules`` read them.
    """

    from .rule_pipeline import eligibility_summary, semantic_eligibility

    run = ctx.require_run("check-semantic-eligibility")
    profile = profile or str(ctx.study.wedd["primary_profile"])
    books = _rulebooks(ctx)
    rdir = run.sub("rules")
    outputs = [rdir / "semantic_eligibility.parquet", _eligibility_path(run), rdir / "rule_anchor_manifest.parquet"]

    def compute() -> dict[str, Any]:
        split = _split_manifest(ctx)
        dev_ids = split.loc[split["partition"] == "acdc_dev", "sample_id"]
        b_all = pd.read_parquet(run.path / "semantics" / "B_ACDC.parquet")
        critical = critical_qc_mask(b_all, ctx.study).to_numpy()
        anchors = split[split["partition"] == "acdc_dev"][["sample_id", "patient_id", "outer_fold"]].copy()
        anchors["critical_qc_excluded"] = anchors["sample_id"].isin(set(b_all.loc[critical, "sample_id"]))
        _write_table(anchors.sort_values("sample_id").reset_index(drop=True), outputs[2])
        reference = b_all[b_all["sample_id"].isin(set(dev_ids))].set_index("sample_id")
        oof = pd.read_parquet(run.path / "oof" / "bhat_oof_ACDC_dev.parquet").set_index("sample_id")
        mean_oof = pd.read_parquet(run.path / "transitions" / "comparator_mean.parquet").set_index("sample_id")
        fit_rows = reference.loc[[i for i in oof.index if i in reference.index]]
        folds = oof.loc[fit_rows.index, "outer_fold"].to_numpy(dtype=float)
        candidates = ctx.study.rule_candidates()
        tables, class_counts = [], {}
        for rb, spec in books.items():
            stability = pd.read_parquet(run.path / "thresholds" / f"{rb}_{profile}_stability.parquet")
            recon = spec["substrate"] != "measured_B"
            tables.append(semantic_eligibility(
                rulebook=rb, reference=reference, fit_rows=fit_rows, oof=oof.loc[fit_rows.index] if recon else None,
                mean_oof=mean_oof.loc[fit_rows.index] if recon else None, folds=folds if recon else None, stability=stability,
                candidates=candidates, rule_policy=ctx.study.rule_policy, registry=ctx.study.registry))
            _, decision = _rule_substrate(ctx, spec)
            class_counts[rb] = decision.value_counts().to_dict()
        table = pd.concat(tables, ignore_index=True)
        _write_table(table, outputs[0], csv=True)
        min_class = int(ctx.study.rule_policy["eligibility_gates"]["consequent_support"]["min_fit_patients_per_class"])
        summary = eligibility_summary(table, class_counts, min_class)
        summary.update({"profile": profile, "role": "development_only", "partition": "acdc_dev"})
        atomic_write_json(outputs[1], summary, allow_nan=True)
        return {rb: len(v["eligible_features"]) for rb, v in summary["rulebooks"].items()}

    inputs = [run.path / "semantics" / "B_ACDC.parquet", run.path / "oof" / "bhat_oof_ACDC_dev.parquet",
              run.path / "transitions" / "comparator_mean.parquet", run.path / "formal" / "d_model_ACDC_dev.parquet",
              _contracts(ctx) / "split_manifest.parquet"] + [run.path / "thresholds" / f"{rb}_{profile}_stability.parquet" for rb in books]
    return _stage(ctx, "check-semantic-eligibility", compute, inputs, outputs, {"profile": profile}, fitting=True)


def induce_rules(ctx: StageContext, *, tier: str = "variable_precision", profile: str | None = None):
    """Variable-precision discovery per rulebook over its eligible antecedents (``rules/<rb>/rulebook_induced.json``)."""

    run = ctx.require_run("induce-rules")
    if tier != "variable_precision":
        raise ValueError("induce-rules: the v6 tier is 'variable_precision' (configs/rule_policy.yaml discovery.tiers.primary)")
    induce = methods_api("tmmri.rough_sets", "induce_vprs_rules")
    profile = profile or str(ctx.study.wedd["primary_profile"])
    books = _rulebooks(ctx)
    outputs = [run.sub("rules") / rb / "rulebook_induced.json" for rb in books]

    def compute() -> dict[str, Any]:
        out = {}
        for rb, spec in books.items():
            states = pd.read_parquet(run.path / "granules" / f"{rb}_{profile}_states_ACDC_dev.parquet").set_index("sample_id")
            _, decision = _rule_substrate(ctx, spec)
            states = states.loc[[i for i in states.index if i in decision.index]]
            anchors = pd.Series([s.split(":", 1)[1] for s in states.index], index=states.index, name="patient_id")
            attributes = eligible_attributes(ctx, rb)
            book = induce(states, decision.loc[states.index], anchors, ctx.study.rule_policy, registry=ctx.study.registry,
                          attributes=attributes, consequent_kind=rb, rulebook_id=rb)
            atomic_write_json(run.path / "rules" / rb / "rulebook_induced.json",
                              {"rulebook": rb, "attributes": attributes, "records": book.to_records(),
                               "class_records": book.class_records, "failing_gates": book.failing_gates}, allow_nan=True)
            out[rb] = len(book.rules)
        return out

    inputs = [run.path / "granules" / f"{rb}_{profile}_states_ACDC_dev.parquet" for rb in books] + [_eligibility_path(run)]
    return _stage(ctx, "induce-rules", compute, inputs, outputs, {"tier": tier, "profile": profile}, fitting=True)


def inner_oof_path(run: RunRoot, fold: int) -> Path:
    return run.path / "oof" / f"inner_oof_ACDC_dev_fold{int(fold)}.parquet"


def _provenance(ctx: StageContext, paths: Mapping[str, Path], stage: str) -> dict[str, Any]:
    code = code_version()
    return {**{f"{k}_sha256": hash_path(v) for k, v in sorted(paths.items())},
            "config_sections_sha256": ctx.study.sections_hash(STAGE_CONFIG_SECTIONS[stage]), "code_commit": code.get("commit"),
            "code_sha256": code.get("code_sha256")}


def gate_rules(ctx: StageContext, *, profile: str | None = None, substrate: str = "nested_inner_oof"):
    """Cross-fitted validation gate over the five acdc_dev outer folds; writes ``rulebook.json`` or ``EMPTY.json``.

    Per outer fold ``f`` (SPECIFICATION.md Section 11.6): the predictor (``d_model`` consequents) is refitted
    without fold ``f``; the WEDD cuts are refitted on the training folds' **inner nested out-of-fold**
    reconstructions (transitions refitted inside outer-train(f) only, lambda by CV over the remaining inner folds;
    :class:`tmmri.rule_pipeline.NestedGateSubstrate`) and discovery is re-run there; the held-out fold is
    discretized from its outer OOF reconstruction.  ``substrate="nested_inner_oof"`` is the default and is recorded
    in the stage record and in each rulebook's provenance (``outer_oof`` is a sensitivity only); the measured-B oracle
    uses measured B for both.  Surviving rules carry ``eligibility = ELIGIBLE`` and their cross-fit statistics
    (:func:`tmmri.rough_sets.apply_gate`).
    """

    from .rule_pipeline import NestedGateSubstrate, acdc_inner_labels, gate_and_apply, rulebook_payload

    if substrate not in {"nested_inner_oof", "outer_oof"}:
        raise ValueError("gate-rules substrate must be nested_inner_oof (default) or outer_oof (sensitivity)")
    run = ctx.require_run("gate-rules")
    rulebook_cls = methods_api("tmmri.rough_sets", "Rulebook")
    fit_predictor_fn = methods_api("tmmri.classification", "fit_formal_predictor")
    profile = profile or str(ctx.study.wedd["primary_profile"])
    books = _rulebooks(ctx)
    outputs = [run.path / "rules" / rb / name for rb in books for name in ("gate.parquet", "rulebook_or_empty.json")]
    n_folds = int(ctx.study.rule_policy["validation_gate"]["folds"])
    nested_any = substrate == "nested_inner_oof" and any(spec["substrate"] != "measured_B" for spec in books.values())
    if nested_any:
        outputs += [inner_oof_path(run, f) for f in range(1, n_folds + 1)]

    def compute() -> dict[str, Any]:
        ids, a, b, folds_all = _dev_matrices(ctx)
        row_ids = ids["sample_id"].astype(str).to_numpy()
        labels_all = _cohort(ctx, "ACDC").set_index("sample_id").loc[row_ids, "acdc_class"].astype(str).to_numpy()
        cfg = ctx.study.formal["audited_predictor"]
        oof = pd.read_parquet(run.path / "oof" / "bhat_oof_ACDC_dev.parquet").set_index("sample_id")
        inner = pd.read_parquet(_contracts(ctx) / "inner_folds.parquet")
        targets = ctx.study.transition_targets()
        candidates = [c for c in ctx.study.rule_candidates() if c in oof.columns]
        shared = NestedGateSubstrate(a, b, targets, ctx.study.transition, ctx.study.registry, folds_all,
                                     acdc_inner_labels(inner, row_ids), oof, candidates, row_ids)
        summary = {}
        for rb, spec in books.items():
            values, _ = _rule_substrate(ctx, spec)
            subset = np.flatnonzero(pd.Index(row_ids).isin(values.index))
            gate_ids = row_ids[subset]
            predictors: dict[tuple[int, ...], Any] = {}

            def d_model_fn(train: np.ndarray, target: np.ndarray, _subset: np.ndarray = subset, _cache: dict = predictors) -> np.ndarray:
                key = tuple(int(i) for i in train)
                if key not in _cache:
                    _cache[key] = fit_predictor_fn(a[_subset[train]], pd.Series(labels_all[_subset[train]]), cfg)
                return predict_labels(_cache[key], a[_subset[target]])

            def label_fn(train: np.ndarray, target: np.ndarray, _subset: np.ndarray = subset) -> np.ndarray:
                return labels_all[_subset[target]]

            nested = substrate == "nested_inner_oof" and spec["substrate"] != "measured_B"
            substrate_fn = shared.with_subset(subset) if nested else None
            induced = json.loads((run.path / "rules" / rb / "rulebook_induced.json").read_text(encoding="utf-8"))
            book = rulebook_cls.from_records(induced["records"], rulebook_id=rb, consequent_kind=rb)
            attributes = list(induced["attributes"]) if "attributes" in induced else eligible_attributes(ctx, rb)
            table, gated = gate_and_apply(values=values.loc[gate_ids], decision_fn=label_fn if spec["consequent"] == "Y_acdc_class" else d_model_fn,
                                          folds=folds_all[subset], rule_policy=ctx.study.rule_policy, book=book, profile=profile,
                                          wedd_cfg=ctx.study.wedd, anchors=np.array([s.split(":", 1)[1] for s in gate_ids], dtype=object),
                                          substrate_fn=substrate_fn, registry=ctx.study.registry, rulebook_id=rb, attributes=attributes)
            _write_table(table, run.path / "rules" / rb / "gate.parquet")
            provenance = _provenance(ctx, {
                "A": run.path / "formal" / "A_ACDC.parquet", "B": run.path / "semantics" / "B_ACDC.parquet",
                "T": run.path / "transitions" / "transition_model.json", "oof_ledger": run.path / "oof" / "bhat_oof_ACDC_dev.parquet",
                "thresholds": run.path / "thresholds" / f"{rb}_{profile}.json", "anchor_manifest": run.path / "rules" / "rule_anchor_manifest.parquet",
                "decision": run.path / ("contracts/cohort_ACDC.parquet" if spec["consequent"] == "Y_acdc_class" else "formal/d_model_ACDC_dev.parquet"),
                "eligibility": _eligibility_path(run)}, "gate-rules")
            provenance["gate_substrate"] = "nested_inner_oof" if nested else ("measured_B" if spec["substrate"] == "measured_B" else "outer_oof")
            name, payload = rulebook_payload(rb, gated, table, provenance)
            for stale in ("rulebook.json", "EMPTY.json"):
                (run.path / "rules" / rb / stale).unlink(missing_ok=True)
            atomic_write_json(run.path / "rules" / rb / name, payload, allow_nan=True)
            atomic_write_json(run.path / "rules" / rb / "rulebook_or_empty.json",
                              {"rulebook": rb, "file": name, "eligible_rules": len(payload.get("records", [])),
                               "gate_substrate": provenance["gate_substrate"]})
            summary[rb] = len(payload.get("records", []))
        if nested_any:   # inner nested OOF ledger of the gate (reused by run-ablations)
            for f in range(1, n_folds + 1):
                _write_table(shared.inner_oof(f).reset_index(names="sample_id"), inner_oof_path(run, f))
        return summary

    inputs = [run.path / "rules" / rb / "rulebook_induced.json" for rb in books] + [
        run.path / "formal" / "A_ACDC.parquet", run.path / "oof" / "bhat_oof_ACDC_dev.parquet", _contracts(ctx) / "inner_folds.parquet",
        _eligibility_path(run)]
    return _stage(ctx, "gate-rules", compute, inputs, outputs, {"profile": profile, "substrate": substrate}, fitting=True)


# ----------------------------------------------------------------------------- freeze-analysis-bundle


def bundle_files(run: RunRoot) -> dict[str, str]:
    """Every fitted object of the analysis bundle (relative path -> SHA-256)."""

    files: dict[str, str] = {}
    for folder in BUNDLE_DIRS:
        base = run.path / folder
        for path in sorted(base.rglob("*")) if base.is_dir() else []:
            rel = path.relative_to(run.path)
            if path.is_file() and not path.name.startswith(".") and not set(rel.parts) & set(SECONDARY_SUBDIRS):
                files[rel.as_posix()] = hash_path(path)
    for rel in BUNDLE_FILES:
        if (run.path / rel).is_file():
            files[rel] = hash_path(run.path / rel)
    return files


def freeze_analysis_bundle(ctx: StageContext, *, required: Iterable[str] = ("transitions/transition_model.json", "formal/d_model_predictor.joblib")) -> dict[str, Any]:
    """Write ``.stages/FROZEN.json``: bundle hash manifest, config hashes, code version (no timestamps inside).

    ``bundle_config`` holds the digests of the configuration sections of the fitting stages (:data:`BUNDLE_STAGES`);
    evaluation-only sections are governed by the protocol tag and the amendments log, not by the bundle.
    Secondary outputs (``*/refit_arm/``, ``*/ablations/``) are never part of the bundle.
    """

    run = ctx.require_run("freeze-analysis-bundle")
    files = bundle_files(run)
    missing = [r for r in required if r not in files]
    books = _rulebooks(ctx)
    for rb in books:
        if f"rules/{rb}/rulebook.json" not in files and f"rules/{rb}/EMPTY.json" not in files:
            missing.append(f"rules/{rb}/rulebook.json|EMPTY.json")
    if missing:
        raise BundleError(f"analysis bundle incomplete: {missing}")
    code = code_version()
    # protocol-v6.0.1 (decision F; outside bundle_sha256): the newest configured protocol tag that exists with HEAD descending
    # from it at freeze time (amendment tags signed, as the M&Ms-2 lock); an unsigned or absent amendment tag is never recorded.
    present = [t for i, t in enumerate(ctx.study.protocol_tags) if protocol_frozen(t, signed=i > 0)]
    payload = {"schema": "tmmri.frozen_bundle/1", "run_id": run.run_id, "bundle": files, "bundle_sha256": sha256_json(files),
               "config": ctx.study.hash_manifest(), "bundle_config": bundle_config(ctx.study), "code": code,
               "protocol_tag": ctx.study.freeze_tag, "protocol_tag_present_at_freeze": ctx.study.freeze_tag in present,
               "protocol_version_tag": present[-1] if present else None, "protocol_version_tag_configured": ctx.study.current_protocol_tag,
               "protocol_version_tag_present_at_freeze": ctx.study.current_protocol_tag in present}
    if run.frozen_path.is_file():
        prior = json.loads(run.frozen_path.read_text(encoding="utf-8"))
        if prior.get("bundle_sha256") == payload["bundle_sha256"] and _bundle_config_matches(prior, ctx.study):
            run.set_status("freeze-analysis-bundle", "BUILT", action="SKIPPED", bundle_sha256=payload["bundle_sha256"])
            return prior
        raise BundleError(f"run {run.run_id} is already frozen with a different bundle or config; use a new run id")
    atomic_write_json(run.frozen_path, payload)
    run.set_status("freeze-analysis-bundle", "BUILT", action="RAN", bundle_sha256=payload["bundle_sha256"])
    run.log({"stage": "freeze-analysis-bundle", "bundle_files": len(files), "code_dirty": code.get("dirty")})
    return payload


def verify_frozen_bundle(ctx: StageContext) -> dict[str, Any]:
    run = ctx.require_run("evaluate")
    if not run.frozen_path.is_file():
        raise BundleError(f"run {run.run_id} has no frozen analysis bundle (run freeze-analysis-bundle first)")
    frozen = json.loads(run.frozen_path.read_text(encoding="utf-8"))
    changed = [rel for rel, digest in frozen["bundle"].items() if hash_path(run.path / rel) != digest]
    if changed:
        raise BundleError(f"frozen bundle changed after the freeze: {changed[:5]}")
    if not _bundle_config_matches(frozen, ctx.study):
        raise BundleError("configs or registry changed after the freeze (bundle configuration sections differ)")
    return frozen


def bundle_config(study: StudyConfig) -> dict[str, Any]:
    sections = list(bundle_sections())
    return {"sections": study.sections_digest(sections), "hash": study.sections_hash(sections)}


def _bundle_config_matches(frozen: Mapping[str, Any], study: StudyConfig) -> bool:
    recorded = frozen.get("bundle_config")
    if recorded is None:        # records written before per-section hashing: the whole configuration must match
        return frozen.get("config", {}).get("config_hash") == study.config_hash
    return recorded.get("hash") == bundle_config(study)["hash"]


# ----------------------------------------------------------------------------- evaluate, refit arm (METHODS)


SAME_DEFINITION = "HARMONIZABLE_SAME_DEFINITION"   # registry label_tables.mms_to_acdc compatibility value


def reference_labels(cohort: pd.DataFrame, dataset: str) -> pd.Series:
    """Evaluation-only class reference: the ACDC class, or the M&Ms crosswalk class of the
    ``HARMONIZABLE_SAME_DEFINITION`` cases only (IHD -> MINF is reported separately, off-ontology has none; D8)."""

    if canonical_dataset(dataset) == "ACDC":
        return cohort["acdc_class"].astype(object)
    same = cohort["crosswalk_compatibility"].astype(str) == SAME_DEFINITION
    return cohort["crosswalk_acdc_class"].astype(object).where(same)


def _dev_reference(ctx: StageContext) -> tuple[pd.Series, pd.Series]:
    """(fit-partition SD, fit-partition mean) of the transition targets on the acdc_dev fit rows."""

    run = ctx.require_run("evaluate")
    split = _split_manifest(ctx)
    targets = ctx.study.transition_targets()
    b_dev = pd.read_parquet(run.path / "semantics" / "B_ACDC.parquet")
    dev_ids = set(split.loc[split["partition"] == "acdc_dev", "sample_id"])
    b_dev = b_dev[b_dev["sample_id"].isin(dev_ids) & ~critical_qc_mask(b_dev, ctx.study)]
    return b_dev[targets].std(ddof=1), b_dev[targets].mean()


def _rho_kwargs(ctx: StageContext) -> dict[str, Any]:
    boot = ctx.study.bootstrap
    return dict(family=ctx.study.primary_family(), n_boot=int(boot["fixed_model_bootstrap"]["replicates"]),
                seed=int(boot["fixed_model_bootstrap"]["seed"]), min_estimable=int(boot["primary_test"]["min_estimable_targets"]),
                min_valid_fraction=float(boot["fixed_model_bootstrap"]["min_valid_fraction"]), n_jobs=ctx.jobs)


def _cluster_ids(source: StageContext | pd.DataFrame, ids: Iterable[str]) -> pd.Series:
    """Bootstrap ``cluster_id`` of ``ids`` from the split manifest (a declared duplicate-image pair shares one cluster;
    every other case is its own cluster, ``cluster_id = sample_id``; configs/cohorts.yaml duplicate_image_pairs)."""

    split = _split_manifest(source) if isinstance(source, StageContext) else source
    table = split.set_index("sample_id") if "sample_id" in split.columns else split
    ids = [str(s) for s in ids]
    found = table["cluster_id"].reindex(ids) if "cluster_id" in table.columns else pd.Series([None] * len(ids), index=ids, dtype=object)
    return pd.Series([c if isinstance(c, str) and c else i for i, c in zip(ids, found.tolist())], index=ids, dtype=object)


def _row_clusters(frame: pd.DataFrame, clusters: pd.Series) -> np.ndarray:
    """``cluster_id`` aligned with the rows of ``frame`` (its ``sample_id`` column, else its index)."""

    ids = frame["sample_id"].astype(str) if "sample_id" in frame.columns else pd.Index(frame.index).astype(str)
    return _cluster_ids(clusters.rename("cluster_id").to_frame(), ids).to_numpy(dtype=object)


def _rho(ctx: StageContext, y_true: pd.DataFrame, y_pred: pd.DataFrame, mean: pd.Series, sd: pd.Series, *,
         min_cases: int | None = None, clusters: pd.Series | None = None) -> dict[str, Any]:
    """Primary estimand on the rows of ``y_true`` (paired cluster bootstrap over ``cluster_id``; ``configs/bootstrap.yaml``).

    Each case weighs once in the point estimate; the bootstrap draws clusters (``clusters``, indexed by sample_id;
    default: read from the split manifest), so a declared duplicate-image pair is resampled together."""

    primary_rho = methods_api("tmmri.metrics", "primary_rho")
    targets = list(y_true.columns)
    y_null = pd.DataFrame([mean[targets]] * len(y_true), index=y_true.index)
    cluster = pd.Series([str(s).split(":", 1)[1] for s in y_true.index], index=y_true.index)
    resample = _cluster_ids(ctx if clusters is None else clusters.rename("cluster_id").to_frame(), y_true.index)
    out = primary_rho(y_true, y_pred[targets], y_null, sd, cluster=cluster, resample_cluster=resample.to_numpy(dtype=object), **_rho_kwargs(ctx))
    out["n_cases"] = int(len(y_true))
    if min_cases is not None:
        below = len(y_true) < min_cases
        out.update({"min_cases_per_stratum": int(min_cases), "descriptive_only": bool(below)})
        if below:   # point estimates kept; interval NE (configs/bootstrap.yaml subgroup_reporting_thresholds)
            status = below_threshold_status(ctx.study)
            out.update({"ci_lower": None, "ci_upper": None, "rho_j_ci": None, "status": status["status"], "reason": status["reason"]})
    return out


def below_threshold_status(study: StudyConfig) -> dict[str, str]:
    cfg = study.bootstrap["subgroup_reporting_thresholds"].get("below_threshold_status") or {}
    return {"status": str(cfg.get("status", "NE")), "reason": str(cfg.get("reason", "STRATUM_BELOW_MIN_CASES"))}


def hypothesis_id(study: StudyConfig, partition: str) -> str | None:
    """The configured hypothesis id of an evaluation partition (``configs/bootstrap.yaml`` ``primary_test``)."""

    pt = study.bootstrap["primary_test"]
    for item in list(pt.get("hierarchy", [])) + list(pt.get("sensitivity", [])):
        if str(item.get("partition")) == partition:
            return str(item["id"])
    h3 = ((study.cohorts.get("mms2") or {}).get("endpoints") or {}).get("h_primary_3") or {}   # protocol-v6.0.1
    if str(h3.get("partition")) == partition:
        return str(h3["id"])
    return None


def _stratum_label(value: Any) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "<missing>"
    if isinstance(value, float) and float(value).is_integer():
        return str(int(value))
    return str(value)


def external_strata(study: StudyConfig, cohort: pd.DataFrame, ids: list[str]) -> dict[str, pd.Series]:
    """Stratum variables of the external evaluation (``configs/bootstrap.yaml``
    ``subgroup_reporting_thresholds.external_strata``); ``refit_vendor_group`` maps the vendor to the seen (A, B) or
    unseen (C, D) group of ``configs/splits.yaml`` ``refit_arm.eval_strata``."""

    names = list(study.bootstrap["subgroup_reporting_thresholds"]["external_strata"])
    groups = study.splits["refit_arm"]["eval_strata"]
    seen, unseen = {str(v) for v in groups["seen_vendors"]}, {str(v) for v in groups["unseen_vendors"]}
    sub = cohort.loc[ids]
    out: dict[str, pd.Series] = {}
    for name in names:
        if name == "refit_vendor_group":
            out[name] = sub["vendor"].astype(str).map(lambda v: "seen_vendors" if v in seen else ("unseen_vendors" if v in unseen else "other"))
        elif name in sub.columns:
            out[name] = sub[name].map(_stratum_label)
        else:
            raise KeyError(f"external stratum {name!r} is not a column of contracts/cohort_MMS.parquet")
    return out


def descriptive_strata(study: StudyConfig, cohort: pd.DataFrame, ids: list[str]) -> dict[str, pd.Series]:
    """Descriptive demographic strata of the external evaluation (SPECIFICATION.md 12.3; Appendix A Q28;
    ``configs/bootstrap.yaml`` ``subgroup_reporting_thresholds.descriptive_strata``): ``sex`` as released and
    ``age_band`` from the release CSV age with the configured edges (``<40``, ``40-64``, ``>=65`` for edges 40, 65).
    They are reported with counts, point estimates and intervals, never tested, fitted or used for selection."""

    cfg = study.bootstrap["subgroup_reporting_thresholds"]
    names = list(cfg.get("descriptive_strata", []))
    sub = cohort.loc[ids]
    out: dict[str, pd.Series] = {}
    for name in names:
        if name == "age_band":
            lo, hi = (float(e) for e in cfg["age_band_edges_years"])
            age = pd.to_numeric(sub["age_years"], errors="coerce")
            labels = np.where(age.isna(), "<missing>", np.where(age < lo, f"<{lo:g}", np.where(age < hi, f"{lo:g}-{hi - 1:g}", f">={hi:g}")))
            out[name] = pd.Series(labels, index=sub.index)
        elif name in sub.columns:
            out[name] = sub[name].map(_stratum_label)
        else:
            raise KeyError(f"descriptive stratum {name!r} is not a column of contracts/cohort_MMS.parquet")
    return out


def _decoder_log(raw: pd.DataFrame, decoded: pd.DataFrame, states: pd.DataFrame, targets: list[str]) -> dict[str, Any]:
    """Every domain-guard block and clip of the decoders (SPECIFICATION.md 9.5), per target."""

    out = {}
    for t in targets:
        st = states[t].astype(str) if t in states.columns else pd.Series("", index=raw.index)
        r = pd.to_numeric(raw[t], errors="coerce").to_numpy(dtype=float)
        d = pd.to_numeric(decoded[t], errors="coerce").to_numpy(dtype=float)
        clipped = np.isfinite(r) & np.isfinite(d) & (r != d)
        out[t] = {"estimated": int((st == "ESTIMATED").sum()), "numeric_domain_blocked": int((st == "NUMERIC_DOMAIN_BLOCKED").sum()),
                  "target_ineligible": int((st == "TARGET_INELIGIBLE").sum()), "input_qc_blocked": int((st == "INPUT_QC_BLOCKED").sum()),
                  "clipped_within_guard": int(clipped.sum()),
                  "blocked_sample_ids": [str(i) for i in st.index[st == "NUMERIC_DOMAIN_BLOCKED"]]}
    return out


def evaluate(ctx: StageContext, split: str, *, no_fit: bool = False):
    """Evaluate the frozen bundle once on ``acdc_test`` or ``mms_external`` (never fits anything).

    Before routing, rows with a critical QC reason or a non-finite formal A (``INPUT_NONFINITE``) are ``QC_ABSTAIN``
    and get no ``d_model``; the M&Ms class reference is the same-definition crosswalk class only.  Reconstructions
    pass the decoders of SPECIFICATION.md 9.5 (registry-domain guard) and the per-object stability of Section 11.4
    (the five frozen outer-fold transitions) before they are discretized for routing; the raw reconstructions stay
    the substrate of every error metric.  ``acdc_test`` writes the F1 tests; ``mms_external`` writes the 220-case
    sensitivity, the fixed-sequence decision, per-stratum errors, routes and rho (strata below the reporting
    threshold: point estimates with NE intervals and tests) and the F3 tests.  BH is applied within each family.
    """

    from .metrics import bh_family_table, family_name, paired_error_tests, stratified_route_metrics
    from .rule_pipeline import decode_reconstructions, qc_route_frame, route_records, routing_substrate

    if split not in EVALUATION_SPLITS + (MMS2_SPLIT,):
        raise ValueError(f"evaluate --split must be one of {EVALUATION_SPLITS + (MMS2_SPLIT,)}")
    if not no_fit:
        raise ValueError("evaluate never fits; pass --no-fit to confirm the frozen-bundle evaluation")
    ds = {"acdc_test": "ACDC", "mms_external": "MMS", MMS2_SPLIT: "MMS2"}[split]
    if ds == "MMS":
        require_mms_unlocked(ctx, "evaluate --split mms_external")
    elif ds == "MMS2":       # protocol-v6.0.1: the unchanged v6.0.0 bundle on mms2_external (H-primary-3)
        require_mms2_unlocked(ctx, f"evaluate --split {MMS2_SPLIT}")
        require_mms2_validated(ctx, f"evaluate --split {MMS2_SPLIT}")
    frozen = verify_frozen_bundle(ctx)
    run = ctx.require_run("evaluate")
    model_cls = methods_api("tmmri.transition_fit", "TransitionModel")
    cutset_cls = methods_api("tmmri.wedd", "CutSet")
    per_target = methods_api("tmmri.metrics", "per_target_errors")
    route_metrics = methods_api("tmmri.metrics", "route_metrics")
    fixed_sequence = methods_api("tmmri.metrics", "fixed_sequence_test")
    books = _rulebooks(ctx)
    profile = str(ctx.study.wedd["primary_profile"])
    mdir, rdir, idir = run.sub("metrics"), run.sub("reconstructed"), run.sub("inference")
    multiplicity = ctx.study.bootstrap["multiplicity"]
    families = [str(f) for f in multiplicity["families"]]
    outputs = [rdir / f"bhat_{split}.parquet", idir / f"d_model_{split}.parquet", mdir / f"primary_rho_{split}.json",
               mdir / f"per_target_{split}.parquet", rdir / f"bhat_{split}_decoded.parquet", mdir / f"decoder_{split}.json"]
    outputs += [p for rb in books for p in (idir / f"routes_{rb}_{split}.parquet", mdir / f"routes_{rb}_{split}.json")]
    outputs += [rdir / f"stability_{rb}_{split}.parquet" for rb, spec in books.items() if spec["substrate"] != "measured_B"]
    h1_path = mdir / "primary_rho_acdc_test.json"
    if ds == "ACDC":
        outputs.append(mdir / "multiplicity_F1.parquet")
    elif ds == "MMS2":
        from .mms2_stages import evaluation_outputs as mms2_evaluation_outputs

        outputs += mms2_evaluation_outputs(mdir)
    else:
        outputs += [mdir / "primary_rho_mms_external_never_analysed.json", mdir / "fixed_sequence.json",
                    mdir / "strata_errors_mms_external.parquet", mdir / "strata_routes_mms_external.parquet",
                    mdir / "strata_rho_mms_external.json", mdir / "multiplicity_F3.parquet"]

    def compute() -> dict[str, Any]:
        import joblib

        from .formal_features import formal_descriptor_names

        study = ctx.study
        split_manifest = _split_manifest(ctx, ds if ds == "MMS2" else None)
        ids = split_manifest[split_manifest["partition"] == split].sort_values("sample_id")
        a = pd.read_parquet(run.path / "formal" / f"A_{ds}.parquet").set_index("sample_id")
        b = pd.read_parquet(run.path / "semantics" / f"B_{ds}.parquet").set_index("sample_id")
        keep = [s for s in ids["sample_id"] if s in a.index and s in b.index]
        clusters = _cluster_ids(split_manifest, keep)   # bootstrap clusters (declared duplicate-image pairs share one)
        x = a.loc[keep, formal_descriptor_names()].to_numpy(dtype=np.float64)
        finite = np.isfinite(x).all(axis=1)
        model = model_cls.from_dict(json.loads((run.path / "transitions" / "transition_model.json").read_text(encoding="utf-8")))
        bhat = model.predict(x)
        bhat.index = keep
        _write_table(bhat.reset_index(names="sample_id"), outputs[0])
        # Decoders (9.5) and the outer-fold mappers (11.4): the routing substrate; raw bhat stays the error substrate.
        decoded, dstates, fold_decoded = decode_reconstructions(model, load_fold_models(ctx, run), x, pd.Index(keep, name="sample_id"),
                                                                study.registry, study.transition)
        targets = study.transition_targets()
        _write_table(decoded[targets].join(dstates[targets].add_suffix("__state")).reset_index(names="sample_id"), outputs[4])
        atomic_write_json(outputs[5], {"split": split, "rule": study.transition["decoders"]["domain_guard"]["rule"],
                                       "targets": _decoder_log(bhat, decoded, dstates, targets)})
        predictor = joblib.load(run.path / "formal" / "d_model_predictor.joblib")
        d_model = pd.Series(predict_labels(predictor, x), index=keep, name="d_model")
        _write_table(pd.DataFrame({"sample_id": keep, "d_model": d_model.to_numpy()}), outputs[1])
        reference_sd, mean = _dev_reference(ctx)
        errors = per_target(b.loc[keep, targets], bhat[targets], reference_sd)
        _write_table(errors.reset_index() if errors.index.name else errors, outputs[3])
        rho = _rho(ctx, b.loc[keep, targets], bhat, mean, reference_sd, clusters=clusters)
        rho.update({"n_formal_nonfinite": int((~finite).sum()), "hypothesis_id": hypothesis_id(study, split), "partition": split})
        atomic_write_json(outputs[2], rho, allow_nan=True)
        cohort = _cohort(ctx, ds).set_index("sample_id")
        reference = reference_labels(cohort, ds).loc[keep]
        qc = qc_route_frame(b.loc[keep], finite, study)
        candidates = study.rule_candidates()
        routes_by_rb: dict[str, pd.DataFrame] = {}
        for rb, spec in books.items():
            book_path = run.path / "rules" / rb / "rulebook.json"
            records = json.loads(book_path.read_text(encoding="utf-8"))["records"] if book_path.is_file() else []
            cuts = cutset_cls.from_dict(json.loads((run.path / "thresholds" / f"{rb}_{profile}.json").read_text(encoding="utf-8")))
            if spec["substrate"] == "measured_B":
                substrate = b.loc[keep]
            else:
                substrate, stability = routing_substrate(decoded, dstates, fold_decoded, cuts, study.wedd, candidates)
                _write_table(stability.reset_index(names="sample_id"), rdir / f"stability_{rb}_{split}.parquet")
            routes = route_records(records, rb, cuts, substrate, qc, study)
            _write_table(routes, idir / f"routes_{rb}_{split}.parquet")
            metrics = route_metrics(routes, reference, d_model, cluster=_row_clusters(routes, clusters) if len(routes) == len(keep) else None)
            metrics.update({"rulebook": rb, "split": split, "reference": "acdc_class" if ds == "ACDC" else "crosswalk_same_definition_only"})
            atomic_write_json(mdir / f"routes_{rb}_{split}.json", metrics, allow_nan=True)
            routes_by_rb[rb] = routes
        y_null = pd.DataFrame([mean[targets]] * len(keep), index=keep)
        q = float(multiplicity["q"])
        if ds == "ACDC":
            f1 = family_name("F1", families)
            tests = paired_error_tests(b.loc[keep, targets], bhat[targets], y_null, targets, reference_sd, family=f1, partition=split,
                                       comparison="mean_vs_ridge_per_target", cluster=clusters.loc[keep].to_numpy(dtype=object))
            _write_table(bh_family_table(tests, q=q, families=families), mdir / "multiplicity_F1.parquet", csv=True)
            return {"rows": len(keep), "bundle_sha256": frozen["bundle_sha256"], "results": [split]}
        if ds == "MMS2":     # protocol-v6.0.1: H-primary-3 (its own one-test family), strata and the M&Ms-2 secondary family
            from .mms2_stages import evaluate_mms2_tail

            return evaluate_mms2_tail(ctx, split=split, b=b, bhat=bhat, keep=keep, targets=targets, reference_sd=reference_sd, mean=mean,
                                      y_null=y_null, clusters=clusters, routes_by_rb=routes_by_rb, reference=reference, d_model=d_model,
                                      rho=rho, cohort=cohort, mdir=mdir, frozen=frozen)
        # mms_external: the 220-case sensitivity and the fixed-sequence decision (configs/bootstrap.yaml primary_test).
        never_id = "mms_external_never_analysed"
        never = set(ids.loc[ids["never_analysed"].astype(bool), "sample_id"])
        sub = [s for s in keep if s in never]
        rho_never = _rho(ctx, b.loc[sub, targets], bhat.loc[sub], mean, reference_sd, clusters=clusters)
        rho_never.update({"hypothesis_id": hypothesis_id(study, never_id), "partition": never_id, "in_hierarchy": False})
        atomic_write_json(mdir / "primary_rho_mms_external_never_analysed.json", rho_never, allow_nan=True)
        h1 = json.loads(h1_path.read_text(encoding="utf-8")) if h1_path.is_file() else {
            "partition": "acdc_test", "ci_lower": None, "status": "NOT_EVALUATED", "reason": "evaluate --split acdc_test not recorded"}
        h1.setdefault("hypothesis_id", hypothesis_id(study, "acdc_test"))
        sesoi = float(study.bootstrap["primary_test"]["smallest_effect_of_interest"])
        sequence = fixed_sequence([h1, rho, rho_never], sesoi)
        atomic_write_json(mdir / "fixed_sequence.json", {"results": sequence, "sesoi": sesoi,
                                                         "hierarchy": study.bootstrap["primary_test"]["hierarchy"],
                                                         "sensitivity": study.bootstrap["primary_test"].get("sensitivity", [])}, allow_nan=True)
        # Strata (vendor, centre, release split, pathology, MMS_100 membership, never-analysed, seen vs unseen vendors).
        min_cases = int(study.bootstrap["subgroup_reporting_thresholds"]["min_cases_per_stratum"])
        status = below_threshold_status(study)
        f3 = family_name("F3", families)
        # Prespecified strata are tested within F3; the descriptive demographic strata (sex, age band; Appendix A Q28)
        # report counts, point estimates and intervals only, never p-values or F3 tests.
        strata = {**{k: ("prespecified", v) for k, v in external_strata(study, cohort, keep).items()},
                  **{k: ("descriptive_demographic", v) for k, v in descriptive_strata(study, cohort, keep).items()}}
        error_rows, route_rows, rho_strata = [], [], {}
        tests = [paired_error_tests(b.loc[keep, targets], bhat[targets], y_null, targets, reference_sd, family=f3, partition=split,
                                    comparison="mean_vs_ridge_per_target", cluster=clusters.loc[keep].to_numpy(dtype=object))]
        for var, (role, series) in strata.items():
            descriptive = role != "prespecified"
            evaluated = "DESCRIPTIVE" if descriptive else "EVALUATED"
            rho_strata[var] = {}
            for value in sorted(series.unique()):
                rows = [s for s in keep if series.loc[s] == value]
                below = len(rows) < min_cases
                e = per_target(b.loc[rows, targets], bhat.loc[rows, targets], reference_sd)
                if below or descriptive:
                    e.loc[:, [c for c in ("pearson_p", "spearman_p") if c in e.columns]] = np.nan
                e.insert(0, "stratum", value)
                e.insert(0, "stratum_variable", var)
                e["stratum_role"] = role
                e["stratum_n"] = len(rows)
                e["stratum_clusters"] = int(clusters.loc[rows].nunique())
                e["descriptive_only"] = below or descriptive
                e["inference_status"] = status["status"] if below else evaluated
                e["inference_reason"] = status["reason"] if below else None
                error_rows.append(e)
                rho_strata[var][value] = {**_rho(ctx, b.loc[rows, targets], bhat.loc[rows], mean, reference_sd, min_cases=min_cases, clusters=clusters),
                                          "stratum_role": role}
                if not descriptive:
                    tests.append(paired_error_tests(b.loc[rows, targets], bhat.loc[rows, targets], y_null.loc[rows], targets, reference_sd,
                                                    family=f3, partition=split, comparison="mean_vs_ridge_per_target", stratum_variable=var,
                                                    stratum=value, min_cases=min_cases, cluster=clusters.loc[rows].to_numpy(dtype=object)))
            for rb, routes in routes_by_rb.items():
                r = stratified_route_metrics(routes, reference.to_numpy(), d_model.to_numpy(), series.loc[keep].to_numpy(), min_cases=min_cases,
                                             cluster=clusters.loc[keep].to_numpy(dtype=object))
                r.insert(0, "stratum_variable", var)
                r.insert(0, "rulebook", rb)
                r["stratum_role"] = role
                r["routes"] = [json.dumps(v, sort_keys=True) for v in r["routes"]]
                r["inference_status"] = np.where(r["descriptive_only"], status["status"], evaluated)
                r["inference_reason"] = [status["reason"] if d else None for d in r["descriptive_only"]]
                route_rows.append(r)
        _write_table(pd.concat(error_rows, ignore_index=True), mdir / "strata_errors_mms_external.parquet", csv=True)
        _write_table(pd.concat(route_rows, ignore_index=True), mdir / "strata_routes_mms_external.parquet", csv=True)
        atomic_write_json(mdir / "strata_rho_mms_external.json", {"min_cases_per_stratum": min_cases, "below_threshold": status,
                                                                  "descriptive_strata": [k for k, (role, _) in strata.items() if role != "prespecified"],
                                                                  "strata": rho_strata}, allow_nan=True)
        _write_table(bh_family_table(pd.concat(tests, ignore_index=True), q=q, families=families), mdir / "multiplicity_F3.parquet", csv=True)
        return {"rows": len(keep), "bundle_sha256": frozen["bundle_sha256"], "results": [split, never_id]}

    inputs = [run.frozen_path, run.path / "formal" / f"A_{ds}.parquet", run.path / "semantics" / f"B_{ds}.parquet",
              _contracts(ctx) / f"cohort_{ds}.parquet"]
    if ds == "MMS" and h1_path.is_file():
        inputs.append(h1_path)
    if ds == "MMS2":
        inputs.append(_contracts(ctx) / "split_manifest_MMS2.parquet")
    return _stage(ctx, f"evaluate:{split}", compute, inputs, outputs, {"split": split}, status="EVALUATED")


# ----------------------------------------------------------------------------- shift audit (Section 12.3; after M0)


def audit_shift(ctx: StageContext, dataset: str = "mms"):
    """Label-free shift audit: out-of-fold AUROC of a two-sample logistic regression on Z (formal A standardized with the
    frozen acdc_dev mean and SD) between ``acdc_dev`` and ``mms_external``, pooled and per vendor
    (``configs/bootstrap.yaml`` ``shift_audit``).  It reads M&Ms A, so it is guarded like every M&Ms endpoint stage
    (M0 tag or the synthetic override) and requires the frozen bundle.  The classifier reads domain membership only
    (no B, no Y), is cross-fitted by a hash rule within domain, is never part of the bundle and never selects anything.
    Per-target external bias of the frozen transition is in ``metrics/per_target_mms_external.parquet`` and, by vendor
    and centre, in ``metrics/strata_errors_mms_external.parquet``.  ``dataset="mms2"`` (protocol-v6.0.1) runs the same
    audit on ``acdc_dev`` versus ``mms2_external`` under the M&Ms-2 lock (record ``audit-shift:MMS2``, outputs
    ``metrics/shift_audit_mms2_external.*`` and ``metrics/shift_audit_oof_scores_mms2_external.parquet``).
    """

    from .classification import domain_shift_audit
    from .formal_features import formal_descriptor_names

    cmp_ds = canonical_dataset(dataset)
    if cmp_ds == "MMS2":
        require_mms2_unlocked(ctx, "audit-shift --dataset mms2")
        require_mms2_validated(ctx, "audit-shift --dataset mms2")
    elif cmp_ds == "MMS":
        require_mms_unlocked(ctx, "audit-shift")
    else:
        raise ValueError("audit-shift --dataset must be mms or mms2")
    frozen = verify_frozen_bundle(ctx)
    run = ctx.require_run("audit-shift")
    model_cls = methods_api("tmmri.transition_fit", "TransitionModel")
    cfg = ctx.study.bootstrap["shift_audit"]
    if cmp_ds == "MMS2":
        cfg = {**cfg, "comparison_partition": MMS2_SPLIT}
    mdir = run.sub("metrics")
    outputs = [mdir / "shift_audit_mms_external.json", mdir / "shift_audit_mms_external.parquet", mdir / "shift_audit_oof_scores.parquet"]
    if cmp_ds == "MMS2":
        outputs = [mdir / f"shift_audit_{MMS2_SPLIT}.json", mdir / f"shift_audit_{MMS2_SPLIT}.parquet",
                   mdir / f"shift_audit_oof_scores_{MMS2_SPLIT}.parquet"]

    def compute() -> dict[str, Any]:
        study = ctx.study
        split = _split_manifest(ctx)
        if cmp_ds == "MMS2":
            split = pd.concat([split, _split_manifest(ctx, "MMS2")], ignore_index=True)
        names = formal_descriptor_names()
        model = model_cls.from_dict(json.loads((run.path / "transitions" / "transition_model.json").read_text(encoding="utf-8")))

        def usable_rows(ds: str, partition: str) -> tuple[list[str], np.ndarray, dict[str, int]]:
            a = pd.read_parquet(run.path / "formal" / f"A_{ds}.parquet").set_index("sample_id")
            b = pd.read_parquet(run.path / "semantics" / f"B_{ds}.parquet").set_index("sample_id")
            ids = [s for s in sorted(split.loc[split["partition"] == partition, "sample_id"]) if s in a.index and s in b.index]
            x = a.loc[ids, names].to_numpy(dtype=float)
            nonfinite = ~np.isfinite(x).all(axis=1)
            critical = critical_qc_mask(b.loc[ids], study).to_numpy(dtype=bool)
            ok = ~nonfinite & ~critical
            return ([ids[i] for i in np.flatnonzero(ok)], x[ok],
                    {"n_partition": len(ids), "n_excluded_critical_qc": int(critical.sum()), "n_excluded_nonfinite": int(nonfinite.sum()), "n_used": int(ok.sum())})

        ref_ids, x_ref, ref_counts = usable_rows("ACDC", str(cfg["reference_partition"]))
        cmp_ids, x_cmp, cmp_counts = usable_rows(cmp_ds, str(cfg["comparison_partition"]))
        z_ref, z_cmp = model.standardizer.transform(x_ref), model.standardizer.transform(x_cmp)
        boot = study.bootstrap["fixed_model_bootstrap"]
        min_cases = int(study.bootstrap["subgroup_reporting_thresholds"]["min_cases_per_stratum"])
        vendor = _cohort(ctx, cmp_ds).set_index("sample_id").loc[cmp_ids, "vendor"].astype(str).to_numpy()
        groups: list[tuple[str, str, np.ndarray]] = []
        if "pooled" in cfg["comparisons"]:
            groups.append(("pooled", "all", np.arange(len(cmp_ids))))
        if "vendor" in cfg["comparisons"]:
            groups += [("vendor", str(v), np.flatnonzero(vendor == v)) for v in sorted(set(vendor.tolist()))]
        results, scores = [], []
        for comparison, group, pos in groups:
            record, table = domain_shift_audit(z_ref, z_cmp[pos], ref_ids, [cmp_ids[i] for i in pos], cfg, n_boot=int(boot["replicates"]),
                                               seed=int(boot["seed"]), min_cases=min_cases, reference_clusters=list(_cluster_ids(split, ref_ids)),
                                               comparison_clusters=list(_cluster_ids(split, [cmp_ids[i] for i in pos])))
            results.append({"comparison": comparison, "group": group, **record})
            scores.append(table.assign(comparison=comparison, group=group))
        _write_table(pd.DataFrame(results), outputs[1], csv=True)
        _write_table(pd.concat(scores, ignore_index=True), outputs[2])
        atomic_write_json(outputs[0], {
            "schema": "tmmri.shift_audit/1", "role": cfg["role"], "label_free": True, "metric": cfg["metric"], "input": cfg["input"],
            "reference_partition": cfg["reference_partition"], "comparison_partition": cfg["comparison_partition"],
            "counts": {"reference": ref_counts, "comparison": cmp_counts}, "results": results, "bundle_sha256": frozen["bundle_sha256"],
            "per_target_external_bias": "metrics/per_target_mms_external.parquet (bias); metrics/strata_errors_mms_external.parquet (by vendor and centre)"
            if cmp_ds == "MMS" else f"metrics/per_target_{MMS2_SPLIT}.parquet (bias); metrics/strata_errors_{MMS2_SPLIT}.parquet (by stratum)"},
            allow_nan=True)
        return {"comparisons": len(results)}

    inputs = [run.frozen_path, run.path / "formal" / "A_ACDC.parquet", run.path / "formal" / f"A_{cmp_ds}.parquet", run.path / "semantics" / "B_ACDC.parquet",
              run.path / "semantics" / f"B_{cmp_ds}.parquet", _contracts(ctx) / "split_manifest.parquet", _contracts(ctx) / f"cohort_{cmp_ds}.parquet"]
    if cmp_ds == "MMS2":
        inputs.append(_contracts(ctx) / "split_manifest_MMS2.parquet")
    name = "audit-shift" if cmp_ds == "MMS" else "audit-shift:MMS2"
    return _stage(ctx, name, compute, inputs, outputs, {"comparisons": list(cfg["comparisons"])})


# ----------------------------------------------------------------------------- refit arm (Decision D17)


def _refit_min_class(study: StudyConfig) -> int:
    text = str(study.splits["refit_arm"]["predictor_classes"])
    match = re.search(r"at least (\d+)", text)
    if not match:
        raise ValueError("configs/splits.yaml refit_arm.predictor_classes must state 'at least <n>' training cases")
    return int(match.group(1))


def _require_external_recorded(run: RunRoot) -> None:
    record = run.read_record("evaluate:mms_external")
    needed = [run.path / "metrics" / "primary_rho_mms_external.json", run.path / "metrics" / "primary_rho_mms_external_never_analysed.json"]
    if record is None or record.status != "EVALUATED" or not all(p.is_file() for p in needed):
        raise BundleError("fit-refit-arm runs only after the primary external evaluation (H-primary-2 and the 220-case "
                          "sensitivity) is recorded (configs/splits.yaml refit_arm.ordering)")


def _mean_oof(b: pd.DataFrame, targets: list[str], folds: np.ndarray) -> pd.DataFrame:
    """Out-of-fold train-mean predictor (the null comparator) on the fit rows of ``b``."""

    from .target_transforms import observed_values

    out = pd.DataFrame(np.nan, index=b.index, columns=targets, dtype=float)
    for f in sorted({int(v) for v in folds}):
        held = folds == f
        for t in targets:
            x = observed_values(b.loc[~held], t)
            x = x[np.isfinite(x)]
            out.loc[held, t] = float(np.mean(x)) if x.size else np.nan
    return out


def fit_refit_arm(ctx: StageContext):
    """D17 refit ceiling: the v6 pipeline refitted on ``mms_refit_train``, evaluated on ``mms_refit_eval``.

    QC outlier references from ``mms_refit_train``; transition with lambda by CV over the refit folds and nested OOF
    reconstructions (inner CV over the remaining refit folds); predictor on the same-definition crosswalk classes with
    at least the configured number of training cases; WEDD cuts, bootstrap stability gate, eligibility, VPRS discovery
    and the cross-fitted gate over the refit folds (nested inner-OOF substrate); routes on ``mms_refit_eval``; the
    fixed-model bootstrap of rho overall and for seen (A, B) / unseen (C, D) vendors beside the frozen ACDC bundle on the
    same cases (identical case order and bootstrap seed, so the draws are paired); and the downstream refit bootstrap
    of the rulebooks.  Every output lives under ``*/refit_arm/``, is never part of the primary bundle and is never
    pooled with the primary arm.
    """

    from .formal_features import formal_descriptor_names
    from .rule_pipeline import (NestedGateSubstrate, decode_reconstructions, downstream_refit_bootstrap, fit_rulebook, qc_route_frame,
                                remaining_fold_labels, routing_substrate, rulebook_payload, route_records)

    require_mms_unlocked(ctx, "fit-refit-arm")
    run = ctx.require_run("fit-refit-arm")
    _require_external_recorded(run)
    fit_ref = methods_api("tmmri.semantics.qc", "fit_outlier_reference")
    apply_ref = methods_api("tmmri.semantics.qc", "apply_outlier_reference")
    fit_v6 = methods_api("tmmri.transition_fit", "fit_transition_v6")
    nested = methods_api("tmmri.transition_fit", "nested_oof_transition")
    fit_predictor_fn = methods_api("tmmri.classification", "fit_formal_predictor")
    per_target = methods_api("tmmri.metrics", "per_target_errors")
    route_metrics = methods_api("tmmri.metrics", "route_metrics")
    arm = "refit_arm"
    d = {k: run.sub(k) / arm for k in ("qc", "semantics", "formal", "transitions", "oof", "reconstructed", "thresholds", "rules", "inference", "metrics")}
    for path in d.values():
        path.mkdir(parents=True, exist_ok=True)
    books = _rulebooks(ctx)
    profile = str(ctx.study.wedd["primary_profile"])
    outputs = [d["qc"] / "outlier_reference.json", d["semantics"] / "B_MMS_refit.parquet", d["transitions"] / "transition_model.json",
               d["transitions"] / "lambda_selection.parquet", d["oof"] / "bhat_oof_mms_refit_train.parquet", d["formal"] / "d_model_predictor.joblib",
               d["formal"] / "d_model_mms_refit.parquet", d["reconstructed"] / "bhat_mms_refit_eval.parquet",
               d["metrics"] / "per_target_mms_refit_eval.parquet", d["metrics"] / "primary_rho_mms_refit_eval.json",
               d["rules"] / "downstream_refit_bootstrap.parquet", d["metrics"] / "refit_arm_summary.json"]
    outputs += [p for rb in books for p in (d["thresholds"] / f"{rb}_{profile}.json", d["thresholds"] / f"{rb}_{profile}_stability.parquet",
                                             d["rules"] / rb / "eligibility.parquet", d["rules"] / rb / "gate.parquet",
                                             d["rules"] / rb / "rulebook_or_empty.json", d["inference"] / f"routes_{rb}_mms_refit_eval.parquet",
                                             d["metrics"] / f"routes_{rb}_mms_refit_eval.json")]

    def compute() -> dict[str, Any]:
        import joblib

        study = ctx.study
        split = _split_manifest(ctx).set_index("sample_id")
        cohort = _cohort(ctx, "MMS").set_index("sample_id")
        names = formal_descriptor_names()
        targets = study.transition_targets()
        a = pd.read_parquet(run.path / "formal" / "A_MMS.parquet").set_index("sample_id")
        b_raw = pd.read_parquet(run.path / "semantics" / "B_MMS.parquet")
        have = set(a.index) & set(b_raw["sample_id"])
        train_all = sorted(s for s in split.index[split["refit_partition"] == "mms_refit_train"] if s in have)
        eval_ids = sorted(s for s in split.index[split["refit_partition"] == "mms_refit_eval"] if s in have)
        # QC outlier references from mms_refit_train only (configs/cohorts.yaml refit_arm_reference_partition).
        reference_qc = fit_ref(b_raw, study.registry, study.cohorts, fit_mask=b_raw["sample_id"].isin(train_all).to_numpy(copy=True),
                               reason_codes=study.reason_codes)
        atomic_write_json(outputs[0], reference_qc)
        b_refit = apply_ref(b_raw, reference_qc, study.cohorts, reason_codes=study.reason_codes)
        _write_table(b_refit, outputs[1])
        b = b_refit.set_index("sample_id")
        critical = set(b.index[critical_qc_mask(b, study).to_numpy()])
        finite_ids = set(a.index[np.isfinite(a[names].to_numpy(dtype=float)).all(axis=1)])
        train = [s for s in train_all if s not in critical and s in finite_ids]
        clusters = _cluster_ids(split, split.index)   # bootstrap clusters (declared duplicate-image pairs share one)
        folds = split.loc[train, "refit_fold"].to_numpy(dtype=float)
        a_tr = a.loc[train, names].to_numpy(dtype=float)
        b_tr = b.loc[train].copy().reset_index()
        cfg, registry = study.transition, study.registry
        model = fit_v6(a_tr, b_tr, targets, cfg, registry, folds)
        atomic_write_json(outputs[2], model.to_dict(), allow_nan=True)
        oof, details = nested(a_tr, b_tr, targets, cfg, registry, folds, lambda f, pos: folds[np.asarray(pos, dtype=int)], return_details=True)
        _write_table(details["lambda_selection"], outputs[3])
        oof.index = train
        _write_table(oof.reset_index(names="sample_id"), outputs[4])
        mean_oof = _mean_oof(b.loc[train], targets, folds)
        # Predictor: same-definition crosswalk classes with >= n training cases; others excluded from predictor fitting.
        labels_all = reference_labels(cohort, "MMS")
        y_tr = labels_all.loc[train]
        counts = y_tr.dropna().astype(str).value_counts()
        min_class = _refit_min_class(study)
        classes = sorted(str(c) for c, n in counts.items() if int(n) >= min_class)
        y_fit = y_tr.where(y_tr.astype(str).isin(classes)).to_numpy(dtype=object)
        pcfg = study.formal["audited_predictor"]
        predictor = fit_predictor_fn(a_tr, pd.Series(y_fit), pcfg)
        joblib.dump(predictor, outputs[5])
        d_train = pd.Series(predict_labels(predictor, a_tr), index=train, dtype=object)
        x_ev = a.loc[eval_ids, names].to_numpy(dtype=float)
        finite_ev = np.isfinite(x_ev).all(axis=1)
        d_eval = pd.Series(predict_labels(predictor, x_ev), index=eval_ids, dtype=object)
        _write_table(pd.DataFrame({"sample_id": train + eval_ids, "refit_partition": ["mms_refit_train"] * len(train) + ["mms_refit_eval"] * len(eval_ids),
                                   "d_model_refit": list(d_train.to_numpy()) + list(d_eval.to_numpy())}), outputs[6])
        predictors: dict[tuple[int, ...], Any] = {}

        def d_model_fn_for(pos: np.ndarray) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
            def fn(tr: np.ndarray, target: np.ndarray) -> np.ndarray:
                key = tuple(int(i) for i in pos[tr])
                if key not in predictors:
                    predictors[key] = fit_predictor_fn(a_tr[pos[tr]], pd.Series(y_fit[pos[tr]]), pcfg)
                return predict_labels(predictors[key], a_tr[pos[target]])
            return fn

        def label_fn_for(pos: np.ndarray) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
            labels = y_tr.to_numpy(dtype=object)[pos]
            return lambda tr, target: labels[target]

        candidates = [c for c in study.rule_candidates() if c in oof.columns]
        anchors = pd.Series([s.split(":", 1)[1] for s in train], index=train)
        substrate_fn = NestedGateSubstrate(a_tr, b_tr, targets, cfg, registry, folds, remaining_fold_labels(folds), oof, candidates, train)
        fitted: dict[str, dict[str, Any]] = {}
        summary_books = {}
        # Evaluation on mms_refit_eval.
        bhat_ev = model.predict(x_ev)
        bhat_ev.index = eval_ids
        _write_table(bhat_ev.reset_index(names="sample_id"), outputs[7])
        # Routing substrate: decoders (9.5) and the refit-arm outer-fold mappers for per-object stability (11.4).
        dec_ev, st_ev, fold_ev = decode_reconstructions(model, [details["models"][f] for f in sorted(details["models"])], x_ev,
                                                        pd.Index(eval_ids, name="sample_id"), registry, cfg)
        qc_ev = qc_route_frame(b.loc[eval_ids], finite_ev, study)
        reference_ev = labels_all.loc[eval_ids]
        for rb, spec in books.items():
            measured = spec["substrate"] == "measured_B"
            is_label = spec["consequent"] == "Y_acdc_class"
            # The dataset-label rulebook's universe is the labelled (same-definition) cases; its transitions still use
            # every fit row (IHD and off-ontology cases are kept for transition fitting; configs/splits.yaml).
            pos = np.flatnonzero(y_tr.notna().to_numpy()) if is_label else np.arange(len(train))
            rows = [train[i] for i in pos]
            if measured:
                cols = candidates + [f"{c}__state" for c in candidates if f"{c}__state" in b.columns]
                values = b.loc[rows, cols]
            else:
                values = oof.loc[rows, candidates]
            decision = y_tr.loc[rows].astype(object) if is_label else d_train.loc[rows]
            fit = fit_rulebook(rulebook_id=rb, values=values, decision=decision, anchors=anchors.loc[rows], folds=folds[pos], profile=profile,
                               study=study, decision_fn=label_fn_for(pos) if is_label else d_model_fn_for(pos),
                               substrate_fn=None if measured else substrate_fn.with_subset(pos), reference=b.loc[rows],
                               availability=b.loc[train_all], oof=None if measured else oof.loc[rows],
                               mean_oof=None if measured else mean_oof.loc[rows], n_jobs=ctx.jobs, clusters=clusters)
            fitted[rb] = fit
            atomic_write_json(d["thresholds"] / f"{rb}_{profile}.json", fit["cuts"].to_dict(), allow_nan=True)
            _write_table(fit["stability"], d["thresholds"] / f"{rb}_{profile}_stability.parquet")
            _write_table(fit["eligibility"], d["rules"] / rb / "eligibility.parquet")
            _write_table(fit["gate"], d["rules"] / rb / "gate.parquet")
            name, payload = rulebook_payload(rb, fit["gated"], fit["gate"], {"arm": arm, "fit_partition": "mms_refit_train",
                                                                             "gate_substrate": "measured_B" if measured else "nested_inner_oof"})
            for stale in ("rulebook.json", "EMPTY.json"):
                (d["rules"] / rb / stale).unlink(missing_ok=True)
            atomic_write_json(d["rules"] / rb / name, payload, allow_nan=True)
            atomic_write_json(d["rules"] / rb / "rulebook_or_empty.json", {"rulebook": rb, "file": name, "eligible_rules": len(payload.get("records", []))})
            substrate = b.loc[eval_ids] if measured else routing_substrate(dec_ev, st_ev, fold_ev, fit["cuts"], study.wedd, candidates)[0]
            routes = route_records(payload.get("records", []), rb, fit["cuts"], substrate, qc_ev, study)
            _write_table(routes, d["inference"] / f"routes_{rb}_mms_refit_eval.parquet")
            metrics = route_metrics(routes, reference_ev, d_eval, cluster=_row_clusters(routes, clusters) if len(routes) == len(eval_ids) else None)
            metrics.update({"rulebook": rb, "arm": arm, "reference": "crosswalk_same_definition_only", "d_model": "refit_arm_predictor"})
            atomic_write_json(d["metrics"] / f"routes_{rb}_mms_refit_eval.json", metrics, allow_nan=True)
            summary_books[rb] = {"eligible_features": fit["attributes"], "induced_rules": len(fit["induced"].rules),
                                 "eligible_rules": len(payload.get("records", []))}
        # Continuous semantics: refit arm vs the frozen ACDC bundle on the same cases (paired draws).
        ref_sd, ref_mean = b.loc[train, targets].std(ddof=1), b.loc[train, targets].mean()
        errors = per_target(b.loc[eval_ids, targets], bhat_ev[targets], ref_sd)
        _write_table(errors.reset_index() if errors.index.name else errors, outputs[8])
        acdc_sd, acdc_mean = _dev_reference(ctx)
        frozen_bhat = pd.read_parquet(run.path / "reconstructed" / "bhat_mms_external.parquet").set_index("sample_id")
        strata_cfg = study.splits["refit_arm"]["eval_strata"]
        vendor = cohort.loc[eval_ids, "vendor"].astype(str)
        min_cases = int(study.bootstrap["subgroup_reporting_thresholds"]["min_cases_per_stratum"])
        strata = {"all": eval_ids, "seen_vendors": [s for s in eval_ids if vendor[s] in {str(v) for v in strata_cfg["seen_vendors"]}],
                  "unseen_vendors": [s for s in eval_ids if vendor[s] in {str(v) for v in strata_cfg["unseen_vendors"]}]}
        rho = {"schema": "tmmri.refit_arm_rho/1", "role": "required_secondary_ceiling", "confirmatory": False, "pooled_with_primary": False,
               "paired": "identical case order and bootstrap seed for both arms", "strata_vendors": {k: v for k, v in strata_cfg.items()},
               "refit_arm": {}, "frozen_acdc_bundle": {}}
        for name, rows in strata.items():
            stratum_min = None if name == "all" else min_cases
            rho["refit_arm"][name] = _rho(ctx, b.loc[rows, targets], bhat_ev.loc[rows], ref_mean, ref_sd, min_cases=stratum_min, clusters=clusters)
            present = [s for s in rows if s in frozen_bhat.index]
            rho["frozen_acdc_bundle"][name] = _rho(ctx, b.loc[present, targets], frozen_bhat.loc[present], acdc_mean, acdc_sd, min_cases=stratum_min,
                                                   clusters=clusters)
        atomic_write_json(outputs[9], rho, allow_nan=True)
        # Downstream refit bootstrap of the rulebooks (configs/bootstrap.yaml downstream_refit_bootstrap).
        dr = study.bootstrap["downstream_refit_bootstrap"]
        specs = {rb: {"substrate": "measured" if spec["substrate"] == "measured_B" else "reconstructed",
                      "decision": (y_tr.to_numpy(dtype=object) if spec["consequent"] == "Y_acdc_class" else d_train.to_numpy(dtype=object)),
                      "attributes": fitted[rb]["attributes"], "reference": fitted[rb]["induced"], "cuts": fitted[rb]["cuts"]} for rb, spec in books.items()}
        table = downstream_refit_bootstrap(A=a_tr, B=b_tr, folds=folds, inner_labels=remaining_fold_labels(folds), row_ids=train,
                                           study=study, rulebooks=specs, profile=profile, n_replicates=int(dr["replicates"]), seed=int(dr["seed"]),
                                           n_jobs=ctx.jobs, clusters=list(clusters.loc[train]))
        _write_table(table, outputs[10])
        summary = {"schema": "tmmri.refit_arm_summary/1", "arm": arm, "role": "required_secondary_ceiling", "pooled_with_primary": False,
                   "n_train_partition": len(train_all), "n_train_fit_rows": len(train), "n_train_critical_or_nonfinite": len(train_all) - len(train),
                   "n_eval": len(eval_ids), "predictor_classes": classes, "predictor_min_class_cases": min_class,
                   "class_counts_same_definition": {str(k): int(v) for k, v in counts.items()}, "rulebooks": summary_books,
                   "downstream_refit_bootstrap": {"replicates": int(dr["replicates"]), "seed": int(dr["seed"]),
                                                  "failed_replicates": table.attrs.get("failed_replicates", [])}}
        atomic_write_json(outputs[11], summary, allow_nan=True)
        return {"train": len(train), "eval": len(eval_ids)}

    inputs = [run.path / "formal" / "A_MMS.parquet", run.path / "semantics" / "B_MMS.parquet", _contracts(ctx) / "split_manifest.parquet",
              _contracts(ctx) / "cohort_MMS.parquet", run.path / "metrics" / "primary_rho_mms_external.json",
              run.path / "metrics" / "primary_rho_mms_external_never_analysed.json", run.path / "reconstructed" / "bhat_mms_external.parquet",
              run.path / "semantics" / "B_ACDC.parquet"]
    return _stage(ctx, "fit-refit-arm", compute, inputs, outputs, {"arm": arm, "profile": profile})


# ----------------------------------------------------------------------------- audited-predictor summary (Section 13.2; read-only)


NOT_EXACT = "HARMONIZABLE_NOT_EXACT"               # registry label_tables.mms_to_acdc: IHD -> MINF, reported separately
OFF_ONTOLOGY = "OFF_ONTOLOGY"
COMPARATOR_NOT_FITTED = "COMPARATOR_NOT_FITTED_IN_RUN"
OOF_PROBABILITIES_NOT_RECORDED = "OOF_PROBABILITIES_NOT_RECORDED"


def _require_evaluated(run: RunRoot, stage: str, requester: str) -> Any:
    record = run.read_record(stage)
    if record is None or record.status != "EVALUATED":
        raise BundleError(f"{requester} runs only after {stage} is recorded (.stages/{run.record_path(stage).name}); run it first")
    return record


def _check_recorded_output(run: RunRoot, record: Any, path: Path, requester: str) -> None:
    """``path`` must still carry the hash its producing stage recorded (the summaries read recorded artifacts only)."""

    if record is None:
        return
    key = run.portable(path)
    if key in record.outputs and record.outputs[key] != hash_path(path):
        raise BundleError(f"{requester}: {key} changed after {record.stage} recorded it; use a new run id")


def _predictor_probabilities(predictor: Any, x: np.ndarray) -> pd.DataFrame:
    """Class probabilities of a frozen predictor on recorded formal A (no fitting); rows with a non-finite descriptor
    have none, exactly as :func:`predict_labels` gives them no decision."""

    proba = predictor.predict_proba(x).reset_index(drop=True)
    proba.loc[~np.isfinite(np.asarray(x, dtype=float)).all(axis=1)] = np.nan
    return proba


def summarize_predictor(ctx: StageContext):
    """Audited-predictor metrics of SPECIFICATION.md 13.2 from recorded artifacts only (never fits or re-evaluates).

    Partitions: ``acdc_dev`` cross-fitted out-of-fold decisions (``formal/d_model_crossfit_ACDC_dev.parquet``; the fold
    predictors are not recorded, so AUROC, ECE and log loss are ``NE`` with ``OOF_PROBABILITIES_NOT_RECORDED``),
    ``acdc_test`` and ``mms_external`` (the recorded ``inference/d_model_<split>.parquet``, with the class probabilities of
    the hash-verified frozen predictor on the recorded formal A, whose argmax must reproduce the recorded decisions), the
    latter restricted to the crosswalk same-definition cases (IHD -> MINF counted separately as ``HARMONIZABLE_NOT_EXACT``,
    never pooled; off-ontology cases only as the distribution of ``d_model``), and the refit-arm predictor on
    ``mms_refit_eval`` when its predictions are recorded.  Critical-QC rows (``QC_ABSTAIN``) are excluded and counted.
    Intervals: the fixed-model cluster bootstrap of ``configs/bootstrap.yaml`` (``cluster_id``; deterministic for any
    ``n_jobs``); ECE with the configured equal-mass bins and the realized bin counts.  The measured-B semantic
    comparator is never fitted in a run: it is reported ``NE`` (``COMPARATOR_NOT_FITTED_IN_RUN``).
    Requires ``.stages/FROZEN.json`` and the ``evaluate`` record of each summarized partition; the M&Ms part obeys
    :func:`require_mms_unlocked`.
    """

    from .classification import label_distribution, predictor_summary_v6
    from .formal_features import formal_descriptor_names
    from .metrics import wilson_interval

    run = ctx.require_run("summarize-predictor")
    if not run.frozen_path.is_file():
        raise BundleError(f"summarize-predictor: run {run.run_id} has no frozen analysis bundle (run freeze-analysis-bundle first)")
    test_record = _require_evaluated(run, "evaluate:acdc_test", "summarize-predictor")
    mms_record = run.read_record("evaluate:mms_external")
    include_mms = mms_record is not None and mms_record.status == "EVALUATED"
    if include_mms:
        require_mms_unlocked(ctx, "summarize-predictor (mms_external)")
    frozen = verify_frozen_bundle(ctx)
    p = run.path
    refit_files = [p / "formal" / "refit_arm" / "d_model_mms_refit.parquet", p / "formal" / "refit_arm" / "d_model_predictor.joblib",
                   p / "semantics" / "refit_arm" / "B_MMS_refit.parquet"]
    refit_record = run.read_record("fit-refit-arm") if include_mms else None
    include_refit = refit_record is not None and all(f.is_file() for f in refit_files)
    mdir = run.sub("metrics")
    outputs = [mdir / "predictor_acdc_dev_oof.json", mdir / "predictor_acdc_test.json", mdir / "predictor_confusion.parquet",
               mdir / "predictor_summary.json"]
    inputs = [run.frozen_path, run.record_path("evaluate:acdc_test"), p / "formal" / "d_model_predictor.joblib",
              p / "formal" / "d_model_crossfit_ACDC_dev.parquet", p / "formal" / "A_ACDC.parquet", p / "semantics" / "B_ACDC.parquet",
              p / "inference" / "d_model_acdc_test.parquet", _contracts(ctx) / "cohort_ACDC.parquet", _contracts(ctx) / "split_manifest.parquet"]
    if include_mms:
        outputs.append(mdir / "predictor_mms_external.json")
        inputs += [run.record_path("evaluate:mms_external"), p / "formal" / "A_MMS.parquet", p / "semantics" / "B_MMS.parquet",
                   p / "inference" / "d_model_mms_external.parquet", _contracts(ctx) / "cohort_MMS.parquet"]
    if include_refit:
        outputs.append(run.sub("metrics") / "refit_arm" / "predictor_mms_refit_eval.json")
        inputs += [run.record_path("fit-refit-arm"), *refit_files]
    partitions = ["acdc_dev_oof", "acdc_test"] + (["mms_external"] if include_mms else []) + (["mms_refit_eval"] if include_refit else [])
    fit_record = run.read_record("fit-predictor")
    _check_recorded_output(run, fit_record, p / "formal" / "d_model_crossfit_ACDC_dev.parquet", "summarize-predictor")
    _check_recorded_output(run, test_record, p / "inference" / "d_model_acdc_test.parquet", "summarize-predictor")
    if include_mms:
        _check_recorded_output(run, mms_record, p / "inference" / "d_model_mms_external.parquet", "summarize-predictor")
    if include_refit:
        for f in refit_files:
            _check_recorded_output(run, refit_record, f, "summarize-predictor")

    def compute() -> dict[str, Any]:
        import joblib

        study = ctx.study
        boot, cal = study.bootstrap["fixed_model_bootstrap"], study.bootstrap["calibration"]
        if str(cal.get("ece_binning", "equal_mass")) != "equal_mass":
            raise ValueError("configs/bootstrap.yaml calibration.ece_binning must be equal_mass (SPECIFICATION.md 13.2)")
        kw = dict(n_boot=int(boot["replicates"]), seed=int(boot["seed"]), n_bins=int(cal["ece_bins"]),
                  min_valid_fraction=float(boot["min_valid_fraction"]), n_jobs=ctx.jobs)
        pcfg = study.formal["audited_predictor"]
        configured = [str(c) for c in pcfg["classes"]]
        ccfg = study.formal.get("comparator_predictor") or {}
        comparator = {"name": str(ccfg.get("name", "measured_b_logistic_semantic_comparator")), "role": str(ccfg.get("role", "semantic_comparator_not_audited")),
                      "status": "NE", "reason": COMPARATOR_NOT_FITTED,
                      "note": "configs/formal.yaml comparator_predictor is not fitted by any stage of this run and is never fitted here"}
        split = _split_manifest(ctx)
        names = formal_descriptor_names()
        predictor = joblib.load(p / "formal" / "d_model_predictor.joblib")
        confusion_rows: list[dict[str, Any]] = []
        base = {"schema": "tmmri.predictor_summary/1", "predictor": str(pcfg["name"]), "bundle_sha256": frozen["bundle_sha256"],
                "semantic_comparator": comparator, "metrics_definition": "SPECIFICATION.md 13.2 Audited predictor; configs/bootstrap.yaml "
                "classification_metrics, calibration, fixed_model_bootstrap", "qc_rule": "critical-QC rows (QC_ABSTAIN) are excluded and counted"}

        def partition_ids(partition: str, recorded: pd.Index, column: str = "partition") -> tuple[list[str], int]:
            members = sorted(split.loc[split[column] == partition, "sample_id"].astype(str))
            return [s for s in members if s in recorded], len(members)

        def without_critical(ids: list[str], b: pd.DataFrame) -> tuple[list[str], int]:
            crit = set(b.index[critical_qc_mask(b, study).to_numpy(dtype=bool)])
            kept = [s for s in ids if s not in crit]
            return kept, len(ids) - len(kept)

        def summary(partition: str, truth: pd.Series, decisions: pd.Series, proba: pd.DataFrame | None, classes: list[str], ids: list[str],
                    extra_counts: dict[str, int], **extra: Any) -> dict[str, Any]:
            result = predictor_summary_v6(truth.reindex(ids).to_numpy(dtype=object), decisions.reindex(ids).to_numpy(dtype=object), proba, classes,
                                          clusters=_cluster_ids(split, ids).to_numpy(dtype=object), **kw, **extra)
            result["counts"] = {**extra_counts, **result["counts"]}
            if partition != "mms_refit_eval" and result.get("companions"):
                cm = result["companions"]["confusion_matrix"]
                for i, ref in enumerate(cm["rows_reference"]):
                    for j, pred in enumerate(cm["columns_d_model"]):
                        confusion_rows.append({"partition": partition, "reference_class": ref, "d_model": pred, "n": int(cm["counts"][i][j])})
            return {**base, "partition": partition, **result}

        # acdc_dev: cross-fitted out-of-fold decisions (fold predictors refitted without the held-out fold; not recorded).
        cohort_acdc = _cohort(ctx, "ACDC").set_index("sample_id")
        b_acdc = pd.read_parquet(p / "semantics" / "B_ACDC.parquet").set_index("sample_id")
        cross = pd.read_parquet(p / "formal" / "d_model_crossfit_ACDC_dev.parquet").set_index("sample_id")
        ids, n_part = partition_ids("acdc_dev", cross.index)
        kept, n_crit = without_critical(ids, b_acdc)
        dev = summary("acdc_dev_oof", cohort_acdc["acdc_class"].astype(object), cross["d_model_crossfit"].astype(object), None, configured, kept,
                      {"n_partition": n_part, "n_recorded_predictions": len(ids), "n_excluded_critical_qc": n_crit},
                      probability_reason=OOF_PROBABILITIES_NOT_RECORDED)
        dev.update({"predictions": "formal/d_model_crossfit_ACDC_dev.parquet (d_model_crossfit)", "reference": "acdc_class",
                    "probability_source": None, "note": "fit-predictor records the cross-fitted decisions only; the fold predictors "
                    "and their probabilities are not part of the run, so AUROC, ECE and log loss are NE (never refitted here)"})
        atomic_write_json(outputs[0], dev, allow_nan=True)
        # acdc_test: recorded d_model, probabilities of the frozen predictor on the recorded formal A.
        a_acdc = pd.read_parquet(p / "formal" / "A_ACDC.parquet").set_index("sample_id")
        d_test = pd.read_parquet(p / "inference" / "d_model_acdc_test.parquet").set_index("sample_id")["d_model"].astype(object)
        ids, n_part = partition_ids("acdc_test", d_test.index)
        kept, n_crit = without_critical(ids, b_acdc)
        test = summary("acdc_test", cohort_acdc["acdc_class"].astype(object), d_test,
                       _predictor_probabilities(predictor, a_acdc.loc[kept, names].to_numpy(dtype=np.float64)), list(predictor.classes), kept,
                       {"n_partition": n_part, "n_recorded_predictions": len(ids), "n_excluded_critical_qc": n_crit})
        probability_source = ("formal/d_model_predictor.joblib (frozen bundle, hash-verified) applied to the recorded formal/A_<DATASET>.parquet; "
                              "its argmax reproduces the recorded d_model on every row")
        test.update({"predictions": "inference/d_model_acdc_test.parquet (d_model)", "reference": "acdc_class", "probability_source": probability_source})
        atomic_write_json(outputs[1], test, allow_nan=True)
        index: dict[str, Any] = {"acdc_dev_oof": "metrics/predictor_acdc_dev_oof.json", "acdc_test": "metrics/predictor_acdc_test.json"}
        not_included: dict[str, str] = {}
        if include_mms:
            cohort_mms = _cohort(ctx, "MMS").set_index("sample_id")
            b_mms = pd.read_parquet(p / "semantics" / "B_MMS.parquet").set_index("sample_id")
            a_mms = pd.read_parquet(p / "formal" / "A_MMS.parquet").set_index("sample_id")
            d_mms = pd.read_parquet(p / "inference" / "d_model_mms_external.parquet").set_index("sample_id")["d_model"].astype(object)
            ids, n_part = partition_ids("mms_external", d_mms.index)
            compat = cohort_mms.loc[ids, "crosswalk_compatibility"].astype(str)
            groups = {g: [s for s in ids if compat[s] == g] for g in (SAME_DEFINITION, NOT_EXACT, OFF_ONTOLOGY)}
            unmapped = [s for s in ids if compat[s] not in groups]
            kept = {g: without_critical(v, b_mms) for g, v in groups.items()}
            same_ids = kept[SAME_DEFINITION][0]
            ext = summary("mms_external", reference_labels(cohort_mms, "MMS").astype(object), d_mms,
                          _predictor_probabilities(predictor, a_mms.loc[same_ids, names].to_numpy(dtype=np.float64)), list(predictor.classes), same_ids,
                          {"n_partition": n_part, "n_recorded_predictions": len(ids), "n_same_definition": len(groups[SAME_DEFINITION]),
                           "n_excluded_critical_qc": kept[SAME_DEFINITION][1]})
            crosswalk = study.registry["label_tables"]["mms_to_acdc"]
            not_exact_ids = kept[NOT_EXACT][0]
            mapped = sorted({str(v) for v in cohort_mms.loc[groups[NOT_EXACT], "crosswalk_acdc_class"].dropna()})
            hits = int(sum(str(d_mms[s]) in mapped for s in not_exact_ids if d_mms[s] is not None and pd.notna(d_mms[s])))
            lo, hi = wilson_interval(hits, len(not_exact_ids)) if not_exact_ids else (None, None)
            codes = sorted({str(v.get("reason_code")) for v in crosswalk["crosswalk"].values() if v.get("compatibility") == NOT_EXACT and v.get("reason_code")})
            off_ids = kept[OFF_ONTOLOGY][0]
            ext.update({
                "predictions": "inference/d_model_mms_external.parquet (d_model)", "probability_source": probability_source,
                "reference": "crosswalk_same_definition_only (registry label_tables.mms_to_acdc HARMONIZABLE_SAME_DEFINITION)",
                "crosswalk": {"n_partition": n_part, "found": {g: len(v) for g, v in groups.items()}, "unmapped": len(unmapped),
                              "excluded_critical_qc": {g: kept[g][1] for g in groups}, "expected_from_registry": crosswalk.get("counts_mms_external"),
                              "evaluable_same_definition": len(same_ids)},
                "ihd_not_exact": {"compatibility": NOT_EXACT, "reason_codes": codes, "mapped_acdc_class": mapped, "pooled": False,
                                  "status": "REPORTED_SEPARATELY", "n": len(not_exact_ids), "n_excluded_critical_qc": kept[NOT_EXACT][1],
                                  "d_model_distribution": label_distribution(d_mms.reindex(not_exact_ids).to_numpy(dtype=object), configured),
                                  "n_d_model_equals_mapped_class": hits, "proportion_d_model_equals_mapped_class": hits / len(not_exact_ids) if not_exact_ids else None,
                                  "wilson_95": [lo, hi], "note": "never pooled into the crosswalk-evaluable metrics"},
                "off_ontology": {"compatibility": OFF_ONTOLOGY, "n": len(off_ids), "n_excluded_critical_qc": kept[OFF_ONTOLOGY][1],
                                 "d_model_distribution": label_distribution(d_mms.reindex(off_ids).to_numpy(dtype=object), configured),
                                 "reporting": "distribution of d_model only; no reference, no accuracy"}})
            atomic_write_json(mdir / "predictor_mms_external.json", ext, allow_nan=True)
            index["mms_external"] = "metrics/predictor_mms_external.json"
        else:
            not_included["mms_external"] = "evaluate:mms_external not recorded"
        if include_refit:
            refit_predictor = joblib.load(refit_files[1])
            refit = pd.read_parquet(refit_files[0]).set_index("sample_id")
            b_refit = pd.read_parquet(refit_files[2]).set_index("sample_id")
            ids, n_part = partition_ids("mms_refit_eval", refit.index[refit["refit_partition"] == "mms_refit_eval"], column="refit_partition")
            kept_refit, n_crit = without_critical(ids, b_refit)
            arm = summary("mms_refit_eval", reference_labels(cohort_mms, "MMS").astype(object), refit["d_model_refit"].astype(object),
                          _predictor_probabilities(refit_predictor, a_mms.loc[kept_refit, names].to_numpy(dtype=np.float64)), list(refit_predictor.classes),
                          kept_refit, {"n_partition": n_part, "n_recorded_predictions": len(ids), "n_excluded_critical_qc": n_crit})
            arm.update({"arm": "refit_arm", "role": "required_secondary_ceiling", "pooled_with_primary": False, "predictor": "refit_arm_predictor",
                        "predictions": "formal/refit_arm/d_model_mms_refit.parquet (d_model_refit)", "reference": "crosswalk_same_definition_only",
                        "probability_source": "formal/refit_arm/d_model_predictor.joblib (fit-refit-arm record, hash-verified) applied to the recorded "
                        "formal/A_MMS.parquet; its argmax reproduces the recorded d_model_refit", "qc_rule": "critical QC of semantics/refit_arm/B_MMS_refit.parquet"})
            atomic_write_json(outputs[-1], arm, allow_nan=True)
            index["mms_refit_eval"] = "metrics/refit_arm/predictor_mms_refit_eval.json"
        elif include_mms:
            not_included["mms_refit_eval"] = "fit-refit-arm predictions not recorded"
        _write_table(pd.DataFrame(confusion_rows, columns=["partition", "reference_class", "d_model", "n"]), outputs[2])
        atomic_write_json(outputs[3], {"schema": "tmmri.predictor_summary_index/1", "partitions": index, "not_included": not_included,
                                       "semantic_comparator": comparator, "bundle_sha256": frozen["bundle_sha256"], "read_only": True,
                                       "fitting": "none (recorded decisions; frozen predictors applied to recorded formal A)"})
        return {"partitions": list(index)}

    return _stage(ctx, "summarize-predictor", compute, inputs, outputs, {"partitions": partitions}, status="EVALUATED")


# ----------------------------------------------------------------------------- ablation matrix (Section 13.3; secondary)


def _policy_variant(policy: Mapping[str, Any], *, beta: float | None = None, max_length: int | None = None) -> dict[str, Any]:
    import copy

    out = copy.deepcopy(dict(policy))
    if beta is not None:
        out["discovery"]["tiers"]["primary"]["beta"] = float(beta)
        out["discovery"]["tiers"]["primary"]["min_confidence"] = 1.0 - float(beta)
    if max_length is not None:
        out["discovery"]["max_length"] = int(max_length)
    return out


def wedd_settings_variants(study: StudyConfig) -> list[tuple[str, dict[str, Any]]]:
    """(name, profile overrides) of the WEDD settings grid (``configs/wedd.yaml`` ``sensitivity_grid``), one factor at a
    time around the primary profile; the primary value of a factor is not repeated.  ``weight_entropy`` w sets the
    density weight to 1 - w; ``max_depth`` values are evaluated with ``max_depth_evaluated_with`` (four bins)."""

    from .wedd import resolve_profile

    grid = study.wedd.get("sensitivity_grid") or {}
    primary = resolve_profile(str(study.wedd["primary_profile"]), study.wedd)
    out: list[tuple[str, dict[str, Any]]] = []
    for w in grid.get("weight_entropy", []):
        if float(w) != float(primary["weight_entropy"]):
            out.append((f"weight_entropy_{float(w):.2f}", {"weight_entropy": float(w), "weight_density": round(1.0 - float(w), 12)}))
    for m in grid.get("bandwidth_multiplier", []):
        if float(m) != float(primary.get("bandwidth_multiplier", 1.0)):
            out.append((f"bandwidth_multiplier_{float(m):g}", {"bandwidth_multiplier": float(m)}))
    for nb in grid.get("max_bins", []):
        if int(nb) != int(primary["max_bins"]):
            out.append((f"max_bins_{int(nb)}", {"max_bins": int(nb)}))
    coupled = {str(k): v for k, v in (grid.get("max_depth_evaluated_with") or {}).items()}
    for depth in grid.get("max_depth", []):
        overrides = {"max_depth": int(depth), **coupled}
        if all(primary.get(k) == v for k, v in overrides.items()):
            continue
        out.append(("max_depth_" + str(int(depth)) + "".join(f"_{k}_{v}" for k, v in sorted(coupled.items())), overrides))
    return out


def _wedd_cfg_variant(wedd_cfg: Mapping[str, Any], profile: str, overrides: Mapping[str, Any]) -> dict[str, Any]:
    import copy

    cfg = copy.deepcopy(dict(wedd_cfg))
    cfg["profiles"][profile] = {**cfg["profiles"][profile], **dict(overrides)}
    return cfg


def linearity_criterion(study: StudyConfig, y: pd.DataFrame, reference: pd.DataFrame, comparator: pd.DataFrame, sd: pd.Series,
                        folds: np.ndarray, cluster: pd.Series, resample_cluster: pd.Series | None = None) -> dict[str, Any]:
    """The frozen linearity criterion of SPECIFICATION.md 9.4 (``configs/transition.yaml`` ``linearity_criterion``) on the
    acdc_dev OOF ledger: the nonlinear comparator is *better* only if its pooled fit-SD-normalized MAE is at least the
    configured relative reduction below the reference in the required number of outer folds and no primary-family
    target degrades with a paired patient-bootstrap CI above zero.  Reported either way; the audit pipeline stays affine."""

    from .metrics import paired_nmae_difference, pooled_nmae_by_fold
    from .rule_pipeline import k_of_n

    lc = study.transition["linearity_criterion"]
    need, _ = k_of_n(lc["folds_required"])
    minimum = float(lc["min_relative_reduction"])
    targets = study.transition_targets()
    ref = pooled_nmae_by_fold(y, reference, targets, sd, folds)
    cmp_ = pooled_nmae_by_fold(y, comparator, targets, sd, folds)
    per_fold = {}
    for f in sorted(ref):
        r, c = ref[f], cmp_.get(f, float("nan"))
        rel = 1.0 - c / r if np.isfinite(r) and r > 0 and np.isfinite(c) else float("nan")
        per_fold[str(f)] = {"reference_pooled_nmae": r, "comparator_pooled_nmae": c, "relative_reduction": rel,
                            "meets_min_relative_reduction": bool(np.isfinite(rel) and rel >= minimum)}
    meets = sum(v["meets_min_relative_reduction"] for v in per_fold.values())
    boot = study.bootstrap["fixed_model_bootstrap"]
    diff = paired_nmae_difference(y, reference, comparator, study.primary_family(), sd, n_boot=int(boot["replicates"]), seed=int(boot["seed"]),
                                  cluster=cluster, resample_cluster=None if resample_cluster is None else resample_cluster.reindex(y.index).to_numpy(dtype=object))
    degraded = [str(r.target) for r in diff.itertuples(index=False) if np.isfinite(r.ci_lower) and float(r.ci_lower) > 0]
    return {"comparator": lc["comparator"], "reference": lc["reference"], "metric": lc["metric"], "min_relative_reduction": minimum,
            "folds_required": need, "per_fold": per_fold, "folds_meeting": int(meets),
            "primary_family_nmae_difference": diff.to_dict("records"), "degraded_primary_family_targets": degraded,
            "nonlinear_declared_better": bool(meets >= need and not degraded), "outcome": lc["outcome"],
            "executed_audit_pipeline": lc["executed_audit_pipeline"]}


def run_ablations(ctx: StageContext):
    """The Section 13.3 ablation matrix on ACDC, each output a labelled secondary result (never a primary output).

    * ``transition_variants.json`` (family F2 and the named comparators): typed ridge on the 57 descriptors (primary)
      versus the 128-d P6 projection, identity target transforms, the pseudoinverse, one global lambda and the RBF kernel
      ridge, against the mean predictor, on the acdc_dev OOF ledger and on acdc_test (variants refitted on acdc_dev
      only; paired bootstrap draws); ``linearity_criterion.json`` (the frozen criterion of Section 9.4);
    * ``discretizers.json`` (F4): ``wedd_gaussian_mm_v1`` (primary, reused) versus ``three_quantile_v1`` and
      ``wedd_uniform_mm_v1`` for the primary ``d_model`` rulebook (cuts, stability, eligibility, discovery, gate,
      acdc_test routes); ``wedd_settings.json`` (F4): the ``configs/wedd.yaml`` ``sensitivity_grid``, one factor at a time;
    * ``rule_policy.json`` (F5): VPRS beta, strict-only and maximum-length sensitivities of
      ``configs/rule_policy.yaml`` ``discovery.sensitivities`` (discovery and gate refitted; acdc_test routes);
    * ``gate_thresholds.json`` (F5): the Wilson lower-bound sensitivities ``validation_gate.gate_sensitivities``;
    * ``multiplicity_F2/F4/F5.parquet``: the per-target (F2) and per-variant (F4, F5) tests of ``configs/bootstrap.yaml``
      ``multiplicity.tests`` with Benjamini-Hochberg q-values within each family;
    * ``downstream_refit_bootstrap.parquet``: rulebook stability over the configured downstream refit replicates.

    acdc_test routes use the same substrate as ``evaluate`` (decoders and per-object stability under each variant's cuts).
    """

    from .formal_features import formal_descriptor_names
    from .metrics import bh_family_table, family_name, paired_error_tests, variant_test_row
    from .rule_pipeline import (NestedGateSubstrate, acdc_inner_labels, decode_reconstructions, downstream_refit_bootstrap, fit_rulebook,
                                gate_and_apply, qc_route_frame, routing_substrate, rulebook_payload, route_records)

    run = ctx.require_run("run-ablations")
    record = run.read_record("evaluate:acdc_test")
    if record is None or record.status != "EVALUATED":
        raise BundleError("run-ablations runs after the primary acdc_test evaluation is recorded (evaluate --split acdc_test --no-fit)")
    verify_frozen_bundle(ctx)
    fit_v6 = methods_api("tmmri.transition_fit", "fit_transition_v6")
    fit_global = methods_api("tmmri.transition_fit", "fit_transition_global_lambda_v6")
    fit_rbf = methods_api("tmmri.transition_fit", "fit_rbf_kernel_ridge_v6")
    model_cls = methods_api("tmmri.transition_fit", "TransitionModel")
    per_target = methods_api("tmmri.metrics", "per_target_errors")
    route_metrics = methods_api("tmmri.metrics", "route_metrics")
    cutset_cls = methods_api("tmmri.wedd", "CutSet")
    rulebook_cls = methods_api("tmmri.rough_sets", "Rulebook")
    induce = methods_api("tmmri.rough_sets", "induce_vprs_rules")
    fit_predictor_fn = methods_api("tmmri.classification", "fit_formal_predictor")
    mdir = run.sub("metrics") / "ablations"
    rdir = run.sub("rules") / "ablations"
    for path in (mdir, rdir):
        path.mkdir(parents=True, exist_ok=True)
    profile = str(ctx.study.wedd["primary_profile"])
    outputs = [mdir / "transition_variants.json", mdir / "transition_variants_per_target.parquet", mdir / "discretizers.json",
               mdir / "rule_policy.json", mdir / "gate_thresholds.json", mdir / "downstream_refit_bootstrap.parquet",
               mdir / "wedd_settings.json", mdir / "linearity_criterion.json", mdir / "multiplicity_F2.parquet",
               mdir / "multiplicity_F4.parquet", mdir / "multiplicity_F5.parquet", mdir / "ablation_index.json"]
    label = {"role": "secondary_ablation", "replaces_primary": False, "selection_forbidden": "no post-test best-configuration claim"}

    def compute() -> dict[str, Any]:
        study = ctx.study
        targets = study.transition_targets()
        split = _split_manifest(ctx)
        multiplicity = study.bootstrap["multiplicity"]
        families = [str(f) for f in multiplicity["families"]]
        q = float(multiplicity["q"])
        f2, f4, f5 = (family_name(k, families) for k in ("F2", "F4", "F5"))
        ids, a, b, folds = _dev_matrices(ctx)
        row_ids = ids["sample_id"].astype(str).to_numpy()
        ids_p6, a_p6, _, _ = _dev_matrices(ctx, "p6")
        assert list(ids_p6["sample_id"]) == list(row_ids), "P6 and formal A rows must align"
        sd, mean = _dev_reference(ctx)
        b_all = pd.read_parquet(run.path / "semantics" / "B_ACDC.parquet").set_index("sample_id")
        b_dev = b_all.loc[row_ids, targets]
        tdir, odir = run.path / "transitions", run.path / "oof"

        def _oof(path: Path) -> pd.DataFrame:
            return pd.read_parquet(path).set_index("sample_id").loc[row_ids]

        mean_oof = _oof(tdir / "comparator_mean.parquet")[targets]
        variants_dev = {"ridge_typed_formal57": _oof(odir / "bhat_oof_ACDC_dev.parquet"), "ridge_typed_p6_128": _oof(odir / "bhat_oof_ACDC_dev_p6.parquet"),
                        "ridge_identity_transform": _oof(tdir / "comparator_identity_transform_ridge.parquet"),
                        "pseudoinverse_p1": _oof(tdir / "comparator_pseudoinverse_p1.parquet")}
        named = [c for c in NAMED_COMPARATORS if c in (study.transition.get("comparators") or {})]
        for name in named:
            variants_dev[name] = _oof(tdir / f"comparator_{name}.parquet")
        # Test partition: variants refitted on all acdc_dev fit rows (lambda by CV over the outer folds).
        test_ids = [s for s in split.loc[split["partition"] == "acdc_test", "sample_id"].sort_values()]
        a_test = pd.read_parquet(run.path / "formal" / "A_ACDC.parquet").set_index("sample_id")
        a_test_p6 = pd.read_parquet(run.path / "formal" / "A_ACDC_p6_128.parquet").set_index("sample_id")
        test_ids = [s for s in test_ids if s in a_test.index and s in b_all.index]
        p6_cols = [c for c in a_test_p6.columns if c.startswith("a_")]
        x_test, x_test_p6 = a_test.loc[test_ids, formal_descriptor_names()].to_numpy(float), a_test_p6.loc[test_ids, p6_cols].to_numpy(float)
        primary_model = model_cls.from_dict(json.loads((tdir / "transition_model.json").read_text(encoding="utf-8")))
        models = {"ridge_typed_formal57": (primary_model, x_test),
                  "ridge_typed_p6_128": (fit_v6(a_p6, b, targets, study.transition, study.registry, folds, representation="p6_projection_128"), x_test_p6),
                  "ridge_identity_transform": (fit_v6(a, b, targets, study.transition, study.registry, folds, identity_transforms=True), x_test),
                  "pseudoinverse_p1": (fit_v6(a, b, targets, study.transition, study.registry, None, lambdas={t: 0.0 for t in targets}), x_test)}
        if "ridge_global_lambda" in named:
            models["ridge_global_lambda"] = (fit_global(a, b, targets, study.transition, study.registry, folds), x_test)
        if "rbf_kernel_ridge" in named:
            models["rbf_kernel_ridge"] = (fit_rbf(a, b, targets, study.transition, study.registry, folds), x_test)
        per_rows, trans = [], {**label, "family": f2, "named_comparators": named, "null_comparator": "mean", "acdc_dev_oof": {}, "acdc_test": {}}
        for name, oof_v in variants_dev.items():
            trans["acdc_dev_oof"][name] = _rho(ctx, b_dev, oof_v, mean, sd)
            per_rows.append(per_target(b_dev, oof_v[targets], sd).assign(variant=name, partition="acdc_dev_oof"))
        trans["acdc_dev_oof"]["mean_oof_reference"] = "transitions/comparator_mean.parquet (out-of-fold null)"
        test_preds: dict[str, pd.DataFrame] = {}
        for name, (m, x) in models.items():
            pred = m.predict(x)
            pred.index = test_ids
            test_preds[name] = pred
            trans["acdc_test"][name] = _rho(ctx, b_all.loc[test_ids, targets], pred, mean, sd)
            per_rows.append(per_target(b_all.loc[test_ids, targets], pred[targets], sd).assign(variant=name, partition="acdc_test"))
        trans["global_lambda"] = models["ridge_global_lambda"][0].lambda_selection if "ridge_global_lambda" in models else None
        atomic_write_json(outputs[0], trans, allow_nan=True)
        per = pd.concat([f.reset_index() if f.index.name else f for f in per_rows], ignore_index=True)
        _write_table(per, outputs[1])
        # F2: per-target paired tests of the primary transition versus the configured comparators on acdc_test.
        members = (multiplicity.get("family_members") or {}).get(f2, {})
        primary_name = str(members.get("primary", "ridge_typed_formal57"))
        f2_rows = [paired_error_tests(b_all.loc[test_ids, targets], test_preds[primary_name], test_preds[str(c)], targets, sd, family=f2,
                                      partition="acdc_test", comparison=f"{c}_vs_{primary_name}",
                                      cluster=_cluster_ids(split, test_ids).to_numpy(dtype=object)) for c in members.get("comparators", [])]
        _write_table(bh_family_table(pd.concat(f2_rows, ignore_index=True), q=q, families=families), outputs[8], csv=True)
        # Linearity criterion (Section 9.4) on the acdc_dev OOF ledger.
        if "rbf_kernel_ridge" in variants_dev and "linearity_criterion" in study.transition:
            lin = linearity_criterion(study, b_dev, variants_dev["ridge_typed_formal57"][targets], variants_dev["rbf_kernel_ridge"][targets], sd,
                                      ids["outer_fold"].to_numpy(dtype=float), pd.Series([s.split(":", 1)[1] for s in row_ids], index=row_ids),
                                      resample_cluster=_cluster_ids(split, row_ids))
        else:
            lin = {"status": "not_configured"}
        atomic_write_json(outputs[7], {**label, "partition": "acdc_dev_oof", **lin}, allow_nan=True)
        # Shared inputs of the rule ablations (primary d_model rulebook).
        books = _rulebooks(ctx)
        spec = books["d_model"]
        values, decision = _rule_substrate(ctx, spec)
        subset = np.flatnonzero(pd.Index(row_ids).isin(values.index))
        gate_ids = row_ids[subset]
        labels_all = _cohort(ctx, "ACDC").set_index("sample_id").loc[row_ids, "acdc_class"].astype(str).to_numpy()
        pcfg = study.formal["audited_predictor"]
        cache: dict[tuple[int, ...], Any] = {}

        def d_model_fn(tr: np.ndarray, target: np.ndarray) -> np.ndarray:
            key = tuple(int(i) for i in tr)
            if key not in cache:
                cache[key] = fit_predictor_fn(a[subset[tr]], pd.Series(labels_all[subset[tr]]), pcfg)
            return predict_labels(cache[key], a[subset[target]])

        oof = pd.read_parquet(odir / "bhat_oof_ACDC_dev.parquet").set_index("sample_id")
        candidates = [c for c in study.rule_candidates() if c in oof.columns]
        inner = pd.read_parquet(_contracts(ctx) / "inner_folds.parquet")
        substrate_fn = NestedGateSubstrate(a, b, targets, study.transition, study.registry, folds, acdc_inner_labels(inner, row_ids), oof,
                                           candidates, row_ids, subset=subset)
        substrate_fn.preload({f: pd.read_parquet(inner_oof_path(run, f)).set_index("sample_id")
                              for f in sorted({int(v) for v in folds}) if inner_oof_path(run, f).is_file()})
        bhat_test = pd.read_parquet(run.path / "reconstructed" / "bhat_acdc_test.parquet").set_index("sample_id")
        d_test = pd.read_parquet(run.path / "inference" / "d_model_acdc_test.parquet").set_index("sample_id")["d_model"]
        a_rows = a_test.loc[bhat_test.index, formal_descriptor_names()].to_numpy(float)
        qc_test = qc_route_frame(b_all.loc[bhat_test.index], np.isfinite(a_rows).all(axis=1), study)
        ref_test = _cohort(ctx, "ACDC").set_index("sample_id").loc[bhat_test.index, "acdc_class"]
        dec_t, st_t, fold_t = decode_reconstructions(primary_model, load_fold_models(ctx, run), a_rows, bhat_test.index, study.registry, study.transition)

        def test_metrics(records: list[dict[str, Any]], cuts: Any) -> tuple[dict[str, Any], np.ndarray]:
            substrate, _ = routing_substrate(dec_t, st_t, fold_t, cuts, study.wedd, candidates)
            routes = route_records(records, "d_model", cuts, substrate, qc_test, study)
            return route_metrics(routes, ref_test, d_test), (routes["route"].astype(str) == "EXACT_ACCEPT").to_numpy()

        folds_gate = folds[subset]
        anchors = pd.Series([s.split(":", 1)[1] for s in gate_ids], index=gate_ids)
        reference = b_all.loc[gate_ids]
        mean_gate = mean_oof.loc[gate_ids]
        availability = b_all.loc[[s for s in split.loc[split["partition"] == "acdc_dev", "sample_id"] if s in b_all.index]]
        primary_cuts = cutset_cls.from_dict(json.loads((run.path / "thresholds" / f"d_model_{profile}.json").read_text(encoding="utf-8")))
        primary_book_path = run.path / "rules" / "d_model" / "rulebook.json"
        primary_records = json.loads(primary_book_path.read_text(encoding="utf-8"))["records"] if primary_book_path.is_file() else []
        primary_stability = pd.read_parquet(run.path / "thresholds" / f"d_model_{profile}_stability.parquet")
        primary_routes, primary_accepted = test_metrics(primary_records, primary_cuts)
        f4_rows: list[dict[str, Any]] = []
        f5_rows: list[dict[str, Any]] = []

        def refit(prof: str, wedd_cfg: Mapping[str, Any] | None, tag: str) -> dict[str, Any]:
            fit = fit_rulebook(rulebook_id="d_model", values=values.loc[gate_ids], decision=decision.loc[gate_ids], anchors=anchors, folds=folds_gate,
                               profile=prof, study=study, decision_fn=d_model_fn, substrate_fn=substrate_fn, reference=reference,
                               availability=availability, oof=oof.loc[gate_ids], mean_oof=mean_gate, n_jobs=ctx.jobs, wedd_cfg=wedd_cfg,
                               clusters=_cluster_ids(split, gate_ids))
            _, payload = rulebook_payload("d_model", fit["gated"], fit["gate"], {"ablation": tag})
            atomic_write_json(rdir / f"d_model_{tag.replace(':', '_')}.json", payload, allow_nan=True)
            routes, accepted = test_metrics(payload.get("records", []), fit["cuts"])
            return {"source": "refitted", "eligible_features": fit["attributes"], "induced_rules": len(fit["induced"].rules),
                    "eligible_rules": len(payload.get("records", [])),
                    "mean_cut_recurrence": float(fit["stability"]["recurrence"].mean()) if len(fit["stability"]) else None,
                    "acdc_test_routes": routes, "_accepted": accepted}

        # Discretizers (F4).
        disc = {**label, "family": f4, "rulebook": "d_model", "profiles": {}}
        for prof in study.wedd["profiles"]:
            if prof == profile:
                entry = {"source": "primary artifacts", "eligible_features": eligible_attributes(ctx, "d_model"),
                         "eligible_rules": len(primary_records), "mean_cut_recurrence": float(primary_stability["recurrence"].mean()),
                         "acdc_test_routes": primary_routes}
            else:
                entry = refit(prof, None, f"discretizer:{prof}")
                f4_rows.append(variant_test_row(f4, "acdc_test", f"discretizer:{prof}", primary_accepted, entry.pop("_accepted")))
            disc["profiles"][prof] = entry
        atomic_write_json(outputs[2], disc, allow_nan=True)
        # WEDD settings grid (F4), one factor at a time around the primary profile (configs/wedd.yaml sensitivity_grid).
        grid = study.wedd.get("sensitivity_grid") or {}
        wedd_out = {**label, "family": f4, "rulebook": "d_model", "primary_profile": profile, "design": grid.get("design"), "variants": {}}
        for name, overrides in wedd_settings_variants(study):
            entry = refit(profile, _wedd_cfg_variant(study.wedd, profile, overrides), f"wedd_settings:{name}")
            f4_rows.append(variant_test_row(f4, "acdc_test", f"wedd_settings:{name}", primary_accepted, entry.pop("_accepted")))
            wedd_out["variants"][name] = {"overrides": overrides, **entry}
        atomic_write_json(outputs[6], wedd_out, allow_nan=True)
        # Rule policy (F5): beta, strict-only and maximum-length sensitivities.
        sens = study.rule_policy["discovery"].get("sensitivities") or {}
        states = pd.read_parquet(run.path / "granules" / f"d_model_{profile}_states_ACDC_dev.parquet").set_index("sample_id").loc[gate_ids]
        attributes = eligible_attributes(ctx, "d_model")
        variants: list[tuple[str, dict[str, Any]]] = [(f"beta_{float(bv):.2f}", _policy_variant(study.rule_policy, beta=float(bv))) for bv in sens.get("beta", [])]
        if sens.get("strict_only"):
            variants.append(("strict_only", _policy_variant(study.rule_policy, beta=0.0)))
        variants += [(f"max_length_{int(m)}", _policy_variant(study.rule_policy, max_length=int(m))) for m in sens.get("max_length", [])]
        policy_out = {**label, "family": f5, "rulebook": "d_model", "variants": {}}
        for name, pol in variants:
            book = induce(states, decision.loc[gate_ids], anchors, pol, registry=study.registry, attributes=attributes,
                          consequent_kind="d_model", rulebook_id="d_model")
            table, gated = gate_and_apply(values=values.loc[gate_ids], decision_fn=d_model_fn, folds=folds_gate, rule_policy=pol, book=book,
                                          profile=profile, wedd_cfg=study.wedd, anchors=anchors.to_numpy(dtype=object), substrate_fn=substrate_fn,
                                          registry=study.registry, rulebook_id="d_model", attributes=attributes)
            _, payload = rulebook_payload("d_model", gated, table, {"ablation": f"rule_policy:{name}"})
            atomic_write_json(rdir / f"d_model_policy_{name}.json", payload, allow_nan=True)
            routes, accepted = test_metrics(payload.get("records", []), primary_cuts)
            f5_rows.append(variant_test_row(f5, "acdc_test", f"rule_policy:{name}", primary_accepted, accepted))
            policy_out["variants"][name] = {"beta": pol["discovery"]["tiers"]["primary"]["beta"], "max_length": pol["discovery"]["max_length"],
                                            "induced_rules": len(book.rules), "eligible_rules": len(payload.get("records", [])),
                                            "acdc_test_routes": routes}
        atomic_write_json(outputs[3], policy_out, allow_nan=True)
        # Gate lower-bound sensitivities (F5): the primary gate table re-thresholded (no refit).
        from .rough_sets import apply_gate

        gate_table = pd.read_parquet(run.path / "rules" / "d_model" / "gate.parquet")
        induced = json.loads((run.path / "rules" / "d_model" / "rulebook_induced.json").read_text(encoding="utf-8"))
        induced_book = rulebook_cls.from_records(induced["records"], rulebook_id="d_model", consequent_kind="d_model")
        gates = {**label, "family": f5, "rulebook": "d_model", "thresholds": {}}
        for level in study.rule_policy["validation_gate"].get("gate_sensitivities", []):
            column = f"eligible_at_{float(level):.2f}"
            table = gate_table.copy()
            if len(table):
                table["eligibility"] = np.where(table[column].astype(bool), "ELIGIBLE", "INELIGIBLE_CROSSFIT_AGREEMENT")
            _, payload = rulebook_payload("d_model", apply_gate(induced_book, table), table, {"ablation": f"gate_lower_bound:{float(level):.2f}"})
            routes, accepted = test_metrics(payload.get("records", []), primary_cuts)
            f5_rows.append(variant_test_row(f5, "acdc_test", f"gate_lower_bound:{float(level):.2f}", primary_accepted, accepted))
            gates["thresholds"][f"{float(level):.2f}"] = {"eligible_rules": len(payload.get("records", [])), "acdc_test_routes": routes}
        atomic_write_json(outputs[4], gates, allow_nan=True)
        for rows, path in ((f4_rows, outputs[9]), (f5_rows, outputs[10])):
            frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["family", "p_value"])
            _write_table(bh_family_table(frame, q=q, families=families) if rows else frame, path, csv=True)
        # Downstream refit bootstrap of the three primary rulebooks on acdc_dev (Section 13.3, rulebook stability).
        dr = study.bootstrap["downstream_refit_bootstrap"]
        specs = {}
        for rb, rb_spec in books.items():
            rb_values, rb_decision = _rule_substrate(ctx, rb_spec)
            aligned = pd.Series(rb_decision.reindex(row_ids).to_numpy(dtype=object), index=row_ids)
            rb_induced = json.loads((run.path / "rules" / rb / "rulebook_induced.json").read_text(encoding="utf-8"))
            specs[rb] = {"substrate": "measured" if rb_spec["substrate"] == "measured_B" else "reconstructed",
                         "decision": aligned.to_numpy(dtype=object), "attributes": eligible_attributes(ctx, rb),
                         "reference": rulebook_cls.from_records(rb_induced["records"], rulebook_id=rb, consequent_kind=rb),
                         "cuts": cutset_cls.from_dict(json.loads((run.path / "thresholds" / f"{rb}_{profile}.json").read_text(encoding="utf-8")))}
        boot_table = downstream_refit_bootstrap(A=a, B=b, folds=folds, inner_labels=acdc_inner_labels(inner, row_ids), row_ids=row_ids, study=study,
                                                rulebooks=specs, profile=profile, n_replicates=int(dr["replicates"]), seed=int(dr["seed"]), n_jobs=ctx.jobs,
                                                clusters=list(_cluster_ids(split, row_ids)))
        _write_table(boot_table.assign(role="secondary_rulebook_stability"), outputs[5])
        index = {**label, "partition": "acdc_dev (fitting) and acdc_test (evaluation of frozen-protocol variants)",
                 "outputs": [p.name for p in outputs[:-1]], "downstream_refit_bootstrap": {"replicates": int(dr["replicates"]), "seed": int(dr["seed"]),
                                                                                           "failed_replicates": boot_table.attrs.get("failed_replicates", [])},
                 "multiplicity": {"method": multiplicity["method"], "q": q, "families": {"F2": f2, "F4": f4, "F5": f5},
                                  "tests": multiplicity.get("tests")},
                 "not_run": []}
        atomic_write_json(outputs[-1], index, allow_nan=True)
        return {"outputs": len(outputs)}

    books = _rulebooks(ctx)
    inputs = [run.frozen_path, run.path / "reconstructed" / "bhat_acdc_test.parquet", run.path / "inference" / "d_model_acdc_test.parquet",
              run.path / "formal" / "A_ACDC_p6_128.parquet"] + [run.path / "rules" / rb / "gate.parquet" for rb in books]
    return _stage(ctx, "run-ablations", compute, inputs, outputs, {"profile": profile})


# ----------------------------------------------------------------------------- equivariance (Decision D19; optional)


def run_equivariance(ctx: StageContext, dataset: str = "acdc"):
    """Optional equivariance/ETM diagnostics (``configs/equivariance.yaml``) on seeded acdc_dev cases.

    Formal and semantic generators (central differences of ``formal_descriptors_v3`` and of measured B on the
    transformed masks) and the frozen transition's reconstruction generator; diagnostic only, never a gate and never
    an input of the primary rulebook.  M&Ms equivariance is not run before the freeze M0 (``mms_policy``).
    """

    import copy

    from .formal_features import formal_descriptors_from_arrays

    ds = canonical_dataset(dataset)
    cfg = ctx.study.configs["equivariance"]
    if ds != "ACDC":
        if not protocol_frozen(ctx.study.freeze_tag):
            raise ProtocolFreezeError("run-equivariance --dataset mms: M&Ms equivariance only after the protocol freeze M0 "
                                      "(configs/equivariance.yaml mms_policy)")
        raise NotImplementedError("run-equivariance --dataset mms: configs/equivariance.yaml selects acdc_dev cases only")
    run = ctx.require_run("run-equivariance")
    select = methods_api("tmmri.equivariance", "select_equivariance_cases")
    diagnostics = methods_api("tmmri.equivariance", "equivariance_diagnostics_v6")
    model_cls = methods_api("tmmri.transition_fit", "TransitionModel")
    case_fn = methods_api("tmmri.semantics.features", "case_semantics_v6")
    edir = run.sub("equivariance")
    outputs = [edir / "equivariance_cases_ACDC.parquet", edir / "equivariance_targets_ACDC.parquet", edir / "equivariance_selection_ACDC.json"]

    def compute() -> dict[str, Any]:
        import functools

        study = ctx.study
        split = _split_manifest(ctx)
        cohort = _cohort(ctx, "ACDC").set_index("case_id")
        dev = split[split["partition"] == cfg["case_selection"]["partition"]]
        chosen = select(sorted(dev["patient_id"].astype(str)), cfg)
        targets = study.transition_targets()
        model = model_cls.from_dict(json.loads((run.path / "transitions" / "transition_model.json").read_text(encoding="utf-8")))
        by_patient = dev.set_index("patient_id")
        metadata = {pid: release_metadata("ACDC", {**cohort.loc[pid].to_dict(), "case_id": pid}, by_patient.loc[pid, "partition"]) for pid in chosen}
        loader = functools.partial(load_case_for_semantics, dataset="ACDC", staged_root=str(ctx.staged_root),
                                   label_map=study.label_map("acdc"), metadata=metadata)
        cases = [loader(pid) for pid in chosen]

        def semantic_fn(case: Any, ed_mask: np.ndarray, es_mask: np.ndarray) -> np.ndarray:
            moved = copy.copy(case)
            moved.ed_mask = np.asarray(ed_mask).astype(case.ed_mask.dtype)
            moved.es_mask = np.asarray(es_mask).astype(case.es_mask.dtype)
            row = case_fn(moved, study.registry, cohorts=study.cohorts, reason_codes=study.reason_codes)
            return np.array([row[t] if row.get(f"{t}__state") == "OBSERVED" and row[t] is not None else np.nan for t in targets], dtype=float)

        def predict_fn(rows: np.ndarray) -> np.ndarray:
            return model.predict(np.asarray(rows, dtype=float))[targets].to_numpy(dtype=float)

        case_table, target_table = diagnostics(cases, formal_descriptors_from_arrays, semantic_fn, predict_fn, cfg, target_names=targets)
        _write_table(case_table, outputs[0])
        _write_table(target_table, outputs[1])
        atomic_write_json(outputs[2], {"dataset": "ACDC", "partition": cfg["case_selection"]["partition"], "selected_patients": chosen,
                                       "seed": int(cfg["case_selection"]["seed"]), "n_cases": int(cfg["case_selection"]["n_cases"]),
                                       "feeds_primary_rulebook": False, "role": "optional_diagnostic_D19"})
        return {"cases": len(chosen)}

    inputs = [_contracts(ctx) / "split_manifest.parquet", _contracts(ctx) / "cohort_ACDC.parquet", run.path / "transitions" / "transition_model.json",
              _staging_manifest_path(ctx, "ACDC")]
    return _stage(ctx, "run-equivariance", compute, inputs, outputs, {"dataset": ds})


# ----------------------------------------------------------------------------- manuscript tables (Section 16; read-only)


def build_manuscript_tables(ctx: StageContext):
    """``manuscript_tables/table01_*`` to ``table10_*`` (CSV and LaTeX booktabs fragments) and ``provenance.json`` from
    recorded artifacts only (:mod:`tmmri.manuscript_tables`); runs after every evaluation, before the release manifest.

    Requires ``.stages/FROZEN.json`` and the ``evaluate:acdc_test`` record; the M&Ms tables are built only when
    ``evaluate:mms_external`` is recorded, and then obey :func:`require_mms_unlocked`.  Nothing is written to
    ``manuscript/``; no case identifier and no per-case M&Ms content appears in any table.
    """

    from .manuscript_tables import PROVENANCE, TABLE_NAMES, TableBuildError, build_tables, registry_file, source_paths

    run = ctx.require_run("build-manuscript-tables")
    if not run.frozen_path.is_file():
        raise BundleError(f"build-manuscript-tables: run {run.run_id} has no frozen analysis bundle (run freeze-analysis-bundle first)")
    _require_evaluated(run, "evaluate:acdc_test", "build-manuscript-tables")
    mms_record = run.read_record("evaluate:mms_external")
    include_mms = mms_record is not None and mms_record.status == "EVALUATED"
    if include_mms:
        require_mms_unlocked(ctx, "build-manuscript-tables (M&Ms tables)")
    # --- protocol v6.0.1 (lane L2): manuscript-2.1 tables beside Tables 1-10, which stay byte-identical
    from .manuscript_tables import PROVENANCE_V601, TABLE_NAMES_V601
    from .predictor_reporting import evaluated, require_partition_unlocked

    include_mms2 = evaluated(run, "evaluate:mms2_external")
    if include_mms2:
        require_partition_unlocked(ctx, "mms2_external", "build-manuscript-tables")
    # --- end protocol v6.0.1
    tdir = run.sub("manuscript_tables")
    spec_path, registry_path = PROJECT_ROOT / "SPECIFICATION.md", registry_file(ctx.study)
    outputs = [tdir / f"{name}.{ext}" for name in TABLE_NAMES for ext in ("csv", "tex")] + [tdir / PROVENANCE]
    outputs += [tdir / f"{name}.{ext}" for name in TABLE_NAMES_V601 for ext in ("csv", "tex")] + [tdir / PROVENANCE_V601]
    inputs = source_paths(run, ctx.study, include_mms=include_mms, spec_path=spec_path, registry_path=registry_path, include_mms2=include_mms2)

    def compute() -> dict[str, Any]:
        try:
            return build_tables(run, ctx.study, include_mms=include_mms, out_dir=tdir, spec_path=spec_path, registry_path=registry_path,
                                include_mms2=include_mms2)
        except TableBuildError as exc:
            raise BundleError(f"build-manuscript-tables: {exc}") from exc

    return _stage(ctx, "build-manuscript-tables", compute, inputs, outputs, {"tables": list(TABLE_NAMES), "include_mms": include_mms,
                                                                             "tables_v6_0_1": list(TABLE_NAMES_V601), "include_mms2": include_mms2})


# ----------------------------------------------------------------------------- release manifest


RELEASE_MANIFEST_EXCLUDED = ("release_manifest.json", "artifact_status.json", ".stages/run_log.jsonl")


def write_release_manifest(ctx: StageContext) -> dict[str, Any]:
    """``release_manifest.json``: SHA-256 of every run-root file, the configuration and registry hashes, the code
    version, the protocol tag state and the normalized ``artifact_status.json`` (content-derived; rewritten only when
    it changes).  Mutable logs (``artifact_status.json``, ``.stages/run_log.jsonl``) are summarized, not hashed."""

    run = ctx.require_run("write-release-manifest")
    files = {}
    for path in sorted(run.path.rglob("*")):
        rel = path.relative_to(run.path).as_posix()
        if path.is_file() and rel not in RELEASE_MANIFEST_EXCLUDED and not (path.name.startswith(".") and ".tmp" in path.name):
            files[rel] = hash_path(path)
    status = {stage: {k: v for k, v in rec.items() if k != "action"} for stage, rec in sorted(run.status().get("stages", {}).items())
              if stage != "write-release-manifest"}
    tag = ctx.study.freeze_tag
    payload = {"schema": "tmmri.release_manifest/1", "run_id": run.run_id, "n_files": len(files), "files": files,
               "files_sha256": sha256_json(files), "config": ctx.study.hash_manifest(), "code": code_version(),
               "protocol_tag": tag, "protocol_frozen": protocol_frozen(tag), "synthetic_fixture": ctx.synthetic_sources(),
               "artifact_status": status}
    path = run.path / "release_manifest.json"
    prior = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    action = "SKIPPED" if prior == json.loads(json.dumps(payload, default=str)) else "RAN"
    if action == "RAN":
        atomic_write_json(path, payload)
    run.set_status("write-release-manifest", "BUILT", action=action, manifest_sha256=hash_path(path))
    run.log({"stage": "write-release-manifest", "action": action, "files": len(files)})
    return {"action": action, "files": len(files), "manifest": str(path)}


# ----------------------------------------------------------------------------- orchestration


def step_id(stage: str, arg: str | None) -> str:
    return f"{stage}:{arg}" if arg else stage


def _select_steps(until: str | None, start: str | None) -> list[tuple[str, str | None]]:
    ids = [step_id(*s) for s in PIPELINE_STEPS]

    def locate(name: str) -> int:
        if name in ids:
            return ids.index(name)
        matches = [i for i, (stage, _) in enumerate(PIPELINE_STEPS) if stage == name]
        if not matches:
            raise ValueError(f"unknown stage {name!r}; steps: {ids}")
        return matches[0]

    first = locate(start) if start else 0
    last = locate(until) if until else len(ids) - 1
    if last < first:
        raise ValueError(f"--until {until} precedes --from {start}")
    return list(PIPELINE_STEPS[first:last + 1])


def run_step(ctx: StageContext, stage: str, arg: str | None) -> Any:
    if stage == "stage-sources":
        report = stage_sources(ctx, str(arg))
        if not report.ok:
            raise RuntimeError(f"stage-sources:{arg} failed for {len(report.failures)} cases: {report.failures[:3]}")
        return {"action": staging_action(report)}
    if stage == "validate-cohort":
        report = validate_cohort(ctx, strict=True)
        action = ctx.run.status()["stages"].get("validate-cohort", {}).get("action") if ctx.run is not None else None
        return {"action": action or "RAN", "passed": report["passed"]}
    if stage == "build-semantics":
        return build_semantics(ctx, str(arg))
    if stage == "extract-formal":
        return extract_formal(ctx, str(arg))
    if stage == "fit-transition":
        return fit_transition(ctx)
    if stage == "fit-predictor":
        return fit_predictor(ctx)
    if stage == "discretize":
        return discretize(ctx, profile=str(ctx.study.wedd["primary_profile"]))
    if stage == "check-semantic-eligibility":
        return check_semantic_eligibility(ctx)
    if stage == "induce-rules":
        return induce_rules(ctx)
    if stage == "gate-rules":
        return gate_rules(ctx)
    if stage == "freeze-analysis-bundle":
        freeze_analysis_bundle(ctx)
        return {"action": ctx.run.status()["stages"]["freeze-analysis-bundle"]["action"]}
    if stage == "evaluate":
        return evaluate(ctx, str(arg), no_fit=True)
    if stage == "audit-shift":
        return audit_shift(ctx)
    if stage == "fit-refit-arm":
        return fit_refit_arm(ctx)
    if stage == "summarize-predictor":
        base = summarize_predictor(ctx)
        # --- protocol v6.0.1 (A2, A4, A5): one extended record per available partition (tmmri.predictor_reporting)
        from .predictor_reporting import summarize_predictor_extended

        extended = summarize_predictor_extended(ctx)
        ran = base.action != "SKIPPED" or any(r.action != "SKIPPED" for r in extended)
        return {"action": "RAN" if ran else "SKIPPED", "extended": [r.stage for r in extended]}
        # --- end protocol v6.0.1
    if stage == "run-ablations":
        return run_ablations(ctx)
    if stage == "run-equivariance":
        return run_equivariance(ctx, str(arg or "acdc"))
    if stage == "build-manuscript-tables":
        return build_manuscript_tables(ctx)
    if stage == "build-manuscript-figures":   # protocol v6.0.1 A3
        from .manuscript_figures import build_manuscript_figures

        return build_manuscript_figures(ctx)
    if stage == "write-release-manifest":
        return write_release_manifest(ctx)
    raise ValueError(f"unknown stage {stage}")


def run_pipeline(ctx: StageContext, *, until: str | None = None, start: str | None = None) -> list[dict[str, Any]]:
    """Run the v6 steps in order (``--from``/``--until`` select a contiguous range); stop at the first failure.

    Every step goes through its stage function, so the once-only guard of a frozen run applies to ``--from``/``--until``
    ranges exactly as to single commands; a refusal leaves the stage's recorded status untouched."""

    run = ctx.require_run("run")
    if ctx.synthetic_sources():
        run.set_field("data", {"kind": "synthetic_fixture", "status": "UNIT_TESTED_SYNTHETIC",
                               "note": "algorithm control on synthetic fixtures; never a cohort result"})
    done: list[dict[str, Any]] = []
    for stage, arg in _select_steps(until, start):
        sid = step_id(stage, arg)
        try:
            value = run_step(ctx, stage, arg)
        except ProtocolFreezeError as exc:
            run.set_status(sid, "SPECIFIED", blocked_reason="awaiting_protocol_freeze_M0", detail=str(exc))
            raise
        except StageRecordLockedError:
            raise                       # the recorded stage and its status stay as they are (once-only guard)
        except NotImplementedError as exc:
            run.set_status(sid, "SPECIFIED", blocked_reason="methods_api_missing", detail=str(exc))
            raise
        except Exception as exc:
            run.set_status(sid, "BLOCKED_COMPUTE", detail=f"{type(exc).__name__}: {exc}")
            raise
        action = getattr(value, "action", None) or (value.get("action") if isinstance(value, dict) else None) or "RAN"
        done.append({"step": sid, "action": action})
    return done
