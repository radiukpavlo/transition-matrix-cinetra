"""Read-only preflight of the protocol-v6.0.1 execution steps (``python -m tmmri.cli preflight --run-id <id> --step <step>``).

The once-only evaluations of protocol-v6.0.1 (SPECIFICATION.md 14.6) are irreversible, so every step of
``docs/V601_EXECUTION_RUNBOOK.md`` is checked before it runs.  Each rule below is the precondition the guarded stage
itself enforces (or the protocol rule its docstring cites), evaluated without running it:

* run root and stage records: :class:`tmmri.runs.RunRoot` opened with ``create=False`` and
  :meth:`~tmmri.runs.RunRoot.read_record`; a required artifact must be a recorded output of some stage and still carry
  the recorded SHA-256 (:func:`tmmri.runs.hash_path`, as ``run_stage`` checks before a skip);
* freezes: ``.stages/FROZEN.json`` through :func:`tmmri.stages.verify_frozen_bundle`; the F8, D20, F9, F9b and
  S1-S4 freeze records through :func:`tmmri.experiments.l57_stages.verify_hash_manifest` (every listed file re-hashed);
* once-only state: a recorded target stage would SKIP when identical, else REFUSE with exit 7 in a frozen run
  (:func:`tmmri.stages.once_only_guard`, :func:`tmmri.neural.image_stages.image_bundle_guard`);
* protocol tags through :func:`tmmri.stages.protocol_frozen` (``protocol-v6.0.1``: annotated, signed, HEAD descends);
* SPECIFICATION.md 14.6 freeze order for the M&Ms-2 steps (:data:`tmmri.stages.MMS2_FREEZE_STAGES`,
  :func:`tmmri.stages.mms2_freeze_order`);
* weights (``registry/weights.yaml``, ``$TORCH_HOME``), the CPE-1 upload set (:func:`tmmri.neural.cloud.cpe1_upload_set`),
  the Modal client pin, the Modal token (existence only: ``MODAL_TOKEN_ID`` or ``%USERPROFILE%/.modal.toml``; contents
  are never read), free disk space and a clean git working tree.

Nothing is written.  Output: one line per check, ``PASS``, ``WARN``, ``FAIL`` or ``UNKNOWN`` (a resource that is not
visible from this side: the Docker volumes from the host, the host's Modal token, M&Ms-2 archive or Windows tar from
the container), then one ``RESULT`` line.  Exit 0 when no check failed, else 1.  The M&Ms-2 archive is hashed only with
``--hash`` and only once the signed tag exists (SPECIFICATION.md 4.6: no M&Ms-2 byte before the tag).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .paths import PROJECT_ROOT

PASS, WARN, FAIL, UNKNOWN = "PASS", "WARN", "FAIL", "UNKNOWN"
STEPS = ("replication", "f8", "b5", "b6", "tensor-cache", "weights", "d20", "modal-qualify", "upload", "f9-train", "f9-import",
         "f9-freeze", "f9b", "trust-fit", "eval-historical", "s5", "s6", "tag", "mms2-extract", "mms2-evaluate", "results-freeze")
# Where each step's command runs (docs/V601_EXECUTION_RUNBOOK.md); the preflight runs in the same place.
STEP_SIDE = {"modal-qualify": "host", "f9-train": "host", "tag": "host", "mms2-extract": "host"}
# Steps whose records are evidence (freezes, once-only evaluations, the tag): a dirty working tree fails them.
STRICT_GIT = frozenset({"replication", "f8", "d20", "f9-freeze", "f9b", "trust-fit", "eval-historical", "s6", "tag", "mms2-extract",
                        "mms2-evaluate", "results-freeze"})
SPLITS = ("acdc_dev", "acdc_test", "mms_external", "mms2_external")
STEP_SPLITS = {"tensor-cache": ("acdc_dev", "acdc_test", "mms_external", "mms2_external"),
               "d20": ("acdc_dev", "acdc_test", "mms_external"), "eval-historical": ("acdc_test", "mms_external")}
DEFAULT_SPLITS = {"tensor-cache": ("acdc_dev", "acdc_test", "mms_external"), "d20": ("acdc_dev", "acdc_test", "mms_external"),
                  "eval-historical": ("acdc_test", "mms_external")}
DATASET_OF = {"acdc_dev": "ACDC", "acdc_test": "ACDC", "mms_external": "MMS", "mms2_external": "MMS2"}
RUN_ID_CONVENTION = re.compile(r"^v6-[a-z0-9][a-z0-9-]*-\d{8}-\d{4}$")   # cli --run-id help: v6-<purpose>-YYYYMMDD-HHMM
HEX64 = re.compile(r"^[0-9a-f]{64}$")
DISK_FAIL_GIB = 5.0
ALT_BUNDLES = ("d20", "f9", "f9b")
FREEZE_BUNDLES = ("F8", "D20", "F9", "F9b", "S1-S4")   # tmmri.stages.MMS2_FREEZE_STAGES


@dataclass(frozen=True)
class Check:
    status: str
    name: str
    detail: str

    def line(self) -> str:
        return f"{self.status:<7} {self.name}: {self.detail}"


def detect_side() -> str:
    """``container`` inside Docker (``/.dockerenv``), else ``host``."""

    return "container" if Path("/.dockerenv").exists() else "host"


def _git(args: Sequence[str], root: Path = PROJECT_ROOT) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(["git", "--no-optional-locks", *args], cwd=root, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None


def git_dirty(root: Path = PROJECT_ROOT) -> tuple[bool | None, list[str]]:
    """Tracked-file changes, as :func:`tmmri.runs.code_version` records ``dirty`` (untracked files ignored)."""

    proc = _git(["status", "--porcelain", "--untracked-files=no"], root)
    if proc is None or proc.returncode != 0:
        return None, []
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    return bool(lines), lines


def git_head(root: Path = PROJECT_ROOT) -> str | None:
    proc = _git(["rev-parse", "HEAD"], root)
    return proc.stdout.strip() if proc is not None and proc.returncode == 0 else None


def tag_object_type(tag: str, root: Path = PROJECT_ROOT) -> str | None:
    """``tag`` (annotated), ``commit`` (lightweight) or None (absent), as ``git cat-file -t`` reports it."""

    proc = _git(["cat-file", "-t", tag], root)
    return proc.stdout.strip() if proc is not None and proc.returncode == 0 else None


def tag_signed(tag: str, root: Path = PROJECT_ROOT) -> bool:
    from .stages import tag_has_signature

    proc = _git(["cat-file", "tag", tag], root)
    return proc is not None and proc.returncode == 0 and tag_has_signature(proc.stdout)


def _rel(path: Path, base: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(base).resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


@dataclass
class Options:
    split: list[str] = field(default_factory=list)
    wave: int = 1
    specs: str | None = None
    source: str | None = None
    archive: str | None = None
    vendor: str = "all"
    hash: bool = False
    min_free_gb: float = 20.0
    explicit_artifact_root: bool = False
    explicit_data_root: bool = False


class Preflight:
    """The checks of one step on one run root; every method only reads."""

    def __init__(self, step: str, ctx: Any, *, side: str, options: Options) -> None:
        self.step = step
        self.ctx = ctx
        self.run = ctx.run
        self.study = ctx.study
        self.side = side
        self.opt = options
        self.checks: list[Check] = []
        self._index: dict[str, list[tuple[str, str | None]]] | None = None
        self._records: dict[str, Any] | None = None

    # ------------------------------------------------------------------ output
    def add(self, status: str, name: str, detail: str) -> str:
        self.checks.append(Check(status, name, detail))
        return status

    @property
    def failed(self) -> bool:
        return any(c.status == FAIL for c in self.checks)

    def lines(self) -> list[str]:
        counts = {s: sum(c.status == s for c in self.checks) for s in (PASS, WARN, UNKNOWN, FAIL)}
        verdict = "NO-GO" if counts[FAIL] else "GO"
        summary = (f"RESULT  {verdict} for step {self.step} of run {self.run.run_id}: {counts[PASS]} PASS, {counts[WARN]} WARN, "
                   f"{counts[UNKNOWN]} UNKNOWN, {counts[FAIL]} FAIL (preflight wrote nothing)")
        return [c.line() for c in self.checks] + [summary]

    # ------------------------------------------------------------------ visibility
    @property
    def artifact_root(self) -> Path:
        return self.run.artifact_root

    @property
    def data_root(self) -> Path:
        return Path(self.ctx.data_root)

    def run_visible(self) -> bool:
        return self.run.path.is_dir() or self.side == "container" or self.opt.explicit_artifact_root

    def data_visible(self) -> bool:
        return self.side == "container" or self.opt.explicit_data_root or (self.data_root / "manifests").is_dir()

    def _not_visible(self, what: str) -> str:
        if STEP_SIDE.get(self.step) == "host":
            return self.add(UNKNOWN, what, f"not visible from the host (Docker volume); run this preflight in the container as well")
        return self.add(FAIL, what, "not visible from the host (Docker volumes tmmri_artifacts/tmmri_data): run this preflight in the "
                                    "container (docker compose run --rm -T tmmri python -m tmmri.cli preflight ...)")

    # ------------------------------------------------------------------ run root and records
    def run_root(self, *, required: bool = True, created_by_step: bool = False) -> bool:
        path = self.run.path
        if path.is_dir():
            n = len(list(self.run.stages_dir.glob("*.json"))) if self.run.stages_dir.is_dir() else 0
            frozen = " (frozen: .stages/FROZEN.json)" if self.run.frozen_path.is_file() else ""
            self.add(PASS, "run root", f"{path.as_posix()} exists with {n} stage files{frozen}")
            return True
        if not self.run_visible():
            self._not_visible("run root")
            return False
        if created_by_step:
            self.add(PASS, "run root", f"{path.as_posix()} absent; the step creates it")
        else:
            self.add(FAIL if required else WARN, "run root", f"{path.as_posix()} does not exist")
        return False

    def records(self) -> dict[str, Any]:
        if self._records is None:
            out: dict[str, Any] = {}
            if self.run.stages_dir.is_dir():
                for p in sorted(self.run.stages_dir.glob("*.json")):
                    if p.name == "FROZEN.json":
                        continue
                    name = p.stem.replace("__", ":")
                    rec = self.run.read_record(name)
                    if rec is not None:
                        out[rec.stage] = rec
            self._records = out
        return self._records

    def index(self) -> dict[str, list[tuple[str, str | None]]]:
        """Recorded outputs of every stage record: portable path -> [(stage, sha256)]."""

        if self._index is None:
            idx: dict[str, list[tuple[str, str | None]]] = {}
            for stage, rec in self.records().items():
                for key, digest in rec.outputs.items():
                    idx.setdefault(key, []).append((stage, digest))
            self._index = idx
        return self._index

    def resolve(self, rel: str | Path) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.run.path / p

    def _portable_to_path(self, key: str) -> Path:
        from .paths import DATA_ROOT

        prefix, _, rest = key.partition(":")
        base = {"run": self.run.path, "data": DATA_ROOT, "repo": PROJECT_ROOT}.get(prefix)
        return base / rest if base is not None and rest else Path(key)

    def artifact(self, rel: str | Path, *, required: bool = True, recorded: bool = True, name: str | None = None,
                 missing_hint: str = "") -> bool:
        """``rel`` exists and (``recorded``) is a recorded stage output whose SHA-256 still matches the record."""

        from .runs import hash_path

        path = self.resolve(rel)
        label = name or f"artifact {_rel(path, self.run.path)}"
        if not path.exists():
            hint = f"; {missing_hint}" if missing_hint else ""
            self.add(FAIL if required else WARN, label, f"missing{hint}")
            return False
        if not recorded:
            self.add(PASS, label, "present")
            return True
        key = self.run.portable(path)
        producers = self.index().get(key)
        if not producers:
            self.add(FAIL if required else WARN, label, "present but not a recorded output of any stage record (.stages); "
                                                        "it cannot be traced, so the reading stage would refuse or record unverified input")
            return False
        digest = hash_path(path)
        matching = [stage for stage, recorded_digest in producers if recorded_digest == digest]
        if matching:
            self.add(PASS, label, f"recorded by {matching[0]}; SHA-256 unchanged ({digest[:12]})")
            return True
        stage = producers[-1][0]
        self.add(FAIL, label, f"changed after {stage} recorded it (recorded {str(producers[-1][1])[:12]}, now {str(digest)[:12]}); "
                              "a changed recorded output needs a new run id")
        return False

    def record(self, stage: str, *, required: bool = True, status: str | None = None, verify_outputs: bool = True,
               hint: str = "") -> Any:
        """The stage record exists (with ``status``) and every output it lists still carries the recorded SHA-256."""

        from .runs import hash_path

        rec = self.records().get(stage)
        name = f"record {stage}"
        if rec is None:
            self.add(FAIL if required else WARN, name, f"not recorded (.stages/{self.run.record_path(stage).name}){'; ' + hint if hint else ''}")
            return None
        if status is not None and rec.status != status:
            self.add(FAIL if required else WARN, name, f"status {rec.status}, expected {status}")
            return None
        if verify_outputs:
            changed = [key for key, digest in rec.outputs.items() if hash_path(self._portable_to_path(key)) != digest]
            if changed:
                self.add(FAIL, name, f"{len(changed)} recorded output(s) missing or changed since the record (e.g. {changed[:3]}); a new run id is required")
                return None
        started = (rec.run_log or {}).get("started_utc")
        self.add(PASS, name, f"{rec.status}, {len(rec.outputs)} outputs{' unchanged' if verify_outputs else ''}" + (f", started {started}" if started else ""))
        return rec

    def once_only(self, stage: str, *, freeze_files: Iterable[str | Path] = ()) -> bool:
        """WARN when the target stage is already recorded: what a re-run would do (once-only guard)."""

        rec = self.records().get(stage)
        name = f"once-only {stage}"
        if rec is None:
            self.add(PASS, name, "not yet recorded; the step will run it")
            return False
        freezes = [p for p in (self.resolve(f) for f in freeze_files) if p.is_file()]
        frozen = self.run.frozen_path.is_file() or bool(freezes)
        started = (rec.run_log or {}).get("started_utc")
        if frozen:
            why = "FROZEN.json" if self.run.frozen_path.is_file() else ", ".join(p.name for p in freezes)
            verdict = (f"would SKIP if inputs, configuration, code version and parameters are unchanged, else REFUSE (once-only guard, "
                       f"{why}; exit 7)")
        else:
            verdict = "would SKIP if unchanged, else RE-RUN and replace the record (no freeze yet)"
        self.add(WARN, name, f"already recorded ({rec.status}{', ' + started if started else ''}); {verdict}")
        return True

    # ------------------------------------------------------------------ freezes
    def frozen_bundle(self, *, required: bool = True) -> dict[str, Any] | None:
        from . import stages

        name = "freeze .stages/FROZEN.json"
        if not self.run.frozen_path.is_file():
            self.add(FAIL if required else WARN, name, "missing: run freeze-analysis-bundle first")
            return None
        try:
            frozen = stages.verify_frozen_bundle(self.ctx)
        except (stages.BundleError, ValueError, KeyError, OSError) as exc:
            self.add(FAIL, name, f"verification failed: {exc}")
            return None
        self.add(PASS, name, f"{len(frozen.get('bundle', {}))} files re-hashed unchanged; bundle configuration matches; "
                             f"bundle_sha256 {str(frozen.get('bundle_sha256'))[:12]}; protocol_version_tag {frozen.get('protocol_version_tag')}")
        return frozen

    def freeze_paths(self) -> dict[str, str]:
        """Run-relative freeze records of F8, D20, F9, F9b and S1-S4 (configs/cart_f8.yaml, configs/probes.yaml)."""

        from .comparators.cart_f8 import CONFIG_KEY, FROZEN_NAME
        from .secondary_stages import lane_config

        probes = lane_config(self.study, "probes")["inputs"]
        f8_dir = str(lane_config(self.study, CONFIG_KEY).get("output_dir", "comparators/cart_f8"))
        return {"F8": f"{f8_dir}/{FROZEN_NAME}", "D20": str(probes["d20"]["freeze_record"]), "F9": str(probes["f9"]["freeze_record"]),
                "F9b": str(probes["f9b"]["freeze_record"]), "S1-S4": "trust/trust_bundle.json"}

    def freeze_file(self, bundle: str, *, required: bool = True) -> bool:
        """The freeze record exists, its own ``bundle_sha256`` holds (D20, F9) and every listed file re-hashes unchanged."""

        from .experiments.l57_stages import verify_hash_manifest
        from .runs import sha256_json
        from .stages import BundleError

        rel = self.freeze_paths()[bundle]
        path = self.run.path / rel
        name = f"freeze {bundle} ({rel})"
        if not path.is_file():
            self.add(FAIL if required else WARN, name, "missing")
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if bundle in ("D20", "F9"):
                body = {k: v for k, v in payload.items() if k != "bundle_sha256"}
                if sha256_json(body) != payload.get("bundle_sha256"):
                    raise BundleError(f"{rel} content does not match its bundle_sha256")
            info = verify_hash_manifest(self.run, rel, f"preflight {bundle}")
        except (BundleError, ValueError, KeyError, OSError) as exc:
            self.add(FAIL, name, f"verification failed: {exc}")
            return False
        self.add(PASS, name, f"{info['n_files']} files re-hashed unchanged; bundle_sha256 {str(info['bundle_sha256'])[:12]}")
        return True

    def freezes_before_extraction(self) -> None:
        """SPECIFICATION.md 14.6 step 6: each freeze record of F8, D20, F9, F9b and S1-S4 must precede the M&Ms-2 extraction."""

        from .stages import MMS2_FREEZE_STAGES

        for bundle in FREEZE_BUNDLES:
            missing = [s for s in MMS2_FREEZE_STAGES[bundle] if s not in self.records()]
            name = f"freeze order {bundle}"
            if missing:
                self.add(FAIL, name, f"freeze record(s) {missing} not recorded; extracting M&Ms-2 now leaves {bundle} without a registered "
                                     "mms2_external result (labelled post hoc, SPECIFICATION.md 14.6); freeze it first or record it as not executed")
            else:
                self.add(PASS, name, f"{', '.join(MMS2_FREEZE_STAGES[bundle])} recorded before the extraction")

    def freeze_order_labels(self) -> None:
        """mms2-evaluate: the evidence label each bundle's mms2_external result will carry (:func:`tmmri.stages.mms2_freeze_order`)."""

        from .stages import mms2_freeze_order

        order = mms2_freeze_order(self.ctx, list(FREEZE_BUNDLES))
        for bundle, info in order["bundles"].items():
            if info["before_extraction"]:
                self.add(PASS, f"freeze order {bundle}", "every freeze record finished before the first M&Ms-2 extraction: prospective")
            else:
                late = [e["stage"] for e in info["freezes"] if not e["before_extraction"]]
                self.add(WARN, f"freeze order {bundle}", f"{late} not recorded or not dated before the extraction "
                                                          f"({order['extraction']['first_extraction_started_utc']}): its mms2_external result is labelled post hoc")

    # ------------------------------------------------------------------ data, sources, tags, git, disk
    def staging_manifest(self, dataset: str, *, required: bool = True) -> bool:
        from .ingest.staging import manifest_path_for

        path = manifest_path_for(self.ctx.manifests_dir, dataset)
        name = f"staging manifest {dataset}"
        if path.is_file():
            self.add(PASS, name, f"{path.as_posix()} present")
            return True
        if not self.data_visible():
            self._not_visible(name)
            return False
        self.add(FAIL if required else WARN, name, f"{path.as_posix()} missing: run stage-sources --dataset {dataset.lower()} first")
        return False

    def source_root(self, dataset: str) -> bool:
        name = f"source root {dataset}"
        try:
            root = self.ctx.source_root(dataset)
        except (KeyError, OSError, ValueError) as exc:
            self.add(FAIL, name, f"cannot resolve: {exc}")
            return False
        if root.is_dir():
            self.add(PASS, name, f"{root.as_posix()} mounted")
            return True
        if self.side == "host":
            self.add(UNKNOWN, name, f"{root.as_posix()} not visible from the host (read-only bind /sources/{dataset} in the container)")
            return False
        self.add(FAIL, name, f"{root.as_posix()} missing (check .env and the compose bind)")
        return False

    def tag_v600(self) -> bool:
        from .stages import protocol_frozen

        tag = self.study.freeze_tag
        if protocol_frozen(tag):
            self.add(PASS, f"tag {tag}", "annotated and HEAD descends from it (M&Ms-1 lock open)")
            return True
        self.add(FAIL, f"tag {tag}", "absent, not annotated or HEAD does not descend from it: every M&Ms-1 endpoint stage refuses (exit 3)")
        return False

    def tag_v601(self, *, required: bool = True) -> bool:
        """The M&Ms-2 lock: :func:`tmmri.stages.protocol_frozen` (annotated, signature block, HEAD descends)."""

        from .stages import mms2_lock_tag, protocol_frozen

        tag = mms2_lock_tag(self.study)
        name = f"tag {tag}"
        if protocol_frozen(tag, signed=True):
            self.add(PASS, name, "annotated, carries a signature block (not verified cryptographically, as stages.protocol_frozen) "
                                 "and HEAD descends from it: M&Ms-2 lock open")
            return True
        kind = tag_object_type(tag)
        if kind is None:
            detail = "absent: the owner signs it (git tag -s); every M&Ms-2 stage refuses (exit 3)"
        elif kind != "tag":
            detail = f"exists as a {kind} (lightweight), not an annotated tag: it never unlocks M&Ms-2"
        elif not tag_signed(tag):
            detail = "annotated but unsigned: it never unlocks M&Ms-2 (protocol_frozen signed=True)"
        else:
            detail = "signed, but HEAD does not descend from it"
        self.add(FAIL if required else WARN, name, detail)
        return False

    def git_clean(self) -> None:
        dirty, lines = git_dirty()
        head = git_head()
        name = "git working tree"
        if dirty is None:
            self.add(WARN, name, "git status unavailable; stage records will carry dirty=None")
        elif dirty:
            strict = self.step in STRICT_GIT
            self.add(FAIL if strict else WARN, name, f"{len(lines)} tracked file(s) modified (e.g. {[l[3:] for l in lines[:3]]}); stage records "
                                                     "would carry code.dirty=true; commit or restore first")
        else:
            self.add(PASS, name, f"clean (tracked files) at {str(head)[:12]}")

    def disk(self, path: Path | None = None) -> None:
        target = Path(path or self.artifact_root)
        probe = target
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        try:
            free = shutil.disk_usage(probe).free / 2 ** 30
        except OSError as exc:
            self.add(WARN, "free disk", f"unknown for {probe.as_posix()}: {exc}")
            return
        status = FAIL if free < DISK_FAIL_GIB else WARN if free < self.opt.min_free_gb else PASS
        self.add(status, "free disk", f"{free:.1f} GiB free at {probe.as_posix()} (FAIL below {DISK_FAIL_GIB:g}, WARN below {self.opt.min_free_gb:g})")

    def run_id_convention(self) -> None:
        rid = self.run.run_id
        if RUN_ID_CONVENTION.fullmatch(rid):
            self.add(PASS, "run id", f"{rid} follows v6-<purpose>-YYYYMMDD-HHMM")
        else:
            self.add(WARN, "run id", f"{rid} does not follow v6-<purpose>-YYYYMMDD-HHMM (accepted by the stages; CLAUDE.md convention)")

    # ------------------------------------------------------------------ weights, torch
    def weights(self, keys: Sequence[str], *, file_required: bool) -> None:
        from .neural.weights import DEEP_ENCODER_CONFIG, PENDING, WEIGHTS, checkpoint_path, file_prefix, load_registry, sha256_of

        registry = (load_registry(self.study.root).get("weights") or {})
        for key in keys:
            entry = registry.get(key) or {}
            sha = str(entry.get("sha256") or "")
            name = f"weights {key}"
            if not HEX64.fullmatch(sha):
                self.add(FAIL, f"{name} registry", f"registry/weights.yaml has no 64-hex sha256 ({sha or 'absent'}): run verify-torch-weights")
                continue
            if entry.get("file") != WEIGHTS[key]["file"] or not sha.startswith(file_prefix(key)):
                self.add(FAIL, f"{name} registry", f"entry file {entry.get('file')} / sha256 prefix does not match {WEIGHTS[key]['file']}")
                continue
            self.add(PASS, f"{name} registry", f"{WEIGHTS[key]['file']} pinned, sha256 {sha[:12]}, {entry.get('bytes')} bytes")
            if not os.environ.get("TORCH_HOME"):
                self.add(UNKNOWN, f"{name} file", "TORCH_HOME is unset here; the weight cache is /cache/torch (volume tmmri_cache) in the "
                                                  "tmmri-gpu service only: run this preflight there")
                continue
            path = checkpoint_path(key)
            if not path.is_file():
                self.add(FAIL if file_required else WARN, f"{name} file",
                         f"{path.as_posix()} absent" + ("" if file_required else "; verify-torch-weights downloads it from download.pytorch.org"))
                continue
            if self.opt.hash:
                observed, _ = sha256_of(path)
                self.add(PASS if observed == sha else FAIL, f"{name} file", f"{path.as_posix()} SHA-256 {'matches' if observed == sha else 'DIFFERS from'} the registry")
            else:
                size_ok = entry.get("bytes") is None or path.stat().st_size == int(entry["bytes"])
                self.add(PASS if size_ok else FAIL, f"{name} file", f"{path.as_posix()} present, size {'matches' if size_ok else 'differs'} "
                                                                   "(SHA-256 with --hash; every encoder re-verifies it before use)")
        if "resnet50_imagenet1k_v2" in keys:
            import yaml

            cfg = yaml.safe_load((Path(self.study.root) / DEEP_ENCODER_CONFIG).read_text(encoding="utf-8"))
            recorded = str((cfg.get("encoder") or {}).get("weights_sha256") or "")
            pinned = str((registry.get("resnet50_imagenet1k_v2") or {}).get("sha256") or "")
            if recorded == pinned and HEX64.fullmatch(recorded):
                self.add(PASS, "configs/deep_encoder.yaml weights_sha256", "equals registry/weights.yaml")
            elif recorded == PENDING:
                self.add(WARN, "configs/deep_encoder.yaml weights_sha256", "pending_first_use: verify-torch-weights records it (a tracked-file edit)")
            else:
                self.add(FAIL, "configs/deep_encoder.yaml weights_sha256", f"{recorded[:12]} differs from the registry ({pinned[:12]}); verify-torch-weights refuses")

    def torch_available(self, what: str) -> None:
        import importlib.util

        if importlib.util.find_spec("torch") is not None:
            self.add(PASS, "torch", f"importable here ({what})")
        else:
            self.add(FAIL, "torch", f"not importable here: {what} runs in the tmmri-gpu service (docker compose --profile gpu run --rm tmmri-gpu ...)")

    # ------------------------------------------------------------------ Modal
    def f9_config(self) -> dict[str, Any]:
        from .neural.f9_config import load_f9_config

        return load_f9_config(self.study.root)

    def modal_pin(self) -> str | None:
        from .neural.cloud import modal_settings, require_client_pin

        modal_cfg = modal_settings(self.f9_config()["backend"])
        try:
            pin = require_client_pin(modal_cfg)
        except ValueError as exc:
            self.add(FAIL, "Modal client pin", str(exc))
            return None
        self.add(PASS, "Modal client pin", f"configs/f9_cnn.yaml backend.modal.client_version = {pin} (uvx --from modal=={pin})")
        gpu = modal_cfg.get("gpu")
        if isinstance(gpu, str) and gpu:
            self.add(PASS, "Modal GPU", f"one pinned type {gpu} (no fallback list)")
        else:
            self.add(FAIL, "Modal GPU", f"backend.modal.gpu must be one GPU type, found {gpu!r}")
        return pin

    def modal_token(self) -> None:
        name = "Modal token"
        if os.environ.get("MODAL_TOKEN_ID"):
            self.add(PASS, name, "MODAL_TOKEN_ID is set in this environment (value not read)")
        elif self.side == "container":
            self.add(UNKNOWN, name, "the host's %USERPROFILE%/.modal.toml is not visible in the container; container commands need "
                                    "-e MODAL_TOKEN_ID -e MODAL_TOKEN_SECRET")
        else:
            profile = Path(os.environ.get("USERPROFILE") or Path.home())
            path = profile / ".modal.toml"
            if path.is_file():
                self.add(PASS, name, f"{path.as_posix()} exists (contents not read)")
            else:
                self.add(FAIL, name, f"neither MODAL_TOKEN_ID nor {path.as_posix()}: the owner runs `uvx modal setup`")

    def uvx(self) -> None:
        found = shutil.which("uvx")
        self.add(PASS if found else FAIL, "uvx", f"{found}" if found else "not on PATH: the pinned Modal client runs through uvx")

    def dockerfile_bases(self) -> None:
        text = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
        found = dict(re.findall(r"^ARG (UBUNTU_IMAGE|UV_IMAGE)=(\S+)$", text, flags=re.MULTILINE))   # scripts/cloud/modal_train.py
        bad = [k for k in ("UBUNTU_IMAGE", "UV_IMAGE") if "@sha256:" not in found.get(k, "")]
        self.add(FAIL if bad else PASS, "Dockerfile bases", f"{bad} not pinned by digest" if bad else "UBUNTU_IMAGE and UV_IMAGE pinned by digest (Modal image)")

    def qualification_report(self, *, required: bool = False) -> None:
        """The latest ``outputs/cloud/qualify_<gpu>_*.json`` of ``modal_train.py --qualify`` for the pinned GPU."""

        from .neural.cloud import modal_settings

        gpu = str(modal_settings(self.f9_config()["backend"]).get("gpu"))
        reports = sorted((PROJECT_ROOT / "outputs" / "cloud").glob(f"qualify_{gpu}_*.json"))
        name = f"Modal qualification ({gpu})"
        if not reports:
            self.add(FAIL if required else WARN, name, f"no outputs/cloud/qualify_{gpu}_*.json: SPECIFICATION.md 14.7 qualifies the pinned GPU on "
                                                     "synthetic tensors before the first acdc_dev upload")
            return
        try:
            report = json.loads(reports[-1].read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.add(FAIL, name, f"{reports[-1].name} unreadable: {exc}")
            return
        keys = ("passed", "determinism", "resume", "ieee_fp32", "tf32_disabled")
        detail = ", ".join(f"{k}={report.get(k) if not isinstance(report.get(k), dict) else report[k].get('passed', '...')}" for k in keys)
        self.add(PASS if report.get("passed") is True else FAIL, name, f"{reports[-1].name}: {detail}")

    def ledger_uploads(self) -> dict[str, str]:
        """relpath -> sha256 of the upload rows of ``cloud/modal_ledger.csv`` (CPE-1 ledger)."""

        import csv

        from .neural.cloud import ledger_path, modal_settings

        path = ledger_path(self.run.path, modal_settings(self.f9_config()["backend"]))
        if not path.is_file():
            return {}
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        failed = {(r.get("local_path"), r.get("sha256")) for r in rows if r.get("action") == "upload_failed"}
        return {str(r.get("local_path")): str(r.get("sha256")) for r in rows
                if r.get("action") == "upload" and (r.get("local_path"), r.get("sha256")) not in failed}

    # ------------------------------------------------------------------ helpers of several steps
    def recorded_inputs(self, rels: Iterable[str], *, required: bool = True, hint: str = "") -> bool:
        ok = True
        for rel in rels:
            ok &= self.artifact(rel, required=required, missing_hint=hint)
        return ok

    def partitions(self) -> list[str]:
        allowed = STEP_SPLITS.get(self.step, ())
        chosen = list(self.opt.split) or list(DEFAULT_SPLITS.get(self.step, ()))
        bad = [s for s in chosen if s not in allowed]
        if bad:
            self.add(FAIL, "--split", f"{bad} not applicable to step {self.step} (choices {list(allowed)})")
            chosen = [s for s in chosen if s in allowed]
        return chosen

    def lock_for(self, partition: str) -> bool:
        if partition == "mms_external":
            return self.tag_v600()
        if partition == "mms2_external":
            return self.tag_v601()
        return True


# ============================================================================ steps


def _contracts(pf: Preflight, *names: str) -> None:
    pf.recorded_inputs([f"contracts/{n}" for n in names], hint="run validate-cohort first")


def step_replication(pf: Preflight) -> None:
    pf.run_id_convention()
    pf.tag_v600()
    pf.git_clean()
    for ds in ("ACDC", "MMS"):
        pf.staging_manifest(ds)
    for ds in ("acdc", "mms"):
        pf.source_root(ds)
    if pf.run_root(created_by_step=True):
        if pf.run.frozen_path.is_file():
            pf.add(WARN, "replication run state", "frozen (.stages/FROZEN.json): each recorded step of `run` would SKIP when identical, "
                                                  "else REFUSE (once-only guard, exit 7); a code change since the records refuses at the first one")
        else:
            pf.add(WARN, "replication run state", f"{len(pf.records())} stage records: `run` resumes; recorded stages skip when unchanged")
        for stage in ("build-manuscript-tables", "build-manuscript-figures"):
            if stage in pf.records():
                pf.add(WARN, f"once-only {stage}", "already recorded: it belongs after every evaluation and the tag (results-freeze)")
    pf.disk()


def step_f8(pf: Preflight) -> None:
    from .comparators.cart_f8 import FIT_STAGE, _mms2_staged
    from .stages import inner_oof_path

    if not pf.run_root():
        return
    pf.frozen_bundle()
    pf.artifact("oof/bhat_oof_ACDC_dev.parquet")
    pf.artifact("formal/d_model_crossfit_ACDC_dev.parquet")
    n_folds = int(pf.study.splits["acdc"]["acdc_dev"]["outer_folds"]["n_folds"])
    for k in range(1, n_folds + 1):
        pf.artifact(_rel(inner_oof_path(pf.run, k), pf.run.path))
    pf.artifact("semantics/B_ACDC.parquet")
    _contracts(pf, "split_manifest.parquet")
    try:
        staged = _mms2_staged(pf.ctx)
    except (OSError, ValueError, KeyError) as exc:
        pf.add(WARN, "M&Ms-2 not yet extracted or staged", f"cannot tell: {exc}")
    else:
        pf.add(FAIL if staged else PASS, "M&Ms-2 not yet extracted or staged",
               "an extraction record or staging manifest exists: fit-cart-surrogate refuses (F8 must be frozen first)" if staged
               else "no extraction record and no MMS2 staging manifest (fit-cart-surrogate's own guard)")
    pf.once_only(FIT_STAGE)
    if (pf.run.path / pf.freeze_paths()["F8"]).is_file():
        pf.freeze_file("F8")
    pf.git_clean()
    pf.disk()


def step_b5(pf: Preflight) -> None:
    from .experiments.exploratory.common import analysis_config

    pf.run_root(created_by_step=True)
    try:
        analysis_config(pf.study, "B5")
        pf.add(PASS, "configs/exploratory.yaml B5", "present (synthetic planted rules only; no cohort data and no frozen bundle needed)")
    except (KeyError, FileNotFoundError, ValueError) as exc:
        pf.add(FAIL, "configs/exploratory.yaml B5", str(exc))
    pf.once_only("run-exploratory:B5")
    pf.git_clean()
    pf.disk()


def step_b6(pf: Preflight) -> None:
    from .experiments.exploratory.common import analysis_config

    if not pf.run_root():
        return
    try:
        analysis_config(pf.study, "B6")
        pf.add(PASS, "configs/exploratory.yaml B6", "present")
    except (KeyError, FileNotFoundError, ValueError) as exc:
        pf.add(FAIL, "configs/exploratory.yaml B6", str(exc))
    pf.frozen_bundle()
    profile = str(pf.study.wedd["primary_profile"])
    pf.recorded_inputs(["formal/A_ACDC.parquet", "semantics/B_ACDC.parquet", "formal/d_model_ACDC_dev.parquet",   # fold_salts.plan inputs
                        "contracts/cohort_ACDC.parquet", "contracts/split_manifest.parquet", "transitions/transition_model.json",
                        f"thresholds/d_model_{profile}.json", "rules/semantic_eligibility.json", "rules/d_model/rulebook_or_empty.json"])
    pf.once_only("run-exploratory:B6")
    pf.git_clean()
    pf.disk()


def step_tensor_cache(pf: Preflight) -> None:
    from .neural.image_stages import F9_FREEZE_RECORD

    parts = pf.partitions()
    if not pf.run_root():
        return
    _contracts(pf, "split_manifest.parquet", "cohort_ACDC.parquet")
    cfg = Path(pf.study.root) / "configs" / "neural_input.yaml"
    pf.add(PASS if ("neural_input" in pf.study.configs or cfg.is_file()) else FAIL, "configs/neural_input.yaml", "present" if cfg.is_file() else "missing")
    for ds in sorted({DATASET_OF[p] for p in parts}):
        pf.staging_manifest(ds)
    for part in parts:
        pf.lock_for(part)
        if part == "mms2_external":       # build_tensor_cache: M&Ms-2 tensors only after freeze-f9 (SPECIFICATION.md 14.6)
            pf.artifact(F9_FREEZE_RECORD, recorded=False, missing_hint="M&Ms-2 tensors are built only after freeze-f9")
            pf.record("validate-cohort:MMS2", hint="run validate-cohort --dataset mms2 first")
        pf.once_only(f"build-tensor-cache:{part}")
    pf.add(WARN, "O1 orientation", "build-tensor-cache records are once-only in a frozen run: a later fix of neural/preprocess.py "
                                   "cannot rebuild them in this run id (exit 7); settle O1 before building the caches")
    pf.git_clean()
    pf.disk()


def step_weights(pf: Preflight) -> None:
    from .neural.f9_config import resolve_weights
    from .neural.weights import D20_WEIGHTS, F9_WEIGHTS

    pf.weights((F9_WEIGHTS, D20_WEIGHTS), file_required=False)
    try:
        resolved = resolve_weights(pf.f9_config(), pf.study.root)
        pf.add(PASS, "configs/f9_cnn.yaml weights", f"{resolved['name']} -> {resolved['file']} (resolve_weights)")
    except (KeyError, ValueError, FileNotFoundError) as exc:
        pf.add(FAIL, "configs/f9_cnn.yaml weights", str(exc))
    if pf.side == "container" and not os.environ.get("TORCH_HOME"):
        pf.add(WARN, "service", "this service has no TORCH_HOME: verify-torch-weights here would download into the container's own "
                                "~/.cache/torch, which --rm discards; run it in tmmri-gpu (TORCH_HOME=/cache/torch)")
    pf.add(WARN, "tracked files", "verify-torch-weights rewrites registry/weights.yaml (and configs/deep_encoder.yaml while pending); "
                                  "check `git status` afterwards")
    pf.disk(Path(os.environ.get("TORCH_HOME") or pf.artifact_root))


def step_d20(pf: Preflight) -> None:
    from .neural.weights import D20_WEIGHTS

    parts = pf.partitions()
    pf.weights((D20_WEIGHTS,), file_required=True)
    pf.torch_available("extract-deep-features")
    if not pf.run_root():
        return
    _contracts(pf, "split_manifest.parquet", "inner_folds.parquet", "cohort_ACDC.parquet")
    pf.artifact("semantics/B_ACDC.parquet")
    for ds in sorted({DATASET_OF[p] for p in parts}):
        pf.staging_manifest(ds)
    deep_freeze = pf.freeze_paths()["D20"]
    for part in parts:
        pf.lock_for(part)
        pf.once_only(f"extract-deep-features:{part}", freeze_files=[deep_freeze])
    pf.once_only("fit-deep-encoder", freeze_files=[deep_freeze])
    if (pf.run.path / deep_freeze).is_file():
        pf.freeze_file("D20")
    _extraction_notice(pf, "D20")
    pf.git_clean()
    pf.disk()


def _extraction_notice(pf: Preflight, bundle: str) -> None:
    from .stages import mms2_extraction_times

    try:
        ext = mms2_extraction_times(pf.ctx)
    except (OSError, ValueError, KeyError):
        return
    if ext["exists"]:
        pf.add(WARN, f"freeze order {bundle}", f"an M&Ms-2 extraction record exists ({ext['first_extraction_started_utc']}): a {bundle} freeze "
                                               "recorded now is late and its mms2_external result is labelled post hoc (SPECIFICATION.md 14.6)")
    else:
        pf.add(PASS, f"freeze order {bundle}", "no M&Ms-2 extraction record yet: a freeze recorded now precedes it")


def step_modal_qualify(pf: Preflight) -> None:
    pf.modal_pin()
    pf.modal_token()
    if pf.side == "host":
        pf.uvx()
    else:
        pf.add(UNKNOWN, "uvx", "modal_train.py runs on the host: run this preflight on the host")
    pf.dockerfile_bases()
    for rel in ("uv.lock", "configs/runtime.env", "scripts/cloud/modal_train.py"):
        pf.add(PASS if (PROJECT_ROOT / rel).is_file() else FAIL, rel, "present" if (PROJECT_ROOT / rel).is_file() else "missing")
    pf.add(PASS, "data", "--qualify uses synthetic tensors only (tmmri.neural.qualify); nothing is read from or written to a Volume")
    pf.disk(PROJECT_ROOT / "outputs")


def step_upload(pf: Preflight) -> None:
    from .neural.cloud import CloudPolicyError, cpe1_upload_set, modal_settings, require_allowed_volume

    pf.modal_pin()
    pf.modal_token()
    if pf.side == "container":
        pf.uvx()
    else:
        pf.add(WARN, "where", "upload-cache runs in the tmmri container (docker compose run --rm -e MODAL_TOKEN_ID -e MODAL_TOKEN_SECRET tmmri ...)")
    modal_cfg = modal_settings(pf.f9_config()["backend"])
    try:
        volume = require_allowed_volume(str(modal_cfg["cache_volume"]), modal_cfg)
        pf.add(PASS, "CPE-1 Volume", f"{volume} is a named CPE-1 Volume")
    except CloudPolicyError as exc:
        pf.add(FAIL, "CPE-1 Volume", str(exc))
    pf.qualification_report(required=True)
    if not pf.run_root():
        return
    pf.artifact("neural_cache/acdc_dev", missing_hint="run build-tensor-cache --split acdc_dev first")
    try:
        items = cpe1_upload_set(pf.run.path)
    except (CloudPolicyError, FileNotFoundError, KeyError, ValueError) as exc:
        pf.add(FAIL, "CPE-1 upload set", str(exc))
        items = []
    else:
        pf.add(PASS, "CPE-1 upload set", f"{len(items)} files ({sum(i['bytes'] for i in items) / 2 ** 20:.1f} MiB): manifest.csv and the listed "
                                         ".npz only, every SHA-256 matches the manifest, all rows acdc_dev")
    uploaded = pf.ledger_uploads()
    if items and uploaded:
        done = [i for i in items if uploaded.get(f"neural_cache/acdc_dev/{i['relpath']}") == i["sha256"]]
        pf.add(WARN, "CPE-1 ledger", f"{len(done)} of {len(items)} files already uploaded with the same SHA-256 (cloud/modal_ledger.csv)")
    elif items:
        pf.add(PASS, "CPE-1 ledger", "no upload recorded yet")
    pf.once_only("upload-cache")
    pf.git_clean()


def _spec_rel(pf: Preflight, purpose: str | None = None) -> str:
    """Run-relative spec file: ``--specs`` (f9-import), else the plan of the wave (F9) or the S5 train spec file."""

    if purpose is None and pf.opt.specs:
        return pf.opt.specs
    if purpose == "s5":
        return f"data_efficiency/f9/train_wave{pf.opt.wave}_specs.json"
    return f"neural_f9/plan/wave{pf.opt.wave}_specs.json"


def step_f9_train(pf: Preflight) -> None:
    from .neural.f9_config import resolve_weights, training_config
    from .neural.fit_spec import read_spec_file
    from .neural.cloud import modal_settings
    from .runs import sha256_file, sha256_json

    pf.modal_pin()
    pf.modal_token()
    if pf.side == "host":
        pf.uvx()
    else:
        pf.add(UNKNOWN, "uvx", "modal_train.py runs on the host: run this preflight on the host as well")
    pf.qualification_report(required=True)
    cfg = pf.f9_config()
    export = Path(pf.opt.specs) if pf.opt.specs else Path("outputs") / "cloud" / f"wave{pf.opt.wave}_specs.json"
    export = export if export.is_absolute() else Path.cwd() / export
    payload = None
    if not export.is_file():
        pf.add(FAIL, "exported spec file", f"{export.as_posix()} missing: export the run's spec file to outputs/cloud first")
    else:
        try:
            payload = read_spec_file(export)
        except (ValueError, KeyError) as exc:
            pf.add(FAIL, "exported spec file", f"{export.as_posix()}: {exc}")
    if payload is not None:
        problems = []
        if int(payload["wave"]) != pf.opt.wave:
            problems.append(f"wave {payload['wave']} != --wave {pf.opt.wave}")
        if payload["run_id"] != pf.run.run_id:
            problems.append(f"planned for run {payload['run_id']}")
        if payload["config_sha256"] != sha256_json(training_config(cfg)):
            problems.append("its training configuration differs from configs/f9_cnn.yaml (train-f9 refuses the same; re-plan)")
        try:
            pinned = resolve_weights(cfg, pf.study.root)["sha256"]
            if str(payload["weights"].get("sha256")).lower() != pinned:
                problems.append("weights sha256 differs from registry/weights.yaml")
        except (KeyError, ValueError, FileNotFoundError) as exc:
            problems.append(f"weights: {exc}")
        modal_cfg = modal_settings(cfg["backend"])
        projected = len(payload["specs"]) * float(modal_cfg["projected_minutes_per_fit"]) / 60.0 * float(modal_cfg["gpu_usd_per_hour"])
        pf.add(FAIL if problems else PASS, "exported spec file",
               "; ".join(problems) if problems else f"{export.name}: wave {payload['wave']}, purpose {payload.get('purpose')}, "
                                                    f"{payload['n_specs']} fits, projected ${projected:.2f} (modal_train.py projection)")
        results = export.with_name(f"wave{pf.opt.wave}_modal_results.json")
        if results.is_file():
            pf.add(WARN, "once-only Modal wave", f"{results.name} exists: this wave was already mapped; a rerun maps every fit again "
                                                 "(completed fit directories on the Volume are returned unchanged)")
    if not pf.run.path.is_dir() and not pf.run_visible():
        pf.add(UNKNOWN, "run root", "not visible from the host: run `preflight --step f9-train` in the container too (spec copy, upload, freeze)")
        return
    if not pf.run_root():
        return
    rel = _spec_rel(pf, str(payload.get("purpose", "f9")) if payload is not None else "f9")
    pf.artifact(rel)
    run_copy = pf.run.path / rel
    if payload is not None and run_copy.is_file():
        same = sha256_file(run_copy) == sha256_file(export)
        pf.add(PASS if same else FAIL, "spec export", f"{export.name} is {'byte-identical to' if same else 'DIFFERENT from'} {rel}")
    if payload is not None and payload.get("purpose", "f9") == "f9" and (pf.run.path / pf.freeze_paths()["F9"]).is_file():
        pf.add(FAIL, "F9 not frozen", "neural_f9/f9_bundle.json exists: F9 fits after the freeze never enter it (import-f9-fits refuses)")
    manifest = pf.run.path / "neural_cache" / "acdc_dev" / "manifest.csv"
    if payload is not None and manifest.is_file():
        same = sha256_file(manifest) == payload["cache_manifest_sha256"]
        pf.add(PASS if same else FAIL, "cache manifest", f"neural_cache/acdc_dev/manifest.csv {'matches' if same else 'DIFFERS from'} the spec's cache_manifest_sha256")
    uploaded = pf.ledger_uploads()
    if manifest.is_file():
        import csv

        with manifest.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        from .neural.cloud import cache_file_name

        wanted = {f"neural_cache/acdc_dev/{(r.get('npz_file') or '').strip() or cache_file_name(r['sample_id'])}": str(r["npz_sha256"]).lower() for r in rows}
        wanted["neural_cache/acdc_dev/manifest.csv"] = sha256_file(manifest)
        missing = [k for k, v in wanted.items() if uploaded.get(k) != v]
        pf.add(FAIL if missing else PASS, "CPE-1 upload", f"{len(missing)} of {len(wanted)} cache files have no upload row with this SHA-256 "
                                                          "in cloud/modal_ledger.csv (run upload-cache)" if missing else
               f"all {len(wanted)} cache files uploaded with matching SHA-256 (cloud/modal_ledger.csv)")


def step_f9_import(pf: Preflight) -> None:
    from .neural.f9_stages import FIT_ID, f9_paths
    from .neural.fit_spec import read_spec_file
    from .neural.selection import NE_INSUFFICIENT_CLASS_SUPPORT, STATUS_COMPLETE
    from .runs import sha256_file

    if not pf.run_root():
        return
    paths = f9_paths(pf.run)
    rel = _spec_rel(pf)
    spec_path = pf.run.path / rel
    if not spec_path.is_file():
        pf.add(FAIL, "spec file", f"{rel} missing")
        return
    try:
        payload = read_spec_file(spec_path)
    except (ValueError, KeyError) as exc:
        pf.add(FAIL, "spec file", f"{rel}: {exc}")
        return
    if int(payload["wave"]) != pf.opt.wave:
        pf.add(FAIL, "spec file", f"{rel} is a wave-{payload['wave']} spec file, not wave {pf.opt.wave}")
        return
    pf.artifact(rel)
    if payload.get("purpose", "f9") == "f9" and paths.bundle.is_file():
        pf.add(FAIL, "F9 not frozen", "neural_f9/f9_bundle.json exists: import-f9-fits refuses F9 fits after the freeze")
    planned = {s.fit_id: s for s in payload["fit_specs"]}
    known = set(planned)
    siblings = [paths.wave_specs(1), paths.wave_specs(2)]
    if pf.opt.specs:
        siblings += [spec_path.with_name(spec_path.name.replace(f"wave{pf.opt.wave}", f"wave{o}")) for o in (1, 2) if o != pf.opt.wave]
    for sibling in siblings:
        if sibling.is_file() and sibling.resolve() != spec_path.resolve():
            known |= {s.fit_id for s in read_spec_file(sibling)["fit_specs"]}
    source = Path(pf.opt.source) if pf.opt.source else Path("outputs") / "cloud" / "runs" / pf.run.run_id
    source = (source if source.is_absolute() else Path.cwd() / source).resolve()
    if not source.is_dir():
        pf.add(FAIL, "download folder", f"{source.as_posix()} is not a directory (uvx --from modal==<pin> modal volume get ...)")
        return
    present = {d.name: d for d in source.iterdir() if d.is_dir() and FIT_ID.fullmatch(d.name)}
    unknown = sorted(set(present) - known)
    missing = sorted(set(planned) - set(present))
    pf.add(FAIL if unknown else PASS, "download folder: unknown fits", f"{len(unknown)} fit directories belong to no planned wave (e.g. {unknown[:3]})"
           if unknown else f"{len(present)} fit directories, all planned")
    pf.add(FAIL if missing else PASS, "download folder: planned fits", f"{len(missing)} of {len(planned)} wave-{pf.opt.wave} fits missing (e.g. {missing[:3]})"
           if missing else f"all {len(planned)} planned fits present")
    bad, conflicts = [], []
    for fid in sorted(set(planned) & set(present)):
        folder, spec = present[fid], planned[fid]
        try:
            record = json.loads((folder / "record.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            bad.append(f"{fid}: record.json missing or unreadable")
            continue
        if record.get("fit_id") != fid or record.get("spec") != spec.to_dict():
            bad.append(f"{fid}: fit id or spec does not match the plan")
        elif record.get("config_sha256") != payload["config_sha256"]:
            bad.append(f"{fid}: fitted with another training configuration")
        elif record.get("status") not in (STATUS_COMPLETE, NE_INSUFFICIENT_CLASS_SUPPORT):
            bad.append(f"{fid}: status {record.get('status')} is not importable")
        elif record["status"] == STATUS_COMPLETE and not (folder / "model.pt").is_file() and not spec.is_selection_fit:
            bad.append(f"{fid}: model.pt missing (refit checkpoints are required)")
        elif pf.opt.hash and (folder / "model.pt").is_file() and sha256_file(folder / "model.pt") != record.get("checkpoint_sha256"):
            bad.append(f"{fid}: model.pt SHA-256 differs from its record")
        dest = paths.fit_dir(fid) / "record.json"
        if dest.is_file() and sha256_file(dest) != sha256_file(folder / "record.json"):
            conflicts.append(fid)
    pf.add(FAIL if bad else PASS, "fit records", f"{len(bad)} not importable: {bad[:3]}" if bad else
           f"{len(set(planned) & set(present))} records match the plan and configuration" + ("" if pf.opt.hash else " (model.pt hashed with --hash; import-f9-fits always hashes)"))
    pf.add(FAIL if conflicts else PASS, "existing checkpoints", f"{len(conflicts)} fits already imported with another record.json (never overwritten): {conflicts[:3]}"
           if conflicts else "no conflicting record.json in neural_f9/checkpoints")
    name = f"import-f9-fits:wave{pf.opt.wave}" if not pf.opt.specs else f"import-f9-fits:{spec_path.stem}"
    pf.once_only(name)
    pf.disk()


def step_f9_freeze(pf: Preflight) -> None:
    from .neural.f9_stages import _load_records, f9_paths
    from .neural.fit_spec import read_spec_file
    from .neural.selection import STATUS_COMPLETE
    from .runs import sha256_file

    if not pf.run_root():
        return
    paths = f9_paths(pf.run)
    payloads = {}
    for wave in (1, 2):
        rel = f"neural_f9/plan/wave{wave}_specs.json"
        if pf.artifact(rel):
            payloads[wave] = read_spec_file(pf.run.path / rel)
    pf.record("plan-f9")
    pf.record("select-f9")
    pf.artifact("neural_f9/selection/selection_table.parquet")
    if 1 in payloads and 2 in payloads:
        same = payloads[2].get("wave1_specs_sha256") == sha256_file(paths.wave_specs(1))
        pf.add(PASS if same else FAIL, "wave-2 derivation", "wave2_specs.json was derived from the current wave-1 spec file" if same
               else "wave2_specs.json was not derived from the current wave-1 spec file (freeze-f9 refuses)")
        for wave, payload in payloads.items():
            try:
                recs = _load_records(paths, payload)
                pf.add(PASS, f"wave-{wave} fit records", f"all {len(recs)} present and match the plan and configuration")
            except (FileNotFoundError, ValueError) as exc:
                pf.add(FAIL, f"wave-{wave} fit records", str(exc))
                continue
            if not any(n in pf.records() for n in (f"import-f9-fits:wave{wave}", f"train-f9:wave{wave}_specs")):
                pf.add(WARN, f"wave-{wave} provenance", f"neither import-f9-fits:wave{wave} nor train-f9:wave{wave}_specs is recorded")
            if wave == 2:
                bad = [fid for fid, r in recs.items() if r.get("status") != STATUS_COMPLETE
                       or not (paths.fit_dir(fid) / "model.pt").is_file() or sha256_file(paths.fit_dir(fid) / "model.pt") != r.get("checkpoint_sha256")]
                pf.add(FAIL if bad else PASS, "wave-2 checkpoints", f"{len(bad)} refits incomplete, missing or changed: {bad[:3]}" if bad
                       else f"all {len(recs)} refits COMPLETE; every model.pt matches checkpoint_sha256")
    pf.add(WARN, "cross-hardware check", "SPECIFICATION.md 14.6 step 4 and 14.7 require the Modal-versus-local logit check (max |delta logit|, "
                                         "argmax agreement) before freeze-f9; no command of this build implements it: record it before freezing")
    if "infer-f9:acdc_dev" in pf.records() and not paths.bundle.is_file():
        pf.add(WARN, "infer-f9:acdc_dev before freeze-f9", "already recorded without the F9 freeze record: its encoders were never checked "
                                                           "against the bundle and, in a frozen run, it cannot be re-run after freeze-f9 (exit 7)")
    pf.once_only("freeze-f9")
    if paths.bundle.is_file():
        pf.freeze_file("F9")
    _extraction_notice(pf, "F9")
    pf.git_clean()


def step_f9b(pf: Preflight) -> None:
    if not pf.run_root():
        return
    pf.freeze_file("F9")
    rec = pf.record("infer-f9:acdc_dev", hint="run infer-f9 --split acdc_dev (tmmri-gpu) after freeze-f9")
    if rec is not None:   # infer_f9 adds the F9 freeze record to its inputs (and checks every encoder against it) only once it exists
        after = pf.run.portable(pf.run.path / pf.freeze_paths()["F9"]) in rec.inputs
        pf.add(PASS if after else WARN, "infer-f9:acdc_dev after freeze-f9", "the record lists the F9 freeze record as an input: its encoders were "
               "checked against the frozen bundle" if after else "recorded before freeze-f9: its encoders were not checked against the frozen "
               "bundle, and in a frozen run it cannot be re-run (exit 7); report it")
    pf.artifact("neural_f9/activations.parquet", recorded=False, missing_hint="infer-f9 writes the combined activations")
    pf.recorded_inputs(["semantics/B_ACDC.parquet", "formal/A_ACDC.parquet", "contracts/split_manifest.parquet", "contracts/inner_folds.parquet",
                        "contracts/cohort_ACDC.parquet"])
    pf.once_only("fit-probes")
    if (pf.run.path / pf.freeze_paths()["F9b"]).is_file():
        pf.freeze_file("F9b")
    _extraction_notice(pf, "F9b")
    pf.git_clean()


def _trust_inputs(pf: Preflight) -> dict[str, list[str]]:
    """acdc_dev inputs of each trust predictor and representation (the files tmmri.experiments.l57_stages reads)."""

    from .secondary_stages import lane_config

    probes = lane_config(pf.study, "probes")
    inp, seed = probes["inputs"], int(probes["f9b"]["seeds"][0])
    fmt = {"partition": "acdc_dev", "dataset": "ACDC", "seed": seed}
    c4 = str(inp["descriptors"]["predictions"])
    audited = [c4] if (pf.run.path / c4).is_file() else ["formal/A_ACDC.parquet", str(inp["descriptors"]["crossfit_dev"]), "semantics/B_ACDC.parquet"]
    return {   # predictor_outputs / representation_matrix on acdc_dev (has_predictor, has_representation)
        "audited_predictor": audited,
        "d20_predictor": [str(inp["d20"]["predictions"])],
        "f9": [str(inp["f9"]["predictions"]).format(**fmt)],
        "descriptors": ["formal/A_ACDC.parquet"],
        "d20": [str(inp["d20"]["features"]).format(**fmt)],
        "f9b": [str(inp["f9"]["activations"]).format(**fmt)],
    }


def step_trust_fit(pf: Preflight) -> None:
    from .secondary_stages import lane_config

    if not pf.run_root():
        return
    trust = lane_config(pf.study, "trust")
    pf.recorded_inputs(["contracts/split_manifest.parquet", "contracts/cohort_ACDC.parquet", "semantics/B_ACDC.parquet", "formal/A_ACDC.parquet"])
    needed = _trust_inputs(pf)
    for item in [*trust["predictors"], *trust["representations"]]:
        files = needed[str(item)]
        absent = [f for f in files if not (pf.run.path / f).is_file()]
        pf.add(FAIL if absent else PASS, f"trust input {item}", f"not recorded: {absent} (fit-trust refuses; --allow-missing records it NE, not executed)"
               if absent else "acdc_dev out-of-fold inputs present")
    for bundle in ("D20", "F9", "F9b"):
        if (pf.run.path / pf.freeze_paths()[bundle]).is_file():
            pf.freeze_file(bundle)
        else:
            pf.add(WARN, f"freeze {bundle}", "not frozen: its trust inputs would be unfrozen or NE")
    pf.once_only("fit-trust")
    if (pf.run.path / pf.freeze_paths()["S1-S4"]).is_file():
        pf.freeze_file("S1-S4")
    _extraction_notice(pf, "S1-S4")
    pf.git_clean()


def _evaluation_stages(pf: Preflight, split: str) -> None:
    """Prerequisites and once-only state of the evaluations of one held-out partition (shared by eval-historical and mms2-evaluate)."""

    from .secondary_stages import lane_config

    probes = lane_config(pf.study, "probes")["inputs"]
    freezes = pf.freeze_paths()
    rec = pf.record(f"evaluate:{split}", status="EVALUATED", verify_outputs=False, hint=f"evaluate --split {split} --no-fit first")
    if rec is not None:
        pf.artifact(str(probes["descriptors"]["bhat"]).format(partition=split))
        pf.artifact(str(probes["descriptors"]["decisions"]).format(partition=split))
    pf.once_only(f"evaluate-cart-surrogate:{split}")
    cache = f"neural_cache/{split}"
    if (pf.run.path / cache / "manifest.csv").is_file():
        pf.artifact(cache)
    else:
        pf.add(WARN, f"cache manifest {split}", f"{cache}/manifest.csv not built yet: infer-f9 --split {split} refuses (exit 1, nothing recorded) "
                                                f"until build-tensor-cache --split {split} has run")
    pf.once_only(f"build-tensor-cache:{split}")
    pf.once_only(f"extract-deep-features:{split}", freeze_files=[freezes["D20"]])
    pf.once_only(f"infer-f9:{split}", freeze_files=[freezes["F9"]])
    inputs = {"d20": [str(probes["d20"]["features"]).format(partition=split)], "f9": [f"neural_f9/predictions_{split}.parquet"],
              "f9b": [f"neural_f9/activations_{split}.parquet"]}
    for bundle in ALT_BUNDLES:
        absent = [f for f in inputs[bundle] if not (pf.run.path / f).is_file()]
        if absent:
            pf.add(WARN, f"inputs evaluate-alt-bundles {bundle}:{split}", f"not yet recorded: {absent}; run the producing stage first (the "
                                                                         "evaluation refuses with exit 1 and records nothing)")
        else:
            pf.recorded_inputs(inputs[bundle])
        pf.once_only(f"evaluate-alt-bundles:{bundle}:{split}")
    pf.once_only(f"evaluate-trust:{split}")


def step_eval_historical(pf: Preflight) -> None:
    parts = pf.partitions()
    if not pf.run_root():
        return
    pf.frozen_bundle()
    for bundle in FREEZE_BUNDLES:      # an absent freeze: that bundle's evaluations refuse and are reported as not executed
        pf.freeze_file(bundle, required=False)
    for split in parts:
        pf.lock_for(split)
        pf.staging_manifest(DATASET_OF[split])
        _evaluation_stages(pf, split)
    pf.git_clean()
    pf.disk()


def step_s5(pf: Preflight) -> None:
    from .experiments.data_centric import data_centric_config

    if not pf.run_root():
        return
    cfg = data_centric_config(pf.study)["data_efficiency"]
    pf.recorded_inputs(["contracts/split_manifest.parquet", "contracts/cohort_ACDC.parquet", "contracts/inner_folds.parquet",
                        "formal/A_ACDC.parquet", "semantics/B_ACDC.parquet"])
    pf.once_only("run-data-efficiency:descriptor")
    pf.artifact("neural_cache/acdc_dev", required=False, missing_hint="f9 steps need build-tensor-cache --split acdc_dev")
    if not (pf.run.path / pf.freeze_paths()["F9"]).is_file():
        pf.add(WARN, "F9 frozen", "neural_f9/f9_bundle.json absent: the S5 F9 waves run after freeze-f9 (SPECIFICATION.md 14.6; scripts/cloud/README.md)")
    else:
        pf.freeze_file("F9")
    for step in ("f9-wave1", "f9-wave2", "f9-infer", "f9-aggregate"):
        pf.once_only(f"run-data-efficiency:{step}")
    full = pf.run.path / str(cfg["f9"]["full_fraction_outputs"])
    have = all((full / n).is_file() for n in ("predictions.parquet", "activations.parquet"))
    pf.add(PASS if have else WARN, "S5 f = 1.0 outputs", f"{cfg['f9']['full_fraction_outputs']} predictions/activations present" if have
           else "infer-f9 --split acdc_dev outputs absent: f9-aggregate refuses until they exist")
    pf.add(WARN, "results freeze", "S5 is reported as not executed if it misses the results freeze (SPECIFICATION.md 14.6)")
    pf.disk()


def step_s6(pf: Preflight) -> None:
    from .experiments.data_centric import data_centric_config

    if not pf.run_root():
        return
    cfg = data_centric_config(pf.study)["simulated_shift"]
    vendors = [str(v) for v in cfg["vendors"]] if pf.opt.vendor == "all" else [pf.opt.vendor]
    unknown = [v for v in vendors if v not in {str(k) for k in cfg["vendors"]}]
    if unknown:
        pf.add(FAIL, "--vendor", f"unknown vendor codes {unknown}; configured {list(cfg['vendors'])}")
        vendors = [v for v in vendors if v not in unknown]
    pf.frozen_bundle()
    pf.record("evaluate:acdc_test", status="EVALUATED", verify_outputs=False)
    pf.recorded_inputs(["contracts/split_manifest.parquet", "contracts/cohort_MMS.parquet", "contracts/cohort_ACDC.parquet", "semantics/B_ACDC.parquet",
                        "reconstructed/bhat_acdc_test.parquet", "inference/d_model_acdc_test.parquet"])
    for ds in ("ACDC", "MMS"):
        pf.staging_manifest(ds)
    pf.once_only("run-simulated-shift:geometry")
    freezes = pf.freeze_paths()
    for bundle in ("D20", "F9", "F9b"):
        if (pf.run.path / freezes[bundle]).is_file():
            pf.freeze_file(bundle)
        else:
            pf.add(WARN, f"freeze {bundle}", f"absent: the --step {'d20' if bundle == 'D20' else 'f9'} legs refuse until it exists")
    for name, stage in (("D20 features of acdc_test", "extract-deep-features:acdc_test"), ("F9 inference of acdc_test", "infer-f9:acdc_test")):
        if stage in pf.records():
            pf.record(stage)
        else:
            pf.add(WARN, name, f"{stage} not recorded: that leg refuses (exit 1, nothing recorded) until it is")
    for v in vendors:
        for suffix in ("", ":tensors", ":d20", ":f9"):
            pf.once_only(f"run-simulated-shift:{v}{suffix}")
    pf.git_clean()
    pf.disk()


def step_tag(pf: Preflight) -> None:
    import importlib.util

    from .stages import mms2_lock_tag

    tag = mms2_lock_tag(pf.study)
    pf.add(PASS if pf.study.current_protocol_tag == tag else FAIL, "configured amendment tag",
           f"configs/study_v6.yaml amendment_tags ends with {pf.study.current_protocol_tag}; configs/cohorts.yaml mms2.lock_tag {tag}")
    kind = tag_object_type(tag)
    if kind is None:
        pf.add(PASS, f"tag {tag}", "not yet created: ready for the owner's git tag -s")
    else:
        pf.tag_v601(required=True)
        if pf.checks[-1].status == PASS:
            pf.checks[-1] = Check(WARN, f"tag {tag}", "already signed and HEAD descends from it: nothing to sign")
    pf.git_clean()
    proc = _git(["config", "--get", "user.signingkey"])
    fmt = _git(["config", "--get", "gpg.format"])
    has_key = proc is not None and proc.returncode == 0 and bool(proc.stdout.strip())
    pf.add(PASS if has_key else WARN, "signing key", f"user.signingkey configured (gpg.format={fmt.stdout.strip() if fmt is not None and fmt.returncode == 0 else 'openpgp'}; "
                                                     "value not printed)" if has_key else "user.signingkey not configured: git tag -s cannot sign")
    amend = (PROJECT_ROOT / "protocol" / "amendments.md").read_text(encoding="utf-8")
    pf.add(PASS if tag in amend else FAIL, "protocol/amendments.md", f"has a {tag} row" if tag in amend else f"no {tag} row")
    spec_head = (PROJECT_ROOT / "SPECIFICATION.md").read_text(encoding="utf-8").splitlines()[:8]
    version = tag.removeprefix("protocol-v")
    pf.add(PASS if any(f"version:** {version}" in line for line in spec_head) else FAIL, "SPECIFICATION.md version",
           f"header states version {version}" if any(f"version:** {version}" in line for line in spec_head) else f"header does not state version {version}")
    try:
        from .ingest.mms2 import external_ids, load_split_manifest

        n = len(external_ids(load_split_manifest()))
        pf.add(PASS, "M&Ms-2 split manifest", f"registry/sources/mms2_split_manifest.csv verified against its receipt; {n} mms2_external subjects")
    except Exception as exc:   # noqa: BLE001 - any failure of the tracked manifest is a FAIL line, not a crash
        pf.add(FAIL, "M&Ms-2 split manifest", str(exc))
    script = PROJECT_ROOT / "scripts" / "maintenance" / "check_repository_layout.py"
    spec = importlib.util.spec_from_file_location("tmmri_preflight_layout", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    violations = module.check_repository(PROJECT_ROOT)
    pf.add(FAIL if violations else PASS, "layout check", f"{len(violations)} violations: {violations[:3]}" if violations else "scripts/maintenance/check_repository_layout.py passes")
    pf.add(WARN, "tests", "the container pytest suite must be green before signing (not run by the preflight)")
    if pf.run.path.is_dir():
        for bundle in FREEZE_BUNDLES:
            path = pf.run.path / pf.freeze_paths()[bundle]
            pf.add(PASS if path.is_file() else WARN, f"freeze {bundle} before extraction", "frozen" if path.is_file()
                   else "not frozen yet: freeze it before extract-mms2 or its mms2_external result is post hoc")
    else:
        pf.add(UNKNOWN, "run-root freezes", "run root not visible here; `preflight --step mms2-extract` in the container lists them")


def step_mms2_extract(pf: Preflight) -> None:
    from .ingest.mms2 import RECEIPT_JSON, archive_expectation
    from .ingest.sources import sha256_file, source_binding

    unlocked = pf.tag_v601()
    binding = source_binding("mms2")
    archive = Path(pf.opt.archive or binding["archive"]["reference_copy"])
    if pf.side == "container" and not archive.exists():
        pf.add(UNKNOWN, "M&Ms-2 archive", f"{archive.as_posix()} is not visible in the container (extract-mms2 runs on the host)")
    else:
        expected = archive_expectation(archive)
        if not archive.is_file():
            pf.add(FAIL, "M&Ms-2 archive", f"{archive.as_posix()} not found")
        else:
            size = archive.stat().st_size
            pf.add(PASS if size == expected["bytes"] else FAIL, "M&Ms-2 archive size",
                   f"{size} bytes {'equals' if size == expected['bytes'] else 'DIFFERS from'} the expected {expected['bytes']}")
            if not pf.opt.hash:
                pf.add(WARN, "M&Ms-2 archive SHA-256", "not computed (pass --hash after the tag; extract-mms2 hashes it before extracting)")
            elif not unlocked:
                pf.add(WARN, "M&Ms-2 archive SHA-256", "not computed: no M&Ms-2 byte is read before the signed tag (SPECIFICATION.md 4.6)")
            else:
                same = sha256_file(archive) == expected["sha256"]
                pf.add(PASS if same else FAIL, "M&Ms-2 archive SHA-256", "matches registry/sources/mms2.yaml" if same else "MISMATCH")
    tar = Path(str(binding["archive"].get("extractor") or "C:/Windows/System32/tar.exe"))
    if pf.side == "container":
        pf.add(UNKNOWN, "extractor", f"{tar.as_posix()} is the host's Windows bsdtar (extract-mms2 runs on the host)")
    else:
        pf.add(PASS if tar.is_file() else FAIL, "extractor", f"{tar.as_posix()} {'present' if tar.is_file() else 'missing (GNU tar cannot read RAR)'}")
    dest = pf.ctx.source_root("mms2")
    if not dest.exists() or (dest.is_dir() and not any(dest.iterdir())):
        pf.add(PASS, "destination", f"{dest.as_posix()} empty or absent: the 720 files land here")
    elif (dest / RECEIPT_JSON).is_file():
        pf.add(WARN, "once-only extraction", f"{dest.as_posix()} holds an extraction receipt: extract-mms2 would SKIP if archive, list and files "
                                             "verify, else refuse (it never overwrites)")
    else:
        pf.add(FAIL, "destination", f"{dest.as_posix()} is not empty and holds no extraction receipt: extract-mms2 refuses")
    if pf.run.path.is_dir():
        pf.freezes_before_extraction()
        for bundle in FREEZE_BUNDLES:
            if (pf.run.path / pf.freeze_paths()[bundle]).is_file():
                pf.freeze_file(bundle)
    else:
        pf.add(UNKNOWN, "freeze order", "run root not visible here: run `preflight --step mms2-extract` in the container for the SPECIFICATION.md "
                                        "14.6 freeze records")
    pf.add(WARN if pf.side == "container" else PASS, "where", "extract-mms2 is host only (Windows tar.exe); pass no --run-id on the host "
                                                              "(it would create a stray host run root)")
    pf.git_clean()
    pf.disk(dest)


def step_mms2_evaluate(pf: Preflight) -> None:
    from .stages import BundleError, CohortValidationError, mms2_extraction_times, require_mms2_validated

    pf.tag_v601()
    if not pf.run_root():
        return
    ext = mms2_extraction_times(pf.ctx)
    pf.add(PASS if ext["exists"] else FAIL, "M&Ms-2 extraction record", f"first extraction started {ext['first_extraction_started_utc']}"
           if ext["exists"] else "no receipt or extraction_MMS2.json: run extract-mms2 on the host first")
    pf.frozen_bundle()
    for ds in ("ACDC", "MMS"):
        pf.staging_manifest(ds)
    pf.staging_manifest("MMS2", required=False)
    for bundle in FREEZE_BUNDLES:
        if (pf.run.path / pf.freeze_paths()[bundle]).is_file():
            pf.freeze_file(bundle)
        else:
            pf.add(WARN, f"freeze {bundle}", "absent: its mms2_external evaluation is not executed")
    if ext["exists"]:
        pf.freeze_order_labels()
    if "validate-cohort:MMS2" in pf.records() and (pf.run.path / "contracts").is_dir():
        try:
            require_mms2_validated(pf.ctx, "preflight")
            pf.add(PASS, "record validate-cohort:MMS2", "passed; its contracts are unchanged (duplicate exclusion in force)")
        except (BundleError, CohortValidationError) as exc:
            pf.add(FAIL, "record validate-cohort:MMS2", str(exc))
    else:
        pf.add(PASS, "once-only validate-cohort:MMS2", "not yet recorded; validate-cohort --dataset mms2 runs first (always strict)")
    for stage in ("build-semantics:MMS2", "extract-formal:MMS2"):
        pf.once_only(stage)
    _evaluation_stages(pf, "mms2_external")
    for stage in ("audit-shift:MMS2", "summarize-predictor:mms2_external", "summarize-predictor-extended:mms2_external", "run-exploratory:B8"):
        pf.once_only(stage)
    pf.git_clean()
    pf.disk()


def step_results_freeze(pf: Preflight) -> None:
    from .manuscript_tables import run_protocol_tag, spec_matches_tag

    pf.tag_v601()
    if not pf.run_root():
        return
    pf.frozen_bundle()
    run_tag = run_protocol_tag(pf.run, pf.study)
    matches = spec_matches_tag(PROJECT_ROOT / "SPECIFICATION.md", run_tag)
    pf.add({True: PASS, False: FAIL, None: WARN}[matches], "SPECIFICATION.md vs run tag",
           {True: f"equals {run_tag}:SPECIFICATION.md (Table 1)", False: f"differs from {run_tag}:SPECIFICATION.md: build-manuscript-tables refuses",
            None: f"git cannot show {run_tag}:SPECIFICATION.md (tag absent?)"}[matches])
    for split in ("acdc_test", "mms_external"):
        pf.record(f"evaluate:{split}", status="EVALUATED", verify_outputs=False)
    if pf.records().get("evaluate:mms2_external") is None:
        pf.add(WARN, "record evaluate:mms2_external", "not recorded: H3 is dropped (go/no-go rule 3) and reported as not executed")
    registered = ["fit-cart-surrogate", "fit-deep-encoder", "freeze-f9", "fit-probes", "fit-trust", "run-data-efficiency:descriptor",
                  "run-data-efficiency:f9-aggregate", "audit-shift:MMS2", "summarize-predictor:mms2_external"]
    for split in ("acdc_test", "mms_external", "mms2_external"):
        registered += [f"evaluate-cart-surrogate:{split}", f"evaluate-trust:{split}"] + [f"evaluate-alt-bundles:{b}:{split}" for b in ALT_BUNDLES]
    registered += [f"run-exploratory:B{k}" for k in range(1, 9)]
    missing = [s for s in registered if s not in pf.records()]
    pf.add(WARN if missing else PASS, "registered and exploratory records", f"{len(missing)} not recorded (reported as not executed, with reasons): {missing}"
           if missing else "all recorded")
    for stage in ("build-comparison-tables", "build-manuscript-tables", "build-manuscript-figures"):
        rec = pf.records().get(stage)
        if stage == "build-manuscript-tables" and rec is not None:
            now = pf.records().get("evaluate:mms2_external") is not None
            if bool(rec.params.get("include_mms2")) != now:
                pf.add(FAIL, f"once-only {stage}", f"recorded with include_mms2={rec.params.get('include_mms2')}, now {now}: a rerun REFUSES (exit 7)")
                continue
        pf.once_only(stage)
    if (pf.run.path / "release_manifest.json").is_file():
        pf.add(WARN, "release_manifest.json", "exists; write-release-manifest rewrites it whenever the run root changed (not once-only)")
    pf.git_clean()
    pf.disk()


STEP_FUNCTIONS: dict[str, Callable[[Preflight], None]] = {
    "replication": step_replication, "f8": step_f8, "b5": step_b5, "b6": step_b6, "tensor-cache": step_tensor_cache,
    "weights": step_weights, "d20": step_d20, "modal-qualify": step_modal_qualify, "upload": step_upload, "f9-train": step_f9_train,
    "f9-import": step_f9_import, "f9-freeze": step_f9_freeze, "f9b": step_f9b, "trust-fit": step_trust_fit,
    "eval-historical": step_eval_historical, "s5": step_s5, "s6": step_s6, "tag": step_tag, "mms2-extract": step_mms2_extract,
    "mms2-evaluate": step_mms2_evaluate, "results-freeze": step_results_freeze,
}


# ============================================================================ CLI


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def add_preflight_parser(sub: argparse._SubParsersAction) -> None:
    text = ("v6.0.1: read-only prerequisite checks of one execution step (docs/V601_EXECUTION_RUNBOOK.md); "
            "PASS/WARN/FAIL/UNKNOWN lines, exit 0 when nothing failed, else 1; writes nothing")
    p = sub.add_parser("preflight", help=text, description=text)
    p.add_argument("--run-id", required=True, help="run id, pattern v6-<purpose>-YYYYMMDD-HHMM")
    p.add_argument("--step", required=True, choices=list(STEPS))
    p.add_argument("--study", default="configs/study_v6.yaml", help="protocol study configuration")
    p.add_argument("--split", action="append", default=[], choices=list(SPLITS),
                   help="tensor-cache, d20, eval-historical: restrict to this partition (repeatable; default the step's partitions)")
    p.add_argument("--wave", type=int, default=1, choices=[1, 2], help="f9-train, f9-import: the wave")
    p.add_argument("--specs", default=None, help="f9-train: the exported spec file (default outputs/cloud/wave<N>_specs.json); "
                                                 "f9-import: the run-relative spec file (default neural_f9/plan/wave<N>_specs.json)")
    p.add_argument("--source", default=None, help="f9-import: the downloaded fit folder (default outputs/cloud/runs/<run_id>)")
    p.add_argument("--archive", default=None, help="mms2-extract: archive path (default configs/source_bindings/mms2.json reference_copy)")
    p.add_argument("--vendor", default="all", help="s6: M&Ms-1 vendor code or all")
    p.add_argument("--hash", action="store_true", help="also hash large files: the M&Ms-2 archive (only after the signed tag), weight files, "
                                                        "downloaded model.pt files")
    p.add_argument("--min-free-gb", type=float, default=20.0, help="WARN below this free space (FAIL below 5 GiB)")
    p.add_argument("--data-root", default=None, help="runtime data root (default TMMRI_DATA_ROOT or data/)")
    p.add_argument("--artifact-root", default=None, help="run-root parent (default TMMRI_ARTIFACT_ROOT or artifacts/)")
    p.add_argument("--acdc-source", default=None)
    p.add_argument("--mms-source", default=None)
    p.add_argument("--mms2-source", default=None, help="M&Ms-2 source area (default TMMRI_MMS2_SOURCE, then ./data/raw/mms2)")


def build_preflight(args: argparse.Namespace, *, side: str | None = None) -> Preflight:
    """The read-only context: ``RunRoot(create=False)`` and a :class:`tmmri.stages.StageContext` that never writes."""

    from .config import load_study
    from .paths import ARTIFACT_ROOT, DATA_ROOT
    from .runs import RunRoot
    from .stages import StageContext

    study = load_study(args.study)
    artifact_root = _path(args.artifact_root) if args.artifact_root else ARTIFACT_ROOT
    data_root = _path(args.data_root) if args.data_root else DATA_ROOT
    run = RunRoot(args.run_id, artifact_root, subfolders=study.study.get("artifact_subfolders"), create=False)
    sources = {k: _path(v) for k, v in (("acdc", args.acdc_source), ("mms", args.mms_source), ("mms2", args.mms2_source)) if v}
    ctx = StageContext(study=study, run=run, data_root=data_root, sources=sources)
    options = Options(split=list(args.split), wave=int(args.wave), specs=args.specs, source=args.source, archive=args.archive, vendor=str(args.vendor),
                      hash=bool(args.hash), min_free_gb=float(args.min_free_gb),
                      explicit_artifact_root=bool(args.artifact_root or os.environ.get("TMMRI_ARTIFACT_ROOT")),
                      explicit_data_root=bool(args.data_root or os.environ.get("TMMRI_DATA_ROOT")))
    return Preflight(args.step, ctx, side=side or detect_side(), options=options)


def run_preflight(args: argparse.Namespace) -> int:
    """Print the check lines of ``args.step``; 0 when no check failed, else 1."""

    try:
        pf = build_preflight(args)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(Check(FAIL, "preflight setup", str(exc)).line())
        print(f"RESULT  NO-GO for step {args.step}: the preflight could not start (preflight wrote nothing)")
        return 1
    try:
        STEP_FUNCTIONS[args.step](pf)
    except Exception as exc:   # noqa: BLE001 - an unexpected error is a failed preflight, never a silent pass
        pf.add(FAIL, "preflight error", f"{type(exc).__name__}: {exc}")
    for line in pf.lines():
        print(line)
    return 1 if pf.failed else 0
