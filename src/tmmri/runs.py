"""Run roots, content-addressed stage records and parallel helpers (protocol CMR-TRANS-AUDIT 6.0.0, Decision D23).

A run root is ``<ARTIFACT_ROOT>/<run_id>/`` (Docker volume ``tmmri_artifacts`` in the container) with the
subfolders of ``configs/study_v6.yaml`` ``artifact_subfolders``, ``artifact_status.json`` and a ``.stages/`` folder.
Every stage writes ``.stages/<stage>.json``: a hashed *payload* (input and output SHA-256, config hash, code
version, library versions, ``n_jobs``, parameters) and a separate, unhashed ``run_log`` (timestamps, duration,
host).  :func:`run_stage` skips a stage when the inputs, the config hash, the code tree hash and the parameters
match the previous record and every recorded output still has its recorded hash; otherwise it reruns the stage,
unless its ``rerun_guard`` refuses (the once-only guard of a frozen run, :func:`tmmri.stages.once_only_guard`).

Paths inside records are portable: ``run:<rel>`` (inside the run root), ``data:<rel>`` (``DATA_ROOT``) or
``repo:<rel>`` (``PROJECT_ROOT``), so a record written on the host and one written in the container agree.
Parallelism: per-case work uses joblib/loky with one BLAS thread per worker; results are gathered in input order,
so they never depend on the worker count.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .paths import ARTIFACT_ROOT, DATA_ROOT, PROJECT_ROOT, validate_run_id

DEFAULT_SUBFOLDERS = (
    "contracts", "qc", "semantics", "formal", "transitions", "oof", "reconstructed", "thresholds", "granules",
    "rules", "inference", "equivariance", "metrics", "figures", "manuscript_tables", ".stages",
)
STAGE_RECORD_SCHEMA = "tmmri.stage_record/1"
DEFAULT_N_JOBS = 12          # configs/runtime.env TMMRI_N_JOBS; results are independent of the value
DEFAULT_SOURCE_READERS = 4   # concurrent source readers (Docker Desktop bind throughput, D: HDD)
LIBRARIES = ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "scikit-image", "nibabel", "joblib", "PyYAML")


# ----------------------------------------------------------------------------- parallel helpers


def _positive_int(value: object) -> int | None:
    try:
        out = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def resolve_n_jobs(explicit: int | None = None) -> int:
    """CPU workers: explicit value, then ``TMMRI_N_JOBS``, then min(12, CPU count)."""

    return _positive_int(explicit) or _positive_int(os.environ.get("TMMRI_N_JOBS")) or max(1, min(DEFAULT_N_JOBS, os.cpu_count() or 1))


def resolve_source_readers(explicit: int | None = None) -> int:
    """Concurrent source readers (independent of ``n_jobs``): explicit value, ``TMMRI_SOURCE_READERS``, then 4."""

    return _positive_int(explicit) or _positive_int(os.environ.get("TMMRI_SOURCE_READERS")) or DEFAULT_SOURCE_READERS


def parallel_map(fn: Callable[[Any], Any], items: Iterable[Any], n_jobs: int = 1) -> list[Any]:
    """Map ``fn`` over ``items`` with joblib/loky (one BLAS thread per worker); results keep the input order."""

    items = list(items)
    if n_jobs <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    from joblib import Parallel, delayed, parallel_config

    with parallel_config(backend="loky", inner_max_num_threads=1):
        return list(Parallel(n_jobs=min(n_jobs, len(items)), batch_size=1)(delayed(fn)(item) for item in items))


def peak_rss_bytes() -> int | None:
    """Peak resident set size of the current process (Linux ``ru_maxrss``; Windows ``PeakWorkingSetSize``)."""

    try:
        import resource  # POSIX only

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value) if sys.platform == "darwin" else int(value) * 1024
    except ImportError:
        pass
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class _Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            counters = _Counters()
            counters.cb = ctypes.sizeof(_Counters)
            kernel32 = ctypes.WinDLL("kernel32")
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.WinDLL("psapi")
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]
            if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return int(counters.PeakWorkingSetSize)
        except (OSError, AttributeError):
            return None
    return None


# ----------------------------------------------------------------------------- hashing and atomic writes


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False, default=str).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _is_temporary(path: Path) -> bool:
    return path.name.startswith(".") and ".tmp" in path.name


def hash_path(path: str | Path) -> str | None:
    """SHA-256 of a file; of a directory the hash of its sorted ``relpath<TAB>sha256`` lines; ``None`` if missing."""

    p = Path(path)
    if p.is_file():
        return sha256_file(p)
    if p.is_dir():
        lines = [f"{f.relative_to(p).as_posix()}\t{sha256_file(f)}" for f in sorted(p.rglob("*"))
                 if f.is_file() and not _is_temporary(f)]
        return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    return None


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    """Write ``data`` to a temporary sibling, then rename it over ``path`` (never a partial file)."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)
    return target


def atomic_write_json(path: str | Path, value: Any, *, allow_nan: bool = False) -> Path:
    """Sorted, indented JSON; ``allow_nan`` only for fitted-model payloads that carry NaN (read back by Python)."""

    text = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=allow_nan, default=str) + "\n"
    return atomic_write_bytes(path, text.encode("utf-8"))


def portable_path(path: str | Path, run_path: Path | None = None) -> str:
    """``run:``/``data:``/``repo:`` prefixed relative path (host/container independent), else absolute POSIX."""

    p = Path(path).resolve()
    for prefix, root in (("run", run_path), ("data", DATA_ROOT), ("repo", PROJECT_ROOT)):
        if root is None:
            continue
        try:
            return f"{prefix}:{p.relative_to(Path(root).resolve()).as_posix()}"
        except ValueError:
            continue
    return p.as_posix()


# ----------------------------------------------------------------------------- code and library versions


def _git(args: Sequence[str], root: Path) -> str | None:
    """``git --no-optional-locks <args>``: read-only, never refreshes or rewrites ``.git/index`` (container bind mounts)."""

    try:
        proc = subprocess.run(["git", "--no-optional-locks", *args], cwd=root, capture_output=True, text=True, check=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout


def code_tree_sha256(root: Path = PROJECT_ROOT) -> str:
    """Hash of every ``src/tmmri/**/*.py`` (line endings normalized, so host and container agree)."""

    base = Path(root) / "src" / "tmmri"
    lines = []
    for path in sorted(base.rglob("*.py")):
        content = path.read_bytes().replace(b"\r\n", b"\n")
        lines.append(f"{path.relative_to(base).as_posix()}\t{hashlib.sha256(content).hexdigest()}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


@functools.lru_cache(maxsize=None)
def _code_version_cached(root: str) -> tuple[str | None, bool | None, str]:
    commit = _git(["rev-parse", "HEAD"], Path(root))
    status = _git(["status", "--porcelain", "--untracked-files=no"], Path(root))
    return (commit.strip() if commit else None, None if status is None else bool(status.strip()), code_tree_sha256(Path(root)))


def code_version(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Commit, dirty flag and source-tree hash, computed once per process and root (``git status`` costs seconds
    on bind mounts; a ``run`` pays it once).  Call ``code_version.cache_clear()`` after editing code in-process."""

    commit, dirty, code_sha = _code_version_cached(str(Path(root).resolve()))
    return {"commit": commit, "dirty": dirty, "code_sha256": code_sha}


code_version.cache_clear = _code_version_cached.cache_clear   # type: ignore[attr-defined]


def library_versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {"python": platform.python_version()}
    for name in LIBRARIES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ----------------------------------------------------------------------------- run roots and stage records


@dataclass
class StageRecord:
    """Content-addressed record of one stage execution (the payload is hashed; ``run_log`` is not)."""

    stage: str
    inputs: dict[str, str | None]
    outputs: dict[str, str | None]
    config_hash: str
    code: dict[str, Any]
    library_versions: dict[str, str | None]
    n_jobs: int
    params: dict[str, Any] = field(default_factory=dict)
    status: str = "BUILT"
    run_log: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return {"stage": self.stage, "inputs": self.inputs, "outputs": self.outputs, "config_hash": self.config_hash,
                "code": self.code, "library_versions": self.library_versions, "n_jobs": self.n_jobs,
                "params": self.params, "status": self.status}

    def skip_key(self) -> str:
        """What must match for a skip: stage, inputs, config hash, code tree hash and parameters."""

        return sha256_json({"stage": self.stage, "inputs": self.inputs, "config_hash": self.config_hash,
                            "code_sha256": self.code.get("code_sha256"), "params": self.params})

    def to_dict(self) -> dict[str, Any]:
        payload = self.payload()
        return {"schema": STAGE_RECORD_SCHEMA, "payload": payload, "payload_sha256": sha256_json(payload),
                "run_log": self.run_log}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StageRecord":
        p = data["payload"]
        return cls(stage=p["stage"], inputs=dict(p["inputs"]), outputs=dict(p["outputs"]), config_hash=p["config_hash"],
                   code=dict(p["code"]), library_versions=dict(p["library_versions"]), n_jobs=int(p["n_jobs"]),
                   params=dict(p.get("params", {})), status=p.get("status", "BUILT"), run_log=dict(data.get("run_log", {})))


class RunRoot:
    """``<artifact_root>/<run_id>/`` with the protocol subfolders, ``artifact_status.json`` and ``.stages/``."""

    def __init__(self, run_id: str, artifact_root: str | Path | None = None,
                 subfolders: Sequence[str] | None = None, create: bool = True) -> None:
        self.run_id = validate_run_id(run_id)
        self.artifact_root = Path(artifact_root) if artifact_root is not None else ARTIFACT_ROOT
        self.path = self.artifact_root / self.run_id
        self.subfolders = tuple(subfolders) if subfolders else DEFAULT_SUBFOLDERS
        if create:
            for name in self.subfolders:
                (self.path / name).mkdir(parents=True, exist_ok=True)
            if not self.status_path.is_file():
                atomic_write_json(self.status_path, {"run_id": self.run_id, "stages": {}})

    @property
    def stages_dir(self) -> Path:
        return self.path / ".stages"

    @property
    def status_path(self) -> Path:
        return self.path / "artifact_status.json"

    @property
    def frozen_path(self) -> Path:
        return self.stages_dir / "FROZEN.json"

    def sub(self, name: str) -> Path:
        path = self.path / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def record_path(self, stage: str) -> Path:
        return self.stages_dir / f"{stage.replace(':', '__')}.json"

    def read_record(self, stage: str) -> StageRecord | None:
        path = self.record_path(stage)
        if not path.is_file():
            return None
        try:
            return StageRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (KeyError, ValueError, TypeError):
            return None

    def status(self) -> dict[str, Any]:
        try:
            return json.loads(self.status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"run_id": self.run_id, "stages": {}}

    def set_status(self, stage: str, status: str, **details: Any) -> None:
        current = self.status()
        current.setdefault("stages", {})[stage] = {"status": status, **details}
        atomic_write_json(self.status_path, current)

    def set_field(self, key: str, value: Any) -> None:
        """Set a top-level field of ``artifact_status.json`` (for example the synthetic-data declaration)."""

        if key in {"run_id", "stages"}:
            raise ValueError(f"artifact_status.json field {key!r} is reserved")
        current = self.status()
        current[key] = value
        atomic_write_json(self.status_path, current)

    def log(self, event: Mapping[str, Any]) -> None:
        self.stages_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"utc": _utc_now(), **event}, sort_keys=True, default=str)
        with (self.stages_dir / "run_log.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")

    def portable(self, path: str | Path) -> str:
        return portable_path(path, self.path)


@dataclass
class StageResult:
    stage: str
    action: str                     # RAN | SKIPPED
    record: StageRecord
    value: Any = None


def record_difference(prior: StageRecord, candidate: StageRecord, *, force: bool = False, outputs_verify: bool = False) -> str:
    """What prevents reusing ``prior`` for ``candidate`` (a comma-separated list; never empty)."""

    parts = [label for label, differs in (
        ("--force", force),
        ("inputs", prior.inputs != candidate.inputs),
        ("config hash", prior.config_hash != candidate.config_hash),
        ("code version", prior.code.get("code_sha256") != candidate.code.get("code_sha256")),
        ("parameters", prior.params != candidate.params),
        ("status", prior.status != candidate.status),
    ) if differs]
    if not parts and not outputs_verify:
        parts.append("outputs (a recorded output is missing or changed)")
    return ", ".join(parts) or "record"


def run_stage(
    run: RunRoot,
    name: str,
    fn: Callable[[], Any],
    inputs: Iterable[str | Path],
    outputs: Iterable[str | Path],
    *,
    config_hash: str,
    params: Mapping[str, Any] | None = None,
    force: bool = False,
    n_jobs: int = 1,
    code: Mapping[str, Any] | None = None,
    status: str = "BUILT",
    rerun_guard: Callable[[str], None] | None = None,
) -> StageResult:
    """Run ``fn`` unless an identical stage record exists (see the module docstring); record the result.

    ``rerun_guard`` is called, before ``fn``, when a record of the stage exists but cannot be reused; it receives a
    short description of what differs (``inputs``, ``config hash``, ``code version``, ``parameters``, ``status``,
    ``outputs`` or ``--force``) and may raise to refuse the re-execution (the once-only guard of a frozen run).
    """

    input_paths = [Path(p) for p in inputs]
    missing = [str(p) for p in input_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"stage {name}: missing inputs {missing}")
    output_paths = [Path(p) for p in outputs]
    in_hashes = {run.portable(p): hash_path(p) for p in input_paths}
    code_info = dict(code) if code is not None else code_version()
    params_clean = json.loads(canonical_json(dict(params or {})))
    candidate = StageRecord(stage=name, inputs=in_hashes, outputs={}, config_hash=config_hash, code=code_info,
                            library_versions=library_versions(), n_jobs=int(n_jobs), params=params_clean, status=status)
    prior = run.read_record(name)
    outputs_verify = False
    if not force and prior is not None and prior.skip_key() == candidate.skip_key() and prior.status == status:
        current = {run.portable(p): hash_path(p) for p in output_paths}
        outputs_verify = current == prior.outputs and all(v is not None for v in current.values())
        if outputs_verify:
            run.log({"stage": name, "action": "SKIPPED"})
            run.set_status(name, prior.status, action="SKIPPED", payload_sha256=sha256_json(prior.payload()))
            return StageResult(name, "SKIPPED", prior)
    if prior is not None and rerun_guard is not None:
        rerun_guard(record_difference(prior, candidate, force=force, outputs_verify=outputs_verify))
    started = time.perf_counter()
    started_utc = _utc_now()
    value = fn()
    seconds = time.perf_counter() - started
    out_hashes = {run.portable(p): hash_path(p) for p in output_paths}
    absent = [k for k, v in out_hashes.items() if v is None]
    if absent:
        raise RuntimeError(f"stage {name} did not produce its declared outputs: {absent}")
    candidate.outputs = out_hashes
    candidate.run_log = {"started_utc": started_utc, "seconds": round(seconds, 3), "host": socket.gethostname(),
                         "platform": platform.platform()}
    atomic_write_json(run.record_path(name), candidate.to_dict())
    run.log({"stage": name, "action": "RAN", "seconds": round(seconds, 3)})
    run.set_status(name, status, action="RAN", payload_sha256=sha256_json(candidate.payload()))
    return StageResult(name, "RAN", candidate, value)
