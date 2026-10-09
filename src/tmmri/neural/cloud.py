"""Cloud-processing exception CPE-1 (``protocol/amendments.md``, 2 October 2026) for the F9/S5 Modal backend.

Only the derived ``acdc_dev`` tensor cache may leave the workstation: ``<run>/neural_cache/acdc_dev/manifest.csv`` and
the ``<sample_id>.npz`` files it lists, every row with partition ``acdc_dev``.  :func:`cpe1_upload_set` enumerates the
directory and refuses (:class:`CloudPolicyError`) when any other file is present, when a listed file is missing or its
SHA-256 differs from the manifest, when a path is a link or escapes the directory, or when a row names another
partition.  Every upload and every Volume deletion is appended to ``<run>/cloud/modal_ledger.csv`` (UTC time, action,
Volume, remote path, run-relative path, bytes, SHA-256) before the Modal client is called.  Only the two Volumes named
in ``configs/f9_cnn.yaml`` (``tmmri-neural-cache``, ``tmmri-neural-runs``) are accepted.

:func:`read_backend_config` parses the ``backend:`` block of ``configs/f9_cnn.yaml`` with PyYAML when it is importable
and otherwise with a small block-mapping reader, so ``scripts/cloud/modal_train.py`` runs under
``uvx --from modal==<pin>`` without extra packages.  Standard library only at import time.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

UPLOAD_PARTITION = "acdc_dev"
LEDGER_COLUMNS = ("utc", "action", "volume", "remote_path", "local_path", "bytes", "sha256", "run_id", "client", "note")
NPZ_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.npz$")
LOCAL_SIDECARS = frozenset({"failures.csv", "cache_record.json"})   # written by build-tensor-cache beside the manifest; never uploaded


def cache_file_name(sample_id: str) -> str:
    """C3 file name of a case; the same rule as :func:`tmmri.neural.preprocess.cache_file_name` (``:`` -> ``__``),
    repeated here because this module must import without PyYAML or the CPU stack (Modal client)."""

    return str(sample_id).replace(":", "__").replace("/", "_") + ".npz"


class CloudPolicyError(RuntimeError):
    """A requested upload or Volume operation falls outside CPE-1."""


# ----------------------------------------------------------------------------- backend configuration


def _scalar(text: str) -> Any:
    value = text.strip()
    if value in ("", "null", "~", "Null", "NULL"):
        return None
    if value in ("true", "True"):
        return True
    if value in ("false", "False"):
        return False
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [] if not inner else [_scalar(part) for part in inner.split(",")]
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?(\d+\.\d*|\.\d+|\d+)([eE][-+]?\d+)?", value):
        return float(value)
    return value


def _strip_comment(line: str) -> str:
    quote = None
    for i, ch in enumerate(line):
        if ch in "\"'":
            quote = None if quote == ch else (ch if quote is None else quote)
        elif ch == "#" and quote is None and (i == 0 or line[i - 1] in " \t"):
            return line[:i].rstrip()
    return line.rstrip()


def parse_block_mapping(text: str) -> dict[str, Any]:
    """Nested block mappings with scalar or flow-list values (the subset used by ``backend:``)."""

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        line = _strip_comment(raw)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, rest = line.strip().partition(":")
        if not sep or not key:
            raise ValueError(f"unsupported YAML line in the backend block: {raw!r}")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if rest.strip():
            parent[key.strip()] = _scalar(rest)
        else:
            child: dict[str, Any] = {}
            parent[key.strip()] = child
            stack.append((indent, child))
    return root


def backend_block(text: str) -> str:
    """The ``backend:`` top-level block of a configuration text."""

    lines = text.splitlines()
    out: list[str] = []
    inside = False
    for line in lines:
        if re.match(r"^backend:\s*(#.*)?$", line):
            inside = True
            out.append(line)
            continue
        if inside:
            if line and not line[0].isspace() and not line.startswith("#"):
                break
            out.append(line)
    if not out:
        raise KeyError("configs/f9_cnn.yaml has no backend block")
    return "\n".join(out)


def read_backend_config(path: str | Path) -> dict[str, Any]:
    """``backend`` of ``configs/f9_cnn.yaml`` (PyYAML when importable, else :func:`parse_block_mapping`)."""

    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        return parse_block_mapping(backend_block(text))["backend"]
    return yaml.safe_load(text)["backend"]


def modal_settings(backend: Mapping[str, Any]) -> dict[str, Any]:
    return dict(backend["modal"])


def require_client_pin(modal_cfg: Mapping[str, Any]) -> str:
    pin = modal_cfg.get("client_version")
    if not pin or not re.fullmatch(r"\d+\.\d+(\.\d+)?([a-z0-9.]*)?", str(pin)):
        raise ValueError("configs/f9_cnn.yaml backend.modal.client_version is not pinned: run `uvx modal setup`, then record the "
                         "version printed by `uvx modal --version` (every Modal command refuses until it is pinned)")
    return str(pin)


def allowed_volumes(modal_cfg: Mapping[str, Any]) -> tuple[str, str]:
    return (str(modal_cfg["cache_volume"]), str(modal_cfg["runs_volume"]))


def require_allowed_volume(volume: str, modal_cfg: Mapping[str, Any]) -> str:
    if volume not in allowed_volumes(modal_cfg):
        raise CloudPolicyError(f"CPE-1: Volume {volume!r} is not one of the named Volumes {allowed_volumes(modal_cfg)}")
    return volume


# ----------------------------------------------------------------------------- CPE-1 upload set


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _inside(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
    except ValueError:
        return False
    return True


def cpe1_upload_set(run_path: str | Path, requested: Sequence[str | Path] | None = None) -> list[dict[str, Any]]:
    """Files of ``<run>/neural_cache/acdc_dev`` that CPE-1 permits to upload, verified; refuses anything else.

    ``requested`` (optional) restricts the upload to these paths; each must be one of the permitted files.  Returns
    one dict per file (``path``, ``relpath`` relative to the cache directory, ``bytes``, ``sha256``), manifest first.
    """

    run_path = Path(run_path)
    base = run_path / "neural_cache" / UPLOAD_PARTITION
    manifest = base / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"{manifest} missing: run build-tensor-cache first")
    if base.is_symlink() or not _inside(base, run_path):
        raise CloudPolicyError(f"CPE-1: {base} is a link or lies outside the run root")
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise CloudPolicyError(f"CPE-1: {manifest} lists no file")
    partitions = sorted({r.get("partition", "") for r in rows})
    if partitions != [UPLOAD_PARTITION]:
        raise CloudPolicyError(f"CPE-1: only {UPLOAD_PARTITION} rows may be uploaded; the manifest lists partitions {partitions}")
    expected: dict[str, str] = {}
    for row in rows:
        name = (row.get("npz_file") or "").strip() or cache_file_name(row["sample_id"])   # C3 naming (':' -> '__')
        if not NPZ_NAME.fullmatch(name):
            raise CloudPolicyError(f"CPE-1: unexpected sample id {row['sample_id']!r} in {manifest}")
        expected[name] = str(row["npz_sha256"]).strip().lower()
    present = sorted(p for p in base.rglob("*"))
    links = [p for p in present if p.is_symlink()]
    if links:
        raise CloudPolicyError(f"CPE-1: links are never uploaded: {[str(p) for p in links[:5]]}")
    others = [p.relative_to(base).as_posix() for p in present
              if p.is_dir() or (p.name != "manifest.csv" and p.name not in expected and p.name not in LOCAL_SIDECARS) or p.parent != base]
    if others:
        raise CloudPolicyError(f"CPE-1: {base} holds files outside the permitted set (manifest.csv and the listed .npz): {others[:10]}")
    missing = sorted(set(expected) - {p.name for p in present})
    if missing:
        raise CloudPolicyError(f"CPE-1: the manifest lists files that are missing: {missing[:5]}")
    files = [manifest] + [base / name for name in sorted(expected)]
    out: list[dict[str, Any]] = []
    for path in files:
        digest = _sha256(path)
        if path.name in expected and digest != expected[path.name]:
            raise CloudPolicyError(f"CPE-1: {path.name} SHA-256 {digest} differs from the manifest ({expected[path.name]})")
        out.append({"path": path, "relpath": path.relative_to(base).as_posix(), "bytes": path.stat().st_size, "sha256": digest})
    if requested:
        permitted = {item["path"].resolve(): item for item in out}
        chosen = []
        for value in requested:
            candidate = Path(value)
            candidate = candidate if candidate.is_absolute() else Path.cwd() / candidate
            item = permitted.get(candidate.resolve())
            if item is None:
                raise CloudPolicyError(f"CPE-1: {value} is not permitted (only {base}/manifest.csv and the .npz files it lists)")
            chosen.append(item)
        out = chosen
    return out


# ----------------------------------------------------------------------------- ledger


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def ledger_path(run_path: str | Path, modal_cfg: Mapping[str, Any] | None = None) -> Path:
    rel = (modal_cfg or {}).get("ledger", "cloud/modal_ledger.csv")
    return Path(run_path) / str(rel)


def append_ledger(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Append rows (LF line endings); writes the header once."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file()
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(LEDGER_COLUMNS), lineterminator="\n", extrasaction="raise")
    if new:
        writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in LEDGER_COLUMNS})
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())
        handle.flush()
        os.fsync(handle.fileno())
    return path


def modal_command(pin: str, *args: str) -> list[str]:
    """``uvx --from modal==<pin> modal <args>`` (the pinned client)."""

    return ["uvx", "--from", f"modal=={pin}", "modal", *args]


def volume_put_commands(pin: str, volume: str, files: Sequence[Mapping[str, Any]], remote_dir: str) -> list[list[str]]:
    return [modal_command(pin, "volume", "put", volume, str(item["path"]), f"{remote_dir.rstrip('/')}/{item['relpath']}") for item in files]


def volume_delete_commands(pin: str, modal_cfg: Mapping[str, Any]) -> list[list[str]]:
    return [modal_command(pin, "volume", "delete", volume) for volume in allowed_volumes(modal_cfg)]


Runner = Callable[[list[str]], Any]
