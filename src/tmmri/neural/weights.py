"""Pinned torchvision ImageNet weights (``verify-torch-weights``; owner-approved download, protocol-v6.0.1).

Two files are used: ``resnet18-f37072fd.pth`` (``ResNet18_Weights.IMAGENET1K_V1``, the F9 backbone of NEURAL-TM-1.0
4.1) and ``resnet50-11ad3fa6.pth`` (``ResNet50_Weights.IMAGENET1K_V2``, the frozen D20 encoder of SPECIFICATION.md
8.4).  :func:`verify_torch_weights` downloads a missing file from ``download.pytorch.org`` only (no other host, no
redirect off it) into ``$TORCH_HOME/hub/checkpoints`` (``/cache/torch`` in the container, volume ``tmmri_cache``),
checks the SHA-256 prefix that torchvision embeds in each file name, and records the full SHA-256 and size in
``registry/weights.yaml`` and, for the D20 encoder, in ``configs/deep_encoder.yaml`` ``encoder.weights_sha256``.
Once recorded, a hash is never rewritten: a file that differs from it fails the check.  :func:`require_weights`
is the only way the encoders open a weight file (verified against the registry; never downloaded at use).
"""

from __future__ import annotations

import hashlib
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping

ALLOWED_HOST = "download.pytorch.org"
WEIGHTS_REGISTRY = "registry/weights.yaml"
DEEP_ENCODER_CONFIG = "configs/deep_encoder.yaml"
PENDING = "pending_first_use"
WEIGHTS: dict[str, dict[str, str]] = {
    "resnet18_imagenet1k_v1": {
        "torchvision_weights": "ResNet18_Weights.IMAGENET1K_V1", "architecture": "resnet18", "file": "resnet18-f37072fd.pth",
        "url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        "used_by": "F9 trained CNN backbone (docs/NEURAL_ENCODER_EXPERIMENT_PROTOCOL.md 4.1; configs/f9_cnn.yaml)",
    },
    "resnet50_imagenet1k_v2": {
        "torchvision_weights": "ResNet50_Weights.IMAGENET1K_V2", "architecture": "resnet50", "file": "resnet50-11ad3fa6.pth",
        "url": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
        "used_by": "D20 frozen deep encoder (SPECIFICATION.md 8.4; configs/deep_encoder.yaml)",
    },
}
D20_WEIGHTS = "resnet50_imagenet1k_v2"
F9_WEIGHTS = "resnet18_imagenet1k_v1"
_HEADER = ("# Pinned torchvision ImageNet weight files (protocol-v6.0.1; owner-approved download from download.pytorch.org only).\n"
           "# Written by `python -m tmmri.cli verify-torch-weights` (src/tmmri/neural/weights.py); a recorded sha256 is never\n"
           "# rewritten, and every encoder verifies its file against it before use.\n")


class WeightsError(ValueError):
    """A weight file is missing, has the wrong hash, or would come from a host other than download.pytorch.org."""


def torch_home() -> Path:
    if os.environ.get("TORCH_HOME"):
        return Path(os.environ["TORCH_HOME"])
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "torch"


def checkpoint_path(key: str, home: str | Path | None = None) -> Path:
    return Path(home if home is not None else torch_home()) / "hub" / "checkpoints" / WEIGHTS[key]["file"]


def file_prefix(key: str) -> str:
    match = re.search(r"-([0-9a-f]{8,})\.pth$", WEIGHTS[key]["file"])
    if not match:
        raise WeightsError(f"{WEIGHTS[key]['file']} carries no SHA-256 prefix")
    return match.group(1)


def sha256_of(path: str | Path) -> tuple[str, int]:
    hasher, size = hashlib.sha256(), 0
    with Path(path).open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            hasher.update(chunk)
            size += len(chunk)
    return hasher.hexdigest(), size


def _check_host(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != ALLOWED_HOST:
        raise WeightsError(f"refusing to download from {url!r}: only https://{ALLOWED_HOST} is approved")


def download(url: str, dest: Path, *, opener: Any = None) -> Path:
    """Stream ``url`` (download.pytorch.org only) to ``dest`` atomically."""

    _check_host(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.part")
    open_url = opener or urllib.request.urlopen
    with open_url(url, timeout=120) as response:
        _check_host(response.geturl())
        with tmp.open("wb") as handle:
            while chunk := response.read(4 * 1024 * 1024):
                handle.write(chunk)
    os.replace(tmp, dest)
    return dest


def load_registry(root: str | Path) -> dict[str, Any]:
    import yaml

    path = Path(root) / WEIGHTS_REGISTRY
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def recorded_sha256(key: str, root: str | Path) -> str | None:
    entry = (load_registry(root).get("weights") or {}).get(key) or {}
    value = entry.get("sha256")
    return str(value) if value and value != PENDING else None


def _write_registry(root: Path, entries: Mapping[str, Mapping[str, Any]]) -> Path:
    import yaml

    from ..runs import atomic_write_bytes

    payload = {"schema": "tmmri.weights_registry/1", "amendment": "protocol-v6.0.1", "source_host": ALLOWED_HOST,
               "cache": "$TORCH_HOME/hub/checkpoints (container: /cache/torch, volume tmmri_cache)",
               "weights": {k: dict(entries[k]) for k in sorted(entries)}}
    text = _HEADER + yaml.safe_dump(payload, sort_keys=False, default_flow_style=False, width=200)
    return atomic_write_bytes(Path(root) / WEIGHTS_REGISTRY, text.encode("utf-8"))


def _record_deep_encoder_hash(root: Path, sha: str) -> str:
    """Set ``encoder.weights_sha256`` of configs/deep_encoder.yaml when pending (line edit; comments kept)."""

    from ..runs import atomic_write_bytes

    path = Path(root) / DEEP_ENCODER_CONFIG
    raw = path.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in raw else "\n"
    pattern = re.compile(r"^(?P<indent>[ \t]+)weights_sha256:[ \t]*(?P<value>[^\s#]+)(?P<rest>[^\r\n]*)(?=\r?\n|\Z)", re.MULTILINE)
    match = pattern.search(raw)
    if match is None:
        raise WeightsError(f"{DEEP_ENCODER_CONFIG} has no encoder.weights_sha256 line")
    value = match.group("value").strip("\"'")
    if value == sha:
        return "unchanged"
    if value != PENDING:
        raise WeightsError(f"{DEEP_ENCODER_CONFIG} records weights_sha256 {value}, the file hashes to {sha}")
    line = f"{match.group('indent')}weights_sha256: \"{sha}\"   # resnet50-11ad3fa6.pth; registry/weights.yaml (verify-torch-weights)"
    atomic_write_bytes(path, (raw[:match.start()] + line + raw[match.end():]).replace("\r\n", "\n").replace("\n", newline).encode("utf-8"))
    return "recorded"


def verify_torch_weights(root: str | Path, *, home: str | Path | None = None, allow_download: bool = True,
                         record: bool = True, opener: Any = None) -> dict[str, Any]:
    """Download (if absent and allowed), verify and record both weight files; returns a report (raises
    :class:`WeightsError` on a missing file, a prefix mismatch or a hash that differs from the record)."""

    base = Path(root)
    registry = (load_registry(base).get("weights") or {})
    entries: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {"torch_home": str(Path(home) if home is not None else torch_home()), "weights": {}}
    for key, spec in WEIGHTS.items():
        path = checkpoint_path(key, home)
        action = "present"
        if not path.is_file():
            if not allow_download:
                raise WeightsError(f"{path} missing (run verify-torch-weights without --no-download)")
            download(spec["url"], path, opener=opener)
            action = "downloaded"
        sha, size = sha256_of(path)
        prefix = file_prefix(key)
        if not sha.startswith(prefix):
            raise WeightsError(f"{spec['file']}: SHA-256 {sha} does not start with the file-name prefix {prefix}")
        known = (registry.get(key) or {}).get("sha256")
        if known and known != PENDING and known != sha:
            raise WeightsError(f"{spec['file']}: SHA-256 {sha} differs from registry/weights.yaml ({known})")
        entries[key] = {**spec, "sha256_prefix": prefix, "sha256": sha, "bytes": size}
        report["weights"][key] = {"file": str(path), "action": action, "sha256": sha, "bytes": size, "prefix_ok": True,
                                  "registry": "matches" if known == sha else "pending"}
    if record:
        _write_registry(base, entries)
        report["deep_encoder_config"] = _record_deep_encoder_hash(base, entries[D20_WEIGHTS]["sha256"])
        report["registry"] = str(base / WEIGHTS_REGISTRY)
    return report


def require_weights(key: str, root: str | Path, home: str | Path | None = None) -> tuple[Path, str]:
    """(path, sha256) of a recorded weight file after verifying it against registry/weights.yaml."""

    expected = recorded_sha256(key, root)
    if expected is None:
        raise WeightsError(f"{key}: no SHA-256 in {WEIGHTS_REGISTRY}; run verify-torch-weights first")
    path = checkpoint_path(key, home)
    if not path.is_file():
        raise WeightsError(f"{path} missing; run verify-torch-weights first")
    sha, _ = sha256_of(path)
    if sha != expected:
        raise WeightsError(f"{path.name}: SHA-256 {sha} differs from {WEIGHTS_REGISTRY} ({expected})")
    return path, sha
