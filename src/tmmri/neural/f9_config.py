"""Loading ``configs/f9_cnn.yaml`` and resolving the pinned ResNet-18 weights (``registry/weights.yaml``, by name).

The training configuration passed to :func:`tmmri.neural.train.train_fit` and embedded in spec files is the parsed file
without its ``backend`` block (:func:`training_config`); its SHA-256 is recorded in every spec file and fit record.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..paths import PROJECT_ROOT
from ..runs import sha256_json

CONFIG_REL = Path("configs") / "f9_cnn.yaml"


def config_path(root: str | Path | None = None) -> Path:
    return Path(root or PROJECT_ROOT) / CONFIG_REL


def load_f9_config(root: str | Path | None = None, *, path: str | Path | None = None) -> dict[str, Any]:
    import yaml

    file = Path(path) if path is not None else config_path(root)
    with file.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if cfg.get("model_id") != "f9":
        raise ValueError(f"{file} is not the F9 configuration")
    return cfg


def training_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Everything except ``backend`` (backend settings never change a fit's result)."""

    return {k: v for k, v in cfg.items() if k != "backend"}


def training_config_sha256(cfg: Mapping[str, Any]) -> str:
    return sha256_json(training_config(cfg))


def _find_entry(data: Any, name: str) -> dict[str, Any] | None:
    if isinstance(data, Mapping):
        if isinstance(data.get(name), Mapping):
            return dict(data[name])
        for key in ("weights", "entries", "models", "artifacts"):
            found = _find_entry(data.get(key), name)
            if found is not None:
                return found
    if isinstance(data, list):
        for item in data:
            if isinstance(item, Mapping) and name in (item.get("name"), item.get("id")):
                return dict(item)
    return None


def torch_hub_checkpoints() -> Path:
    home = os.environ.get("TORCH_HOME") or str(Path.home() / ".cache" / "torch")
    return Path(home) / "hub" / "checkpoints"


def resolve_weights(cfg: Mapping[str, Any], root: str | Path | None = None) -> dict[str, Any]:
    """``{name, sha256, file, url, path}`` of the backbone weights named in ``configs/f9_cnn.yaml`` ``weights``.

    The SHA-256 comes from ``registry/weights.yaml`` (written by ``verify-torch-weights``); a missing entry or SHA-256
    raises, because NEURAL-TM-1.0 4.1 requires the pin before training."""

    import yaml

    wcfg = cfg["weights"]
    registry = Path(root or PROJECT_ROOT) / str(wcfg["registry"])
    if not registry.is_file():
        raise FileNotFoundError(f"{registry} missing: run verify-torch-weights to pin {wcfg['name']} before planning F9 fits")
    entry = _find_entry(yaml.safe_load(registry.read_text(encoding="utf-8")), str(wcfg["name"]))
    if entry is None:
        raise KeyError(f"{registry} has no entry {wcfg['name']!r}")
    sha = entry.get("sha256") or entry.get("file_sha256") or entry.get("weights_sha256")
    if not sha or len(str(sha)) != 64:
        raise ValueError(f"{registry} entry {wcfg['name']!r} has no pinned SHA-256")
    file = str(entry.get("file") or entry.get("filename") or wcfg["file"])
    if file != str(wcfg["file"]):
        raise ValueError(f"{registry} entry {wcfg['name']!r} names file {file}, configs/f9_cnn.yaml expects {wcfg['file']}")
    return {"name": str(wcfg["name"]), "sha256": str(sha).lower(), "file": file, "url": entry.get("url"),
            "path": str(torch_hub_checkpoints() / file)}
