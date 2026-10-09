"""IEEE FP32 deterministic numerics and the environment record of every torch computation (workflow contract C9).

:func:`configure_torch_numerics` must run before CUDA initializes: it sets ``CUBLAS_WORKSPACE_CONFIG=:4096:8``,
turns TF32 off for cuBLAS and cuDNN (the legacy ``allow_tf32`` flags and, where the API exists, every
``fp32_precision`` to ``"ieee"``), enables ``torch.use_deterministic_algorithms(True)`` and disables the cuDNN
benchmark (docs/EXECUTION_ENVIRONMENT_SPEC.md 5.2 rules 5, 6 and 10).  Execution stays in eager mode.
:func:`environment_record` returns the device, GPU name/UUID/capability, driver, torch/torchvision/CUDA/cuDNN versions
and the TF32 state, for every fit, extraction and inference record.
"""

from __future__ import annotations

import os
import platform
import subprocess
from typing import Any

CUBLAS_WORKSPACE = ":4096:8"
FP32_PRECISION_PATHS = ("backends", "backends.cuda.matmul", "backends.cudnn", "backends.cudnn.conv", "backends.cudnn.rnn")


class NumericsError(RuntimeError):
    """The required FP32/determinism state cannot be established (for example CUDA initialized too early)."""


def _resolve(torch: Any, dotted: str) -> Any:
    obj = torch
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def configure_torch_numerics() -> dict[str, Any]:
    """Apply the C9 numerics (idempotent) and return :func:`numerics_state`."""

    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    import torch

    if current != CUBLAS_WORKSPACE:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            raise NumericsError(f"CUDA was initialized before CUBLAS_WORKSPACE_CONFIG={CUBLAS_WORKSPACE} could be set (found {current!r})")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for dotted in FP32_PRECISION_PATHS:
        try:
            target = _resolve(torch, dotted)
        except AttributeError:
            continue
        if hasattr(target, "fp32_precision"):
            target.fp32_precision = "ieee"
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    state = numerics_state()
    if state["tf32_enabled"]:
        raise NumericsError(f"TF32 is still enabled: {state}")
    return state


def numerics_state() -> dict[str, Any]:
    import torch

    precision: dict[str, Any] = {}
    for dotted in FP32_PRECISION_PATHS:
        try:
            precision[dotted] = _resolve(torch, dotted).fp32_precision
        except AttributeError:
            continue
    legacy = {"cuda.matmul.allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
              "cudnn.allow_tf32": bool(torch.backends.cudnn.allow_tf32)}
    tf32 = any(legacy.values()) or any(str(v) == "tf32" for v in precision.values())
    return {
        "precision": "fp32_ieee", "tf32_enabled": bool(tf32), "allow_tf32": legacy, "fp32_precision": precision,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic), "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "eager_mode": True,
    }


def resolve_device(requested: str | None = None) -> Any:
    """``cuda`` when available (or explicitly requested), else ``cpu``; an unavailable explicit request fails."""

    import torch

    if requested in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise NumericsError(f"device {requested!r} requested but CUDA is not available")
    return device


def _nvidia_smi() -> dict[str, Any]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    rows = [[p.strip() for p in line.split(",")] for line in out.strip().splitlines() if line.strip()]
    return {"gpus": [{"name": r[0], "uuid": r[1], "driver_version": r[2]} for r in rows if len(r) >= 3]}


def environment_record(device: Any) -> dict[str, Any]:
    """Device, GPU, driver, library versions and the numerics state (no timestamps)."""

    import torch

    try:
        import torchvision

        torchvision_version = torchvision.__version__
    except ImportError:
        torchvision_version = None
    record: dict[str, Any] = {
        "device": str(device), "torch": torch.__version__, "torchvision": torchvision_version,
        "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "python": platform.python_version(), "platform": platform.platform(), "numerics": numerics_state(),
        "gpu": None, "driver_version": None,
    }
    if getattr(device, "type", str(device)) == "cuda":
        index = device.index if getattr(device, "index", None) is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        smi = _nvidia_smi().get("gpus", [])
        uuid = str(getattr(props, "uuid", "")) or None
        match = next((g for g in smi if uuid and uuid in g["uuid"]), smi[index] if index < len(smi) else {})
        record["gpu"] = {"index": int(index), "name": props.name, "uuid": uuid or match.get("uuid"),
                         "capability": f"{props.major}.{props.minor}", "total_memory_bytes": int(props.total_memory)}
        record["driver_version"] = match.get("driver_version")
    return record
