"""IEEE FP32 and deterministic numerics for all F9 torch code (contract C9; EXECUTION_ENVIRONMENT_SPEC.md 5.2 rules 5,
6 and 10; Modal assessment Section 4 C2).

:func:`configure_numerics` sets ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` (refusing when CUDA was initialized with another
value), turns TF32 off through both the legacy flags (``torch.backends.cuda.matmul.allow_tf32``,
``torch.backends.cudnn.allow_tf32``) and, where the API exists (torch 2.14), ``fp32_precision = "ieee"``, enables
``torch.use_deterministic_algorithms(True)`` and disables ``cudnn.benchmark``.  It asserts the resulting state, which
:func:`environment_record` stores in every fit record together with the device, GPU name and UUID, driver, torch, CUDA
and cuDNN versions.  Execution is eager (``torch.compile`` is never used).
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
from typing import Any

CUBLAS_WORKSPACE_CONFIG = ":4096:8"
FP32_PRECISION_ATTRS = ("backends", "backends.cuda.matmul", "backends.cudnn", "backends.cudnn.conv", "backends.cudnn.rnn")


def _resolve(root: Any, dotted: str) -> Any:
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def configure_numerics(device: Any = "cpu", *, threads: int | None = None) -> dict[str, Any]:
    """Apply C9 before any CUDA work; return :func:`numerics_state` (raises when TF32 would remain enabled)."""

    import torch

    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if current != CUBLAS_WORKSPACE_CONFIG:
        if torch.cuda.is_initialized():
            raise RuntimeError(f"CUDA was initialized with CUBLAS_WORKSPACE_CONFIG={current!r}; set {CUBLAS_WORKSPACE_CONFIG} "
                               "before any CUDA call")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for dotted in FP32_PRECISION_ATTRS:
        obj = _resolve(torch, dotted)
        if obj is not None and hasattr(obj, "fp32_precision"):
            obj.fp32_precision = "ieee"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if threads:
        torch.set_num_threads(int(threads))
    state = numerics_state()
    assert_ieee_fp32(state)
    return state


def numerics_state() -> dict[str, Any]:
    import torch

    state: dict[str, Any] = {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "execution": "eager",
    }
    for dotted in FP32_PRECISION_ATTRS:
        obj = _resolve(torch, dotted)
        if obj is not None and hasattr(obj, "fp32_precision"):
            state[f"fp32_precision.{dotted.removeprefix('backends').lstrip('.') or 'global'}"] = str(obj.fp32_precision)
    state["tf32_enabled"] = bool(state["cuda_matmul_allow_tf32"] or state["cudnn_allow_tf32"]
                                 or any(v == "tf32" for k, v in state.items() if k.startswith("fp32_precision.")))
    return state


def assert_ieee_fp32(state: dict[str, Any] | None = None) -> None:
    state = state or numerics_state()
    problems = [k for k in ("cuda_matmul_allow_tf32", "cudnn_allow_tf32", "cudnn_benchmark") if state.get(k)]
    problems += [k for k, v in state.items() if k.startswith("fp32_precision.") and v != "ieee"]
    if state.get("tf32_enabled") or not state.get("deterministic_algorithms") or state.get("cublas_workspace_config") != CUBLAS_WORKSPACE_CONFIG:
        problems.append("tf32/deterministic/cublas")
    if problems:
        raise RuntimeError(f"C9 numerics violated (IEEE FP32, deterministic, eager): {sorted(set(problems))} in {state}")


def _nvidia_smi(index: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(["nvidia-smi", f"--id={index}", "--query-gpu=name,uuid,driver_version,memory.total",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        return {}
    parts = [p.strip() for p in proc.stdout.strip().split(",")]
    return dict(zip(("smi_name", "smi_uuid", "driver_version", "memory_total"), parts)) if len(parts) == 4 else {}


def environment_record(device: Any = "cpu") -> dict[str, Any]:
    """Device, GPU name/UUID, driver, torch/CUDA/cuDNN versions, TF32 state and host facts for a fit record."""

    import torch

    dev = torch.device(device)
    record: dict[str, Any] = {
        "device": str(dev),
        "torch": torch.__version__,
        "torchvision": _torchvision_version(),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "numerics": numerics_state(),
    }
    if dev.type == "cuda":
        index = dev.index if dev.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        record.update({"gpu_name": props.name, "gpu_uuid": str(getattr(props, "uuid", "")) or None,
                       "gpu_capability": f"{props.major}.{props.minor}", "gpu_memory_bytes": int(props.total_memory),
                       **_nvidia_smi(index)})
    else:
        record.update({"gpu_name": None, "gpu_uuid": None, "driver_version": None})
    return record


def _torchvision_version() -> str | None:
    try:
        import torchvision
    except ImportError:  # pragma: no cover - torchvision is part of the gpu group
        return None
    return torchvision.__version__
