from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

from .paths import PROJECT_ROOT, resolve_path, resolve_recorded_path

PARQUET_ENGINE = "pyarrow"
PARQUET_COMPRESSION = "snappy"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dirs(paths: Iterable[str | Path]) -> None:
    for path in paths:
        resolve_path(path).mkdir(parents=True, exist_ok=True)


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, engine=PARQUET_ENGINE, compression=PARQUET_COMPRESSION, index=False)
    return out


def write_table_pair(df: pd.DataFrame, parquet_path: str | Path, csv_path: str | Path | None = None) -> list[Path]:
    paths = [write_parquet(df, parquet_path)]
    if csv_path is not None:
        out = resolve_path(csv_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        paths.append(out)
    return paths


def write_yaml(data: dict[str, Any], path: str | Path) -> Path:
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(data, f, sort_keys=True, allow_unicode=False)
    return out


def read_yaml(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def write_json(data: dict[str, Any], path: str | Path) -> Path:
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, sort_keys=True, indent=2)
        f.write("\n")
    return out


def sha256_file(path: str | Path) -> str:
    p = resolve_path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def git_commit(root: Path = PROJECT_ROOT) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return proc.stdout.strip() or None


def file_manifest_sha256(paths: Iterable[str | Path], root: Path = PROJECT_ROOT) -> str:
    rows: list[str] = []
    for path in sorted(resolve_path(p, root) for p in paths):
        if not path.exists() or not path.is_file():
            continue
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            rel = path.name
        rows.append(f"{rel}\t{sha256_file(path)}")
    return sha256_text("\n".join(rows))


def write_checkpoint(
    name: str,
    script_name: str,
    input_paths: Iterable[str | Path],
    output_paths: Iterable[str | Path],
    cases_processed: int,
    failures: list[str] | None = None,
    checkpoint_dir: str | Path = "artifacts/checkpoints",
) -> Path:
    failures = failures or []
    payload = {
        "timestamp_utc": utc_now_iso(),
        "script_name": script_name,
        "git_commit": git_commit(),
        "input_paths": [str(resolve_path(p)) for p in input_paths],
        "output_paths": [str(resolve_path(p)) for p in output_paths],
        "number_of_cases_processed": int(cases_processed),
        "number_of_failures": len(failures),
        "failure_summary": failures,
    }
    return write_json(payload, resolve_path(checkpoint_dir) / name)


def runtime_environment() -> dict[str, Any]:
    packages = {}
    for name in [
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "scikit-image",
        "scikit-learn",
        "nibabel",
        "torch",
        "monai",
    ]:
        try:
            from importlib.metadata import version

            packages[name] = version(name)
        except Exception:
            packages[name] = None
    return {
        "timestamp_utc": utc_now_iso(),
        "operating_system": platform.platform(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "packages": packages,
        "cpu_model": platform.processor() or None,
        "ram_gb": None,
        "gpu_model": None,
        "cuda_version": None,
        "cudnn_version": None,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }


def deterministic_seed(seed: int = 0) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    try:
        from monai.utils import set_determinism

        set_determinism(seed=seed)
    except Exception:
        pass
