#!/usr/bin/env python3
"""Qualify the CMR-TRANS-AUDIT execution container (docs/EXECUTION_ENVIRONMENT_SPEC.md 1.0.0, checks C10-C17).

Default command of the Compose service `tmmri` (CPU target `env`):
    docker compose run --rm tmmri
which runs:
    python scripts/verify_environment.py --out artifacts/environment/verify.json
The optional GPU target adds the GPU checks (Decision D20):
    docker compose --profile gpu run --rm tmmri-gpu
which runs the same script with `--out artifacts/environment/verify_gpu.json --gpu`.

Uses only the standard library and the locked third-party packages; it never imports `tmmri`. It never opens a
cohort file: the source mounts are only listed and their mount flags read. It writes the JSON report and one
synthetic NIfTI file in a temporary folder of the data volume, which it deletes. `git_clean` records the Git
state (commit, `dirty`) and fails only if Git cannot be read; a dirty tree never fails qualification (check C17
judges `"dirty"` after the coordinator's commit). Exits with status 1 if any check fails.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

EXPECTED = {"python": (3, 12), "torch": "2.14.0", "torchvision": "0.29.0", "cuda": "12.6"}
VOLUMES = ("/workspace/data", "/workspace/artifacts", "/cache")
SOURCES = ("/sources/acdc", "/sources/mms")
MOUNTS = ("/workspace", "/workspace/data", "/workspace/data/raw", "/workspace/artifacts", "/cache", *SOURCES,
          "/archive")
LIBRARIES = ("numpy", "scipy", "pandas", "pyarrow", "nibabel", "skimage", "sklearn", "yaml", "joblib",
             "threadpoolctl", "matplotlib")
RECORDED_DISTRIBUTIONS = ("psutil", "tabulate", "pytest", "pytest-xdist", "ruff", "python-docx", "reportlab", "lxml")
MIN_FREE_GB = 20
SEED = 20260613  # configs/study_v6.yaml seeds.synthetic_controls
BOOTSTRAP_REPLICATES = 200  # downstream refit bootstrap count (Section 3 of the design brief)
REPLICATES_PER_TASK = 10


def cpu_name() -> str | None:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            return next((ln.split(":", 1)[1].strip() for ln in fh if ln.startswith("model name")), None)
    except OSError:
        return None


def cpu_quota() -> float | None:
    """CPUs granted by the cgroup quota (Compose `cpus`), or None when unlimited."""
    try:
        with open("/sys/fs/cgroup/cpu.max", encoding="utf-8") as fh:
            quota, period = fh.read().split()
        return None if quota == "max" else round(int(quota) / int(period), 2)
    except (OSError, ValueError):
        return None


def meminfo_gib(key: str) -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            kib = next(int(ln.split()[1]) for ln in fh if ln.startswith(key + ":"))
        return round(kib / 2**20, 1)
    except (OSError, StopIteration, ValueError):
        return None


def shm_gib() -> float | None:
    try:
        stat = os.statvfs("/dev/shm")
        return round(stat.f_frsize * stat.f_blocks / 2**30, 2)
    except OSError:
        return None


def mount_info(path: str) -> dict:
    """Mount flags of a path; never opens a file below it."""
    if not os.path.isdir(path):
        return {"exists": False}
    info: dict = {"exists": True, "mount_point": os.path.ismount(path)}
    try:
        info["read_only"] = bool(os.statvfs(path).f_flag & os.ST_RDONLY)
        info["writable"] = os.access(path, os.W_OK)
        with os.scandir(path) as entries:
            info["non_empty"] = next(entries, None) is not None
    except OSError as exc:
        info["error"] = str(exc)
    return info


def sha256_file(path: str) -> str | None:
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


def read_text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return None


def git_state(repo: str = "/workspace") -> dict:
    if shutil.which("git") is None:
        return {"available": False}
    start = time.perf_counter()
    head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    # --no-optional-locks: never refresh .git/index from inside the container (the Windows side owns the index).
    status = subprocess.run(["git", "--no-optional-locks", "-C", repo, "status", "--porcelain"],
                            capture_output=True, text=True, check=False)
    autocrlf = subprocess.run(["git", "-C", repo, "config", "--get", "core.autocrlf"],
                              capture_output=True, text=True, check=False)
    entries = [ln for ln in status.stdout.splitlines() if ln.strip()]
    return {
        "available": True,
        "commit": head.stdout.strip() or None,
        "dirty": bool(entries),
        "dirty_entries": len(entries),
        "dirty_sample": entries[:10],
        "core_autocrlf": autocrlf.stdout.strip() or None,
        "status_seconds": round(time.perf_counter() - start, 2),
        "ok": head.returncode == 0 and status.returncode == 0,
        "error": (head.stderr or status.stderr).strip() or None,
    }


def import_libraries() -> tuple[dict, dict]:
    versions: dict = {}
    errors: dict = {}
    for name in LIBRARIES:
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001 - any import failure fails the check
            errors[name] = f"{type(exc).__name__}: {exc}"
    for dist in RECORDED_DISTRIBUTIONS:
        try:
            versions[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            versions[dist] = None
    return versions, errors


def blas_pools() -> list[dict]:
    import numpy  # noqa: F401  (loads its BLAS)
    import scipy.linalg  # noqa: F401  (loads its BLAS)
    import sklearn  # noqa: F401  (loads OpenMP)
    from threadpoolctl import threadpool_info

    return threadpool_info()


def ridge_bootstrap_task(x, y, lam: float, seed: int, replicates: list[int]):
    """Ridge coefficients for bootstrap replicates; each replicate owns its seed, so worker count cannot matter."""
    import numpy as np
    from threadpoolctl import threadpool_limits

    out = []
    with threadpool_limits(limits=1):  # one BLAS thread per worker (Decision D3)
        for rep in replicates:
            rng = np.random.default_rng([seed, rep])
            idx = rng.integers(0, x.shape[0], size=x.shape[0])
            xb = x[idx] - x[idx].mean(axis=0)
            yb = y[idx] - y[idx].mean(axis=0)
            out.append(np.linalg.solve(xb.T @ xb + lam * np.eye(x.shape[1]), xb.T @ yb))
    return out


def bootstrap_digest(n_jobs: int) -> str:
    import numpy as np
    from joblib import Parallel, delayed

    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((150, 57))                 # ACDC-sized synthetic A (57 formal descriptors)
    y = x @ rng.standard_normal((57, 7)) + 0.1 * rng.standard_normal((150, 7))  # 7 primary-family targets
    reps = list(range(BOOTSTRAP_REPLICATES))
    tasks = [reps[i:i + REPLICATES_PER_TASK] for i in range(0, len(reps), REPLICATES_PER_TASK)]
    parts = Parallel(n_jobs=n_jobs, backend="loky")(delayed(ridge_bootstrap_task)(x, y, 1.0, SEED, t) for t in tasks)
    coefs = np.stack([c for part in parts for c in part]).astype("<f8")
    return hashlib.sha256(coefs.tobytes()).hexdigest()


def parallel_determinism(workers: int) -> dict:
    start = time.perf_counter()
    try:
        digests = {}
        for run in (1, 2):
            digests[f"n_jobs_1_run_{run}"] = bootstrap_digest(1)
            digests[f"n_jobs_{workers}_run_{run}"] = bootstrap_digest(workers)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    values = set(digests.values())
    return {"ok": len(values) == 1, "workers": workers, "replicates": BOOTSTRAP_REPLICATES, "digests": digests,
            "seconds": round(time.perf_counter() - start, 2)}


def nifti_round_trip(folder: str = "/workspace/data") -> dict:
    try:
        import nibabel as nib
        import numpy as np

        labels = (np.arange(16 * 12 * 8, dtype=np.int16).reshape(16, 12, 8) % 4)
        zooms = (1.25, 1.25, 10.0)  # anisotropic short-axis spacing: slice spacing must survive the round trip
        affine = np.diag([*zooms, 1.0])
        with tempfile.TemporaryDirectory(dir=folder if os.path.isdir(folder) else None, prefix=".verify-") as tmp:
            path = os.path.join(tmp, "roundtrip_gt.nii.gz")
            img = nib.Nifti1Image(labels, affine)
            img.header.set_zooms(zooms)
            nib.save(img, path)
            back = nib.load(path)
            data = np.asarray(back.dataobj)
            ok = (np.array_equal(data, labels) and data.dtype == labels.dtype
                  and tuple(float(z) for z in back.header.get_zooms()[:3]) == zooms
                  and np.allclose(back.affine, affine))
        return {"ok": bool(ok), "folder": folder, "zooms": list(zooms)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def disk_free() -> dict:
    out = {}
    for path in VOLUMES:
        try:
            out[path] = round(shutil.disk_usage(path).free / 1e9, 1)
        except OSError:
            out[path] = None
    return out


def nvidia_smi() -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    query = "--query-gpu=name,driver_version,memory.total,memory.used,compute_cap"
    out = subprocess.run([exe, query, "--format=csv,noheader"], capture_output=True, text=True, check=False)
    return (out.stdout or out.stderr).strip()


def kernels_for(major: int, minor: int, arch_list: list[str]) -> bool:
    """True if the wheel ships GPU code this device can run.

    CUDA binaries for sm_XY run on devices of the same major version X with minor >= Y, so the cu126
    wheels' sm_60 code serves compute capability 6.1 (GTX 1080). Architecture-specific targets such as
    sm_90a run only on their exact capability.
    """
    for arch in arch_list:
        if not arch.startswith("sm_"):
            continue
        code = arch[3:]
        digits = code.rstrip("af")
        if len(digits) < 2 or not digits.isdigit():
            continue
        arch_major, arch_minor = int(digits[:-1]), int(digits[-1])
        if arch_major == major and (arch_minor == minor if digits != code else arch_minor <= minor):
            return True
    return False


def gpu_checks(report: dict, checks: dict) -> None:
    try:
        import torch
        import torchvision
    except Exception as exc:  # noqa: BLE001
        report["gpu_error"] = f"{type(exc).__name__}: {exc}"
        checks.update({"torch_version": False, "cuda_runtime": False, "cuda_available": False,
                       "kernels_for_this_gpu": False})
        return
    report["versions"].update({"torch": torch.__version__, "torchvision": torchvision.__version__,
                               "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()})
    report["nvidia_smi"] = nvidia_smi()
    checks["torch_version"] = torch.__version__.split("+")[0] == EXPECTED["torch"]
    checks["torchvision_version"] = torchvision.__version__.split("+")[0] == EXPECTED["torchvision"]
    checks["cuda_runtime"] = (torch.version.cuda or "").startswith(EXPECTED["cuda"])
    checks["cuda_available"] = torch.cuda.is_available()
    checks["kernels_for_this_gpu"] = False
    if not checks["cuda_available"]:
        return
    major, minor = torch.cuda.get_device_capability(0)
    props = torch.cuda.get_device_properties(0)
    arch_list = torch.cuda.get_arch_list()
    report["gpu"] = {"device": props.name, "capability": f"{major}.{minor}",
                     "total_memory_gib": round(props.total_memory / 2**30, 2), "arch_list": arch_list}
    checks["kernels_for_this_gpu"] = kernels_for(major, minor, arch_list)
    try:  # run one convolution on the device and compare it with the CPU result
        gen = torch.Generator().manual_seed(SEED)
        x, w = torch.randn(2, 3, 32, 32, generator=gen), torch.randn(4, 3, 3, 3, generator=gen)
        cpu = torch.nn.functional.conv2d(x, w)
        gpu = torch.nn.functional.conv2d(x.cuda(), w.cuda()).cpu()
        checks["gpu_kernel_runs"] = bool(torch.allclose(cpu, gpu, rtol=1e-4, atol=1e-4))
    except RuntimeError as exc:
        report["gpu_kernel_error"] = str(exc)
        checks["gpu_kernel_runs"] = False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", help="write the JSON report to this path")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("TMMRI_N_JOBS") or os.cpu_count() or 1),
                        help="joblib workers for the determinism check (default: TMMRI_N_JOBS)")
    parser.add_argument("--gpu", action="store_true", help="also check the optional GPU stack (target env-gpu)")
    args = parser.parse_args()

    versions, import_errors = import_libraries()
    report: dict = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "kernel": platform.release(),
        "in_container": os.path.exists("/.dockerenv"),
        "host": {
            "cpu": cpu_name(),
            "logical_cpus": os.cpu_count(),
            "cpu_quota": cpu_quota(),
            "ram_gib": meminfo_gib("MemTotal"),
            "swap_gib": meminfo_gib("SwapTotal"),
            "shm_gib": shm_gib(),
        },
        "mounts": {path: mount_info(path) for path in MOUNTS},
        "git": git_state(),
        "lock": {"uv_lock_sha256": sha256_file("/workspace/uv.lock"),
                 "image_uv_lock_sha256": sha256_file("/opt/project/uv.lock"),
                 "python_version_file": read_text("/workspace/.python-version")},
        "versions": versions,
        "import_errors": import_errors,
        "env": {k: os.environ.get(k) for k in (
            "PYTHONHASHSEED", "PYTHONUTF8", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "MPLBACKEND", "PYTHONPATH")}
               | {k: v for k, v in sorted(os.environ.items()) if k.startswith("TMMRI_")},
    }
    try:
        report["blas"] = blas_pools()
    except Exception as exc:  # noqa: BLE001
        report["blas"] = []
        report["blas_error"] = f"{type(exc).__name__}: {exc}"
    report["parallel_determinism"] = parallel_determinism(max(2, args.workers))
    report["nifti_round_trip"] = nifti_round_trip()
    report["disk_free_gb"] = disk_free()

    mounts = report["mounts"]
    blas = [pool for pool in report["blas"] if pool.get("user_api") == "blas"]
    checks: dict[str, bool] = {
        "python_3_12": sys.version_info[:2] == EXPECTED["python"],
        "linux": platform.system() == "Linux",
        "in_container": report["in_container"],
        "volumes_mounted": all(mounts[p].get("mount_point") for p in VOLUMES),
        "sources_mounted_read_only": all(mounts[p].get("non_empty") and mounts[p].get("read_only")
                                         and not mounts[p].get("writable") for p in SOURCES),
        "libraries_import": not import_errors,
        "blas_single_thread": bool(blas) and all(pool.get("num_threads") == 1 for pool in blas),
        "parallel_determinism": report["parallel_determinism"]["ok"],
        "nifti_round_trip": report["nifti_round_trip"]["ok"],
        "disk_free": all((free or 0) >= MIN_FREE_GB for free in report["disk_free_gb"].values()),
        "git_clean": bool(report["git"].get("ok")),
        "lock_matches_image": (report["lock"]["uv_lock_sha256"] is not None
                               and report["lock"]["uv_lock_sha256"] == report["lock"]["image_uv_lock_sha256"]),
    }
    if args.gpu:
        gpu_checks(report, checks)
    report["checks"] = checks
    report["passed"] = all(checks.values())
    text = json.dumps(report, indent=2, default=str)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
