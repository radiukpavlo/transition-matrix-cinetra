"""Modal app ``tmmri-neural``: F9 and S5 fits under cloud-processing exception CPE-1 (protocol 2.1 Section H;
docs/MODAL_COMPUTE_ASSESSMENT_2026-10-02.md Sections 4-5; configs/f9_cnn.yaml ``backend.modal``).

Run from the repository root on the owner's workstation (token in %USERPROFILE%\\.modal.toml or MODAL_TOKEN_ID /
MODAL_TOKEN_SECRET), with ``<pin>`` = ``backend.modal.client_version``::

    uvx --from modal==<pin> modal run scripts/cloud/modal_train.py --qualify [--gpu-override L4]
    uvx --from modal==<pin> modal run scripts/cloud/modal_train.py --wave 1 --specs <wave1_specs.json>
    uvx --from modal==<pin> modal run scripts/cloud/modal_train.py --wave 2 --specs <wave2_specs.json>
    uvx --from modal==<pin> modal run scripts/cloud/modal_train.py --print-delete-commands

* Image: the repository's digest-pinned Ubuntu and uv bases (read from the Dockerfile), the same ``uv.lock`` synced as
  the local ``env-gpu`` target (``uv sync --locked --no-install-project --group clinical --group gpu``), its
  ``uv pip freeze`` hash recorded at build time, ``configs/runtime.env`` variables, and ``tmmri`` added with
  ``add_local_python_source``.  Every fit record carries the image id, the freeze hash and the GPU facts.
* ``fit_remote`` calls :func:`tmmri.neural.train.train_fit` (the only training entry point) with the tensor cache on
  Volume ``tmmri-neural-cache`` (``/<run_id>/neural_cache/acdc_dev``) and outputs on ``tmmri-neural-runs``
  (``/<run_id>/<fit_id>/``); it commits the Volume after every epoch checkpoint, uses ``modal.Retries`` and resumes
  from the last epoch checkpoint after a preemption.  One pinned GPU type, no fallback list.
* ``--wave`` maps the spec file with ``.map(order_outputs=False)`` and at most ``max_containers`` (10) GPUs and writes
  ``wave<k>_modal_results.json`` beside the spec file.  The local entrypoint never imports torch.
* ``--qualify`` runs synthetic tensors only (:func:`tmmri.neural.qualify.run_qualification`): throughput, two identical
  fits bitwise equal, an interrupted and resumed fit equal to the uninterrupted one, and the IEEE FP32 (TF32 off) state.
* ``--print-delete-commands`` only prints the Volume deletion commands for the owner (after the release backup).

Nothing here uploads data: the cache goes up only through ``python -m tmmri.cli upload-cache`` (CPE-1 ledger).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import modal

SETTINGS_ENV = "TMMRI_MODAL_SETTINGS"
REMOTE_CONFIG = "/opt/tmmri/configs/f9_cnn.yaml"
WEIGHTS_DIR = "/opt/torch/hub/checkpoints"
TORCHVISION_MODELS = "https://download.pytorch.org/models/"
RUNTIME_ENV_KEYS = ("PYTHONHASHSEED", "PYTHONUTF8", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "MPLBACKEND")


def _repo() -> Path:
    return Path(__file__).resolve().parents[2]


def _dockerfile_bases(dockerfile: Path) -> tuple[str, str]:
    found = dict(re.findall(r"^ARG (UBUNTU_IMAGE|UV_IMAGE)=(\S+)$", dockerfile.read_text(encoding="utf-8"), flags=re.MULTILINE))
    for key in ("UBUNTU_IMAGE", "UV_IMAGE"):
        if "@sha256:" not in found.get(key, ""):
            raise SystemExit(f"{dockerfile}: {key} is not pinned by digest")
    return found["UBUNTU_IMAGE"], found["UV_IMAGE"]


def _runtime_env(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and not key.startswith("#") and key in RUNTIME_ENV_KEYS:
            values[key] = value
    return values


def _local_settings() -> dict:
    repo = _repo()
    sys.path.insert(0, str(repo / "src"))
    from tmmri.neural.cloud import read_backend_config, require_client_pin

    modal_cfg = dict(read_backend_config(repo / "configs" / "f9_cnn.yaml")["modal"])
    pin = require_client_pin(modal_cfg)
    if modal.__version__ != pin:
        raise SystemExit(f"Modal client {modal.__version__} differs from the pinned {pin}: use uvx --from modal=={pin} modal run ...")
    if not isinstance(modal_cfg["gpu"], str) or not modal_cfg["gpu"]:
        raise SystemExit("configs/f9_cnn.yaml backend.modal.gpu must be one GPU type (no fallback list)")
    ubuntu, uv = _dockerfile_bases(repo / "Dockerfile")
    return {"modal": modal_cfg, "ubuntu_image": ubuntu, "uv_image": uv, "runtime_env": _runtime_env(repo / "configs" / "runtime.env")}


SETTINGS = _local_settings() if modal.is_local() else json.loads(os.environ[SETTINGS_ENV])
MODAL_CFG = SETTINGS["modal"]
REPO = _repo() if modal.is_local() else Path("/nonexistent-local-repo")

image = (
    modal.Image.from_registry(SETTINGS["ubuntu_image"])
    .dockerfile_commands([
        # Modal's builder rejects a COPY source with both a tag and a digest (qualification of 3 October 2026), so the
        # digest alone is used: the same pinned image bytes as the Dockerfile's UV_IMAGE.
        f"COPY --from={re.sub(r':[^:@/]+@', '@', SETTINGS['uv_image'])} /uv /uvx /usr/local/bin/",
        ("ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 TZ=Etc/UTC UV_PYTHON_INSTALL_DIR=/opt/uv/python "
         "UV_PYTHON_PREFERENCE=only-managed UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy MPLBACKEND=Agg TORCH_HOME=/opt/torch"),
        "ENV PATH=/opt/venv/bin:$PATH",
        ("RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git git-lfs unzip pigz pv jq rsync time "
         "&& rm -rf /var/lib/apt/lists/*"),
        "RUN uv python install 3.12",
        "ENV UV_PYTHON_DOWNLOADS=never",
    ])
    .add_local_file(REPO / "pyproject.toml", "/opt/project/pyproject.toml", copy=True)
    .add_local_file(REPO / "uv.lock", "/opt/project/uv.lock", copy=True)
    .add_local_file(REPO / ".python-version", "/opt/project/.python-version", copy=True)
    .run_commands(
        # Modal's builder may inject package-index variables (a mirror); they make uv see a source other than the one in
        # uv.lock, so they are listed (names only) and cleared for the locked sync (qualification of 3 October 2026).
        "env | grep -E '^(UV|PIP)_' | sed 's/=.*//' || true",
        ("cd /opt/project && env -u UV_INDEX_URL -u UV_EXTRA_INDEX_URL -u UV_DEFAULT_INDEX -u UV_INDEX -u PIP_INDEX_URL "
         "-u PIP_EXTRA_INDEX_URL -u PIP_TRUSTED_HOST uv sync --locked --no-install-project --group clinical --group gpu"),
        "uv pip freeze --python /opt/venv/bin/python > /opt/project/freeze.txt",
        "sha256sum /opt/project/freeze.txt | cut -d' ' -f1 > /opt/project/freeze.sha256",
    )
    .env({**SETTINGS["runtime_env"], "CUBLAS_WORKSPACE_CONFIG": ":4096:8", SETTINGS_ENV: json.dumps(SETTINGS, sort_keys=True)})
    .add_local_file(REPO / "configs" / "f9_cnn.yaml", REMOTE_CONFIG)
    .add_local_python_source("tmmri")
)

app = modal.App(str(MODAL_CFG["app"]))
cache_volume = modal.Volume.from_name(str(MODAL_CFG["cache_volume"]))
_runs_kwargs: dict = {"create_if_missing": True}
if MODAL_CFG.get("runs_volume_version"):
    _runs_kwargs["version"] = int(MODAL_CFG["runs_volume_version"])
runs_volume = modal.Volume.from_name(str(MODAL_CFG["runs_volume"]), **_runs_kwargs)
FUNCTION_ARGS = {
    "image": image,
    "gpu": str(MODAL_CFG["gpu"]),
    "cpu": float(MODAL_CFG["cpu"]),
    "memory": int(MODAL_CFG["memory_mib"]),
    "timeout": int(MODAL_CFG["timeout_s"]),
    "retries": modal.Retries(max_retries=int(MODAL_CFG["retries"]), backoff_coefficient=float(MODAL_CFG["retry_backoff_coefficient"]),
                             initial_delay=float(MODAL_CFG["retry_initial_delay_s"])),
    "max_containers": int(MODAL_CFG["max_containers"]),
}


def _backend_facts() -> dict:
    freeze = Path("/opt/project/freeze.sha256")
    return {"name": "modal", "app": MODAL_CFG["app"], "gpu_type": MODAL_CFG["gpu"], "client_version": MODAL_CFG["client_version"],
            "image_id": os.environ.get("MODAL_IMAGE_ID"), "task_id": os.environ.get("MODAL_TASK_ID"),
            "region": os.environ.get("MODAL_REGION"), "cloud_provider": os.environ.get("MODAL_CLOUD_PROVIDER"),
            "uv_pip_freeze_sha256": freeze.read_text(encoding="utf-8").strip() if freeze.is_file() else None}


def _ensure_weights(info: dict) -> dict:
    """The pinned backbone file on the container disk (downloaded once per container; SHA-256 always verified)."""

    import urllib.request

    target = Path(WEIGHTS_DIR) / str(info["file"])
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        url = info.get("url") or f"{TORCHVISION_MODELS}{info['file']}"
        if not str(url).startswith(TORCHVISION_MODELS):
            raise ValueError(f"weights URL {url} is not a torchvision model URL")
        tmp = target.with_name(f".{target.name}.tmp")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != str(info["sha256"]).lower():
        target.unlink(missing_ok=True)
        raise ValueError(f"{target}: SHA-256 {digest} differs from the pinned {info['sha256']}")
    return {"name": info["name"], "sha256": digest, "path": str(target)}


@app.function(**FUNCTION_ARGS, volumes={str(MODAL_CFG["cache_mount"]): cache_volume, str(MODAL_CFG["runs_mount"]): runs_volume})
def fit_remote(item: dict) -> dict:
    from tmmri.neural.train import train_fit

    run_id, spec = item["run_id"], item["spec"]
    out_dir = Path(MODAL_CFG["runs_mount"]) / run_id / spec["fit_id"]
    cache_dir = Path(MODAL_CFG["cache_mount"]) / run_id / "neural_cache"
    backend = {**_backend_facts(), "spec_file_sha256": item["spec_file_sha256"], "wave": item["wave"]}
    record = train_fit(spec, cache_dir, out_dir, "cuda", config=item["config"], weights=_ensure_weights(item["weights"]),
                       expected_manifest_sha256=item["cache_manifest_sha256"], backend=backend, on_checkpoint=runs_volume.commit)
    runs_volume.commit()
    out = {"fit_id": record.fit_id, "status": record.status, "checkpoint_sha256": record.checkpoint_sha256,
           "selected_epoch": record.selected_epoch, "seconds": record.timings.get("seconds_total"),
           "attempts": record.resume.get("attempts"), "gpu_name": record.environment.get("gpu_name"), "image_id": backend["image_id"]}
    return _plain(out)


def _plain(value: dict) -> dict:
    """JSON round trip: the local client has no torch, so no torch or numpy object may cross back (qualification of
    3 October 2026: a torch version object made the report undecodable)."""

    return json.loads(json.dumps(value, default=str))


@app.function(**FUNCTION_ARGS)
def qualify_remote(settings: dict) -> dict:
    """Synthetic tensors only; nothing is read from or written to a Volume."""

    from tmmri.neural.f9_config import load_f9_config, training_config
    from tmmri.neural.qualify import run_qualification

    cfg = training_config(load_f9_config(path=REMOTE_CONFIG))
    report = run_qualification("/tmp/tmmri-qualify", "cuda", config=cfg, **settings)
    report["backend"] = _backend_facts()
    return _plain(report)


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


@app.local_entrypoint()
def main(specs: str = "", wave: int = 0, qualify: bool = False, gpu_override: str = "", out: str = "", budget_usd: float = 0.0,
         print_delete_commands: bool = False) -> None:
    from tmmri.neural.cloud import volume_delete_commands

    pin = str(MODAL_CFG["client_version"])
    if print_delete_commands:
        for command in volume_delete_commands(pin, MODAL_CFG):
            print(" ".join(command))
        print("After each deletion record it: docker compose run --rm tmmri python -m tmmri.cli upload-cache --run-id <run_id> "
              "--record-deletion <volume>")
        return
    if qualify:
        function = qualify_remote.with_options(gpu=gpu_override) if gpu_override else qualify_remote
        report = function.remote({"size": 224, "n_per_class": 3, "slices": (8, 10), "epochs": 3, "bench_steps": 20})
        report["gpu_requested"] = gpu_override or MODAL_CFG["gpu"]
        target = Path(out) if out else _repo() / "outputs" / "cloud" / f"qualify_{report['gpu_requested']}_{_utc_stamp()}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps({k: report.get(k) for k in ("passed", "benchmark", "determinism", "resume", "ieee_fp32", "tf32_disabled")},
                         indent=2, default=str))
        print(f"report: {target}")
        if not report.get("passed"):
            raise SystemExit(1)
        return
    if not specs or wave not in (1, 2):
        raise SystemExit("--specs <file> and --wave 1|2 are required (or --qualify / --print-delete-commands)")
    from tmmri.neural.fit_spec import read_spec_file

    spec_path = Path(specs)
    payload = read_spec_file(spec_path)
    if int(payload["wave"]) != int(wave):
        raise SystemExit(f"{spec_path} is wave {payload['wave']}, not {wave}")
    spec_sha = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    projected = len(payload["specs"]) * float(MODAL_CFG["projected_minutes_per_fit"]) / 60.0 * float(MODAL_CFG["gpu_usd_per_hour"])
    print(f"{len(payload['specs'])} fits on {MODAL_CFG['gpu']} (max {MODAL_CFG['max_containers']} containers); projected ${projected:.2f}")
    if budget_usd and projected > budget_usd:
        raise SystemExit(f"projected ${projected:.2f} exceeds --budget-usd {budget_usd:.2f}; nothing started")
    items = [{"run_id": payload["run_id"], "wave": int(wave), "spec": spec, "config": payload["config"], "weights": payload["weights"],
              "cache_manifest_sha256": payload["cache_manifest_sha256"], "spec_file_sha256": spec_sha} for spec in payload["specs"]]
    results, failures = [], []
    for result in fit_remote.map(items, order_outputs=False, return_exceptions=True):
        if isinstance(result, BaseException):
            failures.append(repr(result))
            print(f"FAILED: {result!r}")
            continue
        results.append(result)
        print(f"{result['fit_id']} {result['status']} {result['checkpoint_sha256']}")
    summary = spec_path.with_name(f"wave{int(wave)}_modal_results.json")
    summary.write_text(json.dumps({"run_id": payload["run_id"], "wave": int(wave), "spec_file_sha256": spec_sha, "n_specs": len(items),
                                   "completed": len(results), "failed": failures, "results": sorted(results, key=lambda r: r["fit_id"]),
                                   "utc": _utc_stamp()}, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"summary: {summary}")
    print(f"download: uvx --from modal=={pin} modal volume get {MODAL_CFG['runs_volume']} /{payload['run_id']}/ <local folder>; "
          f"then: python -m tmmri.cli import-f9-fits --run-id {payload['run_id']} --wave {int(wave)} --source <local folder>/{payload['run_id']}")
    if failures:
        raise SystemExit(1)
