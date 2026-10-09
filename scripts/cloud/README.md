> Public edition of the registered file `scripts/cloud/README.md` (protocol-v6.0.1, git blob `8fd2878ac4cd`); operational details are replaced as listed in EDITIONS.md.

# Modal backend for F9 and S5 (CPE-1)

`modal_train.py` runs the F9 trained-CNN fits (and the S5 data-efficiency fits) of the `protocol-v6.0.1` amendment on the owner's Modal workspace. The amendment is protocol 2.1, Section D item 10, Section E (S5) and Section H. The settings are in `configs/f9_cnn.yaml` under `backend.modal`. The backend works only inside cloud-processing exception CPE-1 ([protocol/amendments.md](../../protocol/amendments.md), 2 October 2026):

- **What may go up.** Only the derived `acdc_dev` tensor cache: `neural_cache/acdc_dev/manifest.csv` and the `.npz` files it lists. `python -m tmmri.cli upload-cache` is the only upload path. It refuses every other file and writes each upload to `cloud/modal_ledger.csv` in the run root before it calls the Modal client.
- **Where.** Only the Volumes `tmmri-neural-cache` (inputs) and `tmmri-neural-runs` (fit outputs), in the personal workspace `<workspace>`.
- **What stays local.** Evaluation (`infer-f9`), every test or external partition, raw files and `clinical_validation/`. All reported numbers come from local inference.
- **Evidence.** Fit records are downloaded and imported with SHA-256 checks. Modal dashboard logs are never evidence.

Nothing in this folder runs on Modal until the owner has a token. No data have been uploaded.

## One-time setup (owner)

1. `uvx modal setup` stores the token in `%USERPROFILE%\.modal.toml`. Never put the token in the repository.
2. `uvx modal --version` prints the client version. Record it as `backend.modal.client_version` in `configs/f9_cnn.yaml`. Every command below refuses while the pin is `null` or differs from the running client.
3. All commands run from the repository root with the pinned client: `uvx --from modal==<pin> modal ...`.

## Sequence

| Step | Command | Writes |
|---|---|---|
| Qualification on synthetic tensors (about $2) | `uvx --from modal==<pin> modal run scripts/cloud/modal_train.py --qualify` (benchmark another type with `--gpu-override L4`) | `outputs/cloud/qualify_<gpu>_<utc>.json` |
| Plan wave 1 (324 specs) | `docker compose run --rm tmmri python -m tmmri.cli plan-f9 --run-id <run_id>` | `neural_f9/plan/wave1_specs.json` |
| Upload the cache (CPE-1, ledgered) | `docker compose run --rm -e MODAL_TOKEN_ID -e MODAL_TOKEN_SECRET tmmri python -m tmmri.cli upload-cache --run-id <run_id> --create-volume` (try `--dry-run` first) | `cloud/modal_ledger.csv` |
| Export the spec file to the host | `docker compose run --rm -v "${PWD}/outputs/cloud:/export" tmmri cp artifacts/<run_id>/neural_f9/plan/wave1_specs.json /export/` | `outputs/cloud/wave1_specs.json` |
| Wave 1 | `uvx --from modal==<pin> modal run scripts/cloud/modal_train.py --wave 1 --specs outputs/cloud/wave1_specs.json` | Volume `tmmri-neural-runs:/<run_id>/<fit_id>/`; `outputs/cloud/wave1_modal_results.json` |
| Download and import | `uvx --from modal==<pin> modal volume get tmmri-neural-runs /<run_id>/ outputs/cloud/runs`, then `docker compose run --rm tmmri python -m tmmri.cli import-f9-fits --run-id <run_id> --wave 1 --source outputs/cloud/runs/<run_id>` | `neural_f9/checkpoints/<fit_id>/`, `neural_f9/import_ledger.csv` |
| Select (18 refits) | `docker compose run --rm tmmri python -m tmmri.cli select-f9 --run-id <run_id>` | `neural_f9/plan/wave2_specs.json`, `neural_f9/selection/` |
| Wave 2, download, import | as wave 1 with `--wave 2` | the 18 selected checkpoints |
| Freeze | `docker compose run --rm tmmri python -m tmmri.cli freeze-f9 --run-id <run_id>` | `neural_f9/f9_bundle.json` |
| S5 waves (after the F9 freeze) | `run-data-efficiency --step f9-wave1` writes `data_efficiency/f9/train_wave1_specs.json` (purpose `s5`); run it with `modal_train.py --wave 1 --specs <that file>`, download, then `import-f9-fits --wave 1 --specs data_efficiency/f9/train_wave1_specs.json --source ...`; `--step f9-wave2` and wave 2 likewise; `--step f9-infer` runs the refits locally | `neural_f9/checkpoints/<fit_id>/` (never part of `f9_bundle.json`), `data_efficiency/f9/oof/<fit_id>/` |
| Volume deletion (after the release backup) | `uvx --from modal==<pin> modal run scripts/cloud/modal_train.py --print-delete-commands` prints the commands; the owner runs them and records each one with `upload-cache --record-deletion <volume>` | `cloud/modal_ledger.csv` |

`--source` is the folder that holds the downloaded `<fit_id>/` directories. Only `record.json` is required for a selection fit, and `model.pt` as well for a refit. An imported file is never overwritten with different content. `--budget-usd` on a wave refuses to start when the printed projection exceeds it (the workspace cap is $180 per month).

**Local fallback.** `docker compose --profile gpu run --rm tmmri-gpu python -m tmmri.cli train-f9 --run-id <run_id> --backend local --specs neural_f9/plan/wave1_specs.json`. It runs the same `train_fit` sequentially on the GTX 1080 (EXECUTION_ENVIRONMENT_SPEC.md 5.2 rules 7-8), and its fits need no import.

## Design notes

- **Image.** The image uses the Dockerfile's digest-pinned `UBUNTU_IMAGE` and `UV_IMAGE`, the same apt packages and `uv python install 3.12`, and `uv sync --locked --no-install-project --group clinical --group gpu`, the command of the local `env-gpu` target. The freeze hash must equal the local `tmmri-env-gpu` (protocol 2.1 H.2). The `uv pip freeze` hash is written at build time (`/opt/project/freeze.sha256`) and stored, with `MODAL_IMAGE_ID`, in every fit record. Compare it with `docker compose --profile gpu run --rm tmmri-gpu sh -c "uv pip freeze --python /opt/venv/bin/python | sha256sum"`. The image also receives the `configs/runtime.env` thread and seed variables, `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `tmmri` (`add_local_python_source`). Still to confirm in qualification: whether Modal's builder accepts `COPY --from=<digest>` in `dockerfile_commands`, and whether it runs an image whose Python is the uv-managed interpreter on `PATH` (`add_python` is not used).
- **GPU.** One pinned type (`L40S`), with no fallback list. If an A100 or H100 is ever pinned, it needs the `!` suffix so that Modal cannot upgrade it. Avoid sm_100 and sm_120 cards, which have no kernels in the cu126 wheels.
- **Numerics (C9).** Training uses IEEE FP32: TF32 is off through the legacy flags and `fp32_precision = "ieee"`. Algorithms are deterministic, `cudnn.benchmark` is off and execution is eager. Every fit record stores the state together with the GPU name, UUID and driver.
- **Retries and preemption.** `modal.Retries(3, backoff 2.0)` restarts a failed input. `train_fit` resumes from `resume/last.pt` (model, optimizer, scheduler, early-stopping state and RNG). The epoch RNG is re-derived from `(seed, epoch)`, so the resumed fit equals an uninterrupted one bit for bit on the same hardware. A completed fit directory is returned unchanged. Non-preemptible execution (3x the price) is not used.
- **Volumes.** The function reads `tmmri-neural-cache:/<run_id>/neural_cache/acdc_dev` and verifies the manifest and every `.npz` against SHA-256 before training. It writes `tmmri-neural-runs:/<run_id>/<fit_id>/` and commits after every epoch. The runs Volume is created as Volumes v2 (many concurrent writers of distinct files).
- **Backbone weights.** Each container downloads `resnet18-f37072fd.pth` once from `download.pytorch.org/models/` and verifies it against the SHA-256 pinned in `registry/weights.yaml`, which the spec file carries. The weights are never uploaded.
