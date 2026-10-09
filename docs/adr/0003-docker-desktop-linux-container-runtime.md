> Public edition of `docs/adr/0003-docker-desktop-linux-container-runtime.md` of the authors' development repository (commit `7d2adbe546f7`); operational details are replaced as listed in EDITIONS.md.

# ADR-0003: Execution runtime: a CPU-first Linux container on Docker Desktop

**Status:** accepted, September 28, 2026 (pre-freeze). **Relates to:** Decisions D3, D22 and D23 of [SPECIFICATION.md](../../SPECIFICATION.md) 6.0.0, all unchanged. **Details:** docs/EXECUTION_ENVIRONMENT_SPEC.md 1.0.0.

## Context

Up to protocol v5 the pipeline ran in a Windows virtual environment on the workstation (CPython 3.13.6 for `v5_native_image_dss_100`), with a constraints file but no lock. Four weaknesses followed from that:

- **Unlocked stack.** Package versions were recorded after the run, not fixed before it.
- **Platform coupling.** Paths were resolved from the Windows checkout, 260-character limits applied, and 532 hash-locked historical files keep CRLF bytes because the checkout uses `core.autocrlf=true`.
- **Thread counts.** BLAS threading was not fixed per worker, so parallel runs could not promise worker-independent results.
- **Tracked outputs.** In-place runs rewrote tracked registry, feature and split files.

D3 binds a CPU-first design: one Windows 11 workstation with 16 logical CPUs and 32 GB RAM; per-case parallelism with one BLAS thread per worker; the GTX 1080 unused by the primary pipeline. D22 binds a Linux container on Docker Desktop with the digest pins of the CXR-TRANS-AUDIT image, CPython 3.12.14 managed by `uv`, and a committed `uv.lock`. D23 binds one-pass verify-and-stage into a data volume.

The same workstation already runs Docker Desktop 4.91.0 on the WSL 2 back end for the CXR-TRANS-AUDIT project (its own ADR-0003), with the Docker data disk on `F:`. Three realizations were compared:

- the Windows-native virtual environment, kept as it was;
- WSL 2 with a user-managed Ubuntu distribution;
- a Linux container on the existing Docker Desktop installation.

## Decision

Realize D3, D22 and D23 as a Linux container on Docker Desktop.

- **Execution.** Every protocol run executes in the Compose project `tmmri`, from the image `tmmri-env:local`. The image is built from the committed `uv.lock` (`uv sync --locked`) on `ubuntu:24.04` and `uv` 0.12.18, both pinned by digest, with CPython 3.12.14 in an immutable `/opt/venv`.
- **CPU first.** The default target `env` has no GPU access. Per-case work runs in `joblib` workers (`TMMRI_N_JOBS=12`) with one BLAS thread each (`configs/runtime.env`), inside a CPU cap of 14 of the 16 logical CPUs (`TMMRI_CPUS`).
- **Optional GPU target.** The target `env-gpu` (service `tmmri-gpu`, Compose profile `gpu`) adds `torch 2.14.0+cu126` and `torchvision 0.29.0+cu126` from the explicit `cu126` index, only for the optional deep-encoder branch (D20).
- **Storage.** Code is bind-mounted from `<local path>`. The staging cache, manifests, run outputs and caches live in the named volumes `tmmri_data`, `tmmri_artifacts` and `tmmri_cache` on Docker's disk on `F:`. ACDC and the complete M&Ms-1 release stay where they are and are bound read-only at `/sources/acdc` and `/sources/mms`; they are never copied into the repository. Environment records and image tarballs go to `<local path>`.
- **Line endings.** The checkout keeps `core.autocrlf=true`, so the hash-locked CRLF bytes stay valid. The image sets the same Git option, so Git inside the container reports an unmodified checkout as clean.
- **Host virtual environment.** `<local path>` (CPython 3.13) stays a development convenience for editing and quick tests. It is not an execution environment for protocol runs.

## Consequences

- **Provenance.** Every manifest records the image identifier, the base-image digests, the `uv.lock` SHA-256, the exact Python patch and the host facts (Windows build, WSL, Docker Desktop, kernel, driver). Qualification is `scripts/verify_environment.py` plus `scripts/host/host_facts.ps1` (environment specification §11).
- **Worker-independent results.** The qualification checks that a seeded bootstrap gives one SHA-256 digest with 1 and with 12 workers. Pipeline code must seed per replicate or per case, never per worker.
- **Lock ceilings.** The lock holds the seven v5-constrained packages to their tested release series (for example NumPy 2.4, SciPy 1.17, pyarrow 24), so the container and the host virtual environment compute with the same libraries. Lifting a ceiling before M0 needs a dated record; after M0, an amendment.
- **Coexistence with CXR-TRANS-AUDIT.** The projects share Docker Desktop, the WSL 2 VM (20 GB RAM, 16 processors) and the GPU. They use separate Compose projects, images and volumes. The CPU container never opens the GPU. CPU-heavy MRI jobs do not run while a CXR-TRANS-AUDIT GPU job runs unless `TMMRI_CPUS` is lowered, and that project's GPU-isolation check reports "not isolated" while any `tmmri` container runs (environment specification §9.5).
- **Paths.** Inside the container, Windows path limits and drive letters no longer reach the code. Legacy flows that write in place would write into the bind-mounted checkout, so protocol runs write only below `/workspace/artifacts/<run_id>` and `/workspace/data`.
- **Updates.** Docker Desktop, WSL, NVIDIA driver and Windows updates are applied only between milestones and followed by the qualification. Docker Desktop's subscription terms apply; the investigator confirms eligibility.
