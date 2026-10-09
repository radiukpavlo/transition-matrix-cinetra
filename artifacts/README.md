> Public edition of the registered file `artifacts/README.md` (protocol-v6.0.1, git blob `7236d3785b32`); operational details are replaced as listed in EDITIONS.md.

# Artifact runtime root

`artifacts/` is a git-ignored runtime root (ADR-0004). In the container runtime it is the Docker volume `tmmri_artifacts`, mounted at `/workspace/artifacts`; on the host it resolves to `TMMRI_ARTIFACT_ROOT` (default: this folder). Nothing written here is tracked; only this README is.

- **One folder per run.** A protocol-v6 run writes only below `artifacts/<run_id>/` (`tmmri.paths.run_root`). Run ids match `^[a-z0-9][a-z0-9._-]{2,80}$`; the policy is `v6-<purpose>-YYYYMMDD-HHMM`. A run root holds the stage folders (`contracts/`, `qc/`, `semantics/`, `formal/`, `transitions/`, `oof/`, `reconstructed/`, `thresholds/`, `granules/`, `rules/`, `inference/`, `equivariance/`, `metrics/`, `figures/`, `manuscript_tables/`), the stage records in `.stages/`, `artifact_status.json` and `release_manifest.json`.
- **Environment and benchmarks.** `environment/` and `bench/` hold the container qualification and timing records.
- **Legacy commands.** The v4/v5 commands still write their flat layout (`checkpoints/`, `models/`, `provenance/`, `releases/`, `reports/`, `rules/`, `transforms/`) directly below this root.

Results are released from a run root through its release manifest; they are never committed from here. The tracked v4/v5 outputs that used to live here are historical, pre-fix evidence in legacy/v5/artifacts/ (legacy/v5/README.md). Missing recorded `artifacts/...` paths fall back to those copies through `tmmri.paths.resolve_recorded_path`, while release and provenance writers never use that fallback.
