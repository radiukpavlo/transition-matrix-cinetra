> Public edition of the registered file `data/README.md` (protocol-v6.0.1, git blob `e6b1ad56cf18`); operational details are replaced as listed in EDITIONS.md.

# Data runtime root

`data/` is a git-ignored runtime root (ADR-0004). In the container runtime it is the Docker volume `tmmri_data`, mounted at `/workspace/data`; on the host it resolves to `TMMRI_DATA_ROOT` (default: this folder). Nothing written here is tracked.

- `raw/`: local raw ACDC and M&Ms release data. The payloads are ignored and must never be committed; only a few small release documents in `raw/` are tracked, and they stay where they are. Rebuild the payloads on any machine with `scripts/data/acquire_datasets.py`.
- `staged/<DATASET>/<case_id>/`: the verify-and-stage cache of protocol v6. Each source file is read once, verified by SHA-256 against its tracked manifest in [registry/sources/](../registry/sources/) and staged as ED/ES image and mask frames; `manifests/staging_<DATASET>.json` keys the cache by source SHA-256. Runs read the cache and never re-hash raw data.
- `sources/`: default mount point of the read-only source bindings (`TMMRI_SOURCES_ROOT`; `/sources` in the container).
- `features/`, `derived/`, `interim/`, `for_rules/`, `splits/`, `raw_registry/`: scratch outputs of the legacy v4/v5 commands (`run-minimal`, `validate-native-data`, `run-full-validation`). They are recreated on demand and never tracked.

Tracked evidence lives elsewhere: active source manifests in [registry/sources/](../registry/sources/) and the relocated v4/v5 tables in legacy/v5/data/, which are pre-fix and never v6 inputs (legacy/v5/README.md). Missing recorded `data/...` paths fall back to those tracked copies through `tmmri.paths.resolve_recorded_path`. Governed cohort handling is described in clinical_validation/README.md.
