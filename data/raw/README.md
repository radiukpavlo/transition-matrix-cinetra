# Raw datasets (not included)

This folder is bound read-only into the containers (`compose.yaml`). Put your own copies of the datasets here, or
point `.env` to them:

- ACDC: `data/raw/acdc/database/` (the default of `TMMRI_ACDC_SOURCE`; `python scripts/data/acquire_datasets.py acdc`
  downloads it here and checks every file against the registered manifest);
- M&Ms-1: anywhere, with `TMMRI_MMS_SOURCE` set in `.env`;
- M&Ms-2: the archive `MnM2.rar` anywhere; `extract-mms2` writes the 720 extracted files to `data/raw/mms2/`.

Nothing in this folder is ever committed. See [docs/REPRODUCTION.md](../../docs/REPRODUCTION.md).
