# Public editions and hash-only files

The release publishes most registered files byte for byte (their git blob ids equal those of the registered commit
`d04f0908e66766335609c938f06d51c53421a45a`, tag `protocol-v6.0.1`). The files below are the exceptions.

## Public editions

These files contain operational details of the original workstation and accounts. In the public editions, those
details are replaced; nothing else changes. The kinds of replacement are: local paths, payment remarks, storage locations, workspace names, and links to
documents that are not part of this release. Each edition starts with a notice line (except JSON and code files).
The registered blob of each file is proven to sit in the registered commit, so anyone given the original file can
check it with `git hash-object`.

| File | Registered blob | Replaced |
|---|---|---|
| `SPECIFICATION.md` | `bc26520059284a4a62f20dad11d17747bd80abbd` | links to unreleased documents (1); local paths (9); storage locations (2) |
| `protocol/amendments.md` | `21159e502e7585863e4b53ffbefa7c9d5f0627bc` | links to unreleased documents (1); local paths (5); payment remarks (2); storage locations (5); workspace names (1) |
| `protocol/neural_encoder_v6.0.1.design.json` | `2dc485bbc67afc39a61c3d5c3730d5d067a300b8` | storage locations (1); workspace names (1) |
| `configs/cohorts.yaml` | `8a2920739495acd9067aafee6fcc04fee4a84435` | local paths (1) |
| `configs/f9_cnn.yaml` | `c575500d8062946e1eb6ac344631d381dac9fc16` | payment remarks (1); workspace names (2) |
| `configs/study_v6.yaml` | `6eca666768adfb8fddf2f3c61a899d23a1c88d8b` | local paths (1) |
| `configs/source_bindings/mms.json` | `f87aa67c4370663d691ead68e98d9011c2710116` | local paths (1) |
| `configs/source_bindings/mms2.json` | `bd53d8672a66dab9c5541bd47d3e7432ac43bd4e` | local paths (2); storage locations (1) |
| `registry/sources/mms2.yaml` | `70e7b7196c0fd37592d831519212d686eb8eddf3` | local paths (2); storage locations (1) |
| `configs/README.md` | `ee7031684f4dc0f66675c4ea8ca85a7239144a0a` | links to unreleased documents (5) |
| `data/README.md` | `e6b1ad56cf18579bdbf07e15ffff337def281002` | links to unreleased documents (4) |
| `artifacts/README.md` | `7236d3785b326cc95ae72c304558922116b10246` | links to unreleased documents (3) |
| `scripts/cloud/README.md` | `8fd2878ac4cd37d75bb901503c577d8ef0fff6cb` | links to unreleased documents (2); workspace names (1) |
| `scripts/data/build_mms_full_release_manifest.py` | `53f3a512f00f753e38b4f80d1882fa0d3eeebb68` | local paths (2) |
| `.env.example` | `6df9adb4cfeec3b55869519a88964d753f2b5b5b` | local paths (2) |
| `registry/sources/mms_full_release_receipt.yaml` | `11727ee97713f2a4b8ae9e22568e6b0ced3e3de4` | local paths (2) |

## Development documents

These documents are not part of the registration; they record decisions and the executed timeline. They are taken
from the authors' development repository, with the same kinds of replacement.

| File | Replaced |
|---|---|
| `docs/V601_RESULTS_FREEZE_2026-10-03.md` | links to unreleased documents (1); local paths (2); payment remarks (1) |
| `docs/NEURAL_ENCODER_EXPERIMENT_PROTOCOL.md` | links to unreleased documents (3); storage locations (1) |
| `docs/adr/0001-acdc-development-full-mms-external.md` | links to unreleased documents (1) |
| `docs/adr/0002-v6-protocol-decisions.md` | links to unreleased documents (1) |
| `docs/adr/0003-docker-desktop-linux-container-runtime.md` | links to unreleased documents (2); local paths (3) |

## Hash-only files

These registered files are not published: they record the legacy (v5) acquisition on the original workstation and
play no part in the v6 pipeline. Their blob ids are proven to sit in the registered commit.

| File | Registered blob |
|---|---|
| `registry/sources/acdc_native.yaml` | `0e460af9585ab024d6a0b0b1accf96bf87073343` |
