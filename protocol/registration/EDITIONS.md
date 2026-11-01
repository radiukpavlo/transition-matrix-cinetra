# Public editions and hash-only files

The release publishes most registered files byte for byte (their git blob ids equal those of the registered commit
`e9d6ccdf8bef8deb9deabf72613abd025691c1df`, tag `protocol-v6.0.5`). The files below are the exceptions.

## Public editions

These files contain operational details of the original workstation and accounts. In the public editions, those
details are replaced; nothing else changes. The kinds of replacement are: local paths, payment remarks, storage locations, workspace names, and links to
documents that are not part of this release. Each edition starts with a notice line (except JSON and code files).
The registered blob of each file is proven to sit in the registered commit, so anyone given the original file can
check it with `git hash-object`.

| File | Registered blob | Replaced |
|---|---|---|
| `SPECIFICATION.md` | `0cd698578fa17d130f4ab0de0c04e362e2ad3552` | local paths (9); storage locations (2) |
| `protocol/amendments.md` | `93f779d32f20a987e0c462488025e31a7f2e4d28` | local paths (5); payment remarks (2); storage locations (5); workspace names (1) |
| `protocol/neural_encoder_v6.0.1.design.json` | `2dc485bbc67afc39a61c3d5c3730d5d067a300b8` | storage locations (1); workspace names (1) |
| `configs/cohorts.yaml` | `8a2920739495acd9067aafee6fcc04fee4a84435` | local paths (1) |
| `configs/f9_cnn.yaml` | `c575500d8062946e1eb6ac344631d381dac9fc16` | payment remarks (1); workspace names (2) |
| `configs/study_v6.yaml` | `6eca666768adfb8fddf2f3c61a899d23a1c88d8b` | local paths (1) |
| `configs/source_bindings/mms.json` | `f87aa67c4370663d691ead68e98d9011c2710116` | local paths (1) |
| `configs/source_bindings/mms2.json` | `bd53d8672a66dab9c5541bd47d3e7432ac43bd4e` | local paths (2); storage locations (1) |
| `registry/sources/mms2.yaml` | `70e7b7196c0fd37592d831519212d686eb8eddf3` | local paths (2); storage locations (1) |
| `configs/README.md` | `1f38164912bc579b0b15e0c2f400a79a23f8258e` | links to unreleased documents (4) |
| `data/README.md` | `e6b1ad56cf18579bdbf07e15ffff337def281002` | links to unreleased documents (3) |
| `artifacts/README.md` | `7236d3785b326cc95ae72c304558922116b10246` | links to unreleased documents (2) |
| `scripts/cloud/README.md` | `8fd2878ac4cd37d75bb901503c577d8ef0fff6cb` | links to unreleased documents (1); workspace names (1) |
| `scripts/data/build_mms_full_release_manifest.py` | `53f3a512f00f753e38b4f80d1882fa0d3eeebb68` | local paths (2) |
| `.env.example` | `6df9adb4cfeec3b55869519a88964d753f2b5b5b` | local paths (2) |
| `registry/sources/mms_full_release_receipt.yaml` | `11727ee97713f2a4b8ae9e22568e6b0ced3e3de4` | local paths (2) |

## Development documents

These documents are not part of the registration; they record decisions and the executed timeline. They are taken
from the authors' development repository, with the same kinds of replacement.

| File | Replaced |
|---|---|
| `docs/V601_RESULTS_FREEZE_2026-10-03.md` | local paths (2); payment remarks (1) |
| `docs/NEURAL_ENCODER_EXPERIMENT_PROTOCOL.md` | links to unreleased documents (2); storage locations (1) |
| `docs/adr/0001-acdc-development-full-mms-external.md` | notice line only |
| `docs/adr/0002-v6-protocol-decisions.md` | notice line only |
| `docs/adr/0003-docker-desktop-linux-container-runtime.md` | links to unreleased documents (1); local paths (3) |
| `docs/public_release_supplement.md` | notice line only |

## Evidence packs

The aggregate tables and figures of the post-freeze exploratory blocks are published byte for byte, except two
files per pack: the README, a public edition with the same kinds of replacement (its links point at their place in
this release or, for documents outside it, become plain text; it starts with a notice line), and `SHA256SUMS`, whose
README line carries the digest of the public edition. The table gives the SHA-256 of each original file.

| File | Original SHA-256 | Replaced |
|---|---|---|
| `results/v6.0.2/tables/README.md` | `e5df8fd956665c7ab23b7d243263353d801bfb2699a893545b00a5e4bb0ba466` | links to unreleased documents (3) |
| `results/v6.0.2/tables/SHA256SUMS` | `33d5ece6629beacb51d2eb0922c1c3600d5a9ef506c341a2d929969f0c85ac98` | digest of the README edition (1) |
| `figures/v6.0.2/README.md` | `2f40752fca0d29c65123523600ed53ed70eedd77b614b5f835a4f2de9e6ecf55` | links pointed at their place in this release (1); links to unreleased documents (7); local paths (1) |
| `figures/v6.0.2/SHA256SUMS` | `4d84525fb90a25929ffb9939e95511fb15387fe27e6543a910154d18d291b61a` | digest of the README edition (1) |
| `results/v6.0.3/tables/README.md` | `a942ed24f37853aa6ed3fcc0cc6f816f5ac64f1131c396842a38f45349caeffa` | links pointed at their place in this release (1); links to unreleased documents (3) |
| `results/v6.0.3/tables/SHA256SUMS` | `db094aad4e925e98b01bfac608cfc5673f36321ae5a9dc750807cb94712fe336` | digest of the README edition (1) |
| `results/v6.0.4/tables/README.md` | `59e9061fb0334efb4b8b24e4210998da6a63bd84ff8e562d1404083e023a4777` | links pointed at their place in this release (2); links to unreleased documents (3) |
| `results/v6.0.4/tables/SHA256SUMS` | `e623eedacd75cdd591b99f0eb411a136b108b530528a292cc40a64474a0aa495` | digest of the README edition (1) |

## Hash-only files

These registered files are not published: they record the legacy (v5) acquisition on the original workstation and
play no part in the v6 pipeline. Their blob ids are proven to sit in the registered commit.

| File | Registered blob |
|---|---|
| `registry/sources/acdc_native.yaml` | `0e460af9585ab024d6a0b0b1accf96bf87073343` |
