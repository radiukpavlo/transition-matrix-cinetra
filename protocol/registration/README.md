# Protocol registration

CINETRA was evaluated under a registered protocol, CMR-TRANS-AUDIT. The authors registered it as two signed,
annotated git tags in their development repository:

| Tag | What it fixed | Signed (signer's clock) |
|---|---|---|
| `protocol-v6.0.0` | Cohorts, the formal representation A, the semantic matrix B, the transition, the rule layer, the metrics and the confirmatory hypotheses H1 and H2 | 29 September 2026, 22:46 +0300 |
| `protocol-v6.0.1` | The amendment: the M&Ms-2 hash split, the confirmatory hypothesis H3, the H4 family, the comparators D20, F9, F9b and F8, and the analyses S1–S6 | 3 October 2026, 17:31 +0300, before any M&Ms-2 image was extracted |

The development repository is private, because it also holds material outside this study, such as the governance
files of a planned reader benchmark. This folder lets anyone check the registration without it.

## Check it

```bash
python tools/verify_registration.py
```

The script needs Python 3.9 or later, OpenSSH `ssh-keygen` 8.0 or later (for the signatures) and, on a clone, git. It
checks four things:

1. **Signatures.** Each tag object in [tags/](tags/) hashes to its recorded id, and its SSH signature verifies
   against the key in [allowed_signers](allowed_signers).
2. **Chain.** Each tag points to a commit, and the commit and tree objects in [objects/](objects/) hash to their ids.
   A git object id is the SHA-1 of its content, so none of them can be altered without changing its id.
3. **Inclusion.** Walking those trees, each registered path resolves to the blob id recorded in
   [PROOFS.json](PROOFS.json).
4. **Identity.** Every file this repository publishes verbatim (code, tests, configs and registry definitions) has
   exactly that blob id, so it is byte-identical to the registered file.

## What it does not show

- **The dates** are the signer's own (the tagger line), not third-party timestamps. They are corroborated by the
  push history of the development repository, which editors and reviewers can inspect on request.
- **Public editions.** A few registered files contain operational details of the authors' workstation and accounts
  (local paths, a workspace name, storage locations, payment remarks). They are published as public editions, with
  only those details replaced; [EDITIONS.md](EDITIONS.md) lists every file and every kind of replacement. Their
  registered blob ids are proven, so anyone given an original file can check it with `git hash-object`.
- **Hash-only file.** One registered file is not published: the receipt of the legacy (v5) acquisition, which holds
  workstation paths and plays no part in the v6 pipeline. Its registered blob id is proven.
- **Dataset manifests.** The run-time manifests of the three dataset releases (file names, sizes, SHA-256 digests and
  the study's own eligibility, duplicate and subset flags) are published verbatim; they contain no image, mask,
  label, pathology or demographic value. [tools/verify_data.py](../../tools/verify_data.py) checks your downloads
  against them.
- **The trees reveal file names** of the development repository, but no file content beyond what is published here.

## The code of the frozen run

The frozen run `v6-replication-20261003-1241`, whose results this repository reports, records its code commit as
`d04f0908`, the commit of `protocol-v6.0.1`. The code in `src/`, `tests/` and `configs/` is exported from that commit,
so step 4 also shows that it is the code that produced the results.
