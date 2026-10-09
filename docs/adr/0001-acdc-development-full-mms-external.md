> Public edition of `docs/adr/0001-acdc-development-full-mms-external.md` of the authors' development repository (commit `2426d0b7d39c`); operational details are replaced as listed in EDITIONS.md.

# ADR-0001: ACDC development with the complete labelled M&Ms-1 release as the external cohort

**Status:** accepted, September 28, 2026 (pre-freeze; the use of the complete M&Ms dataset is the owner's instruction of that date; the exclusions, strata and evaluation protocol are pending owner confirmation at M0). *Amended September 29, 2026:* the owner confirmed the exclusions, strata and evaluation protocol, and added descriptive demographic strata (Q28; `protocol/amendments.md`). **Relates to:** Decisions D1, D2, D8, D16, D17 and D24 of [SPECIFICATION.md](../../SPECIFICATION.md) 6.0.0. **Supersedes:** the MMS_100 external cohort of the v5 native experiment (`v5_native_image_dss_100`) and the `prohibited_fallback` of `registry/sources/mms_100_acquisition_blocker.yaml`, for computational analyses only.

## Context

Protocol v5 fitted every object on ACDC and evaluated it externally on MMS_100, a stratified 100-case subset of M&Ms-1 selected one case per vendor, centre and pathology group. The same 100 cases became the frozen reader-study cohort of the governed clinical-validation package (protocol v2). The full M&Ms-1 OpenDataset release (345 cases, 691 files) is held locally and hash-verified. Of its cases, 320 are labelled and eligible; the 25 Training/Unlabeled cases carry placeholder frame indices and contain two voxel-identical duplicate pairs. On September 28, 2026 the owner instructed that the specification be remade with the complete M&Ms dataset. The v5 external volume and mass bias, which the manuscript attributed to cross-vendor shift, turned out to be largely the ACDC slice-increment defect fixed in commit `8ed5a40`. The v5 external evidence therefore cannot be reused.

## Decision

- **Development and internal test.** ACDC stays the only fitting source: the 100 official training patients form `acdc_dev` with five hash-assigned folds, and the 50 official testing patients form `acdc_test`, which is declared previously analysed. *Amended September 28, 2026:* by PI decision `patient055` (its ED and ES images equal those of the test patient `patient118`) and `patient076` (its images equal those of `patient074`; the lower identifier is kept) are excluded from `acdc_dev`, which has 98 patients; the folds are assigned on the 98 by the unchanged hash rule (20, 20, 20, 20 and 18 patients); `patient118` stays in `acdc_test` (SPECIFICATION.md §4.5, D15; `protocol/amendments.md`).
- **External cohort.** The complete labelled M&Ms-1 release, 320 cases (vendors A 95, B 125, C 50, D 50; centres 1–5), is the external cohort `mms_external`. It is evaluated once with the frozen ACDC bundle, after the protocol tag and the frozen-bundle record exist.
- **Exclusions.** The 25 Training/Unlabeled cases are excluded with `MMS_RELEASE_UNLABELED`.
- **Strata.** The external cohort is stratified by vendor, centre, release split, pathology and MMS_100 membership. The 220 never-analysed cases repeat the external primary test as a confirmatory sensitivity. The 100 previously analysed cases are reported separately as previously unblinded.
- **Refit arm.** A refit on M&Ms Training/Labeled (150), evaluated on Validation and Testing (170), is a required secondary arm that is never pooled with the primary arm. *Amended September 28, 2026:* `K3R0Y7`, whose images equal those of the training case `A7G0P5`, is excluded from `mms_refit_eval`, which has 169 cases; both cases stay in `mms_external` (320) as one bootstrap cluster, and so do `C8J7L5` and `C8O0P2` (identical ED images). Every bootstrap resamples clusters: `mms_external` has 320 cases in 318 clusters (D16, D17).
- **Unchanged.** The MMS_100 reader cohort and the governed clinical-validation package stay frozen under protocol v2. v6 never reads their restricted material or responses.

## Consequences

- The external test grows from 100 to 320 cases, and GE and Canon cases (vendors C and D, absent from M&Ms training) are represented by 50 cases each.
- Off-ontology pathologies (43 cases) and IHD (4) have no class-accuracy endpoint. Crosswalk-evaluable class accuracy uses 273 cases.
- Every M&Ms computation waits for M0. Only staging and hash verification may run earlier.
- BSA-indexed quantities are unsupported externally, because height is missing for vendors B and C.
- Staging reads each source file once and keeps only ED/ES frames, so full decompression (about 67 GB) is never needed.
- The tracked copy of the M&Ms information CSV and the legacy per-case MMS_100 tables must not reach a public mirror (M&Ms Research Use Agreement, clause 5).
