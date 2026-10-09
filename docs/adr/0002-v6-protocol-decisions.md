> Public edition of `docs/adr/0002-v6-protocol-decisions.md` of the authors' development repository (commit `2426d0b7d39c`); operational details are replaced as listed in EDITIONS.md.

# ADR-0002: CMR-TRANS-AUDIT protocol 6.0.0 decisions

**Status:** accepted, September 28, 2026 (pre-freeze; pending owner confirmation at M0). *Amended September 29, 2026:* the owner confirmed D5, D9, D12, D18, the exclusion of the 25 unlabelled cases (D1) and the SV-discrepancy severity, and decided D4 (`protocol/amendments.md`); the other decisions remain pending at M0. *Amended September 29, 2026 (later):* the owner confirmed every remaining decision, the environment decisions DEC-E1 to DEC-E5, and the new descriptive demographic strata (Q28); no decision is pending. **Relates to:** Decisions D1–D25 of [SPECIFICATION.md](../../SPECIFICATION.md) 6.0.0 and the decision record in its Appendix A. **Supersedes:** the implementation specifications `AGENTS_v1`–`AGENTS_v5` (now in `docs/history/implementation_specs/`) and the v5 native experiment design. ADR-0001 records the cohort decision; ADR-0003 records the container runtime; ADR-0004 records the repository refactor and legacy relocation.

## Context

The owner instructed on September 28, 2026 that the project be analysed and refactored completely, that the specification and documentation be remade with the complete M&Ms dataset, that future experiments be made efficient and effective, and that the structure of the CXR-TRANS-AUDIT protocol (`transition-matrix-x-ray`, `SPECIFICATION.md` 3.0.0) be followed. A coordinating agent session produced version 6.0.0 from three read-only audits (the X-ray template, the MRI method and code, and the local M&Ms release) and a binding design brief. No design interview took place. The audits found several mismatches between the v5 manuscript and the code:

- WEDD was described but training tertiles were implemented.
- Rough-set approximations were described but frequency rules were implemented.
- The rulebook was truncated at a 200-rule cap.
- Soft matching could emit labels inside evaluation.
- The λ objective drifted between runs.
- Missing-structure QC reasons defaulted to moderate.
- The ACDC slice-increment defect made every tracked ACDC volume and mass 5–10 times too small.

## Decisions

1. **Cohorts (D1).** ACDC 150 (development 100, internal test 50; *Amended September 28, 2026:* development 98 after the duplicate-image exclusions of item 15) and the complete labelled M&Ms-1 release, 320 cases, as the external cohort; the 25 unlabelled cases excluded (ADR-0001).
2. **Data access (D2).** Both releases held locally, verified against tracked SHA-256 manifests and mounted read-only; tracked manifests carry identifiers, release split, relative paths, sizes, SHA-256 and eligibility, exclusion, duplicate and MMS_100-membership flags only (vendor and centre are read from the mounted information CSV at run time).
3. **Compute (D3).** CPU-first on the Windows 11 workstation: per-case parallelism, one BLAS thread per worker, results independent of the worker count; the GPU serves only the optional deep-encoder branch.
4. **Venue (D4).** The MDPI class already used by the manuscript and a Q1/Q2 target; no journal is named yet, and the manuscript is remade only after v6 results exist. *Amended September 29, 2026:* the owner selected *Big Data and Cognitive Computing* (MDPI, ISSN 2504-2289); the manuscript uses the class option `bdcc`, and the target is the Special Issue *Data-Centric and Trustworthy AI for Biomedical Signal and Image Analysis* (deadline 31 January 2027; SPECIFICATION.md §18.1–18.3).
5. **Audited predictor (D5).** A frozen multinomial logistic regression on standardized formal A (L2, C = 1.0, five ACDC classes). Its decision `d_model` is audited; the measured-B classifier is a semantic comparator.
6. **Primary semantic family (D6).** Ventricular function and size (LV and RV EDV, ESV, EF; LV mass at ED). ED wall thickness and wall thickening are key secondary targets.
7. **Masks (D7).** Reference segmentations condition both A and B (a mask-conditioned audit); no automatic segmentation is evaluated.
8. **Ontology crosswalk (D8).** ACDC classes are the consequent vocabulary. NOR, DCM, HCM and ARV→RV map with the same definition; IHD→MINF is not exact and is reported separately; HHD, LVNC, AHS and Other are off-ontology.
9. **Formal representation (D9).** The 57 unprojected deterministic descriptors, standardized on the fit partition. The P6 128-d projection is a continuity comparator; semantic-proxy A is forbidden.
10. **Transition (D10).** Masked ridge with an unpenalized intercept and typed target transforms (`log`, `logit_percent`, `identity`). Per-target λ comes from inner 5-fold CV over 21 log-spaced values from 1e-6 to 1e4, with grid-edge flags. Comparators: identity-transform ridge, the pseudoinverse and the train mean.
11. **Discretization (D11).** `wedd_gaussian_mm_v1` with the 500-replicate stability gate; tertiles and the uniform kernel are sensitivities.
12. **Rule tier (D12).** Variable-precision rough sets, β = 0.10, with at least 5 anchor patients, no rule cap and deterministic ranking. Eligibility uses a cross-fitted gate over the five outer folds: at least 5 pooled matches and a Wilson lower bound ≥ 0.60.
13. **Rule consequent (D13).** `d_model` primary; the dataset-label rulebook is a companion and the measured-B oracle a third artifact.
14. **Inference routes (D14).** Seven partitioned routes with reason codes; only `EXACT_ACCEPT` emits a label; there is no conflict resolution and no base-decision tie-breaker.
15. **Splits (D15).** Five outer folds by rank of `SHA256("acdc-fold-v6|" + patient_id)` within class; inner CV inside each outer training set; the final bundle is refitted on all 100 development patients. *Amended September 28, 2026:* by PI decision the declared duplicate images `patient055` (`DUPLICATE_IMAGE_OF_TEST_CASE`) and `patient076` (`DUPLICATE_IMAGE_WITHIN_PARTITION`) are excluded from `acdc_dev`, so the folds and the final bundle use 98 patients (folds of 20, 20, 20, 20 and 18).
16. **External protocol (D16).** The frozen bundle is evaluated once on M&Ms-320 with strata. The 220 never-analysed cases form the confirmatory sensitivity stratum; the 100 previously analysed cases are declared previously unblinded. *Amended September 28, 2026:* the two M&Ms duplicate-image pairs stay in `mms_external`, each as one bootstrap cluster (320 cases, 318 clusters).
17. **Refit arm (D17).** An M&Ms refit ceiling is a required secondary arm: train on Training/Labeled, evaluate on Validation + Testing, report seen versus unseen vendors, never pool with the primary arm. *Amended September 28, 2026:* `K3R0Y7` (`DUPLICATE_IMAGE_OF_REFIT_TRAIN_CASE`) is excluded from the evaluation partition, which has 169 cases in 168 clusters.
18. **Primary estimand (D18).** The family-mean relative MAE reduction of ridge versus the train mean on fit-SD-normalized targets, with a 2,000-replicate paired bootstrap. *Amended September 28, 2026:* every patient- or case-level bootstrap, interval and cluster-level test resamples `cluster_id` clusters (a declared duplicate-image pair is one cluster, every other case its own). Fixed-sequence testing runs on ACDC internal test, then on M&Ms-320, with a smallest effect of interest of 0.10.
19. **Equivariance (D19).** Optional: ±1 px, ±2° and ±5 % intensity actions on 12 seeded ACDC development cases, reporting generator residuals and the finite-action defect; it never feeds the primary rulebook.
20. **Deep encoder (D20).** An optional frozen ResNet-50 (`IMAGENET1K_V2`) image-only branch in the GPU container target; not required for the primary claim.
21. **QC severity (D21).** Four severities. Missing structure is critical, and an unregistered reason fails closed.
22. **Runtime (D22).** A digest-pinned Linux container on Docker Desktop (WSL 2) running CPython 3.12.14 through uv with `uv.lock`; an optional GPU target serves D20 (ADR-0003).
23. **Efficiency (D23).** One-pass verify-and-stage into a content-addressed cache and a stage DAG with content-addressed stage records; runs never re-hash raw data.
24. **Reader-study firewall (D24).** The governed clinical-validation package and the MMS_100 reader cohort stay frozen under protocol v2; v6 never reads restricted material or responses.
25. **Freeze and file plan (D25).** A new root `SPECIFICATION.md` 6.0.0, frozen with the registry and configs by the signed tag `protocol-v6.0.0` at M0 (owner action) and changed afterwards only through `protocol/amendments.md`. Historical documents move to `docs/history/`, and v4/v5 outputs move to `legacy/v5/` as pre-fix evidence (ADR-0004).

## Consequences

- The registry defines 83 features (25 transition targets, 9 rule candidates) and 323 typed B-table columns. Block E (BSA and BSA-indexed values), block F (long axis, sphericity) and the 48 AHA-16 columns are excluded or unsupported.
- The fact corrections of SPECIFICATION.md Section 2.3 must not be reverted by copying older documents. This includes the reattribution of the v5 external volume and mass bias to the slice-increment defect.
- Every engineering detail marked [PROPOSED] in the specification, and every row of Appendix A sourced to the coordinator, becomes binding only when the owner confirms or amends it at M0. Amendments are logged in `protocol/amendments.md` before the tag.
- The manuscript still describes v5 and must be remade from v6 metrics.
- Nothing empirical exists yet. Every claim remains conditional on executing the frozen pipeline with the locally held data after M0.
