> Public edition of the registered file `SPECIFICATION.md` (protocol-v6.0.1, git blob `bc2652005928`); operational details are replaced as listed in EDITIONS.md.

# CMR-TRANS-AUDIT: unified implementation specification

## 0. Document control and non-negotiable execution rules

**Specification version:** 6.0.1 (amendment of 6.0.0; pending the owner's signature). **Decision dates:** September 28, 2026 (6.0.0); October 2, 2026 (the owner decisions registered by 6.0.1, in the amendment commit dated October 3, 2026). **Status:** authoritative research protocol. Version 6.0.0 is frozen by the annotated tag `protocol-v6.0.0`, and its primary run `v6-primary-20260929-2307` is recorded; those results stand as recorded. Version 6.0.1 keeps every 6.0.0 value and adds the registered items of Section 0.3. The additions bind only when the owner signs the annotated tag `protocol-v6.0.1` on the amendment commit; until then no M&Ms-2 byte is read. **Supersedes:** the implementation specifications `AGENTS_v1`–`AGENTS_v5` (kept as historical documents in `docs/history/implementation_specs/`), the v5 native experiment design (`v5_native_image_dss_100`) and the MMS_100 selection rule *for computational analyses*. Where any passage in those documents, in `docs/history/IMPLEMENTATION_PLAN_repository_restructure_v1.md`, elsewhere in `docs/history/` or in the current manuscript conflicts with this document, this document controls the computational research design. **Not superseded:** the clinical-validation protocol v2 (`clinical_validation/protocol/protocol_v2.md`, `statistical_analysis_plan_v2.md`) and its frozen MMS_100 reader cohort are unchanged; they are outside this protocol's analyses (Decision D24). **Intended users:** the principal investigator (PI, the repository owner) and research-engineering agents implementing the study end to end. **Intended use:** retrospective, research-only auditing of a frozen formal-representation predictor on cardiac cine MRI and of symbolic explanations derived from it; not diagnosis, triage, treatment selection, segmentation or patient management.

**Provenance of this version.** Version 6.0.0 was produced on September 28, 2026 by a coordinating agent session acting on the owner's instruction to analyse and completely refactor the project, to remake the specification and documentation with the **complete M&Ms dataset**, to make future computational experiments as efficient and effective as possible, and to adhere to the structure and specification of the CXR-TRANS-AUDIT protocol (`<local path>` 3.0.0). The session worked from three read-only audits of September 28, 2026 (the chest X-ray template, the current MRI method and code, and the full local M&Ms-1 release), a binding design brief, and a primary-source check of the dataset and standards citations. No design interview with the PI took place. The use of the complete labelled M&Ms-1 release, the Docker Desktop realization and the X-ray structure are owner instructions; **every other decision in the ledger is a coordinator proposal pending owner confirmation at milestone M0**. *Amended September 29, 2026:* the owner confirmed every proposal of Appendix A, with decisions on D4 (journal and Special Issue) and on the duplicate-image pairs (Q27), and adopted Q28 (`protocol/amendments.md`). Appendix A records each decision with its source. *Amended October 3, 2026 (version 6.0.1, pending the owner's signature):* an amendment agent session recorded, as additions only, the owner decisions of October 2, 2026 in manuscript protocol 2.1 (`docs/MANUSCRIPT_PROTOCOL_V2_2026-10-02.md`, Sections C, D, E, H, I, J and O) and roadmap item A1 (`docs/COMPUTATIONAL_ROADMAP.md`). They cover the once-only wording, the M&Ms-2 cohort and H-primary-3, the registered comparators F8, D20, F9 and F9b, the registered secondary analyses S1–S6, the exploratory register B1–B8, exposure labels, the freeze order and data governance (Section 0.3; `protocol/amendments.md`). No 6.0.0 value changes, and nothing in this version is an owner signature or approval. *Amended October 3, 2026 (owner issues O1 and O3):* D20 and F9 are made invariant to the in-plane dihedral group D4 (Sections 8.4 and 8.6) and the exploratory analysis B9 joins the register (Section 13.6). This is the one change to a definition frozen in 6.0.0 (the D20 patient pooling); it enters `configs/deep_encoder.yaml` as the amendment-scoped key `tta`, so the v6.0.0 section digest of that file is unchanged.

**Labels used throughout.** **[INHERITED]** marks a mechanism supported by the six supplied papers; **[VERIFIED]** marks a publicly documented fact confirmed against a primary source on September 28, 2026 (a date suffix is added where the verification was made in the X-ray protocol instead); **[PROPOSED]** marks the cardiac MRI adaptation specified here; **[BLOCKED]** marks an artifact that requires unavailable data, permissions or compute. **[PRESPEC]** marks a numerical setting that must be copied into `configs/` and frozen before any held-out evaluation; such values are engineering settings, not validated clinical cutoffs and not promised results. Unmarked engineering prescriptions are proposed decisions. **[LOCAL]** marks a fact measured on the locally held data by the read-only audits of September 28, 2026; it is a data property, not a result.

**Cohorts.** The two image cohorts are **ACDC** (150 patients: 98 of the 100 official training patients form the development cohort, two being excluded as declared duplicate images (Section 4.5), and the 50 official testing patients the internal test cohort) and the **complete labelled M&Ms-1 OpenDataset release** (320 cases: Training/Labeled 150, Validation 34, Testing 136; vendors A Siemens 95, B Philips 125, C GE 50, D Canon 50; centres 1–5), which is the external cohort. The 25 Training/Unlabeled cases are excluded (Section 4.5). The frozen 100-case MMS_100 subset is not a separate cohort in v6: it is a declared stratum (`mms100_member`) of the external cohort, previously analysed under v5 and therefore previously unblinded. M&Ms-2, Sunnybrook, the Kaggle Data Science Bowl, UK Biobank, CMRxRecon and CMRxMotion may not be added. ImageNet initialization in the optional deep-encoder branch (Decision D20) is general-purpose model initialization, not a cardiac cohort. *Amended in 6.0.1 (Decision D27):* one half of the M&Ms-2 release, chosen by a hash rule fixed before any of its bytes are read, is added as the prospective external cohort `mms2_external`; the other half, `mms2_sealed`, is never extracted under v6 (Section 4.6). The other resources stay excluded. The ImageNet initialization of the F9 comparator (ResNet-18 `IMAGENET1K_V1`, Section 8.6) is likewise model initialization.

**Agents MUST** preserve source labels, identifiers, frame indices, geometry metadata, reference masks, partition boundaries, quality-control states and explicit failures. **Agents MUST NOT** manufacture patient data, semantic values, masks, rules, clinical thresholds, external compatibility, access permissions, physician responses, approvals, signatures or results; recruit clinicians; open, import, analyse or package reader-study responses or anything under `clinical_validation/restricted/`; modify any file under `clinical_validation/`; compute B, A, reconstructions, rules or metrics for any M&Ms case before M0 (staging and hash verification are allowed); present an automatic segmentation as a reference standard; or represent existing public annotations as a new clinical validation. A technically valid empty rulebook is an acceptable outcome. An ineligible feature, class, rule or metric is reported as such with its failing criterion, never silently dropped or relaxed.

**Reading convention.** `[P1]`–`[P6]` identify the six supplied methodological papers (Section 19.3, with SHA-256 of the exact reviewed files). `[D…]`, `[S…]`, `[J…]` and `[M…]` identify dataset, standards, journal and implementation sources (Sections 19.4–19.7). Physical PDF page numbers include appended supplements even when printed numbering restarts. Citations to supplied author manuscripts (P3 as supplied, P5, P6) do not establish that the supplied text equals a published version. P6 is this project's own XAI-CRI article; its LaTeX source is `manuscript/article/main.tex`, which still describes the v5 results.

**Execution boundary.** Both datasets are held locally and are hash-verifiable. ACDC: 1,056 release files under `data/raw/acdc/database` (all listed with SHA-256 in the tracked ACDC release manifest). M&Ms-1: the full OpenDataset release, 345 cases and 691 files including the information CSV, at `<local path>`, with SHA-256 for every file in `<local path>` [LOCAL]. Sources are mounted read-only into a digest-pinned Linux container on Docker Desktop's WSL 2 back end (Decision D22; ADR-0003; `docs/EXECUTION_ENVIRONMENT_SPEC.md` 1.0.0). No protocol-v6 `B`, `A`, transition, rulebook or metric exists yet; every such artifact is `NOT_GENERATED` (Section 19.2). Before M0, computing any endpoint on an M&Ms case is forbidden; synthetic data and ACDC-development timing probes are allowed for benchmarks and tests. Tracked v4/v5 outputs are legacy evidence (`legacy/v5/`), were computed before the ACDC slice-increment correction, and are never v6 inputs. *Amended in 6.0.1:* the M&Ms-2 release archive is held unopened until the `protocol-v6.0.1` tag (Section 4.6). Derived `acdc_dev` training tensors and the F9 and S5 development fits may be processed in the owner's Modal workspace under cloud-processing exception CPE-1; every evaluation runs locally (Section 14.7).

**Freeze policy.** At milestone M0 this file, `registry/semantic_registry.yaml`, `registry/reason_codes.yaml`, `registry/pattern_templates.yaml`, `registry/estimator_partition_registry.csv`, the active cohort manifests under `registry/sources/` and every v6 file under `configs/` (all except `configs/clinical/`, which the clinical-validation protocol v2 governs, and `configs/legacy/`, which serves legacy flows only) are frozen by an annotated, signed git tag `protocol-v6.0.0`; the tag's commit hash and the SHA-256 of each frozen file are copied into every downstream manifest. No external preregistration service is used (Decision D25). Any later change to a frozen value requires a dated entry in `protocol/amendments.md`, a new tag `protocol-v6.0.x`, and, if any held-out or external partition has already been unblinded, an explicit statement that the change is post hoc. *Amended in 6.0.1:* the annotated, signed tag `protocol-v6.0.1` on the amendment commit freezes this file at version 6.0.1, the `protocol-v6.0.0` files with the additions of Section 0.3, and the new 6.0.1 files under `configs/`, `registry/` and `registry/sources/` (listed in `protocol/amendments.md`). Only the owner creates it. The tag precedes every M&Ms-2 byte (Section 4.6) and every 6.0.1 bundle freeze (Section 14.6). It does not activate the neural study NEURAL-TM-1.0, which is planned under `protocol-v7.0.0`.

**Pre-freeze revision (September 28, 2026).** Two corrections precede the freeze and are recorded here and in `protocol/amendments.md`. (1) **ACDC slice increment.** All 150 ACDC ED/ES frame files carry unit-scaled affines (column norms 1, 1, 1; `sform_code` 2) while the header `pixdim` gives the true spacing (slice spacing 10 mm in 127 patients, 5 mm in 21, 6.5 mm in 1, 7 mm in 1) [LOCAL]. The geometry reader took the slice increment from the affine, so every tracked ACDC volume and mass used 1 mm. Commit `8ed5a40` makes the reader use the header `pixdim` whenever the affine is not voxel-scaled (`affine_is_unscaled`). M&Ms affines are voxel-scaled in all 345 cases and are unaffected. No tracked table was regenerated. (2) **Legacy relocation.** Tracked v4/v5 outputs (`artifacts/**`, `data/{derived,features,for_rules,interim,splits}/**`) are relocated with `git mv` to `legacy/v5/` and declared pre-fix; `legacy/v5/README.md` states their status (ADR-0004). No binding value of an earlier protocol is carried into v6 by these moves, and no v6 partition had been unblinded.

### 0.1 How to read this document

| If you are… | Start with… |
|---|---|
| deciding whether the design is sound | Sections 0.2, 1, 3, 17, Appendix A |
| implementing staging, adapters and the reference tables **B** | Sections 4, 6, 7, 12 |
| implementing the formal matrix **A** and the audited predictor | Section 8 |
| implementing the transition **T**, WEDD, rough sets and routes | Sections 9, 10, 11 |
| implementing evaluation, statistics, the refit arm and ablations | Sections 12.3, 13 |
| building the container, repository, tests and release | Sections 14, 15, 16; `docs/EXECUTION_ENVIRONMENT_SPEC.md`; `docs/EXPERIMENT_EXECUTION_PLAN.md` |
| writing the manuscript | Sections 2.3, 16, 17, 18 |
| implementing the version 6.0.1 additions | Sections 0.3, 4.6, 8.6, 13.5, 13.6, 14.6, 14.7, 15.1, 15.2 (tests 25–33) |

Conventions: **A**, **B**, **T**, **S** denote matrices; `n` is the number of aligned analysis objects (rows); `k` the formal dimension; `l` the number of numeric semantic targets. All sample tables are row-oriented (Section 9.1). File and folder names are literal. Equations are numbered (E1), (E2), … within this document; source-paper equation numbers are quoted as "P4 Eq. (6)". Canonical mask labels are 0 background, 1 LV cavity, 2 LV myocardium, 3 RV cavity (the M&Ms convention; ACDC labels 1 and 3 are swapped on load, Section 7.3).

### 0.2 Decision ledger (binding; every row confirmed or decided by the owner by September 29, 2026)

Source of every row: owner instruction (2026-09-28) for D1 (complete M&Ms), D22 (Docker Desktop) and D25 (structure and file plan); owner decision (2026-09-29) for D4 (journal and Special Issue); owner confirmation (2026-09-28 for the duplicate-image pairs; 2026-09-29 for every other value) of the coordinator's proposals (Appendix A; `protocol/amendments.md`).

| ID | Decision | Binding value | Where implemented |
|---|---|---|---|
| D1 | Cohort scope | ACDC 150 (development 98 of the 100 training patients + internal test 50; two duplicate-image exclusions, §4.5) and the complete labelled M&Ms-1 release, 320 cases (Training/Labeled 150, Validation 34, Testing 136; vendors A 95, B 125, C 50, D 50; centres 1–5) as the external cohort; the 25 Training/Unlabeled cases are excluded (`MMS_RELEASE_UNLABELED`) | §4.3, §4.5, §12.2; `configs/cohorts.yaml`; `registry/sources/mms_full_release_manifest.csv` |
| D2 | Data access | Both datasets held locally and hash-verifiable (ACDC 1,056 files; M&Ms 345 cases, 691 files, SHA-256 in `BACKUP_MANIFEST.sha256`); ACDC CC BY-NC-SA 4.0 with mandatory citation; M&Ms Research Use Agreement (personal research copy; no redistribution of any portion); sources mounted read-only; tracked manifests carry identifiers, release split, relative paths, sizes, SHA-256 and eligibility/membership flags only (vendor and centre come from the mounted information CSV) | §4.4, §7.3, §14.5; `configs/source_bindings/{acdc,mms}.json`; `configs/cohorts.yaml` |
| D3 | Compute | CPU-first: one Windows 11 workstation, 16 logical CPUs, 32 GB RAM (WSL 2 VM about 20 GB), GTX 1080 8 GB unused by the primary pipeline; per-case parallelism, one BLAS thread per worker, results independent of worker count | §15.3; `configs/study_v6.yaml` (`runtime`); `configs/runtime.env`; `docs/EXPERIMENT_EXECUTION_PLAN.md` |
| D4 | Venue | *Big Data and Cognitive Computing* (MDPI; ISSN 2504-2289), Special Issue *Data-Centric and Trustworthy AI for Biomedical Signal and Image Analysis* (submission deadline 31 January 2027), owner decisions of September 29, 2026; `manuscript/article/main.tex` uses the MDPI class option `bdcc`; manuscript remade only after v6 results | §18; `configs/study_v6.yaml` (`venue`) |
| D5 | Audited predictor | Frozen multinomial logistic regression on standardized formal A (classes DCM, HCM, MINF, NOR, RV), fitted on ACDC development; its decision `d_model` is audited; the measured-B logistic classifier is a semantic comparator | §8.2; `configs/formal.yaml` (`audited_predictor`) |
| D6 | Primary semantic family | Ventricular function and size: `lv_edv_ml`, `lv_esv_ml`, `lv_ef_percent`, `rv_edv_ml`, `rv_esv_ml`, `rv_ef_percent`, `lv_myocardial_mass_ed_g`; key secondary `global_wall_thickness_ed_mm`, `global_wall_thickening_percent`; degradation of absolute volumes and mass under cross-vendor shift is a prespecified hypothesis, never fixed post hoc | §3.3, §6.2; `registry/semantic_registry.yaml` (`primary_target_set`, `key_secondary_targets`) |
| D7 | Masks | Reference segmentations condition both A and B in the primary branch: a mask-conditioned audit; no automatic segmentation is evaluated or claimed | §1.1, §7, §8.1, §17 |
| D8 | Ontology crosswalk | ACDC classes are the consequent vocabulary; M&Ms NOR→NOR, DCM→DCM, HCM→HCM, ARV→RV (`HARMONIZABLE_SAME_DEFINITION`); IHD→MINF (`HARMONIZABLE_NOT_EXACT`, reported separately); HHD, LVNC, AHS, Other → `OFF_ONTOLOGY`; crosswalk-evaluable external n = 273 (+4 IHD) | §12.3; `registry/semantic_registry.yaml` (`label_tables.mms_to_acdc`); `configs/cohorts.yaml`; `registry/reason_codes.yaml` |
| D9 | Formal representation A | `formal_descriptors_v3`: the 57 unprojected deterministic descriptors of `native_deterministic_baseline_v2`, standardized on the fit partition only; continuity comparator: the P6 128-d Gaussian projection (seed 20260611, first 32 raw retained); semantic-proxy A forbidden | §8.1; `configs/formal.yaml` (`primary`, `comparators`) |
| D10 | Transition T | Ridge with fit-only preprocessing; typed target transforms (`log`, `logit_percent`, `identity`) then fit-SD standardization; per-target λ by inner 5-fold CV on 21 log-spaced values 1e-6…1e4; grid-edge selections flagged; identity-transform ridge sensitivity; pseudoinverse historical baseline; train-mean null comparator | §9; `configs/transition.yaml` |
| D11 | Discretization | `wedd_gaussian_mm_v1` [INHERITED P4] fitted on the fit partition with the P5 500-replicate stability gate; sensitivities `three_quantile_v1` (P6/v5 tertiles) and `wedd_uniform_mm_v1` | §11.3, §11.4; `configs/wedd.yaml`; `configs/bootstrap.yaml` |
| D12 | Rule tier | Variable-precision rough sets, β = 0.10: granule confidence ≥ 0.90, ≥ 5 anchor patients; confidence-1 rules flagged `STRICT`; no rule cap; deterministic ranking; cross-fitted gate over the 5 outer folds: pooled out-of-fold matched anchors ≥ 5 and Wilson 95 % lower bound of agreement ≥ 0.60 | §11.5, §11.6; `configs/rule_policy.yaml` |
| D13 | Rule consequent | `d_model` primary; dataset-label rulebook companion; measured-B oracle rulebook third | §11.1; `configs/rule_policy.yaml` |
| D14 | Inference routes | `EXACT_ACCEPT`, `CONFLICT_ABSTAIN`, `SOFT_CANDIDATES_ONLY` (never a label), `UNAVAILABLE_FEATURE_ABSTAIN`, `NO_MATCH_ABSTAIN`, `QC_ABSTAIN`, `EMPTY_RULEBOOK`; reason codes in `registry/reason_codes.yaml`; `REFERENCE_OFF_ONTOLOGY` evaluation-only; no conflict resolution; base decision never a tie-breaker | §11.7; `registry/reason_codes.yaml`; `configs/rule_policy.yaml` |
| D15 | Splits | ACDC development = the 100 official training patients minus the declared duplicate-image exclusions `patient055` (`DUPLICATE_IMAGE_OF_TEST_CASE`) and `patient076` (`DUPLICATE_IMAGE_WITHIN_PARTITION`) = **98** (DCM 20, HCM 20, MINF 19, NOR 19, RV 20); 5 outer folds assigned on the 98 eligible patients by rank of `SHA256("acdc-fold-v6|" + patient_id)` within class (folds 1–4: 20 patients, 4 per class; fold 5: 18, with 3 MINF and 3 NOR); λ by inner CV inside each outer training set; final bundle refit on all 98; ACDC internal test = 50 official testing patients (`patient118` kept); unit = patient (ACDC) or case (M&Ms). *Amended September 28, 2026:* PI decision on the four duplicate-image pairs (§4.5; `protocol/amendments.md`) | §12.1, §12.2; `configs/splits.yaml`; `registry/estimator_partition_registry.csv` |
| D16 | External protocol | M&Ms-320 evaluated once with the frozen ACDC bundle; no refit, threshold or QC-reference change; strata vendor, centre, release split, pathology, `mms100_member`; confirmatory sensitivity stratum = the 220 never-analysed cases; the 100 previously analysed cases reported separately and declared previously unblinded. *Amended September 28, 2026:* both members of the duplicate-image pairs A7G0P5/K3R0Y7 and C8J7L5/C8O0P2 stay in `mms_external`, each pair as one bootstrap cluster: 320 cases in 318 clusters (never-analysed 220 in 220; `mms100_member` 100 in 99) | §12.3; `configs/splits.yaml`; `configs/cohorts.yaml` |
| D17 | Secondary arm | M&Ms refit ceiling (required secondary, never pooled): the same pipeline fitted on Training/Labeled (150, vendors A/B) with inner CV only, evaluated on Validation + Testing minus K3R0Y7 (**169**; `DUPLICATE_IMAGE_OF_REFIT_TRAIN_CASE`, its images equal those of the training case A7G0P5), seen vendors (A, B; 70) versus unseen (C, D; 99). *Amended September 28, 2026:* K3R0Y7 excluded from `mms_refit_eval`; 169 cases in 168 clusters | §12.3; `configs/splits.yaml`; `configs/bootstrap.yaml` |
| D18 | Primary estimand | ρ = 1 − MAE_ridge / MAE_mean on fit-SD-normalized primary-family targets (family mean; paired cluster bootstrap over `cluster_id` (a declared duplicate-image pair is one cluster, every other patient or case its own), 2,000 replicates, percentile 95 % CI); fixed-sequence testing: H-primary-1 on ACDC internal test (CI lower bound > 0.10, the smallest effect of interest), then H-primary-2 on M&Ms-320; the 220-case stratum repeats H-primary-2 as sensitivity | §3.3, §13.1, §13.3; `configs/bootstrap.yaml` (`primary_test`) |
| D19 | Equivariance / ETM | Optional [P3]: ±1 px translation, ±2° rotation, ±5 % intensity on 12 ACDC development cases selected with seed 20260611; generator residuals and finite-action defect; M&Ms only after M0; never feeds the primary rulebook | §10; `configs/equivariance.yaml` |
| D20 | Deep-encoder branch | Optional: frozen image-only ResNet-50 `IMAGENET1K_V2`, 2D short-axis slices at ED/ES, 2048-d penultimate pooled per patient, as alternative A and alternative audited predictor; GPU container target; not required; tests skip until implemented | §8.4; `configs/deep_encoder.yaml`; `src/tmmri/deep_encoder.py` (pending) |
| D21 | QC severity [P6] | critical (excluded from fitting; `QC_ABSTAIN` at inference), high (feature-level masking), moderate (reported), informational; missing-structure reasons are critical | §12.4; `configs/cohorts.yaml` (`qc_severity`); `registry/reason_codes.yaml` |
| D22 | Runtime | Linux container on Docker Desktop (WSL 2); Ubuntu 24.04 and uv 0.12.18 digest-pinned; CPython 3.12.14 via uv; `uv.lock`; CPU target `env` default; optional `env-gpu` with torch 2.14.0+cu126 and torchvision 0.29.0+cu126 for D20 | §14.2, §15.3; `Dockerfile`; `compose.yaml`; `configs/runtime.env`; ADR-0003; `docs/EXECUTION_ENVIRONMENT_SPEC.md` |
| D23 | Efficiency | One-pass verify-and-stage into a content-addressed staging cache; runs never re-hash raw data; stage DAG with content-addressed stage records; per-case parallelism; measured time model | §7.4, §14.2, §15.3; `configs/study_v6.yaml`; `src/tmmri/ingest/staging.py`; `src/tmmri/runs.py`; `src/tmmri/stages.py` |
| D24 | Reader-study firewall | The governed clinical-validation package and the MMS_100 reader cohort stay frozen under protocol v2; v6 never reads restricted material or responses; v6 results are not reader-study results | §0, §1.5, §12.3; `configs/study_v6.yaml` (`reader_study_firewall`); `clinical_validation/README.md` |
| D25 | Freeze and file plan | Signed tag `protocol-v6.0.0` at M0 (owner action); amendments log; `SPECIFICATION.md` new at root; historical documents in `docs/history/`; v4/v5 outputs in `legacy/v5/`, pre-fix and never v6 inputs | §0, §14.1; `protocol/amendments.md`; ADR-0004 |

Seeds [PRESPEC]: `primary_model` 20260611; folds by hash rule (no seed); `bootstrap` 20260928; `synthetic_controls` 20260613; `equivariance_selection` 20260611. Bootstrap counts [PRESPEC]: fixed-model 2,000; WEDD stability 500; downstream refit 200. Multiplicity: Benjamini–Hochberg q = 0.05 within the prespecified families F1–F5 (Section 13.3); merged ranking forbidden.

### 0.3 Version 6.0.1 additions (owner decisions of October 2, 2026; binding when the owner signs `protocol-v6.0.1`)

Source of every row: the owner decisions of October 2, 2026 recorded in manuscript protocol 2.1 (`docs/MANUSCRIPT_PROTOCOL_V2_2026-10-02.md`, Section O), and roadmap item A1 (`docs/COMPUTATIONAL_ROADMAP.md`, Phase A); the dated row of October 3, 2026 in `protocol/amendments.md`. No row changes a value of Section 0.2. Configuration files and stage names are those of the 6.0.1 implementation; the file list is finalized at integration.

| ID | Decision | Binding value | Where implemented |
|---|---|---|---|
| D26 | Once-only wording (roadmap A1) | In a frozen run (`.stages/FROZEN.json` present) a stage that already has a record may only be skipped. If its inputs, code version, configuration hash, parameters or recorded outputs differ, or `--force` is given, it refuses with exit 7; a new run id is required. The wording now matches the guard of commit `b7a579b`; no behaviour changes | §14.2; §15.1; §15.2 test 23; `src/tmmri/stages.py` (`once_only_guard`) |
| D27 | Prospective external cohort | M&Ms-2 release `MnM2.rar` (9,824,397,247 bytes; SHA-256 in §4.6). Subjects `001`–`360` are ranked by `SHA256("mms2-split-v6.0.1\|" + subject_id)`: the first 180 form `mms2_external` (minus duplicate-image exclusions), the other 180 form `mms2_sealed` (never extracted under v6). Short-axis ED/ES images and masks only; identity labels; the M&Ms-2 lock (exit 3 before `protocol-v6.0.1`) | §4.6, §8.5, §12.2; `configs/cohorts.yaml` and `configs/splits.yaml` (`mms2` sections); `configs/source_bindings/mms2.json`; `registry/sources/mms2*` |
| D28 | H-primary-3 | On `mms2_external` the unchanged v6.0.0 bundle has ρ (E1) with a 95 % cluster-bootstrap lower bound > 0.10. It is its own one-test family, outside the fixed sequence of D18. Per-target secondary tests form family F6 | §3.3, §3.4, §12.3, §13.3; `configs/bootstrap.yaml` (`primary_test`; values unchanged) |
| D29 | Registered comparators | F8 CART surrogate; D20 executed as frozen (status implemented) except the D4 symmetrization of its patient feature (owner issue O1, October 3, 2026); F9 trained CNN (NEURAL-TM-1.0 §4.1–4.2, 342 fits) with the D4 augmentation and test-time symmetrization of §8.6; F9b concept probe | §8.4, §8.6; `configs/cart_f8.yaml`; `configs/deep_encoder.yaml`; `configs/neural_input.yaml`; `configs/f9_cnn.yaml`; `configs/probes.yaml`; `registry/weights.yaml` |
| D30 | Registered secondary analyses | S1 selective prediction; S2 label-free OOD detection; S3 label-free harmonization; S4 calibration transfer; S5 data-efficiency curves; S6 simulated acquisition shift | §13.5; `configs/trust.yaml` (S1–S4); `configs/data_centric.yaml` (S5, S6) |
| D31 | Statistical status | Confirmatory: H-primary-1 and H-primary-2 (recorded), H-primary-3, and the H4 family (H4a and H4b; Decision D36) only. Benjamini–Hochberg families F1–F5 and F6. Estimation only (paired intervals, no significance language) for F8, D20, F9, F9b and S1–S6. Exploratory register B1–B9 (B9 added on October 3, 2026, owner issue O1), labelled `EXPLORATORY_POST_HOC`. No analysis outside this table may be called registered | §13.3, §13.6; `configs/exploratory.yaml` |
| D32 | Exposure labels | `mms2_external`: prospective. `acdc_test` and `mms_external`: every analysis first registered or first evaluated in 6.0.1 is "registered after unblinding". `acdc_dev` out-of-fold results: development | §12.2 |
| D33 | Freeze order and once-only locks | The tag, then the F8, D20, F9 and F9b bundles and the S1, S2 and S4 parameters frozen with their hashes recorded before M&Ms-2 extraction; each partition evaluated once per bundle; then the results freeze | §14.6 |
| D34 | Data governance | Cloud-processing exception CPE-1 and cloud-storage exception CSE-1 (`protocol/amendments.md`, October 2, 2026) apply unchanged. Modal holds only the `acdc_dev` tensor cache and the F9 and S5 development fits, under one pinned GPU type, IEEE FP32, the upload ledger and local inference for every reported number. The Modal client is pinned through `uvx` (`scripts/cloud/README.md`), so `uv.lock` and the images are unchanged | §14.7 |
| D35 | Claim boundary | No explanation of the CNN's decisions is claimed. No rules are fitted on CNN decisions; that work is deferred to protocol v7 (NEURAL-TM-1.0). F9b measures linear decodability of concepts only | §1.5, §17 |
| D36 | Second confirmatory family H4 (owner risk management, October 3, 2026) | **Rationale:** H-primary-3 is likely to fail (M&Ms-1: ρ 0.192, 95 % interval [0.032, 0.298]), yet on M&Ms-1 the per-target `ρ_j` of LVEF (0.63) and RVEF (0.50) transferred while the absolute volumes did not. H4 turns these historically generated hypotheses into prospective tests on `mms2_external`, declared before any M&Ms-2 byte is read. **H4a (ratio concepts transfer):** `ρ_EF`, the estimand (E1) of the unchanged v6.0.0 bundle computed with the H-primary-3 code and restricted to the ejection-fraction targets of the registry primary family (`lv_ef_percent`, `rv_ef_percent`); it passes when the two-sided 97.5 % cluster-bootstrap lower bound exceeds 0.10; both EF targets must be estimable, otherwise NE. **H4b (label-free harmonization helps):** Δρ = ρ(S3 variant H-a: A re-standardized with the mean and SD of the unlabelled target partition; T, λ and the predictor frozen) − ρ(frozen bundle), over the primary family on the same rows, with a paired cluster bootstrap; it passes when the two-sided 97.5 % lower bound exceeds 0. **Multiplicity:** Bonferroni within the family: two tests, each judged by its two-sided 97.5 % interval (α = 0.025 per test, family-wise α = 0.05); the family is separate from H-primary-3 and from F6, with no further adjustment. **Bootstrap:** 2,000 replicates, seed 20260928 with a stable sub-seed per hypothesis (the first 4 bytes, big-endian, of `SHA256("20260928\|" + id)`: H4a 1958159060, H4b 1591327988), the cluster unit and the NE rules of Section 3.5. **Scope:** confirmatory only on `mms2_external`; the same code runs descriptively on `acdc_test` and `mms_external`, labelled "registered after unblinding" | §3.3, §3.4, §13.3; `configs/cohorts.yaml` (`mms2.endpoints.h4`); `evaluate-h4`; `src/tmmri/confirmatory_h4.py` |

Seeds and salts [PRESPEC] added by 6.0.1: F9 seeds 20261011 (primary), 20261012 and 20261013 (distinct from the NEURAL-TM-1.0 seeds 20261001–20261003); F8 `random_state` 20260928; bootstrap seed 20260928 with 2,000 replicates for every 6.0.1 interval (the H4 sub-seeds of D36 derive from it); salts `f9-select-v6.0.1|` (F9 selection folds), `lc-v6.0.1|` (S5 subsamples) and `mms2-split-v6.0.1|` (M&Ms-2 split). Multiplicity: Benjamini–Hochberg q = 0.05 within F6 (Section 13.3), separately from F1–F5; Bonferroni within the confirmatory H4 family (D36).

---

## 1. Executive research concept and novelty claim

### 1.1 Research concept

Develop a **mask-conditioned post-hoc audit** of a frozen formal-representation predictor on cardiac cine MRI. The formal representation A is a fixed set of deterministic image-and-mask descriptors of the end-diastolic (ED) and end-systolic (ES) short-axis frames; the audited predictor is a frozen multinomial classifier on A, fitted on ACDC development patients. After freezing the predictor, reconstruct standardized CMR quantities from A by an inspectable affine transition, discretize the reconstruction into symbolic states, induce variable-precision rules that explain the predictor's decision, gate them by cross-fitting, and test the unchanged bundle once on the complete labelled M&Ms-1 release:

`formal A (image and mask descriptors) -> transition T -> reconstructed CMR semantics (volumes, function, mass, wall) -> WEDD states -> variable-precision rules explaining d_model -> accepted consequent or reason-coded abstention`.

The question is **which parts of the predictor's behaviour can be reconstructed in standardized, reference-supported CMR terms, and which parts cease to be verifiable under vendor, centre and pathology shift?** The reference masks are expert annotations released with each dataset; they condition both A and the reference semantics B (Decision D7). The audit therefore does not test a segmentation model and says nothing about automatic contouring. Because A is unitless and not resampled, absolute volumes and mass are expected to degrade across vendors with different fields of view, matrices and spacings, while ratios such as ejection fraction may transfer better; both are hypotheses to test (Section 3.4). [D1–D3; P6]

The predictor and its explanations are separate outputs. A predicted class can remain available while the audit abstains. A reconstructed semantic can be evaluated even when the rulebook is empty. Measured reference features are never substituted for reconstructed features at inference.

### 1.2 Central research question

> Can a frozen, mask-conditioned formal representation of cardiac cine MRI be transitioned, by a leakage-controlled affine operator, into standardized ventricular volumes, function and mass with quantified per-target fidelity in physical units, and does the reconstructed description support a global, cross-fitted, conflict-aware and abstention-capable rulebook whose coverage, covered fidelity to the audited predictor, conflict rate and abstention rate remain interpretable when the frozen bundle meets four scanner vendors and five centres it never saw?

### 1.3 Contribution to claim, conditional on the results

The intended contribution is an **evidence-typed, two-cohort CMR audit** that connects a frozen representation to standardized CMR quantities, quantifies reconstruction separately from classification and rule fidelity, admits only cross-fitted rules, and reports vendor-, centre- and pathology-specific failure honestly, including the part of the external gap that a same-cohort refit can and cannot close (Decision D17). The methodological emphasis is the typed semantic contract, the corrected geometry chain, the leakage-controlled estimation, the once-only external evaluation with a never-analysed replication stratum, and the reproducible evidence chain.

Do not claim invention of the transition matrix, ridge regression, Lie-algebra regularization, entropy-density discretization or rough sets. Those mechanisms are inherited [P1–P6]. A publishable advance must be shown through the implemented semantic contract, the controlled comparisons, the external failure analysis and the reproducible evidence chain.

### 1.4 Primary and secondary outputs

**Primary output.** Patient-level reconstruction of the ventricular function-and-size family on the ACDC internal test set, then (only if that passes) on the M&Ms-320 external cohort, summarized by the estimand of Section 3.3 and judged against the smallest effect of interest (Decision D18).

**Key secondary output.** Reconstruction of `global_wall_thickness_ed_mm` and `global_wall_thickening_percent` with the same metrics.

**Required secondary outputs.** Per-target reconstruction for every eligible target; the frozen primary rulebook or its explicit empty result with accepted coverage, covered fidelity, conflict and abstention by reason; audited-predictor performance internally and on crosswalk-evaluable external cases; the label-companion and measured-B oracle rulebooks; external strata by vendor, centre, release split, pathology and MMS_100 membership; the never-analysed 220-case replication; the M&Ms refit ceiling (Decision D17); discretizer, transform and representation ablations; the affine-versus-nonlinear comparison under a frozen criterion.

**Optional branches.** Equivariance and ETM (Decision D19); the deep-encoder branch (Decision D20). They have separate artifact paths and no power to change the primary claim.

**Outside scope.** Clinical utility, physician preference, diagnostic safety, patient outcomes, workflow benefit, causal explanation, automatic segmentation quality and any reader-study endpoint.

### 1.5 Intended use and non-claims

- Outputs are audit artifacts: matrices, transition weights, per-target fidelity tables, rulebooks, route statistics. They are not diagnostic outputs and not segmentation outputs.
- Three vocabularies are kept apart and never conflated: the **dataset label** (ACDC `Group`; M&Ms `Pathology`), the **imaging measurement** (EDV, EF, mass, wall thickness computed from reference masks) and the **clinical diagnosis** (never claimed). An ACDC class is a dataset label assigned by the dataset's own criteria, not a diagnosis established here.
- The reader study is a separate governed package (clinical-validation protocol v2). v6 computations never read its restricted material or responses, and no v6 metric is a reader-study result (Decision D24).
- Transition weights are fitted associations under the development distribution, not causal or physiological statements [P6, Appendix C].
- A reason-coded abstention states that no eligible rule evidence verifies the case; it is not evidence of clinically safe deferral [P5 §3.5].
- *Amended in 6.0.1 (Decision D35):* D20 and F9 are comparators, and F9b is a linear concept probe of F9's activations. None of them explains the CNN's decisions, and no rule is fitted on CNN decisions in v6 (Section 17).

---

## 2. Methodological lineage from all six papers

### 2.1 Source-by-source reconstruction

| Source and reviewed artifact | Scientific objective, inputs, A and B | Estimation and downstream mechanisms | Evidence boundary and what this study inherits |
|---|---|---|---|
| **P1**, *Mathematics* 2024;12:1024, 32 pp. | Relate formal deep-model features to an understandable representation on the same objects (MNIST, FNC-1, Iris). | Operationally `AT ≈ B` with `T ≈ A⁺B` (Eqs. 5–9), no centering, regularization or intercept; equations printed in column style while computed in row form. | Inherit aligned object rows and feature-to-feature reconstruction; the raw pseudoinverse is only the historical baseline (it interpolates fitted rows exactly when under-determined). Do not inherit "change of basis" wording or subjective trust as validation. |
| **P2**, *Front. Artif. Intell.* 2024;7:1482141, 13 pp. | Build B from expert-defined measurable features; one application is cardiac MRI (m = 100, k = 1,024, l = 20), an under-determined fit. | Same pseudoinverse estimator; six admissible sources of feature values; four selection criteria; class-level Cohen's κ. | Inherit the registry as design object and measurable definitions with units; replace expert consensus by standards-based grounding (Section 5); the under-determined MRI fit motivates a regularized estimator. Do not inherit class-level κ as the only fidelity measure. |
| **P3**, ETM, supplied author manuscript (37 pp.); published *Mach. Learn. Knowl. Extr.* 2026;8(4):92 | Formal-to-mental map that approximately intertwines a Lie-group action; executed for SO(2) rotation on MNIST. | Finite-difference generators; objective with a commutator penalty whose stacked form squares the weight; SVD or LSQR without materializing Kronecker products. | Inherit an optional symmetry experiment only with actual image actions and declared semantic actions (Section 10); penalty bookkeeping and step sizes are project-defined. |
| **P4**, SEMTRA, *Mach. Learn. Knowl. Extr.* 2026;8(7):181, 43 pp. | Frozen representation → continuous semantics → symbolic audit. | Ridge with one global α; WEDD with Gaussian KDE, entropy base N_c, candidate-wise min–max normalization, weight 0.65; greedy constrained reducts; conflicts resolved by score and soft matches emit labels. | Inherit the staged pipeline, target-explicit metrics, the WEDD profile and the audit-tax framing. Depart: no label on conflict or soft match, explicit route partition, per-target λ, a stability gate, a maximum rule length. |
| **P5**, CARDIA-X, supplied manuscript (60 pp.) | ECG audit with a 52-target typed contract and an external branch. | Typed transforms; masked per-target ridge with one global λ; WEDD with a 500-replicate stability gate; variable-precision tier β = 0.10; estimator-to-partition registry; replay boundary; frozen nonlinear criterion met but affine pipeline retained. | Inherit typed units, masks, the stability gate, the variable-precision tier, the empty-result discipline, the registries, the replay boundary and the linearity criterion. Numeric class gates, reason codes and per-target λ are project definitions. |
| **P6**, XAI-CRI, supplied September 2, 2026 manuscript (28 pp.); source `manuscript/article/main.tex` in this repository | Cardiac MRI: ACDC development, 100-case M&Ms external; deterministic image/mask descriptors as A (not a trained neural latent), measured CMR quantities as B. | `B̂ = AT`, ridge with λ by internal CV; training-only median imputation, 1st/99th percentile clipping, z-scaling of A; NRMSE by evaluation-split SD; four QC severities with feature-level exclusion; WEDD described but tertiles implemented; a 200-rule book induced on measured B; perturbation diagnostics; synthetic 10-d → 6-d → 3-rule control with seed 20260613; SHA-256 release manifest. | Direct precedent. Inherit row alignment by `sample_id`, the deterministic descriptors of A, QC severities with feature-level exclusion, external freezing, unsupported-predicate handling and the release manifest. Correct its method/implementation mismatches and its geometry defect (Sections 2.2–2.3); its external volume and mass findings are not precedent. |

### 2.2 Reconciliation table: never silently combine these definitions

| Issue | Source definitions or discrepancy | Binding decision for this project | Status |
|---|---|---|---|
| Matrix orientation | P1/P2 print `B = TA` yet compute `AT ≈ B`; P3 uses `T ∈ ℝ^{l×k}`; P4–P6 use `T ∈ ℝ^{k×l}`. | Row convention everywhere: `A ∈ ℝ^{n×k}`, `B ∈ ℝ^{n×l}`, `T ∈ ℝ^{k×l}`, predictions `AT + intercept`. | Explicit dimensional repair |
| Intercept | P4 double-centers if applied literally; P5 fixes the intercept at zero although masks differ per target; v5 centred B only. | Unpenalized intercept per target on that target's observed fitting rows (Section 9.2); one formulation, never both corrections. | Explicit correction |
| Regularization selection | P4 one global α; P5 one global λ = 100; P6 internal CV; v5 one λ for all 28 targets (λ = 3; v4 λ = 100 at the grid edge). | Per-target λ by inner CV within each outer training set; a single global λ is a sensitivity; grid-edge selections are flagged (D10). | Project extension |
| λ objective drift (v5) | The tracked v4/v5 and baseline CV tables used a raw mixed-unit RMSE objective; later code used per-target RMSE divided by the training-fold SD. | Per-target selection by inner-validation MAE in physical units (Section 9.3); legacy λ values are never reused. | Correction |
| Target transforms | P4 min–max; P5 typed transforms; v5 none (median imputation and 1st/99th clipping of B in physical units, centring only). | `log` for strictly positive volumes, mass and areas, `logit_percent` for EF, `identity` otherwise; fit-SD standardization; no imputation of B (per-target observation masks); fitting targets winsorized at their 1st/99th fit percentiles in transformed space, evaluation against unclipped references (D10; `configs/transition.yaml`). | Project choice |
| Normalized error | P6 Eq. (13) and v5 `normalized_rmse_by_split_sd` (misnamed `train_sd`) use the evaluated split's SD. | Fit-partition SD for every split; evaluation-SD normalization is a named sensitivity. | Explicit choice |
| Discretization (v5) | P6 and the manuscript describe WEDD; `rules.py` implements training tertiles labelled `training_quantile_tertiles_wedd_approximation`; a real WEDD exists only in a synthetic manuscript script. | Primary `wedd_gaussian_mm_v1`; tertiles become the named sensitivity `three_quantile_v1` (D11). | Correction |
| Rule induction (v5) | The manuscript claims rough-set approximations and reducts; v5 counts singles, pairs and triples among the first six features in column order, filters support ≥ 3 and confidence ≥ 0.60. | Variable-precision rough sets with explicit lower and upper approximations, greedy constrained reduction and order-independent candidate generation (Section 11.5). | Correction |
| Rule cap (v5) | `max_rules = 200`; v4 and v5 both hit the cap, so rulebooks were truncated by sort order. | No arbitrary cap; deterministic ranking; every candidate and its gate outcome stored (D12). | Correction |
| Substrate (v5) | v5 rules were induced on measured B. | Primary rules use reconstructed OOF B̂; measured-B rules form the oracle rulebook with separate paths and thresholds (D13). | No mixing |
| Soft matching (v5) | `evaluate_rulebook` used `allow_soft = True`, so a rule with one mismatching antecedent could emit a label inside evaluation. | Soft candidates never emit a label (`SOFT_CANDIDATES_ONLY`, D14). | Correction |
| Conflict handling | P4 resolves conflicts by score; v5 scored by Σ confidence × support with a 0.95 confidence and 0.50 margin rule; the clinical package uses a margin rule. | Conflicting exact consequents route to `CONFLICT_ABSTAIN`; no conflict resolution; the base decision is never a tie-breaker (D14). | Conservative adaptation |
| Model-selection wording (v5) | The manuscript states that validation data were used for model selection; λ was actually chosen by CV inside the training split. | v6 has no validation-for-selection partition: every selection is by CV inside `acdc_dev` and is stated as such. | Wording correction |
| WEDD kernel | P4 Gaussian KDE, base N_c, min–max; P5 uniform kernel, base 2; P6 unnormalized raw entropy plus density. | `wedd_gaussian_mm_v1` exactly as Section 11.3.1 (copied from the CXR-TRANS-AUDIT profile); `wedd_uniform_mm_v1` sensitivity; P6's unnormalized form not used. | Selected implementation (D11) |
| Rule purity | P4 lower approximations are pure but its rules satisfy only τ = 0.84; P5 defines β = 0.10. | Primary tier variable precision β = 0.10; confidence-1 rules flagged `STRICT`; no certainty wording for non-strict rules. | Separated definitions (D12) |
| Rule validation | v5 had no independent rule gate; the development cohort (98 patients) cannot spare a gate partition. | Cross-fitted gate over the five outer folds (Section 11.6). | Project definition |
| QC defaults (v5) | Missing-structure reasons were absent from the severity map and defaulted to moderate; outlier references used every row including test and external rows. | Missing structure is critical; outlier references fitted on `acdc_dev` only (D21, D16). | Correction |
| Predicate similarity (P6, v5) | P6 Eq. (9) penalized similarity; v5 direction `normal` never matched state `mid`, and the score ignored the rule's label. | Not used. CMR pattern templates are descriptive, without a numeric alignment score (Section 11.8). | Not adopted |
| Linear interpretability | P5 nonlinear criterion met but affine retained. | Frozen criterion (Section 9.4); the executed audit remains affine regardless of the outcome. | Restricted inheritance |
| External evidence | P5 fits a separate external transition; P6 and v5 froze ACDC and applied it to 100 M&Ms cases. | The complete labelled release (320) is evaluated once with the frozen ACDC bundle; a separately fitted M&Ms transition exists only as the declared refit arm (D16, D17). | P6-style boundary, widened |

### 2.3 Fact corrections applied relative to v5, the manuscript and earlier project documents

| Item | Earlier statement | Corrected statement (source) |
|---|---|---|
| ACDC slice spacing | Tracked ACDC B computed with the affine slice increment | All 150 ACDC frame files have unit-scaled affines; the tracked B used dz = 1 mm for every patient, so every ACDC volume, stroke volume, myocardial volume, mass and long-axis value is about 5–10 times too small (tracked mean LVEDV 18.7 mL; the M&Ms-100 mean is 150.8 mL). Fixed in commit `8ed5a40`; nothing tracked regenerated [LOCAL] |
| M&Ms volume and mass bias | The manuscript attributes the "large negative volume and mass bias" on M&Ms to cross-centre and cross-vendor shift (`main.tex`, Discussion) | The v5 external LVEDV bias of −136 mL (RMSE 144.7 mL) is largely the ACDC slice-spacing defect: the transition was fitted on ACDC targets that were too small. The shift component is unknown until v6 runs [LOCAL] |
| Affected targets | — | 11 of the 28 v5 targets are affected (LV and RV EDV, ESV, SV; myocardial volume ED and ES; mass; SV discrepancy; long axis). The QC flag for SV discrepancy > 15 mL changes from 1 to about 79 of 150 ACDC patients after the fix, and robust outlier flags change [LOCAL] |
| Discretization | "WEDD" (manuscript §2.1) | Training tertiles in the implementation (see §2.2) |
| Rough sets | "Lower/upper approximations and reducts" (manuscript) | Frequency and association counting; no approximations or reducts |
| Rulebook size | "200 rules" | Truncated at the cap of 200, not a natural rulebook size |
| External cohort size | "100-case M&Ms external" (P6) | The v5 external set was a stratified 100-case subset; v6 uses the complete labelled release, 320 cases [D4–D6] |
| M&Ms release versus paper | "375 subjects" (challenge paper) | The challenge paper reports 375 subjects [D5, **VERIFIED**]; the OpenDataset release held locally has 345 cases (320 labelled + 25 unlabelled). The 30 missing cases are all vendor A: validation A has 4 and testing A 16 cases, against 10 and 40 for every other vendor [LOCAL]. This is consistent with a balanced 10/40 design per vendor; it was not re-verified against the paper text |
| MMS_100 selection seed | Seed 20260611 recorded for the MMS_100 selection | The seed has no effect; the selection is a deterministic round-robin over sorted (vendor, centre, pathology, SHA-256(case_id)) groups (`splits.py`) [LOCAL] |
| Equivariance cases | "12 sampled cases" | Selected with `head()` per split, not at random; configured epsilon keys were unread. v6 selects with seed 20260611 (D19) |
| BSA-indexed targets | Implicitly excluded | The substring `index` in the exclusion list silently removed every BSA-indexed column from the v5 target set; v6 excludes them explicitly, with a registered reason (height is absent for M&Ms vendors B and C), in registry block E (Section 6.4) |
| Full-package prohibition | `mms_100_acquisition_blocker.yaml`, `docs/history/experiment_machine_v5/datasets.md` and `docs/history/experiment_machine_v5/transfer_archive.md` forbid the full 345-case package (bodies kept unchanged as historical records; `docs/history/README.md` states the supersession) | Superseded by the owner's instruction of September 28, 2026 to use the complete M&Ms dataset; recorded as a dated pre-freeze row in `protocol/amendments.md` |
| Hash reproducibility | Legacy manifests verify on the source machine | Several legacy hashes depend on CRLF worktree line endings (`scripts/maintenance/crlf_worktree_paths.txt`); the v5 release manifest has 4 pre-existing hash mismatches [LOCAL]. v6 uses LF (`.gitattributes`) |

### 2.4 Feature-construction lessons carried forward

P1 shows that the mental representation can be physical measurements rather than another embedding. P2 makes the registry the central design object and gives the cardiac MRI example of an under-determined fit. P3 requires that the extraction function be rerun on actually transformed inputs. P4 shows that compact rules can have far lower fidelity than the predictor. P5 adds typed transforms, observation masks, threshold stability and the replay boundary. P6 adds native-image provenance, per-feature QC and explicit unsupported predicates, and, through its implementation history, the lesson that geometry must be verified at the header level: an affine that looks valid can be unit-scaled.

For cardiac MRI these become: A computed from verified native ED/ES frames and reference masks without resampling; B computed by one audited measurement code path for both cohorts; spacing taken from a verified geometry rule with the method recorded per case; BSA-indexed quantities only where anthropometrics exist; and no rule promoted because a clinically plausible pattern exists.

---

## 3. Research questions, hypotheses, and estimands

### 3.1 Endpoints and target separation

**Reconstruction endpoints.** The reference semantics B are measured from the released reference masks at the ED and ES frames with verified geometry (Section 7). The primary family is ventricular function and size (Decision D6); the key secondary targets are ED wall thickness and wall thickening. Every other eligible target is secondary.

**Audited decision.** `d_model(x) = argmax_c p_c(x)` of the frozen multinomial logistic regression on standardized A (Decision D5), with classes {DCM, HCM, MINF, NOR, RV}. The primary rulebook explains `d_model`; it never explains a dataset label.

**Dataset labels.** `Y_acdc` is the ACDC `Group` from `Info.cfg` (DCM, HCM, MINF, NOR, RV). `Y_mms` is the M&Ms `Pathology` from the information CSV, mapped to the ACDC vocabulary by the frozen crosswalk of Section 12.3; off-ontology labels have no class-accuracy endpoint. Labels are decision attributes and stratifiers; they never enter A or B.

### 3.2 Prespecified questions

| Question | Hypothesis to test, not assume | Estimand and population |
|---|---|---|
| RQ1: semantic recoverability | Ridge on formal A reconstructs the primary family better than the development mean by at least the smallest effect of interest. | ρ of Section 3.3 on `acdc_test` patients with valid references (H-primary-1). |
| RQ2: external transport | The frozen bundle keeps a reconstruction gain on the complete external cohort. | ρ on `mms_external` (H-primary-2, only if H-primary-1 passes); per target, vendor, centre, release split and pathology. |
| RQ3: unitless representation | Because A is unitless and not resampled, unit-free targets (EF) transfer better than absolute volumes and mass across vendors. | Paired difference of external per-target ρ between the EF targets and the absolute targets of the primary family. |
| RQ4: vendor shift | Vendors C (GE) and D (Canon), whose matrices, fields of view and spacings differ most from ACDC, degrade more than A (Siemens) and B (Philips). | Difference in family-mean normalized MAE between vendors C ∪ D and A ∪ B on `mms_external`. |
| RQ5: replication | The never-analysed 220 cases replicate the conclusion on all 320. | H-primary-2 repeated on `mms_external_never_analysed`; the previously analysed 100 are reported separately. |
| RQ6: symbolic audit | Some predictor behaviour admits compact, cross-fitted rules over reconstructed semantics. | Accepted coverage, covered fidelity to `d_model`, conflict and abstention by reason on `acdc_test` and `mms_external`; covered accuracy on crosswalk-evaluable cases. |
| RQ7: development limitation versus vendor shift | Refitting on M&Ms vendors A/B closes part of the external gap on seen vendors but less on unseen vendors. | Paired ρ of the refit arm versus the frozen ACDC bundle on `mms_refit_eval`, seen (A, B) versus unseen (C, D) (Decision D17). |
| RQ8: adequacy of linearity | The affine bridge may be competitive with a small nonlinear mapper, but need not be. | Frozen criterion of Section 9.4 on identical rows, targets, A and masks. |
| RQ9: discretization | WEDD may improve threshold stability or the coverage–fidelity trade-off relative to tertiles and the uniform kernel. | Paired differences under identical rule gates. |
| RQ10 (optional): equivariance | Small in-plane actions leave reconstructed volumetric semantics nearly invariant, and rotation is more disruptive than whole-pixel translation. | Finite-action defect and generator residuals (Section 10). |
| RQ11 (optional): deep encoder | An image-only frozen CNN representation reconstructs the primary family less or more faithfully than the mask-conditioned A. | The same estimand with the D20 A (Section 8.4). *Amended in 6.0.1:* executed as a registered comparator, estimation only (Section 8.6). |
| RQ12 (6.0.1): prospective transport | The unchanged v6.0.0 bundle keeps a reconstruction gain on a cohort that no analysis has touched. | ρ on `mms2_external` (H-primary-3, Section 3.3); per target (F6), vendor, centre and pathology strata (Section 12.3). |
| RQ13 (6.0.1): representations | How the mask-conditioned descriptors compare with frozen (D20) and trained (F9) image representations, as concept sources and as classifiers. | ρ and `ρ_j` per representation (descriptors, D20, F9b); classifier metrics per predictor (audited, D20, F9); paired differences; estimation only (Section 8.6). |
| RQ14 (6.0.1): trust under shift | Whether abstention thresholds, OOD flags and temperatures fitted in development still hold under shift. | S1, S2 and S4 (Section 13.5); estimation only. |
| RQ15 (6.0.1): data-centric mechanisms | Whether label-free harmonization recovers ρ, whether acquisition geometry alone reproduces the external drop, and which pipeline is more data-efficient at n ≤ 98. | S3, S6 and S5 (Section 13.5); estimation only. |

### 3.3 Primary estimand, smallest effect of interest and test [PRESPEC]

**Primary target set (Decision D6):** `lv_edv_ml`, `lv_esv_ml`, `lv_ef_percent`, `rv_edv_ml`, `rv_esv_ml`, `rv_ef_percent`, `lv_myocardial_mass_ed_g`, measured on reference masks as defined in Section 6.2. Their eligibility is frozen from development data only (Section 11.2); an ineligible target stays in the family with its failure recorded and contributes no term.

**Estimand (Decision D18).** For each target `j` of the family `F`, compute on the evaluation patients with an observed reference the mean absolute error of ridge (`MAE_ridge,j`) and of the train-mean null predictor (`MAE_mean,j`, the constant fit-partition mean of the target in physical units), both in physical units after the inverse transform and both divided by the fit-partition standard deviation `s_j` of the target. The per-target relative reduction and the family estimand are

\[
\rho_j = 1 - \frac{MAE_{ridge,j}/s_j}{MAE_{mean,j}/s_j}, \qquad
\rho = \operatorname{mean}_{j \in F,\ \mathrm{estimable}}\ \rho_j. \tag{E1}
\]

The normalization cancels inside each `ρ_j`; it is kept because the normalized MAEs are reported beside `ρ` (`configs/bootstrap.yaml`, `primary_test`). The fit partition is `acdc_dev` for the frozen bundle; `s_j` and the mean predictor are never re-estimated on `acdc_test` or on any M&Ms case. At least five of the seven targets must be estimable [PRESPEC]; otherwise per-target results are reported and the aggregate is withdrawn without redefinition.

**Primary test (fixed sequence).** A paired cluster bootstrap of `ρ` with the same resampled clusters for ridge and mean (Section 13.1): 2,000 replicates, seed 20260928, percentile 95 % interval (Section 13.1). **H-primary-1** is tested on `acdc_test`; success means the 95 % lower bound of `ρ` exceeds **0.10**, the smallest effect of interest: the transition reduces family-mean normalized error by more than ten percent with confidence. **H-primary-2** is tested on `mms_external` with the same criterion **only if H-primary-1 succeeded**; otherwise it is reported descriptively with the statement that the fixed sequence stopped. The 220-case stratum repeats H-primary-2 as a sensitivity and is never a rescue test. Plain superiority (lower bound > 0) and the point estimate are reported alongside. Neither outcome establishes clinical relevance.

**H-primary-3 (version 6.0.1; Decision D28) [PRESPEC].**
- **Bundle:** the unchanged v6.0.0 analysis bundle, whose `.stages/FROZEN.json` hashes equal those of run `v6-primary-20260929-2307`. A replication run that reproduces them qualifies. The H-primary-3 record names the bundle hashes; a result from any other bundle is not an H-primary-3 result.
- **Population:** `mms2_external` (Section 4.6), evaluated once.
- **Estimand:** ρ of (E1) over the same primary target set, with `s_j` and the mean predictor of `acdc_dev`. At least five of the seven targets must be estimable.
- **Test:** the paired cluster bootstrap above (2,000 replicates, seed 20260928, percentile 95 % interval). **Success means the lower bound of ρ exceeds 0.10.**
- **Family:** H-primary-3 is its own one-test family. It is declared before any M&Ms-2 byte is read, it is not a step of the fixed sequence of Decision D18, it does not depend on the recorded H-primary-1 or H-primary-2 decisions, and no multiplicity adjustment applies.
- **Precision, stated before data:** with about 180 clusters, H-primary-3 passes only if ρ is about 0.28 or more. This is a planning approximation, not a power calculation: the M&Ms-1 interval half-width (318 clusters) scaled by √(318/180) gives a half-width near 0.18.
- **Reporting:** plain superiority and the point estimate are reported alongside, and either outcome is reported.

**H4 family (version 6.0.1; Decision D36) [PRESPEC; `configs/cohorts.yaml` `mms2.endpoints.h4`].**
- **Origin and status:** hypotheses generated on the historical M&Ms-1 evaluation (per-target `ρ_j`: LVEF 0.63 and RVEF 0.50 transferred; the absolute volumes did not), tested prospectively and confirmatorily on `mms2_external` only. The family is declared before any M&Ms-2 byte is read and is independent of the H-primary-3 decision.
- **Bundle and population:** the unchanged v6.0.0 bundle of H-primary-3 on `mms2_external`, evaluated once, on the rows of `evaluate --split mms2_external`; `s_j` and the mean predictor of `acdc_dev`.
- **H4a, ratio concepts transfer:** `ρ_EF` = ρ of (E1), computed with the H-primary-3 code, over the ejection-fraction targets of the primary family {`lv_ef_percent`, `rv_ef_percent`}. **Success means the two-sided 97.5 % cluster-bootstrap lower bound of `ρ_EF` exceeds 0.10.** Both EF targets must be estimable; otherwise H4a is NE (reported as not passed).
- **H4b, label-free harmonization helps:** Δρ = ρ(S3 variant H-a) − ρ(frozen bundle) over the primary family on the same rows, where H-a re-standardizes A with the mean and SD of the unlabelled `mms2_external` partition and T, λ and the predictor stay frozen (Section 13.5). **Success means the two-sided 97.5 % paired cluster-bootstrap lower bound of Δρ exceeds 0.** At least five of the seven targets must be estimable for both members; otherwise NE.
- **Test and multiplicity:** paired cluster bootstrap, 2,000 replicates, seed 20260928, stable sub-seeds H4a 1958159060 and H4b 1591327988 (first 4 bytes, big-endian, of `SHA256("20260928|" + id)`), percentile intervals and the NE rules of Section 3.5. Bonferroni within the family: each of the two tests is judged by its two-sided 97.5 % interval, so the family-wise error is at most 0.05. The family is separate from H-primary-3 (its own one-test family) and from F6; no other adjustment applies.
- **Other partitions:** the same code runs on `acdc_test` and `mms_external`, descriptively only, labelled "registered after unblinding"; no decision is derived there.
- **Record:** `evaluate-h4 --split <partition>` writes the once-only decision record `metrics/confirmatory_h4/h4_<partition>.json` from the recorded `evaluate` outputs and the hash-verified bundle (the frozen T must reproduce the recorded B̂); a recorded decision is never recomputed or overwritten. Manuscript Table 3 reports H4a and H4b beside H-primary-1 to H-primary-3.

**Key secondary endpoints.** `ρ_j` with bootstrap intervals for `global_wall_thickness_ed_mm` and `global_wall_thickening_percent` on `acdc_test` and `mms_external`, without a success criterion.

**Secondary endpoints.** Per-target metrics for every eligible target (Section 13.2); the linearity criterion (Section 9.4); rulebook metrics; predictor metrics; external strata and the replication stratum (Section 12.3); the refit arm (Section 12.3); ablations (Section 13.3); optional branches if executed.

### 3.4 Falsifiable predictions

| ID | Hypothesis | Prediction that would refute it |
|---|---|---|
| H1 (H-primary-1) | Primary-family reconstruction on `acdc_test` beats the development mean by more than 10 % (RQ1). | Bootstrap lower bound of `ρ` ≤ 0.10. |
| H2 (H-primary-2) | The frozen bundle keeps that gain on `mms_external` (RQ2). | H1 fails (sequence stops) or the lower bound of `ρ` on `mms_external` ≤ 0.10. |
| H3 | Unit-free targets transfer better: on `mms_external` the mean `ρ_j` of {`lv_ef_percent`, `rv_ef_percent`} exceeds the mean `ρ_j` of the five absolute targets (RQ3). | Paired bootstrap interval of the difference includes zero or lies below it. |
| H4 | Vendors C and D degrade more than A and B (RQ4). | Bootstrap interval of the family-mean normalized MAE difference (C ∪ D minus A ∪ B) includes zero or lies below it. |
| H5 | The never-analysed 220 replicate the 320 (RQ5). | The H-primary-2 decision on the 220 stratum differs from that on the 320. |
| H6 | The affine transition is adequate: the nonlinear comparator does not meet the frozen criterion of Section 9.4 (RQ8). | Criterion met → reported as such; the executed pipeline stays affine. |
| H7 | The primary rulebook is non-empty with accepted coverage ≥ 0.30 and covered fidelity ≥ 0.80 on `acdc_test` [PROPOSED thresholds] (RQ6). | Either bound violated → reported as audit tax; an empty rulebook is reported with its failing gates. |
| H8 | On `mms_refit_eval` the refit arm improves `ρ` over the frozen ACDC bundle, and the improvement is larger on seen vendors (A, B) than on unseen vendors (C, D) (RQ7). | No paired improvement, or an improvement on unseen vendors at least as large as on seen vendors. |
| H9 (optional) | The finite-action defect of the reconstructed primary family is smaller under ±1 px translation than under ±2° rotation (RQ10). | The paired bootstrap interval of the difference over the 12 selected cases includes zero or lies below it. |
| H10 (H-primary-3; 6.0.1) | The unchanged v6.0.0 bundle keeps the gain on the prospective `mms2_external` (RQ12). Manuscript protocol 2.1 calls this hypothesis "H3"; in this table H3 is the unit-free-target hypothesis. | Bootstrap lower bound of `ρ` on `mms2_external` ≤ 0.10. |
| H11 (H4a; 6.0.1, D36) | Ratio concepts transfer: on `mms2_external` the unchanged bundle reconstructs the EF targets better than the development mean by more than 10 % (the prospective, confirmatory form of H3 and RQ3). Manuscript protocol 2.1 and D36 call it "H4a"; in this table H4 is the vendor hypothesis. | Two-sided 97.5 % lower bound of `ρ_EF` ≤ 0.10, or NE. |
| H12 (H4b; 6.0.1, D36) | Label-free harmonization helps: S3 variant H-a raises ρ on `mms2_external` over the frozen bundle (RQ15). | Two-sided 97.5 % paired lower bound of Δρ ≤ 0, or NE. |

### 3.5 Non-estimable results

Every estimand is reported with its denominator and eligibility status. A quantity whose denominator is empty, whose reference is unsupported for a cohort, or whose bootstrap yields fewer than 95 % valid replicates is written as `NE` with a reason code, never as zero and never omitted [P5 convention].

---

## 4. Dataset comparison and final cohort decision

### 4.1 Requirements the cohorts must satisfy

R-D1 public, versioned, citable, with documented access terms. R-D2 cine short-axis stacks with released expert reference segmentations of LV cavity, LV myocardium and RV cavity at ED and ES (required by the mask-conditioned design, Decision D7). R-D3 a diagnostic-group label for every development case, in a vocabulary that can serve as the consequent. R-D4 an external cohort from other centres and vendors with the same three structures annotated. R-D5 one case per patient or identifiers sufficient to prevent leakage. R-D6 no new clinician involvement. R-D7 terms that permit a local research copy and publication of aggregate results.

### 4.2 Comparison of resources

Counts describe published resources or locally held releases, not this project's eligible sample. Facts marked **VERIFIED** were confirmed against a primary source or the publisher's record on September 28, 2026; facts marked *not verified* come from secondary descriptions and are not used for any decision.

| Resource and version | Cohort and projections | Reference evidence and uncertainty | Identifiers, reports, access; implication for this study |
|---|---|---|---|
| **ACDC** (Automated Cardiac Diagnosis Challenge; Bernard et al., IEEE TMI 2018;37(11):2514–2525) | 150 patients, 100 official training and 50 official testing, five evenly represented groups (DCM, HCM, MINF, NOR, RV; 20 per group in training, 10 in testing) [LOCAL]; single centre; cine short-axis 4D series plus separate ED and ES frames | Expert reference masks (RV cavity, myocardium, LV cavity) at ED and ES; `Info.cfg` with ED, ES, `Group`, `Height`, `Weight`, `NbFrame`. Frame files have unit-scaled affines; header `pixdim` holds the spacing [LOCAL] | Patient folders `patientNNN`; one study per patient. CC BY-NC-SA 4.0, non-commercial scientific research only, mandatory citation (tracked `data/raw/acdc/database/LICENSE_TERMS.md`). Development and internal test. [D1–D3; **VERIFIED** citation] |
| **M&Ms-1 OpenDataset**, complete release (Campello et al., IEEE TMI 2021;40(12):3543–3554) | The challenge paper reports 375 subjects from centres in Spain, Germany and Canada scanned on Siemens, Philips, GE and Canon systems [D5, **VERIFIED**]. The local release has 345 cases: Training/Labeled 150 (A 75, B 75), Training/Unlabeled 25 (C 25), Validation 34 (A 4, B 10, C 10, D 10), Testing 136 (A 16, B 40, C 40, D 40) [LOCAL] | Expert masks (LV, myocardium, RV) on exactly the ED and ES frames for all 320 labelled cases; information CSV with vendor, centre, 0-based ED/ES indices, age, pathology, sex, height, weight [LOCAL] | Case codes (`External code`); one case per subject. M&Ms Research Use Agreement: clause 4 permits a verbatim copy for personal research use, clause 5 forbids distributing any portion. Official site https://www.ub.edu/mnms/ (unavailable on September 26 and 28, 2026); unofficial mirrors are forbidden. External cohort (320). [D4–D6] |
| **MMS_100** (frozen subset of M&Ms-1) | 100 cases: Testing 50, Training/Labeled 32, Validation 18; vendors A 24, B 22, C 29, D 25; one case per (vendor, centre, pathology) group round-robin over 23 groups [LOCAL] | Same masks and CSV as the full release | Frozen reader-study cohort of clinical-validation protocol v2; analysed computationally under v5. In v6 a declared, previously unblinded stratum of `mms_external` (D16, D24) |
| **M&Ms-2** (Martín-Isla et al., IEEE JBHI 2023, DOI 10.1109/JBHI.2023.3267857) | Multi-disease, multi-view (short-axis and long-axis) right-ventricle segmentation challenge [**VERIFIED** identity]; subject counts in secondary sources disagree (360 versus 450), *not verified* (official site unavailable) | RV-focused annotations; *not verified* in detail | Not selected: RV-focused challenge with a different label contract, not held locally, and no owner instruction to add it. *Amended in 6.0.1:* the owner downloaded the release on October 1, 2026, and the owner decisions of October 2, 2026 select half of it as the prospective external cohort; the other half stays sealed (Section 4.6) |
| **Sunnybrook Cardiac Data** (2009 LV Segmentation Challenge; Radau et al., MIDAS Journal 2009) | 45 cine-MRI studies from four groups (healthy, hypertrophy, heart failure with and without infarction); 1.5 T GE Signa [**VERIFIED**, Cardiac Atlas Project page] | Expert endocardial and epicardial contours at ED, endocardial only at ES; no RV [**VERIFIED**] | Public-domain licence on the Cardiac Atlas Project [**VERIFIED**]. Not selected: no RV, no ES myocardium, too small |
| **Second Annual Data Science Bowl** (Kaggle, 2016) | Multi-hospital 2D cine DICOM, about 30 frames per series, varying slices and views [**VERIFIED**, competition page] | LV end-diastolic and end-systolic volumes only; no segmentations [**VERIFIED**]; case counts *not verified* | Kaggle competition terms. Not selected: no masks, so B cannot be measured on the same code path (R-D2) |
| **UK Biobank CMR** | Population imaging study targeting 100,000 participants; cine long- and short-axis among other sequences [**VERIFIED**, UK Biobank protocol literature] | Returned datasets from applicants and consortia; access conditions, fees and derived-segmentation coverage *not verified* | Application-based access with an obligation to return results [**VERIFIED**]. Not selected: no approved application, predominantly healthy population, no ACDC-compatible group labels |
| **CMRxRecon** (Wang et al., *Sci Data* 2024;11:687) | 300 subjects, multi-contrast, multi-view, multi-coil raw k-space (cine and T1/T2 mapping); 120/60/120 split [**VERIFIED**] | Segmentation labels reported for the cine data [**VERIFIED**]; diagnostic groups *not verified* | Not selected: reconstruction benchmark, no diagnostic-group consequent, not held locally |
| **CMRxMotion** (MICCAI 2022) | 40 healthy volunteers, 320 cine series on one 3 T Siemens scanner under four breath-hold behaviours [**VERIFIED**] | Motion-artifact grades and segmentations; no disease groups | Not selected: healthy volunteers only; potentially useful as a future robustness stress set |

### 4.3 Binding selection

**Development cohort: ACDC development (`acdc_dev`).** The 100 official training patients minus the two declared duplicate-image exclusions (`patient055`, `patient076`; Section 4.5), 98 patients, with the released ED and ES frames and reference masks, `Info.cfg` metadata and the header geometry rule of Section 7.3. Every fitted object (standardizers, target transforms, λ, the transition, the audited predictor, WEDD cuts, granules, rules, QC outlier references) is fitted on `acdc_dev` or on subsets of it. [D1–D3]

**Internal test cohort: ACDC testing (`acdc_test`).** The 50 official testing patients, used for evaluation only. These patients were evaluated under v4/v5 with the pre-fix geometry and a different A, transition and rule policy; the investigators saw those results. v6 therefore declares `acdc_test` **previously analysed** (`configs/splits.yaml`, `previously_unblinded`); no v6 setting was tuned on it, and H-primary-1 is interpreted with that limitation (Section 17).

**External cohort: complete labelled M&Ms-1 release (`mms_external`).** All 320 cases of Training/Labeled, Validation and Testing that meet the eligibility checks of Section 4.5, regardless of their challenge split, evaluated once with the frozen ACDC bundle (Decision D16). The release split, vendor, centre, pathology and MMS_100 membership are strata, never fitting inputs.

**Excluded: Training/Unlabeled (25 cases, vendor C, centre 4).** Reason code `MMS_RELEASE_UNLABELED`: their CSV ED and ES are 0 placeholders although the masks carry labels on two frames, and two pairs are voxel-identical duplicates (C5Q2Y5 = E9L4N2, A8C9H8 = Q0Q1Y4) [LOCAL]. They are listed in `cohort_exclusions.parquet` and never read after staging verification.

**Rejected alternatives.** M&Ms-2, Sunnybrook, the Data Science Bowl, UK Biobank, CMRxRecon and CMRxMotion, for the reasons in Section 4.2; a pooled ACDC + M&Ms development set (would destroy the external test); automatic segmentations as masks (would turn the audit into a segmentation study, Decision D7). *Amended in 6.0.1:* M&Ms-2 is no longer rejected; half of it becomes the prospective external cohort `mms2_external` (Section 4.6), which is never pooled with ACDC or M&Ms-1.

### 4.4 Access and compatibility gates

Before any v6 measurement, archive the licence and agreement texts, the release file manifests with SHA-256 and the staging manifests. Confirm what may be redistributed under each agreement; do not infer permission from public discoverability. A failed gate blocks the corresponding branch, not schema generation, synthetic tests or mathematical verification.

| Gate | Condition | Blocked consequence |
|---|---|---|
| ACDC release integrity | All 1,056 release files match the tracked release manifest; the 300 ED/ES image and mask files stage without error | `B_ACDC`, A, predictor and everything downstream |
| ACDC geometry | Every ED/ES frame passes the geometry rule of Section 7.3 with the method recorded (`nifti_header_pixdim_unscaled_affine` expected for all 150) | Volumetric targets for the failing patients (critical QC) |
| M&Ms release integrity | All 691 files match `BACKUP_MANIFEST.sha256`; 345 CSV rows match 345 case folders; the CSV equals the tracked copy byte for byte | `B_MMS`, all external analyses |
| M&Ms frames and labels | For all 320 labelled cases the annotated mask frames equal the CSV {ED, ES} and labels 1–3 are present at both phases [LOCAL: 320/320 on September 28, 2026] | The failing case is excluded with its reason |
| Protocol freeze | Tag `protocol-v6.0.0` resolves and the ACDC analysis bundle carries `.stages/FROZEN.json` | Any M&Ms B, A, reconstruction, rule or metric (Section 8.5) |
| Licence of published material | Each figure or table that shows case-level content is permitted by the relevant agreement | That asset is replaced by an aggregate or omitted |
| Protocol amendment freeze (6.0.1) | Tag `protocol-v6.0.1` is an annotated tag and HEAD descends from it | Any M&Ms-2 read, extraction, staging, B, A, reconstruction, rule or metric (Sections 4.6, 8.5) |
| M&Ms-2 archive integrity (6.0.1) | After the tag: `MnM2.rar` at the reference path has 9,824,397,247 bytes and the SHA-256 of Section 4.6; its listing holds the subjects `001`–`360` and the 720 `mms2_external` short-axis ED/ES image and mask members; extracted byte counts equal the listing | `extract-mms2` and every M&Ms-2 stage; no file is written |

### 4.5 Known cohort characteristics that shape the design

- **Vendor, centre and split structure** [LOCAL]. A = Siemens, centre 1 (95 labelled cases); B = Philips, centres 2 (74) and 3 (51); C = GE, centre 4 (50 labelled; the other 25 are the excluded unlabelled cases); D = Canon, centre 5 (50). Split × centre for the 320: Training/Labeled C1 75, C2 50, C3 25; Validation C1 4, C2 5, C3 5, C4 10, C5 10; Testing C1 16, C2 19, C3 21, C4 40, C5 40. GE and Canon have no labelled training cases.
- **Pathology** (320 labelled) [LOCAL]: DCM 97, HCM 82, NOR 79, Other 25, HHD 15, ARV 15, IHD 4, LVNC 2, AHS 1. Rare classes (IHD 4, LVNC 2, AHS 1) support no class-level estimate and are reported as counts only.
- **Never-analysed stratum** [LOCAL]: 220 cases (A 71, B 103, C 21, D 25; 118 Training/Labeled, 16 Validation, 86 Testing).
- **Anthropometrics**: height is recorded for 39.1 % of the 345 release rows (A 89.5 %, B 0 %, C 0 %, D 100 %); weight for 99.4 % [LOCAL]. BSA and every BSA-indexed quantity are therefore unsupported for M&Ms external claims (Section 6.4).
- **Geometry** [LOCAL]: M&Ms images are 4D float32 with voxel-scaled affines (`sform_code` 1) in all 345 cases, and mask geometry equals image geometry in all 345. Matrices, slice counts, frame counts and spacings differ by vendor (A 196–256 × 192–288, 5–16 slices, 0.977–1.625 mm in plane, 8–10 mm slices; B 256–384 square, 8–20 slices, 5–10 mm; C 256–512 square, 0.684–1.797 mm, 5–10 mm; D 220–548 × 192–512, 0.795–1.823 mm, 10 mm). ACDC frame files have unit-scaled affines (Section 0, pre-freeze revision).
- **Mask encoding** [LOCAL]: 213 M&Ms masks are float32 integer-valued; 132 are uint8 with `scl_slope` 3/255, so scaled reads return values such as 1.00000002. Labels are always rounded with `np.rint` before comparison (acceptance test 6).
- **Frame indices** [LOCAL]: M&Ms ED and ES are 0-based indices, present for all rows; no labelled case has ED = ES. ACDC `Info.cfg` ED and ES are 1-based frame numbers (file names `patientNNN_frameXX`), converted to 0-based indices on load.
- **Flags, not failures** [LOCAL]: D3O9U9 (5 slices at 8 mm; RV on two slices at ES) and B6I0T4 (RV on two slices at ES) are retained and flagged informational.
- **Duplicates** [LOCAL]: no duplicate mask signatures among the 320 labelled cases.
- **Duplicate images** [LOCAL]. `validate-cohort` compares the SHA-256 of the decoded staged ED and ES image voxels within and across cohorts. Four pairs of labelled cases have identical decoded images but different reference masks and different source file hashes: ACDC `patient055` (training, MINF) and `patient118` (testing), ED and ES; ACDC `patient074` and `patient076` (both training, NOR), ED and ES; M&Ms `A7G0P5` (Training/Labeled) and `K3R0Y7` (Validation; an MMS_100 member), ED and ES; M&Ms `C8J7L5` (Validation) and `C8O0P2` (Testing), ED only (both MMS_100 members, same vendor). The information CSV records different vendor and centre values for the identical images of `A7G0P5` and `K3R0Y7` (a release metadata inconsistency, reported and not corrected). *Amended September 28, 2026:* by PI decision the pairs are declared in `configs/cohorts.yaml` (`duplicate_image_pairs`) with their actions: `patient055` is excluded from `acdc_dev` (`DUPLICATE_IMAGE_OF_TEST_CASE`) and `patient118` stays in `acdc_test`; `patient076` is excluded from `acdc_dev` and `patient074` kept (the lower identifier; `DUPLICATE_IMAGE_WITHIN_PARTITION`); `K3R0Y7` is excluded from `mms_refit_eval` (`DUPLICATE_IMAGE_OF_REFIT_TRAIN_CASE`) and both it and `A7G0P5` stay in `mms_external`; `C8J7L5` and `C8O0P2` stay everywhere. The two members of a declared pair share one bootstrap cluster wherever both appear (Section 13.1). `validate-cohort --strict` fails on any detected pair that is not declared, or whose declared frames or action do not match (acceptance test 24).

### 4.6 M&Ms-2 prospective cohort (version 6.0.1; Decision D27) [PRESPEC; `configs/cohorts.yaml` and `configs/splits.yaml` (`mms2`), `configs/source_bindings/mms2.json`, `registry/sources/mms2*`]

**Source receipt.**

| Item | Value |
|---|---|
| Release archive | `MnM2.rar`, 9,824,397,247 bytes, SHA-256 `934076fe388f0747c80b7f0fdf5ebc57cd480ab737be8228639480e61515362b`; downloaded by the owner on October 1, 2026 |
| Reference copy | `<local path>`, read-only; the only extraction source, hash-verified before use |
| Other copy | off-site storage, storage only, under cloud-storage exception CSE-1 (`protocol/amendments.md`, October 2, 2026). It is never extracted, previewed, processed or used as a source |
| Terms | M&Ms-2 Research Use Agreement (`<local path>`): a personal research copy; no distribution (Section 14.5) |
| Tracked receipt | `registry/sources/mms2*`: identifiers, member paths, sizes, SHA-256 and membership flags only (Decision D2 rules); no per-case metadata |
| Identity | Martín-Isla et al., IEEE JBHI 2023 (Section 4.2) |

**Split rule and manifest.**
- **Rule:** sort the subject identifiers `001`–`360` by the hexadecimal digest `SHA256("mms2-split-v6.0.1|" + subject_id)` ascending, ties by `subject_id`. Ranks 1–180 (the `rank` column of the manifest) form `mms2_external`; ranks 181–360 form `mms2_sealed`, which is reserved for protocol v7 and never extracted under v6. The hash uses the raw zero-padded code.
- **No data read:** the rule reads no image, mask, CSV row or label. It is computed from the declared identifier range before the tag.
- **Manifest:** the membership manifest `registry/sources/mms2_split_manifest.csv` (columns `subject_id, rank, sha256, partition`; 360 rows) is committed in the amendment commit; its SHA-256, computed on the file with LF line endings, is `f427d9723b4055024ba92e47bc652388d615c54bbf4968de9088255dfa1a072d` (also in `registry/sources/mms2.yaml`). `load_split_manifest` refuses a file that does not reproduce the rule.
- **Check after the tag:** the archive listing must hold exactly the subjects `001`–`360` with the expected short-axis members. Any difference refuses extraction and writes no file; the split is never recomputed from the listing, and a new dated amendment is required.

**Release facts read before the tag (October 2, 2026, with the owner's approval).** To avoid a later amendment, the header line of the release CSV and the member names of the archive listing were read before the tag (logged by the owner in `<local path>`). Only column names and file names were exposed: no subject row, image, mask or label.
- **Release CSV header:** `MnM2/dataset_information.csv` has exactly the columns `SUBJECT_CODE,DISEASE,VENDOR,SCANNER,FIELD` (comma-separated, CRLF line endings). There is no `SEX` and no `AGE` column.
- **Member layout:** `MnM2/dataset/<id>/<id>_SA_ED.nii.gz`, `<id>_SA_ED_gt.nii.gz`, `<id>_SA_ES.nii.gz` and `<id>_SA_ES_gt.nii.gz`, plus short-axis cine and long-axis files that are never extracted; `MnM2/dataset_information.csv` and `MnM2/readme.txt`; 3,964 entries in total. Extraction selects members by unique file name; the folder template and the entry count are compared and recorded in the extraction receipt, never blocking (`configs/cohorts.yaml` `mms2.inputs.archive_layout`).

**Ingest contract.**
- **Members:** for each `mms2_external` subject, only the short-axis ED and ES images and their `_gt` masks (720 files). Long-axis and cine members and every `mms2_sealed` member are never written to any disk or volume; only the member names and sizes of the listing are read.
- **Extraction:** `extract-mms2` runs Windows `tar.exe` (bsdtar) with `-T` and the list of the 720 member names on the hash-verified D: copy. It writes a local source tree that is then bound read-only (`configs/source_bindings/mms2.json`); the extracted byte counts equal the listing.
- **Labels:** identity: 1 LV cavity, 2 LV myocardium, 3 RV cavity (the canonical convention; no swap), rounded with `np.rint`.
- **Geometry:** the v6 rule of Section 7.3: the header `pixdim` when the affine is unit-scaled; inconsistent geometry is rejected as critical QC. The method is recorded per case.
- **Phases:** ED and ES from the member file names.
- **Release CSV:** read after the tag, into memory, only by `src/tmmri/ingest/mms2.py`; `mms2_sealed` rows are dropped before any write and their values are never parsed, and only the external rows reach the source area (`MMS2_INFO_EXTERNAL.csv`). The contract (`configs/cohorts.yaml` `mms2.info_csv`) is exactly the five columns above.
  - **Fail closed (exit 8, `HUMAN_GATE`, before any image is extracted):** an unknown or missing column; a malformed or repeated `SUBJECT_CODE`; a `DISEASE` value outside the declared synonym table. The table maps each code and its spelled-out forms, compared case-insensitively after trimming white space: NOR/Normal; LV/Dilated left ventricle; HCM/Hypertrophic cardiomyopathy; ARR/Congenital arrhythmogenesis/Arrhythmogenic; FALL/Tetralogy of Fallot; CIA/Interatrial communication; TRI/Tricuspid regurgitation; DRV/Dilated right ventricle.
  - **Data-driven, never fail closed:** `VENDOR`, `SCANNER` and `FIELD` are unverified before the tag. Every value is accepted and recorded verbatim (`vendor_release`, `scanner`, `field_strength`). The vendor is also normalized case-insensitively to Siemens, GE or Philips when the string names exactly one of them (GE as a word or "General Electric"), else kept as is; only Siemens, GE and Philips enter the typed-B `scanner_vendor`.
- **Identifiers:** the split hash uses the raw zero-padded code (`001`–`360`). Every downstream identifier is prefixed: `case_id` = `patient_id` = `MMS2_<code>` and `sample_id` = `MMS2:MMS2_<code>`, so no purely numeric token is a case identifier for the guard of Section 14.5.
- **QC:** the v6 severity model of Section 12.4, unchanged, with references from `acdc_dev`.

**Duplicate-image exclusions.** `validate-cohort` compares the SHA-256 of the decoded ED and ES image voxels of every `mms2_external` case with every staged ACDC case, every staged M&Ms-1 case and every other `mms2_external` case.
- **Rule:** a case with any match, at ED or ES, is excluded from `mms2_external` with a reason code of the `DUPLICATE_IMAGE_*` family: `DUPLICATE_IMAGE_OF_ACDC_CASE`, `DUPLICATE_IMAGE_OF_MMS1_CASE` or `DUPLICATE_IMAGE_WITHIN_MMS2` (`configs/cohorts.yaml` `mms2.reason_codes`; the frozen `registry/reason_codes.yaml` is unchanged). Both members of a pair inside `mms2_external` are excluded.
- **Sealed half:** it is never decoded under v6, so no comparison with it is made. The exclusion ledger records each excluded case with its matched case, so that a later ingest of the sealed half can repeat the check.
- **Size:** the analysed `mms2_external` is 180 cases minus these exclusions; critical QC is counted in the flow diagram. Each remaining case is its own cluster.

**Crosswalk (Decision D8, extended).** ACDC classes stay the consequent vocabulary. The release codes and their spelled-out synonyms are bound in `configs/cohorts.yaml` `mms2.info_csv` and `mms2.crosswalk`.

| M&Ms-2 pathology | ACDC class | Crosswalk status | External class endpoint |
|---|---|---|---|
| Normal | NOR | `HARMONIZABLE_SAME_DEFINITION` | accuracy, macro-F1, macro one-versus-rest AUROC and ECE on the NOR + HCM subset |
| Hypertrophic cardiomyopathy | HCM | `HARMONIZABLE_SAME_DEFINITION` | as for Normal |
| Dilated left ventricle | DCM | `HARMONIZABLE_NOT_EXACT` (the ACDC DCM class also requires LVEF < 40 %) | distribution of `d_model`; reported separately, never pooled |
| Dilated right ventricle, tricuspid regurgitation, tetralogy of Fallot, inter-atrial communication, congenital arrhythmogenesis (the RV family) | RV | `HARMONIZABLE_NOT_EXACT`; pooled as one predeclared stratum | distribution of `d_model`; reported separately, never pooled |
| — | MINF | no M&Ms-2 category | none |

The crosswalk is frozen with the tag and never revised after the M&Ms-2 evaluation. It asserts label compatibility of dataset-defined categories, not diagnostic equivalence.

**M&Ms-2 lock.** Until the annotated tag `protocol-v6.0.1` exists and HEAD descends from it, every M&Ms-2 command refuses with exit 3 and writes nothing. This covers:
- `extract-mms2`;
- `stage-sources --dataset mms2` and `validate-cohort` on M&Ms-2;
- `build-semantics --dataset mms2` and `extract-formal --dataset mms2`;
- `evaluate --split mms2_external`, and `audit-shift` and `summarize-predictor` on `mms2_external`;
- every 6.0.1 stage that reads `mms2_external`: F8, D20, F9, F9b and S1–S4 evaluation.

Unlike M&Ms-1 under 6.0.0, verification and staging are locked too: no M&Ms-2 byte, including the archive hash, is read before the tag (`extract-mms2` checks the lock before it hashes the archive). The single exception is the owner-approved read of the CSV header line and the member names on October 2, 2026, recorded above. The only override is `--allow-unfrozen-synthetic` on a source carrying the `SYNTHETIC_FIXTURE` marker. Endpoint stages also require the frozen analysis bundle (Section 8.5) and run once per bundle (Section 14.6).

---

## 5. Standards and protocols evidence table

Three categories are kept separate: CMR acquisition and post-processing conventions, anatomical nomenclature, and AI reporting guidance. A reporting checklist is not a clinical standard. Bibliographic identities were verified on September 28, 2026 (Section 19.5).

| Category | Source, version and scope | Applicability | Exact constrained decision |
|---|---|---|---|
| CMR post-processing | **SCMR standardized image interpretation and post-processing, 2020 update**; Schulz-Menger J et al., *J Cardiovasc Magn Reson* 2020;22:19. | Primary convention for volumes, function and mass from short-axis stacks. | Volumes by voxel summation over the short-axis stack (disc summation equivalent); EF = 100 · (EDV − ESV)/EDV; mass = myocardial volume × 1.050 g/mL; papillary and trabecular handling and basal-slice choice follow each dataset's reference annotations and are recorded (`papillary_inclusion_status`), never converted; no normal-range cutoff becomes a rule threshold. [S1] |
| CMR acquisition protocols | **SCMR standardized CMR protocols, 2020 update**; Kramer CM et al., *J Cardiovasc Magn Reson* 2020;22:17. | Background for cine short-axis acquisition. | Slice thickness, gap and increment are recorded per case; conformance of either dataset to the protocol recommendations is not assumed or claimed. [S4] |
| CMR reporting | **SCMR guidelines for reporting CMR examinations**; Hundley WG et al., *J Cardiovasc Magn Reson* 2022;24:29. | Wording boundary. | Outputs are research quantities, never clinical reports; units and definitions stated with every quantity. [S5] |
| Body-surface area | **Mosteller RD.** Simplified calculation of body-surface area. *N Engl J Med* 1987;317(17):1098. | Only for the descriptive block-E composites. | BSA = √(height_cm · weight_kg / 3600), computed only where both exist; every BSA-indexed quantity is excluded from v6 targets and rules (Section 6.4). [S3] |
| Myocardial nomenclature | **AHA standardized myocardial segmentation and nomenclature for tomographic imaging**; Cerqueira MD et al., *Circulation* 2002;105(4):539–542. | Required for any segmental claim. | AHA-16 segmental features need RV insertion points and long-axis orientation that are not verified; all AHA-16 columns are reserved unsupported (Section 6.5). [S2] |
| AI imaging reporting | **CLAIM 2024 Update**; Tejani AS et al., *Radiol Artif Intell* 2024;6(4):e240300. | Primary reporting checklist. | Report data sources, partitions, reference standards, preprocessing, model selection, error analysis and reproducibility item by item; use "reference standard", never the alternative term. Checklist filed as `manuscript/CLAIM2024_checklist.md` at M15. [S6] |
| Diagnostic-accuracy reporting | **STARD-AI 2025**; Sounderajah V et al., *Nat Med* 2025;31(10):3283–3289; author correction 2026 [**VERIFIED 2026-09-24**, CXR-TRANS-AUDIT]. | The predictor's class-accuracy estimand only. | Flow diagram through eligibility, QC and crosswalk; explicit handling of abstained outputs and off-ontology references; applicability crosswalk. [S7] |
| Prediction-model reporting | **TRIPOD+AI**; Collins GS et al., *BMJ* 2024;385:e078378. | The audited predictor's development and evaluation. | Model specification, partitions, calibration and uncertainty in the predictor table. [S8] |
| Early live clinical evaluation | **DECIDE-AI**; Vasey B et al., *Nat Med* 2022;28:924–933. | **Not applied**: no live clinical evaluation occurs in this protocol (the governed reader study is outside it). | Recorded as not applicable with reason; the v5 methodology note (formerly `docs/methodology/`, now `docs/history/methodology_v5/`) framed v5 as DECIDE-AI early-stage research, which v6 does not repeat. [S9] |

CLAIM, STARD-AI and TRIPOD+AI are reporting guidance, not diagnostic standards, clinical certification or evidence of utility. Ethics and privacy review and compliance with each data agreement remain necessary; no exemption or approval is invented.

---

## 6. Cardiac MRI semantic feature registry

### 6.1 Evidence tiers, states and typing conventions

The registry is a versioned contract (`registry/semantic_registry.yaml`, version 6.0.0, source schema `B_v2.0`, typed schema `B_v6.0`). A feature cell is a nullable typed value plus two companions, `__state` and `__evidence`. Registry fields per feature: `name`, `family`, `mental_model_tier`, `meaning`, `dtype`, `unit`, `domain`, `evidence_sources` (per cohort), `algorithm_id`, `source_requirement`, `target_transform`, `discretization`, `transition_role` (`candidate`, `candidate_secondary`, `excluded`), `rule_role` (`candidate`, `reconstruction_only`, `excluded`), `primary_family`, `key_secondary`, `harmonization` (for M&Ms), `dependencies`, `standard_ref`, `notes`.

**Evidence tiers.**

| Tier | Meaning | Cohorts |
|---|---|---|
| `E1_EXPERT_MASK_GEOMETRY` | Deterministic computation from the released expert segmentation (ED and ES label maps) combined with the header geometry of the same file | ACDC, M&Ms |
| `E1H_HEADER_GEOMETRY` | NIfTI header geometry: in-plane spacing and slice increment (affine column norms, or header `pixdim` when the frame affine is unit-scaled; commit `8ed5a40`) | ACDC, M&Ms |
| `E2_RELEASE_METADATA` | Field published with the release (`Info.cfg`; the M&Ms information CSV) | ACDC, M&Ms |
| `E3_DERIVED_COMPOSITE` | Arithmetic composite of E1/E1H/E2 values (Mosteller BSA, BMI, BSA-indexed values) | ACDC; M&Ms only where height exists |
| `QC_DERIVED` | Automated quality-control computation; computational triage, not expert adjudication | ACDC, M&Ms |
| `E0_PROJECT_PROVENANCE` | Project identifier, split assignment, algorithm tag or path; not evidence about the patient | ACDC, M&Ms |
| `UNSUPPORTED` | Prerequisites absent in the release; the column is reserved and never populated | all |

A mask-derived measurement is a research quantity computed from a reference-standard annotation; it never becomes a clinical measurement claim, and a released segmentation is never replaced by an automatic one.

**States.** `OBSERVED` (the only state with a non-null value), `NOT_APPLICABLE` (undefined by construction, for example EF when EDV < 1 mL), `UNSUPPORTED` (the cohort cannot supply the tier), `QC_BLOCKED` (a critical or high QC reason masks the cell), `MISSING_SOURCE` (a required release field or file is absent, for example height for M&Ms vendors B and C), `CONFLICTING` (two release sources disagree, for example CSV frames and annotated mask frames). No numeric sentinel is permitted. Reconstructed **B̂** carries its own states (Section 11.2).

### 6.2 Registry blocks A–C: ventricles, myocardium and wall

`dx`, `dy` are the in-plane spacing and `dz` the verified slice increment (Section 7.3); volumes are voxel counts × `dx · dy · dz / 1000` in mL. Rule candidates use `wedd_gaussian_mm_v1`; every other block member is `not_discretized`.

| Name | Meaning | dtype, unit, domain | Transform / discretization | Roles |
|---|---|---|---|---|
| `lv_edv_ml`, `lv_esv_ml` | LV cavity volume at ED and ES | float64, mL, (0, ∞) | `log`; WEDD | **primary targets**; rule candidates |
| `lv_ef_percent` | `100 · (EDV − ESV)/EDV`; `NOT_APPLICABLE` if EDV < 1 mL | float64, %, [0, 100] | `logit_percent`; WEDD | **primary target**; rule candidate |
| `rv_edv_ml`, `rv_esv_ml` | RV cavity volume at ED and ES | float64, mL, (0, ∞) | `log`; WEDD | **primary targets**; rule candidates |
| `rv_ef_percent` | RV ejection fraction | float64, %, [0, 100] | `logit_percent`; WEDD | **primary target**; rule candidate |
| `lv_sv_ml`, `rv_sv_ml` | EDV − ESV | float64, mL, (0, ∞) | `log`; not discretized | `candidate_secondary`; `reconstruction_only` |
| `lv_myocardial_mass_ed_g` | Myocardial volume at ED × 1.050 g/mL | float64, g, (0, ∞) | `log`; WEDD | **primary target**; rule candidate |
| `lv_myo_volume_es_ml` | Myocardial volume at ES | float64, mL, (0, ∞) | `log`; not discretized | `candidate_secondary`; `reconstruction_only` |
| `lv_myo_volume_ed_ml` | Myocardial volume at ED | float64, mL, (0, ∞) | `none` | `excluded` (equals mass / 1.050) |
| `global_wall_thickness_ed_mm` | Area-weighted mean myocardial thickness over 72 radial rays per slice at ED | float64, mm, (0, ∞) | `identity`; WEDD | **key secondary**; rule candidate |
| `global_wall_thickness_es_mm` | The same at ES | float64, mm, (0, ∞) | `identity`; not discretized | `candidate_secondary`; `reconstruction_only` |
| `global_wall_thickening_percent` | `100 · (thickness_ES − thickness_ED)/thickness_ED` | float64, %, ℝ | `identity`; WEDD | **key secondary**; rule candidate |

Block counts: A `ventricular_volume_function` 8; B `myocardium` 3; C `wall_thickness` 3. Primary rulebook antecedent candidates (registry `primary_rulebook_antecedent_candidates`): exactly the seven primary targets and the two key secondary targets, 9 in total; `lv_rv_sv_discrepancy_ml` (a v5 rule feature) is a QC column in v6. Exclusion rule: never `lv_ef_percent` together with both `lv_edv_ml` and `lv_esv_ml`, and never `rv_ef_percent` together with both RV volumes, in one rule (EF is a deterministic function of the two volumes).

### 6.3 Registry block D: in-plane shape

`lv_area_mean_ed_mm2`, `lv_area_mean_es_mm2`, `rv_area_mean_ed_mm2`, `rv_area_mean_es_mm2` (mean per-slice cavity area; float64, mm², `log`); `lv_compactness_mean_ed`, `lv_compactness_mean_es`, `rv_compactness_mean_ed`, `rv_compactness_mean_es` (mean per-slice `4πA/P²`; ratio, `identity`); `lv_eccentricity_mean_ed`, `lv_eccentricity_mean_es`, `rv_eccentricity_mean_ed`, `rv_eccentricity_mean_es` (mean per-slice moment eccentricity; ratio, `identity`). All are `candidate_secondary` transition targets and `reconstruction_only` (never antecedents of the primary rulebook). Block D count: 12. Transition targets in total: 25 (9 `candidate` + 16 `candidate_secondary`).

### 6.4 Registry blocks E and F: body-size composites and invalid geometric approximations

**Block E, `body_size_composite` (7):** `bmi_kg_m2`, `bsa_m2`, `lv_edv_indexed_ml_m2`, `lv_esv_indexed_ml_m2`, `rv_edv_indexed_ml_m2`, `rv_esv_indexed_ml_m2`, `lv_myocardial_mass_indexed_ed_g_m2` (Mosteller BSA [S3]; a BMI above 40 sets a caution flag). All are `transition_role = excluded`, `rule_role = excluded`, with `harmonization = UNSUPPORTED` for M&Ms because height is missing for every vendor B and C case. They are descriptive columns only. The v5 defect that removed them silently by substring match is replaced by this explicit, reasoned exclusion.

**Block F, `geometric_approximation_not_valid` (2):** `lv_long_axis_length_mm` (slice count × dz; the short-axis stack does not measure the long axis) and `lv_sphericity_index_3d`; both `excluded`.

### 6.5 Reserved unsupported targets, metadata, QC columns and label tables

**Reserved unsupported targets.** The 48 AHA-16 segmental columns (`aha16_seg01_wall_thickness_ed_mm` … `aha16_seg16_wall_thickening_percent`: thickness at ED and ES and thickening per segment) have `evidence_sources = UNSUPPORTED`, `transition_role = excluded` and `rule_role = excluded` (Section 5, [S2]), so that no adapter can populate them.

**Metadata and context columns (42; not features; never targets).** The `B_v2.0` names, among them `schema_version`, `sample_id`, `case_id`, `patient_id`, `study_id`, `dataset_source`, `subset_name`, `split`, `disease_label`, `ed_frame_index`, `es_frame_index`, `scanner_vendor`, `center_id`, `height_cm`, `weight_kg`, `papillary_inclusion_status`, `spacing_x_mm`, `spacing_y_mm`, `slice_thickness_mm`, `slice_gap_mm`, `slice_increment_mm`, `z_spacing_method`, `myocardial_density_g_per_ml` (constant 1.050) and the raw path columns. `disease_label` is context only: it never enters the numeric projection, A or any antecedent. The v6 stratum columns (`vendor`, `centre`, `release_split`, `mms100_member`) are joined from the cohort manifests by `sample_id`.

**QC columns (32).** Presence flags per structure and phase, label, spacing and overlap validity, EF validity, the identical-mask flag, the robust-outlier flag and features, failure reasons, wall-thickness and shape validity flags, AHA feasibility, and `lv_rv_sv_discrepancy_ml` with its review flag (Section 12.4).

**Label tables.** `Y_<dataset>.parquet` carries `sample_id`, the source value (`Info.cfg` `Group`; CSV `Pathology`), the crosswalked ACDC class or null, and the crosswalk status and reason code (registry `label_tables`; Section 12.3). Labels never enter B as features. Development class counts: 20 per class in `acdc_dev`, 10 per class in `acdc_test`.

**Registry totals:** 8 + 3 + 3 + 12 + 7 + 2 + 48 = **83 semantic features**, each as `name`, `name__state`, `name__evidence`, plus 42 metadata and context columns and 32 QC columns: 42 + 32 + 3 × 83 = **323 typed B-table columns** (the `B_v2.0` source table has 157 columns). The machine-readable registry emits these counts; `tests/test_registry_check.py` asserts them.

### 6.6 Cross-dataset feature mapping (harmonization contract; pooling is forbidden)

| Feature family | ACDC → M&Ms compatibility | Permitted external claim |
|---|---|---|
| Volumes, SV, EF, mass (blocks A, B) | `HARMONIZABLE_SAME_ALGORITHM`: identical code path on reference masks; different annotation protocols (basal slice, papillary and trabecular handling), vendors and spacings | Per-target reconstruction transport with the annotation-protocol caveat; EF versus absolute quantities compared (H3) |
| Wall thickness and thickening (block C) | `HARMONIZABLE_SAME_ALGORITHM`, in-plane spacing dependent | Per-target transport |
| Cavity area, compactness, eccentricity (block D) | `HARMONIZABLE_SAME_ALGORITHM`; area depends on in-plane spacing, the ratios do not | Per-target transport |
| BSA and BSA-indexed quantities (block E) | `UNSUPPORTED` (height missing for vendors B and C) | None |
| Long axis, sphericity (block F); AHA-16 | Excluded or `UNSUPPORTED` in both cohorts | None |
| Diagnostic label | Crosswalk of Section 12.3 | Class accuracy only on crosswalk-evaluable cases; IHD separately; off-ontology `d_model` and route distributions |
| Vendor, centre | M&Ms only (ACDC is single-centre) | Strata, never covariates in fitting |

The primary rulebook may be routed on every technically eligible external case using reconstructed B̂ without reference values; reference-based evaluation uses the measured M&Ms B. The two B tables are never pooled.

---

## 7. Exact schemas, adapters and construction contract for the reference tables B

### 7.1 Canonical artifacts and the numeric projection

Under `artifacts/<run_id>/semantics/<dataset>/` with `dataset ∈ {ACDC, MMS}` (the legacy alias `MMS_100` is kept only for governed and legacy flows):

```text
B_<dataset>.parquet               # metadata + registry features as (name__value, name__state, name__evidence)
B_<dataset>.schema.json           # generated from registry/semantic_registry.yaml; never hand-edited
B_<dataset>.provenance.parquet    # one row per (sample_id, feature): state, tier, algorithm id, geometry method, source and staged hashes
B_<dataset>.preview.csv           # header-only until an authorized build; then 12 rows, no pixels
B_<dataset>.config.json           # dataset version, supported features, bindings hash, registry version
B_<dataset>.manifest.json         # status, counts, ordered-row hash, parent hashes, environment, code commit
Y_<dataset>.parquet               # dataset labels with crosswalk status (Section 12.3)
cohort_exclusions.parquet         # every excluded source object with its reason code
```

The staging cache lives outside the run root, in the data volume (Decision D23): `/workspace/data/staged/<DATASET>/<case_id>/{ed,es}_{image,mask}.nii` plus `/workspace/data/manifests/staging_<DATASET>.json`, keyed by source SHA-256.

The typed table is not an all-real matrix. Its numeric projection `B_numeric` with the same-shaped mask `M` (Section 9.1) selects columns **by registry identity** (`transition_role ∈ {candidate, candidate_secondary}`), never by dtype or by substring matching (the v5 `index` defect). Rows are sorted by `sample_id` and the ordered identifier list is hashed; `A`, `B` and `Y` must join one-to-one. A case with an unrecoverable decode error lives in `cohort_exclusions.parquet`, not in an analysis table with invented values.

### 7.2 Normalized ledger interface

Each adapter emits `data/normalized/normalized_<DATASET>.jsonl`, one object per included case:

```text
meta: sample_id, dataset, dataset_version, patient_id | case_id, release_split, vendor, vendor_name,
      centre, height_cm, weight_kg, age_years and sex (M&Ms only; used only for the descriptive demographic strata of Section 12.3, Q28; never features, targets, fitting or selection)
frames: {ed_index, es_index, index_base: 0 | 1, n_frames, annotated_frames: [..]}
geometry: {shape, spacing_x_mm, spacing_y_mm, slice_increment_mm, z_spacing_method, affine_is_unscaled,
           sform_code, qform_code, mask_geometry_equal: bool}
mask_encoding: {dtype, scl_slope, scl_inter, rounded: true, labels_present_ed: [..], labels_present_es: [..]}
source_refs: [{role: image | mask | info, path: relative path under /sources, sha256}]
staged_refs: [{role: ed_image | ed_mask | es_image | es_mask, path, sha256}]
label: {source_value, crosswalk_class, crosswalk_status}
```

A missing key, unrecognized frame convention, geometry mismatch, unexpected label value or hash mismatch raises an error; no heuristic fallback exists. Every raw-key binding lives in `configs/source_bindings/<dataset>.json`.

### 7.3 Source adapters and verify-on-mount bindings

Both releases were inspected read-only on September 28, 2026 [LOCAL]. The binding files record `documented_on`, `observed_on_mount` and each item's `expected`, `bound` and `note`; items not yet re-observed inside the container stay `verify_on_mount` until M2.

**ACDC adapter (`src/tmmri/ingest/acdc.py`; binding `configs/source_bindings/acdc.json`).**

| Item | Publicly documented or locally observed expectation | Binding status |
|---|---|---|
| Release files | `database/{training,testing}/patientNNN/` with `Info.cfg`, `patientNNN_4d.nii.gz`, `patientNNN_frameXX.nii.gz`, `patientNNN_frameXX_gt.nii.gz`; 1,056 files, 2,456,816,223 B; plus `MANDATORY_CITATION.md` and licence terms | observed; SHA-256 per file in the tracked release manifest |
| `Info.cfg` | Keys `ED`, `ES`, `Group`, `Height`, `NbFrame`, `Weight` | observed; `ED`/`ES` are 1-based frame numbers matching the `_frameXX` suffix; stored as 0-based `ed_frame_index` = ED − 1 |
| Labels | 0 background, 1 RV cavity, 2 myocardium, 3 LV cavity (ACDC evaluation page) | observed; swapped 1 ↔ 3 on load to the canonical convention |
| Geometry | Frame files: unit-scaled affines (`sform_code` 2), spacing in header `pixdim`; 4D files mixed (36 voxel-scaled, 114 unit) | observed; rule below; the 4D file is not read by the primary pipeline |
| Rules | ED and ES frames from `Info.cfg`; reference masks at exactly those frames; every patient has one study; `Group` is the dataset label | — |

**M&Ms adapter (`src/tmmri/ingest/mms.py`; binding `configs/source_bindings/mms.json`).**

| Item | Publicly documented or locally observed expectation | Binding status |
|---|---|---|
| Release files | `<split>/<CASE>/<CASE>_sa.nii.gz` (4D cine) and `<CASE>_sa_gt.nii.gz` (4D mask) for 345 cases; information CSV `211230_M&Ms_Dataset_information_diagnosis_opendataset.csv` (14,139 B), byte-identical to the tracked `data/raw/mms/211230_MnMs_Dataset_information_diagnosis_opendataset.csv` | observed; SHA-256 per file in `BACKUP_MANIFEST.sha256`; the tracked `registry/sources/mms_full_release_manifest.csv` carries identifiers, release split, relative paths, sizes, SHA-256 and eligibility, exclusion, duplicate and MMS_100-membership flags only (vendor and centre are read from the mounted information CSV at run time) |
| CSV | 12 columns: unnamed index, `External code`, `VendorName`, `Vendor`, `Centre`, `ED`, `ES`, `Age`, `Pathology`, `Sex`, `Height`, `Weight`; 345 rows, no duplicate identifiers | observed |
| Frames | `ED`, `ES` are 0-based indices into the 4D time axis | observed; acceptance test 7 |
| Labels | 1 LV cavity, 2 myocardium, 3 RV cavity | observed; identity map |
| Mask encoding | float32 integer-valued (213) or uint8 with `scl_slope` 3/255 (132) | observed; `np.rint` before comparison; acceptance test 6 |
| Geometry | Voxel-scaled affines in all 345 cases (`sform_code` 1, `qform_code` 0); mask geometry equals image geometry | observed |
| Rules | Eligible (`labelled_320`, `configs/cohorts.yaml`) iff release split ∈ {Training/Labeled, Validation, Testing}, CSV ED ≠ ES, both indices lie inside the frame count and the mask is annotated on exactly those frames; structure presence, EDV > ESV and the other critical checks are QC (Section 12.4), and all 320 pass them [LOCAL]; Training/Unlabeled → `MMS_RELEASE_UNLABELED`, and the four duplicate cases also carry `MMS_DUPLICATE_CASE`; `Age` and `Sex` are carried in the normalized ledger for flow reporting only and never enter any table used for fitting or stratified inference | — |

**Geometry rule (commit `8ed5a40`) [PRESPEC].** For each staged 3D frame: if the affine's column norms equal the header zooms (tolerance 1e-4 mm), take `dx`, `dy`, `dz` from the affine (`z_spacing_method = nifti_affine_projection`); if the affine is unit-scaled while `pixdim` differs from 1, take them from `pixdim` (`nifti_header_pixdim_unscaled_affine`, informational QC `z_spacing_header_pixdim_fallback`); a non-positive, non-finite or unavailable spacing is critical (`INPUT_GEOMETRY_INVALID`). The slice increment is `dz`; slice thickness and gap are recorded when present and never substituted for the increment.

### 7.4 Construction kernel contract

The kernel ports the measurement code of `src/tmmri/semantics/measurements.py` and `semantics/features.py` under the 6.0.0 registry; `src/tmmri/semantics/contract.py` validates B against the registry.

- `verify_and_stage(dataset, source_root, manifest, cache_root) → StagingRecord` (`ingest/staging.py`): reads each source file once, streams its SHA-256 and compares it with the tracked manifest; for a match, writes only the ED and ES image and mask frames as uncompressed 3D NIfTI to the content-addressed cache and records source hash → staged hash; a mismatch aborts that case with `SOURCE_HASH_MISMATCH`. Re-staging skips a case whose staging record matches. Runs never re-hash raw data; they verify staged hashes (acceptance test 16).
- `resolve_geometry(img) → (dx, dy, dz, method)`: the geometry rule of Section 7.3 (acceptance test 5).
- `select_frames(case) → (ed, es)`: ACDC from `Info.cfg`, M&Ms from the CSV; requires the annotated mask frames to equal {ED, ES}; rejects placeholders (acceptance test 7).
- `load_masks(case) → (mask_ed, mask_es)`: `np.rint`, canonical label map, error on unexpected labels (acceptance test 6).
- `measure(case) → (row, provenance)`: block A–D features with states; a missing structure at either phase sets the dependent cells `QC_BLOCKED` and the row's severity critical (Decision D21).
- `build(dataset, run_id)`: fails on mixed versions, duplicate identifiers, unfrozen bindings, partition leakage or a missing normalized ledger; writes Parquet with the registry schema and `status = BUILT_NOT_YET_ANALYSIS_ELIGIBLE` until Section 7.5 passes.
- `self_test()`: synthetic cylinders and ellipsoids with known volumes under both affine conventions, a uint8-scaled mask, a planted missing structure, and the registry count. Synthetic objects never enter cohort tables.

Per-case work runs in parallel (joblib, `loky` back end) with one BLAS thread per worker; outputs are sorted by `sample_id` and must be byte-identical for any worker count (acceptance test 15).

### 7.5 Mandatory validation before promotion

| Test | Acceptance condition |
|---|---|
| Source completeness | Every ACDC patient (150) and every M&Ms case (345) appears exactly once in the cohort ledger or the exclusion ledger (M&Ms: 320 + 25) |
| Hashes | Every source hash equals the tracked manifest; every staged file equals its staging record |
| Geometry | Every frame has a recorded `z_spacing_method`; ACDC volumes are in the physically plausible order of magnitude (no dz = 1 mm artefact: the build fails if any ACDC frame is measured with a unit slice increment while `pixdim` differs) |
| Frames and labels | Annotated frames equal {ED, ES}; labels 1–3 present at both phases or the row is critical; no fractional label survives rounding |
| Semantic nulls | No unsupported, missing-source or not-applicable cell holds a value; BSA cells are `MISSING_SOURCE` without height |
| Label integrity | `Y` recomputed independently from `Info.cfg` and the CSV equals the exported labels; class-by-partition counts published before any fitting |
| Schema and provenance | Every registry feature present; every cell has state and provenance; counts equal the registry |
| Round trip | Parquet write and read preserve identifiers, nulls, states, types and the row-order hash; a shuffled A table is rejected |
| Leakage | The A builder has no access to labels, B or predictions; no label or file-name string enters A |

### 7.6 No-access behaviour

If a source is not mounted or fails its integrity gate, the kernel writes `B_<dataset>.schema.json`, header-only previews and `artifact_status.json` with `BLOCKED_DATA_ACCESS`; it never writes a file named `B_<dataset>.parquet`, and synthetic self-test objects never appear in any cohort artifact.

---

## 8. Formal matrix A and the audited predictor

### 8.1 Formal representation A (Decision D9)

**Definition.** `formal_descriptors_v3` = the 57 unprojected descriptors of `native_deterministic_baseline_v2` (implemented in `src/tmmri/formal_features.py`, formerly `a_extraction.py`). For each phase (ED, ES) the image is normalized within the image to its 1st and 99th intensity percentiles and clipped to [0, 1]; for each canonical label 0–3 the descriptor records the mean, standard deviation, median, 25th and 75th percentiles of the normalized intensity inside the label and the label's voxel fraction of the image (6 × 4 × 2 = 48); then, for labels 1–3, the ED fraction, the ES fraction and their difference (9). **Row:** one patient (ACDC) or case (M&Ms). No resampling, no physical units, no learned parameters.

**Known structure [LOCAL, code inspection].** The nine label-fraction terms repeat six voxel-fraction columns of the 48 exactly and add three differences of them; in addition the four label fractions of each phase sum to one. The 57 columns therefore have rank at most 46. This is harmless for ridge and is reported with the condition diagnostics; it is not a reason to change D9. An empty label yields zeros for its six statistics, but such a row is critical QC and never enters fitting (Decision D21).

**Standardization.** Zero-variance columns on the fit partition are removed (retained indices `K`); the remaining columns are standardized with fit-partition means and SDs (Section 9.1). No PCA is applied [PRESPEC; `configs/transition.yaml`, `pca: none`].

**Continuity comparator.** The P6/v5 128-d representation: the same 57 descriptors multiplied by a Gaussian random matrix (seed 20260611, scale 1/√57), with its first 32 columns overwritten by the first 32 raw descriptors. It is an ablation only (Section 13.3). **Semantic-proxy A** (the v4 metadata and hashed pseudo-features mode) is forbidden in v6.

**Mask conditioning (Decision D7).** A uses the reference masks, so it encodes the same segmentation that defines B. Reconstruction quality therefore measures how well intensity-and-label-fraction descriptors, which carry no spacing, predict physical quantities; it is not evidence about any image-only model.

**Verification.** Exactly 57 finite values per included object; `A`, `B` and `Y` identifier vectors agree element-wise; toggling any downstream setting leaves A's hash unchanged.

### 8.2 Audited predictor (Decision D5)

**Model [PRESPEC; `configs/formal.yaml`, `audited_predictor`].** Multinomial logistic regression (scikit-learn `LogisticRegression`, solver `lbfgs`, L2 penalty, `C = 1.0` fixed with no hyperparameter search, `class_weight = balanced`, `max_iter = 10000`, `tol = 1e-8`, `random_state = 20260611`) on standardized A, classes {DCM, HCM, MINF, NOR, RV}. The final predictor is fitted on all 98 development patients and frozen; `d_model` is its argmax decision, ties broken by the fixed class order. Fold predictors refitted without a held-out outer fold supply the out-of-fold development predictions and the held-out consequent of the cross-fitted rule gate (Section 11.6). No recalibration is applied; calibration is evaluated (ECE) and never refitted externally.

**Outputs stored, never part of A:** class probabilities, `d_model`, the out-of-fold development predictions and the model card with coefficients.

**Semantic comparator.** A multinomial logistic regression with the same settings on measured B transition candidates (fit-partition median imputation and standardization; `comparator_predictor`) is reported beside the audited predictor; it is never audited by the rulebooks.

**Evaluation.** OOF on `acdc_dev`, then `acdc_test`; on `mms_external` only for crosswalk-evaluable cases (Section 12.3).

### 8.3 Extraction and freeze

A and the predictor outputs are written as Float64 Parquet with `sample_id` first (`formal/A_ACDC.parquet`, `formal/A_MMS.parquet`) plus an ordered-identifier hash. `freeze-analysis-bundle` writes the stage record `artifacts/<run_id>/.stages/FROZEN.json` containing the hashes of the standardizers, the predictor, the transition, the decoders, the thresholds, the rulebooks and the configuration; any later write to those objects is refused.

### 8.4 Optional deep-encoder branch (Decision D20)

A frozen, image-only torchvision ResNet-50 with `ResNet50_Weights.IMAGENET1K_V2` (weight-file SHA-256 recorded at first use) supplies an alternative A: every 2D short-axis slice of the ED and ES frames is normalized per slice to its 1st/99th percentiles and clipped to [0, 1], resized to 224 × 224 (bilinear), replicated to three channels and ImageNet-normalized; the 2048-d global-average-pooled penultimate vectors are averaged over all slices of both phases to one vector per patient (`configs/deep_encoder.yaml`). No mask enters this branch. It produces its own standardization, transition and alternative audited predictor (same settings as Sections 8.2 and 9) under `artifacts/<run_id>/deep_encoder/`, runs in the `env-gpu` container target with deterministic algorithms, FP32 and batch 32, and never mixes with primary artifacts. The module `src/tmmri/deep_encoder.py` does not exist yet; its tests skip with a uniform message until it does.

*Amended in 6.0.1 (Decision D29):* D20 becomes a registered comparator, executed exactly as frozen above.
- **Values and status:** no value changes other than the D4 symmetrization below. Its status becomes implemented: `src/tmmri/deep_encoder.py`, with the stages `verify-torch-weights`, `extract-deep-features` and `fit-deep-encoder`.
- **Weights:** the weight files and their SHA-256 are listed in `registry/weights.yaml`. They are downloaded once from download.pytorch.org (`resnet50-11ad3fa6.pth`, and `resnet18-f37072fd.pth` for F9), with the owner's approval of October 2, 2026.
- **Preprocessing:** D20 keeps the preprocessing of this section and does not read the F9 tensor cache (Section 8.6).
- **Partitions:** features are extracted for `acdc_dev`, `acdc_test`, `mms_external` and, after the tag and the D20 freeze, `mms2_external`.
- **Downstream:** the v6 standardization, ridge T and audited-predictor settings with the v6 outer folds. This gives out-of-fold development results, the out-of-fold `d20_predictor` and a frozen bundle that is evaluated once per partition.
- **Status of results:** estimation only (Section 13.3); they never feed the primary claim or the primary rulebook.

*Amended October 3, 2026 (owner issues O1 and O3):*
- **Orientation (O3, stated explicitly):** before normalization, every slice is reoriented in plane by `tmmri.neural.preprocess.canonical_orientation` of the staged ED image header, the frame of the F9 tensor cache. It is a lossless transpose and flip: columns point toward patient left, rows toward posterior, and the stored slice order is kept.
- **Resize (O3, stated explicitly):** after the per-slice normalization, `torchvision.transforms.v2.functional.resize` to 224 × 224, bilinear, `antialias=True`, on CPU float32, then clipping to [0, 1].
- **Reason for the D4 symmetrization (O1):** after `canonical_orientation` the mask-derived direction from the LV to the RV centroid on a mid-ventricular ED slice is consistent within ACDC (all 98 `acdc_dev` cases have orientation code `rows=+1,cols=+0`, median angle −144°, RV left of LV in 100 %) but not within M&Ms-1 (`rows=+1,cols=+0`, 193 cases, median −146°; `rows=+0,cols=-1`, 80 cases, +128°; `rows=-1,cols=+0`, 45 cases, +154°; 2 others; RV left of LV in 100 %), so about 40 % of the M&Ms-1 slices are vertically mirrored relative to ACDC in the canonical frame (measured by the owner on the staged data on October 3, 2026; angles with y up). The v6 mask descriptors are orientation-invariant; only D20 and F9 are affected.
- **D4 symmetrization (O1):** D4 has 8 elements, the rotations by 0°, 90°, 180° and 270°, each with or without a transpose. Each element is an exact pixel permutation of the resized 224 × 224 input. The 2048-d patient feature is the Float64 mean of the FP32 slice vectors over all ED and ES slices and all 8 elements. No mask is used.
- **Configuration and records:** `configs/deep_encoder.yaml` `tta: d4_mean_8` (`src/tmmri/neural/dihedral.py`). The key is amendment-scoped: it is excluded from the v6.0.0 digest of the `deep_encoder` section (`src/tmmri/config.py`), and `extract-deep-features` and `fit-deep-encoder` hash `deep_encoder.tta` explicitly. The extractor record `deep_encoder/features/A_D20_<partition>_extractor.json` records `tta`, the orientation, the resize and the pooling rule. `tta: none` reproduces the 6.0.0 definition.

### 8.5 External extraction lock

`build-semantics --dataset mms`, `extract-formal --dataset mms` and `evaluate --split mms_external` refuse to run unless the tag `protocol-v6.0.0` resolves in the repository and the named run carries `.stages/FROZEN.json` whose hashes match the bundle; the only override, `--allow-unfrozen-synthetic`, accepts synthetic fixtures and nothing else [PROPOSED extension of D16 to B and A: no M&Ms quantity is computed before the ACDC bundle is frozen]. Staging and hash verification of M&Ms sources are allowed before M0. *Amended in 6.0.1:* the M&Ms-2 lock of Section 4.6 requires the tag `protocol-v6.0.1` for every M&Ms-2 command, including extraction, hash verification and staging; endpoint stages on `mms2_external` also require the frozen bundle, as above. The M&Ms-1 lock is unchanged.

### 8.6 Registered comparators of version 6.0.1 (Decision D29) [PRESPEC]

All four comparators are fitted on `acdc_dev` only, are estimation only (Section 13.3), never feed the primary claim or rulebook, and follow the freeze order of Section 14.6.

| ID | Definition | Evaluation and exposure label | Configuration and stages |
|---|---|---|---|
| F8 CART surrogate | A decision tree (depth ≤ 3, minimum leaf 5, balanced class weights, `random_state` 20260928) fitted on the out-of-fold B̂ of the registry rule-antecedent candidates against the out-of-fold `d_model`. Two versions: **full coverage**; and **selective**, which keeps only leaves passing the v6 gate (out-of-fold fidelity ≥ 0.90, ≥ 5 matches, Wilson 95 % lower bound ≥ 0.60) and lets the others abstain. Leaf fidelity is nested within the v6 outer folds: for each fold the tree is refitted on the inner nested OOF B̂ of the other folds, and a held-out row routed to final leaf L counts as a match of L only when the fold tree gives it L's label (reproduced in that fold, as the v6 gate counts matches only where a rule is reproduced); it agrees when L's label equals its out-of-fold `d_model`; matches and agreements are pooled over the five folds (`configs/cart_f8.yaml` `selective.nesting.match_rule`) | Frozen as a supplementary bundle (`comparators/cart_f8/`) before M&Ms-2 extraction. Prospective on `mms2_external`; registered after unblinding on `acdc_test` and `mms_external` | `configs/cart_f8.yaml`; `fit-cart-surrogate`, `evaluate-cart-surrogate` |
| D20 | Section 8.4, executed as frozen | Out-of-fold on `acdc_dev`; frozen bundle evaluated once on `acdc_test` and `mms_external` (registered after unblinding) and `mms2_external` (prospective) | `configs/deep_encoder.yaml`; `registry/weights.yaml`; `verify-torch-weights`, `extract-deep-features`, `fit-deep-encoder` |
| F9 trained CNN | NEURAL-TM-1.0 §4.1–4.2 verbatim (`docs/NEURAL_ENCODER_EXPERIMENT_PROTOCOL.md`), except for the seeds, the selection salt and the D4 symmetrization of October 3, 2026. See the list below this table | Fold encoders give the out-of-fold development predictions. The final encoder of each seed is frozen and evaluated once on `acdc_test` and `mms_external` (registered after unblinding) and `mms2_external` (prospective). The primary seed is the F9 of S1–S4 and F9b; the three seeds are reported as range and SD, with no best-seed choice and no ensembling | `configs/neural_input.yaml`, `configs/f9_cnn.yaml`, `registry/weights.yaml`; `build-tensor-cache`, `plan-f9`, `upload-cache`, `train-f9`, `import-f9-fits`, `select-f9`, `freeze-f9`, `infer-f9`; `scripts/cloud/modal_train.py` |
| F9b concept probe | Primary seed only. For each outer fold k, the 64-d activations of encoder k are standardized on its fitting patients, and the v6 typed ridge T (Section 9, `transition_fit`; same transforms, λ grid and inner CV) is fitted on them and decodes fold k, giving out-of-fold ρ and `ρ_j`. The final encoder with its own T, fitted on all 98, is the frozen probe | Out-of-fold on `acdc_dev`; frozen probe evaluated once on `acdc_test` and `mms_external` (registered after unblinding) and `mms2_external` (prospective) | `configs/probes.yaml`; `fit-probes`, `evaluate-alt-bundles`, `build-comparison-tables` |

**F9 design.**
- **Backbone and inputs:** ResNet-18 `IMAGENET1K_V1`; all ED and ES short-axis slices; two geometry scalars per slice.
- **Head:** attention and maximum pooling per phase; ED, ES and ED − ES concatenated; a 64-d bottleneck (the activations A); `Linear(64, 5)`.
- **Training:** class-weighted cross entropy, AdamW, at most 80 epochs with the early stopping of §4.2, and the §4.2 augmentation.
- **D4 augmentation (amended October 3, 2026; owner issue O1):** for each patient, one of the 8 in-plane D4 elements (Section 8.4) is drawn uniformly from the per-patient augmentation RNG. It is the first draw, before the angle, translations, intensity factor and noise. It applies identically to all slices of both phases, before the §4.2 rotation, translation, intensity and noise steps. The geometry scalars are unchanged, and determinism and resume stay bitwise. NEURAL-TM-1.0 §4.2 forbids reflections; here they enter only as D4 elements (`configs/f9_cnn.yaml` `augmentation.dihedral: d4_uniform_per_patient`).
- **Test-time symmetrization (amended October 3, 2026; owner issue O1):** every F9 prediction and activation averages the FP32 logits over the 8 D4 transforms of the ED and ES images, then takes the Float64 softmax. The 64-d activations A are averaged the same way. This covers `infer-f9`, the validation outputs used for selection and early stopping, S5, S6, the qualification and the cross-hardware check (`inference.tta: d4_mean_8`). No mask is used, and fit and inference records carry `tta`. The reason is the measured orientation inconsistency of Section 8.4.
- **Candidates:** six, the unfrozen blocks {`layer4`, `layer3+layer4`} × backbone learning rate {1e-5, 3e-5, 1e-4}.
- **Selection:** 3-fold class-stratified selection within each fitting set, with folds from the salt `f9-select-v6.0.1|` (key composition in `configs/f9_cnn.yaml`). Ranking: mean macro-F1, then mean cross entropy, then fewer unfrozen blocks, then the lower learning rate.
- **Refit:** on the whole fitting set for the rounded-up median of the selected epoch counts.
- **Fitting sets:** the five v6 outer folds and the final set of all 98.
- **Seeds:** 20261011 (primary), 20261012 and 20261013.
- **Fit count:** 6 fitting sets × (6 candidates × 3 selection folds + 1 refit) × 3 seeds = **342 fits**.
- **Technical-validity gate** (`configs/f9_cnn.yaml` `validity_gate`; a pre-evaluation bug check, never a tuning loop): before `freeze-f9` writes the bundle it requires, for the primary seed, (i) a development out-of-fold macro-F1 of at least 0.30 over all five classes (chance 0.20), from the `is_oof` predictions of the fold encoders recorded by `infer-f9 --split acdc_dev`, which must cover every `acdc_dev` patient exactly once; (ii) in every refit, a training loss at the final epoch below that of epoch 1 (a single-epoch refit fails); (iii) the recorded nonzero-change check of training (a nonzero change in a convolution of every unfrozen block; frozen weights and BN running statistics unchanged) in every selected fit, that is the refits and the selection fits of each winning candidate. On failure `freeze-f9` refuses (exit 7) with reason `PRE_EVALUATION_BUG_CHECK`, writes no bundle and writes the report `neural_f9/validity_gate.json` (also written, and hashed into the bundle, when the gate passes). The pre-evaluation bug rule of NEURAL-TM-1.0 applies: fix the code, invalidate the affected development artifacts and rerun them under recorded code changes. A failure never changes a hyperparameter, a seed or a value of the gate: `freeze-f9` refuses any `configs/f9_cnn.yaml` training configuration other than that of the wave-1 plan.

**F9 tensor cache** (`neural_cache/<partition>/`, written by `build-tensor-cache`).
- **Arrays:** float32 ED and ES slice arrays of 224 × 224 in [0, 1]. Each phase is clipped at its 1st and 99th percentiles and scaled, resampled bilinearly in plane to 1.5 mm, then centre-cropped or zero-padded; slice order is kept.
- **Geometry scalars:** the centred physical slice position divided by 100 mm and the corrected slice increment (Section 7.3) divided by 10 mm.
- **Manifest:** per-sample `npz_sha256`, crop fractions, in-plane spacing and transform version. The label column is empty wherever labels are not allowed.
- **Load-time steps:** grey-to-three-channel replication and the ImageNet mean and SD are applied at load time and never stored.
- **Masks:** no mask enters F9.

**Prediction contract.** Every classifier (`audited_predictor`, `d20_predictor`, `f9`) writes `predictions.parquet` in its model directory. Each row holds `sample_id`, `cluster_id`, `partition`, `model_id`, `seed`, `encoder_id`, `is_oof`, `y_true`, `y_pred`, `prob_<CLASS>` for the five classes in registry order, and `logit_<CLASS>`. Activations are stored as `activations.parquet` with the same keys.

**Comparative estimands (estimation only).**
- **Per representation** (descriptors, D20, F9b): ρ and `ρ_j`.
- **Per predictor** (audited predictor, D20 predictor, F9): accuracy, macro-F1, macro one-versus-rest AUROC, ECE (as Section 13.2: 10 equal-mass bins, the v6.0.0 metric, is primary for every predictor; ECE with 10 equal-width bins is reported beside it as a companion; owner decision of 3 October 2026), Brier score and log loss.
- **Differences:** paired cluster-bootstrap differences against the descriptor pipeline (2,000 replicates, seed 20260928).
- **M&Ms class metrics:** only on the same-definition crosswalk subsets (M&Ms-1: the 273 cases of Section 12.3; M&Ms-2: NOR + HCM).

**Firewall.**
- Latent coordinates are never pooled across encoders.
- Every analysed activation and every reported F9 and F9b number is computed by local inference (Section 14.7).
- F9 results cannot change the frozen design of NEURAL-TM-1.0, and no rule is fitted on CNN decisions in v6.

---

## 9. Transition-matrix mathematics and algorithms

### 9.1 Dimensions, fitted coordinates and typed transforms

Let `n` be the number of aligned objects, `k = 57` the formal dimension, `l` the number of registered numeric targets and `M ∈ {0,1}^{n×l}` the reference-observation indicator (`M_ij = 1` iff the state is `OBSERVED` and no high-severity QC mask applies). Metadata, states, tiers, anthropometrics and reserved columns never enter the numeric projection.

\[
A \in \mathbb{R}^{n\times k},\quad B \in \mathbb{R}^{n\times l},\quad M \in \{0,1\}^{n\times l},\quad T \in \mathbb{R}^{k\times l}. \tag{E2}
\]

**Formal preprocessing (fit rows only; frozen).** Remove zero-variance columns (`K`, `k_K = |K|`) and standardize with fit means `μ_A` and SDs `D_A = diag(s_A)`:

\[
Z = (A_K - \mathbf{1}\mu_A^\top) D_A^{-1} \in \mathbb{R}^{n\times k_K}. \tag{E3}
\]

**Typed target transforms (fit on observed fitting rows; frozen) [PRESPEC].**

| Transform | Registry dtypes | `g_j(x)` | Inverse | Standardize |
|---|---|---|---|---|
| `log` (strictly positive volumes, SV, mass and cavity areas) | mL, g, mm², domain (0, ∞) | `ln x` | `exp(z)` | fit mean and SD of `ln x` |
| `logit_percent` (primary for EF) | %, domain [0, 100] | `logit(clip(x/100, 0.001, 0.999))` | `100 · sigmoid(z)` | fit mean and SD |
| `identity` (lengths, percent changes that may be negative, dimensionless shape ratios) | mm, %, ratio | `x` | `x` | fit mean and SD |
| `identity_sensitivity` (every candidate target as `identity`; D10 sensitivity) | all | `x` | `x` | fit mean and SD |

\[
y_{ij} = \frac{g_j(B_{ij}) - \mu_{B,j}}{s_{B,j}},\qquad i \in O_j = \{i : M_{ij} = 1\}. \tag{E4}
\]

A target with fewer than 10 observed fitting rows is excluded from fitting with a reason [INHERITED, P5]; a constant target is retained as descriptive and never standardized by an artificial denominator. No B value is imputed (per-target observation masks replace the v5 median imputation). Fitting targets are winsorized at their 1st/99th fit-row percentiles in transformed space, on fitting rows only; evaluation always uses the unclipped reference [PRESPEC; `configs/transition.yaml`]. A non-finite A value is an extraction failure (`INPUT_NONFINITE`) and excludes the row.

### 9.2 Historical pseudoinverse and the preferred masked ridge

**Historical baseline [INHERITED, P1–P2].** The minimum-norm least-squares solution (`numpy.linalg.pinv`, λ = 0) on the same standardized A and transformed, standardized targets; never invert normal equations.

**Masked ridge with an unpenalized intercept [PROPOSED correction of P4/P5/P6].** For each target `j`, with `z̄_j = mean(Z[O_j])`, `ȳ_j = mean(y[O_j, j])`, `Z_{c,j} = Z[O_j] − 1 z̄_jᵀ`, `y_{c,j} = y[O_j, j] − ȳ_j`:

\[
(w_j, c_j) = \arg\min_{w, c}\ \sum_{i\in O_j}(z_i w + c - y_{ij})^2 + \lambda_j\|w\|_2^2. \tag{E5}
\]

With the thin SVD `Z_{c,j} = U_j Σ_j V_jᵀ`,

\[
w_j = V_j\,\operatorname{diag}\!\left(\frac{\sigma_a}{\sigma_a^2 + \lambda_j}\right) U_j^\top y_{c,j},\qquad c_j = \bar y_j - \bar z_j w_j. \tag{E6}
\]

The sum-loss convention of P6 and v5 is kept (`configs/transition.yaml`, `objective_convention: sum_loss`, which is `(ZᵀZ + λI)⁻¹ZᵀY` on centred standardized A), so λ is not scaled by `|O_j|`; under a mean-loss objective the same fit corresponds to `λ/|O_j|`. Singular values below `max(|O_j|, k_K)·ε_machine·σ_1` are treated as zero. Stacking `W` and `c`,

\[
\widehat Y = Z W + \mathbf{1}c^\top,\qquad T_K = D_A^{-1} W \in \mathbb{R}^{k_K\times l},\qquad \widehat B_{ij} = g_j^{-1}\big(s_{B,j}\widehat Y_{ij} + \mu_{B,j}\big). \tag{E7}
\]

The raw-coordinate operator has zeros in dropped columns and composed intercept `c − μ_Aᵀ T_K`; use either form, never both corrections. With a nonlinear `g_j⁻¹` the map is affine in transformed coordinates, not in physical units. Persist `W`, `c`, the preprocessing objects, the transform parameters, `K`, `T_K` and the composed intercept (`transitions/T_ACDC_ridge.parquet`, `.preproc.npz`, human-readable JSON).

**Salience diagnostic [INHERITED, P4, descriptive only].** `η_j = ‖T_{K,:j}‖₂` is reported; it never decides eligibility.

### 9.3 Leakage-controlled nested fitting algorithm [PRESPEC]

```text
fit_transition(acdc_dev_ids, A_frozen, B, M, registry):
    verify identical A/B/Y identifier vectors, source and staged hashes, masks, QC severities
    drop critical-QC rows from fitting; apply high-severity feature masks to M
    outer folds f = 1..5 and inner folds g = 1..5 from configs/splits.yaml (hash rules, Section 12.2)
    for each outer fold f:
        for lambda in numpy.logspace(-6, 4, 21):                       # 1e-6 ... 1e4, half-decade steps
            for each inner fold g of outer-train(f):
                fit variance filter, standardizer, target transforms and winsorization on inner-train(g)
                solve (E5)-(E6) per target on inner-train observed rows
                score MAE in physical units per target on inner-validation observed rows
        lambda_j*(f) = argmin of the mean inner MAE; ties -> larger lambda
        flag LAMBDA_AT_GRID_EDGE if lambda_j*(f) is 1e-6 or 1e4
        refit on outer-train(f); predict outer-held-out rows; store the inner-OOF predictions of outer-train(f)
        write OOF predictions, masks, fold and fitting-identifier hashes to the OOF ledger
    evaluate eligibility (Section 11.2) from the OOF ledger only
    select lambda_j by 5-fold CV over the outer folds on all 98; refit the single frozen transition
    predict acdc_test (and, after M0 and .stages/FROZEN.json, mms_external) with no fitting
```

A grid-edge selection is never silently accepted: it is kept (the grid is frozen and never extended post hoc), recorded per target and fold in `transitions/lambda_selection.parquet` and reported in every table that uses the target. A single global λ chosen by pooled normalized MAE is a named sensitivity.

### 9.4 Required comparators and the frozen linearity criterion

Fit on identical inputs, rows and masks: (a) the train-mean null predictor (`B̂_ij = b̄_j`, the fit-partition mean in physical units; D10); (b) the historical pseudoinverse; (c) masked ridge with typed transforms and per-target λ (primary); (d) the same ridge with every transform replaced by `identity` (`identity_sensitivity`); (e) ridge on the 128-d P6 projection (`p6_projection_ridge`); (f) ridge with one global λ, the grid value minimizing the mean over targets of the inner-CV MAE divided by each target's fit-row SD, ties to the larger λ (`configs/transition.yaml` `comparators.ridge_global_lambda`) [PRESPEC]; (g) a nonlinear comparator for the linearity criterion: RBF kernel ridge regression on Z with the typed transforms of the primary ridge, the bandwidth set by the median pairwise-distance heuristic on the fitting rows (fallback 1.0 when that median is zero) and a per-target penalty chosen on the same inner folds and grid, with an unpenalized mean intercept (`comparators.rbf_kernel_ridge`) [PRESPEC]. No comparator reads test or external losses during selection. `fit-transition --nested-oof` writes the out-of-fold ledgers of (a), (b), (d), (f) and (g) (`transitions/comparator_*.parquet`, same rows, masks and folds as the primary); `run-ablations` refits every comparator on all 98 for `acdc_test` and evaluates the criterion below (`metrics/ablations/linearity_criterion.json`).

**Frozen decision criterion [INHERITED from P5, with the core subset defined here; `configs/transition.yaml` `linearity_criterion`].** The nonlinear comparator is declared *better* only if it achieves ≥ 2 % relative reduction in pooled normalized MAE (the mean over transition targets of MAE_j / s_j on the fold's out-of-fold rows) in ≥ 4 of 5 outer folds **and** no primary-family target degrades with a paired CI excluding zero (the 2,000-replicate paired patient-bootstrap 95 % interval of the normalized MAE difference, comparator minus ridge, lies entirely above zero). The outcome is reported either way; **the executed audit pipeline remains affine**, and adopting a nonlinear transition requires a new frozen protocol version.

### 9.5 Decoders

- `log` and `logit_percent` inverses keep reconstructed volumes, mass, areas and EF inside their domains by construction.
- Domain guard [PRESPEC]: a reconstruction outside the registry domain by more than 5 % of the fit-partition range is `NUMERIC_DOMAIN_BLOCKED` (`FEATURE_DOMAIN_BLOCKED`) for symbolic use; within the guard the decoded value is clipped to the domain; raw values are kept for error metrics and every block is logged. `evaluate` writes the raw reconstructions (`reconstructed/bhat_<split>.parquet`, the substrate of every error metric), the decoded values with their reconstructed states (`reconstructed/bhat_<split>_decoded.parquet`, the routing substrate) and every block and clip per target (`metrics/decoder_<split>.json`); the refit arm and the acdc_test routes of `run-ablations` use the same decoders.
- **Derived-consistency diagnostic [PROPOSED].** Reconstructed EF is compared with `100 · (EDV̂ − ESV̂)/EDV̂` from reconstructed volumes, and reconstructed SV with `EDV̂ − ESV̂`; discrepancies are reported, never enforced.
- Decoders never conceal transition failure.

### 9.6 Per-feature evaluation

For every target `j` and evaluation partition `Q` on observed rows: MAE, RMSE, median absolute error, signed bias, fit-SD-normalized MAE and RMSE, Pearson and Spearman correlation, Lin's concordance, Bland–Altman bias and 95 % limits of agreement for the primary and key secondary targets (descriptive method-comparison plots), and `ρ_j`. Mixed-unit means across targets are descriptive only [INHERITED, P6]. Every metric carries a cluster bootstrap CI (Section 13.1), and a failed-feature manifest lists every `NE`.

---

## 10. Equivariance and ETM branch (optional; Decision D19)

### 10.1 Scope and admissible actions [PRESPEC; `configs/equivariance.yaml`]

The branch runs only after every mandatory endpoint is complete; it is a diagnostic, never a gate, and never alters the primary rulebook. It uses actual transformed images **and masks**, because A depends on both; B is re-measured on the transformed masks.

| Transformation | Policy | Semantic action and limitation |
|---|---|---|
| In-plane translation | ±1 px in x and in y; image linear, mask nearest-neighbour interpolation; zero padding | Volumes, mass, EF and wall thickness nominally invariant; whole-pixel shifts are exact unless structures reach the border |
| In-plane rotation | ±2° about the slice centre; image linear, mask nearest-neighbour; zero padding | Nominally invariant; voxel counts change through interpolation, so the measured change of B is the reference action |
| Intensity scaling | factors 0.95 and 1.05 on the image; mask unchanged | B unchanged exactly; A nearly invariant because of within-image percentile normalization |
| Flips, crops, elastic deformation, through-plane actions | Excluded | Not semantics-preserving or not defined for a short-axis stack |

**Case selection.** `numpy.random.default_rng(20260611).choice(sorted acdc_dev patient_ids, size=12, replace=False)`, recorded in `equivariance/selection.json` before any transform is applied (this replaces the v5 `head()` selection). M&Ms cases enter this branch only after M0.

### 10.2 Generators

With actual transformed inputs `g_i(±ε)x`, formal and semantic generators are estimated by central differences

\[
\dot a_i(x) \approx \frac{a(g_i(+\varepsilon)x) - a(g_i(-\varepsilon)x)}{2\varepsilon} \approx J^A_i\, a(x),\qquad
\dot b_i(x) \approx \frac{b(g_i(+\varepsilon)x) - b(g_i(-\varepsilon)x)}{2\varepsilon} \approx J^B_i\, b(x), \tag{E8}
\]

with steps in physical units (1 px; 2° in radians; 5 % intensity), `a` the standardized formal vector and `b` the standardized primary-family targets measured on the transformed masks. Each generator is fitted by least squares on the selected cases and reported with its residual [INHERITED, P3 Supplement Eq. 4]

\[
\eta_i = \frac{\|\Delta A_i - A (J^A_i)^\top\|_F}{\|\Delta A_i\|_F}. \tag{E9}
\]

P3's step sizes (0.1 rad in the main text, 1e-4 in the released code) are not inherited. A generator-regularized ETM fit (P3 Eq. 17 with the corrected `√γ` stacking) is **not configured** in v6; running it requires an amendment that fixes its γ grid and Pareto rule before any result is seen.

### 10.3 Metrics

The held-out finite-action defect compares the reconstructed action with the measured action on actually transformed inputs:

\[
D_g = \frac{\sum_x \|(\widehat b(gx) - \widehat b(x)) - (b(gx) - b(x))\|_2^2}{\sum_x \|b(x) - \bar b\|_2^2 + \varepsilon_0}, \tag{E10}
\]

on standardized primary-family targets, reported per transformation and target together with the generator residuals and the measured reference change `‖b(gx) − b(x)‖`. The v5 figures (48 runs; rotation more disruptive than translation) are legacy exploratory results and are not reused.

---

## 11. WEDD, rough sets, rules, conflict, soft matching, and abstention

### 11.1 Rulebooks, consequents and anchors

| Rulebook | Directory | Antecedent substrate | Consequent | Purpose |
|---|---|---|---|---|
| **Primary: model fidelity** | `rules/d_model/` | Nested-OOF reconstructed B̂ of the registry's `primary_rulebook_antecedent_candidates` that pass Section 11.2: the seven primary-family targets and the two key secondary targets (never EF together with both volumes of the same ventricle) | `d_model` | SEMTRA fidelity audit of the frozen predictor (Decision D13); routable on every technically eligible case |
| Companion: dataset label | `rules/dataset_label/` | As the primary rulebook, with its own WEDD thresholds supervised by the label | ACDC `Group` (`Y_acdc_class`) | Label-semantics audit; caveat "dataset label, not diagnosis" |
| Oracle: measured B | `rules/measured_oracle/` | Measured **B** of the same features, with its own WEDD thresholds fitted on measured values | `d_model` | Isolates the transition-induced loss (P6 analogue) |

No threshold, rule or result migrates between rulebooks. No rulebook is selected after inspecting test performance. Every development patient is one anchor (one study per patient), so no anchor selection is needed; the anchor manifest `rule_anchor_manifest.parquet` is still written, before any label or prediction is read, to fix the universe and its fold membership. There is no feature budget: every eligible candidate (at most 9) enters discovery.

### 11.2 Eligibility before and after discretization [PRESPEC; `configs/rule_policy.yaml`]

| Gate | Prespecified engineering condition |
|---|---|
| Definition and provenance | Valid registry definition, source binding, domain and units; no unsupported prerequisite |
| Reference availability | Observed reference on ≥ 80 % of the development patients |
| Variability | ≥ 3 distinct observed values; fit SD > 0 |
| Reconstruction | OOF point estimate improves MAE over the train-mean predictor by ≥ 5 % with improvement in ≥ 4 of 5 outer folds (applies to the `d_model` and `dataset_label` rulebooks) |
| Post-discretization stability | ≥ 60 % bootstrap recurrence of each retained cut (Section 11.4); applied per candidate feature: a feature with no cut or with any cut below 0.60 is ineligible (`INELIGIBLE_NO_CUT`, `INELIGIBLE_UNSTABLE_CUT`), so the fold and final rules keep comparable bin counts |
| Consequent support | ≥ 5 development patients per represented consequent class; a class below it is `CLASS_INELIGIBLE` and a rulebook missing classes is labelled as such |
| No tautology | Metadata, labels and QC columns are never antecedents; any feature copied from or deterministically encoding the consequent is excluded; the dependency audit is stored |

`check-semantic-eligibility` evaluates these gates per rulebook after `discretize` and writes `rules/semantic_eligibility.{parquet,csv,json}` and the anchor manifest `rules/rule_anchor_manifest.parquet`; `induce-rules` and the cross-fitted gate use only the eligible features.

**Two state systems.** Reference **B** carries the states of Section 6.1. Reconstructed **B̂** carries `ESTIMATED`, `INPUT_QC_BLOCKED`, `TARGET_INELIGIBLE`, `NUMERIC_DOMAIN_BLOCKED`, `UNSTABLE_ESTIMATE`. At inference, guards inspect only the staged image and mask, permitted metadata, the frozen models and their outputs; they never read a reference B value or `Y`.

### 11.3 Discretization into symbolic states

**Order of precedence [INHERITED, P6].** (1) native symbolic states; (2) registry-defined cuts; (3) WEDD. The primary registry has no native Boolean targets and no clinical cutoffs are registered (Section 5), so every rule antecedent is discretized by WEDD; comparators run in parallel.

Symbolic state matrix `S ∈ 𝒱^{n×q}` with `S_ij = q_j(B̂_ij)` (or of `B_ij` for the oracle) and the extra state `?` for masked, blocked or ineligible cells.

#### 11.3.1 Exact WEDD profile `wedd_gaussian_mm_v1` [INHERITED from P4, with the P5 stability gate; Decision D11]

This is the CXR-TRANS-AUDIT profile of the same name, copied unchanged except that the entropy sums over the classes of the consequent (five ACDC classes for `d_model`) instead of a binary label.

Work on reconstructed training-anchor values from the OOF ledger after the fixed decoder; never discretize measured B and deploy those bins on differently distributed B̂ (the oracle learns its own thresholds). For one attribute and a current node containing `n` eligible anchor objects, candidates `Θ_j` are the midpoints between adjacent unique values; a cut `t` induces `L_t = {x : z_j(x) ≤ t}`, `R_t = {x : z_j(x) > t}`; require `max(10, ceil(0.02·N_anchor_train))` patients in both children; constant or missing-only attributes produce no cut.

\[
H(G) = -\sum_{y\in\mathcal Y} p_G(y)\log_2 p_G(y),\qquad H_j(t) = \frac{|L_t|}{n}H(L_t) + \frac{|R_t|}{n}H(R_t),\qquad 0\log 0 = 0. \tag{E11}
\]

Base 2 equals P4's base `N_c` for a binary decision; for the five-class consequent the candidate-wise min–max normalization below makes the selected cut independent of the base, while the gain threshold stays in bits. Reject candidates with information gain `H(node) − H_j(t) < 10⁻³` bits [PRESPEC]. With the Gaussian kernel `K(u) = exp(−u²/2)/√(2π)` and the robust bandwidth

\[
\widehat p_j(t) = \frac{1}{n h_j}\sum_{i=1}^{n} K\!\left(\frac{t - z_{ij}}{h_j}\right),\qquad h_j = 0.9\,\min\!\left(s_j, \frac{IQR_j}{1.349}\right) n^{-1/5}, \tag{E12}
\]

falling back to `0.9·s_j·n^{−1/5}` when the IQR is zero (logged) and refusing to split when the SD is zero. Compute `H_j` and `p̂_j` on the same valid candidate set, min–max normalize each over that set (a zero range maps to zero), and minimize

\[
J_j(t) = 0.65\,\widetilde H_j(t) + 0.35\,\widetilde p_j(t). \tag{E13}
\]

Ties are broken by the larger child minimum support, then the smaller `t`. Grow best-first with at most three bins and depth two per feature [PRESPEC; depth three in sensitivity, evaluated at four bins because a tree with at most three bins never exceeds depth two (`configs/wedd.yaml` `sensitivity_grid`)]; equal node priorities are broken lexicographically by node path. Evaluate Gaussian sums in chunks; a histogram approximation is allowed only as a separately named implementation with an equivalence test. Store each split node's valid candidates, `H`, density, their min–max normalizations, `J`, normalization extrema, bandwidth, supports, gain, selected cut, node path and fit hash in `thresholds/wedd_candidates.parquet` (one row per candidate and rulebook, written by `discretize`) and in the rulebook threshold JSON (`thresholds/<rulebook>_<profile>.json`). Bins are `(−∞, t₁], (t₁, t₂], …, (t_last, +∞)` with exact right-closed boundaries; unit tests check equality at every cut and at adjacent floating-point values.

With `N_anchor_train = 98` (78 or 80 inside an outer fold) the minimum child size is 10 patients.

**Comparators** under identical support and bin limits: `three_quantile_v1` (training tertiles with duplicate cuts removed; the P6/v5 continuity discretizer) and `wedd_uniform_mm_v1` (P5's uniform counting kernel, otherwise identical). WEDD is not assumed superior [P4 p. 17].

### 11.4 Bootstrap and per-object state stability [INHERITED, P5]

Refit thresholds on 500 cluster bootstrap resamples (Section 13.1) of `acdc_dev` with the identical configuration (seed 20260928); a cut recurs if a cut within the same feature and node path lies within `0.1 × training IQR` (fallback `0.1 × training SD`) of the original; absence of the node or cut counts as non-recurrence; retain a cut only if its recurrence is ≥ 0.60 [PRESPEC]. Report recurrence and numerical spread; recurrence is not clinical reliability.

Per-object stability: apply the five stored outer-fold transitions to the object's A vector, decode with the final quantizer, and require at least four of five valid predictions to agree with the final transition's symbol; otherwise the feature is `UNSTABLE_ESTIMATE` for that object and unavailable for exact matching (`FEATURE_UNSTABLE`). The outer-fold transitions are frozen with the bundle (`transitions/fold_models/`), are decoded with the same domain guard, and `evaluate` writes the per-object states of each reconstructed rulebook (`reconstructed/stability_<rulebook>_<split>.parquet`); the refit arm applies its own refit-fold transitions.

### 11.5 Rough-set decision table and variable-precision discovery (Decision D12)

The decision system is `𝒟 = (U, C, d, V, f)` with `U` the development anchors, `C` the eligible reconstructed symbol columns, `d = d_model`, and

\[
\operatorname{IND}(P) = \{(x, x') : s_j(x) = s_j(x')\ \forall j \in P\},\quad
\underline{P}(Y_y) = \bigcup_{[x]_P \subseteq Y_y}[x]_P,\quad
\overline{P}(Y_y) = \bigcup_{[x]_P \cap Y_y \neq \varnothing}[x]_P,\quad
BN_P(Y_y) = \overline{P}(Y_y)\setminus\underline{P}(Y_y). \tag{E14}
\]

Equivalence classes are formed by hashing a fixed ordered signature; missing or unavailable states cannot act as literals; for a fixed attribute subset the indiscernibility is computed on its complete-symbol sub-universe and excluded anchors are counted separately.

**Discovery.** Candidate conjunctions are enumerated exhaustively over every candidate subset of size 1 to `L_max` and every bin combination, so that the result is independent of column order (the v5 first-six defect); missing or unavailable states are never literals. A granule with majority class `y` is admitted as **variable-precision** if at most `β = 0.10` of its members disagree with `y` [INHERITED, P5], hence confidence ≥ 0.90, and its antecedent support is ≥ 5 anchor patients [PRESPEC]; it is **strict** if no member disagrees (confidence 1). For each candidate store `antecedent_n`, `joint_n`, `confidence = joint_n / antecedent_n` and class support from the matching rows; there is no unlabeled "support" field. Maximum length `L_max = 3` [PRESPEC]. Reduction: greedy backward deletion of antecedents while the discovery gate still holds, ties broken by larger resulting support, then registry order; identical rules are deduplicated and strictly redundant supersets with the same consequent are removed. This is a greedy constrained reduct, not a claim to the globally minimal reduct [INHERITED, P4]. No arbitrary cap is applied (the v5 cap of 200 is removed). Deterministic ranking: confidence descending, `antecedent_n` descending, length ascending, registry order, `rule_id`. Every rule carries `tier ∈ {STRICT, VARIABLE_PRECISION}`; only `STRICT` rules may be described as certain on development granules. A strict-only rulebook is a sensitivity. The positive-region diagnostic is reported separately.

### 11.6 Cross-fitted rule eligibility and provenance [PRESPEC; `configs/rule_policy.yaml`, `validation_gate`]

The 98 development patients cannot spare a separate gate partition, so the gate is cross-fitted over the five outer folds (Decision D12). Let `R_final` be the rulebook discovered on all 98 anchors from the nested-OOF ledger. For each outer fold `f`:

1. refit the transition, the audited predictor, the WEDD thresholds and rule discovery on the four training folds only (thresholds on their inner-OOF reconstructions: for each inner fold `g` of `outer-train(f)` the transition is refitted on the remaining inner folds, with λ by CV over them, and predicts fold `g`; `gate-rules --substrate nested_inner_oof`, the default, recorded in the stage record and in every rulebook's provenance; the ledger is `oof/inner_oof_ACDC_dev_fold<f>.parquet`), giving the fold rulebook `R_f`;
2. discretize the outer-OOF reconstructions of the held-out fold-`f` patients with the fold thresholds;
3. take as held-out consequent the `d_model` of the fold predictor, which never saw the held-out fold (the dataset label for the companion rulebook);
4. a rule `r ∈ R_final` is **reproduced** in fold `f` if `R_f` contains a rule with the same feature set, the same bin index per feature under the same bin count, and the same consequent; a bin-count mismatch means not reproduced in that fold; if reproduced, count the held-out patients it matches and those whose held-out consequent equals its consequent.

A rule is **eligible** iff its pooled out-of-fold matched patients are ≥ 5 and the Wilson 95 % lower bound of agreement on them is ≥ 0.60 [PRESPEC]:

\[
L_{W} = \frac{\hat p + \frac{z^2}{2m} - z\sqrt{\frac{\hat p(1-\hat p)}{m} + \frac{z^2}{4m^2}}}{1 + \frac{z^2}{m}},\qquad z = 1.96, \tag{E15}
\]

with `m` pooled matches and `p̂` their agreement. Worked values: 6 of 6 gives 0.61 (eligible); 5 of 5 gives 0.57 and 9 of 10 gives 0.596 (both ineligible); 10 of 11 gives 0.62. The effective minimum for a perfectly concordant rule is therefore six matches. The gate is applied once; gate bounds 0.50 and 0.70 are sensitivities. **Semantic support** (agreement between each antecedent's reconstructed symbol and the symbol of the measured reference under the same thresholds, on the pooled matches) is reported per antecedent with its Wilson interval as a diagnostic, not a gate [PROPOSED]; an antecedent whose reference is not `OBSERVED` is `REFERENCE_SEMANTIC_UNVERIFIABLE`. Rule selection completes before `acdc_test` or external inference; label accuracy is reported but not optimized.

Every rule record MUST contain (`rule_record_required_fields`): `rule_id` (content-derived), `rulebook`, `tier`, `consequent_endpoint`, `antecedents` (feature, bin, threshold digest, numeric bounds in physical units), `consequent`, `fit_antecedent_n_patients`, `fit_joint_n_patients`, `fit_confidence`, `crossfit_matched_n_patients`, `crossfit_agreement`, `crossfit_wilson_lower`, `crossfit_folds_reproduced`, `length`, `bootstrap_recurrence`, `eligibility` (`ELIGIBLE` or an explicit failure code), `caveat` ("model decision, not diagnosis" or "dataset label, not diagnosis"), and `provenance` with the digests of `A`, `B`, `T`, the OOF ledger, thresholds, the anchor manifest, the decision vector, the configuration and the code commit. If no rule survives, `rules/<rulebook>/EMPTY.json` records every failing gate.

### 11.7 Inference routes, soft matching and reason codes (Decision D14)

```text
infer(case, permitted_metadata, frozen_bundle):
    if staging, geometry, frames or critical QC prevent a valid A:      return QC_ABSTAIN(reason)
    compute the predictor output independently; it is never a rule-availability gate
    if the eligible rulebook is empty:                                  return EMPTY_RULEBOOK
    reconstruct B-hat; apply decoders, high-severity feature masks and per-object stability
    discretize with frozen thresholds; never read reference B or Y
    exact = eligible rules whose EVERY antecedent is available and equal
    if exact has exactly one distinct consequent:                       return EXACT_ACCEPT(consequent, rule ids)
    if exact has more than one distinct consequent:                     return CONFLICT_ABSTAIN(consequents, rule ids)
    candidates = soft candidates (below)
    if candidates:                                                      return SOFT_CANDIDATES_ONLY(candidates, mismatches)
    if some rule could not be evaluated because an antecedent symbol is unavailable:
                                                                        return UNAVAILABLE_FEATURE_ABSTAIN(features)
    return NO_MATCH_ABSTAIN(MATCH_NONE)
```

Routes partition every attempted input: `QC_ABSTAIN + EMPTY_RULEBOOK + EXACT_ACCEPT + CONFLICT_ABSTAIN + SOFT_CANDIDATES_ONLY + UNAVAILABLE_FEATURE_ABSTAIN + NO_MATCH_ABSTAIN = 1`, in the precedence of `configs/rule_policy.yaml` (`route_precedence`). Only `EXACT_ACCEPT` emits a label. There is no conflict resolution; a predictor disagreement with an exact rule is reported and never cancels the rule; neither the base decision nor any reference may act as a tie-breaker.

**Soft matching** [adapted from P5/P6]: with the full antecedent count `L_r`,

\[
\operatorname{sim}(x, r) = \frac{1}{L_r}\sum_{j\in R_r}\mathbf 1\{s_j(x)\ \text{available and}\ s_j(x) = v_{rj}\}. \tag{E16}
\]

Unavailable antecedents stay in the denominator. A soft candidate requires `L_r ≥ 3`, similarity ≥ 0.66 (two of three antecedents) and at most one mismatching or unavailable antecedent [PRESPEC]; rank by similarity, then discovery support, then `rule_id`; return up to five candidates with their mismatches. A soft match **never returns an accepted class**; no criticality weights are invented.

**Reason codes** (`registry/reason_codes.yaml`, version 6.0.0; the file is authoritative):

| Code | Scope | Meaning |
|---|---|---|
| `INPUT_DECODE_FAILURE`, `INPUT_NONFINITE` | input | Image or mask unreadable or header inconsistent; non-finite voxels or descriptors (→ `QC_ABSTAIN`) |
| `INPUT_GEOMETRY_INVALID` | input | In-plane spacing non-positive or non-finite, or slice increment unavailable (→ `QC_ABSTAIN`) |
| `INPUT_MISSING_ED_ES_FRAME` | input | ED or ES missing, out of range, ED = ES, or mask not annotated on that frame (→ `QC_ABSTAIN`) |
| `QC_MISSING_LV_BLOOD_POOL`, `QC_MISSING_RV_BLOOD_POOL`, `QC_MISSING_LV_MYOCARDIUM` | qc | Structure absent at ED or ES; critical (→ `QC_ABSTAIN`) |
| `QC_UNEXPECTED_LABEL_VALUE`, `QC_INVALID_LABEL_MAPPING`, `QC_LABEL_SMOOTHED`, `QC_MASK_OVERLAP` | qc | Label-map integrity failures; critical (→ `QC_ABSTAIN`) |
| `QC_LV_EDV_LE_ESV`, `QC_RV_EDV_LE_ESV`, `QC_LV_EF_OUT_OF_RANGE`, `QC_RV_EF_OUT_OF_RANGE`, `QC_MASS_NONPOSITIVE` | qc | Physically inconsistent measurements; critical (→ `QC_ABSTAIN`) |
| `QC_UNREGISTERED_REASON` | qc | A QC token not registered in `configs/cohorts.yaml`; fails closed (→ `QC_ABSTAIN`) |
| `FEATURE_TARGET_INELIGIBLE`, `FEATURE_QC_MASKED`, `FEATURE_DOMAIN_BLOCKED`, `FEATURE_UNSTABLE`, `FEATURE_MISSING_SOURCE` | eligibility | A required antecedent is ineligible, QC-masked, outside the guard, unstable, or (oracle only) missing (→ `UNAVAILABLE_FEATURE_ABSTAIN`) |
| `MATCH_MISSING_REQUIRED_SYMBOL` | match | No exact rule fired and a rule could not be evaluated for an unavailable symbol (→ `UNAVAILABLE_FEATURE_ABSTAIN`) |
| `MATCH_CONFLICTING_CONSEQUENTS` | match | Exact rules with different consequents fired; no label (→ `CONFLICT_ABSTAIN`) |
| `MATCH_SOFT_ONLY` | match | Only soft candidates exist; never a label (→ `SOFT_CANDIDATES_ONLY`) |
| `MATCH_NONE` | match | Every rule evaluable and none fired exactly or softly (→ `NO_MATCH_ABSTAIN`) |
| `RULEBOOK_EMPTY` | rulebook | No rule survived discovery and the cross-fitted gate (→ `EMPTY_RULEBOOK`) |
| `CLASS_INELIGIBLE` | rulebook | A consequent class failed its support gate (rulebook-level record) |
| `REFERENCE_OFF_ONTOLOGY`, `REFERENCE_NOT_EXACT_IHD_MINF`, `REFERENCE_SEMANTIC_UNVERIFIABLE` | evaluation | Evaluation-only annotations; never change a route |
| `MMS_RELEASE_UNLABELED`, `MMS_DUPLICATE_CASE` | external | Excluded M&Ms release cases (Section 4.3) |
| `SOURCE_HASH_MISMATCH`, `EXCLUDED_UPSTREAM` | flow | Staging refused (run `BLOCKED_DATA_ACCESS`); removed by a manifest before inference |

### 11.8 CMR pattern templates (descriptive; no numeric alignment score)

`registry/pattern_templates.yaml` lists research templates written by the research team from the registry vocabulary (no clinician authorship; descriptive only). Conditions use symbolic bin positions of the frozen WEDD cuts (`lowest_bin`, `highest_bin`, `not_lowest_bin`, `not_highest_bin`), never clinical cutoffs. Supported: `PT_DILATED_LV_REDUCED_EF`, `PT_THICK_WALL_PRESERVED_EF`, `PT_INCREASED_LV_MASS`, `PT_DILATED_RV_REDUCED_RVEF`, `PT_REDUCED_THICKENING`, `PT_REDUCED_LVEF_REDUCED_THICKENING`, `PT_NORMAL_SIZE_FUNCTION`. Unsupported and left unmatched: `PT_LV_NONCOMPACTION`, `PT_VALVULAR_DISEASE`, `PT_MYOCARDIAL_INFLAMMATION`, `PT_MYOCARDIAL_SCAR`, `PT_REGIONAL_WALL_MOTION`, `PT_RESTRICTIVE_PHYSIOLOGY`. A template is matched when an eligible rule's antecedent set structurally implies it (exact inclusion); for each template report the count of such rules and their coverage. Templates never become rules and never map to diagnoses.

---

## 12. Data splits, leakage prevention, and external testing

### 12.1 Unit of analysis

The **patient** (ACDC; one study per patient) or **case** (M&Ms; one per subject) is the analysis object. The bootstrap cluster is `cluster_id` (split manifest): the two members of a declared duplicate-image pair (Section 4.5) share one cluster, and every other patient or case is its own cluster (`cluster_id = sample_id`). *Amended September 28, 2026:* clusters replace the patient as the resampling unit (Section 13.1). No split is made by slice, frame or phase. Patient identity cannot be linked across the two datasets; the residual possibility of shared subjects is reported as unverifiable.

### 12.2 Split plan and allowed uses [PRESPEC]

| Partition | Source | Permitted estimation | Forbidden use |
|---|---|---|---|
| `acdc_dev` | 98 of the 100 official ACDC training patients (`patient055` and `patient076` excluded as declared duplicate images; DCM 20, HCM 20, MINF 19, NOR 19, RV 20), with `fold ∈ {1..5}`; 98 clusters | All fitting and selection: standardizers, target transforms, λ, transition, audited predictor, WEDD cuts, stability bootstraps, granules, rules, the cross-fitted gate, QC outlier references | Any choice informed by `acdc_test` or M&Ms outcomes |
| `acdc_test` | The 50 official ACDC testing patients (previously analysed under v4/v5), including `patient118`; 50 clusters | Internal evaluation with the frozen bundle only (H-primary-1) | Preprocessing, selection, threshold search, rule induction, QC reference fitting, model replacement |
| `mms_external` | The 320 labelled M&Ms-1 cases in 318 clusters (A7G0P5 + K3R0Y7 and C8J7L5 + C8O0P2 are one cluster each); strata `vendor`, `centre`, `release_split`, `pathology`, `mms100_member` | Once-only frozen evaluation (H-primary-2); prespecified strata and error analysis | Fitting of anything, calibration, normalization, QC reference change, feature pruning, rule induction, crosswalk revision |
| `mms_external_never_analysed` | The 220 cases of `mms_external` with `mms100_member = false`; 220 clusters | Confirmatory sensitivity repeat of H-primary-2 | Any use as a rescue test or a selection criterion |
| `mms_refit_train` | Release Training/Labeled, 150 cases (vendors A and B); 150 clusters | Refit arm only (D17): the same pipeline with inner CV | Any contact with the primary bundle; pooling with primary results |
| `mms_refit_eval` | Release Validation + Testing minus K3R0Y7 (`DUPLICATE_IMAGE_OF_REFIT_TRAIN_CASE`), 169 cases (A 20, B 50, C 49, D 50) in 168 clusters | Refit-arm evaluation, seen (A, B) versus unseen (C, D) vendors | Selection of any refit setting; pooling with primary results |
| `mms2_external` (6.0.1) | The first 180 M&Ms-2 subjects by the split rule of Section 4.6, minus duplicate-image exclusions; one cluster per case | Once-only frozen evaluation of the v6.0.0 bundle (H-primary-3) and of the frozen F8, D20, F9, F9b and S1–S4 bundles; prespecified strata (Section 12.3) | Any read before the tag; fitting of anything, calibration, normalization (except the declared transductive S3 statistics), QC reference change, feature pruning, rule induction, crosswalk revision |
| `mms2_sealed` (6.0.1) | The other 180 M&Ms-2 subjects | None under v6; reserved for protocol v7 | Extraction, decoding or any analysis; only member names and sizes are read, by the post-tag listing check of Section 4.6 |

**Exposure labels (version 6.0.1; Decision D32).** Every 6.0.1 result record and table carries one of these labels for its partition:
- **Prospective:** every `mms2_external` result. The partition stays unseen until its once-only evaluation.
- **Registered after unblinding:** every result on `acdc_test` or `mms_external` of an analysis first registered or first evaluated in 6.0.1. This covers F8, D20, F9, F9b, S1–S4 and S6, and the measured-B comparator, which Section 8.2 prespecifies but which was first evaluated after unblinding. Both partitions were evaluated by run `v6-primary-20260929-2307` and under v5.
- **Development:** `acdc_dev` out-of-fold results, including S5.
- **`EXPLORATORY_POST_HOC`:** B1–B9.

No 6.0.1 analysis on `acdc_test` or `mms_external` is confirmatory.

The 25 Training/Unlabeled cases belong to no partition (`MMS_RELEASE_UNLABELED`). `patient055` and `patient076` belong to no partition (*Amended September 28, 2026:* PI decision; `configs/cohorts.yaml` `duplicate_image_pairs`).

**Independent clusters per stratum** (`validate-cohort` on the staged sources, September 28, 2026 [LOCAL]). `mms_external`: vendor A 95/95 (cases/clusters), B 125/125, C 50/49, D 50/50; centre 4 50/49, every other centre cases = clusters; release split Training/Labeled 150/150, Validation 34/34, Testing 136/136; `mms100_member` 100/99; never-analysed 220/220. `mms_refit_eval`: vendor C 49/48; seen vendors (A, B) 70/70; unseen vendors (C, D) 99/98. Every ACDC class and fold has as many clusters as patients.

**Outer folds (Decision D15).** Within each ACDC class, sort the eligible development patients (20 per class; 19 MINF and 19 NOR) by the hexadecimal digest `SHA256("acdc-fold-v6|" + patient_id)` ascending, with `patient_id` the release folder name (for example `patient001`) and ties broken by `patient_id`; with rank `r = 0, 1, …`, `fold = (r mod 5) + 1`. The excluded duplicate-image patients take no rank. On the 98 eligible patients folds 1–4 hold 20 patients (4 per class) and fold 5 holds 18 (DCM 4, HCM 4, MINF 3, NOR 3, RV 4), as computed by `validate-cohort` [PRESPEC; `configs/splits.yaml` `outer_folds.expected`]. *Amended September 28, 2026:* fold sizes follow from the 98 eligible patients (previously 20 per fold on 100). The rule reads no image, mask, B value or prediction.

**Inner folds.** Inside `outer-train(f)` (78 patients for f = 1–4, 80 for f = 5), within class, rank the 15 or 16 patients by `SHA256("acdc-inner-fold-v6|" + str(f) + "|" + patient_id)` ascending and set `inner_fold = (r mod 5) + 1` [PRESPEC]. For the final bundle, λ is selected by 5-fold CV over the outer folds on all 98.

**Refit-arm folds.** Inside `mms_refit_train`, within vendor, rank by `SHA256("mms-refit-fold-v6|" + case_id)` ascending and set `fold = (r mod 5) + 1` [PRESPEC].

Freeze the fold manifests and counts in `configs/splits.yaml` and `artifacts/<run_id>/contracts/` before any fitting. Publish actual counts.

### 12.3 Leakage firewall and external protocol

- `registry/estimator_partition_registry.csv` names every estimated object with its permitted fitting, selection and evaluation partitions (fold assignments, standardizers, transforms, λ, predictor weights, transition, comparators, WEDD cuts, granules, rules, gate statistics, QC outlier references, equivariance generators, refit-arm objects); `tests/test_registry_check.py` rejects any fitted object whose fitting or selection partitions include `acdc_test` or an M&Ms partition (refit-arm objects on `mms_refit_train` excepted).
- Stage records and caches are keyed by split-manifest, configuration, code and input hashes; a cache built with another fold's standardizer is unusable.
- Reference **B** and **Y** of evaluation partitions live behind an evaluation-only loader; acceptance test 13 replaces them with sentinels and asserts identical predictions and routes.
- Duplicate control: exact-file SHA-256 of every source and decoded-voxel hashes of the ED/ES arrays, within and across cohorts; the known M&Ms Training/Unlabeled duplicate pairs are excluded; the four duplicate-image pairs of labelled cases (Section 4.5) are declared in `configs/cohorts.yaml` (`duplicate_image_pairs`) with their frames, action and reason code, and `validate-cohort --strict` fails on any undeclared pair, a declared pair whose frames no longer match, or an action that is not applied (*Amended September 28, 2026:* PI decision); patient identity across ACDC and M&Ms cannot be verified and is reported as such.
- The reader-study firewall (Decision D24): no v6 command reads `clinical_validation/restricted/`, reader responses, `<local path>` or any `git_lost_found_*` path; v6 tables do not carry reader-study pseudonyms.
- Legacy evidence under `legacy/v5/` is never a v6 input; a v6 stage that resolves a path under `legacy/` fails.

**External protocol (Decision D16).** The frozen ACDC bundle is evaluated on `mms_external` **once**: no refit, no threshold, calibration or crosswalk change, and no QC reference change (robust outlier references are fitted on `acdc_dev` only). All five strata are reported with their counts; vendor and centre panels are mandatory. Descriptive demographic strata [PRESPEC; Q28; `configs/bootstrap.yaml` `descriptive_strata`, `age_band_edges_years`]: sex as released and age band (< 40, 40–64, ≥ 65 years) from the release CSV, reported on `mms_external` with counts, point estimates and cluster-bootstrap intervals (below the subgroup threshold: counts and point estimates only); they carry no p-values, never enter the F3 family and are never used for fitting or selection. ACDC `Info.cfg` has no age or sex. The **220 never-analysed cases** form the confirmatory sensitivity stratum; the **100 previously analysed cases** (the MMS_100 reader cohort, computed under v5) are reported as a separate stratum and declared previously unblinded. External reports keep separate: (a) semantic reconstruction on all 320 (reference B measured with the same code path); (b) rule fidelity to the frozen predictor on all technically eligible cases; (c) predictor and rule compatibility with the crosswalked label; (d) QC, availability and shift. An external failure is a result, never an instruction to remove a feature and rerun. Shift audit (`audit-shift`, after M0 under the same lock; `configs/bootstrap.yaml` `shift_audit`): two-sample classifier AUROC (L2 logistic regression on Z, the formal A standardized with the frozen `acdc_dev` scaler, cross-fitted over five hash-rule folds of clusters within domain, `SHA256("shift-audit-v6|" + cluster_id)` (the `sample_id` for a singleton, so both members of a declared duplicate pair share a fold); out-of-fold AUROC with a percentile bootstrap resampling clusters (`cluster_id`) within domain) between `acdc_dev` and `mms_external`, pooled and per vendor (`metrics/shift_audit_mms_external.{json,parquet}`); the classifier reads domain membership only, never B or Y, is never part of the bundle and never selects anything (a label-free diagnostic in the estimator registry); per-target external bias of the frozen transition [INHERITED, P6] (`metrics/per_target_mms_external.parquet`, by vendor and centre in `metrics/strata_errors_mms_external.parquet`).

**External label contract and ontology crosswalk (Decision D8; registry `label_tables.mms_to_acdc`).**

| M&Ms `Pathology` | n (of 320) [LOCAL] | ACDC class | Crosswalk status | External class endpoint |
|---|---:|---|---|---|
| NOR | 79 | NOR | `HARMONIZABLE_SAME_DEFINITION` | accuracy, per-class recall |
| DCM | 97 | DCM | `HARMONIZABLE_SAME_DEFINITION` | accuracy, per-class recall |
| HCM | 82 | HCM | `HARMONIZABLE_SAME_DEFINITION` | accuracy, per-class recall |
| ARV | 15 | RV (the broader ACDC abnormal-RV category) | `HARMONIZABLE_SAME_DEFINITION` | accuracy, per-class recall |
| IHD | 4 | MINF | `HARMONIZABLE_NOT_EXACT` (`REFERENCE_NOT_EXACT_IHD_MINF`) | reported separately, never pooled |
| HHD, LVNC, AHS, Other | 15, 2, 1, 25 | — | `OFF_ONTOLOGY` (`REFERENCE_OFF_ONTOLOGY`) | none; distribution of `d_model` and of routes |

Crosswalk-evaluable n = 273 (+4 IHD); off-ontology 43. The crosswalk is frozen with the registry before M0 and never revised after unblinding. It asserts label compatibility of dataset-defined categories, not diagnostic equivalence.

**M&Ms refit ceiling (Decision D17; required secondary).** The complete v6 pipeline is refitted on `mms_refit_train` (150 cases, vendors A and B) with the refit-arm folds as the only CV and QC outlier references from `mms_refit_train`, and evaluated on `mms_refit_eval` (169 cases; K3R0Y7 excluded, *Amended September 28, 2026:* its images equal those of the training case A7G0P5). The refit predictor uses the ACDC classes whose `HARMONIZABLE_SAME_DEFINITION` crosswalk has at least 5 training cases; IHD and off-ontology cases are excluded from predictor fitting and kept for transition fitting (`configs/splits.yaml`, `refit_arm`). Required outputs: per-target metrics and `ρ` (with the mean predictor and `s_j` of `mms_refit_train`) on `mms_refit_eval`, by seen (A, B; 70 cases) and unseen (C, D; 99 cases in 98 clusters) vendors, beside the frozen ACDC bundle's results on the same 169 cases (paired; the same cluster draws); the refit predictor and rulebook are reported descriptively. The arm runs only after the primary external evaluation and its 220-case sensitivity are recorded; refit-arm objects are never loaded by the primary bundle, and the two arms are never pooled. It quantifies how much of the ACDC → M&Ms gap is development-cohort limitation and how much is vendor shift; it is not a second external test.

**Prospective external protocol (M&Ms-2; version 6.0.1; Decisions D27, D28).**
- **Once only:** the v6.0.0 bundle (H-primary-3, Section 3.3) and each frozen 6.0.1 bundle (Section 8.6; S1–S4) are evaluated once on `mms2_external`. There is no refit, and no threshold, calibration, crosswalk or QC-reference change.
- **Secondary endpoints:** per-target `ρ_j`, with the F6 family of Section 13.3. Prespecified stratum: the pathology group, with the RV family pooled as one predeclared stratum. Vendor, scanner and field strength are data-driven descriptive strata (their values are unverified before the tag; recorded verbatim and the vendor normalized as in Section 4.6): counts, estimates and intervals, never tested. The release CSV carries no centre, release split, sex or age (header read before the tag), so these strata are reported as not available. Strata below 20 clusters follow Section 13.2.
- **Other measures:**
  - the label-free shift-audit AUROC between `acdc_dev` and `mms2_external` (`audit-shift`, with the classifier, salt and bootstrap of the M&Ms-1 audit);
  - audited-predictor accuracy, macro-F1, macro one-versus-rest AUROC and ECE on the NOR + HCM subset;
  - `d_model` distributions for the non-equivalent strata;
  - routes, coverage and fidelity of the `d_model` and dataset-label rulebooks.
- **Refit arm:** not extended to M&Ms-2.
- **Estimator registry:** `registry/estimator_partition_registry.csv` lists every 6.0.1 estimator with fitting and selection on `acdc_dev` only. This covers the F8 trees; the D20 standardizer, transition and predictor; the F9 encoders, heads and candidate selection; the F9b standardizers and transitions; the S1 thresholds; the S2 references, PCA and flag thresholds; the S4 temperature; and the S5 subsample fits.
- **Transductive statistics:** the S3 standardizations H-a and H-b are label-free statistics of the evaluated partition. They are declared like the shift-audit classifier and are never part of a frozen bundle.

### 12.4 Quality-control severity model (Decision D21) [INHERITED from P6; PRESPEC; `configs/cohorts.yaml`, `qc_severity`]

| Check | Severity | Consequence |
|---|---|---|
| Missing ED/ES mask or frame out of range; ED = ES; LV, RV or myocardium absent at ED or ES; unexpected label value or invalid label mapping; smoothed integer labels; high overlap before resolution; non-positive spacing or slice increment unavailable; LV or RV EDV ≤ ESV; LV or RV EF outside 0–100; myocardial mass ≤ 0 | critical (4) | Row excluded from every fit; `QC_ABSTAIN` at inference; counted in the flow diagram |
| Identical ED and ES masks; wall-thickness method invalid globally or at ES | high (3) | Feature-level masking (cells `QC_BLOCKED`, `FEATURE_QC_MASKED`) of the registered dependent features (`feature_masks`); other features retained |
| Extreme robust outlier (`|0.6745 · (x − median)/MAD| > 3.5`, references from `acdc_dev` only, at least 5 reference values, zero MAD never flags); temporal metadata review flag; LV SV − RV SV discrepancy > 15 mL | moderate (2) | Retained; reported and used as a stratifier or sensitivity; masks nothing |
| Short-axis sphericity approximation; AHA mapping not feasible; BSA unavailable; BMI > 40 Mosteller caution; slice increment from the header `pixdim` fallback | informational (1) | Retained |

Missing-structure reasons are critical and an unregistered reason fails closed (`QC_UNREGISTERED_REASON`): the v5 default-to-moderate defect is removed. The SV-discrepancy check masks nothing: after the slice-increment fix about 79 of 150 ACDC patients exceed 15 mL [LOCAL estimate], and the mismatch also reflects regurgitation and basal-slice conventions; 15 mL is an engineering review trigger, not a clinical cutoff. QC panels (all critical and high cases plus 20 further cases per partition, ranked by `SHA256("qc-panel-v6|" + sample_id)` because no seed is configured; `configs/cohorts.yaml` `qc_panels`) are rendered into `artifacts/<run_id>/qc/panels/<DATASET>/` by `build-semantics` with a panel manifest `qc/panels_<DATASET>.json`; `qc/manual_review.parquet` (ACDC partitions; `qc/manual_review_MMS.parquet` for `mms_external`, after M0 only) lists every moderate-or-higher case and every panel case and is initialized with `adjudication = pending_review`. Automated QC is computational triage, not expert adjudication.

---

## 13. Metrics, statistical tests, confidence intervals, and ablations

### 13.1 Evaluation units and uncertainty [PRESPEC]

Every metric is computed over patients (ACDC) or cases (M&Ms), one row each, and every case weighs once in a point estimate. Uncertainty: 2,000 bootstrap replicates of **clusters** (`cluster_id` of the split manifest; a declared duplicate-image pair is one cluster that is drawn with both its cases, every other patient or case is its own cluster), seed 20260928, percentile 95 % intervals; identical resampled cluster lists for paired differences; resampling is unstratified within the evaluated partition, and within the stratum for stratum-level intervals (a pair split across two strata is a singleton in each) [PROPOSED]. *Amended September 28, 2026:* by PI decision every patient- or case-level bootstrap, interval and cluster-level test of v6 resamples clusters: the primary `ρ` and its per-target `ρ_j` intervals, the stratum results, the refit-arm `ρ`, the paired nMAE differences of the linearity criterion, the WEDD threshold stability bootstrap, the downstream refit bootstrap and the shift-audit bootstrap (within domain); the per-target Wilcoxon tests of F1–F3 rank the per-cluster means of the paired differences; the F4/F5 McNemar tests run on `acdc_test`, whose 50 patients are 50 singleton clusters. With singleton clusters the cluster bootstrap equals the patient bootstrap draw for draw (clusters are numbered by their first patient), so it reduces to the patient bootstrap wherever no pair is present (`acdc_dev`, `acdc_test`, `mms_refit_train`, the never-analysed stratum). Independent clusters (cases/clusters): `acdc_dev` 98/98, `acdc_test` 50/50, `mms_external` 320/318, `mms_external_never_analysed` 220/220, `mms100_member` 100/99, `mms_refit_train` 150/150, `mms_refit_eval` 169/168 (strata in Section 12.2); every result record reports `n_clusters` beside `n`. Report valid replicate counts; if fewer than 95 % are valid the interval is `NE`. Proportions with zero or all events are supplemented with Wilson intervals. Uncertainty from patients and from downstream refits is reported separately, never pooled.

### 13.2 Metric definitions

| Stage | Required metrics | Interpretation and denominator |
|---|---|---|
| Audited predictor | Accuracy, balanced accuracy, macro-F1, macro one-versus-rest AUROC, ECE with 10 equal-mass bins and realized bin counts; per-class recall, log loss and confusion matrix as descriptive companions | OOF on `acdc_dev`, `acdc_test`; on `mms_external` only for the 273 crosswalk-evaluable cases (IHD counts separately); off-ontology cases get the distribution of `d_model` only |
| Continuous semantics | Section 9.6 set, including `ρ_j` and the family `ρ` | Physical units; fit-SD normalization; observed counts and QC exclusions stated |
| Rules | Activation, accepted coverage, covered fidelity to `d_model`, covered label accuracy, faithful coverage, conflict and conflict-encounter rates, abstention by route and reason code, per-class coverage, rule count, tier composition, length | Consequent and route in every row; no coverage claim from a merely fired rule |
| External strata | All of the above per vendor, centre, release split, pathology and MMS_100 membership, plus the never-analysed flag and the seen (A, B) versus unseen (C, D) vendor groups (`external_strata`; `metrics/strata_{errors,routes,rho}_mms_external.*`) | Strata with < 20 cases are descriptive counts and point estimates only (`configs/bootstrap.yaml`): their intervals and tests are `NE` with reason `STRATUM_BELOW_MIN_CASES`, and no stratum is dropped |
| Refit arm | Continuous semantics on `mms_refit_eval`, seen versus unseen vendors, paired with the frozen bundle | Never pooled with the primary arm |
| Optional equivariance | Generator residuals, finite-action defect, measured reference change | Actual transformed inputs |

For observed set `Q_j`:

\[
MAE_j = \frac{1}{|Q_j|}\sum_{i\in Q_j}|\widehat b_{ij} - b_{ij}|,\qquad RMSE_j = \sqrt{\frac{1}{|Q_j|}\sum_{i\in Q_j}(\widehat b_{ij} - b_{ij})^2},\qquad NRMSE_j = \frac{RMSE_j}{s_{j}}, \tag{E17}
\]

with `s_j` the fit-partition SD. A target with zero fit SD has no normalized error. Correlation is not agreement; bias and error are always shown.

With `F(x)` the set of fired eligible exact rules, `a(x) = 1` for `EXACT_ACCEPT` and `o(x)` the accepted consequent, over `N` inference-processed inputs:

\[
Activation = \frac{1}{N}\sum_x \mathbf 1\{|F(x)|>0\},\qquad Coverage = \frac{1}{N}\sum_x a(x),\qquad Abstention = 1 - Coverage, \tag{E18}
\]

\[
Fid_{cov} = \frac{\sum_x a(x)\mathbf 1\{o(x) = d_{model}(x)\}}{\sum_x a(x)},\qquad Acc_{cov} = \frac{\sum_{x: Y\ evaluable} a(x)\mathbf 1\{o(x) = Y(x)\}}{\sum_{x: Y\ evaluable} a(x)}, \tag{E19}
\]

\[
FaithfulCoverage = \frac{1}{N}\sum_x a(x)\mathbf 1\{o(x) = d_{model}(x)\},\qquad CR = \frac{1}{N}\sum_x \mathbf 1\{\text{route}(x) = \text{CONFLICT\_ABSTAIN}\}. \tag{E20}
\]

The conflict-encounter rate counts inputs for which any two fired eligible rules disagree, including inputs that then route elsewhere [INHERITED, P6]. If no rule is accepted, covered metrics are `NE`, coverage is 0 and abstention is 1. A denominator reconciliation table lists attempted inputs, QC exclusions, processed inputs, accepted outputs and label-evaluable outputs.

**Audit tax.** On exactly the accepted subset `𝒜`,

\[
AuditTax = Acc_{predictor}(\mathcal A) - Acc_{cov}(\mathcal A), \tag{E21}
\]

reported beside coverage; never subtract covered accuracy from whole-cohort predictor accuracy [P4 framing].

### 13.3 Tests, multiplicity and the ablation matrix

The primary comparison is the fixed-sequence bootstrap test of Section 3.3. No image-level tests on repeated structures; no independent-proportion tests on paired predictions. Effect sizes and intervals precede p-values. Benjamini–Hochberg control at q = 0.05 is applied within each prespecified family (`configs/bootstrap.yaml`): (F1) per-target reconstruction, ridge versus mean, on `acdc_test`; (F2) per-target ridge versus the comparators of Section 9.4 (pseudoinverse, identity transforms, P6 projection); (F3) per-target external transport on `mms_external` and its strata; (F4) discretizers; (F5) rule-policy sensitivities. No merged ranking across families [INHERITED, P5]. The tests entering the BH step [PRESPEC; `multiplicity.tests`, `multiplicity.family_members`]: in F1–F3 a two-sided Wilcoxon signed-rank test per target of the paired per-case differences `|error of the comparator| − |error of the primary transition|` (effect: the MAE difference, also divided by the fit-partition SD; F3 on `mms_external` and on every stratum of `external_strata` with ≥ 20 cases, the others `NE`); in F4 (the discretizer profiles and the WEDD settings grid) and F5 (the rule-policy and gate-bound sensitivities) an exact McNemar test per variant of the paired `EXACT_ACCEPT` indicators of the variant and the primary configuration of the `d_model` rulebook on `acdc_test`. F1 is written by `evaluate --split acdc_test` (`metrics/multiplicity_F1.parquet`), F3 by `evaluate --split mms_external` (`metrics/multiplicity_F3.parquet`), F2, F4 and F5 by `run-ablations` (`metrics/ablations/multiplicity_F{2,4,5}.parquet`); `NE` rows are kept with an empty q-value.

*Amended in 6.0.1 (Decisions D28, D31, D36):*
- **Family F6:** per-target external transport on `mms2_external` and its strata. It uses the F3 test: a two-sided Wilcoxon signed-rank test per target of the paired per-cluster differences, ridge versus mean, on strata with ≥ 20 clusters, the others `NE`. It is written by `evaluate --split mms2_external`, with Benjamini–Hochberg q = 0.05 within F6, never merged with F1–F5.
- **H-primary-3:** a one-test family outside the Benjamini–Hochberg families.
- **H4 family (D36):** H4a and H4b form a second confirmatory family on `mms2_external` with Bonferroni control within the family (each test judged by its two-sided 97.5 % interval; Section 3.3), outside the Benjamini–Hochberg families and separate from H-primary-3. On `acdc_test` and `mms_external` the H4 estimands are estimation only, registered after unblinding.
- **Estimation only:** F8, D20, F9, F9b and S1–S6 report point estimates and paired cluster-bootstrap intervals (2,000 replicates, seed 20260928), with no p-value, no significance language and no success criterion. For F9 the three-seed range and SD are reported, and no seed is chosen.
- **Exploratory:** the analyses of Section 13.6 report effect sizes only, with no confirmatory p-value and no post-hoc power.
- **Registration rule:** no analysis outside Section 0.3 may be called registered; anything added later is labelled post hoc.

| Axis | Frozen comparisons | Required output |
|---|---|---|
| Formal representation | 57 unprojected descriptors (primary) versus the 128-d P6 projection | per-target losses, condition diagnostics, `ρ` |
| Target transforms | typed (`log`, `logit_percent`, `identity`) versus `identity_sensitivity` | per-target losses, domain violations |
| Transition | mean, pseudoinverse, ridge per-target λ, ridge global λ, RBF kernel ridge | matched per-target losses, linearity criterion |
| Symbolic source | OOF reconstructed B̂ versus measured-B oracle | separate rulebooks; coverage, fidelity, accuracy |
| Discretizer | `wedd_gaussian_mm_v1` versus `three_quantile_v1` versus `wedd_uniform_mm_v1` | cut stability, acceptance, fidelity, abstention |
| WEDD settings | weight {0, 0.25, 0.5, 0.65, 0.75, 1}, bandwidth multiplier {0.5, 1, 2}, max bins {2, 3, 4}, max depth 3 (at four bins) | one factor at a time around the primary configuration (`metrics/ablations/wedd_settings.json`) |
| Rule policy | β {0.05, 0.10, 0.20}; strict-only; `L_max` {2, 3} (`configs/rule_policy.yaml` `discovery.sensitivities`); gate lower bound {0.50, 0.60, 0.70} (`validation_gate.gate_sensitivities`) | per-class coverage, rule count, route rates |
| External cohort | frozen ACDC bundle versus refit arm (D17) | paired `ρ` and per-target losses, seen versus unseen vendors |
| External strata | vendor, centre, release split, pathology, MMS_100 membership | support-qualified errors and route rates |
| Optional branches | equivariance generators and defects; deep-encoder A | as Sections 8.4 and 10 |

`run-ablations` writes every row of this matrix that is computable on ACDC (formal representation, target transforms, pseudoinverse, global-λ ridge, RBF kernel ridge and the linearity criterion, discretizers, the WEDD settings grid, rule policy, gate bounds) as labelled secondary outputs under `metrics/ablations/` and `rules/ablations/`; `metrics/ablations/ablation_index.json` lists them (`not_run` is empty). Its acdc_test routes use the substrate of `evaluate` (decoders and per-object stability under each variant's cuts). Rulebook stability: a 200-replicate **downstream refit bootstrap** on `acdc_dev` (`metrics/ablations/downstream_refit_bootstrap.parquet`; the refit arm writes its own under `rules/refit_arm/`) (target transforms, transition, thresholds and rule induction refitted), separate from the 500-replicate threshold bootstrap and the 2,000-replicate fixed-model bootstrap; rules are matched by consequent and antecedent-interval overlap with exact and relaxed recurrence reported.

### 13.4 Failure-mode analyses

Stratify reconstruction and rule failures by vendor, centre, release split, pathology, QC severity, slice increment, slice count, in-plane spacing and predictor confidence decile. Report confidently incorrect predictions, semantically plausible but model-unfaithful rules, and numerically stable but reference-inaccurate semantics. **Scale audit:** relate per-case external volume and mass errors to in-plane spacing, slice increment and field of view; this is the direct test of the unitless-A mechanism behind H3. **Threshold-within-error flag:** any WEDD cut lying within the feature's OOF MAE of a neighbouring cut is flagged `threshold_within_error = true` [P5 §5.2 lesson]. Subgroups below 20 cases are descriptive and underpowered, not fairness certification.

### 13.5 Registered secondary analyses S1–S6 (version 6.0.1; Decision D30) [PRESPEC; `configs/trust.yaml` (S1–S4), `configs/data_centric.yaml` (S5, S6)]

**Common rules.**
- **Intervals:** paired cluster bootstrap, 2,000 replicates, seed 20260928; proportions with Wilson intervals, numerator and denominator.
- **Fitting:** every threshold, reference, PCA and temperature is fitted on `acdc_dev` out-of-fold outputs only and frozen before any evaluation (Section 14.6).
- **Exposure labels:** prospective on `mms2_external`; registered after unblinding on `acdc_test` and `mms_external` (Section 12.2).
- **Estimation only:** Section 13.3.
- **Not executed:** a registered analysis that cannot run is reported as not executed, with its reason; it is never replaced.

| ID | Method | Outputs per partition | Stages and run-root folder |
|---|---|---|---|
| S1 Selective prediction | Confidence = the maximum softmax of the audited predictor, the D20 predictor and F9 (primary seed). τ₅₀ and τ₈₀ are the `acdc_dev` out-of-fold confidence quantiles that give 50 % and 80 % coverage | Risk–coverage curve and AURC (cluster bootstrap); realised coverage and accuracy at τ₅₀ and τ₈₀; the v6 rule-route coverage and fidelity alongside. Wording: "computational abstention" | `fit-trust`, `evaluate-trust`; `trust/` |
| S2 Label-free OOD detection | Per representation: the Mahalanobis distance to the `acdc_dev` distribution, with Ledoit–Wolf shrinkage on standardized A; for D20, computed after a 50-component PCA fitted on `acdc_dev`. Flag: above the 95th percentile of the `acdc_dev` out-of-fold scores | Proportion flagged per partition and vendor; AUROC of the score for detecting misclassified cases; Spearman correlation with the per-case normalized reconstruction error (the mean over primary targets of \|B̂ − B\| / `s_j`); accuracy and ρ in flagged against unflagged cases | `fit-trust`, `evaluate-trust`; `trust/` |
| S3 Label-free harmonization | T, λ and the predictor stay frozen; only the standardization of A changes. **H-a:** the mean and SD of the unlabelled evaluated partition. **H-b:** per-vendor mean and SD within that partition, from vendor metadata only (partitions with vendor metadata). B, labels and outcomes are never used | ρ, `ρ_j` and predictor accuracy against the frozen bundle, with paired differences. Direction (descriptive): recovery of ρ indicates a location–scale shift of A; no recovery indicates a change in the A → B relation. Reported as transductive adaptation | `evaluate-trust`; `trust/` |
| S4 Calibration transfer | A single temperature fitted by negative log-likelihood on the `acdc_dev` out-of-fold logits of each predictor (audited, D20, F9) | Reliability diagrams; ECE (as Section 13.2), Brier score and log loss before and after | `fit-trust`, `evaluate-trust`; `trust/` |
| S5 Data-efficiency curves | Development only, so no new held-out exposure. Fractions f ∈ {0.5, 0.75, 1.0} of each outer fitting set, class-stratified, selected within each class by the rank of `SHA256("lc-v6.0.1|" + repr(float(f)) + "|" + str(r) + "|" + patient_id)` ascending (ties by `patient_id`), with f written as the Python float representation (`0.5`, `0.75`, `1.0`), r the 0-based repeat (0, 1, 2) and `patient_id` the canonical patient key of the v6 hash rules (for example `patient001`; `configs/data_centric.yaml` `data_efficiency.subsample`); each class keeps floor(f · n_c + 0.5) members. Descriptor and F9 fits use the same subsamples. Three repeats for f < 1; f = 1.0 reuses the full fits. Pipelines: the descriptor pipeline (CPU) and F9 (primary seed, full selection; 2 × 3 × 5 × 19 = **570 fits**) | Out-of-fold ρ and macro-F1 on the held-out outer fold per fraction; a missing class gives `NE_INSUFFICIENT_CLASS_SUPPORT` | `run-data-efficiency`; `data_efficiency/` |
| S6 Simulated acquisition shift | Interventional, on `acdc_test`: images and masks are resampled to the median in-plane spacing and slice increment of each M&Ms-1 vendor, read from headers only (images linear, masks nearest-neighbour). A, B, the D20 features and the F9 inputs are recomputed, and the frozen bundles are applied | Δρ, Δ`ρ_j` and Δaccuracy for the descriptor, D20 and F9/F9b representations against unmodified `acdc_test` | `run-simulated-shift`; `simulated_shift/` |

### 13.6 Exploratory register B1–B9 (version 6.0.1; post hoc; B9 added October 3, 2026) [`configs/exploratory.yaml`]

These analyses explain results and never confirm them. Each runs as `run-exploratory --analysis B<n>`, writes `exploratory/<B#>/` with a provenance record, and labels every output `EXPLORATORY_POST_HOC`. They report effect sizes only.

| ID | Question and method | Data |
|---|---|---|
| B1 | Shift attribution: per-descriptor AUROC and Kolmogorov–Smirnov distance between `acdc_dev` and each M&Ms vendor; the shift audit repeated without each descriptor group | `acdc_dev`, `mms_external` formal A |
| B2 | Error decomposition: external volume error regressed on in-plane spacing, slice thickness, slice count, matrix and field of view; bias against spread per vendor | `mms_external` |
| B3 | Vendor-A note: pathology mix, per-target bias and geometry against ACDC | `mms_external`, `acdc_dev` |
| B4 | Cut stability: recurrence over the 500 bootstrap replicates as a function of `max_bins`, and its relation to class separation | `acdc_dev` |
| B5 | Gate power: planted rules with agreement 0.80–0.95 and support 5–20, 2,000 synthetic datasets per cell, the v6 gate and alternatives; P(a true rule passes) and the false-pass rate. Optionally on Modal CPU, because the data are synthetic | synthetic |
| B6 | Fold variance: 20 alternative fold salts; spread of λ, cuts, eligible features and rules | `acdc_dev` only |
| B7 | Segmentation sensitivity: 1-pixel erosion and dilation per structure, and the basal slice dropped; ΔB̂, Δρ, `d_model` flip rate and route changes | `acdc_test` |
| B8 | Annotation convention: per-vendor target bias on M&Ms-1 against M&Ms-2; descriptive and confounded by centre; run only after the once-only `mms2_external` evaluations are recorded | `mms_external`, `mms2_external` |
| B9 | Orientation audit of the image-branch input frame (owner issue O1; label-free): the staged ED reference mask is reoriented like the D20 and F9 inputs (`canonical_orientation`). On the mid-ventricular ED slice (the middle, in stored order, of the slices with both LV and RV cavity) the LV → RV centroid direction is measured in mm, with y up. Per dataset, partition, vendor (where the cohort table has one) and orientation code: n, median and 10th–90th percentile angle, fraction RV-left, fraction RV-above. Masks serve only for the two centroids; no class label is read | `acdc_dev`, `acdc_test`, `mms_external`; `mms2_external` only through the M&Ms-2 lock and its passed validation |

---

## 14. Reproducibility and provenance

### 14.1 Repository layout

Files marked ● exist at the end of the refactor session of September 28, 2026 (created, moved or kept by the coordinated agents); ● marks file presence, not completion of a v6 stage on real data. Files marked ○ are implementation targets or runtime outputs.

```text
transition-matrix-mri/
  SPECIFICATION.md ●               # this protocol, version 6.0.1 (6.0.0 values unchanged)
  README.md ●  CONTEXT.md ●  AGENTS.md ●  CLAUDE.md ●  CONTRIBUTING.md ●  LICENSE ●
  pyproject.toml ●  uv.lock ●  .python-version ●  Dockerfile ●  compose.yaml ●  .dockerignore ●
  .env.example ●  .gitattributes ●  .gitignore ●  CITATION.cff ○
  docs/adr/0001-acdc-development-full-mms-external.md ●
  docs/adr/0002-v6-protocol-decisions.md ●
  docs/adr/0003-docker-desktop-linux-container-runtime.md ●   # realization of D22
  docs/adr/0004-repository-refactor-and-legacy-relocation.md ●   # D25 file moves
  docs/EXECUTION_ENVIRONMENT_SPEC.md ●   # container environment, version 1.0.0
  docs/EXPERIMENT_EXECUTION_PLAN.md ●    # measured job order and budgets, version 1.0.0
  docs/INSTALLATION_AGENTS_PROMPT.md ●  docs/DATA_ACQUISITION_AGENTS_PROMPT.md ●  docs/EXPERIMENT_AGENTS_PROMPT.md ●
  docs/references/external_sources.bib ●
  docs/history/ ●   # implementation_specs/ (AGENTS_v1-v5), execution_reports/, codex_conversations/,
                    # experiment_machine_v5/, methodology_v5/, reproducibility_v5.md, repository_layout_v5.md,
                    # IMPLEMENTATION_PLAN_repository_restructure_v1.md
  protocol/amendments.md ●  protocol/decision_log.jsonl ○  protocol/source_review_manifest.json ○
  registry/
    semantic_registry.yaml ●       # Section 6 contract, version 6.0.0
    reason_codes.yaml ●            # Section 11.7
    pattern_templates.yaml ●       # Section 11.8
    estimator_partition_registry.csv ●   # Section 12.3, one row per estimated object; hashes empty until M0
    sources/ ●                     # acdc_* manifests, mms_full_release_manifest.csv, mms_100_* (frozen reader cohort)
  configs/
    study_v6.yaml ●  cohorts.yaml ●  splits.yaml ●  formal.yaml ●  transition.yaml ●  wedd.yaml ●
    rule_policy.yaml ●  equivariance.yaml ●  deep_encoder.yaml ●  bootstrap.yaml ●  runtime.env ●
    source_bindings/{acdc,mms}.json ●
    clinical/ ● (governed, unchanged)   legacy/ ● (former pipeline/, labels/, segmentation/ for legacy flows)
  scripts/verify_environment.py ●  scripts/data/ ●  scripts/host/ ●  scripts/bench/ ●  scripts/experiments/ ●
  scripts/clinical_validation/ ●  scripts/maintenance/ ●
  src/tmmri/ ●
    __init__.py  cli.py  paths.py  config.py  io.py  provenance.py  release.py  runs.py  stages.py (v6 stage DAG)
    ingest/{acdc,mms,inventory,native_data,geometry,labels,staging,sources}.py
    semantics/{measurements,features,qc,manual_review,visual_qc,contract}.py
    formal_features.py  transition.py  transition_fit.py  target_transforms.py
    wedd.py  rough_sets.py  rule_pipeline.py (eligibility, nested gate substrate, refit bootstrap)  rules.py  inference.py  predicates.py  rule_audit.py
    metrics.py  classification.py  equivariance.py  pipeline.py  splits.py  for_rules_import.py (legacy)
    experiments/transition_grid.py   clinical/ (governed package code; behaviour unchanged)
    deep_encoder.py ○ (optional D20 branch; not yet implemented)
  tests/ ●  (conftest.py with the registry fixture and require_module skip helper; fixtures_v6.py synthetic sources;
             test_v6_*.py protocol tests; test_registry_check.py)
  data/ (git-ignored runtime root; Docker volume tmmri_data)  staged/ ○  manifests/ ○  normalized/ ○
  artifacts/ (git-ignored runtime root; Docker volume tmmri_artifacts)
    <run_id>/ ○  contracts/ qc/ semantics/ formal/ transitions/ oof/ reconstructed/ thresholds/ granules/
                 rules/ inference/ equivariance/ metrics/ figures/ manuscript_tables/ .stages/
                 artifact_status.json  release_manifest.json  .stages/FROZEN.json (after M10)
                 */refit_arm/ and */ablations/ (secondary outputs; never part of the frozen bundle)
  legacy/v5/ ●   # relocated v4/v5 evidence, pre-fix, never a v6 input (README states status)
  clinical_validation/ ● (unchanged)   manuscript/ ● (content unchanged; describes v5)   third_party/benjamin-plus/ ●
  manuscript/CLAIM2024_checklist.md ○  manuscript/STARD-AI_crosswalk.md ○  manuscript/TRIPOD-AI_checklist.md ○
  # version 6.0.1 additions (○ until integration; file list finalized at integration)
  configs/cart_f8.yaml  exploratory.yaml  neural_input.yaml  f9_cnn.yaml  probes.yaml  trust.yaml  data_centric.yaml
  configs/source_bindings/mms2.json   registry/weights.yaml   registry/sources/mms2*   (split manifest, receipt)
  scripts/cloud/modal_train.py  scripts/cloud/README.md   # Modal app; client pinned through uvx (Section 14.7)
  src/tmmri/deep_encoder.py  src/tmmri/neural/{fit_spec,model,data}.py  src/tmmri/experiments/trust/
  artifacts/<run_id>/ neural_cache/<partition>/ deep_encoder/ neural_f9/ probes/ trust/ data_efficiency/
                      simulated_shift/ comparators/cart_f8/ exploratory/<B#>/ figures/ cloud/
```

Source-to-result graph:

```text
ACDC release + M&Ms release + tracked manifests + licence texts
  -> verify-and-stage (one read per source; SHA-256 against manifest) -> staged ED/ES frames + staging manifests
  -> cohort, fold and stratum manifests; exclusion ledger
  -> geometry rule, frame selection, label canonicalization, QC severities
  -> reference B + M + Y (one measurement code path)   |   formal A (57 descriptors)
  -> frozen audited predictor -> d_model
  -> nested fold transforms, per-target lambda, OOF ledger, frozen T -> B-hat + estimated-state ledger
  -> eligibility -> WEDD thresholds -> S -> granules -> VPRS rules -> cross-fitted gate -> rulebook or EMPTY
  -> .stages/FROZEN.json -> acdc_test evaluation -> (tag + frozen bundle) M&Ms B and A -> once-only external evaluation
  -> refit arm (after external) -> metrics with numerators, denominators and cohort hashes
  -> generated tables, figures and macros -> remade manuscript
```

### 14.2 Provenance requirements

Every artifact manifest records artifact type and schema version, dataset release, ordered-row hash, source and staged hashes, all parent-artifact hashes, code commit and dirty-tree status (a dirty tree makes a run ineligible for reporting), configuration snapshot and hash, the protocol tag, seeds, package versions, container image identifier and base-image digests, Docker Desktop, WSL and kernel versions, thread counts and BLAS back end, command, timestamp and status. Timestamps are recorded outside content hashes (the v5 `generated_at_utc` fields made hashes non-reproducible). Byte hashes and semantic-content hashes are both retained.

**Stage DAG (Decision D23).** `python -m tmmri.cli run --study configs/study_v6.yaml` executes the stages of Section 15.1 in dependency order (`src/tmmri/stages.py`). Each stage writes `.stages/<stage>.json` with its input hashes, code version, configuration hash and output hashes; a stage is skipped only when its inputs, code tree hash, configuration hash and parameters match an existing record whose outputs verify. The configuration hash of a stage covers only the parsed configuration *sections* the stage reads (`STAGE_CONFIG_SECTIONS`, listed in the record as `config_sections`), so a change to an evaluation-only section (for example `bootstrap.fixed_model_bootstrap`) re-runs exactly the stages downstream of it, and a comment-only edit re-runs nothing. `.stages/FROZEN.json` records the digests of the sections of the fitting stages; any later change to one of them makes every fitting stage of that run refuse (a new run id is required). `release_manifest.json` (`write-release-manifest`) holds the SHA-256 of every run-root file, the configuration and registry file hashes, the code version, the protocol-tag state and the normalized `artifact_status.json`. Promotion from a staging directory to the run root is atomic; an interrupted run never leaves a completed manifest. v6 stages never write into tracked paths (the v5 in-place runs that rewrote tracked registry and feature files are legacy behaviour).

**Once-only guard** (*amended in 6.0.1*, roadmap A1, Decision D26). In a frozen run (`.stages/FROZEN.json` present), a stage that already has a record **may only be skipped**. If its inputs, code version, configuration hash, parameters or recorded outputs differ from the record, or `--force` is given, it refuses with exit 7 (`StageRecordLockedError`). It is never re-executed, and a new run id is required. Stages without a record run normally. `write-release-manifest` is not a recorded stage and always runs. The refusal of fitting stages after a change of the frozen sections, described above, is a special case of this rule. Every 6.0.1 stage that evaluates a frozen bundle writes a stage record and is subject to this guard.

**Determinism.** Seeds as in Section 0.2; `PYTHONHASHSEED=0`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `NUMEXPR_NUM_THREADS` = 1 are set at process start by `configs/runtime.env` (setting them in-process, as v5 did, is ineffective). Serialize Parquet with PyArrow, Snappy compression and no index; linear algebra in Float64; matrices in NPZ plus human-readable JSON; no untrusted pickle. The container runs CPython 3.12.14 from `uv.lock` (Decision D22); the host virtual environment (Python 3.13) remains only for the existing test suite and legacy flows. The repository is checked out with LF line endings so frozen-file hashes agree inside the container.

### 14.3 Replay boundary [INHERITED, P5]

Deterministic replay is claimed **from the staged ED/ES frames onward** for the primary branch, because A is computed by deterministic code with no learned parameters, under fixed inputs, configurations, seeds, the pinned container and single-threaded BLAS; results must be byte-identical for any worker count. For the optional deep-encoder branch, replay is claimed from the frozen activations onward; bitwise GPU extraction across hardware is not claimed. The manuscript states this boundary. *Amended in 6.0.1:* for F9, F9b and S5, replay is claimed from the frozen checkpoints and the locally computed predictions and activations onward. F9 training is deterministic only on one GPU model and software stack; bitwise equality of training across hardware is not claimed, and the cross-hardware check of Section 14.7 is reported instead.

### 14.4 Blocked-artifact protocol (never fabricate)

If a required resource or compute is unavailable, the stage writes `artifacts/<run_id>/artifact_status.json` with `status ∈ {BLOCKED_DATA_ACCESS, BLOCKED_COMPUTE}`, the blocked resources, the produced items (schemas, registry, tests, code) and the not-produced items, emits header-only previews and zero-byte `*.BLOCKED` markers in place of every data-bearing artifact, and never writes a file named `B_<dataset>.parquet`, `A_<dataset>.parquet`, `T_*` or a rulebook. Synthetic fixtures run the code path without data and are never presented as cohort results. A command refused by the lock of Section 8.5 exits with an error and writes no data-bearing artifact; it is not a blocked-access status.

### 14.5 Data-sharing policy

| Artifact | Sharing |
|---|---|
| Code, registry, configs, tests, synthetic fixtures | Public under the repository `LICENSE` (a code licence, not a data licence) |
| Transition operators, thresholds, rulebooks, pattern-template reports, aggregate metrics, figures without case content | Public; contain no per-case content |
| Per-case `B`, `A`, `B̂`, `S`, route traces, staged frames | ACDC: CC BY-NC-SA 4.0, non-commercial scientific research, mandatory citation [D3]; M&Ms: Research Use Agreement clause 5 forbids distributing any portion [D6]. Per-case M&Ms tables are never published; ACDC per-case tables only under the licence terms. Otherwise provide reproducible construction instructions |
| Example images and QC panels | ACDC only where the licence permits; M&Ms images are not redistributed |
| Tracked manifests | Identifiers, release split, relative paths, sizes, SHA-256 and eligibility/membership flags only; no vendor, centre, pathology, age, sex, height or weight (vendor and centre are read from the mounted information CSV at run time; Decision D2) |

**Licence risk for any public mirror.** `data/raw/mms/211230_MnMs_Dataset_information_diagnosis_opendataset.csv` is tracked in git. It is a verbatim portion of the M&Ms release with per-case vendor, centre, pathology, age, sex, height and weight, so clause 5 of the M&Ms Research Use Agreement forbids distributing it. Tracked legacy per-case tables derived from MMS_100 (for example under `legacy/v5/` and `clinical_validation/runs/cv_baseline_v1_20260809/data/features/`) carry the same risk. The repository must not be made public, mirrored or archived publicly until the owner decides how to remove these files from the published history [PROPOSED; owner decision]. v6 code reads the CSV from the mounted source, not from the tracked copy.

*Amended in 6.0.1:* M&Ms-2 follows the M&Ms-1 rules above. Its Research Use Agreement allows a personal research copy and forbids distribution, so no M&Ms-2 image, per-case table or per-case metadata is published or tracked. Tracked M&Ms-2 manifests carry identifiers, member paths, sizes, SHA-256 and membership flags only, and the release CSV is never copied into the repository.

### 14.6 Freeze order and once-only locks (version 6.0.1; Decision D33) [PRESPEC]

| Step | Object frozen or evaluated | Precondition | Record |
|---|---|---|---|
| 1 | The amendment commit, holding this file, the 6.0.1 configurations, the registry rows and the M&Ms-2 split manifest; the annotated tag `protocol-v6.0.1`, signed by the owner | Tests green; layout check passes | Tag object; the `protocol/amendments.md` row |
| 2 | The F8 supplementary bundle | The `acdc_dev` out-of-fold outputs of the v6 bundle | `comparators/cart_f8/` freeze record with hashes |
| 3 | The D20 bundle (standardizer, transition, predictor) | Weights verified (`verify-torch-weights`) | `deep_encoder/` freeze record |
| 4 | F9 (fold and final encoders of all three seeds) and F9b; the S1 thresholds, the S2 references and flag thresholds and the S4 temperatures, fitted on `acdc_dev` out-of-fold outputs | Fit records imported and verified (`import-f9-fits`); cross-hardware check done (Section 14.7); the F9 technical-validity gate passed (Section 8.6) | `neural_f9/` (`freeze-f9`), `probes/` and `trust/` freeze records |
| 5 | Once-only evaluations on `acdc_test` and `mms_external` (registered after unblinding); S6 on `acdc_test` | The freeze record of the bundle | Stage records |
| 6 | M&Ms-2, in order: hash check, extraction of the 720 files, staging, `validate-cohort --strict`, semantics, formal A. Then once-only evaluation on `mms2_external` of the v6.0.0 bundle (H-primary-3 and the H4 family, D36), F8, D20, F9, F9b and S1–S4; the shift audit; summaries | The step-1 tag; the freeze records of steps 2–4 dated before the M&Ms-2 extraction record | Stage records; `extract-mms2` record |
| 7 | B1–B9; tables, figures, release manifest; the results freeze, approved by the owner | — | `release_manifest.json` |

**Rules.**
- **Once per bundle:** each partition is evaluated once per frozen bundle. A repeat needs a new run id and is reported as a repeat, never as a replacement.
- **Late freeze:** a comparator or parameter set whose freeze record is dated after the M&Ms-2 extraction record has no registered `mms2_external` result; any such result is labelled post hoc.
- **Not executed:** a registered item that cannot run is reported as not executed, with its reason, and does not block the others. S5 runs after F9 and is reported as not executed if it misses the results freeze.
- **After the results freeze:** nothing is recomputed. A later bug is reported as a post-hoc correction showing both values.
- **No rules on CNN decisions:** no rule is fitted on CNN decisions (Section 17).

### 14.7 Data governance and the cloud backend (version 6.0.1; Decision D34)

Cloud-processing exception CPE-1 and cloud-storage exception CSE-1 (`protocol/amendments.md`, October 2, 2026) apply unchanged and are authoritative.
- **CSE-1:** covers the owner's off-site copy of `MnM2.rar`. Extraction reads only the hash-verified D: copy (Section 4.6).
- **CPE-1:** recorded for the neural study, and used here, within the same scope, for the F9 and S5 development fits. Those fits consume the same derived `acdc_dev` inputs: preprocessed slice tensors, the two geometry scalars and the labels.
- **Local-only stages:** v6.0.0 stages, D20, F8, F9b, S1–S4, S6, B1–B4 and B6–B9, and every evaluation, run locally.
- **Never uploaded:** any `acdc_test`, `mms_external`, `mms2_external` or `mms2_sealed` tensor, prediction or derivative; raw files; ACDC height and weight; anything under `clinical_validation/`; v6 run roots.

| Condition | Binding value |
|---|---|
| Data in the cloud | Only the `acdc_dev` tensor cache (`neural_cache/acdc_dev/` with its manifest), the outputs of F9 and S5 development fits, and synthetic tensors (qualification; B5). Location: the owner's personal Modal workspace and its two named Volumes, as recorded in CPE-1 |
| GPU type | One pinned type with no fallback list, fixed in `configs/f9_cnn.yaml` after the synthetic qualification and before the first upload of `acdc_dev` tensors |
| Numerics | IEEE FP32: `torch.backends.cudnn.allow_tf32 = False`, `torch.backends.cuda.matmul.allow_tf32 = False` and `fp32_precision = "ieee"` where the API exists; `torch.use_deterministic_algorithms(True)`; `cudnn.benchmark = False`; `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before CUDA initializes; eager mode |
| Qualification (synthetic tensors only) | Repeated fits are bitwise identical; a resumed fit equals an uninterrupted one; the TF32 state is asserted |
| Fit records | Each fit records its fit spec and `fit_id`, seeds, input SHA-256, image identifier and `uv pip freeze` hash, device, GPU name and UUID, driver, torch, CUDA and cuDNN versions, the TF32 state, timings, retries and the checkpoint SHA-256. `import-f9-fits` downloads and verifies every record; dashboard logs are never evidence |
| Ledger | `cloud/modal_ledger.csv` in the run root records every upload and deletion (path, bytes, SHA-256, Volume, UTC time) and is archived with the run. Jobs verify input SHA-256 before use |
| Inference | Every reported F9 and F9b number comes from local inference in the `tmmri-gpu` service (`infer-f9`). A development-only cross-hardware check compares Modal and local logits on the same checkpoints (maximum \|Δlogit\|, argmax agreement) and is reported |
| Image and client | The Modal image is built from the pinned base and `uv.lock` (`--group gpu`); its `uv pip freeze` hash must equal that of the local `tmmri-env-gpu`. The Modal client is pinned through `uvx` (`scripts/cloud/README.md`) instead of a uv `cloud` group, so `pyproject.toml`, `uv.lock`, the Dockerfile and both images are unchanged |
| Credentials and spending | The token stays in `%USERPROFILE%\.modal.toml` or in run-time environment variables, never in the repository. The workspace spending cap applies, with at most 10 concurrent containers |
| Fallback | If qualification fails, F9 trains on the local GTX 1080; S5 follows it if time allows and is otherwise reported as not executed |
| Retention | After the release backup is verified, the owner deletes both Volumes and the deletion is entered in the ledger |

---

## 15. Implementation roadmap, acceptance tests and resources

Owners: **PI** = the repository owner; **agent lane** = an automated research-engineering lane run under `docs/EXPERIMENT_AGENTS_PROMPT.md` or `docs/DATA_ACQUISITION_AGENTS_PROMPT.md`. No clinical staff role exists.

| Milestone | Owner and ordered tasks | Required artifacts | Acceptance or blocking condition |
|---|---|---|---|
| **M0 Freeze** | PI: confirm or amend every proposal of Appendix A; record the decisions in `protocol/amendments.md`; sign tag `protocol-v6.0.0`. Agent lane: environment qualification (`docs/EXECUTION_ENVIRONMENT_SPEC.md`), registry and configuration validation, full test suite | Tag, `uv.lock`, image identifiers, qualification report, amendments row | No undecided default; every frozen value in `configs/`; tests green; qualification passes |
| **M1 Access and staging** | Agent lane: `stage-sources --dataset acdc`; `stage-sources --dataset mms` (verification and ED/ES staging only); archive licence texts | Staging manifests, exclusion ledger draft | Every hash matches; ACDC 1,056 files; M&Ms 691 files; 345 cases accounted as 320 + 25 |
| **M2 Source bindings** | Agent lane: finalize `configs/source_bindings/{acdc,mms}.json` inside the container | Bindings with observed values | No guessed key; every verify-on-mount item resolved or blocked |
| **M3 Cohort and splits** | Agent lane: `validate-cohort --strict`; fold, stratum and `mms100_member` manifests | Cohort, split and stratum manifests | Counts equal Section 4.5; no patient in two partitions; fold rule reproduces |
| **M4 ACDC semantics** | Agent lane: `build-semantics --dataset acdc`; QC severities; panels | `B_ACDC`, `Y_ACDC`, provenance, QC tables | Contract tests pass; `z_spacing_method` recorded for 150/150; registry counts match |
| **M5 Formal A and predictor** | Agent lane: `extract-formal --dataset acdc`; `fit-predictor` | `A_ACDC`, predictor, OOF predictions, model card | 57 finite descriptors per row; identifier hashes agree |
| **M6 Transition** | Agent lane: `fit-transition --profile ridge --nested-oof` with every comparator | OOF ledger, λ table with grid-edge flags, final T, comparators | Recovery, SVD and transform tests pass |
| **M7 Eligibility** | Agent lane: `check-semantic-eligibility` (gates of Section 11.2 from the OOF ledger and the stability tables) | Eligible and failed feature manifests | Ineligibility explicit; no relaxed rule |
| **M8 WEDD and S** | Agent lane: `discretize --profile wedd_gaussian_mm_v1` and comparators; stability bootstrap | Candidates, cuts, bandwidths, S, recurrence | Cut-edge tests pass; no test or external fitting |
| **M9 Rules and gate** | Agent lane: `induce-rules --tier variable_precision`; `gate-rules` | Candidate, rejected and eligible rulebooks or `EMPTY.json`; cross-fit ledger | Support recomputes from anchors; gate applied once |
| **M10 Freeze bundle** | Agent lane: inference firewall tests; `freeze-analysis-bundle` | `.stages/FROZEN.json`, route logs on synthetic cases | Reference-substitution test passes; routes partition |
| **M11 Internal test** | Agent lane: `evaluate --split acdc_test --no-fit` | Metrics, H-primary-1 decision, denominator reconciliation | Evaluated once; every invalid denominator `NE` |
| **M12 External** | Agent lane: `build-semantics` and `extract-formal` for `mms` under the lock; `evaluate --split mms_external --no-fit`; `audit-shift` | External metrics by stratum, H-primary-2 decision, 220 replication, shift audit | Lock satisfied; evaluated once; no fitting or pooling |
| **M13 Refit arm and ablations** | Agent lane: `fit-refit-arm` (includes its downstream refit bootstrap); `run-ablations` (ablation grid and the `acdc_dev` downstream refit bootstrap) | Refit-arm report, ablation tables, bootstrap provenance | Never pooled; no post-test best-configuration claim |
| **M14 Optional branches** | Agent lane: `run-equivariance --dataset acdc` (D19); deep encoder (D20) in `env-gpu` (not implemented). *Amended in 6.0.1:* D20 is registered and implemented (Section 8.4) | Branch manifests and metrics | Reported whatever the outcome; primary artifacts unchanged |
| **M16 Amendment 6.0.1** | PI: review the amendment commit and sign the tag `protocol-v6.0.1`. Agent lane: the 6.0.1 stages, configurations and tests (Section 0.3), then the freeze order of Section 14.6 | Tag; the freeze records of Section 14.6; the M&Ms-2 extraction record; the results freeze | Tests 25–33 green; no M&Ms-2 byte read before the tag; every partition evaluated once per bundle |
| **M15 Release** | PI: remake the manuscript from v6 metrics; CLAIM, STARD-AI and TRIPOD+AI crosswalks; licence check of every asset; agent lane: `write-release-manifest` | Figures, tables, claim ledger, release manifest | Every number resolves to an artifact and partition; no clinical claim |

Wall-time budgets are measured, not assumed: they live in `docs/EXPERIMENT_EXECUTION_PLAN.md` 1.0.0.

### 15.1 Command interfaces

```bash
# M1-M3 (allowed for M&Ms before M0: verification and staging only)
python -m tmmri.cli stage-sources --dataset acdc|mms [--verify-only] [--source <root>] [--out <staged root>] [--case <id> ...] [--source-readers <n>]
python -m tmmri.cli validate-cohort --strict [--run-id <id>]
# M4-M10 (ACDC development)
python -m tmmri.cli build-semantics --dataset acdc|mms --run-id <id>
python -m tmmri.cli extract-formal --dataset acdc|mms --run-id <id>
python -m tmmri.cli fit-transition --profile ridge --nested-oof --run-id <id>
python -m tmmri.cli fit-predictor --run-id <id>
python -m tmmri.cli discretize --profile wedd_gaussian_mm_v1 --run-id <id>
python -m tmmri.cli check-semantic-eligibility --run-id <id>                  # M7 gates of Section 11.2
python -m tmmri.cli induce-rules --tier variable_precision --run-id <id>
python -m tmmri.cli gate-rules [--substrate nested_inner_oof|outer_oof] --run-id <id>   # default nested_inner_oof (Section 11.6)
python -m tmmri.cli freeze-analysis-bundle --run-id <id>
# M11-M15 (evaluation, secondary arms, release)
python -m tmmri.cli evaluate --split acdc_test --no-fit --run-id <id>
python -m tmmri.cli evaluate --split mms_external --no-fit --run-id <id>     # refuses unless bundle frozen and protocol tag present (override only --allow-unfrozen-synthetic)
python -m tmmri.cli audit-shift --run-id <id>                                # label-free shift audit (Section 12.3); M&Ms lock as evaluate
python -m tmmri.cli fit-refit-arm --run-id <id>                              # D17; refuses before the recorded external evaluation
python -m tmmri.cli run-ablations --run-id <id>                              # Section 13.3 matrix; secondary outputs under */ablations/
python -m tmmri.cli run-equivariance --dataset acdc --run-id <id>            # D19; ACDC development only before M0
python -m tmmri.cli write-release-manifest --run-id <id>                     # release_manifest.json of the run root
python -m tmmri.cli run --study configs/study_v6.yaml --run-id <id> [--until <step>] [--from <step>]
python -m tmmri.cli bench --cases 10 --out artifacts/bench/<id>.json
python -m pytest -q
# Version 6.0.1 (stage names fixed here; options finalized at integration; M&Ms-2 lock of Section 4.6: exit 3 before protocol-v6.0.1)
python -m tmmri.cli extract-mms2 ...                                        # selective extraction of the 720 mms2_external short-axis files
python -m tmmri.cli stage-sources|validate-cohort|build-semantics|extract-formal --dataset mms2 ...
python -m tmmri.cli evaluate|audit-shift|summarize-predictor --split mms2_external --run-id <id>   # H-primary-3 and F6 (Section 3.3)
python -m tmmri.cli evaluate-h4 --split acdc_test|mms_external|mms2_external --run-id <id>      # H4 family (D36): confirmatory on mms2_external only
python -m tmmri.cli fit-predictor --run-id <id>                              # A2: also stores the five fold predictors
python -m tmmri.cli summarize-predictor|build-manuscript-tables|build-manuscript-figures --run-id <id>   # reporting (Section 16)
python -m tmmri.cli fit-cart-surrogate|evaluate-cart-surrogate --run-id <id>                          # F8
python -m tmmri.cli verify-torch-weights|extract-deep-features|fit-deep-encoder --run-id <id>         # D20
python -m tmmri.cli build-tensor-cache|plan-f9|upload-cache|train-f9|import-f9-fits|select-f9|freeze-f9|infer-f9 --run-id <id>   # F9
python -m tmmri.cli fit-probes|evaluate-alt-bundles|build-comparison-tables --run-id <id>             # F9b; D20, F9, F9b evaluation
python -m tmmri.cli fit-trust|evaluate-trust --run-id <id>                                            # S1-S4
python -m tmmri.cli run-data-efficiency --step descriptor|f9-wave1|f9-wave2|f9-infer|f9-aggregate --run-id <id>   # S5 (f9-wave<N> also write data_efficiency/f9/train_wave<N>_specs.json, purpose s5)
python -m tmmri.cli train-f9 --specs data_efficiency/f9/train_wave<N>_specs.json | import-f9-fits --wave <N> --specs <same> --run-id <id>   # S5 fits after the F9 freeze
python -m tmmri.cli run-simulated-shift --vendor A|B|C|D|all --step descriptors|tensors|d20|f9 --run-id <id>         # S6
python -m tmmri.cli run-exploratory --analysis B1|...|B9 --run-id <id>                               # EXPLORATORY_POST_HOC
# Modal app scripts/cloud/modal_train.py: launched through the uvx-pinned Modal client exactly as scripts/cloud/README.md documents (Section 14.7)
```

Every v6 subcommand accepts `--study`, `--n-jobs` (default `TMMRI_N_JOBS`, then 12), `--source-readers` (concurrent source readers for staging, default `TMMRI_SOURCE_READERS`, then 4), `--data-root`, `--artifact-root`, `--acdc-source`, `--mms-source`, `--force` (rerun despite a matching stage record; *amended in 6.0.1:* in a frozen run every recorded stage refuses it with exit 7, Section 14.2) and `--allow-unfrozen-synthetic` (accepted only for sources with the `SYNTHETIC_FIXTURE` marker; it also waives the cohort-size checks of `validate-cohort`, which cannot hold on a fixture, while every structural check stays strict). `stage-sources` writes `<data-root>/staged/<DATASET>/<case_id>/{ed,es}_{image,mask}.nii` and `<data-root>/manifests/staging_<DATASET>.json` (`verification_<DATASET>.json` with `--verify-only`; an append-only `staging_<DATASET>.log.jsonl`); `validate-cohort` writes `<run>/contracts/` with a run id and `<data-root>/manifests/cohort/` without one. The `run` steps, in order: `stage-sources:acdc`, `stage-sources:mms`, `validate-cohort`, `build-semantics:acdc`, `extract-formal:acdc`, `fit-transition`, `fit-predictor`, `discretize`, `check-semantic-eligibility`, `induce-rules`, `gate-rules`, `freeze-analysis-bundle`, `evaluate:acdc_test`, `build-semantics:mms`, `extract-formal:mms`, `evaluate:mms_external`, `audit-shift`, `fit-refit-arm`, `run-ablations`, `run-equivariance:acdc`, `write-release-manifest`. Before M0 the run stops at `build-semantics:mms` (exit 3); run `--until evaluate:acdc_test`, then `run-ablations` and `run-equivariance` separately.

| Exit code | Meaning |
|---:|---|
| 0 | Success (a stage whose record matches is `SKIPPED`, not an error) |
| 1 | Usage error or missing input (for example a prerequisite stage has not run) |
| 2 | `stage-sources`: at least one case failed verification or staging (`BLOCKED_DATA_ACCESS`) |
| 3 | Refused by the protocol lock: M&Ms endpoint stage before M0 (Section 8.5); any M&Ms-2 command (extraction, verification, staging or endpoint) before the annotated tag `protocol-v6.0.1` is contained in HEAD (Section 4.6; *amended in 6.0.1*); or `--allow-unfrozen-synthetic` on non-synthetic sources |
| 4 | A METHODS API is missing (`NotImplementedError`; the stage stays `SPECIFIED`) |
| 5 | Typed-B contract violation (`tmmri.semantics.contract.ContractViolation`) |
| 6 | `validate-cohort --strict` (or `run`) found a failing check |
| 7 | Analysis bundle error: incomplete, changed after the freeze, a fitting stage in a frozen run, or `fit-refit-arm`/`run-ablations` before their prerequisite evaluation. *Amended in 6.0.1:* also a recorded stage of a frozen run whose inputs, code version, configuration hash, parameters or outputs differ, or that receives `--force` (once-only guard, Section 14.2; a new run id is required) |
| 8 | *Added in 6.0.1:* `HUMAN_GATE`: a release fact differs from the declared M&Ms-2 contract (archive layout, a CSV column, a subject code or a `DISEASE` value outside the synonym table of Section 4.6); fail closed for an owner decision |

`--no-fit` refuses estimator fitting at runtime; evaluation receives read-only access to the frozen bundle. The `mms` variants of `build-semantics` and `extract-formal`, and `audit-shift`, obey the same lock as `evaluate --split mms_external` (Section 8.5). Legacy subcommands (`run-minimal`, `run-full-validation`, and the clinical-validation commands) keep working for the governed and legacy flows and are not v6 stages.

### 15.2 Load-bearing acceptance tests

Each test names the file and function that implements it (● implemented and passing on the host suite of September 28, 2026, and for tests 22 and 25–33 on the integration suites of October 2, 2026: host, the `tmmri` container and the `tmmri-gpu` image on CPU; ○ pending).

1. **Affine recovery:** on synthetic full-rank A and `B = A T_true + 1 b_trueᵀ`, unregularized fitting recovers predictions to relative error < 1e-10 in Float64; with per-target masks the intercept does not drift. → `tests/test_v6_transition.py::test_affine_recovery_with_masks` ●
2. **SVD ridge equivalence:** (E6) agrees with a trusted regularized least-squares solver to relative tolerance 1e-8 under the mean-loss convention. → `tests/test_v6_transition.py::test_svd_ridge_matches_reference_solver` ●
3. **Row alignment:** a permuted identifier table, duplicate identifier, missing case or mismatched row hash causes a hard failure. → partial: duplicate and unsorted identifiers `tests/test_v6_contract.py::test_contract_catches_violations`, misaligned A/B rows raise in `fit_transition_v6`; permuted-table and row-hash tests ○
4. **Typed transforms:** `log`, `logit_percent` (including the clip bounds) and `identity` round-trip to 1e-12; a non-positive value under `log` raises. → `tests/test_v6_transition.py::test_typed_transforms_round_trip` ●
5. **Slice-increment fallback:** a synthetic frame with a unit-scaled affine and `pixdim` (1.5, 1.5, 10) yields `dz = 10` and method `nifti_header_pixdim_unscaled_affine`; a voxel-scaled affine yields the affine norms; an inconsistent pair is critical; the known volume of a synthetic cylinder is recovered under both conventions. → `tests/test_acdc_geometry.py::test_unit_scaled_affine_uses_header_slice_spacing`, `::test_voxel_scaled_oblique_affine_keeps_affine_projection`, `tests/test_v6_ingest.py::test_staged_geometry_uses_the_header_pixdim_fallback`, `tests/test_v6_semantics.py::test_cylinder_phantom_volumes_ef_and_legacy_parity` ●
6. **uint8 mask scaling round trip:** a mask stored as uint8 with `scl_slope` 3/255 reads back, after `np.rint`, as exactly the original labels 0–3; unrounded comparison is shown to fail. → `tests/test_v6_ingest.py::test_labels_are_canonicalized` ●
7. **ED/ES frame selection:** ACDC `Info.cfg` frames and M&Ms 0-based CSV indices select the annotated frames; a case whose annotated frames differ from {ED, ES}, or whose indices are the 0/0 placeholder, is rejected with its reason code. → `tests/test_v6_ingest.py::test_frame_selection_is_zero_based`, `tests/test_v6_staging.py::test_unannotated_es_frame_is_refused` ●
8. **Fold hash rule:** the fold assignment is deterministic, independent of input order, gives 4 patients per class per fold on 100 synthetic patients, is computed on the eligible patients only (on the 98 eligible patients: folds of 20, 20, 20, 20 and 18 with the configured class counts), and changes if the salt changes. → `tests/test_v6_ingest.py::test_v6_fold_rule_is_deterministic_and_stratified`, `::test_v6_folds_are_assigned_on_the_eligible_patients`, `::test_v6_inner_and_refit_folds` ●
9. **Rough sets:** lower, upper and boundary sets match hand-computed sets on a tiny table; variable-precision granules never appear in the strict tier; enumeration is independent of column order. → `tests/test_v6_wedd_roughsets.py::test_lower_upper_boundary_match_hand_computation`, `::test_variable_precision_granules_are_not_strict`, `::test_vprs_discovery_is_exhaustive_deterministic_and_uncapped` ●
10. **WEDD:** planted cuts on a synthetic Gaussian mixture are recovered within 0.02 IQR at zero noise; bins are right-closed at every cut; `wedd_uniform_mm_v1` and `three_quantile_v1` run on the same fixture. → `tests/test_v6_wedd_roughsets.py::test_planted_cut_recovered_at_zero_noise`, `::test_bins_are_right_closed_at_every_cut`, `::test_uniform_and_tertile_profiles_run_on_same_fixture` ●
11. **Routing:** synthetic cases for each of the seven routes; routes partition exactly; conflict and soft routes never emit a label; a domain-blocked or per-object-unstable reconstruction routes `UNAVAILABLE_FEATURE_ABSTAIN` with `FEATURE_DOMAIN_BLOCKED` or `FEATURE_UNSTABLE`. → `tests/test_v6_inference.py` (one test per route, `::test_routes_partition_and_codes_are_registered`), `tests/test_v6_integration.py::test_gated_rulebook_payload_routes_exact_accept_and_qc_first`, `tests/test_v6_acceptance.py::test_domain_guard_and_unstable_estimates_route_with_registered_codes` ●
12. **No tautology:** consequent columns and their deterministic functions never appear as antecedents. → `tests/test_v6_wedd_roughsets.py::test_tautology_guard`, `tests/test_registry_check.py::test_qc_and_metadata_are_never_features_or_antecedents` ●
13. **No reference at inference:** substituting every evaluation B and Y with sentinels leaves predictions and routes unchanged (QC tokens, derived from the staged image and mask, are kept; the measured-B oracle consumes measured B by definition and is excluded); the evaluation-only metrics change. → `tests/test_v6_acceptance.py::test_reference_substitution_leaves_predictions_and_routes_unchanged` ●
14. **Leakage and external lock:** the estimator-to-partition registry lists no `acdc_test` or M&Ms partition under fitting or selection (refit-arm objects excepted on `mms_refit_train`); `evaluate --split mms_external`, `build-semantics --dataset mms` and `extract-formal --dataset mms` refuse without the tag and `.stages/FROZEN.json`. → `tests/test_registry_check.py::test_estimator_partition_registry_has_no_forbidden_fitting`, `tests/test_v6_cli.py::test_mms_endpoint_stages_refuse_before_the_freeze`, `::test_cli_exit_code_for_refused_mms_stage` ●
15. **Worker-count invariance:** B, A and the transition built with 1 and with 4 workers are byte-identical. → `tests/test_v6_staging.py::test_staging_and_formal_extraction_are_identical_for_n_jobs_1_and_3`, `tests/test_v6_semantics.py::test_semantics_table_is_sorted_and_invariant_to_n_jobs`, `tests/test_v6_metrics.py::test_primary_rho_point_ci_and_n_jobs_invariance`, `tests/test_v6_wedd_roughsets.py::test_stability_gate_is_invariant_to_n_jobs` ●
16. **Staging cache verification:** a tampered staged file is detected; re-staging skips a case whose record matches and re-reads none of its sources. → `tests/test_v6_staging.py::test_staging_is_idempotent_and_content_addressed` ●
17. **Synthetic end-to-end control:** a 10-d formal → 6-d semantic → 3-rule generator with known T and cuts (seed 20260613) recovers all three rules at zero noise and reports degradation at semantic noise σ ∈ {0.03, 0.06, 0.10, 0.15, 0.20}. → `tests/test_v6_synthetic_control.py::test_exact_recovery_at_zero_noise`, `::test_degradation_is_reported_not_hidden` ●
18. **Cross-fitted gate:** on a toy table the pooled matches and Wilson bounds equal hand computation (6/6 → 0.61 eligible; 5/5 → 0.57 ineligible). → `tests/test_v6_wedd_roughsets.py::test_cross_fitted_gate` ●
19. **Registry integrity:** the YAML registry validates; counts equal the registry totals asserted in `tests/test_registry_check.py`; every reason code maps to a declared route. → `tests/test_registry_check.py` ●
20. **No fabrication:** blocked mode writes no data-bearing artifact. → `tests/test_v6_integration.py::test_blocked_staging_writes_no_data_bearing_artifact` ●
21. **Optional equivariance:** a whole-pixel translation of image and mask leaves B identical; skipped until `equivariance` implements the v6 branch. → `tests/test_v6_integration.py::test_whole_pixel_translation_leaves_b_identical`, `tests/test_v6_semantics.py::test_equivariance_v6_selection_and_config_epsilons` ●
22. **Optional deep encoder:** skipped with a uniform message until the D20 module exists. → ○ (`src/tmmri/deep_encoder.py` not implemented). *Amended in 6.0.1:* D20 is registered. On synthetic volumes, the test checks the frozen preprocessing of Section 8.4, the 2048-d per-patient vector, the recorded weight SHA-256 and the downstream v6 settings. → `tests/test_deep_encoder.py::test_per_slice_normalization_follows_the_frozen_rule`, `::test_fit_deep_encoder_writes_frozen_d20_bundle`, `::test_d20_extraction_is_deterministic_on_a_random_resnet50` (GPU image, CPU), `tests/test_v601_neural_chain.py::test_neural_chain_on_a_frozen_synthetic_run` ●
23. **Stage DAG:** `tmmri.cli run` on synthetic fixtures runs every stage of Section 15.1 including the shift audit, the refit arm, the ablations (with the linearity criterion and the WEDD settings grid), equivariance and the release manifest, and writes the external strata with `NE` below the reporting threshold, the fixed-sequence decision with the 220-case sensitivity and the BH tables of F1–F5; a second run skips every stage; an evaluation-only configuration change re-runs only the downstream stages; a fitting-stage configuration change in a frozen run is refused. *Amended in 6.0.1 (roadmap A1):* in a frozen run, a recorded stage may only be skipped. A change of its inputs, code version, configuration hash or parameters, a changed recorded output, or `--force` is refused with exit 7, and the stage is not re-executed → `tests/test_v6_reporting.py::test_once_only_guard_refuses_recorded_stages_of_a_frozen_run`, `tests/test_v6_cli.py::test_cli_run_synthetic_end_to_end`, `tests/test_v6_runs.py::test_run_stage_skips_on_identical_inputs_and_reruns_on_changes`, `tests/test_v6_integration.py::test_stage_sections_resolve_and_bundle_excludes_evaluation_only_sections`; QC panels, WEDD candidates, decoder logs, the F1 table, the within-family BH rule, the shift-audit classifier and the WEDD grid in `tests/test_v6_acceptance.py` ●
24. **Declared duplicate images and cluster bootstraps** (*Amended September 28, 2026:* PI decision): a declared pair is excluded or clustered as configured (`patient055` and `patient076` leave `acdc_dev` with their reason codes; K3R0Y7 leaves `mms_refit_eval` only; each pair shares one `cluster_id`); an undeclared voxel-identical pair fails `validate-cohort --strict`, and so does a declared pair whose detected frames differ; the cluster bootstrap equals the patient bootstrap when every cluster is a singleton and draws the two members of a pair together otherwise (20 duplicated pairs reproduce the 20-case interval exactly); the cluster-level Wilcoxon test counts a pair once; results are identical for 1 and 3 workers. → `tests/test_v6_duplicates.py`, `tests/test_registry_check.py::test_declared_duplicate_images_are_registered_and_sizes_consistent` ●

Version 6.0.1 acceptance tests (Section 0.3). Synthetic fixtures only. The cross-lane chain (tensor cache → F9 plan, local training, selection, refits and freeze → local inference → D20 → F9b, S1–S4 and the alternative-bundle evaluations on `acdc_test` and an external partition → S5 F9 after the freeze) runs on the CPU in `tests/test_v601_neural_chain.py::test_neural_chain_on_a_frozen_synthetic_run`.

25. **M&Ms-2 lock:** before an annotated `protocol-v6.0.1` contained in HEAD, every M&Ms-2 command of Section 4.6 exits 3 and writes nothing. `--allow-unfrozen-synthetic` is accepted only on a `SYNTHETIC_FIXTURE` source. → `tests/test_mms2_ingest.py::test_mms2_compute_stages_exit_3_before_the_tag`, `::test_extraction_refuses_before_the_tag_then_verifies_the_hash`, `::test_mms2_lock_override_only_for_synthetic_fixtures` ●
26. **M&Ms-2 split and ingest:**
    - The split is deterministic and independent of input order. It gives 180/180 on `001`–`360`, changes if the salt changes, and reads no data.
    - On a synthetic NIfTI fixture: labels map by identity; a unit-scaled affine takes the header `pixdim`; phases come from the file names; only the short-axis ED/ES members are extracted; sealed rows and members are never written.
    - The duplicate rule excludes matches against ACDC, M&Ms-1 and within `mms2_external`.
    - → `tests/test_mms2_ingest.py::test_split_manifest_is_reproducible_from_the_rule`, `::test_extraction_list_is_exactly_the_720_external_short_axis_files`, `::test_release_csv_contract_drops_sealed_rows_before_parsing_values`, `::test_vendor_scanner_field_are_data_driven_and_disease_synonyms_fail_closed_only_outside_the_table`, `::test_case_ids_are_prefixed_downstream_and_the_split_keeps_the_raw_code`, `::test_geometry_rule_and_label_identity`, `::test_extraction_writes_only_the_external_files_and_rows`, `::test_duplicate_rule_excludes_matches_against_acdc_mms1_and_within_mms2`, `tests/test_mms2_pipeline.py::test_mms2_stages_on_a_frozen_synthetic_run` ●
27. **Registry:** every 6.0.1 estimator has a row with no fitting or selection on `acdc_test`, `mms_external` or `mms2_external`. The S3 statistics and the shift audit are admitted only as declared label-free diagnostics. → `tests/test_registry_check.py::test_estimator_partition_registry_has_no_forbidden_fitting` ●
28. **Once-only on 6.0.1 stages:** every 6.0.1 stage that evaluates a frozen bundle refuses re-execution in a frozen run with exit 7. → `tests/test_mms2_pipeline.py::test_mms2_stages_on_a_frozen_synthetic_run`, `tests/test_l57_stages.py::test_l57_stages_end_to_end_and_once_only`, `tests/test_exploratory.py::test_lane_l3_stages_on_a_synthetic_run`, `::test_b5_stage_provenance_and_once_only_guard`, `tests/test_f9_stages.py::test_select_import_freeze_and_once_only`, `tests/test_deep_encoder.py::test_fit_deep_encoder_writes_frozen_d20_bundle`, `tests/test_v601_reporting.py::test_extended_summary_refuses_a_changed_comparator_and_obeys_the_mms2_lock`, `tests/test_data_centric_stages.py::test_s6_vendor_geometry_descriptor_deltas_and_once_only` ●
29. **F9 fit specification and numerics:**
    - `fit_id` is the first 16 hexadecimal digits of the SHA-256 of the canonical JSON of the other fields.
    - The planner yields 342 F9 and 570 S5 fit specifications.
    - Repeated synthetic CPU fits are bitwise identical, and a resumed fit equals an uninterrupted one.
    - Every record carries the device, the versions and the TF32 state of Section 14.7.
    - → `tests/test_f9_selection.py::test_fit_id_is_stable_and_order_independent`, `::test_plan_counts_324_and_18`, `::test_refit_and_s5_specs`, `tests/test_data_efficiency.py::test_f9_spec_counts_are_540_selection_plus_30_refits`, `tests/test_f9_train.py::test_qualification_bitwise_repeat_resume_and_ieee`, `::test_selection_fit_record_and_idempotence` (GPU image, CPU) ●
30. **Tensor cache:** on synthetic volumes of known geometry, the cache checks shapes, dtype float32, the [0, 1] range, the 1.5 mm in-plane resampling, the 224 × 224 crop or pad, the geometry scalars and the manifest columns. Labels are empty where they are not allowed. → `tests/test_neural_preprocess.py::test_build_tensor_cache_round_trip_from_staged_nifti`, `::test_physical_size_is_preserved_across_source_spacings`, `::test_geometry_scalars_orientation_and_determinism`, `::test_mms_cache_refuses_before_the_freeze_and_has_no_labels`, `tests/test_v601_integration.py::test_s6_hook_is_the_build_tensor_cache_transform` ●
31. **Trust:** S1 recovers a known coverage on synthetic confidences; S2 flags a planted shifted cluster; S4 recovers a known temperature. → `tests/test_l57_trust.py::test_thresholds_give_the_planned_development_coverage`, `::test_mahalanobis_flags_and_auroc_detect_a_planted_shift`, `::test_temperature_recovers_a_planted_temperature_and_log_probabilities_agree` ●
32. **Data centric:** S5 subsamples are deterministic and class-stratified, and a missing class gives `NE_INSUFFICIENT_CLASS_SUPPORT`. S6 resampling round-trips the geometry of a synthetic phantom. → `tests/test_data_efficiency.py::test_subsample_hash_rule_is_deterministic_and_order_free`, `::test_subsample_is_class_stratified_with_round_half_up_counts`, `::test_insufficient_class_support_is_ne_and_emits_no_specs`, `tests/test_simulated_shift.py::test_round_trip_restores_geometry_and_masks_exactly_through_a_finer_grid` ●
33. **Comparators and exploratory register:**
    - F8 trees have depth ≤ 3 and leaves ≥ 5, and the selective leaves satisfy the gate.
    - F9b uses the v6 `transition_fit` and never pools latent coordinates across encoders.
    - Each of B1–B9 writes a provenance record labelled `EXPLORATORY_POST_HOC`; B9 measures the orientation by hand-checked rules.
    - D4 (October 3, 2026): the eight elements form a closed group; the F9 and D20 outputs for an input and for any D4 transform of it agree within 1e-6; the D4 augmentation draw is seeded, first and shared by both phases; the `tta` key leaves the v6.0.0 `deep_encoder` digest unchanged.
    - → `tests/test_cart_f8.py::test_settings_are_the_registered_ones`, `::test_nested_ledger_and_selective_gate_by_hand`, `tests/test_l57_stages.py::test_probes_never_pool_latent_coordinates_across_encoders`, `tests/test_exploratory.py::test_b5_stage_provenance_and_once_only_guard`, `::test_lane_l3_stages_on_a_synthetic_run`, `::test_b9_orientation_measurement_by_hand`, `tests/test_dihedral.py` (torch parts in the GPU image, CPU) ●

### 15.3 Resource policy (Decision D3)

The primary pipeline is CPU-only. Per-case stages (staging, B, A) run in parallel with `TMMRI_N_JOBS` workers (12 by default in `configs/runtime.env`) and one BLAS thread per worker; downstream stages are small (100 × 57 design, a few thousand candidate conjunctions) and run in one process. Staging reads each source once: the 320 labelled M&Ms cases occupy 14.621 GB compressed (67.173 GB decompressed), while the staged ED/ES-only cache is expected to be about 1–3 GB [LOCAL estimate]; full decompression is never needed. The GTX 1080 is used only by the optional deep-encoder branch in the `env-gpu` target. `python -m tmmri.cli bench` measures per-case rates; the time model and budgets are recorded in `docs/EXPERIMENT_EXECUTION_PLAN.md`, not here. A compute failure is `BLOCKED_COMPUTE`; a smoke subset is never the published cohort.

**Execution on the host.**
- **Container:** every v6 stage runs in the `tmmri` compose service (target `env`); the optional D20 branch runs in `tmmri-gpu` (profile `gpu`).
- **Volumes:** `tmmri_data` → `/workspace/data`, `tmmri_artifacts` → `/workspace/artifacts`, `tmmri_cache` → `/cache`; the X-ray project's `cxr-audit_*` volumes are never touched.
- **Sources:** `${TMMRI_ACDC_SOURCE}` → `/sources/acdc` and `${TMMRI_MMS_SOURCE}` → `/sources/mms`, both read-only.
- **Run identifiers:** `v6-<purpose>-YYYYMMDD-HHMM`.

---

## 16. Expected tables, figures, supplementary artifacts, and matrix files

All assets are generated from frozen metrics by the manuscript asset script; none is hand-typed. *Amended in 6.0.1:* Table 1 is built only when this file equals `SPECIFICATION.md` of the protocol tag the run was executed under: `protocol-v6.0.0` for a v6.0.0 run, `protocol-v6.0.1` for a run frozen with 6.0.1 code (`FROZEN.json` `protocol_version_tag`) or holding a record of a 6.0.1 stage, such as `evaluate:mms2_external` on the frozen v6.0.0 bundle (`tmmri.manuscript_tables.run_protocol_tag`).

| Asset | Required content and provenance |
|---|---|
| Table 1 | Methodological reconciliation and distinct contribution relative to P1–P6 and to v5 (Sections 2.2–2.3) |
| Table 2 | Cohort flow: ACDC 150 (100 + 50); M&Ms 345 → 320 + 25 excluded; strata by vendor, centre, release split, pathology and MMS_100 membership; QC exclusions |
| Table 3 | Final registry: supported and unsupported features, tiers, transforms, harmonization |
| Table 4 | Primary reconstruction on `acdc_test`: per target versus mean, pseudoinverse and nonlinear comparators with units and CIs; `ρ` and the H-primary-1 decision |
| Table 5 | External reconstruction on `mms_external`: per target, `ρ`, H-primary-2, by vendor and centre; the 220 and 100 strata |
| Table 6 | Audited predictor: OOF, `acdc_test`, crosswalk-evaluable M&Ms; IHD counts; off-ontology `d_model` distribution; semantic comparator |
| Table 7 | Rulebook metrics for primary, companion and oracle rulebooks; routes by reason code; denominator reconciliation |
| Table 8 | Refit-arm comparison on `mms_refit_eval`, seen versus unseen vendors, paired with the frozen bundle |
| Table 9 | Ablations: formal A 57 versus 128, typed versus identity transforms, discretizers, λ policy, rule policy |
| Table 10 | Failures and non-estimable results: ineligible features and classes, empty rulebooks, `NE` metrics, grid-edge λ, thresholds within error |
| Figure 1 | Artifact and evidence flow: masks → A and B, reconstruction, S, gates, routes |
| Figure 2 | Cohort flow and partition diagram with the untouched external branch and the never-analysed stratum |
| Figure 3 | ACDC ED/ES frames with reference masks and the geometry method (ACDC only, licence permitting) |
| Figure 4 | Predicted versus reference primary targets and Bland–Altman plots, internal and external |
| Figure 5 | **Per-vendor external figure**: per-target normalized error and bias by vendor and centre, with spacing and field-of-view covariates |
| Figure 6 | **Refit-arm comparison**: frozen ACDC bundle versus refit arm on `mms_refit_eval`, seen versus unseen vendors |
| Figure 7 | **Never-analysed replication**: `ρ` and per-target `ρ_j` on 320, 220 and 100, with intervals |
| Figure 8 | Coverage versus covered fidelity with conflict and abstention visible; gate sensitivity |
| Figure 9 | WEDD candidate objectives and bootstrap cut stability beside the comparators |
| Optional Figure 10 | Equivariance defects and, if run, deep-encoder comparison |
| Table 11 (6.0.1) | M&Ms-2 cohort flow (split, duplicate-image exclusions, QC) and H-primary-3: `ρ` with its interval and decision, per-target `ρ_j` with F6 q-values, strata; exposure label "prospective" |
| Table 12 (6.0.1) | Three representations (descriptors, D20, F9b): `ρ` and `ρ_j`; three predictors (audited, D20, F9; F9 three-seed range and SD): accuracy, macro-F1, AUROC, ECE, Brier score, log loss; paired differences; exposure label per partition |
| Table 13 (6.0.1) | Trust and data-centric results: S1 realised coverage and accuracy at τ₅₀ and τ₈₀ and AURC; S2 flag rates and error-detection AUROC; S3 paired differences; S4 calibration before and after; S6 deltas; F8 coverage and fidelity |
| Figures 11–14 (6.0.1) | 11: the prospective M&Ms-2 result beside ACDC test and M&Ms-1. 12: the representation panel. 13: risk–coverage with OOD flags. 14: harmonization and simulated shift. Generated by `build-manuscript-figures` from recorded artifacts. No M&Ms image, no case identifier, and cells with n < 20 suppressed |
| Supplement (6.0.1) | S5 learning curves, reliability diagrams, F9 seeds and fit records, cross-hardware check, B1–B9 (labelled exploratory), the protocol-deviation log |

Qualitative cases are chosen by a documented sampling policy, never cherry-picked; M&Ms image content is not published (Section 14.5). **Supplementary release:** registry, crosswalk, configurations, fold and stratum hashes, the A extraction specification, fitted T and intercepts, OOF ledgers, WEDD candidates and cuts, granule memberships (as counts where licences require), candidate, rejected and eligible rulebooks, route statistics, bootstrap seed lists, tests. Artifact status vocabulary: `SPECIFIED`, `SCHEMA_READY`, `UNIT_TESTED_SYNTHETIC`, `BLOCKED_DATA_ACCESS`, `BLOCKED_COMPUTE`, `BUILT`, `INELIGIBLE`, `EVALUATED`, `RELEASED`; an empty rulebook after a real gate differs from an absent rulebook.

---

## 17. Risks, limitations, unsupported claims, and mitigation

| Risk or limitation | Mitigation and reporting boundary |
|---|---|
| **Mask-conditioned A** | A and B are computed from the same reference masks; reconstruction measures what spacing-free descriptors of image and mask predict, not what an image-only model knows. No segmentation or image-only claim; the D20 branch is the only image-only A and is optional |
| **Unitless A** | A carries no spacing, so absolute volumes and mass cannot be recovered across vendors except through correlates; this is hypothesis H3, reported whatever the outcome, and never fixed post hoc by adding spacing to A |
| **Small single-centre development cohort** | ACDC development has 98 patients from one centre (20 per class; 19 MINF and 19 NOR after the duplicate-image exclusions); estimates are imprecise; cross-fitting replaces a gate partition; support gates may leave classes or the rulebook empty |
| **Internal test previously analysed** | `acdc_test` was evaluated under v4/v5 (pre-fix pipeline); its results were seen; H-primary-1 is therefore not a fully blinded test and is reported with this caveat |
| **M&Ms 100 previously unblinded** | The MMS_100 cases were analysed under v5; they form a separate stratum; the 220 never-analysed cases are the confirmatory sensitivity stratum |
| **Off-ontology external classes** | 43 external cases have no ACDC class (HHD, LVNC, AHS, Other) and 4 IHD are not exact; no class accuracy there; route and `d_model` distributions only |
| **Annotation-protocol differences** | Reference masks follow different dataset protocols (basal slice, papillary handling); same-algorithm measurement does not erase them; stated in every external table |
| **Missing anthropometrics** | BSA-indexed quantities unsupported for M&Ms external claims |
| **CRLF-pinned historical hashes** | Legacy hash manifests depend on CRLF worktrees and 4 v5 release hashes already mismatch; legacy evidence is not re-verified under LF and is never a v6 input |
| **Manuscript describes v5** | `manuscript/article/main.tex` reports pre-fix v5 results (including the misattributed M&Ms bias); it must be remade after v6 and must not be submitted as is |
| **Licence exposure** | The tracked M&Ms CSV and legacy per-case tables must not reach a public mirror (Section 14.5) |
| **Linear misspecification** | Mean and nonlinear comparators; bias and domain failures reported; inspectability is not adequacy |
| **Post-hoc rationalization** | Fidelity to the predictor and agreement with references reported separately; no causal claim from a coefficient or rule |
| **Variable-precision wording** | Non-strict rules never described as certain |
| **Abstention presented as safe referral** | Called computational abstention only [P5] |
| **Proposals not yet confirmed** | Every coordinator proposal is listed in Appendix A and must be confirmed or amended at M0; none is presented as a PI decision |
| **CNN comparators are black boxes** (6.0.1) | F9 is reported as a classifier, and F9b as a linear probe that measures how well concepts can be decoded from F9's activations, not whether the CNN uses them. No explanation of the CNN's decisions is claimed. Rules over CNN decisions are not fitted in v6; they are deferred to protocol v7 (NEURAL-TM-1.0, planned as `protocol-v7.0.0`), whose frozen design F9 results cannot change |
| **Registered after unblinding** (6.0.1) | Every 6.0.1 analysis on `acdc_test` and `mms_external` carries this label and is never confirmatory; only `mms2_external` results are prospective (Section 12.2) |
| **Small prospective cohort** (6.0.1) | About 180 clusters; H-primary-3 passes only if ρ is about 0.28 or more (Section 3.3); a failure is reported as a prospective replication of the limitation |
| **Transductive and simulated analyses** (6.0.1) | S3 uses unlabelled statistics of the evaluated partition and is reported as transductive adaptation, not as a deployment claim. S6 simulates geometry only, not acquisition physics |
| **Cloud training and hardware** (6.0.1) | Modal fits are deterministic only on the pinned GPU type and software stack; every reported number comes from local inference, and the cross-hardware differences are reported (Section 14.7) |

**Explicitly unsupported claims:** clinically validated; diagnostic accuracy for clinical use; automatic segmentation performance; physician-approved rulebook; reader-study result; causal explanation; AHA segmental or regional wall-motion findings; scar or tissue characterization; BSA-indexed external findings; generalization to unseen vendors beyond the four M&Ms vendors; first-ever method. *Added in 6.0.1:* an explanation of the CNN's decisions; rules over CNN decisions; prospective status for any `acdc_test` or `mms_external` result; deployment value of label-free harmonization; confirmatory status for any comparator, S-analysis or B-analysis.

---

## 18. Journal-fit strategy and manuscript framing

### 18.1 Evidence

The owner selected *Big Data and Cognitive Computing* (MDPI; ISSN 2504-2289) on September 29, 2026 (Decision D4; `protocol/amendments.md`). `manuscript/article/main.tex` now uses the MDPI class option `bdcc` instead of the placeholder `journal`; the class's journal list maps it to the full name and the ISSN [J1]. The companion CXR-TRANS-AUDIT protocol targets *Journal of Imaging* [J2]; that choice does not bind this protocol.

| Journal | Verified scope and indexing | Citation evidence | Requirements and fit |
|---|---|---|---|
| **Big Data and Cognitive Computing** (MDPI; ISSN 2504-2289) — owner decision | Indexed in Scopus, ESCI (Web of Science), dblp, Inspec and Ei Compendex; JCR Q1 (Computer Science, Theory and Methods) and CiteScore Q1 (Computer Science Applications); median first decision about 23.3 days after submission (first half of 2026) [J3; **VERIFIED** 2026-09-29] | Impact Factor 5.3 (2025) and 5-year Impact Factor 4.9 (2025) as displayed by the publisher [J3; **VERIFIED** 2026-09-29]; career documentation still needs dated provider exports | Sections *Data Mining and Machine Learning*, *Cognitive System* and *Artificial Intelligence and Multi-Agent Systems* host explainable-AI and medical-image work; reproducible methods; data, code, ethics and AI-use statements |

**Candidate Special Issues** (open for submission on September 29, 2026; deadlines are the publisher's and may change) [J4–J8; **VERIFIED** 2026-09-29]:

| Special Issue (Section) | Submission deadline | Fit to this protocol | Status |
|---|---|---|---|
| *Data-Centric and Trustworthy AI for Biomedical Signal and Image Analysis* (Artificial Intelligence and Multi-Agent Systems) [J5] | 31 January 2027 | Its listed topics are data curation and quality control (harmonization), robustness under multi-centre dataset shift, reliability assessment, leakage prevention, provenance and reproducible benchmarks: Sections 7, 12.3–12.4, 13 and 14 of this protocol (verify-and-stage provenance, duplicate-image control, the QC severity model, the four-vendor external test and the shift audit) | **Selected** (owner decision, September 29, 2026); open through the end of October 2026 and beyond |
| *Next-Generation Medical Image Analysis: Multimodal, Decentralized, Fair and Reasoning-Centric Approaches* (Cognitive System) [J6] | 18 December 2026 | Medical imaging with transparent, clinically meaningful reasoning, robustness and uncertainty; oriented to foundation models, vision-language models and agentic systems, which this protocol does not use | Not selected |
| *Beyond Diagnosis: Machine Learning in Prognosis, Prevention, Healthcare, Neurosciences, and Precision Medicine* (Data Mining and Machine Learning) [J7] | 1 November 2026 | Lists explainable and interpretable machine learning for clinical decision making and medical images, but centres on prognosis, prevention and precision medicine rather than audits of diagnostic-class decisions | Not selected; the only healthcare issue whose deadline falls at the end of October 2026 |
| *Artificial Intelligence Models and Cognitive Computing: Innovations from Algorithms to Intelligent Systems* (Cognitive System) [J8] | 31 December 2026 | Neuro-symbolic reasoning and explainable, trustworthy AI in general; healthcare is one application area | Not selected |

The owner selected the Special Issue *Data-Centric and Trustworthy AI for Biomedical Signal and Image Analysis* (Section *Artificial Intelligence and Multi-Agent Systems*; Guest Editor Dr. Yizhe Zhang; deadline 31 January 2027) on September 29, 2026 [J5]. A journal or Special Issue choice is a submission-fit judgment, not a guarantee of acceptance; the deadline is the publisher's and is re-checked before submission.

### 18.2 Contribution framing

Working title [PROPOSED]: **"Data-Centric Auditing of Cardiac MRI Decisions with Transition-Matrix Semantics and Cross-Fitted Rough-Set Rules: ACDC Development and Complete Multi-Vendor M&Ms Evaluation."** Narrative: the audit problem; the typed A/B/T/S architecture with the corrected geometry chain; exact estimation and matched comparators; internal and external reconstruction with the fixed-sequence smallest-effect decision; vendor, centre and pathology strata with the never-analysed replication; the refit ceiling; rule coverage, fidelity, conflict and abstention; explicit failures, including the correction of the v5 external bias attribution. Cite each source at its actual contribution boundary; treat P5 and P6 as reviewed manuscripts unless publication metadata is verified. For *Big Data and Cognitive Computing* the framing stresses the cognitive-computing contribution (reason-coded symbolic rules over transitioned semantics, with abstention) and the data-centric reliability contribution (hash-verified provenance, duplicate-image control, QC severities and the four-vendor shift analysis).

**Special Issue crosswalk.** The manuscript addresses each research area listed by the selected Special Issue [J5] only where this protocol produces evidence for it; areas outside the protocol are stated as such and never claimed.

| Listed research area of the Special Issue | Where this protocol addresses it | Manuscript evidence |
|---|---|---|
| Data curation and quality control (noise, artifacts, missingness, harmonization) | §6.1 typed states (values null unless `OBSERVED`), §7 verify-and-stage and contract validation, §12.4 four-level QC severity with feature masking, §4.5 and D15–D17 duplicate-image control | QC and harmonization table, staging and contract reports, the declared duplicate pairs |
| Annotation-efficient learning (weak, semi- or self-supervised, active learning) | Not addressed: reference masks and labels are used as released (D7) | Stated as outside scope |
| Robustness under dataset shift (multi-centre generalization, OOD detection, domain adaptation) | §12.3 once-only external test on four vendors and five centres, seen versus unseen vendors in the refit arm (D17) and the label-free shift audit; no domain adaptation (D16 forbids refitting) | Per-vendor and per-centre panels, refit-ceiling comparison, shift-audit AUROC |
| Uncertainty quantification, calibration and reliability assessment (including ground-truth-free QA) | §13.1 cluster bootstrap intervals, §13.2 calibration (ECE) of the audited predictor, §11.4 WEDD stability gate, §11.7 reason-coded abstention; the shift audit is ground-truth-free | Interval tables, calibration (ECE) of the audited predictor, abstention by reason code |
| Bias and fairness analysis, leakage prevention, provenance and reproducible benchmarks | §12.3 leakage firewall and estimator-to-partition registry, duplicate-image control, §14 provenance, replay boundary and content-addressed stage records, stratified reporting by vendor, centre and pathology; descriptive demographic strata by sex and age band (Q28; Section 12.3) | Leakage and provenance statement, release manifest, stratified tables |
| Privacy-preserving learning and analytics (federated learning, differential privacy, secure computation) | Not addressed: data stay local and are never redistributed (§14.5); no federated or private learning | Stated as outside scope |
| Deployable pipelines for real-time or streaming signals and clinical translation | Partly: a digest-pinned container and a content-addressed pipeline (D22, D23); no real-time, streaming or clinical deployment (§1.5) | Reproducibility section; non-claims |

### 18.3 Submission checklist

Complete CLAIM 2024, STARD-AI (for the predictor's class accuracy) and TRIPOD+AI (for the predictor) with manuscript locations; include the no-new-clinician statement, reference-standard provenance, licence and redistribution restrictions, the ethics determination and the limitations of Section 17; make every number traceable to a generated artifact; state that the reader study is separate and unreported. For the selected Special Issue: prepare the manuscript with the MDPI template option `bdcc`; submit through the publisher's submission system to *Data-Centric and Trustworthy AI for Biomedical Signal and Image Analysis* before its deadline of 31 January 2027 (re-check the deadline first [J5]); in the cover letter, map the contribution to the Special Issue crosswalk of Section 18.2; state data availability within the ACDC and M&Ms licences (no redistribution; code, configurations, manifests and hashes are shared, images are not).

---

## 19. Final checklist and source registers

### 19.1 Requirement-to-section audit

`Specified` means an actionable requirement is present; it does not mean an experiment was executed.

| Requirement | Where implemented | Status |
|---|---|---|
| Six sources with verified file hashes | §2, §19.3 | Hashes recomputed September 28, 2026; identical to the CXR register |
| Lineage, v5 discrepancies and fact corrections | §2.2–2.3 | Specified |
| Research question, outputs, non-claims | §1, §3 | Specified |
| Complete M&Ms cohort decision with exclusions and strata | §4, §12 | Specified; data access verified locally (D2) |
| Standards separated by category | §5 | Specified |
| Typed registry with tiers, states, units, transforms, roles | §6, `registry/semantic_registry.yaml` | 83 features, 42 metadata and 32 QC columns, 323 typed columns; 25 transition targets; 9 rule candidates |
| Staging, adapters, geometry rule, validation, blocked behaviour | §7 | Specified |
| Formal A, audited predictor, deep-encoder branch, external lock | §8 | Specified |
| Transition with intercept, masks, nested folds, comparators, linearity criterion | §9 | Specified |
| Optional equivariance/ETM | §10 | Specified |
| WEDD, VPRS, cross-fitted gate, routes, reason codes, templates | §11 | Specified |
| Splits, leakage firewall, external protocol, crosswalk, refit arm, QC | §12 | Specified |
| Metrics, fixed-sequence primary test, multiplicity, ablations | §13 | Specified |
| Provenance, stage DAG, replay boundary, blocked protocol, sharing | §14 | Specified |
| Roadmap, CLI, acceptance tests, resources | §15 | Specified; tests delivered by the implementation lanes |
| Assets, risks, journal fit | §16–18 | Specified; journal not selected |
| Decision provenance | §0.2, Appendix A, ADR-0002 | Recorded; every proposal confirmed by the owner (2026-09-29) |
| Version 6.0.1 additions: A1 wording, M&Ms-2, H-primary-3, F8, D20, F9, F9b, S1–S6, B1–B9, D4 symmetrization of D20 and F9, exposure labels, freeze order, data governance, claim boundary | §0.3, §3.3, §4.6, §8.4–8.6, §12.2–12.3, §13.3, §13.5–13.6, §14.2, §14.6–14.7, §15.1–15.2, §16, §17 | Specified; binding only after the owner signs `protocol-v6.0.1`; tests 25–33 delivered at integration |

### 19.2 Artifact-status declaration

```yaml
specification: 6.0.0_complete_for_implementation
freeze_record: git_tag_protocol-v6.0.0_pending_at_M0
decision_status: owner_instructions_plus_coordinator_proposals_pending_M0
research_scope: retrospective_research_only
new_healthcare_professional_involvement: prohibited
selected_cmr_datasets: [acdc, mms1_opendataset_labelled]
cohort_counts: {acdc_dev: 98, acdc_test: 50, mms_external: 320, mms_external_never_analysed: 220, mms_refit_train: 150, mms_refit_eval: 169}   # amended 2026-09-28 (declared duplicate images)
cluster_counts: {acdc_dev: 98, acdc_test: 50, mms_external: 318, mms_external_never_analysed: 220, mms100_member: 99, mms_refit_train: 150, mms_refit_eval: 168}
excluded_sources: {MMS_RELEASE_UNLABELED: 25, DUPLICATE_IMAGE_OF_TEST_CASE: 1, DUPLICATE_IMAGE_WITHIN_PARTITION: 1, DUPLICATE_IMAGE_OF_REFIT_TRAIN_CASE: 1}   # the last excludes K3R0Y7 from mms_refit_eval only
external_policy: once_only_frozen_evaluation_fixed_sequence
secondary_arm: mms_refit_ceiling_never_pooled
reader_study_package: clinical_validation_protocol_v2_outside_this_protocol
semantic_contract: {features: 83, metadata_context_columns: 42, qc_columns: 32, total_typed_B_columns: 323, transition_targets: 25, rule_antecedent_candidates: 9}
data_access:
  acdc_release_1056_files: VERIFIED_LOCAL
  mms_full_release_691_files: VERIFIED_LOCAL
real_dataset_artifacts:
  B_ACDC.parquet: NOT_GENERATED
  B_MMS.parquet: NOT_GENERATED        # locked until the M0 tag and .stages/FROZEN.json
  A_matrices: NOT_GENERATED
  audited_predictor: NOT_GENERATED
  fitted_transitions: NOT_GENERATED
  rulebooks: NOT_GENERATED
  empirical_metrics: NOT_GENERATED
synthetic_end_to_end: UNIT_TESTED_SYNTHETIC   # tests/test_v6_cli.py::test_cli_run_synthetic_end_to_end (every stage incl. audit-shift, strata, BH F1-F5, skips, reruns); never a cohort result
acceptance_tests_15_2: {implemented: 22, partial: [3], pending: [22]}   # 22 = optional D20 deep encoder (not implemented)
prespecified_outputs_implemented: [external_strata_with_NE_below_20, fixed_sequence_with_220_sensitivity, bh_within_F1_F5, decoder_domain_guard_and_per_object_stability_in_routing, qc_panels_and_manual_review, wedd_candidates, wedd_settings_grid, global_lambda_and_rbf_comparators, linearity_criterion, shift_audit]
legacy_evidence: {path: legacy/v5, status: PRE_FIX_NEVER_A_V6_INPUT}
access_dependent_bindings:
  acdc_info_cfg_frame_base: verify_on_mount_in_container
  mms_csv_and_mask_encoding: verify_on_mount_in_container
publication_evidence_limit: {target_journal: NOT_SELECTED, journal_metrics: NOT_INDEPENDENTLY_VERIFIED}
licence_risk: {tracked_mms_information_csv: FLAGGED_NO_PUBLIC_MIRROR}
# version 6.0.1 additions (the 6.0.0 declaration above is kept as written on September 28-29, 2026)
amendment_6_0_1:
  specification: 6.0.1_additions_only_pending_owner_signature
  freeze_record: git_tag_protocol-v6.0.1_pending_owner_signature   # created by the owner only
  mms2_source: {archive: MnM2.rar, bytes: 9824397247, sha256: 934076fe388f0747c80b7f0fdf5ebc57cd480ab737be8228639480e61515362b, status: UNOPENED_UNTIL_TAG_EXCEPT_OWNER_APPROVED_HEADER_AND_MEMBER_NAMES_2026-10-02, drive_copy: CSE-1_storage_only}
  mms2_split: {salt: "mms2-split-v6.0.1|", subjects: 360, mms2_external: 180_minus_duplicate_exclusions, mms2_sealed: 180_never_extracted_under_v6, manifest_sha256: "f427d9723b4055024ba92e47bc652388d615c54bbf4968de9088255dfa1a072d"}
  confirmatory: [H-primary-1_recorded, H-primary-2_recorded, H-primary-3, H4a, H4b]   # H4: Decision D36, Bonferroni within the family
  registered_estimation_only: [F8, D20, F9, F9b, S1, S2, S3, S4, S5, S6]
  exploratory_post_hoc: [B1, B2, B3, B4, B5, B6, B7, B8, B9]   # B9 added 2026-10-03 (owner issue O1)
  image_branch_d4: {D20: {tta: d4_mean_8}, F9: {augmentation: d4_uniform_per_patient, tta: d4_mean_8}, deep_encoder_v6_0_0_digest: unchanged_amendment_scoped_key}
  real_dataset_artifacts: {mms2_external: NOT_GENERATED, D20: NOT_GENERATED, F8: NOT_GENERATED, F9: NOT_GENERATED, F9b: NOT_GENERATED, S1_S6: NOT_GENERATED, B1_B9: NOT_GENERATED}
  cloud: {exceptions: [CPE-1, CSE-1], modal_scope: acdc_dev_tensor_cache_and_development_fits_only, modal_client: uvx_pinned_no_uv_group, reported_numbers: LOCAL_INFERENCE_ONLY}
  acceptance_tests_15_2_added: {implemented: [22, 25, 26, 27, 28, 29, 30, 31, 32, 33], pending: []}   # integration of October 2, 2026; synthetic fixtures only
  mms2_release_facts_read_before_tag: {date: "2026-10-02", approved_by: owner, scope: csv_header_line_and_member_names_only, header: "SUBJECT_CODE,DISEASE,VENDOR,SCANNER,FIELD", listing_entries: 3964}
```

### 19.3 Primary methodological source register (SHA-256 of the reviewed files, recomputed September 28, 2026)

The six PDFs are not copied into this repository (P5 and P6 are unpublished manuscripts); they are held at `<local path>`. The digests below were recomputed read-only on September 28, 2026 and equal those of the CXR-TRANS-AUDIT register §19.3.

| ID | Bibliographic identity and reviewed artifact | Pages | SHA-256 |
|---|---|---:|---|
| **P1** | Radiuk P, Barmak O, Manziuk E, Krak I. Explainable Deep Learning: A Visual Analytics Approach with Transition Matrices. *Mathematics* 2024;12(7):1024. DOI 10.3390/math12071024. `sources/01_Explainable-deep-learning.pdf` | 32 | `6fe241d84e8773c98eff3638d013fcbd5cf74f1fc99a02c5619c1219c1910881` |
| **P2** | Barmak O, Krak I, Yakovlev S, Manziuk E, Radiuk P, Kuznetsov V. Toward explainable deep learning in healthcare through transition matrix and user-friendly features. *Front Artif Intell* 2024;7:1482141. DOI 10.3389/frai.2024.1482141. `sources/02_Toward-explainable-deep-learning.pdf` | 13 | `4f85cb7b197a6a0526600854d58fec39383594a75a3dec088c5b8a999329fb4d` |
| **P3** | Radiuk P, Barmak O, Bedratyuk L, Krak I. Equivariant Transition Matrices for Explainable Deep Learning: A Lie Group Linearization Approach. Reviewed: author manuscript (main text February 16, 2026; supplement April 4, 2026). Published: *Mach Learn Knowl Extr* 2026;8(4):92, DOI 10.3390/make8040092. `sources/03_Equivariant-transition-matrices.pdf` | 37 | `dfa48d7625616f3ce58c74afadbbfa73080f2c2aa7e59cd5286eadc4d6727a33` |
| **P4** | Radiuk P, Barmak O, Krak I. SEMTRA: Global Semantic Transition and Rough-Set Rules for Auditable Post-Hoc Explainability. *Mach Learn Knowl Extr* 2026;8(7):181. DOI 10.3390/make8070181. `sources/04_SEMTRA.pdf` | 43 | `9995c16297235af37526da29699afc6addb2ded824657035e40ce2aee0a140e0` |
| **P5** | Radiuk P, Barmak O, Klymenko L, Krak I. CARDIA-X: Global Semantic Transition and Rough-Set Rules for Auditable Post-hoc Electrocardiographic Explainability. Author manuscript submitted to *AI* (MDPI), version September 6, 2026. `sources/05_CARDIA-X.pdf` | 60 | `5f4d7a9b0d8f11c667d8d9ffe54514557689bd7236519c7b280820991ce0798e` |
| **P6** | Radiuk P, Barmak O, Krak I. Explainable Deep Learning for Cardiac MRI: Transition-Matrix Semantics and Auditable Production Rules. Author manuscript, version September 2, 2026; LaTeX source in this repository at `manuscript/article/main.tex`. `sources/06_XAI-CRI.pdf` | 28 | `ca42613d853f8b22e8fe1f25f6d2fba9762f2d9c97d94bbf778f8ef9b3474210` |

### 19.4 Dataset primary sources

- **D1** ACDC database and evaluation pages (label values 0 background, 1 RV, 2 myocardium, 3 LV): https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html ; https://www.creatis.insa-lyon.fr/Challenge/acdc/evaluation.html
- **D2** Bernard O, Lalande A, Zotti C, Cervenansky F, et al. Deep Learning Techniques for Automatic MRI Cardiac Multi-Structures Segmentation and Diagnosis: Is the Problem Solved? *IEEE Trans Med Imaging* 2018;37(11):2514–2525, DOI 10.1109/TMI.2018.2837502 [**VERIFIED**]
- **D3** ACDC licence: CC BY-NC-SA 4.0 with specific terms (non-commercial scientific research; mandatory citation), tracked at `data/raw/acdc/database/LICENSE_TERMS.md` and `MANDATORY_CITATION.md`; https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
- **D4** M&Ms challenge official site: https://www.ub.edu/mnms/ (returned the "temporarily unavailable" page on September 26 and September 28, 2026)
- **D5** Campello VM, Gkontra P, Izquierdo C, Martín-Isla C, et al. Multi-Centre, Multi-Vendor and Multi-Disease Cardiac Segmentation: The M&Ms Challenge. *IEEE Trans Med Imaging* 2021;40(12):3543–3554, DOI 10.1109/TMI.2021.3090082 [**VERIFIED**; 375 subjects]
- **D6** M&Ms Dataset Research Use Agreement (clause 4 verbatim personal research copy; clause 5 no distribution of any portion), as recorded in `docs/history/experiment_machine_v5/datasets.md` and the acquisition receipt under `registry/sources/`; the agreement text is archived at M1
- **D7** Martín-Isla C et al. Deep Learning Segmentation of the Right Ventricle in Cardiac MRI: The M&Ms Challenge. *IEEE J Biomed Health Inform* 2023, DOI 10.1109/JBHI.2023.3267857 [**VERIFIED** identity]; https://www.ub.edu/mnms-2/
- **D8** Sunnybrook Cardiac Data, Cardiac Atlas Project: https://www.cardiacatlas.org/sunnybrook-cardiac-data/ ; Radau P, Lu Y, Connelly K, Paul G, Dick AJ, Wright GA. Evaluation Framework for Algorithms Segmenting Short Axis Cardiac MRI. *MIDAS Journal* 2009
- **D9** Second Annual Data Science Bowl: https://www.kaggle.com/c/second-annual-data-science-bowl
- **D10** UK Biobank CMR: Petersen SE et al. UK Biobank's cardiovascular magnetic resonance protocol. *J Cardiovasc Magn Reson* 2016, DOI 10.1186/s12968-016-0227-4
- **D11** Wang C et al. CMRxRecon: A publicly available k-space dataset and benchmark to advance deep learning for cardiac MRI. *Sci Data* 2024;11:687, DOI 10.1038/s41597-024-03525-4
- **D12** CMRxMotion: Wang S et al., arXiv 2210.06385; challenge results arXiv 2507.19165; Zenodo record 6362258
- **D13** Local release manifests: the tracked ACDC release manifest and `registry/sources/mms_full_release_manifest.csv`; `<local path>` (not tracked)

### 19.5 Standards and reporting sources

- **S1** Schulz-Menger J, Bluemke DA, Bremerich J, et al. Standardized image interpretation and post-processing in cardiovascular magnetic resonance – 2020 update. *J Cardiovasc Magn Reson* 2020;22:19, DOI 10.1186/s12968-020-00610-6 [**VERIFIED**]
- **S2** Cerqueira MD, Weissman NJ, Dilsizian V, et al. Standardized myocardial segmentation and nomenclature for tomographic imaging of the heart. *Circulation* 2002;105(4):539–542, DOI 10.1161/hc0402.102975 [**VERIFIED**]
- **S3** Mosteller RD. Simplified calculation of body-surface area. *N Engl J Med* 1987;317(17):1098, DOI 10.1056/NEJM198710223171717 (identity as registered in `registry/semantic_registry.yaml`; not re-verified on September 28, 2026)
- **S4** Kramer CM, Barkhausen J, Bucciarelli-Ducci C, Flamm SD, Kim RJ, Nagel E. Standardized cardiovascular magnetic resonance imaging (CMR) protocols: 2020 update. *J Cardiovasc Magn Reson* 2020;22:17, DOI 10.1186/s12968-020-00607-1 [**VERIFIED**]
- **S5** Hundley WG, Bluemke DA, Bogaert J, et al. Society for Cardiovascular Magnetic Resonance (SCMR) guidelines for reporting cardiovascular magnetic resonance examinations. *J Cardiovasc Magn Reson* 2022;24:29, DOI 10.1186/s12968-021-00827-z [**VERIFIED**]
- **S6** Tejani AS et al. Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 Update. *Radiol Artif Intell* 2024;6(4):e240300, DOI 10.1148/ryai.240300 [**VERIFIED 2026-09-24**, CXR-TRANS-AUDIT]
- **S7** Sounderajah V et al. The STARD-AI reporting guideline for diagnostic accuracy studies using artificial intelligence. *Nat Med* 2025;31(10):3283–3289, DOI 10.1038/s41591-025-03953-8; Author Correction *Nat Med* 2026;32(9):3493, DOI 10.1038/s41591-026-04570-9 [**VERIFIED 2026-09-24**, CXR-TRANS-AUDIT]
- **S8** Collins GS et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378, DOI 10.1136/bmj-2023-078378 [**VERIFIED 2026-09-24**, CXR-TRANS-AUDIT]
- **S9** Vasey B et al. DECIDE-AI. *Nat Med* 2022;28:924–933, DOI 10.1038/s41591-022-01772-9 [**VERIFIED 2026-09-24**, CXR-TRANS-AUDIT]

### 19.6 Journal evidence sources

- **J1** `manuscript/article/main.tex` line 4 (MDPI class option `bdcc` since September 29, 2026; the placeholder `journal` before) and `manuscript/article/Definitions/journalnames.tex` (option `bdcc`: *Big Data and Cognitive Computing*, ISSN 2504-2289), inspected September 29, 2026
- **J2** CXR-TRANS-AUDIT `SPECIFICATION.md` 3.0.0 §18.1 and §19.6 (Journal of Imaging pages https://www.mdpi.com/journal/jimaging/about and /instructions, checked there on September 24, 2026)
- **J3** *Big Data and Cognitive Computing* home page (journal description, indexing, rank, impact factors, first-decision time), https://www.mdpi.com/journal/BDCC, checked September 29, 2026
- **J4** Special Issues listing (open issues ordered by submission deadline, pages 1–2), https://www.mdpi.com/journal/BDCC/special_issues, checked September 29, 2026
- **J5** *Data-Centric and Trustworthy AI for Biomedical Signal and Image Analysis*, https://www.mdpi.com/journal/BDCC/special_issues/83V9MMIOJC, checked September 29, 2026
- **J6** *Next-Generation Medical Image Analysis: Multimodal, Decentralized, Fair and Reasoning-Centric Approaches*, https://www.mdpi.com/journal/BDCC/special_issues/56N7IB859Z, checked September 29, 2026
- **J7** *Beyond Diagnosis: Machine Learning in Prognosis, Prevention, Healthcare, Neurosciences, and Precision Medicine*, https://www.mdpi.com/journal/BDCC/special_issues/E57BWR9J52, checked September 29, 2026
- **J8** *Artificial Intelligence Models and Cognitive Computing: Innovations from Algorithms to Intelligent Systems*, https://www.mdpi.com/journal/BDCC/special_issues/P2957DV101, checked September 29, 2026

### 19.7 Implementation documentation

Documentation locations; not re-fetched on September 28, 2026.

- **M1** NiBabel NIfTI images, header scaling (`scl_slope`, `scl_inter`) and affines: https://nipy.org/nibabel/nifti_images.html
- **M2** scikit-learn `LogisticRegression`: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- **M3** joblib `Parallel` with the `loky` back end: https://joblib.readthedocs.io/
- **M4** threadpoolctl (BLAS thread control and reporting): https://github.com/joblib/threadpoolctl
- **M5** torchvision ResNet-50 and `ResNet50_Weights.IMAGENET1K_V2`: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html
- **M6** uv project and lockfile documentation: https://docs.astral.sh/uv/
- **M7** Commit `8ed5a40` of this repository (geometry reader: header `pixdim` when the affine is not voxel-scaled)

---

## Appendix A. Decision record (September 28, 2026)

No design interview took place. Each row records the question, the options the coordinating session considered, the value bound in Section 0.2 and its source. Rows sourced "coordinator proposal" become binding only when the PI confirms or amends them at M0; amendments are logged in `protocol/amendments.md` before the tag. *Amended September 29, 2026:* the owner confirmed every remaining row (`protocol/amendments.md`); no row is pending.

| Question | Options considered | Decision | Ledger | Source |
|---|---|---|---|---|
| Q1 External cohort | MMS_100 as in v5; complete labelled M&Ms-1 (320); 320 + unlabelled 25 | Complete labelled release, 320; unlabelled 25 excluded | D1 | owner instruction (2026-09-28) for the complete dataset; exclusion of the 25: owner confirmation (2026-09-29) |
| Q2 Data access | re-download; local backup with manifest; transfer archive | Local release, hash-verified, mounted read-only | D2 | owner confirmation (2026-09-29) |
| Q3 Compute | CPU-only; CPU-first with optional GPU; GPU-first | CPU-first; GPU only for D20 | D3 | owner confirmation (2026-09-29) |
| Q4 Venue | name a journal now; keep MDPI class and decide later | *Big Data and Cognitive Computing* (MDPI, ISSN 2504-2289); Special Issue *Data-Centric and Trustworthy AI for Biomedical Signal and Image Analysis* (deadline 31 January 2027; §18.1) | D4 | owner decision (2026-09-29) |
| Q5 Audited predictor | measured-B classifier (v5); classifier on A; deep CNN | Multinomial logistic regression on standardized A (L2, C = 1.0 fixed); measured-B classifier as comparator | D5 | owner confirmation (2026-09-29) |
| Q6 Primary family | all 28 v5 targets; ventricular function and size; wall only | Ventricular function and size (7); wall thickness and thickening key secondary | D6 | owner confirmation (2026-09-29) |
| Q7 Masks | reference masks (P6 continuity); automatic segmentation; image-only primary | Reference masks condition A and B | D7 | owner confirmation (2026-09-29) |
| Q8 Crosswalk | ACDC vocabulary with off-ontology route; M&Ms vocabulary; no class endpoint externally | ACDC vocabulary; IHD not exact; four classes off-ontology | D8 | owner confirmation (2026-09-29) |
| Q9 Formal A | 128-d projection (P6); 57 unprojected; semantic proxy | 57 unprojected; 128-d as continuity comparator; proxy forbidden | D9 | owner confirmation (2026-09-29) |
| Q10 Transition | one global λ (v5); per-target λ with typed transforms; nonlinear primary | Per-target λ ridge with typed transforms; comparators | D10 | owner confirmation (2026-09-29) |
| Q11 Discretization | tertiles (v5); WEDD Gaussian with stability gate; entropy-only | `wedd_gaussian_mm_v1` + 500-replicate gate; tertiles and uniform as sensitivities | D11 | owner confirmation (2026-09-29) |
| Q12 Rule tier and gate | frequency rules with cap (v5); VPRS with a held-out gate; VPRS with a cross-fitted gate | VPRS β = 0.10, ≥ 5 anchors, cross-fitted Wilson gate ≥ 0.60 | D12 | owner confirmation (2026-09-29) |
| Q13 Rule consequent | dataset label; `d_model`; both | `d_model` primary; label companion; measured-B oracle | D13 | owner confirmation (2026-09-29) |
| Q14 Routes | margin-based conflict resolution (v5); partitioned routes without resolution | Seven partitioned routes; no resolution; soft never labels | D14 | owner confirmation (2026-09-29) |
| Q15 Splits | v5 85/15/50 seeded split; 5 hash folds on the 100 + official test 50 | Hash folds (salt `acdc-fold-v6|`); official test | D15 | owner confirmation (2026-09-29) |
| Q16 External protocol | evaluate 320 once; evaluate only the 220; pool | 320 once; 220 confirmatory sensitivity; 100 separate | D16 | owner confirmation (2026-09-29) |
| Q17 Refit arm | none; M&Ms refit ceiling as required secondary | Required secondary, never pooled | D17 | owner confirmation (2026-09-29) |
| Q18 Primary estimand | per-target superiority; family ρ with SESOI 0.10 and fixed sequence | Family ρ, SESOI 0.10, ACDC then M&Ms | D18 | owner confirmation (2026-09-29) |
| Q19 Equivariance | drop; optional with seeded selection; required | Optional, seeded 12 cases | D19 | owner confirmation (2026-09-29) |
| Q20 Deep encoder | none; optional ResNet-50 branch; primary | Optional branch, GPU target | D20 | owner confirmation (2026-09-29) |
| Q21 QC severity | v5 defaults; missing structure critical | Four severities; missing structure critical | D21 | SV-discrepancy severity (moderate, §12.4): owner confirmation (2026-09-29); other values: owner confirmation (2026-09-29) |
| Q22 Runtime | host venv; Linux container on Docker Desktop | Digest-pinned container; CPython 3.12.14 via uv | D22 | owner instruction (2026-09-28) for Docker Desktop; pins: owner confirmation (2026-09-29) |
| Q23 Efficiency | re-read raw per run (v5); verify-and-stage with stage DAG | One-pass verify-and-stage; content-addressed stage records | D23 | owner instruction (2026-09-28) to make experiments efficient; mechanism: owner confirmation (2026-09-29) |
| Q24 Reader-study firewall | merge with v6; keep governed and separate | Separate and frozen under protocol v2 | D24 | owner confirmation (2026-09-29) |
| Q25 Freeze and file plan | edit old specs; new root specification with history and legacy moves | New `SPECIFICATION.md` 6.0.0; `docs/history/`; `legacy/v5/`; signed tag at M0 | D25 | owner instruction (2026-09-28) to follow the transition-matrix-x-ray structure; tag: owner action at M0 |
| Q26 Engineering details beyond the ledger | — | Every item marked [PROPOSED] in Sections 3–15 | — | owner confirmation (2026-09-29) |
| Q27 Duplicate images of labelled cases (four pairs with identical decoded images and different masks; Section 4.5) | keep all as independent; exclude both members; exclude one member per pair; keep both as one bootstrap cluster | `patient055` out of `acdc_dev` (`DUPLICATE_IMAGE_OF_TEST_CASE`), `patient118` kept in `acdc_test`; `patient076` out of `acdc_dev`, `patient074` kept (lower identifier; `DUPLICATE_IMAGE_WITHIN_PARTITION`); `acdc_dev` 98; K3R0Y7 out of `mms_refit_eval` (`DUPLICATE_IMAGE_OF_REFIT_TRAIN_CASE`), 169; both M&Ms pairs stay in `mms_external` as one cluster each (320 cases, 318 clusters); every bootstrap, interval and cluster-level test resamples clusters | D15, D16, D17, D18 | owner instruction (2026-09-28) |
| Q28 Demographic strata (Special Issue fit, bias and fairness) | none, as now (age and sex are never used, §7.2); descriptive external strata by sex and age band; prespecified subgroup tests | Proposed: descriptive `mms_external` strata by sex (F, M) and age band (< 40, 40–64, ≥ 65 years) from the mounted release CSV, with the subgroup reporting threshold and no tests, fitting or selection; ACDC `Info.cfg` has no age or sex | — | owner confirmation (2026-09-29) |

---

**End of specification.** All empirical claims remain conditional on executing the frozen pipeline with the locally held, hash-verified data on the declared hardware after the M0 freeze. The required outcome is a reproducible and falsifiable audit, including explicit ineligibility when the evidence does not support a feature, class, rule or route.
