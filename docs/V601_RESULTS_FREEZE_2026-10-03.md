> Public edition of `docs/V601_RESULTS_FREEZE_2026-10-03.md` of the authors' development repository (commit `590196aa9241`); operational details are replaced as listed in EDITIONS.md.

# protocol-v6.0.1 results freeze (3 October 2026)

This is the record of the v6.0.1 execution and of the owner's approval of the results freeze. It contains aggregate results only: no case identifiers and no per-case values (SPECIFICATION.md §14.5). It changes no protocol value.

**Approval.** The owner approved the results freeze in the working session on 3 October 2026, at about 20:50 local time. From then on nothing in run `v6-replication-20261003-1241` is recomputed (SPECIFICATION.md 6.0.1 §14.6; manuscript protocol 2.1, section M, rule 5). A bug found later is reported as a post-hoc correction and never replaces a recorded result.

## 1. Identification

| Item | Value |
|---|---|
| Run root | `v6-replication-20261003-1241` (Docker volume `tmmri_artifacts`) |
| Protocol | `protocol-v6.0.1`: an annotated tag signed by the owner (SSH) on commit `d04f0908e66766335609c938f06d51c53421a45a` and pushed to `origin` at 14:32 UTC. The M&Ms-2 extraction started at 14:34 UTC |
| Bundle under test | The v6.0.0 bundle, reproduced by this run: 242 of 252 output files are byte-identical to `v6-primary-20260929-2307`, and the other 10 differ in provenance only |
| Release manifest | `release_manifest.json` SHA-256 `9260dd64b8cf1738e8c7b6cc7304053f930c27ae874903aa25e5158ea78afb4d`. It records `protocol_frozen` true and `code.dirty` false, at commit `d04f0908`. Its `protocol_tag` field names `protocol-v6.0.0`, the tag in force when the bundle was frozen; the 6.0.1 stages carry `provenance_v6_0_1.json` |

Freeze records, all frozen before the M&Ms-2 extraction (SPECIFICATION §14.6). Each is given as the SHA-256 of its record file:

| Bundle | Record | SHA-256 |
|---|---|---|
| F8 | `comparators/cart_f8/FROZEN_F8.json` | `a210c91735665677541812383ef5b491649bd9bd96648962a9f0b53dd07e9ac1` |
| D20 | `deep_encoder/FROZEN_D20.json` | `abb4cf7062966ea6620814c29b3d2a12e595b692c2ee3ae422ffc2fcdb22ba80` |
| F9 | `neural_f9/f9_bundle.json` (internal `bundle_sha256` `d0ab5a17…`) | `004ebcca36f9a3550a29c66a35f861f6030201f2cd01d93303ac93c74b9f86d4` |
| F9b | `probes/f9b/bundle.json` | `ea6c16e9476427cea6f8c3a8e594a4588ca6acbd37b53e58980ffd902b8e6fbe` |
| S1, S2, S4 | `trust/trust_bundle.json` | `6a002bb5cb183ce398a2f102f1599622813ae441b51860d58188d2b2c113c36d` |
| F9 cross-hardware QC | `neural_f9/cross_hardware_qc.json` | `c693ee8a21d8e959bb31923db16e1487539e5a113268b78ce17c31942a245e25` |

**Backups.** Both copies were made with `rsync --checksum`. A checksum dry run after copying showed 0 differences.
- `<local path>`: 43 GB.
- `<local path>`: 43 GB.

**Cloud (CPE-1).**
- Uploaded: only `neural_cache/acdc_dev` (99 files), with every upload ledgered in `cloud/modal_ledger.csv`.
- Training: F9 ran 342 fits on Modal L40S and S5 ran 570; every fit completed and none was retried.
- Volumes: not yet deleted (owner decision under the CPE-1 retention clause).

## 2. Results

Intervals are 95 % percentile cluster-bootstrap intervals (2,000 replicates) unless stated otherwise. Labels:
- **prospective:** `mms2_external`, 180 subjects never seen before this run;
- **historical:** registered after unblinding; `acdc_test` and `mms_external`.

### 2.1 Confirmatory

| Test | Partition | Estimate | Decision |
|---|---|---|---|
| H-primary-1 (recorded, v6.0.0) | `acdc_test` (50) | ρ 0.553 [0.473, 0.619] | passed |
| H-primary-2 (recorded, v6.0.0) | `mms_external` (320 cases / 318 clusters) | ρ 0.192 [0.032, 0.298] | failed |
| **H-primary-3** | `mms2_external` (180 clusters) | **ρ 0.311 [0.233, 0.382]** | **passed** (lower bound > 0.10) |
| **H4a**, ρ_EF (LVEF, RVEF) | `mms2_external` | 97.5 % interval **[0.558, 0.652]** | **passed** (lower bound > 0.10) |
| **H4b**, Δρ from label-free harmonization H-a | `mms2_external` | 97.5 % interval [−0.019, 0.022] | failed (lower bound ≤ 0) |

H4 is Bonferroni within the family (2 tests, 97.5 % intervals) and is separate from H-primary-3 and F6.

Per-target ρ_j on `mms2_external`:

| Target | ρ_j | 95 % interval |
|---|---|---|
| LVEF | 0.67 | [0.63, 0.70] |
| RVEF | 0.55 | [0.48, 0.61] |
| LVESV | 0.35 | [0.17, 0.51] |
| LV mass | 0.28 | [0.15, 0.38] |
| RVESV | 0.19 | [0.06, 0.31] |
| LVEDV | 0.06 | [−0.10, 0.20] |
| RVEDV | 0.08 | [−0.06, 0.20] |

### 2.2 Registered secondary results (estimation only)

**Predictors.** Accuracy on the same-definition crosswalk subsets: M&Ms-1, 272 cases; M&Ms-2, NOR + HCM, 68 cases.

| Predictor | `acdc_test` | `mms_external` (historical) | `mms2_external` (prospective) |
|---|---|---|---|
| Audited descriptor predictor | 0.88 | 0.53 [0.47, 0.59]; AUROC 0.78 | **0.71** (48/68); AUROC 0.84 |
| F9 trained CNN, primary seed | 0.58 [0.44, 0.72] | 0.56 [0.50, 0.62]; AUROC 0.78 | **0.72**; AUROC 0.85 |
| D20 frozen ResNet-50 | 0.36 [0.24, 0.50] | 0.16 [0.12, 0.21] | recorded |

- F9 development OOF macro-F1 (primary seed) is 0.542.
- The F9 seeds vary widely: seed 20261012 reaches 0.375 on `mms_external` and 0.515 on `mms2_external`. All seeds are reported, with no best-seed choice.

**Other registered analyses.**
- **F9b:** ρ on `mms_external` per seed is about 0.17, −0.10 and 0.05 (descriptors: 0.19).
- **S1, selective prediction:** thresholds fixed on development OOF outputs.

  | Predictor | Partition | Coverage | Accuracy on accepted cases |
  |---|---|---|---|
  | Audited | `mms2_external` | 32 % | 0.86 (19/22) |
  | Audited | `mms2_external` | 78 % | 0.75 |
  | F9 | `mms2_external` | 43 % | 0.83 (24/29) |
  | F9 | `mms2_external` | 78 % | 0.79 |
  | F9 | `mms_external` | 45 % | 0.72 |

  Under shift, the realised coverage falls below the development target (32 % against 50 %).
- **S2, OOD flags:** on `mms2_external`, D20 flags 23 % of cases (GE 43 %, Siemens 22 %, Philips 13 %).
- **S4, calibration:** the development-fitted temperature does not transfer. For the audited predictor on `mms2_external`, equal-width ECE goes from 0.091 to 0.140. Equal-mass ECE (primary) is in `trust/mms2_external/s4_calibration.json`.
- **S5, data efficiency:** descriptor and F9 curves are recorded (7 points each; 570 F9 fits).
- **S6, simulated acquisition geometry:** for every vendor and representation, |Δρ| and |Δaccuracy| are ≤ about 0.06. Acquisition geometry alone does not reproduce the external drop.
- **F8 (CART surrogate):** evaluated on all three partitions; it is prospective on `mms2_external`.

### 2.3 Exploratory (post hoc, labelled)

All of these are recorded under `exploratory/<B#>/`.
- **B1–B4, B6, B7:** recorded as specified.
- **B5:** 48 cells of gate power. A perfect 5/5 rule cannot pass the v6 gate; it needs at least 6/6.
- **B8:**
  - On M&Ms-1, Siemens (vendor A) shows large positive volume biases (LVESV +1.0 and LVEDV +0.7 fit-SD). On M&Ms-2 these are near +0.1.
  - GE and Philips under-estimate EDV in both cohorts.
  - Vendor and centre are confounded.
  - The cross-cohort difference table was not computed (vendor labels A–D against names). A post-hoc comparison using SPECIFICATION's mapping (A Siemens, B Philips, C GE, D Canon) is derived from the recorded `vendor_bias.csv`.
- **B9:**
  - The orientation audit covers 468 cases on the historical partitions.
  - `mms2_external` was not executed before its validation; it is not re-run after the freeze.

## 3. Deviations and notes after signing

These affect no recorded value.
1. **Figure 5** (`build-manuscript-figures`) was OMITTED. The figure builder searched `probes/` and `trust/` for the comparison inputs, but `build-comparison-tables` wrote them to `manuscript_tables/comparison/figure05_representation_panel.parquet`, which is recorded and hashed. Figure 5 is rendered from that file in the writing phase and labelled "rendered after the results freeze from recorded data".
2. **The M&Ms-2 evaluate preflight** reported NO-GO before `evaluate:mms2_external` existed. This is expected ordering; every stage then ran with exit 0.
3. **Download retries.** Two S5 download attempts failed on the host (a carriage return in Windows-generated fit IDs, and a missing final newline). The retried downloads were complete, and the preflight verified every import.
4. **Pre-signing deviations** are listed in `protocol/amendments.md` (items 1–11), including the D20 GPU-isolation note.

## 4. Open items

- **Manuscript (paper 1):** first complete draft due Wednesday 7 October 2026. Classification accuracy is reported in context (chance level 0.20, label non-equivalence, AUROC, S1 abstention); the prospective confirmatory results lead.
- **Reader-study track:** pending the owner's results and the P1–P4 checks (manuscript protocol 2.1, section F).
- **Modal Volumes:** owner decision of 3 October 2026: `tmmri-neural-cache` and `tmmri-neural-runs` are **retained for now**, for reuse in paper 2. This is consistent with the CPE-1 retention clause: they are deleted after the neural release and its verified local backup, and the deletion is recorded in the ledger.
- **Paper 2 (protocol v7):** multi-centre development and physically calibrated features, prospective on the sealed 180 M&Ms-2 subjects. `mms2_sealed` remains unextracted.
