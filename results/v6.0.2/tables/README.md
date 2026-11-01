> Public edition of `manuscript/article/tables/v602/README.md` of the authors' development repository (commit `e9d6ccdf8bef`); operational details are replaced as listed in [EDITIONS.md](../../../protocol/registration/EDITIONS.md).

# Evidence pack: post-freeze exploratory block B10–B22 (v6.0.2)

This folder holds the aggregate outputs of run `v6-postfreeze-20261028-0201`, the post-freeze block B10–B22. The block is exploratory and post hoc (label `EXPLORATORY_POST_HOC`): it ran under the signed amendment tag `protocol-v6.0.2` (code commit `72c5395`, clean tree) and read the frozen run `v6-replication-20261003-1241` read-only. It answers the editor's pre-check comments (EPC). It does not replace or change the recorded results of the v6.0.1 freeze in `../v601/`; nothing here is confirmatory.
- Written on 2026-10-28 by `manuscript/scripts/export_v602_evidence_pack.py` from the run root `artifacts/v6-postfreeze-20261028-0201/postfreeze/` (Docker volume `artifacts`).
- `provenance.json` gives, for every file, its run-relative source path(s) and their SHA-256, the code commit and `code_dirty` of each analysis, and the identifier-scan result.
- `SHA256SUMS` lists every file; `sha256sum -c SHA256SUMS` run here must report every file OK.

**Contents** (rows in brackets). Tables of B10–B20 are verbatim CSV copies of the run's parquet tables (same columns and rows, full float precision, empty cell = not recorded). B21 and B22 are flattened from `results.json` into one row per representation or seed, partition, quantity or metric, and target or class. `summaries.json` holds the scalars of each `summary.json` that the tables do not hold (for example the B13 development mean and median, the B14 thresholds, the B17 large-error thresholds) and the configuration of B21 and B22.
- **B10** (EPC-1, coupling): analytical ejection fraction from the voxel fractions (no learning).
  - `b10_exactness_counts.csv` (8): per partition and EF target, the cases where the analytical EF from the voxel fractions reproduces the measured EF within the tolerance, and the counts of rows that are not applicable.
  - `b10_rho_ef.csv` (12): rho_EF of the analytical arm and of the primary ridge per partition, and their difference.
  - `b10_paired_difference_vs_primary_ridge.csv` (20): paired differences, analytical arm minus primary ridge, in nMAE, rho_EF and rho_j.
  - `b10_per_target_mae_bias_loa.csv` (16): per EF target and arm: MAE, bias, limits of agreement, nMAE and rho_j with intervals.
- **B11** (EPC-1, coupling): descriptor-family ablations of the typed transition.
  - `b11_rho.csv` (60): rho, rho_volumes and rho_EF of the primary ridge and of the ridge refitted on the 17 voxel-fraction or the 40 intensity descriptors, with differences vs the primary ridge.
  - `b11_paired_difference_vs_primary_ridge.csv` (136): paired differences of each descriptor-family arm vs the primary ridge (rho, rho_volumes, rho_EF, rho_j, nMAE).
  - `b11_per_target_mae_bias_loa.csv` (84): per target, partition and arm: MAE, bias, limits of agreement, nMAE and rho_j.
- **B12** (EPC-1, coupling): geometry-aware reconstruction and error decomposition.
  - `b12_rho.csv` (60): rho, rho_volumes and rho_EF of the analytical arm, the primary ridge and the geometry-aware ridge (formal 57 descriptors plus 3 geometry covariates), with differences vs the geometry-aware ridge.
  - `b12_decomposition_table.csv` (70): per target and partition, the normalized error split into analytical reference, approximation, missing information and shift.
  - `b12_analytical_exactness.csv` (20): how closely the analytical volumes (voxel fraction x voxel count x voxel size) reproduce the measured ones.
  - `b12_paired_differences.csv` (136): paired differences of the analytical arm and the primary ridge vs the geometry-aware ridge.
  - `b12_per_target_mae_bias_loa.csv` (84): per target, partition and arm: MAE, bias, limits of agreement, nMAE and rho_j.
- **B13** (EPC-6, rho baseline): development-median baseline.
  - `b13_side_by_side_with_mean_null.csv` (40): the recorded mean-null rho beside the recomputed mean null and the development-median null.
  - `b13_rho_median.csv` (42): rho, rho_volumes and rho_EF under the mean and median nulls (and their outer-training variants), with differences vs the mean null.
  - `b13_rho_j_median.csv` (63): per target, partition and null: MAE, bias, limits of agreement, nMAE and rho_j.
  - `b13_paired_difference_median_minus_mean.csv` (85): paired differences, median null minus mean null, in rho, rho_volumes, rho_EF, rho_j and nMAE.
- **B14** (EPC-6, rho baseline): smallest effect of interest in physical units and a decision sweep.
  - `b14_translation_table.csv` (28): the smallest effect of interest (0.10) in physical units per target and partition (0.10 x mean-null MAE), beside the null and ridge MAE and rho_j.
  - `b14_sweep_table.csv` (49): for each recorded hypothesis estimate, whether its 95 % lower bound exceeds each threshold of the sweep, and the largest threshold exceeded.
- **B15** (EPC-4, rulebook): candidate-level rule ledger.
  - `b15_candidate_ledger.csv` (103): every candidate rule of the three rulebooks with the result of each gate condition and the first condition it fails.
  - `b15_rejection_counts_by_reason.csv` (12): candidates per rulebook by first failed condition (PASSED_ALL for those that pass).
  - `b15_concept_table.csv` (27): concept eligibility per rulebook: availability, variability, reconstruction, cut stability and tautology checks.
- **B16** (EPC-4, rulebook): coverage-fidelity comparison of rule systems.
  - `b16_coverage_fidelity_table.csv` (96): per rule system, variant and partition: coverage and fidelity to d_model with Wilson intervals and k/n, accuracy, and d_model accuracy on the same rows.
  - `b16_figure_data.csv` (96): the coverage-fidelity subset plotted in the coverage-fidelity figure.
- **B17** (EPC-7, reliability): reliability of reconstructed concepts.
  - `b17_signal_auroc_table.csv` (72): AUROC of the continuous signals (OOD score, uncertainty, unstable bins) for a large reconstruction error, per target and for any-large.
  - `b17_binary_signal_table.csv` (72): sensitivity, specificity and PPV of the binary signals (not accepted at tau50 or tau80, OOD flag) for a large reconstruction error.
  - `b17_symbol_error_table.csv` (21): symbol errors (bin of B-hat differs from bin of B) by stability flag per concept, with the risk ratio.
- **B18** (EPC-2, decision link): decision-link test (how much of the audited decisions the concepts carry).
  - `b18_decision_link_table.csv` (12): how much of the audited decisions d_model the concepts carry: fidelity, accuracy, majority rate and macro-F1 of a logistic model on bhat7, bhat9 or measured7, beside the recorded F8 fidelity.
- **B19** (EPC-A2, majority class): majority-class references.
  - `b19_majority_class_table.csv` (3): majority-class rate with Wilson interval, class counts and uniform chance per evaluation partition.
- **B20** (EPC-A4, worked example): real processed cases for the worked example.
  - `b20_case_slots.csv` (5): five ACDC test cases keyed only by slot (a, b1-b4): measured B, raw and decoded B-hat, bins and stability, class probabilities, acceptance, OOD score and flag, and the routes of each system.
- **B21** (EPC-3, matched comparators): phase-preserving and mask-pooled frozen image representations (D20-P, D20-M).
  - `b21_representations.csv` (2): the two frozen ResNet-50 representations: D20-M (mask-pooled, 12288 columns) and D20-P (phase-preserving, 4096 columns), with fit rows and bundle hash.
  - `b21_rho.csv` (136): rho, rho_volumes and rho_EF (family rows) and rho_j (target rows) of the transition fitted on each representation, per partition, with intervals.
  - `b21_rho_paired_vs_primary_ridge.csv` (64): paired difference in rho (family) and rho_j (targets), representation minus the primary ridge of the source run, on the same rows and cluster draws.
  - `b21_predictor_metrics.csv` (80): classifier on each representation: accuracy, balanced accuracy, macro-F1, macro AUROC, Brier, ECE (equal mass; equal width as companion), log loss, plus majority rate and chance, per partition.
  - `b21_predictor_per_class_recall.csv` (40): recall per class of each representation's classifier.
  - `b21_predictor_confusion_matrix.csv` (200): confusion counts (reference x predicted class) of each representation's classifier.
  - `b21_counts.csv` (8): rows per representation and partition: partition rows, evaluated rows, critical-QC exclusions, predictor counts and status.
  - `b21_d20_replication_diagnostic.csv` (4): recomputed D20 features vs the recorded D20 of the source run (maximum absolute difference, cases compared); diagnostic, reported and never used.
- **B22** (EPC-3, matched comparators): mask-aware fine-tuned CNN (F9-M) and its concept probe (F9b-M).
  - `b22_inputs.csv` (4): per partition: cases, cases with an absent structure, and the image alignment check of the neural cache.
  - `b22_fits.csv` (18): per seed and fit (five outer folds and final): selected candidate, refit epochs, maximum mask-stem weight, seconds and checkpoint hash.
  - `b22_validity_gate.csv` (3): per seed validity gate of F9-M (development out-of-fold macro-F1 >= 0.3, mask stem trained, training checks); reported, never used.
  - `b22_f9m_predictor_metrics_per_seed.csv` (120): F9-M classifier per seed and partition: accuracy, balanced accuracy, macro-F1, macro AUROC, Brier, ECE, log loss, plus majority rate and chance.
  - `b22_f9m_per_class_recall_per_seed.csv` (60): F9-M recall per class, seed and partition.
  - `b22_f9m_confusion_matrix_per_seed.csv` (300): F9-M confusion counts (reference x predicted class) per seed and partition.
  - `b22_f9bm_probe_rho_per_seed.csv` (204): F9b-M concept probe (transition fitted on the 64 F9-M activations) per seed: rho, rho_volumes and rho_EF (family rows) and rho_j (target rows), with intervals.
  - `b22_f9bm_probe_paired_vs_primary_ridge_per_seed.csv` (96): paired difference in rho and rho_j, F9b-M probe minus the primary ridge of the source run, per seed.
  - `b22_counts_per_seed.csv` (12): rows per seed and partition: partition rows, evaluated rows, critical-QC exclusions, predictor counts and status.
  - `b22_seed_mean.csv` (24): mean, minimum and maximum over the three seeds (with each seed's value) of rho, rho_volumes, rho_EF, accuracy, macro-F1 and macro AUROC.

**Recorded rho check.** The primary ridge recomputed inside B11, B12 and B13 equals the recorded values on `acdc_dev_oof`, `acdc_test` and `mms2_external`. On `mms_external` the block scores 319 cases (it excludes the one critical-QC case), which gives rho = 0.183 against the recorded 0.192 on 320 cases; the recorded value stays the reference.

**Identifier scan.** Before writing, all 51 files of this pack were checked against all 1358 case, patient and cluster identifiers of the source run's contracts (ACDC, M&Ms-1 and M&Ms-2: cohort tables, split manifests, inner folds, cohort-validation and duplicate records) and the `sample_id` values of the B11 and B12 development out-of-fold files, as whole tokens, ignoring case: no identifier occurs, and no file has a `sample_id` (or other identifier) column.

**Rules:**
- **Never edit these files.** They are evidence, not manuscript text. They change only by rerunning the script on the recorded run; a new run needs an owner decision.
- **Exploratory, post hoc.** Every number from this pack is reported as exploratory post hoc evidence (`EXPLORATORY_POST_HOC`) beside, never instead of, the v6.0.1 results.
- **Aggregates only.** No per-case file is exported (`oof_*`, `features/`, `mask_cache/`, `inference/`, `evaluated/`, `fits/`, `probes/*.parquet`, network weights), nor `B20/restricted/`, the B20 summary or any per-analysis `provenance.json`. The one exception is `b20_case_slots.csv`: five ACDC test cases keyed only by slot (a, b1–b4), whose values the owner approved for publication without identifiers.
- Copy each value exactly as recorded and record its source file in `source_map.md`.

**Regenerate** (from the repository root; in Git Bash prefix `MSYS_NO_PATHCONV=1`):

```
docker compose run --rm -T tmmri python manuscript/scripts/export_v602_evidence_pack.py
```
