> Public edition of `manuscript/article/tables/v604/README.md` of the authors' development repository (commit `e9d6ccdf8bef`); operational details are replaced as listed in [EDITIONS.md](../../../protocol/registration/EDITIONS.md).

# Evidence pack: post-freeze exploratory block B28–B31 (v6.0.4)

This folder holds the aggregate outputs of run `v6-postfreeze-20261029-1008`, the post-freeze block B28–B31. The block is exploratory, post hoc and descriptive (label `EXPLORATORY_POST_HOC`): it ran under the signed amendment tag `protocol-v6.0.4` (code commit `fdad2c5`, clean tree except `code_dirty` true for B30: uncommitted edits to tracked files outside the analysis code (`.gitattributes` and `scripts/maintenance/check_repository_layout.py`) were on disk when it started; the code-tree SHA-256 equals that of the other analyses and the lock found `configs/exploratory_postfreeze_v604.yaml` and `src/` equal to the tag) and read the frozen run `v6-replication-20261003-1241` and the B23–B27 run `v6-postfreeze-20261028-2135` read-only. It does not replace or change the recorded results of the v6.0.1 freeze in `../v601/` or the packs [`../v602/`](../../v6.0.2/tables/README.md) and [`../v603/`](../../v6.0.3/tables/README.md); nothing here is confirmatory, and no analysis has a reading.
- Written on 2026-10-30 by `manuscript/scripts/export_v604_evidence_pack.py` from the run root `artifacts/v6-postfreeze-20261029-1008/postfreeze/` (Docker volume `artifacts`).
- `provenance.json` gives, for every file, its run-relative source path(s) and their SHA-256, the code commit and `code_dirty` of each analysis, and the identifier-scan result.
- `SHA256SUMS` lists every file; `sha256sum -c SHA256SUMS` run here must report every file OK.

**Contents** (rows in brackets). Tables are verbatim CSV copies of the run's parquet tables (same columns and rows, full float precision, empty cell = not recorded). `summaries.json` holds each `summary.json` without its lists of rows, which repeat the tables.
- **B28**: agreement and ceilings of the decision-link test.
  - `b28_b23_reproduction.csv` (18): the re-applied B23 fidelities beside the recorded ones per partition and input (numerator and denominator), and whether they were reproduced exactly.
  - `b28_kappa_table.csv` (12): Cohen's kappa against d_model for each B23 input (B-hat7, PCA-7, all 57 descriptors, the constant development majority decision) per partition, with cluster-bootstrap intervals and the fidelity counts.
  - `b28_kappa_null_summary.csv` (6): median and 95th percentile of kappa over the 1,000 random projections and the 1,000 permutations per partition, beside B-hat7 on the same rows.
  - `b28_majority_constant_table.csv` (6): the fidelity of a constant at each partition's most frequent d_model decision (an outcome-dependent reference, not a method), with Wilson intervals.
  - `b28_ceiling_table.csv` (3): the supervised four-dimensional ceiling (LDA-4 on the 57 descriptors, fitted on acdc_dev, then the B18 model): fidelity with Wilson interval and kappa with cluster-bootstrap interval per partition.
- **B29**: ejection-fraction categories, measured against reconstructed.
  - `b29_ef_category_crosstab.csv` (52): measured against reconstructed ejection-fraction categories (LVEF <40, 40 to <50, >=50 %; RVEF <40, >=40 %) per partition.
  - `b29_ef_category_agreement.csv` (8): proportion of cases in the same EF category (Wilson interval) and Cohen's kappa (cluster-bootstrap interval) per partition and target.
- **B30**: audited decisions by reference category.
  - `b30_confusion_table.csv` (50): the recorded confusion of the audited predictor on acdc_test and mms_external, checked against the recorded decisions and labels.
  - `b30_recall_table.csv` (12): per-class recall with Wilson intervals: acdc_test and mms_external, and the same-definition subset of mms2_external.
  - `b30_mms2_category_table.csv` (8): d_model decisions per released M&Ms-2 pathology category, with the cases whose measured LVEF or RVEF is below 40 % per category.
  - `b30_b25_covered_table.csv` (4): the mms2_external cases accepted by the re-applied B25 rulebook: released categories in cells of at least five cases, and the d_model decisions where rule and d_model disagree (counts).
- **B31**: reconstruction plausibility.
  - `b31_plausibility_counts.csv` (48): cases with an internal EF inconsistency (> 5 points), a reconstructed ESV at or above EDV, or a target outside the plausibility bounds, per partition, with Wilson intervals and the counts left out.
  - `b31_ef_difference_distribution.csv` (8): median, 2.5th and 97.5th percentiles of EF implied by the reconstructed volumes minus reconstructed EF, LV and RV, per partition.
  - `b31_clipped_skill_table.csv` (32): rho_j and rho with each reconstruction clipped to the plausibility bounds beside the unclipped values, with the paired cluster-bootstrap difference, per partition (acdc_dev_oof in-sample).

**Identifier scan.** Before writing, all 18 files of this pack were checked against all 1358 case, patient and cluster identifiers of the frozen run's contracts (ACDC, M&Ms-1 and M&Ms-2) and the anchors of the B25 rulebook, as whole tokens, ignoring case: no identifier occurs, and no file has an identifier column.

**Rules:**
- **Never edit these files.** They are evidence, not manuscript text. They change only by rerunning the script on the recorded run; a new run needs an owner decision.
- **Exploratory, post hoc.** Every number from this pack is reported as exploratory post hoc evidence (`EXPLORATORY_POST_HOC`) beside, never instead of, the v6.0.1 results.
- **Aggregates only.** No per-case file, fitted model, cut file or rulebook is exported, nor any per-analysis `provenance.json`.
- Copy each value exactly as recorded and record its source file in `source_map.md`.

**Regenerate** (from the repository root; in Git Bash prefix `MSYS_NO_PATHCONV=1`):

```
docker compose run --rm -T tmmri python manuscript/scripts/export_v604_evidence_pack.py
```
