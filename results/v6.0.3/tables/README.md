> Public edition of `manuscript/article/tables/v603/README.md` of the authors' development repository (commit `e9d6ccdf8bef`); operational details are replaced as listed in [EDITIONS.md](../../../protocol/registration/EDITIONS.md).

# Evidence pack: post-freeze exploratory block B23–B27 (v6.0.3)

This folder holds the aggregate outputs of run `v6-postfreeze-20261028-2135`, the post-freeze block B23–B27. The block is exploratory and post hoc (label `EXPLORATORY_POST_HOC`): it ran under the signed amendment tag `protocol-v6.0.3` (code commit `152e315`, clean tree) and read the frozen run `v6-replication-20261003-1241` and the B10–B22 run `v6-postfreeze-20261028-0201` read-only. It answers the readiness review of the revised manuscript. It does not replace or change the recorded results of the v6.0.1 freeze in `../v601/` or the B10–B22 pack in [`../v602/`](../../v6.0.2/tables/README.md); nothing here is confirmatory. Only B23 has a reading fixed before computation (D43).
- Written on 2026-10-28 by `manuscript/scripts/export_v603_evidence_pack.py` from the run root `artifacts/v6-postfreeze-20261028-2135/postfreeze/` (Docker volume `artifacts`).
- `provenance.json` gives, for every file, its run-relative source path(s) and their SHA-256, the code commit and `code_dirty` of each analysis, and the identifier-scan result.
- `SHA256SUMS` lists every file; `sha256sum -c SHA256SUMS` run here must report every file OK.

**Contents** (rows in brackets). Tables are verbatim CSV copies of the run's parquet tables (same columns and rows, full float precision, empty cell = not recorded). `b23_reading.json` is the D43 reading as recorded. `b25_rule.csv` is the admitted rule with its anchor patients replaced by their count per source dataset. `summaries.json` holds each `summary.json` without its lists of rows, which repeat the tables.
- **B23** (EPC-2; readiness review R2): null references for the decision-link test.
  - `b23_b18_reproduction.csv` (4): the refitted B-hat7 fidelity beside the recorded B18 fidelity per partition (k, n, estimate) and whether it was reproduced (tolerance 1e-12).
  - `b23_null_fidelity_table.csv` (18): fidelity to d_model of each decision-link input per partition (B-hat7, PCA-7, all 57 descriptors, the constant majority decision; median and 95th percentile of the random-projection and permutation nulls), with Wilson intervals, and accuracy on the same-definition subsets.
  - `b23_delta_table.csv` (3): Delta = fidelity(B-hat7) - fidelity(PCA-7) per partition with its paired cluster-bootstrap interval and counts.
  - `b23_null_distributions.csv` (6000): the fidelity of each of the 1,000 random projections and 1,000 permutations per partition.
  - `b23_reading.json`: the D43 reading per external partition and overall, as recorded.
- **B24** (EPC-2; readiness review R1): rule audit of the image networks (F9, F9-M).
  - `b24_concept_table.csv` (18): per network and concept: observed fraction, reconstruction gate, cut stability and eligibility, with the reasons for ineligibility.
  - `b24_candidate_ledger.csv` (3): every candidate rule of the network rulebooks with its counts and first failed condition.
  - `b24_rejection_counts_by_reason.csv` (1): candidate rules per network rulebook by first failed condition.
  - `b24_coverage_fidelity_table.csv` (6): coverage, fidelity to the network's decisions and accuracy of the (empty) network rulebooks per partition, with the routing note.
  - `b24_decision_link_table.csv` (24): decision link of each network: fidelity of the probe concepts, PCA-7 and the random-projection null on the network's activations, and the probe-minus-PCA difference, per partition.
  - `b24_decision_link_nulls.csv` (6000): the fidelity of each of the 1,000 random projections of the activations per network and partition.
- **B25** (EPC-4; readiness review R3): rule admission on a pooled development sample (sample-size test of the empty rulebook).
  - `b25_admission_counts.csv` (3): the pooled admission sample by source (acdc_dev, mms_external, pooled): cases, decisions per class, cases per fold and unstable cells.
  - `b25_candidate_ledger.csv` (54): every candidate rule of the pooled admission with its counts, the gate quantities and its first failed condition.
  - `b25_rejection_counts_by_reason.csv` (3): candidate rules by first failed condition.
  - `b25_coverage_fidelity_mms2.csv` (1): coverage, fidelity to d_model and accuracy of the pooled rulebook on mms2_external, with Wilson intervals.
  - `b25_gate.csv` (1): the cross-fitted gate of each induced rule: out-of-fold matches and agreements, Wilson bounds, folds reproduced and eligibility.
  - `b25_rule.csv` (1): the admitted rule: antecedent, consequent, fit and cross-fitted counts, and anchor patients counted per source dataset.
- **B26** (EPC-6; readiness review R5): robust agreement and outlier description.
  - `b26_robust_agreement_table.csv` (21): per target and partition: median bias and the 2.5th and 97.5th percentiles of B-hat - B with cluster-bootstrap intervals, beside the recorded Bland-Altman bias and limits.
  - `b26_outlier_counts.csv` (21): per target and partition: scored cases, outliers, median, MAD and the outlier threshold.
  - `b26_outlier_description.csv` (175): outliers against the other cases by vendor, center, slice count, voxel volume and imaged volume, in cells of at least five cases (smaller categories merged into 'other' or suppressed in both groups).
  - `b26_geometry_check.csv` (21): per target and partition: outliers that remain, or no longer are, outliers under the geometry-aware transition of B12.
- **B27** (EPC-7; readiness review R7): fold-spread concept-uncertainty signal.
  - `b27_spread_auroc_table.csv` (48): AUROC of the fold-spread signal for large concept errors (development and within-partition definitions; per target and any-large) with cluster-bootstrap intervals.
  - `b27_spread_spearman_table.csv` (21): Spearman correlation of the fold-spread signal with the normalized absolute error, per target and partition.
  - `b27_b17_reread_table.csv` (72): the B17 signals (uncertainty, OOD score, unstable bins) re-read under the within-partition definition.

**Identifier scan.** Before writing, all 28 files of this pack were checked against all 1358 case, patient and cluster identifiers of the frozen run's contracts (ACDC, M&Ms-1 and M&Ms-2) and the B25 anchors, as whole tokens, ignoring case: no identifier occurs, and no file has an identifier column.

**Rules:**
- **Never edit these files.** They are evidence, not manuscript text. They change only by rerunning the script on the recorded run; a new run needs an owner decision.
- **Exploratory, post hoc.** Every number from this pack is reported as exploratory post hoc evidence (`EXPLORATORY_POST_HOC`) beside, never instead of, the v6.0.1 results.
- **Aggregates only.** No per-case file, fitted model, projection or permutation array, cut file or rulebook (which lists anchor patients) is exported, nor any per-analysis `provenance.json`.
- Copy each value exactly as recorded and record its source file in `source_map.md`.

**Regenerate** (from the repository root; in Git Bash prefix `MSYS_NO_PATHCONV=1`):

```
docker compose run --rm -T tmmri python manuscript/scripts/export_v603_evidence_pack.py
```
