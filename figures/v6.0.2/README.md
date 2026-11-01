> Public edition of `manuscript/article/figures/v602/README.md` of the authors' development repository (commit `e9d6ccdf8bef`); operational details are replaced as listed in [EDITIONS.md](../../protocol/registration/EDITIONS.md).

# Evidence renders: post-freeze exploratory block B10–B22 (v6.0.2)

This folder holds the data figures of run `v6-postfreeze-20261028-0201`, the post-freeze block B10–B22 (signed tag `protocol-v6.0.2`, label `EXPLORATORY_POST_HOC`). They are drawn from the aggregates of the evidence pack [`../../tables/v602/`](../../results/v6.0.2/tables/README.md) and, for the recorded representations, from the v6.0.1 pack `../../tables/v601/comparison/`. Nothing is fitted, resampled, re-estimated or recomputed; every plotted value is a recorded estimate or interval bound, copied exactly. Nothing here is confirmatory, and nothing changes a value of the v6.0.1 freeze.
- Rendered by `manuscript/scripts/render_v602_figures.py` in the approved house style of `render_postfreeze_figures.py`, whose style kit it imports.
- Each figure comes as `.pdf` (vector, fonts embedded; use this in LaTeX), `.png` (400 dpi, the house setting; for previewing) and `.provenance.json` (input paths and SHA-256, script and house-style SHA-256, render date, output hashes, the plotted values and the rules applied).
- Multi-panel figures are one file per panel, without panel letters or titles, for LaTeX `\subfloat`; the captions describe the panels.
- `SHA256SUMS` lists every file except itself; `sha256sum -c SHA256SUMS` run here must report every file OK.

**These are evidence renders.** The article includes none of them directly: during the rewrite the manuscript copies the files it shows into `../paper1/`, named with their article figure number, and `\graphicspath` lists only that folder. Never edit these files by hand; they change only by rerunning the script.

## Files

- **`figure03_median_baseline_b13`** (Figure 3 with the median baseline; EPC-6). Forest plot, printed at 0.92 linewidth like the house Figure 3. Per partition (ACDC development out of fold, ACDC test H1, M&Ms-1 H2, M&Ms-2 H3): rho against the development mean (filled teal circle) above rho against the development median (open teal circle), with 95 % intervals and the values at the right; reference lines at 0 (no gain) and 0.10 (smallest effect).
  - Sources: `b13_side_by_side_with_mean_null.csv` (recorded mean null, median null), `b13_rho_median.csv` (arms; cross-checked).
  - **M&Ms-1 uses the recomputed mean null.** The block scores 319 of the 320 cases (one critical-QC case excluded; 317 clusters), so the filled marker there is the recomputed mean null of B13, 0.183 [0.021, 0.290], not the recorded 0.192 [0.032, 0.298] on 320 cases. The recorded value stays the reference of H2 and of the house Figure 3. On the other partitions the recomputed mean null equals the recorded one, which is plotted.
  - On ACDC development the median null is the outer-training median of each fold (arm `median_null_outer_training`); the mean null is the recorded full-development value of the house Figure 3 (0.532). The outer-training mean null, 0.535 [0.478, 0.582], is in `b13_rho_median.csv`.
  - Not drawn: rho_volumes and rho_EF (the house Figure 3 shows the overall rho only; both are in `b13_rho_median.csv`) and the H2 sensitivity row on never-analyzed M&Ms-1 cases (B13 has no such row).
  - Exposure wording follows D38 of `protocol-v6.0.2`: retrospective (ACDC test, M&Ms-1), preregistered held-out (M&Ms-2); "prospective" is withdrawn.
- **`figure05_matched_comparators_b21_b22_{rho,accuracy,macro_f1,macro_ovr_auroc}`** (Figure 5 extended; EPC-3, EPC-A2). Four half panels (3.4 × 2.5 in, placed at 0.49 linewidth, as a 2 × 2 grid) on ACDC test, M&Ms-1 and M&Ms-2. Each partition holds, left to right: mask descriptors (audited predictor) | D20, D20-P, D20-M | F9 seeds 20261011, 20261012, 20261013 | F9-M seeds in the same order. The rho panel shows the concept probes (F9b, F9b-M) and carries the legend; the classification panels show the classifiers (F9, F9-M), as in the house Figure 5.
  - Sources: recorded representations from `../../tables/v601/comparison/figure05_representation_panel.parquet` (estimate rows); D20-P and D20-M from `b21_rho.csv` (family rho) and `b21_predictor_metrics.csv`; F9-M from `b22_f9m_predictor_metrics_per_seed.csv`; F9b-M from `b22_f9bm_probe_rho_per_seed.csv` (family rho); majority-class rates from `b19_majority_class_table.csv`.
  - **F9b rho of the seeds 20261012 and 20261013** is not in the v6.0.1 pack (it holds the primary seed only). As the house Figure 5 does, the script reads it from the frozen run's aggregate records `probes/evaluations/f9b/<partition>.json` (backup `<local path>`, read-only, or `TMMRI_RUN_ROOT`), and stops unless the primary seed there equals the parquet. Their paths and hashes are in the provenance files.
  - Accuracy panel: the majority-class rate of each partition as a solid segment (a majority-class predictor has exactly that accuracy): ACDC test 0.20 (balanced, equal to chance), M&Ms-1 0.353 (DCM, 96 of 272), M&Ms-2 0.574 (NOR, 39 of 68); uniform chance 0.20 dotted. The B19 rates equal the `majority_rate` rows of B21 and B22 (checked).
  - Macro-F1 panel: **no reference line.** Neither recorded reference is a macro-F1 value: a majority-class predictor's macro-F1 is not its accuracy, and 0.20 is the macro-F1 of uniform guessing only on balanced partitions (M&Ms-1 has no MINF case, M&Ms-2 only HCM and NOR). Drawing one would need a new statistic.
  - Macro AUROC panel: chance 0.5 dotted; rho panel: 0 solid, smallest effect 0.10 dashed.
  - B21 and B22 score M&Ms-1 rho on 319 cases (one critical-QC exclusion); the recorded representations on 320. ACDC development out-of-fold values of D20-P, D20-M, F9-M and F9b-M are in the pack but not drawn, because the recorded representations have no development evaluation (NE in `table04_representation_comparison.csv`).
- **`figure07c_coverage_fidelity_b16_{acdc_dev_oof,acdc_test,mms_external,mms2_external}`** (Figure 7c, coverage and fidelity of the rule systems; EPC-4). One half panel per partition. Coverage (accepted / processed) against fidelity to `d_model` among the accepted cases, each with its recorded Wilson 95 % interval. The first panel carries the legend.
  - Sources: `b16_figure_data.csv`, checked column by column against `b16_coverage_fidelity_table.csv`.
  - Systems that admit no rule (coverage 0, fidelity undefined) share one marker, the primary rulebook, in a lane below the fidelity axis, drawn with their common coverage interval and their count: on every partition the primary rulebook and 17 variants (F4 discretizers `wedd_uniform_mm_v1`, `three_quantile_v1`; F4 settings `weight_entropy_0.00`–`1.00`, `bandwidth_multiplier_0.5`, `bandwidth_multiplier_2`, `max_bins_4`, `max_depth_3_max_bins_4`; F5 `beta_0.05`, `beta_0.10`, `strict_only`, `max_length_2`, `max_length_3`; F5 gate bound 0.70).
  - **Skipped (status not EVALUATED):** `f8_cart_selective / selective` on `acdc_dev_oof`, status NE ("F8 selective leaves are not recorded per outer fold").
  - On ACDC test, F5 gate bound 0.50 and the companion rulebook coincide (8 of 50 accepted, fidelity 8 of 8); their markers are offset by ±2.1 printed points for display only.
  - The house Figure 7 panels are 0.32 linewidth; with up to six systems and two-way intervals per partition these panels are drawn at 0.49 linewidth instead.
- **`figureS_decision_sweep_b14`** (supplementary decision sweep; EPC-6). Printed at 0.92 linewidth. For each recorded hypothesis estimate (H1, H2, its never-analyzed sensitivity, H3, and H4a on three partitions): a dot at each threshold of the sweep 0–0.30, filled where the recorded lower bound exceeds it and open where it does not, the lower bound as a vertical bar, 0 solid and the smallest effect 0.10 dashed, and at the right the lower bound and the recorded largest threshold exceeded.
  - The lower bounds are those recorded: 95 % for ρ (H1, H2, H3) and, for H4a, the 97.5 % Bonferroni interval recorded for H4 (exploratory record §3, item 3); the axis label says so.
  - Sources: `b14_sweep_table.csv`; thresholds and the smallest effect from `summaries.json` (B14). The flags are the recorded `lower_bound_exceeds_threshold`, checked equal to lower bound > threshold.

## Encoding

The house system is unchanged: Century Gothic; white background with a light grid at the major ticks; no spines; neutral labelled reference lines (solid: no gain, dashed: smallest effect, dotted: chance); halo intervals; opaque borderless legends; every font printed at 6 pt or more. Entities keep their colours: mask descriptors and the audited predictor teal `#008FA8` circles, D20 vermilion `#E35B2C` diamonds, F9 / F9b violet `#7D55C7` squares, primary seed filled and the other seeds hollow triangles. Each new variant differs from its parent in a channel other than hue:

| Variant | Parent | Encoding |
|---|---|---|
| Development median (B13) | development mean, filled teal circle | open teal circle |
| D20-P, phase-preserving (B21) | D20, filled vermilion diamond | open vermilion diamond |
| D20-M, mask-pooled (B21) | D20 | filled dark-vermilion `#9C3510` diamond |
| F9-M / F9b-M, mask-aware (B22) | F9 / F9b, violet; seed 20261011 filled square, 20261012 open up triangle, 20261013 open down triangle | dark violet `#4B2B90`; seed 20261011 filled square, 20261012 open left triangle, 20261013 open right triangle |
| B16 rulebook variants | primary rulebook, filled teal circle | teal: square F4 at most 2 bins, up-triangle F5 β = 0.20, down-triangle F5 gate bound 0.50, open circle companion rulebook |
| F8 CART surrogate (B16) | — | dark teal `#015B6C` (darkest step of the house teal ramp) hexagon, filled full tree, open selective tree |

Slot order within each Figure 5 partition is fixed, so position also identifies the variant. Palette check with the dataviz validator (all pairs, light surface): teal, vermilion, violet, `#9C3510` and `#4B2B90` pass the colour-vision-deficiency separation (worst ΔE 10.4, deutan, the existing teal–violet pair), the normal-vision floor (worst ΔE 15.8, `#4B2B90` vs `#7D55C7`) and contrast against white; the two dark shades sit by design below the categorical lightness band (OKLCH L 0.39–0.40). Each parent and its dark shade also passes the ordinal check (monotone lightness, step ≥ 0.06, one hue).

## Rules

- **Exploratory, post hoc.** Every figure is reported as `EXPLORATORY_POST_HOC` evidence beside, never instead of, the v6.0.1 figures in `../v601/` and `../paper1/`.
- **Never edit these files.** They change only by rerunning the script on the recorded packs; record each plotted value's source in `source_map.md` when the manuscript uses it.
- **Deterministic.** The PDF creation date and PNG software tag are removed and the provenance date is fixed in the script (`RENDER_DATE`), so a rerun writes byte-identical files and an identical `SHA256SUMS`.

## Regenerate

From the repository root (host venv; Century Gothic is a Windows font):

```
.venv/Scripts/python manuscript/scripts/render_v602_figures.py
```

The script rewrites every figure, its provenance and `SHA256SUMS` (which covers this README); edit the README first, then rerun.
