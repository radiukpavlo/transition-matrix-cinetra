> Public edition of `manuscript/reports/paper1/public_release_supplement.md` of the authors' development repository (commit `40a27db05c9d`); operational details are replaced as listed in [EDITIONS.md](../protocol/registration/EDITIONS.md).

*This document accompanies the Supplementary Materials of the article "CINETRA: A Multi-Vendor Audit of Semantic Concept Transfer in Cardiac MRI" and documents three of its sections in the public release: the notation registry, the data-efficiency curves and the cross-hardware quality control.*

## 1. Notation Registry

Table R1 lists the symbols of the article. The centered **A** has rank at most 46, because its nine fraction terms are linear combinations of the voxel fractions in the 48 per-label statistics, and per-phase label fractions sum to one.

**Table R1.** Notation; each matrix row is a case.

| Symbol | Meaning |
|---|---|
| *n*, *k*, *l* | Number of cases, formal descriptors (*k* = 57) and semantic targets |
| **A** ∈ ℝ<sup>*n*×*k*</sup> | Formal representation: 57 mask descriptors |
| **B** ∈ ℝ<sup>*n*×*l*</sup> | Semantic matrix: measured concepts in physical units |
| **M** | Observation mask; *M*<sub>*ij*</sub> = 1 if concept *j* of case *i* is observed and not QC-masked |
| *K*, **Z** | Retained descriptor columns and the standardized descriptors |
| *g*<sub>*j*</sub>, *y*<sub>*ij*</sub> | Typed transform of target *j* and its standardized transformed value |
| **w**<sub>*j*</sub>, *w*<sub>0,*j*</sub>, *λ*<sub>*j*</sub> | Ridge weights, unpenalized intercept and penalty of target *j* |
| **T**, **B̂** | Transition matrix and reconstruction in physical units |
| *O*<sub>*j*</sub>, *Q*<sub>*j*</sub> | Observed fitting rows and evaluable cases of target *j* |
| *H*<sub>*j*</sub>, *p̂*<sub>*j*</sub>, *J*<sub>*j*</sub>, *σ̂*<sub>*j*</sub> | WEDD split entropy, kernel density, criterion and sample standard deviation of a reconstructed concept |
| *s*<sub>*j*</sub>(*x*) | Symbol (bin) of concept *j* for case *x* |
| 𝒟, [*x*]<sub>*P*</sub>, *Y*<sub>*c*</sub>, *β* | Decision system, indiscernibility class, decision class *c* and variable-precision tolerance |
| *L*<sub>*W*</sub>, *m*, *π̂* | Wilson lower bound, pooled held-out matches and agreement rate of a rule |
| *a*(*x*), sim(*x*, *r*), *R*<sub>*r*</sub> | Acceptance indicator, soft similarity and antecedent set of rule *r* |
| *d*<sub>model</sub> | Decision of the audited predictor |
| *γ*(**x**), *τ*<sub>*θ*</sub> | Confidence and selective-prediction threshold at *θ*% development coverage, *θ* ∈ {50, 80} (analysis S1) |
| *D*(**x**) | Mahalanobis OOD score (analysis S2) |
| **Z**<sup>H</sup> | Harmonized standardization (analysis S3) |
| *ϑ* | Temperature (analysis S4); written *ϑ* to avoid confusion with **T** |
| *ρ*<sub>*j*</sub>, *ρ*, *ρ*<sub>EF</sub>, Δ*ρ* | Per-target and family skill scores, ejection-fraction score and harmonization effect; *ρ* is not a correlation |

## 2. Data-Efficiency Curves

Figure R1 and Table R2 show analysis S5. Both pipelines were refitted on class-stratified fractions of each outer fitting set of the ACDC development set and scored out of fold on all 98 patients; F9 used its primary seed with the full selection procedure (570 fits). No class lacked support at any fraction.

**Figure R1.** Data efficiency on the ACDC development set: Subfigure (a) presents *ρ* with 95% intervals and Subfigure (b) the macro-F1. Figure files: `manuscript/article/figures/paper1/figureS3_data_efficiency_a.pdf` (a) and `manuscript/article/figures/paper1/figureS3_data_efficiency_b.pdf` (b).

**Table R2.** Data efficiency, ACDC development set (out of fold; 98 clusters; bootstrap CI).

| Fraction | Repeat | Descriptors: *ρ* [95% CI] | Descriptors: macro-F1 | F9: *ρ* [95% CI] | F9: macro-F1 |
|---:|---:|---|---:|---|---:|
| 0.5 | 0 | 0.455 [0.394, 0.507] | 0.712 | 0.097 [0.035, 0.156] | 0.312 |
| 0.5 | 1 | 0.453 [0.387, 0.511] | 0.745 | 0.077 [0.028, 0.127] | 0.361 |
| 0.5 | 2 | 0.459 [0.393, 0.514] | 0.806 | 0.120 [0.059, 0.175] | 0.299 |
| 0.75 | 0 | 0.521 [0.462, 0.570] | 0.819 | 0.193 [0.128, 0.253] | 0.439 |
| 0.75 | 1 | 0.500 [0.441, 0.550] | 0.878 | 0.206 [0.146, 0.261] | 0.460 |
| 0.75 | 2 | 0.505 [0.447, 0.555] | 0.799 | 0.228 [0.155, 0.294] | 0.488 |
| 1.0 | 0 | 0.532 [0.474, 0.579] | 0.868 | 0.239 [0.186, 0.289] | 0.542 |

## 3. Cross-Hardware Quality Control

Before the F9 freeze, the 54 winning selection fits were evaluated both on the cloud graphics processing unit (GPU; NVIDIA L40S) and locally (NVIDIA GeForce GTX 1080; NVIDIA Corporation, Santa Clara, CA, USA) with the same trained weights and the eight-transform D4 average. The largest absolute difference in validation cross-entropy was 7.95 × 10<sup>−7</sup> against a tolerance of 10<sup>−3</sup>, and the macro-F1 was identical in every fit (tolerance 10<sup>−9</sup>); the check passed. This check deviated from the protocol, which had specified a comparison of logits and argmax agreement.
