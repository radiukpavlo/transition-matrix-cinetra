> Public edition of `docs/NEURAL_ENCODER_EXPERIMENT_PROTOCOL.md` of the authors' development repository (commit `5d88a3041ca9`); operational details are replaced as listed in EDITIONS.md.

# Neural classifier, Transition Matrices and SEMTRA: additional experiment protocol

**Design version:** NEURAL-TM-1.0, 1 October 2026. **Implementation status:** specified, not implemented or executed. **Planned activation:** `protocol-v7.0.0`; that tag has not been created. *Amended 2 October 2026 (protocol/amendments.md, row 2026-10-03):* NEURAL-TM-1.0 as a whole is not activated by `protocol-v6.0.1`; it is planned under `protocol-v7.0.0`. The 6.0.1 comparator F9 implements Sections 4.1-4.2 with its own seeds (SPECIFICATION.md 8.6). Numerical design constants are mirrored in [the design contract](../protocol/neural_encoder_v6.0.1.design.json). This document specifies the new study; it is not evidence that any success criterion has been met.

The owner explicitly requires a trained neural encoder and the objective: **Determine how faithfully rules expressed in clinical concepts reproduce the neural classifier's decisions.** The implementation plan is EXPERIMENT_EXECUTION_PLAN.md. This study replaces the descriptor pipeline as the intended primary research study. The frozen v6.0.0 descriptor experiment remains a historical protocol and a separately identified comparator.

The owner reports that additional held-out exposure is **unknown**. No claim of prospective validation is permissible until the exposure audit below establishes its scope. No held-out outputs were inspected in preparing this design. The audit that motivated it is the ECG/MRI assessment.

## 1. Authority, activation and claim boundaries

The current executable authority is still [SPECIFICATION.md](../SPECIFICATION.md) 6.0.0. Do not silently interpret its descriptor matrix, logistic predictor or optional frozen ResNet-50 branch as this study. The dated entry in [the amendment log](../protocol/amendments.md) records the change in research direction, without pretending that the new executable contract is frozen.

Before any reported neural experiment, implement this design, reconcile the specification, estimator/partition registry, configurations, reason codes, CLI and environment documentation, and freeze them together at a new annotated protocol tag. After the protocol freeze, development selection follows the finite search below. Before evaluation, a second freeze fixes the selected checkpoint, transition, rulebook and all comparator bundles. Creating a protocol tag on today's unchanged descriptor implementation would not satisfy either requirement.

Separate the following claims:

1. **Prediction:** the neural model discriminates the released dataset classes. These labels are research endpoints, not adjudicated clinical diagnoses.
2. **Semantic decoding:** clinically named quantities can be reconstructed from its learned representation.
3. **Decision fidelity:** accepted production rules reproduce that same model's decisions on independent cases.
4. **Encoder alignment:** the concepts used by the rules retain information used by the neural decision head, and controlled perturbations show coherent model/concept/rule responses.

The combined intended claim requires all four. High reconstruction accuracy alone does not establish decision relevance. High fidelity alone can faithfully repeat a model's errors. None of these observational or computational tests establishes that a clinical concept causally determines disease, or that the entire neural network has been explained exactly.

## 2. Mandatory model and matrix definitions

For patient i, let x_i be the ordered ED and ES short-axis image stacks and the acquisition geometry needed to interpret them. Let y_i be the five-class research label and b_i the independently measured clinical-concept vector. Expert contours are used to measure b_i and audit image regions; they are not neural inputs.

The primary model is trained on development patients:

\[
a_i=E_\theta(x_i)\in\mathbb{R}^{64},\qquad
\ell_i=a_iW+c,\qquad d_i=\arg\max_k\ell_{ik}.
\]

The **formal matrix A is the patient-by-64 matrix of activations from the trained encoder**. It is neither a matrix of descriptors nor a matrix of network weights. The encoder's last trainable bottleneck is upstream of the exact linear head that produces d. No separately trained classifier is substituted when defining the rule consequent.

For fitting set S, remove zero-variance A columns and standardize the remaining columns using S only, obtaining Z. Let U be the per-target transformed and standardized measured B, using the existing typed semantic transforms and observation masks. Fit the Transition Matrix T column by column:

\[
(t_j,q_j)=\arg\min_{t,q}\sum_{i\in S_j}(u_{ij}-z_it-q)^2+\lambda_j\lVert t\rVert_2^2,
\qquad \widehat b_{ij}=g_j^{-1}\{s_j(z_it_j+q_j)+\mu_j\}.
\]

S_j contains only permitted, observed target cells; the intercept is unpenalized. Thus **T is estimated from neural activations A and measured concepts B**. The primary decoder remains affine in transformed concept space, preserving a literal transition matrix. A nonlinear concept decoder is a comparator and cannot be presented as this T.

SEMTRA receives cross-fitted reconstructed concepts in original units and the neural decisions. It learns WEDD cuts and production rules of the form `clinical concept intervals -> d_neural`. Measured B and y are forbidden in the inference router. The dataset-label companion and measured-B oracle have separate identifiers and cannot replace the primary rulebook.

Each A file must identify checkpoint, architecture, bottleneck layer, phase order, preprocessing, feature order and fit-patient hashes. Each T must identify the same checkpoint and coordinate system. **Latent columns from different independently trained encoders cannot be pooled, averaged or treated as corresponding coordinates.** Only decoded concepts and decisions, which have shared meanings, may be pooled across folds.

## 3. Cohorts, exposure audit and independent evaluation

Preserve the existing source bindings, geometry correction, duplicate controls, five-class order, semantic registry and crosswalk unless a specific additional amendment changes them. Intended development is the 98 independent ACDC development patients; retain the exclusions of patient055 and patient076. The 50 ACDC test patients are a historical internal evaluation cohort because v5 outputs were already analysed. They cannot become a pristine test set by adopting a new encoder.

The 320 M&Ms cases comprise 318 declared image-identity clusters. The previously analysed 100-case stratum remains historical. The named 220-case stratum is only a **candidate** independent cohort until exposure is reconciled. It is not a development resource, including for self-supervised pretraining, normalization, architecture selection or visually motivated model repair.

Before neural development, create `contracts/exposure_audit.json` from run inventories, stage timestamps, evaluation-access logs and the owner's account. Read metadata and prior access declarations first; do not reopen endpoint tables to decide whether they were viewed. For each cluster and source partition record `historical_exposure`, `known_unexposed`, or `unknown`, the supporting records, earliest access and intended role. Unknown is not equivalent to unexposed. Exposure includes evaluation-driven choices by other agents, not merely viewing final accuracy.

Use every cluster in the predeclared 220-case stratum that is demonstrably unexposed and not linked to an exposed/development cluster as the prospective cohort; freeze its membership **before development model selection**. Record attrition by exposure status and its implications for representativeness. There is no performance-based subset selection. If no credible independent set can be established, complete development and report the available evaluations as historical validation; acquire and preregister a new independent cohort for the prospective claim. Do not randomly split already inspected cases and call the remainder unseen.

All images, slices, phases, augmented versions and duplicate-image partners of a patient/cluster stay together. Fit nothing on ACDC test or M&Ms primary-evaluation cases. A future M&Ms refit arm starts only after the fixed-bundle evaluation, uses the existing 150/169 case partition with its duplicate exclusions, and is explicitly a secondary adaptation study.

For clinical-label compatibility retain the frozen crosswalk: same-definition labels support the specified class metrics; IHD-to-MINF is a separate non-equivalent stratum; unsupported pathologies remain in reconstruction, fidelity and routing analyses. Do not discard a case from model-fidelity evaluation because its diagnosis cannot be crosswalked.

The evaluated target is one record per patient. Primary confidence intervals use independent clusters. For duplicated external records, report all case-level outputs but give each cluster total weight one in aggregate endpoints; the two members each receive weight one-half. No resampled slice counts as an independent patient.

For this new image-based branch, distinguish input/geometry QC from reference-semantic QC. Only image/geometry validity and numerical inference failures may determine neural evaluability or the primary router's QC state. A bad/missing reference mask or concept masks that target's T fit and semantic evaluation; it does not enter the neural input, suppress its prediction or alter the route. Apply this separation consistently during fitting and inference, and report reference availability separately. This requires an explicit branch-specific QC amendment at activation rather than inheriting mask-dependent v6 routing unchanged.

**Data location (CPE-1, CSE-1).** All cohort data stay on the owner's workstation, with two exceptions recorded in `protocol/amendments.md` (2 October 2026). Under CSE-1, off-site storage keeps the unopened original M&Ms-2 archive as storage only. It is never extracted or shared there, so the sealed half stays sealed. Under CPE-1, derived training inputs of the 98 development patients may be processed in the owner's personal Modal workspace for N3/N4 training: preprocessed slice tensors, the two geometry scalars, labels and locally computed concept targets B. The outputs of development fits may be held there too. Raw ACDC files, the ACDC test patients, every M&Ms-1 and M&Ms-2 file or derivative, and `clinical_validation/` are never processed outside the workstation. Evaluation (N5) and audits of evaluation cases (N6) run locally. Every upload and deletion is recorded in a hash ledger.

## 4. Image pipeline and the neural classifier

### 4.1 Inputs and architecture

Use a shared ResNet-18 initialized from the explicitly named torchvision `IMAGENET1K_V1` weights. Pin the downloaded artifact's SHA-256 before training. ImageNet initialization is a prior; the encoder must subsequently learn from the allowed MRI training patients. A frozen pretrained extractor alone fails the mandatory-training requirement.

Use all acquired slices of both ED and ES, retaining their separate phase identities. Phase indices come from the documented source metadata; automated phase detection is outside this experiment's claim. Do not use masks to select slices, find a bounding box, suppress background or create extra channels. No filename, patient identifier, pathology, vendor or centre code enters the network.

Apply the documented ACDC slice-increment correction. Reorient arrays consistently using physical geometry while preserving each short-axis plane and slice order; reject inconsistent geometry rather than silently substituting unit spacing. Within each phase, clip finite image intensities at the 1st/99th percentiles and scale to [0,1]; constant/nonfinite inputs follow explicit input-QC rules. Resample each slice in plane to 1.5 mm isotropic spacing using bilinear interpolation, then center-crop or zero-pad to 224 by 224 pixels, a 336 mm field of view. Replicate grayscale to three channels and apply the fixed ImageNet channel means/SDs. No through-plane interpolation or resize that silently changes physical scale is allowed. Save transform metadata and crop fractions. A development-only image review checks whether this fixed field of view is usable; any redesign must be amended before training selection, never after held-out inspection.

For each slice, concatenate its 512-dimensional ResNet global-pool output with two geometry scalars: centered physical slice position divided by 100 mm and the corrected inter-slice increment divided by 10 mm. Pass these 514 values through a shared `Linear(514,128)` plus GELU. These acquisition geometry values are inputs to trainable layers, not substitutes for clinical B. Use a trainable attention scorer `Linear(128,64) -> tanh -> Linear(64,1)` with softmax over available slices, and concatenate attention-weighted mean and elementwise maximum to obtain 256 features per phase. Padding positions never contribute. Concatenate ED, ES and ED-minus-ES phase vectors (768 dimensions), then `Linear(768,256) -> GELU -> Dropout(0.25) -> Linear(256,64) -> GELU`. The last 64 activations are A. The classification head is exactly `Linear(64,5)`; no hidden skip connection, measured concept, metadata bypass or separately trained decision head is permitted.

This model is **image plus acquisition-geometry based**, with source-selected cardiac phases. Do not describe it as fully automatic raw-cine diagnosis. A geometry-input ablation removes the two geometry scalars while keeping physical resampling and phase order; it tests how much the network uses that additional information.

### 4.2 Training and finite selection

Primary training minimizes class-weighted cross entropy on the five ACDC classes, with inverse-frequency weights calculated on fitting patients and normalized to mean one. No semantic auxiliary loss is used in the primary branch. The concept-supervised ablation below is a different model, not evidence that the primary network independently learned those concepts.

There are exactly six network candidates: train ResNet `layer4` or `layer3+layer4`, crossed with backbone learning rate {0.00001, 0.00003, 0.0001}. All newly added encoder layers and the classifier head are always trained with ten times the backbone learning rate. Other ResNet layers remain frozen. Batch-normalization running statistics remain at their pretrained values in every candidate; affine parameters train only in unfrozen blocks. Save both initial and final parameter hashes and nonzero changes in an unfrozen convolutional block. Updating only the final classifier is insufficient.

AdamW: weight decay 0.0001, betas (0.9,0.999), epsilon 1e-8, gradient norm clipping at 5; cosine learning-rate schedule over 80 epochs; effective batch size four patients; patient order reshuffled each epoch. Use FP32 and deterministic algorithms. Gradient accumulation and deterministic slice chunking/checkpointing may fit the fixed effective batch on the qualified GPU; they cannot alter patient weighting or discard slices. Maximum 80 epochs; earliest early stopping at epoch 20, patience 12 on validation cross entropy, minimum improvement 0.0001; ties choose the earlier checkpoint. No training-set metric selects a checkpoint.

Training augmentation is one transform per patient, shared across phases and slices: rotation uniform [-10,10] degrees, in-plane translation independently uniform [-10,10] mm, intensity multiplication uniform [0.9,1.1], and additive Gaussian noise SD 0.01 in [0,1] intensity units followed by clipping. No reflections, elastic deformation, arbitrary scaling, phase exchange or mixup. Evaluation uses the deterministic transform only.

Select candidates only by three-fold patient/cluster-stratified CV within the current fitting set: highest mean macro-F1, then lowest mean cross entropy, then fewer unfrozen backbone blocks, then lower learning rate. Macro-F1 includes all five classes; a missing class makes a required selection fold invalid, not an excuse to drop that class. Repartition deterministically from allowed patients if necessary according to the fold-generation rule, or declare `NE_INSUFFICIENT_CLASS_SUPPORT` if impossible. Refit the selected candidate on all fitting patients for the rounded-up median of its three selected epoch counts. Do not monitor an outer validation/test set during this refit. Record train/validation losses and F1, calibration and per-class support.

The main seed is 20261001. Seeds 20261002 and 20261003 are mandatory robustness replications of the same selection procedure. No best-seed selection or seed ensembling. The first seed defines the primary frozen model; all three results are reported. Calibration metrics use raw softmax, 10 fixed equal-width bins, multiclass Brier score and log loss; no temperature is chosen from held-out data.

## 5. Full nesting and checkpoint-coordinate integrity

Using an encoder trained on all 98 patients to create apparently out-of-fold transition inputs is prohibited. The supervised encoder, head, early stopping, candidate selection, latent scaler, transition transforms, transition penalties, concept eligibility, cuts and rule discovery must all respect the relevant assessment boundary.

Use three explicitly different validation roles:

1. **Five outer assessment folds** on ACDC development estimate the performance of the entire selection-and-rule procedure. No output from an outer assessment fold chooses a setting. The existing duplicate-aware outer membership is retained. These estimates are development evidence, not independent confirmation after choosing the research design.
2. **Five rule-gate folds inside each outer training set**, and the same five-fold mechanism on the complete development set for the final bundle, construct cross-fitted concepts/decisions and candidate-rule gate evidence.
3. **Three selection folds inside each rule-gate training set** select network candidates and transition penalties and supply the inner out-of-fold semantic substrate for local eligibility/discretization. Every training set at every level uses a deterministic class-stratified cluster allocation: order clusters within class by SHA256 of `neural-tm-v1|role|parent_fit_hash|cluster_id`, distribute round-robin to the requested fold count, and archive all memberships. Do not let a child fold access its parent's held-out cases.

For a fitting set S, the implementation follows this finite procedure:

- `learn_encoder_transition(S)` trains/evaluates the six candidates in the three selection folds. For each candidate/fold checkpoint, fit each candidate ridge penalty on that checkpoint's **own fitting-patient activations**, and decode validation activations using that same checkpoint. Choose the network by the classification criterion; choose each target's penalty by pooled original-unit validation MAE for that selected network, ties to larger penalty. Refit the network on S and fit T in its new latent coordinate system using the selected penalties. The penalty grid, typed transforms and per-target masking inherit `configs/transition.yaml` at v6.0.0. All fitted preprocessing is fold-local. Reuse the selected candidate's three validation concept/decision ledgers as an inner substrate; it is selection data, not an unbiased performance estimate.
- For each rule-gate fold V, run `learn_encoder_transition(S\V)`. Within S\V, select candidate concepts and cuts using only its three-fold inner substrate; reconstruction improvement must occur in at least two of three selection folds. Fit cut stability, WEDD and local rules inside S\V. Evaluate those local rules and their local encoder/head/T on V. Neither eligibility nor cut-stability filters may be inherited from a calculation that included V.
- Pool the five rule-gate validation ledgers **in concept space**, select final concept eligibility using four of five fold reconstruction improvements, and fit the final WEDD cuts and candidate rules. Fit `learn_encoder_transition(S)` to provide the deployable encoder/head/T. Assess each final rule template using the five independently discovered local rulebooks on their respective held-out folds. Template identity is feature names, bin indices/counts and consequent; actual local cut values and their spread must also be retained.
- This pooled gate is an internal selection device: final template selection used pooled development information. Its Wilson bounds are **nominal development bounds**, not independent per-rule confirmation. The enclosing outer assessment fold tests the resulting complete selected bundle, including its final numeric cuts. The separately frozen prospective cohort ultimately evaluates the final development bundle.

This nesting depth is fixed; do not recurse indefinitely. Internal selection scores may be optimistic within their selection level, but no outer assessment or prospective case participates in any training/selection at a lower level. Persist the fit graph so this property is mechanically checkable. A cached neural model is reusable only when its exact fit IDs, selection boundary, candidate, seed, weights and preprocessing match.

Per-object stability on a new case requires five **encoder-plus-transition pairs**, each taking the original images through its own checkpoint and scaler. Decode each into clinical units, then apply the final bundle's cuts. Require agreement with the final reconstruction's bin in at least four of five pairs for every used antecedent. Never feed final-encoder A into a different fold's T. A near-boundary, unavailable or unstable concept causes abstention according to the route policy; the neural class does not break a rule conflict.

## 6. Concepts, rule eligibility and semantic support

Retain all 25 declared transition targets, the seven-target ventricular function/size primary family, and the registry's allowed primary antecedents (at most nine). This fixed registry prevents choosing clinical concepts after viewing held-out fidelity. Do not silently add indexed volumes with absent BSA, duplicate algebraic targets, QC columns, metadata or class labels.

Report reference availability, target-state reasons, original-unit MAE/RMSE/bias, normalized MAE, robust error quantiles, Bland–Altman summaries and nonconstant-target performance. A constant or almost-always-zero target cannot establish useful decoding merely through low error. Raw, unguarded reconstructions determine error metrics; guarded values determine routing. Always save both.

Development concept eligibility requires at least 80% observed reference cells, at least three distinct values, positive fit SD, and at least 5% relative MAE improvement over the fit-mean baseline in four of five gate folds (two of three inside local selection sets). Each WEDD cut must recur in at least 60% of 500 cluster-bootstrap cut refits, with the inherited 0.10-fit-IQR tolerance and SD fallback. This cut bootstrap is conditional on its saved substrate; it must not be described as retraining the neural encoder.

Use the existing Gaussian WEDD profile: entropy/density weights 0.65/0.35, at most three bins, depth two, and the inherited minimum-leaf and tie rules. Discover VPRS rules at confidence >=0.90, beta 0.10, minimum five development patients, length at most three, exhaustive enumeration of eligible subsets, deterministic redundancy reduction, and no top-K cap.

The new neural-study development rule gate requires at least 10 pooled held-out matched clusters, Wilson two-sided 95% agreement lower bound >=0.70, and template reproduction in at least three of five rule-gate folds. These are explicitly revised developmental criteria, not silent changes to v6.0.0's 5/0.60 gate. Archive rejected rules and every failed condition. Do not relax a gate to force a nonempty rulebook.

**Semantic-support audit:** for each fired rule and each antecedent, compare the reconstructed-concept bin with the measured reference bin under the same frozen numeric cuts. An observation is semantically valid only when all antecedents of the rule are observed and agree. For an accepted case with several consistent fired rules, the primary strict case definition requires agreement for the union of all their antecedents; also report the less stringent any-valid-rule diagnostic separately. Missing/blocked reference concepts count as unsupported in the strict endpoint and are also reported separately, not removed from its denominator. Measured B is available to the evaluator only and must never change the inference route.

Maintain three distinct statuses: development-eligible rule, independently supported rule, and unsupported/non-estimable rule. A fixed final rule is independently supported only with >=20 prospective matched clusters, agreement point estimate >=0.90, simultaneous lower bound >=0.80 for fidelity and >=0.70 for all-antecedent semantic support. Use Bonferroni simultaneous two-sided Wilson bounds across **2R** proportions, where R is the complete number of rules frozen before evaluation; use alpha=0.05/(2R) per interval. This conservative rule-level label is secondary; failure does not license deleting rules and recomputing the primary result. Report fold cut spread, overlap and redundancy, and all R intervals, including zero-support rules as NE.

For this rule-level certification only, a cluster counts once when at least one member matches the fixed rule; it is a fidelity success only if every matching member agrees with its neural decision, and a semantic-support success only if every matching member has all antecedents supported. Thus each Wilson numerator is an integer number of independent successful clusters, never a fractional count of duplicated records. The primary aggregate endpoints retain the equal-cluster-weight definition in Section 3. Wilson intervals are nominal approximate intervals; the Bonferroni adjustment does not make their finite-sample coverage exact.

## 7. Primary estimands and predeclared success criteria

All thresholds here are **research acceptance targets chosen for this new design**, not established clinical tolerances or reported achievements. The uncertainty requirements are deliberately separate from point targets. Classification, fidelity, coverage and semantics have different denominators.

Let N be all intended evaluation clusters, including input-QC failures. Let Q indicate that a neural prediction can be made, C indicate an exact, nonconflicting accepted rule decision, D indicate equality of that decision to the same frozen neural classifier, and S indicate strict case-level semantic support as defined above. Set C=0 for QC failures and every abstention. Then:

\[
\text{coverage}=E[C],\quad
\text{covered fidelity}=E[CD]/E[C],\quad
\text{semantic support}=E[CS]/E[C],\quad
\text{faithful semantic coverage}=E[CDS].
\]

Report overall neural evaluability E[Q], all route counts, classifier accuracy/F1 versus evaluable research labels, and rule label compatibility separately. A zero accepted denominator is NE and fails the relevant success requirement, not fidelity 1. All class-, vendor- and centre-specific support remains visible.

**G0 — provenance and validity.** Source-label roundtrip checks, duplicate exclusion, exposure status, partition/fit graph, encoder training evidence, bundle identity, no-inference-reference checks and frozen evaluation all pass. Unknown exposure prevents a prospective success label even if numerical metrics are high.

**G1 — a useful neural classifier.** On the prospective same-definition label subset: neural macro-F1 >=0.70, its cluster-bootstrap 95% lower bound >=0.60, balanced accuracy >=0.70, and per-class recalls/support reported. Compute macro-F1 over the frozen externally compatible class set, including a zero when a supported class receives no correct predictions. If a class in that set is absent, the full endpoint is NE; report the observed-class diagnostic without substituting it. Five-class ACDC historical/OOF results remain mandatory separate outputs. These research cutoffs do not establish clinical readiness.

Under the current crosswalk the same-definition external class set is DCM, HCM, NOR and RV (ARV mapped to RV). A neural prediction of MINF on one of those reference classes remains an error; do not renormalize probabilities or discard that prediction to obtain a four-class metric. Retain the original five-class probabilities for log loss/Brier scoring with the reference labels embedded in that class order. Verify these source bindings at activation rather than infer them from evaluated outcomes.

**G2 — concept reconstruction.** At least five of the seven registered primary-family targets are estimable, and relative reduction rho in pooled fit-SD-normalized MAE versus the development-mean predictor is >=0.10 with paired cluster-bootstrap 95% lower bound >0. Calculate the comparator and scales from training only. Report every individual target even when the pooled criterion passes.

Precisely, for the fixed estimable primary target set J, rho = 1 - [mean over j of MAE_j / s_j] / [mean over j of MAE_mean,j / s_j], where s_j is the original-unit reference SD on development fitting patients. Both methods use the same observed evaluation cells and cluster weights. Determine target estimability from the frozen training contract and reference availability, never from which errors improve.

**G3 — fidelity at useful coverage.** On the prospective cohort: coverage >=0.50 with two-sided 95% lower bound >=0.40; covered fidelity >=0.90 with lower bound >=0.85; and at least 30 accepted independent clusters. These conditions must hold together on the unchanged frozen router. Report faithful coverage E[CD] as well.

**G4 — meanings of the fired rules are supported.** Strict semantic support among all accepted cases >=0.85 with lower bound >=0.75; faithful semantic coverage E[CDS] >=0.35 with lower bound >=0.25. Missing reference concepts remain failures for these strict rates. This gate protects against faithful rules whose reconstructed clinical statements are false.

**G5 — encoder alignment.** Require the decision-subspace test in Section 8.1 to meet its prespecified thresholds using the union of actual deployed rule antecedents. Require the randomization controls and perturbation/localization audits to be completed and reported. Their descriptive nature is explicit; positive saliency pictures are not a substitute for G5's numerical test.

**Study outcomes:** `SUPPORTED_NEURAL_SEMANTIC_AUDIT` requires G0–G5 on the prospective cohort. `FAITHFUL_RULE_SURROGATE_ONLY` requires G0/G3/G4 but fails or cannot estimate G5. `RECONSTRUCTABLE_CONCEPTS_ONLY` requires G0/G2 without G3/G4. `INCONCLUSIVE_SUPPORT` denotes an uncertainty/support/exposure limitation; `CRITERION_NOT_MET` denotes an estimable numerical failure. Always publish the full vector of gate outcomes; these labels must not conceal failed prediction or reconstruction gates. Historical data can receive descriptive gate results but never the prospective success label.

## 8. Mandatory encoder-to-rule alignment experiments

### 8.1 Decision-relevant concept subspace: primary alignment test

Exploit the actual linear neural head rather than fitting a more flattering substitute. Rewrite it in standardized latent coordinates, \(\ell=zW_z+c_z\), accounting exactly for removed constant columns and the A scaler. Remove the class-common logit component with \(W_c=W_z(I-\mathbf{1}\mathbf{1}^T/K)\) and the analogous centered intercept. Let T_R contain the transition columns for the union of concepts occurring in the deployed primary rules. Define the orthogonal projector \(P_R=T_RT_R^+\) using relative SVD tolerance 1e-10.

Archive rank, singular values, condition number and

\[
\eta=\lVert(I-P_R)W_c\rVert_F^2/\lVert W_c\rVert_F^2.
\]

On independent cases compare original logits to \(zP_RW_c+c_c\), retaining the original intercept and class tie order. Since zT_R is the transformed concept output before decoding, this tests whether those rule concepts span the head's decision directions. Require **eta <=0.10**, projected-head decision agreement >=0.95 with a two-sided 95% lower bound >=0.90, and all eligible neural cases included. If there are no deployed rules, zero head norm, or no retained concept rank, G5 is NE and cannot pass. Also report the same quantities for all eligible concepts as a diagnostic; their success cannot rescue failure of the actual rule-antecedent union.

Report per-case decision margins and the remaining-logit contribution. Exact algebraic equivalence of all class logits up to a shared offset would require zero residual in this coordinate system; the 0.10 research criterion is an approximation, not such a proof. Collinear concepts cannot be assigned independent causal importance from T coefficients. Include a development-fitted semantic-only surrogate and the residual-latent probe below to expose additional nonlinear predictability, without replacing the original-head test.

### 8.2 Semantic sufficiency and residual information

Fit multinomial logistic probes with C in {0.01,0.1,1,10}, selected within the same development selection boundaries, to reproduce neural d using (a) transformed reconstructed concepts, (b) latent residual z(I-P_R), and (c) both. Fit probes on inner cross-fitted neural decisions and concepts, never on test decisions. Report independent fidelity, log loss and coverage-free agreement, plus the gain from adding the residual. Use all classes and missing-state indicators from training-only preprocessing, with no reference B at inference.

These probes are secondary diagnostics; the residual can carry class information correlated with concepts even when it is orthogonal geometrically. A residual probe's success therefore demonstrates predictive information, not an independent disease mechanism. Do not claim full representation completeness from the projection metric alone.

### 8.3 Controlled image and concept perturbations

Predeclare transformations without looking at held-out examples. On all technically eligible evaluation cases run, one factor at a time: translations (+5 and -5 mm along each in-plane axis), rotations (+5 and -5 degrees), intensity multipliers 0.9 and 1.1, and Gaussian noise SD 0.02 with seed derived from cluster ID. Use the same spatial perturbation across phases/slices. Compute prediction flip rate, change in original predicted-class logit margin, clinical reconstruction changes, route changes, and fidelity conditional on unchanged original neural decisions. Report all cases; do not retain only unchanged cases for the main endpoint. These are nuisance-stability tests, not disease counterfactuals.

Run ED-only, ES-only and phase-swap diagnostics, and physically calibrated segmentation-only shape perturbations for the semantic measurement code. Phase ablations probe model dependence; arbitrary phase swaps need not preserve diagnosis. Changes to masks alone should change measured reference B but must leave encoder A and d byte-identical. They must not be presented as changes to the primary neural input.

For model relevance/localization, compute integrated gradients of the frozen original predicted-class logit using 64 integration steps and a zero image in normalized input space as the fixed baseline; report completeness residuals. Use 16-by-16-pixel occlusion at stride 16, replacing image content with that same baseline, across each phase. Aggregate at patient level. Quantify attribution concentration in LV/RV/myocardium and equal-area background controls using reference masks only after predictions; report size normalization and background dependence. Repeat attribution on randomly reinitialized head, then head plus unfrozen blocks, on the fixed first 12 development clusters ranked by hash. Report whether maps change and prediction degradation; no selected illustrative image counts as a quantitative pass.

Perform latent subspace removal z -> z(I-P_R) and retention z -> zP_R as algebraic model probes, with random equal-rank orthogonal subspaces (20 fixed seeded draws) as controls. Do not label these off-manifold edits as anatomically realizable interventions. A concept-bottleneck comparator can support concept-value interventions, but its behavior is not an intervention on the original classifier.

## 9. Comparators and experiments required for completion

Every comparison uses identical patient boundaries, target definitions and inference denominators. Relearn each comparator's own preprocessing, cuts and gates on development. Record independent hashes and identical input manifests. No comparator is promoted to the primary based on external performance.

**Prediction/representation controls:** the existing 57-descriptor logistic pipeline; the specified architecture with a completely frozen pretrained backbone but trainable pooling/bottleneck/head; a randomly initialized model with all backbone blocks trainable and the same three learning rates as its finite search; and the primary model without geometry scalars. The frozen-backbone comparator selects the new-layer learning rate from ten times the three primary backbone rates. Other training/selection settings are unchanged, except that the from-scratch control updates batch-normalization running statistics using fitting patients only. Preserve ED/ES separation in every neural control. Parameter/layer training differences are part of each contrast and must be disclosed.

**Transition controls:** training mean, Moore–Penrose affine mapping, identity-transform ridge, global-penalty ridge and the existing RBF kernel-ridge comparator. All primary/comparator A matrices originate from their identified encoder; descriptor-based transitions are explicitly marked descriptor controls. Report paired target errors and complete downstream rule results for each declared transition, not only improved reconstruction scores.

**Rule/discretization controls:** primary WEDD versus training tertiles and Gaussian-WEDD entropy weights {0.5,0.8}; VPRS beta {0.05,0.20} and maximum length two as one-factor changes. Rerun eligibility, cuts, discovery, gates and routing for each. Retain soft candidates without labels and conflicts as abstentions. Fit a depth-three CART surrogate on the same reconstructed concepts and neural decisions, minimum leaf five, balanced class weights, seed fixed. Its selective version accepts only leaves with development OOF fidelity >=0.90, >=10 matches and nominal Wilson lower bound >=0.70; external results cannot choose the leaf threshold. Report its full-coverage and selective versions.

**Reference and semantic controls:** measured-B oracle rules for the same neural d; dataset-label rules on reconstructed B; a direct concept-bottleneck network with the same image backbone, typed concept head, and label head on predicted concepts only. Train the bottleneck comparator with weighted class CE plus mean observed-cell standardized concept squared error at weights {0.1,1.0}, selected on development macro-F1 with reconstruction error as the tie-breaker. No residual latent bypass. Report concept interventions only for this comparator. It is not the primary model and cannot be used to say that the primary classifier reasons exclusively through concepts.

For that comparator, retain the 256-dimensional shared hidden layer and replace the final 64-dimensional layer/GELU with a linear output for the 25 transformed standardized concepts; use `Linear(25,5)` as its classification head. Fit concept transforms and scales only on fitting patients and mask unavailable reference cells in the concept loss. Retain all finite concept predictions at inference. Use the same network training search and nested boundaries, crossed with the two auxiliary-loss weights. No evaluation reference concept may silently replace a prediction except in the explicitly labelled intervention diagnostic.

**Negative controls and reproducibility:** three primary seeds; 20 within-development permutations of patient-to-B correspondence, keeping missingness patterns with their B rows, for the transition/rule pipeline with the encoder fixed; and three development label permutations for the complete neural training procedure. Preserve class counts and patient grouping. Evaluate label-permutation controls on outer development folds only; never train on a held-out label permutation. B-permutation controls retain the authentic neural d and expose chance semantic decoding. These controls have finite Monte Carlo resolution; do not report precision unsupported by their replicate count.

Derive every non-training RNG seed as the first eight hexadecimal digits of SHA256 of `neural-tm-v1|experiment_id|replicate_index|fit_id_hash` interpreted as an unsigned 32-bit integer; add the cluster ID to that string for patient-specific perturbations. Record the string and resolved seed, use NumPy PCG64 for statistical/permutation draws, and seed the pinned PyTorch generator separately with that value for neural operations. No seed is retried for favorable outcomes. Primary training seeds and the primary bootstrap seed remain the explicit integers already specified.

Before final-bundle freeze, complete every development ablation and freeze the entire comparator register. Held-out evaluation is one predeclared batch of fixed models. If compute prevents a mandatory comparison, mark study completion incomplete and report the missing comparison; do not quietly narrow the experimental claim.

## 10. Statistical analysis, multiplicity and support planning

Use two-sided 95% uncertainty intervals throughout. For primary proportions on independent singleton clusters, use Wilson intervals, including all-success and zero-success cases. For a cluster with multiple records, aggregate its numerator/denominator contribution and use 10,000 paired cluster-bootstrap replicates, seed 20261011; report singleton-stratum Wilson intervals separately. Never give duplicated records independent Bernoulli weight. For MAE differences, rho, macro-F1 and paired metric differences, use the same 10,000 cluster resamples and percentile intervals, preserving paired predictions. More than 5% undefined bootstrap replicates makes that interval NE, with the count reported; do not silently discard them. Below 20 clusters, subgroup intervals/tests are NE and counts/point estimates remain visible.

G1–G5 form an **intersection-union success claim**: every required condition must pass. Each bound is assessed at its prespecified level; success is never obtained by selecting one favorable endpoint from the set. Other positive claims are separately controlled. After G0–G5 pass, test the two planned secondary superiority contrasts for faithful semantic coverage: primary versus tertile SEMTRA and primary versus selective CART. Require a point improvement >=0.05 and reject the paired equal-coverage-success null with exact McNemar tests, Holm-adjusted across the two at family alpha 0.05, using one `C*D*S` outcome per singleton cluster. For duplicate clusters use a cluster sign-flip permutation test of paired cluster-weighted contributions (100,000 draws, plus-one correction) rather than treating records independently. Report two-sided paired intervals and all unadjusted/adjusted p-values. If the primary gates fail, these contrasts remain descriptive.

All remaining ablations, attribution results, probes, seed variation and subgroup contrasts are descriptive with paired intervals. Do not claim superiority from overlapping/nonoverlapping marginal CIs. Rule-level simultaneous bounds use the separate 2R family in Section 6. Never calculate post-hoc observed power.

The fixed inventory (98 development, 50 historical internal, up to 220 candidate prospective cases) limits certainty. No claim of adequate power is inferred from those counts. Before freeze, calculate design sensitivity at N={50,100,220}, coverage={0.3,0.5,0.7}, true covered fidelity={0.90,0.95,0.98}, and strict semantic support={0.80,0.90,0.95}. Enumerate the binomial acceptance count and conditional binomial successes to calculate the probability of meeting each marginal point/bound criterion. Report marginal power, expected evaluable counts and the assumption of independent singleton cases; correlated joint outcomes and QC loss require a separate sensitivity simulation with assumptions stated. These are planning scenarios, not empirical forecasts. Repeat planning for the exposure-audited fixed N without inspecting endpoints.

Useful exact boundary checks for the interval implementation: with all successes, the two-sided 95% Wilson lower bound is n/(n+1.959963984540054^2). It first exceeds 0.85 at 22/22, 0.90 at 35/35, and 0.95 at 73/73. These counts do not imply adequate power when errors occur and do not include rule-selection multiplicity. In particular, the 50-case historical internal set is too small to promise well-supported class-specific rule certification. A rulebook may pass aggregate fidelity while most individual rules remain insufficiently supported; report that distinction.

## 11. Provenance, label integrity, failure and release

Before fitting, assert exact identifier joins among images, labels, concepts and folds; compare source-normalized versus serialized-and-reread labels **row by row and class by class**; check masks/phase conventions and geometry units; reject missing, duplicated, truncated or re-ordered target materializations. A hash of incorrectly serialized labels is not a semantic validation. Save aggregate mismatch counts and private row-level ledgers in the licensed runtime root. Nonzero label mismatch blocks the experiment. No source diagnosis is inferred from a filename.

The pre-fitting roundtrip audit covers development labels. Held-out label roundtrip validation occurs inside the frozen evaluator after bundle freeze, before metrics, with no access from model-selection code. Its failure blocks that evaluation and is logged as an exposure event if endpoint-derived information is revealed. Do not open held-out label tables early under the name of a serialization audit.

Freeze protocol and implementation before development selection. Freeze selected models, all preprocessing, every T, cuts, rules, inference thresholds, metrics code, comparator list and exposure-audited evaluation IDs before evaluation. All three primary-seed bundles are fixed before opening any result. A single evaluation plan hash covers every predeclared partition/model. A new run ID never restores blindness. Do not use historical internal-test results to repair the model before the prospective batch.

Required provenance follows the whole path: `image -> preprocessing -> encoder checkpoint -> A -> scaler -> T -> decoded Bhat -> cuts -> matched rules -> route -> neural decision comparison`. The final release must contain:

- Exposure and source-label/geometry audits, source hashes, split/cluster/fold manifests, dependency/weight hashes and complete estimator fit graph.
- Model candidates, epoch-selection records, losses/learning curves, initial/final weights and trainable-layer deltas, all three seeds and failed runs.
- Per-checkpoint A matrices and neural logits, per-target T/intercept/scalers/transforms, raw and guarded Bhat, nested concept/decision ledgers and comparator results.
- Concept/rule gate attrition, candidate and final rules, numeric cut bootstrap results, rejected rules, pooled nominal gate evidence and independent per-rule results.
- Per-case matched rule IDs, fired intervals and values, five-pair stability votes, route/reasons, class probabilities, semantic-support results in a separate evaluator-only file, and cluster IDs.
- Reconstruction, prediction, fidelity/coverage, semantic validity, projection/probe/perturbation/localization results, stratified results, uncertainty, multiplicity decisions and a complete G0–G5 outcome record.
- A release manifest and explicit limitations, including any NE, empty rulebook, failed criterion, missing experiment or historical exposure.

A production-rule record names its physical-unit intervals, neural consequent and checkpoint hash; gives discovery, cross-fit and prospective support separately; reports empirical fidelity and semantic support with the appropriate bounds; states whether it is development-eligible or independently supported; and links to the exact inference trace. “High LV volume” alone is insufficient: provide the interval, units, domain, reconstruction error and evidence. Rules are research decision explanations and do not prescribe care.

On a pre-evaluation bug, invalidate affected development artifacts and rerun under recorded code changes. On a post-evaluation bug, preserve the failed output, document the correction and exposure, and report corrected evaluation as post hoc for those cases. A genuine crash before metrics or endpoint-derived diagnostics can be resumed only with an identical bundle and recorded failure boundary. No silent cutoff change, seed selection, class removal, rule pruning or test-set retry is allowed.

## 12. Methodological grounding and procedural assistance

The distinction between reconstructing concepts and testing their sufficiency for a model decision follows the problem addressed by [Yeh et al., On Completeness-aware Concept-Based Explanations in Deep Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2020/hash/ecb287ff763c169694f682af52c1f309-Abstract.html). Section 8.1's linear-head projection test is this protocol's explicitly defined diagnostic, not a claim to implement that paper's completeness estimator.

The separately identified bottleneck comparator follows [Koh et al., Concept Bottleneck Models](https://proceedings.mlr.press/v119/koh20a.html). Its concept interventions must be interpreted with the procedural limitations discussed by [Shin et al., A Closer Look at the Intervention Procedure of Concept Bottleneck Models](https://proceedings.mlr.press/v202/shin23a.html). Neither paper licenses interpreting post-hoc decoding as a causal explanation of an unconstrained encoder.

The architecture/weight identifier is supported by [torchvision's ResNet-18 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html); the MRI preprocessing and trained pooling/bottleneck above are new protocol choices, not torchvision defaults. Existing Transition Matrices/SEMTRA source bindings remain those in SPECIFICATION.md and the prior audit.

Experimental-design, statistical-analysis and statistical-power skill guidance informed this design. Procedural reference: Kassis T, Agarwal V, He Y, Patel D, Brueckner AM. *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents* (2026), [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065); current metadata checked on 1 October 2026. This reference documents assistance, not evidence of experimental performance.
