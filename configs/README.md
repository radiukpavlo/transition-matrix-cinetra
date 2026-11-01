> Public edition of the registered file `configs/README.md` (protocol-v6.0.5, git blob `1f38164912bc`); operational details are replaced as listed in [EDITIONS.md](../protocol/registration/EDITIONS.md).

# Configurations

Configuration files are grouped by protocol and purpose. Paths inside them are project-relative.

**Protocol v6 (CMR-TRANS-AUDIT, `SPECIFICATION.md` 6.0.0).** The study configuration is written at the configs root and frozen with the protocol tag `protocol-v6.0.0` at milestone M0:

- `study_v6.yaml`: study entry point for `tmmri.cli run --study` (added with the v6 pipeline).
- `cohorts.yaml`, `splits.yaml`: ACDC development and internal test, the M&Ms-320 external cohort and the refit arm; fold assignment.
- `formal.yaml`, `transition.yaml`, `wedd.yaml`, `rule_policy.yaml`: formal descriptors A, ridge transition T, discretization and the rule tier.
- `equivariance.yaml`, `deep_encoder.yaml`, `bootstrap.yaml`: optional branches and resampling (`deep_encoder.yaml`: the D20 branch, implemented with protocol-v6.0.1).
- `runtime.env`: container runtime variables (threads, seeds and roots).
- `source_bindings/`: `acdc.json` and `mms.json`, binding each source to its tracked manifest in `registry/sources/`.

**Protocol v6.0.1 amendment (`SPECIFICATION.md` 6.0.1).** These files carry `specification_version: "6.0.1"` (files governed by the frozen v6.0.0 bundle keep `"6.0.0"`, because the field lies inside their hashed sections). They are read by their stages from the study root and are never part of the v6.0.0 analysis-bundle sections:

- `study_v6.yaml` `freeze.amendment_tags`: the amendment tags in order; a run frozen with this code records the newest (`FROZEN.json` `protocol_version_tag`), against which Table 1 is checked.
- M&Ms-2: the `mms2` sections of `cohorts.yaml` and `splits.yaml` and `source_bindings/mms2.json` (archive, extraction, the release CSV contract and member layout read before the tag with the owner's approval). The `mms2` sections are read only by the M&Ms-2 stages and are left out of the digests of the whole `cohorts` and `splits` sections, so v6.0.0 stage records and frozen bundles stay valid.
- `neural_input.yaml`: image input path of the neural branches (tensor cache, input QC, local F9 inference); its `input` block mirrors `protocol/neural_encoder_v6.0.1.design.json`. The pinned torchvision weight hashes live in `registry/weights.yaml`.
- `cart_f8.yaml`: the registered comparator F8 (CART surrogate of `d_model`; `fit-cart-surrogate`, `evaluate-cart-surrogate`).
- `exploratory.yaml`: the exploratory register B1-B8 (`run-exploratory --analysis B<k>`), labelled `EXPLORATORY_POST_HOC`.
- `f9_cnn.yaml`: the F9 trained-CNN comparator (architecture, six candidates, training, selection and refit rules, seeds 20261011-20261013, fit counts) and its local and Modal backends under CPE-1 ([scripts/cloud/README.md](../scripts/cloud/README.md)).
- `probes.yaml`, `trust.yaml` (amendment items 11-14): the F9b concept probe, the evaluation of the alternative bundles (D20, F9, F9b) with Table 4 and Figure 5, and the trust analyses S1-S4. Read by `src/tmmri/experiments/l57_stages.py` from the same configuration tree; never part of the v6.0.0 analysis bundle.
- `data_centric.yaml`: the registered data-centric analyses S5 (data-efficiency curves on `acdc_dev`; `run-data-efficiency`, whose F9 waves write purpose-s5 spec files for `train-f9`, `import-f9-fits --specs` and the Modal app) and S6 (simulated acquisition shift on `acdc_test`; `run-simulated-shift`).

**Protocol v6.0.2 amendment (`SPECIFICATION.md` 6.0.2, Sections 0.4 and 13.7; pending the owner's signature of `protocol-v6.0.2`).** `exploratory_postfreeze.yaml` holds the post-freeze exploratory block B10–B22 (`run-postfreeze --analysis B<k>`), labelled `EXPLORATORY_POST_HOC`. It is a separate lane file, so `exploratory.yaml` and every file hashed by the frozen run stay unchanged. It reads the source run `v6-replication-20261003-1241` read-only and writes to a new run id.

**Protocol v6.0.3 amendment (`SPECIFICATION.md` 6.0.3, Sections 0.5 and 13.8; pending the owner's signature of `protocol-v6.0.3`).** `exploratory_postfreeze_v603.yaml` holds the second post-freeze exploratory block B23–B27 (`run-postfreeze --analysis B<k>`), labelled `EXPLORATORY_POST_HOC`. It is a separate lane file, so `exploratory_postfreeze.yaml` (frozen by `protocol-v6.0.2`) and every file hashed by the frozen run stay unchanged. It reads the source runs `v6-replication-20261003-1241` and `v6-postfreeze-20261028-0201` read-only and writes to a new run id. B25 is the one analysis in which `mms_external` cases enter a fit, as rule-admission data only (Decision D44).

**Protocol v6.0.4 amendment (`SPECIFICATION.md` 6.0.4, Sections 0.6 and 13.9; pending the owner's signature of `protocol-v6.0.4`).** `exploratory_postfreeze_v604.yaml` holds the third post-freeze exploratory block B28–B31 (`run-postfreeze --analysis B<k>`), descriptive analyses of recorded outputs, labelled `EXPLORATORY_POST_HOC`. It is a separate lane file; `exploratory_postfreeze.yaml`, `exploratory_postfreeze_v603.yaml` and every file hashed by the frozen run stay unchanged. It reads the source runs `v6-replication-20261003-1241`, `v6-postfreeze-20261028-0201` and `v6-postfreeze-20261028-2135` read-only and writes to a new run id. The one new fit is the supervised ceiling of B28, on `acdc_dev`.

**Governed.** clinical/ holds the clinical-validation settings of protocol v2. They are hash-locked with clinical_validation/ and are not changed by v6.

**Legacy.** legacy/ holds the v4/v5 configurations used by the legacy commands (`run-minimal`, `validate-native-data`, `run-full-validation`): `legacy/pipeline/` (default, full-validation and native-validation settings; the CLI default is legacy/pipeline/default.yaml), `legacy/labels/` (label-mapping notes) and `legacy/segmentation/`. Recorded commands that name `configs/pipeline/...` resolve to these files. Their output paths point into the git-ignored runtime roots `data/` and `artifacts/`.

A structural change never alters experiment identifiers, seeds, thresholds or cohort definitions.
