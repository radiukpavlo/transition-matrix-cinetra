# Reproducing CINETRA

This guide re-runs the frozen study from the public code and your own copies of the three datasets, then compares
your results with the published ones. The study itself ran once, as run `v6-replication-20261003-1241` on
3–4 October 2026; this repository contains its code (byte-identical to the registered commit), configuration,
aggregate results and ACDC-fitted bundles, but no image, mask or other case-level data.

Every command of the original execution is listed in order below. Where a step needs a GPU, the original ran on an
NVIDIA GeForce GTX 1080 (local) and, for the F9 training waves only, on NVIDIA L40S GPUs in a cloud workspace
(the cloud-processing exception CPE-1 of the protocol). Both paths are described.

## 1. What you need

| Item | Requirement |
|---|---|
| Operating system | Linux, macOS or Windows with a Linux container runtime |
| Containers | Docker Engine or Docker Desktop with Compose v2; for GPU steps, the NVIDIA Container Toolkit |
| CPU and memory | 8 or more cores and 32 GB of memory recommended; the original used 14 of 16 logical CPUs |
| GPU | Any CUDA GPU supported by the pinned PyTorch 2.14 wheels (cu126; Pascal or newer), for D20, F9 inference, F9 training (if not on Modal), S5 and S6 |
| Disk | About 60 GB for the datasets, the caches and one run root (the original F9 checkpoints alone take about 40 GB) |
| Host tools | git; Python 3.9 or later (for `tools/`); OpenSSH 8.0 or later (signature check); `bsdtar` (libarchive) to read the M&Ms-2 RAR archive |
| Time | Under one hour of CPU work for the core pipeline, about 5 CPU-hours for the exploratory analyses B1–B9, and GPU time for F9: about 14 h for the 342 F9 fits and about 19 h for the 570 S5 fits on a GTX 1080 (about 2 h on cloud L40S GPUs) |

## 2. Get the code and check the registration

```bash
git clone https://github.com/radiukpavlo/transition-matrix-cinetra.git
cd transition-matrix-cinetra
python tools/verify_registration.py
```

The last command must end with `PASS`: it verifies the signatures of the two protocol tags and that every published
code, test, config and registry file is byte-identical to the registered commit
([protocol/registration/](../protocol/registration/README.md)).

## 3. Get the data

The datasets are not redistributed here. Obtain them from their organizers and accept their terms:

| Cohort | Source | Terms | What the study uses |
|---|---|---|---|
| ACDC | [Creatis, University of Lyon](https://www.creatis.insa-lyon.fr/Challenge/acdc/) | CC BY-NC-SA 4.0 | The full release (100 training and 50 testing patients, 1,056 files) |
| M&Ms-1 | [M&Ms challenge, University of Barcelona](https://www.ub.edu/mnms/) | Research Use Agreement | The complete release (345 cases, 691 files) |
| M&Ms-2 | [M&Ms-2 challenge, University of Barcelona](https://www.ub.edu/mnms-2/) | Research Use Agreement | The archive `MnM2.rar` (360 subjects) |

The ACDC release can be downloaded and verified with the acquisition script (it reads the registered file manifest):

```bash
python scripts/data/acquire_datasets.py acdc
```

Then check your copies against the registered manifests:

```bash
python tools/verify_data.py --acdc data/raw/acdc/database
python tools/verify_data.py --mms <path to the complete M&Ms-1 release>
python tools/verify_data.py --mms2 <path to MnM2.rar>
```

Only the ED and ES frames and masks of each case are used. The M&Ms-2 archive is never unpacked as a whole:
`extract-mms2` (step 7) extracts only the 720 ED and ES files of the 180 held-out subjects, and the sealed half
(ranks 181–360 of the registered hash split) stays unread. `python tools/verify_data.py --split` recomputes that
split from the code and checks it against `registry/sources/mms2_split_manifest.csv`.

## 4. Configure

1. Copy `.env.example` to `.env` and set `TMMRI_ACDC_SOURCE`, `TMMRI_MMS_SOURCE` and `TMMRI_ARCHIVE` (an existing
   folder for environment records) to your paths. Use forward slashes.
2. For M&Ms-2, pass the archive path to `extract-mms2 --archive` (step 7), or set `archive.reference_copy` in
   `configs/source_bindings/mms2.json`, which holds a placeholder in this public edition.

## 5. Build the environment and run the tests

```bash
docker compose build tmmri
docker compose --profile gpu build tmmri-gpu
docker compose run --rm -T tmmri python -m pytest -q
```

The images pin Ubuntu 24.04, CPython 3.12.14 and every Python package by digest (`uv.lock`); see
`pyproject.toml`. Tests that need data or a GPU skip themselves when those are absent.

## 6. Create local protocol tags

The pipeline protects the registered design with locks: M&Ms-1 endpoints refuse to run unless an annotated tag
`protocol-v6.0.0` is contained in HEAD, and M&Ms-2 refuses unless a signed annotated tag `protocol-v6.0.1` is
(`stages.protocol_frozen`; exit code 3). These locks guarded the authors' own blinding. A reproduction happens after
the registration, so create the two tags on the commit you run, signing the second with your own key:

```bash
git tag -a protocol-v6.0.0 -m "local tag for a reproduction of CMR-TRANS-AUDIT 6.0.0"
git tag -s protocol-v6.0.1 -m "local tag for a reproduction of CMR-TRANS-AUDIT 6.0.1"
```

The registered tags themselves are checked in step 2. Do not push your local tags.

## 7. Run the study

Set a run id once and run every command from the repository root. `$CPU` and `$GPU` are shorthands:

```bash
RUN_ID=v6-reproduction-$(date +%Y%m%d-%H%M)
CPU="docker compose run --rm -T tmmri python -m tmmri.cli"
GPU="docker compose --profile gpu run --rm -T tmmri-gpu python -m tmmri.cli"
```

Before any step, `$CPU preflight --run-id $RUN_ID --step <step>` lists its preconditions (exit 0 means none failed).
Stage records are once-only: a recorded stage is skipped, and a changed input makes it refuse (exit 7) instead of
overwriting. Never use `--force`; start a new run id instead.

**7.1 Core pipeline (ACDC development fit, ACDC test and M&Ms-1 evaluation).**

```bash
$CPU run --run-id $RUN_ID --until run-equivariance
$CPU fit-cart-surrogate --run-id $RUN_ID
$CPU run-exploratory --analysis B6 --run-id $RUN_ID
$CPU run-exploratory --analysis B5 --run-id $RUN_ID          # synthetic gate power; about 2.6 CPU-hours
```

**7.2 Image comparators D20 and F9 (GPU).**

```bash
$GPU verify-torch-weights --run-id $RUN_ID                     # ImageNet weights from download.pytorch.org
for s in acdc_dev acdc_test mms_external; do $CPU build-tensor-cache --split $s --run-id $RUN_ID; done
for s in acdc_dev acdc_test mms_external; do $GPU extract-deep-features --split $s --run-id $RUN_ID; done
$GPU fit-deep-encoder --run-id $RUN_ID
$CPU plan-f9 --run-id $RUN_ID
$GPU train-f9 --run-id $RUN_ID --backend local --specs neural_f9/plan/wave1_specs.json
$CPU select-f9 --run-id $RUN_ID
$GPU train-f9 --run-id $RUN_ID --backend local --specs neural_f9/plan/wave2_specs.json
$CPU freeze-f9 --run-id $RUN_ID
$GPU infer-f9 --split acdc_dev --run-id $RUN_ID
$CPU fit-probes --run-id $RUN_ID
```

The original trained the F9 waves on cloud GPUs (`scripts/cloud/modal_train.py`, with the procedure in
[scripts/cloud/README.md](../scripts/cloud/README.md)); local training is the protocol's registered fallback and
gives the same selection and refit rules.

**7.3 Data efficiency S5 (development only).**

```bash
$CPU run-data-efficiency --step descriptor --run-id $RUN_ID
$CPU run-data-efficiency --step f9-wave1 --run-id $RUN_ID
$GPU train-f9 --run-id $RUN_ID --backend local --specs data_efficiency/f9/train_wave1_specs.json
$CPU run-data-efficiency --step f9-wave2 --run-id $RUN_ID
$GPU train-f9 --run-id $RUN_ID --backend local --specs data_efficiency/f9/train_wave2_specs.json
$GPU run-data-efficiency --step f9-infer --run-id $RUN_ID
$CPU run-data-efficiency --step f9-aggregate --run-id $RUN_ID
```

**7.4 Trust bundle and the evaluations on ACDC test and M&Ms-1.**

```bash
$CPU fit-trust --run-id $RUN_ID
for s in acdc_test mms_external; do
  $CPU evaluate-cart-surrogate --split $s --run-id $RUN_ID
  $GPU infer-f9 --split $s --run-id $RUN_ID
  $CPU evaluate-alt-bundles --split $s --run-id $RUN_ID
  $CPU evaluate-trust --split $s --run-id $RUN_ID
done
$CPU run-simulated-shift --vendor all --step descriptors --run-id $RUN_ID
$CPU run-simulated-shift --vendor all --step tensors --run-id $RUN_ID
$GPU run-simulated-shift --vendor all --step d20 --run-id $RUN_ID
$GPU run-simulated-shift --vendor all --step f9 --run-id $RUN_ID
```

**7.5 M&Ms-2 (prospective cohort).** The extraction runs on the host, because it needs `bsdtar` to read the RAR
archive; install the package on the host first (`python -m pip install -e .`).

```bash
PYTHONPATH=src python -m tmmri.cli extract-mms2 --archive <path to MnM2.rar> --tar bsdtar
$CPU stage-sources --dataset mms2
$CPU validate-cohort --dataset mms2 --run-id $RUN_ID
$CPU build-semantics --dataset mms2 --run-id $RUN_ID
$CPU extract-formal --dataset mms2 --run-id $RUN_ID
$CPU evaluate --split mms2_external --no-fit --run-id $RUN_ID         # H3 and the H4 family
$CPU audit-shift --dataset mms2 --run-id $RUN_ID
$CPU summarize-predictor --split mms2_external --run-id $RUN_ID
$CPU summarize-predictor --partition mms2_external --extended-only --run-id $RUN_ID
$CPU evaluate-cart-surrogate --split mms2_external --run-id $RUN_ID
$CPU build-tensor-cache --split mms2_external --run-id $RUN_ID
$GPU extract-deep-features --split mms2_external --run-id $RUN_ID
$GPU infer-f9 --split mms2_external --run-id $RUN_ID
$CPU evaluate-alt-bundles --split mms2_external --run-id $RUN_ID
$CPU evaluate-trust --split mms2_external --run-id $RUN_ID
```

`extract-mms2` stops with exit 8 (`HUMAN_GATE`) after a successful extraction, so that a person confirms it before
anything reads the images; continue with the next command.

**7.6 Exploratory analyses and the release manifest.**

```bash
for b in B1 B2 B3 B4 B7 B8 B9; do $CPU run-exploratory --analysis $b --run-id $RUN_ID; done
$CPU build-comparison-tables --run-id $RUN_ID
$CPU build-manuscript-tables --run-id $RUN_ID
$CPU build-manuscript-figures --run-id $RUN_ID
$CPU write-release-manifest --run-id $RUN_ID
```

## 8. Compare with the published results

Copy the run root out of the `artifacts` volume and compare it with [results/v6.0.1](../results/v6.0.1/):

```bash
docker compose run --rm -T -v "$PWD/outputs:/export" tmmri cp -r artifacts/$RUN_ID /export/
python tools/compare_results.py outputs/$RUN_ID
```

What to expect:

- **Deterministic stages** (the descriptor pipeline, transition, rules, metrics, trust analyses and exploratory
  analyses) run in IEEE FP32 or Float64 with fixed seeds and reproduced the earlier run of the same code byte for
  byte (242 of 252 outputs; the other 10 differed only in run metadata). Their values should match exactly.
- **GPU stages** (D20 features, F9 and everything downstream of them) depend on the GPU model, the driver and cuDNN.
  The protocol's cross-hardware tolerance is 10⁻³ on the validation cross-entropy, and `compare_results.py` uses a
  relative tolerance of 10⁻³ for these outputs (`--rtol-neural`). A different GPU can change a selected F9
  candidate and therefore the F9 numbers more than that; such differences are expected and are not failures of the
  descriptor audit.
- **Provenance fields** (run ids, times, hosts, paths, commits and input digests) always differ and are not counted.
  The configs that are public editions also hash differently from the registered ones, which changes only those
  digests.

## 9. Use the frozen bundles

[bundles/v6.0.1](../bundles/v6.0.1/) holds the objects fitted on the ACDC development set in the original run: the
audited predictor and its fold models, the transition T, the WEDD cut points, the rulebooks (the primary one is
empty), the S1–S4 trust references, the F8 tree and the fitted maps of D20 and F9b. The three final F9 networks are
attached to the GitHub release (`f9_final_seed*.pt`; digests in
[bundles/v6.0.1/neural_f9/final_weights.json](../bundles/v6.0.1/neural_f9/final_weights.json)). Files whose content
names a case are not included. The bundles are the frozen run's own files, laid out as in its run root; the pipeline
reads them from a run root. They are licensed CC BY-NC-SA 4.0, because they derive from ACDC. Some are
scikit-learn objects stored with `joblib` (pickle): load them only after checking their digests against
`EXPORT_MANIFEST.json`.
