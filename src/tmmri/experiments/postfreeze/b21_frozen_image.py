"""B21 of the post-freeze exploratory block (protocol-v6.0.2; ``configs/exploratory_postfreeze.yaml`` ``B21``;
SPECIFICATION.md 13.7, Decisions D37, D39 and D41): the phase-preserving (D20-P) and mask-pooled (D20-M) representations
of the frozen D20 encoder through the D20 downstream.  Label ``EXPLORATORY_POST_HOC``; estimation only, no test.

Encoder and image input: exactly D20 (:mod:`tmmri.deep_encoder`; ``configs/deep_encoder.yaml``): the verified frozen
ResNet-50 ``IMAGENET1K_V2``, the staged ED and ES images in the canonical orientation, per-slice 1st/99th percentile
normalization, bilinear antialiased resize to 224 x 224, grey replicated to three channels, ImageNet normalization and
the eight in-plane D4 transforms of the resized input (:mod:`tmmri.neural.dihedral`).  The layer-4 map is computed once
per slice and transform; its global average pool (the encoder's own ``avgpool``) is exactly the D20 slice vector.

* **D20-P** (4,096-d): the Float64 mean of the FP32 2048-d slice vectors over the slices of one phase and the D4
  transforms (element order 0..7, then slice order, as :func:`tmmri.deep_encoder.patient_vector`); ``[ED, ES]``.
* **D20-M** (12,288-d): the layer-4 map (2048 x 7 x 7) upsampled bilinearly to the input size
  (``torch.nn.functional.interpolate``, ``align_corners=False``); the reference mask (staging label map: 1 LV cavity,
  2 myocardium, 3 RV cavity) oriented like the image, resized to the same size by nearest neighbour with pixel-centre
  alignment (``InterpolationMode.NEAREST_EXACT``, the nearest-neighbour counterpart of the image's bilinear geometry)
  and D4-transformed with the image; per transform, phase and structure the feature mean over all structure pixels of
  all slices, then the Float64 mean over the transforms.  The interpolation is linear, so each masked sum is computed
  exactly as the FP32 map times the Float64 interpolation weights of the mask (``interpolate`` of the 49 unit maps,
  summed over the structure pixels) instead of materializing the 2048 x 224 x 224 map (:data:`UPSAMPLE_RULE`).  Layout:
  phase-major ``[ED-LV, ED-MYO, ED-RV, ES-LV, ES-MYO, ES-RV]``, 2048 channels each.  A structure absent from a phase
  (no pixel on the resized grid of any slice) gives NE (``INPUT_STRUCTURE_ABSENT``): the case has no D20-M vector and
  is excluded and counted.

Downstream (:func:`fit_representation`): the D20 downstream of :func:`tmmri.deep_encoder.fit_d20_models` with the
representation's own columns: zero-variance removal and fit-row standardization inside the typed transition of
``configs/transition.yaml`` (per-target inner-CV lambda over the recorded v6 inner folds, nested out-of-fold B-hat over
the recorded outer folds, final T with lambda by CV over the outer folds) and the audited-predictor settings of
``configs/formal.yaml`` (fold predictors for the out-of-fold development rows, a final predictor on all development
fit rows); critical-QC rows are excluded from every fit; no PCA.

Order (lane ``policy.freeze_before_evaluation``): ``acdc_dev`` features, fit, write and hash both bundles
(``<rep>/FROZEN.json``); then per evaluated partition: extract once, re-verify the bundle hashes, evaluate once.

Evaluation (shared with B22; lane ``statistics``): rho, rho_j (Equation E1, primary family, acdc_dev fit-row mean null
and SD, at least five estimable targets), rho_EF (both EF targets), rho of the five volume targets, each with the paired
cluster bootstrap (2,000 replicates, seed 20260928, unit ``cluster_id``, percentile 95 %) and the paired difference
against the primary ridge of the source run on the same rows and draws; accuracy, macro-F1 and macro OvR AUROC (plus
the other v6 predictor metrics) on the same-definition class reference; chance (1/5) and majority-class rates with
Wilson intervals.  Critical-QC rows are excluded from every metric and counted (lane ``policy.qc``); fewer than 20
cases give point estimates with NE intervals (``STRATUM_BELOW_MIN_CASES``).

Every source-run file is read through ``PostfreezeContext.verified`` (hash check); staged images and masks are read
with the staging hash check of ``case.json``.  Outputs: ``postfreeze/B21/`` of the new run root.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ANALYSIS = "B21"
LABEL = "EXPLORATORY_POST_HOC"
SCHEMA = "tmmri.postfreeze_b21/1"
DATASETS = {"acdc_dev": "ACDC", "acdc_test": "ACDC", "mms_external": "MMS", "mms2_external": "MMS2"}
PHASES = ("ed", "es")
STRUCTURES = ((1, "lv"), (2, "myo"), (3, "rv"))       # v6 label convention after the staging label map (ACDC 1 <-> 3)
STRUCTURE_ABSENT = "INPUT_STRUCTURE_ABSENT"
STRATUM_BELOW_MIN_CASES = "STRATUM_BELOW_MIN_CASES"
FEATURE_DIM = 2048
REPRESENTATIONS = {
    "d20p": {"id": "d20p_resnet50_imagenet1k_v2_phase_4096", "estimator": "d20p_transition_and_predictor", "dimension": 2 * FEATURE_DIM,
             "layout": "[ED, ES] x 2048 (global-average-pooled layer-4 vector; mean over the slices of the phase and the 8 D4 transforms)"},
    "d20m": {"id": "d20m_resnet50_imagenet1k_v2_mask_pooled_12288", "estimator": "d20m_transition_and_predictor",
             "dimension": len(PHASES) * len(STRUCTURES) * FEATURE_DIM,
             "layout": "[ED-LV, ED-MYO, ED-RV, ES-LV, ES-MYO, ES-RV] x 2048 (layer-4 map upsampled bilinearly; mean over the structure "
                       "pixels of all slices of the phase; mean over the 8 D4 transforms)"},
}
MASK_RESIZE_RULE = ("torchvision.transforms.v2.functional.resize(size=[224, 224], interpolation=NEAREST_EXACT) of the label slices "
                    "oriented with the image (pixel-centre alignment of the image's bilinear resize); then the image's D4 element")
UPSAMPLE_RULE = ("torch.nn.functional.interpolate(layer4 map, size=[224, 224], mode='bilinear', align_corners=False); the masked sums "
                 "are computed by linearity as map x (interpolate of the 49 unit maps . mask) in Float64, without materializing the "
                 "2048 x 224 x 224 map")


# ============================================================================= source run (read-only, hash-checked)


def source_stage(rel: str) -> str:
    """The source-run stage that recorded ``rel`` (the ``stage`` argument of ``PostfreezeContext.verified``)."""

    rel = str(rel).replace("\\", "/")
    parts = rel.split("/")
    stem = Path(rel).stem
    if rel.startswith("semantics/B_"):
        return f"build-semantics:{stem[2:]}"
    if rel.startswith("contracts/"):
        return "validate-cohort:MMS2" if "MMS2" in stem else "validate-cohort"
    if rel.startswith("oof/") or rel.startswith("transitions/"):
        return "fit-transition"
    if rel.startswith("reconstructed/bhat_"):
        return f"evaluate:{stem[len('bhat_'):]}"
    if rel.startswith("deep_encoder/features/"):
        return f"extract-deep-features:{_partition_suffix(stem, 'A_D20_')}"
    if rel.startswith("deep_encoder/"):
        return "fit-deep-encoder"
    if rel.startswith("neural_cache/"):
        return f"build-tensor-cache:{parts[1]}"
    if rel.startswith("neural_f9/selection/") or rel.startswith("neural_f9/plan/wave2"):
        return "select-f9"
    if rel.startswith("neural_f9/plan/"):
        return "plan-f9"
    if rel.startswith("neural_f9/checkpoints/"):
        return "import-f9-fits:wave2"
    if rel.startswith("neural_f9/predictions_") or rel.startswith("neural_f9/activations_") or rel.startswith("neural_f9/inference_"):
        return f"infer-f9:{_partition_suffix(stem, stem.split('_', 1)[0] + '_')}"
    if rel.startswith("neural_f9/"):
        return "freeze-f9"
    if rel.startswith("probes/f9b/"):
        return "fit-probes"
    return "release_manifest"


def _partition_suffix(stem: str, prefix: str) -> str:
    rest = stem[len(prefix):] if stem.startswith(prefix) else stem
    for partition in DATASETS:
        if rest == partition or rest.startswith(partition + "_"):
            return partition
    return rest


def source_path(pctx: Any, rel: str) -> Path:
    """Hash-checked path of a source-run file (``PostfreezeContext.verified``; a mismatch refuses the analysis)."""

    return Path(pctx.verified(str(rel), source_stage(rel)))


def source_table(pctx: Any, rel: str) -> pd.DataFrame:
    frame = pd.read_parquet(source_path(pctx, rel))
    if "sample_id" in frame.columns:
        frame["sample_id"] = frame["sample_id"].astype(str)
    return frame


def source_json(pctx: Any, rel: str) -> Any:
    return json.loads(source_path(pctx, rel).read_text(encoding="utf-8"))


def staged_root(pctx: Any) -> Path:
    """The staging root below ``staged_data_root`` (``/workspace/data/staged``; the data root itself when it is one)."""

    root = Path(pctx.staged_data_root)
    return root / "staged" if (root / "staged").is_dir() else root


def lane_partitions(pctx: Any) -> tuple[str, list[str]]:
    """(fit partition, evaluated partitions) of the lane ``policy``; a partition of ``policy.never`` refuses."""

    policy = pctx.lane_cfg["policy"]
    fit = str(policy["fit_partition"])
    evaluated = [str(p) for p in policy["evaluated_partitions"]]
    never = {str(p) for p in policy.get("never", [])}
    bad = sorted(({fit} | set(evaluated)) & never | ({fit} | set(evaluated)) - set(DATASETS))
    if bad or fit != "acdc_dev":
        raise ValueError(f"lane policy names partitions this analysis cannot process: {bad or fit}")
    return fit, evaluated


def require_mms2_lock(pctx: Any) -> dict[str, Any]:
    """Lane ``policy.locks``: M&Ms-2 only through its passed ``validate-cohort --dataset mms2`` record."""

    record = source_json(pctx, "contracts/cohort_validation_MMS2.json")
    if record.get("passed") is not True:
        raise PermissionError("mms2_external refused: contracts/cohort_validation_MMS2.json of the source run has not passed")
    return {"record": "contracts/cohort_validation_MMS2.json", "passed": True}


@dataclass
class PartitionInputs:
    """One partition of the source run: split rows, typed B, class reference, critical QC and bootstrap clusters."""

    partition: str
    dataset: str
    rows: pd.DataFrame                  # split-manifest rows (sample_id, dataset, case_id, outer_fold, cluster_id), sorted
    b: pd.DataFrame                     # typed B of the partition rows present in B, index sample_id
    truth: pd.Series                    # ACDC class; M&Ms same-definition crosswalk class, else None
    critical: pd.Series                 # critical QC (bool) by sample_id
    clusters: pd.Series                 # cluster_id by sample_id

    @property
    def ids(self) -> list[str]:
        return list(self.b.index)

    @property
    def eval_ids(self) -> list[str]:
        """Rows of every fit and metric: in B and without a critical QC reason (lane ``policy.qc``)."""

        return [s for s in self.b.index if not bool(self.critical[s])]


def load_partition(pctx: Any, partition: str) -> PartitionInputs:
    from ... import stages

    if partition not in DATASETS:
        raise ValueError(f"unknown partition {partition!r}")
    ds = DATASETS[partition]
    if ds == "MMS2":
        require_mms2_lock(pctx)
    split = source_table(pctx, "contracts/split_manifest_MMS2.parquet" if ds == "MMS2" else "contracts/split_manifest.parquet")
    rows = split[split["partition"].astype(str) == partition].copy()
    if rows.empty:
        raise ValueError(f"no rows of partition {partition!r} in the source split manifest")
    rows["cluster_id"] = stages._cluster_ids(split, rows["sample_id"]).to_numpy(dtype=object)
    rows = rows.sort_values("sample_id").reset_index(drop=True)
    b = source_table(pctx, f"semantics/B_{ds}.parquet").set_index("sample_id")
    ids = [s for s in rows["sample_id"] if s in b.index]
    cohort = source_table(pctx, f"contracts/cohort_{ds}.parquet").set_index("sample_id")
    truth = stages.reference_labels(cohort, "ACDC" if ds == "ACDC" else "MMS").reindex(ids).astype(object)
    truth = truth.where(truth.notna(), None)
    critical = stages.critical_qc_mask(b.loc[ids], pctx.study).astype(bool)
    clusters = rows.set_index("sample_id")["cluster_id"].astype(object)
    return PartitionInputs(partition, ds, rows, b.loc[ids], truth, critical, clusters)


def dev_reference(pctx: Any, dev: PartitionInputs) -> tuple[pd.Series, pd.Series]:
    """(SD with ddof 1, mean) of the transition targets on the acdc_dev rows without critical QC (as ``stages._dev_reference``)."""

    targets = pctx.study.transition_targets()
    b = dev.b.loc[dev.eval_ids, targets]
    return b.std(ddof=1), b.mean()


def inner_fold_fn(pctx: Any, ids: pd.DataFrame) -> Callable[..., np.ndarray]:
    """``fn(outer_fold, train_positions=None)``: the recorded v6 inner folds (as ``stages._inner_fold_fn``)."""

    inner = source_table(pctx, "contracts/inner_folds.parquet")

    def fn(outer_fold: int, train_positions: np.ndarray | None = None) -> np.ndarray:
        rows = inner[inner["outer_fold_heldout"] == int(outer_fold)].set_index("sample_id")["inner_fold"]
        if train_positions is None:
            train = ids.loc[ids["outer_fold"] != int(outer_fold), "sample_id"]
        else:
            train = ids["sample_id"].iloc[np.asarray(train_positions, dtype=int)]
        return rows.loc[train.astype(str)].to_numpy(dtype=int)

    return fn


def class_order(pctx: Any) -> list[str]:
    return [str(c) for c in pctx.study.formal["audited_predictor"]["classes"]]


def statistics_settings(pctx: Any) -> dict[str, Any]:
    """Lane ``statistics`` (bootstrap, minimum stratum, target families) plus the v6 settings they leave implicit."""

    stats = pctx.lane_cfg["statistics"]
    boot = stats["bootstrap"]
    study_boot = pctx.study.bootstrap
    if str(boot.get("unit", "cluster_id")) != "cluster_id" or str(boot.get("interval", "percentile")) != "percentile":
        raise ValueError("lane statistics.bootstrap must resample cluster_id with percentile intervals")
    return {"n_boot": int(boot["replicates"]), "seed": int(boot["seed"]), "level": float(boot["level"]),
            "min_valid_fraction": float(study_boot["fixed_model_bootstrap"]["min_valid_fraction"]),
            "n_bins": int(study_boot["calibration"]["ece_bins"]),
            "min_estimable": int(study_boot["primary_test"]["min_estimable_targets"]),
            "min_cases": int(stats["min_cases_per_stratum"]),
            "ef_targets": [str(t) for t in stats["ef_targets"]], "volume_targets": [str(t) for t in stats["volume_targets"]]}


# ============================================================================= evaluation (shared with B22)


def _short_cluster(rows: Sequence[str]) -> np.ndarray:
    return np.array([s.split(":", 1)[1] if ":" in s else s for s in rows], dtype=object)


def _below_min_cases(out: dict[str, Any], n: int, min_cases: int) -> dict[str, Any]:
    out["min_cases_per_stratum"] = int(min_cases)
    if n < int(min_cases):
        out.update({"ci_lower": None, "ci_upper": None, "rho_j_ci": None, "status": "NE", "reason": STRATUM_BELOW_MIN_CASES})
    return out


def rho_report(pctx: Any, pin: PartitionInputs, bhat: pd.DataFrame, rows: Iterable[str], mean: pd.Series, sd: pd.Series, *,
               reference: pd.DataFrame | None = None) -> dict[str, Any]:
    """rho and rho_j (primary family), rho_EF, rho of the volume targets and, with ``reference`` (the primary ridge's
    B-hat of the source run), the paired difference comparator minus ridge on the same rows and draws."""

    from ...confirmatory_h4 import rho_ef
    from ...metrics import primary_rho
    from ..alt_bundles import paired_rho_difference

    st = statistics_settings(pctx)
    targets = pctx.study.transition_targets()
    family = pctx.study.primary_family()
    rows = [s for s in rows if s in bhat.index and s in pin.b.index]
    if not rows:
        return {"rho": None, "status": "NE", "reason": "no_rows_with_reconstructions", "n_cases": 0}
    y_true, y_pred = pin.b.loc[rows, targets], bhat.loc[rows, targets].astype(float)
    null = pd.DataFrame([mean[targets]] * len(rows), index=rows)
    cluster, resample = _short_cluster(rows), pin.clusters.reindex(rows).to_numpy(dtype=object)
    boot = dict(n_boot=st["n_boot"], seed=st["seed"], level=st["level"], min_valid_fraction=st["min_valid_fraction"])
    n_clusters = int(pd.Series(resample).nunique())
    out = {"n_cases": len(rows), "n_clusters": n_clusters}
    out["rho"] = _below_min_cases(primary_rho(y_true, y_pred, null, sd, family, cluster=cluster, min_estimable=st["min_estimable"],
                                              resample_cluster=resample, **boot), len(rows), st["min_cases"])
    out["rho_ef"] = _below_min_cases(rho_ef(y_true, y_pred, mean, sd, st["ef_targets"], family=family, cluster=cluster, resample_cluster=resample,
                                            **boot), len(rows), st["min_cases"])
    out["rho_volumes"] = _below_min_cases(primary_rho(y_true, y_pred, null, sd, st["volume_targets"], cluster=cluster,
                                                      min_estimable=len(st["volume_targets"]), resample_cluster=resample, **boot),
                                          len(rows), st["min_cases"])
    if reference is not None:
        paired = [s for s in rows if s in reference.index]
        ref = reference.loc[paired, family].astype(float)
        out["paired_vs_primary_ridge"] = {
            "n_cases": len(paired), "definition": "rho(comparator) - rho(primary ridge of the source run), same rows and cluster draws",
            **paired_rho_difference(y_true.loc[paired, family], ref, y_pred.loc[paired, family], null.loc[paired, family], sd, family,
                                    cluster=_short_cluster(paired), resample_cluster=pin.clusters.reindex(paired).to_numpy(dtype=object),
                                    min_estimable=st["min_estimable"], **boot)}
    return out


def predictor_report(pctx: Any, pin: PartitionInputs, proba: pd.DataFrame, ids: Iterable[str]) -> dict[str, Any]:
    """v6 predictor metrics (accuracy, macro-F1, macro OvR AUROC, ...) on the same-definition reference, plus the chance
    rate and the majority-class rate (class named; Wilson interval) of the same evaluable rows."""

    from ...metrics import wilson_interval
    from ..alt_bundles import evaluable_rows, predictor_metrics

    st = statistics_settings(pctx)
    classes = class_order(pctx)
    ids = [s for s in ids if s in proba.index]
    if not ids:
        return {"status": "NE", "reason": "no_rows_with_predictions", "n_rows": 0}
    truth = pin.truth.reindex(ids).to_numpy(dtype=object)
    frame = proba.reindex(ids)[classes]
    out = predictor_metrics(truth, frame, classes, pin.clusters.reindex(ids).to_numpy(dtype=object), n_boot=st["n_boot"], seed=st["seed"],
                            n_bins=st["n_bins"], min_valid_fraction=st["min_valid_fraction"], level=st["level"])
    keep = evaluable_rows(truth, frame, classes)
    counts = {c: int(sum(str(t) == c for t in truth[keep])) for c in classes}
    n = int(keep.sum())
    rates: dict[str, Any] = {"chance": 1.0 / len(classes), "n_evaluable": n, "class_counts": counts}
    if n:
        top = max(classes, key=lambda c: (counts[c], -classes.index(c)))       # ties: first in class order
        lo, hi = wilson_interval(counts[top], n, st["level"])
        rates.update({"majority_class": top, "majority_rate": counts[top] / n, "majority_wilson": [lo, hi], "majority_numerator": counts[top]})
    out["chance_and_majority_rates"] = rates
    out["n_rows"] = len(ids)
    return out


def partition_counts(pin: PartitionInputs, rows_with_features: Iterable[str]) -> dict[str, int]:
    have = set(rows_with_features)
    return {"n_partition_rows": int(len(pin.rows)), "n_rows_in_b": int(len(pin.ids)), "n_excluded_critical_qc": int(pin.critical.sum()),
            "n_eval_rows": int(len(pin.eval_ids)), "n_eval_rows_with_features": int(sum(s in have for s in pin.eval_ids))}


def ridge_reference(pctx: Any, partition: str) -> pd.DataFrame:
    """The primary ridge's B-hat of the source run: out-of-fold on acdc_dev, the frozen final T elsewhere."""

    rel = "oof/bhat_oof_ACDC_dev.parquet" if partition == "acdc_dev" else f"reconstructed/bhat_{partition}.parquet"
    return source_table(pctx, rel).set_index("sample_id")


# ============================================================================= fitting and freezing (shared with B22)


@dataclass
class RepresentationFit:
    representation: str
    columns: list[str]
    model: Any                                   # TransitionModel (final; lambda by CV over the outer folds)
    oof: pd.DataFrame                            # nested out-of-fold B-hat of the transition rows (sample_id column)
    lambda_ledger: pd.DataFrame
    fold_models: dict[int, Any]
    predictor: Any                               # FormalPredictor on all development fit rows
    fold_predictors: dict[int, Any]
    oof_predictions: pd.DataFrame                # sample_id, outer_fold, prob_<C>, logit_<C>, y_true, y_pred (fold predictors)
    transition_ids: pd.DataFrame
    summary: dict[str, Any] = field(default_factory=dict)

    def oof_bhat(self) -> pd.DataFrame:
        return self.oof.set_index("sample_id")

    def oof_proba(self, classes: Sequence[str]) -> pd.DataFrame:
        frame = self.oof_predictions.set_index("sample_id")[[f"prob_{c}" for c in classes]].astype(float)
        frame.columns = list(classes)
        return frame


def fit_representation(features: pd.DataFrame, dev: pd.DataFrame, b: pd.DataFrame, labels: pd.Series, critical_ids: set[str], *,
                       representation: str, targets: list[str], transition_cfg: Mapping[str, Any], registry: Mapping[str, Any],
                       predictor_cfg: Mapping[str, Any], make_inner_fn: Callable[[pd.DataFrame], Callable[..., np.ndarray]],
                       classes: Sequence[str]) -> RepresentationFit:
    """:func:`tmmri.deep_encoder.fit_d20_models` with the representation's own columns (``features``: index sample_id,
    the ``a_*`` columns; acdc_dev rows only)."""

    from functools import partial

    from ...classification import fit_formal_predictor
    from ...deep_encoder import _logits_and_proba
    from ...transition_fit import fit_transition_v6, nested_oof_transition

    cols = [c for c in features.columns if str(c).startswith("a_")]
    feats = features.copy()
    feats.index = feats.index.astype(str)
    dev = dev[["sample_id", "outer_fold", "cluster_id"]].copy()
    dev["sample_id"] = dev["sample_id"].astype(str)
    outside = sorted(set(feats.index) - set(dev["sample_id"]))
    if outside:
        raise ValueError(f"{representation}: fitting received rows outside acdc_dev ({len(outside)})")
    b = b.copy()
    b["sample_id"] = b["sample_id"].astype(str)
    b_fit = b[~b["sample_id"].isin(critical_ids)]
    ids = (dev.merge(pd.DataFrame({"sample_id": feats.index}), on="sample_id").merge(b_fit[["sample_id"]], on="sample_id")
           .sort_values("sample_id").reset_index(drop=True))
    a_rows = feats.loc[ids["sample_id"], cols].to_numpy(dtype=np.float64)
    b_rows = b_fit.set_index("sample_id").loc[ids["sample_id"]].copy().reset_index()
    folds = ids["outer_fold"].to_numpy(dtype=int)
    tcfg, reg = dict(transition_cfg), dict(registry)
    model = fit_transition_v6(a_rows, b_rows, targets, tcfg, reg, folds, representation=representation)
    fitter = partial(fit_transition_v6, representation=representation)
    oof, details = nested_oof_transition(a_rows, b_rows, targets, tcfg, reg, folds, make_inner_fn(ids), return_details=True, fitter=fitter)
    oof = oof.assign(sample_id=ids["sample_id"].to_numpy()) if "sample_id" not in oof else oof
    final = pd.DataFrame([{"outer_fold": pd.NA, "target": t, "lambda": model.lambdas.get(t, np.nan),
                           "lambda_at_grid_edge": bool(model.edge_flags.get(t, False)),
                           "flag": "LAMBDA_AT_GRID_EDGE" if model.edge_flags.get(t, False) else "",
                           "eligible": t in model.fitted_targets, "exclusion_reason": model.excluded.get(t, ""),
                           "n_fit_rows": model.n_fit_rows} for t in targets])
    ledger = pd.concat([details["lambda_selection"].assign(scope="outer_fold"), final.assign(scope="final_bundle")], ignore_index=True)
    ledger["outer_fold"] = ledger["outer_fold"].astype("Int64")

    # Predictor rows: every acdc_dev row with features; critical-QC, non-finite and unlabelled rows are not fitted.
    prows = dev.merge(pd.DataFrame({"sample_id": feats.index}), on="sample_id").sort_values("sample_id").reset_index(drop=True)
    x = feats.loc[prows["sample_id"], cols].to_numpy(dtype=np.float64)
    y = labels.reindex(prows["sample_id"]).astype(object).reset_index(drop=True)
    fit_rows = np.isfinite(x).all(axis=1) & ~prows["sample_id"].isin(critical_ids).to_numpy() & y.notna().to_numpy()
    pcfg = dict(predictor_cfg)
    predictor = fit_formal_predictor(x, y, pcfg, fit_mask=fit_rows)
    pfolds = prows["outer_fold"].to_numpy(dtype=int)
    fold_predictors: dict[int, Any] = {}
    blocks = []
    for fold in sorted(set(pfolds.tolist())):
        held = pfolds == fold
        fold_predictor = fit_formal_predictor(x, y, pcfg, fit_mask=fit_rows & ~held)
        fold_predictors[int(fold)] = fold_predictor
        logits, proba = _logits_and_proba(fold_predictor, x[held], classes)
        block = pd.DataFrame({"sample_id": prows.loc[held, "sample_id"].to_numpy(), "outer_fold": pfolds[held]})
        for j, c in enumerate(classes):
            block[f"prob_{c}"] = proba[:, j]
        for j, c in enumerate(classes):
            block[f"logit_{c}"] = logits[:, j]
        block["y_true"] = y[held].to_numpy()
        finite = np.isfinite(proba).all(axis=1)
        block["y_pred"] = [classes[int(np.argmax(p))] if ok else None for p, ok in zip(proba, finite)]
        blocks.append(block)
    oof_predictions = pd.concat(blocks, ignore_index=True).sort_values("sample_id").reset_index(drop=True)
    summary = {"transition_rows": int(len(ids)), "predictor_rows": int(len(prows)), "predictor_fit_rows": int(fit_rows.sum()),
               "n_columns": len(cols), "fitted_targets": list(model.fitted_targets), "excluded_targets": dict(model.excluded),
               "missing_classes": list(predictor.missing_classes)}
    return RepresentationFit(representation=representation, columns=cols, model=model, oof=oof, lambda_ledger=ledger,
                             fold_models=dict(details["models"]), predictor=predictor, fold_predictors=fold_predictors,
                             oof_predictions=oof_predictions, transition_ids=ids, summary=summary)


def write_table(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(tmp, engine="pyarrow", index=False)
    tmp.replace(path)
    return path


def write_fit(fit: RepresentationFit, directory: Path) -> list[Path]:
    """Every fitted object of ``fit`` below ``directory`` (the D20 bundle layout)."""

    import joblib

    from ...runs import atomic_write_json

    d = Path(directory)
    paths = [atomic_write_json(d / "transitions" / "transition_model.json", fit.model.to_dict(), allow_nan=True),
             atomic_write_json(d / "transitions" / "lambda_selection.json", {"lambdas": dict(fit.model.lambdas), "edge_flags": dict(fit.model.edge_flags)},
                               allow_nan=True),
             write_table(fit.lambda_ledger, d / "transitions" / "lambda_selection.parquet"),
             write_table(fit.oof, d / "oof" / "bhat_oof_ACDC_dev.parquet")]
    for k in sorted(fit.fold_models):
        paths.append(atomic_write_json(d / "transitions" / "fold_models" / f"transition_fold{k}.json", fit.fold_models[k].to_dict(), allow_nan=True))
    for k in sorted(fit.fold_predictors):
        paths.append(atomic_write_json(d / "formal" / "fold_predictors" / f"predictor_fold{k}.json", fit.fold_predictors[k].to_dict(), allow_nan=True))
    (d / "formal").mkdir(parents=True, exist_ok=True)
    tmp = d / "formal" / ".predictor.joblib.tmp"
    joblib.dump(fit.predictor, tmp)
    tmp.replace(d / "formal" / "predictor.joblib")
    paths += [d / "formal" / "predictor.joblib",
              atomic_write_json(d / "formal" / "predictor.json", fit.predictor.to_dict(), allow_nan=True),
              write_table(fit.oof_predictions, d / "formal" / "oof_predictions_ACDC_dev.parquet"),
              write_table(fit.transition_ids, d / "transitions" / "transition_rows_ACDC_dev.parquet")]
    return paths


def freeze(directory: Path, files: Iterable[Path], provenance: Mapping[str, Any], *, schema: str, name: str = "FROZEN.json") -> dict[str, Any]:
    """``<directory>/<name>``: SHA-256 of every file (relative to ``directory``), the provenance and ``bundle_sha256``."""

    from ...runs import atomic_write_json, sha256_file, sha256_json

    d = Path(directory)
    hashes = {Path(p).relative_to(d).as_posix(): sha256_file(p) for p in sorted(Path(p) for p in files)}
    payload = {"schema": schema, "label": LABEL, "files": hashes, "provenance": dict(provenance)}
    frozen = {**payload, "bundle_sha256": sha256_json(payload)}
    atomic_write_json(d / name, frozen)
    return frozen


def verify_frozen(directory: Path, name: str = "FROZEN.json") -> dict[str, Any]:
    """Re-hash a freeze record before an evaluation; any change refuses."""

    from ...runs import sha256_file, sha256_json

    d = Path(directory)
    frozen = json.loads((d / name).read_text(encoding="utf-8"))
    payload = {k: v for k, v in frozen.items() if k != "bundle_sha256"}
    if sha256_json(payload) != frozen.get("bundle_sha256"):
        raise RuntimeError(f"{d / name}: content does not match its bundle_sha256")
    changed = [rel for rel, sha in frozen["files"].items() if not (d / rel).is_file() or sha256_file(d / rel) != sha]
    if changed:
        raise RuntimeError(f"{d / name}: files changed after the freeze: {changed}")
    return frozen


# ============================================================================= D20-P and D20-M extraction


def mask_input_labels(case: Any, masks: Mapping[str, np.ndarray], *, size: Sequence[int]) -> Any:
    """``[n_ed + n_es, H, W]`` uint8 labels (ED slices first): the reference masks oriented like the D20 image input and
    resized to ``size`` by nearest neighbour with pixel-centre alignment (:data:`MASK_RESIZE_RULE`)."""

    import torch
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.v2 import functional as TF

    from ...neural.preprocess import InputRejected, canonical_orientation, oriented_slices

    orientation = canonical_orientation(case.ed_geometry)
    stacks = []
    for phase in PHASES:
        image = np.asarray(getattr(case, f"{phase}_image"))
        mask = np.asarray(masks[phase])
        if mask.shape != image.shape:
            raise InputRejected("INPUT_DECODE_FAILURE", f"{phase}: mask shape {mask.shape} differs from the image {image.shape}")
        labels = np.clip(np.rint(oriented_slices(mask, orientation)), 0, 255).astype(np.uint8)
        tensor = torch.from_numpy(np.ascontiguousarray(labels)).unsqueeze(1)
        stacks.append(TF.resize(tensor, [int(size[0]), int(size[1])], interpolation=InterpolationMode.NEAREST_EXACT, antialias=False)[:, 0])
    return torch.cat(stacks, dim=0).contiguous()


def layer4_maps(encoder: Any, x: Any) -> Any:
    """The layer-4 map of a torchvision ResNet (the stages of its ``forward`` up to ``avgpool``)."""

    y = encoder.maxpool(encoder.relu(encoder.bn1(encoder.conv1(x))))
    return encoder.layer4(encoder.layer3(encoder.layer2(encoder.layer1(y))))


def pooled_vectors(encoder: Any, maps: Any) -> Any:
    """The encoder's own global average pool of layer-4 maps: exactly ``encoder(x)`` with ``fc`` = identity."""

    import torch

    return torch.flatten(encoder.avgpool(maps), 1)


def case_vectors(encoder: Any, case: Any, masks: Mapping[str, np.ndarray] | None, *, device: Any, batch_size: int = 32, tta: str = "d4_mean_8",
                 input_kwargs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """D20-P (always) and D20-M (when ``masks`` is given and every structure is present) of one case.

    Returns ``{"d20p": [4096] float64, "d20m": [12288] float64 | None, "absent": [(phase, structure)], "n_ed", "n_es",
    "d20": [2048] float64}``; ``d20`` is the D20 patient vector of the same pass (all slices of both phases)."""

    import torch
    import torch.nn.functional as F

    from ...deep_encoder import d20_input_tensor, patient_vector
    from ...neural.dihedral import d4_apply, tta_elements

    elements = tta_elements(tta)
    x, counts = d20_input_tensor(case, **dict(input_kwargs or {}))
    n_ed, n_es = counts["n_ed"], counts["n_es"]
    size = (int(x.shape[-2]), int(x.shape[-1]))
    labels = None if masks is None else mask_input_labels(case, masks, size=size)
    absent: list[tuple[str, str]] = []
    if labels is not None:
        for p, (lo, hi) in zip(PHASES, ((0, n_ed), (n_ed, n_ed + n_es))):
            absent += [(p, name) for k, name in STRUCTURES if not bool((labels[lo:hi] == k).any())]
    want_m = labels is not None and not absent
    phase_of = np.array([0] * n_ed + [1] * n_es)
    pooled: list[np.ndarray] = []                                   # per element: [n, 2048] float32
    sums = np.zeros((len(elements), len(PHASES), len(STRUCTURES), FEATURE_DIM), dtype=np.float64)
    pixels = np.zeros((len(elements), len(PHASES), len(STRUCTURES)), dtype=np.int64)
    basis = None                                                    # [h*w, H*W] Float64: the upsampled unit maps
    with torch.inference_mode():
        for e, g in enumerate(elements):
            xg = d4_apply(x, g)
            lg = None if not want_m else d4_apply(labels, g)
            rows = []
            for start in range(0, int(xg.shape[0]), int(batch_size)):
                maps = layer4_maps(encoder, xg[start:start + int(batch_size)].to(device))
                rows.append(pooled_vectors(encoder, maps).to(dtype=torch.float32).cpu().numpy())
                if not want_m:
                    continue
                b, c, h, w = (int(v) for v in maps.shape)
                if basis is None:
                    unit = torch.eye(h * w, dtype=torch.float64, device=maps.device).reshape(h * w, 1, h, w)
                    basis = F.interpolate(unit, size=size, mode="bilinear", align_corners=False).reshape(h * w, -1)
                onehot = torch.stack([(lg[start:start + b] == k) for k, _ in STRUCTURES], dim=1).reshape(b, len(STRUCTURES), -1)
                onehot = onehot.to(device=maps.device, dtype=torch.float64)
                # masked sum of the upsampled map = map x (upsampled unit maps . mask), by linearity of the interpolation
                weights = torch.einsum("qp,bsp->bqs", basis, onehot)                                    # [b, h*w, 3]
                masked = torch.bmm(maps.to(torch.float64).reshape(b, c, h * w), weights).cpu().numpy()  # [b, 2048, 3]
                counts_b = onehot.sum(dim=2).to(torch.int64).cpu().numpy()                               # [b, 3]
                for i in range(b):
                    sums[e, phase_of[start + i]] += masked[i].T
                    pixels[e, phase_of[start + i]] += counts_b[i]
            pooled.append(np.concatenate(rows, axis=0))
    stacked = np.stack(pooled)                                      # [elements, n, 2048]
    d20p = np.concatenate([patient_vector(stacked[:, :n_ed].reshape(-1, FEATURE_DIM)),
                           patient_vector(stacked[:, n_ed:].reshape(-1, FEATURE_DIM))])
    out: dict[str, Any] = {"d20p": d20p, "d20m": None, "absent": absent, "n_ed": n_ed, "n_es": n_es,
                           "d20": patient_vector(stacked.reshape(-1, FEATURE_DIM))}
    if want_m:
        if not (pixels == pixels[:1]).all():
            raise RuntimeError("structure pixel counts differ between D4 transforms (the transforms must permute pixels)")
        means = sums / pixels[..., None].astype(np.float64)              # [elements, phase, structure, 2048]
        out["d20m"] = (means.sum(axis=0) / float(len(elements))).reshape(-1)
    return out


def feature_frame(rows: list[dict[str, Any]], dimension: int) -> pd.DataFrame:
    from ...neural.prediction_table import activation_columns

    keys = ["sample_id", "cluster_id", "partition", "dataset", "case_id", "n_slices_ed", "n_slices_es"]
    columns = activation_columns(dimension)
    records = [{**{k: r[k] for k in keys}, **dict(zip(columns, r["vector"].tolist()))} for r in rows]
    return pd.DataFrame(records, columns=keys + columns).sort_values("sample_id").reset_index(drop=True)


def staged_case_loader(root: Path, row: Mapping[str, Any], ni_cfg: Mapping[str, Any]) -> tuple[Any, dict[str, np.ndarray]]:
    """(CaseVolumes, {"ed": mask, "es": mask}) of a staged case: the D20 image loader and the label-mapped reference
    masks (``tmmri.ingest.staging.load_staged_case``), both after the staging hash check of ``case.json``."""

    from ...ingest.staging import load_staged_case
    from ...neural.preprocess import load_staged_volumes

    case = load_staged_volumes(root, str(row["dataset"]), str(row["case_id"]), sample_id=str(row["sample_id"]),
                               cluster_id=str(row["cluster_id"]), cfg=ni_cfg, verify=True)
    staged = load_staged_case(str(row["dataset"]), str(row["case_id"]), root, verify=True)
    return case, {"ed": np.asarray(staged.ed_mask), "es": np.asarray(staged.es_mask)}


def extract_partition(rows: pd.DataFrame, partition: str, loader: Callable[[Mapping[str, Any]], tuple[Any, dict[str, np.ndarray]]], encoder: Any, *,
                      device: Any, batch_size: int, tta: str, input_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """D20-P and D20-M feature tables, failures (reason codes) and the D20 vectors of the same pass, for ``rows``."""

    from ...neural.preprocess import InputRejected

    p_rows, m_rows, d20_rows, failures = [], [], [], []
    for row in rows.to_dict("records"):
        keys = {"sample_id": str(row["sample_id"]), "cluster_id": str(row["cluster_id"]), "partition": partition,
                "dataset": str(row["dataset"]), "case_id": str(row["case_id"])}
        try:
            case, masks = loader(row)
            res = case_vectors(encoder, case, masks, device=device, batch_size=batch_size, tta=tta, input_kwargs=input_kwargs)
        except InputRejected as exc:
            failures += [{**keys, "representation": rep, "reason_code": exc.reason_code, "detail": exc.detail} for rep in REPRESENTATIONS]
            continue
        base = {**keys, "n_slices_ed": res["n_ed"], "n_slices_es": res["n_es"]}
        if not np.isfinite(res["d20p"]).all():
            failures.append({**keys, "representation": "d20p", "reason_code": "INPUT_NONFINITE", "detail": "non-finite feature vector"})
        else:
            p_rows.append({**base, "vector": res["d20p"]})
            d20_rows.append({**base, "vector": res["d20"]})
        if res["absent"]:
            failures.append({**keys, "representation": "d20m", "reason_code": STRUCTURE_ABSENT,
                             "detail": "; ".join(f"{p.upper()} {s}" for p, s in res["absent"])})
        elif res["d20m"] is None or not np.isfinite(res["d20m"]).all():
            failures.append({**keys, "representation": "d20m", "reason_code": "INPUT_NONFINITE", "detail": "non-finite feature vector"})
        else:
            m_rows.append({**base, "vector": res["d20m"]})
    fail = pd.DataFrame(failures, columns=["sample_id", "cluster_id", "partition", "dataset", "case_id", "representation", "reason_code", "detail"])
    return {"d20p": feature_frame(p_rows, REPRESENTATIONS["d20p"]["dimension"]), "d20m": feature_frame(m_rows, REPRESENTATIONS["d20m"]["dimension"]),
            "d20": feature_frame(d20_rows, FEATURE_DIM), "failures": fail.sort_values(["sample_id", "representation"]).reset_index(drop=True)}


def d20_recombination_check(pctx: Any, partition: str, d20: pd.DataFrame) -> dict[str, Any]:
    """Diagnostic only: the D20 vector of this pass against the recorded ``A_D20_<partition>`` (same encoder and input;
    the difference measures replication of the D20 extraction, never used for anything)."""

    from ...neural.prediction_table import activation_columns

    recorded = source_table(pctx, f"deep_encoder/features/A_D20_{partition}.parquet").set_index("sample_id")
    cols = activation_columns(FEATURE_DIM)
    here = d20.set_index("sample_id")
    common = [s for s in here.index if s in recorded.index]
    if not common:
        return {"n_compared": 0}
    diff = np.abs(here.loc[common, cols].to_numpy(dtype=np.float64) - recorded.loc[common, cols].to_numpy(dtype=np.float64))
    return {"n_compared": len(common), "n_recorded": int(len(recorded)), "n_here": int(len(here)), "max_abs_difference": float(diff.max()),
            "median_case_max_abs_difference": float(np.median(diff.max(axis=1))), "role": "diagnostic; reported, never used"}


# ============================================================================= run


def _write_features(out: Path, partition: str, extracted: Mapping[str, Any]) -> list[Path]:
    paths = [write_table(extracted["d20p"], out / "features" / f"A_D20P_{partition}.parquet"),
             write_table(extracted["d20m"], out / "features" / f"A_D20M_{partition}.parquet"),
             write_table(extracted["failures"], out / "features" / f"failures_{partition}.parquet")]
    return paths


def _failure_counts(failures: pd.DataFrame) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {rep: {} for rep in REPRESENTATIONS}
    for (rep, reason), n in failures.groupby(["representation", "reason_code"]).size().items():
        out[str(rep)][str(reason)] = int(n)
    return out


def run(pctx: Any, *, device: Any = None, encoder: Any = None, loader: Callable[..., Any] | None = None,
        input_size: Sequence[int] | None = None) -> dict[str, Any]:
    """B21 on the source run of ``pctx`` (see the module docstring).  ``encoder``, ``loader`` and ``input_size`` are
    test injections (a random-weight ResNet-50, synthetic cases and a small input); a real run uses none of them."""

    from ...deep_encoder import _logits_and_proba
    from ...neural.dihedral import TTA_NONE
    from ...neural.preprocess import load_neural_input_config
    from ...neural.torch_numerics import configure_torch_numerics, environment_record, resolve_device
    from ...runs import atomic_write_json, code_version

    study = pctx.study
    acfg = dict(pctx.analysis_cfg(ANALYSIS) or {})
    fit_partition, evaluated = lane_partitions(pctx)
    out = Path(pctx.out_dir(ANALYSIS))
    results_path = out / "results.json"
    pctx.require_new_output(results_path)
    de_cfg = dict(study.configs["deep_encoder"])
    ni_cfg = dict(study.configs.get("neural_input") or load_neural_input_config(study.root))
    tta = str(de_cfg.get("tta") or TTA_NONE)
    batch = int(de_cfg["runtime"]["batch_size"])
    size = [int(v) for v in (input_size or de_cfg["input"]["resize"])]
    input_kwargs = {"size": size, "mean": ni_cfg["input"]["channel_mean"], "sd": ni_cfg["input"]["channel_sd"]}
    numerics = configure_torch_numerics()
    dev = resolve_device(device)
    frozen_d20 = source_json(pctx, "deep_encoder/FROZEN_D20.json")
    weights: dict[str, Any] = {"recorded_d20_weights_sha256": frozen_d20.get("provenance", {}).get("weights_sha256")}
    weights_path: Path | None = None
    if encoder is None:
        from ...deep_encoder import build_d20_encoder
        from ...neural.weights import D20_WEIGHTS, require_weights

        weights_path, weights_sha = require_weights(D20_WEIGHTS, study.root)
        if weights_sha != str(de_cfg["encoder"]["weights_sha256"]) or weights_sha != weights["recorded_d20_weights_sha256"]:
            raise ValueError("the D20 weight file differs from configs/deep_encoder.yaml or the source run's FROZEN_D20.json")
        encoder = build_d20_encoder(weights_path)
        weights.update({"file": weights_path.name, "sha256": weights_sha})
    else:
        weights.update({"file": None, "sha256": None, "note": "injected encoder (tests)"})
    encoder = encoder.to(dev).eval()
    root = staged_root(pctx)
    load = loader or (lambda row: staged_case_loader(root, row, ni_cfg))
    targets = study.transition_targets()
    classes = class_order(pctx)

    def extract(pin: PartitionInputs) -> dict[str, Any]:
        rows = pin.rows[pin.rows["sample_id"].isin(pin.eval_ids)]
        res = extract_partition(rows, pin.partition, load, encoder, device=dev, batch_size=batch, tta=tta, input_kwargs=input_kwargs)
        written.extend(_write_features(out, pin.partition, res))
        return res

    written: list[Path] = []
    dev_in = load_partition(pctx, fit_partition)
    sd, mean = dev_reference(pctx, dev_in)
    dev_feats = extract(dev_in)
    critical_ids = set(dev_in.critical.index[dev_in.critical.to_numpy()])
    dev_split = dev_in.rows[["sample_id", "outer_fold", "cluster_id"]]
    b_dev = dev_in.b.reset_index()
    labels = dev_in.truth
    fits: dict[str, RepresentationFit] = {}
    freezes: dict[str, dict[str, Any]] = {}
    code = code_version()
    for rep, spec in REPRESENTATIONS.items():
        feats = dev_feats[rep].set_index("sample_id")
        fit = fit_representation(feats[[c for c in feats.columns if c.startswith("a_")]], dev_split, b_dev, labels, critical_ids,
                                 representation=spec["id"], targets=targets, transition_cfg=study.transition, registry=study.registry,
                                 predictor_cfg=study.formal["audited_predictor"], make_inner_fn=lambda ids: inner_fold_fn(pctx, ids),
                                 classes=classes)
        freezes[rep] = freeze(out / rep, write_fit(fit, out / rep), {
            "analysis": ANALYSIS, "estimator": spec["estimator"], "representation": spec["id"], "fit_partition": fit_partition,
            "dev_features_sha256": _sha(out / "features" / f"A_{rep.upper()}_{fit_partition}.parquet"), "weights": weights,
            "code": {"commit": code.get("commit"), "code_sha256": code.get("code_sha256"), "dirty": code.get("dirty")},
            "config_sections": ["deep_encoder", "neural_input", "transition", "formal.audited_predictor", "registry"]},
            schema="tmmri.postfreeze_b21_bundle/1")
        fits[rep] = fit
    results: dict[str, Any] = {"schema": SCHEMA, "label": LABEL, "analysis": ANALYSIS, "answers": acfg.get("answers"),
                               "estimation_only": True, "classes": classes, "statistics": statistics_settings(pctx),
                               "representations": {rep: {"id": spec["id"], "layout": spec["layout"], "estimator": spec["estimator"],
                                                         "bundle_sha256": freezes[rep]["bundle_sha256"], "fit_summary": fits[rep].summary,
                                                         "partitions": {}} for rep, spec in REPRESENTATIONS.items()},
                               "extraction": {"tta": tta, "input_size": size, "batch_size": batch, "mask_resize": MASK_RESIZE_RULE,
                                              "layer4_upsampling": UPSAMPLE_RULE, "structures": dict((k, n) for k, n in STRUCTURES),
                                              "weights": weights, "numerics": numerics, "environment": environment_record(dev)},
                               "failures": {}, "d20_replication_diagnostic": {}}
    results["failures"][fit_partition] = _failure_counts(dev_feats["failures"])
    results["d20_replication_diagnostic"][fit_partition] = d20_recombination_check(pctx, fit_partition, dev_feats["d20"])
    ridge = ridge_reference(pctx, fit_partition)
    for rep in REPRESENTATIONS:
        fit = fits[rep]
        results["representations"][rep]["partitions"]["acdc_dev_oof"] = {
            "evidence": "development (out-of-fold)", "counts": partition_counts(dev_in, dev_feats[rep]["sample_id"]),
            "representation": rho_report(pctx, dev_in, fit.oof_bhat(), fit.transition_ids["sample_id"].tolist(), mean, sd, reference=ridge),
            "predictor": predictor_report(pctx, dev_in, fit.oof_proba(classes), dev_in.eval_ids)}
    for partition in evaluated:
        pin = load_partition(pctx, partition)
        for rep in REPRESENTATIONS:
            verify_frozen(out / rep)
        feats = extract(pin)
        ridge = ridge_reference(pctx, partition)
        results["failures"][partition] = _failure_counts(feats["failures"])
        results["d20_replication_diagnostic"][partition] = d20_recombination_check(pctx, partition, feats["d20"])
        for rep in REPRESENTATIONS:
            fit = fits[rep]
            frame = feats[rep].set_index("sample_id")
            x = frame[fit.columns].to_numpy(dtype=np.float64)
            index = pd.Index(frame.index, name="sample_id")
            bhat = fit.model.predict(x, index=index)
            proba = pd.DataFrame(_logits_and_proba(fit.predictor, x, classes)[1], index=index, columns=classes)
            written.append(write_table(bhat.reset_index(), out / rep / "evaluated" / f"bhat_{partition}.parquet"))
            written.append(write_table(proba.reset_index(), out / rep / "evaluated" / f"proba_{partition}.parquet"))
            results["representations"][rep]["partitions"][partition] = {
                "evidence": "registered after unblinding" if partition != "mms2_external" else "prospective (M&Ms-2 lock of protocol-v6.0.1)",
                "counts": partition_counts(pin, frame.index),
                "representation": rho_report(pctx, pin, bhat, pin.eval_ids, mean, sd, reference=ridge),
                "predictor": predictor_report(pctx, pin, proba, pin.eval_ids)}
    atomic_write_json(results_path, results, allow_nan=True)
    outputs = sorted({*written, results_path, *[out / rep / "FROZEN.json" for rep in REPRESENTATIONS]})
    rels = sorted({"contracts/split_manifest.parquet", "contracts/inner_folds.parquet", "deep_encoder/FROZEN_D20.json",
                   "oof/bhat_oof_ACDC_dev.parquet",
                   *(["contracts/split_manifest_MMS2.parquet", "contracts/cohort_validation_MMS2.json"] if "mms2_external" in evaluated else []),
                   *[f"contracts/cohort_{DATASETS[p]}.parquet" for p in [fit_partition, *evaluated]],
                   *[f"semantics/B_{DATASETS[p]}.parquet" for p in [fit_partition, *evaluated]],
                   *[f"reconstructed/bhat_{p}.parquet" for p in evaluated],
                   *[f"deep_encoder/features/A_D20_{p}.parquet" for p in [fit_partition, *evaluated]]})
    inputs = {f"source:{rel}": source_path(pctx, rel) for rel in rels}
    inputs.update(dict(getattr(pctx, "verified_log", None) or {}))   # every hash-checked read of the context (same labels)
    if weights.get("file"):
        inputs["weights:resnet50_imagenet1k_v2"] = weights_path
    summary = {"analysis": ANALYSIS, "label": LABEL,
               "rho": {rep: {p: (v["representation"].get("rho") or {}).get("rho") for p, v in results["representations"][rep]["partitions"].items()}
                       for rep in REPRESENTATIONS},
               "macro_f1": {rep: {p: ((v["predictor"].get("metrics") or {}).get("macro_f1") or {}).get("estimate")
                                  for p, v in results["representations"][rep]["partitions"].items()} for rep in REPRESENTATIONS},
               "failures": results["failures"], "bundle_sha256": {rep: freezes[rep]["bundle_sha256"] for rep in REPRESENTATIONS},
               "staged_inputs": "staged images and masks of every processed case, verified against case.json (staging hash check)"}
    pctx.write_provenance(ANALYSIS, inputs, outputs, summary)
    return summary


def _sha(path: Path) -> str:
    from ...runs import sha256_file

    return sha256_file(path)
