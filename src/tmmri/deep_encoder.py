"""Decision D20: the frozen deep-encoder branch (SPECIFICATION.md 8.4; ``configs/deep_encoder.yaml``), executed exactly
as frozen under the protocol-v6.0.1 amendment (docs/MANUSCRIPT_PROTOCOL_V2_2026-10-02.md, D item 9).

Encoder.  torchvision ResNet-50 with ``ResNet50_Weights.IMAGENET1K_V2`` (the file ``resnet50-11ad3fa6.pth``, verified
against ``registry/weights.yaml`` before use; never downloaded at use), frozen, ``fc`` replaced by the identity, so
every slice gives the 2048-d global-average-pooled penultimate vector.  Inputs are image-only: all short-axis slices
of the staged ED and ES frames (no mask), reoriented in plane like the neural tensor cache
(:func:`tmmri.neural.preprocess.canonical_orientation`; lossless, slice order kept), normalized **per slice** to the
1st/99th percentiles of its voxels and clipped to [0, 1] (a constant slice uses scale 1, the formal-A convention of
``configs/formal.yaml``), resized to 224 x 224 (``torchvision.transforms.v2.functional.resize``, bilinear,
``antialias=True``, CPU float32), replicated to three channels and ImageNet-normalized.  *6.0.1 (owner issues O1 and
O3):* every slice is encoded under the eight in-plane D4 transforms of the resized 224 x 224 input
(:mod:`tmmri.neural.dihedral`, ``configs/deep_encoder.yaml`` ``tta: d4_mean_8``), because the canonical orientation
leaves about 40 % of the M&Ms-1 slices vertically mirrored relative to ACDC; the patient vector is the Float64 mean of
the FP32 slice vectors over all slices of both phases and all eight transforms (``tta: none``, the v6.0.0 definition,
averages the untransformed slices only).  Runtime: ``env-gpu``, FP32 with TF32 off, deterministic algorithms, eager
mode, at most 32 slices per forward pass, one patient at a time (:mod:`tmmri.neural.torch_numerics`).

Downstream (``fit-deep-encoder``; acdc_dev only).  The v6 machinery unchanged, with the D20 vectors as an alternative
A: zero-variance removal and fit-partition standardization, the typed masked ridge T with per-target lambda by inner
CV (:func:`tmmri.transition_fit.fit_transition_v6` and :func:`~tmmri.transition_fit.nested_oof_transition` over the v6
outer and inner folds) and the audited-predictor settings (:func:`tmmri.classification.fit_formal_predictor`,
``configs/formal.yaml`` ``audited_predictor``) with out-of-fold predictors per v6 outer fold and a final predictor
on all development rows.  Critical-QC rows are excluded from every fit as in the primary pipeline.  The bundle is then
frozen (``deep_encoder/FROZEN_D20.json``: SHA-256 of every fitted object, the development features, the
configuration digest, the code version and the weight hash).

Layout below ``artifacts/<run_id>/deep_encoder/`` (the primary layout rooted at ``deep_encoder/``)::

    features/A_D20_<partition>.parquet (+ _failures.parquet, _extractor.json)
    transitions/transition_model.json, lambda_selection.json, lambda_selection.parquet, fold_models/transition_fold<k>.json
    oof/bhat_oof_ACDC_dev.parquet
    formal/d_model_predictor.joblib (+ .json), d_model_ACDC_dev.parquet, d_model_crossfit_ACDC_dev.parquet,
    formal/fold_predictors/predictor_fold<k>.json
    predictions.parquet          (contract C4, model_id d20_predictor; acdc_dev out-of-fold and final rows)
    FROZEN_D20.json

:func:`load_d20_bundle`, :func:`d20_predictions` and :func:`d20_reconstructions` evaluate the frozen bundle on another
partition's features (``evaluate-alt-bundles``); they never fit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .neural.prediction_table import (
    CLASS_ORDER,
    activation_columns,
    prediction_frame,
    sort_frame,
)
from .neural.dihedral import D4_TTA, TTA_NONE, d4_apply, tta_elements
from .neural.preprocess import (
    CaseVolumes,
    InputRejected,
    canonical_orientation,
    check_phase_pair,
    oriented_slices,
)

D20_REPRESENTATION = "d20_resnet50_imagenet1k_v2_mean2048"
FEATURE_DIM = 2048
MODEL_ID = "d20_predictor"
FROZEN_NAME = "FROZEN_D20.json"
RESIZE_IMPL = "torchvision.transforms.v2.functional.resize(size=[224, 224], interpolation=BILINEAR, antialias=True) on CPU float32"
PER_SLICE_RULE = "per slice: lo, hi = numpy.percentile(voxels, [1, 99]); x' = clip((x - lo) / (hi - lo), 0, 1); hi <= lo -> scale 1"
ORIENTATION_RULE = ("tmmri.neural.preprocess.canonical_orientation of the ED image geometry, in plane and lossless (transpose/flip; "
                    "columns toward patient left, rows toward posterior), slice order kept")
POOLING_RULE = {D4_TTA: "Float64 mean of the FP32 2048-d slice vectors over all ED and ES slices and the 8 in-plane D4 transforms "
                        "(tmmri.neural.dihedral; applied to the resized 224 x 224 input)",
                TTA_NONE: "Float64 mean of the FP32 2048-d slice vectors over all ED and ES slices (v6.0.0 definition)"}


class D20BundleError(RuntimeError):
    """The frozen D20 bundle is missing, incomplete or changed after its freeze."""


def feature_columns() -> list[str]:
    return activation_columns(FEATURE_DIM)


# ----------------------------------------------------------------------------- preprocessing (image only)


def normalize_slice(values: np.ndarray, percentiles: Sequence[float] = (1.0, 99.0)) -> np.ndarray:
    """Per-slice percentile clip to [0, 1] (Float64; a constant slice uses scale 1)."""

    x = np.asarray(values, dtype=np.float64)
    lo, hi = (float(v) for v in np.percentile(x, [float(percentiles[0]), float(percentiles[1])]))
    scale = hi - lo if hi > lo else 1.0
    return np.clip((x - lo) / scale, 0.0, 1.0)


def d20_phase_slices(case: CaseVolumes) -> dict[str, np.ndarray]:
    """Oriented ``(n, rows, cols)`` slices of both phases (image only; non-finite voxels reject the case)."""

    check_phase_pair(case.ed_geometry, case.es_geometry)
    orientation = canonical_orientation(case.ed_geometry)
    out: dict[str, np.ndarray] = {}
    for phase in ("ed", "es"):
        volume = np.asarray(getattr(case, f"{phase}_image"))
        if volume.ndim != 3 or volume.shape[2] < 1:
            raise InputRejected("INPUT_DECODE_FAILURE", f"{phase}: expected a non-empty 3D volume, got {volume.shape}")
        if not np.isfinite(volume).all():
            raise InputRejected("INPUT_NONFINITE", f"{phase}: {int((~np.isfinite(volume)).sum())} non-finite voxels")
        out[phase] = oriented_slices(volume, orientation)
    return out


def d20_input_tensor(case: CaseVolumes, *, size: Sequence[int] = (224, 224), mean: Sequence[float] = (0.485, 0.456, 0.406),
                     sd: Sequence[float] = (0.229, 0.224, 0.225), percentiles: Sequence[float] = (1.0, 99.0)) -> tuple[Any, dict[str, int]]:
    """``[n_ed + n_es, 3, H, W]`` float32 encoder input (ED slices first, then ES) and the slice counts."""

    import torch
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.v2 import functional as TF

    phases = d20_phase_slices(case)
    stacks = []
    for phase in ("ed", "es"):
        normalized = np.stack([normalize_slice(s, percentiles) for s in phases[phase]]).astype(np.float32)
        tensor = torch.from_numpy(normalized).unsqueeze(1)
        stacks.append(TF.resize(tensor, [int(size[0]), int(size[1])], interpolation=InterpolationMode.BILINEAR, antialias=True))
    x = torch.cat(stacks, dim=0).clamp_(0.0, 1.0).repeat(1, 3, 1, 1)
    m = torch.tensor([float(v) for v in mean], dtype=torch.float32).view(1, 3, 1, 1)
    s = torch.tensor([float(v) for v in sd], dtype=torch.float32).view(1, 3, 1, 1)
    return ((x - m) / s).contiguous(), {"n_ed": int(phases["ed"].shape[0]), "n_es": int(phases["es"].shape[0])}


# ----------------------------------------------------------------------------- encoder


def build_d20_encoder(weights_path: str | Path | None = None, *, model: Any = None) -> Any:
    """Frozen ResNet-50 feature extractor (``fc`` = identity, eval mode, no gradients).

    ``weights_path`` is the verified ``resnet50-11ad3fa6.pth``; ``model`` injects a ready network (tests use a
    random-weight ResNet-50; ``weights_path`` is then ignored)."""

    import torch
    import torchvision

    net = model if model is not None else torchvision.models.resnet50(weights=None)
    if model is None:
        if weights_path is None:
            raise ValueError("build_d20_encoder needs the verified weight file (registry/weights.yaml)")
        state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        net.load_state_dict(state)
    net.fc = torch.nn.Identity()
    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net


def encode_slices(encoder: Any, x: Any, *, device: Any, batch_size: int = 32) -> np.ndarray:
    """``[n, d]`` float32 penultimate vectors, at most ``batch_size`` slices per forward pass."""

    import torch

    out = []
    with torch.inference_mode():
        for start in range(0, int(x.shape[0]), int(batch_size)):
            chunk = x[start:start + int(batch_size)].to(device)
            out.append(encoder(chunk).to(dtype=torch.float32).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, FEATURE_DIM), dtype=np.float32)


def encode_slices_tta(encoder: Any, x: Any, *, device: Any, batch_size: int = 32, tta: str = D4_TTA) -> np.ndarray:
    """``[n_elements * n, d]`` float32 slice vectors of every D4 transform of ``tta`` (element order 0..7, then slice
    order), at most ``batch_size`` slices per forward pass."""

    return np.concatenate([encode_slices(encoder, d4_apply(x, g), device=device, batch_size=batch_size) for g in tta_elements(tta)], axis=0)


def patient_vector(slice_features: np.ndarray) -> np.ndarray:
    """Float64 mean over all rows: the ED and ES slices of every D4 transform (fixed summation order)."""

    f = np.asarray(slice_features, dtype=np.float64)
    return f.sum(axis=0) / float(f.shape[0])


def extract_d20_features(cases: Iterable[CaseVolumes | tuple[str, Callable[[], CaseVolumes]]], encoder: Any, *, device: Any,
                         partition: str, batch_size: int = 32, input_kwargs: Mapping[str, Any] | None = None,
                         dimension: int = FEATURE_DIM, tta: str = D4_TTA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(features, failures): one row per case with ``a_0000`` .. ``a_2047`` (Float64), sorted by sample_id.

    ``cases`` holds :class:`CaseVolumes` or lazy ``(sample_id, loader)`` pairs; an
    :class:`~tmmri.neural.preprocess.InputRejected` from a loader or from preprocessing is recorded as a failure with
    its reason code.  ``tta`` is the D4 symmetrization of the patient vector (:data:`POOLING_RULE`)."""

    tta_elements(tta)                                             # an unknown setting refuses before any case is read
    rows, failures = [], []
    columns = activation_columns(dimension)
    for item in cases:
        sid = item[0] if isinstance(item, tuple) else item.sample_id
        try:
            case = item[1]() if isinstance(item, tuple) else item
            x, counts = d20_input_tensor(case, **dict(input_kwargs or {}))
            vector = patient_vector(encode_slices_tta(encoder, x, device=device, batch_size=batch_size, tta=tta))
        except InputRejected as exc:
            failures.append({"sample_id": str(sid), "partition": partition, "reason_code": exc.reason_code, "detail": exc.detail})
            continue
        if vector.shape != (dimension,) or not np.isfinite(vector).all():
            failures.append({"sample_id": case.sample_id, "partition": partition, "reason_code": "INPUT_NONFINITE",
                             "detail": "non-finite or mis-sized feature vector"})
            continue
        rows.append({"sample_id": str(case.sample_id), "cluster_id": str(case.cluster_id or case.sample_id), "partition": partition,
                     "dataset": case.dataset, "case_id": case.case_id, "n_slices_ed": counts["n_ed"], "n_slices_es": counts["n_es"],
                     **dict(zip(columns, vector.tolist()))})
    keys = ["sample_id", "cluster_id", "partition", "dataset", "case_id", "n_slices_ed", "n_slices_es"]
    features = pd.DataFrame(rows, columns=keys + columns).sort_values("sample_id").reset_index(drop=True)
    failed = pd.DataFrame(failures, columns=["sample_id", "partition", "reason_code", "detail"])
    return features, failed.sort_values("sample_id", na_position="last").reset_index(drop=True)


# ----------------------------------------------------------------------------- fitting (acdc_dev only)


@dataclass
class D20Fit:
    model: Any                                   # TransitionModel (final; lambda by CV over the outer folds)
    oof: pd.DataFrame                            # nested out-of-fold B-hat on the transition rows
    lambda_ledger: pd.DataFrame
    fold_models: dict[int, Any]
    predictor: Any                               # FormalPredictor on all development fit rows
    fold_predictors: dict[int, Any]
    d_model: pd.DataFrame
    crossfit: pd.DataFrame
    predictions: pd.DataFrame                    # C4 (acdc_dev)
    transition_ids: pd.DataFrame
    summary: dict[str, Any] = field(default_factory=dict)


def _logits_and_proba(predictor: Any, x: np.ndarray, classes: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """(logits, probabilities) mapped to ``classes``; a class absent from the fit gets logit NaN and probability 0."""

    n = x.shape[0]
    logits = np.full((n, len(classes)), np.nan)
    proba = np.full((n, len(classes)), np.nan)
    finite = np.all(np.isfinite(x[:, predictor.standardizer.keep]), axis=1)
    if finite.any():
        decision = predictor.decision_function(x[finite])
        p = predictor.predict_proba(x[finite]).to_numpy(dtype=float)
        for j, cls in enumerate(classes):
            if cls in predictor.classes:
                k = list(predictor.classes).index(cls)
                logits[finite, j] = decision[:, k]
                proba[finite, j] = p[:, k]
            else:
                proba[finite, j] = 0.0
    return logits, proba


def fit_d20_models(features: pd.DataFrame, dev: pd.DataFrame, b: pd.DataFrame, labels: pd.Series, critical_ids: set[str], *,
                   targets: list[str], transition_cfg: Mapping[str, Any], registry: Mapping[str, Any],
                   predictor_cfg: Mapping[str, Any], make_inner_fn: Callable[[pd.DataFrame], Callable[..., np.ndarray]],
                   classes: Sequence[str] = CLASS_ORDER) -> D20Fit:
    """Fit the D20 transition and predictor on acdc_dev exactly like ``fit-transition`` and ``fit-predictor``.

    ``features``: ``sample_id`` + ``a_*`` (acdc_dev only); ``dev``: split-manifest rows ``sample_id, outer_fold,
    cluster_id`` of acdc_dev; ``b``: typed B of ACDC; ``labels``: ACDC class by sample_id; ``critical_ids``: rows
    with critical QC (excluded from every fit); ``make_inner_fn(ids)``: the v6 inner-fold function for ``ids``."""

    from functools import partial

    from .classification import fit_formal_predictor
    from .transition_fit import fit_transition_v6, nested_oof_transition

    cols = [c for c in features.columns if c.startswith("a_")]
    if len(cols) != FEATURE_DIM:
        raise ValueError(f"D20 features must have {FEATURE_DIM} columns, found {len(cols)}")
    feats = features.set_index("sample_id")
    dev = dev[["sample_id", "outer_fold", "cluster_id"]].copy()
    outside = sorted(set(feats.index) - set(dev["sample_id"]))
    if outside:
        raise ValueError(f"D20 fitting received rows outside acdc_dev (e.g. {outside[:3]})")
    # Transition rows: acdc_dev with features and B, critical QC removed (stages._dev_matrices).
    b_fit = b[~b["sample_id"].isin(critical_ids)]
    ids = (dev.merge(feats.reset_index()[["sample_id"]], on="sample_id").merge(b_fit[["sample_id"]], on="sample_id")
           .sort_values("sample_id").reset_index(drop=True))
    a_rows = feats.loc[ids["sample_id"], cols].to_numpy(dtype=np.float64)
    b_rows = b_fit.set_index("sample_id").loc[ids["sample_id"]].copy().reset_index()
    folds = ids["outer_fold"].to_numpy(dtype=int)
    tcfg, reg = dict(transition_cfg), dict(registry)
    model = fit_transition_v6(a_rows, b_rows, targets, tcfg, reg, folds, representation=D20_REPRESENTATION)
    fitter = partial(fit_transition_v6, representation=D20_REPRESENTATION)
    oof, details = nested_oof_transition(a_rows, b_rows, targets, tcfg, reg, folds, make_inner_fn(ids), return_details=True, fitter=fitter)
    oof = oof.assign(sample_id=ids["sample_id"].to_numpy()) if "sample_id" not in oof else oof
    final = pd.DataFrame([{"outer_fold": pd.NA, "target": t, "lambda": model.lambdas.get(t, np.nan),
                           "lambda_at_grid_edge": bool(model.edge_flags.get(t, False)),
                           "flag": "LAMBDA_AT_GRID_EDGE" if model.edge_flags.get(t, False) else "",
                           "eligible": t in model.fitted_targets, "exclusion_reason": model.excluded.get(t, ""),
                           "n_fit_rows": model.n_fit_rows} for t in targets])
    ledger = pd.concat([details["lambda_selection"].assign(scope="outer_fold"), final.assign(scope="final_bundle")], ignore_index=True)
    ledger["outer_fold"] = ledger["outer_fold"].astype("Int64")

    # Predictor rows: every acdc_dev row with features; critical-QC or non-finite rows are not fitted (fit-predictor).
    prows = dev.merge(feats.reset_index()[["sample_id"]], on="sample_id").sort_values("sample_id").reset_index(drop=True)
    x = feats.loc[prows["sample_id"], cols].to_numpy(dtype=np.float64)
    y = labels.reindex(prows["sample_id"]).astype(object).reset_index(drop=True)
    fit_rows = np.isfinite(x).all(axis=1) & ~prows["sample_id"].isin(critical_ids).to_numpy() & y.notna().to_numpy()
    pcfg = dict(predictor_cfg)
    predictor = fit_formal_predictor(x, y, pcfg, fit_mask=fit_rows)
    pfolds = prows["outer_fold"].to_numpy(dtype=int)
    crossfit = np.empty(len(prows), dtype=object)
    fold_predictors: dict[int, Any] = {}
    blocks = []
    for fold in sorted(set(pfolds)):
        held = pfolds == fold
        fold_predictor = fit_formal_predictor(x, y, pcfg, fit_mask=fit_rows & ~held)
        fold_predictors[int(fold)] = fold_predictor
        logits, proba = _logits_and_proba(fold_predictor, x[held], classes)
        crossfit[held] = fold_predictor.predict(x[held])
        blocks.append(prediction_frame(sample_id=prows.loc[held, "sample_id"].tolist(), cluster_id=prows.loc[held, "cluster_id"].tolist(),
                                       probabilities=proba, logits=logits, model_id=MODEL_ID, partition="acdc_dev",
                                       encoder_id=f"fold{int(fold)}", is_oof=True, y_true=y[held].tolist(), classes=classes))
    logits, proba = _logits_and_proba(predictor, x, classes)
    blocks.append(prediction_frame(sample_id=prows["sample_id"].tolist(), cluster_id=prows["cluster_id"].tolist(), probabilities=proba,
                                   logits=logits, model_id=MODEL_ID, partition="acdc_dev", encoder_id="final", is_oof=False,
                                   y_true=y.tolist(), classes=classes))
    d_model = pd.DataFrame({"sample_id": prows["sample_id"].to_numpy(), "d_model": predictor.predict(x)})
    crossfit_frame = pd.DataFrame({"sample_id": prows["sample_id"].to_numpy(), "outer_fold": pfolds, "d_model_crossfit": crossfit})
    summary = {"transition_rows": int(len(ids)), "predictor_rows": int(len(prows)), "predictor_fit_rows": int(fit_rows.sum()),
               "fitted_targets": list(model.fitted_targets), "excluded_targets": dict(model.excluded),
               "missing_classes": list(predictor.missing_classes)}
    return D20Fit(model=model, oof=oof, lambda_ledger=ledger, fold_models=dict(details["models"]), predictor=predictor,
                  fold_predictors=fold_predictors, d_model=d_model, crossfit=crossfit_frame,
                  predictions=sort_frame(pd.concat(blocks, ignore_index=True)), transition_ids=ids, summary=summary)


# ----------------------------------------------------------------------------- bundle files and freeze


def bundle_paths(deep_dir: str | Path, n_folds: int = 5) -> dict[str, Path]:
    d = Path(deep_dir)
    paths = {
        "features_dev": d / "features" / "A_D20_acdc_dev.parquet",
        "transition_model": d / "transitions" / "transition_model.json",
        "lambda_selection_json": d / "transitions" / "lambda_selection.json",
        "lambda_selection": d / "transitions" / "lambda_selection.parquet",
        "oof": d / "oof" / "bhat_oof_ACDC_dev.parquet",
        "predictor_joblib": d / "formal" / "d_model_predictor.joblib",
        "predictor_json": d / "formal" / "d_model_predictor.json",
        "d_model_dev": d / "formal" / "d_model_ACDC_dev.parquet",
        "d_model_crossfit": d / "formal" / "d_model_crossfit_ACDC_dev.parquet",
        "predictions": d / "predictions.parquet",
        "fit_record": d / "fit_record.json",
    }
    for k in range(1, int(n_folds) + 1):
        paths[f"transition_fold{k}"] = d / "transitions" / "fold_models" / f"transition_fold{k}.json"
        paths[f"predictor_fold{k}"] = d / "formal" / "fold_predictors" / f"predictor_fold{k}.json"
    return paths


def write_d20_fit(fit: D20Fit, deep_dir: str | Path, *, n_folds: int, record: Mapping[str, Any]) -> dict[str, Path]:
    """Write every fitted object of :class:`D20Fit` in the primary layout below ``deep_dir``."""

    import joblib

    from .runs import atomic_write_json

    paths = bundle_paths(deep_dir, n_folds)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    def table(frame: pd.DataFrame, path: Path) -> None:
        tmp = path.with_name(f".{path.name}.tmp")
        frame.to_parquet(tmp, engine="pyarrow", index=False)
        tmp.replace(path)

    atomic_write_json(paths["transition_model"], fit.model.to_dict(), allow_nan=True)
    atomic_write_json(paths["lambda_selection_json"], {"lambdas": dict(fit.model.lambdas), "edge_flags": dict(fit.model.edge_flags)}, allow_nan=True)
    table(fit.lambda_ledger, paths["lambda_selection"])
    table(fit.oof, paths["oof"])
    for k in range(1, int(n_folds) + 1):
        if k not in fit.fold_models or k not in fit.fold_predictors:
            raise ValueError(f"outer fold {k} has no fitted D20 transition or predictor")
        atomic_write_json(paths[f"transition_fold{k}"], fit.fold_models[k].to_dict(), allow_nan=True)
        atomic_write_json(paths[f"predictor_fold{k}"], fit.fold_predictors[k].to_dict(), allow_nan=True)
    tmp = paths["predictor_joblib"].with_name(".d_model_predictor.joblib.tmp")
    joblib.dump(fit.predictor, tmp)
    tmp.replace(paths["predictor_joblib"])
    atomic_write_json(paths["predictor_json"], fit.predictor.to_dict(), allow_nan=True)
    table(fit.d_model, paths["d_model_dev"])
    table(fit.crossfit, paths["d_model_crossfit"])
    table(fit.predictions, paths["predictions"])
    atomic_write_json(paths["fit_record"], {**dict(record), "summary": fit.summary, "representation": D20_REPRESENTATION})
    return paths


def freeze_d20(deep_dir: str | Path, *, n_folds: int, provenance: Mapping[str, Any]) -> dict[str, Any]:
    """``FROZEN_D20.json``: SHA-256 of every bundle file plus ``provenance``; ``bundle_sha256`` over both."""

    from .runs import atomic_write_json, sha256_file, sha256_json

    d = Path(deep_dir)
    files = {path.relative_to(d).as_posix(): sha256_file(path) for path in sorted(bundle_paths(d, n_folds).values())}
    payload = {"schema": "tmmri.d20_bundle/1", "representation": D20_REPRESENTATION, "model_id": MODEL_ID, "n_folds": int(n_folds),
               "files": files, "provenance": dict(provenance)}
    frozen = {**payload, "bundle_sha256": sha256_json(payload)}
    atomic_write_json(d / FROZEN_NAME, frozen)
    return frozen


def load_d20_bundle(deep_dir: str | Path) -> tuple[Any, Any, dict[str, Any]]:
    """(TransitionModel, FormalPredictor, frozen record) after re-hashing every frozen file."""

    from .classification import FormalPredictor
    from .runs import sha256_file, sha256_json
    from .transition_fit import TransitionModel

    d = Path(deep_dir)
    path = d / FROZEN_NAME
    if not path.is_file():
        raise D20BundleError(f"{path} missing: run fit-deep-encoder first")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    payload = {k: v for k, v in frozen.items() if k != "bundle_sha256"}
    if sha256_json(payload) != frozen.get("bundle_sha256"):
        raise D20BundleError("FROZEN_D20.json content does not match its bundle_sha256")
    changed = [rel for rel, sha in frozen["files"].items() if not (d / rel).is_file() or sha256_file(d / rel) != sha]
    if changed:
        raise D20BundleError(f"D20 bundle files changed after the freeze: {changed}")
    model = TransitionModel.from_dict(json.loads((d / "transitions" / "transition_model.json").read_text(encoding="utf-8")))
    predictor = FormalPredictor.from_dict(json.loads((d / "formal" / "d_model_predictor.json").read_text(encoding="utf-8")))
    return model, predictor, frozen


def d20_predictions(deep_dir: str | Path, features: pd.DataFrame, partition: str, *, y_true: pd.Series | None = None,
                    classes: Sequence[str] = CLASS_ORDER) -> pd.DataFrame:
    """C4 rows of the frozen final D20 predictor on ``features`` of ``partition`` (no fitting)."""

    _, predictor, _ = load_d20_bundle(deep_dir)
    frame = features.sort_values("sample_id").reset_index(drop=True)
    x = frame[feature_columns()].to_numpy(dtype=np.float64)
    logits, proba = _logits_and_proba(predictor, x, classes)
    truth = None if y_true is None else y_true.reindex(frame["sample_id"]).tolist()
    return prediction_frame(sample_id=frame["sample_id"].tolist(), cluster_id=frame["cluster_id"].tolist(), probabilities=proba,
                            logits=logits, model_id=MODEL_ID, partition=partition, encoder_id="final", is_oof=False,
                            y_true=truth, classes=classes)


def d20_reconstructions(deep_dir: str | Path, features: pd.DataFrame) -> pd.DataFrame:
    """B-hat (original units) of the frozen final D20 transition, indexed by sample_id (no fitting)."""

    model, _, _ = load_d20_bundle(deep_dir)
    frame = features.sort_values("sample_id").reset_index(drop=True)
    return model.predict(frame[feature_columns()].to_numpy(dtype=np.float64), index=pd.Index(frame["sample_id"], name="sample_id"))
