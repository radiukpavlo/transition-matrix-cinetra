"""B22 of the post-freeze exploratory block (protocol-v6.0.2; ``configs/exploratory_postfreeze.yaml`` ``B22``;
SPECIFICATION.md 13.7, Decisions D37, D39 and D41): the mask-aware fine-tuned CNN F9-M and its concept probe F9b-M.
Label ``EXPLORATORY_POST_HOC``; estimation only, no test, no selection.

* **Inputs.**  The recorded C3 image tensors of the source run (``neural_cache/<partition>/``; every file hash-checked
  through ``PostfreezeContext.verified`` and the cache manifest) plus the staged reference masks on the same grid
  (:func:`tmmri.neural.f9m.mask_label_slices`; staging hash check of ``case.json``).  Alignment check: the staged images
  through :func:`tmmri.neural.preprocess.preprocess_case` must reproduce the recorded tensors within
  :data:`ALIGNMENT_TOLERANCE` (the cache transform is deterministic Float64 numpy cast to FP32, so a recomputation differs
  at most by FP32 rounding, below 1e-6, while a geometry or orientation mismatch moves edges by whole pixels, about
  0.1-1 in [0, 1] units); a larger difference refuses the analysis.  The label slices are written to
  ``postfreeze/B22/mask_cache/<partition>/`` with a manifest.
* **Networks.**  For each F9 seed (20261011-20261013) and fitting set (outer folds 1-5 and the final fit): the winning
  F9 candidate, the refit epoch count and the fitting patients recorded by ``select-f9`` (``selection_summary.json``,
  cross-checked against ``f9_bundle.json`` and the refit's ``record.json`` spec, whose fit id is recomputed from its
  content), trained as F9-M by :func:`tmmri.neural.f9m.train_refit` (18 refits; no selection fit, no early stopping;
  local GPU, IEEE FP32, deterministic algorithms).  The training configuration must hash to the recorded F9
  ``config_sha256``.  F9-M trains on F9's recorded fitting patients (the comparison isolates the mask channels; F9's
  fitting sets keep critical-QC rows); every fit and every metric of this analysis excludes critical-QC rows.
* **Order** (lane ``policy.freeze_before_evaluation``): all refits are written and hashed (``fits/FROZEN.json``), the
  ``acdc_dev`` activations of all six encoders per seed are computed and the F9b-M probes fitted and hashed
  (``probes/seed<k>/FROZEN.json``) before any evaluated partition is read; each evaluated partition is then processed
  once with the final encoders after re-verifying both freezes.
* **F9b-M** (lane ``B22.probe``): :func:`tmmri.experiments.probes.fit_f9b_probes` per seed: per outer fold k the typed
  transition on encoder k's standardized 64 activations of its fitting rows (acdc_dev without critical QC), the
  out-of-fold B-hat of fold k, and the final T on the final encoder's activations; latent coordinates are never pooled
  across encoders.
* **Outputs** per seed and as the seed mean (range beside it): rho, rho_EF and rho of the volumes (F9b-M), accuracy,
  macro-F1 and macro OvR AUROC with the chance and majority-class rates (F9-M head, same-definition reference) on
  ``acdc_dev`` out of fold and the evaluated partitions (:mod:`.b21_frozen_image` evaluation), and the validity gate:
  development out-of-fold macro-F1 over all five classes >= 0.30 per seed (with F9's training-loss and parameter-change
  checks), reported and never used to select.
* **Fallback** (lane ``B22.fallback``): a seed whose refits do not all complete is reported ``NOT_EXECUTED`` with the
  reason; the completed seeds are reported.  Completed refit directories are reused when the analysis is resumed in the
  same run root (:func:`tmmri.neural.f9m.train_refit` is idempotent).
"""

from __future__ import annotations

import io
import time
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .b21_frozen_image import (
    DATASETS,
    LABEL,
    PartitionInputs,
    class_order,
    dev_reference,
    freeze,
    inner_fold_fn,
    lane_partitions,
    load_partition,
    partition_counts,
    predictor_report,
    rho_report,
    ridge_reference,
    source_json,
    source_path,
    staged_case_loader,
    staged_root,
    verify_frozen,
    write_table,
)

ANALYSIS = "B22"
SCHEMA = "tmmri.postfreeze_b22/1"
ALIGNMENT_TOLERANCE = 1e-4
GATE_MIN_MACRO_F1 = 0.30
NOT_EXECUTED = "NOT_EXECUTED"
ACTIVATION_DIM = 64
ZIP_DATE = (1980, 1, 1, 0, 0, 0)


# ============================================================================= refit plan


def refit_plan(pctx: Any, cfg: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The 18 recorded F9 refits (seed, fitting set, winning candidate, refit epochs, fitting patients), each checked
    against ``f9_bundle.json`` and its refit record; ``cfg`` must hash to the recorded training configuration."""

    from ...neural.fit_spec import Candidate, FitSpec
    from ...neural.prediction_table import encoder_id
    from ...runs import sha256_json

    summary = source_json(pctx, "neural_f9/selection/selection_summary.json")
    bundle = source_json(pctx, "neural_f9/f9_bundle.json")
    fits = {str(f["fit_id"]): f for f in bundle.get("fits", [])}
    seeds = [int(s) for s in cfg["seeds"]["values"]]
    cfg_sha = sha256_json(dict(cfg))
    plan: list[dict[str, Any]] = []
    for w in summary["winners"]:
        fid = str(w["fit_id"])
        record = source_json(pctx, f"neural_f9/checkpoints/{fid}/record.json")
        spec = FitSpec.from_dict(record["spec"])                  # recomputes the fit id from the recorded content
        recorded = fits.get(fid)
        problems = [name for name, ok in (
            ("record fit_id", spec.fit_id == fid == str(record.get("fit_id"))), ("refit spec", spec.selection_fold is None),
            ("seed", spec.seed == int(w["seed"]) and spec.seed in seeds), ("outer_fold", str(spec.outer_fold) == str(w["outer_fold"])),
            ("candidate", spec.candidate.label == str(w["candidate"])), ("refit_epochs", spec.refit_epochs == int(w["refit_epochs"])),
            ("f9_bundle", recorded is not None and Candidate.coerce(recorded["candidate"]) == spec.candidate
             and int(recorded["refit_epochs"]) == spec.refit_epochs and int(recorded["seed"]) == spec.seed),
            ("config_sha256", str(record.get("config_sha256")) == cfg_sha)) if not ok]
        if problems:
            raise ValueError(f"recorded F9 refit {fid} is inconsistent ({', '.join(problems)}); B22 refuses")
        plan.append({"seed": spec.seed, "outer_fold": spec.outer_fold, "encoder_id": encoder_id(spec.outer_fold), "source_fit_id": fid,
                     "candidate": spec.candidate.label, "refit_epochs": int(spec.refit_epochs), "n_fitting_patients": len(spec.fitting_patients),
                     "spec": spec})
    expected = int(cfg["fit_counts"]["wave2_refit"])
    if len(plan) != expected and not summary.get("not_estimable"):
        raise ValueError(f"selection_summary.json lists {len(plan)} refits, configs/f9_cnn.yaml expects {expected}")
    order = {s: i for i, s in enumerate(seeds)}
    plan.sort(key=lambda p: (order[p["seed"]], 99 if p["encoder_id"] == "final" else int(p["outer_fold"])))
    meta = {"winners": len(plan), "not_estimable": list(summary.get("not_estimable") or []), "seeds": seeds,
            "training_config_sha256": cfg_sha, "total_refit_epochs": int(sum(p["refit_epochs"] for p in plan))}
    return plan, meta


def check_fitting_sets(plan: Sequence[Mapping[str, Any]], dev_folds: Mapping[str, int]) -> None:
    """Every fitting patient belongs to acdc_dev, and a fold encoder never saw its held-out fold."""

    for p in plan:
        patients = set(p["spec"].fitting_patients)
        outside = patients - set(dev_folds)
        if outside:
            raise ValueError(f"refit {p['source_fit_id']}: {len(outside)} fitting patients outside acdc_dev")
        if p["encoder_id"] != "final":
            leak = {s for s in patients if int(dev_folds[s]) == int(p["outer_fold"])}
            if leak:
                raise ValueError(f"refit {p['source_fit_id']}: fitted on {len(leak)} patients of its held-out outer fold")


# ============================================================================= inputs (recorded images + staged masks)


def _labels_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("ed_labels", "es_labels"):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.ascontiguousarray(arrays[name], dtype=np.uint8), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload.getvalue())
    return buffer.getvalue()


def partition_items(pctx: Any, partition: str, loader: Callable[[Mapping[str, Any]], tuple[Any, dict[str, np.ndarray]]], ni_cfg: Mapping[str, Any],
                    out: Path) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    """(items, mask manifest, summary) of one partition: the recorded C3 arrays of every cache row (hash-checked) with
    the reference-mask label slices on the same grid, after the alignment check; label slices written below ``out``."""

    import hashlib

    from ...neural.data import load_npz, read_manifest
    from ...neural.f9m import MASK_STRUCTURES, mask_label_slices
    from ...neural.preprocess import CACHE_ARRAYS, cache_file_name, preprocess_case
    from ...runs import atomic_write_bytes

    manifest_rel = f"neural_cache/{partition}/manifest.csv"
    manifest_path = source_path(pctx, manifest_rel)
    read_manifest(manifest_path)                                    # column and uniqueness checks
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)   # identifiers stay strings
    target = out / "mask_cache" / partition
    items: dict[str, dict[str, Any]] = {}
    rows = []
    worst = 0.0
    for r in manifest.to_dict("records"):
        sid = str(r["sample_id"])
        name = str(r.get("npz_file") or "").strip() or cache_file_name(sid)
        arrays = load_npz(source_path(pctx, f"neural_cache/{partition}/{name}"), expected_sha256=str(r["npz_sha256"]))
        case, masks = loader({"sample_id": sid, "cluster_id": str(r["cluster_id"]), "dataset": str(r["dataset"]), "case_id": str(r["case_id"])})
        again = preprocess_case(case, ni_cfg).arrays
        diff = 0.0
        for key in CACHE_ARRAYS:
            if again[key].shape != arrays[key].shape:
                raise ValueError(f"{partition}: the staged images do not reproduce the recorded tensor shapes ({key}); B22 refuses")
            diff = max(diff, float(np.max(np.abs(again[key].astype(np.float64) - arrays[key].astype(np.float64)))) if arrays[key].size else 0.0)
        if diff > ALIGNMENT_TOLERANCE:
            raise ValueError(f"{partition}: staged images differ from the recorded tensors by {diff:.3g} > {ALIGNMENT_TOLERANCE}; B22 refuses")
        worst = max(worst, diff)
        labels = mask_label_slices(case, masks, ni_cfg)
        for phase in ("ed", "es"):
            if labels[f"{phase}_labels"].shape != arrays[phase].shape:
                raise ValueError(f"{partition}: mask grid {labels[f'{phase}_labels'].shape} differs from the image grid {arrays[phase].shape}")
        data = _labels_npz(labels)
        atomic_write_bytes(target / name, data)
        absent = [f"{phase.upper()}:{k}" for phase in ("ed", "es") for k in MASK_STRUCTURES if not (labels[f"{phase}_labels"] == k).any()]
        rows.append({"sample_id": sid, "cluster_id": str(r["cluster_id"]), "partition": partition, "label": r.get("label", ""),
                     "n_ed": int(arrays["ed"].shape[0]), "n_es": int(arrays["es"].shape[0]), "labels_npz_file": name,
                     "labels_npz_sha256": hashlib.sha256(data).hexdigest(), "image_npz_sha256": str(r["npz_sha256"]),
                     "image_max_abs_difference": diff, "absent_structures": ";".join(absent)})
        items[sid] = {"sample_id": sid, **arrays, **labels}
    frame = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True) if rows else pd.DataFrame(rows)
    write_table(frame, out / "mask_cache" / f"mask_manifest_{partition}.parquet")
    summary = {"n_cases": len(items), "image_alignment_max_abs_difference": worst, "alignment_tolerance": ALIGNMENT_TOLERANCE,
               "n_cases_with_absent_structure": int((frame["absent_structures"] != "").sum()) if len(frame) else 0,
               "image_cache_manifest": manifest_rel}
    return items, frame, summary


# ============================================================================= inference


def infer(models: Mapping[str, Any], items: Mapping[str, Mapping[str, Any]], ids: Sequence[str], *, partition: str, seed: int,
          clusters: Mapping[str, Any], truth: Mapping[str, Any], dev_folds: Mapping[str, int], device: Any,
          classes: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(predictions, activations) of every encoder in ``models`` (encoder id -> F9MNet) on ``ids`` (8-transform TTA;
    probabilities = Float64 softmax of the mean logits); ``is_oof`` marks a fold encoder's held-out patients."""

    import torch

    from ...neural.f9m import MODEL_ID, predict_patients
    from ...neural.prediction_table import activation_columns

    preds, acts = [], []
    acols = activation_columns(ACTIVATION_DIM)
    for enc, model in models.items():
        model.to(device)
        logits, activations = predict_patients(model, [items[s] for s in ids], device)
        proba = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        fold = None if enc == "final" else int(enc[len("fold"):])
        keys = pd.DataFrame({"sample_id": list(ids), "cluster_id": [str(clusters.get(s, s)) for s in ids], "partition": partition,
                             "model_id": MODEL_ID, "seed": int(seed), "encoder_id": enc,
                             "is_oof": [fold is not None and int(dev_folds.get(s, -1)) == fold for s in ids],
                             "y_true": [truth.get(s) for s in ids]})
        p = keys.copy()
        for j, c in enumerate(classes):
            p[f"prob_{c}"] = proba[:, j]
        for j, c in enumerate(classes):
            p[f"logit_{c}"] = logits[:, j]
        p["y_pred"] = [classes[int(k)] for k in proba.argmax(axis=1)]
        preds.append(p)
        acts.append(pd.concat([keys, pd.DataFrame(activations, columns=acols)], axis=1))
        model.to("cpu")
    return pd.concat(preds, ignore_index=True), pd.concat(acts, ignore_index=True)


def _proba(frame: pd.DataFrame, classes: Sequence[str]) -> pd.DataFrame:
    out = frame.set_index("sample_id")[[f"prob_{c}" for c in classes]].astype(float)
    out.columns = list(classes)
    return out


def validity_gate(predictions: pd.DataFrame, classes: Sequence[str], dev_ids: Sequence[str], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Lane ``B22.validity_gate`` (reported, never used to select): development out-of-fold macro-F1 over all five
    classes of the fold encoders (every development patient exactly once) >= 0.30, beside F9's training-loss and
    parameter-change checks of every refit (``configs/f9_cnn.yaml`` ``validity_gate`` (ii) and (iii))."""

    from sklearn.metrics import f1_score

    from ...neural.f9_stages import parameter_change_check, training_loss_check

    oof = predictions[predictions["is_oof"].astype(bool)]
    counts = oof["sample_id"].value_counts()
    dev = set(dev_ids)
    complete = set(counts.index) == dev and bool((counts == 1).all())
    known = oof["y_true"].map(lambda v: v is not None and not pd.isna(v) and str(v) in classes).to_numpy(dtype=bool)
    score = (float(f1_score([str(t) for t in oof["y_true"][known]], [str(p) for p in oof["y_pred"][known]], labels=list(classes),
                            average="macro", zero_division=0)) if known.any() else None)
    loss = [training_loss_check(r) for r in records]
    change = [parameter_change_check(r) for r in records]
    return {"rule": f"development out-of-fold macro-F1 over all five classes >= {GATE_MIN_MACRO_F1}; reported, never used to select",
            "oof_macro_f1": score, "passed": bool(complete and score is not None and score >= GATE_MIN_MACRO_F1),
            "oof_rows_complete": complete, "n_oof_rows": len(oof), "n_dev_patients": len(dev), "chance_level": 1.0 / len(classes),
            "training_loss_check_passed": all(c["passed"] for c in loss), "parameter_change_check_passed": all(c["passed"] for c in change),
            "mask_stem_trained": all(bool((r.get("parameters") or {}).get("mask_stem_trained")) for r in records)}


def _estimate(block: Mapping[str, Any] | None, *path: str) -> float | None:
    value: Any = block
    for key in path:
        value = value.get(key) if isinstance(value, Mapping) else None
    if isinstance(value, Mapping):
        value = value.get("estimate")
    return None if value is None else float(value)


SEED_MEAN_FIELDS = {"rho": ("representation", "rho", "rho"), "rho_ef": ("representation", "rho_ef", "rho"),
                    "rho_volumes": ("representation", "rho_volumes", "rho"), "accuracy": ("predictor", "metrics", "accuracy"),
                    "macro_f1": ("predictor", "metrics", "macro_f1"), "macro_auroc": ("predictor", "metrics", "macro_ovr_auroc")}


def seed_mean(per_seed: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Mean and range over the completed seeds of every reported estimate (no interval; seeds are not resampled)."""

    out: dict[str, Any] = {}
    for name, path in SEED_MEAN_FIELDS.items():
        values = {s: _estimate(v, *path) for s, v in per_seed.items()}
        finite = [v for v in values.values() if v is not None and np.isfinite(v)]
        out[name] = {"mean": float(np.mean(finite)) if finite else None, "min": float(np.min(finite)) if finite else None,
                     "max": float(np.max(finite)) if finite else None, "n_seeds": len(finite), "per_seed": values}
    return out


# ============================================================================= run


def run(pctx: Any, *, device: Any = None, loader: Callable[..., Any] | None = None, config: Mapping[str, Any] | None = None,
        weights: Any = None) -> dict[str, Any]:
    """B22 on the source run of ``pctx`` (see the module docstring).  ``device``, ``loader``, ``config`` and ``weights``
    are test injections (CPU, synthetic cases, a tiny configuration, random weights); a real run uses none of them and
    refuses without CUDA (lane ``B22.backend: local_gpu``)."""

    import torch

    from ...experiments.probes import fit_f9b_probes
    from ...neural.f9_config import load_f9_config, resolve_weights, training_config
    from ...neural.f9m import load_f9m, train_refit
    from ...neural.numerics import configure_numerics, environment_record
    from ...neural.prediction_table import activation_columns
    from ...neural.preprocess import load_neural_input_config
    from ...neural.train import WeightsRef
    from ...runs import atomic_write_json, code_version, sha256_file

    study = pctx.study
    acfg = dict(pctx.analysis_cfg(ANALYSIS) or {})
    if str(acfg.get("backend", "local_gpu")) != "local_gpu":
        raise ValueError(f"lane B22 backend must be local_gpu, found {acfg.get('backend')!r}")
    fit_partition, evaluated = lane_partitions(pctx)
    out = Path(pctx.out_dir(ANALYSIS))
    results_path = out / "results.json"
    pctx.require_new_output(results_path)
    numerics = configure_numerics()
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("B22 trains on the local GPU (lane B22 backend local_gpu); CUDA is not available, nothing was run")
        device = "cuda"
    dev = torch.device(device)
    cfg = dict(config) if config is not None else training_config(load_f9_config(study.root))
    ni_cfg = dict(study.configs.get("neural_input") or load_neural_input_config(study.root))
    classes = [str(c) for c in cfg["class_order"]]
    if classes != class_order(pctx):
        raise ValueError("configs/f9_cnn.yaml class_order differs from the audited-predictor classes")
    plan, plan_meta = refit_plan(pctx, cfg)
    bundle = source_json(pctx, "neural_f9/f9_bundle.json")
    if weights is None:
        w = resolve_weights(cfg, study.root)
        if str(w["sha256"]) != str((bundle.get("weights") or {}).get("sha256")):
            raise ValueError("the pinned ResNet-18 weights differ from the source run's f9_bundle.json")
        weights = WeightsRef(path=w["path"], sha256=w["sha256"], name=w["name"])
    wref = WeightsRef.coerce(weights)
    root = staged_root(pctx)
    load = loader or (lambda row: staged_case_loader(root, row, ni_cfg))
    targets = study.transition_targets()
    code = code_version()

    # ---------------------------------------------------------------- development inputs and the 18 refits
    dev_in = load_partition(pctx, fit_partition)
    dev_folds = {str(r.sample_id): int(r.outer_fold) for r in dev_in.rows.itertuples(index=False)}
    check_fitting_sets(plan, dev_folds)
    dev_items, dev_masks, dev_summary = partition_items(pctx, fit_partition, load, ni_cfg, out)
    labels = {str(r.sample_id): str(r.label) for r in dev_masks.itertuples(index=False)}
    inputs_record = {"partition": fit_partition, "image_cache_manifest_sha256": sha256_file(source_path(pctx, f"neural_cache/{fit_partition}/manifest.csv")),
                     "mask_manifest_sha256": sha256_file(out / "mask_cache" / f"mask_manifest_{fit_partition}.parquet")}
    seeds: dict[int, dict[str, Any]] = {int(s): {"status": "COMPLETE", "fits": {}} for s in plan_meta["seeds"]}
    fits_dir = out / "fits"
    for i, p in enumerate(plan, start=1):
        state = seeds[p["seed"]]
        if state["status"] != "COMPLETE":
            continue
        t0 = time.perf_counter()
        target = fits_dir / f"seed{p['seed']}" / p["encoder_id"]
        try:
            record = train_refit(p["spec"], dev_items, labels, target, dev, config=cfg, weights=wref, mask_channels=True,
                                 inputs={**inputs_record, "source_fit_id": p["source_fit_id"]}, backend={"name": "local", "lane": "B22"})
        except (RuntimeError, ValueError, OSError) as exc:          # fallback: the seed is reported as not executed
            state.update({"status": NOT_EXECUTED, "reason": f"{p['encoder_id']}: {type(exc).__name__}: {str(exc)[:300]}"})
            continue
        if record.status != "COMPLETE":
            state.update({"status": NOT_EXECUTED, "reason": f"{p['encoder_id']}: {record.status}: {record.reason}"})
            continue
        state["fits"][p["encoder_id"]] = {"dir": target, "record": record.to_dict(), "source_fit_id": p["source_fit_id"],
                                          "candidate": p["candidate"], "refit_epochs": p["refit_epochs"]}
        print(f"B22 refit {i}/{len(plan)} seed {p['seed']} {p['encoder_id']}: {p['refit_epochs']} epochs, "
              f"{time.perf_counter() - t0:.0f} s", flush=True)
    done = {s: v for s, v in seeds.items() if v["status"] == "COMPLETE"}
    fit_files = [d["dir"] / name for v in done.values() for d in v["fits"].values() for name in ("model.pt", "record.json")]
    fits_frozen = freeze(fits_dir, fit_files, {"analysis": ANALYSIS, "estimator": "f9m_networks_and_f9bm_probes", "model_id": "f9m",
                                               "completed_seeds": sorted(done), "plan": {k: v for k, v in plan_meta.items()},
                                               "weights": {"name": wref.name, "sha256": wref.sha256},
                                               "code": {"commit": code.get("commit"), "code_sha256": code.get("code_sha256"), "dirty": code.get("dirty")}},
                         schema="tmmri.postfreeze_b22_networks/1") if fit_files else None

    # ---------------------------------------------------------------- acdc_dev inference (six encoders) and the probes
    dev_ids = sorted(dev_items)
    dev_eval = [s for s in dev_in.eval_ids if s in dev_items]
    probe_ids = dev_in.rows[dev_in.rows["sample_id"].isin(dev_eval)][["sample_id", "outer_fold"]].sort_values("sample_id").reset_index(drop=True)
    sd, mean = dev_reference(pctx, dev_in)
    acols = activation_columns(ACTIVATION_DIM)
    dev_pred, dev_act, probes, gates = {}, {}, {}, {}
    for s, v in done.items():
        models = {enc: load_f9m(d["dir"]) for enc, d in v["fits"].items()}
        dev_pred[s], dev_act[s] = infer(models, dev_items, dev_ids, partition=fit_partition, seed=s, clusters=dev_in.clusters,
                                        truth=dev_in.truth.to_dict(), dev_folds=dev_folds, device=dev, classes=classes)
        res = fit_f9b_probes(probe_ids, dev_in.b.loc[probe_ids["sample_id"]].reset_index(), dev_act[s], acols, targets, study.transition,
                             study.registry, inner_fold_fn(pctx, probe_ids))
        pdir = out / "probes" / f"seed{s}"
        files = [write_table(res["oof"], pdir / "bhat_oof_ACDC_dev.parquet"), write_table(res["lambda_selection"], pdir / "lambda_selection.parquet"),
                 atomic_write_json(pdir / "transition_final.json", res["final"].to_dict(), allow_nan=True)]
        files += [atomic_write_json(pdir / f"transition_fold{k}.json", m.to_dict(), allow_nan=True) for k, m in sorted(res["fold_models"].items())]
        probes[s] = {"result": res, "frozen": freeze(pdir, files, {"analysis": ANALYSIS, "representation": "f9bm_activations_64", "seed": s,
                                                                    "networks_bundle_sha256": fits_frozen["bundle_sha256"] if fits_frozen else None},
                                                     schema="tmmri.postfreeze_b22_probe/1")}
        gates[s] = validity_gate(dev_pred[s], classes, dev_ids, [d["record"] for d in v["fits"].values()])
        del models
    infer_dir = out / "inference"
    outputs = [*fit_files, *([fits_dir / "FROZEN.json"] if fits_frozen else [])]
    if done:
        outputs += [write_table(pd.concat(dev_pred.values(), ignore_index=True), infer_dir / f"predictions_{fit_partition}.parquet"),
                    write_table(pd.concat(dev_act.values(), ignore_index=True), infer_dir / f"activations_{fit_partition}.parquet")]
    ridge = ridge_reference(pctx, fit_partition)
    per_seed: dict[int, dict[str, Any]] = {s: {"acdc_dev_oof": {
        "evidence": "development (out-of-fold)", "counts": partition_counts(dev_in, dev_ids),
        "representation": rho_report(pctx, dev_in, probes[s]["result"]["oof"].set_index("sample_id"), probe_ids["sample_id"].tolist(), mean, sd,
                                     reference=ridge),
        "predictor": predictor_report(pctx, dev_in, _proba(dev_pred[s][dev_pred[s]["is_oof"].astype(bool)], classes), dev_in.eval_ids)}}
        for s in done}

    # ---------------------------------------------------------------- evaluated partitions (final encoders), once each
    summaries = {fit_partition: dev_summary}
    for partition in evaluated:
        if fits_frozen:
            verify_frozen(fits_dir)
        for s in done:
            verify_frozen(out / "probes" / f"seed{s}")
        pin: PartitionInputs = load_partition(pctx, partition)
        if not done:
            continue
        items, _, summaries[partition] = partition_items(pctx, partition, load, ni_cfg, out)
        ids = sorted(items)
        ridge = ridge_reference(pctx, partition)
        preds, acts = [], []
        for s, v in done.items():
            model = {"final": load_f9m(v["fits"]["final"]["dir"])}
            pred, act = infer(model, items, ids, partition=partition, seed=s, clusters=pin.clusters, truth=pin.truth.to_dict(),
                              dev_folds={}, device=dev, classes=classes)
            preds.append(pred)
            acts.append(act)
            x = act.set_index("sample_id")[acols]
            bhat = probes[s]["result"]["final"].predict(x.to_numpy(dtype=np.float64), index=pd.Index(x.index, name="sample_id"))
            outputs.append(write_table(bhat.reset_index(), out / "evaluated" / f"bhat_{partition}_seed{s}.parquet"))
            per_seed[s][partition] = {
                "evidence": "registered after unblinding" if partition != "mms2_external" else "prospective (M&Ms-2 lock of protocol-v6.0.1)",
                "counts": partition_counts(pin, ids),
                "representation": rho_report(pctx, pin, bhat, pin.eval_ids, mean, sd, reference=ridge),
                "predictor": predictor_report(pctx, pin, _proba(pred, classes), pin.eval_ids)}
        outputs += [write_table(pd.concat(preds, ignore_index=True), infer_dir / f"predictions_{partition}.parquet"),
                    write_table(pd.concat(acts, ignore_index=True), infer_dir / f"activations_{partition}.parquet")]

    partitions = ["acdc_dev_oof", *evaluated]
    results = {
        "schema": SCHEMA, "label": LABEL, "analysis": ANALYSIS, "answers": acfg.get("answers"), "estimation_only": True, "classes": classes,
        "model_id": "f9m", "plan": {**plan_meta, "refits": [{k: v for k, v in p.items() if k != "spec"} for p in plan]},
        "seeds": {str(s): {"status": v["status"], "reason": v.get("reason"),
                           "fits": {enc: {"source_fit_id": d["source_fit_id"], "candidate": d["candidate"], "refit_epochs": d["refit_epochs"],
                                          "checkpoint_sha256": d["record"]["checkpoint_sha256"], "seconds": d["record"]["timings"].get("seconds_total"),
                                          "mask_stem_max_abs_weight": d["record"]["parameters"].get("mask_stem_max_abs_weight")}
                                    for enc, d in v["fits"].items()}} for s, v in seeds.items()},
        "networks_bundle_sha256": fits_frozen["bundle_sha256"] if fits_frozen else None,
        "probe_bundle_sha256": {str(s): probes[s]["frozen"]["bundle_sha256"] for s in probes},
        "validity_gate": {str(s): g for s, g in gates.items()},
        "per_seed": {str(s): v for s, v in per_seed.items()},
        "seed_mean": {p: seed_mean({str(s): per_seed[s].get(p, {}) for s in per_seed}) for p in partitions},
        "inputs": summaries, "numerics": numerics, "environment": environment_record(dev),
        "fitting_rows": "F9-M networks: the recorded F9 fitting patients; F9b-M probes and every metric: rows without critical QC",
    }
    atomic_write_json(results_path, results, allow_nan=True)
    mask_outputs = sorted((out / "mask_cache").rglob("*"))
    outputs = sorted({*outputs, results_path, *[p for p in mask_outputs if p.is_file()],
                      *[out / "probes" / f"seed{s}" / "FROZEN.json" for s in probes]})
    rels = sorted({"neural_f9/selection/selection_summary.json", "neural_f9/f9_bundle.json",
                   *[f"neural_f9/checkpoints/{p['source_fit_id']}/record.json" for p in plan],
                   "contracts/split_manifest.parquet", "contracts/inner_folds.parquet", "oof/bhat_oof_ACDC_dev.parquet",
                   *[f"neural_cache/{p}/manifest.csv" for p in [fit_partition, *evaluated]],
                   *[f"contracts/cohort_{DATASETS[p]}.parquet" for p in [fit_partition, *evaluated]],
                   *[f"semantics/B_{DATASETS[p]}.parquet" for p in [fit_partition, *evaluated]],
                   *[f"reconstructed/bhat_{p}.parquet" for p in evaluated],
                   *(["contracts/split_manifest_MMS2.parquet", "contracts/cohort_validation_MMS2.json"] if "mms2_external" in evaluated else [])})
    inputs = {f"source:{rel}": source_path(pctx, rel) for rel in rels}    # every npz was hash-checked when read (manifest rows)
    inputs.update(dict(getattr(pctx, "verified_log", None) or {}))      # every verified read of the source run (real context)
    if wref.path and Path(wref.path).is_file():
        inputs[f"weights:{wref.name}"] = Path(wref.path)
    summary = {"analysis": ANALYSIS, "label": LABEL, "completed_seeds": sorted(done),
               "not_executed_seeds": sorted(s for s, v in seeds.items() if v["status"] != "COMPLETE"),
               "validity_gate_passed": {str(s): g["passed"] for s, g in gates.items()},
               "rho_seed_mean": {p: results["seed_mean"][p]["rho"]["mean"] for p in partitions},
               "macro_f1_seed_mean": {p: results["seed_mean"][p]["macro_f1"]["mean"] for p in partitions},
               "networks_bundle_sha256": results["networks_bundle_sha256"],
               "staged_inputs": "staged images and masks of every processed case, verified against case.json (staging hash check)"}
    pctx.write_provenance(ANALYSIS, inputs, outputs, summary)
    return summary
