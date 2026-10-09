"""Local F9 inference (``infer-f9``): predictions and 64-d activations of the trained CNN comparator (contract C4).

Checkpoints follow contract C8: ``neural_f9/checkpoints/<fit_id>/model.pt`` (state_dict) and ``record.json`` (the
fit record of the C5 fit spec), loaded only through ``tmmri.neural.model.load_f9``; tensors come from the C3 cache
through ``tmmri.neural.data.load_npz`` and ``collate_patients`` (the trainer's own loader, so channel replication and
ImageNet normalization are exactly those of training).  Only refit encoders are used (``purpose == "f9"``,
``selection_fold`` null, ``fraction`` 1, ``repeat`` 0): on ``acdc_dev`` the five outer-fold encoders and the final one
(``fold<k>`` predicts every development patient, ``is_oof`` only for the patients of outer fold k, so the F9b probe
can fit T on fold k's fitting-patient activations); on every other partition the final encoder only.  Each fold
encoder must not have been fitted on its held-out fold, and no encoder may have seen a patient of an evaluation
partition; violations refuse the run.  Patients run one at a time in eval mode under the C9 numerics.  Every
prediction and activation is D4-symmetrized (6.0.1, owner issue O1; ``configs/f9_cnn.yaml`` ``inference.tta:
d4_mean_8``; :func:`tmmri.neural.dihedral.f9_forward_tta`): the logits and the 64-d activations A are the Float64 means
over the eight in-plane D4 transforms of the ED and ES images of the FP32 outputs, and the probabilities are the Float64
softmax of the mean logits; ``details["tta"]`` records the setting.  All reported F9 numbers come from this local
inference.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .prediction_table import (
    CLASS_ORDER,
    activation_frame,
    encoder_id,
    prediction_frame,
    sort_frame,
)
from .dihedral import D4_TTA, FORWARD_KEYS, f9_forward_tta
from .preprocess import read_cache_manifest, verify_cache_entry

ACTIVATION_DIM = 64
MODEL_ID = "f9"


class CheckpointError(ValueError):
    """Missing, duplicated or leaking F9 checkpoints."""


@dataclass(frozen=True)
class EncoderCheckpoint:
    fit_id: str
    seed: int
    encoder_id: str
    outer_fold: int | None
    path: Path
    fitting_patients: frozenset[str]


def _field(record: Mapping[str, Any], key: str) -> Any:
    for container in (record, record.get("spec") or {}, record.get("fit_spec") or {}):
        if isinstance(container, Mapping) and key in container:
            return container[key]
    return None


def discover_checkpoints(root: str | Path) -> list[EncoderCheckpoint]:
    """Refit F9 encoders below ``neural_f9/checkpoints`` (one per seed and fitting set; duplicates refuse)."""

    found: dict[tuple[int, str], EncoderCheckpoint] = {}
    base = Path(root)
    for record_path in sorted(base.glob("*/record.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if str(_field(record, "purpose")) != "f9" or _field(record, "selection_fold") is not None:
            continue
        fraction, repeat = _field(record, "fraction"), _field(record, "repeat")
        if (fraction is not None and float(fraction) != 1.0) or (repeat is not None and int(repeat) != 0):
            continue
        if not (record_path.parent / "model.pt").is_file():
            raise CheckpointError(f"{record_path.parent}: model.pt missing")
        outer = _field(record, "outer_fold")
        enc = encoder_id(outer)
        seed = int(_field(record, "seed"))
        key = (seed, enc)
        if key in found:
            raise CheckpointError(f"two refit checkpoints for seed {seed}, {enc}: {found[key].path.name} and {record_path.parent.name}")
        found[key] = EncoderCheckpoint(fit_id=str(_field(record, "fit_id") or record_path.parent.name), seed=seed, encoder_id=enc,
                                       outer_fold=None if enc == "final" else int(outer), path=record_path.parent,
                                       fitting_patients=frozenset(str(s) for s in (_field(record, "fitting_patients") or [])))
    return [found[k] for k in sorted(found)]


def select_encoders(checkpoints: Sequence[EncoderCheckpoint], wanted: Sequence[str], *, seeds: Iterable[int] | None = None) -> list[EncoderCheckpoint]:
    chosen = [c for c in checkpoints if c.encoder_id in set(wanted) and (seeds is None or c.seed in set(seeds))]
    present = {(c.seed, c.encoder_id) for c in chosen}
    for seed in sorted({c.seed for c in chosen}):
        missing = [e for e in wanted if (seed, e) not in present]
        if missing:
            raise CheckpointError(f"seed {seed}: refit encoders {missing} missing")
    if not chosen:
        raise CheckpointError(f"no refit F9 checkpoints for encoders {list(wanted)}")
    return chosen


def check_leakage(encoders: Sequence[EncoderCheckpoint], partition: str, partition_ids: set[str], dev_folds: Mapping[str, int],
                  ) -> None:
    """Refuse a fold encoder fitted on its held-out fold, any encoder fitted outside acdc_dev, and any encoder that saw
    a patient of an evaluation partition."""

    dev_ids = set(dev_folds)
    for enc in encoders:
        if not enc.fitting_patients:
            raise CheckpointError(f"{enc.fit_id}: record.json lists no fitting_patients")
        outside = sorted(enc.fitting_patients - dev_ids)
        if outside:
            raise CheckpointError(f"{enc.fit_id}: fitted on patients outside acdc_dev (e.g. {outside[:3]})")
        if enc.outer_fold is not None:
            held = {s for s, f in dev_folds.items() if int(f) == enc.outer_fold}
            leak = sorted(enc.fitting_patients & held)
            if leak:
                raise CheckpointError(f"{enc.fit_id} ({enc.encoder_id}): fitted on its held-out outer fold (e.g. {leak[:3]})")
        if partition != "acdc_dev":
            leak = sorted(enc.fitting_patients & partition_ids)
            if leak:
                raise CheckpointError(f"{enc.fit_id}: fitted on {partition} patients (e.g. {leak[:3]})")


def _to_device(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}


def run_f9_inference(cache_dir: str | Path, encoders: Sequence[EncoderCheckpoint], *, partition: str, device: Any,
                     dev_folds: Mapping[str, int] | None = None, y_true: Mapping[str, Any] | None = None,
                     classes: Sequence[str] = CLASS_ORDER, tta: str = D4_TTA) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """(predictions, activations, details) of ``encoders`` on every case of the C3 cache in ``cache_dir`` (``tta``: the
    test-time symmetrization of :func:`tmmri.neural.dihedral.f9_forward_tta`; every reported number uses the default)."""

    import torch

    from ..runs import sha256_file

    model_api = importlib.import_module("tmmri.neural.model")
    data_api = importlib.import_module("tmmri.neural.data")
    manifest = read_cache_manifest(cache_dir)
    rows = manifest.to_dict("records")
    paths = {str(r["sample_id"]): verify_cache_entry(cache_dir, r) for r in rows}
    folds = dict(dev_folds or {})
    truth = dict(y_true or {})
    pred_blocks, act_blocks, used = [], [], []
    for enc in encoders:
        model = model_api.load_f9(str(enc.path))
        model.to(device)
        model.eval()
        logits_rows, act_rows = [], []
        with torch.inference_mode():
            for r in rows:
                batch = _to_device(data_api.collate_patients([data_api.load_npz(str(paths[str(r["sample_id"])]))]), device)
                logits, activations = f9_forward_tta(model, batch, tta=tta)
                logits_rows.append(logits.numpy().reshape(-1))
                act_rows.append(activations.numpy().reshape(-1))
        logit = np.vstack(logits_rows) if logits_rows else np.zeros((0, len(classes)))
        acts = np.vstack(act_rows) if act_rows else np.zeros((0, ACTIVATION_DIM))
        if logit.shape[1] != len(classes) or acts.shape[1] != ACTIVATION_DIM:
            raise CheckpointError(f"{enc.fit_id}: model returned logits {logit.shape} and activations {acts.shape}")
        proba = torch.softmax(torch.from_numpy(logit), dim=1).numpy()
        ids = [str(r["sample_id"]) for r in rows]
        clusters = [str(r["cluster_id"]) for r in rows]
        is_oof = [enc.outer_fold is not None and int(folds.get(s, -1)) == enc.outer_fold for s in ids]
        labels = [truth.get(s) for s in ids]
        keys = dict(sample_id=ids, cluster_id=clusters, model_id=MODEL_ID, partition=partition, encoder_id=enc.encoder_id,
                    is_oof=is_oof, seed=enc.seed)
        pred_blocks.append(prediction_frame(probabilities=proba, logits=logit, y_true=labels, classes=classes, **keys))
        act_blocks.append(activation_frame(activations=acts, **keys))
        used.append({"fit_id": enc.fit_id, "seed": enc.seed, "encoder_id": enc.encoder_id, "checkpoint": enc.path.name,
                     "model_pt_sha256": sha256_file(enc.path / "model.pt"), "record_sha256": sha256_file(enc.path / "record.json"),
                     "n_fitting_patients": len(enc.fitting_patients)})
    predictions = sort_frame(pd.concat(pred_blocks, ignore_index=True))
    activations = sort_frame(pd.concat(act_blocks, ignore_index=True))
    details = {"partition": partition, "n_cases": len(rows), "tta": tta, "encoders": used, "cache_manifest_rows": len(manifest),
               "transform_versions": sorted({str(v) for v in manifest["transform_version"]})}
    return predictions, activations, details
