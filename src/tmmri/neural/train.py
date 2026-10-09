"""F9 training: :func:`train_fit` is the only training entry point (contract C5); ``train-f9 --backend local`` and the
Modal function of ``scripts/cloud/modal_train.py`` both call it.

One call trains one :class:`tmmri.neural.fit_spec.FitSpec` on the ``acdc_dev`` tensor cache (NEURAL-TM-1.0 4.2;
``configs/f9_cnn.yaml``):

* data: the spec's fitting patients; a selection fit trains on two selection folds and validates on the third
  (:func:`tmmri.neural.selection.selection_folds`), a refit trains on all fitting patients for ``refit_epochs``
  without monitoring anything; the cache manifest and every npz are SHA-256 verified;
* loss: class-weighted cross entropy, inverse-frequency weights on the training patients normalized to mean one,
  weighted mean over the patients of each effective batch of four (micro-batches accumulate to the same mean);
* optimizer: AdamW (weight decay 1e-4, betas (0.9, 0.999), eps 1e-8) with the unfrozen backbone blocks at the
  candidate rate and the new layers and head at ten times it; gradient-norm clipping at 5; cosine schedule over 80
  epochs stepped per epoch; patient order reshuffled every epoch;
* augmentation: :func:`tmmri.neural.data.augment_batch` (one D4 element per patient first, 6.0.1 owner issue O1);
* evaluation: every validation output is the D4 test-time symmetrization of inference (:func:`predict_logits`;
  ``inference.tta: d4_mean_8``, recorded as ``tta`` in the fit record), so selection, early stopping and the
  cross-hardware check see the outputs that local inference reports;
* early stopping (selection fits): validation CE, earliest stop at epoch 20, patience 12, minimum improvement 1e-4,
  ties keep the earlier checkpoint; the selected epoch's weights are the fit's ``model.pt``;
* resume: after every epoch ``resume/last.pt`` holds the model, optimizer, scheduler, early-stopping state, epoch
  history and RNG states (``resume/best.pt`` the best weights), written atomically; the epoch RNG is re-derived from
  ``(seed, epoch)``, so an interrupted and resumed fit equals an uninterrupted one bit for bit on the same hardware;
* checks: initial and final parameter hashes, a nonzero change in a convolution of every unfrozen block, unchanged
  frozen weights and unchanged BN running statistics;
* record: ``record.json`` (:class:`FitRecord`) with losses, macro-F1, calibration and per-class support, the selected
  epoch, timings, the C9 environment record and the backend; ``model.pt`` is the state dict.  A completed fit
  directory is returned as is (idempotent; Modal retries and re-runs are safe).
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..runs import atomic_write_json, sha256_file, sha256_json
from .data import (
    TRAINING_PARTITION,
    augment_batch,
    batch_to,
    collate_patients,
    load_npz,
    npz_path,
    read_manifest,
)
from .dihedral import D4_TTA, f9_forward_tta
from .fit_spec import FitSpec, coerce_spec
from .model import (
    CHANGE_WATCH,
    FROZEN_WATCH,
    F9Net,
    build_model,
    cpu_state,
    state_sha256,
)
from .numerics import configure_numerics, environment_record
from .selection import (
    NE_INSUFFICIENT_CLASS_SUPPORT,
    STATUS_COMPLETE,
    EarlyStopper,
    InsufficientClassSupport,
    class_weights,
    require_class_support,
    selection_folds,
)

FIT_RECORD_SCHEMA = "tmmri.f9_fit_record/1"
STATUS_INTERRUPTED = "INTERRUPTED"
RESUME_SCHEMA = "tmmri.f9_resume/1"


@dataclass
class WeightsRef:
    """Backbone initialization: a torchvision ResNet-18 state dict and its pinned SHA-256."""

    path: str | None
    sha256: str | None
    name: str = "resnet18_imagenet1k_v1"

    @classmethod
    def coerce(cls, value: WeightsRef | Mapping[str, Any] | None) -> WeightsRef:
        if isinstance(value, WeightsRef):
            return value
        if value is None:
            raise ValueError("train_fit needs the backbone weights (path and pinned SHA-256)")
        return cls(path=value.get("path"), sha256=value.get("sha256"), name=str(value.get("name", "resnet18_imagenet1k_v1")))


@dataclass
class FitRecord:
    """``record.json`` of one fit."""

    fit_id: str
    spec: dict[str, Any]
    status: str
    reason: str | None = None
    config_sha256: str | None = None
    class_order: list[str] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)
    train_ids: list[str] = field(default_factory=list)
    val_ids: list[str] = field(default_factory=list)
    support_train: dict[str, int] = field(default_factory=dict)
    support_val: dict[str, int] = field(default_factory=dict)
    class_weights: list[float] = field(default_factory=list)
    max_epochs: int | None = None
    epochs: list[dict[str, Any]] = field(default_factory=list)
    selected_epoch: int | None = None
    stopped_epoch: int | None = None
    stop_reason: str | None = None
    val_metrics: dict[str, Any] = field(default_factory=dict)
    val_predictions: list[dict[str, Any]] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    architecture: dict[str, Any] = field(default_factory=dict)
    checkpoint_file: str | None = None
    checkpoint_sha256: str | None = None
    state_sha256: str | None = None
    environment: dict[str, Any] = field(default_factory=dict)
    backend: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, Any] = field(default_factory=dict)
    resume: dict[str, Any] = field(default_factory=dict)
    tta: str | None = None                       # test-time symmetrization of every evaluation output (6.0.1: d4_mean_8)
    schema: str = FIT_RECORD_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FitRecord:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def write(self, path: str | Path) -> Path:
        return atomic_write_json(path, self.to_dict(), allow_nan=True)

    @classmethod
    def load(cls, path: str | Path) -> FitRecord:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def deterministic_view(self) -> dict[str, Any]:
        """The content that must be identical across repeated or resumed fits (timings and host facts excluded)."""

        epochs = [{k: v for k, v in e.items() if k != "seconds"} for e in self.epochs]
        return {"fit_id": self.fit_id, "status": self.status, "epochs": epochs, "selected_epoch": self.selected_epoch,
                "stopped_epoch": self.stopped_epoch, "val_metrics": self.val_metrics, "val_predictions": self.val_predictions,
                "state_sha256": self.state_sha256, "parameters": self.parameters, "class_weights": self.class_weights}


def _utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def epoch_seed(seed: int, epoch: int) -> int:
    import hashlib

    return int.from_bytes(hashlib.sha256(f"f9-epoch|{int(seed)}|{int(epoch)}".encode()).digest()[:8], "big") % (2**63)


def resolve_device(device: Any) -> torch.device:
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# ----------------------------------------------------------------------------- metrics


def _log_softmax64(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64)
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def classification_summary(logits: np.ndarray, y: np.ndarray, classes: Sequence[str], *, n_bins: int = 10) -> dict[str, Any]:
    """Cross entropy (unweighted mean), macro-F1 over all classes, accuracy, equal-width ECE (10 bins), multiclass
    Brier score, log loss and the confusion matrix (rows: reference, columns: prediction; class order)."""

    from sklearn.metrics import confusion_matrix, f1_score

    k = len(classes)
    logp = _log_softmax64(logits)
    prob = np.exp(logp)
    pred = prob.argmax(axis=1)
    onehot = np.eye(k)[y]
    ce = float(-(onehot * logp).sum(axis=1).mean())
    conf = prob.max(axis=1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        sel = (conf > edges[b]) & (conf <= edges[b + 1]) if b else (conf >= edges[0]) & (conf <= edges[1])
        if sel.any():
            ece += sel.mean() * abs(float((pred[sel] == y[sel]).mean()) - float(conf[sel].mean()))
    return {"n": len(y), "cross_entropy": ce, "log_loss": ce,
            "macro_f1": float(f1_score(y, pred, labels=list(range(k)), average="macro", zero_division=0)),
            "accuracy": float((pred == y).mean()), "ece_equal_width_10": float(ece),
            "brier": float(((prob - onehot) ** 2).sum(axis=1).mean()),
            "confusion": confusion_matrix(y, pred, labels=list(range(k))).tolist()}


# ----------------------------------------------------------------------------- loops


def _labels(rows: Any, ids: Sequence[str], index: Mapping[str, int]) -> np.ndarray:
    lookup = dict(zip(rows["sample_id"], rows["label"]))
    return np.array([index[lookup[i]] for i in ids], dtype=np.int64)


@torch.no_grad()
def predict_logits(model: F9Net, items: Sequence[Mapping[str, Any]], device: torch.device, batch_size: int = 4, *,
                   tta: str = D4_TTA) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic evaluation (no random augmentation, eval mode): logits and activations A in input order, each the
    mean over the D4 transforms of ``tta`` (:func:`tmmri.neural.dihedral.f9_forward_tta`), returned as float32."""

    model.eval()
    logits, acts = [], []
    for start in range(0, len(items), batch_size):
        batch = batch_to(collate_patients(items[start:start + batch_size]), device)
        out, a = f9_forward_tta(model, batch, tta=tta)
        logits.append(out.numpy())
        acts.append(a.numpy())
    return np.concatenate(logits).astype(np.float32), np.concatenate(acts).astype(np.float32)


def _train_epoch(model: F9Net, optimizer: torch.optim.Optimizer, items: Mapping[str, Mapping[str, Any]], ids: Sequence[str],
                 y: Mapping[str, int], weights: torch.Tensor, rng: np.random.Generator, cfg: Mapping[str, Any],
                 device: torch.device, n_classes: int) -> float:
    batch_size = int(cfg["batch"]["effective_patients"])
    micro = max(1, int(cfg["batch"].get("micro_batch_patients", batch_size)))
    spacing = float(cfg["input"]["inplane_spacing_mm"])
    clip = float(cfg["gradient_clip_norm"])
    eye = torch.eye(n_classes, dtype=torch.float32)
    order = rng.permutation(len(ids))
    model.train()
    loss_sum, weight_sum = 0.0, 0.0
    trainable = [p for p in model.parameters() if p.requires_grad]
    for start in range(0, len(ids), batch_size):
        chosen = [ids[i] for i in order[start:start + batch_size]]
        batch = augment_batch(batch_to(collate_patients([items[i] for i in chosen]), device), rng, cfg["augmentation"], spacing_mm=spacing)
        onehot = eye[torch.tensor([y[i] for i in chosen], dtype=torch.long)].to(device)
        w_y = (onehot * weights).sum(dim=1)
        total = w_y.sum()
        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0
        for m in range(0, len(chosen), micro):
            sl = slice(m, m + micro)
            sub = {k: v[sl] for k, v in batch.items()}
            logits, _ = model.forward_batch(sub)
            logp = torch.log_softmax(logits, dim=1)
            loss = -(w_y[sl] * (onehot[sl] * logp).sum(dim=1)).sum() / total
            loss.backward()
            batch_loss += float(loss.detach().to("cpu"))
        torch.nn.utils.clip_grad_norm_(trainable, clip)
        optimizer.step()
        loss_sum += batch_loss * float(total.detach().to("cpu"))
        weight_sum += float(total.detach().to("cpu"))
    return loss_sum / weight_sum


# ----------------------------------------------------------------------------- persistence


def _atomic_torch_save(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(obj, tmp)
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path


def _completed(out: Path, spec: FitSpec) -> FitRecord | None:
    path = out / "record.json"
    if not path.is_file():
        return None
    record = FitRecord.load(path)
    if record.fit_id != spec.fit_id:
        raise ValueError(f"{path} belongs to fit {record.fit_id}, not {spec.fit_id}")
    if record.status == STATUS_COMPLETE:
        digest = sha256_file(out / str(record.checkpoint_file))
        if digest != record.checkpoint_sha256:
            raise ValueError(f"{out / str(record.checkpoint_file)}: SHA-256 differs from record.json; the fit directory is corrupt")
    return record


def _bump_attempts(out: Path) -> int:
    path = out / "attempts.json"
    try:
        count = int(json.loads(path.read_text(encoding="utf-8")).get("attempts", 0))
    except (OSError, ValueError):
        count = 0
    atomic_write_json(path, {"attempts": count + 1, "last_start_utc": _utc()})
    return count + 1


def _rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {"torch_cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: Mapping[str, Any], device: torch.device) -> None:
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda" and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


# ----------------------------------------------------------------------------- entry point


def train_fit(spec: FitSpec | Mapping[str, Any], cache_dir: str | Path, out_dir: str | Path, device: Any = "auto", *,
              config: Mapping[str, Any] | None = None, weights: WeightsRef | Mapping[str, Any] | None = None,
              expected_manifest_sha256: str | None = None, backend: Mapping[str, Any] | None = None,
              on_checkpoint: Callable[[], Any] | None = None, stop_after_epoch: int | None = None) -> FitRecord:
    """Train one fit (see the module docstring) and return its :class:`FitRecord`.

    ``cache_dir`` contains ``acdc_dev/manifest.csv``; outputs go to ``out_dir`` (``model.pt``, ``record.json``,
    ``resume/`` while running).  ``config`` is the training configuration (default ``configs/f9_cnn.yaml`` without
    ``backend``).  ``on_checkpoint`` runs after every epoch checkpoint (Modal: ``Volume.commit``).  ``stop_after_epoch``
    simulates a preemption after that epoch's checkpoint (qualification and tests): the call returns an
    ``INTERRUPTED`` record without writing ``record.json``, and the next call resumes."""

    spec = coerce_spec(spec)
    if config is None:
        from .f9_config import load_f9_config, training_config

        config = training_config(load_f9_config())
    cfg = dict(config)
    cfg_sha = sha256_json(cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = _completed(out, spec)
    if done is not None:
        return done
    t_start, started_utc = time.perf_counter(), _utc()
    attempts = _bump_attempts(out)
    configure_numerics()                          # before any CUDA call (CUBLAS_WORKSPACE_CONFIG); asserts IEEE FP32
    dev = resolve_device(device)
    if weights is None:
        from .f9_config import resolve_weights

        weights = resolve_weights(cfg)
    classes = list(cfg["class_order"])
    index = {c: i for i, c in enumerate(classes)}
    partition = str(cfg.get("training_partition", TRAINING_PARTITION))
    if partition != TRAINING_PARTITION:
        raise ValueError(f"F9 fits train on {TRAINING_PARTITION} only, configuration names {partition}")
    cache = Path(cache_dir) / TRAINING_PARTITION
    manifest_path = cache / "manifest.csv"
    manifest_sha = sha256_file(manifest_path)
    if expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256:
        raise ValueError(f"{manifest_path}: SHA-256 {manifest_sha} differs from the spec file ({expected_manifest_sha256})")
    manifest = read_manifest(manifest_path)
    rows = manifest[manifest["sample_id"].isin(spec.fitting_patients)]
    absent = sorted(set(spec.fitting_patients) - set(rows["sample_id"]))
    if absent:
        raise ValueError(f"{len(absent)} fitting patients are not in {manifest_path} (e.g. {absent[:3]})")
    if set(rows["partition"]) != {TRAINING_PARTITION}:
        raise ValueError(f"fitting patients must all belong to {TRAINING_PARTITION}; found {sorted(set(rows['partition']))}")
    unlabeled = sorted(rows.loc[~rows["label"].isin(classes), "sample_id"])
    if unlabeled:
        raise ValueError(f"fitting patients without a class label in {classes}: {unlabeled[:3]}")
    tta = str((cfg.get("inference") or {}).get("tta", D4_TTA))
    if tta != D4_TTA:
        raise ValueError(f"configs/f9_cnn.yaml inference.tta must be {D4_TTA!r} (6.0.1, owner issue O1), got {tta!r}")
    record = FitRecord(fit_id=spec.fit_id, spec=spec.to_dict(), status=STATUS_COMPLETE, config_sha256=cfg_sha, class_order=classes,
                       backend=dict(backend or {"name": "local"}), tta=tta,
                       architecture={"dropout": float(cfg["architecture"]["bottleneck"]["dropout_after_first_gelu"]),
                                     "channel_mean": list(cfg["input"]["channel_mean"]), "channel_sd": list(cfg["input"]["channel_sd"])})
    record.environment = environment_record(dev)
    try:
        if spec.is_selection_fit:
            folds = selection_folds(rows, n_folds=int(cfg["selection"]["n_folds"]), classes=classes, salt=str(cfg["selection"]["salt"]))
            val_ids = sorted(folds.loc[folds["selection_fold"] == spec.selection_fold, "sample_id"])
            train_ids = sorted(folds.loc[folds["selection_fold"] != spec.selection_fold, "sample_id"])
        else:
            train_ids, val_ids = sorted(rows["sample_id"]), []
        lookup = dict(zip(rows["sample_id"], rows["label"]))
        record.support_train = require_class_support([lookup[i] for i in train_ids], classes, where="training patients")
        if val_ids:
            record.support_val = require_class_support([lookup[i] for i in val_ids], classes, where="validation patients")
        cw = class_weights([lookup[i] for i in train_ids], classes)
    except InsufficientClassSupport as exc:
        record.status, record.reason = NE_INSUFFICIENT_CLASS_SUPPORT, str(exc)
        record.timings = {"started_utc": started_utc, "finished_utc": _utc(), "seconds_total": round(time.perf_counter() - t_start, 3)}
        record.write(out / "record.json")
        return record
    record.train_ids, record.val_ids, record.class_weights = train_ids, val_ids, [float(v) for v in cw]

    shas = dict(zip(manifest["sample_id"], manifest["npz_sha256"]))
    verify = bool(cfg["input"].get("verify_npz_sha256", True))
    rows_by_id = {str(r["sample_id"]): r for r in manifest.to_dict("records")}
    items = {sid: load_npz(npz_path(cache, rows_by_id[sid]), expected_sha256=shas[sid] if verify else None) for sid in train_ids + val_ids}
    wref = WeightsRef.coerce(weights)
    if not wref.sha256:
        raise ValueError("the backbone weights SHA-256 is not pinned (registry/weights.yaml); training refuses")
    record.inputs = {"partition": TRAINING_PARTITION, "cache_manifest_sha256": manifest_sha,
                     "npz_sha256": {sid: shas[sid] for sid in sorted(items)}, "weights": {"name": wref.name, "sha256": wref.sha256}}

    model = build_model(spec.candidate, wref.path, seed=spec.seed, expected_sha256=wref.sha256, dropout=record.architecture["dropout"],
                        channel_mean=record.architecture["channel_mean"], channel_sd=record.architecture["channel_sd"]).to(dev)
    initial = cpu_state(model)
    record.parameters["initial_state_sha256"] = state_sha256(initial)
    optimizer = torch.optim.AdamW(model.parameter_groups(spec.candidate.lr_backbone, float(cfg["new_layer_lr_multiplier"])),
                                  lr=spec.candidate.lr_backbone, betas=tuple(cfg["optimizer"]["betas"]), eps=float(cfg["optimizer"]["eps"]),
                                  weight_decay=float(cfg["optimizer"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["schedule"]["t_max_epochs"]),
                                                           eta_min=float(cfg["schedule"]["eta_min"]))
    selection = spec.is_selection_fit
    max_epochs = int(cfg["epochs"]["max"]) if selection else int(spec.refit_epochs)
    if max_epochs > int(cfg["schedule"]["t_max_epochs"]):
        raise ValueError(f"{max_epochs} epochs exceed the {cfg['schedule']['t_max_epochs']}-epoch schedule")
    record.max_epochs = max_epochs
    stopper = EarlyStopper.from_config(cfg) if selection else None
    history: list[dict[str, Any]] = []
    start_epoch, resumed_from = 1, None
    best_state: dict[str, torch.Tensor] | None = None
    resume_dir = out / "resume"
    if (resume_dir / "last.pt").is_file():
        saved = torch.load(resume_dir / "last.pt", map_location="cpu", weights_only=False)
        if saved.get("schema") != RESUME_SCHEMA or saved.get("fit_id") != spec.fit_id or saved.get("config_sha256") != cfg_sha:
            raise ValueError(f"{resume_dir / 'last.pt'} belongs to another fit or configuration")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history = list(saved["history"])
        if stopper is not None:
            stopper = EarlyStopper.from_state(saved["early_stopping"])
            if (resume_dir / "best.pt").is_file():
                best_state = torch.load(resume_dir / "best.pt", map_location="cpu", weights_only=True)
        _restore_rng(saved["rng"], dev)
        start_epoch = resumed_from = int(saved["epoch"]) + 1
        if saved.get("stopped"):
            start_epoch = max_epochs + 1
    weights_t = torch.tensor(cw, dtype=torch.float32, device=dev)
    y_index = {sid: index[lookup[sid]] for sid in train_ids + val_ids}
    val_items = [items[i] for i in val_ids]
    y_val = np.array([y_index[i] for i in val_ids], dtype=np.int64)
    stopped = bool(stopper.stopped) if stopper is not None and resumed_from else False

    for epoch in range(start_epoch, max_epochs + 1):
        t_epoch = time.perf_counter()
        torch.manual_seed(epoch_seed(spec.seed, epoch))
        rng = np.random.default_rng([int(spec.seed), int(epoch)])
        lrs = [float(g["lr"]) for g in optimizer.param_groups]
        train_loss = _train_epoch(model, optimizer, items, train_ids, y_index, weights_t, rng, cfg, dev, len(classes))
        scheduler.step()
        entry: dict[str, Any] = {"epoch": epoch, "lr_backbone": lrs[0], "lr_new": lrs[1], "train_loss": train_loss}
        if selection:
            logits, _ = predict_logits(model, val_items, dev, tta=tta)
            summary = classification_summary(logits, y_val, classes)
            entry.update({"val_cross_entropy": summary["cross_entropy"], "val_macro_f1": summary["macro_f1"], "val_accuracy": summary["accuracy"]})
            improved, stopped = stopper.update(epoch, summary["cross_entropy"])
            if improved:
                best_state = cpu_state(model)
                _atomic_torch_save(best_state, resume_dir / "best.pt")
        entry["seconds"] = round(time.perf_counter() - t_epoch, 3)
        history.append(entry)
        _atomic_torch_save({"schema": RESUME_SCHEMA, "fit_id": spec.fit_id, "config_sha256": cfg_sha, "epoch": epoch,
                            "model": cpu_state(model), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                            "early_stopping": stopper.state() if stopper is not None else None, "stopped": bool(stopped),
                            "history": history, "rng": _rng_state(dev)}, resume_dir / "last.pt")
        if on_checkpoint is not None:
            on_checkpoint()
        if stopped:
            break
        if stop_after_epoch is not None and epoch >= int(stop_after_epoch) and epoch < max_epochs:
            record.status, record.epochs = STATUS_INTERRUPTED, history
            record.resume = {"attempts": attempts, "resumed_from_epoch": resumed_from, "interrupted_after_epoch": epoch}
            return record

    record.epochs = history
    if selection:
        if best_state is None:
            raise RuntimeError(f"fit {spec.fit_id}: no finite validation cross entropy in {len(history)} epochs")
        model.load_state_dict(best_state)
        record.selected_epoch = int(stopper.best_epoch)
        record.stopped_epoch = int(history[-1]["epoch"])
        record.stop_reason = "early_stopping" if stopper.stopped else "max_epochs"
        logits, _ = predict_logits(model, val_items, dev, tta=tta)
        record.val_metrics = classification_summary(logits, y_val, classes)
        probs = np.exp(_log_softmax64(logits))
        record.val_predictions = [{"sample_id": sid, "y_true": classes[int(y_val[j])], "y_pred": classes[int(probs[j].argmax())],
                                   **{f"prob_{c}": float(probs[j, k]) for k, c in enumerate(classes)},
                                   **{f"logit_{c}": float(logits[j, k]) for k, c in enumerate(classes)}} for j, sid in enumerate(val_ids)]
    else:
        record.selected_epoch = record.stopped_epoch = int(history[-1]["epoch"])
        record.stop_reason = "refit_epochs"
    final = cpu_state(model)
    record.parameters.update(_parameter_checks(initial, final, model))
    checkpoint = _atomic_torch_save(final, out / "model.pt")
    record.checkpoint_file, record.checkpoint_sha256, record.state_sha256 = "model.pt", sha256_file(checkpoint), state_sha256(final)
    record.timings = {"started_utc": started_utc, "finished_utc": _utc(), "seconds_total": round(time.perf_counter() - t_start, 3),
                      "seconds_per_epoch": [e["seconds"] for e in history]}
    record.resume = {"attempts": attempts, "resumed_from_epoch": resumed_from}
    record.write(out / "record.json")
    if not bool(cfg.get("outputs", {}).get("keep_resume_state_after_completion", False)):
        shutil.rmtree(resume_dir, ignore_errors=True)
    if on_checkpoint is not None:
        on_checkpoint()
    return record


def _parameter_checks(initial: Mapping[str, torch.Tensor], final: Mapping[str, torch.Tensor], model: F9Net) -> dict[str, Any]:
    """Nonzero change in a convolution of every unfrozen block; frozen weights and BN running statistics unchanged."""

    changes = {}
    for block in model.unfrozen_blocks:
        key = CHANGE_WATCH[block]
        changes[key] = float((final[key] - initial[key]).abs().max())
    running = [k for k in initial if k.startswith("backbone.") and k.endswith(("running_mean", "running_var"))]
    bn_unchanged = all(torch.equal(initial[k], final[k]) for k in running)
    frozen_unchanged = torch.equal(initial[FROZEN_WATCH], final[FROZEN_WATCH])
    if min(changes.values()) <= 0.0:
        raise RuntimeError(f"no parameter change in an unfrozen convolutional block: {changes}")
    if not (bn_unchanged and frozen_unchanged):
        raise RuntimeError(f"frozen parameters or BN running statistics changed (bn {bn_unchanged}, frozen {frozen_unchanged})")
    return {"final_state_sha256": state_sha256(final), "unfrozen_conv_max_abs_change": changes,
            "bn_running_statistics_unchanged": bn_unchanged, "frozen_weights_unchanged": frozen_unchanged,
            "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "backbone_trainable_parameters": int(sum(p.numel() for p in model.backbone_trainable_parameters()))}
