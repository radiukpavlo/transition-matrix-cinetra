"""F9-M: F9 with three reference-mask channels (protocol-v6.0.2 analysis B22; ``configs/exploratory_postfreeze.yaml``
``B22``; SPECIFICATION.md 13.7, Decision D39).  Exploratory comparator only; F9 itself (:mod:`tmmri.neural.model`,
:mod:`~tmmri.neural.train`, :mod:`~tmmri.neural.data`) is not changed, and with the mask path off every function here
reproduces F9 exactly.

* **Mask input** (:func:`mask_label_slices`): the staged reference mask (staging label map: 1 LV cavity, 2 myocardium,
  3 RV cavity) on exactly the grid of the C3 image tensor (:func:`tmmri.neural.preprocess.preprocess_case`): the same
  canonical orientation of the ED image geometry, the same physical 1.5 mm grid centred on the field of view and the
  same zero padding outside it (:func:`tmmri.neural.preprocess.axis_weights`), with the nearest source pixel
  ``clip(floor(u + 0.5), 0, n - 1)`` in place of the bilinear pair.  Stored as uint8 label slices ``[n, 224, 224]``
  and expanded to three binary channels (values 0 or 1, never ImageNet-normalized) by :func:`collate_with_masks`.
* **Stem** (:class:`F9MNet`): ``conv1`` output = the frozen pretrained ``conv1`` on the three image channels + a new
  3 -> 64 7 x 7 convolution with conv1's stride and padding on the mask channels (:class:`MaskStem`; no bias, created
  at zero without drawing from the torch RNG, so every other layer is initialized exactly as F9), then ``bn1`` and
  everything after it unchanged.  The mask stem trains in F9's new-layer group (10 x the backbone rate, AdamW weight
  decay as every trainable parameter).
* **Augmentation** (:func:`augment_with_masks`): :func:`tmmri.neural.data.augment_batch` unchanged for the images (the
  same RNG stream as F9), then the same per-patient D4 element and rotation/translation replayed from a copy of the RNG
  state and applied to the mask channels by nearest-neighbour ``grid_sample`` with zero padding (no intensity factor,
  no noise).
* **Inference** (:func:`f9m_forward_tta`): the D4 test-time symmetrization of F9 with the same element applied to the
  images and the masks.
* **Refits** (:func:`train_refit`): the refit path of :func:`tmmri.neural.train.train_fit` (per-epoch seeds, AdamW
  groups, cosine schedule, clipping, class weights, batches, parameter checks, ``record.json`` + ``model.pt``, resume
  state removed at completion) on in-memory patients; with ``mask_channels=False`` it reproduces ``train_fit``.
"""

from __future__ import annotations

import copy
import json
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..runs import sha256_file, sha256_json
from .data import (
    _theta,
    augment_batch,
    augmentation_parameters,
    batch_to,
    collate_patients,
)
from .dihedral import D4_ORDER, D4_TTA, d4_apply, tta_elements, uses_d4_augmentation
from .fit_spec import Candidate, FitSpec, coerce_spec
from .model import (
    IMAGENET_MEAN,
    IMAGENET_SD,
    F9Net,
    _check_prefix,
    cpu_state,
    file_sha256,
    state_sha256,
)
from .numerics import configure_numerics, environment_record
from .selection import (
    NE_INSUFFICIENT_CLASS_SUPPORT,
    STATUS_COMPLETE,
    InsufficientClassSupport,
    class_weights,
    require_class_support,
)
from .train import (
    RESUME_SCHEMA,
    STATUS_INTERRUPTED,
    FitRecord,
    WeightsRef,
    _atomic_torch_save,
    _bump_attempts,
    _completed,
    _parameter_checks,
    _restore_rng,
    _rng_state,
    _utc,
    epoch_seed,
    resolve_device,
)

MODEL_ID = "f9m"
MASK_STRUCTURES = (1, 2, 3)                      # LV cavity, myocardium, RV cavity (v6 labels after the staging label map)
SEG_KEYS = ("ed_seg", "es_seg")
LABEL_KEYS = ("ed_labels", "es_labels")
MASK_STEM_KEY = "mask_stem.weight"
NEAREST_RULE = ("nearest source pixel clip(floor(u + 0.5), 0, n - 1) at the positions u of tmmri.neural.preprocess.axis_weights "
                "(1.5 mm grid centred on the field of view); zero outside [-0.5, n - 0.5]")


# ----------------------------------------------------------------------------- mask input on the C3 grid


def nearest_axis_index(n_source: int, spacing_mm: float, target_mm: float, size: int) -> tuple[np.ndarray, np.ndarray]:
    """``(index, inside)`` of the nearest source pixel for every output pixel of the grid of :func:`axis_weights`."""

    j = np.arange(int(size), dtype=np.float64)
    u = (j - (size - 1) / 2.0) * (float(target_mm) / float(spacing_mm)) + (n_source - 1) / 2.0
    inside = (u >= -0.5) & (u <= n_source - 0.5)
    index = np.clip(np.floor(u + 0.5), 0, n_source - 1).astype(np.int64)
    return index, inside


def resample_labels(slices: np.ndarray, spacing_rc: Sequence[float], target_mm: float, size_rc: Sequence[int]) -> np.ndarray:
    """Nearest-neighbour counterpart of :func:`tmmri.neural.preprocess.resample_slices` for label slices ``(n, r, c)``
    -> uint8 ``(n, H, W)`` (zero outside the source field of view)."""

    arr = np.asarray(slices)
    ir, inside_r = nearest_axis_index(arr.shape[1], spacing_rc[0], target_mm, size_rc[0])
    ic, inside_c = nearest_axis_index(arr.shape[2], spacing_rc[1], target_mm, size_rc[1])
    out = arr[:, ir][:, :, ic]
    out = np.where(inside_r[None, :, None] & inside_c[None, None, :], out, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def mask_label_slices(case: Any, masks: Mapping[str, np.ndarray], cfg: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """``{"ed_labels", "es_labels"}``: the reference masks of ``case`` on the C3 grid of ``preprocess_case``."""

    from .preprocess import (
        InputRejected,
        canonical_orientation,
        check_phase_pair,
        oriented_slices,
    )

    icfg = cfg["input"]
    target = float(icfg["in_plane_spacing_mm"])
    size = tuple(int(v) for v in icfg["size_pixels"])
    check_phase_pair(case.ed_geometry, case.es_geometry, cfg)
    orientation = canonical_orientation(case.ed_geometry)
    out: dict[str, np.ndarray] = {}
    for phase in ("ed", "es"):
        geom = getattr(case, f"{phase}_geometry")
        mask = np.rint(np.asarray(masks[phase])).astype(np.int64)
        if tuple(mask.shape) != tuple(geom.shape):
            raise InputRejected("INPUT_DECODE_FAILURE", f"{phase}: mask shape {mask.shape} does not match the image geometry {geom.shape}")
        out[f"{phase}_labels"] = resample_labels(oriented_slices(mask, orientation), orientation.spacing(geom.spacing_mm), target, size)
    return out


def label_channels(labels: np.ndarray, structures: Sequence[int] = MASK_STRUCTURES) -> np.ndarray:
    """``[n, H, W]`` labels -> ``[n, 3, H, W]`` float32 binary channels (LV cavity, myocardium, RV cavity)."""

    arr = np.asarray(labels)
    return np.stack([(arr == int(k)) for k in structures], axis=1).astype(np.float32)


# ----------------------------------------------------------------------------- batches and augmentation


def collate_with_masks(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """:func:`tmmri.neural.data.collate_patients` plus ``ed_seg/es_seg [B, S, 3, H, W]`` (zero padding)."""

    out = collate_patients(items)
    for phase in ("ed", "es"):
        images = out[phase]
        seg = torch.zeros((images.shape[0], images.shape[1], len(MASK_STRUCTURES), images.shape[2], images.shape[3]), dtype=torch.float32)
        for i, item in enumerate(items):
            labels = np.asarray(item[f"{phase}_labels"])
            if labels.shape != tuple(np.asarray(item[phase]).shape):
                raise ValueError(f"{phase}_labels {labels.shape} differ from the {phase} images {np.asarray(item[phase]).shape}")
            seg[i, :labels.shape[0]] = torch.from_numpy(label_channels(labels))
        out[f"{phase}_seg"] = seg
    return out


def augment_with_masks(batch: Mapping[str, Any], rng: np.random.Generator, cfg: Mapping[str, Any], *, spacing_mm: float) -> dict[str, Any]:
    """F9's augmentation of the images (``augment_batch`` unchanged) and the same spatial transform of the masks."""

    state = copy.deepcopy(rng.bit_generator.state)
    out = augment_batch(batch, rng, cfg, spacing_mm=spacing_mm)
    replay = np.random.Generator(type(rng.bit_generator)())
    replay.bit_generator.state = state
    ed = batch["ed"]
    size = (int(ed.shape[-2]), int(ed.shape[-1]))
    dihedral = uses_d4_augmentation(cfg)
    thetas, elements = [], []
    for i in range(int(ed.shape[0])):                             # the draws of augment_batch, in its order
        elements.append(int(replay.integers(D4_ORDER)) if dihedral else 0)
        params = augmentation_parameters(replay, cfg)
        thetas.append(_theta(params, spacing_mm, size))
        for phase in ("ed", "es"):
            n = int(batch[f"{phase}_mask"][i].sum().item())
            replay.standard_normal((n, *size), dtype=np.float32)
    if replay.bit_generator.state != rng.bit_generator.state:
        raise RuntimeError("mask augmentation replay diverged from augment_batch's random stream")
    theta = torch.from_numpy(np.stack(thetas)).to(ed.device)
    with torch.no_grad():
        for phase in ("ed", "es"):
            seg = batch[f"{phase}_seg"]
            if any(elements):
                seg = torch.stack([d4_apply(seg[i], g) for i, g in enumerate(elements)])
            b, s, c = int(seg.shape[0]), int(seg.shape[1]), int(seg.shape[2])
            flat = seg.reshape(b * s, c, *size)
            grid = F.affine_grid(theta.repeat_interleave(s, dim=0), [b * s, c, *size], align_corners=False)
            moved = F.grid_sample(flat, grid, mode="nearest", padding_mode="zeros", align_corners=False).reshape(b, s, c, *size)
            out[f"{phase}_seg"] = moved * batch[f"{phase}_mask"].to(moved.dtype).view(b, s, 1, 1, 1)
    return out


# ----------------------------------------------------------------------------- network


class MaskStem(nn.Module):
    """3 -> 64 convolution with ``conv1``'s kernel, stride and padding; no bias; zero at creation (no RNG draw)."""

    def __init__(self, conv1: nn.Conv2d, in_channels: int = len(MASK_STRUCTURES)) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(int(conv1.out_channels), int(in_channels), *conv1.kernel_size))
        self.stride, self.padding, self.dilation = conv1.stride, conv1.padding, conv1.dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, None, self.stride, self.padding, self.dilation)


class F9MNet(F9Net):
    """``forward(ed, es, ed_geom, es_geom, ed_mask, es_mask, ed_seg=None, es_seg=None) -> (logits, a)``; F9 when the
    mask path is off (``mask_channels=False``: no extra parameter, every call delegated to :class:`F9Net`)."""

    def __init__(self, backbone: nn.Module, unfreeze: str, *, mask_channels: bool = True, **kwargs: Any) -> None:
        super().__init__(backbone, unfreeze, **kwargs)
        self.mask_channels = bool(mask_channels)
        self.mask_stem = MaskStem(self.backbone.conv1) if self.mask_channels else None

    def new_modules(self) -> list[nn.Module]:
        return super().new_modules() + ([self.mask_stem] if self.mask_channels else [])

    def _stem_and_trunk(self, slices: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        b = self.backbone
        x = b.conv1(slices) + self.mask_stem(seg)
        x = b.maxpool(b.relu(b.bn1(x)))
        x = b.layer4(b.layer3(b.layer2(b.layer1(x))))
        return torch.flatten(b.avgpool(x), 1)

    def _encode(self, ed: torch.Tensor, es: torch.Tensor, ed_geom: torch.Tensor, es_geom: torch.Tensor, ed_mask: torch.Tensor,
                es_mask: torch.Tensor, ed_seg: torch.Tensor | None = None, es_seg: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.mask_channels:
            return super()._encode(ed, es, ed_geom, es_geom, ed_mask, es_mask)
        if ed_seg is None or es_seg is None:
            raise ValueError("F9-M needs the mask channels ed_seg and es_seg")
        counts_ed, counts_es = _check_prefix(ed_mask), _check_prefix(es_mask)
        pieces = [ed[b, :n] for b, n in enumerate(counts_ed)] + [es[b, :n] for b, n in enumerate(counts_es)]
        geoms = [ed_geom[b, :n] for b, n in enumerate(counts_ed)] + [es_geom[b, :n] for b, n in enumerate(counts_es)]
        segs = [ed_seg[b, :n] for b, n in enumerate(counts_ed)] + [es_seg[b, :n] for b, n in enumerate(counts_es)]
        slices = torch.cat(pieces, dim=0).unsqueeze(1).expand(-1, 3, -1, -1)
        slices = (slices - self.channel_mean) / self.channel_sd
        pooled = self._stem_and_trunk(slices, torch.cat(segs, dim=0).to(slices.dtype))
        h = F.gelu(self.slice_proj(torch.cat([pooled, torch.cat(geoms, dim=0).to(pooled.dtype)], dim=1)))
        n_ed = sum(counts_ed)
        return (self._pad(h[:n_ed], counts_ed, int(ed_mask.shape[1])), self._pad(h[n_ed:], counts_es, int(es_mask.shape[1])))

    def forward(self, ed: torch.Tensor, es: torch.Tensor, ed_geom: torch.Tensor, es_geom: torch.Tensor, ed_mask: torch.Tensor,
                es_mask: torch.Tensor, ed_seg: torch.Tensor | None = None, es_seg: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.mask_channels:
            return super().forward(ed, es, ed_geom, es_geom, ed_mask, es_mask)
        h_ed, h_es = self._encode(ed, es, ed_geom, es_geom, ed_mask, es_mask, ed_seg, es_seg)
        v_ed, v_es = self._pool(h_ed, ed_mask), self._pool(h_es, es_mask)
        z = torch.cat([v_ed, v_es, v_ed - v_es], dim=1)
        a = F.gelu(self.fc2(self.dropout(F.gelu(self.fc1(z)))))
        return self.head(a), a

    def forward_batch(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.mask_channels:
            return super().forward_batch(batch)
        return self(batch["ed"], batch["es"], batch["ed_geom"], batch["es_geom"], batch["ed_mask"], batch["es_mask"], batch["ed_seg"], batch["es_seg"])


def build_f9m(candidate: Candidate | Mapping[str, Any], weights_path: str | Path | None, *, seed: int | None = None,
              expected_sha256: str | None = None, dropout: float = 0.25, channel_mean: Sequence[float] = IMAGENET_MEAN,
              channel_sd: Sequence[float] = IMAGENET_SD, mask_channels: bool = True) -> F9MNet:
    """:func:`tmmri.neural.model.build_model` for :class:`F9MNet` (identical backbone loading and seeding)."""

    import torchvision

    cand = Candidate.coerce(candidate)
    backbone = torchvision.models.resnet18(weights=None)
    if weights_path is not None:
        if expected_sha256 is not None:
            digest = file_sha256(weights_path)
            if digest != str(expected_sha256).lower():
                raise ValueError(f"backbone weights {weights_path}: SHA-256 {digest} differs from the pinned {expected_sha256}")
        backbone.load_state_dict(torch.load(Path(weights_path), map_location="cpu", weights_only=True))
    return F9MNet(backbone, cand.unfreeze, dropout=dropout, channel_mean=channel_mean, channel_sd=channel_sd, seed=seed,
                  mask_channels=mask_channels)


def load_f9m(checkpoint_dir: str | Path, map_location: Any = "cpu") -> F9MNet:
    """``<checkpoint_dir>/model.pt`` + ``record.json`` of :func:`train_refit` -> F9MNet in eval mode (SHA-256 verified)."""

    directory = Path(checkpoint_dir)
    record = json.loads((directory / "record.json").read_text(encoding="utf-8"))
    weights = directory / "model.pt"
    digest = file_sha256(weights)
    if record.get("checkpoint_sha256") != digest:
        raise ValueError(f"{weights}: SHA-256 {digest} differs from record.json ({record.get('checkpoint_sha256')})")
    spec = FitSpec.from_dict(record["spec"])
    if spec.fit_id != record.get("fit_id"):
        raise ValueError(f"{directory}: record fit_id {record.get('fit_id')} does not match its spec ({spec.fit_id})")
    arch = record.get("architecture", {})
    model = build_f9m(spec.candidate, None, dropout=float(arch.get("dropout", 0.25)), channel_mean=arch.get("channel_mean", IMAGENET_MEAN),
                      channel_sd=arch.get("channel_sd", IMAGENET_SD), mask_channels=bool(arch.get("mask_channels", True)))
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.f9m_record = record  # type: ignore[attr-defined]
    return model.to(map_location).eval()


def f9m_forward_tta(model: F9MNet, batch: Mapping[str, Any], *, tta: str = D4_TTA) -> tuple[torch.Tensor, torch.Tensor]:
    """(logits, activations) as CPU Float64: the mean over the elements of ``tta`` of the FP32 outputs on the transformed
    images and masks (same element for both; :func:`tmmri.neural.dihedral.f9_forward_tta` with masks)."""

    logit_sum = act_sum = None
    elements = tta_elements(tta)
    for g in elements:
        segs = [d4_apply(batch[k], g) for k in SEG_KEYS] if model.mask_channels else [None, None]
        logits, acts = model(d4_apply(batch["ed"], g), d4_apply(batch["es"], g), batch["ed_geom"], batch["es_geom"], batch["ed_mask"],
                             batch["es_mask"], *segs)
        logits = logits.detach().to(device="cpu", dtype=torch.float32).to(torch.float64)
        acts = acts.detach().to(device="cpu", dtype=torch.float32).to(torch.float64)
        logit_sum = logits if logit_sum is None else logit_sum + logits
        act_sum = acts if act_sum is None else act_sum + acts
    n = float(len(elements))
    return logit_sum / n, act_sum / n


@torch.no_grad()
def predict_patients(model: F9MNet, items: Sequence[Mapping[str, Any]], device: Any, *, tta: str = D4_TTA) -> tuple[np.ndarray, np.ndarray]:
    """Float64 TTA logits ``[n, 5]`` and activations ``[n, 64]`` in input order, one patient per forward pass (as
    :func:`tmmri.neural.inference.run_f9_inference`)."""

    model.eval()
    collate = collate_with_masks if model.mask_channels else collate_patients
    logits, acts = [], []
    for item in items:
        out, a = f9m_forward_tta(model, batch_to(collate([item]), device), tta=tta)
        logits.append(out.numpy().reshape(-1))
        acts.append(a.numpy().reshape(-1))
    return np.vstack(logits), np.vstack(acts)


# ----------------------------------------------------------------------------- refits


def _train_epoch(model: F9MNet, optimizer: torch.optim.Optimizer, items: Mapping[str, Mapping[str, Any]], ids: Sequence[str],
                 y: Mapping[str, int], weights: torch.Tensor, rng: np.random.Generator, cfg: Mapping[str, Any], device: torch.device,
                 n_classes: int, *, collate: Callable[..., Any], augment: Callable[..., Any]) -> float:
    """:func:`tmmri.neural.train._train_epoch` with the collate and augmentation functions as parameters."""

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
        batch = augment(batch_to(collate([items[i] for i in chosen]), device), rng, cfg["augmentation"], spacing_mm=spacing)
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


def train_refit(spec: FitSpec | Mapping[str, Any], items: Mapping[str, Mapping[str, Any]], labels: Mapping[str, str], out_dir: str | Path,
                device: Any = "auto", *, config: Mapping[str, Any], weights: WeightsRef | Mapping[str, Any], mask_channels: bool = True,
                inputs: Mapping[str, Any] | None = None, backend: Mapping[str, Any] | None = None,
                stop_after_epoch: int | None = None) -> FitRecord:
    """One refit (``spec.refit_epochs`` epochs on all fitting patients, nothing monitored) of F9-M (or of F9 with
    ``mask_channels=False``) on in-memory patients ``items`` (sample id -> C3 arrays plus ``ed_labels``/``es_labels``).
    Outputs as :func:`tmmri.neural.train.train_fit`: ``model.pt``, ``record.json``; a completed directory is returned
    as is and an interrupted one resumes from ``resume/last.pt``."""

    spec = coerce_spec(spec)
    if spec.is_selection_fit:
        raise ValueError(f"train_refit trains refits only; {spec.fit_id} is a selection fit")
    cfg = dict(config)
    cfg_sha = sha256_json(cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = _completed(out, spec)
    if done is not None:
        return done
    t_start, started_utc = time.perf_counter(), _utc()
    attempts = _bump_attempts(out)
    configure_numerics()
    dev = resolve_device(device)
    classes = list(cfg["class_order"])
    index = {c: i for i, c in enumerate(classes)}
    tta = str((cfg.get("inference") or {}).get("tta", D4_TTA))
    if tta != D4_TTA:
        raise ValueError(f"configs/f9_cnn.yaml inference.tta must be {D4_TTA!r}, got {tta!r}")
    collate, augment = (collate_with_masks, augment_with_masks) if mask_channels else (collate_patients, augment_batch)
    record = FitRecord(fit_id=spec.fit_id, spec=spec.to_dict(), status=STATUS_COMPLETE, config_sha256=cfg_sha, class_order=classes,
                       backend=dict(backend or {"name": "local"}), tta=tta,
                       architecture={"dropout": float(cfg["architecture"]["bottleneck"]["dropout_after_first_gelu"]),
                                     "channel_mean": list(cfg["input"]["channel_mean"]), "channel_sd": list(cfg["input"]["channel_sd"]),
                                     "model_id": MODEL_ID if mask_channels else "f9", "mask_channels": bool(mask_channels),
                                     "mask_stem": "3->64 7x7, conv1 stride and padding, no bias, zero init, new-layer rate" if mask_channels else None})
    record.environment = environment_record(dev)
    train_ids = sorted(spec.fitting_patients)
    missing = [s for s in train_ids if s not in items or s not in labels]
    if missing:
        raise ValueError(f"{len(missing)} fitting patients have no inputs or labels (e.g. {missing[:3]})")
    unlabeled = [s for s in train_ids if labels[s] not in classes]
    if unlabeled:
        raise ValueError(f"fitting patients without a class label in {classes}: {unlabeled[:3]}")
    try:
        record.support_train = require_class_support([labels[i] for i in train_ids], classes, where="training patients")
        cw = class_weights([labels[i] for i in train_ids], classes)
    except InsufficientClassSupport as exc:
        record.status, record.reason = NE_INSUFFICIENT_CLASS_SUPPORT, str(exc)
        record.timings = {"started_utc": started_utc, "finished_utc": _utc(), "seconds_total": round(time.perf_counter() - t_start, 3)}
        record.write(out / "record.json")
        return record
    record.train_ids, record.val_ids, record.class_weights = train_ids, [], [float(v) for v in cw]
    wref = WeightsRef.coerce(weights)
    if not wref.sha256:
        raise ValueError("the backbone weights SHA-256 is not pinned (registry/weights.yaml); training refuses")
    record.inputs = {**dict(inputs or {}), "weights": {"name": wref.name, "sha256": wref.sha256}}

    model = build_f9m(spec.candidate, wref.path, seed=spec.seed, expected_sha256=wref.sha256, dropout=record.architecture["dropout"],
                      channel_mean=record.architecture["channel_mean"], channel_sd=record.architecture["channel_sd"],
                      mask_channels=mask_channels).to(dev)
    initial = cpu_state(model)
    record.parameters["initial_state_sha256"] = state_sha256(initial)
    optimizer = torch.optim.AdamW(model.parameter_groups(spec.candidate.lr_backbone, float(cfg["new_layer_lr_multiplier"])),
                                  lr=spec.candidate.lr_backbone, betas=tuple(cfg["optimizer"]["betas"]), eps=float(cfg["optimizer"]["eps"]),
                                  weight_decay=float(cfg["optimizer"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["schedule"]["t_max_epochs"]),
                                                           eta_min=float(cfg["schedule"]["eta_min"]))
    max_epochs = int(spec.refit_epochs)
    if max_epochs > int(cfg["schedule"]["t_max_epochs"]):
        raise ValueError(f"{max_epochs} epochs exceed the {cfg['schedule']['t_max_epochs']}-epoch schedule")
    record.max_epochs = max_epochs
    history: list[dict[str, Any]] = []
    start_epoch, resumed_from = 1, None
    resume_dir = out / "resume"
    if (resume_dir / "last.pt").is_file():
        saved = torch.load(resume_dir / "last.pt", map_location="cpu", weights_only=False)
        if saved.get("schema") != RESUME_SCHEMA or saved.get("fit_id") != spec.fit_id or saved.get("config_sha256") != cfg_sha \
                or saved.get("mask_channels") != bool(mask_channels):
            raise ValueError(f"{resume_dir / 'last.pt'} belongs to another fit or configuration")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history = list(saved["history"])
        _restore_rng(saved["rng"], dev)
        start_epoch = resumed_from = int(saved["epoch"]) + 1
    weights_t = torch.tensor(cw, dtype=torch.float32, device=dev)
    y_index = {sid: index[labels[sid]] for sid in train_ids}

    for epoch in range(start_epoch, max_epochs + 1):
        t_epoch = time.perf_counter()
        torch.manual_seed(epoch_seed(spec.seed, epoch))
        rng = np.random.default_rng([int(spec.seed), int(epoch)])
        lrs = [float(g["lr"]) for g in optimizer.param_groups]
        train_loss = _train_epoch(model, optimizer, items, train_ids, y_index, weights_t, rng, cfg, dev, len(classes),
                                  collate=collate, augment=augment)
        scheduler.step()
        entry: dict[str, Any] = {"epoch": epoch, "lr_backbone": lrs[0], "lr_new": lrs[1], "train_loss": train_loss,
                                 "seconds": round(time.perf_counter() - t_epoch, 3)}
        history.append(entry)
        _atomic_torch_save({"schema": RESUME_SCHEMA, "fit_id": spec.fit_id, "config_sha256": cfg_sha, "mask_channels": bool(mask_channels),
                            "epoch": epoch, "model": cpu_state(model), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                            "early_stopping": None, "stopped": False, "history": history, "rng": _rng_state(dev)}, resume_dir / "last.pt")
        if stop_after_epoch is not None and epoch >= int(stop_after_epoch) and epoch < max_epochs:
            record.status, record.epochs = STATUS_INTERRUPTED, history
            record.resume = {"attempts": attempts, "resumed_from_epoch": resumed_from, "interrupted_after_epoch": epoch}
            return record

    record.epochs = history
    record.selected_epoch = record.stopped_epoch = int(history[-1]["epoch"])
    record.stop_reason = "refit_epochs"
    final = cpu_state(model)
    record.parameters.update(_parameter_checks(initial, final, model))
    if mask_channels:
        record.parameters["mask_stem_max_abs_weight"] = float(final[MASK_STEM_KEY].abs().max())
        record.parameters["mask_stem_trained"] = bool(record.parameters["mask_stem_max_abs_weight"] > 0.0)
    checkpoint = _atomic_torch_save(final, out / "model.pt")
    record.checkpoint_file, record.checkpoint_sha256, record.state_sha256 = "model.pt", sha256_file(checkpoint), state_sha256(final)
    record.timings = {"started_utc": started_utc, "finished_utc": _utc(), "seconds_total": round(time.perf_counter() - t_start, 3),
                      "seconds_per_epoch": [e["seconds"] for e in history]}
    record.resume = {"attempts": attempts, "resumed_from_epoch": resumed_from}
    record.write(out / "record.json")
    if not bool(cfg.get("outputs", {}).get("keep_resume_state_after_completion", False)):
        shutil.rmtree(resume_dir, ignore_errors=True)
    return record
