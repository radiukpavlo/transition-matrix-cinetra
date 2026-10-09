"""F9 network (contract C8; NEURAL-TM-1.0 4.1 verbatim; ``configs/f9_cnn.yaml`` ``architecture``).

A shared torchvision ResNet-18 (``IMAGENET1K_V1`` weights, SHA-256 pinned before training) encodes every acquired ED and
ES slice.  Its 512-d global pool plus the two geometry scalars (514) pass a shared ``Linear(514,128)`` + GELU.  A shared
attention scorer ``Linear(128,64) -> tanh -> Linear(64,1)`` gives a softmax over the valid slices of a phase; the
attention-weighted mean and the elementwise maximum (256) form each phase vector.  ED, ES and ED-ES (768) feed
``Linear(768,256) -> GELU -> Dropout(0.25) -> Linear(256,64) -> GELU``; these 64 activations are A, and the head is
exactly ``Linear(64,5)``.  Padding positions never contribute: only the valid (prefix) slices enter the backbone, and the
softmax and the maximum exclude padding.  The input is grey in [0, 1]; the model replicates it to three channels and
applies the fixed ImageNet mean/SD.

Freezing: only the candidate's blocks (``layer4`` or ``layer3+layer4``) of the backbone train; batch-normalization
running statistics stay at their pretrained values in every candidate (all backbone BN layers stay in eval mode), and
BN affine parameters train only inside the unfrozen blocks.  The global pool is written as a spatial mean (identical to
``AdaptiveAvgPool2d(1)``) because the CUDA backward of the adaptive pool has no deterministic implementation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .fit_spec import UNFREEZE_OPTIONS, Candidate, FitSpec

N_CLASSES = 5
SLICE_FEATURES = 512
GEOMETRY_FEATURES = 2
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_SD = (0.229, 0.224, 0.225)
CHANGE_WATCH = {"layer4": "backbone.layer4.1.conv2.weight", "layer3": "backbone.layer3.1.conv2.weight"}
FROZEN_WATCH = "backbone.layer1.0.conv1.weight"


class SpatialMean(nn.Module):
    """``x.mean((2, 3), keepdim=True)``: the ResNet global average pool with a deterministic backward."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(2, 3), keepdim=True)


def _check_prefix(mask: torch.Tensor) -> list[int]:
    if mask.dtype != torch.bool or mask.ndim != 2:
        raise ValueError("masks must be boolean [B, S]")
    counts = mask.sum(dim=1)
    positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
    if not torch.equal(mask, positions < counts.unsqueeze(1)):
        raise ValueError("masks must be prefix masks (valid slices first; collate_patients)")
    values = [int(v) for v in counts.tolist()]
    if min(values) < 1:
        raise ValueError("every patient needs at least one valid slice per phase")
    return values


class F9Net(nn.Module):
    """``forward(ed, es, ed_geom, es_geom, ed_mask, es_mask) -> (logits [B,5], a [B,64])``."""

    def __init__(self, backbone: nn.Module, unfreeze: str, *, n_classes: int = N_CLASSES, dropout: float = 0.25,
                 channel_mean: Sequence[float] = IMAGENET_MEAN, channel_sd: Sequence[float] = IMAGENET_SD, seed: int | None = None) -> None:
        super().__init__()
        backbone.fc = nn.Identity()
        backbone.avgpool = SpatialMean()
        self.backbone = backbone
        if unfreeze not in UNFREEZE_OPTIONS:
            raise ValueError(f"unfreeze must be one of {UNFREEZE_OPTIONS}, got {unfreeze!r}")
        self.unfrozen_blocks = tuple(unfreeze.split("+"))
        if seed is not None:
            torch.manual_seed(int(seed))
        self.slice_proj = nn.Linear(SLICE_FEATURES + GEOMETRY_FEATURES, 128)
        self.att_hidden = nn.Linear(128, 64)
        self.att_score = nn.Linear(64, 1)
        self.fc1 = nn.Linear(768, 256)
        self.dropout = nn.Dropout(float(dropout))
        self.fc2 = nn.Linear(256, 64)
        self.head = nn.Linear(64, int(n_classes))
        self.register_buffer("channel_mean", torch.tensor(list(channel_mean), dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("channel_sd", torch.tensor(list(channel_sd), dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        for name, param in self.backbone.named_parameters():
            param.requires_grad_(name.split(".", 1)[0] in self.unfrozen_blocks)

    # ------------------------------------------------------------------ freezing and parameter groups

    def train(self, mode: bool = True) -> F9Net:
        super().train(mode)
        for module in self.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def new_modules(self) -> list[nn.Module]:
        return [self.slice_proj, self.att_hidden, self.att_score, self.fc1, self.fc2, self.head]

    def backbone_trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def new_parameters(self) -> list[nn.Parameter]:
        return [p for m in self.new_modules() for p in m.parameters()]

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)

    def parameter_groups(self, lr_backbone: float, multiplier: float) -> list[dict[str, Any]]:
        return [{"params": self.backbone_trainable_parameters(), "lr": float(lr_backbone), "name": "backbone_unfrozen"},
                {"params": self.new_parameters(), "lr": float(lr_backbone) * float(multiplier), "name": "new_layers_and_head"}]

    # ------------------------------------------------------------------ forward

    def _encode(self, ed: torch.Tensor, es: torch.Tensor, ed_geom: torch.Tensor, es_geom: torch.Tensor,
                ed_mask: torch.Tensor, es_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        counts_ed, counts_es = _check_prefix(ed_mask), _check_prefix(es_mask)
        pieces = [ed[b, :n] for b, n in enumerate(counts_ed)] + [es[b, :n] for b, n in enumerate(counts_es)]
        geoms = [ed_geom[b, :n] for b, n in enumerate(counts_ed)] + [es_geom[b, :n] for b, n in enumerate(counts_es)]
        slices = torch.cat(pieces, dim=0).unsqueeze(1).expand(-1, 3, -1, -1)
        slices = (slices - self.channel_mean) / self.channel_sd
        pooled = self.backbone(slices).flatten(1)
        h = F.gelu(self.slice_proj(torch.cat([pooled, torch.cat(geoms, dim=0).to(pooled.dtype)], dim=1)))
        n_ed = sum(counts_ed)
        return (self._pad(h[:n_ed], counts_ed, int(ed_mask.shape[1])), self._pad(h[n_ed:], counts_es, int(es_mask.shape[1])))

    @staticmethod
    def _pad(h: torch.Tensor, counts: list[int], width: int) -> torch.Tensor:
        parts = torch.split(h, counts, dim=0)
        return torch.stack([F.pad(part, (0, 0, 0, width - part.shape[0])) for part in parts])

    def _pool(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        score = self.att_score(torch.tanh(self.att_hidden(h))).squeeze(-1)
        weights = torch.softmax(score.masked_fill(~mask, float("-inf")), dim=1)
        mean = (weights.unsqueeze(-1) * h).sum(dim=1)
        peak = h.masked_fill(~mask.unsqueeze(-1), float("-inf")).amax(dim=1)
        return torch.cat([mean, peak], dim=1)

    def forward(self, ed: torch.Tensor, es: torch.Tensor, ed_geom: torch.Tensor, es_geom: torch.Tensor,
                ed_mask: torch.Tensor, es_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_ed, h_es = self._encode(ed, es, ed_geom, es_geom, ed_mask, es_mask)
        v_ed, v_es = self._pool(h_ed, ed_mask), self._pool(h_es, es_mask)
        z = torch.cat([v_ed, v_es, v_ed - v_es], dim=1)
        a = F.gelu(self.fc2(self.dropout(F.gelu(self.fc1(z)))))
        return self.head(a), a

    def forward_batch(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        return self(batch["ed"], batch["es"], batch["ed_geom"], batch["es_geom"], batch["ed_mask"], batch["es_mask"])


# ----------------------------------------------------------------------------- construction and loading


def file_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_model(candidate: Candidate | Mapping[str, Any], weights_path: str | Path | None, *, seed: int | None = None,
                expected_sha256: str | None = None, dropout: float = 0.25, channel_mean: Sequence[float] = IMAGENET_MEAN,
                channel_sd: Sequence[float] = IMAGENET_SD) -> F9Net:
    """ResNet-18 backbone with ``weights_path`` (torchvision state dict; SHA-256 checked when ``expected_sha256`` is
    given) and fresh new layers (seeded with ``seed``).  ``weights_path=None`` leaves the backbone uninitialized for a
    state dict that is loaded afterwards (:func:`load_f9`)."""

    import torchvision

    cand = Candidate.coerce(candidate)
    backbone = torchvision.models.resnet18(weights=None)
    if weights_path is not None:
        if expected_sha256 is not None:
            digest = file_sha256(weights_path)
            if digest != str(expected_sha256).lower():
                raise ValueError(f"backbone weights {weights_path}: SHA-256 {digest} differs from the pinned {expected_sha256}")
        backbone.load_state_dict(torch.load(Path(weights_path), map_location="cpu", weights_only=True))
    return F9Net(backbone, cand.unfreeze, dropout=dropout, channel_mean=channel_mean, channel_sd=channel_sd, seed=seed)


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Content hash of a state dict: sorted keys, dtype, shape and the raw bytes of each tensor."""

    hasher = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().to("cpu").contiguous()
        hasher.update(f"{key}|{tensor.dtype}|{tuple(tensor.shape)}|".encode())
        hasher.update(tensor.numpy().tobytes() if tensor.dtype != torch.bfloat16 else tensor.float().numpy().tobytes())
    return hasher.hexdigest()


def cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}


def load_f9(checkpoint_dir: str | Path, map_location: Any = "cpu") -> F9Net:
    """``<checkpoint_dir>/model.pt`` + ``record.json`` -> F9Net in eval mode (SHA-256 of ``model.pt`` verified)."""

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
    model = build_model(spec.candidate, None, dropout=float(arch.get("dropout", 0.25)),
                        channel_mean=arch.get("channel_mean", IMAGENET_MEAN), channel_sd=arch.get("channel_sd", IMAGENET_SD))
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.f9_record = record  # type: ignore[attr-defined]
    return model.to(map_location).eval()
