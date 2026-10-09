"""F9 fit specification (contract C5 of the protocol-v6.0.1 implementation; ``configs/f9_cnn.yaml``).

A :class:`FitSpec` names one network fit completely: purpose (``f9`` or ``s5``), seed, outer fold (1-5 or ``"final"``),
the sorted fitting patients, the selection fold (1-3; ``None`` for a refit), the candidate (unfrozen blocks and backbone
learning rate), the refit epoch count (``None`` for a selection fit), the S5 fraction and repeat.  ``fit_id`` is the
first 16 hex characters of the SHA-256 of the canonical JSON (:func:`tmmri.runs.canonical_json`) of all other fields,
so the same fit has the same id on every backend.  :func:`tmmri.neural.train.train_fit` is the only training entry
point; the local backend (``train-f9``) and the Modal function (``scripts/cloud/modal_train.py``) both call it.

Spec files (``plan-f9``, ``select-f9``) wrap the specs with the run id, the wave, the training configuration and its
hash, the tensor-cache manifest hash and the backbone weights reference; jobs verify all three before training.  This
module is standard-library only, so the Modal local entrypoint can read and verify spec files without numpy or torch.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..runs import atomic_write_json, canonical_json, sha256_json

PURPOSES = ("f9", "s5")
UNFREEZE_OPTIONS = ("layer4", "layer3+layer4")
LR_BACKBONE_OPTIONS = (0.00001, 0.00003, 0.0001)
OUTER_FOLDS = (1, 2, 3, 4, 5, "final")
SELECTION_FOLDS = (1, 2, 3)
F9_SEEDS = (20261011, 20261012, 20261013)        # protocol 2.1 D item 10; distinct from paper 2
PRIMARY_SEED = 20261011
SPEC_FILE_SCHEMA = "tmmri.f9_specs/1"
FIT_ID_LENGTH = 16


@dataclass(frozen=True)
class Candidate:
    """One of the six network candidates (NEURAL-TM-1.0 4.2)."""

    unfreeze: str
    lr_backbone: float

    def __post_init__(self) -> None:
        if self.unfreeze not in UNFREEZE_OPTIONS:
            raise ValueError(f"candidate unfreeze must be one of {UNFREEZE_OPTIONS}, got {self.unfreeze!r}")
        lr = float(self.lr_backbone)
        if lr not in LR_BACKBONE_OPTIONS:
            raise ValueError(f"candidate lr_backbone must be one of {LR_BACKBONE_OPTIONS}, got {self.lr_backbone!r}")
        object.__setattr__(self, "lr_backbone", lr)

    @property
    def blocks(self) -> tuple[str, ...]:
        return tuple(self.unfreeze.split("+"))

    @property
    def n_unfrozen_blocks(self) -> int:
        return len(self.blocks)

    @property
    def label(self) -> str:
        return f"{self.unfreeze}@{self.lr_backbone:g}"

    def to_dict(self) -> dict[str, Any]:
        return {"unfreeze": self.unfreeze, "lr_backbone": float(self.lr_backbone)}

    @classmethod
    def coerce(cls, value: Candidate | Mapping[str, Any]) -> Candidate:
        if isinstance(value, Candidate):
            return value
        return cls(unfreeze=str(value["unfreeze"]), lr_backbone=float(value["lr_backbone"]))


CANDIDATES: tuple[Candidate, ...] = tuple(Candidate(u, lr) for u in UNFREEZE_OPTIONS for lr in LR_BACKBONE_OPTIONS)


def candidates_from_config(cfg: Mapping[str, Any]) -> tuple[Candidate, ...]:
    """The six candidates of ``configs/f9_cnn.yaml`` (checked against the protocol's fixed set)."""

    found = tuple(Candidate.coerce(c) for c in cfg["candidates"])
    if set(found) != set(CANDIDATES) or len(found) != len(CANDIDATES):
        raise ValueError(f"configs/f9_cnn.yaml candidates must be exactly the six protocol candidates, got {[c.label for c in found]}")
    return found


def _outer_fold(value: Any) -> int | str:
    if isinstance(value, str) and value != "final":
        value = int(value)
    if isinstance(value, bool) or value not in OUTER_FOLDS:
        raise ValueError(f"outer_fold must be 1..5 or 'final', got {value!r}")
    return value


@dataclass(frozen=True)
class FitSpec:
    """Contract C5.  ``fit_id`` is derived; it is never an input."""

    purpose: str
    seed: int
    outer_fold: int | str
    fitting_patients: tuple[str, ...]
    selection_fold: int | None
    candidate: Candidate
    refit_epochs: int | None
    fraction: float = 1.0
    repeat: int = 0
    fit_id: str = field(init=False)

    def __post_init__(self) -> None:
        patients = tuple(sorted(str(p) for p in self.fitting_patients))
        if not patients or len(set(patients)) != len(patients):
            raise ValueError("fitting_patients must be a non-empty set of sample ids")
        object.__setattr__(self, "fitting_patients", patients)
        object.__setattr__(self, "candidate", Candidate.coerce(self.candidate))
        object.__setattr__(self, "outer_fold", _outer_fold(self.outer_fold))
        if self.purpose not in PURPOSES:
            raise ValueError(f"purpose must be one of {PURPOSES}, got {self.purpose!r}")
        if isinstance(self.seed, bool) or int(self.seed) != self.seed:
            raise ValueError(f"seed must be an integer, got {self.seed!r}")
        object.__setattr__(self, "seed", int(self.seed))
        allowed_seeds = F9_SEEDS if self.purpose == "f9" else (PRIMARY_SEED,)
        if self.seed not in allowed_seeds:
            raise ValueError(f"{self.purpose} fits use seeds {allowed_seeds}, got {self.seed}")
        if self.selection_fold is not None:
            if self.selection_fold not in SELECTION_FOLDS or isinstance(self.selection_fold, bool):
                raise ValueError(f"selection_fold must be 1, 2, 3 or None, got {self.selection_fold!r}")
            if self.refit_epochs is not None:
                raise ValueError("a selection fit has refit_epochs = None")
        else:
            if self.refit_epochs is None or isinstance(self.refit_epochs, bool) or int(self.refit_epochs) < 1:
                raise ValueError("a refit (selection_fold None) needs refit_epochs >= 1")
            object.__setattr__(self, "refit_epochs", int(self.refit_epochs))
        fraction = float(self.fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must lie in (0, 1], got {self.fraction!r}")
        object.__setattr__(self, "fraction", fraction)
        if isinstance(self.repeat, bool) or int(self.repeat) != self.repeat or int(self.repeat) < 0:
            raise ValueError(f"repeat must be a non-negative integer, got {self.repeat!r}")
        object.__setattr__(self, "repeat", int(self.repeat))
        if self.purpose == "f9" and (fraction != 1.0 or self.repeat != 0):
            raise ValueError("F9 fits use fraction 1.0 and repeat 0")
        object.__setattr__(self, "fit_id", hashlib.sha256(canonical_json(self.identity())).hexdigest()[:FIT_ID_LENGTH])

    def identity(self) -> dict[str, Any]:
        """All fields except ``fit_id`` (the hashed content)."""

        return {"purpose": self.purpose, "seed": self.seed, "outer_fold": self.outer_fold,
                "fitting_patients": list(self.fitting_patients), "selection_fold": self.selection_fold,
                "candidate": self.candidate.to_dict(), "refit_epochs": self.refit_epochs,
                "fraction": self.fraction, "repeat": self.repeat}

    def to_dict(self) -> dict[str, Any]:
        return {"fit_id": self.fit_id, **self.identity()}

    @property
    def is_selection_fit(self) -> bool:
        return self.selection_fold is not None

    @property
    def group_key(self) -> tuple[Any, ...]:
        """Fits that compete in one selection: same purpose, fitting set, seed, fraction and repeat."""

        return (self.purpose, str(self.outer_fold), self.seed, self.fraction, self.repeat, self.fitting_patients)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, verify: bool = True) -> FitSpec:
        spec = cls(purpose=str(data["purpose"]), seed=data["seed"], outer_fold=data["outer_fold"],
                   fitting_patients=tuple(data["fitting_patients"]), selection_fold=data.get("selection_fold"),
                   candidate=Candidate.coerce(data["candidate"]), refit_epochs=data.get("refit_epochs"),
                   fraction=data.get("fraction", 1.0), repeat=data.get("repeat", 0))
        if verify and data.get("fit_id") not in (None, spec.fit_id):
            raise ValueError(f"fit_id {data.get('fit_id')!r} does not match its content (expected {spec.fit_id})")
        return spec


def coerce_spec(spec: FitSpec | Mapping[str, Any]) -> FitSpec:
    return spec if isinstance(spec, FitSpec) else FitSpec.from_dict(spec)


# ----------------------------------------------------------------------------- spec files


def write_spec_file(path: str | Path, specs: Iterable[FitSpec], *, run_id: str, wave: int, config: Mapping[str, Any],
                    cache_manifest_sha256: str, weights: Mapping[str, Any], purpose: str = "f9",
                    extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Write a spec file (sorted JSON; the specs keep plan order) and return its payload."""

    items = [s.to_dict() for s in specs]
    ids = [s["fit_id"] for s in items]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate fit ids in one spec file")
    payload = {"schema": SPEC_FILE_SCHEMA, "run_id": run_id, "wave": int(wave), "purpose": purpose,
               "config": dict(config), "config_sha256": sha256_json(dict(config)),
               "cache_manifest_sha256": cache_manifest_sha256, "weights": dict(weights),
               "n_specs": len(items), "specs": items, **dict(extra or {})}
    atomic_write_json(path, payload)
    return payload


def read_spec_file(path: str | Path) -> dict[str, Any]:
    """Read and verify a spec file: schema, count, configuration hash and every fit id.  ``payload["fit_specs"]``
    holds the :class:`FitSpec` objects in file order."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != SPEC_FILE_SCHEMA:
        raise ValueError(f"{path}: not an F9 spec file (schema {payload.get('schema')!r})")
    if sha256_json(payload["config"]) != payload["config_sha256"]:
        raise ValueError(f"{path}: the embedded training configuration does not match config_sha256")
    specs = [FitSpec.from_dict(item) for item in payload["specs"]]
    if len(specs) != int(payload["n_specs"]) or len({s.fit_id for s in specs}) != len(specs):
        raise ValueError(f"{path}: n_specs or fit-id uniqueness check failed")
    payload["fit_specs"] = specs
    return payload
