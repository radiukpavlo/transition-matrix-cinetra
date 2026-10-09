"""F9 finite selection (NEURAL-TM-1.0 4.2; protocol 2.1 D item 10; ``configs/f9_cnn.yaml`` ``selection``).

* **Selection folds.**  Three patient/cluster-stratified folds inside each fitting set: within each class the clusters
  are ordered by ``SHA256("f9-select-v6.0.1|" + fitting_set_hash + "|" + cluster_id)`` (ties by cluster id) and dealt
  round-robin (:func:`tmmri.splits.hash_rank_folds`, the v6 fold rule).  ``fitting_set_hash`` is the first 16 hex of the
  SHA-256 of the newline-joined sorted sample ids, so every fitting set (outer folds, the final set and S5 subsamples)
  has its own deterministic partition.  Macro-F1 needs all five classes in every validation fold; a class with fewer
  clusters than folds makes that impossible and the fit set is ``NE_INSUFFICIENT_CLASS_SUPPORT``.
* **Rule.**  Highest mean macro-F1 over the three folds, then lowest mean validation cross entropy, then fewer unfrozen
  blocks, then the lower backbone learning rate.  No seed is selected; each seed runs its own selection.
* **Refit.**  The winner is refitted on all fitting patients for ``ceil(median)`` of its three selected epochs.
* **Early stopping** (:class:`EarlyStopper`): an epoch improves when its validation CE is below the best minus
  ``min_delta``; stopping is allowed from ``min_epoch`` once ``patience`` epochs passed without improvement; ties keep
  the earlier checkpoint.

Standard library, numpy and pandas only (no torch): ``plan-f9`` and ``select-f9`` run in the CPU container.
"""

from __future__ import annotations

import hashlib
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..splits import hash_rank_folds
from .fit_spec import CANDIDATES, Candidate, FitSpec

CLASSES = ("DCM", "HCM", "MINF", "NOR", "RV")      # configs/f9_cnn.yaml class_order (registry label_tables.Y_acdc)
SELECT_SALT = "f9-select-v6.0.1|"
N_SELECTION_FOLDS = 3
NE_INSUFFICIENT_CLASS_SUPPORT = "NE_INSUFFICIENT_CLASS_SUPPORT"
STATUS_COMPLETE = "COMPLETE"


class InsufficientClassSupport(ValueError):
    """A required class is missing from a fitting or validation set (``NE_INSUFFICIENT_CLASS_SUPPORT``)."""

    code = NE_INSUFFICIENT_CLASS_SUPPORT


def fitting_set_hash(sample_ids: Iterable[str]) -> str:
    ids = sorted(str(s) for s in sample_ids)
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()[:16]


def selection_salt(sample_ids: Iterable[str], salt: str = SELECT_SALT) -> str:
    return f"{salt}{fitting_set_hash(sample_ids)}|"


def _cluster_labels(patients: pd.DataFrame, classes: Sequence[str]) -> pd.DataFrame:
    frame = patients[["sample_id", "cluster_id", "label"]].astype(str)
    unknown = sorted(set(frame["label"]) - set(classes))
    if unknown:
        raise ValueError(f"labels outside the class order {list(classes)}: {unknown}")
    per_cluster = frame.groupby("cluster_id")["label"].nunique()
    mixed = per_cluster[per_cluster > 1]
    if len(mixed):
        raise ValueError(f"clusters with more than one class: {list(mixed.index[:5])}")
    return frame.drop_duplicates("cluster_id")[["cluster_id", "label"]].reset_index(drop=True)


def selection_folds(patients: pd.DataFrame, *, n_folds: int = N_SELECTION_FOLDS, classes: Sequence[str] = CLASSES,
                    salt: str = SELECT_SALT) -> pd.DataFrame:
    """``sample_id, cluster_id, label, selection_fold`` for one fitting set (columns ``sample_id``, ``cluster_id``,
    ``label``).  Raises :class:`InsufficientClassSupport` when some class has fewer clusters than folds."""

    clusters = _cluster_labels(patients, classes)
    counts = clusters["label"].value_counts()
    short = {c: int(counts.get(c, 0)) for c in classes if int(counts.get(c, 0)) < int(n_folds)}
    if short:
        raise InsufficientClassSupport(f"{NE_INSUFFICIENT_CLASS_SUPPORT}: classes with fewer than {n_folds} clusters in the "
                                       f"fitting set: {short}")
    clusters["selection_fold"] = hash_rank_folds(clusters, "cluster_id", "label", selection_salt(patients["sample_id"], salt),
                                                 int(n_folds)).astype(int)
    out = patients[["sample_id", "cluster_id", "label"]].astype(str).merge(clusters[["cluster_id", "selection_fold"]], on="cluster_id")
    return out.sort_values("sample_id").reset_index(drop=True)


def require_class_support(labels: Iterable[str], classes: Sequence[str] = CLASSES, *, where: str = "fitting set") -> dict[str, int]:
    """Per-class counts; raises :class:`InsufficientClassSupport` when a class is absent."""

    values = [str(v) for v in labels]
    support = {c: values.count(c) for c in classes}
    missing = [c for c, n in support.items() if n == 0]
    if missing:
        raise InsufficientClassSupport(f"{NE_INSUFFICIENT_CLASS_SUPPORT}: {where} lacks classes {missing}")
    return support


def class_weights(labels: Iterable[str], classes: Sequence[str] = CLASSES) -> np.ndarray:
    """Inverse-frequency class weights on the fitting patients, normalized to mean one (class order)."""

    support = require_class_support(labels, classes)
    inverse = np.array([1.0 / support[c] for c in classes], dtype=np.float64)
    return inverse / inverse.mean()


def refit_epochs(selected_epochs: Sequence[int]) -> int:
    """Rounded-up median of the winner's selected epochs (one per selection fold)."""

    values = [int(e) for e in selected_epochs]
    if not values or min(values) < 1:
        raise ValueError(f"selected epochs must be positive integers, got {values}")
    return math.ceil(statistics.median(values))


@dataclass
class EarlyStopper:
    """Validation-CE early stopping (``configs/f9_cnn.yaml`` ``early_stopping``)."""

    min_epoch: int = 20
    patience: int = 12
    min_delta: float = 0.0001
    best_value: float = math.inf
    best_epoch: int = 0
    stopped: bool = False
    history: list[float] = field(default_factory=list)

    def update(self, epoch: int, value: float) -> tuple[bool, bool]:
        """Record ``value`` for ``epoch``; return ``(improved, stop)``."""

        self.history.append(float(value))
        improved = bool(np.isfinite(value)) and value < self.best_value - self.min_delta
        if improved:
            self.best_value, self.best_epoch = float(value), int(epoch)
        self.stopped = epoch >= self.min_epoch and epoch - self.best_epoch >= self.patience
        return improved, self.stopped

    def state(self) -> dict[str, Any]:
        return {"min_epoch": self.min_epoch, "patience": self.patience, "min_delta": self.min_delta,
                "best_value": self.best_value, "best_epoch": self.best_epoch, "stopped": self.stopped, "history": list(self.history)}

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> EarlyStopper:
        es = cfg["early_stopping"]
        return cls(min_epoch=int(es["min_epoch"]), patience=int(es["patience"]), min_delta=float(es["min_delta"]))

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> EarlyStopper:
        return cls(min_epoch=int(state["min_epoch"]), patience=int(state["patience"]), min_delta=float(state["min_delta"]),
                   best_value=float(state["best_value"]), best_epoch=int(state["best_epoch"]), stopped=bool(state["stopped"]),
                   history=[float(v) for v in state["history"]])


def rank_key(mean_macro_f1: float, mean_cross_entropy: float, candidate: Candidate) -> tuple[float, float, int, float]:
    """Sort key of the selection rule (smallest first = winner)."""

    return (-float(mean_macro_f1), float(mean_cross_entropy), candidate.n_unfrozen_blocks, float(candidate.lr_backbone))


def select_candidate(rows: Iterable[Mapping[str, Any]], *, n_folds: int = N_SELECTION_FOLDS) -> tuple[Candidate, pd.DataFrame]:
    """Apply the rule to per-(candidate, fold) rows with ``candidate`` (Candidate or mapping), ``selection_fold``,
    ``macro_f1``, ``cross_entropy`` and ``selected_epoch``.  Every candidate needs all ``n_folds`` folds.  Returns the
    winner and the ranked table (one row per candidate)."""

    by_candidate: dict[Candidate, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_candidate.setdefault(Candidate.coerce(row["candidate"]), []).append(row)
    table = []
    for cand, items in by_candidate.items():
        folds = sorted(int(r["selection_fold"]) for r in items)
        if folds != list(range(1, int(n_folds) + 1)):
            raise ValueError(f"candidate {cand.label} has selection folds {folds}, expected 1..{n_folds}")
        items = sorted(items, key=lambda r: int(r["selection_fold"]))
        f1 = [float(r["macro_f1"]) for r in items]
        ce = [float(r["cross_entropy"]) for r in items]
        epochs = [int(r["selected_epoch"]) for r in items]
        table.append({"candidate": cand.label, "unfreeze": cand.unfreeze, "lr_backbone": cand.lr_backbone,
                      "n_unfrozen_blocks": cand.n_unfrozen_blocks, "mean_macro_f1": float(np.mean(f1)),
                      "mean_cross_entropy": float(np.mean(ce)), "macro_f1_folds": f1, "cross_entropy_folds": ce,
                      "selected_epochs": epochs, "refit_epochs": refit_epochs(epochs), "_candidate": cand})
    if not table:
        raise ValueError("no candidate rows to select from")
    table.sort(key=lambda r: rank_key(r["mean_macro_f1"], r["mean_cross_entropy"], r["_candidate"]))
    for rank, row in enumerate(table, start=1):
        row["rank"] = rank
        row["selected"] = rank == 1
    winner = table[0]["_candidate"]
    frame = pd.DataFrame([{k: v for k, v in r.items() if k != "_candidate"} for r in table])
    return winner, frame


# ----------------------------------------------------------------------------- plans (waves)


def plan_wave1(fitting_sets: Mapping[int | str, Sequence[str]], *, seeds: Sequence[int],
               candidates: Sequence[Candidate] = CANDIDATES, purpose: str = "f9", fraction: float = 1.0, repeat: int = 0,
               n_folds: int = N_SELECTION_FOLDS) -> list[FitSpec]:
    """Selection fits: fitting set x seed x candidate x selection fold (plan order)."""

    specs = []
    for outer_fold, patients in fitting_sets.items():
        for seed in seeds:
            for cand in candidates:
                for fold in range(1, int(n_folds) + 1):
                    specs.append(FitSpec(purpose=purpose, seed=int(seed), outer_fold=outer_fold, fitting_patients=tuple(patients),
                                         selection_fold=fold, candidate=cand, refit_epochs=None, fraction=fraction, repeat=repeat))
    return specs


def plan_wave2(wave1: Sequence[FitSpec], records: Mapping[str, Mapping[str, Any]], *,
               n_folds: int = N_SELECTION_FOLDS) -> tuple[list[FitSpec], pd.DataFrame, list[dict[str, Any]]]:
    """Refit specs from the wave-1 fit records (``fit_id`` -> record dict of :class:`tmmri.neural.train.FitRecord`).

    Returns ``(refit_specs, selection_table, not_estimable)``.  A group with any NE record is not estimable (no refit);
    a missing record raises ``KeyError``."""

    groups: dict[tuple[Any, ...], list[FitSpec]] = {}
    for spec in wave1:
        if not spec.is_selection_fit:
            raise ValueError(f"wave-1 spec {spec.fit_id} is not a selection fit")
        groups.setdefault(spec.group_key, []).append(spec)
    missing = [s.fit_id for s in wave1 if s.fit_id not in records]
    if missing:
        raise KeyError(f"{len(missing)} wave-1 fit records are missing (e.g. {missing[:3]})")
    refits: list[FitSpec] = []
    tables: list[pd.DataFrame] = []
    not_estimable: list[dict[str, Any]] = []
    for specs in groups.values():
        first = specs[0]
        group = {"purpose": first.purpose, "outer_fold": str(first.outer_fold), "seed": first.seed,
                 "fraction": first.fraction, "repeat": first.repeat, "fitting_set_hash": fitting_set_hash(first.fitting_patients)}
        statuses = {records[s.fit_id].get("status") for s in specs}
        if statuses != {STATUS_COMPLETE}:
            reasons = sorted({str(records[s.fit_id].get("reason") or records[s.fit_id].get("status")) for s in specs
                              if records[s.fit_id].get("status") != STATUS_COMPLETE})
            not_estimable.append({**group, "status": NE_INSUFFICIENT_CLASS_SUPPORT if statuses <= {STATUS_COMPLETE, NE_INSUFFICIENT_CLASS_SUPPORT}
                                  else "NOT_COMPLETE", "reasons": reasons})
            continue
        rows = []
        for s in specs:
            rec = records[s.fit_id]
            rows.append({"candidate": s.candidate, "selection_fold": s.selection_fold, "macro_f1": rec["val_metrics"]["macro_f1"],
                         "cross_entropy": rec["val_metrics"]["cross_entropy"], "selected_epoch": rec["selected_epoch"]})
        winner, table = select_candidate(rows, n_folds=n_folds)
        chosen = table.iloc[0]
        refit = FitSpec(purpose=first.purpose, seed=first.seed, outer_fold=first.outer_fold, fitting_patients=first.fitting_patients,
                        selection_fold=None, candidate=winner, refit_epochs=int(chosen["refit_epochs"]),
                        fraction=first.fraction, repeat=first.repeat)
        refits.append(refit)
        tables.append(table.assign(**{k: v for k, v in group.items()}, refit_fit_id=refit.fit_id))
    table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    return refits, table, not_estimable
