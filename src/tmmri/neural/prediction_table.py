"""Prediction and activation tables of every classifier (workflow contract C4).

``<model_dir>/predictions.parquet``: ``sample_id, cluster_id, partition, model_id`` (``audited_predictor``,
``d20_predictor`` or ``f9``), ``seed`` (int; -1 if none), ``encoder_id`` (``fold1`` .. ``fold5`` or ``final``),
``is_oof`` (bool), ``y_true`` (nullable str), ``y_pred`` (str), ``prob_<CLASS>`` for the five classes in the registry
class order and ``logit_<CLASS>`` (nullable).  ``<model_dir>/activations.parquet`` has the same key columns plus
``a_000`` .. ``a_<d-1>`` (zero-padded to at least three digits).  ``y_pred`` is the argmax of the probabilities with
ties broken by the class order.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

CLASS_ORDER = ("DCM", "HCM", "MINF", "NOR", "RV")   # configs/formal.yaml audited_predictor.classes; design JSON class_order
MODEL_IDS = ("audited_predictor", "d20_predictor", "f9")
ENCODER_IDS = ("fold1", "fold2", "fold3", "fold4", "fold5", "final")
KEY_COLUMNS = ("sample_id", "cluster_id", "partition", "model_id", "seed", "encoder_id", "is_oof")
SORT_COLUMNS = ["partition", "model_id", "seed", "encoder_id", "sample_id"]


def encoder_id(outer_fold: Any) -> str:
    """``fold<k>`` for an outer fold ``k``, ``final`` for the refit on all development patients."""

    if outer_fold is None or str(outer_fold).lower() == "final":
        return "final"
    return f"fold{int(outer_fold)}"


def activation_columns(dimension: int) -> list[str]:
    width = max(3, len(str(int(dimension) - 1)))
    return [f"a_{i:0{width}d}" for i in range(int(dimension))]


def _keys(n: int, *, sample_id: Sequence[str], cluster_id: Sequence[str] | None, partition: str | Sequence[str], model_id: str,
          seed: int | None, encoder_id: str | Sequence[str], is_oof: bool | Sequence[bool]) -> pd.DataFrame:
    if model_id not in MODEL_IDS:
        raise ValueError(f"model_id {model_id!r} not in {MODEL_IDS}")
    enc = [encoder_id] * n if isinstance(encoder_id, str) else list(encoder_id)
    bad = sorted(set(enc) - set(ENCODER_IDS))
    if bad:
        raise ValueError(f"encoder_id values {bad} not in {ENCODER_IDS}")
    frame = pd.DataFrame({
        "sample_id": pd.array([str(s) for s in sample_id], dtype="string"),
        "cluster_id": pd.array([str(c) for c in (cluster_id if cluster_id is not None else sample_id)], dtype="string"),
        "partition": pd.array([partition] * n if isinstance(partition, str) else [str(p) for p in partition], dtype="string"),
        "model_id": pd.array([model_id] * n, dtype="string"),
        "seed": np.full(n, -1 if seed is None else int(seed), dtype=np.int64),
        "encoder_id": pd.array(enc, dtype="string"),
        "is_oof": np.asarray([is_oof] * n if isinstance(is_oof, (bool, np.bool_)) else list(is_oof), dtype=bool),
    })
    if len(frame) != n:
        raise ValueError("key columns are not aligned")
    return frame


def prediction_frame(*, sample_id: Sequence[str], probabilities: np.ndarray, model_id: str, partition: str | Sequence[str],
                     encoder_id: str | Sequence[str], is_oof: bool | Sequence[bool], seed: int | None = None,
                     cluster_id: Sequence[str] | None = None, y_true: Iterable[Any] | None = None, logits: np.ndarray | None = None,
                     classes: Sequence[str] = CLASS_ORDER) -> pd.DataFrame:
    """One C4 block; ``probabilities``/``logits`` are ``(n, len(classes))`` in ``classes`` order (NaN = no output)."""

    p = np.asarray(probabilities, dtype=np.float64)
    n = p.shape[0]
    if p.shape != (n, len(classes)):
        raise ValueError(f"probabilities must be (n, {len(classes)}), got {p.shape}")
    frame = _keys(n, sample_id=sample_id, cluster_id=cluster_id, partition=partition, model_id=model_id, seed=seed,
                  encoder_id=encoder_id, is_oof=is_oof)
    truth = [None] * n if y_true is None else [None if v is None or (not isinstance(v, str) and pd.isna(v)) or str(v) == "" else str(v)
                                                for v in y_true]
    frame["y_true"] = pd.array(truth, dtype="string")
    finite = np.all(np.isfinite(p), axis=1)
    pred = np.array([None] * n, dtype=object)
    pred[finite] = np.asarray(classes, dtype=object)[np.argmax(p[finite], axis=1)]
    frame["y_pred"] = pd.array(pred, dtype="string")
    for j, cls in enumerate(classes):
        frame[f"prob_{cls}"] = p[:, j]
    lg = np.full((n, len(classes)), np.nan) if logits is None else np.asarray(logits, dtype=np.float64)
    if lg.shape != (n, len(classes)):
        raise ValueError(f"logits must be (n, {len(classes)}), got {lg.shape}")
    for j, cls in enumerate(classes):
        frame[f"logit_{cls}"] = lg[:, j]
    return frame


def activation_frame(*, sample_id: Sequence[str], activations: np.ndarray, model_id: str, partition: str | Sequence[str],
                     encoder_id: str | Sequence[str], is_oof: bool | Sequence[bool], seed: int | None = None,
                     cluster_id: Sequence[str] | None = None) -> pd.DataFrame:
    a = np.asarray(activations, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("activations must be a 2D array")
    frame = _keys(a.shape[0], sample_id=sample_id, cluster_id=cluster_id, partition=partition, model_id=model_id, seed=seed,
                  encoder_id=encoder_id, is_oof=is_oof)
    values = pd.DataFrame(a, columns=activation_columns(a.shape[1]), index=frame.index)
    return pd.concat([frame, values], axis=1)


def sort_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(SORT_COLUMNS, kind="mergesort").reset_index(drop=True)


def validate_prediction_frame(frame: pd.DataFrame, classes: Sequence[str] = CLASS_ORDER) -> None:
    """Raise ``ValueError`` unless ``frame`` carries the C4 columns with consistent argmax decisions."""

    required = list(KEY_COLUMNS) + ["y_true", "y_pred"] + [f"prob_{c}" for c in classes] + [f"logit_{c}" for c in classes]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"prediction table lacks columns {missing}")
    p = frame[[f"prob_{c}" for c in classes]].to_numpy(dtype=float)
    finite = np.all(np.isfinite(p), axis=1)
    if not np.allclose(p[finite].sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("probabilities do not sum to one")
    expected = np.asarray(classes, dtype=object)[np.argmax(p[finite], axis=1)]
    if not np.array_equal(frame.loc[finite, "y_pred"].astype(str).to_numpy(), expected.astype(str)):
        raise ValueError("y_pred is not the argmax of the probabilities")
    duplicated = frame.duplicated(list(KEY_COLUMNS[:1]) + ["model_id", "seed", "encoder_id", "partition"])
    if duplicated.any():
        raise ValueError("duplicate (sample_id, model_id, seed, encoder_id, partition) rows")
