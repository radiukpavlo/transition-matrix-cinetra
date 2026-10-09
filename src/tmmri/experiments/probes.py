"""F9b concept probe (protocol-v6.0.1 item 11): the v6 typed ridge T on the trained CNN's 64-d activations.

For every outer fold k the transition (zero-variance filter, standardization, typed target transforms, per-target
lambda by inner CV on the v6 grid with the v6 inner folds of outer fold k) is fitted on **encoder k's** activations of
its fitting patients and decodes **encoder k's** activations of fold k, giving the out-of-fold B-hat.  The final T is
fitted on the final encoder's activations of all acdc_dev fit rows (lambda by CV over the outer folds, as
``fit-transition``).  Latent coordinates are never pooled across encoders: each T only ever sees the coordinates of
the encoder it belongs to.  :func:`tmmri.transition_fit.fit_transition_v6` is the only fitter.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from ..transition_fit import LAMBDA_AT_GRID_EDGE, TransitionModel, fit_transition_v6

REPRESENTATION = "f9b_activations_64"


def encoder_matrix(activations: pd.DataFrame, encoder_id: str, sample_ids: list[str], columns: list[str]) -> np.ndarray:
    """Activations of ``encoder_id`` for ``sample_ids`` (row order kept); every sample must be present exactly once."""

    rows = activations[activations["encoder_id"].astype(str) == str(encoder_id)].set_index("sample_id")
    if rows.index.duplicated().any():
        raise ValueError(f"encoder {encoder_id}: several activation rows per sample_id")
    missing = [s for s in sample_ids if s not in rows.index]
    if missing:
        raise ValueError(f"encoder {encoder_id}: no activations for {len(missing)} required patients (e.g. {missing[:3]})")
    return rows.loc[sample_ids, columns].to_numpy(dtype=np.float64)


def fit_f9b_probes(ids: pd.DataFrame, b: pd.DataFrame, activations: pd.DataFrame, columns: list[str], targets: list[str],
                   cfg: dict[str, Any], registry: dict[str, Any], inner_fold_fn: Callable[..., np.ndarray]) -> dict[str, Any]:
    """Per-fold probes, the out-of-fold B-hat and the final probe for one seed.

    ``ids`` holds ``sample_id`` and ``outer_fold`` of the acdc_dev fit rows (critical QC removed), ``b`` their typed B
    (row-aligned), ``activations`` the C4 activation rows of one seed (encoder_id ``fold1..fold5`` and ``final``) and
    ``inner_fold_fn(k, train_positions)`` the v6 inner-fold labels of outer fold k."""

    sample_ids = ids["sample_id"].astype(str).tolist()
    folds = ids["outer_fold"].to_numpy(dtype=int)
    oof = pd.DataFrame(np.nan, index=pd.Index(sample_ids, name="sample_id"), columns=list(targets), dtype=float)
    fold_models: dict[int, TransitionModel] = {}
    ledger: list[dict[str, Any]] = []
    for k in sorted(set(folds.tolist())):
        train = np.flatnonzero(folds != k)
        held = np.flatnonzero(folds == k)
        encoder = f"fold{k}"
        a_train = encoder_matrix(activations, encoder, [sample_ids[i] for i in train], columns)
        a_held = encoder_matrix(activations, encoder, [sample_ids[i] for i in held], columns)
        model = fit_transition_v6(a_train, b.iloc[train], list(targets), cfg, registry, np.asarray(inner_fold_fn(int(k), train)),
                                  representation=f"{REPRESENTATION}:{encoder}")
        fold_models[int(k)] = model
        oof.iloc[held, :] = model.predict(a_held).to_numpy()
        ledger += _ledger_rows(model, targets, scope="outer_fold", fold=int(k), encoder=encoder)
    a_final = encoder_matrix(activations, "final", sample_ids, columns)
    final = fit_transition_v6(a_final, b, list(targets), cfg, registry, folds, representation=f"{REPRESENTATION}:final")
    ledger += _ledger_rows(final, targets, scope="final_bundle", fold=None, encoder="final")
    oof["outer_fold"] = pd.array(folds, dtype="Int64")
    lam = pd.DataFrame(ledger)
    lam["outer_fold"] = lam["outer_fold"].astype("Int64")
    return {"oof": oof.reset_index(), "fold_models": fold_models, "final": final, "lambda_selection": lam}


def _ledger_rows(model: TransitionModel, targets: list[str], *, scope: str, fold: int | None, encoder: str) -> list[dict[str, Any]]:
    return [{"scope": scope, "outer_fold": fold, "encoder_id": encoder, "target": t, "lambda": model.lambdas.get(t, np.nan),
             "lambda_at_grid_edge": bool(model.edge_flags.get(t, False)), "flag": LAMBDA_AT_GRID_EDGE if model.edge_flags.get(t, False) else "",
             "eligible": t in model.fitted_targets, "exclusion_reason": model.excluded.get(t, ""), "n_fit_rows": model.n_fit_rows} for t in targets]
