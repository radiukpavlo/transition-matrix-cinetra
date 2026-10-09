"""In-plane dihedral group D4 of the image branches D20 and F9 (SPECIFICATION.md 6.0.1, Sections 8.4 and 8.6; owner
issue O1 of 3 October 2026).

Measured reason.  After :func:`tmmri.neural.preprocess.canonical_orientation` the mask-derived direction from the LV to
the RV centroid on a mid-ventricular ED slice is consistent within ACDC (all 98 ``acdc_dev`` cases: orientation code
``rows=+1,cols=+0``, median angle -144 degrees, RV left of LV in 100 %) but not within M&Ms-1 (``rows=+1,cols=+0``,
193 cases, median -146 degrees; ``rows=+0,cols=-1``, 80 cases, +128 degrees; ``rows=-1,cols=+0``, 45 cases, +154
degrees; 2 others; RV left of LV in 100 %): about 40 % of the M&Ms-1 slices are vertically mirrored relative to ACDC in
the canonical frame, because the stored affines do not fix the in-plane orientation reliably.  The descriptors of the
v6 pipeline are orientation-invariant; the image branches are made invariant to the eight in-plane symmetries of the
square slice instead, without masks at inference (``run-exploratory --analysis B9`` re-measures the audit).

Element ``g`` in 0..7 is ``k = g % 4`` quarter turns over the last two axes (``numpy.rot90`` / ``torch.rot90`` with
the axes (rows, columns), from the row axis toward the column axis), followed by a transpose of the last two axes when
``g >= 4``.  Element 0 is the identity.  Each element permutes the pixels of a square slice exactly (no interpolation).

* :func:`d4_apply` applies one element to a numpy array or a torch tensor ``[..., H, H]``.
* ``tta`` ``"d4_mean_8"`` (:data:`D4_TTA`) averages the eight transforms; ``"none"`` is the identity only (the v6.0.0
  definition of D20, kept for reading older records).  :func:`tta_elements` maps a ``tta`` value to its elements.
* :func:`f9_forward_tta` is the F9 test-time symmetrization: the Float64 mean over the transforms of the FP32 logits and
  of the 64-d activations A of a model with the C8 signature; geometry scalars and slice masks are unchanged.
* Training augmentation ``"d4_uniform_per_patient"`` (:data:`D4_AUGMENTATION`): one element per patient drawn uniformly
  from the per-patient augmentation RNG (:func:`tmmri.neural.data.augment_batch`).

torch is imported inside the functions that need it, so the numpy helpers run in the CPU container.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

D4_ORDER = 8
D4_ELEMENTS = tuple(range(D4_ORDER))
D4_TTA = "d4_mean_8"
TTA_NONE = "none"
D4_AUGMENTATION = "d4_uniform_per_patient"
FORWARD_KEYS = ("ed", "es", "ed_geom", "es_geom", "ed_mask", "es_mask")


def d4_element(g: int) -> tuple[int, bool]:
    """``(quarter turns, transpose)`` of element ``g``."""

    g = int(g)
    if not 0 <= g < D4_ORDER:
        raise ValueError(f"D4 element must be in 0..{D4_ORDER - 1}, got {g}")
    return g % 4, g >= 4


def d4_apply(x: Any, g: int) -> Any:
    """Element ``g`` applied to the last two axes of a numpy array or torch tensor (square slices only)."""

    k, transpose = d4_element(g)
    if k == 0 and not transpose:
        return x
    shape = tuple(x.shape)
    if len(shape) < 2 or shape[-1] != shape[-2]:
        raise ValueError(f"D4 transforms need square slices [..., H, H], got shape {shape}")
    if isinstance(x, np.ndarray):
        out = np.rot90(x, k, axes=(x.ndim - 2, x.ndim - 1)) if k else x
        return np.ascontiguousarray(np.swapaxes(out, -1, -2) if transpose else out)
    import torch

    dims = (x.dim() - 2, x.dim() - 1)
    out = torch.rot90(x, k, dims=dims) if k else x
    return (out.transpose(*dims) if transpose else out).contiguous()


def tta_elements(tta: str) -> tuple[int, ...]:
    """The D4 elements averaged by a ``tta`` setting (``d4_mean_8``: all eight; ``none``: the identity)."""

    if tta == D4_TTA:
        return D4_ELEMENTS
    if tta == TTA_NONE:
        return (0,)
    raise ValueError(f"tta must be {D4_TTA!r} or {TTA_NONE!r}, got {tta!r}")


def uses_d4_augmentation(cfg: Mapping[str, Any]) -> bool:
    """``augmentation.dihedral`` of ``configs/f9_cnn.yaml``: ``d4_uniform_per_patient``; absent or ``none``: no D4 draw."""

    value = cfg.get("dihedral")
    if value in (None, TTA_NONE):
        return False
    if value == D4_AUGMENTATION:
        return True
    raise ValueError(f"augmentation dihedral must be {D4_AUGMENTATION!r} or {TTA_NONE!r}, got {value!r}")


def f9_forward_tta(model: Any, batch: Mapping[str, Any], *, tta: str = D4_TTA) -> tuple[Any, Any]:
    """(logits [B, C], activations [B, 64]) as CPU Float64 tensors: the means over the elements of ``tta`` of the FP32
    outputs of ``model(ed, es, ed_geom, es_geom, ed_mask, es_mask)`` on the transformed ED and ES images (summed in
    element order 0..7, then divided by the element count)."""

    import torch

    elements = tta_elements(tta)
    logit_sum = act_sum = None
    for g in elements:
        logits, acts = model(d4_apply(batch["ed"], g), d4_apply(batch["es"], g), *[batch[k] for k in FORWARD_KEYS[2:]])
        logits = logits.detach().to(device="cpu", dtype=torch.float32).to(torch.float64)
        acts = acts.detach().to(device="cpu", dtype=torch.float32).to(torch.float64)
        logit_sum = logits if logit_sum is None else logit_sum + logits
        act_sum = acts if act_sum is None else act_sum + acts
    n = float(len(elements))
    return logit_sum / n, act_sum / n
