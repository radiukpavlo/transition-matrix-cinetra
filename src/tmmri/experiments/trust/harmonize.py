"""S3 label-free harmonization: only the standardization of A changes; T, lambda and the predictor stay frozen.

* **H-a**: the per-column mean and SD of the unlabelled target partition (all its rows with finite A).
* **H-b**: the per-vendor mean and SD within the target partition (vendor metadata only); a vendor group with fewer
  than ``min_rows_per_vendor`` finite rows uses the H-a statistics (recorded).

A frozen model (``TransitionModel``, ``FormalPredictor``: dataclasses with a ``standardizer``) is applied with a
standardizer whose kept columns carry the target statistics (a kept column whose target SD is not above 1e-12 keeps
its frozen SD, recorded as degenerate).  For a target that is a pure location-scale image of the fit rows
(``a' = c + s * a``, ``s > 0``) H-a reproduces the frozen predictions exactly.

B, labels and outcomes are never inputs: :func:`fit_harmonization` accepts a frame of representation columns only and
:func:`assert_label_free` refuses any other column (:class:`LabelLeakError`), so the adjustment cannot read them.
This is transductive adaptation and is reported as such.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ...transition import FormalStandardizer

VARIANTS = ("H-a", "H-b")
SD_FLOOR = 1e-12                       # the FormalStandardizer zero-variance rule
VENDOR_COLUMN = "vendor"


class LabelLeakError(ValueError):
    """A harmonization input carries something other than representation columns (B, labels or outcomes)."""


def assert_label_free(columns: Iterable[str], allowed: Iterable[str], *, forbidden: Iterable[str] = ()) -> None:
    """Refuse any column outside ``allowed`` (the representation's own columns) and any ``forbidden`` name."""

    cols = [str(c) for c in columns]
    allowed_set = {str(c) for c in allowed}
    leaked = sorted(set(cols) & {str(f) for f in forbidden})
    extra = sorted(set(cols) - allowed_set)
    if leaked or extra:
        raise LabelLeakError(f"harmonization may read representation columns only; refused {leaked + [c for c in extra if c not in leaked]}")


@dataclass(frozen=True)
class LocationScale:
    mu: np.ndarray
    sd: np.ndarray
    n_rows: int
    group: str

    def to_dict(self) -> dict[str, Any]:
        return {"group": self.group, "n_rows": int(self.n_rows), "mu": self.mu.tolist(), "sd": self.sd.tolist()}


def location_scale(a: np.ndarray, group: str = "all") -> LocationScale:
    """Per-column mean and population SD (ddof = 0, as the v6 standardizer) over the rows with finite A."""

    x = np.asarray(a, dtype=float)
    rows = x[np.all(np.isfinite(x), axis=1)]
    if rows.shape[0] == 0:
        raise ValueError(f"harmonization group {group!r} has no finite rows")
    return LocationScale(rows.mean(axis=0), rows.std(axis=0, ddof=0), int(rows.shape[0]), str(group))


def fit_harmonization(a_frame: pd.DataFrame, columns: list[str], *, variant: str, vendor: pd.Series | None = None,
                      min_rows_per_vendor: int = 10, forbidden: Iterable[str] = ()) -> dict[str, Any]:
    """Target statistics of ``variant`` from the representation columns of the target partition.

    ``a_frame`` must hold exactly the representation ``columns`` (index = sample_id); ``vendor`` (H-b only) is the
    vendor metadata aligned with it.  Returns ``{"variant", "groups": {name: LocationScale}, "row_group": Series,
    "fallback_groups": [...]}``."""

    if variant not in VARIANTS:
        raise ValueError(f"unknown harmonization variant {variant!r}")
    assert_label_free(a_frame.columns, columns, forbidden=forbidden)
    x = a_frame[list(columns)].to_numpy(dtype=float)
    overall = location_scale(x, "all")
    row_group = pd.Series("all", index=a_frame.index, dtype=object)
    groups: dict[str, LocationScale] = {"all": overall}
    fallback: list[dict[str, Any]] = []
    if variant == "H-b":
        if vendor is None:
            raise ValueError("H-b needs the vendor metadata of the target partition")
        if vendor.name not in (VENDOR_COLUMN, None):
            raise LabelLeakError(f"H-b reads vendor metadata only, not {vendor.name!r}")
        v = vendor.reindex(a_frame.index).astype(object).where(vendor.reindex(a_frame.index).notna(), "<missing>").astype(str)
        finite = np.all(np.isfinite(x), axis=1)
        for name in sorted(v.unique()):
            member = (v == name).to_numpy()
            n_finite = int((member & finite).sum())
            if n_finite < int(min_rows_per_vendor):
                fallback.append({"vendor": name, "n_finite_rows": n_finite, "statistics": "H-a (all rows of the target partition)"})
                continue
            groups[f"vendor:{name}"] = location_scale(x[member], f"vendor:{name}")
            row_group[member] = f"vendor:{name}"
    return {"variant": variant, "groups": groups, "row_group": row_group, "fallback_groups": fallback}


def harmonized_standardizer(frozen: FormalStandardizer, stats: LocationScale) -> tuple[FormalStandardizer, list[int]]:
    """The frozen standardizer with the target mean and SD on its kept columns (the kept set never changes)."""

    if stats.mu.size != frozen.mu.size:
        raise ValueError("target statistics and the frozen standardizer have different widths")
    mu = frozen.mu.astype(float).copy()
    sd = frozen.sd.astype(float).copy()
    keep = frozen.keep.astype(bool)
    degenerate = [int(j) for j in np.flatnonzero(keep & ~(np.isfinite(stats.sd) & (stats.sd > SD_FLOOR)))]
    ok = keep & np.isfinite(stats.sd) & (stats.sd > SD_FLOOR)
    mu[keep] = stats.mu[keep]
    sd[ok] = stats.sd[ok]
    return FormalStandardizer(keep.copy(), mu, sd), degenerate


def with_standardizer(model: Any, standardizer: FormalStandardizer) -> Any:
    """A copy of a frozen dataclass model (``TransitionModel``, ``FormalPredictor``) with another standardizer."""

    if not dataclasses.is_dataclass(model) or not hasattr(model, "standardizer"):
        raise TypeError(f"{type(model).__name__} has no A standardization to harmonize")
    return dataclasses.replace(model, standardizer=standardizer)


def apply_harmonized(model: Any, a: np.ndarray, harmonization: dict[str, Any], index: pd.Index,
                     predict: str = "predict") -> tuple[Any, dict[str, Any]]:
    """Apply ``model`` with the harmonized standardizer of each row's group; returns (predictions, record).

    ``predict`` names the method: ``predict`` (TransitionModel -> DataFrame of B-hat) or ``predict_proba``
    (FormalPredictor -> DataFrame of probabilities).  Row order of the result follows ``index``."""

    row_group = harmonization["row_group"].reindex(index)
    a = np.asarray(a, dtype=float)
    parts: list[pd.DataFrame] = []
    record: dict[str, Any] = {"variant": harmonization["variant"], "groups": {}, "fallback_groups": harmonization["fallback_groups"]}
    for name in sorted(row_group.unique()):
        member = (row_group == name).to_numpy()
        std, degenerate = harmonized_standardizer(model.standardizer, harmonization["groups"][name])
        out = getattr(with_standardizer(model, std), predict)(a[member])
        out = pd.DataFrame(out)
        out.index = index[member]
        parts.append(out)
        record["groups"][name] = {"n_rows": int(member.sum()), "n_fit_rows": harmonization["groups"][name].n_rows,
                                  "degenerate_kept_columns": degenerate}
    return pd.concat(parts).reindex(index), record
