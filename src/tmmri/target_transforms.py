"""Registry-driven typed target transforms for the v6 transition (SPECIFICATION.md 6.0.0, Section 9.1; Decision D10).

Every transform, its clip and the clipping quantiles are read from ``registry/semantic_registry.yaml`` (per-feature
``target_transform``) and ``configs/transition.yaml`` (``target_transforms``); no PRESPEC value is hard-coded here.
Parameters (transformed-space clip bounds, mean and SD) are fitted on the observed fit rows only and frozen.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

TRANSFORM_KINDS = ("log", "logit_percent", "identity")
OBSERVED = "OBSERVED"
STATE_SUFFIX = "__state"


def registry_feature_map(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Feature entries of the semantic registry keyed by name (YAML merge keys already resolved)."""
    return {str(entry["name"]): entry for entry in registry.get("features", [])}


def registry_transform_kind(registry: dict[str, Any], feature: str) -> str:
    entry = registry_feature_map(registry).get(feature)
    if entry is None:
        raise KeyError(f"{feature} is not a registry feature")
    kind = str(entry.get("target_transform"))
    if kind not in TRANSFORM_KINDS:
        raise ValueError(f"{feature} has target_transform {kind!r}; only {TRANSFORM_KINDS} are transition targets")
    return kind


def logit_clip_from_config(cfg: dict[str, Any] | None) -> tuple[float, float]:
    try:
        lo, hi = (cfg or {})["target_transforms"]["logit_percent"]["clip"]
    except (KeyError, TypeError, ValueError) as exc:
        raise KeyError("configs/transition.yaml target_transforms.logit_percent.clip is required for logit_percent") from exc
    lo, hi = float(lo), float(hi)
    if not 0.0 < lo < hi < 1.0:
        raise ValueError(f"logit_percent clip must satisfy 0 < lo < hi < 1, got {(lo, hi)}")
    return lo, hi


def clipping_quantiles_from_config(cfg: dict[str, Any] | None) -> tuple[float, float] | None:
    clipping = ((cfg or {}).get("target_transforms") or {}).get("clipping")
    if not clipping:
        return None
    if clipping.get("space", "transformed") != "transformed":
        raise ValueError("Only transformed-space clipping is implemented (configs/transition.yaml target_transforms.clipping.space)")
    lo, hi = clipping["quantiles"]
    return float(lo), float(hi)


@dataclass(frozen=True)
class TypedTransform:
    """One elementwise transform ``g`` and its inverse; values outside the domain map to NaN."""

    kind: str
    clip: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.kind not in TRANSFORM_KINDS:
            raise ValueError(f"Unknown transform {self.kind!r}")
        if self.kind == "logit_percent" and self.clip is None:
            raise ValueError("logit_percent requires the configured clip")

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.kind == "identity":
                return x.copy()
            if self.kind == "log":
                return np.where(x > 0, np.log(np.where(x > 0, x, 1.0)), np.nan)
            lo, hi = self.clip  # type: ignore[misc]
            p = np.clip(x / 100.0, lo, hi)
            return np.where(np.isfinite(x), np.log(p) - np.log1p(-p), np.nan)

    def inverse(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        if self.kind == "identity":
            return z.copy()
        if self.kind == "log":
            with np.errstate(over="ignore"):
                return np.exp(z)
        with np.errstate(over="ignore"):
            return 100.0 / (1.0 + np.exp(-z))


def get_transform(kind: str, cfg: dict[str, Any] | None = None, *, clip: tuple[float, float] | None = None) -> TypedTransform:
    if kind == "logit_percent":
        return TypedTransform(kind, clip if clip is not None else logit_clip_from_config(cfg))
    return TypedTransform(kind)


def observed_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Float values with NaN for every cell that is not finite or whose ``__state`` companion is not OBSERVED."""
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float) if column in frame.columns else np.full(len(frame), np.nan)
    state_col = f"{column}{STATE_SUFFIX}"
    if state_col in frame.columns:
        values = np.where(frame[state_col].astype("string").fillna("").to_numpy() == OBSERVED, values, np.nan)
    return np.where(np.isfinite(values), values, np.nan)


@dataclass
class TargetFit:
    name: str
    kind: str
    clip: tuple[float, float] | None
    clip_quantiles: tuple[float, float] | None
    lo: float | None          # transformed-space clip bounds (fit rows)
    hi: float | None
    mu: float                 # mean of the (clipped) transformed fit values
    sd: float                 # population SD (ddof = 0) of the (clipped) transformed fit values
    n_fit: int
    fit_mean_original: float  # unclipped observed fit values, original units (mean-baseline comparator)
    fit_sd_original: float    # sample SD (ddof = 1), original units (metric normalization)
    fit_min_original: float
    fit_max_original: float

    @property
    def transform(self) -> TypedTransform:
        return TypedTransform(self.kind, self.clip)

    @property
    def usable(self) -> bool:
        return bool(self.n_fit > 0 and np.isfinite(self.sd) and self.sd > 0)


class TargetTransformSet:
    """Typed transform, transformed-space clipping and standardization per target, fitted on fit rows only."""

    def __init__(self, fits: dict[str, TargetFit], order: list[str]):
        self.fits = fits
        self.order = list(order)

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        registry: dict[str, Any],
        columns: list[str],
        *,
        cfg: dict[str, Any] | None = None,
        identity: bool = False,
    ) -> "TargetTransformSet":
        clip_quantiles = clipping_quantiles_from_config(cfg)
        logit_clip = None
        fits: dict[str, TargetFit] = {}
        for col in columns:
            kind = "identity" if identity else registry_transform_kind(registry, col)
            if kind == "logit_percent" and logit_clip is None:
                logit_clip = logit_clip_from_config(cfg)
            clip = logit_clip if kind == "logit_percent" else None
            transform = TypedTransform(kind, clip)
            x = observed_values(frame, col)
            z = transform.forward(x)
            valid = np.isfinite(z)
            z_fit = z[valid]
            x_fit = x[valid]
            lo = hi = None
            if z_fit.size and clip_quantiles is not None:
                lo, hi = (float(v) for v in np.quantile(z_fit, clip_quantiles))
                z_fit = np.clip(z_fit, lo, hi)
            n_fit = int(z_fit.size)
            fits[col] = TargetFit(
                name=col,
                kind=kind,
                clip=clip,
                clip_quantiles=clip_quantiles,
                lo=lo,
                hi=hi,
                mu=float(np.mean(z_fit)) if n_fit else float("nan"),
                sd=float(np.std(z_fit, ddof=0)) if n_fit else float("nan"),
                n_fit=n_fit,
                fit_mean_original=float(np.mean(x_fit)) if n_fit else float("nan"),
                fit_sd_original=float(np.std(x_fit, ddof=1)) if n_fit > 1 else float("nan"),
                fit_min_original=float(np.min(x_fit)) if n_fit else float("nan"),
                fit_max_original=float(np.max(x_fit)) if n_fit else float("nan"),
            )
        return cls(fits, list(columns))

    # -- elementwise -----------------------------------------------------------------------------------------
    def transform(self, frame: pd.DataFrame, *, clip: bool = False) -> pd.DataFrame:
        """``g(x)`` (optionally clipped to the fit-row quantiles) without standardization; NaN when unobserved."""
        out = {}
        for col in self.order:
            fit = self.fits[col]
            z = fit.transform.forward(observed_values(frame, col))
            if clip and fit.lo is not None and fit.hi is not None:
                z = np.where(np.isfinite(z), np.clip(z, fit.lo, fit.hi), np.nan)
            out[col] = z
        return pd.DataFrame(out, index=frame.index, columns=self.order)

    def forward(self, frame: pd.DataFrame, *, clip: bool = False) -> pd.DataFrame:
        """Standardized transformed values ``(g(x) - mu) / sd``; ``clip=True`` only for fitting rows."""
        z = self.transform(frame, clip=clip)
        for col in self.order:
            fit = self.fits[col]
            z[col] = (z[col] - fit.mu) / fit.sd if fit.usable else np.nan
        return z

    def inverse(self, frame: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Original units from standardized transformed values."""
        if isinstance(frame, np.ndarray):
            frame = pd.DataFrame(frame, columns=self.order)
        out = {}
        for col in self.order:
            fit = self.fits[col]
            y = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
            out[col] = fit.transform.inverse(y * fit.sd + fit.mu) if fit.usable else np.full(len(y), np.nan)
        return pd.DataFrame(out, index=frame.index, columns=self.order)

    # -- summaries -------------------------------------------------------------------------------------------
    def fit_sd_original(self) -> pd.Series:
        return pd.Series({c: self.fits[c].fit_sd_original for c in self.order}, dtype=float)

    def fit_mean_original(self) -> pd.Series:
        return pd.Series({c: self.fits[c].fit_mean_original for c in self.order}, dtype=float)

    def fit_range_original(self) -> dict[str, tuple[float, float]]:
        return {c: (self.fits[c].fit_min_original, self.fits[c].fit_max_original) for c in self.order}

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "target_transforms_v6", "order": list(self.order), "fits": {c: asdict(self.fits[c]) for c in self.order}}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TargetTransformSet":
        fits = {}
        for name, raw in d["fits"].items():
            raw = dict(raw)
            for key in ("clip", "clip_quantiles"):
                if raw.get(key) is not None:
                    raw[key] = tuple(float(v) for v in raw[key])
            fits[name] = TargetFit(**raw)
        return cls(fits, list(d["order"]))
