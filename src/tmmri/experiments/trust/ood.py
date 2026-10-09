"""S2 label-free OOD scores: Mahalanobis distance to the ``acdc_dev`` distribution with Ledoit-Wolf shrinkage.

The reference standardizes the representation with the fit-partition mean and SD (zero-variance columns removed, the
v6 :class:`tmmri.transition.FormalStandardizer`), optionally projects onto a PCA fitted on the same rows (D20: 50
components), and estimates the Ledoit-Wolf covariance there.  The score is ``sqrt((y - m)' P (y - m))``.  Nothing here
reads B or labels; the flag threshold is a quantile of development out-of-fold scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.covariance import LedoitWolf

from ...transition import FormalStandardizer


@dataclass
class MahalanobisReference:
    standardizer: FormalStandardizer
    location: np.ndarray                 # mean of the (projected) standardized fit rows
    precision: np.ndarray                # Ledoit-Wolf precision
    shrinkage: float
    n_fit: int
    pca_mean: np.ndarray | None = None
    pca_components: np.ndarray | None = None   # (k, n_kept)

    @property
    def dimension(self) -> int:
        return int(self.location.size)

    def _project(self, x: np.ndarray) -> np.ndarray:
        z = self.standardizer.transform(np.asarray(x, dtype=float))
        if self.pca_components is not None:
            z = (z - self.pca_mean) @ self.pca_components.T
        return z

    def score(self, x: np.ndarray) -> np.ndarray:
        """Mahalanobis distance per row; NaN for rows with a non-finite kept value."""

        x = np.asarray(x, dtype=float)
        out = np.full(x.shape[0], np.nan)
        finite = np.all(np.isfinite(x[:, self.standardizer.keep]), axis=1)
        if finite.any():
            d = self._project(x[finite]) - self.location
            out[finite] = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, self.precision, d), 0.0))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "tmmri.mahalanobis_reference/1", "standardizer": self.standardizer.to_dict(), "location": self.location.tolist(),
                "precision": self.precision.tolist(), "shrinkage": float(self.shrinkage), "n_fit": int(self.n_fit),
                "pca_mean": None if self.pca_mean is None else self.pca_mean.tolist(),
                "pca_components": None if self.pca_components is None else self.pca_components.tolist()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MahalanobisReference":
        return cls(standardizer=FormalStandardizer.from_dict(d["standardizer"]), location=np.asarray(d["location"], dtype=float),
                   precision=np.asarray(d["precision"], dtype=float), shrinkage=float(d["shrinkage"]), n_fit=int(d["n_fit"]),
                   pca_mean=None if d.get("pca_mean") is None else np.asarray(d["pca_mean"], dtype=float),
                   pca_components=None if d.get("pca_components") is None else np.asarray(d["pca_components"], dtype=float))


def _pca(z: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic PCA by SVD of the centred rows; each component's largest-magnitude loading is positive."""

    mean = z.mean(axis=0)
    _, _, vt = np.linalg.svd(z - mean, full_matrices=False)
    k = int(min(n_components, vt.shape[0], max(z.shape[0] - 1, 1)))
    comps = vt[:k].copy()
    signs = np.sign(comps[np.arange(k), np.argmax(np.abs(comps), axis=1)])
    signs[signs == 0] = 1.0
    return mean, comps * signs[:, None]


def fit_mahalanobis_reference(x_fit: np.ndarray, *, pca_components: int | None = None) -> MahalanobisReference:
    """Reference on the fit rows (finite rows only): standardize, optional PCA, Ledoit-Wolf."""

    x = np.asarray(x_fit, dtype=float)
    x = x[np.all(np.isfinite(x), axis=1)]
    if x.shape[0] < 3:
        raise ValueError("the Mahalanobis reference needs at least three finite fit rows")
    std = FormalStandardizer.fit(x)
    z = std.transform(x)
    pca_mean = comps = None
    if pca_components:
        pca_mean, comps = _pca(z, int(pca_components))
        z = (z - pca_mean) @ comps.T
    lw = LedoitWolf(store_precision=True, assume_centered=False).fit(z)
    return MahalanobisReference(standardizer=std, location=np.asarray(lw.location_, dtype=float), precision=np.asarray(lw.precision_, dtype=float),
                                shrinkage=float(lw.shrinkage_), n_fit=int(x.shape[0]), pca_mean=pca_mean, pca_components=comps)


def flag_threshold(oof_scores: np.ndarray, quantile: float = 0.95, method: str = "linear") -> float:
    """The flag threshold: the ``quantile`` of the development out-of-fold scores."""

    s = np.asarray(oof_scores, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        raise ValueError("no finite out-of-fold scores")
    return float(np.quantile(s, float(quantile), method=method))
