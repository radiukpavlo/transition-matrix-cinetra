from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


LAMBDA_GRID = np.array([1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10, 30, 100], dtype=float)


@dataclass(frozen=True)
class PreprocessingState:
    b_medians: dict[str, float]
    b_clip_lower: dict[str, float]
    b_clip_upper: dict[str, float]
    a_mu: dict[str, float]
    a_sigma: dict[str, float]
    b_mu: dict[str, float]
    a_columns: list[str]
    b_columns: list[str]


def fit_train_only_preprocessing(a_train: pd.DataFrame, b_train: pd.DataFrame, a_columns: list[str], b_columns: list[str]) -> PreprocessingState:
    b_numeric = b_train[b_columns].apply(pd.to_numeric, errors="coerce").astype(float)
    medians = b_numeric.median(skipna=True).fillna(0.0).to_dict()
    b_imputed = b_numeric.fillna(medians)
    lower = b_imputed.quantile(0.01).to_dict()
    upper = b_imputed.quantile(0.99).to_dict()
    b_clipped = b_imputed.clip(lower=pd.Series(lower), upper=pd.Series(upper), axis=1)
    a_numeric = a_train[a_columns].apply(pd.to_numeric, errors="coerce").astype(float)
    a_mu = a_numeric.mean(skipna=True).fillna(0.0).to_dict()
    a_sigma = a_numeric.std(skipna=True, ddof=0).replace(0, 1.0).fillna(1.0).to_dict()
    b_mu = b_clipped.mean(skipna=True).fillna(0.0).to_dict()
    return PreprocessingState(medians, lower, upper, a_mu, a_sigma, b_mu, list(a_columns), list(b_columns))


def transform_a(a: pd.DataFrame, state: PreprocessingState) -> np.ndarray:
    arr = a[state.a_columns].apply(pd.to_numeric, errors="coerce").astype(float).to_numpy(dtype=float)
    mu = np.array([state.a_mu[c] for c in state.a_columns], dtype=float)
    sigma = np.array([state.a_sigma[c] if state.a_sigma[c] != 0 else 1.0 for c in state.a_columns], dtype=float)
    arr = np.where(np.isfinite(arr), arr, mu)
    return (arr - mu) / sigma


def transform_b_centered(b: pd.DataFrame, state: PreprocessingState) -> np.ndarray:
    df = b[state.b_columns].apply(pd.to_numeric, errors="coerce").astype(float).fillna(state.b_medians)
    df = df.clip(lower=pd.Series(state.b_clip_lower), upper=pd.Series(state.b_clip_upper), axis=1)
    arr = df.to_numpy(dtype=float)
    mu = np.array([state.b_mu[c] for c in state.b_columns], dtype=float)
    return arr - mu


def inverse_b_centered(b_centered: np.ndarray, state: PreprocessingState) -> np.ndarray:
    mu = np.array([state.b_mu[c] for c in state.b_columns], dtype=float)
    return b_centered + mu


def fit_ridge_transition(a_z: np.ndarray, b_c: np.ndarray, lambda_value: float) -> np.ndarray:
    if a_z.ndim != 2 or b_c.ndim != 2:
        raise ValueError("A and B must be 2D matrices.")
    if a_z.shape[0] != b_c.shape[0]:
        raise ValueError(f"Row mismatch: A has {a_z.shape[0]}, B has {b_c.shape[0]}.")
    k = a_z.shape[1]
    lhs = a_z.T @ a_z + float(lambda_value) * np.eye(k)
    rhs = a_z.T @ b_c
    return np.linalg.solve(lhs, rhs)


def predict_b_hat(a: pd.DataFrame, t_matrix: np.ndarray, state: PreprocessingState) -> pd.DataFrame:
    a_z = transform_a(a, state)
    b_hat = inverse_b_centered(a_z @ t_matrix, state)
    out = pd.DataFrame(b_hat, columns=state.b_columns)
    for col in ["sample_id", "patient_id", "dataset_source", "split"]:
        if col in a.columns:
            out.insert(0 if col == "sample_id" else len(out.columns), col, a[col].to_numpy())
    return out


def assert_transition_shape(a: np.ndarray, t_matrix: np.ndarray, b: np.ndarray) -> None:
    pred = a @ t_matrix
    if pred.shape != b.shape:
        raise AssertionError(f"A @ T shape {pred.shape} does not match B shape {b.shape}.")


# ---------------------------------------------------------------------------------------------------------------
# Protocol v6 ridge core (SPECIFICATION.md 6.0.0, Section 9.2; Decision D10). The legacy functions above are unchanged.
# ---------------------------------------------------------------------------------------------------------------

RIDGE_CONVENTIONS = ("sum_loss", "mean_loss")


def svd_rank_cutoff(singular_values: np.ndarray, n_rows: int, n_cols: int) -> float:
    """Singular values at or below max(n, p) * eps * sigma_max are treated as zero (numpy.linalg.pinv default)."""
    s = np.asarray(singular_values, dtype=float)
    if s.size == 0:
        return 0.0
    return float(max(int(n_rows), int(n_cols)) * np.finfo(float).eps * float(np.max(s)))


def ridge_from_svd(
    singular_values: np.ndarray,
    vt: np.ndarray,
    projected_targets: np.ndarray,
    penalty: float,
    keep: np.ndarray | None = None,
) -> np.ndarray:
    """Ridge coefficients from a precomputed thin SVD ``Z_c = U S V'`` and ``projected_targets = U' y_c``.

    ``w = V diag(s / (s^2 + penalty)) U' y_c``. ``keep`` masks singular values treated as zero (rank cutoff); with
    ``keep=None`` no value is masked, which is the exact arithmetic of the v5 experiment grid.
    """
    s = np.asarray(singular_values, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        factors = s / (s**2 + float(penalty))
    if keep is not None:
        factors = np.where(np.asarray(keep, dtype=bool), factors, 0.0)
    if np.ndim(projected_targets) == 1:
        return vt.T @ (factors * projected_targets)
    return vt.T @ (factors[:, None] * projected_targets)


def ridge_penalty(lambda_value: float, n_rows: int, convention: str) -> float:
    """Penalty on the normal equations: lambda (sum loss, P6/v5 convention) or n * lambda (mean loss)."""
    if convention not in RIDGE_CONVENTIONS:
        raise ValueError(f"Unknown ridge objective convention {convention!r}; expected one of {RIDGE_CONVENTIONS}")
    lam = float(lambda_value)
    if lam < 0 or not np.isfinite(lam):
        raise ValueError("Ridge lambda must be finite and non-negative")
    return lam * (int(n_rows) if convention == "mean_loss" else 1.0)


def ridge_svd(
    z: np.ndarray,
    y: np.ndarray,
    lambda_value: float,
    *,
    convention: str = "sum_loss",
    cutoff: bool = True,
) -> tuple[np.ndarray, np.ndarray | float]:
    """Ridge with an unpenalized intercept via the thin SVD of the centred design (X-ray E4-E5 structure).

    Returns ``(w, c)`` with ``y_hat = z @ w + c``. ``lambda_value = 0`` with ``cutoff=True`` is the minimum-norm
    least-squares (pseudoinverse) solution.
    """
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    if z.ndim != 2 or y.ndim not in (1, 2) or z.shape[0] != y.shape[0]:
        raise ValueError("ridge_svd requires a 2D design and aligned 1D or 2D targets")
    if z.shape[0] == 0:
        raise ValueError("ridge_svd requires at least one row")
    z_mean = z.mean(axis=0)
    y_mean = y.mean(axis=0)
    zc = z - z_mean
    yc = y - y_mean
    u, s, vt = np.linalg.svd(zc, full_matrices=False)
    keep = s > svd_rank_cutoff(s, zc.shape[0], zc.shape[1]) if cutoff else None
    w = ridge_from_svd(s, vt, u.T @ yc, ridge_penalty(lambda_value, zc.shape[0], convention), keep)
    c = y_mean - z_mean @ w
    return w, c


@dataclass(frozen=True)
class MaskedRidgeFit:
    """Per-target ridge fitted on each target's observed rows; ``y_hat = z @ coef + intercept``."""

    coef: np.ndarray
    intercept: np.ndarray
    lambdas: np.ndarray
    fitted: np.ndarray
    n_observed: np.ndarray
    excluded_targets: dict[int, str]
    convention: str

    @property
    def intercept_raw(self) -> np.ndarray:
        return self.intercept


def fit_masked_ridge(
    z: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray | None,
    lam: float | np.ndarray,
    *,
    convention: str = "sum_loss",
    min_rows: int = 10,
    cutoff: bool = True,
) -> MaskedRidgeFit:
    """Fit each target on its own observed rows (``mask[i, j]``); targets with fewer than ``min_rows`` are excluded."""
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    n, p = z.shape
    l = y.shape[1]
    m = np.isfinite(y) if mask is None else (np.asarray(mask, dtype=bool) & np.isfinite(y))
    lambdas = np.broadcast_to(np.asarray(lam, dtype=float), (l,)).astype(float)
    coef = np.zeros((p, l), dtype=float)
    intercept = np.full(l, np.nan, dtype=float)
    fitted = np.zeros(l, dtype=bool)
    excluded: dict[int, str] = {}
    counts = m.sum(axis=0)
    for j in range(l):
        if counts[j] < int(min_rows):
            excluded[j] = f"fewer_than_{int(min_rows)}_observed_rows"
            continue
        rows = m[:, j]
        w, c = ridge_svd(z[rows], y[rows, j], float(lambdas[j]), convention=convention, cutoff=cutoff)
        coef[:, j] = w
        intercept[j] = float(c)
        fitted[j] = True
    return MaskedRidgeFit(coef, intercept, lambdas, fitted, counts.astype(int), excluded, convention)


def predict_masked_ridge(fit: MaskedRidgeFit, z: np.ndarray) -> np.ndarray:
    out = np.asarray(z, dtype=float) @ fit.coef + fit.intercept
    out[:, ~fit.fitted] = np.nan
    return out


def predict(fit: MaskedRidgeFit, z: np.ndarray) -> np.ndarray:
    return predict_masked_ridge(fit, z)


def pseudoinverse_transition(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Historical P1 baseline on complete rows: T = pinv(A_c) B_c with the same centring convention as ridge."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ac = a - a.mean(axis=0)
    bc = b - b.mean(axis=0)
    u, s, vt = np.linalg.svd(ac, full_matrices=False)
    keep = s > svd_rank_cutoff(s, ac.shape[0], ac.shape[1])
    return ridge_from_svd(s, vt, u.T @ bc, 0.0, keep) if np.any(keep) else np.zeros((a.shape[1], b.shape[1]))


@dataclass(frozen=True)
class FormalStandardizer:
    """Zero-variance removal and fit-partition mean/SD standardization of formal A (configs/transition.yaml
    latent_preprocessing; configs/formal.yaml standardization). Population SD (ddof = 0); a column is dropped when its
    fit SD is not finite or not above 1e-12."""

    keep: np.ndarray
    mu: np.ndarray
    sd: np.ndarray

    @classmethod
    def fit(cls, a: np.ndarray) -> "FormalStandardizer":
        a = np.asarray(a, dtype=float)
        if a.ndim != 2 or a.shape[0] == 0:
            raise ValueError("FormalStandardizer.fit requires a non-empty 2D matrix")
        if not np.all(np.isfinite(a)):
            raise ValueError("Formal A contains non-finite values on fit rows (INPUT_NONFINITE rows must be excluded)")
        mu = a.mean(axis=0)
        sd = a.std(axis=0, ddof=0)
        keep = np.isfinite(sd) & (sd > 1e-12)
        return cls(keep, mu, sd)

    @property
    def n_kept(self) -> int:
        return int(self.keep.sum())

    def transform(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        return (a[:, self.keep] - self.mu[self.keep]) / self.sd[self.keep]

    def to_dict(self) -> dict[str, list]:
        return {"keep": self.keep.astype(bool).tolist(), "mu": self.mu.tolist(), "sd": self.sd.tolist()}

    @classmethod
    def from_dict(cls, d: dict[str, list]) -> "FormalStandardizer":
        return cls(np.asarray(d["keep"], dtype=bool), np.asarray(d["mu"], dtype=float), np.asarray(d["sd"], dtype=float))
