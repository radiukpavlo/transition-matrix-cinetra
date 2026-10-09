"""Discretization of rule-antecedent candidates into symbolic states (SPECIFICATION.md 6.0.0, Sections 11.3-11.4;
Decision D11; configs/wedd.yaml).

``wedd_gaussian_mm_v1`` is the CXR-TRANS-AUDIT 3.0.0 Section 11.3.1 definition [INHERITED from P4, with the P5
stability gate]: supervised by the rulebook consequent (entropy over its classes), candidates are midpoints of adjacent
unique values, both children need ``max(10, ceil(0.02 N_fit))`` patients, candidates with information gain below
``min_gain_bits`` are rejected, the conditional entropy and the kernel density at the cut are min-max normalized over the
valid candidates and ``J = w_H H~ + w_p p~`` is minimized; growth is best-first up to ``max_bins`` and ``max_depth``.
``wedd_uniform_mm_v1`` swaps the Gaussian kernel for P5's uniform counting kernel; ``three_quantile_v1`` uses the
fit-partition tertiles (P6/v5 continuity). Bins are right-closed: ``(-inf, t1], (t1, t2], ..., (t_last, +inf)``.

This module reconciles the synthetic ablation in manuscript/scripts/generate_manuscript_assets.py (one cut per feature,
81 quantile candidates, bandwidth 0.20 SD, alpha 0.75, unnormalized costs), which is NOT this profile: v6 uses the
configured profile only.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

MISSING_STATE = "missing"
UNAVAILABLE_PREFIX = "UNAVAILABLE:"
STATE_PATTERN = re.compile(r"^b(\d+)of(\d+)$")
# Reconstructed / measured cell states that make an antecedent symbol unavailable, with the reason code routed at
# inference (registry/reason_codes.yaml eligibility scope).
STATE_TO_UNAVAILABLE_CODE = {
    "NUMERIC_DOMAIN_BLOCKED": "FEATURE_DOMAIN_BLOCKED",
    "TARGET_INELIGIBLE": "FEATURE_TARGET_INELIGIBLE",
    "UNSTABLE_ESTIMATE": "FEATURE_UNSTABLE",
    "QC_BLOCKED": "FEATURE_QC_MASKED",
    "MISSING_SOURCE": "FEATURE_MISSING_SOURCE",
    "NOT_APPLICABLE": "FEATURE_MISSING_SOURCE",
}
_TIE_RTOL = 1e-12
_TIE_ATOL = 1e-15


def state_token(index: int, n_bins: int) -> str:
    """Symbol of bin ``index`` (0-based) of ``n_bins``; the rule identity uses the bin index and the bin count."""
    return f"b{int(index)}of{int(n_bins)}"


def parse_state_token(token: object) -> tuple[int, int] | None:
    match = STATE_PATTERN.match(str(token)) if token is not None else None
    return (int(match.group(1)), int(match.group(2))) if match else None


def unavailable_token(code: str) -> str:
    return f"{UNAVAILABLE_PREFIX}{code}"


def is_available_state(token: object) -> bool:
    """A symbol usable as a literal: not null, not ``missing`` and not ``UNAVAILABLE:<code>``."""
    if token is None or (isinstance(token, float) and math.isnan(token)) or token is pd.NA:
        return False
    text = str(token)
    return bool(text) and text != MISSING_STATE and not text.startswith(UNAVAILABLE_PREFIX) and text not in ("nan", "<NA>", "None")


# ----------------------------------------------------------------------------------------------------------------
# Profiles
# ----------------------------------------------------------------------------------------------------------------
def resolve_profile(profile_name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    profiles = cfg["profiles"]
    if profile_name not in profiles:
        raise KeyError(f"Unknown discretization profile {profile_name!r} (configs/wedd.yaml profiles)")
    raw = dict(profiles[profile_name])
    if "inherits" in raw:
        base = resolve_profile(str(raw.pop("inherits")), cfg)
        base.update(raw)
        raw = base
    raw["name"] = profile_name
    return raw


def _profile_kind(profile: dict[str, Any]) -> str:
    if profile.get("cuts") == "fit_partition_tertiles":
        return "tertiles"
    if profile.get("kernel") in ("gaussian", "uniform_counting"):
        return "wedd"
    raise ValueError(f"Profile {profile.get('name')!r} is neither a WEDD kernel profile nor fit_partition_tertiles")


_MIN_LEAF = re.compile(r"^\s*max\(\s*(\d+)\s*,\s*ceil\(\s*([0-9.]+)\s*\*\s*N_\w+\s*\)\s*\)\s*$")
_BANDWIDTH = re.compile(r"^\s*([0-9.]+)\s*\*\s*min\(\s*sd\s*,\s*IQR\s*/\s*([0-9.]+)\s*\)\s*\*\s*n\^\(\s*(-?\d+)\s*/\s*(\d+)\s*\)\s*$")
_BANDWIDTH_FALLBACK = re.compile(r"^\s*([0-9.]+)\s*\*\s*sd\s*\*\s*n\^\(\s*(-?\d+)\s*/\s*(\d+)\s*\)\s*$")


@dataclass(frozen=True)
class WeddParams:
    kernel: str
    bw_multiplier: float
    iqr_divisor: float
    bw_exponent: float
    fallback_multiplier: float
    fallback_exponent: float
    entropy_base: float
    weight_entropy: float
    weight_density: float
    min_leaf_floor: int
    min_leaf_fraction: float
    min_gain: float
    max_depth: int
    max_bins: int
    bandwidth_scale: float = 1.0

    @classmethod
    def from_profile(cls, profile: dict[str, Any], overrides: dict[str, Any] | None = None) -> "WeddParams":
        p = dict(profile)
        p.update(overrides or {})
        bw = _BANDWIDTH.match(str(p["bandwidth"]))
        fb = _BANDWIDTH_FALLBACK.match(str(p["bandwidth_iqr_zero_fallback"]))
        ml = _MIN_LEAF.match(str(p["min_leaf"]))
        if not (bw and fb and ml):
            raise ValueError("Unrecognized WEDD bandwidth or min_leaf rule in configs/wedd.yaml (fail closed)")
        if p.get("sd_zero", "no_split") != "no_split":
            raise ValueError("Only sd_zero: no_split is implemented")
        if p.get("normalization", "candidate_wise_minmax_zero_range_to_zero") != "candidate_wise_minmax_zero_range_to_zero":
            raise ValueError("Only candidate_wise_minmax_zero_range_to_zero normalization is implemented")
        if p.get("candidates", "midpoints_adjacent_unique_values") != "midpoints_adjacent_unique_values":
            raise ValueError("Only midpoint candidates are implemented")
        if p.get("growth", "best_first") != "best_first" or p.get("bins", "right_closed") != "right_closed":
            raise ValueError("Only best_first growth with right_closed bins is implemented")
        return cls(
            kernel=str(p["kernel"]),
            bw_multiplier=float(bw.group(1)),
            iqr_divisor=float(bw.group(2)),
            bw_exponent=float(bw.group(3)) / float(bw.group(4)),
            fallback_multiplier=float(fb.group(1)),
            fallback_exponent=float(fb.group(2)) / float(fb.group(3)),
            entropy_base=float(p["entropy_base"]),
            weight_entropy=float(p["weight_entropy"]),
            weight_density=float(p["weight_density"]),
            min_leaf_floor=int(ml.group(1)),
            min_leaf_fraction=float(ml.group(2)),
            min_gain=float(p["min_gain_bits"]),
            max_depth=int(p["max_depth"]),
            max_bins=int(p["max_bins"]),
            bandwidth_scale=float(p.get("bandwidth_multiplier", 1.0)),
        )

    def min_leaf(self, n_fit_patients: int) -> int:
        return max(self.min_leaf_floor, int(math.ceil(self.min_leaf_fraction * int(n_fit_patients) - 1e-12)))


# ----------------------------------------------------------------------------------------------------------------
# Core WEDD split search
# ----------------------------------------------------------------------------------------------------------------
def _entropy_rows(counts: np.ndarray, base: float) -> np.ndarray:
    """Entropy of each row of a class-count matrix (0 log 0 = 0)."""
    counts = np.asarray(counts, dtype=float)
    totals = counts.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(totals > 0, counts / totals, 0.0)
        logs = np.where(p > 0, np.log(p), 0.0)
    return -(p * logs).sum(axis=-1) / math.log(base)


def _bandwidth(x: np.ndarray, params: WeddParams) -> tuple[float, str]:
    n = x.size
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    q75, q25 = np.percentile(x, [75, 25])
    iqr = float(q75 - q25)
    if iqr > 0:
        h = params.bw_multiplier * min(sd, iqr / params.iqr_divisor) * n ** params.bw_exponent
        rule = "robust_min_sd_iqr"
    else:
        h = params.fallback_multiplier * sd * n ** params.fallback_exponent
        rule = "iqr_zero_fallback_sd"
    return float(h * params.bandwidth_scale), rule


def _density(candidates: np.ndarray, x: np.ndarray, h: float, kernel: str, chunk: int = 4096) -> np.ndarray:
    out = np.empty(candidates.size, dtype=float)
    n = x.size
    for start in range(0, candidates.size, chunk):
        t = candidates[start:start + chunk, None]
        u = (t - x[None, :]) / h
        if kernel == "gaussian":
            out[start:start + chunk] = np.exp(-0.5 * u**2).sum(axis=1) / (math.sqrt(2.0 * math.pi) * n * h)
        elif kernel == "uniform_counting":
            out[start:start + chunk] = (np.abs(u) <= 1.0).sum(axis=1) / (2.0 * n * h)
        else:
            raise ValueError(f"Unknown kernel {kernel!r}")
    return out


def _minmax(v: np.ndarray) -> tuple[np.ndarray, float, float]:
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi - lo <= 0:
        return np.zeros_like(v), lo, hi
    return (v - lo) / (hi - lo), lo, hi


def _best_split(x: np.ndarray, y: np.ndarray, n_classes: int, params: WeddParams, min_leaf: int, record_candidates: bool) -> dict[str, Any] | None:
    """Best WEDD cut of one node (x sorted ascending, y class codes aligned) or None."""
    n = x.size
    info: dict[str, Any] = {"n": int(n)}
    if n < 2 * min_leaf:
        return None
    if not float(np.std(x, ddof=1)) > 0:
        return None
    h, rule = _bandwidth(x, params)
    if not (np.isfinite(h) and h > 0):
        return None
    uniq = np.unique(x)
    if uniq.size < 2:
        return None
    cands = (uniq[:-1] + uniq[1:]) / 2.0
    n_left = np.searchsorted(x, cands, side="right")
    onehot = np.zeros((n, n_classes), dtype=float)
    onehot[np.arange(n), y] = 1.0
    cum = np.cumsum(onehot, axis=0)
    total = cum[-1]
    left = cum[n_left - 1]
    right = total[None, :] - left
    h_node = float(_entropy_rows(total[None, :], params.entropy_base)[0])
    h_split = (n_left / n) * _entropy_rows(left, params.entropy_base) + ((n - n_left) / n) * _entropy_rows(right, params.entropy_base)
    gain = h_node - h_split
    valid = (n_left >= min_leaf) & ((n - n_left) >= min_leaf) & (gain >= params.min_gain)
    if not valid.any():
        return None
    vc = cands[valid]
    vh = h_split[valid]
    vp = _density(vc, x, h, params.kernel)
    hn, h_lo, h_hi = _minmax(vh)
    pn, p_lo, p_hi = _minmax(vp)
    j = params.weight_entropy * hn + params.weight_density * pn
    j_min = float(np.min(j))
    tied = np.flatnonzero(np.isclose(j, j_min, rtol=_TIE_RTOL, atol=_TIE_ATOL))
    vl = n_left[valid]
    child_min = np.minimum(vl, n - vl)
    order = sorted(tied, key=lambda i: (-int(child_min[i]), float(vc[i])))
    k = order[0]
    info.update(
        {
            "bandwidth": h,
            "bandwidth_rule": rule,
            "n_candidates": int(cands.size),
            "n_valid_candidates": int(valid.sum()),
            "entropy_node": h_node,
            "entropy_min": h_lo,
            "entropy_max": h_hi,
            "density_min": p_lo,
            "density_max": p_hi,
            "selected_cut": float(vc[k]),
            "selected_objective": float(j[k]),
            "selected_entropy": float(vh[k]),
            "selected_density": float(vp[k]),
            "gain_bits": float(gain[valid][k]),
            "support_left": int(vl[k]),
            "support_right": int(n - vl[k]),
            "left_positions_end": int(vl[k]),
        }
    )
    if record_candidates:
        info["candidates"] = {
            "cut": vc.tolist(),
            "entropy": vh.tolist(),
            "density": vp.tolist(),
            "gain_bits": gain[valid].tolist(),
            "support_left": vl.tolist(),
            "objective": j.tolist(),
        }
    return info


def _wedd_cuts(x: np.ndarray, y: np.ndarray, n_classes: int, params: WeddParams, n_fit_patients: int, record_candidates: bool) -> tuple[list[float], list[str], list[dict[str, Any]], int]:
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ys = y[order]
    min_leaf = params.min_leaf(n_fit_patients)
    n_root = xs.size
    leaves: dict[str, tuple[int, int, int]] = {"root": (0, n_root, 0)}  # path -> (start, stop, depth) in sorted order
    split_cache: dict[str, dict[str, Any] | None] = {}
    cuts: list[float] = []
    paths: list[str] = []
    records: list[dict[str, Any]] = []
    while len(cuts) + 1 < params.max_bins:
        best_path = None
        best_priority = -np.inf
        for path in sorted(leaves):
            start, stop, depth = leaves[path]
            if depth >= params.max_depth:
                continue
            if path not in split_cache:
                split_cache[path] = _best_split(xs[start:stop], ys[start:stop], n_classes, params, min_leaf, record_candidates)
            split = split_cache[path]
            if split is None:
                continue
            priority = (stop - start) / n_root * split["gain_bits"]
            if best_path is None or (priority > best_priority and not np.isclose(priority, best_priority, rtol=_TIE_RTOL, atol=_TIE_ATOL)):
                best_path, best_priority = path, priority
        if best_path is None:
            break
        start, stop, depth = leaves.pop(best_path)
        split = split_cache[best_path]
        mid = start + int(split["left_positions_end"])
        leaves[f"{best_path}.L"] = (start, mid, depth + 1)
        leaves[f"{best_path}.R"] = (mid, stop, depth + 1)
        cuts.append(float(split["selected_cut"]))
        paths.append(best_path)
        rec = {k: v for k, v in split.items() if k != "left_positions_end"}
        rec.update({"node_path": best_path, "depth": depth, "priority_weighted_gain": float(best_priority), "min_leaf": min_leaf})
        records.append(rec)
    return cuts, paths, records, min_leaf


def _tertile_cuts(x: np.ndarray) -> tuple[list[float], list[str]]:
    q = np.quantile(x, [1.0 / 3.0, 2.0 / 3.0])
    cuts: list[float] = []
    paths: list[str] = []
    for label, t in zip(("q1", "q2"), q):
        t = float(t)
        if t in cuts:  # duplicate_cuts_removed
            continue
        if (x <= t).any() and (x > t).any():
            cuts.append(t)
            paths.append(label)
    return cuts, paths


# ----------------------------------------------------------------------------------------------------------------
# Cut containers
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class FeatureCuts:
    feature: str
    profile: str
    cuts: list[float]
    node_paths: list[str]
    status: str
    n_fit: int
    fit_iqr: float
    fit_sd: float
    min_leaf: int | None = None
    fit_hash: str = ""
    records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n_bins(self) -> int:
        return len(self.cuts) + 1

    def bin_index(self, x: np.ndarray) -> np.ndarray:
        """Right-closed bin index (0-based); -1 for non-finite values."""
        x = np.asarray(x, dtype=float)
        idx = np.searchsorted(np.asarray(self.cuts, dtype=float), x, side="left")
        return np.where(np.isfinite(x), idx, -1).astype(int)


def _fit_hash(profile: dict[str, Any], feature: str, x: np.ndarray, y_codes: np.ndarray, classes: list[str], n_fit: int) -> str:
    h = hashlib.sha256()
    h.update(json.dumps({"profile": profile, "feature": feature, "classes": classes, "n_fit_patients": n_fit}, sort_keys=True, default=str).encode())
    h.update(np.ascontiguousarray(x, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(y_codes, dtype=np.int64).tobytes())
    return h.hexdigest()


def _decision_codes(decision: pd.Series | np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray]:
    # Vectorized (called once per feature and bootstrap replicate); same classes and codes as the per-row loop:
    # classes sorted by their string form, -1 for unknown decisions.
    d = np.asarray(decision, dtype=object).reshape(-1)
    known = ~np.asarray(pd.isna(d), dtype=bool)
    text = np.array([str(v) for v in d[known]], dtype=str)
    classes, inverse = np.unique(text, return_inverse=True)
    codes = np.full(d.shape[0], -1, dtype=int)
    codes[known] = inverse.reshape(-1)
    return codes, [str(c) for c in classes], known


def fit_feature_cuts(
    x: np.ndarray,
    decision: pd.Series | np.ndarray | None,
    profile: dict[str, Any],
    *,
    feature: str,
    n_fit_patients: int,
    overrides: dict[str, Any] | None = None,
    record_candidates: bool = True,
) -> FeatureCuts:
    x = np.asarray(x, dtype=float)
    kind = _profile_kind(profile)
    finite = np.isfinite(x)
    if kind == "wedd":
        if decision is None:
            raise ValueError("WEDD is supervised; a decision vector is required")
        codes, classes, known = _decision_codes(decision)
        rows = finite & known
    else:
        codes, classes = np.zeros(x.size, dtype=int), []
        rows = finite
    xv = x[rows]
    yv = codes[rows]
    fit_iqr = float(np.subtract(*np.percentile(xv, [75, 25]))) if xv.size else float("nan")
    fit_sd = float(np.std(xv, ddof=1)) if xv.size > 1 else float("nan")
    fhash = _fit_hash({**profile, **(overrides or {})}, feature, xv, yv, classes, int(n_fit_patients))
    base = dict(feature=feature, profile=str(profile["name"]), n_fit=int(xv.size), fit_iqr=fit_iqr, fit_sd=fit_sd, fit_hash=fhash)
    if xv.size == 0:
        return FeatureCuts(cuts=[], node_paths=[], status="NO_CUT_MISSING_ONLY", **base)
    if not np.ptp(xv) > 0:
        return FeatureCuts(cuts=[], node_paths=[], status="NO_CUT_CONSTANT", **base)
    if kind == "tertiles":
        cuts, paths = _tertile_cuts(xv)
        return FeatureCuts(cuts=sorted(cuts), node_paths=[p for _, p in sorted(zip(cuts, paths))], status="CUT" if cuts else "NO_CUT_NO_VALID_CANDIDATE", **base)
    params = WeddParams.from_profile(profile, overrides)
    cuts, paths, records, min_leaf = _wedd_cuts(xv, yv, len(classes), params, int(n_fit_patients), record_candidates)
    pairs = sorted(zip(cuts, paths))
    return FeatureCuts(
        cuts=[c for c, _ in pairs],
        node_paths=[p for _, p in pairs],
        status="CUT" if cuts else "NO_CUT_NO_VALID_CANDIDATE",
        min_leaf=min_leaf,
        records=records,
        **base,
    )


CANDIDATE_COLUMNS = ["feature", "profile", "fit_hash", "node_path", "depth", "node_n", "min_leaf", "bandwidth", "bandwidth_rule",
                     "entropy_node", "entropy_min", "entropy_max", "density_min", "density_max", "n_candidates", "n_valid_candidates",
                     "cut", "entropy", "density", "entropy_normalized", "density_normalized", "objective", "gain_bits",
                     "support_left", "support_right", "selected"]


class CutSet:
    """Frozen cuts per feature; ``transform`` maps values to state tokens ``b<i>of<n>`` (right-closed bins)."""

    def __init__(self, features: dict[str, FeatureCuts], profile_name: str, order: list[str] | None = None):
        self.features = features
        self.profile_name = profile_name
        self.order = list(order) if order is not None else list(features)

    def cuts(self) -> dict[str, list[float]]:
        return {f: list(self.features[f].cuts) for f in self.order}

    def bin_indices(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for f in self.order:
            x = pd.to_numeric(frame[f], errors="coerce").to_numpy(dtype=float) if f in frame.columns else np.full(len(frame), np.nan)
            out[f] = self.features[f].bin_index(x)
        return pd.DataFrame(out, index=frame.index, columns=self.order)

    def transform(self, frame: pd.DataFrame, *, states: pd.DataFrame | None = None) -> pd.DataFrame:
        """State tokens per feature. A cell is ``missing`` when non-finite, or ``UNAVAILABLE:<code>`` when its
        reconstructed/measured state (``states`` or the frame's ``<feature>__state`` companion) is a blocking state
        (NUMERIC_DOMAIN_BLOCKED, TARGET_INELIGIBLE, UNSTABLE_ESTIMATE, QC_BLOCKED, MISSING_SOURCE, NOT_APPLICABLE)."""
        idx = self.bin_indices(frame)
        out = {}
        for f in self.order:
            n_bins = self.features[f].n_bins
            tokens = np.array([state_token(i, n_bins) if i >= 0 else MISSING_STATE for i in idx[f].to_numpy()], dtype=object)
            cell_states = None
            if states is not None and f in states.columns:
                cell_states = states[f].astype("string").fillna("").to_numpy()
            elif f"{f}__state" in frame.columns:
                cell_states = frame[f"{f}__state"].astype("string").fillna("").to_numpy()
            if cell_states is not None:
                for i, s in enumerate(cell_states):
                    code = STATE_TO_UNAVAILABLE_CODE.get(str(s))
                    if code is not None:
                        tokens[i] = unavailable_token(code)
                    elif str(s) not in ("", "OBSERVED", "ESTIMATED"):
                        tokens[i] = MISSING_STATE
            out[f] = tokens
        return pd.DataFrame(out, index=frame.index, columns=self.order)

    def retain(self, stability: pd.DataFrame, min_recurrence: float | None = None) -> "CutSet":
        """Drop cuts that fail the bootstrap gate (``retained`` column, or recurrence below ``min_recurrence`` when
        given); adjacent bins merge. A cut absent from the stability table is dropped (non-recurrence)."""
        new = {}
        for f in self.order:
            fc = self.features[f]
            rows = stability[(stability["feature"] == f) & stability["cut"].notna()]
            kept = []
            for cut, path in zip(fc.cuts, fc.node_paths):
                match = rows[(rows["node_path"] == path) & np.isclose(rows["cut"].astype(float), cut, rtol=0, atol=1e-12)]
                if match.empty:
                    continue
                ok = bool(match["retained"].iloc[0]) if min_recurrence is None else float(match["recurrence"].iloc[0]) >= float(min_recurrence)
                if ok:
                    kept.append((cut, path))
            new[f] = FeatureCuts(**{**asdict(fc), "cuts": [c for c, _ in kept], "node_paths": [p for _, p in kept], "status": fc.status if kept or not fc.cuts else "NO_CUT_UNSTABLE"})
        return CutSet(new, self.profile_name, self.order)

    def records_frame(self) -> pd.DataFrame:
        rows = []
        for f in self.order:
            fc = self.features[f]
            for rec in fc.records:
                row = {k: v for k, v in rec.items() if k != "candidates"}
                row.update({"feature": f, "profile": fc.profile, "fit_hash": fc.fit_hash})
                rows.append(row)
        return pd.DataFrame(rows)

    def candidates_frame(self) -> pd.DataFrame:
        """Every valid candidate cut of every split node with its objective terms (``wedd_candidates.parquet``;
        SPECIFICATION.md 11.3.1): entropy, density, their candidate-wise min-max normalizations, objective J, gain,
        supports, bandwidth, normalization extrema, node path and fit hash; ``selected`` marks the chosen cut."""
        rows = []
        for f in self.order:
            fc = self.features[f]
            for rec in fc.records:
                cand = rec.get("candidates") or {}
                n = int(rec.get("n", 0))
                e_lo, e_hi = float(rec.get("entropy_min", np.nan)), float(rec.get("entropy_max", np.nan))
                d_lo, d_hi = float(rec.get("density_min", np.nan)), float(rec.get("density_max", np.nan))
                for i, cut in enumerate(cand.get("cut", [])):
                    e, d = float(cand["entropy"][i]), float(cand["density"][i])
                    left = int(cand["support_left"][i])
                    rows.append({"feature": f, "profile": fc.profile, "fit_hash": fc.fit_hash, "node_path": rec.get("node_path"),
                                 "depth": rec.get("depth"), "node_n": n, "min_leaf": rec.get("min_leaf"), "bandwidth": rec.get("bandwidth"),
                                 "bandwidth_rule": rec.get("bandwidth_rule"), "entropy_node": rec.get("entropy_node"),
                                 "entropy_min": e_lo, "entropy_max": e_hi, "density_min": d_lo, "density_max": d_hi,
                                 "n_candidates": rec.get("n_candidates"), "n_valid_candidates": rec.get("n_valid_candidates"),
                                 "cut": float(cut), "entropy": e, "density": d,
                                 "entropy_normalized": (e - e_lo) / (e_hi - e_lo) if e_hi > e_lo else 0.0,
                                 "density_normalized": (d - d_lo) / (d_hi - d_lo) if d_hi > d_lo else 0.0,
                                 "objective": float(cand["objective"][i]), "gain_bits": float(cand["gain_bits"][i]),
                                 "support_left": left, "support_right": n - left,
                                 "selected": bool(float(cut) == float(rec.get("selected_cut", np.nan)))})
        return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "cutset_v6", "profile": self.profile_name, "order": list(self.order), "features": {f: asdict(self.features[f]) for f in self.order}}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CutSet":
        return cls({f: FeatureCuts(**raw) for f, raw in d["features"].items()}, str(d["profile"]), list(d["order"]))


def fit_cuts(
    values: pd.DataFrame,
    decision: pd.Series | np.ndarray | None,
    profile_name: str,
    cfg: dict[str, Any],
    *,
    features: Iterable[str] | None = None,
    n_fit_patients: int | None = None,
    overrides: dict[str, Any] | None = None,
    record_candidates: bool = True,
) -> CutSet:
    """Fit cuts per feature on the fit rows given (the substrate of configs/wedd.yaml), supervised by ``decision``.

    ``n_fit_patients`` defaults to the number of rows (the min_leaf rule uses the fit-partition patient count)."""
    profile = resolve_profile(profile_name, cfg)
    cols = list(features) if features is not None else [c for c in values.columns if not str(c).endswith(("__state", "__evidence"))]
    n_fit = int(n_fit_patients) if n_fit_patients is not None else len(values)
    d = None if decision is None else np.asarray(decision, dtype=object)
    out = {}
    for f in cols:
        x = pd.to_numeric(values[f], errors="coerce").to_numpy(dtype=float)
        state_col = f"{f}__state"
        if state_col in values.columns:
            x = np.where(values[state_col].astype("string").fillna("").to_numpy() == "OBSERVED", x, np.nan)
        out[f] = fit_feature_cuts(x, d, profile, feature=f, n_fit_patients=n_fit, overrides=overrides, record_candidates=record_candidates)
    return CutSet(out, profile_name, cols)


# ----------------------------------------------------------------------------------------------------------------
# Bootstrap stability gate [INHERITED, P5] and per-object stability
# ----------------------------------------------------------------------------------------------------------------
_TOLERANCE = re.compile(r"^\s*([0-9.]+)_fit_iqr_fallback_([0-9.]+)_fit_sd\s*$")


def _tolerance(fc: FeatureCuts, rule: str) -> float:
    m = _TOLERANCE.match(str(rule))
    if not m:
        raise ValueError(f"Unrecognized stability match_tolerance {rule!r}")
    if np.isfinite(fc.fit_iqr) and fc.fit_iqr > 0:
        return float(m.group(1)) * fc.fit_iqr
    return float(m.group(2)) * (fc.fit_sd if np.isfinite(fc.fit_sd) else 0.0)


def _bootstrap_chunk(
    x: np.ndarray,
    d: np.ndarray | None,
    group_rows: list[np.ndarray],
    seeds: list[np.random.SeedSequence],
    profile: dict[str, Any],
    feature_names: list[str],
    n_fit: int,
    overrides: dict[str, Any] | None,
) -> list[dict[str, list[tuple[str, float]]]]:
    results = []
    n_groups = len(group_rows)
    for ss in seeds:
        rng = np.random.default_rng(ss)
        draw = rng.integers(0, n_groups, size=n_groups)
        rows = np.concatenate([group_rows[g] for g in draw])
        rep: dict[str, list[tuple[str, float]]] = {}
        for j, f in enumerate(feature_names):
            fc = fit_feature_cuts(x[rows, j], None if d is None else d[rows], profile, feature=f, n_fit_patients=n_fit, overrides=overrides, record_candidates=False)
            rep[f] = list(zip(fc.node_paths, fc.cuts))
        results.append(rep)
    return results


def stability_gate(
    values: pd.DataFrame,
    decision: pd.Series | np.ndarray | None,
    profile_name: str,
    cfg: dict[str, Any],
    seed: int,
    n_replicates: int,
    n_jobs: int = 1,
    *,
    features: Iterable[str] | None = None,
    n_fit_patients: int | None = None,
    groups: pd.Series | np.ndarray | None = None,
    reference: CutSet | None = None,
    overrides: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Refit the cuts on ``n_replicates`` cluster bootstrap resamples (identical configuration) and report, per
    reference cut, the recurrence: a cut within the same feature and node path lying within the configured tolerance
    (0.10 x fit IQR, fallback 0.10 x fit SD). A missing node or cut is non-recurrence. Replicate ``r`` draws from
    ``SeedSequence(seed).spawn(n_replicates)[r]``, so the result is identical for any ``n_jobs``.  ``groups`` are the
    row-aligned bootstrap clusters (``cluster_id``; default: every row its own), ordered by first appearance, so
    singleton groups reproduce the row bootstrap draw for draw."""
    gate = cfg["stability_gate"]
    profile = resolve_profile(profile_name, cfg)
    ref = reference or fit_cuts(values, decision, profile_name, cfg, features=features, n_fit_patients=n_fit_patients, overrides=overrides)
    names = list(ref.order)
    n_fit = int(n_fit_patients) if n_fit_patients is not None else len(values)
    x = np.column_stack([pd.to_numeric(values[f], errors="coerce").to_numpy(dtype=float) for f in names]) if names else np.empty((len(values), 0))
    for j, f in enumerate(names):
        if f"{f}__state" in values.columns:
            x[:, j] = np.where(values[f"{f}__state"].astype("string").fillna("").to_numpy() == "OBSERVED", x[:, j], np.nan)
    d = None if decision is None else np.asarray(decision, dtype=object)
    if groups is None:
        group_rows = [np.array([i]) for i in range(len(values))]
    else:
        codes, uniques = pd.factorize(pd.Series(np.asarray(groups, dtype=object)).astype(str))
        group_rows = [np.flatnonzero(codes == k) for k in range(len(uniques))]
    children = np.random.SeedSequence(int(seed)).spawn(int(n_replicates))
    n_chunks = max(1, min(int(n_replicates), max(1, int(n_jobs)) * 4))
    chunks = [c for c in np.array_split(np.arange(int(n_replicates)), n_chunks) if c.size]
    if int(n_jobs) == 1:
        parts = [_bootstrap_chunk(x, d, group_rows, [children[i] for i in c], profile, names, n_fit, overrides) for c in chunks]
    else:
        from joblib import Parallel, delayed

        parts = Parallel(n_jobs=int(n_jobs), backend="loky")(
            delayed(_bootstrap_chunk)(x, d, group_rows, [children[i] for i in c], profile, names, n_fit, overrides) for c in chunks
        )
    replicates = [rep for part in parts for rep in part]
    rows = []
    for f in names:
        fc = ref.features[f]
        tol = _tolerance(fc, gate["match_tolerance"])
        if not fc.cuts:
            rows.append({"profile": profile_name, "feature": f, "node_path": None, "cut": np.nan, "status": fc.status, "tolerance": tol,
                         "n_replicates": int(n_replicates), "n_recurred": 0, "recurrence": np.nan, "recurrence_min": float(gate["recurrence_min"]),
                         "retained": False, "matched_mean": np.nan, "matched_sd": np.nan, "matched_q025": np.nan, "matched_q975": np.nan, "seed": int(seed)})
            continue
        for cut, path in zip(fc.cuts, fc.node_paths):
            matched = []
            for rep in replicates:
                near = [c for p, c in rep[f] if p == path and abs(c - cut) <= tol]
                if near:
                    matched.append(min(near, key=lambda c: (abs(c - cut), c)))
            m = np.asarray(matched, dtype=float)
            recurrence = m.size / float(n_replicates) if n_replicates else np.nan
            rows.append(
                {
                    "profile": profile_name,
                    "feature": f,
                    "node_path": path,
                    "cut": float(cut),
                    "status": fc.status,
                    "tolerance": tol,
                    "n_replicates": int(n_replicates),
                    "n_recurred": int(m.size),
                    "recurrence": recurrence,
                    "recurrence_min": float(gate["recurrence_min"]),
                    "retained": bool(recurrence >= float(gate["recurrence_min"])),
                    "matched_mean": float(m.mean()) if m.size else np.nan,
                    "matched_sd": float(m.std(ddof=1)) if m.size > 1 else np.nan,
                    "matched_q025": float(np.quantile(m, 0.025)) if m.size else np.nan,
                    "matched_q975": float(np.quantile(m, 0.975)) if m.size else np.nan,
                    "seed": int(seed),
                }
            )
    return pd.DataFrame(rows)


_AGREEMENT = re.compile(r"^\s*(\d+)_of_(\d+)\s*$")


def per_object_stability(fold_predictions: list[pd.DataFrame], final_prediction: pd.DataFrame, cutset: CutSet, cfg: dict[str, Any]) -> pd.DataFrame:
    """Per-object symbol stability (configs/wedd.yaml per_object_stability): the object's A vector is mapped by each
    outer-fold transition, decoded with the final cuts, and at least ``k`` of ``m`` valid fold symbols must equal the
    final symbol; otherwise the cell state is UNSTABLE_ESTIMATE (else ESTIMATED)."""
    pcfg = cfg["per_object_stability"]
    m = _AGREEMENT.match(str(pcfg["agreement_required"]))
    if not m:
        raise ValueError("Unrecognized per_object_stability.agreement_required")
    required, n_mappers = int(m.group(1)), int(m.group(2))
    if len(fold_predictions) != n_mappers or n_mappers != int(pcfg["fold_mappers"]):
        raise ValueError(f"Expected {pcfg['fold_mappers']} fold mappers, got {len(fold_predictions)}")
    final_idx = cutset.bin_indices(final_prediction)
    fold_idx = [cutset.bin_indices(p) for p in fold_predictions]
    out = pd.DataFrame(index=final_prediction.index)
    for f in cutset.order:
        agree = np.zeros(len(final_prediction), dtype=int)
        for fi in fold_idx:
            agree += ((fi[f].to_numpy() >= 0) & (fi[f].to_numpy() == final_idx[f].to_numpy())).astype(int)
        stable = agree >= required
        out[f] = np.where(final_idx[f].to_numpy() < 0, "", np.where(stable, "ESTIMATED", str(pcfg["failure_state"])))
    return out
