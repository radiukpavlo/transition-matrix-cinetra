"""Shared pieces of the exploratory register B1-B9 (protocol-v6.0.1 amendment item 8; B9 for owner issue O1;
``configs/exploratory.yaml``).

Every analysis is the stage ``run-exploratory:<B#>`` of a v6 run root (once-only guard, content-addressed record) and
writes ``exploratory/<B#>/`` with a ``provenance.json`` labelled ``EXPLORATORY_POST_HOC``: the input hashes, the
configuration digests, the code version and the frozen bundle hash.  Inputs are recorded run artifacts whose hashes
are checked against the record of the stage that wrote them; no analysis re-runs an evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from ...runs import code_version, hash_path
from ...secondary_stages import input_hashes, lane_config, lane_config_digest, require_partition_unlocked

CONFIG_KEY = "exploratory"
LABEL = "EXPLORATORY_POST_HOC"
STAGE = "run-exploratory"
ANALYSES = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9")
DECLARED = "protocol-v6.0.1 amendment item 8: exploratory register B1-B8, declared post hoc (explains results, never confirms them)"
DECLARED_BY_ANALYSIS = {"B9": "protocol-v6.0.1 exploratory register, B9 added on 3 October 2026 for owner issue O1 (image-branch orientation), "
                              "declared post hoc (explains results, never confirms them)"}


def config(study: Any) -> dict[str, Any]:
    cfg = lane_config(study, CONFIG_KEY)
    if str(cfg.get("label")) != LABEL:
        raise ValueError(f"configs/{CONFIG_KEY}.yaml label must be {LABEL}")
    return cfg


def analysis_config(study: Any, analysis: str) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = config(study)
    if analysis not in ANALYSES or analysis not in cfg["analyses"]:
        raise ValueError(f"run-exploratory --analysis must be one of {list(ANALYSES)} (configs/{CONFIG_KEY}.yaml analyses)")
    return cfg, dict(cfg["analyses"][analysis])


def out_dir(run: Any, cfg: Mapping[str, Any], analysis: str) -> Path:
    return run.sub(f"{cfg.get('output_root', 'exploratory')}/{analysis}")


def provenance(ctx: Any, analysis: str, cfg: Mapping[str, Any], inputs: Mapping[str, Path], outputs: Iterable[Path],
               **extra: Any) -> dict[str, Any]:
    """The ``EXPLORATORY_POST_HOC`` provenance record of one analysis (no timestamps; deterministic)."""

    run = ctx.require_run(STAGE)
    code = code_version()
    frozen = None
    if run.frozen_path.is_file():
        import json

        frozen = json.loads(run.frozen_path.read_text(encoding="utf-8")).get("bundle_sha256")
    return {"schema": "tmmri.exploratory_provenance/1", "label": LABEL, "analysis": analysis, "title": cfg["analyses"][analysis]["title"],
            "declared": DECLARED_BY_ANALYSIS.get(analysis, DECLARED), "role": "exploratory_post_hoc", "confirmatory": False, "inference": "effect sizes and descriptive intervals only",
            "reads": "recorded run artifacts only; no evaluation re-run", "run_id": run.run_id, "v6_bundle_sha256": frozen,
            "inputs": input_hashes(run, {k: v for k, v in inputs.items() if Path(v).exists()}),
            "config": {"file": f"configs/{CONFIG_KEY}.yaml", "sha256": lane_config_digest(ctx.study, CONFIG_KEY),
                       "analysis_settings": cfg["analyses"][analysis]},
            "outputs": [run.portable(p) for p in outputs], "code_commit": code.get("commit"), "code_sha256": code.get("code_sha256"), **extra}


def check_recorded(run: Any, stage: str, paths: Iterable[Path], requester: str) -> None:
    """Every path must carry the hash its producing stage recorded (the summaries read recorded artifacts only)."""

    from ...stages import _check_recorded_output

    record = run.read_record(stage)
    for path in paths:
        if Path(path).is_file():
            _check_recorded_output(run, record, Path(path), requester)


def evaluated(run: Any, split: str) -> bool:
    record = run.read_record(f"evaluate:{split}")
    return record is not None and record.status == "EVALUATED"


def external_partitions(ctx: Any, cfg: Mapping[str, Any], requester: str, wanted: Iterable[str] | None = None) -> tuple[dict[str, str], dict[str, str]]:
    """``({split: dataset} included, {split: reason} not executed)``: a partition is included when its
    ``evaluate:<split>`` record exists; its M&Ms lock is then enforced (fail closed, exit 3)."""

    run = ctx.require_run(STAGE)
    parts = dict(cfg["external_partitions"])
    included, skipped = {}, {}
    for split in (list(wanted) if wanted is not None else list(parts)):
        if split not in parts:
            skipped[split] = "not configured in configs/exploratory.yaml external_partitions"
        elif not evaluated(run, split):
            skipped[split] = f"evaluate:{split} not recorded"
        else:
            require_partition_unlocked(ctx, split, f"{requester} ({split})")
            included[split] = str(parts[split]["dataset"])
    return included, skipped


def observed(b: pd.DataFrame, target: str) -> pd.Series:
    """Measured B of ``target`` on its OBSERVED cells (NaN elsewhere)."""

    values = pd.to_numeric(b[target], errors="coerce") if target in b.columns else pd.Series(np.nan, index=b.index)
    state = f"{target}__state"
    if state in b.columns:
        values = values.where(b[state].astype("string").fillna("") == "OBSERVED")
    return values.astype(float)


def normalized_errors(bhat: pd.DataFrame, b: pd.DataFrame, targets: Iterable[str], sd: pd.Series) -> pd.DataFrame:
    """(B-hat - measured B) / acdc_dev SD per target on the rows of ``bhat`` (NaN where either side is missing)."""

    out = {}
    for t in targets:
        pred = pd.to_numeric(bhat[t], errors="coerce") if t in bhat.columns else pd.Series(np.nan, index=bhat.index)
        scale = float(sd.get(t, np.nan))
        out[t] = (pred - observed(b.reindex(bhat.index), t)) / scale if np.isfinite(scale) and scale > 0 else np.nan
    return pd.DataFrame(out, index=bhat.index)


def cluster_bootstrap(values: np.ndarray, clusters: np.ndarray, statistic, *, n_boot: int, seed: int, level: float = 0.95) -> tuple[float, float | None, float | None]:
    """Point estimate (each row once) and percentile interval of ``statistic(values, weights)`` under a cluster
    bootstrap (replicate r seeded by ``SeedSequence(seed).spawn(n_boot)[r]``; a drawn cluster brings all its rows)."""

    from ...metrics import _cluster_codes, _cluster_draw_weights

    values = np.asarray(values)
    codes, n_clusters = _cluster_codes(clusters, len(values))
    estimate = float(statistic(values, np.ones(len(values))))
    if len(values) == 0 or n_boot <= 0:
        return estimate, None, None
    boot = []
    for ss in np.random.SeedSequence(int(seed)).spawn(int(n_boot)):
        w = _cluster_draw_weights(np.random.default_rng(ss), codes, n_clusters)
        boot.append(statistic(values, w))
    arr = np.asarray(boot, dtype=float)
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return estimate, None, None
    alpha = (1.0 - level) / 2.0
    return estimate, float(np.quantile(arr, alpha)), float(np.quantile(arr, 1.0 - alpha))


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & (w > 0)
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else float("nan")


def weighted_sd(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & (w > 0)
    if w[ok].sum() <= 1:
        return float("nan")
    m = np.sum(v[ok] * w[ok]) / np.sum(w[ok])
    return float(np.sqrt(np.sum(w[ok] * (v[ok] - m) ** 2) / (np.sum(w[ok]) - 1.0)))


def mean_with_ci(values: pd.Series, clusters: pd.Series, boot: Mapping[str, Any], min_cases: int) -> dict[str, Any]:
    """Mean of the finite values with a cluster-bootstrap interval; NE interval below ``min_cases``."""

    ok = np.isfinite(values.to_numpy(dtype=float))
    v = values.to_numpy(dtype=float)[ok]
    c = clusters.reindex(values.index).to_numpy(dtype=object)[ok]
    if not v.size:
        return {"n": 0, "n_clusters": 0, "estimate": None, "ci_lower": None, "ci_upper": None, "status": "NE", "reason": "NO_OBSERVED_VALUES"}
    n_clusters = int(pd.Series(c).astype(str).nunique())
    below = v.size < int(min_cases)
    est, lo, hi = cluster_bootstrap(v, c, weighted_mean, n_boot=0 if below else int(boot["replicates"]), seed=int(boot["seed"]),
                                    level=float(boot.get("level", 0.95)))
    return {"n": int(v.size), "n_clusters": n_clusters, "estimate": est, "sd": float(np.std(v, ddof=1)) if v.size > 1 else None,
            "ci_lower": lo, "ci_upper": hi, "status": "NE" if below else "EVALUATED", "reason": "STRATUM_BELOW_MIN_CASES" if below else None}


def manifest_path(ctx: Any, ds: str) -> Path | None:
    """The staging manifest of ``ds`` (``<data-root>/manifests/staging_<DS>.json``) when it exists."""

    from ...ingest.staging import manifest_path_for

    try:
        path = manifest_path_for(ctx.manifests_dir, ds)
    except (KeyError, ValueError):
        path = Path(ctx.manifests_dir) / f"staging_{ds}.json"
    return path if path.is_file() else None


GEOMETRY_COLUMNS = ("inplane_spacing_mm", "slice_thickness_mm", "slice_increment_mm", "num_slices", "matrix_rows", "matrix_cols",
                    "matrix_pixels", "fov_mm")


def geometry_frame(b: pd.DataFrame, manifest: Mapping[str, Any] | None) -> pd.DataFrame:
    """Acquisition geometry per row of ``b`` (index sample_id): in-plane spacing, slice thickness and increment from
    measured B; matrix and slice count from the staging-manifest geometry (``image_shape``, ``num_slices``); FOV."""

    cases = dict((manifest or {}).get("cases") or {})
    rows = []
    for sid in b.index.astype(str):
        r = b.loc[sid]
        dx, dy = pd.to_numeric(r.get("spacing_x_mm"), errors="coerce"), pd.to_numeric(r.get("spacing_y_mm"), errors="coerce")
        geom = (cases.get(sid.split(":", 1)[-1]) or {}).get("geometry") or {}
        shape = list(geom.get("image_shape") or [])
        nx, ny = (float(shape[0]), float(shape[1])) if len(shape) >= 2 else (np.nan, np.nan)
        nz = geom.get("num_slices", shape[2] if len(shape) >= 3 else None)
        rows.append({"sample_id": sid, "inplane_spacing_mm": float(np.nanmean([dx, dy])) if np.isfinite([dx, dy]).any() else np.nan,
                     "slice_thickness_mm": pd.to_numeric(r.get("slice_thickness_mm"), errors="coerce"),
                     "slice_increment_mm": pd.to_numeric(r.get("slice_increment_mm"), errors="coerce"),
                     "num_slices": float(nz) if nz is not None else np.nan, "matrix_rows": nx, "matrix_cols": ny,
                     "matrix_pixels": nx * ny, "fov_mm": float(max(nx * dx, ny * dy)) if np.isfinite([nx, ny, dx, dy]).all() else np.nan})
    return pd.DataFrame(rows).set_index("sample_id").astype(float)


def partition_inputs(ctx: Any, split: str, ds: str) -> dict[str, Path]:
    """Recorded artifacts of an evaluated partition read by B2, B3 and B8 (plus the optional staging manifest)."""

    from ...stages import _contracts

    run = ctx.require_run(STAGE)
    p = run.path
    out = {f"bhat_{split}": p / "reconstructed" / f"bhat_{split}.parquet", f"B_{ds}": p / "semantics" / f"B_{ds}.parquet",
           f"cohort_{ds}": _contracts(ctx) / f"cohort_{ds}.parquet", f"evaluate_{split}_record": run.record_path(f"evaluate:{split}")}
    manifest = manifest_path(ctx, ds)
    if manifest is not None:
        out[f"staging_manifest_{ds}"] = manifest
    return out


def partition_errors(ctx: Any, cfg: Mapping[str, Any], split: str, ds: str, targets: Iterable[str]) -> dict[str, Any]:
    """Normalized errors, vendor, clusters and geometry of the non-critical rows of a recorded evaluation partition."""

    import json

    from ...stages import _cluster_ids, _cohort, _dev_reference, _split_manifest, critical_qc_mask

    run = ctx.require_run(STAGE)
    check_recorded(run, f"evaluate:{split}", [run.path / "reconstructed" / f"bhat_{split}.parquet"], f"{STAGE} ({split})")
    bhat = pd.read_parquet(run.path / "reconstructed" / f"bhat_{split}.parquet").set_index("sample_id")
    b = pd.read_parquet(run.path / "semantics" / f"B_{ds}.parquet").set_index("sample_id").reindex(bhat.index)
    critical = critical_qc_mask(b, ctx.study).to_numpy(dtype=bool)
    ids = [str(s) for s, c in zip(bhat.index, critical) if not c]
    sd, _ = _dev_reference(ctx)
    errors = normalized_errors(bhat.loc[ids], b.loc[ids], list(targets), sd)
    vendor_col = str((cfg.get("external_partitions") or {}).get(split, {}).get("vendor_column", "vendor"))
    cohort = _cohort(ctx, ds).set_index("sample_id").reindex(ids)
    vendor = cohort[vendor_col].map(lambda v: "<missing>" if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v)) \
        if vendor_col in cohort.columns else pd.Series("<missing>", index=ids)
    manifest = manifest_path(ctx, ds)
    geometry = geometry_frame(b.loc[ids], json.loads(manifest.read_text(encoding="utf-8")) if manifest is not None else None)
    return {"ids": ids, "errors": errors, "vendor": vendor, "clusters": _cluster_ids(_split_manifest(ctx), ids), "geometry": geometry,
            "cohort": cohort, "n_excluded_critical_qc": int(critical.sum()), "sd": sd}


def describe(values: pd.Series) -> dict[str, Any]:
    v = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if not v.size:
        return {"n": 0, "median": None, "q25": None, "q75": None, "min": None, "max": None, "mean": None}
    return {"n": int(v.size), "median": float(np.median(v)), "q25": float(np.quantile(v, 0.25)), "q75": float(np.quantile(v, 0.75)),
            "min": float(v.min()), "max": float(v.max()), "mean": float(v.mean())}


def file_hashes(paths: Iterable[Path]) -> dict[str, str | None]:
    return {Path(p).name: hash_path(p) for p in paths}
