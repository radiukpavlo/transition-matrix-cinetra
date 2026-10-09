"""B9 orientation audit of the image-branch input frame (exploratory, post hoc; owner issue O1 of 3 October 2026).

The image branches D20 and F9 see every slice after :func:`tmmri.neural.preprocess.canonical_orientation` (an in-plane,
lossless transpose/flip chosen from the staged ED image header).  B9 measures, label-free, whether that frame puts the
heart the same way up in every cohort: for every staged case of the configured partitions (``acdc_dev``, ``acdc_test``,
``mms_external``; ``mms2_external`` only through the M&Ms-2 lock and its passed ``validate-cohort --dataset mms2``
record) the ED reference mask is reoriented exactly like the images (:func:`~tmmri.neural.preprocess.oriented_slices`),
and on the mid-ventricular ED slice (the middle, in stored order, of the ED slices with both LV-cavity and RV-cavity
voxels) the direction from the LV to the RV centroid is measured in millimetres, x along the columns and y up (minus the
row offset).  Masks are read only for these two centroids; no class label, prediction or evaluation output is read
(vendor comes from the cohort table's vendor column alone).

Per dataset, partition, vendor (where the cohort table has one; ``all`` pools them) and orientation code
(:attr:`~tmmri.neural.preprocess.Orientation.code`; ``all`` pools them) it reports n, the median and the 10th and 90th
percentiles of the angle (degrees in (-180, 180]), the fraction with the RV left of the LV (negative column offset) and
the fraction with the RV above it (negative row offset).  Descriptive only: it documents why D20 and F9 are D4-symmetrized
(:mod:`tmmri.neural.dihedral`) and never feeds a bundle, a threshold or a selection.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

OUTPUTS = ("per_case.parquet", "strata.parquet", "strata.csv", "summary.json")
LANE_KEYS = ("neural_input",)               # the geometry settings of canonical_orientation (as the tensor cache reads them)
MEASURED = "MEASURED"
NO_LV_RV_SLICE = "NO_LV_RV_SLICE"
ALL = "all"
NO_VENDOR = "<not available>"
CASE_COLUMNS = ["sample_id", "dataset", "partition", "vendor", "orientation_code", "status", "n_slices_ed", "n_lv_rv_slices", "slice_index",
                "d_col_mm", "d_row_mm", "angle_deg", "rv_left", "rv_above"]
STRATA_COLUMNS = ["dataset", "partition", "vendor", "orientation_code", "n", "angle_median_deg", "angle_q10_deg", "angle_q90_deg",
                  "fraction_rv_left", "fraction_rv_above"]


# ----------------------------------------------------------------------------- pure measurement


def mid_ventricular_slice(mask_slices: np.ndarray, *, lv: int, rv: int) -> tuple[int | None, int]:
    """(index, count) of the ED slices ``[n, rows, cols]`` holding both ``lv`` and ``rv`` voxels: the middle one in
    stored order (``(count - 1) // 2``), or ``None`` when no slice holds both."""

    both = [k for k in range(int(mask_slices.shape[0])) if (mask_slices[k] == lv).any() and (mask_slices[k] == rv).any()]
    return (both[(len(both) - 1) // 2] if both else None), len(both)


def lv_rv_direction(mask_slice: np.ndarray, spacing_rc: Sequence[float], *, lv: int, rv: int) -> dict[str, Any]:
    """LV -> RV centroid offsets (mm) of one canonical slice, the angle (degrees, x = columns, y up, in (-180, 180])
    and the RV-left / RV-above flags."""

    def centroid(label: int) -> np.ndarray:
        return np.argwhere(mask_slice == label).mean(axis=0)

    d = centroid(rv) - centroid(lv)
    d_row, d_col = float(d[0]) * float(spacing_rc[0]), float(d[1]) * float(spacing_rc[1])
    angle = math.degrees(math.atan2(-d_row, d_col))
    if angle <= -180.0:
        angle += 360.0
    return {"d_col_mm": d_col, "d_row_mm": d_row, "angle_deg": angle, "rv_left": bool(d_col < 0), "rv_above": bool(d_row < 0)}


def case_orientation(ed_mask: np.ndarray, geometry: Any, *, lv: int, rv: int) -> dict[str, Any]:
    """The B9 row of one case: ``ed_mask`` in staged voxel order (X, Y, Z), ``geometry`` the
    :class:`~tmmri.neural.preprocess.PhaseGeometry` of the staged ED image."""

    from ...neural.preprocess import canonical_orientation, oriented_slices

    orientation = canonical_orientation(geometry)
    slices = oriented_slices(np.asarray(ed_mask), orientation)
    k, count = mid_ventricular_slice(slices, lv=lv, rv=rv)
    row: dict[str, Any] = {"orientation_code": orientation.code, "n_slices_ed": int(slices.shape[0]), "n_lv_rv_slices": int(count)}
    if k is None:
        return {**row, "status": NO_LV_RV_SLICE, "slice_index": None}
    return {**row, "status": MEASURED, "slice_index": int(k),
            **lv_rv_direction(slices[k], orientation.spacing(geometry.spacing_mm), lv=lv, rv=rv)}


def _staged_case_row(job: Mapping[str, Any]) -> dict[str, Any]:
    """One staged case (worker function): label-mapped ED mask and the ED image geometry of the tensor cache."""

    import nibabel as nib

    from ...ingest.staging import load_staged_case
    from ...neural.preprocess import InputRejected, phase_geometry

    keys = {"sample_id": job["sample_id"], "dataset": job["dataset"], "partition": job["partition"], "vendor": job["vendor"]}
    case = load_staged_case(job["dataset"], job["case_id"], job["staged_root"], label_map=job["label_map"])
    try:
        geometry = phase_geometry(nib.load(str(Path(job["staged_root"]) / job["dataset"] / job["case_id"] / "ed_image.nii")), job["geometry_cfg"])
    except InputRejected as exc:
        return {**keys, "status": exc.reason_code}
    return {**keys, **case_orientation(case.ed_mask, geometry, lv=int(job["lv"]), rv=int(job["rv"]))}


def strata_table(per_case: pd.DataFrame, quantiles: Sequence[float] = (0.10, 0.50, 0.90)) -> pd.DataFrame:
    """Per dataset and partition: every (vendor or ``all``) x (orientation code or ``all``) cell of the measured cases."""

    q_lo, q_mid, q_hi = (float(q) for q in quantiles)
    measured = per_case[per_case["status"] == MEASURED]
    rows = []
    for (ds, part), block in measured.groupby(["dataset", "partition"], sort=True):
        vendors = [ALL] + sorted(set(block["vendor"].astype(str)) - {NO_VENDOR})
        for vendor in vendors:
            vb = block if vendor == ALL else block[block["vendor"].astype(str) == vendor]
            for code in [ALL] + sorted(set(vb["orientation_code"].astype(str))):
                cell = vb if code == ALL else vb[vb["orientation_code"].astype(str) == code]
                angles = cell["angle_deg"].to_numpy(dtype=float)
                lo, mid, hi = (float(v) for v in np.quantile(angles, [q_lo, q_mid, q_hi]))
                rows.append({"dataset": ds, "partition": part, "vendor": vendor, "orientation_code": code, "n": int(len(cell)),
                             "angle_median_deg": mid, "angle_q10_deg": lo, "angle_q90_deg": hi,
                             "fraction_rv_left": float(cell["rv_left"].astype(bool).mean()),
                             "fraction_rv_above": float(cell["rv_above"].astype(bool).mean())})
    return pd.DataFrame(rows, columns=STRATA_COLUMNS)


# ----------------------------------------------------------------------------- stage


def _vendors(ctx: Any, ds: str, column: str | None, ids: Sequence[str]) -> pd.Series:
    """The cohort table's vendor column alone (no class label is read); ``<not available>`` without one."""

    from ...stages import _contracts

    path = _contracts(ctx) / f"cohort_{ds}.parquet"
    if not column or not path.is_file():
        return pd.Series(NO_VENDOR, index=list(ids), dtype=object)
    import pyarrow.parquet as pq

    if column not in pq.read_schema(path).names:
        return pd.Series(NO_VENDOR, index=list(ids), dtype=object)
    vendor = pd.read_parquet(path, columns=["sample_id", column]).set_index("sample_id")[column].reindex(list(ids))
    return vendor.map(lambda v: NO_VENDOR if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v)).astype(object)


def plan(ctx: Any, cfg: Mapping[str, Any], acfg: Mapping[str, Any]) -> dict[str, Any]:
    from ... import stages
    from ...secondary_stages import require_partition_unlocked
    from .common import STAGE, manifest_path

    run = ctx.require_run(STAGE)
    requester = f"{STAGE} B9"
    contracts = stages._contracts(ctx)
    inputs: dict[str, Path] = {"split_manifest": contracts / "split_manifest.parquet"}
    included: dict[str, str] = {}
    skipped: dict[str, str] = {}
    rows: dict[str, pd.DataFrame] = {}
    for part, ds in dict(acfg["partitions"]).items():
        part, ds = str(part), str(ds)
        if ds == "MMS2":
            if run.read_record(stages.MMS2_VALIDATION_STAGE) is None:
                skipped[part] = f"{stages.MMS2_VALIDATION_STAGE} not recorded"
                continue
            require_partition_unlocked(ctx, part, f"{requester} ({part})")          # the M&Ms-2 lock (exit 3 before protocol-v6.0.1)
            inputs["validate_cohort_MMS2_record"] = stages.require_mms2_validated(ctx, requester)
            inputs["split_manifest_MMS2"] = contracts / "split_manifest_MMS2.parquet"
        split = stages.split_manifest_for(ctx, part)
        members = split[split["partition"].astype(str) == part]
        staging = manifest_path(ctx, ds)
        if members.empty:
            skipped[part] = "no rows in the split manifest"
            continue
        if staging is None:
            skipped[part] = f"staging manifest of {ds} not recorded"
            continue
        if ds == "MMS":
            require_partition_unlocked(ctx, part, f"{requester} ({part})")          # the M&Ms-1 lock (protocol-v6.0.0)
        stages._require_staged(ctx, ds, sorted(members["case_id"].astype(str)))
        inputs[f"staging_manifest_{ds}"] = staging
        cohort = contracts / f"cohort_{ds}.parquet"
        if cohort.is_file() and (acfg.get("vendor_columns") or {}).get(ds):
            inputs[f"cohort_{ds}"] = cohort
        included[part] = ds
        rows[part] = members[["sample_id", "case_id"]].astype(str).sort_values("sample_id").reset_index(drop=True)
    return {"inputs": inputs, "included": included, "not_executed": skipped, "rows": rows}


def compute(ctx: Any, cfg: Mapping[str, Any], acfg: Mapping[str, Any], plan: Mapping[str, Any], out) -> dict[str, Any]:
    from ...ingest.sources import cohort_key
    from ...neural.image_stages import neural_input_config
    from ...runs import atomic_write_json, parallel_map
    from ...stages import _write_table

    labels = dict(acfg["labels"])
    geometry_cfg = neural_input_config(ctx)
    jobs = []
    for part, ds in plan["included"].items():
        members = plan["rows"][part]
        vendor = _vendors(ctx, ds, (acfg.get("vendor_columns") or {}).get(ds), members["sample_id"])
        label_map = ctx.study.label_map(cohort_key(ds))
        jobs += [{"sample_id": sid, "case_id": cid, "dataset": ds, "partition": part, "vendor": vendor.get(sid, NO_VENDOR),
                  "staged_root": str(ctx.staged_root), "label_map": label_map, "geometry_cfg": geometry_cfg, "lv": labels["lv"], "rv": labels["rv"]}
                 for sid, cid in zip(members["sample_id"], members["case_id"])]
    per_case = pd.DataFrame(parallel_map(_staged_case_row, jobs, n_jobs=int(ctx.jobs or 1)), columns=CASE_COLUMNS)
    for col in ("d_col_mm", "d_row_mm", "angle_deg"):
        per_case[col] = pd.to_numeric(per_case[col], errors="coerce").astype(float)
    for col in ("n_slices_ed", "n_lv_rv_slices", "slice_index"):
        per_case[col] = pd.to_numeric(per_case[col], errors="coerce").astype("Int64")
    for col in ("rv_left", "rv_above"):
        per_case[col] = per_case[col].astype("boolean")
    per_case = per_case.sort_values(["partition", "sample_id"]).reset_index(drop=True)
    strata = strata_table(per_case, acfg.get("quantiles", (0.10, 0.50, 0.90)))
    _write_table(per_case, out / "per_case.parquet")
    _write_table(strata, out / "strata.parquet", csv=True)
    partitions = {}
    for part in plan["included"]:
        block = per_case[per_case["partition"] == part]
        pooled = strata[(strata["partition"] == part) & (strata["vendor"] == ALL) & (strata["orientation_code"] == ALL)]
        partitions[part] = {"dataset": plan["included"][part], "n_cases": int(len(block)),
                            "status": {str(k): int(v) for k, v in block["status"].value_counts().sort_index().items()},
                            "orientation_codes": {str(k): int(v) for k, v in block["orientation_code"].dropna().value_counts().sort_index().items()},
                            "pooled": pooled.drop(columns=["dataset", "partition", "vendor", "orientation_code"]).to_dict("records")[0] if len(pooled) else None}
    summary = {"analysis": "B9", "included": plan["included"], "not_executed": plan["not_executed"], "partitions": partitions,
               "definitions": {k: acfg[k] for k in ("orientation", "slice", "angle", "rv_left", "rv_above") if k in acfg},
               "note": "descriptive and label-free: reference masks are read only for the LV and RV centroids; no class label is read"}
    atomic_write_json(out / "summary.json", summary, allow_nan=True)
    return {"included": list(plan["included"]), "not_executed": plan["not_executed"],
            "n_measured": int((per_case["status"] == MEASURED).sum()), "n_cases": int(len(per_case))}
