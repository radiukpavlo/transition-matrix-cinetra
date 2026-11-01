"""B12: geometry-aware reconstruction and error decomposition (EPC-1; ``configs/exploratory_postfreeze.yaml``
analyses.B12).

(a) Analytical physical volumes V_k,phase = f_k,phase * N_phase * dx * dy * dz / 1000 mL from the voxel fractions of
formal A, with N the voxel count of the phase array (staging-manifest ``image_shape``; the manifest records one geometry
per case, read from the ED mask, so N_ES = N_ED), dx and dy the in-plane spacing and dz the slice increment as in B
(never the slice thickness); LV mass = V_myo,ED * 1.05 g/mL; the analytical EF of B10.  The largest absolute
difference from B is reported per partition and target.
(b) ``ridge_typed_formal57_plus_geometry3``: the 57 descriptors plus log voxel volume, log imaged volume and log slice
count, fitted exactly as B11.
Decomposition per primary target in normalized MAE (divided by s_j): approximation (geometry-aware ridge on the volumes,
primary ridge on EF; internal partitions, with the analytical error as reference), missing information (primary ridge
minus geometry-aware ridge, paired, every partition) and shift (geometry-aware ridge on each external partition minus on
acdc_test, independent cluster bootstraps).  The staging manifests are read from the staged data root and verified
against the hash recorded as an input of ``build-semantics:<DS>``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .b10_analytical_ef import analytical_ef
from .common import (
    EVALUATED,
    PostfreezeContext,
    clean,
    formal_a,
    included_partitions,
    write_json,
    write_table,
)
from .estimates import interval_from_draws, nmae_draws
from .transition_variants import (
    PRIMARY,
    dev_design,
    difference_rows,
    evaluate_variants,
    fit_and_freeze,
)

ANALYSIS = "B12"
ESTIMATOR = "ridge_typed_formal57_plus_geometry3"
MYOCARDIAL_DENSITY_G_PER_ML = 1.05
VOLUMES = {"lv_edv_ml": ("fd_l1_fraction_ed", 1.0), "lv_esv_ml": ("fd_l1_fraction_es", 1.0), "rv_edv_ml": ("fd_l3_fraction_ed", 1.0),
           "rv_esv_ml": ("fd_l3_fraction_es", 1.0), "lv_myocardial_mass_ed_g": ("fd_l2_fraction_ed", MYOCARDIAL_DENSITY_G_PER_ML)}
COVARIATES = ("log_voxel_volume_ml", "log_imaged_volume_ml", "log_slice_count")
INTERNAL = ("acdc_dev_oof", "acdc_test")


def staging_geometry(pctx: PostfreezeContext, ds: str) -> dict[str, dict[str, Any]]:
    """``{case_id: geometry}`` of the staging manifest of ``ds`` (hash-checked against ``build-semantics:<DS>``)."""

    import json

    from ...ingest.staging import manifest_path_for

    try:
        path = manifest_path_for(pctx.stage_ctx.manifests_dir, ds)
    except (KeyError, ValueError):
        path = pctx.stage_ctx.manifests_dir / f"staging_{ds}.json"
    manifest = json.loads(pctx.verified(path, f"build-semantics:{ds}").read_text(encoding="utf-8"))
    return {str(cid): dict(rec.get("geometry") or {}) for cid, rec in (manifest.get("cases") or {}).items()}


def geometry(b: pd.DataFrame, cases: dict[str, dict[str, Any]], ids: list[str]) -> pd.DataFrame:
    """dx, dy, dz (B), N (voxels of the ED array), slice count and the three log covariates per row."""

    rows = []
    for sid in ids:
        geom = cases.get(sid.split(":", 1)[-1]) or {}
        shape = [int(v) for v in (geom.get("image_shape") or [])]
        n_vox = float(np.prod(shape[:3])) if len(shape) >= 3 else np.nan
        n_slices = float(shape[2]) if len(shape) >= 3 else float(geom["num_slices"]) if geom.get("num_slices") is not None else np.nan
        dx, dy, dz = (pd.to_numeric(b.at[sid, c], errors="coerce") if c in b.columns else np.nan
                      for c in ("spacing_x_mm", "spacing_y_mm", "slice_increment_mm"))
        voxel_ml = dx * dy * dz / 1000.0
        rows.append({"sample_id": sid, "dx_mm": dx, "dy_mm": dy, "dz_mm": dz, "n_voxels": n_vox, "n_slices": n_slices, "voxel_volume_ml": voxel_ml})
    frame = pd.DataFrame(rows).set_index("sample_id").astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["log_voxel_volume_ml"] = np.log(frame["voxel_volume_ml"].where(frame["voxel_volume_ml"] > 0))
        frame["log_imaged_volume_ml"] = np.log((frame["n_voxels"] * frame["voxel_volume_ml"]).where(frame["n_voxels"] * frame["voxel_volume_ml"] > 0))
        frame["log_slice_count"] = np.log(frame["n_slices"].where(frame["n_slices"] > 0))
    return frame


def analytical_volumes(a: pd.DataFrame, geo: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for target, (column, density) in VOLUMES.items():
        f = pd.to_numeric(a[column], errors="coerce").reindex(geo.index)
        out[target] = f * geo["n_voxels"] * geo["voxel_volume_ml"] * density
    return pd.DataFrame(out, index=geo.index)


def run(pctx: Any) -> dict[str, Any]:
    from ...formal_features import formal_descriptor_names

    acfg = pctx.analysis_cfg(ANALYSIS)
    out = pctx.out_dir(ANALYSIS)
    if tuple(acfg["geometry_covariates"]) != COVARIATES or acfg["learned_variant"] != ESTIMATOR:
        raise ValueError(f"{ANALYSIS}: configs/exploratory_postfreeze.yaml geometry_covariates must be {list(COVARIATES)} and learned_variant {ESTIMATOR}")
    targets = [str(t) for t in pctx.lane_cfg["statistics"]["primary_targets"]]
    volume_targets = [str(t) for t in pctx.lane_cfg["statistics"]["volume_targets"]]
    names = formal_descriptor_names()
    splits, skipped = included_partitions(pctx, ["acdc_dev", *EVALUATED], f"run-postfreeze {ANALYSIS}")
    cases: dict[str, dict[str, dict[str, Any]]] = {}

    def geo_for(ds: str, b: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
        if ds not in cases:
            cases[ds] = staging_geometry(pctx, ds)
        return geometry(b, cases[ds], ids)

    # (b) fit and freeze the geometry-aware ridge on acdc_dev before any evaluated partition is processed.
    ids, a, b, folds, inner_fn = dev_design(pctx)
    dev_ids = ids["sample_id"].astype(str).tolist()
    geo_dev = geo_for("ACDC", b.set_index("sample_id"), dev_ids)
    x_dev = np.column_stack([a, geo_dev.loc[dev_ids, list(COVARIATES)].to_numpy(dtype=float)])
    entry = {**fit_and_freeze(pctx, out, ESTIMATOR, x_dev, b, ids, folds, inner_fn, [*names, *COVARIATES])}
    a_cache: dict[str, pd.DataFrame] = {}
    b_cache: dict[str, pd.DataFrame] = {}

    def frames(ds: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        if ds not in a_cache:
            a_cache[ds] = formal_a(pctx, ds)
            b_cache[ds] = pctx.read_parquet(f"semantics/B_{ds}.parquet").set_index("sample_id")
        return a_cache[ds], b_cache[ds]

    def design(split: str, ds: str, ids_p: list[str], _entry: dict[str, Any]) -> np.ndarray:
        a_p, b_p = frames(ds)
        geo = geo_for(ds, b_p, ids_p)
        return np.column_stack([a_p.reindex(ids_p)[names].to_numpy(dtype=float), geo[list(COVARIATES)].to_numpy(dtype=float)])

    exact_rows: list[dict[str, Any]] = []

    def analytical_arm(split: str, part: Any) -> dict[str, pd.DataFrame]:
        a_p, _ = frames(part.dataset)
        a_rows = a_p.reindex(part.ids)
        geo = geo_for(part.dataset, part.b, part.ids)
        vol = analytical_volumes(a_rows, geo)
        ef, _ = analytical_ef(a_rows, part.b)
        label = "acdc_dev_oof" if split == "acdc_dev" else split
        for target in volume_targets:
            b_raw = pd.to_numeric(part.b[target], errors="coerce") if target in part.b.columns else pd.Series(np.nan, index=part.ids)
            both = vol[target].notna() & b_raw.notna()
            diff = (vol[target] - b_raw).abs()
            exact_rows.append({"partition": label, "target": target, "n_rows": len(part.ids), "n_both_finite": int(both.sum()),
                               "n_missing_geometry": int(geo["n_voxels"].isna().sum()), "max_abs_difference": float(diff[both].max()) if both.any() else None,
                               "n_abs_difference_gt_1e-6": int((both & (diff > 1e-6)).sum())})
        return {"analytical": pd.concat([vol, ef], axis=1)}

    ev = evaluate_variants(pctx, splits, {ESTIMATOR: entry}, design, extra_arms=analytical_arm, reference=ESTIMATOR)
    results = ev["results"]
    # Decomposition in normalized MAE.
    rows = []
    for label, result in results.items():
        geo_arm, prim_arm, ana_arm = result["arms"][ESTIMATOR], result["arms"][PRIMARY], result["arms"]["analytical"]
        missing = result["differences"][PRIMARY]["nmae"]
        for t in targets:
            if label in INTERNAL:
                source = (ESTIMATOR, geo_arm) if t in volume_targets else (PRIMARY, prim_arm)
                rows.append({"target": t, "component": "approximation", "partition": label, "definition": f"nMAE of {source[0]}",
                             **{k: source[1]["nmae"][t].get(k) for k in ("estimate", "ci_lower", "ci_upper", "status", "reason")}})
                rows.append({"target": t, "component": "analytical_reference", "partition": label, "definition": "nMAE of the analytical estimate",
                             **{k: ana_arm["nmae"][t].get(k) for k in ("estimate", "ci_lower", "ci_upper", "status", "reason")}})
            rows.append({"target": t, "component": "missing_information", "partition": label, "definition": f"nMAE {PRIMARY} minus nMAE {ESTIMATOR} (paired)",
                         **{k: missing[t].get(k) for k in ("estimate", "ci_lower", "ci_upper", "status", "reason")}})
    shift_reference = "acdc_test"
    if shift_reference in ev["predictions"]:
        draws = {}
        for label, (part, y, preds) in ev["predictions"].items():
            if label in (shift_reference, "mms_external", "mms2_external"):
                draws[label] = nmae_draws(y, preds[ESTIMATOR], ev["sd"], targets, units=part.units, clusters=part.clusters, boot=pctx.boot,
                                         stream=EVALUATED.index(label))
        level = float(pctx.boot.get("level", 0.95))
        for label in ("mms_external", "mms2_external"):
            if label not in draws:
                continue
            (p_ext, d_ext), (p_ref, d_ref) = draws[label], draws[shift_reference]
            for j, t in enumerate(targets):
                r = interval_from_draws(p_ext[j] - p_ref[j], d_ext[:, j] - d_ref[:, j], level)
                rows.append({"target": t, "component": "shift", "partition": label,
                             "definition": f"nMAE {ESTIMATOR} on {label} minus on {shift_reference} (independent cluster bootstraps)",
                             **{k: r.get(k) for k in ("estimate", "ci_lower", "ci_upper", "status", "reason")}})
    paths = {"exact": out / "analytical_exactness.parquet", "rho": out / "rho.parquet", "per_target": out / "per_target_mae_bias_loa.parquet",
             "decomposition": out / "decomposition_table.parquet", "diff": out / "paired_differences.parquet", "summary": out / "summary.json"}
    write_table(pd.DataFrame(exact_rows), paths["exact"], pctx)
    write_table(ev["rho"], paths["rho"], pctx)
    write_table(ev["per_target"], paths["per_target"], pctx)
    write_table(pd.DataFrame(rows), paths["decomposition"], pctx)
    write_table(difference_rows(results), paths["diff"], pctx)
    labels = list(results)
    summary = clean({"analysis": ANALYSIS, "analytical_volumes": acfg["analytical_volumes"], "myocardial_density_g_per_ml": MYOCARDIAL_DENSITY_G_PER_ML,
                     "n_phase_source": "staging-manifest image_shape of the ED mask (one geometry per case; N_ES = N_ED)",
                     "geometry_covariates": dict(acfg["geometry_covariates"]), "frozen": entry["frozen"], "included": labels, "not_executed": skipped,
                     "exactness": exact_rows, "decomposition": acfg["decomposition"], "results": results})
    write_json(summary, paths["summary"], pctx)
    pctx.write_provenance(ANALYSIS, dict(pctx.verified_log), [*entry["outputs"], *paths.values()],
                          {"included": labels, "not_executed": skipped, "estimator": ESTIMATOR, "frozen_sha256": entry["frozen"]["files"],
                           "max_abs_difference": {f"{r['partition']}:{r['target']}": r["max_abs_difference"] for r in exact_rows}})
    return {"included": labels, "not_executed": skipped, "estimator": ESTIMATOR}
