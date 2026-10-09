"""B7 segmentation sensitivity on acdc_test (exploratory, post hoc; after the recorded acdc_test evaluation).

Each staged acdc_test case is perturbed (1-pixel in-plane erosion or dilation of one structure, or removal of the basal
slice) and B and A are recomputed through the existing code (:func:`tmmri.semantics.features.case_semantics_v6` with the
frozen QC outlier reference, :func:`tmmri.formal_features.formal_descriptors_from_arrays`); the frozen bundle (transition,
fold mappers, predictor, rulebooks and cuts) is applied unchanged.  Against the unperturbed recomputation it reports the
change of B-hat (mean |delta| / acdc_dev SD per target), the change of rho, the ``d_model`` flip rate and the route and
label changes of every rulebook (Wilson intervals).  Nothing is refitted and no recorded output is rewritten.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

OUTPUTS = ("per_case.parquet", "per_target.parquet", "per_target.csv", "summary.json")
LABELS = {"lv": 1, "myo": 2, "rv": 3}
PERTURBATIONS = ("erode_lv", "dilate_lv", "erode_myo", "dilate_myo", "erode_rv", "dilate_rv", "drop_basal_slice")


# ----------------------------------------------------------------------------- mask perturbations


def _neighbours(slice2d: np.ndarray) -> list[np.ndarray]:
    """4-connected neighbour labels of every pixel (outside the image: background 0)."""

    padded = np.pad(slice2d, 1, mode="constant", constant_values=0)
    return [padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:]]


def erode(mask: np.ndarray, label: int) -> np.ndarray:
    """1-pixel in-plane erosion of ``label`` per slice: a boundary pixel takes the most frequent other label among its
    4 neighbours (ties: the smallest label)."""

    out = np.array(mask, copy=True)
    for z in range(mask.shape[2]):
        s = mask[..., z]
        nb = _neighbours(s)
        inside = s == label
        boundary = inside & np.any([n != label for n in nb], axis=0)
        if not boundary.any():
            continue
        others = sorted({int(v) for v in np.unique(np.stack(nb)) if int(v) != label} | {0})
        counts = np.stack([sum((n == o).astype(int) for n in nb) for o in others])
        choice = np.asarray(others)[np.argmax(counts, axis=0)]
        out[..., z] = np.where(boundary, choice, s)
    return out


def dilate(mask: np.ndarray, label: int) -> np.ndarray:
    """1-pixel in-plane dilation of ``label`` per slice (4-connected; overwrites the neighbouring pixels of other labels)."""

    out = np.array(mask, copy=True)
    for z in range(mask.shape[2]):
        s = mask[..., z]
        grow = (s != label) & np.any([n == label for n in _neighbours(s)], axis=0)
        out[..., z] = np.where(grow, label, s)
    return out


def basal_slice(ed_mask: np.ndarray) -> int | None:
    """Of the two outermost slices with any label at ED, the one with the larger ED LV area (ties: the lower index)."""

    labelled = [z for z in range(ed_mask.shape[2]) if np.any(ed_mask[..., z] > 0)]
    if not labelled:
        return None
    ends = sorted({labelled[0], labelled[-1]})
    areas = [int(np.sum(ed_mask[..., z] == LABELS["lv"])) for z in ends]
    return ends[int(np.argmax(areas))]


def perturb_masks(ed: np.ndarray, es: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    if name == "none":
        return np.array(ed, copy=True), np.array(es, copy=True)
    if name == "drop_basal_slice":
        z = basal_slice(ed)
        ed2, es2 = np.array(ed, copy=True), np.array(es, copy=True)
        if z is not None:
            ed2[..., z] = 0
            es2[..., z] = 0
        return ed2, es2
    op, _, structure = name.partition("_")
    if op not in {"erode", "dilate"} or structure not in LABELS:
        raise ValueError(f"unknown perturbation {name!r}; expected one of {PERTURBATIONS}")
    fn = erode if op == "erode" else dilate
    return fn(ed, LABELS[structure]).astype(ed.dtype), fn(es, LABELS[structure]).astype(es.dtype)


def perturbed_case(case: Any, name: str) -> Any:
    ed, es = perturb_masks(case.ed_mask, case.es_mask, name)
    moved = copy.copy(case)
    moved.ed_mask, moved.es_mask = ed, es
    return moved


# ----------------------------------------------------------------------------- recompute and apply the frozen bundle


def recompute(cases: Iterable[Any], name: str, *, semantics_fn: Callable[[Any], dict], formal_fn: Callable[..., np.ndarray],
              finalize_b: Callable[[pd.DataFrame], pd.DataFrame], descriptor_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(B, A) of every case under perturbation ``name`` (index sample_id)."""

    rows, a_rows = [], []
    for case in cases:
        moved = perturbed_case(case, name)
        rows.append(semantics_fn(moved))
        a_rows.append(np.asarray(formal_fn(moved.ed_image, moved.es_image, moved.ed_mask, moved.es_mask), dtype=float))
    ids = [str(c.sample_id) for c in cases]
    b = finalize_b(pd.DataFrame(rows)).set_index("sample_id").reindex(ids)
    return b, pd.DataFrame(np.vstack(a_rows) if a_rows else np.empty((0, len(descriptor_names))), index=ids, columns=descriptor_names)


def apply_bundle(b: pd.DataFrame, a: pd.DataFrame, bundle: Mapping[str, Any], study: Any) -> dict[str, Any]:
    """B-hat, decoded substrate, ``d_model`` and routes of every rulebook of the frozen bundle (nothing fitted)."""

    from ...rule_pipeline import decode_reconstructions, qc_route_frame, route_records, routing_substrate
    from ...stages import predict_labels

    x = a.to_numpy(dtype=float)
    index = pd.Index(a.index, name="sample_id")
    bhat = bundle["model"].predict(x)
    bhat.index = index
    decoded, states, fold_decoded = decode_reconstructions(bundle["model"], bundle["fold_models"], x, index, study.registry, study.transition)
    d_model = pd.Series(predict_labels(bundle["predictor"], x), index=index, dtype=object)
    qc = qc_route_frame(b, np.isfinite(x).all(axis=1), study)
    routes = {}
    for rb, spec in bundle["rulebooks"].items():
        if spec["substrate"] == "measured_B":
            substrate = b
        else:
            substrate, _ = routing_substrate(decoded, states, fold_decoded, spec["cuts"], study.wedd, study.rule_candidates())
        routes[rb] = route_records(spec["records"], rb, spec["cuts"], substrate, qc, study).set_index("sample_id")
    return {"bhat": bhat, "d_model": d_model, "routes": routes}


def compare(base_b: pd.DataFrame, base: Mapping[str, Any], pert_b: pd.DataFrame, pert: Mapping[str, Any], *, targets: list[str], sd: pd.Series,
            rho_fn: Callable[[pd.DataFrame, pd.DataFrame], dict[str, Any]]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """(summary, per-target deltas, per-case table) of one perturbation against the unperturbed recomputation."""

    from ...comparators.cart_f8 import proportion

    ids = list(base_b.index)
    per_target = []
    for t in targets:
        scale = float(sd.get(t, np.nan))
        d_hat = (pd.to_numeric(pert["bhat"][t], errors="coerce") - pd.to_numeric(base["bhat"][t], errors="coerce")).abs() / scale
        d_b = (pd.to_numeric(pert_b[t], errors="coerce") - pd.to_numeric(base_b[t], errors="coerce")).abs() / scale
        per_target.append({"target": t, "n_bhat": int(d_hat.notna().sum()), "mean_abs_delta_bhat_sd": float(d_hat.mean()) if d_hat.notna().any() else np.nan,
                           "max_abs_delta_bhat_sd": float(d_hat.max()) if d_hat.notna().any() else np.nan, "n_b": int(d_b.notna().sum()),
                           "mean_abs_delta_b_sd": float(d_b.mean()) if d_b.notna().any() else np.nan})
    per_target = pd.DataFrame(per_target)
    rho_base, rho_pert = rho_fn(base_b[targets], base["bhat"]), rho_fn(pert_b[targets], pert["bhat"])
    dm0, dm1 = base["d_model"].reindex(ids), pert["d_model"].reindex(ids)
    both = dm0.notna() & dm1.notna()
    flips = both & (dm0.astype(str) != dm1.astype(str))
    case = pd.DataFrame({"sample_id": ids, "d_model_base": dm0.to_numpy(), "d_model_perturbed": dm1.to_numpy(), "d_model_flip": flips.to_numpy()})
    routes = {}
    for rb in base["routes"]:
        r0, r1 = base["routes"][rb].reindex(ids), pert["routes"][rb].reindex(ids)
        route_change = (r0["route"].astype(str) != r1["route"].astype(str))
        label_change = (r0["label"].astype(object).fillna("<none>").astype(str) != r1["label"].astype(object).fillna("<none>").astype(str))
        routes[rb] = {"route_change": proportion(int(route_change.sum()), len(ids)), "label_change": proportion(int(label_change.sum()), len(ids)),
                      "accepted_base": int((r0["route"] == "EXACT_ACCEPT").sum()), "accepted_perturbed": int((r1["route"] == "EXACT_ACCEPT").sum())}
        case[f"route_base_{rb}"] = r0["route"].to_numpy()
        case[f"route_perturbed_{rb}"] = r1["route"].to_numpy()
    primary = [t for t in targets if t in set(per_target["target"])]
    summary = {"n_cases": len(ids), "d_model_flip": proportion(int(flips.sum()), int(both.sum())),
               "rho_base": rho_base.get("rho"), "rho_perturbed": rho_pert.get("rho"),
               "delta_rho": (rho_pert["rho"] - rho_base["rho"]) if rho_base.get("rho") is not None and rho_pert.get("rho") is not None else None,
               "rho_perturbed_record": rho_pert, "mean_abs_delta_bhat_sd": float(per_target.loc[per_target["target"].isin(primary), "mean_abs_delta_bhat_sd"].mean()),
               "routes": routes}
    return summary, per_target, case


# ----------------------------------------------------------------------------- stage


def plan(ctx: Any, cfg: Mapping[str, Any], acfg: Mapping[str, Any]) -> dict[str, Any]:
    from ...stages import BundleError, _contracts, _rulebooks, _staging_manifest_path, fold_model_paths
    from .common import evaluated

    run = ctx.require_run("run-exploratory")
    if not evaluated(run, str(acfg["partition"])):
        raise BundleError(f"run-exploratory B7 runs after evaluate:{acfg['partition']} is recorded")
    p = run.path
    profile = str(ctx.study.wedd["primary_profile"])
    inputs = {"B_ACDC": p / "semantics" / "B_ACDC.parquet", "A_ACDC": p / "formal" / "A_ACDC.parquet", "cohort_ACDC": _contracts(ctx) / "cohort_ACDC.parquet",
              "split_manifest": _contracts(ctx) / "split_manifest.parquet", "outlier_reference": p / "qc" / "outlier_reference.json",
              "transition_model": p / "transitions" / "transition_model.json", "predictor": p / "formal" / "d_model_predictor.joblib",
              "staging_manifest_ACDC": _staging_manifest_path(ctx, "ACDC"), "evaluate_record": run.record_path(f"evaluate:{acfg['partition']}")}
    for i, path in enumerate(fold_model_paths(ctx, run), start=1):
        inputs[f"transition_fold{i}"] = path
    for rb in _rulebooks(ctx):
        inputs[f"thresholds_{rb}"] = p / "thresholds" / f"{rb}_{profile}.json"
        inputs[f"rulebook_or_empty_{rb}"] = p / "rules" / rb / "rulebook_or_empty.json"
    return {"inputs": inputs, "included": {str(acfg["partition"]): "ACDC"}, "not_executed": {}}


def load_bundle(ctx: Any) -> dict[str, Any]:
    import json

    import joblib

    from ...stages import _rulebooks, load_fold_models, methods_api
    from ...wedd import CutSet

    run = ctx.require_run("run-exploratory")
    profile = str(ctx.study.wedd["primary_profile"])
    model = methods_api("tmmri.transition_fit", "TransitionModel").from_dict(
        json.loads((run.path / "transitions" / "transition_model.json").read_text(encoding="utf-8")))
    books = {}
    for rb, spec in _rulebooks(ctx).items():
        path = run.path / "rules" / rb / "rulebook.json"
        books[rb] = {"substrate": spec["substrate"], "records": json.loads(path.read_text(encoding="utf-8"))["records"] if path.is_file() else [],
                     "cuts": CutSet.from_dict(json.loads((run.path / "thresholds" / f"{rb}_{profile}.json").read_text(encoding="utf-8")))}
    return {"model": model, "fold_models": load_fold_models(ctx, run), "predictor": joblib.load(run.path / "formal" / "d_model_predictor.joblib"),
            "rulebooks": books}


def compute(ctx: Any, cfg: Mapping[str, Any], acfg: Mapping[str, Any], plan: Mapping[str, Any], out) -> dict[str, Any]:
    import functools
    import json

    from ...formal_features import formal_descriptor_names, formal_descriptors_from_arrays
    from ...ingest.sources import cohort_key
    from ...metrics import primary_rho
    from ...runs import atomic_write_json
    from ...semantics.features import case_semantics_v6
    from ...semantics.qc import apply_outlier_reference
    from ...stages import _cluster_ids, _cohort, _dev_reference, _rho_kwargs, _split_manifest, _write_table, load_case_for_semantics, release_metadata

    run = ctx.require_run("run-exploratory")
    study = ctx.study
    partition = str(acfg["partition"])
    split = _split_manifest(ctx)
    recorded_b = pd.read_parquet(run.path / "semantics" / "B_ACDC.parquet").set_index("sample_id")
    ids = [s for s in sorted(split.loc[split["partition"] == partition, "sample_id"].astype(str)) if s in recorded_b.index]
    cohort = _cohort(ctx, "ACDC").set_index("case_id")
    rows = split.set_index("sample_id")
    case_ids = [s.split(":", 1)[1] for s in ids]
    metadata = {cid: release_metadata("ACDC", {**cohort.loc[cid].to_dict(), "case_id": cid}, rows.loc[f"ACDC:{cid}", "partition"]) for cid in case_ids}
    loader = functools.partial(load_case_for_semantics, dataset="ACDC", staged_root=str(ctx.staged_root), label_map=study.label_map(cohort_key("ACDC")),
                               metadata=metadata)
    cases = [loader(cid) for cid in case_ids]
    reference = json.loads((run.path / "qc" / "outlier_reference.json").read_text(encoding="utf-8"))
    names = formal_descriptor_names()
    sd, mean = _dev_reference(ctx)
    targets = study.transition_targets()
    family = study.primary_family()
    clusters = _cluster_ids(split, ids)
    kw = _rho_kwargs(ctx)
    kw["n_boot"] = int(acfg.get("rho_bootstrap_replicates", kw["n_boot"]))

    def semantics_fn(case: Any) -> dict:
        return case_semantics_v6(case, study.registry, cohorts=study.cohorts, reason_codes=study.reason_codes)

    def finalize_b(frame: pd.DataFrame) -> pd.DataFrame:
        return apply_outlier_reference(frame, reference, study.cohorts, reason_codes=study.reason_codes)

    def rho_fn(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict[str, Any]:
        y_null = pd.DataFrame([mean[family]] * len(y_true), index=y_true.index)
        cluster = pd.Series([str(s).split(":", 1)[1] for s in y_true.index], index=y_true.index)
        return primary_rho(y_true[family], y_pred[family], y_null, sd, cluster=cluster,
                           resample_cluster=clusters.reindex(y_true.index).to_numpy(dtype=object), **kw)

    bundle = load_bundle(ctx)
    base_b, base_a = recompute(cases, "none", semantics_fn=semantics_fn, formal_fn=formal_descriptors_from_arrays, finalize_b=finalize_b, descriptor_names=names)
    recorded_a = pd.read_parquet(run.path / "formal" / "A_ACDC.parquet").set_index("sample_id").reindex(ids)[names]
    consistency = float(np.nanmax(np.abs(base_a.to_numpy(dtype=float) - recorded_a.to_numpy(dtype=float)))) if len(ids) else None
    base = apply_bundle(base_b, base_a, bundle, study)
    summaries, per_target_rows, per_case_rows = {}, [], []
    for name in acfg["perturbations"]:
        pert_b, pert_a = recompute(cases, str(name), semantics_fn=semantics_fn, formal_fn=formal_descriptors_from_arrays, finalize_b=finalize_b,
                                   descriptor_names=names)
        pert = apply_bundle(pert_b, pert_a, bundle, study)
        summary, per_target, per_case = compare(base_b, base, pert_b, pert, targets=targets, sd=sd, rho_fn=rho_fn)
        summaries[str(name)] = summary
        per_target_rows.append(per_target.assign(perturbation=str(name)))
        per_case_rows.append(per_case.assign(perturbation=str(name)))
    per_case = pd.concat(per_case_rows, ignore_index=True) if per_case_rows else pd.DataFrame()
    _write_table(per_case.astype({c: object for c in per_case.columns if c.startswith(("route_", "d_model_base", "d_model_perturbed"))}), out / "per_case.parquet")
    _write_table(pd.concat(per_target_rows, ignore_index=True) if per_target_rows else pd.DataFrame(), out / "per_target.parquet", csv=True)
    atomic_write_json(out / "summary.json", {
        "analysis": "B7", "partition": partition, "n_cases": len(ids), "perturbations": summaries, "morphology": acfg["morphology"],
        "basal_slice": acfg["basal_slice"], "rho_base": rho_fn(base_b[targets], base["bhat"]),
        "baseline_recompute_max_abs_diff_formal_A_vs_recorded": consistency,
        "note": "frozen bundle applied to recomputed A and B; nothing refitted; per-case table holds identifiers and stays in the run root"}, allow_nan=True)
    return {"perturbations": list(summaries), "cases": len(ids)}
