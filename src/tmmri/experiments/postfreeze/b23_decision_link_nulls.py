"""B23: null references for the decision-link test (EPC-2, readiness review R2; ``configs/exploratory_postfreeze_v603.yaml``
analyses.B23; reading fixed before computation by Decision D43).

The model of B18 (:func:`tmmri.classification.fit_formal_predictor` with the audited-predictor settings of
``configs/formal.yaml``: multinomial L2 logistic regression, C = 1.0, no search, inputs standardized on its fit rows) is
fitted on acdc_dev to the final ``d_model`` decisions (``formal/d_model_ACDC_dev.parquet``) from six inputs:

* bhat7 - the seven primary targets of the nested out-of-fold B-hat under the frozen typed transforms, refitted exactly
  as B18 (:func:`b18_decision_link.design`, the same fit rows and outer-fold refits).  Its acdc_dev out-of-fold fidelity
  and its fidelity on every evaluated partition must equal the recorded B18 values (``postfreeze/B18/
  decision_link_table.parquet`` of the second source run) to 1e-12 with the same numerator and denominator; otherwise
  the analysis refuses (BundleError) before any output table is written;
* pca7 - the first seven principal components of the 57 formal descriptors standardized on acdc_dev (mean and SD of
  the finite development rows; a zero SD becomes 1), fitted on acdc_dev;
* random7 - the standardized descriptors times each of the 1,000 standard-normal 57 x 7 matrices drawn in order from
  ``numpy.random.default_rng(20261028)`` (lane ``statistics.random_projections``);
* permuted_bhat7 - bhat7 with its development fit rows permuted against ``d_model`` (1,000 permutations of the fit rows
  drawn in order from ``default_rng(20261029)``, lane ``statistics.permutations``); each model is applied to the
  unpermuted partitions;
* constant - the majority development ``d_model`` (ties by name) for every case;
* formal57 - the 57 standardized descriptors (upper reference).

Every fitted object (predictors, standardizer and PCA, projections, permutations) is written and hashed (``FROZEN.json``)
before any evaluated partition is processed, and the evaluation reads the frozen files back.  Fidelity is the agreement
with ``d_model`` of the partition on every case with a prediction (as B18; rows with a non-finite input have no
prediction and are counted), with Wilson intervals; random7 and permuted_bhat7 are summarized by the median and the 95th
percentile (numpy linear interpolation) of their 1,000 fidelities, each on the cases that bhat7 and the draw both
predict.  Delta = fidelity(bhat7) - fidelity(pca7) on the cases both predict, with the paired cluster bootstrap of the
lane (2,000 replicates, seed 20260928, unit cluster_id); the comparison of fidelity(bhat7) with P95 also uses the cases
bhat7 and the projections both predict, and the reading row reports both counts (``analyses.B23.rows``).
The D43 reading is made per external partition and the lower outcome of mms_external and mms2_external is the reading
of the block (O3 < O2b < O2 < O1); acdc_test is reported but does not enter it.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .b18_decision_link import design, flat, label_metrics
from .common import COMMAND, EVALUATED, clean, formal_a, included_partitions, load_partition, reference_labels, write_json, write_table
from .lane_v603 import (
    agreement,
    check_frozen,
    fit_link,
    freeze,
    null_summary,
    paired_difference,
    pca_scores,
    principal_components,
    random_projections,
    read_predictors,
    require_registered,
    standardize,
    standardizer,
    write_array,
    write_predictors,
)

ANALYSIS = "B23"
ESTIMATORS = {"bhat7": "lr_decision_link_bhat7_refit", "pca7": "lr_decision_link_pca7", "random7": "lr_decision_link_random7_x1000",
              "permuted_bhat7": "lr_decision_link_permuted_bhat7_x1000", "formal57": "lr_decision_link_formal57"}
PCA_ESTIMATOR = "pca7_formal57"
N_COMPONENTS = 7
SINGLE = ("bhat7", "pca7", "formal57", "constant")
NULLS = ("random7", "permuted_bhat7")
EXTERNAL = ("mms_external", "mms2_external")
ORDER = ("O3", "O2b", "O2", "O1")              # D43: the lower outcome of the two external partitions is the reading
STATEMENTS = {
    "O1": "the reconstructed concepts carry most of the audited decisions",
    "O2": "the concepts carry as much of the decision as the best seven-dimensional linear summary of the descriptors",
    "O2b": "less than the best seven-dimensional summary of the descriptors but more than random seven-dimensional projections",
    "O3": "the descriptor subspace carries the decision; the concepts add nothing beyond their dimension",
}
CLAIM = {"O1": "the claim stays", "O2": "the claim is replaced by the O2 statement", "O2b": "the claim is replaced by the O2b statement",
         "O3": "the claim is withdrawn and replaced by the O3 statement"}
TOLERANCE = 1e-12


def outcome(fidelity_bhat7: float | None, delta_lower: float | None, delta_upper: float | None, p95_random7: float | None) -> str | None:
    """The D43 outcome of one partition: O3 when fidelity(bhat7) <= P95(random7); otherwise O1 when the lower bound of
    Delta is above 0, O2b when its upper bound is below 0, O2 when the interval contains 0.  None when a needed
    quantity is not estimable."""

    def ok(v: float | None) -> bool:
        return v is not None and np.isfinite(v)

    if not (ok(fidelity_bhat7) and ok(p95_random7)):
        return None
    if not fidelity_bhat7 > p95_random7:
        return "O3"
    if not (ok(delta_lower) and ok(delta_upper)):
        return None
    if delta_lower > 0:
        return "O1"
    if delta_upper < 0:
        return "O2b"
    return "O2"


def lower_outcome(outcomes: Mapping[str, str | None]) -> str | None:
    """The lower outcome over mms_external and mms2_external (None unless both are determined)."""

    values = [outcomes.get(p) for p in EXTERNAL]
    if any(v is None for v in values):
        return None
    return min(values, key=ORDER.index)


def check_reproduction(recorded: pd.DataFrame, refit: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rows comparing the refitted bhat7 fidelity of every partition label with the recorded B18 row; any difference
    (numerator, denominator, or estimate beyond 1e-12) or a missing recorded row refuses (BundleError, exit 7)."""

    from ...stages import BundleError

    table = recorded[recorded["input"].astype(str) == "bhat7"].set_index("partition")
    rows, failures = [], []
    for label, metrics in refit.items():
        new = metrics["fidelity"]
        row = {"partition": label, "refit_fidelity": new.get("estimate"), "refit_k": int(new["k"]), "refit_n": int(new["n"])}
        if label not in table.index:
            row.update({"recorded_fidelity": None, "recorded_k": None, "recorded_n": None, "reproduced": False})
            failures.append(f"{label}: no recorded B18 bhat7 row")
        else:
            rec = table.loc[label]
            r_est = None if pd.isna(rec["fidelity"]) else float(rec["fidelity"])
            same_value = (r_est is None and new.get("estimate") is None) or (
                r_est is not None and new.get("estimate") is not None and abs(r_est - float(new["estimate"])) <= TOLERANCE)
            same = same_value and int(rec["fidelity_k"]) == int(new["k"]) and int(rec["fidelity_n"]) == int(new["n"])
            row.update({"recorded_fidelity": r_est, "recorded_k": int(rec["fidelity_k"]), "recorded_n": int(rec["fidelity_n"]), "reproduced": bool(same)})
            if not same:
                failures.append(f"{label}: refit {new['k']}/{new['n']} vs recorded {int(rec['fidelity_k'])}/{int(rec['fidelity_n'])}")
        rows.append(row)
    if failures:
        raise BundleError(f"{COMMAND} {ANALYSIS}: the refitted bhat7 does not reproduce the recorded B18 fidelities to {TOLERANCE:g} "
                          f"({'; '.join(failures)}); refused")
    return rows


def null_comparison(pred_bhat7: pd.Series, null_preds: Iterable[pd.Series], decision: pd.Series) -> dict[str, Any]:
    """A null's draws against bhat7 on the cases both predict (analyses.B23.rows): each draw's fidelity on the cases
    that bhat7 and the draw predict and that have a decision, their median and P95, and fidelity(bhat7) on the same
    cases.  The draws of one null predict the same cases by construction (projections of the same standardized
    descriptors, or bhat7 itself); draws with different cases would give fidelity(bhat7) several values (ValueError)."""

    d = decision.reindex(pred_bhat7.index)
    own = pred_bhat7.notna() & d.notna()
    ks, ns, masks, n_null = [], [], set(), None
    for pred in null_preds:
        predicted = pred.reindex(pred_bhat7.index).notna() & d.notna()
        masks.add(predicted.to_numpy().tobytes())
        n_null = int(predicted.sum())
        rows = pred_bhat7.index[(own & predicted).to_numpy()]
        k, n = agreement(pred.reindex(rows), d)
        ks.append(k)
        ns.append(n)
    if len(masks) > 1:
        raise ValueError(f"{ANALYSIS}: the draws of one null predict different cases ({len(masks)} row sets); fidelity(bhat7) on the common cases "
                         "would not be unique")
    common = own & (np.frombuffer(next(iter(masks)), dtype=bool) if masks else own.to_numpy())
    k_b, n_b = agreement(pred_bhat7.reindex(pred_bhat7.index[np.asarray(common, dtype=bool)]), d)
    fids = [k / n if n else np.nan for k, n in zip(ks, ns)]
    return {**null_summary(fids), "fidelities": fids, "k": ks, "n": ns, "n_bhat7": int(own.sum()), "n_null": n_null, "n_common": int(n_b),
            "fidelity_bhat7_common": k_b / n_b if n_b else None}


def reading(split: str, fid_bhat7: float | None, delta: Mapping[str, Any], random7: Mapping[str, Any]) -> dict[str, Any]:
    """The D43 row of one partition: fidelity(bhat7) against P95(random7) on the cases bhat7 and the projections both
    predict, and Delta on the cases bhat7 and PCA-7 both predict, each with both counts (analyses.B23.rows)."""

    compared = random7.get("fidelity_bhat7_common")
    o = outcome(compared, delta.get("ci_lower"), delta.get("ci_upper"), random7.get("p95"))
    return {"partition": split, "enters_reading": split in EXTERNAL, "fidelity_bhat7_own_cases": fid_bhat7, "n_bhat7": random7.get("n_bhat7"),
            "fidelity_bhat7": compared, "p95_random7": random7.get("p95"), "n_random7": random7.get("n_null"), "n_common_random7": random7.get("n_common"),
            "delta": delta.get("estimate"), "delta_ci_lower": delta.get("ci_lower"), "delta_ci_upper": delta.get("ci_upper"),
            "delta_status": delta.get("status"), "n_pca7": delta.get("n_pca7"), "n_common_delta": delta.get("n_common"), "outcome": o,
            "statement": STATEMENTS.get(o) if o else None}


def run(pctx: Any) -> dict[str, Any]:
    from ...formal_features import formal_descriptor_names
    from ...stages import predict_labels
    from ...transition_fit import TransitionModel

    acfg = pctx.analysis_cfg(ANALYSIS)
    out = pctx.out_dir(ANALYSIS)
    study = pctx.study
    require_registered(pctx, ANALYSIS, [*ESTIMATORS.values(), PCA_ESTIMATOR])
    stats = pctx.lane_cfg["statistics"]
    min_cases = int(stats["min_cases_per_stratum"])
    rp, pm = dict(stats["random_projections"]), dict(stats["permutations"])
    pcfg = dict(study.formal)
    primary = [str(t) for t in stats["primary_targets"]]
    names = formal_descriptor_names()
    model = TransitionModel.from_dict(pctx.read_json("transitions/transition_model.json"))
    recorded_b18 = pctx.read_second_parquet("postfreeze/B18/decision_link_table.parquet")
    dev = load_partition(pctx, "acdc_dev")
    d_dev = pctx.read_parquet("formal/d_model_ACDC_dev.parquet").set_index("sample_id")["d_model"].reindex(dev.ids)
    y_dev = d_dev.astype(object).to_numpy()
    folds = dev.bhat["outer_fold"].reindex(dev.ids)
    # 1. Inputs on acdc_dev.
    x_b7, untransformed = design(dev.bhat, primary, model.transforms, measured=False)
    a_dev = formal_a(pctx, "ACDC").reindex(dev.ids)[names].to_numpy(dtype=float)
    scaler = standardizer(a_dev)
    z_dev = standardize(a_dev, scaler)
    pca = principal_components(z_dev, N_COMPONENTS)
    projections = random_projections(int(rp["seed"]), int(rp["draws"]), len(names), N_COMPONENTS)
    fit_rows = np.flatnonzero(np.isfinite(x_b7).all(axis=1) & d_dev.notna().to_numpy())
    rng = np.random.default_rng(int(pm["seed"]))
    permutations = np.stack([rng.permutation(len(fit_rows)) for _ in range(int(pm["draws"]))]) if int(pm["draws"]) else np.zeros((0, len(fit_rows)), int)
    counts = d_dev.dropna().astype(str).value_counts()
    majority = str(sorted(counts.index[counts == counts.max()])[0])
    # 2. Fit and freeze every object on acdc_dev before any evaluated partition.
    fitted = {"bhat7": fit_link(x_b7, y_dev, pcfg), "pca7": fit_link(pca_scores(z_dev, pca), y_dev, pcfg), "formal57": fit_link(z_dev, y_dev, pcfg)}
    random_models = [fit_link(z_dev @ r, y_dev, pcfg) for r in projections]
    permuted_models = [fit_link(x_b7[fit_rows][p], y_dev[fit_rows], pcfg) for p in permutations]
    frozen: dict[str, dict[str, Any]] = {}
    files: list = []

    def store(estimator: str, written: list, meta: dict[str, Any]) -> None:
        mdir = out / "models" / estimator
        frozen[estimator] = freeze(pctx, mdir, written, {"estimator_id": estimator, "fit_partition": "acdc_dev", **meta})
        files.extend([*written, mdir / "FROZEN.json"])

    mdir = out / "models" / PCA_ESTIMATOR
    mdir.mkdir(parents=True, exist_ok=True)
    store(PCA_ESTIMATOR, [write_json({"columns": names, "standardizer": scaler, "pca": pca}, mdir / "pca.json", pctx)],
          {"columns": "57 formal descriptors", "n_components": N_COMPONENTS, "n_fit_rows": pca["n_rows"]})
    for key in ("bhat7", "pca7", "formal57"):
        mdir = out / "models" / ESTIMATORS[key]
        mdir.mkdir(parents=True, exist_ok=True)
        store(ESTIMATORS[key], [write_predictors([fitted[key]], mdir / "predictor.json", pctx)],
              {"input": key, "target": "d_model (formal/d_model_ACDC_dev.parquet)", "untransformed_columns": untransformed if key == "bhat7" else []})
    mdir = out / "models" / ESTIMATORS["random7"]
    mdir.mkdir(parents=True, exist_ok=True)
    store(ESTIMATORS["random7"], [write_array(projections, mdir / "projections.npy", pctx), write_predictors(random_models, mdir / "predictors.json", pctx)],
          {"input": "random7", "draws": int(rp["draws"]), "generator": rp.get("generator"), "seed": int(rp["seed"]), "shape": [len(names), N_COMPONENTS]})
    mdir = out / "models" / ESTIMATORS["permuted_bhat7"]
    mdir.mkdir(parents=True, exist_ok=True)
    store(ESTIMATORS["permuted_bhat7"], [write_array(permutations, mdir / "permutations.npy", pctx),
                                         write_predictors(permuted_models, mdir / "predictors.json", pctx)],
          {"input": "permuted_bhat7", "draws": int(pm["draws"]), "generator": pm.get("generator"), "seed": int(pm["seed"]),
           "permuted": "the acdc_dev fit rows of bhat7 against d_model", "n_fit_rows": int(len(fit_rows))})
    mdir = out / "models" / "constant_majority"
    mdir.mkdir(parents=True, exist_ok=True)
    store("constant_majority", [write_json({"decision": majority, "rule": "majority acdc_dev d_model, ties by name"}, mdir / "constant.json", pctx)],
          {"input": "constant"})
    # 3. Read the frozen objects back after their hash check.
    for estimator, record in frozen.items():
        check_frozen(out / "models" / estimator, record)
    models = {k: read_predictors(out / "models" / ESTIMATORS[k] / "predictor.json")[0] for k in ("bhat7", "pca7", "formal57")}
    random_models = read_predictors(out / "models" / ESTIMATORS["random7"] / "predictors.json")
    permuted_models = read_predictors(out / "models" / ESTIMATORS["permuted_bhat7"] / "predictors.json")
    projections = np.load(out / "models" / ESTIMATORS["random7"] / "projections.npy")
    frozen_pca = json.loads((out / "models" / PCA_ESTIMATOR / "pca.json").read_text(encoding="utf-8"))
    scaler, pca = frozen_pca["standardizer"], frozen_pca["pca"]
    majority = json.loads((out / "models" / "constant_majority" / "constant.json").read_text(encoding="utf-8"))["decision"]
    # 4. The B18 reproduction on acdc_dev (outer-fold refits of bhat7, as B18) before any evaluated partition.
    from ...classification import fit_formal_predictor

    oof = pd.Series(None, index=dev.ids, dtype=object)
    fit_mask = np.isfinite(x_b7).all(axis=1) & d_dev.notna().to_numpy()
    for f in sorted(set(folds.dropna().astype(int))):
        held = (folds == f).to_numpy()
        fold_model = fit_formal_predictor(x_b7, y_dev, pcfg, fit_mask=fit_mask & ~held)
        oof[held] = predict_labels(fold_model, x_b7[held])
    refit = {"acdc_dev_oof": label_metrics(oof, d_dev, reference_labels(pctx, "acdc_dev"), dev.clusters, pctx.boot, min_cases)}
    check_reproduction(recorded_b18, refit)
    # 5. Each evaluated partition once, with the frozen objects.
    splits, skipped = included_partitions(pctx, list(EVALUATED), f"{COMMAND} {ANALYSIS}")
    rows, draws, deltas, readings, excluded = [], [], [], {}, {}
    for split in splits:
        part = load_partition(pctx, split)
        d_p = pctx.read_parquet(f"inference/d_model_{split}.parquet").set_index("sample_id")["d_model"].reindex(part.ids)
        reference = reference_labels(pctx, split)
        x_p, _ = design(part.bhat, primary, model.transforms, measured=False)
        z_p = standardize(formal_a(pctx, part.dataset).reindex(part.ids)[names].to_numpy(dtype=float), scaler)
        inputs = {"bhat7": x_p, "pca7": pca_scores(z_p, pca), "formal57": z_p}
        preds = {k: pd.Series(predict_labels(models[k], x), index=part.ids, dtype=object) for k, x in inputs.items()}
        preds["constant"] = pd.Series(majority, index=part.ids, dtype=object)
        excluded[split] = {k: int((~np.isfinite(x).all(axis=1)).sum()) for k, x in inputs.items()}
        metrics = {k: label_metrics(preds[k], d_p, reference, part.clusters, pctx.boot, min_cases) for k in SINGLE}
        refit[split] = metrics["bhat7"]
        for k in SINGLE:
            rows.append({"partition": split, "input": k, "estimator_id": ESTIMATORS.get(k, "constant_majority"), "n_cases": len(part.ids),
                         "n_excluded_nonfinite": excluded[split].get(k, 0), **flat(metrics[k]), "enters_reading": split in EXTERNAL})
        summaries = {}
        for null, fitted_models, x_null in (("random7", random_models, [z_p @ r for r in projections]),
                                            ("permuted_bhat7", permuted_models, [x_p] * len(permuted_models))):
            cmp = null_comparison(preds["bhat7"], (pd.Series(predict_labels(m, x), index=part.ids, dtype=object) for m, x in zip(fitted_models, x_null)), d_p)
            for i, (k, n) in enumerate(zip(cmp["k"], cmp["n"])):
                draws.append({"partition": split, "null": null, "draw": i, "fidelity_k": k, "fidelity_n": n, "fidelity": k / n if n else None})
            summaries[null] = cmp
            rows.append({"partition": split, "input": null, "estimator_id": ESTIMATORS[null], "n_cases": len(part.ids), "n_excluded_nonfinite": None,
                         "null_median": cmp["median"], "null_p95": cmp["p95"], "null_n_draws": cmp["n_draws"], "rows": "cases bhat7 and the draw both predict",
                         "n_null": cmp["n_null"], "n_common_with_bhat7": cmp["n_common"], "fidelity_bhat7_common": cmp["fidelity_bhat7_common"],
                         "enters_reading": split in EXTERNAL})
        delta = paired_difference(preds["bhat7"], preds["pca7"], d_p, part.clusters, pctx.boot, min_cases)
        delta["n_pca7"] = agreement(preds["pca7"], d_p)[1]
        deltas.append({"partition": split, "difference": "fidelity(bhat7) - fidelity(pca7)", "rows": "cases both predict", **delta,
                       "n_bhat7": agreement(preds["bhat7"], d_p)[1], "enters_reading": split in EXTERNAL})
        readings[split] = reading(split, metrics["bhat7"]["fidelity"].get("estimate"), delta, summaries["random7"])
    reproduction = check_reproduction(recorded_b18, refit)        # refuses before any table is written
    block = lower_outcome({p: (readings.get(p) or {}).get("outcome") for p in EXTERNAL})
    reading_doc = {"decision": "D43", "fixed_before_computation": True, "order": "O3 < O2b < O2 < O1", "statements": STATEMENTS,
                   "rule": acfg["reading"], "partitions": {p: readings.get(p) for p in EXTERNAL},
                   "descriptive_only": {p: readings[p] for p in readings if p not in EXTERNAL},
                   "not_executed": {p: skipped[p] for p in EXTERNAL if p in skipped},
                   "reading": block, "statement": STATEMENTS.get(block) if block else None, "claim": CLAIM.get(block) if block else None,
                   "status": "DETERMINED" if block else "NOT_DETERMINED"}
    paths = {"fidelity": out / "null_fidelity_table.parquet", "nulls": out / "null_distributions.parquet", "delta": out / "delta_table.parquet",
             "reproduction": out / "b18_reproduction.parquet", "reading": out / "reading.json", "summary": out / "summary.json"}
    write_table(pd.DataFrame(rows), paths["fidelity"], pctx)
    write_table(pd.DataFrame(draws, columns=["partition", "null", "draw", "fidelity_k", "fidelity_n", "fidelity"]), paths["nulls"], pctx)
    write_table(pd.DataFrame(deltas), paths["delta"], pctx)
    write_table(pd.DataFrame(reproduction), paths["reproduction"], pctx)
    write_json(clean(reading_doc), paths["reading"], pctx)
    summary = clean({"analysis": ANALYSIS, "model": acfg["model"], "target": acfg["target"], "inputs": acfg["inputs"], "metric": acfg["metric"],
                     "comparison": acfg["comparison"], "frozen": frozen, "majority_decision": majority, "excluded_nonfinite": excluded,
                     "b18_reproduction": reproduction, "included": splits, "not_executed": skipped, "reading": reading_doc,
                     "rows": json.loads(pd.DataFrame(rows).to_json(orient="records"))})
    write_json(summary, paths["summary"], pctx)
    pctx.write_provenance(ANALYSIS, dict(pctx.verified_log), [*files, *paths.values()],
                          {"included": splits, "not_executed": skipped, "reading": block, "b18_reproduced": True,
                           "outcomes": {p: (readings.get(p) or {}).get("outcome") for p in readings}})
    return {"included": splits, "not_executed": skipped, "reading": block}
