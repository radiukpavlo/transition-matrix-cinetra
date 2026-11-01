"""B18: the decision-link test (EPC-2; ``configs/exploratory_postfreeze.yaml`` analyses.B18).

How much of the audited decisions the concepts carry: a multinomial logistic regression with the audited-predictor
settings (``configs/formal.yaml`` audited_predictor through :func:`tmmri.classification.fit_formal_predictor`: L2,
C = 1.0, no search, standardized on its fit rows) predicts the final ``d_model`` decisions of acdc_dev from

* bhat7: the seven primary targets of the nested out-of-fold B-hat,
* bhat9: the nine rule concepts of the nested out-of-fold B-hat,
* measured7: the seven primary targets of measured B (OBSERVED cells),

each under the frozen typed transforms of the transition (``transitions/transition_model.json``, no clipping; a column
without a frozen transform enters untransformed and is listed).  Rows with a non-finite input are excluded and counted.
Each final model is written and frozen by hash before it is applied once to the raw recorded B-hat (bhat7, bhat9) or the
measured B (measured7) of every evaluated partition; acdc_dev is reported out of fold (outer-fold refits).
Metrics: fidelity (agreement with ``d_model`` of the same partition, Wilson); accuracy (Wilson), macro-F1 (cluster
bootstrap) and the majority-class rate (Wilson) on the same-definition reference subsets; the recorded F8 fidelity
(full coverage) is quoted beside.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .common import (
    EVALUATED,
    clean,
    included_partitions,
    load_partition,
    reference_labels,
    write_json,
    write_table,
)
from .estimates import wilson

ANALYSIS = "B18"
ESTIMATORS = {"bhat7": "lr_decision_link_bhat7", "bhat9": "lr_decision_link_bhat9", "measured7": "lr_decision_link_measured7"}


def design(frame: pd.DataFrame, columns: list[str], transforms: Any, *, measured: bool) -> tuple[np.ndarray, list[str]]:
    """Inputs under the frozen typed transforms; measured B keeps only OBSERVED cells, B-hat its values (state companions
    dropped).  Returns the matrix and the columns without a frozen transform (entered untransformed)."""

    from ...target_transforms import observed_values

    src = frame if measured else frame[[c for c in frame.columns if not str(c).endswith("__state")]]
    covered = [c for c in columns if c in transforms.order]
    typed = transforms.transform(src, clip=False) if covered else pd.DataFrame(index=src.index)
    out = pd.DataFrame({c: (typed[c].to_numpy(dtype=float) if c in covered else observed_values(src, c)) for c in columns}, index=src.index)
    return out.to_numpy(dtype=float), [c for c in columns if c not in covered]


def label_metrics(pred: pd.Series, d_model: pd.Series, reference: pd.Series, clusters: pd.Series, boot: dict[str, Any], min_cases: int) -> dict[str, Any]:
    from sklearn.metrics import f1_score

    from ..alt_bundles import bootstrap_statistic

    pred = pred.astype(object).where(pred.notna(), None)
    fid = pred.notna() & d_model.reindex(pred.index).notna()
    ref = reference.reindex(pred.index)
    acc = pred.notna() & ref.notna()
    k_fid = int((pred[fid].astype(str) == d_model.reindex(pred.index)[fid].astype(str)).sum())
    y_t, y_p = ref[acc].astype(str).to_numpy(), pred[acc].astype(str).to_numpy()
    out: dict[str, Any] = {"n_predicted": int(pred.notna().sum()), "fidelity": wilson(k_fid, int(fid.sum())),
                           "accuracy": wilson(int((y_t == y_p).sum()), int(acc.sum()))}
    if acc.sum():
        counts = pd.Series(y_t).value_counts().sort_index()
        top = counts[counts == counts.max()].index[0]
        out["majority_class"] = str(top)
        out["majority_rate"] = wilson(int(counts.max()), int(acc.sum()))
        cl = clusters.reindex(pred.index)[acc].astype(str).to_numpy(dtype=object)
        below = int(acc.sum()) < int(min_cases)
        f1 = bootstrap_statistic(lambda r: f1_score(y_t[r], y_p[r], average="macro", zero_division=0), int(acc.sum()), cl,
                                 n_boot=0 if below else int(boot["replicates"]), seed=int(boot["seed"]), level=float(boot.get("level", 0.95)),
                                 min_valid_fraction=0.0)
        if below and f1.get("estimate") is not None:   # point estimate kept, interval NE (as every v6 stratum below the minimum)
            f1.update({"ci_lower": None, "ci_upper": None, "status": "NE", "reason": "STRATUM_BELOW_MIN_CASES"})
        out["macro_f1"] = f1
    else:
        out["majority_class"] = None
        out["majority_rate"] = wilson(0, 0)
        out["macro_f1"] = {"estimate": None, "ci_lower": None, "ci_upper": None, "status": "NE", "reason": "NO_REFERENCE_ROWS"}
    return out


def flat(m: dict[str, Any]) -> dict[str, Any]:
    out = {"n_predicted": m["n_predicted"], "majority_class": m["majority_class"]}
    for key in ("fidelity", "accuracy", "majority_rate", "macro_f1"):
        out.update({key: m[key].get("estimate"), f"{key}_ci_lower": m[key].get("ci_lower"), f"{key}_ci_upper": m[key].get("ci_upper")})
        if "k" in m[key]:
            out.update({f"{key}_k": m[key]["k"], f"{key}_n": m[key]["n"]})
    out["macro_f1_status"] = m["macro_f1"].get("status")
    return out


def run(pctx: Any) -> dict[str, Any]:
    from ...classification import FormalPredictor, fit_formal_predictor
    from ...runs import sha256_file
    from ...stages import predict_labels
    from ...transition_fit import TransitionModel

    acfg = pctx.analysis_cfg(ANALYSIS)
    out = pctx.out_dir(ANALYSIS)
    study = pctx.study
    registry = {str(e["id"]): e for e in pctx.lane_cfg["estimators"]}
    if any(ESTIMATORS[k] not in registry or registry[ESTIMATORS[k]].get("analysis") != ANALYSIS for k in ESTIMATORS):
        raise ValueError(f"{ANALYSIS}: the decision-link estimators must be registered in the lane estimators")
    min_cases = int(pctx.lane_cfg["statistics"]["min_cases_per_stratum"])
    pcfg = dict(study.formal)
    primary = [str(t) for t in pctx.lane_cfg["statistics"]["primary_targets"]]
    concepts = list(study.rule_candidates())
    columns = {"bhat7": primary, "bhat9": concepts, "measured7": primary}
    model = TransitionModel.from_dict(pctx.read_json("transitions/transition_model.json"))
    dev = load_partition(pctx, "acdc_dev")
    d_dev = pctx.read_parquet("formal/d_model_ACDC_dev.parquet").set_index("sample_id")["d_model"].reindex(dev.ids)
    folds = dev.bhat["outer_fold"].reindex(dev.ids)
    splits, skipped = included_partitions(pctx, list(EVALUATED), f"run-postfreeze {ANALYSIS}")
    rows, frozen, untransformed, excluded, models = [], {}, {}, {}, {}
    # 1. Fit and freeze every model on acdc_dev (with its out-of-fold development report) before any evaluated partition.
    for key, cols in columns.items():
        measured = key == "measured7"
        x_dev, raw_cols = design(dev.b if measured else dev.bhat, cols, model.transforms, measured=measured)
        untransformed[key] = raw_cols
        finite = np.isfinite(x_dev).all(axis=1)
        fit_rows = finite & d_dev.notna().to_numpy()
        excluded[key] = {"acdc_dev": int((~finite).sum())}
        final = fit_formal_predictor(x_dev, d_dev.astype(object).to_numpy(), pcfg, fit_mask=fit_rows)
        mdir = out / "models" / ESTIMATORS[key]
        mdir.mkdir(parents=True, exist_ok=True)
        path = mdir / "predictor.json"
        write_json(final.to_dict(), path, pctx)
        frozen[key] = {"estimator_id": ESTIMATORS[key], "file": pctx.portable(path), "sha256": sha256_file(path), "columns": cols,
                       "n_fit_rows": int(fit_rows.sum()), "untransformed_columns": raw_cols}
        write_json(frozen[key], mdir / "FROZEN.json", pctx)
        if sha256_file(path) != frozen[key]["sha256"]:
            raise RuntimeError(f"{ANALYSIS}: frozen predictor {path} changed")
        models[key] = FormalPredictor.from_dict(json.loads(path.read_text(encoding="utf-8")))
        # acdc_dev out of fold: the same model refitted on the outer-training rows of each fold.
        oof = pd.Series(None, index=dev.ids, dtype=object)
        for f in sorted(set(folds.dropna().astype(int))):
            held = (folds == f).to_numpy()
            fold_model = fit_formal_predictor(x_dev, d_dev.astype(object).to_numpy(), pcfg, fit_mask=fit_rows & ~held)
            oof[held] = predict_labels(fold_model, x_dev[held])
        m = label_metrics(oof, d_dev, reference_labels(pctx, "acdc_dev"), dev.clusters, pctx.boot, min_cases)
        rows.append({"input": key, "estimator_id": ESTIMATORS[key], "partition": "acdc_dev_oof", "n_cases": len(dev.ids),
                     "n_excluded_nonfinite": excluded[key]["acdc_dev"], **flat(m), "f8_recorded_fidelity": None,
                     "fidelity_reference": "final d_model (formal/d_model_ACDC_dev.parquet)"})
    # 2. Each evaluated partition once, with the three frozen models.
    for split in splits:
        part = load_partition(pctx, split)
        d_p = pctx.read_parquet(f"inference/d_model_{split}.parquet").set_index("sample_id")["d_model"].reindex(part.ids)
        reference = reference_labels(pctx, split)
        rel = f"comparators/cart_f8/evaluation/metrics_{split}.json"
        f8 = ((pctx.read_json(rel).get("versions") or {}).get("full_coverage") or {}).get("fidelity_to_d_model") if pctx.exists(rel) else None
        for key, cols in columns.items():
            measured = key == "measured7"
            x, _ = design(part.b if measured else part.bhat, cols, model.transforms, measured=measured)
            pred = pd.Series(predict_labels(models[key], x), index=part.ids, dtype=object)
            excluded[key][split] = int((~np.isfinite(x).all(axis=1)).sum())
            m = label_metrics(pred, d_p, reference, part.clusters, pctx.boot, min_cases)
            rows.append({"input": key, "estimator_id": ESTIMATORS[key], "partition": split, "n_cases": len(part.ids),
                         "n_excluded_nonfinite": excluded[key][split], **flat(m),
                         "f8_recorded_fidelity": (f8 or {}).get("estimate"), "f8_recorded_fidelity_ci_lower": (f8 or {}).get("wilson_lower"),
                         "f8_recorded_fidelity_ci_upper": (f8 or {}).get("wilson_upper"), "f8_source": rel if f8 else None,
                         "fidelity_reference": f"d_model (inference/d_model_{split}.parquet)"})
    paths = {"table": out / "decision_link_table.parquet", "summary": out / "summary.json"}
    write_table(pd.DataFrame(rows), paths["table"], pctx)
    labels = ["acdc_dev_oof", *splits]
    summary = clean({"analysis": ANALYSIS, "model": acfg["model"], "target": acfg["target"], "inputs": acfg["inputs"], "metrics": acfg["metrics"],
                     "frozen_models": frozen, "untransformed_columns": untransformed, "excluded_nonfinite": excluded, "included": labels,
                     "not_executed": skipped, "rows": json.loads(pd.DataFrame(rows).to_json(orient="records"))})
    write_json(summary, paths["summary"], pctx)
    outputs = list(paths.values())
    outputs += [out / "models" / e / n for e in ESTIMATORS.values() for n in ("predictor.json", "FROZEN.json")]
    pctx.write_provenance(ANALYSIS, dict(pctx.verified_log), outputs, {"included": labels, "not_executed": skipped, "frozen_models": frozen})
    return {"included": labels, "not_executed": skipped, "frozen_models": len(frozen)}
