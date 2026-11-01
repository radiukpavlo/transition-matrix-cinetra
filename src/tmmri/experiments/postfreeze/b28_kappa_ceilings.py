"""B28: agreement and ceilings of the decision-link test (``configs/exploratory_postfreeze_v604.yaml``
analyses.B28; SPECIFICATION.md 6.0.4 Section 13.9; no reading, D48).

* **(a) Kappa.**  The frozen B23 objects of the third source run (``postfreeze/B23/models`` of v6-postfreeze-20261028-2135:
  the bhat7, pca7 and formal57 link models, the PCA-7 standardizer, the development-majority constant, the 1,000 random
  projections and the 1,000 permuted-bhat7 models, each hash-checked through the B23 provenance record) are applied to
  the recorded inputs of every evaluated partition exactly as B23 applied them.  Every fidelity must reproduce the
  recorded B23 value with the same numerator and denominator (``null_fidelity_table.parquet`` for bhat7, pca7,
  formal57 and constant; ``null_distributions.parquet`` draw by draw for random7 and permuted_bhat7), otherwise the
  analysis refuses (BundleError) before any table is written.  Reported: Cohen's kappa (unweighted) against ``d_model``
  for each single input with the cluster bootstrap of the lane, and the median and 95th percentile (numpy linear
  interpolation) of kappa over the draws of each null, each draw on the cases that bhat7 and the draw both predict (the
  rows of the recorded B23 draws); the recorded B23 fidelities beside.
* **(b) Cohort-majority constant.**  Per partition, a constant at that partition's most frequent ``d_model`` decision
  (ties by name): its fidelity (Wilson) and kappa, an outcome-dependent reference, not a method; the development
  majority of B23 beside.
* **(c) Supervised ceiling.**  Linear discriminant analysis (four components, SVD solver) of ``d_model``
  (``formal/d_model_ACDC_dev.parquet``) on the 57 formal descriptors standardized on acdc_dev (mean and SD of the finite
  development rows; a zero SD becomes 1), followed by the B18 model (:func:`tmmri.classification.fit_formal_predictor`
  with the audited-predictor settings: multinomial L2 logistic regression, C = 1) on the four discriminant scores; both
  fitted on acdc_dev and written and hashed (``FROZEN.json``) before any evaluated partition is processed, then read
  back.  Fidelity with a Wilson interval and kappa with the cluster bootstrap; rows with a non-finite descriptor have no
  prediction and are counted.

Values derived from exact counts carry a half-up companion (BLD-1).  No output carries a case or subject identifier.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from . import b23_decision_link_nulls as b23
from .b18_decision_link import design
from .common import COMMAND, OUTPUT_ROOT, clean, formal_a, load_partition
from .lane_v603 import agreement, check_frozen, fit_link, freeze, pca_scores, read_predictors, standardize, standardizer, write_predictors
from .lane_v604 import (
    block_partitions,
    cohen_kappa,
    half_up,
    identifier_tokens,
    kappa_counts,
    kappa_row,
    proportion,
    quantiles,
    require_registered,
    write_checked_json,
    write_checked_table,
)

ANALYSIS = "B28"
SOURCE = "B23"
LDA_ESTIMATOR = "lda4_formal57"
LINK_ESTIMATOR = "lr_decision_link_lda4"
N_COMPONENTS = 4
SINGLE = b23.SINGLE                             # bhat7, pca7, formal57, constant (development majority)
NULLS = b23.NULLS                               # random7, permuted_bhat7
NULL_QUANTILES = {"median": 0.5, "p95": 0.95}


def b23_rel(*parts: str) -> str:
    return "/".join([OUTPUT_ROOT, SOURCE, *parts])


def model_rel(estimator: str, name: str) -> str:
    return b23_rel("models", estimator, name)


# ----------------------------------------------------------------------------------------------- the supervised ceiling


def fit_lda(z: np.ndarray, y: np.ndarray, k: int = N_COMPONENTS) -> dict[str, Any]:
    """LDA (SVD solver, ``k`` components) of ``y`` on the rows of ``z`` with finite inputs and a label; the frozen form
    is the centre ``xbar`` and the first ``k`` columns of ``scalings`` (scores = (z - xbar) @ scalings, as sklearn)."""

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    z = np.asarray(z, dtype=float)
    labels = pd.Series(np.asarray(y, dtype=object))
    rows = np.isfinite(z).all(axis=1) & labels.notna().to_numpy()
    classes = sorted(set(labels[rows].astype(str)))
    if len(classes) <= k:
        raise ValueError(f"{ANALYSIS}: LDA with {k} components needs at least {k + 1} classes of d_model on the fit rows, found {len(classes)}")
    lda = LinearDiscriminantAnalysis(n_components=k, solver="svd").fit(z[rows], labels[rows].astype(str).to_numpy())
    if lda.scalings_.shape[1] < k:
        raise ValueError(f"{ANALYSIS}: the discriminant has rank {lda.scalings_.shape[1]} < {k} on the fit rows")
    doc = {"solver": "svd", "n_components": int(k), "classes": [str(c) for c in lda.classes_], "xbar": [float(v) for v in lda.xbar_],
           "scalings": np.asarray(lda.scalings_[:, :k], dtype=float).tolist(), "explained_variance_ratio": [float(v) for v in lda.explained_variance_ratio_[:k]],
           "n_rows": int(rows.sum())}
    if not np.allclose(lda_scores(z[rows], doc), lda.transform(z[rows]), rtol=0.0, atol=1e-9):
        raise RuntimeError(f"{ANALYSIS}: the frozen discriminant does not reproduce sklearn's transform")
    return doc


def lda_scores(z: np.ndarray, doc: Mapping[str, Any]) -> np.ndarray:
    """Discriminant scores of the frozen LDA (non-finite rows stay non-finite and get no prediction)."""

    return (np.asarray(z, dtype=float) - np.asarray(doc["xbar"], dtype=float)) @ np.asarray(doc["scalings"], dtype=float)


# ----------------------------------------------------------------------------------------------- reproduction of B23


def check_b23_reproduction(recorded_single: pd.DataFrame, recorded_draws: pd.DataFrame, single: Mapping[str, Mapping[str, tuple[int, int]]],
                           draws: Mapping[str, Mapping[str, tuple[list[int], list[int]]]]) -> list[dict[str, Any]]:
    """Rows comparing every re-applied B23 fidelity with the recorded one: the single inputs by (partition, input), the
    nulls draw by draw.  A missing recorded row, another number of draws, or any numerator or denominator that differs
    refuses (BundleError, exit 7); the message names the partition and input, never a value."""

    from ...stages import BundleError

    table = recorded_single.assign(partition=recorded_single["partition"].astype(str), input=recorded_single["input"].astype(str))
    nulls = recorded_draws.assign(partition=recorded_draws["partition"].astype(str), null=recorded_draws["null"].astype(str))
    rows, failures = [], []
    for split, per_input in single.items():
        for name, (k, n) in per_input.items():
            rec = table[(table["partition"] == split) & (table["input"] == name)]
            row = {"partition": split, "input": name, "n_draws": None, "k": int(k), "n": int(n)}
            if len(rec) != 1:
                row.update({"recorded_k": None, "recorded_n": None, "reproduced": False})
                failures.append(f"{split}/{name}: {len(rec)} recorded rows")
            else:
                rk, rn = int(rec["fidelity_k"].iloc[0]), int(rec["fidelity_n"].iloc[0])
                row.update({"recorded_k": rk, "recorded_n": rn, "reproduced": rk == int(k) and rn == int(n)})
                if not row["reproduced"]:
                    failures.append(f"{split}/{name}: numerator or denominator differs")
            rows.append(row)
    for split, per_null in draws.items():
        for name, (ks, ns) in per_null.items():
            rec = nulls[(nulls["partition"] == split) & (nulls["null"] == name)].sort_values("draw")
            same_draws = len(rec) == len(ks) and [int(d) for d in rec["draw"]] == list(range(len(ks)))
            differing = (int(sum(int(a) != int(b) or int(c) != int(d) for a, b, c, d in zip(rec["fidelity_k"], ks, rec["fidelity_n"], ns)))
                         if same_draws else None)
            ok = same_draws and differing == 0
            rows.append({"partition": split, "input": name, "n_draws": len(ks), "n_draws_recorded": int(len(rec)), "n_draws_differing": differing,
                         "reproduced": bool(ok)})
            if not ok:
                failures.append(f"{split}/{name}: " + ("draws differ in number or order" if not same_draws else f"{differing} draws differ"))
    if failures:
        raise BundleError(f"{COMMAND} {ANALYSIS}: the frozen B23 objects do not reproduce the recorded B23 fidelities with the same numerator "
                          f"and denominator ({'; '.join(failures)}); refused")
    return rows


def draw_kappas(pred_bhat7: pd.Series, null_preds: Iterable[pd.Series], decision: pd.Series) -> list[float | None]:
    """Kappa of each draw against ``d_model`` on the cases that bhat7 and the draw both predict (the rows of the B23
    draw fidelities, :func:`b23_decision_link_nulls.null_comparison`)."""

    d = decision.reindex(pred_bhat7.index)
    own = pred_bhat7.notna() & d.notna()
    out = []
    for pred in null_preds:
        rows = (own & pred.reindex(pred_bhat7.index).notna()).to_numpy()
        ids = pred_bhat7.index[rows]
        out.append(cohen_kappa(pred.reindex(ids).astype(str), d.reindex(ids).astype(str)))
    return out


def cohort_majority(decision: pd.Series) -> str | None:
    counts = decision.dropna().astype(str).value_counts()
    return str(sorted(counts.index[counts == counts.max()])[0]) if len(counts) else None


def recorded_beside(table: pd.DataFrame, split: str, name: str, columns: Iterable[str], prefix: str = "recorded_b23_") -> dict[str, Any]:
    rec = table[(table["partition"].astype(str) == split) & (table["input"].astype(str) == name)]
    return {f"{prefix}{c}": (None if rec.empty or c not in rec.columns or pd.isna(rec[c].iloc[0]) else rec[c].iloc[0]) for c in columns}


def run(pctx: Any) -> dict[str, Any]:
    from ...formal_features import formal_descriptor_names
    from ...stages import predict_labels
    from ...transition_fit import TransitionModel

    acfg = pctx.analysis_cfg(ANALYSIS)
    out = pctx.out_dir(ANALYSIS)
    study = pctx.study
    require_registered(pctx, ANALYSIS, [LDA_ESTIMATOR, LINK_ESTIMATOR])
    stats = pctx.lane_cfg["statistics"]
    boot = pctx.boot
    pcfg = dict(study.formal)
    primary = [str(t) for t in stats["primary_targets"]]
    names = formal_descriptor_names()
    model = TransitionModel.from_dict(pctx.read_json("transitions/transition_model.json"))
    # 1. The frozen B23 objects and the recorded B23 fidelities (third source run, hash-checked).
    est = b23.ESTIMATORS
    b23_models = {k: read_predictors(pctx.third_file(model_rel(est[k], "predictor.json")))[0] for k in ("bhat7", "pca7", "formal57")}
    random_models = read_predictors(pctx.third_file(model_rel(est["random7"], "predictors.json")))
    projections = np.load(pctx.third_file(model_rel(est["random7"], "projections.npy")), allow_pickle=False)
    permuted_models = read_predictors(pctx.third_file(model_rel(est["permuted_bhat7"], "predictors.json")))
    frozen_pca = pctx.read_third_json(model_rel(b23.PCA_ESTIMATOR, "pca.json"))
    if [str(c) for c in frozen_pca["columns"]] != list(names):
        raise ValueError(f"{ANALYSIS}: the frozen B23 PCA columns differ from the 57 formal descriptors of this build")
    scaler, pca = frozen_pca["standardizer"], frozen_pca["pca"]
    dev_majority = str(pctx.read_third_json(model_rel("constant_majority", "constant.json"))["decision"])
    if len(projections) != len(random_models):
        raise ValueError(f"{ANALYSIS}: {len(projections)} frozen projections but {len(random_models)} random7 models")
    recorded_single = pctx.read_third_parquet(b23_rel("null_fidelity_table.parquet"))
    recorded_draws = pctx.read_third_parquet(b23_rel("null_distributions.parquet"))
    # 2. The supervised ceiling, fitted on acdc_dev and frozen before any evaluated partition is processed.
    dev = load_partition(pctx, "acdc_dev")
    d_dev = pctx.read_parquet("formal/d_model_ACDC_dev.parquet").set_index("sample_id")["d_model"].reindex(dev.ids)
    y_dev = d_dev.astype(object).to_numpy()
    a_dev = formal_a(pctx, "ACDC").reindex(dev.ids)[names].to_numpy(dtype=float)
    dev_scaler = standardizer(a_dev)
    z_dev = standardize(a_dev, dev_scaler)
    lda = fit_lda(z_dev, y_dev, N_COMPONENTS)
    link = fit_link(lda_scores(z_dev, lda), y_dev, pcfg)
    frozen, files = {}, []
    for estimator, name, writer, meta in (
            (LDA_ESTIMATOR, "lda.json", lambda p: write_checked_json({"columns": names, "standardizer": dev_scaler, "lda": lda}, p, pctx, set()),
             {"input": "57 formal descriptors standardized on acdc_dev", "n_components": N_COMPONENTS, "solver": "svd", "n_fit_rows": lda["n_rows"],
              "target": "d_model (formal/d_model_ACDC_dev.parquet)"}),
            (LINK_ESTIMATOR, "predictor.json", lambda p: write_predictors([link], p, pctx),
             {"input": f"the {N_COMPONENTS} scores of {LDA_ESTIMATOR}", "model": "B18 (configs/formal.yaml audited_predictor settings)",
              "target": "d_model (formal/d_model_ACDC_dev.parquet)"})):
        mdir = out / "models" / estimator
        mdir.mkdir(parents=True, exist_ok=True)
        written = [writer(mdir / name)]
        frozen[estimator] = freeze(pctx, mdir, written, {"estimator_id": estimator, "fit_partition": "acdc_dev", **meta})
        files.extend([*written, mdir / "FROZEN.json"])
    for estimator, record in frozen.items():
        check_frozen(out / "models" / estimator, record)
    ceiling_doc = json.loads((out / "models" / LDA_ESTIMATOR / "lda.json").read_text(encoding="utf-8"))
    ceiling_scaler, lda = ceiling_doc["standardizer"], ceiling_doc["lda"]
    link = read_predictors(out / "models" / LINK_ESTIMATOR / "predictor.json")[0]
    # 3. Each evaluated partition once.
    included, skipped = block_partitions(pctx, [str(p) for p in acfg["partitions"]], f"{COMMAND} {ANALYSIS}")
    single_counts, draw_counts = {}, {}
    kappa_rows, null_rows, majority_rows, ceiling_rows = [], [], [], []
    tokens = identifier_tokens(dev.ids)
    for label, split in included:
        part = load_partition(pctx, split)
        tokens |= identifier_tokens(part.ids)
        d_p = pctx.read_parquet(f"inference/d_model_{split}.parquet").set_index("sample_id")["d_model"].reindex(part.ids)
        x_p, _ = design(part.bhat, primary, model.transforms, measured=False)
        a_p = formal_a(pctx, part.dataset).reindex(part.ids)[names].to_numpy(dtype=float)
        z_p = standardize(a_p, scaler)
        inputs = {"bhat7": x_p, "pca7": pca_scores(z_p, pca), "formal57": z_p}
        preds = {k: pd.Series(predict_labels(b23_models[k], x), index=part.ids, dtype=object) for k, x in inputs.items()}
        preds["constant"] = pd.Series(dev_majority, index=part.ids, dtype=object)
        single_counts[label] = {k: agreement(preds[k], d_p) for k in SINGLE}
        nonfinite = {k: int((~np.isfinite(x).all(axis=1)).sum()) for k, x in inputs.items()}
        for k in SINGLE:
            kk, nn = single_counts[label][k]
            kappa_rows.append({"partition": label, "input": k, "estimator_id": est.get(k, "constant_majority"), "n_cases": len(part.ids),
                               "n_excluded_nonfinite": nonfinite.get(k, 0), "n_no_decision": int(d_p.isna().sum()),
                               **proportion("fidelity", kk, nn), **kappa_row(preds[k], d_p, part.clusters, boot),
                               **recorded_beside(recorded_single, label, k, ("fidelity", "fidelity_ci_lower", "fidelity_ci_upper"))})
        draw_counts[label] = {}
        for null, fitted, x_null in (("random7", random_models, [z_p @ r for r in projections]), ("permuted_bhat7", permuted_models, [x_p] * len(permuted_models))):
            null_preds = [pd.Series(predict_labels(m, x), index=part.ids, dtype=object) for m, x in zip(fitted, x_null)]
            cmp = b23.null_comparison(preds["bhat7"], null_preds, d_p)
            draw_counts[label][null] = ([int(v) for v in cmp["k"]], [int(v) for v in cmp["n"]])
            kappas = draw_kappas(preds["bhat7"], null_preds, d_p)
            common = (preds["bhat7"].notna() & d_p.notna() & null_preds[0].notna()).to_numpy() if null_preds else np.zeros(len(part.ids), bool)
            ids_c = pd.Index(part.ids)[common]
            k_b, n_b = agreement(preds["bhat7"].reindex(ids_c), d_p)
            if n_b != cmp["n_common"] or (n_b and k_b / n_b != cmp["fidelity_bhat7_common"]):
                raise RuntimeError(f"{ANALYSIS}: the common cases of bhat7 and {null} differ from those of the B23 comparison")
            kappa_b = kappa_counts(preds["bhat7"].reindex(ids_c).astype(str), d_p.reindex(ids_c).astype(str))
            null_rows.append({"partition": label, "null": null, "estimator_id": est[null], "rows": "cases bhat7 and the draw both predict",
                              "n_draws": len(kappas), "n_draws_kappa_defined": int(sum(v is not None for v in kappas)), "n_common_with_bhat7": cmp["n_common"],
                              **{f"kappa_{k}": v for k, v in quantiles([np.nan if v is None else v for v in kappas], NULL_QUANTILES).items() if k != "n"},
                              "kappa_bhat7_common": kappa_b[0] / kappa_b[1] if kappa_b[1] else None, "kappa_bhat7_common_half_up": half_up(*kappa_b),
                              "fidelity_null_median": cmp["median"], "fidelity_null_p95": cmp["p95"], **proportion("fidelity_bhat7_common", k_b, n_b),
                              **recorded_beside(recorded_single, label, null, ("null_median", "null_p95", "null_n_draws", "fidelity_bhat7_common"))})
        majority = cohort_majority(d_p)
        for reference, decision in (("development_majority_b23", dev_majority), ("cohort_majority", majority)):
            pred = pd.Series(decision, index=part.ids, dtype=object) if decision is not None else pd.Series(None, index=part.ids, dtype=object)
            kk, nn = agreement(pred, d_p)
            majority_rows.append({"partition": label, "constant": reference, "decision": decision,
                                  "role": "outcome-dependent reference, not a method" if reference == "cohort_majority" else "B23 constant (acdc_dev majority)",
                                  **proportion("fidelity", kk, nn), **kappa_row(pred, d_p, part.clusters, boot)})
        scores = lda_scores(standardize(a_p, ceiling_scaler), lda)
        pred = pd.Series(predict_labels(link, scores), index=part.ids, dtype=object)
        kk, nn = agreement(pred, d_p)
        ceiling_rows.append({"partition": label, "estimator_id": LINK_ESTIMATOR, "input": f"{LDA_ESTIMATOR} scores", "n_cases": len(part.ids),
                             "n_excluded_nonfinite": int((~np.isfinite(scores).all(axis=1)).sum()), "n_predicted": int(pred.notna().sum()),
                             **proportion("fidelity", kk, nn), **kappa_row(pred, d_p, part.clusters, boot),
                             **recorded_beside(recorded_single, label, "formal57", ("fidelity",), "recorded_b23_formal57_"),
                             **recorded_beside(recorded_single, label, "bhat7", ("fidelity",), "recorded_b23_bhat7_")})
    reproduction = check_b23_reproduction(recorded_single, recorded_draws, single_counts, draw_counts)   # refuses before any table
    paths = {"kappa": out / "kappa_table.parquet", "nulls": out / "kappa_null_summary.parquet", "majority": out / "majority_constant_table.parquet",
             "ceiling": out / "ceiling_table.parquet", "reproduction": out / "b23_reproduction.parquet", "summary": out / "summary.json"}
    for key, rows in (("kappa", kappa_rows), ("nulls", null_rows), ("majority", majority_rows), ("ceiling", ceiling_rows), ("reproduction", reproduction)):
        write_checked_table(pd.DataFrame(rows), paths[key], pctx, tokens)
    summary = clean({"analysis": ANALYSIS, "label": "EXPLORATORY_POST_HOC", "reading": None, "kappa": acfg["kappa"],
                     "cohort_majority_constant": acfg["cohort_majority_constant"], "supervised_ceiling": acfg["supervised_ceiling"],
                     "b23_source": f"{pctx.third_root.name}:{b23_rel()}", "b23_reproduced": True, "reproduction": reproduction, "frozen": frozen,
                     "development_majority": dev_majority, "included": [lab for lab, _ in included], "not_executed": skipped,
                     "kappa_rows": json.loads(pd.DataFrame(kappa_rows).to_json(orient="records")),
                     "ceiling_rows": json.loads(pd.DataFrame(ceiling_rows).to_json(orient="records"))})
    write_checked_json(summary, paths["summary"], pctx, tokens)
    pctx.write_provenance(ANALYSIS, dict(pctx.verified_log), [*files, *paths.values()],
                          {"included": [lab for lab, _ in included], "not_executed": skipped, "b23_reproduced": True, "reading": None})
    return {"included": [lab for lab, _ in included], "not_executed": skipped}
