"""B25: rule admission on a pooled development sample, the sample-size test of the empty rulebook (EPC-4, readiness
review R3; ``configs/exploratory_postfreeze_v603.yaml`` analyses.B25; Decision D44, which supersedes D37 for B25 only).

* **Frozen.**  The recorded cuts and bins of the ``d_model`` rulebook (``thresholds/d_model_<profile>.json``), its concept
  eligibility (``rules/semantic_eligibility.json``), the gate bounds of ``configs/rule_policy.yaml``, the transition and
  the predictor.  Nothing of these is refitted: the gate's cut function returns the recorded cuts in every fold.
* **Admission sample.**  acdc_dev: the rows of the frozen Section 11.6 gate of the d_model rulebook (the non-critical-QC
  development cases with the nested out-of-fold B-hat) with the final ``d_model`` decision
  (``formal/d_model_ACDC_dev.parquet``, in sample, the consequent of the frozen rule layer's own discovery) in their
  recorded outer folds, so that B25 differs from the frozen rule layer only in its admission sample; mms_external: the non-critical-QC cases
  with the frozen-bundle B-hat (``reconstructed/bhat_mms_external.parquet``) and the final ``d_model`` decision, in the
  outer folds of D46 (every mms_external cluster of the split manifest, before any exclusion, sorted by the SHA-256 hex
  digest of the salt ``protocol-v6.0.3/B25`` followed by the ``cluster_id`` and assigned to folds 1-5 round robin;
  every case of a cluster in one fold).  An mms_external case without a decision is excluded and counted.
* **Gate folds.**  In outer fold f the acdc_dev rows take the substrate and the decisions of the frozen gate
  (:func:`frozen_gate_parts`: the recorded nested inner OOF B-hat in outer-train(f), the outer OOF B-hat held out, and
  the decisions of the predictor refitted without fold f, which reproduce the recorded cross-fitted decisions); the
  mms_external rows keep the frozen-bundle B-hat and the final ``d_model`` in every fold.
* **Symbols.**  The recorded cuts applied to B-hat; a cell flagged ``UNSTABLE_ESTIMATE`` in the recorded stability table
  of its partition (``reconstructed/stability_d_model_mms_external.parquet``; none is recorded for the development
  out-of-fold B-hat) satisfies no literal.
* **Fitted with the pooled sample** (D44): granules, variable-precision discovery over the frozen eligible concepts,
  the cross-fitted gate of Section 11.6 (:func:`tmmri.rough_sets.cross_fitted_gate` with the recorded cuts in every
  fold), pooled held-out matches and Wilson bounds; a candidate ledger as B15.  The rulebook is written
  and hashed before ``mms2_external`` is processed.
* **Evaluation.**  ``mms2_external`` only (``mms_external`` is never an evaluation partition of B25): the admitted rules
  with the recorded cuts and the routing of Section 11.7 (decoders and per-object stability of the recorded outer-fold
  transitions, as ``evaluate``); coverage, fidelity to ``d_model`` among accepted cases and accuracy among accepted
  cases on the same-definition subset, with Wilson intervals.  An empty rulebook is reported as such.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from ...runs import hash_path
from .b16_coverage_fidelity import _flat, route_labels, system_metrics
from .candidate_ledger import ledger
from .common import COMMAND, DATASET, clean, formal_a, included_partitions, load_partition, reference_labels, split_manifest, write_json, write_table
from .lane_v603 import check_frozen, freeze, require_registered

ANALYSIS = "B25"
ESTIMATOR = "rulebook_pooled_d_model"
RULEBOOK = "d_model"
UNSTABLE = "UNSTABLE_ESTIMATE"
EVALUATION = ("mms2_external",)


def fold_assignment(clusters: Iterable[Any], salt: str, n_folds: int) -> dict[str, int]:
    """D46: the distinct ``cluster_id`` values sorted by the SHA-256 hex digest of ``salt + cluster_id`` (UTF-8; ties by
    cluster_id) and assigned to folds 1..n_folds round robin in that order."""

    unique = sorted({str(c) for c in clusters}, key=lambda c: (hashlib.sha256((str(salt) + c).encode("utf-8")).hexdigest(), c))
    return {c: i % int(n_folds) + 1 for i, c in enumerate(unique)}


def mms_fold_assignment(manifest: pd.DataFrame, salt: str, n_folds: int) -> dict[str, int]:
    """D46 over every mms_external cluster of the split manifest, before any exclusion (critical QC, no decision);
    ``cluster_id`` as the v6 bootstrap (a case without one is its own cluster)."""

    from ...stages import _cluster_ids

    rows = manifest[manifest["partition"].astype(str) == "mms_external"]
    return fold_assignment(_cluster_ids(rows, rows["sample_id"].astype(str)).tolist(), salt, n_folds)


def frozen_gate_parts(pctx: Any, candidates: list[str], n_folds: int) -> tuple[pd.DataFrame, Callable[..., np.ndarray], Any]:
    """The acdc_dev part of the admission sample as the frozen Section 11.6 gate of the d_model rulebook builds it
    (:func:`tmmri.stages.gate_rules`):

    * its rows and order: the d_model rule substrate on the acdc_dev fit rows (``_rule_substrate``, critical QC removed);
    * its substrate: :class:`tmmri.rule_pipeline.NestedGateSubstrate` preloaded with the recorded nested inner OOF B-hat
      (``oof/inner_oof_ACDC_dev_fold<f>.parquet``) for outer-train(f) and the outer OOF B-hat for the held-out fold; a
      missing file refuses (nothing is refitted);
    * its fold decisions: the audited predictor refitted without fold f on the gate rows (``gate_rules``' d_model_fn),
      which must reproduce the recorded cross-fitted decisions (``formal/d_model_crossfit_ACDC_dev.parquet``) on every
      held-out fold, else BundleError.

    Returns the frame (the final in-sample ``d_model`` of ``formal/d_model_ACDC_dev.parquet`` as the consequent of
    discovery and the ledger, as the frozen rule layer; recorded outer fold; anchor; B-hat with ESTIMATED states),
    ``fold_predictor(train, target)`` over its positions and the substrate."""

    from ...classification import fit_formal_predictor
    from ...rule_pipeline import NestedGateSubstrate, acdc_inner_labels
    from ...stages import BundleError, _cohort, _dev_matrices, _rule_substrate, _rulebooks, predict_labels

    study = pctx.study
    for rel in ("formal/A_ACDC.parquet", "semantics/B_ACDC.parquet", "contracts/split_manifest.parquet", "contracts/cohort_ACDC.parquet",
                "contracts/inner_folds.parquet", "oof/bhat_oof_ACDC_dev.parquet", "formal/d_model_ACDC_dev.parquet"):
        pctx.source_file(rel)
    inner_rel = {f: f"oof/inner_oof_ACDC_dev_fold{f}.parquet" for f in range(1, int(n_folds) + 1)}
    missing = [rel for rel in inner_rel.values() if not pctx.exists(rel)]
    if missing:
        raise BundleError(f"{COMMAND} {ANALYSIS}: the frozen gate's nested inner out-of-fold B-hat is not recorded ({missing}); nothing is refitted; "
                          "refused")
    ids_frame, a_dev, b_rows, folds_all = _dev_matrices(pctx.stage_ctx)
    row_ids = ids_frame["sample_id"].astype(str).to_numpy()
    gate_values, final_decision = _rule_substrate(pctx.stage_ctx, _rulebooks(pctx.stage_ctx)[RULEBOOK])
    subset = np.flatnonzero(pd.Index(row_ids).isin(gate_values.index))
    gate_ids = row_ids[subset]
    gate_folds = np.asarray(folds_all, dtype=int)[subset]
    labels_all = _cohort(pctx.stage_ctx, "ACDC").set_index("sample_id").loc[row_ids, "acdc_class"].astype(str).to_numpy()
    oof = pd.read_parquet(pctx.source_root / "oof" / "bhat_oof_ACDC_dev.parquet").set_index("sample_id")
    inner = pd.read_parquet(pctx.source_root / "contracts" / "inner_folds.parquet")
    substrate = NestedGateSubstrate(a_dev, b_rows, study.transition_targets(), study.transition, study.registry, folds_all,
                                    acdc_inner_labels(inner, row_ids), oof, candidates, row_ids, subset=subset)
    substrate.preload({f: pd.read_parquet(pctx.source_file(rel)).set_index("sample_id") for f, rel in inner_rel.items()})
    pcfg = study.formal["audited_predictor"]
    predictors: dict[tuple[int, ...], Any] = {}

    def fold_predictor(train: np.ndarray, target: np.ndarray) -> np.ndarray:
        train, target = np.asarray(train, dtype=int), np.asarray(target, dtype=int)
        key = tuple(int(i) for i in train)
        if key not in predictors:
            predictors[key] = fit_formal_predictor(a_dev[subset[train]], pd.Series(labels_all[subset[train]]), pcfg)
        return np.asarray(predict_labels(predictors[key], a_dev[subset[target]]), dtype=object)

    cross = pctx.read_parquet("formal/d_model_crossfit_ACDC_dev.parquet").set_index("sample_id")
    cross.index = cross.index.astype(str)
    recorded = cross["d_model_crossfit"].reindex(gate_ids)
    if recorded.isna().any() or not (cross["outer_fold"].reindex(gate_ids).astype(float).to_numpy() == gate_folds.astype(float)).all():
        raise ValueError(f"{ANALYSIS}: the recorded cross-fitted decisions do not cover the frozen gate rows in their outer folds")
    mismatched = {}
    for f in sorted(set(gate_folds.tolist())):
        held = np.flatnonzero(gate_folds == f)
        pred = fold_predictor(np.flatnonzero(gate_folds != f), held)
        bad = int(sum(str(p) != str(r) for p, r in zip(pred, recorded.iloc[held])))
        if bad:
            mismatched[int(f)] = bad
    if mismatched:
        raise BundleError(f"{COMMAND} {ANALYSIS}: the fold predictors of the frozen gate do not reproduce the recorded cross-fitted decisions "
                          f"(differing cases per held-out fold: {mismatched}); refused")
    frame = pd.DataFrame({"source": "acdc_dev", "decision": final_decision.reindex(gate_ids).to_numpy(dtype=object), "fold": gate_folds.astype(float),
                          "anchor": [f"ACDC:{s.split(':', 1)[1]}" for s in gate_ids]}, index=pd.Index(gate_ids, name="sample_id"))
    return frame.join(admission_states(gate_values.loc[gate_ids], None, candidates)), fold_predictor, substrate


def admission_states(values: pd.DataFrame, unstable: pd.DataFrame | None, candidates: list[str]) -> pd.DataFrame:
    """B-hat of the candidates with ``<feature>__state`` companions: UNSTABLE_ESTIMATE where the recorded stability table
    flags the cell, ESTIMATED otherwise (the frozen cuts then give the cell no literal)."""

    out = values.reindex(columns=candidates).apply(pd.to_numeric, errors="coerce")
    for c in candidates:
        flag = unstable[c].astype(object).reindex(out.index) == UNSTABLE if unstable is not None and c in unstable.columns else pd.Series(False, index=out.index)
        out[f"{c}__state"] = np.where(flag.fillna(False).to_numpy(dtype=bool), UNSTABLE, "ESTIMATED")
    return out


def run(pctx: Any) -> dict[str, Any]:
    from ...formal_features import formal_descriptor_names
    from ...rough_sets import apply_gate, cross_fitted_gate, induce_vprs_rules
    from ...rule_pipeline import decode_reconstructions, qc_route_frame, route_records, routing_substrate, rulebook_payload
    from ...stages import fold_model_paths, load_fold_models
    from ...transition_fit import TransitionModel
    from ...wedd import CutSet

    acfg = pctx.analysis_cfg(ANALYSIS)
    out = pctx.out_dir(ANALYSIS)
    study = pctx.study
    require_registered(pctx, ANALYSIS, [ESTIMATOR])
    if [str(p) for p in acfg["partitions"]] != list(EVALUATION):
        raise ValueError(f"{ANALYSIS}: the only evaluation partition is mms2_external (D44), got {acfg['partitions']}")
    policy = dict(study.rule_policy)
    profile = str(study.wedd["primary_profile"])
    n_folds = int(policy["validation_gate"]["folds"])
    assignment = dict(pctx.lane_cfg["statistics"]["b25_fold_assignment"])
    # Frozen objects of the d_model rulebook (read, never refitted).
    cuts_rel = f"thresholds/d_model_{profile}.json"
    cuts_path = pctx.source_file(cuts_rel)
    cuts_recorded = json.loads(cuts_path.read_text(encoding="utf-8"))
    cuts = CutSet.from_dict(cuts_recorded)
    stability = pctx.read_parquet(f"thresholds/d_model_{profile}_stability.parquet")
    attributes = [str(a) for a in pctx.read_json("rules/semantic_eligibility.json")["rulebooks"][RULEBOOK]["eligible_features"]]
    induced_attributes = [str(a) for a in pctx.read_json(f"rules/{RULEBOOK}/rulebook_induced.json").get("attributes", attributes)]
    if induced_attributes != attributes:
        raise ValueError(f"{ANALYSIS}: the recorded eligible concepts {attributes} differ from the induced attributes {induced_attributes}")
    candidates = [c for c in study.rule_candidates() if c in cuts.features]
    # Admission sample, acdc_dev part: the rows, substrate and fold decisions of the frozen Section 11.6 gate of the d_model
    # rulebook (stages.gate_rules), read from the frozen run; the recorded nested inner OOF B-hat, never refitted.
    dev, fold_predictor, substrate = frozen_gate_parts(pctx, candidates, n_folds)
    n_dev = len(dev.index)
    gate_source = {"acdc_dev_rows": n_dev, "substrate": f"recorded oof/inner_oof_ACDC_dev_fold1-{n_folds}.parquet in outer-train(f), outer OOF B-hat held out",
                   "decisions": "audited predictor refitted without fold f on the frozen gate rows (gate_rules); reproduces "
                                "formal/d_model_crossfit_ACDC_dev.parquet on every held-out fold", "fold_decisions_reproduced": True,
                   "mms_external": "frozen-bundle B-hat and final d_model in every fold"}
    reason = pctx.partition_status("mms_external", f"{COMMAND} {ANALYSIS}")
    if reason is not None:
        raise ValueError(f"{ANALYSIS}: the pooled admission sample needs mms_external (D44): {reason}")
    # mms_external part (D44, D46): folds over every mms_external cluster of the split manifest, before any exclusion.
    mms = load_partition(pctx, "mms_external")
    unstable = pctx.read_parquet("reconstructed/stability_d_model_mms_external.parquet").set_index("sample_id")
    d_mms = pctx.read_parquet("inference/d_model_mms_external.parquet").set_index("sample_id")["d_model"]
    folds_mms = mms_fold_assignment(split_manifest(pctx, "mms_external"), str(assignment["salt"]), n_folds)
    mms_frame = pd.DataFrame({"source": "mms_external", "decision": d_mms.reindex(mms.ids).to_numpy(dtype=object),
                              "fold": [float(folds_mms[str(c)]) for c in mms.clusters.reindex(mms.ids)], "anchor": [f"{mms.dataset}:{u}" for u in mms.units]},
                             index=pd.Index(mms.ids, name="sample_id")).join(admission_states(mms.bhat.loc[mms.ids], unstable, candidates))
    has_decision = mms_frame["decision"].notna().to_numpy()
    excluded = {"mms_external": int((~has_decision).sum())}
    pooled = pd.concat([dev, mms_frame.loc[has_decision]])              # acdc_dev rows first, in the frozen gate's order
    decision = pooled["decision"].astype(str)
    values = pooled[[*candidates, *[f"{c}__state" for c in candidates]]]
    anchors = pooled["anchor"].to_numpy(dtype=object)
    folds = pooled["fold"].to_numpy(dtype=float)
    states = cuts.transform(values)
    decisions = decision.to_numpy(dtype=object)
    induce_kwargs = {"registry": dict(study.registry), "rulebook_id": RULEBOOK, "attributes": attributes}

    def frozen_cuts(_values: pd.DataFrame, _decision: np.ndarray, _n: int) -> Any:
        return cuts                                                     # D44: the cuts are never refitted

    def decision_fn(train: np.ndarray, target: np.ndarray) -> np.ndarray:
        """acdc_dev rows: the frozen gate's decision in fold f (the predictor without fold f); mms_external: the final d_model."""

        train, target = np.asarray(train, dtype=int), np.asarray(target, dtype=int)
        out = decisions[target].copy()
        is_dev = target < n_dev
        if is_dev.any():
            out[is_dev] = fold_predictor(train[train < n_dev], target[is_dev])
        return out

    def substrate_fn(f: int, train: np.ndarray, held: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
        """acdc_dev rows: the frozen gate's substrate (recorded inner OOF B-hat in outer-train(f), outer OOF B-hat held out);
        mms_external rows: the frozen-bundle B-hat with its recorded unstable cells in every fold."""

        train, held = np.asarray(train, dtype=int), np.asarray(held, dtype=int)
        tr_dev, he_dev = substrate(int(f), train[train < n_dev], held[held < n_dev])
        return (pd.concat([admission_states(tr_dev, None, candidates), values.iloc[train[train >= n_dev]]]),
                pd.concat([admission_states(he_dev, None, candidates), values.iloc[held[held >= n_dev]]]))

    def gate(book: Any) -> pd.DataFrame:
        return cross_fitted_gate(values, decision_fn, folds, policy, final_rulebook=book, cutset_fn=frozen_cuts, anchors=anchors,
                                 induce_kwargs=induce_kwargs, substrate_fn=substrate_fn)

    book = induce_vprs_rules(states, decision, pd.Series(anchors, index=states.index), policy, registry=dict(study.registry), attributes=attributes,
                             consequent_kind=RULEBOOK, rulebook_id=RULEBOOK)
    gate_table = gate(book)
    gated = apply_gate(book, gate_table)
    provenance = {"analysis": ANALYSIS, "estimator_id": ESTIMATOR, "admission_sample": "acdc_dev + mms_external (D44)",
                  "cuts": f"recorded {cuts_rel}", "eligibility": "recorded rules/semantic_eligibility.json", "gate_bounds": "configs/rule_policy.yaml"}
    name, payload = rulebook_payload(RULEBOOK, gated, gate_table, provenance)
    table, ledger_summary = ledger(rulebook="d_model_pooled", states=states, decision=decision, attributes=attributes, book=book, gate_table=gate_table,
                                   gate_fn=gate, policy=policy, registry=dict(study.registry), cuts=cuts, stability=stability,
                                   agreement_reference="final d_model (acdc_dev in sample, as the frozen rule layer; mms_external out of sample)")
    # Admission counts (no identifiers).
    count_rows = []
    for source, rows in [*pooled.groupby("source"), ("pooled", pooled)]:
        unstable_cells = int((rows[[f"{c}__state" for c in candidates]] == UNSTABLE).to_numpy().sum())
        count_rows.append({"source": source, "n_cases": int(len(rows)), "n_excluded_no_decision": int(excluded.get(source, 0)) if source != "pooled"
                           else int(sum(excluded.values())), "n_unstable_cells": unstable_cells,
                           **{f"n_fold{f}": int((rows["fold"] == f).sum()) for f in range(1, n_folds + 1)},
                           **{f"n_decision_{c}": int(n) for c, n in rows["decision"].astype(str).value_counts().sort_index().items()}})
    # Freeze before mms2_external is processed.
    mdir = out / "models" / ESTIMATOR
    mdir.mkdir(parents=True, exist_ok=True)
    written = [write_table(gate_table, mdir / "gate.parquet", pctx), write_json(payload, mdir / name, pctx)]
    frozen = freeze(pctx, mdir, written, {"estimator_id": ESTIMATOR, "fit": "acdc_dev + mms_external, rule admission only (D44)", "rulebook_file": name,
                                          "n_eligible_rules": len(payload.get("records", [])), "attributes": attributes,
                                          "cuts_file": cuts_rel, "cuts_sha256": hash_path(cuts_path),
                                          "cuts_equal_recorded": cuts.to_dict() == CutSet.from_dict(cuts_recorded).to_dict(),
                                          "fold_assignment": assignment})
    check_frozen(mdir, frozen)
    records = list(json.loads((mdir / name).read_text(encoding="utf-8")).get("records", []))
    # Evaluation on mms2_external only.
    splits, skipped = included_partitions(pctx, list(EVALUATION), f"{COMMAND} {ANALYSIS}")
    rows, checks = [], []
    if splits:
        names = formal_descriptor_names()
        model = TransitionModel.from_dict(pctx.read_json("transitions/transition_model.json"))
        for rel in [p.relative_to(pctx.source_root).as_posix() for p in fold_model_paths(pctx.stage_ctx, pctx.stage_ctx.run)]:
            pctx.source_file(rel)
        fold_models = load_fold_models(pctx.stage_ctx, pctx.stage_ctx.run)
    for split in splits:
        keep = [str(s) for s in pctx.read_parquet(f"reconstructed/bhat_{split}.parquet")["sample_id"]]
        ds = DATASET[split]
        a_p = formal_a(pctx, ds).reindex(keep)[names].to_numpy(dtype=float)
        b_p = pctx.read_parquet(f"semantics/B_{ds}.parquet").set_index("sample_id").reindex(keep)
        d_model = pctx.read_parquet(f"inference/d_model_{split}.parquet").set_index("sample_id")["d_model"].reindex(keep)
        qc = qc_route_frame(b_p, np.isfinite(a_p).all(axis=1), study)
        dec, st, fold_dec = decode_reconstructions(model, fold_models, a_p, pd.Index(keep, name="sample_id"), study.registry, study.transition)
        substrate, stab = routing_substrate(dec, st, fold_dec, cuts, study.wedd, study.rule_candidates())
        recorded = pctx.read_parquet(f"reconstructed/stability_d_model_{split}.parquet").set_index("sample_id").reindex(keep)
        cols = [c for c in stab.columns if c in recorded.columns]
        checks.append({"partition": split, "per_object_stability_equals_recorded":
                       bool(stab[cols].astype(str).reset_index(drop=True).equals(recorded[cols].astype(str).reset_index(drop=True)))})
        routes = route_records(records, RULEBOOK, cuts, substrate, qc, study)
        label, processed = route_labels(routes)
        m = system_metrics(label, processed, d_model, reference_labels(pctx, split).reindex(keep))
        rows.append({"rulebook": ESTIMATOR, "partition": split, "n_rules": len(records), **_flat(m), "fidelity_reference": "d_model (final frozen predictor)",
                     "status": "EVALUATED" if records else "EMPTY_RULEBOOK"})
    for split, why in skipped.items():
        rows.append({"rulebook": ESTIMATOR, "partition": split, "n_rules": len(records), "status": "NOT_EXECUTED", "reason": why})
    paths = {"counts": out / "admission_counts.parquet", "ledger": out / "candidate_ledger.parquet", "rejections": out / "rejection_counts_by_reason.parquet",
             "coverage": out / "coverage_fidelity_mms2.parquet", "summary": out / "summary.json"}
    write_table(pd.DataFrame(count_rows), paths["counts"], pctx)
    write_table(table, paths["ledger"], pctx)
    write_table(pd.DataFrame(ledger_summary["rejection_counts"], columns=["rulebook", "first_failed_condition", "n_candidates"]), paths["rejections"], pctx)
    write_table(pd.DataFrame(rows), paths["coverage"], pctx)
    summary = clean({"analysis": ANALYSIS, "decision": "D44", "admission_sample": acfg["admission_sample"], "frozen_objects": acfg["frozen"],
                     "symbols": acfg["symbols"], "gate": acfg["gate"], "gate_source": gate_source, "evaluation": acfg["evaluation"],
                     "attributes": attributes, "frozen": frozen,
                     "admission_counts": count_rows, "ledger": ledger_summary, "n_eligible_rules": len(records), "empty_rulebook": not records,
                     "checks": checks, "included": splits, "not_executed": skipped, "rows": rows})
    write_json(summary, paths["summary"], pctx)
    pctx.write_provenance(ANALYSIS, dict(pctx.verified_log), [*written, mdir / "FROZEN.json", *paths.values()],
                          {"included": splits, "not_executed": skipped, "n_eligible_rules": len(records), "n_candidates": ledger_summary["n_candidates"]})
    return {"included": splits, "not_executed": skipped, "n_eligible_rules": len(records)}
