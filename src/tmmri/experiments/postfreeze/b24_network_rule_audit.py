"""B24: rule audit of the image networks F9 and F9-M (EPC-2, readiness review R1; ``configs/exploratory_postfreeze_v603.yaml``
analyses.B24).  Primary seed 20261011 only (F9: ``neural_f9/f9_bundle.json`` primary_seed; F9-M: B22 of the second
source run); no seed is selected.

* **Consequent.**  The network's own decision, the argmax of its class probabilities (ties to the first class): on
  acdc_dev the out-of-fold decision of the outer-fold networks (``is_oof``), on an evaluated partition the final
  network.  Disagreements with the recorded ``y_pred`` are counted.
* **Concepts.**  The network's probe reconstruction of the same seed (F9b: ``probes/f9b/seed20261011/``; F9b-M:
  ``postfreeze/B22/probes/seed20261011/``): the recorded development out-of-fold B-hat for fitting, and the frozen final
  probe on the final encoder's activations of each evaluated partition.  That reconstruction must reproduce the record
  (``reconstruction_check``) before any frozen object is applied, else the analysis refuses (BundleError): for F9b-M the
  recorded ``postfreeze/B22/evaluated`` B-hat (maximum absolute difference at most 1e-6, the same rows and missing
  cells), for F9b the recorded metrics ``probes/evaluations/f9b/<partition>.json`` of the seed, recomputed with the
  metric code that recorded them (:func:`f9b_metric_check`; every number within 1e-6).
* **Rule layer.**  Section 11 through :func:`tmmri.rule_pipeline.fit_rulebook` with ``configs/wedd.yaml`` and
  ``configs/rule_policy.yaml``, the settings of the primary model-decision rulebook (``d_model``) with the network's
  decisions in place of ``d_model``: WEDD cuts, the cut-stability gate, eligibility (the reconstruction gate against the
  recorded out-of-fold mean comparator), granules, variable-precision discovery and the cross-fitted gate over the
  recorded outer folds.  In fold f of the gate the consequent is the decision of the fold-f network (trained without
  fold f) and the concepts of the training folds are the probe refitted inside outer-train(f) on encoder f's activations
  with the recorded inner folds (:class:`ProbeGateSubstrate`, the probe counterpart of the nested gate substrate); the
  held-out fold keeps its recorded out-of-fold probe reconstruction.  Every object is written and hashed before any
  evaluated partition is processed.  A candidate ledger as B15 (:mod:`candidate_ledger`).
* **Application.**  The frozen rulebook routes each evaluated partition with the frozen cuts and the routing of
  Section 11.7 (decoders of the final probe; critical QC and non-finite activations abstain).  Per-object stability
  (Section 11.4) is not applied: the five outer-fold probes map the activations of the five outer-fold encoders, which
  are recorded on acdc_dev only, so no cell becomes UNSTABLE_ESTIMATE (reported as ``per_object_stability``).
  Coverage and fidelity to the network's decisions and accuracy among accepted cases on the same-definition subsets,
  with Wilson intervals, as B16.
* **Decision link** (descriptive, no reading).  As B18, fitted on acdc_dev to the network's out-of-fold decisions (the
  consequent; the final network's in-sample decisions are close to the labels) and evaluated against the final
  network's decisions: the seven primary targets of the development out-of-fold probe reconstruction under the frozen
  typed transforms, and the PCA-7 and random-projection references of B23 on the final encoder's 64 activations
  standardized on acdc_dev (one encoder, so latent coordinates are never pooled across encoders), with the paired
  difference probe7 - pca7.  Critical-QC cases are excluded from the link, as in B18 (``load_partition``); the rule
  coverage keeps every processed case.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

import numpy as np
import pandas as pd

from ...rule_pipeline import NestedGateSubstrate
from .b16_coverage_fidelity import _flat, route_labels, system_metrics
from .b18_decision_link import design, flat, label_metrics
from .candidate_ledger import ledger
from .common import (
    COMMAND,
    DATASET,
    EVALUATED,
    clean,
    dev_reference,
    included_partitions,
    load_partition,
    reference_labels,
    split_manifest,
    write_json,
    write_table,
)
from .lane_v603 import (
    RECONSTRUCTION_TOLERANCE,
    agreement,
    argmax_decisions,
    check_frozen,
    check_reconstruction,
    compare_recorded,
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

ANALYSIS = "B24"
SEED = 20261011
NETWORKS = ("f9", "f9m")
RULE_LAYER = {"f9": f"rule_layer_f9_seed{SEED}", "f9m": f"rule_layer_f9m_seed{SEED}"}
LINK = {"f9": "lr_decision_link_f9b7_and_activation_nulls", "f9m": "lr_decision_link_f9bm7_and_activation_nulls"}
PROBE = {"f9": "F9b", "f9m": "F9b-M"}
RULEBOOK = "d_model"            # the settings of the primary model-decision rulebook; the network's decisions replace d_model
N_COMPONENTS = 7
ACTIVATION = re.compile(r"a_\d+")
EVALUATION_RECORD = "probes/evaluations/f9b/{}.json"       # stage evaluate-alt-bundles of the frozen source run
NO_PER_OBJECT_STABILITY = ("not applied: the outer-fold probes map the outer-fold encoders' activations, which are recorded on acdc_dev "
                           "only; no cell is UNSTABLE_ESTIMATE")


class ProbeGateSubstrate(NestedGateSubstrate):
    """The nested gate substrate of the probe: inside outer-train(f) the probe (the v6 typed ridge) is refitted on
    encoder f's activations with the recorded inner folds of outer fold f, giving the training rows' inner out-of-fold
    concepts; the held-out fold keeps the recorded out-of-fold probe reconstruction (probe f on encoder f)."""

    def __init__(self, activations_by_fold: dict[int, np.ndarray], **kwargs: Any) -> None:
        super().__init__(A=next(iter(activations_by_fold.values())), **kwargs)
        self.by_fold = {int(k): np.asarray(v, dtype=float) for k, v in activations_by_fold.items()}

    def inner_oof(self, f: int) -> pd.DataFrame:
        from ...transition_fit import nested_oof_transition

        if f not in self._cache:
            train = np.flatnonzero(np.isfinite(self.folds) & (self.folds != f))
            inner = np.asarray(self.inner_labels(int(f)), dtype=float)[train]
            if not np.isfinite(inner).all():
                raise ValueError(f"inner fold labels missing for outer-train({f})")
            oof = nested_oof_transition(self.by_fold[int(f)][train], self.B.iloc[train].reset_index(drop=True), self.targets, self.cfg,
                                        self.registry, inner, lambda g, pos, _inner=inner: _inner[np.asarray(pos, dtype=int)])
            oof.index = self.row_ids[train]
            self._cache[f] = oof
        return self._cache[f]


def network_source(pctx: Any, key: str) -> dict[str, Any]:
    """Readers and record names of one network (seed 20261011)."""

    if key == "f9":
        bundle = pctx.read_json("neural_f9/f9_bundle.json")
        if int(bundle["primary_seed"]) != SEED:
            raise ValueError(f"{ANALYSIS}: neural_f9/f9_bundle.json primary_seed {bundle['primary_seed']} is not {SEED}; seed selection is forbidden")
        return {"read": pctx.read_parquet, "read_json": pctx.read_json, "predictions": "neural_f9/predictions_{}.parquet",
                "activations": "neural_f9/activations_{}.parquet", "probe": f"probes/f9b/seed{SEED}", "evaluated": None,
                "meta": {"network": "F9", "probe": "F9b", "source": "frozen source run (neural_f9/, probes/f9b/)", "primary_seed": int(bundle["primary_seed"])}}
    fits = pctx.read_second_json("postfreeze/B22/fits/FROZEN.json")
    completed = [int(s) for s in ((fits.get("provenance") or {}).get("completed_seeds") or [])]
    if SEED not in completed:
        raise ValueError(f"{ANALYSIS}: F9-M seed {SEED} is not a completed seed of B22 ({completed})")
    return {"read": pctx.read_second_parquet, "read_json": pctx.read_second_json, "predictions": "postfreeze/B22/inference/predictions_{}.parquet",
            "activations": "postfreeze/B22/inference/activations_{}.parquet", "probe": f"postfreeze/B22/probes/seed{SEED}",
            "evaluated": "postfreeze/B22/evaluated/bhat_{}_seed" + str(SEED) + ".parquet",
            "meta": {"network": "F9-M", "probe": "F9b-M", "source": "second source run (postfreeze/B22/)", "completed_seeds": completed}}


def f9b_metric_check(pctx: Any, split: str, bhat: pd.DataFrame) -> dict[str, Any]:
    """analyses.B24.reconstruction_check for F9b: the recorded evaluation metrics of the primary seed
    (``representations.f9b.seeds.<seed>.rho`` of ``probes/evaluations/f9b/<partition>.json``, stage
    evaluate-alt-bundles of the frozen source run) recomputed from ``bhat`` with the metric code that recorded them
    (:func:`tmmri.experiments.l57_stages.rho_block`: the same rows, development reference and bootstrap).  Every key
    must agree and every number within 1e-6, else BundleError (exit 7)."""

    from .. import l57_stages as l57
    from ...stages import BundleError

    rel = EVALUATION_RECORD.format(split)
    seeds = ((pctx.read_json(rel).get("representations") or {}).get("f9b") or {}).get("seeds") or {}
    recorded = (seeds.get(str(SEED)) or {}).get("rho")
    if recorded is None:
        raise BundleError(f"{COMMAND} {ANALYSIS}: {rel} records no F9b metrics of seed {SEED}; the reconstruction check cannot run; refused")
    ds = DATASET[split]
    split_manifest(pctx, split)                                  # every file the metric code reads, hash-checked first
    for rel_input in (f"semantics/B_{ds}.parquet", f"contracts/cohort_{ds}.parquet", f"formal/A_{ds}.parquet"):
        pctx.source_file(rel_input)
    sd, mean = dev_reference(pctx)
    ctx = pctx.stage_ctx
    pdata = l57.partition_data(ctx, split)
    recomputed = json.loads(json.dumps(l57._clean(l57.rho_block(ctx, pdata, bhat, l57._rho_rows(ctx, pdata), mean, sd))))
    diffs = compare_recorded(recorded, recomputed, RECONSTRUCTION_TOLERANCE)
    if diffs:
        raise BundleError(f"{COMMAND} {ANALYSIS}: the recomputed F9b reconstruction on {split} does not reproduce the recorded evaluation metrics "
                          f"of {rel} to {RECONSTRUCTION_TOLERANCE:g} ({'; '.join(diffs[:5])}); refused")
    return {"reproduced": True, "against": f"{rel} representations.f9b.seeds.{SEED}.rho", "metric_code": "tmmri.experiments.l57_stages.rho_block",
            "n_cases": recomputed.get("n_cases"), "tolerance": RECONSTRUCTION_TOLERANCE}


def seed_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[pd.to_numeric(frame["seed"], errors="coerce") == SEED]


def encoder(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    return frame[frame["encoder_id"].astype(str) == name]


def run(pctx: Any) -> dict[str, Any]:
    from ..probes import encoder_matrix
    from ...rule_pipeline import acdc_inner_labels, fit_rulebook, gate_and_apply, qc_route_frame, route_records, rulebook_payload
    from ...stages import _cluster_ids, _dev_matrices, predict_labels
    from ...transition_fit import TransitionModel
    from ...wedd import CutSet

    acfg = pctx.analysis_cfg(ANALYSIS)
    out = pctx.out_dir(ANALYSIS)
    study = pctx.study
    require_registered(pctx, ANALYSIS, [*RULE_LAYER.values(), *LINK.values()])
    stats = pctx.lane_cfg["statistics"]
    min_cases = int(stats["min_cases_per_stratum"])
    rp = dict(stats["random_projections"])
    primary = [str(t) for t in stats["primary_targets"]]
    pcfg = dict(study.formal)
    profile = str(study.wedd["primary_profile"])
    policy = study.rule_policy
    for rel in ("formal/A_ACDC.parquet", "semantics/B_ACDC.parquet", "contracts/split_manifest.parquet", "contracts/inner_folds.parquet",
                "contracts/cohort_ACDC.parquet"):
        pctx.source_file(rel)
    ids_frame, _, b_rows, folds_all = _dev_matrices(pctx.stage_ctx)
    row_ids = ids_frame["sample_id"].astype(str).to_numpy()
    fold_ids = sorted({int(v) for v in folds_all})
    inner = pd.read_parquet(pctx.source_root / "contracts" / "inner_folds.parquet")
    manifest = split_manifest(pctx, "acdc_dev")
    dev_members = manifest.loc[manifest["partition"].astype(str) == "acdc_dev", "sample_id"].astype(str).tolist()
    b_all = pctx.read_parquet("semantics/B_ACDC.parquet").set_index("sample_id")
    mean_oof = pctx.read_parquet("transitions/comparator_mean.parquet").set_index("sample_id")
    model = TransitionModel.from_dict(pctx.read_json("transitions/transition_model.json"))
    dev = load_partition(pctx, "acdc_dev")
    targets = study.transition_targets()
    sources, fitted, frozen, files, concept_rows, ledger_rows, ledger_summary, counts = {}, {}, {}, [], [], [], {}, []
    # 1. Rule layer and decision link of each network, fitted on acdc_dev and frozen before any evaluated partition.
    for key in NETWORKS:
        src = sources[key] = network_source(pctx, key)
        pred_dev = seed_rows(src["read"](src["predictions"].format("acdc_dev")))
        act_dev = seed_rows(src["read"](src["activations"].format("acdc_dev")))
        columns = sorted(c for c in act_dev.columns if ACTIVATION.fullmatch(str(c)))
        oof_dec, oof_mismatch = argmax_decisions(pred_dev[pred_dev["is_oof"].astype(bool)])
        final_dev, final_mismatch = argmax_decisions(encoder(pred_dev, "final"))
        fold_dec = {k: argmax_decisions(encoder(pred_dev, f"fold{k}"))[0] for k in fold_ids}
        probe_oof = src["read"](f"{src['probe']}/bhat_oof_ACDC_dev.parquet").set_index("sample_id")
        probe_final = TransitionModel.from_dict(src["read_json"](f"{src['probe']}/transition_final.json"))
        candidates = [c for c in study.rule_candidates() if c in probe_oof.columns]
        keep = pd.Index(row_ids).isin([s for s in probe_oof.index if oof_dec.get(s) is not None])
        subset = np.flatnonzero(keep)
        gate_ids = row_ids[subset]
        gate_folds = folds_all[subset]
        substrate = ProbeGateSubstrate({k: encoder_matrix(act_dev, f"fold{k}", list(row_ids), columns) for k in fold_ids},
                                       B=b_rows, targets=targets, transition_cfg=study.transition, registry=study.registry, folds=folds_all,
                                       inner_labels=acdc_inner_labels(inner, row_ids), heldout=probe_oof, columns=candidates, row_ids=row_ids,
                                       subset=subset)

        def decision_fn(train: np.ndarray, target: np.ndarray, _folds: np.ndarray = gate_folds, _ids: np.ndarray = gate_ids,
                        _dec: dict = fold_dec) -> np.ndarray:
            held = sorted(set(fold_ids) - {int(v) for v in _folds[np.asarray(train, dtype=int)]})
            if len(held) != 1:
                raise ValueError(f"{ANALYSIS}: the training rows of a gate fold must miss exactly one outer fold, got {held}")
            return _dec[held[0]].reindex(_ids[np.asarray(target, dtype=int)]).to_numpy(dtype=object)

        anchors = pd.Series([s.split(":", 1)[1] for s in gate_ids], index=gate_ids)
        values = probe_oof.loc[gate_ids, candidates]
        decision = oof_dec.loc[gate_ids].astype(str)
        fit = fit_rulebook(rulebook_id=RULEBOOK, values=values, decision=decision, anchors=anchors, folds=gate_folds, profile=profile, study=study,
                           decision_fn=decision_fn, substrate_fn=substrate, reference=b_all.loc[gate_ids], oof=probe_oof.loc[gate_ids],
                           mean_oof=mean_oof.reindex(gate_ids), availability=b_all.loc[[s for s in dev_members if s in b_all.index]], n_jobs=pctx.jobs,
                           clusters=_cluster_ids(manifest, gate_ids))
        provenance = {"analysis": ANALYSIS, "network": src["meta"]["network"], "seed": SEED, "consequent": "network decision (argmax)",
                      "concepts": f"{PROBE[key]} probe reconstruction", "gate_substrate": "nested_inner_oof of the probe (encoder f, inner folds of fold f)"}
        name, payload = rulebook_payload(RULEBOOK, fit["gated"], fit["gate"], provenance)
        mdir = out / "models" / RULE_LAYER[key]
        mdir.mkdir(parents=True, exist_ok=True)
        written = [write_json(fit["cuts"].to_dict(), mdir / f"cuts_{profile}.json", pctx), write_table(fit["stability"], mdir / "stability.parquet", pctx),
                   write_table(fit["eligibility"], mdir / "eligibility.parquet", pctx), write_table(fit["gate"], mdir / "gate.parquet", pctx),
                   write_json(payload, mdir / name, pctx)]
        frozen[RULE_LAYER[key]] = freeze(pctx, mdir, written, {"estimator_id": RULE_LAYER[key], "fit_partition": "acdc_dev", "rulebook_file": name,
                                                              "n_rows": int(len(gate_ids)), "attributes": list(fit["attributes"]),
                                                              "n_eligible_rules": len(payload.get("records", [])), **provenance})
        files += [*written, mdir / "FROZEN.json"]
        for row in fit["eligibility"].to_dict("records"):
            concept_rows.append({"network": key, "probe": PROBE[key], **row, "in_discovery_attributes": row["feature"] in fit["attributes"]})

        def gate_fn(book: Any, _values: pd.DataFrame = values, _fn: Callable = decision_fn, _folds: np.ndarray = gate_folds, _anchors: pd.Series = anchors,
                    _sub: Any = substrate, _attrs: list = list(fit["attributes"])) -> pd.DataFrame:
            return gate_and_apply(values=_values, decision_fn=_fn, folds=_folds, rule_policy=policy, book=book, profile=profile, wedd_cfg=study.wedd,
                                  anchors=_anchors.to_numpy(dtype=object), substrate_fn=_sub, registry=study.registry, rulebook_id=RULEBOOK,
                                  attributes=_attrs)[0]

        table, summary = ledger(rulebook=key, states=fit["states"], decision=decision, attributes=list(fit["attributes"]), book=fit["induced"],
                                gate_table=fit["gate"], gate_fn=gate_fn, policy=policy, registry=study.registry, cuts=fit["cuts"],
                                stability=fit["stability"], agreement_reference="the fold-specific network (outer-fold network trained without the held-out fold)")
        ledger_rows.append(table)
        ledger_summary[key] = summary
        counts.append(pd.DataFrame(summary["rejection_counts"]))
        # Decision link: the network's final decisions on acdc_dev from the probe's seven primary targets and the activations.
        y_dev = oof_dec.reindex(dev.ids).astype(object).to_numpy()   # the consequent (out of fold), not the in-sample final decisions
        x_probe, untransformed = design(probe_oof.reindex(dev.ids), primary, model.transforms, measured=False)
        act = encoder_matrix(act_dev, "final", dev.ids, columns)
        scaler = standardizer(act)
        z = standardize(act, scaler)
        pca = principal_components(z, N_COMPONENTS)
        projections = random_projections(int(rp["seed"]), int(rp["draws"]), len(columns), N_COMPONENTS)
        ldir = out / "models" / LINK[key]
        ldir.mkdir(parents=True, exist_ok=True)
        written = [write_predictors([fit_link(x_probe, y_dev, pcfg)], ldir / "predictor_probe7.json", pctx),
                   write_json({"columns": columns, "standardizer": scaler, "pca": pca, "encoder": "final"}, ldir / "pca.json", pctx),
                   write_predictors([fit_link(pca_scores(z, pca), y_dev, pcfg)], ldir / "predictor_pca7.json", pctx),
                   write_array(projections, ldir / "projections.npy", pctx),
                   write_predictors([fit_link(z @ r, y_dev, pcfg) for r in projections], ldir / "predictors_random7.json", pctx)]
        frozen[LINK[key]] = freeze(pctx, ldir, written, {"estimator_id": LINK[key], "fit_partition": "acdc_dev",
                                                        "target": "out-of-fold network decisions (the consequent); evaluated against the final network",
                                                        "untransformed_columns": untransformed, "draws": int(rp["draws"]), "seed": int(rp["seed"]),
                                                        "activations": f"final encoder, {len(columns)} columns"})
        files += [*written, ldir / "FROZEN.json"]
        fitted[key] = {"columns": columns, "candidates": candidates, "probe_final": probe_final, "cuts": fit["cuts"], "records": payload.get("records", []),
                       "rulebook_file": name, "decision_mismatch_y_pred": {"acdc_dev_oof": oof_mismatch, "acdc_dev_final": final_mismatch},
                       "n_gate_rows": int(len(gate_ids)), "attributes": list(fit["attributes"])}
    # 2. Read the frozen objects back after their hash check.
    for estimator, record in frozen.items():
        check_frozen(out / "models" / estimator, record)
    for key in NETWORKS:
        mdir, ldir = out / "models" / RULE_LAYER[key], out / "models" / LINK[key]
        fitted[key]["cuts"] = CutSet.from_dict(json.loads((mdir / f"cuts_{profile}.json").read_text(encoding="utf-8")))
        fitted[key]["records"] = list(json.loads((mdir / fitted[key]["rulebook_file"]).read_text(encoding="utf-8")).get("records", []))
        fitted[key]["link"] = {"probe7": read_predictors(ldir / "predictor_probe7.json")[0], "pca7": read_predictors(ldir / "predictor_pca7.json")[0],
                               "random7": read_predictors(ldir / "predictors_random7.json"), "projections": np.load(ldir / "projections.npy"),
                               **json.loads((ldir / "pca.json").read_text(encoding="utf-8"))}
    # 3. The recomputed reconstructions of the evaluated partitions must reproduce the recorded ones (analyses.B24
    #    reconstruction_check) before any frozen object is applied: F9b-M its B22 B-hat, F9b the recorded F9b metrics.
    splits, skipped = included_partitions(pctx, list(EVALUATED), f"{COMMAND} {ANALYSIS}")
    recon: dict[tuple[str, str], pd.DataFrame] = {}
    recon_checks: dict[tuple[str, str], dict[str, Any]] = {}
    for split in splits:
        for key in NETWORKS:
            src = sources[key]
            act = encoder(seed_rows(src["read"](src["activations"].format(split))), "final")
            if "partition" in act.columns:
                act = act[act["partition"].astype(str) == split]
            ids_all = act["sample_id"].astype(str).tolist()
            raw_all = fitted[key]["probe_final"].predict(encoder_matrix(act, "final", ids_all, fitted[key]["columns"]),
                                                         index=pd.Index(ids_all, name="sample_id"))
            if src["evaluated"] is not None:
                recorded = src["read"](src["evaluated"].format(split)).set_index("sample_id")
                recon_checks[(split, key)] = {"against": f"{src['evaluated'].format(split)} (second source run)",
                                              **check_reconstruction(raw_all, recorded, f"{ANALYSIS} {PROBE[key]} on {split}")}
            else:
                recon_checks[(split, key)] = f9b_metric_check(pctx, split, raw_all)
            recon[(split, key)] = raw_all
    # 4. Each evaluated partition once per frozen object.
    coverage_rows, link_rows, draw_rows, checks = [], [], [], []
    note = str(acfg.get("note") or "")
    for split in splits:
        members = split_manifest(pctx, split)
        members = set(members.loc[members["partition"].astype(str) == split, "sample_id"].astype(str))
        b_p = pctx.read_parquet(f"semantics/B_{DATASET[split]}.parquet").set_index("sample_id")
        reference = reference_labels(pctx, split)
        link_members = list(load_partition(pctx, split).ids)      # the decision link's cases: critical QC excluded (as B18)
        for key in NETWORKS:
            src, fx = sources[key], fitted[key]
            pred = encoder(seed_rows(src["read"](src["predictions"].format(split))), "final")
            dec, mismatch = argmax_decisions(pred)
            ids = [s for s in dec.index if s in members and s in b_p.index]
            dec = dec.reindex(ids)
            x = encoder_matrix(encoder(seed_rows(src["read"](src["activations"].format(split))), "final"), "final", ids, fx["columns"])
            index = pd.Index(ids, name="sample_id")
            raw = recon[(split, key)].reindex(ids)
            checks.append({"network": key, "partition": split, "decision_mismatch_y_pred": mismatch, "reconstruction_check": recon_checks[(split, key)]})
            # Rule layer: decoders of the final probe, the frozen cuts and the routing of Section 11.7.
            decoded, states = fx["probe_final"].decode(x, dict(study.registry), dict(study.transition), index=index)
            substrate = decoded[[c for c in fx["candidates"] if c in decoded.columns]].copy()
            for c in list(substrate.columns):
                substrate[f"{c}__state"] = states[c].astype(object).to_numpy() if c in states.columns else "ESTIMATED"
            qc = qc_route_frame(b_p.reindex(ids), np.isfinite(x).all(axis=1), study)
            routes = route_records(fx["records"], RULEBOOK, fx["cuts"], substrate, qc, study)
            label, processed = route_labels(routes)
            m = system_metrics(label, processed, dec, reference.reindex(ids))
            extra_note = note if key == "f9m" and split == "mms2_external" else None
            coverage_rows.append({"network": key, "probe": PROBE[key], "seed": SEED, "partition": split, "n_rules": len(fx["records"]), **_flat(m),
                                  "fidelity_reference": "the network's final decision", "per_object_stability": NO_PER_OBJECT_STABILITY,
                                  "note": extra_note, "status": "EVALUATED"})
            # Decision link on the partition against the final network's decisions; critical-QC cases excluded, as B18.
            link = fx["link"]
            in_link = np.isin(ids, link_members)
            lids = [s for s, keep in zip(ids, in_link) if keep]
            dec_l, x_l = dec.reindex(lids), x[in_link]
            checks[-1].update({"n_rule_cases": len(ids), "n_link_cases": len(lids)})
            clusters = _cluster_ids(split_manifest(pctx, split), lids)
            x_probe, _ = design(raw.reindex(lids), primary, model.transforms, measured=False)
            z = standardize(x_l, link["standardizer"])
            preds = {"probe7": pd.Series(predict_labels(link["probe7"], x_probe), index=lids, dtype=object),
                     "pca7": pd.Series(predict_labels(link["pca7"], pca_scores(z, link["pca"])), index=lids, dtype=object)}
            for name_ in ("probe7", "pca7"):
                lm = label_metrics(preds[name_], dec_l, reference, clusters, pctx.boot, min_cases)
                link_rows.append({"network": key, "partition": split, "input": name_, "estimator_id": LINK[key], "n_cases": len(lids), **flat(lm),
                                  "note": extra_note})
            fids = []
            for i, (mdl, r) in enumerate(zip(link["random7"], link["projections"])):
                k, n = agreement(pd.Series(predict_labels(mdl, z @ r), index=lids, dtype=object), dec_l)
                fids.append(k / n if n else np.nan)
                draw_rows.append({"network": key, "partition": split, "null": "random7", "draw": i, "fidelity_k": k, "fidelity_n": n,
                                  "fidelity": k / n if n else None})
            s = null_summary(fids)
            link_rows.append({"network": key, "partition": split, "input": "random7", "estimator_id": LINK[key], "n_cases": len(lids),
                              "null_median": s["median"], "null_p95": s["p95"], "null_n_draws": s["n_draws"], "note": extra_note})
            delta = paired_difference(preds["probe7"], preds["pca7"], dec_l, clusters, pctx.boot, min_cases)
            link_rows.append({"network": key, "partition": split, "input": "probe7_minus_pca7", "estimator_id": LINK[key], "n_cases": len(lids),
                              "difference": delta["estimate"], "difference_ci_lower": delta["ci_lower"], "difference_ci_upper": delta["ci_upper"],
                              "difference_status": delta["status"], "n_common": delta["n_common"], "note": extra_note})
    concept = pd.DataFrame(concept_rows)
    ledger_frame = pd.concat([t for t in ledger_rows if len(t)], ignore_index=True) if any(len(t) for t in ledger_rows) else pd.DataFrame()
    count_frame = pd.concat([c for c in counts if len(c)], ignore_index=True) if any(len(c) for c in counts) else \
        pd.DataFrame(columns=["rulebook", "first_failed_condition", "n_candidates"])
    paths = {"concepts": out / "concept_table.parquet", "ledger": out / "candidate_ledger.parquet", "counts": out / "rejection_counts_by_reason.parquet",
             "coverage": out / "coverage_fidelity_table.parquet", "link": out / "decision_link_table.parquet", "nulls": out / "decision_link_nulls.parquet",
             "summary": out / "summary.json"}
    write_table(concept, paths["concepts"], pctx)
    write_table(ledger_frame, paths["ledger"], pctx)
    write_table(count_frame, paths["counts"], pctx)
    write_table(pd.DataFrame(coverage_rows), paths["coverage"], pctx)
    write_table(pd.DataFrame(link_rows), paths["link"], pctx)
    write_table(pd.DataFrame(draw_rows, columns=["network", "partition", "null", "draw", "fidelity_k", "fidelity_n", "fidelity"]), paths["nulls"], pctx)
    summary = clean({"analysis": ANALYSIS, "networks": {k: {**sources[k]["meta"], "seed": SEED, "probe": PROBE[k], "n_gate_rows": fitted[k]["n_gate_rows"],
                                                           "attributes": fitted[k]["attributes"], "n_eligible_rules": len(fitted[k]["records"]),
                                                           "rulebook_file": fitted[k]["rulebook_file"],
                                                           "decision_mismatch_y_pred": fitted[k]["decision_mismatch_y_pred"]} for k in NETWORKS},
                     "consequent": acfg["consequent"], "concepts": acfg["concepts"], "rule_layer": acfg["rule_layer"], "application": acfg["application"],
                     "decision_link": acfg["decision_link"], "per_object_stability": NO_PER_OBJECT_STABILITY, "ledger": ledger_summary, "frozen": frozen,
                     "checks": checks, "included": splits, "not_executed": skipped, "note": note,
                     "coverage": json.loads(pd.DataFrame(coverage_rows).to_json(orient="records")) if coverage_rows else []})
    write_json(summary, paths["summary"], pctx)
    pctx.write_provenance(ANALYSIS, dict(pctx.verified_log), [*files, *paths.values()],
                          {"included": splits, "not_executed": skipped, "n_eligible_rules": {k: len(fitted[k]["records"]) for k in NETWORKS},
                           "n_candidates": {k: ledger_summary[k]["n_candidates"] for k in NETWORKS}})
    return {"included": splits, "not_executed": skipped, "n_eligible_rules": {k: len(fitted[k]["records"]) for k in NETWORKS}}
