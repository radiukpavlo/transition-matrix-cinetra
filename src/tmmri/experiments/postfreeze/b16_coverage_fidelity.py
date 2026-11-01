"""B16: coverage and fidelity of the rule systems (EPC-4; ``configs/exploratory_postfreeze.yaml`` analyses.B16).

Systems: the primary ``d_model`` rulebook (empty), the F4 discretizer and WEDD-setting variants, the F5 rule-policy
variants and gate bounds 0.50 and 0.70 (``run-ablations`` rulebooks, ``rules/ablations/``), the companion
``dataset_label`` rulebook (consequents are reference labels) and F8 (full and selective).  Each system is applied with
its own frozen cuts; the F4 variants, whose cuts are not recorded, are refitted on acdc_dev with their recorded settings
(the WEDD cuts are deterministic; no seed or salt enters them).  On the evaluated partitions the rules route the same
substrate as ``evaluate`` (decoders and per-object stability under each system's cuts; the recorded routes of the primary
and companion rulebooks are used, and recomputing them is reported as a reproduction check); on acdc_dev the substrate
is the recorded nested out-of-fold B-hat, on which the rules were induced (development, in-sample for the rule layer).
F8 on acdc_dev uses its recorded nested out-of-fold ledger (full version only).
Metrics: coverage (accepted / processed) and fidelity (agreement with ``d_model`` among accepted), with Wilson
intervals; accuracy against the reference among accepted on the same-definition subsets, beside ``d_model``'s accuracy
on the same rows.  No system is chosen or ranked by an evaluated partition.
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd

from .common import (
    DATASET,
    clean,
    formal_a,
    included_partitions,
    reference_labels,
    write_json,
    write_table,
)
from .estimates import wilson

ANALYSIS = "B16"
ACCEPT = "EXACT_ACCEPT"
QC_ABSTAIN = "QC_ABSTAIN"


def system_metrics(label: pd.Series, processed: pd.Series, d_model: pd.Series, reference: pd.Series) -> dict[str, Any]:
    """Coverage, fidelity and accuracy (Wilson) of accepted labels ``label`` (None: not accepted) on the rows of ``label``."""

    idx = label.index
    lab = label.astype(object).where(label.notna(), None)
    proc = processed.reindex(idx).fillna(False).astype(bool)
    dm = d_model.reindex(idx).astype(object).where(d_model.reindex(idx).notna(), None)
    ref = reference.reindex(idx).astype(object).where(reference.reindex(idx).notna(), None)
    accepted = proc & lab.notna()
    fid = accepted & dm.notna()
    acc = accepted & ref.notna()
    same = acc & dm.notna()
    return {"n_attempted": int(len(idx)), "n_processed": int(proc.sum()), "n_accepted": int(accepted.sum()),
            "coverage": wilson(int(accepted.sum()), int(proc.sum())),
            "fidelity": wilson(int((lab[fid] == dm[fid]).sum()), int(fid.sum())),
            "accuracy": wilson(int((lab[acc] == ref[acc]).sum()), int(acc.sum())),
            "d_model_accuracy_same_rows": wilson(int((dm[same] == ref[same]).sum()), int(same.sum()))}


def route_labels(routes: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    frame = routes.set_index(routes["sample_id"].astype(str))
    route = frame["route"].astype(str)
    return frame["label"].where(route == ACCEPT), route != QC_ABSTAIN


def run(pctx: Any) -> dict[str, Any]:
    from ...formal_features import formal_descriptor_names
    from ...rough_sets import Rulebook, apply_gate
    from ...rule_pipeline import (
        decode_reconstructions,
        qc_route_frame,
        route_records,
        routing_substrate,
        rulebook_payload,
    )
    from ...stages import (
        _dev_matrices,
        _rule_substrate,
        _rulebooks,
        _wedd_cfg_variant,
        fold_model_paths,
        load_fold_models,
        wedd_settings_variants,
    )
    from ...transition_fit import TransitionModel
    from ...wedd import CutSet, fit_cuts

    acfg = pctx.analysis_cfg(ANALYSIS)
    out = pctx.out_dir(ANALYSIS)
    study = pctx.study
    profile = str(study.wedd["primary_profile"])
    books = _rulebooks(pctx.stage_ctx)
    folds_rel = [p.relative_to(pctx.source_root).as_posix() for p in fold_model_paths(pctx.stage_ctx, pctx.stage_ctx.run)]
    for rel in ("oof/bhat_oof_ACDC_dev.parquet", "semantics/B_ACDC.parquet", "formal/A_ACDC.parquet", "formal/d_model_ACDC_dev.parquet",
                "contracts/cohort_ACDC.parquet", "contracts/split_manifest.parquet", "transitions/transition_model.json", *folds_rel):
        pctx.source_file(rel)
    values, decision = _rule_substrate(pctx.stage_ctx, books["d_model"])
    candidates = [c for c in study.rule_candidates() if c in values.columns]
    # The F4 refits of run-ablations: fit_cuts on the gate rows in the order of the acdc_dev design matrix.
    row_ids = _dev_matrices(pctx.stage_ctx)[0]["sample_id"].astype(str).tolist()
    gate_ids = [s for s in row_ids if s in values.index]
    cuts_of = {rb: CutSet.from_dict(pctx.read_json(f"thresholds/{rb}_{profile}.json")) for rb in ("d_model", "dataset_label")}

    def records_of(rb: str) -> list[dict[str, Any]]:
        rel = f"rules/{rb}/rulebook.json"
        return list(pctx.read_json(rel).get("records", [])) if pctx.exists(rel) else []

    def ablation(name: str) -> tuple[list[dict[str, Any]], str]:
        rel = f"rules/ablations/d_model_{name}.json"
        payload = pctx.read_json(rel, "run-ablations")
        return list(payload.get("records", [])), rel

    # Every rule system: (system, variant) -> {records, cuts, rulebook, recorded}.
    systems: dict[tuple[str, str], dict[str, Any]] = {}
    for system in acfg["systems"]:
        if system == "primary_d_model":
            systems[(system, "primary")] = {"records": records_of("d_model"), "cuts": cuts_of["d_model"], "rulebook": "d_model", "recorded": "d_model",
                                            "source": "rules/d_model (EMPTY.json or rulebook.json)", "cuts_source": "recorded"}
        elif system == "companion_dataset_label":
            systems[(system, "companion")] = {"records": records_of("dataset_label"), "cuts": cuts_of["dataset_label"], "rulebook": "dataset_label",
                                              "recorded": "dataset_label", "source": "rules/dataset_label", "cuts_source": "recorded"}
        elif system == "f4_discretizer_variants":
            for prof in study.wedd["profiles"]:
                if prof != profile:
                    recs, rel = ablation(f"discretizer_{prof}")
                    cuts = fit_cuts(values.loc[gate_ids, candidates], decision.loc[gate_ids], prof, study.wedd)
                    systems[(system, prof)] = {"records": recs, "cuts": cuts, "rulebook": "d_model", "source": rel, "cuts_source": "refitted on acdc_dev"}
        elif system == "f4_wedd_setting_variants":
            for name, overrides in wedd_settings_variants(study):
                recs, rel = ablation(f"wedd_settings_{name}")
                cuts = fit_cuts(values.loc[gate_ids, candidates], decision.loc[gate_ids], profile, _wedd_cfg_variant(study.wedd, profile, overrides))
                systems[(system, name)] = {"records": recs, "cuts": cuts, "rulebook": "d_model", "source": rel, "cuts_source": "refitted on acdc_dev"}
        elif system == "f5_rule_policy_variants":
            sens = study.rule_policy["discovery"].get("sensitivities") or {}
            names = [f"beta_{float(bv):.2f}" for bv in sens.get("beta", [])] + (["strict_only"] if sens.get("strict_only") else [])
            names += [f"max_length_{int(m)}" for m in sens.get("max_length", [])]
            for name in names:
                recs, rel = ablation(f"policy_{name}")
                systems[(system, name)] = {"records": recs, "cuts": cuts_of["d_model"], "rulebook": "d_model", "source": rel, "cuts_source": "primary (recorded)"}
        elif system.startswith("f5_gate_bounds_"):
            levels = [float(v) for v in re.findall(r"\d+\.\d+", system)]
            induced = pctx.read_json("rules/d_model/rulebook_induced.json")
            book = Rulebook.from_records(induced["records"], rulebook_id="d_model", consequent_kind="d_model")
            gate = pctx.read_parquet("rules/d_model/gate.parquet")
            for level in levels:
                table = gate.copy()
                if len(table):
                    table["eligibility"] = np.where(table[f"eligible_at_{level:.2f}"].astype(bool), "ELIGIBLE", "INELIGIBLE_CROSSFIT_AGREEMENT")
                _, payload = rulebook_payload("d_model", apply_gate(book, table), table, {"ablation": f"gate_lower_bound:{level:.2f}"})
                systems[(system, f"{level:.2f}")] = {"records": list(payload.get("records", [])), "cuts": cuts_of["d_model"], "rulebook": "d_model",
                                                     "source": "rules/d_model/gate.parquet re-thresholded", "cuts_source": "primary (recorded)"}
        elif system in ("f8_cart_full", "f8_cart_selective"):
            systems[(system, system.rsplit("_", 1)[1])] = {"f8": system.rsplit("_", 1)[1], "source": "comparators/cart_f8"}
        else:
            raise ValueError(f"{ANALYSIS}: unknown system {system!r}")
    # Partitions (acdc_dev_oof is the development report).
    wanted = ["acdc_dev" if p == "acdc_dev_oof" else str(p) for p in acfg["partitions"]]
    splits, skipped = included_partitions(pctx, wanted, f"run-postfreeze {ANALYSIS}")
    model = TransitionModel.from_dict(pctx.read_json("transitions/transition_model.json"))
    rows, checks = [], []
    for split in splits:
        label_p = "acdc_dev_oof" if split == "acdc_dev" else split
        if split == "acdc_dev":
            keep = [str(s) for s in values.index]
            d_model = pctx.read_parquet("formal/d_model_ACDC_dev.parquet").set_index("sample_id")["d_model"].reindex(keep)
            reference = reference_labels(pctx, "acdc_dev").reindex(keep)
            qc = pd.DataFrame({"sample_id": keep, "qc_abstain": False, "qc_reason_code": None})
            substrate_for = {rb: values.loc[keep] for rb in ("d_model", "dataset_label")}
        else:
            ds = DATASET[split]
            keep = [str(s) for s in pctx.read_parquet(f"reconstructed/bhat_{split}.parquet")["sample_id"]]
            a_p = formal_a(pctx, ds).reindex(keep)[formal_descriptor_names()].to_numpy(dtype=float)
            b_p = pctx.read_parquet(f"semantics/B_{ds}.parquet").set_index("sample_id").reindex(keep)
            d_model = pctx.read_parquet(f"inference/d_model_{split}.parquet").set_index("sample_id")["d_model"].reindex(keep)
            reference = reference_labels(pctx, split).reindex(keep)
            qc = qc_route_frame(b_p, np.isfinite(a_p).all(axis=1), study)
            dec, st, fold_dec = decode_reconstructions(model, load_fold_models(pctx.stage_ctx, pctx.stage_ctx.run), a_p, pd.Index(keep, name="sample_id"),
                                                       study.registry, study.transition)
            substrate_for = None
        for (system, variant), spec in systems.items():
            base = {"system": system, "variant": variant, "partition": label_p, "source": spec["source"], "cuts_source": spec.get("cuts_source"),
                    "n_rules": len(spec["records"]) if "records" in spec else None}
            if "f8" in spec:
                if split == "acdc_dev":
                    ledger = pctx.read_parquet("comparators/cart_f8/nested_oof_ledger.parquet").set_index("sample_id")
                    if spec["f8"] == "selective":
                        rows.append({**base, "status": "NE", "reason": "F8 selective leaves are not recorded per outer fold"})
                        continue
                    lab = ledger["fold_label"].astype(object)
                    m = system_metrics(lab, pd.Series(True, index=lab.index), ledger["d_model_oof"], reference_labels(pctx, "acdc_dev").reindex(lab.index))
                    fid_ref = "d_model out of fold (recorded nested ledger)"
                else:
                    pred = pctx.read_parquet(f"comparators/cart_f8/evaluation/predictions_{split}.parquet").set_index("sample_id")
                    lab = pred[f"label_{spec['f8']}"].astype(object)
                    m = system_metrics(lab, pred["route_full"].astype(str) != QC_ABSTAIN, pred["d_model"], pred["reference"])
                    fid_ref = "d_model (recorded F8 evaluation)"
                rows.append({**base, **_flat(m), "fidelity_reference": fid_ref, "status": "EVALUATED"})
                continue
            rb = spec["rulebook"]
            if split == "acdc_dev":
                routes = route_records(spec["records"], rb, spec["cuts"], substrate_for[rb], qc, study)
            else:
                substrate, _ = routing_substrate(dec, st, fold_dec, spec["cuts"], study.wedd, candidates)
                routes = route_records(spec["records"], rb, spec["cuts"], substrate, qc, study)
                if spec.get("recorded"):
                    recorded = pctx.read_parquet(f"inference/routes_{spec['recorded']}_{split}.parquet")
                    same = recorded.set_index("sample_id")[["route"]].astype(str).equals(routes.set_index("sample_id")[["route"]].astype(str))
                    checks.append({"system": system, "partition": label_p, "recomputed_routes_equal_recorded": bool(same)})
                    routes = recorded
            lab, processed = route_labels(routes)
            m = system_metrics(lab, processed, d_model, reference)
            rows.append({**base, **_flat(m), "fidelity_reference": "d_model (final frozen predictor)", "status": "EVALUATED"})
    table = pd.DataFrame(rows)
    figure = table[[c for c in ("system", "variant", "partition", "n_processed", "n_accepted", "coverage", "coverage_ci_lower", "coverage_ci_upper",
                                "fidelity", "fidelity_ci_lower", "fidelity_ci_upper", "status") if c in table.columns]] if len(table) else table
    paths = {"table": out / "coverage_fidelity_table.parquet", "figure": out / "figure_data.parquet", "summary": out / "summary.json"}
    write_table(table, paths["table"], pctx)
    write_table(figure, paths["figure"], pctx)
    labels = [("acdc_dev_oof" if s == "acdc_dev" else s) for s in splits]
    summary = clean({"analysis": ANALYSIS, "application": acfg["application"], "metrics": acfg["metrics"], "included": labels, "not_executed": skipped,
                     "systems": {f"{s}:{v}": {k: spec.get(k) for k in ("source", "cuts_source", "rulebook")} | {"n_rules": len(spec.get("records", []))}
                                 for (s, v), spec in systems.items()},
                     "reproduction_checks": checks, "rows": json.loads(table.to_json(orient="records")) if len(table) else []})
    write_json(summary, paths["summary"], pctx)
    pctx.write_provenance(ANALYSIS, dict(pctx.verified_log), list(paths.values()),
                          {"included": labels, "not_executed": skipped, "n_systems": len(systems), "reproduction_checks": checks})
    return {"included": labels, "not_executed": skipped, "n_systems": len(systems)}


def _flat(m: dict[str, Any]) -> dict[str, Any]:
    out = {k: m[k] for k in ("n_attempted", "n_processed", "n_accepted")}
    for key in ("coverage", "fidelity", "accuracy", "d_model_accuracy_same_rows"):
        out.update({key: m[key]["estimate"], f"{key}_k": m[key]["k"], f"{key}_n": m[key]["n"], f"{key}_ci_lower": m[key]["ci_lower"],
                    f"{key}_ci_upper": m[key]["ci_upper"]})
    return out
