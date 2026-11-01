"""B20: real processed cases for the worked example (EPC-A4; ``configs/exploratory_postfreeze.yaml`` analyses.B20).

The five ACDC test scans already shown in the worked-example figure: the slot order is the case order of
``manuscript/article/figures/paper1/spec/figure02.yaml`` (``example_<n>``: the first case is slot a, the next four b1-b4)
and each ``example_<n>`` names its ACDC test folder in ``schematic_scans/provenance.json``.  Every slot must be an
acdc_test case of the source run (never an M&Ms case).  Every field is read from the recorded run: measured B, raw and
decoded B-hat, the bins of the decoded B-hat and of B under the frozen ``d_model`` cuts, the per-object stability flags,
the audited class probabilities, acceptance at tau_50 / tau_80, the S2 descriptor score and flag, ``d_model``, the
reference label, the routes of the three rulebooks and the F8 prediction.
Outputs are keyed by figure slot only; the slot-to-case mapping is written once, to ``postfreeze/B20/restricted/``
of the new run root (owner exception to manuscript evidence rule 06, 27 October 2026), and never to a table, the summary
or the provenance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import clean, reference_labels, write_json, write_table

ANALYSIS = "B20"
SLOTS = ("a", "b1", "b2", "b3", "b4")
SPEC = "manuscript/article/figures/paper1/spec/figure02.yaml"
SCANS = "manuscript/article/figures/paper1/schematic_scans/provenance.json"
SPLIT = "acdc_test"


def slot_cases(repo: Path) -> dict[str, str]:
    """``{slot: sample_id}`` from the figure specification and the scan provenance of the repository."""

    import yaml

    from ...ingest.sources import sample_id

    spec = yaml.safe_load((repo / SPEC).read_text(encoding="utf-8"))
    scans = json.loads((repo / SCANS).read_text(encoding="utf-8"))
    if "ACDC" not in str(scans.get("dataset", "")):
        raise ValueError(f"{ANALYSIS}: {SCANS} must describe ACDC test scans only")
    keys = [str(c["key"]) for c in spec["cases"]]
    if len(keys) != len(SLOTS):
        raise ValueError(f"{ANALYSIS}: {SPEC} must list {len(SLOTS)} cases, not {len(keys)}")
    folders = {f"example_{int(e['example'])}": str(e["acdc_folder"]) for e in scans["examples"]}
    out = {}
    for slot, key in zip(SLOTS, keys):
        folder = folders.get(key)
        if folder is None or "/testing/" not in folder.replace("\\", "/"):
            raise ValueError(f"{ANALYSIS}: slot {slot} ({key}) has no ACDC testing folder in {SCANS}")
        out[slot] = sample_id("ACDC", Path(folder.replace("\\", "/")).name)
    if len(set(out.values())) != len(out):
        raise ValueError(f"{ANALYSIS}: the figure slots must name five distinct cases")
    return out


def run(pctx: Any) -> dict[str, Any]:
    from ...paths import PROJECT_ROOT
    from ...wedd import CutSet
    from ..exploratory.common import observed
    from ..trust.selective import max_softmax

    acfg = pctx.analysis_cfg(ANALYSIS)
    out = pctx.out_dir(ANALYSIS)
    study = pctx.study
    reason = pctx.partition_status(SPLIT, f"run-postfreeze {ANALYSIS}")
    if reason is not None:
        raise ValueError(f"{ANALYSIS}: {SPLIT} is not available ({reason})")
    repo = Path(PROJECT_ROOT)
    slots = slot_cases(repo)
    manifest = pctx.read_parquet("contracts/split_manifest.parquet").set_index("sample_id")
    if any(s not in manifest.index or str(manifest.at[s, "partition"]) != SPLIT for s in slots.values()):
        raise ValueError(f"{ANALYSIS}: every figure slot must be an {SPLIT} case of the source run")
    ids = list(slots.values())
    fields = [str(f) for f in acfg["fields"]]
    targets = list(dict.fromkeys([*[str(t) for t in pctx.lane_cfg["statistics"]["primary_targets"]], *study.rule_candidates()]))
    concepts = list(study.rule_candidates())
    profile = str(study.wedd["primary_profile"])
    b = pctx.read_parquet("semantics/B_ACDC.parquet").set_index("sample_id").reindex(ids)
    raw = pctx.read_parquet(f"reconstructed/bhat_{SPLIT}.parquet").set_index("sample_id").reindex(ids)
    decoded = pctx.read_parquet(f"reconstructed/bhat_{SPLIT}_decoded.parquet").set_index("sample_id").reindex(ids)
    stability = pctx.read_parquet(f"reconstructed/stability_d_model_{SPLIT}.parquet").set_index("sample_id").reindex(ids)
    cuts = CutSet.from_dict(pctx.read_json(f"thresholds/d_model_{profile}.json"))
    frame = pd.DataFrame(index=pd.Index(ids))
    for t in targets:
        frame[f"measured_b__{t}"] = observed(b, t) if t in b.columns else np.nan
        frame[f"bhat_raw__{t}"] = pd.to_numeric(raw[t], errors="coerce") if t in raw.columns else np.nan
        frame[f"bhat_decoded__{t}"] = pd.to_numeric(decoded[t], errors="coerce") if t in decoded.columns else np.nan
        frame[f"bhat_decoded_state__{t}"] = decoded[f"{t}__state"].astype(object) if f"{t}__state" in decoded.columns else None
    bins_hat = cuts.transform(decoded.reindex(columns=[c for c in decoded.columns if c.replace("__state", "") in cuts.order]))
    bins_b = cuts.transform(pd.DataFrame({c: observed(b, c) for c in cuts.order}, index=b.index))
    for c in concepts:
        if c in cuts.order:
            frame[f"bin_bhat_decoded__{c}"] = bins_hat[c].astype(object)
            frame[f"bin_measured_b__{c}"] = bins_b[c].astype(object)
        frame[f"stability__{c}"] = stability[c].astype(object) if c in stability.columns else None
    not_recorded = []
    rel = f"inference/audited_predictor/{SPLIT}/predictions.parquet"
    if pctx.exists(rel):
        pred = pctx.read_parquet(rel).set_index("sample_id").reindex(ids)
        prob_cols = [c for c in pred.columns if str(c).startswith("prob_")]
        for c in prob_cols:
            frame[c] = pd.to_numeric(pred[c], errors="coerce")
        conf = pd.Series(max_softmax(pred[prob_cols].to_numpy(dtype=float)), index=ids)
        tau = ((pctx.read_json("trust/s1_thresholds.json").get("predictors") or {}).get("audited_predictor") or {}) \
            if pctx.exists("trust/s1_thresholds.json") else {}
        for name in ("tau50", "tau80"):
            t = (tau.get(name) or {}).get("tau")
            frame[f"accept_{name}"] = [None if t is None or not np.isfinite(v) else bool(v >= float(t)) for v in conf]
            if t is None:
                not_recorded.append(f"accept_{name}")
    else:
        not_recorded += ["class_probabilities", "accept_tau50", "accept_tau80"]
    rel = f"trust/{SPLIT}/s2_scores.parquet"
    if pctx.exists(rel):
        s2 = pctx.read_parquet(rel)
        s2 = s2[s2["representation"].astype(str) == "descriptors"].set_index("sample_id").reindex(ids)
        frame["ood_score"] = pd.to_numeric(s2["score"], errors="coerce")
        frame["ood_flag"] = s2["flagged"].astype(object)
    else:
        not_recorded += ["ood_score", "ood_flag"]
    frame["d_model_decision"] = pctx.read_parquet(f"inference/d_model_{SPLIT}.parquet").set_index("sample_id")["d_model"].reindex(ids).astype(object)
    frame["reference_label"] = reference_labels(pctx, SPLIT).reindex(ids).astype(object)
    for rb in ("d_model", "dataset_label", "measured_oracle"):
        routes = pctx.read_parquet(f"inference/routes_{rb}_{SPLIT}.parquet").set_index("sample_id").reindex(ids)
        for col in ("route", "reason_code", "label"):
            frame[f"route_{rb}__{col}"] = routes[col].astype(object) if col in routes.columns else None
    rel = f"comparators/cart_f8/evaluation/predictions_{SPLIT}.parquet"
    if pctx.exists(rel):
        f8 = pctx.read_parquet(rel).set_index("sample_id").reindex(ids)
        for col in ("route_full", "label_full", "route_selective", "label_selective"):
            frame[f"f8__{col}"] = f8[col].astype(object)
    else:
        not_recorded.append("f8_prediction")
    frame.insert(0, "slot", list(slots))
    frame = frame.reset_index(drop=True)
    restricted = out / "restricted"
    restricted.mkdir(parents=True, exist_ok=True)
    paths = {"table": out / "case_slots.parquet", "summary": out / "summary.json"}
    slot_map = restricted / "slot_map.json"
    write_table(frame, paths["table"], pctx)
    summary = clean({"analysis": ANALYSIS, "cases": acfg["cases"], "identifiers": acfg["identifiers"], "fields": fields, "slots": list(SLOTS),
                     "partition": SPLIT, "slot_order": f"{SPEC} cases (first = a, then b1-b4)", "not_recorded": not_recorded,
                     "columns": list(frame.columns), "cuts": f"thresholds/d_model_{profile}.json"})
    text = json.dumps(summary) + frame.to_json()
    if any(s in text or s.split(":", 1)[1] in text for s in ids):
        raise RuntimeError(f"{ANALYSIS}: a case identifier would leave the restricted slot map")
    write_json(summary, paths["summary"], pctx)
    write_json({"schema": "tmmri.postfreeze_slot_map/1", "restricted": True, "note": acfg["identifiers"], "slots": slots}, slot_map, pctx)
    pctx.write_provenance(ANALYSIS, {**dict(pctx.verified_log), "repo:figure_spec": repo / SPEC, "repo:scan_provenance": repo / SCANS},
                          [*paths.values(), slot_map], {"partition": SPLIT, "slots": list(SLOTS), "not_recorded": not_recorded})
    return {"slots": list(SLOTS), "not_recorded": not_recorded}
