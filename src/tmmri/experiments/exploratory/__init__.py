"""Exploratory register B1-B9 of the protocol-v6.0.1 amendment (item 8; B9 added for owner issue O1 of 3 October 2026),
declared post hoc (``configs/exploratory.yaml``).

``run-exploratory --analysis B<k>`` runs one analysis as the stage ``run-exploratory:B<k>`` of a v6 run root:

* B1 shift attribution (:mod:`.shift`), B2 external volume-error decomposition (:mod:`.errors`), B3 vendor-A profile
  (:mod:`.vendor_profile`), B4 cut stability against ``max_bins`` and class separation (:mod:`.cut_stability`), B5 gate
  power on synthetic planted rules (:mod:`.gate_power`), B6 fold variance under alternative fold salts on acdc_dev
  (:mod:`.fold_salts`), B7 segmentation sensitivity on acdc_test (:mod:`.segmentation`), B8 annotation convention of
  M&Ms-1 against M&Ms-2 (:mod:`.annotation`), B9 label-free orientation audit of the image-branch input frame
  (:mod:`.orientation`; staged masks for the LV and RV centroids only).

B5 (synthetic only) and B9 (staged masks and headers only, no fitted object) need no frozen bundle.

Every other analysis requires the frozen, hash-verified v6 bundle and reads recorded artifacts
whose hashes are checked against their producing stage; external partitions are used only when their evaluation is
recorded and their lock allows it (M&Ms: ``protocol-v6.0.0``; M&Ms-2: ``protocol-v6.0.1``).  Outputs go to
``exploratory/<B#>/`` beside a ``provenance.json`` labelled ``EXPLORATORY_POST_HOC``; the stage record carries the same
label.  They explain results and never confirm them: effect sizes with descriptive intervals, no confirmatory p-values.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

from .common import ANALYSES, CONFIG_KEY, LABEL, STAGE, analysis_config, out_dir, provenance

# Configuration sections of the study read by each analysis (beside configs/exploratory.yaml).
SECTIONS: dict[str, tuple[str, ...]] = {
    "B1": ("bootstrap.shift_audit", "cohorts.qc_severity"),
    "B2": ("cohorts.qc_severity", "registry.features"),
    "B3": ("cohorts.qc_severity", "registry.features", "registry.primary_families"),
    "B4": ("wedd", "bootstrap.threshold_bootstrap", "rule_policy.rulebooks", "cohorts.qc_severity"),
    "B5": ("rule_policy", "wedd", "registry.primary_rulebook_antecedent_candidates", "registry.features"),
    "B6": ("splits.acdc.acdc_dev", "transition", "formal.audited_predictor", "wedd", "rule_policy", "bootstrap.threshold_bootstrap", "registry",
           "cohorts.qc_severity"),
    "B7": ("registry", "cohorts", "reason_codes", "transition", "wedd", "rule_policy", "bootstrap.fixed_model_bootstrap", "bootstrap.primary_test"),
    "B8": ("cohorts.qc_severity", "registry.features"),
    "B9": ("cohorts.acdc.label_map", "cohorts.mms.label_map", "cohorts.mms2.label_map"),
}
REQUIRES_FROZEN_BUNDLE = {"B1", "B2", "B3", "B4", "B6", "B7", "B8"}


def analysis_module(analysis: str) -> ModuleType:
    from . import annotation, cut_stability, errors, fold_salts, gate_power, orientation, segmentation, shift, vendor_profile

    return {"B1": shift, "B2": errors, "B3": vendor_profile, "B4": cut_stability, "B5": gate_power, "B6": fold_salts, "B7": segmentation,
            "B8": annotation, "B9": orientation}[analysis]


def run_exploratory(ctx: Any, analysis: str):
    """``run-exploratory --analysis B<k>``: one exploratory analysis as a once-only, content-addressed stage."""

    from ...runs import atomic_write_json
    from ...secondary_stages import run_secondary_stage
    from ...stages import verify_frozen_bundle

    analysis = str(analysis).upper()
    cfg, acfg = analysis_config(ctx.study, analysis)
    run = ctx.require_run(STAGE)
    if analysis in REQUIRES_FROZEN_BUNDLE:
        verify_frozen_bundle(ctx)
    module = analysis_module(analysis)
    plan = module.plan(ctx, cfg, acfg)
    inputs = dict(plan["inputs"])
    if run.frozen_path.is_file():
        inputs["frozen_bundle"] = run.frozen_path
    out = out_dir(run, cfg, analysis)
    outputs = [out / name for name in module.OUTPUTS] + [out / "provenance.json"]

    def compute() -> dict[str, Any]:
        result = module.compute(ctx, cfg, acfg, plan, out)
        atomic_write_json(out / "provenance.json", provenance(ctx, analysis, cfg, inputs, outputs, included=plan.get("included"),
                                                              not_executed=plan.get("not_executed"), result=result), allow_nan=True)
        return {"analysis": analysis, "label": LABEL, **result}

    return run_secondary_stage(ctx, f"{STAGE}:{analysis}", compute, list(inputs.values()), outputs, sections=SECTIONS[analysis],
                               lane_keys=[CONFIG_KEY, *getattr(module, "LANE_KEYS", ())], params={"analysis": analysis, "label": LABEL, "included": sorted(plan.get("included") or {})})


__all__ = ["ANALYSES", "LABEL", "REQUIRES_FROZEN_BUNDLE", "SECTIONS", "analysis_module", "run_exploratory"]
