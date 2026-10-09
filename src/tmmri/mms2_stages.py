"""M&Ms-2 stages of the protocol-v6.0.1 amendment (``configs/cohorts.yaml`` ``mms2``; :mod:`tmmri.ingest.mms2`).

The shared stages accept ``--dataset mms2`` / ``--split mms2_external`` (``stage-sources``, ``build-semantics``,
``extract-formal``, ``evaluate``, ``audit-shift``); this module holds the M&Ms-2 parts that have no M&Ms-1
counterpart:

* :func:`extract_mms2` (CLI ``extract-mms2``, host only): archive SHA-256 first, then the lock, then exactly the 720
  ``mms2_external`` short-axis ED/ES files into the read-only source area of ``configs/source_bindings/mms2.json``.
* :func:`validate_cohort_mms2` (``validate-cohort --dataset mms2``; record ``validate-cohort:MMS2``): cohort table,
  the split manifest checked against its rule and the receipt, staging cross-checks, and the decoded-voxel duplicate
  rule against ACDC, M&Ms-1 and within M&Ms-2 (every matching ``mms2_external`` case is excluded with a reason code;
  the link to the never-extracted ``mms2_sealed`` half is recorded for v7).  Writes ``contracts/cohort_MMS2.parquet``
  and ``contracts/split_manifest_MMS2.parquet`` (never part of the frozen v6.0.0 bundle).
* :func:`evaluate_mms2_tail` (called by ``evaluate --split mms2_external``): H-primary-3 as its own one-test family
  (the unchanged v6.0.0 bundle; 95 % cluster-bootstrap lower bound of rho > 0.10), strata (prespecified: pathology
  group with the RV family pooled; descriptive and data-driven: vendor, scanner and field strength, recorded verbatim;
  the release CSV has no sex, age, centre or release split) and the M&Ms-2 secondary family.
* :func:`summarize_predictor_mms2` (``summarize-predictor --split mms2_external``): audited-predictor metrics on the
  same-definition subset (NOR + HCM) and the ``d_model`` distributions of the non-equivalent strata (dilated LV -> DCM;
  RV family -> RV).

Every compute stage obeys :func:`tmmri.stages.require_mms2_unlocked` (exit 3 before the tag); evaluations run once
(``.stages`` records, once-only guard of a frozen run).  Nothing here fits on M&Ms-2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ingest.sources import is_synthetic_fixture, sha256_file
from .runs import atomic_write_json
from .stages import (
    COUNT_CHECK_WAIVER,
    MMS2_EXTRACTION_MANIFEST,
    MMS2_SPLIT,
    SAME_DEFINITION,
    NOT_EXACT,
    STAGE_CONFIG_SECTIONS,
    BundleError,
    CohortValidationError,
    StageContext,
    _check,
    _check_recorded_output,
    _cluster_ids,
    _cohort,
    _contracts,
    _contracts_dir,
    _predictor_probabilities,
    _require_evaluated,
    _rho,
    _split_manifest,
    _stage,
    _staging_manifest,
    _stratum_label,
    _write_table,
    below_threshold_status,
    critical_qc_mask,
    parse_utc,
    reference_labels,
    require_mms2_unlocked,
    verify_frozen_bundle,
)

DS = "MMS2"
MMS2_COUNT_CHECKS = frozenset({"mms2_external_size"})
SPLIT_RULE_VERSION = "mms2-split-v6.0.1"


# ----------------------------------------------------------------------------- extract-mms2


def extract_mms2(ctx: StageContext, *, archive: str | Path | None = None, dest: str | Path | None = None,
                 tar: str | Path | None = None) -> dict[str, Any]:
    """Selective extraction of the ``mms2_external`` files (:func:`tmmri.ingest.mms2.extract_external`).

    Defaults: the archive and extractor of ``configs/source_bindings/mms2.json`` and its source area (host default).
    The receipt is also copied to ``<data-root>/manifests/extraction_MMS2.json`` with ``first_extraction_started_utc``,
    the dated extraction record of SPECIFICATION.md 14.6 (:func:`tmmri.stages.mms2_freeze_order`).
    """

    from .ingest.mms2 import DEFAULT_TAR, archive_expectation, extract_external
    from .ingest.sources import source_binding

    binding = source_binding("mms2")
    archive = Path(archive) if archive else Path(binding["archive"]["reference_copy"])
    dest = Path(dest) if dest else ctx.source_root("mms2")
    tar = tar or binding["archive"].get("extractor") or DEFAULT_TAR
    synthetic = bool(archive_expectation(archive)["synthetic_fixture"])
    result = extract_external(archive, dest, cohort_cfg=ctx.study.cohorts["mms2"], tar=tar,
                              require_unlocked=lambda: require_mms2_unlocked(ctx, "extract-mms2", synthetic=synthetic))
    # SPECIFICATION.md 14.6: the dated extraction record; the earliest extraction start is kept when it is written again.
    manifest = ctx.manifests_dir / MMS2_EXTRACTION_MANIFEST
    starts = [result.get("extraction_started_utc")]
    if manifest.is_file():
        prior = json.loads(manifest.read_text(encoding="utf-8"))
        starts += [prior.get("first_extraction_started_utc"), prior.get("extraction_started_utc")]
    parsed = sorted(t for t in (parse_utc(v) for v in starts if v) if t is not None)
    atomic_write_json(manifest, {**result, "destination": str(dest), "first_extraction_started_utc": parsed[0].isoformat() if parsed else None})
    if ctx.run is not None:
        ctx.run.set_status("extract-mms2", "BUILT", action=result["action"], files=result["list_file"]["entries"])
        ctx.run.log({"stage": "extract-mms2", "action": result["action"], "archive_sha256": result["archive"]["sha256"]})
    return result


# ----------------------------------------------------------------------------- validate-cohort --dataset mms2


def _duplicate_exclusions(table: pd.DataFrame, detected: dict[tuple[str, str], list[str]], codes: dict[str, str]) -> tuple[pd.DataFrame, list[dict]]:
    """Exclude every ``mms2_external`` case that matches ACDC, M&Ms-1 or another M&Ms-2 case (decoded image voxels)."""

    out = table.copy()
    rows: list[dict[str, Any]] = []
    matches: dict[str, list[tuple[str, list[str], str]]] = {}
    for (a, b), frames in detected.items():
        for mine, other in ((a, b), (b, a)):
            if mine.startswith(f"{DS}:"):
                other_ds = other.split(":", 1)[0]
                code = codes["within_mms2" if other_ds == DS else "acdc" if other_ds == "ACDC" else "mms1"]
                matches.setdefault(mine, []).append((other, frames, code))
    by_sample = {str(s): i for i, s in zip(out.index, out["sample_id"])}
    for sample, found in sorted(matches.items()):
        found = sorted(found)
        code = sorted({c for _, _, c in found}, key=lambda c: [codes["acdc"], codes["mms1"], codes["within_mms2"]].index(c))[0]
        link = {"partition": "mms2_sealed", "status": "NOT_CHECKABLE_UNDER_V6", "case": sample,
                "rule": "the v7 ingest applies the same decoded-voxel rule to every mms2_sealed case against this case and its matches"}
        rows.append({"sample_id": sample, "matches": [{"sample_id": o, "frames": f, "reason_code": c} for o, f, c in found],
                     "reason_code": code, "excluded_from": MMS2_SPLIT, "sealed_link": link})
        if sample in by_sample:
            i = by_sample[sample]
            out.at[i, "eligible_v6"] = False
            out.at[i, "mms2_external"] = False
            out.at[i, "exclusion_reason"] = code
            out.at[i, "duplicate_reason_code"] = code
            out.at[i, "duplicate_of"] = ";".join(o for o, _, _ in found)
            out.at[i, "duplicate_frames"] = json.dumps({o: f for o, f, _ in found}, sort_keys=True)
            out.at[i, "sealed_link"] = json.dumps(link, sort_keys=True)
    return out, rows


def build_split_manifest_mms2(table: pd.DataFrame) -> pd.DataFrame:
    """One row per M&Ms-2 cohort row: partition ``mms2_external`` or ``excluded``; every case its own cluster."""

    out = table.assign(release_split=None, partition=np.where(table["mms2_external"].astype(bool), MMS2_SPLIT, "excluded"),
                       stratum=table["vendor"], outer_fold=pd.NA, cluster_id=table["sample_id"],
                       duplicate_pair_id=None, duplicate_excluded_from=np.where(table["duplicate_reason_code"].notna(), MMS2_SPLIT, None))
    columns = ["sample_id", "dataset", "case_id", "patient_id", "release_split", "partition", "stratum", "outer_fold", "split_rank",
               "eligible_v6", "exclusion_reason", "cluster_id", "duplicate_pair_id", "duplicate_excluded_from", "duplicate_reason_code"]
    out = out.reindex(columns=columns)
    out["outer_fold"] = out["outer_fold"].astype("Int64")
    out["split_rule_version"] = SPLIT_RULE_VERSION
    return out.sort_values("sample_id").reset_index(drop=True)


def _validate_mms2_checks(ctx: StageContext, *, strict: bool, waive_counts: bool) -> dict[str, Any]:
    from .ingest import mms2
    from .splits import cluster_counts, detect_image_duplicates

    study = ctx.study
    cfg = study.cohorts["mms2"]
    out_dir = _contracts_dir(ctx)
    checks: list[dict[str, Any]] = []
    try:
        split = mms2.load_split_manifest()
        reproduces = True
    except Exception:     # noqa: BLE001 - recorded as a failed check; the tracked file is then not used
        split, reproduces = mms2.build_split_manifest(), False
    _check(checks, "mms2_split_manifest_reproduces_rule", True, reproduces)
    receipt = mms2.load_receipt()
    _check(checks, "mms2_split_manifest_sha256_matches_receipt", str(receipt.get("split_manifest", {}).get("sha256")),
           mms2.lf_sha256(mms2.SPLIT_MANIFEST) if mms2.SPLIT_MANIFEST.is_file() else None)
    _check(checks, "mms2_split_partition_sizes", {str(k): int(v) for k, v in cfg["partitions"].items()},
           {str(k): int(v) for k, v in split["partition"].value_counts().sort_index().items()})
    root = ctx.source_root("mms2")
    manifest = ctx.source_manifest(DS)
    table = mms2.mms2_cohort_table(root, manifest, cohort_cfg=cfg, split=split)
    sealed = {mms2.case_key(s) for s in mms2.sealed_ids(split)}      # case ids MMS2_<code> (owner decision C)
    staging = _staging_manifest(ctx, DS)
    staged = (staging or {}).get("cases", {})
    seen = set(table["case_id"]) | set(manifest.loc[manifest["role"] != mms2.INFO_ROLE, "case_id"]) | set(staged)
    _check(checks, "mms2_no_sealed_subject", [], sorted(seen & sealed))
    _check(checks, "mms2_external_size", int(cfg["partitions"][MMS2_SPLIT]), int(len(table)))
    _check(checks, "mms2_info_csv_covers_every_case", [], sorted(table.loc[table["pathology"].isna(), "case_id"]))
    _check(checks, "mms2_inputs_complete", [], sorted(table.loc[~table["eligible_v6"], "case_id"]))
    # Decoded-voxel duplicates against ACDC, M&Ms-1 and within M&Ms-2 (staged ED/ES image voxels).
    records, missing = [], []
    for ds in ("ACDC", "MMS", DS):
        recs = (_staging_manifest(ctx, ds) or {}).get("cases")
        if recs is None:
            missing.append(ds)
        records += list((recs or {}).values())
    _check(checks, "mms2_duplicate_reference_cohorts_staged", [], missing)
    detected = {pair: frames for pair, frames in detect_image_duplicates(records).items() if any(s.startswith(f"{DS}:") for s in pair)}
    table, duplicates = _duplicate_exclusions(table, detected, {k: str(v) for k, v in cfg["duplicates"]["reason_codes"].items()})
    _check(checks, "mms2_duplicates_excluded", [], sorted(table.loc[table["duplicate_reason_code"].notna() & table["eligible_v6"], "case_id"]))
    # Cases rejected at staging (for example INPUT_GEOMETRY_INCONSISTENT) leave mms2_external with their reason code.
    failures = {str(k): str(v.get("reason_code")) for k, v in ((staging or {}).get("failures") or {}).items()}
    rejected = table["case_id"].isin(set(failures)) & table["eligible_v6"]
    table.loc[rejected, "exclusion_reason"] = table.loc[rejected, "case_id"].map(failures)
    table.loc[rejected, ["eligible_v6", "mms2_external"]] = False
    _check(checks, "mms2_staging_rejections_excluded", [], sorted(table.loc[table["case_id"].isin(set(failures)) & table["eligible_v6"], "case_id"]))
    expected_ids = sorted(table.loc[table["eligible_v6"], "case_id"])
    _check(checks, "mms2_staged", expected_ids, sorted(c for c in staged if c in set(expected_ids)) if staging is not None else "not_staged")
    tracked = {str(r.relpath): str(r.sha256) for r in manifest.itertuples(index=False)}
    _check(checks, "mms2_staged_sources_match_file_manifest", [],
           sorted(cid for cid, rec in staged.items() for src in rec.get("sources", {}).values() if tracked.get(src["relpath"]) != src["sha256"])[:10])
    allowed = {int(k) for k in cfg["label_map"]}
    _check(checks, "mms2_mask_labels_identity_domain", [],
           sorted(cid for cid, rec in staged.items() if not set(int(v) for v in rec.get("mask_values_released", [])) <= allowed)[:10])
    _check(checks, "mms2_phase_from_file_names", [], sorted(cid for cid, rec in staged.items() if rec.get("phase_source") != "file_name")[:10])
    split_manifest = build_split_manifest_mms2(table)
    ext = split_manifest[split_manifest["partition"] == MMS2_SPLIT]
    clusters = {"mms2_external": cluster_counts(split_manifest, split_manifest["partition"] == MMS2_SPLIT),
                "by_pathology_group": {str(g): int(n) for g, n in table.loc[table["mms2_external"], "pathology_group"].value_counts().sort_index().items()},
                "n_excluded_duplicates": int(table["duplicate_reason_code"].notna().sum()), "n_external_cases": int(len(ext))}
    if waive_counts:
        for c in checks:
            if not c["passed"] and c["check"] in MMS2_COUNT_CHECKS:
                c["waived"] = COUNT_CHECK_WAIVER
    passed = all(c["passed"] or c.get("waived") for c in checks)
    report = {"schema": "tmmri.cohort_validation/1", "dataset": DS, "passed": passed, "contracts_written": passed,
              "checks": checks, "synthetic_count_waiver": bool(waive_counts), "cluster_counts": clusters,
              "outputs": {"cohort_MMS2": "cohort_MMS2.parquet", "split_manifest_MMS2": "split_manifest_MMS2.parquet",
                          "mms2_duplicates": "mms2_duplicates.json"},
              "duplicate_exclusions": [{"sample_id": d["sample_id"], "reason_code": d["reason_code"]} for d in duplicates],
              "staging_rejections": dict(sorted(failures.items())),
              "config_hash": study.sections_hash(STAGE_CONFIG_SECTIONS["validate-cohort:MMS2"])}
    contracts = [out_dir / "cohort_MMS2.parquet", out_dir / "split_manifest_MMS2.parquet", out_dir / "mms2_duplicates.json"]
    if passed:
        _write_table(table, contracts[0])
        _write_table(split_manifest, contracts[1])
        atomic_write_json(contracts[2], {
            "schema": "tmmri.mms2_duplicates/1", "rule": cfg["duplicates"]["rule"], "detection": cfg["duplicates"]["detection"],
            "reference_cohorts": ["ACDC", "MMS", DS], "missing_reference_cohorts": missing, "exclusions": duplicates,
            "sealed_links": [d["sealed_link"] for d in duplicates]})
    else:
        # A failed validation leaves no M&Ms-2 contract (stale copies are removed): the duplicate exclusion against ACDC
        # and M&Ms-1 is a leakage control, so nothing downstream may read an unvalidated cohort.  The report stays for diagnosis.
        for path in contracts:
            path.unlink(missing_ok=True)
    atomic_write_json(out_dir / "cohort_validation_MMS2.json", report)
    if strict and not passed:
        raise CohortValidationError(report)
    return report


def validate_cohort_mms2(ctx: StageContext, *, strict: bool = True) -> dict[str, Any]:
    """``validate-cohort --dataset mms2`` (locked like every M&Ms-2 command before the protocol-v6.0.1 tag, exit 3;
    SPECIFICATION.md 4.6; it computes no endpoint).

    With a run id the outputs go to ``<run>/contracts/`` through the record ``validate-cohort:MMS2`` (not a fitting
    stage, so it may run in a frozen v6.0.0 run; once recorded there it may only be skipped).  With
    ``--allow-unfrozen-synthetic`` and a synthetic M&Ms-2 source, the cohort-size check is waived.

    Strict by default and always strict with a run id: any failed check raises :class:`CohortValidationError` (exit 6),
    writes no contract (``cohort_MMS2.parquet``, ``split_manifest_MMS2.parquet``, ``mms2_duplicates.json``) and no stage
    record, so the cause (for example a reference cohort that is not staged) can be corrected and the validation repeated
    even in a frozen run; only the failed report ``cohort_validation_MMS2.json`` is written.  Every later M&Ms-2 stage
    requires the passed record (:func:`tmmri.stages.require_mms2_validated`).
    """

    from .ingest import mms2
    from .ingest.staging import manifest_path_for

    require_mms2_unlocked(ctx, "validate-cohort --dataset mms2", synthetic=is_synthetic_fixture(ctx.source_root("mms2")))
    waive = bool(ctx.allow_unfrozen_synthetic and is_synthetic_fixture(ctx.source_root("mms2")))
    strict = True if ctx.run is not None else bool(strict)
    if ctx.run is None:
        return _validate_mms2_checks(ctx, strict=strict, waive_counts=waive)
    out_dir = _contracts_dir(ctx)
    outputs = [out_dir / "cohort_validation_MMS2.json", out_dir / "cohort_MMS2.parquet", out_dir / "split_manifest_MMS2.parquet",
               out_dir / "mms2_duplicates.json"]
    inputs = [p for p in (manifest_path_for(ctx.manifests_dir, ds) for ds in ("ACDC", "MMS", DS)) if p.is_file()]
    inputs += [mms2.SPLIT_MANIFEST, mms2.RECEIPT, mms2.file_manifest_path(ctx.source_root("mms2"))]
    binding = Path(ctx.study.root) / "configs" / "source_bindings" / "mms2.json"
    params = {"strict": bool(strict), "synthetic_count_waiver": waive, "source_manifest_sha256": ctx.source_manifest_sha256(DS),
              "source_binding_sha256": sha256_file(binding) if binding.is_file() else None}
    holder: dict[str, Any] = {}

    def compute() -> dict[str, Any]:
        holder["report"] = _validate_mms2_checks(ctx, strict=strict, waive_counts=waive)
        return {"passed": holder["report"]["passed"]}

    _stage(ctx, "validate-cohort:MMS2", compute, inputs, outputs, params)
    return holder.get("report") or json.loads((out_dir / "cohort_validation_MMS2.json").read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- evaluate --split mms2_external


def evaluation_outputs(mdir: Path) -> list[Path]:
    """Outputs of the M&Ms-2 tail of ``evaluate`` beyond the shared ones (B-hat, d_model, rho, per-target, routes)."""

    return [mdir / "h_primary_3.json", mdir / f"strata_errors_{MMS2_SPLIT}.parquet", mdir / f"strata_routes_{MMS2_SPLIT}.parquet",
            mdir / f"strata_rho_{MMS2_SPLIT}.json", mdir / "multiplicity_MMS2.parquet"]


def mms2_strata(ctx: StageContext, cohort: pd.DataFrame, ids: list[str]) -> dict[str, tuple[str, pd.Series]]:
    """Prespecified and descriptive strata (``mms2.strata``).  The release CSV has no sex or age column (header read
    before the tag), so the descriptive demographic strata of configs/bootstrap.yaml are not available for M&Ms-2."""

    cfg = ctx.study.cohorts["mms2"]["strata"]
    sub = cohort.loc[ids]
    out: dict[str, tuple[str, pd.Series]] = {}
    for role, names in (("prespecified", cfg.get("prespecified", [])), ("descriptive_demographic", cfg.get("descriptive", []))):
        for name in names:
            if name not in sub.columns:
                raise KeyError(f"M&Ms-2 stratum {name!r} is not a column of contracts/cohort_MMS2.parquet")
            out[str(name)] = (role, sub[name].map(_stratum_label))
    return out


def evaluate_mms2_tail(ctx: StageContext, *, split: str, b: pd.DataFrame, bhat: pd.DataFrame, keep: list[str], targets: list[str],
                       reference_sd: pd.Series, mean: pd.Series, y_null: pd.DataFrame, clusters: pd.Series,
                       routes_by_rb: dict[str, pd.DataFrame], reference: pd.Series, d_model: pd.Series, rho: dict[str, Any],
                       cohort: pd.DataFrame, mdir: Path, frozen: dict[str, Any]) -> dict[str, Any]:
    """H-primary-3, strata and the M&Ms-2 secondary family (see the module docstring); called inside ``evaluate``."""

    from .metrics import bh_family_table, fixed_sequence_test, paired_error_tests, per_target_errors, stratified_route_metrics

    study = ctx.study
    cfg = study.cohorts["mms2"]
    h3 = dict(cfg["endpoints"]["h_primary_3"])
    family = str(cfg["endpoints"]["secondary_family"])
    q = float(study.bootstrap["multiplicity"]["q"])
    decision = fixed_sequence_test([{**rho, "hypothesis_id": h3["id"], "partition": split, "in_hierarchy": True}], float(h3["threshold"]))[0]
    atomic_write_json(mdir / "h_primary_3.json", {
        "schema": "tmmri.h_primary_3/1", "hypothesis": h3, "family": "own_one_test_family", "result": decision,
        "n_cases": int(len(keep)), "n_clusters": int(clusters.loc[keep].nunique()), "bundle": "unchanged v6.0.0 bundle",
        "bundle_sha256": frozen["bundle_sha256"], "rho_file": f"metrics/primary_rho_{split}.json"}, allow_nan=True)
    min_cases = int(study.bootstrap["subgroup_reporting_thresholds"]["min_cases_per_stratum"])
    status = below_threshold_status(study)
    strata = mms2_strata(ctx, cohort, keep)
    error_rows, route_rows, rho_strata = [], [], {}
    tests = [paired_error_tests(b.loc[keep, targets], bhat[targets], y_null, targets, reference_sd, family=family, partition=split,
                                comparison="mean_vs_ridge_per_target", cluster=clusters.loc[keep].to_numpy(dtype=object))]
    for var, (role, series) in strata.items():
        descriptive = role != "prespecified"
        evaluated = "DESCRIPTIVE" if descriptive else "EVALUATED"
        rho_strata[var] = {}
        for value in sorted(series.unique()):
            rows = [s for s in keep if series.loc[s] == value]
            below = len(rows) < min_cases
            e = per_target_errors(b.loc[rows, targets], bhat.loc[rows, targets], reference_sd)
            if below or descriptive:
                e.loc[:, [c for c in ("pearson_p", "spearman_p") if c in e.columns]] = np.nan
            e.insert(0, "stratum", value)
            e.insert(0, "stratum_variable", var)
            e["stratum_role"] = role
            e["stratum_n"] = len(rows)
            e["stratum_clusters"] = int(clusters.loc[rows].nunique())
            e["descriptive_only"] = below or descriptive
            e["inference_status"] = status["status"] if below else evaluated
            e["inference_reason"] = status["reason"] if below else None
            error_rows.append(e)
            rho_strata[var][value] = {**_rho(ctx, b.loc[rows, targets], bhat.loc[rows], mean, reference_sd, min_cases=min_cases, clusters=clusters),
                                      "stratum_role": role}
            if not descriptive:
                tests.append(paired_error_tests(b.loc[rows, targets], bhat.loc[rows, targets], y_null.loc[rows], targets, reference_sd,
                                                family=family, partition=split, comparison="mean_vs_ridge_per_target", stratum_variable=var,
                                                stratum=value, min_cases=min_cases, cluster=clusters.loc[rows].to_numpy(dtype=object)))
        for rb, routes in routes_by_rb.items():
            r = stratified_route_metrics(routes, reference.to_numpy(), d_model.to_numpy(), series.loc[keep].to_numpy(), min_cases=min_cases,
                                         cluster=clusters.loc[keep].to_numpy(dtype=object))
            r.insert(0, "stratum_variable", var)
            r.insert(0, "rulebook", rb)
            r["stratum_role"] = role
            r["routes"] = [json.dumps(v, sort_keys=True) for v in r["routes"]]
            r["inference_status"] = np.where(r["descriptive_only"], status["status"], evaluated)
            r["inference_reason"] = [status["reason"] if d else None for d in r["descriptive_only"]]
            route_rows.append(r)
    _write_table(pd.concat(error_rows, ignore_index=True), mdir / f"strata_errors_{split}.parquet", csv=True)
    _write_table(pd.concat(route_rows, ignore_index=True), mdir / f"strata_routes_{split}.parquet", csv=True)
    atomic_write_json(mdir / f"strata_rho_{split}.json", {"min_cases_per_stratum": min_cases, "below_threshold": status,
                                                          "descriptive_strata": [k for k, (role, _) in strata.items() if role != "prespecified"],
                                                          "strata": rho_strata}, allow_nan=True)
    _write_table(bh_family_table(pd.concat(tests, ignore_index=True), q=q, families=[family]), mdir / "multiplicity_MMS2.parquet", csv=True)
    return {"rows": len(keep), "bundle_sha256": frozen["bundle_sha256"], "results": [split, str(h3["id"])], "h_primary_3": decision["status"]}


# ----------------------------------------------------------------------------- summarize-predictor --split mms2_external


def summarize_predictor_mms2(ctx: StageContext):
    """Audited-predictor metrics on ``mms2_external`` from recorded artifacts only (never fits or re-evaluates).

    Reference: the same-definition crosswalk subset (NOR, HCM; ``mms2.crosswalk``), with the class probabilities of the
    hash-verified frozen predictor on the recorded formal A.  Non-equivalent strata (``HARMONIZABLE_NOT_EXACT``: dilated
    LV -> DCM, RV family -> RV) are reported per pathology group as ``d_model`` distributions with the share of decisions
    equal to the mapped class (Wilson 95 %), never pooled.  Critical-QC rows are excluded and counted.
    """

    from .classification import label_distribution, predictor_summary_v6
    from .formal_features import formal_descriptor_names
    from .metrics import wilson_interval

    name = f"summarize-predictor:{MMS2_SPLIT}"
    require_mms2_unlocked(ctx, f"summarize-predictor --split {MMS2_SPLIT}")
    run = ctx.require_run(name)
    if not run.frozen_path.is_file():
        raise BundleError(f"{name}: run {run.run_id} has no frozen analysis bundle")
    record = _require_evaluated(run, f"evaluate:{MMS2_SPLIT}", name)
    frozen = verify_frozen_bundle(ctx)
    p = run.path
    output = run.sub("metrics") / f"predictor_{MMS2_SPLIT}.json"
    decisions_path = p / "inference" / f"d_model_{MMS2_SPLIT}.parquet"
    inputs = [run.frozen_path, run.record_path(f"evaluate:{MMS2_SPLIT}"), p / "formal" / "d_model_predictor.joblib", p / "formal" / f"A_{DS}.parquet",
              p / "semantics" / f"B_{DS}.parquet", decisions_path, _contracts(ctx) / f"cohort_{DS}.parquet", _contracts(ctx) / "split_manifest_MMS2.parquet"]
    _check_recorded_output(run, record, decisions_path, name)

    def compute() -> dict[str, Any]:
        import joblib

        study = ctx.study
        boot, cal = study.bootstrap["fixed_model_bootstrap"], study.bootstrap["calibration"]
        kw = dict(n_boot=int(boot["replicates"]), seed=int(boot["seed"]), n_bins=int(cal["ece_bins"]),
                  min_valid_fraction=float(boot["min_valid_fraction"]), n_jobs=ctx.jobs)
        configured = [str(c) for c in study.formal["audited_predictor"]["classes"]]
        names = formal_descriptor_names()
        predictor = joblib.load(p / "formal" / "d_model_predictor.joblib")
        split = _split_manifest(ctx, DS)
        cohort = _cohort(ctx, DS).set_index("sample_id")
        b = pd.read_parquet(p / "semantics" / f"B_{DS}.parquet").set_index("sample_id")
        a = pd.read_parquet(p / "formal" / f"A_{DS}.parquet").set_index("sample_id")
        d = pd.read_parquet(decisions_path).set_index("sample_id")["d_model"].astype(object)
        members = sorted(split.loc[split["partition"] == MMS2_SPLIT, "sample_id"].astype(str))
        ids = [s for s in members if s in d.index]
        compat = cohort.loc[ids, "crosswalk_compatibility"].astype(str)
        crit = set(b.index[critical_qc_mask(b, study).to_numpy(dtype=bool)])
        same_all = [s for s in ids if compat[s] == SAME_DEFINITION]
        same = [s for s in same_all if s not in crit]
        result = predictor_summary_v6(reference_labels(cohort, DS).reindex(same).to_numpy(dtype=object), d.reindex(same).to_numpy(dtype=object),
                                      _predictor_probabilities(predictor, a.loc[same, names].to_numpy(dtype=np.float64)), list(predictor.classes),
                                      clusters=_cluster_ids(split, same).to_numpy(dtype=object), **kw)
        result["counts"] = {"n_partition": len(members), "n_recorded_predictions": len(ids), "n_same_definition": len(same_all),
                            "n_excluded_critical_qc": len(same_all) - len(same), **result["counts"]}
        groups: dict[str, Any] = {}
        not_exact = [s for s in ids if compat[s] == NOT_EXACT]
        for group in sorted({str(g) for g in cohort.loc[not_exact, "pathology_group"].dropna()}):
            members_g = [s for s in not_exact if str(cohort.at[s, "pathology_group"]) == group]
            kept = [s for s in members_g if s not in crit]
            mapped = sorted({str(v) for v in cohort.loc[members_g, "crosswalk_acdc_class"].dropna()})
            hits = int(sum(str(d[s]) in mapped for s in kept if d[s] is not None and pd.notna(d[s])))
            lo, hi = wilson_interval(hits, len(kept)) if kept else (None, None)
            groups[group] = {"compatibility": NOT_EXACT, "mapped_acdc_class": mapped, "pooled": False, "n": len(kept),
                             "n_excluded_critical_qc": len(members_g) - len(kept),
                             "pathologies": sorted({str(v) for v in cohort.loc[members_g, "pathology"].dropna()}),
                             "d_model_distribution": label_distribution(d.reindex(kept).to_numpy(dtype=object), configured),
                             "n_d_model_equals_mapped_class": hits, "proportion_d_model_equals_mapped_class": hits / len(kept) if kept else None,
                             "wilson_95": [lo, hi], "status": "REPORTED_SEPARATELY"}
        atomic_write_json(output, {
            "schema": "tmmri.predictor_summary/1", "predictor": str(study.formal["audited_predictor"]["name"]), "partition": MMS2_SPLIT,
            "bundle_sha256": frozen["bundle_sha256"], **result,
            "predictions": f"inference/d_model_{MMS2_SPLIT}.parquet (d_model)",
            "probability_source": "formal/d_model_predictor.joblib (frozen bundle, hash-verified) applied to the recorded formal/A_MMS2.parquet",
            "reference": "crosswalk_same_definition_only (configs/cohorts.yaml mms2.crosswalk: NOR, HCM)",
            "non_equivalent_strata": groups, "qc_rule": "critical-QC rows (QC_ABSTAIN) are excluded and counted",
            "fitting": "none (recorded decisions; frozen predictor applied to recorded formal A)"}, allow_nan=True)
        return {"partitions": [MMS2_SPLIT], "same_definition": len(same), "non_equivalent_groups": sorted(groups)}

    return _stage(ctx, name, compute, inputs, [output], {"partition": MMS2_SPLIT}, status="EVALUATED")
