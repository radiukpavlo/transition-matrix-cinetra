"""Protocol-v6.0.1 image stages (lane L4a): ``build-tensor-cache``, ``verify-torch-weights``, ``extract-deep-features``,
``fit-deep-encoder`` and ``infer-f9``, on a run root (``artifacts/<run_id>``).

Every recorded stage goes through :func:`tmmri.runs.run_stage` like the v6 stages: inputs, outputs, code version,
parameters and a configuration hash (the parsed sections of the study configuration the stage reads plus the digest
of ``configs/neural_input.yaml``) are written to ``.stages/<stage>.json``.  The rerun guard is the v6 once-only
guard (:func:`tmmri.stages.once_only_guard`: in a run with ``.stages/FROZEN.json`` a recorded stage may only be
skipped) extended by the freeze records of the image bundles (``deep_encoder/FROZEN_D20.json``): once a bundle is
frozen, a recorded stage that feeds or evaluates it may only be skipped.  Stages that read M&Ms images call
:func:`tmmri.stages.require_mms_unlocked` (``mms2_external``: the L1 guard ``require_mms2_unlocked`` when present,
else the ``protocol-v6.0.1`` tag).  Nothing here fits on a test or external partition: the tensor cache and the D20
features are label-free transforms, ``fit-deep-encoder`` reads acdc_dev rows only, and ``infer-f9`` only applies
recorded checkpoints (evaluation partitions only after ``freeze-f9``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd

from .. import stages
from ..runs import RunRoot, atomic_write_json, code_version, run_stage
from .preprocess import config_digest, labels_allowed, load_neural_input_config

CACHE_DIR = "neural_cache"
DEEP_DIR = "deep_encoder"
F9_DIR = "neural_f9"
F9_FREEZE_RECORD = "neural_f9/f9_bundle.json"   # written by freeze-f9 (lane L4b; tmmri.neural.f9_stages.F9Paths.bundle)
MMS2_TAG = "protocol-v6.0.1"
DEV = "acdc_dev"
# ``deep_encoder.tta`` (6.0.1, owner issue O1) is amendment scoped (tmmri.config AMENDMENT_SCOPED_KEYS): it is not part of
# the whole-file ``deep_encoder`` digest, so the D20 stages name it explicitly.
STAGE_SECTIONS: dict[str, tuple[str, ...]] = {
    "build-tensor-cache": (),
    "extract-deep-features": ("deep_encoder", "deep_encoder.tta"),
    "fit-deep-encoder": ("deep_encoder", "deep_encoder.tta", "formal.audited_predictor", "transition", "registry", "cohorts.qc_severity"),
    "infer-f9": (),
}


# ----------------------------------------------------------------------------- configuration and guards


def neural_input_config(ctx: stages.StageContext) -> dict[str, Any]:
    cfg = ctx.study.configs.get("neural_input")
    return dict(cfg) if cfg else load_neural_input_config(ctx.study.root)


def deep_encoder_config(ctx: stages.StageContext) -> dict[str, Any]:
    return dict(ctx.study.configs["deep_encoder"])


def image_bundle_guard(run: RunRoot, name: str, freeze_files: Iterable[Path] = ()) -> Callable[[str], None]:
    """The v6 once-only guard plus the image-bundle freeze records (a recorded stage of a frozen bundle may only skip)."""

    base = stages.once_only_guard(run, name)
    files = list(freeze_files)

    def guard(difference: str) -> None:
        frozen = [p for p in files if p.is_file()]
        if frozen:
            run.log({"stage": name, "action": "REFUSED_ONCE_ONLY", "difference": difference, "frozen": [run.portable(p) for p in frozen]})
            raise stages.StageRecordLockedError(
                f"{name}: refused by the once-only guard: the image bundle is frozen ({', '.join(p.name for p in frozen)}) and {name} "
                f"is already recorded; the current call differs from that record in: {difference}. The record and its outputs are "
                f"unchanged; re-executing {name} needs a new run id.")
        base(difference)

    return guard


def _stage(ctx: stages.StageContext, name: str, fn: Callable[[], Any], inputs: list[Path], outputs: list[Path], *,
           cfg: Mapping[str, Any] | None = None, params: Mapping[str, Any] | None = None, freeze_files: Iterable[Path] = ()):
    run = ctx.require_run(name)
    sections = STAGE_SECTIONS[name.split(":", 1)[0]]
    digest = ctx.study.sections_digest(sections) if sections else {}
    if cfg is not None:
        digest["configs/neural_input.yaml"] = config_digest(cfg)
    config_hash = hashlib.sha256("\n".join(f"{k}\t{v}" for k, v in sorted(digest.items())).encode("utf-8")).hexdigest()
    return run_stage(run, name, fn, inputs, outputs, config_hash=config_hash,
                     params={**dict(params or {}), "config_sections": list(sections), "neural_input": cfg is not None},
                     force=ctx.force, n_jobs=ctx.jobs, rerun_guard=image_bundle_guard(run, name, freeze_files))


def _staged_synthetic(ctx: stages.StageContext, rows: pd.DataFrame) -> bool:
    for r in rows.itertuples(index=False):
        try:
            record = json.loads((ctx.staged_root / str(r.dataset) / str(r.case_id) / "case.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if record.get("synthetic_fixture") is not True:
            return False
    return bool(len(rows))


def require_partition_unlocked(ctx: stages.StageContext, partition: str, stage: str, rows: pd.DataFrame | None = None) -> str:
    """M&Ms images only after the freeze (``mms2*``: after ``protocol-v6.0.1``); synthetic fixtures with the override."""

    if partition.startswith("mms2"):
        hook = getattr(stages, "require_mms2_unlocked", None)
        if hook is not None:
            result = str(hook(ctx, stage))
            stages.require_mms2_validated(ctx, stage)     # the passed validate-cohort:MMS2 record (duplicate exclusion)
            return result
        if stages.protocol_frozen(MMS2_TAG):
            return "protocol_frozen"
        if ctx.allow_unfrozen_synthetic and rows is not None and _staged_synthetic(ctx, rows):
            return "synthetic_override"
        raise stages.ProtocolFreezeError(f"{stage}: no M&Ms-2 computation before the {MMS2_TAG} tag")
    if partition.startswith("mms"):
        return stages.require_mms_unlocked(ctx, stage)
    return "acdc"


def split_inputs(ctx: stages.StageContext, partition: str) -> list[Path]:
    """The split manifests a stage on ``partition`` reads (stage inputs): ``contracts/split_manifest.parquet`` and, for an
    M&Ms-2 partition, ``contracts/split_manifest_MMS2.parquet`` (its membership; written by validate-cohort --dataset mms2)."""

    contracts = stages._contracts(ctx)
    return [contracts / "split_manifest.parquet"] + ([contracts / "split_manifest_MMS2.parquet"] if partition.startswith("mms2") else [])


def partition_rows(ctx: stages.StageContext, partition: str) -> pd.DataFrame:
    """Split-manifest rows of ``partition`` (``sample_id, dataset, case_id, cluster_id, outer_fold``), by sample_id."""

    split = stages.split_manifest_for(ctx, partition)       # M&Ms-2 rows: contracts/split_manifest_MMS2.parquet
    rows = split[split["partition"].astype(str) == partition].copy()
    if rows.empty:
        raise ValueError(f"no rows of partition {partition!r} in the split manifest")
    rows["cluster_id"] = stages._cluster_ids(split, rows["sample_id"]).to_numpy(dtype=object)
    return rows.sort_values("sample_id").reset_index(drop=True)


def _dev_labels(ctx: stages.StageContext) -> pd.Series:
    cohort = stages._cohort(ctx, "ACDC")
    return cohort.set_index("sample_id")["acdc_class"].astype(object)


def _staging_inputs(ctx: stages.StageContext, rows: pd.DataFrame) -> list[Path]:
    datasets = sorted({str(d) for d in rows["dataset"]})
    for ds in datasets:
        stages._require_staged(ctx, ds, sorted(rows.loc[rows["dataset"].astype(str) == ds, "case_id"].astype(str)))
    return [stages._staging_manifest_path(ctx, ds) for ds in datasets]


def _write_table(frame: pd.DataFrame, path: Path) -> Path:
    return stages._write_table(frame, path)


# ----------------------------------------------------------------------------- build-tensor-cache


def build_tensor_cache(ctx: stages.StageContext, partition: str):
    """Contract C3 for one partition from the staged images (labels only where label access is granted)."""

    from .preprocess import build_tensor_cache_from_staged

    run = ctx.require_run("build-tensor-cache")
    cfg = neural_input_config(ctx)
    rows = partition_rows(ctx, partition)
    require_partition_unlocked(ctx, partition, f"build-tensor-cache --split {partition}", rows)
    inputs = [*split_inputs(ctx, partition), *_staging_inputs(ctx, rows)]
    if partition.startswith("mms2"):           # SPECIFICATION.md 14.6: M&Ms-2 tensors only after the F9 freeze (step 4 before step 6)
        f9_freeze = run.path / F9_FREEZE_RECORD
        if not f9_freeze.is_file():
            raise stages.BundleError(f"build-tensor-cache --split {partition}: M&Ms-2 tensors are built only after freeze-f9 ({F9_FREEZE_RECORD})")
        inputs.append(f9_freeze)
    allowed = labels_allowed(cfg, partition)
    if allowed:
        inputs.append(stages._contracts(ctx) / "cohort_ACDC.parquet")
    out = run.path / CACHE_DIR / partition

    def compute() -> dict[str, Any]:
        labels = _dev_labels(ctx) if allowed else pd.Series(dtype=object)
        jobs = [{"sample_id": str(r.sample_id), "cluster_id": str(r.cluster_id), "dataset": str(r.dataset), "case_id": str(r.case_id),
                 "label": (None if pd.isna(labels.get(str(r.sample_id))) else str(labels.get(str(r.sample_id)))) if allowed else None}
                for r in rows.itertuples(index=False)]
        result = build_tensor_cache_from_staged(jobs, out, partition, cfg, staged_root=ctx.staged_root, n_jobs=ctx.jobs,
                                                extra_record={"run_id": run.run_id, "code": code_version()})
        return {"cases": int(len(result["manifest"])), "failures": int(len(result["failures"])),
                "transform_version": result["record"]["transform_version"]}

    return _stage(ctx, f"build-tensor-cache:{partition}", compute, inputs, [out], cfg=cfg, params={"partition": partition})


# ----------------------------------------------------------------------------- verify-torch-weights


def verify_torch_weights(ctx: stages.StageContext, *, allow_download: bool = True) -> dict[str, Any]:
    """Download (download.pytorch.org only) and pin both weight files (``registry/weights.yaml``)."""

    from .weights import verify_torch_weights as verify

    report = verify(ctx.study.root, allow_download=allow_download, record=True)
    if ctx.run is not None:
        atomic_write_json(ctx.run.path / DEEP_DIR / "weights_verification.json", report)
        ctx.run.log({"stage": "verify-torch-weights", "weights": {k: v["sha256"] for k, v in report["weights"].items()}})
    return report


# ----------------------------------------------------------------------------- D20


def _deep_dir(run: RunRoot) -> Path:
    return run.path / DEEP_DIR


def d20_weights(ctx: stages.StageContext) -> tuple[Path, str]:
    """(path, SHA-256) of the verified D20 weight file (``registry/weights.yaml``)."""

    from .weights import D20_WEIGHTS, require_weights

    return require_weights(D20_WEIGHTS, ctx.study.root)


def d20_extract(ctx: stages.StageContext, loaders: Sequence[tuple[str, Callable[[], Any]]], partition: str, weights: tuple[Path, str], *,
                device: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """(features, failures, extractor record) of the D20 extractor of ``extract-deep-features`` (frozen ResNet-50 with the
    verified weights; FP32; deterministic; ``configs/deep_encoder.yaml`` batch, resize and ``tta``) on lazy
    ``(sample_id, loader)`` pairs returning :class:`~tmmri.neural.preprocess.CaseVolumes`.  S6 calls it on the resampled
    ``acdc_test`` cases (``run-simulated-shift --step d20``)."""

    from ..deep_encoder import ORIENTATION_RULE, PER_SLICE_RULE, POOLING_RULE, RESIZE_IMPL, build_d20_encoder, extract_d20_features
    from .dihedral import TTA_NONE
    from .torch_numerics import configure_torch_numerics, environment_record, resolve_device

    de_cfg, ni_cfg = deep_encoder_config(ctx), neural_input_config(ctx)
    weights_path, weights_sha = weights
    batch = int(de_cfg["runtime"]["batch_size"])
    tta = str(de_cfg.get("tta") or TTA_NONE)                 # absent: the v6.0.0 definition (no symmetrization)
    icfg = ni_cfg["input"]
    numerics = configure_torch_numerics()
    dev = resolve_device(device)
    encoder = build_d20_encoder(weights_path).to(dev)
    features, failures = extract_d20_features(loaders, encoder, device=dev, partition=partition, batch_size=batch, tta=tta,
                                              input_kwargs={"size": de_cfg["input"]["resize"], "mean": icfg["channel_mean"],   # SPEC 8.4 resize (configs/deep_encoder.yaml)
                                                            "sd": icfg["channel_sd"]})
    record = {
        "schema": "tmmri.d20_features/1", "partition": partition, "representation": "d20_resnet50_imagenet1k_v2_mean2048",
        "encoder": de_cfg["encoder"], "weights_file": weights_path.name, "weights_sha256": weights_sha,
        "input": {"phases": ["ED", "ES"], "slices": "all", "per_slice_normalization": PER_SLICE_RULE, "resize": RESIZE_IMPL,
                  "orientation": ORIENTATION_RULE, "channels": "grey_replicated_to_3", "channel_mean": icfg["channel_mean"],
                  "channel_sd": icfg["channel_sd"]},
        "tta": tta, "pooling": de_cfg["pooling"], "pooling_rule": POOLING_RULE[tta], "batch_size": batch, "rows": int(len(features)),
        "failures": int(len(failures)),
        "environment": environment_record(dev), "numerics_at_start": numerics, "code": code_version(),
    }
    return features, failures, record


def extract_deep_features(ctx: stages.StageContext, partition: str, *, device: str | None = None):
    """D20 2048-d patient vectors of one partition (frozen ResNet-50; FP32; deterministic; batch 32)."""

    from .preprocess import load_staged_volumes

    run = ctx.require_run("extract-deep-features")
    de_cfg, ni_cfg = deep_encoder_config(ctx), neural_input_config(ctx)
    rows = partition_rows(ctx, partition)
    require_partition_unlocked(ctx, partition, f"extract-deep-features --split {partition}", rows)
    inputs = [*split_inputs(ctx, partition), *_staging_inputs(ctx, rows)]
    if partition.startswith("mms2"):           # SPECIFICATION.md 8.4 and 14.6: mms2_external features only after the D20 freeze
        from ..deep_encoder import D20BundleError, load_d20_bundle

        d20_freeze = _deep_dir(run) / "FROZEN_D20.json"
        try:
            load_d20_bundle(_deep_dir(run))                       # present and every frozen file unchanged
        except D20BundleError as exc:
            raise stages.BundleError(f"extract-deep-features --split {partition}: M&Ms-2 features are extracted only after the D20 freeze "
                                     f"(run fit-deep-encoder first): {exc}") from exc
        inputs.append(d20_freeze)
    weights_path, weights_sha = weights = d20_weights(ctx)
    out = _deep_dir(run) / "features"
    paths = [out / f"A_D20_{partition}.parquet", out / f"A_D20_{partition}_failures.parquet", out / f"A_D20_{partition}_extractor.json"]
    batch = int(de_cfg["runtime"]["batch_size"])

    def compute() -> dict[str, Any]:
        loaders = [(str(r.sample_id), (lambda r=r: load_staged_volumes(ctx.staged_root, str(r.dataset), str(r.case_id), sample_id=str(r.sample_id),
                                                                        cluster_id=str(r.cluster_id), cfg=ni_cfg)))
                   for r in rows.itertuples(index=False)]
        features, failures, record = d20_extract(ctx, loaders, partition, weights, device=device)
        _write_table(features, paths[0])
        _write_table(failures, paths[1])
        atomic_write_json(paths[2], record)
        return {"rows": int(len(features)), "failures": int(len(failures)), "device": record["environment"].get("device", str(device))}

    return _stage(ctx, f"extract-deep-features:{partition}", compute, inputs + [weights_path], paths, cfg=ni_cfg,
                  params={"partition": partition, "weights_sha256": weights_sha, "batch_size": batch, "tta": de_cfg.get("tta")},
                  freeze_files=[_deep_dir(run) / "FROZEN_D20.json"])


def fit_deep_encoder(ctx: stages.StageContext):
    """D20 standardization, ridge T (alternative A) and audited predictor on acdc_dev; OOF and final; frozen."""

    from ..deep_encoder import bundle_paths, fit_d20_models, freeze_d20, write_d20_fit

    run = ctx.require_run("fit-deep-encoder")
    deep = _deep_dir(run)
    n_folds = int(ctx.study.splits["acdc"]["acdc_dev"]["outer_folds"]["n_folds"])
    features_path = deep / "features" / "A_D20_acdc_dev.parquet"
    contracts = stages._contracts(ctx)
    inputs = [features_path, run.path / "semantics" / "B_ACDC.parquet", contracts / "split_manifest.parquet",
              contracts / "inner_folds.parquet", contracts / "cohort_ACDC.parquet"]
    outputs = sorted(bundle_paths(deep, n_folds).values()) + [deep / "FROZEN_D20.json"]
    outputs.remove(features_path)

    def compute() -> dict[str, Any]:
        features = pd.read_parquet(features_path)
        dev = partition_rows(ctx, DEV)[["sample_id", "outer_fold", "cluster_id"]]
        b = pd.read_parquet(run.path / "semantics" / "B_ACDC.parquet")
        critical = set(b.loc[stages.critical_qc_mask(b, ctx.study).to_numpy(), "sample_id"].astype(str))
        fit = fit_d20_models(features, dev, b, _dev_labels(ctx), critical, targets=ctx.study.transition_targets(),
                             transition_cfg=ctx.study.transition, registry=ctx.study.registry,
                             predictor_cfg=ctx.study.formal["audited_predictor"], make_inner_fn=lambda ids: stages._inner_fold_fn(ctx, ids),
                             classes=[str(c) for c in ctx.study.formal["audited_predictor"]["classes"]])
        sections = STAGE_SECTIONS["fit-deep-encoder"]
        provenance = {"run_id": run.run_id, "code": code_version(), "config_sections": ctx.study.sections_digest(sections),
                      "weights_sha256": deep_encoder_config(ctx)["encoder"].get("weights_sha256"), "tta": deep_encoder_config(ctx).get("tta"),
                      "inputs": {run.portable(p): stages.hash_path(p) for p in inputs}}
        write_d20_fit(fit, deep, n_folds=n_folds, record=provenance)
        frozen = freeze_d20(deep, n_folds=n_folds, provenance=provenance)
        return {**fit.summary, "bundle_sha256": frozen["bundle_sha256"]}

    return _stage(ctx, "fit-deep-encoder", compute, inputs, outputs, freeze_files=[deep / "FROZEN_D20.json"])


# ----------------------------------------------------------------------------- infer-f9


def _combine(directory: Path, prefix: str) -> Path:
    """Rebuild ``<prefix>.parquet`` as the union of the recorded per-partition files ``<prefix>_<partition>.parquet``."""

    from .prediction_table import sort_frame

    parts = sorted(p for p in directory.glob(f"{prefix}_*.parquet"))
    frames = [pd.read_parquet(p) for p in parts]
    combined = sort_frame(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame()
    return _write_table(combined, directory / f"{prefix}.parquet")


def _reference_labels(ctx: stages.StageContext, rows: pd.DataFrame, partition: str, manifest: pd.DataFrame) -> dict[str, Any]:
    """y_true: the cache label on acdc_dev; on evaluation partitions the evaluation reference (joined after inference)."""

    if partition == DEV:
        return {str(s): (None if pd.isna(v) else str(v)) for s, v in zip(manifest["sample_id"], manifest["label"])}
    out: dict[str, Any] = {}
    for ds in sorted({str(d) for d in rows["dataset"]}):
        try:
            cohort = stages._cohort(ctx, ds)
            refs = stages.reference_labels(cohort, ds)
        except (FileNotFoundError, KeyError, ValueError):
            continue
        out.update({str(s): (None if pd.isna(v) else str(v)) for s, v in zip(cohort["sample_id"], refs)})
    return out


def infer_f9(ctx: stages.StageContext, partition: str, *, device: str | None = None, seeds: Iterable[int] | None = None):
    """F9 predictions and 64-d activations of one partition (contract C4) from the recorded checkpoints (C8)."""

    from .dihedral import D4_TTA
    from .inference import (
        check_leakage,
        discover_checkpoints,
        run_f9_inference,
        select_encoders,
    )
    from .preprocess import read_cache_manifest
    from .torch_numerics import (
        configure_torch_numerics,
        environment_record,
        resolve_device,
    )

    run = ctx.require_run("infer-f9")
    cfg = neural_input_config(ctx)
    rows = partition_rows(ctx, partition)
    require_partition_unlocked(ctx, partition, f"infer-f9 --split {partition}", rows)
    freeze = run.path / F9_FREEZE_RECORD
    if partition != DEV and not freeze.is_file():
        raise stages.BundleError(f"infer-f9 --split {partition}: evaluation partitions need the frozen F9 bundle ({F9_FREEZE_RECORD})")
    cache = run.path / CACHE_DIR / partition
    ckpt_root = run.path / F9_DIR / "checkpoints"
    if not (cache / "manifest.csv").is_file():
        raise FileNotFoundError(f"{cache / 'manifest.csv'} missing: run build-tensor-cache --split {partition} first")
    encoder_cfg = (cfg.get("inference") or {}).get("encoders") or {}
    wanted = [str(e) for e in (encoder_cfg.get(partition) or encoder_cfg.get("default") or ["final"])]
    encoders = select_encoders(discover_checkpoints(ckpt_root), wanted, seeds=seeds)
    if freeze.is_file():
        check_frozen_encoders(run, encoders, json.loads(freeze.read_text(encoding="utf-8")))
    dev_rows = partition_rows(ctx, DEV)
    dev_folds = {str(s): int(f) for s, f in zip(dev_rows["sample_id"], dev_rows["outer_fold"])}
    check_leakage(encoders, partition, set(rows["sample_id"].astype(str)), dev_folds)
    out = run.path / F9_DIR
    paths = [out / f"predictions_{partition}.parquet", out / f"activations_{partition}.parquet", out / f"inference_{partition}.json"]
    inputs = [cache, *split_inputs(ctx, partition)] + [e.path for e in encoders] + ([freeze] if freeze.is_file() else [])
    order = stages.mms2_freeze_order(ctx, ["F9"]) if partition.startswith("mms2") else None    # SPECIFICATION.md 14.6, recorded

    def compute() -> dict[str, Any]:
        numerics = configure_torch_numerics()
        dev = resolve_device(device)
        manifest = read_cache_manifest(cache)
        predictions, activations, details = run_f9_inference(cache, encoders, partition=partition, device=dev, dev_folds=dev_folds,
                                                             y_true=_reference_labels(ctx, rows, partition, manifest))
        _write_table(predictions, paths[0])
        _write_table(activations, paths[1])
        atomic_write_json(paths[2], {"schema": "tmmri.f9_inference/1", **details, "environment": environment_record(dev),
                                     "numerics_at_start": numerics, "code": code_version(),
                                     "f9_bundle": json.loads(freeze.read_text(encoding="utf-8")).get("bundle_sha256") if freeze.is_file() else None,
                                     "freeze_order": order,
                                     "predictions_sha256": stages.hash_path(paths[0]), "activations_sha256": stages.hash_path(paths[1])})
        return {"rows": int(len(predictions)), "encoders": len(encoders), "device": str(dev)}

    result = _stage(ctx, f"infer-f9:{partition}", compute, inputs, paths, cfg=cfg,
                    params={"partition": partition, "encoders": sorted(f"{e.seed}:{e.encoder_id}:{e.fit_id}" for e in encoders), "tta": D4_TTA},
                    freeze_files=[freeze])
    _combine(out, "predictions")
    _combine(out, "activations")
    return result


def check_frozen_encoders(run: Any, encoders: Sequence[Any], bundle: Mapping[str, Any]) -> None:
    """Integration of L4a and L4b: once ``freeze-f9`` has written ``neural_f9/f9_bundle.json``, every encoder used by
    ``infer-f9`` must be one of its fits, with the frozen checkpoint SHA-256 (no refit outside the bundle is used)."""

    frozen = {str(f["fit_id"]): f for f in bundle.get("fits") or []}
    for enc in encoders:
        fit = frozen.get(str(enc.fit_id))
        if fit is None:
            raise stages.BundleError(f"infer-f9: checkpoint {enc.fit_id} ({enc.encoder_id}, seed {enc.seed}) is not in the frozen F9 bundle")
        if stages.hash_path(enc.path / "model.pt") != str(fit.get("checkpoint_sha256")):
            raise stages.BundleError(f"infer-f9: checkpoint {enc.fit_id} differs from the frozen F9 bundle (checkpoint_sha256)")


def dispatch(ctx: stages.StageContext, args: Any) -> Any:
    """CLI dispatch of the L4a commands (``tmmri.cli``)."""

    command = args.command
    if command == "build-tensor-cache":
        return build_tensor_cache(ctx, args.split)
    if command == "verify-torch-weights":
        return verify_torch_weights(ctx, allow_download=not args.no_download)
    if command == "extract-deep-features":
        return extract_deep_features(ctx, args.split, device=args.device)
    if command == "fit-deep-encoder":
        return fit_deep_encoder(ctx)
    if command == "infer-f9":
        return infer_f9(ctx, args.split, device=args.device, seeds=args.seed or None)
    raise SystemExit(f"Unknown command: {command}")
