"""S6 simulated acquisition shift (manuscript protocol 2.1, Section E; ``configs/data_centric.yaml`` ``simulated_shift``).

An interventional mechanism test on ``acdc_test`` ("registered after unblinding"): the staged ED/ES images (linear) and
masks (nearest neighbour) are resampled to the median in-plane spacing and slice increment of each M&Ms-1 vendor, read
from NIfTI headers only (:func:`tmmri.ingest.geometry.geometry_from_image`; no voxel and no label of M&Ms is read).
Resampling preserves the physical centre of the field of view: output voxel i of an axis with n_out voxels samples the
input at ``c_in + (i - c_out) * s_out / s_in`` (``c = (n - 1) / 2``), with ``n_out = floor(n_in * s_in / s_out + 0.5)``
and nearest-edge boundary handling; the affine is updated consistently (orientation-only ACDC affines stay
orientation-only).  The case's ``z_spacing_method`` is kept, so the QC tokens do not change by design.

Steps (``run-simulated-shift --vendor V --step ...``; each vendor condition is once-only in a frozen run):

* ``geometry`` (run first automatically): per-case header geometry of ``mms_external`` and the vendor medians.
* ``descriptors``: typed B (:func:`tmmri.semantics.features.build_semantics_table_v6` with the frozen acdc_dev outlier
  reference) and formal A (:func:`tmmri.formal_features.formal_descriptors_v3`) on the perturbed cases; the frozen
  transition and audited predictor; delta rho, delta rho_j and delta accuracy against the recorded ``evaluate:acdc_test``
  outputs with a paired cluster bootstrap (:mod:`tmmri.experiments.data_centric`).
* ``tensors``: the C3 tensor inputs of F9 for the perturbed cases through the hook ``simulated_shift.tensor_inputs.hook``
  (``tmmri.neural.preprocess``); refuses (exit 4, nothing recorded) while the hook is unavailable.
* ``d20`` (``tmmri-gpu``): the D20 extractor of ``extract-deep-features`` (:func:`tmmri.neural.image_stages.d20_extract`)
  on the resampled volumes; the frozen final D20 transition and predictor (``deep_encoder/FROZEN_D20.json``, re-hashed)
  on the recorded unperturbed features (``extract-deep-features --split acdc_test``) and on the perturbed ones.
* ``f9`` (``tmmri-gpu``): the frozen final F9 encoder of the primary seed (``neural_f9/f9_bundle.json``; the inference
  code of ``infer-f9``) on the perturbed C3 cache of ``--step tensors``; the frozen F9b final T of that seed
  (``probes/f9b/bundle.json``, re-hashed) on the recorded unperturbed activations (``infer-f9 --split acdc_test``) and
  on the perturbed ones.
* For ``d20`` and ``f9``: delta rho, delta rho_j and delta accuracy as for the descriptors; every perturbed output is
  written under ``simulated_shift/<vendor>/``.  A missing input refuses with nothing recorded.

Research use only; synthetic fixtures are algorithm controls, never cohort results.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import io
import json
import math
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import nibabel as nib
import numpy as np
import pandas as pd

from .data_centric import (
    REGISTERED_AFTER_UNBLINDING,
    bootstrap_settings,
    config_sha256_params,
    data_centric_config,
    paired_accuracy_difference,
    paired_rho_difference,
    record_stage,
)
from ..ingest.geometry import affine_is_unscaled, geometry_from_image
from ..runs import atomic_write_bytes, atomic_write_json, sha256_file

DC_SECTION = "simulated_shift"
STEPS = ("descriptors", "tensors", "d20", "f9")
C3_MANIFEST_COLUMNS = ["sample_id", "cluster_id", "partition", "label", "n_ed", "n_es", "npz_sha256", "crop_fraction_ed", "crop_fraction_es",
                       "inplane_spacing_mm", "transform_version", "npz_file"]   # npz_file: C3 file name (L4a readers)
ORDERS = {"linear": 1, "nearest": 0}


# ----------------------------------------------------------------------------- resampling


def resampled_shape(shape: Sequence[int], spacing_in: Sequence[float], spacing_out: Sequence[float]) -> tuple[int, ...]:
    """``floor(n_in * s_in / s_out + 0.5)`` per axis (at least one voxel)."""

    return tuple(max(1, int(math.floor(int(n) * float(si) / float(so) + 0.5))) for n, si, so in zip(shape, spacing_in, spacing_out))


def index_map(shape_in: Sequence[int], shape_out: Sequence[int], spacing_in: Sequence[float], spacing_out: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """(scale, offset) with input index = scale * output index + offset (physical centre preserved)."""

    scale = np.array([float(so) / float(si) for si, so in zip(spacing_in, spacing_out)], dtype=float)
    c_in = (np.asarray(shape_in, dtype=float) - 1.0) / 2.0
    c_out = (np.asarray(shape_out, dtype=float) - 1.0) / 2.0
    return scale, c_in - scale * c_out


def resample_volume(volume: np.ndarray, spacing_in: Sequence[float], spacing_out: Sequence[float], *, interpolation: str,
                    shape_out: Sequence[int] | None = None) -> np.ndarray:
    """Resample a 3-D (x, y, z) volume to ``spacing_out``: ``linear`` for images (float output of the input dtype),
    ``nearest`` for label masks (labels are copied, never blended; the dtype is kept)."""

    from scipy.ndimage import affine_transform

    vol = np.asarray(volume)
    if vol.ndim != 3:
        raise ValueError(f"resample_volume expects a 3-D volume, got shape {vol.shape}")
    if any(not (np.isfinite(float(s)) and float(s) > 0) for s in (*spacing_in, *spacing_out)):
        raise ValueError(f"inconsistent geometry: spacings {tuple(spacing_in)} -> {tuple(spacing_out)}; never substituted")
    order = ORDERS[interpolation]
    shape = tuple(int(n) for n in (shape_out or resampled_shape(vol.shape, spacing_in, spacing_out)))
    scale, offset = index_map(vol.shape, shape, spacing_in, spacing_out)
    source = vol if order == 0 else vol.astype(np.float64)
    out = affine_transform(source, scale, offset=offset, output_shape=shape, order=order, mode="nearest", prefilter=False)
    return out.astype(vol.dtype if order == 0 or np.issubdtype(vol.dtype, np.floating) else np.float32)


def resampled_affine(affine: np.ndarray, spacing_in: Sequence[float], shape_in: Sequence[int], spacing_out: Sequence[float],
                     shape_out: Sequence[int]) -> np.ndarray:
    """The voxel-to-world affine of the resampled grid.  A scaled affine becomes ``A @ M`` (M = the index map); an
    orientation-only affine (``affine_is_unscaled``: voxel size only in the header pixdim, as in ACDC) keeps its unit
    columns and moves its origin by the physical offset of the new first voxel."""

    a = np.asarray(affine, dtype=float)
    scale, offset = index_map(shape_in, shape_out, spacing_in, spacing_out)
    out = a.copy()
    if affine_is_unscaled(a, tuple(float(s) for s in spacing_in)):
        out[:3, 3] = a[:3, 3] + a[:3, :3] @ (np.asarray(spacing_in, dtype=float) * offset)
    else:
        m = np.eye(4)
        m[:3, :3] = np.diag(scale)
        m[:3, 3] = offset
        out = a @ m
    return out


def label_volumes_ml(mask: np.ndarray, spacing: Sequence[float], labels: Sequence[int] = (1, 2, 3)) -> dict[int, float]:
    voxel_ml = float(np.prod([float(s) for s in spacing])) / 1000.0
    return {int(lab): float((np.asarray(mask) == lab).sum()) * voxel_ml for lab in labels}


def perturb_case(case: Any, target_spacing: Sequence[float], *, vendor: str | None = None) -> Any:
    """A copy of a staged case (:class:`tmmri.ingest.staging.StagedCase`) resampled to ``target_spacing`` (dx, dy, dz)."""

    dx, dy, dz = (tuple(case.spacing_mm) + (None, None, None))[:3]
    if dz is None:
        raise ValueError(f"{case.sample_id}: no slice increment; inconsistent geometry is rejected, never substituted")
    s_in = (float(dx), float(dy), float(dz))
    s_out = tuple(float(s) for s in target_spacing)
    shape_in = tuple(int(n) for n in case.ed_mask.shape)
    for name in ("ed_image", "es_image", "es_mask"):
        if tuple(getattr(case, name).shape) != shape_in:
            raise ValueError(f"{case.sample_id}: {name} shape {getattr(case, name).shape} differs from ed_mask {shape_in}")
    shape_out = resampled_shape(shape_in, s_in, s_out)
    arrays = {name: resample_volume(getattr(case, name), s_in, s_out, interpolation="linear" if name.endswith("image") else "nearest",
                                    shape_out=shape_out) for name in ("ed_image", "es_image", "ed_mask", "es_mask")}
    affine = resampled_affine(case.affine, s_in, shape_in, s_out, shape_out)
    geometry = dataclasses.replace(case.geometry, image_shape=tuple(shape_out), spacing_x_mm=s_out[0], spacing_y_mm=s_out[1],
                                   slice_thickness_mm=s_out[2], slice_increment_mm=s_out[2], slice_increment_variability_mm=0.0,
                                   affine_matrix=affine.tolist(), num_slices=int(shape_out[2]))
    meta = {**case.meta, "slice_thickness_mm": s_out[2],
            "simulated_shift": {"vendor": vendor, "spacing_in_mm": list(s_in), "spacing_out_mm": list(s_out), "shape_in": list(shape_in),
                                "shape_out": list(shape_out), "image_interpolation": "linear", "mask_interpolation": "nearest"}}
    return dataclasses.replace(case, **arrays, spacing_mm=s_out, affine=affine, geometry=geometry, meta=meta)


# ----------------------------------------------------------------------------- vendor geometry (headers only)


def header_geometry(path: str | Path):
    """Geometry of a NIfTI file from its header (shape, pixdim, affine); the voxel data are never read."""

    return geometry_from_image(nib.load(str(path)))


def vendor_geometry(ctx: Any, cfg: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    from .. import stages

    split = stages._split_manifest(ctx)
    ids = set(split.loc[split["partition"] == cfg["geometry"]["source_partition"], "sample_id"].astype(str))
    cohort = stages._cohort(ctx, "MMS")
    cohort = cohort[cohort["sample_id"].astype(str).isin(ids)].sort_values("sample_id")
    rows = []
    for r in cohort.itertuples(index=False):
        staged = Path(ctx.staged_root) / "MMS" / str(r.case_id) / "ed_image.nii"
        path, source = (staged, "staged_ed_image_header") if staged.is_file() else (ctx.source_root("mms") / str(r.image_relpath), "release_image_header")
        geom = header_geometry(path)
        rows.append({"sample_id": str(r.sample_id), "case_id": str(r.case_id), "vendor": str(r.vendor), "vendor_name": str(r.vendor_name),
                     "spacing_x_mm": float(geom.spacing_x_mm), "spacing_y_mm": float(geom.spacing_y_mm),
                     "slice_increment_mm": None if geom.slice_increment_mm is None else float(geom.slice_increment_mm),
                     "z_spacing_method": geom.z_spacing_method, "header_source": source})
    table = pd.DataFrame(rows, columns=["sample_id", "case_id", "vendor", "vendor_name", "spacing_x_mm", "spacing_y_mm", "slice_increment_mm",
                                       "z_spacing_method", "header_source"])
    fields = [str(f) for f in cfg["geometry"]["fields"]]
    summary: dict[str, Any] = {}
    for code, name in cfg["vendors"].items():
        sub = table[table["vendor"] == str(code)]
        med = {f: (float(np.nanmedian(pd.to_numeric(sub[f], errors="coerce"))) if len(sub) and pd.to_numeric(sub[f], errors="coerce").notna().any() else None)
               for f in fields}
        ok = len(sub) > 0 and all(v is not None and v > 0 for v in med.values())
        summary[str(code)] = {"vendor_name": str(name), "n_cases": int(len(sub)), "median": med, "status": "EVALUATED" if ok else "NE",
                              "reason": None if ok else "NO_HEADER_GEOMETRY_FOR_VENDOR"}
    return table, summary


def target_spacing(summary: Mapping[str, Any], vendor: str) -> tuple[float, float, float]:
    entry = summary.get(str(vendor))
    if entry is None or entry["status"] != "EVALUATED":
        raise ValueError(f"run-simulated-shift: no header geometry for vendor {vendor!r}")
    med = entry["median"]
    return float(med["spacing_x_mm"]), float(med["spacing_y_mm"]), float(med["slice_increment_mm"])


# ----------------------------------------------------------------------------- per-case workers


def load_perturbed_case(case_id: str, *, staged_root: str, label_map: dict[int, int], metadata: Mapping[str, Mapping[str, Any]],
                        target: Sequence[float], vendor: str):
    """Loader of :func:`build_semantics_table_v6`: the staged ACDC case with release metadata, then resampled."""

    from ..stages import load_case_for_semantics

    case = load_case_for_semantics(case_id, dataset="ACDC", staged_root=staged_root, label_map=label_map, metadata=metadata)
    return perturb_case(case, target, vendor=vendor)


def _formal_job(job: Mapping[str, Any]) -> dict[str, Any]:
    from ..formal_features import formal_descriptors_v3

    out = {"sample_id": f"ACDC:{job['case_id']}", "case_id": job["case_id"], "values": None, "reason_code": None, "detail": None, "log": None}
    try:
        case = job["loader"](job["case_id"])
    except (FileNotFoundError, OSError, ValueError) as exc:
        out.update(reason_code="INPUT_DECODE_FAILURE", detail=f"{type(exc).__name__}: {exc}")
        return out
    shift = case.meta["simulated_shift"]
    log = {"sample_id": out["sample_id"], **{k: json.dumps(v) for k, v in shift.items() if k != "vendor"},
           "fov_in_mm": json.dumps([n * s for n, s in zip(shift["shape_in"], shift["spacing_in_mm"])]),
           "fov_out_mm": json.dumps([n * s for n, s in zip(shift["shape_out"], shift["spacing_out_mm"])]),
           "affine_out": json.dumps(np.asarray(case.affine).tolist()),
           "label_volumes_ed_ml_out": json.dumps(label_volumes_ml(case.ed_mask, case.spacing_mm)),
           "label_volumes_es_ml_out": json.dumps(label_volumes_ml(case.es_mask, case.spacing_mm)),
           "mask_labels_out": json.dumps(sorted({int(v) for m in (case.ed_mask, case.es_mask) for v in np.unique(m)}))}
    out["log"] = log
    values = formal_descriptors_v3(case)
    if not np.isfinite(values).all() or not np.isfinite(case.ed_image).all() or not np.isfinite(case.es_image).all():
        out.update(reason_code="INPUT_NONFINITE", detail="non-finite voxel or descriptor")
        return out
    out["values"] = values.tolist()
    return out


# ----------------------------------------------------------------------------- tensor inputs (C3)


def resolve_tensor_hook(spec: str) -> Callable[..., Mapping[str, Any]] | None:
    """``module:function`` of the C3 preprocessing hook, or None when it is not importable."""

    module, _, name = str(spec).partition(":")
    try:
        mod = importlib.import_module(module)
    except ImportError:
        return None
    fn = getattr(mod, name, None)
    return fn if callable(fn) else None


def _neural_input_config(study: Any, rel: str) -> dict[str, Any]:
    key = Path(rel).relative_to("configs").with_suffix("").as_posix() if str(rel).startswith("configs/") else Path(rel).stem
    if key in study.configs:
        return dict(study.configs[key])
    path = Path(study.root) / rel
    if path.is_file():
        from ..config import _read_any

        return dict(_read_any(path))
    return {}


def write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    """Deterministic ``.npz`` (fixed member timestamps, sorted members); returns its SHA-256."""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in sorted(arrays):
            member = io.BytesIO()
            np.lib.format.write_array(member, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            zf.writestr(zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)), member.getvalue())
    atomic_write_bytes(path, buffer.getvalue())
    return sha256_file(path)


def npz_filename(sample_id: str) -> str:
    """C3 file name of a case: :func:`tmmri.neural.preprocess.cache_file_name` (``:`` -> ``__`` on every platform), the
    rule of ``build-tensor-cache`` and of every cache reader (integration of L89 and L4a)."""

    from ..neural.preprocess import cache_file_name

    return cache_file_name(sample_id)


def c3_entry(sample_id: str, tensors: Mapping[str, Any], out_dir: Path, *, cluster_id: str, partition: str, label: str | None,
             size: Sequence[int] = (224, 224)) -> dict[str, Any]:
    """Validate one hook result against C3 (slices of ``size``: ``configs/neural_input.yaml`` ``input.size_pixels``) and
    write ``<out_dir>/<sample_id>.npz``; returns its manifest row."""

    arrays = {k: np.asarray(tensors[k], dtype=np.float32) for k in ("ed", "es", "ed_geom", "es_geom")}
    size = tuple(int(n) for n in size)
    for phase in ("ed", "es"):
        x, g = arrays[phase], arrays[f"{phase}_geom"]
        if x.ndim != 3 or x.shape[1:] != size or g.shape != (x.shape[0], 2):
            raise ValueError(f"{sample_id}: C3 {phase} must be [n,{size[0]},{size[1]}] with {phase}_geom [n,2], got {x.shape} and {g.shape}")
        if not np.isfinite(x).all() or x.min() < 0.0 or x.max() > 1.0:
            raise ValueError(f"{sample_id}: C3 {phase} intensities must be finite in [0, 1]")
    from ..neural.preprocess import write_npz as write_c3_npz

    digest, _ = write_c3_npz(Path(out_dir) / npz_filename(sample_id), arrays)   # the byte-deterministic writer of build-tensor-cache
    return {"sample_id": str(sample_id), "cluster_id": str(cluster_id), "partition": partition, "label": label or "",
            "n_ed": int(arrays["ed"].shape[0]), "n_es": int(arrays["es"].shape[0]), "npz_sha256": digest,
            "crop_fraction_ed": float(tensors.get("crop_fraction_ed", np.nan)), "crop_fraction_es": float(tensors.get("crop_fraction_es", np.nan)),
            "inplane_spacing_mm": float(tensors.get("inplane_spacing_mm", np.nan)), "transform_version": str(tensors.get("transform_version", "")),
            "npz_file": npz_filename(sample_id)}


# ----------------------------------------------------------------------------- stages


def _vendor_dir(run: Any, vendor: str) -> Path:
    return run.sub("simulated_shift") / str(vendor)


def _test_cases(ctx: Any, cfg: Mapping[str, Any]) -> tuple[pd.DataFrame, list[str], dict[str, dict[str, Any]]]:
    from .. import stages

    split = stages._split_manifest(ctx)
    test = split[split["partition"] == cfg["partition"]].sort_values("sample_id")
    cohort = stages._cohort(ctx, "ACDC").set_index("case_id")
    case_ids = [str(c) for c in test["case_id"]]
    parts = test.set_index("case_id")["partition"]
    metadata = {cid: stages.release_metadata("ACDC", {**cohort.loc[cid].to_dict(), "case_id": cid}, parts.loc[cid]) for cid in case_ids}
    return split, case_ids, metadata


def geometry_stage(ctx: Any, cfg: Mapping[str, Any]):
    from .. import stages

    run = ctx.require_run("run-simulated-shift")
    sdir = run.sub("simulated_shift")
    outputs = [sdir / "vendor_geometry_cases.parquet", sdir / "vendor_geometry.json"]

    def compute() -> dict[str, Any]:
        table, summary = vendor_geometry(ctx, cfg)
        stages._write_table(table, outputs[0])
        atomic_write_json(outputs[1], {"analysis": "S6_simulated_acquisition_shift", "role": "intervention target geometry",
                                       "source": cfg["geometry"], "vendors": summary, "metadata_only": True, **config_sha256_params(ctx.study)})
        return {"vendors": {k: v["status"] for k, v in summary.items()}}

    inputs = [stages._contracts(ctx) / "split_manifest.parquet", stages._contracts(ctx) / "cohort_MMS.parquet", stages._staging_manifest_path(ctx, "MMS")]
    return record_stage(ctx, "run-simulated-shift:geometry", compute, inputs, outputs, sections=(), dc_section=DC_SECTION, params={"step": "geometry"})


def _bundle_inputs(ctx: Any, run: Any) -> list[Path]:
    from .. import stages

    return [run.frozen_path, run.path / "transitions" / "transition_model.json", run.path / "formal" / "d_model_predictor.joblib",
            run.path / "qc" / "outlier_reference.json", stages._contracts(ctx) / "split_manifest.parquet", stages._contracts(ctx) / "cohort_ACDC.parquet",
            run.path / "semantics" / "B_ACDC.parquet", run.path / "formal" / "A_ACDC.parquet", run.path / "simulated_shift" / "vendor_geometry.json"]


def _rho_inputs(ctx: Any, keep: list[str], split: pd.DataFrame) -> dict[str, Any]:
    from .. import stages

    study = ctx.study
    sd, mean = stages._dev_reference(ctx)
    kw = stages._rho_kwargs(ctx)
    boot = bootstrap_settings(data_centric_config(study))
    return {"null_mean": mean, "reference_sd": sd, "family": study.primary_family(), "min_estimable": kw["min_estimable"],
            "cluster": np.array([s.split(":", 1)[1] for s in keep], dtype=object),
            "resample_cluster": stages._cluster_ids(split, keep).to_numpy(dtype=object), "n_boot": boot["n_boot"], "seed": boot["seed"],
            "level": boot["level"], "min_valid_fraction": boot["min_valid_fraction"]}


def descriptors_stage(ctx: Any, cfg: Mapping[str, Any], vendor: str):
    """Perturbed B and formal A of ``acdc_test`` for one vendor condition; descriptor deltas against the recorded outputs."""

    import joblib

    from .. import stages
    from ..formal_features import formal_descriptor_names
    from ..runs import parallel_map
    from ..semantics.contract import validate_typed_b

    run = ctx.require_run("run-simulated-shift")
    frozen = stages.verify_frozen_bundle(ctx)
    stages._require_evaluated(run, "evaluate:acdc_test", "run-simulated-shift")
    summary = json.loads((run.path / "simulated_shift" / "vendor_geometry.json").read_text(encoding="utf-8"))["vendors"]
    target = target_spacing(summary, vendor)
    vdir = _vendor_dir(run, vendor)
    outputs = [vdir / "B_acdc_test.parquet", vdir / "A_acdc_test.parquet", vdir / "A_failures.parquet", vdir / "bhat_acdc_test.parquet",
               vdir / "d_model_acdc_test.parquet", vdir / "resample_log.parquet", vdir / "delta_descriptors.json"]
    recorded = [run.path / "reconstructed" / "bhat_acdc_test.parquet", run.path / "inference" / "d_model_acdc_test.parquet",
                run.path / "metrics" / "primary_rho_acdc_test.json"]

    def compute() -> dict[str, Any]:
        study = ctx.study
        build_table = stages.methods_api("tmmri.semantics.features", "build_semantics_table_v6")
        apply_ref = stages.methods_api("tmmri.semantics.qc", "apply_outlier_reference")
        model_cls = stages.methods_api("tmmri.transition_fit", "TransitionModel")
        split, case_ids, metadata = _test_cases(ctx, cfg)
        loader = functools.partial(load_perturbed_case, staged_root=str(ctx.staged_root), label_map=study.label_map("acdc"), metadata=metadata,
                                   target=target, vendor=str(vendor))
        frame = build_table((loader, case_ids), study.registry, ctx.jobs, cohorts=study.cohorts, reason_codes=study.reason_codes)
        frame = frame.sort_values("sample_id").reset_index(drop=True)
        reference = json.loads((run.path / "qc" / "outlier_reference.json").read_text(encoding="utf-8"))
        frame = apply_ref(frame, reference, study.cohorts, reason_codes=study.reason_codes)
        validate_typed_b(frame, study.registry, cohorts_cfg=study.cohorts)
        stages._write_table(frame, outputs[0])
        results = parallel_map(_formal_job, [{"case_id": c, "loader": loader} for c in case_ids], n_jobs=ctx.jobs)
        names = formal_descriptor_names()
        ok = [r for r in results if r["values"] is not None]
        a1 = pd.DataFrame([r["values"] for r in ok], columns=names, dtype=np.float64)
        a1.insert(0, "sample_id", [r["sample_id"] for r in ok])
        stages._write_table(a1, outputs[1])
        stages._write_table(pd.DataFrame([{k: r[k] for k in ("sample_id", "case_id", "reason_code", "detail")} for r in results if r["values"] is None],
                                         columns=["sample_id", "case_id", "reason_code", "detail"]), outputs[2])
        stages._write_table(pd.DataFrame([r["log"] for r in results if r["log"] is not None]), outputs[5])
        # Frozen bundle on the perturbed inputs.
        b0 = pd.read_parquet(run.path / "semantics" / "B_ACDC.parquet").set_index("sample_id")
        b1 = frame.set_index("sample_id")
        bhat0 = pd.read_parquet(recorded[0]).set_index("sample_id")
        d0 = pd.read_parquet(recorded[1]).set_index("sample_id")["d_model"]
        keep = [s for s in (f"ACDC:{c}" for c in case_ids) if s in bhat0.index and s in d0.index and s in b0.index and s in b1.index]
        x1 = a1.set_index("sample_id").reindex(keep)[names].to_numpy(dtype=np.float64)
        model = model_cls.from_dict(json.loads((run.path / "transitions" / "transition_model.json").read_text(encoding="utf-8")))
        bhat1 = model.predict(x1, index=pd.Index(keep, name="sample_id"))
        stages._write_table(bhat1.reset_index(names="sample_id"), outputs[3])
        predictor = joblib.load(run.path / "formal" / "d_model_predictor.joblib")
        d1 = pd.Series(stages.predict_labels(predictor, x1), index=keep, name="d_model")
        stages._write_table(pd.DataFrame({"sample_id": keep, "d_model": d1.to_numpy()}), outputs[4])
        targets = study.transition_targets()
        kw = _rho_inputs(ctx, keep, split)
        rho = paired_rho_difference(b0.loc[keep, kw["family"]], bhat0.loc[keep, kw["family"]], b1.loc[keep, kw["family"]], bhat1[kw["family"]],
                                    kw["null_mean"], kw["reference_sd"], kw["family"], cluster=kw["cluster"], resample_cluster=kw["resample_cluster"],
                                    n_boot=kw["n_boot"], seed=kw["seed"], min_estimable=kw["min_estimable"], level=kw["level"],
                                    min_valid_fraction=kw["min_valid_fraction"])
        cohort = stages._cohort(ctx, "ACDC").set_index("sample_id")
        truth = stages.reference_labels(cohort, "ACDC").reindex(keep)
        acc = paired_accuracy_difference(truth.to_numpy(), d0.loc[keep].to_numpy(), d1.to_numpy(), clusters=kw["resample_cluster"],
                                         n_boot=kw["n_boot"], seed=kw["seed"], level=kw["level"], min_valid_fraction=kw["min_valid_fraction"])
        shift_rows = {}
        for t in targets:
            v0 = pd.to_numeric(b0.loc[keep, t], errors="coerce").to_numpy(dtype=float)
            v1 = pd.to_numeric(b1.loc[keep, t], errors="coerce").to_numpy(dtype=float)
            both = np.isfinite(v0) & np.isfinite(v1)
            s = float(kw["reference_sd"].get(t, np.nan))
            shift_rows[t] = {"n": int(both.sum()), "mean_difference": float(np.mean(v1[both] - v0[both])) if both.any() else None,
                             "mean_abs_difference_over_fit_sd": float(np.mean(np.abs(v1[both] - v0[both])) / s) if both.any() and s > 0 else None}
        recorded_rho = json.loads(recorded[2].read_text(encoding="utf-8"))
        atomic_write_json(outputs[6], {
            "analysis": "S6_simulated_acquisition_shift", "analysis_label": REGISTERED_AFTER_UNBLINDING, "registration": "protocol-v6.0.1 item 15",
            "partition": cfg["partition"], "vendor": str(vendor), "vendor_name": summary[str(vendor)]["vendor_name"], "target_spacing_mm": list(target),
            "representation": "descriptors", "bundle_sha256": frozen["bundle_sha256"], "n_cases": len(keep),
            "n_formal_failures": int(sum(r["values"] is None for r in results)), "rho": rho, "accuracy": acc,
            "measurement_shift_B": shift_rows, "recorded_rho_acdc_test": recorded_rho.get("rho"),
            "comparison": "perturbed (A and B recomputed on the resampled images and masks) minus unperturbed (recorded evaluate:acdc_test)",
            **config_sha256_params(study)}, allow_nan=True)
        return {"vendor": str(vendor), "rows": len(keep), "delta_rho": rho.get("delta_rho"), "delta_accuracy": acc.get("delta_accuracy")}

    inputs = _bundle_inputs(ctx, run) + recorded + [stages._staging_manifest_path(ctx, "ACDC")]
    sections = ("registry", "cohorts", "reason_codes", "transition", "formal", "bootstrap.fixed_model_bootstrap", "bootstrap.primary_test")
    return record_stage(ctx, f"run-simulated-shift:{vendor}", compute, inputs, outputs, sections=sections, dc_section=DC_SECTION,
                        params={"vendor": str(vendor), "step": "descriptors", "target_spacing_mm": list(target)}, status="EVALUATED")


def tensors_stage(ctx: Any, cfg: Mapping[str, Any], vendor: str):
    """C3 tensor inputs of the perturbed ``acdc_test`` cases through the configured preprocessing hook."""

    from .. import stages

    run = ctx.require_run("run-simulated-shift")
    stages.verify_frozen_bundle(ctx)
    spec = str(cfg["tensor_inputs"]["hook"])
    hook = resolve_tensor_hook(spec)
    if hook is None:
        raise NotImplementedError(f"run-simulated-shift --step tensors: the C3 preprocessing hook {spec} is not importable "
                                  "(configs/data_centric.yaml simulated_shift.tensor_inputs.hook; signature fn(case, cfg) -> C3 arrays)")
    summary = json.loads((run.path / "simulated_shift" / "vendor_geometry.json").read_text(encoding="utf-8"))["vendors"]
    target = target_spacing(summary, vendor)
    cache = run.path / str(cfg["tensor_inputs"]["out"]).format(vendor=vendor)
    outputs = [cache / "manifest.csv", _vendor_dir(run, vendor) / "tensor_inputs.json"]

    def compute() -> dict[str, Any]:
        study = ctx.study
        split, case_ids, metadata = _test_cases(ctx, cfg)
        clusters = stages._cluster_ids(split, [f"ACDC:{c}" for c in case_ids])
        ncfg = _neural_input_config(study, str(cfg["tensor_inputs"]["config"]))
        rows = []
        for cid in case_ids:
            case = load_perturbed_case(cid, staged_root=str(ctx.staged_root), label_map=study.label_map("acdc"), metadata=metadata,
                                       target=target, vendor=str(vendor))
            rows.append(c3_entry(case.sample_id, hook(case, ncfg), cache, cluster_id=clusters.loc[case.sample_id], partition=str(cfg["partition"]),
                                 label=None, size=(ncfg.get("input") or {}).get("size_pixels") or (224, 224)))
        manifest = pd.DataFrame(rows, columns=C3_MANIFEST_COLUMNS).sort_values("sample_id")
        manifest.to_csv(outputs[0], index=False, lineterminator="\n")
        atomic_write_json(outputs[1], {"analysis": "S6_simulated_acquisition_shift", "analysis_label": REGISTERED_AFTER_UNBLINDING, "vendor": str(vendor),
                                       "target_spacing_mm": list(target), "hook": spec, "cache": run.portable(cache), "n_cases": len(rows),
                                       "labels": "not stored (evaluation partition)", **config_sha256_params(study)})
        return {"vendor": str(vendor), "cases": len(rows)}

    inputs = _bundle_inputs(ctx, run) + [stages._staging_manifest_path(ctx, "ACDC")]
    return record_stage(ctx, f"run-simulated-shift:{vendor}:tensors", compute, inputs, outputs, sections=("cohorts",), dc_section=DC_SECTION,
                        params={"vendor": str(vendor), "step": "tensors", "hook": spec, "target_spacing_mm": list(target)})


def _c4_decisions(frame: pd.DataFrame, flt: Mapping[str, Any], partition: str) -> pd.Series:
    keep = frame["partition"].astype(str) == partition if "partition" in frame.columns else pd.Series(True, index=frame.index)
    for col in ("model_id", "encoder_id"):
        keep &= frame[col].astype(str) == str(flt[col])
    keep &= pd.to_numeric(frame["seed"], errors="coerce").fillna(-1).astype(int) == int(flt["seed"])
    if "is_oof" in frame.columns:
        keep &= ~frame["is_oof"].astype(bool)
    return frame[keep].drop_duplicates("sample_id").set_index("sample_id")["y_pred"]


def _c4_rows(frame: pd.DataFrame, flt: Mapping[str, Any], partition: str) -> pd.DataFrame:
    """The rows of one encoder (``filter``) on ``partition`` of a C4 table, indexed by sample_id."""

    keep = (frame["partition"].astype(str) == partition) & ~frame["is_oof"].astype(bool)
    for col in ("model_id", "encoder_id"):
        keep &= frame[col].astype(str) == str(flt[col])
    keep &= pd.to_numeric(frame["seed"], errors="coerce").fillna(-1).astype(int) == int(flt["seed"])
    out = frame[keep].copy()
    out["sample_id"] = out["sample_id"].astype(str)
    return out.drop_duplicates("sample_id").set_index("sample_id")


def _rep_paths(run: Any, rep: Mapping[str, Any], vendor: str, seed: int | None = None) -> dict[str, Any]:
    """Run-root paths of a representation entry of ``configs/data_centric.yaml`` (``{vendor}`` and ``{seed}`` filled)."""

    def fill(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {k: fill(v) for k, v in value.items()}
        return run.path / str(value).format(vendor=vendor, seed=seed)

    return {k: fill(v) for k, v in rep.items() if k != "filter"}


def _require_recorded(run: Any, stage: str, path: Path, requester: str) -> Path:
    """``path`` must be an unchanged output of the recorded ``stage`` (its producer); returns that record's path."""

    from .. import stages

    record = run.read_record(stage)
    if record is None or run.portable(path) not in record.outputs:
        raise FileNotFoundError(f"{requester}: {run.portable(path)} is not a recorded output of {stage}; run that stage first (nothing is recorded)")
    stages._check_recorded_output(run, record, path, requester)
    return run.record_path(stage)


def _delta_payload(ctx: Any, cfg: Mapping[str, Any], vendor: str, representation: str, b1: pd.DataFrame, r0: pd.DataFrame, r1: pd.DataFrame,
                   p0: pd.Series, p1: pd.Series) -> dict[str, Any]:
    """Delta rho, delta rho_j and delta accuracy (perturbed minus unperturbed; paired cluster bootstrap)."""

    from .. import stages

    run = ctx.require_run("run-simulated-shift")
    split, case_ids, _ = _test_cases(ctx, cfg)
    b0 = pd.read_parquet(run.path / "semantics" / "B_ACDC.parquet").set_index("sample_id")
    ids = [f"ACDC:{c}" for c in case_ids]
    keep = [s for s in ids if s in b0.index and s in b1.index and s in r0.index and s in r1.index]
    kw = _rho_inputs(ctx, keep, split)
    fam = kw["family"]
    rho = paired_rho_difference(b0.loc[keep, fam], r0.reindex(keep).reindex(columns=fam), b1.loc[keep, fam], r1.reindex(keep).reindex(columns=fam),
                                kw["null_mean"], kw["reference_sd"], fam, cluster=kw["cluster"], resample_cluster=kw["resample_cluster"],
                                n_boot=kw["n_boot"], seed=kw["seed"], min_estimable=kw["min_estimable"], level=kw["level"],
                                min_valid_fraction=kw["min_valid_fraction"])
    cohort = stages._cohort(ctx, "ACDC").set_index("sample_id")
    truth = stages.reference_labels(cohort, "ACDC").reindex(ids)
    acc_kw = _rho_inputs(ctx, ids, split)
    acc = paired_accuracy_difference(truth.to_numpy(), p0.reindex(ids).to_numpy(), p1.reindex(ids).to_numpy(), clusters=acc_kw["resample_cluster"],
                                     n_boot=acc_kw["n_boot"], seed=acc_kw["seed"], level=acc_kw["level"], min_valid_fraction=acc_kw["min_valid_fraction"])
    return {"analysis": "S6_simulated_acquisition_shift", "analysis_label": REGISTERED_AFTER_UNBLINDING, "vendor": str(vendor),
            "representation": representation, "partition": cfg["partition"], "n_cases": len(keep), "rho": rho, "accuracy": acc,
            "comparison": "perturbed (the frozen representation on the resampled images) minus unperturbed (its recorded outputs on acdc_test)"}


def _d20_leg(ctx: Any, cfg: Mapping[str, Any], vendor: str, rep: Mapping[str, Any], device: str | None):
    """(inputs, outputs, compute) of ``--step d20``; compute returns (delta payload, extra record fields)."""

    from .. import stages
    from ..deep_encoder import d20_predictions, d20_reconstructions, load_d20_bundle
    from ..neural import image_stages
    from ..neural.preprocess import case_volumes

    run = ctx.require_run("run-simulated-shift")
    partition = str(cfg["partition"])
    paths = _rep_paths(run, rep, vendor)
    deep_dir = paths["freeze_record"].parent
    features0, features1 = paths["features"]["unperturbed"], paths["features"]["perturbed"]
    vdir = _vendor_dir(run, vendor)
    geometry = run.path / "simulated_shift" / "vendor_geometry.json"
    needed = [vdir / "B_acdc_test.parquet", paths["freeze_record"], features0, geometry]
    missing = [run.portable(q) for q in needed if not q.is_file()]
    if missing:
        raise FileNotFoundError(f"run-simulated-shift --vendor {vendor} --step d20: inputs not recorded yet {missing} (nothing is recorded; "
                                f"run --step descriptors, extract-deep-features --split {partition} and fit-deep-encoder first)")
    load_d20_bundle(deep_dir)                                     # the D20 freeze: every frozen file re-hashed
    produced = _require_recorded(run, f"extract-deep-features:{partition}", features0, "run-simulated-shift --step d20")
    weights = image_stages.d20_weights(ctx)
    failures1 = features1.with_name(features1.stem + "_failures.parquet")
    extractor1 = features1.with_name(features1.stem + "_extractor.json")
    outputs = [features1, failures1, extractor1, paths["reconstruction"]["unperturbed"], paths["reconstruction"]["perturbed"],
               paths["predictions"]["unperturbed"], paths["predictions"]["perturbed"], vdir / "delta_d20.json"]
    inputs = [*needed, produced, weights[0], stages._staging_manifest_path(ctx, "ACDC")]

    def compute() -> tuple[dict[str, Any], dict[str, Any]]:
        study = ctx.study
        ni_cfg = image_stages.neural_input_config(ctx)
        target = target_spacing(json.loads(geometry.read_text(encoding="utf-8"))["vendors"], vendor)
        split, case_ids, metadata = _test_cases(ctx, cfg)
        clusters = stages._cluster_ids(split, [f"ACDC:{c}" for c in case_ids])

        def loader(cid: str):
            case = load_perturbed_case(cid, staged_root=str(ctx.staged_root), label_map=study.label_map("acdc"), metadata=metadata,
                                       target=target, vendor=str(vendor))
            return dataclasses.replace(case_volumes(case, ni_cfg), cluster_id=str(clusters.loc[case.sample_id]))

        feats1, fails1, record = image_stages.d20_extract(ctx, [(f"ACDC:{c}", functools.partial(loader, c)) for c in case_ids], partition,
                                                          weights, device=device)
        stages._write_table(feats1, features1)
        stages._write_table(fails1, failures1)
        atomic_write_json(extractor1, {**record, "simulated_shift": {"vendor": str(vendor), "target_spacing_mm": list(target)}})
        tables = {}
        for cond, feats in (("unperturbed", pd.read_parquet(features0)), ("perturbed", feats1)):
            bhat = d20_reconstructions(deep_dir, feats)
            stages._write_table(bhat.reset_index(names="sample_id"), paths["reconstruction"][cond])
            preds = d20_predictions(deep_dir, feats, partition)
            stages._write_table(preds, paths["predictions"][cond])
            tables[cond] = (bhat, _c4_decisions(preds, rep["filter"], partition))
        b1 = pd.read_parquet(vdir / "B_acdc_test.parquet").set_index("sample_id")
        payload = _delta_payload(ctx, cfg, vendor, "d20", b1, tables["unperturbed"][0], tables["perturbed"][0], tables["unperturbed"][1],
                                 tables["perturbed"][1])
        return payload, {"d20_bundle_sha256": json.loads(paths["freeze_record"].read_text(encoding="utf-8")).get("bundle_sha256"),
                         "target_spacing_mm": list(target), "n_feature_failures": int(len(fails1)), "tta": record["tta"],
                         "environment": record["environment"]}

    return inputs, outputs, compute


def _f9_leg(ctx: Any, cfg: Mapping[str, Any], vendor: str, rep: Mapping[str, Any], device: str | None):
    """(inputs, outputs, compute) of ``--step f9``; compute returns (delta payload, extra record fields)."""

    from .. import stages
    from ..neural import image_stages
    from ..neural.inference import check_leakage, discover_checkpoints, run_f9_inference, select_encoders
    from ..neural.preprocess import read_cache_manifest
    from ..transition_fit import TransitionModel
    from .alt_bundles import activation_columns
    from .l57_stages import rel_path, verify_hash_manifest

    run = ctx.require_run("run-simulated-shift")
    partition = str(cfg["partition"])
    seed = int(rep["filter"]["seed"])
    paths = _rep_paths(run, rep, vendor, seed)
    vdir = _vendor_dir(run, vendor)
    cache = paths["cache"]
    needed = [vdir / "B_acdc_test.parquet", paths["freeze_record"], paths["probe_freeze_record"], paths["probe_transition"],
              paths["predictions"]["unperturbed"], paths["activations"]["unperturbed"], cache / "manifest.csv"]
    missing = [run.portable(q) for q in needed if not q.is_file()]
    if missing:
        raise FileNotFoundError(f"run-simulated-shift --vendor {vendor} --step f9: inputs not recorded yet {missing} (nothing is recorded; run "
                                f"freeze-f9, infer-f9 --split {partition}, fit-probes, --step descriptors and --step tensors first)")
    produced = [_require_recorded(run, f"infer-f9:{partition}", paths[k]["unperturbed"], "run-simulated-shift --step f9")
                for k in ("predictions", "activations")]
    produced.append(_require_recorded(run, f"run-simulated-shift:{vendor}:tensors", cache / "manifest.csv", "run-simulated-shift --step f9"))
    probe_freeze = verify_hash_manifest(run, rel_path(run, paths["probe_freeze_record"]), "run-simulated-shift --step f9")
    if rel_path(run, paths["probe_transition"]) not in (json.loads(paths["probe_freeze_record"].read_text(encoding="utf-8")).get("bundle") or {}):
        raise stages.BundleError(f"run-simulated-shift --step f9: {run.portable(paths['probe_transition'])} is not in the F9b freeze record")
    encoders = select_encoders(discover_checkpoints(run.path / image_stages.F9_DIR / "checkpoints"), ["final"], seeds=[seed])
    image_stages.check_frozen_encoders(run, encoders, json.loads(paths["freeze_record"].read_text(encoding="utf-8")))
    outputs = [paths["predictions"]["perturbed"], paths["activations"]["perturbed"], paths["predictions"]["perturbed"].with_name("inference.json"),
               paths["reconstruction"]["unperturbed"], paths["reconstruction"]["perturbed"], vdir / "delta_f9.json"]
    inputs = [*needed, *produced, *[e.path for e in encoders]]

    def compute() -> tuple[dict[str, Any], dict[str, Any]]:
        from ..neural.torch_numerics import configure_torch_numerics, environment_record, resolve_device
        from ..runs import code_version

        numerics = configure_torch_numerics()
        dev = resolve_device(device)
        rows = image_stages.partition_rows(ctx, partition)
        dev_rows = image_stages.partition_rows(ctx, image_stages.DEV)
        dev_folds = {str(s): int(f) for s, f in zip(dev_rows["sample_id"], dev_rows["outer_fold"])}
        check_leakage(encoders, partition, set(rows["sample_id"].astype(str)), dev_folds)
        y_true = image_stages._reference_labels(ctx, rows, partition, read_cache_manifest(cache))
        preds1, acts1, details = run_f9_inference(cache, encoders, partition=partition, device=dev, dev_folds=dev_folds, y_true=y_true)
        stages._write_table(preds1, paths["predictions"]["perturbed"])
        stages._write_table(acts1, paths["activations"]["perturbed"])
        atomic_write_json(outputs[2], {"schema": "tmmri.f9_inference/1", **details, "simulated_shift": {"vendor": str(vendor), "cache": run.portable(cache)},
                                       "environment": environment_record(dev), "numerics_at_start": numerics, "code": code_version()})
        model = TransitionModel.from_dict(json.loads(paths["probe_transition"].read_text(encoding="utf-8")))
        tables = {}
        for cond in ("unperturbed", "perturbed"):
            acts = _c4_rows(pd.read_parquet(paths["activations"][cond]), rep["filter"], partition)
            cols = activation_columns(acts.reset_index(), "a_")
            bhat = model.predict(acts[cols].to_numpy(dtype=np.float64), index=pd.Index(acts.index, name="sample_id"))
            stages._write_table(bhat.reset_index(names="sample_id"), paths["reconstruction"][cond])
            tables[cond] = (bhat, _c4_decisions(pd.read_parquet(paths["predictions"][cond]), rep["filter"], partition))
        b1 = pd.read_parquet(vdir / "B_acdc_test.parquet").set_index("sample_id")
        payload = _delta_payload(ctx, cfg, vendor, "f9", b1, tables["unperturbed"][0], tables["perturbed"][0], tables["unperturbed"][1],
                                 tables["perturbed"][1])
        return payload, {"f9_bundle_sha256": json.loads(paths["freeze_record"].read_text(encoding="utf-8")).get("bundle_sha256"),
                         "f9b_bundle": probe_freeze, "seed": seed, "encoders": [e.fit_id for e in encoders], "tta": details["tta"],
                         "environment": environment_record(dev)}

    return inputs, outputs, compute


def deep_stage(ctx: Any, cfg: Mapping[str, Any], vendor: str, representation: str, *, device: str | None = None):
    """Delta rho, delta rho_j and delta accuracy of D20 or F9: the frozen representation runs here on the perturbed inputs
    and is compared with its recorded outputs on the unperturbed ``acdc_test`` (module docstring)."""

    from .. import stages

    run = ctx.require_run("run-simulated-shift")
    frozen = stages.verify_frozen_bundle(ctx)
    stages._require_evaluated(run, "evaluate:acdc_test", "run-simulated-shift")
    rep = cfg["representations"][representation]
    inputs, outputs, inner = (_d20_leg if representation == "d20" else _f9_leg)(ctx, cfg, vendor, rep, device)
    out = outputs[-1]

    def compute() -> dict[str, Any]:
        payload, extra = inner()
        atomic_write_json(out, {**payload, "bundle_sha256": frozen["bundle_sha256"], **extra, "filter": rep["filter"],
                                "outputs": [run.portable(q) for q in outputs[:-1]], **config_sha256_params(ctx.study)}, allow_nan=True)
        return {"vendor": str(vendor), "representation": representation, "delta_rho": payload["rho"].get("delta_rho"),
                "delta_accuracy": payload["accuracy"].get("delta_accuracy")}

    inputs = [run.frozen_path, stages._contracts(ctx) / "split_manifest.parquet", stages._contracts(ctx) / "cohort_ACDC.parquet",
              run.path / "semantics" / "B_ACDC.parquet", *inputs]
    return record_stage(ctx, f"run-simulated-shift:{vendor}:{representation}", compute, list(dict.fromkeys(inputs)), outputs,
                        sections=("bootstrap.fixed_model_bootstrap", "bootstrap.primary_test", "transition.targets", "registry", "cohorts"),
                        dc_section=DC_SECTION, params={"vendor": str(vendor), "step": representation}, status="EVALUATED")


def run_simulated_shift(ctx: Any, vendor: str = "all", step: str = "descriptors", *, device: str | None = None) -> dict[str, Any]:
    """``run-simulated-shift --vendor A|B|C|D|all --step descriptors|tensors|d20|f9 [--device]`` (S6 on ``acdc_test``;
    ``device`` is the torch device of the d20 and f9 steps)."""

    if step not in STEPS:
        raise ValueError(f"run-simulated-shift --step must be one of {STEPS}")
    cfg = data_centric_config(ctx.study)[DC_SECTION]
    vendors = [str(v) for v in cfg["vendors"]] if vendor == "all" else [str(vendor)]
    unknown = [v for v in vendors if v not in {str(k) for k in cfg["vendors"]}]
    if unknown:
        raise ValueError(f"run-simulated-shift --vendor: unknown vendor codes {unknown}; configured {list(cfg['vendors'])}")
    geometry = geometry_stage(ctx, cfg)
    results = []
    for v in vendors:
        if step == "descriptors":
            res = descriptors_stage(ctx, cfg, v)
        elif step == "tensors":
            res = tensors_stage(ctx, cfg, v)
        else:
            res = deep_stage(ctx, cfg, v, step, device=device)
        results.append({"stage": res.stage, "action": res.action, "value": res.value, "status": res.record.status})
    return {"action": "RAN" if any(r["action"] == "RAN" for r in results) else "SKIPPED", "geometry": geometry.action, "step": step,
            "conditions": results, "analysis_label": REGISTERED_AFTER_UNBLINDING}
