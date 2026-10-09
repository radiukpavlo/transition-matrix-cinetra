"""F9 stages on a run root (protocol-v6.0.1; protocol 2.1 D item 10 and H; ``configs/f9_cnn.yaml``).

``plan-f9``       wave-1 selection specs: 6 fitting sets (v6 outer folds 1-5 and the final set of all 98 ``acdc_dev``
                  patients) x 3 seeds x 6 candidates x 3 selection folds = 324, plus the archived selection-fold
                  memberships (``neural_f9/plan/``).
``train-f9``      runs a spec file sequentially on this machine (``--backend local``; the Modal backend is
                  ``scripts/cloud/modal_train.py``); fits land in ``neural_f9/checkpoints/<fit_id>/``.
``select-f9``     applies the selection rule to the wave-1 records and writes the 18 wave-2 refit specs.
``import-f9-fits`` copies downloaded Modal fit directories into ``neural_f9/checkpoints/`` after SHA-256 checks
                  (never overwriting a different file) and logs every file in ``neural_f9/import_ledger.csv``.
``freeze-f9``     hashes every selected (refit) checkpoint and the selection evidence into ``neural_f9/f9_bundle.json``;
                  seed 20261011 is primary; there is no best-seed choice.  After the freeze the F9 fitting stages refuse.
                  Before anything is written it applies the technical-validity gate of ``configs/f9_cnn.yaml``
                  ``validity_gate`` to the primary seed (development out-of-fold macro-F1 from ``infer-f9 --split
                  acdc_dev`` >= 0.30, a falling training loss in every refit, the recorded nonzero-change check of every
                  selected fit): a pre-evaluation bug check, never a tuning loop.  A failure refuses with reason
                  ``PRE_EVALUATION_BUG_CHECK`` and writes ``neural_f9/validity_gate.json``.
``upload-cache``  CPE-1: uploads only ``neural_cache/acdc_dev`` (manifest and listed npz) to the Volume
                  ``tmmri-neural-cache`` with the pinned Modal client, after writing each file to ``cloud/modal_ledger.csv``.

Every stage is content addressed through :func:`tmmri.runs.run_stage` with the once-only guard of
:func:`tmmri.stages.once_only_guard`; its configuration hash is the SHA-256 of the parsed ``configs/f9_cnn.yaml``.
Only ``acdc_dev`` rows are read; nothing here touches an evaluation partition.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..runs import RunRoot, atomic_write_json, run_stage, sha256_file, sha256_json
from ..stages import BundleError, StageContext, once_only_guard
from .cloud import (
    append_ledger,
    cpe1_upload_set,
    ledger_path,
    modal_command,
    modal_settings,
    require_allowed_volume,
    require_client_pin,
    utc_now,
    volume_put_commands,
)
from .data import TRAINING_PARTITION, read_manifest
from .f9_config import (
    config_path,
    load_f9_config,
    resolve_weights,
    torch_hub_checkpoints,
    training_config,
    training_config_sha256,
)
from .fit_spec import FitSpec, candidates_from_config, read_spec_file, write_spec_file
from .selection import (
    NE_INSUFFICIENT_CLASS_SUPPORT,
    STATUS_COMPLETE,
    InsufficientClassSupport,
    fitting_set_hash,
    plan_wave1,
    plan_wave2,
    selection_folds,
)

F9_DIR = "neural_f9"
FIT_ID = re.compile(r"^[0-9a-f]{16}$")
BUNDLE_SCHEMA = "tmmri.f9_bundle/1"
IMPORT_LEDGER_COLUMNS = ("utc", "wave", "fit_id", "file", "bytes", "sha256", "source", "action")
VALIDITY_REASON = "PRE_EVALUATION_BUG_CHECK"
VALIDITY_SCHEMA = "tmmri.f9_validity_gate/1"
INFER_DEV_STAGE = f"infer-f9:{TRAINING_PARTITION}"


@dataclass(frozen=True)
class F9Paths:
    root: Path

    @property
    def plan(self) -> Path:
        return self.root / "plan"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def selection(self) -> Path:
        return self.root / "selection"

    @property
    def bundle(self) -> Path:
        return self.root / "f9_bundle.json"

    def wave_specs(self, wave: int) -> Path:
        return self.plan / f"wave{int(wave)}_specs.json"

    def fit_dir(self, fit_id: str) -> Path:
        return self.checkpoints / fit_id


def f9_paths(run: RunRoot) -> F9Paths:
    return F9Paths(run.path / F9_DIR)


def _config(ctx: StageContext) -> dict[str, Any]:
    return load_f9_config(ctx.study.root)


def _f9_stage(ctx: StageContext, name: str, fn: Callable[[], Any], inputs: Sequence[Path], outputs: Sequence[Path],
              cfg: Mapping[str, Any], params: Mapping[str, Any] | None = None):
    run = ctx.require_run(name)
    return run_stage(run, name, fn, inputs, outputs, config_hash=sha256_json(dict(cfg)), params=dict(params or {}),
                     force=ctx.force, n_jobs=1, rerun_guard=once_only_guard(run, name))


def require_f9_not_frozen(run: RunRoot, stage: str) -> None:
    if f9_paths(run).bundle.is_file():
        raise BundleError(f"{stage}: the F9 bundle of run {run.run_id} is frozen ({F9_DIR}/f9_bundle.json); F9 fitting stages need a new run id")


def _in_run(run: RunRoot, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    inside = run.path / path
    return inside if inside.exists() else Path.cwd() / path


def _cache_manifest(run: RunRoot) -> Path:
    return run.path / "neural_cache" / TRAINING_PARTITION / "manifest.csv"


# ----------------------------------------------------------------------------- plan-f9


def _development_rows(run: RunRoot, classes: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_path = run.path / "contracts" / "split_manifest.parquet"
    split = pd.read_parquet(split_path, columns=["sample_id", "partition", "outer_fold", "cluster_id"])
    dev = split[split["partition"] == TRAINING_PARTITION].copy()
    dev["sample_id"] = dev["sample_id"].astype(str)
    dev["cluster_id"] = dev["cluster_id"].astype(str)
    if dev.empty or dev["outer_fold"].isna().any():
        raise ValueError(f"{split_path}: acdc_dev rows without outer folds")
    manifest = read_manifest(_cache_manifest(run))
    if set(manifest["partition"]) != {TRAINING_PARTITION}:
        raise ValueError(f"{_cache_manifest(run)} must list only {TRAINING_PARTITION} rows")
    if set(manifest["sample_id"]) != set(dev["sample_id"]):
        diff = sorted(set(manifest["sample_id"]) ^ set(dev["sample_id"]))
        raise ValueError(f"the tensor cache and the split manifest disagree on acdc_dev membership (e.g. {diff[:5]})")
    bad = sorted(manifest.loc[~manifest["label"].isin(classes), "sample_id"])
    if bad:
        raise ValueError(f"acdc_dev cache rows without a class label in {list(classes)}: {bad[:5]}")
    merged = dev.merge(manifest[["sample_id", "cluster_id", "label"]], on="sample_id", suffixes=("", "_cache"))
    if not (merged["cluster_id"] == merged["cluster_id_cache"]).all():
        raise ValueError("cluster ids of the tensor cache differ from the split manifest")
    return dev, merged[["sample_id", "cluster_id", "label", "outer_fold"]].sort_values("sample_id").reset_index(drop=True)


def fitting_sets(dev_rows: pd.DataFrame, cfg: Mapping[str, Any]) -> dict[int | str, list[str]]:
    sets: dict[int | str, list[str]] = {}
    for fold in cfg["fitting_sets"]["outer_folds"]:
        sets[int(fold)] = sorted(dev_rows.loc[dev_rows["outer_fold"].astype(int) != int(fold), "sample_id"])
    sets["final"] = sorted(dev_rows["sample_id"])
    return sets


def plan_f9(ctx: StageContext):
    run = ctx.require_run("plan-f9")
    require_f9_not_frozen(run, "plan-f9")
    cfg = _config(ctx)
    paths = f9_paths(run)
    weights = resolve_weights(cfg, ctx.study.root)
    split_path, manifest_path = run.path / "contracts" / "split_manifest.parquet", _cache_manifest(run)
    outputs = [paths.wave_specs(1), paths.plan / "selection_folds.parquet"]

    def compute() -> dict[str, Any]:
        classes = list(cfg["class_order"])
        _, rows = _development_rows(run, classes)
        sets = fitting_sets(rows, cfg)
        n_folds = int(cfg["selection"]["n_folds"])
        specs = plan_wave1(sets, seeds=[int(s) for s in cfg["seeds"]["values"]], candidates=candidates_from_config(cfg), n_folds=n_folds)
        if len(specs) != int(cfg["fit_counts"]["wave1_selection"]):
            raise ValueError(f"plan-f9 produced {len(specs)} wave-1 specs, configs/f9_cnn.yaml expects {cfg['fit_counts']['wave1_selection']}")
        folds, not_estimable = [], []
        for key, ids in sets.items():
            members = rows[rows["sample_id"].isin(ids)]
            try:
                part = selection_folds(members, n_folds=n_folds, classes=classes, salt=str(cfg["selection"]["salt"]))
            except InsufficientClassSupport as exc:
                not_estimable.append({"fitting_set": str(key), "status": NE_INSUFFICIENT_CLASS_SUPPORT, "reason": str(exc)})
                continue
            folds.append(part.assign(fitting_set=str(key), fitting_set_hash=fitting_set_hash(ids)))
        table = pd.concat(folds, ignore_index=True) if folds else pd.DataFrame(columns=["sample_id", "cluster_id", "label", "selection_fold"])
        paths.plan.mkdir(parents=True, exist_ok=True)
        table.to_parquet(outputs[1], index=False)
        table.to_csv(outputs[1].with_suffix(".csv"), index=False, lineterminator="\n")
        write_spec_file(outputs[0], specs, run_id=run.run_id, wave=1, config=training_config(cfg),
                        cache_manifest_sha256=sha256_file(manifest_path), weights={k: weights[k] for k in ("name", "sha256", "file", "url")},
                        extra={"fitting_sets": {str(k): v for k, v in sets.items()}, "not_estimable": not_estimable})
        return {"wave1_specs": len(specs), "fitting_set_sizes": {str(k): len(v) for k, v in sets.items()},
                "not_estimable": not_estimable, "spec_file": str(outputs[0])}

    return _f9_stage(ctx, "plan-f9", compute, [split_path, manifest_path, config_path(ctx.study.root)], outputs, cfg)


# ----------------------------------------------------------------------------- train-f9 (local backend)


def local_weights(payload: Mapping[str, Any]) -> dict[str, Any]:
    weights = dict(payload["weights"])
    return {"name": weights["name"], "sha256": weights["sha256"], "path": str(torch_hub_checkpoints() / str(weights["file"]))}


def train_f9(ctx: StageContext, specs: str | Path, *, backend: str = "local", device: str = "auto"):
    if backend != "local":
        raise ValueError("train-f9 runs the local backend only; the Modal backend is scripts/cloud/modal_train.py")
    run = ctx.require_run("train-f9")
    cfg = _config(ctx)
    paths = f9_paths(run)
    spec_path = _in_run(run, specs)
    payload = read_spec_file(spec_path)
    if payload.get("purpose", "f9") == "f9":          # S5 spec files (purpose s5) train after the F9 freeze; their fits never enter it
        require_f9_not_frozen(run, "train-f9")
    if payload["run_id"] != run.run_id:
        raise ValueError(f"{spec_path} was planned for run {payload['run_id']}, not {run.run_id}")
    if payload["config_sha256"] != sha256_json(training_config(cfg)):
        raise ValueError(f"{spec_path}: its training configuration differs from configs/f9_cnn.yaml; re-plan")
    fit_specs: list[FitSpec] = payload["fit_specs"]
    outputs = [paths.fit_dir(s.fit_id) / "record.json" for s in fit_specs]
    weights = local_weights(payload)

    def compute() -> dict[str, Any]:
        from .train import train_fit

        statuses: Counter[str] = Counter()
        for spec in fit_specs:
            record = train_fit(spec, run.path / "neural_cache", paths.fit_dir(spec.fit_id), device, config=payload["config"],
                               weights=weights, expected_manifest_sha256=payload["cache_manifest_sha256"],
                               backend={"name": "local", "order": "sequential", "spec_file": spec_path.name})
            statuses[record.status] += 1
        return {"fits": len(fit_specs), "statuses": dict(statuses)}

    name = f"train-f9:{spec_path.stem}"
    return _f9_stage(ctx, name, compute, [spec_path, _cache_manifest(run)], outputs, cfg, {"backend": backend, "device": device})


# ----------------------------------------------------------------------------- select-f9


def _load_records(paths: F9Paths, payload: Mapping[str, Any], *, required: bool = True) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    missing = []
    for spec in payload["fit_specs"]:
        path = paths.fit_dir(spec.fit_id) / "record.json"
        if not path.is_file():
            missing.append(spec.fit_id)
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("fit_id") != spec.fit_id or record.get("spec") != spec.to_dict():
            raise ValueError(f"{path}: the record does not belong to spec {spec.fit_id}")
        if record.get("config_sha256") != payload["config_sha256"]:
            raise ValueError(f"{path}: fitted with another training configuration")
        records[spec.fit_id] = record
    if missing and required:
        raise FileNotFoundError(f"{len(missing)} of {len(payload['fit_specs'])} fit records are missing under {paths.checkpoints} "
                                f"(e.g. {missing[:3]}); train or import them first")
    return records


def select_f9(ctx: StageContext, specs: str | Path = f"{F9_DIR}/plan/wave1_specs.json"):
    run = ctx.require_run("select-f9")
    require_f9_not_frozen(run, "select-f9")
    cfg = _config(ctx)
    paths = f9_paths(run)
    spec_path = _in_run(run, specs)
    payload = read_spec_file(spec_path)
    if int(payload["wave"]) != 1:
        raise ValueError(f"{spec_path} is not a wave-1 spec file")
    records = _load_records(paths, payload)
    outputs = [paths.wave_specs(2), paths.selection / "selection_table.parquet", paths.selection / "selection_summary.json"]

    def compute() -> dict[str, Any]:
        refits, table, not_estimable = plan_wave2(payload["fit_specs"], records, n_folds=int(cfg["selection"]["n_folds"]))
        groups = len({s.group_key for s in payload["fit_specs"]})
        if payload["purpose"] == "f9" and not not_estimable and len(refits) != int(cfg["fit_counts"]["wave2_refit"]):
            raise ValueError(f"select-f9 produced {len(refits)} refit specs, configs/f9_cnn.yaml expects {cfg['fit_counts']['wave2_refit']}")
        paths.selection.mkdir(parents=True, exist_ok=True)
        table.to_parquet(outputs[1], index=False)
        table.drop(columns=[c for c in ("macro_f1_folds", "cross_entropy_folds", "selected_epochs") if c in table.columns]).to_csv(
            outputs[1].with_suffix(".csv"), index=False, lineterminator="\n")
        write_spec_file(outputs[0], refits, run_id=run.run_id, wave=2, config=payload["config"],
                        cache_manifest_sha256=payload["cache_manifest_sha256"], weights=payload["weights"], purpose=payload["purpose"],
                        extra={"wave1_specs_sha256": sha256_file(spec_path), "not_estimable": not_estimable})
        summary = {"groups": groups, "refits": len(refits), "not_estimable": not_estimable,
                   "winners": [{"outer_fold": s.outer_fold, "seed": s.seed, "candidate": s.candidate.label, "refit_epochs": s.refit_epochs,
                                "fit_id": s.fit_id} for s in refits]}
        atomic_write_json(outputs[2], summary)
        return summary

    inputs = [spec_path] + [paths.fit_dir(fid) / "record.json" for fid in sorted(records)]
    return _f9_stage(ctx, "select-f9", compute, inputs, outputs, cfg)


# ----------------------------------------------------------------------------- import-f9-fits


def _copy_verified(src: Path, dest: Path, digest: str) -> str:
    """Copy ``src`` to ``dest`` atomically; return ``imported``/``identical``; refuse a different existing file."""

    if dest.is_file():
        if sha256_file(dest) == digest:
            return "identical"
        raise BundleError(f"{dest} exists with other content; an imported fit file is never overwritten")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    shutil.copyfile(src, tmp)
    if sha256_file(tmp) != digest:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"{src}: the copy's SHA-256 differs (read error)")
    os.replace(tmp, dest)
    return "imported"


def import_f9_fits(ctx: StageContext, source: str | Path, *, wave: int, specs: str | Path | None = None):
    """Copy verified Modal fits into ``neural_f9/checkpoints``.  ``specs`` names another spec file of the run (the S5 spec
    files ``data_efficiency/f9/train_wave<N>_specs.json``); the default is the F9 plan of ``wave``."""

    run = ctx.require_run("import-f9-fits")
    cfg = _config(ctx)
    paths = f9_paths(run)
    spec_path = _in_run(run, specs) if specs else paths.wave_specs(wave)
    payload = read_spec_file(spec_path)
    if payload.get("purpose", "f9") == "f9":
        require_f9_not_frozen(run, "import-f9-fits")
    if int(payload["wave"]) != int(wave):
        raise ValueError(f"{spec_path} is a wave-{payload['wave']} spec file, not wave {wave}")
    planned = {s.fit_id: s for s in payload["fit_specs"]}
    known = set(planned)
    siblings = [paths.wave_specs(1), paths.wave_specs(2)]
    if specs:                                           # the other wave of the same spec family (wave1 <-> wave2 in the name)
        siblings += [spec_path.with_name(spec_path.name.replace(f"wave{int(wave)}", f"wave{other}")) for other in (1, 2) if other != int(wave)]
    for sibling in siblings:
        if sibling.is_file() and sibling.resolve() != spec_path.resolve():
            known |= {s.fit_id for s in read_spec_file(sibling)["fit_specs"]}
    src_root = Path(source).resolve()
    if not src_root.is_dir():
        raise FileNotFoundError(f"{src_root} is not a directory")
    present = {d.name: d for d in src_root.iterdir() if d.is_dir() and FIT_ID.fullmatch(d.name)}
    unknown = sorted(set(present) - known)
    if unknown:
        raise ValueError(f"{len(unknown)} fit directories in {src_root} belong to no planned wave (e.g. {unknown[:3]})")
    missing = sorted(set(planned) - set(present))
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {len(planned)} wave-{wave} fits are missing in {src_root} (e.g. {missing[:3]})")
    outputs = [paths.fit_dir(fid) / "record.json" for fid in sorted(planned)]

    def compute() -> dict[str, Any]:
        verified: list[tuple[str, Path, list[tuple[str, Path, str]]]] = []   # every fit is checked before any file is copied
        for fid in sorted(planned):
            folder = present[fid]
            record_path = folder / "record.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            spec = planned[fid]
            if record.get("fit_id") != fid or FitSpec.from_dict(record["spec"]).fit_id != fid or record["spec"] != spec.to_dict():
                raise ValueError(f"{record_path}: fit id or spec does not match the plan")
            if record.get("config_sha256") != payload["config_sha256"]:
                raise ValueError(f"{record_path}: fitted with another training configuration")
            if record.get("status") not in (STATUS_COMPLETE, NE_INSUFFICIENT_CLASS_SUPPORT):
                raise ValueError(f"{record_path}: status {record.get('status')} is not importable")
            files = [("record.json", record_path, sha256_file(record_path))]
            model_path = folder / "model.pt"
            if record["status"] == STATUS_COMPLETE:
                if model_path.is_file():
                    digest = sha256_file(model_path)
                    if digest != record.get("checkpoint_sha256"):
                        raise ValueError(f"{model_path}: SHA-256 {digest} differs from its record ({record.get('checkpoint_sha256')})")
                    files.append(("model.pt", model_path, digest))
                elif not spec.is_selection_fit:
                    raise FileNotFoundError(f"{model_path} missing: refit checkpoints are required")
            verified.append((fid, folder, files))
        rows, actions = [], Counter()
        for fid, folder, files in verified:
            for name, path, digest in sorted(files, key=lambda f: f[0] != "model.pt"):   # model first, record last
                action = _copy_verified(path, paths.fit_dir(fid) / name, digest)
                actions[action] += 1
                rows.append({"utc": utc_now(), "wave": int(wave), "fit_id": fid, "file": name, "bytes": path.stat().st_size,
                             "sha256": digest, "source": str(folder), "action": action})
        ledger = paths.root / "import_ledger.csv"
        frame = pd.DataFrame(rows, columns=list(IMPORT_LEDGER_COLUMNS))
        frame.to_csv(ledger, mode="a", header=not ledger.is_file(), index=False, lineterminator="\n")
        return {"wave": int(wave), "fits": len(planned), "files": dict(actions), "ledger": str(ledger)}

    name = f"import-f9-fits:wave{int(wave)}" if not specs else f"import-f9-fits:{spec_path.stem}"
    return _f9_stage(ctx, name, compute, [spec_path], outputs, cfg, {"source": str(src_root)})


# ----------------------------------------------------------------------------- freeze-f9: technical-validity gate


class ValidityGateError(BundleError):
    """freeze-f9 refused by the technical-validity gate (reason ``PRE_EVALUATION_BUG_CHECK``); ``report`` is the written report."""

    def __init__(self, message: str, report: Mapping[str, Any]):
        super().__init__(message)
        self.reason = VALIDITY_REASON
        self.report = dict(report)


def oof_macro_f1(predictions: pd.DataFrame, classes: Sequence[str], *, seed: int, dev_ids: Sequence[str]) -> dict[str, Any]:
    """Development out-of-fold macro-F1 of one seed: the ``is_oof`` ``acdc_dev`` rows of the fold encoders (contract C4),
    which must cover every development patient exactly once; macro-F1 over all classes (a missing class scores 0)."""

    from sklearn.metrics import f1_score

    classes = [str(c) for c in classes]
    frame = predictions[(predictions["partition"].astype(str) == TRAINING_PARTITION) & (predictions["model_id"].astype(str) == "f9")
                        & (predictions["seed"].astype(int) == int(seed)) & predictions["is_oof"].astype(bool)]
    counts = frame["sample_id"].astype(str).value_counts()
    dev = {str(s) for s in dev_ids}
    missing, repeated, foreign = sorted(dev - set(counts.index)), sorted(counts.index[counts > 1]), sorted(set(counts.index) - dev)
    truth, pred = frame["y_true"].astype(object).to_numpy(), frame["y_pred"].astype(object).to_numpy()
    known = np.array([t is not None and not pd.isna(t) and str(t) in classes for t in truth], dtype=bool)
    out: dict[str, Any] = {"n_rows": int(len(frame)), "n_dev_patients": len(dev), "missing": missing[:10], "n_missing": len(missing),
                           "repeated": repeated[:10], "n_repeated": len(repeated), "not_development": foreign[:10], "n_not_development": len(foreign),
                           "n_without_reference": int((~known).sum()), "classes": classes, "chance_level": 1.0 / len(classes)}
    if not known.any():
        return {**out, "macro_f1": None}
    score = f1_score([str(t) for t in truth[known]], [str(p) for p in pred[known]], labels=classes, average="macro", zero_division=0)
    return {**out, "macro_f1": float(score)}


def training_loss_check(record: Mapping[str, Any]) -> dict[str, Any]:
    """(ii) The refit's ``train_loss`` of its final epoch is below that of epoch 1 (fewer than two epochs fail)."""

    epochs = sorted((e for e in record.get("epochs") or [] if isinstance(e, Mapping)), key=lambda e: int(e.get("epoch", 0)))
    losses = [float(e["train_loss"]) if e.get("train_loss") is not None else float("nan") for e in epochs]
    ok = len(losses) >= 2 and bool(np.all(np.isfinite(losses))) and losses[-1] < losses[0]
    return {"passed": bool(ok), "n_epochs": len(losses), "epoch_1": losses[0] if losses else None, "final_epoch": losses[-1] if losses else None,
            "reason": None if ok else ("fewer_than_two_epochs" if len(losses) < 2 else "non_finite_loss" if not np.all(np.isfinite(losses))
                                       else "loss_did_not_decrease")}


def parameter_change_check(record: Mapping[str, Any]) -> dict[str, Any]:
    """(iii) The training check of :func:`tmmri.neural.train._parameter_checks` as recorded: a nonzero change in a
    convolution of every unfrozen block, frozen weights and BN running statistics unchanged."""

    params = record.get("parameters") or {}
    changes = {str(k): float(v) for k, v in (params.get("unfrozen_conv_max_abs_change") or {}).items()}
    ok = bool(changes) and min(changes.values()) > 0.0 and params.get("bn_running_statistics_unchanged") is True \
        and params.get("frozen_weights_unchanged") is True
    return {"passed": bool(ok), "unfrozen_conv_max_abs_change": changes, "bn_running_statistics_unchanged": params.get("bn_running_statistics_unchanged"),
            "frozen_weights_unchanged": params.get("frozen_weights_unchanged"),
            "reason": None if ok else ("not_recorded" if not changes else "zero_change_or_frozen_parameters_changed")}


def validity_gate(gate: Mapping[str, Any], *, classes: Sequence[str], primary: int, refits: Mapping[str, Mapping[str, Any]],
                  selected_fits: Mapping[str, Mapping[str, Any]], predictions: pd.DataFrame, dev_ids: Sequence[str]) -> dict[str, Any]:
    """Checks (i)-(iii) of ``configs/f9_cnn.yaml`` ``validity_gate`` for the primary seed (``refits`` and ``selected_fits``:
    fit id -> record).  Returns the report; ``passed`` is the conjunction.  Nothing here may change a hyperparameter."""

    threshold = float(gate["min_oof_macro_f1"])
    oof = oof_macro_f1(predictions, classes, seed=primary, dev_ids=dev_ids)
    coverage = oof["n_missing"] == 0 and oof["n_repeated"] == 0 and oof["n_not_development"] == 0 and oof["n_without_reference"] == 0
    f1_ok = oof["macro_f1"] is not None and coverage and oof["macro_f1"] >= threshold
    losses = {fid: training_loss_check(rec) for fid, rec in sorted(refits.items())}
    changes = {fid: parameter_change_check(rec) for fid, rec in sorted({**refits, **selected_fits}.items())}
    checks = [
        {"check": "oof_macro_f1", "passed": bool(f1_ok), "threshold": threshold, "observed": oof["macro_f1"], "coverage_complete": bool(coverage),
         "detail": oof},
        {"check": "refit_training_loss_decreased", "passed": all(c["passed"] for c in losses.values()) and bool(losses), "rule": str(gate["training_loss"]),
         "failed_fits": sorted(f for f, c in losses.items() if not c["passed"]), "fits": losses},
        {"check": "nonzero_parameter_change", "passed": all(c["passed"] for c in changes.values()) and bool(changes), "rule": str(gate["parameter_change"]),
         "failed_fits": sorted(f for f, c in changes.items() if not c["passed"]), "fits": changes},
    ]
    return {"schema": VALIDITY_SCHEMA, "seed": int(primary), "passed": all(c["passed"] for c in checks), "checks": checks,
            "failed_checks": [c["check"] for c in checks if not c["passed"]], "role": "pre-evaluation bug check; never a tuning loop"}


def _oof_predictions(run: RunRoot, paths: F9Paths, gate: Mapping[str, Any], primary_refits: Sequence[FitSpec], primary: int) -> pd.DataFrame:
    """The recorded ``infer-f9:acdc_dev`` predictions, produced by exactly the primary seed's refit encoders."""

    path = run.path / str(gate["oof_predictions"])
    record = run.read_record(INFER_DEV_STAGE)
    if record is None or not path.is_file():
        raise FileNotFoundError(f"freeze-f9: the technical-validity gate needs the development out-of-fold predictions ({run.portable(path)}); "
                                f"run infer-f9 --split {TRAINING_PARTITION} before freeze-f9")
    if record.outputs.get(run.portable(path)) != sha256_file(path):
        raise BundleError(f"freeze-f9: {run.portable(path)} differs from the output recorded by {INFER_DEV_STAGE}")
    used = {str(e) for e in record.params.get("encoders") or [] if str(e).startswith(f"{primary}:")}
    expected = {f"{primary}:{'final' if s.outer_fold == 'final' else f'fold{s.outer_fold}'}:{s.fit_id}" for s in primary_refits}
    if used != expected:
        raise BundleError(f"freeze-f9: {INFER_DEV_STAGE} did not use exactly the primary seed's refits (recorded {sorted(used)[:3]}, expected "
                          f"{sorted(expected)[:3]}); rerun infer-f9 --split {TRAINING_PARTITION} on the current refits")
    return pd.read_parquet(path)


def _selected_fits(wave1: Mapping[str, Any], primary_refits: Sequence[FitSpec]) -> list[FitSpec]:
    """The selection fits of each winning candidate (same seed, fitting set and candidate as a refit)."""

    keys = {(s.seed, str(s.outer_fold), s.fitting_patients, s.candidate) for s in primary_refits}
    return [s for s in wave1["fit_specs"] if (s.seed, str(s.outer_fold), s.fitting_patients, s.candidate) in keys]


def run_validity_gate(run: RunRoot, paths: F9Paths, cfg: Mapping[str, Any], *, wave1: Mapping[str, Any], wave2: Mapping[str, Any],
                      records1: Mapping[str, Mapping[str, Any]], records2: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Run the gate and write ``neural_f9/validity_gate.json``; refuse with :class:`ValidityGateError` when it fails."""

    gate = cfg["validity_gate"]
    primary = int(cfg["seeds"]["primary"])
    report_path = run.path / str(gate["on_failure"]["report"])
    current = training_config_sha256(cfg)
    if wave1.get("config_sha256") != current:
        raise BundleError("freeze-f9: configs/f9_cnn.yaml differs from the training configuration of the wave-1 plan; a hyperparameter or "
                          "a value of the technical-validity gate never changes after planning (NEURAL-TM-1.0 pre-evaluation bug rule)")
    if report_path.is_file():
        prior = json.loads(report_path.read_text(encoding="utf-8"))
        if prior.get("passed") is False and prior.get("training_config_sha256") != current:
            raise BundleError(f"freeze-f9: {run.portable(report_path)} records a {VALIDITY_REASON} failure under another training configuration; "
                              "a failure never changes hyperparameters")
    refit_specs = [s for s in wave2["fit_specs"] if int(s.seed) == primary]
    selected = _selected_fits(wave1, refit_specs)
    predictions = _oof_predictions(run, paths, gate, refit_specs, primary)
    dev_rows, _ = _development_rows(run, list(cfg["class_order"]))
    report = validity_gate(gate, classes=list(cfg["class_order"]), primary=primary, refits={s.fit_id: records2[s.fit_id] for s in refit_specs},
                           selected_fits={s.fit_id: records1[s.fit_id] for s in selected}, predictions=predictions, dev_ids=list(dev_rows["sample_id"]))
    report.update({"run_id": run.run_id, "training_config_sha256": current, "gate": dict(gate),
                   "oof_predictions": {"path": run.portable(run.path / str(gate["oof_predictions"])),
                                       "sha256": sha256_file(run.path / str(gate["oof_predictions"])), "stage": INFER_DEV_STAGE},
                   "refits": sorted(s.fit_id for s in refit_specs), "selected_fits": sorted(s.fit_id for s in selected)})
    if not report["passed"]:
        report.update({"reason": VALIDITY_REASON, "rule": str(gate["on_failure"]["rule"]),
                       "affected_development_artifacts": [*(f"{F9_DIR}/checkpoints/{fid}/" for fid in sorted({*report["refits"], *report["selected_fits"]})),
                                                          report["oof_predictions"]["path"], f"{F9_DIR}/selection/", f"{F9_DIR}/plan/wave2_specs.json"]})
    paths.root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report_path, report, allow_nan=True)
    if not report["passed"]:
        raise ValidityGateError(f"freeze-f9 refused: {VALIDITY_REASON} ({', '.join(report['failed_checks'])}); report {run.portable(report_path)}. "
                                f"{gate['on_failure']['rule']}", report)
    return report


# ----------------------------------------------------------------------------- freeze-f9


def freeze_f9(ctx: StageContext):
    run = ctx.require_run("freeze-f9")
    cfg = _config(ctx)
    paths = f9_paths(run)
    wave1, wave2 = read_spec_file(paths.wave_specs(1)), read_spec_file(paths.wave_specs(2))
    if wave2.get("wave1_specs_sha256") != sha256_file(paths.wave_specs(1)):
        raise BundleError("the wave-2 specs were not derived from the current wave-1 spec file")
    records1 = _load_records(paths, wave1)
    records2 = _load_records(paths, wave2)
    table_path = paths.selection / "selection_table.parquet"
    primary = int(cfg["seeds"]["primary"])
    gate_path = run.path / str(cfg["validity_gate"]["oof_predictions"])
    report_path = run.path / str(cfg["validity_gate"]["on_failure"]["report"])

    def compute() -> dict[str, Any]:
        # Technical-validity gate (SPECIFICATION.md 8.6): before anything is frozen; a failure refuses and writes the report.
        gate = run_validity_gate(run, paths, cfg, wave1=wave1, wave2=wave2, records1=records1, records2=records2)
        fits = []
        for spec in wave2["fit_specs"]:
            record = records2[spec.fit_id]
            if record.get("status") != STATUS_COMPLETE:
                raise BundleError(f"refit {spec.fit_id} is {record.get('status')}; the bundle needs every refit")
            model = paths.fit_dir(spec.fit_id) / "model.pt"
            if not model.is_file() or sha256_file(model) != record.get("checkpoint_sha256"):
                raise BundleError(f"{model} is missing or differs from its record")
            fits.append({"fit_id": spec.fit_id, "outer_fold": spec.outer_fold, "seed": spec.seed, "primary": spec.seed == primary,
                         "candidate": spec.candidate.to_dict(), "refit_epochs": spec.refit_epochs,
                         "encoder_id": "final" if spec.outer_fold == "final" else f"fold{spec.outer_fold}",
                         "checkpoint": f"{F9_DIR}/checkpoints/{spec.fit_id}/model.pt", "checkpoint_sha256": record["checkpoint_sha256"],
                         "record_sha256": sha256_file(paths.fit_dir(spec.fit_id) / "record.json"),
                         "backend": record.get("backend", {}).get("name"), "gpu_name": record.get("environment", {}).get("gpu_name")})
        wave1_lines = "\n".join(f"{fid}\t{sha256_file(paths.fit_dir(fid) / 'record.json')}" for fid in sorted(records1))
        body = {"schema": BUNDLE_SCHEMA, "run_id": run.run_id, "model_id": "f9", "protocol_tag": cfg.get("protocol_tag"),
                "training_config_sha256": wave1["config_sha256"], "cache_manifest_sha256": wave1["cache_manifest_sha256"],
                "weights": wave1["weights"], "primary_seed": primary, "seeds": [int(s) for s in cfg["seeds"]["values"]],
                "seed_selection": "forbidden", "ensemble": False,
                "wave1_specs_sha256": sha256_file(paths.wave_specs(1)), "wave2_specs_sha256": sha256_file(paths.wave_specs(2)),
                "selection_table_sha256": sha256_file(table_path), "wave1_records_sha256": sha256_json(wave1_lines),
                "n_wave1_records": len(records1), "not_estimable": wave2.get("not_estimable", []),
                "validity_gate": {"passed": gate["passed"], "seed": gate["seed"], "report": run.portable(report_path), "report_sha256": sha256_file(report_path),
                                  "oof_macro_f1": gate["checks"][0]["observed"], "oof_predictions_sha256": gate["oof_predictions"]["sha256"]},
                "fits": sorted(fits, key=lambda f: (str(f["outer_fold"]), f["seed"]))}
        bundle = {**body, "bundle_sha256": sha256_json(body)}
        if paths.bundle.is_file():
            prior = json.loads(paths.bundle.read_text(encoding="utf-8"))
            if prior.get("bundle_sha256") != bundle["bundle_sha256"]:
                raise BundleError(f"{paths.bundle} is frozen with other content (once-only)")
        else:
            atomic_write_json(paths.bundle, bundle)
        return {"bundle_sha256": bundle["bundle_sha256"], "fits": len(fits), "primary_fits": sum(f["primary"] for f in fits),
                "validity_gate": {"passed": gate["passed"], "oof_macro_f1": gate["checks"][0]["observed"]}}

    inputs = [paths.wave_specs(1), paths.wave_specs(2), table_path] + [paths.fit_dir(fid) / "record.json" for fid in sorted(records2)]
    inputs += [p for p in (gate_path, run.record_path(INFER_DEV_STAGE)) if p.exists()]       # absent: the gate refuses (missing input)
    return _f9_stage(ctx, "freeze-f9", compute, inputs, [paths.bundle, report_path], cfg)


# ----------------------------------------------------------------------------- upload-cache (CPE-1)


def _run_checked(command: list[str]) -> None:
    subprocess.run(command, check=True)


def upload_cache(ctx: StageContext, *, files: Sequence[str | Path] | None = None, dry_run: bool = False, create_volume: bool = False,
                 runner: Callable[[list[str]], Any] | None = None):
    """CPE-1 upload of ``neural_cache/acdc_dev`` (see :mod:`tmmri.neural.cloud`)."""

    run = ctx.require_run("upload-cache")
    cfg = _config(ctx)
    modal_cfg = modal_settings(cfg["backend"])
    items = cpe1_upload_set(run.path, requested=files)          # refuses anything outside CPE-1 before anything else
    pin = require_client_pin(modal_cfg)
    volume = require_allowed_volume(str(modal_cfg["cache_volume"]), modal_cfg)
    remote_dir = str(modal_cfg["cache_remote_prefix"]).replace("<run_id>", run.run_id)
    commands = volume_put_commands(pin, volume, items, remote_dir)
    summary = {"volume": volume, "remote_dir": remote_dir, "files": len(items), "bytes": int(sum(i["bytes"] for i in items)),
               "client": f"modal=={pin}"}
    if dry_run:
        return {**summary, "dry_run": True, "commands": [" ".join(c) for c in commands]}
    ledger = ledger_path(run.path, modal_cfg)
    execute = runner or _run_checked
    base = f"neural_cache/{TRAINING_PARTITION}"

    def compute() -> dict[str, Any]:
        if create_volume:
            command = modal_command(pin, "volume", "create", volume)
            append_ledger(ledger, [{"utc": utc_now(), "action": "create_volume", "volume": volume, "run_id": run.run_id,
                                    "client": f"modal=={pin}", "note": "CPE-1 named Volume"}])
            execute(command)
        for item, command in zip(items, commands):
            row = {"utc": utc_now(), "action": "upload", "volume": volume, "remote_path": command[-1], "local_path": f"{base}/{item['relpath']}",
                   "bytes": item["bytes"], "sha256": item["sha256"], "run_id": run.run_id, "client": f"modal=={pin}",
                   "note": "CPE-1 acdc_dev tensor cache"}
            append_ledger(ledger, [row])
            try:
                execute(command)
            except Exception as exc:
                append_ledger(ledger, [{**row, "utc": utc_now(), "action": "upload_failed", "note": str(exc)[:200]}])
                raise
        return {**summary, "ledger": str(ledger)}

    return _f9_stage(ctx, "upload-cache", compute, [item["path"] for item in items], [ledger], cfg,
                     {"volume": volume, "remote_dir": remote_dir, "client": f"modal=={pin}"})


def record_volume_deletion(ctx: StageContext, volume: str) -> dict[str, Any]:
    """Ledger entry for a Volume the owner deleted (CPE-1 (5)); calls nothing on Modal."""

    run = ctx.require_run("upload-cache")
    modal_cfg = modal_settings(_config(ctx)["backend"])
    require_allowed_volume(volume, modal_cfg)
    ledger = append_ledger(ledger_path(run.path, modal_cfg), [{"utc": utc_now(), "action": "delete_volume", "volume": volume,
                                                               "remote_path": "/", "run_id": run.run_id,
                                                               "note": "owner deleted the Volume after the verified local backup"}])
    return {"recorded": "delete_volume", "volume": volume, "ledger": str(ledger)}
