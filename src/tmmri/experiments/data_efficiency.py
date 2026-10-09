"""S5 data-efficiency curves (manuscript protocol 2.1, Section E; ``configs/data_centric.yaml`` ``data_efficiency``).

Development only: every fit and every evaluation uses ``acdc_dev`` rows, so no held-out partition is exposed.  For each
v6 outer fold k the fitting set is ``acdc_dev`` minus fold k; a fraction f of it is drawn per ACDC class by ranking
``SHA256("lc-v6.0.1|" + f + "|" + r + "|" + patient_id)`` (f as Python float repr, repeat r 0-based) and keeping the
first ``floor(f * n_c + 0.5)`` members of each class.  Both pipelines fit on identical subsamples:

* ``run-data-efficiency --step descriptor`` (local CPU): the v6 transition (:func:`tmmri.transition_fit.fit_transition_v6`,
  per-target lambda by inner CV over the v6 inner folds restricted to the subsample) and the audited predictor
  (:func:`tmmri.classification.fit_formal_predictor`) per (f, r, k); out-of-fold rho (fit-partition mean and SD as in
  ``run-ablations``) and macro-F1 on the held-out outer folds.  With f = 1.0 this reproduces the v6 nested OOF B-hat
  and the cross-fitted d_model exactly.
* ``--step f9-wave1`` writes the 540 C5 selection fit specs (purpose ``s5``, primary seed, 2 fractions x 3 repeats x 5
  folds x 6 candidates x 3 selection folds); ``--step f9-wave2`` reads their FitRecords, applies the NEURAL-TM-1.0 4.2
  selection rule and writes the 30 refit specs (median-epoch refit); ``--step f9-aggregate`` reads the refits' C4
  out-of-fold predictions and activations and reports macro-F1 and the F9b-style probe rho on the held-out folds.

A class with fewer members than ``class_support.min_per_class`` in a subsample gives ``NE_INSUFFICIENT_CLASS_SUPPORT``.
Research use only; synthetic fixtures are algorithm controls, never cohort results.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .data_centric import (
    NE_INSUFFICIENT_CLASS_SUPPORT,
    NOT_EXECUTED,
    DevelopmentOnlyError,
    config_sha256_params,
    data_centric_config,
    record_stage,
)
from ..runs import atomic_write_bytes, atomic_write_json, hash_path, sha256_json

DC_SECTION = "data_efficiency"
STEPS = ("descriptor", "f9-wave1", "f9-wave2", "f9-infer", "f9-aggregate")
SPEC_FIELDS = ("purpose", "seed", "outer_fold", "fitting_patients", "selection_fold", "candidate", "refit_epochs", "fraction", "repeat")
UNFROZEN_BLOCKS = {"layer4": 1, "layer3+layer4": 2}


# ----------------------------------------------------------------------------- subsampling (shared by both pipelines)


def fraction_token(fraction: float) -> str:
    """The fraction as it enters the hash: Python float repr (``0.5``, ``0.75``, ``1.0``)."""

    return repr(float(fraction))


def subsample_digest(salt: str, fraction: float, repeat: int, key: str) -> str:
    return hashlib.sha256(f"{salt}{fraction_token(fraction)}|{int(repeat)}|{key}".encode("utf-8")).hexdigest()


def per_class_count(fraction: float, n: int) -> int:
    """``floor(f * n + 0.5)`` (round half up), at most ``n``."""

    return min(int(n), int(math.floor(float(fraction) * int(n) + 0.5)))


def repeats_for(fraction: float, cfg: Mapping[str, Any]) -> list[int]:
    rep = cfg["repeats"]
    return list(range(int(rep["at_one"] if float(fraction) >= 1.0 else rep["below_one"])))


def class_stratified_subsample(frame: pd.DataFrame, *, fraction: float, repeat: int, salt: str, key: str = "patient_id",
                               strata: str = "acdc_class") -> pd.DataFrame:
    """Rank the rows of ``frame`` within each stratum by ``subsample_digest`` (ties by key) and select the first
    :func:`per_class_count` of each.  Returns ``frame`` sorted by key with ``rank`` (0-based, within stratum),
    ``digest`` and ``selected``; the result does not depend on the row order of ``frame``."""

    if frame[key].duplicated().any():
        raise ValueError(f"duplicate {key} values in the subsampling frame")
    out = frame.copy()
    out["digest"] = [subsample_digest(salt, fraction, repeat, str(v)) for v in out[key]]
    out["rank"] = -1
    out["selected"] = False
    for _, group in out.groupby(strata, dropna=False, sort=True):
        order = sorted(group.index, key=lambda i: (out.at[i, "digest"], str(out.at[i, key])))
        m = per_class_count(fraction, len(order))
        for rank, index in enumerate(order):
            out.at[index, "rank"] = rank
            out.at[index, "selected"] = rank < m
    return out.sort_values(key).reset_index(drop=True)


def development_frame(split: pd.DataFrame, cohort: pd.DataFrame, partition: str = "acdc_dev") -> pd.DataFrame:
    """``sample_id, patient_id, acdc_class, outer_fold`` of the eligible development rows (split manifest)."""

    dev = split[split["partition"] == partition][["sample_id", "patient_id", "outer_fold"]].copy()
    classes = cohort.set_index("sample_id")["acdc_class"]
    dev["acdc_class"] = classes.reindex(dev["sample_id"]).to_numpy()
    if dev["outer_fold"].isna().any() or dev["acdc_class"].isna().any():
        raise ValueError(f"{partition} rows need an outer fold and an ACDC class")
    dev["outer_fold"] = dev["outer_fold"].astype(int)
    return dev.sort_values("sample_id").reset_index(drop=True)


def assert_development_only(ids: Iterable[str], split: pd.DataFrame, cfg: Mapping[str, Any]) -> None:
    """Refuse any identifier that is not an ``acdc_dev`` row of the split manifest (no test or external row is reachable)."""

    part = split.set_index("sample_id")["partition"]
    ids = [str(i) for i in ids]
    unknown = [i for i in ids if i not in part.index]
    outside = sorted({str(part[i]) for i in ids if i in part.index and part[i] != cfg["partition"]})
    if unknown or outside:
        raise DevelopmentOnlyError(f"S5 is development only: {len(unknown)} unknown identifiers {unknown[:3]} and rows of partitions {outside} "
                                   f"(forbidden: {cfg['forbidden_partitions']})")


def subsample_table(dev: pd.DataFrame, cfg: Mapping[str, Any], fractions: Sequence[float] | None = None) -> pd.DataFrame:
    """Membership of every (fraction, repeat, outer fold): one row per fitting-set member with ``selected``."""

    sub = cfg["subsample"]
    rows = []
    for f in (cfg["fractions"] if fractions is None else fractions):
        for r in repeats_for(f, cfg):
            for k in sorted(dev["outer_fold"].unique()):
                fitting = dev[dev["outer_fold"] != k]
                ranked = class_stratified_subsample(fitting, fraction=float(f), repeat=r, salt=str(sub["salt"]), key=str(sub["key"]),
                                                    strata=str(sub["strata"]))
                rows.append(ranked.assign(fraction=float(f), repeat=int(r), heldout_fold=int(k)))
    cols = ["fraction", "repeat", "heldout_fold", "sample_id", "acdc_class", "outer_fold", "rank", "digest", "selected"]
    return pd.concat(rows, ignore_index=True)[cols]


def class_support(classes_of_rows: Iterable[str], classes: Sequence[str], min_per_class: int) -> dict[str, Any]:
    counts = pd.Series(list(classes_of_rows), dtype=object).value_counts()
    per = {c: int(counts.get(c, 0)) for c in classes}
    short = [c for c, n in per.items() if n < int(min_per_class)]
    return {"class_counts": per, "min_per_class": int(min_per_class), "ok": not short, "short_classes": short,
            "status": "EVALUATED" if not short else "NE", "reason": None if not short else NE_INSUFFICIENT_CLASS_SUPPORT}


def _macro_f1(truth: Sequence[str], pred: Sequence[Any]) -> float | None:
    from sklearn.metrics import f1_score

    t = pd.Series(list(truth), dtype=object)
    p = pd.Series(list(pred), dtype=object)
    keep = t.notna().to_numpy() & p.map(lambda v: v is not None and str(v) not in ("", "nan", "None")).to_numpy()
    if not keep.any():
        return None
    t, p = t[keep].astype(str), p[keep].astype(str)
    return float(f1_score(t, p, labels=sorted(set(t)), average="macro", zero_division=0))


# ----------------------------------------------------------------------------- descriptor pipeline


def descriptor_curves(ctx: Any, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Fit the v6 transition and audited predictor per (f, r, k) on the subsample; pooled OOF rho and macro-F1."""

    from .. import stages
    from ..classification import fit_formal_predictor
    from ..formal_features import formal_descriptor_names
    from ..transition_fit import fit_transition_v6

    run = ctx.require_run("run-data-efficiency")
    study = ctx.study
    split = stages._split_manifest(ctx)
    dev = development_frame(split, stages._cohort(ctx, "ACDC"), cfg["partition"])
    assert_development_only(dev["sample_id"], split, cfg)
    subs = subsample_table(dev, cfg)
    targets = study.transition_targets()
    classes = [str(c) for c in cfg["class_support"]["classes"]]
    min_class = int(cfg["class_support"]["min_per_class"]["descriptor"])
    # Transition rows: acdc_dev fit rows (critical QC removed), exactly as fit-transition (tmmri.stages._dev_matrices).
    ids, a, b, folds = stages._dev_matrices(ctx)
    assert_development_only(ids["sample_id"], split, cfg)
    inner_fn = stages._inner_fold_fn(ctx, ids)
    # Predictor rows: acdc_dev A rows; fit rows are finite and not critical QC, exactly as fit-predictor.
    a_all = pd.read_parquet(run.path / "formal" / "A_ACDC.parquet").merge(dev[["sample_id", "outer_fold", "acdc_class"]], on="sample_id")
    a_all = a_all.sort_values("sample_id").reset_index(drop=True)
    x = a_all[formal_descriptor_names()].to_numpy(dtype=np.float64)
    y = a_all["acdc_class"].astype(str).reset_index(drop=True)
    b_all = pd.read_parquet(run.path / "semantics" / "B_ACDC.parquet").set_index("sample_id")
    critical_ids = set(b_all.index[stages.critical_qc_mask(b_all, study).to_numpy()])
    pred_fit_rows = np.isfinite(x).all(axis=1) & ~a_all["sample_id"].isin(critical_ids).to_numpy()
    pcfg = study.formal["audited_predictor"]
    sd, mean = stages._dev_reference(ctx)
    b_dev = b.set_index("sample_id").loc[ids["sample_id"], targets]
    bhat_rows, pred_rows, unit_rows, curves = [], [], [], []
    for (f, r), group in subs.groupby(["fraction", "repeat"], sort=True):
        oof = pd.DataFrame(np.nan, index=ids["sample_id"].to_numpy(), columns=targets)
        decisions = pd.Series([None] * len(a_all), index=a_all["sample_id"].to_numpy(), dtype=object)
        proba = pd.DataFrame(np.nan, index=a_all["sample_id"].to_numpy(), columns=classes)
        unit_status = []
        for k, members in group.groupby("heldout_fold", sort=True):
            chosen = set(members.loc[members["selected"], "sample_id"])
            assert_development_only(chosen, split, cfg)
            train_pos = np.flatnonzero((folds != int(k)) & ids["sample_id"].isin(chosen).to_numpy())
            held_pos = np.flatnonzero(folds == int(k))
            model = fit_transition_v6(a[train_pos], b.iloc[train_pos], targets, study.transition, study.registry,
                                      np.asarray(inner_fn(int(k), train_pos)), fit_mask=np.ones(train_pos.size, dtype=bool))
            oof.iloc[held_pos, :] = model.predict(a[held_pos]).to_numpy()
            fit_mask = pred_fit_rows & a_all["sample_id"].isin(chosen).to_numpy() & (a_all["outer_fold"] != int(k)).to_numpy()
            support = class_support(y[fit_mask], classes, min_class)
            held = (a_all["outer_fold"] == int(k)).to_numpy()
            fold_f1 = None
            if support["ok"]:
                predictor = fit_formal_predictor(x, y, pcfg, fit_mask=fit_mask)
                labels = stages.predict_labels(predictor, x[held])
                decisions.iloc[np.flatnonzero(held)] = labels
                p = predictor.predict_proba(x[held])
                for c in p.columns:
                    proba.iloc[np.flatnonzero(held), classes.index(str(c))] = p[c].to_numpy()
                fold_f1 = _macro_f1(y[held], labels)
            unit_status.append(support["ok"])
            unit_rows.append({"fraction": float(f), "repeat": int(r), "heldout_fold": int(k), "n_fitting_set": int(len(members)),
                              "n_subsample": int(len(chosen)), "n_transition_fit_rows": int(model.n_fit_rows),
                              "n_predictor_fit_rows": int(fit_mask.sum()), "class_counts": json.dumps(support["class_counts"], sort_keys=True),
                              "fitted_targets": json.dumps(model.fitted_targets), "lambdas": json.dumps(model.lambdas, sort_keys=True),
                              "macro_f1_heldout_fold": fold_f1, "status": support["status"], "reason": support["reason"]})
        bhat_rows.append(oof.reset_index(names="sample_id").assign(fraction=float(f), repeat=int(r)))
        pred_rows.append(pd.DataFrame({"sample_id": a_all["sample_id"].to_numpy(), "outer_fold": a_all["outer_fold"].to_numpy(),
                                       "y_true": y.to_numpy(), "d_model": decisions.to_numpy(), "fraction": float(f), "repeat": int(r),
                                       **{f"prob_{c}": proba[c].to_numpy() for c in classes}}))
        rho = stages._rho(ctx, b_dev, oof, mean, sd)
        point: dict[str, Any] = {"fraction": float(f), "repeat": int(r), "rho": rho}
        if all(unit_status):
            point["macro_f1"] = {"estimate": _macro_f1(y, decisions.to_numpy()), "status": "EVALUATED", "reason": None,
                                 "n": int(decisions.notna().sum())}
        else:
            point["macro_f1"] = {"estimate": None, "status": "NE", "reason": NE_INSUFFICIENT_CLASS_SUPPORT,
                                 "folds": [u["heldout_fold"] for u in unit_rows if u["fraction"] == float(f) and u["repeat"] == int(r) and u["status"] == "NE"]}
        curves.append(point)
    return {"subsamples": subs, "bhat": pd.concat(bhat_rows, ignore_index=True), "predictions": pd.concat(pred_rows, ignore_index=True),
            "units": pd.DataFrame(unit_rows), "curves": curves, "targets": targets}


def summarize_curves(points: Sequence[Mapping[str, Any]], key: str = "rho") -> list[dict[str, Any]]:
    """Per fraction: mean, minimum and maximum over repeats of the point estimates (repeat variability, descriptive)."""

    out = []
    for f in sorted({float(p["fraction"]) for p in points}):
        vals = [p[key].get("rho" if key == "rho" else "estimate") for p in points if float(p["fraction"]) == f]
        finite = [float(v) for v in vals if v is not None and np.isfinite(v)]
        out.append({"fraction": f, "n_repeats": len(vals), "n_estimable": len(finite), "mean": float(np.mean(finite)) if finite else None,
                    "min": min(finite) if finite else None, "max": max(finite) if finite else None})
    return out


# ----------------------------------------------------------------------------- F9 fit specs (C5)


def _fit_spec_module():
    try:
        return importlib.import_module("tmmri.neural.fit_spec")
    except ImportError:
        return None


def compute_fit_id(fields: Mapping[str, Any]) -> str:
    """First 16 hex of SHA-256 of the canonical JSON (:func:`tmmri.runs.canonical_json`) of every C5 field but fit_id."""

    return sha256_json({k: fields[k] for k in SPEC_FIELDS})[:16]


def make_fit_spec(fields: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """A C5 fit spec as a plain dict and its source: ``tmmri.neural.fit_spec`` (owned by the F9 lane) when importable,
    else the C5 JSON written exactly (:func:`compute_fit_id`)."""

    clean = {k: fields[k] for k in SPEC_FIELDS}
    module = _fit_spec_module()
    if module is not None:
        cls = getattr(module, "FitSpec", None)
        for build in (getattr(cls, "create", None), getattr(module, "make_fit_spec", None), cls):
            if build is None:
                continue
            try:
                spec = build(**clean)
            except TypeError:
                continue
            data = spec.to_dict() if hasattr(spec, "to_dict") else dataclasses.asdict(spec) if dataclasses.is_dataclass(spec) else dict(spec)
            if data.get("fit_id"):
                return dict(data), "tmmri.neural.fit_spec"
    return {"fit_id": compute_fit_id(clean), **clean}, "c5_json"


def _candidates(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"unfreeze": str(c["unfreeze"]), "lr_backbone": float(c["lr_backbone"])} for c in cfg["f9"]["candidates"]]


def f9_units(dev: pd.DataFrame, cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every (fraction, repeat, outer fold) of the F9 path with its subsample and class-support check."""

    f9 = cfg["f9"]
    subs = subsample_table(dev, cfg, fractions=[float(f) for f in f9["fractions"]])
    min_class = int(cfg["class_support"]["min_per_class"]["f9"])
    classes = [str(c) for c in cfg["class_support"]["classes"]]
    units = []
    for (f, r, k), members in subs.groupby(["fraction", "repeat", "heldout_fold"], sort=True):
        if int(k) not in {int(v) for v in f9["outer_folds"]}:
            continue
        chosen = members[members["selected"]]
        units.append({"fraction": float(f), "repeat": int(r), "outer_fold": int(k), "fitting_patients": sorted(chosen["sample_id"].astype(str)),
                      **class_support(chosen["acdc_class"].astype(str), classes, min_class)})
    return units


def wave1_specs(dev: pd.DataFrame, cfg: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Selection fit specs (6 candidates x 3 selection folds per supported unit) and the unit ledger."""

    f9 = cfg["f9"]
    specs, source = [], "c5_json"
    units = f9_units(dev, cfg)
    for unit in units:
        if not unit["ok"]:
            continue
        for cand in _candidates(cfg):
            for s in range(1, int(f9["selection_folds"]) + 1):
                spec, source = make_fit_spec({"purpose": str(f9["purpose"]), "seed": int(f9["seed"]), "outer_fold": unit["outer_fold"],
                                              "fitting_patients": unit["fitting_patients"], "selection_fold": s, "candidate": cand,
                                              "refit_epochs": None, "fraction": unit["fraction"], "repeat": unit["repeat"]})
                specs.append(spec)
    return specs, units, source


def _dotted(record: Mapping[str, Any], path: str) -> Any:
    value: Any = record
    for part in str(path).split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def select_candidate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """NEURAL-TM-1.0 4.2: highest mean macro-F1, then lowest mean cross entropy, then fewer unfrozen backbone blocks,
    then lower backbone learning rate.  ``rows``: one per selection fit with candidate, macro_f1, cross_entropy, epoch."""

    table = pd.DataFrame([{"unfreeze": r["candidate"]["unfreeze"], "lr_backbone": float(r["candidate"]["lr_backbone"]),
                           "macro_f1": float(r["macro_f1"]), "cross_entropy": float(r["cross_entropy"]), "epoch": int(r["epoch"])} for r in rows])
    agg = table.groupby(["unfreeze", "lr_backbone"], sort=True).agg(mean_macro_f1=("macro_f1", "mean"), mean_cross_entropy=("cross_entropy", "mean"),
                                                                       n_folds=("epoch", "size"), epochs=("epoch", lambda e: sorted(int(v) for v in e)))
    agg = agg.reset_index()
    agg["blocks"] = agg["unfreeze"].map(UNFROZEN_BLOCKS)
    agg = agg.sort_values(["mean_macro_f1", "mean_cross_entropy", "blocks", "lr_backbone"], ascending=[False, True, True, True], kind="mergesort")
    best = agg.iloc[0]
    return {"candidate": {"unfreeze": str(best["unfreeze"]), "lr_backbone": float(best["lr_backbone"])},
            "refit_epochs": int(math.ceil(float(np.median(best["epochs"])))), "selected_epochs": list(best["epochs"]),
            "table": agg.drop(columns=["blocks"]).assign(epochs=lambda d: d["epochs"].map(json.dumps)).to_dict("records")}


def wave2_specs(wave1: Sequence[Mapping[str, Any]], records: Mapping[str, Mapping[str, Any]], cfg: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Refit specs (one per unit) from the wave-1 FitRecords; every selection fit of a unit must have a record."""

    fields = cfg["f9"]["record_fields"]
    by_unit: dict[tuple, list[dict[str, Any]]] = {}
    for spec in wave1:
        by_unit.setdefault((float(spec["fraction"]), int(spec["repeat"]), int(spec["outer_fold"])), []).append(dict(spec))
    specs, ledger, source = [], [], "c5_json"
    missing: list[str] = []
    for unit, members in sorted(by_unit.items()):
        rows = []
        for spec in members:
            rec = records.get(spec["fit_id"])
            values = None if rec is None else {k: _dotted(rec, fields[k]) for k in ("macro_f1", "cross_entropy", "epoch")}
            if values is None or any(v is None or (isinstance(v, float) and not np.isfinite(v)) for v in values.values()):
                missing.append(spec["fit_id"])
                continue
            rows.append({"candidate": spec["candidate"], **values})
        if len(rows) != len(members):
            continue
        choice = select_candidate(rows)
        spec, source = make_fit_spec({"purpose": members[0]["purpose"], "seed": int(members[0]["seed"]), "outer_fold": unit[2],
                                      "fitting_patients": list(members[0]["fitting_patients"]), "selection_fold": None,
                                      "candidate": choice["candidate"], "refit_epochs": choice["refit_epochs"], "fraction": unit[0],
                                      "repeat": unit[1]})
        specs.append(spec)
        ledger.append({"fraction": unit[0], "repeat": unit[1], "outer_fold": unit[2], "refit_fit_id": spec["fit_id"], **choice})
    if missing:
        raise FileNotFoundError(f"run-data-efficiency --step f9-wave2: {len(missing)} wave-1 selection fits have no usable FitRecord "
                                f"(e.g. {missing[:3]}); import the wave-1 records first (refits need every selection fit)")
    return specs, ledger, source


def train_spec_prerequisites(run: Any, study: Any) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """(acdc_dev cache manifest, F9 training configuration, pinned backbone weights) of an S5 spec file; raises before
    anything is recorded, so a once-only S5 step never locks without its train-ready spec file."""

    from ..neural.f9_config import load_f9_config, resolve_weights

    manifest = run.path / "neural_cache" / "acdc_dev" / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"{manifest} missing: run build-tensor-cache --split acdc_dev first (S5 F9 spec files pin its SHA-256)")
    f9cfg = load_f9_config(study.root)
    return manifest, f9cfg, resolve_weights(f9cfg, study.root)


def write_train_spec_file(path: Path, specs: Sequence[Mapping[str, Any]], *, run_id: str, wave: int, manifest: Path,
                          f9cfg: Mapping[str, Any], weights: Mapping[str, Any], source: str) -> dict[str, Any]:
    """Integration of L89 and L4b (C5): the S5 specs as an F9 spec file (``tmmri.neural.fit_spec.write_spec_file``,
    purpose ``s5``), the format ``train-f9 --backend local``, ``import-f9-fits --specs`` and ``scripts/cloud/modal_train.py``
    consume."""

    from ..neural.f9_config import training_config
    from ..neural.fit_spec import FitSpec, write_spec_file
    from ..runs import sha256_file

    return write_spec_file(path, [FitSpec.from_dict(s) for s in specs], run_id=run_id, wave=int(wave), config=training_config(f9cfg),
                           cache_manifest_sha256=sha256_file(manifest), weights={k: weights[k] for k in ("name", "sha256", "file", "url")},
                           purpose="s5", extra={"analysis": "S5_data_efficiency", "source": source})


def _spec_index(specs: Sequence[Mapping[str, Any]], wave: int) -> pd.DataFrame:
    return pd.DataFrame([{"fit_id": s["fit_id"], "wave": int(wave), "purpose": s["purpose"], "seed": int(s["seed"]), "fraction": float(s["fraction"]),
                          "repeat": int(s["repeat"]), "outer_fold": s["outer_fold"], "selection_fold": s["selection_fold"],
                          "unfreeze": s["candidate"]["unfreeze"], "lr_backbone": float(s["candidate"]["lr_backbone"]),
                          "refit_epochs": s["refit_epochs"], "n_fitting_patients": len(s["fitting_patients"])} for s in specs])


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    data = "".join(json.dumps(r, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for r in rows)
    return atomic_write_bytes(path, data.encode("utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


# ----------------------------------------------------------------------------- F9 aggregation (C4 outputs)


def _c4_frame(directory: Path, name: str, *, k: int, seed: int, split: pd.DataFrame, cfg: Mapping[str, Any], strict: bool) -> pd.DataFrame | None:
    """The ``fold<k>`` F9 rows of a C4 file restricted to ``acdc_dev``.  ``strict`` (the S5 refit outputs) refuses a file
    that names any other partition or identifier; otherwise (the F9 primary outputs, which also hold evaluation rows of
    other encoders) only development rows are kept, so no test or external row is ever read into S5."""

    path = Path(directory) / name
    if not path.is_file():
        return None
    frame = pd.read_parquet(path)
    if strict:
        assert_development_only(frame["sample_id"], split, cfg)
        if "partition" in frame.columns and set(frame["partition"].astype(str)) - {str(cfg["partition"])}:
            raise DevelopmentOnlyError(f"S5 F9 outputs {path} name partitions other than {cfg['partition']}")
    dev_ids = set(split.loc[split["partition"] == cfg["partition"], "sample_id"].astype(str))
    keep = frame["sample_id"].astype(str).isin(dev_ids)
    if "partition" in frame.columns:
        keep &= frame["partition"].astype(str) == str(cfg["partition"])
    keep &= (frame["model_id"].astype(str) == "f9") & (frame["encoder_id"].astype(str) == f"fold{int(k)}")
    keep &= pd.to_numeric(frame["seed"], errors="coerce").fillna(-1).astype(int) == int(seed)
    return frame[keep].copy()


def probe_rho_inputs(acts: pd.DataFrame, fitting: Sequence[str], heldout: Sequence[str]) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    cols = sorted(c for c in acts.columns if c.startswith("a_"))
    table = acts.drop_duplicates("sample_id").set_index("sample_id")
    fit_ids = [s for s in fitting if s in table.index]
    held_ids = [s for s in heldout if s in table.index]
    return table.loc[fit_ids, cols].to_numpy(float), fit_ids, table.loc[held_ids, cols].to_numpy(float), held_ids


def aggregate_f9(ctx: Any, cfg: Mapping[str, Any], refits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pooled OOF macro-F1 (C4 predictions) and F9b-style probe rho (C4 activations) per (fraction, repeat)."""

    from .. import stages
    from ..transition_fit import fit_transition_v6

    run = ctx.require_run("run-data-efficiency")
    study = ctx.study
    f9 = cfg["f9"]
    split = stages._split_manifest(ctx)
    dev = development_frame(split, stages._cohort(ctx, "ACDC"), cfg["partition"])
    truth = dev.set_index("sample_id")["acdc_class"].astype(str)
    targets = study.transition_targets()
    b_all = pd.read_parquet(run.path / "semantics" / "B_ACDC.parquet")
    b_all = b_all[b_all["sample_id"].isin(set(dev["sample_id"])) & ~stages.critical_qc_mask(b_all, study)].set_index("sample_id")
    inner = pd.read_parquet(stages._contracts(ctx) / "inner_folds.parquet")
    sd, mean = stages._dev_reference(ctx)
    units = [{"fraction": float(s["fraction"]), "repeat": int(s["repeat"]), "outer_fold": int(s["outer_fold"]), "fit_id": s["fit_id"],
              "fitting_patients": list(s["fitting_patients"]), "dir": run.path / str(f9["oof_outputs"]).format(fit_id=s["fit_id"])} for s in refits]
    full_dir = run.path / str(f9["full_fraction_outputs"])
    for k in sorted(dev["outer_fold"].unique()):
        units.append({"fraction": 1.0, "repeat": 0, "outer_fold": int(k), "fit_id": None,
                      "fitting_patients": sorted(dev.loc[dev["outer_fold"] != k, "sample_id"]), "dir": full_dir})
    unit_rows, curves, pred_rows = [], [], []
    grouped: dict[tuple[float, int], list[dict[str, Any]]] = {}
    for unit in units:
        grouped.setdefault((unit["fraction"], unit["repeat"]), []).append(unit)
    for (f, r), group in sorted(grouped.items()):
        decisions = pd.Series([None] * len(dev), index=dev["sample_id"].to_numpy(), dtype=object)
        oof = pd.DataFrame(np.nan, index=list(b_all.index), columns=targets)
        complete_pred, complete_probe = True, True
        for unit in group:
            k = int(unit["outer_fold"])
            held_ids = sorted(dev.loc[dev["outer_fold"] == k, "sample_id"])
            strict = unit["fit_id"] is not None          # S5 refit outputs are S5-only files; the F9 primary file is shared
            pred, acts = (_c4_frame(unit["dir"], name, k=k, seed=int(f9["seed"]), split=split, cfg=cfg, strict=strict)
                          for name in ("predictions.parquet", "activations.parquet"))
            row = {"fraction": float(f), "repeat": int(r), "outer_fold": k, "fit_id": unit["fit_id"], "outputs": run.portable(unit["dir"])}
            if pred is None:
                complete_pred = False
                row.update({"macro_f1_heldout_fold": None, "status": NOT_EXECUTED, "reason": "INPUTS_NOT_RECORDED"})
            else:
                sel = pred[pred["is_oof"].astype(bool)].drop_duplicates("sample_id").set_index("sample_id")
                got = [s for s in held_ids if s in sel.index]
                decisions.loc[got] = sel.loc[got, "y_pred"].astype(str).to_numpy()
                row.update({"n_heldout_predicted": len(got), "macro_f1_heldout_fold": _macro_f1(truth.loc[got], sel.loc[got, "y_pred"]),
                            "status": "EVALUATED" if len(got) == len(held_ids) else "NE",
                            "reason": None if len(got) == len(held_ids) else "HELDOUT_FOLD_INCOMPLETE"})
                complete_pred &= len(got) == len(held_ids)
            if acts is None:
                complete_probe = False
                row["probe"] = "INPUTS_NOT_RECORDED"
            else:
                # C4: encoder fold<k> has is_oof=False rows for its fitting patients and is_oof=True rows for held-out fold k
                # only, so the probe is fitted on the former and decodes the latter (as fit_f9b_probes with encoder_matrix).
                oof_rows = acts["is_oof"].astype(bool)
                fit_pool = [s for s in unit["fitting_patients"] if s in b_all.index]
                a_fit, fit_ids, _, _ = probe_rho_inputs(acts[~oof_rows], fit_pool, [])
                _, _, a_held, held_got = probe_rho_inputs(acts[oof_rows], [], [s for s in held_ids if s in b_all.index])
                inner_k = inner[inner["outer_fold_heldout"] == k].set_index("sample_id")["inner_fold"]
                model = fit_transition_v6(a_fit, b_all.loc[fit_ids], targets, study.transition, study.registry,
                                          inner_k.reindex(fit_ids).to_numpy(dtype=float), representation="f9b_activations")
                if held_got:
                    oof.loc[held_got, targets] = model.predict(a_held).to_numpy()
                row["probe"] = "EVALUATED"
                complete_probe &= len(held_got) == len([s for s in held_ids if s in b_all.index])
            unit_rows.append(row)
        pred_rows.append(pd.DataFrame({"sample_id": decisions.index, "y_pred": decisions.to_numpy(), "fraction": float(f), "repeat": int(r)}))
        point: dict[str, Any] = {"fraction": float(f), "repeat": int(r)}
        point["macro_f1"] = ({"estimate": _macro_f1(truth.loc[decisions.index], decisions.to_numpy()), "status": "EVALUATED", "reason": None}
                             if complete_pred else {"estimate": None, "status": NOT_EXECUTED, "reason": "INPUTS_NOT_RECORDED"})
        point["rho"] = (stages._rho(ctx, b_all.loc[oof.index, targets], oof, mean, sd) if complete_probe
                        else {"rho": None, "status": NOT_EXECUTED, "reason": "INPUTS_NOT_RECORDED"})
        curves.append(point)
    return {"units": pd.DataFrame(unit_rows), "curves": curves, "predictions": pd.concat(pred_rows, ignore_index=True)}


# ----------------------------------------------------------------------------- stage


def _write_table(frame: pd.DataFrame, path: Path) -> Path:
    from ..stages import _write_table as write

    return write(frame, path, csv=False)


def _infer_refits(ctx: Any, cfg: Mapping[str, Any], refits: Sequence[Mapping[str, Any]], wave2_path: Path, label: Mapping[str, Any], *,
                  device: str | None):
    """Integration of L89 and L4a (C4): the local inference of every S5 refit on the ``acdc_dev`` cache with the
    trained-encoder code of ``infer-f9`` (:func:`tmmri.neural.inference.run_f9_inference`): ``fold<k>`` rows for every
    development patient, ``is_oof`` only for outer fold k, written to ``data_efficiency/f9/oof/<fit_id>/``.  Refuses, with
    nothing recorded, while a refit record or checkpoint is missing."""

    from .. import stages
    from ..neural.dihedral import D4_TTA
    from ..neural.inference import EncoderCheckpoint, check_leakage, run_f9_inference
    from ..neural.preprocess import read_cache_manifest

    run = ctx.require_run("run-data-efficiency")
    f9 = cfg["f9"]
    cache = run.path / "neural_cache" / "acdc_dev"
    checkpoints = {s["fit_id"]: run.path / "neural_f9" / "checkpoints" / s["fit_id"] for s in refits}
    missing = [fid for fid, d in checkpoints.items() if not ((d / "record.json").is_file() and (d / "model.pt").is_file())]
    if missing or not (cache / "manifest.csv").is_file():
        raise FileNotFoundError(f"run-data-efficiency --step f9-infer: {len(missing)} S5 refit checkpoints are not recorded (e.g. {missing[:3]}) "
                                "or the acdc_dev tensor cache is missing; train or import the wave-2 refits first")
    outputs = [run.path / str(f9["oof_outputs"]).format(fit_id=s["fit_id"]) / name for s in refits for name in ("predictions.parquet", "activations.parquet")]
    outputs.append(run.path / "data_efficiency" / "f9" / "inference_s5.json")
    inputs = [wave2_path, cache / "manifest.csv", stages._contracts(ctx) / "split_manifest.parquet"]
    inputs += [d / name for d in checkpoints.values() for name in ("record.json", "model.pt")]

    def compute() -> dict[str, Any]:
        from ..neural.torch_numerics import configure_torch_numerics, environment_record, resolve_device

        numerics = configure_torch_numerics()
        dev = resolve_device(device)
        split = stages._split_manifest(ctx)
        frame = development_frame(split, stages._cohort(ctx, "ACDC"), cfg["partition"])
        folds = {str(s): int(k) for s, k in zip(frame["sample_id"], frame["outer_fold"])}
        manifest = read_cache_manifest(cache)
        assert_development_only(manifest["sample_id"], split, cfg)
        truth = dict(zip(frame["sample_id"].astype(str), frame["acdc_class"].astype(str)))
        used = []
        for spec in refits:
            directory = checkpoints[spec["fit_id"]]
            record = json.loads((directory / "record.json").read_text(encoding="utf-8"))
            if record.get("fit_id") != spec["fit_id"] or record.get("status") != "COMPLETE":
                raise ValueError(f"{directory}: the record is not the completed refit {spec['fit_id']}")
            k = int(spec["outer_fold"])
            enc = EncoderCheckpoint(fit_id=str(spec["fit_id"]), seed=int(spec["seed"]), encoder_id=f"fold{k}", outer_fold=k, path=directory,
                                    fitting_patients=frozenset(str(s) for s in spec["fitting_patients"]))
            check_leakage([enc], cfg["partition"], set(), folds)
            preds, acts, details = run_f9_inference(cache, [enc], partition=str(cfg["partition"]), device=dev, dev_folds=folds, y_true=truth)
            out = run.path / str(f9["oof_outputs"]).format(fit_id=spec["fit_id"])
            _write_table(preds, out / "predictions.parquet")
            _write_table(acts, out / "activations.parquet")
            used.extend(details["encoders"])
        atomic_write_json(outputs[-1], {**label, "pipeline": "f9", "step": "f9-infer", "refits": len(refits), "encoders": used, "tta": D4_TTA,
                                        "environment": environment_record(dev), "numerics_at_start": numerics})
        return {"refits": len(refits), "device": str(dev)}

    return record_stage(ctx, "run-data-efficiency:f9-infer", compute, inputs, outputs, sections=(), dc_section=DC_SECTION,
                        params={"step": "f9-infer"})


def run_data_efficiency(ctx: Any, step: str = "descriptor", *, device: str | None = None):
    """``run-data-efficiency --step descriptor | f9-wave1 | f9-wave2 | f9-infer | f9-aggregate`` (S5; development only)."""

    from .. import stages

    if step not in STEPS:
        raise ValueError(f"run-data-efficiency --step must be one of {STEPS}")
    run = ctx.require_run("run-data-efficiency")
    cfg = data_centric_config(ctx.study)[DC_SECTION]
    contracts = stages._contracts(ctx)
    ddir = run.sub("data_efficiency")
    base_inputs = [contracts / "split_manifest.parquet", contracts / "cohort_ACDC.parquet"]
    label = {"analysis": "S5_data_efficiency", "registration": "protocol-v6.0.1 item 15", "partition": cfg["partition"],
             "development_only": True, **config_sha256_params(ctx.study)}

    if step == "descriptor":
        out = ddir / "descriptor"
        outputs = [out / "subsamples.parquet", out / "oof_bhat.parquet", out / "oof_d_model.parquet", out / "units.parquet", out / "curves.json"]

        def compute() -> dict[str, Any]:
            res = descriptor_curves(ctx, cfg)
            for frame, path in zip((res["subsamples"], res["bhat"], res["predictions"], res["units"]), outputs[:4]):
                _write_table(frame, path)
            atomic_write_json(outputs[4], {**label, "pipeline": "descriptor", "points": res["curves"],
                                           "summary_rho": summarize_curves(res["curves"], "rho"),
                                           "summary_macro_f1": summarize_curves(res["curves"], "macro_f1"),
                                           "class_support": cfg["class_support"], "fractions": cfg["fractions"]}, allow_nan=True)
            return {"points": len(res["curves"]), "units": int(len(res["units"]))}

        inputs = base_inputs + [run.path / "formal" / "A_ACDC.parquet", run.path / "semantics" / "B_ACDC.parquet", contracts / "inner_folds.parquet"]
        sections = ("transition", "registry", "cohorts.qc_severity", "formal.audited_predictor", "bootstrap.fixed_model_bootstrap",
                    "bootstrap.primary_test", "splits.acdc.acdc_dev")
        return record_stage(ctx, "run-data-efficiency:descriptor", compute, inputs, outputs, sections=sections, dc_section=DC_SECTION,
                            params={"step": step})

    fdir = ddir / "f9"
    if step == "f9-wave1":
        outputs = [fdir / "specs_wave1.jsonl", fdir / "specs_wave1.parquet", fdir / "subsamples.parquet", fdir / "plan_wave1.json",
                   fdir / "train_wave1_specs.json"]
        cache_manifest, f9cfg, weights = train_spec_prerequisites(run, ctx.study)

        def compute() -> dict[str, Any]:
            split = stages._split_manifest(ctx)
            dev = development_frame(split, stages._cohort(ctx, "ACDC"), cfg["partition"])
            assert_development_only(dev["sample_id"], split, cfg)
            specs, units, source = wave1_specs(dev, cfg)
            for spec in specs:
                assert_development_only(spec["fitting_patients"], split, cfg)
            _write_jsonl(outputs[0], specs)
            _write_table(_spec_index(specs, 1), outputs[1])
            _write_table(subsample_table(dev, cfg, fractions=[float(f) for f in cfg["f9"]["fractions"]]), outputs[2])
            atomic_write_json(outputs[3], {**label, "pipeline": "f9", "wave": 1, "fit_spec_source": source, "n_specs": len(specs),
                                           "expected": cfg["f9"]["expected_fits"],
                                           "units": [{k: v for k, v in u.items() if k != "fitting_patients"} for u in units],
                                           "n_units": len(units), "n_units_supported": sum(u["ok"] for u in units)})
            write_train_spec_file(outputs[4], specs, run_id=run.run_id, wave=1, manifest=cache_manifest, f9cfg=f9cfg, weights=weights,
                                  source=run.portable(outputs[0]))
            return {"specs": len(specs), "units": len(units)}

        return record_stage(ctx, "run-data-efficiency:f9-wave1", compute, base_inputs + [cache_manifest], outputs, sections=("splits.acdc.acdc_dev",),
                            dc_section=DC_SECTION, params={"step": step})

    wave1_path = fdir / "specs_wave1.jsonl"
    if not wave1_path.is_file():
        raise FileNotFoundError(f"{wave1_path} missing: run run-data-efficiency --step f9-wave1 first")
    if step == "f9-wave2":
        wave1 = read_jsonl(wave1_path)
        pattern = str(cfg["f9"]["records"])
        record_paths = {s["fit_id"]: run.path / pattern.format(fit_id=s["fit_id"]) for s in wave1}
        present = [p for p in record_paths.values() if p.is_file()]
        outputs = [fdir / "specs_wave2.jsonl", fdir / "specs_wave2.parquet", fdir / "selection_wave2.json", fdir / "train_wave2_specs.json"]
        cache_manifest, f9cfg, weights = train_spec_prerequisites(run, ctx.study)

        def compute() -> dict[str, Any]:
            records = {fid: json.loads(p.read_text(encoding="utf-8")) for fid, p in record_paths.items() if p.is_file()}
            specs, ledger, source = wave2_specs(wave1, records, cfg)
            _write_jsonl(outputs[0], specs)
            write_train_spec_file(outputs[3], specs, run_id=run.run_id, wave=2, manifest=cache_manifest, f9cfg=f9cfg, weights=weights,
                                  source=run.portable(outputs[0]))
            _write_table(_spec_index(specs, 2), outputs[1])
            atomic_write_json(outputs[2], {**label, "pipeline": "f9", "wave": 2, "fit_spec_source": source, "n_specs": len(specs),
                                           "n_wave1_records": len(records), "selection_rule": cfg["f9"]["selection_rule"],
                                           "refit_epochs": cfg["f9"]["refit_epochs"], "units": ledger})
            return {"specs": len(specs)}

        return record_stage(ctx, "run-data-efficiency:f9-wave2", compute, [wave1_path, *present], outputs, sections=(),
                            dc_section=DC_SECTION, params={"step": step})

    wave2_path = fdir / "specs_wave2.jsonl"
    if not wave2_path.is_file():
        raise FileNotFoundError(f"{wave2_path} missing: run run-data-efficiency --step f9-wave2 first")
    refits = read_jsonl(wave2_path)
    if step == "f9-infer":
        return _infer_refits(ctx, cfg, refits, wave2_path, label, device=device)
    dirs = [run.path / str(cfg["f9"]["oof_outputs"]).format(fit_id=s["fit_id"]) for s in refits] + [run.path / str(cfg["f9"]["full_fraction_outputs"])]
    expected = [d / n for d in dirs for n in ("predictions.parquet", "activations.parquet")]
    missing = [run.portable(p) for p in expected if not p.is_file()]
    if missing:   # nothing is recorded, so the once-only guard of a frozen run never locks a partial aggregate
        raise FileNotFoundError(f"run-data-efficiency --step f9-aggregate: {len(missing)} C4 outputs not recorded yet (e.g. {missing[:3]}); "
                                "run the local inference of every wave-2 refit and of the F9 primary-seed outer folds first")
    present = expected
    outputs = [fdir / "aggregate_units.parquet", fdir / "aggregate_oof_predictions.parquet", fdir / "curves.json"]

    def compute() -> dict[str, Any]:
        res = aggregate_f9(ctx, cfg, refits)
        _write_table(res["units"].assign(outputs=lambda d: d["outputs"].astype(str)), outputs[0])
        _write_table(res["predictions"], outputs[1])
        atomic_write_json(outputs[2], {**label, "pipeline": "f9", "points": res["curves"], "summary_rho": summarize_curves(res["curves"], "rho"),
                                       "summary_macro_f1": summarize_curves(res["curves"], "macro_f1"),
                                       "inputs_recorded": {run.portable(p): hash_path(p) for p in present}}, allow_nan=True)
        return {"points": len(res["curves"])}

    inputs = base_inputs + [wave2_path, run.path / "semantics" / "B_ACDC.parquet", contracts / "inner_folds.parquet", *present]
    return record_stage(ctx, "run-data-efficiency:f9-aggregate", compute, inputs, outputs,
                        sections=("transition", "registry", "cohorts.qc_severity", "bootstrap.fixed_model_bootstrap", "bootstrap.primary_test"),
                        dc_section=DC_SECTION, params={"step": step})
