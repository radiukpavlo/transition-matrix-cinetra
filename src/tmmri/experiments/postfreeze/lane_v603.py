"""The second post-freeze exploratory block B23-B27 (protocol-v6.0.3 amendment; SPECIFICATION.md 6.0.3, Sections 0.5
(D42-D46) and 13.8; ``configs/exploratory_postfreeze_v603.yaml``).

``run-postfreeze --analysis B23..B27 --source-run v6-replication-20261003-1241 --run-id v6-postfreeze-<YYYYMMDD>-<HHMM>``
runs one analysis in a new run root with :class:`PostfreezeV603Context`: the context of B10-B22 (``common.py``) with

* its own lane file, ``exploratory_postfreeze_v603.yaml`` (``specification_version`` 6.0.3, else refused);
* its own lock: the signed annotated tag ``protocol-v6.0.3`` contained in HEAD and the two source runs named in the
  lane (``source_runs``); ``--allow-unfrozen-synthetic`` admits synthetic fixtures only;
* a second read-only source run, ``v6-postfreeze-20261028-0201`` (B10-B22).  Every file read from it is checked against
  the SHA-256 in the provenance record of the analysis that wrote it (``postfreeze/<B#>/provenance.json``), directly or
  through a ``FROZEN.json`` that the record lists and that is verified first (B22's networks and probes); that record
  must carry the EXPLORATORY_POST_HOC label, specification version 6.0.2, its own run id, the frozen source run and the
  SHA-256 of that run's ``release_manifest.json`` (and, under the signed tag, ``protocol_tag_state`` protocol_frozen).
  Nothing is written below it.

The frozen source run is read and hash-checked exactly as in B10-B22.  The shared pieces of B23 and B24 (the decision
link with its dimension-matched references) live here as well.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable, Mapping

import numpy as np
import pandas as pd

from ...paths import ARTIFACT_ROOT, DATA_ROOT
from ...runs import RunRoot, hash_path
from ...secondary_stages import lane_config
from .common import (
    COMMAND,
    LABEL,
    OUTPUT_ROOT,
    PostfreezeContext,
    _inside,
    _string_keys,
    amendment_tag,
    run_id_pattern,
    synthetic_source,
    write_json,
)
from .estimates import BELOW_MIN_CASES, wilson

CONFIG_KEY = "exploratory_postfreeze_v603"
SPECIFICATION_VERSION = "6.0.3"
ANALYSES = ("B23", "B24", "B25", "B26", "B27")
MODULES = {"B23": "b23_decision_link_nulls", "B24": "b24_network_rule_audit", "B25": "b25_pooled_admission", "B26": "b26_robust_agreement",
           "B27": "b27_fold_spread"}
SECOND = "postfreeze_v602"                      # lane key of the second source run (v6-postfreeze-20261028-0201)
SECOND_SPECIFICATION_VERSION = "6.0.2"
DECLARED = ("protocol-v6.0.3 amendment (SPECIFICATION.md 6.0.3 Section 13.8, Decisions D42-D46): second post-freeze exploratory block "
            "B23-B27, declared post hoc (designed after the B10-B22 results; explains recorded results, never confirms them; changes no "
            "recorded decision)")


def config(study: Any) -> dict[str, Any]:
    cfg = _string_keys(lane_config(study, CONFIG_KEY))
    if str(cfg.get("label")) != LABEL:
        raise ValueError(f"configs/{CONFIG_KEY}.yaml label must be {LABEL}")
    if str(cfg.get("specification_version")) != SPECIFICATION_VERSION:
        raise ValueError(f"configs/{CONFIG_KEY}.yaml specification_version must be {SPECIFICATION_VERSION}")
    return cfg


@dataclass
class PostfreezeV603Context(PostfreezeContext):
    """The context of B10-B22 with the v6.0.3 lane, its lock and the second read-only source run."""

    second_root: Path | None = None
    _second_records: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    config_key: ClassVar[str] = CONFIG_KEY
    analyses: ClassVar[tuple[str, ...]] = ANALYSES
    declared: ClassVar[str] = DECLARED

    def require_unlocked(self) -> str:
        return require_v603_unlocked(self)

    def runner(self, analysis: str) -> Callable[[PostfreezeContext], dict[str, Any]]:
        return importlib.import_module(f"{__package__}.{MODULES[analysis]}").run

    # ------------------------------------------------------------------------------------------------ second source run
    def second_record(self, analysis: str) -> dict[str, Any]:
        """The provenance record ``postfreeze/<analysis>/provenance.json`` of the second source run after checking that
        it belongs to B10-B22 over the frozen source run (BundleError, exit 7, otherwise)."""

        from ...stages import BundleError

        analysis = str(analysis).upper()
        if analysis not in self._second_records:
            path = self.second_root / OUTPUT_ROOT / analysis / "provenance.json"
            if not path.is_file():
                raise FileNotFoundError(f"{COMMAND}: {SECOND}:{OUTPUT_ROOT}/{analysis}/provenance.json not found in {self.second_root}")
            record = json.loads(path.read_text(encoding="utf-8"))
            source = record.get("source_run") or {}
            checks = {"label": record.get("label") == LABEL, "analysis": record.get("analysis") == analysis,
                      "specification_version": str(record.get("specification_version")) == SECOND_SPECIFICATION_VERSION,
                      "run_id": record.get("run_id") == self.second_root.name, "source_run": source.get("run_id") == self.source_root.name,
                      "release_manifest_sha256": source.get("release_manifest_sha256") == hash_path(self.source_root / "release_manifest.json")}
            if self.tag_state == "protocol_frozen":
                checks["protocol_tag_state"] = record.get("protocol_tag_state") == "protocol_frozen"
            failed = [k for k, ok in checks.items() if not ok]
            if failed:
                raise BundleError(f"{COMMAND}: {SECOND}:{OUTPUT_ROOT}/{analysis}/provenance.json of {self.second_root.name} does not belong to "
                                  f"B10-B22 over {self.source_root.name} ({', '.join(failed)}); refused")
            self._second_records[analysis] = record
            self.verified_log[f"{SECOND}:{OUTPUT_ROOT}/{analysis}/provenance.json"] = path
        return self._second_records[analysis]

    def second_file(self, rel: str) -> Path:
        """The second-run file ``rel`` (``postfreeze/<B#>/...``) after checking its SHA-256 against the provenance record
        of ``<B#>`` or a ``FROZEN.json`` listed there; an unrecorded or changed file refuses (BundleError, exit 7)."""

        from ...stages import BundleError

        parts = str(rel).split("/")
        if len(parts) < 3 or parts[0] != OUTPUT_ROOT:
            raise BundleError(f"{COMMAND}: {SECOND}:{rel} is not an output of a post-freeze analysis")
        outputs = self.second_record(parts[1]).get("outputs") or {}
        expected, anchor = outputs.get(f"run:{rel}"), f"{OUTPUT_ROOT}/{parts[1]}/provenance.json"
        if expected is None:
            expected, anchor = self._frozen_anchor(parts, outputs)
        if expected is None:
            raise BundleError(f"{COMMAND}: {SECOND}:{rel} is recorded neither in {OUTPUT_ROOT}/{parts[1]}/provenance.json of {self.second_root.name} "
                              "nor in a FROZEN.json listed there; an unrecorded input is refused")
        path = self.second_root / rel
        key = path.resolve().as_posix()
        if key not in self._hashes:
            self._hashes[key] = hash_path(path) if path.is_file() else None
        if self._hashes[key] != expected:
            raise BundleError(f"{COMMAND}: {SECOND}:{rel} differs from the SHA-256 recorded in {anchor} of {self.second_root.name}; refused")
        self.verified_log[f"{SECOND}:{rel}"] = path
        return path

    def _frozen_anchor(self, parts: list[str], outputs: Mapping[str, Any]) -> tuple[str | None, str | None]:
        """(SHA-256, anchor) of a file listed in the ``files`` of the ``FROZEN.json`` of an enclosing directory that the
        provenance record lists and that is verified first (B22 records its network and probe files through
        ``fits/FROZEN.json`` and ``probes/seed<k>/FROZEN.json``)."""

        for depth in range(len(parts) - 1, 1, -1):
            frozen_rel = "/".join([*parts[:depth], "FROZEN.json"])
            if f"run:{frozen_rel}" not in outputs:
                continue
            files = json.loads(self.second_file(frozen_rel).read_text(encoding="utf-8")).get("files") or {}
            name = "/".join(parts[depth:])
            if name in files:
                return str(files[name]), f"{frozen_rel} (listed in {OUTPUT_ROOT}/{parts[1]}/provenance.json)"
        return None, None

    def read_second_parquet(self, rel: str) -> pd.DataFrame:
        return pd.read_parquet(self.second_file(rel))

    def read_second_json(self, rel: str) -> Any:
        return json.loads(self.second_file(rel).read_text(encoding="utf-8"))

    def portable(self, path: str | Path) -> str:
        if self.second_root is not None and _inside(Path(path), self.second_root):
            return f"{SECOND}:{Path(path).resolve().relative_to(self.second_root.resolve()).as_posix()}"
        return super().portable(path)

    def _read_only_roots(self) -> list[Path]:
        return [self.source_root, *([self.second_root] if self.second_root is not None else [])]

    def _provenance_extra(self) -> dict[str, Any]:
        second = {"run_id": self.second_root.name,
                  "provenance_sha256": {a: hash_path(self.second_root / OUTPUT_ROOT / a / "provenance.json") for a in sorted(self._second_records)}}
        return {"source_runs": {"frozen": self._source_record(), SECOND: second}}


def make_v603_context(study_path: str | Path, source_run: str, run_id: str, *, artifact_root: str | Path | None = None,
                      data_root: str | Path | None = None, n_jobs: int | None = None, allow_unfrozen_synthetic: bool = False,
                      sources: Mapping[str, Path] | None = None) -> PostfreezeV603Context:
    """The context of one ``run-postfreeze --analysis B23..B27`` call (nothing is created yet)."""

    from ...config import load_study
    from ...stages import StageContext

    study = load_study(study_path)
    lane = config(study)
    if not run_id_pattern(lane).fullmatch(str(run_id)):
        raise ValueError(f"{COMMAND} --run-id {run_id!r} must match {lane['output']['run_id_template']} (configs/{CONFIG_KEY}.yaml output)")
    second_id = str(lane["source_runs"][SECOND]["run_id"])
    if str(run_id) in (str(source_run), second_id):
        raise ValueError(f"{COMMAND}: --run-id must be a new run root, not a source run ({source_run}, {second_id})")
    artifacts = Path(artifact_root) if artifact_root is not None else ARTIFACT_ROOT
    source = RunRoot(str(source_run), artifacts, create=False)
    if not source.path.is_dir() or not source.stages_dir.is_dir():
        raise FileNotFoundError(f"{COMMAND}: source run {source.path} not found (no .stages/)")
    second = artifacts / second_id
    if not (second / OUTPUT_ROOT).is_dir():
        raise FileNotFoundError(f"{COMMAND}: second source run {second} not found (no {OUTPUT_ROOT}/; configs/{CONFIG_KEY}.yaml source_runs)")
    run_root = artifacts / str(run_id)
    if _inside(run_root, source.path) or _inside(run_root, second):
        raise ValueError(f"{COMMAND}: the new run root may not lie below a source run")
    data = Path(data_root) if data_root is not None else DATA_ROOT
    stage_ctx = StageContext(study=study, run=source, data_root=data, n_jobs=n_jobs, allow_unfrozen_synthetic=allow_unfrozen_synthetic,
                             sources=dict(sources or {}))
    return PostfreezeV603Context(source_root=source.path, run_root=run_root, staged_data_root=data, study=study, lane_cfg=lane,
                                 stage_ctx=stage_ctx, n_jobs=n_jobs, second_root=second)


def require_v603_unlocked(pctx: PostfreezeV603Context) -> str:
    """The lock of B23-B27 (Decision D42): the signed annotated tag ``protocol-v6.0.3`` contained in HEAD and both source
    runs named in ``configs/exploratory_postfreeze_v603.yaml``; ``--allow-unfrozen-synthetic`` admits synthetic fixtures
    only.  Refuses with ProtocolFreezeError (exit 3)."""

    from ...stages import ProtocolFreezeError, protocol_frozen

    tag = amendment_tag(pctx.lane_cfg)
    runs = pctx.lane_cfg["source_runs"]
    frozen, second = str(runs["frozen"]["run_id"]), str(runs[SECOND]["run_id"])
    if pctx.stage_ctx.allow_unfrozen_synthetic:
        if synthetic_source(pctx):
            return "synthetic_override"
        raise ProtocolFreezeError(f"{COMMAND}: --allow-unfrozen-synthetic is accepted only for a synthetic source run (release_manifest.json "
                                  "synthetic_fixture and a synthetic ACDC staging manifest)")
    if pctx.source_root.name != frozen:
        raise ProtocolFreezeError(f"{COMMAND}: the source run is fixed to {frozen} (configs/{CONFIG_KEY}.yaml source_runs.frozen); "
                                  f"got {pctx.source_root.name}")
    if pctx.second_root is None or pctx.second_root.name != second:
        raise ProtocolFreezeError(f"{COMMAND}: the second source run is fixed to {second} (configs/{CONFIG_KEY}.yaml source_runs.{SECOND})")
    if not protocol_frozen(tag, signed=True):
        raise ProtocolFreezeError(f"{COMMAND}: nothing in the block B23-B27 runs on cohort data before the owner signs {tag} (signed "
                                  "annotated tag absent, unsigned, or HEAD does not descend from it)")
    return "protocol_frozen"


# ----------------------------------------------------------------------------------------------- shared helpers


def require_registered(pctx: PostfreezeContext, analysis: str, estimator_ids: Iterable[str]) -> None:
    registry = {str(e["id"]): e for e in pctx.lane_cfg["estimators"]}
    missing = [e for e in estimator_ids if e not in registry or str(registry[e].get("analysis")) != analysis]
    if missing:
        raise ValueError(f"{analysis}: estimators {missing} are not registered for {analysis} in configs/{CONFIG_KEY}.yaml estimators")


def write_array(array: np.ndarray, path: Path, pctx: PostfreezeContext) -> Path:
    """A new ``.npy`` output (refuses an existing file)."""

    pctx.require_new_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, np.asarray(array), allow_pickle=False)
    return path


def freeze(pctx: PostfreezeContext, mdir: Path, files: Iterable[Path], meta: Mapping[str, Any]) -> dict[str, Any]:
    """``FROZEN.json`` of one fitted object: its settings and the SHA-256 of every file (written before evaluation)."""

    record = {**dict(meta), "files": {Path(p).relative_to(mdir).as_posix(): hash_path(p) for p in files}}
    write_json(record, mdir / "FROZEN.json", pctx)
    return record


def check_frozen(mdir: Path, record: Mapping[str, Any]) -> None:
    """Refuse (BundleError) when a frozen file changed after its freeze."""

    from ...stages import BundleError

    for name, sha in record["files"].items():
        if hash_path(mdir / name) != sha:
            raise BundleError(f"{COMMAND}: {mdir.name}/{name} changed after its freeze")


def argmax_decisions(frame: pd.DataFrame) -> tuple[pd.Series, int]:
    """(decision per ``sample_id`` = argmax of the ``prob_<class>`` columns, ties to the first class in column order;
    the number of rows where ``y_pred`` disagrees).  A row without finite probabilities has no decision."""

    cols = [c for c in frame.columns if str(c).startswith("prob_")]
    if not cols:
        raise ValueError("network predictions carry no prob_<class> columns")
    if frame["sample_id"].duplicated().any():
        raise ValueError("network predictions: several rows per sample_id for one encoder")
    probs = frame[cols].to_numpy(dtype=float)
    finite = np.isfinite(probs).all(axis=1)
    labels = np.array([str(c)[len("prob_"):] for c in cols], dtype=object)
    out = np.full(len(frame), None, dtype=object)
    out[finite] = labels[np.argmax(probs[finite], axis=1)]
    dec = pd.Series(out, index=frame["sample_id"].astype(str).to_numpy(), dtype=object)
    mismatch = int((pd.Series(out).astype(str).to_numpy() != frame["y_pred"].astype(str).to_numpy()).sum()) if "y_pred" in frame.columns else 0
    return dec, mismatch


# ----------------------------------------------------------------------------------------------- decision link (B23, B24)


def standardizer(x: np.ndarray) -> dict[str, list[float]]:
    """Mean and SD (ddof = 0) of every column on the finite rows of ``x``; a zero or undefined SD becomes 1."""

    x = np.asarray(x, dtype=float)
    rows = np.isfinite(x).all(axis=1)
    mean = x[rows].mean(axis=0) if rows.any() else np.zeros(x.shape[1])
    sd = x[rows].std(axis=0) if rows.any() else np.ones(x.shape[1])
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    return {"mean": [float(v) for v in mean], "scale": [float(v) for v in sd], "n_rows": int(rows.sum())}


def standardize(x: np.ndarray, scaler: Mapping[str, Any]) -> np.ndarray:
    return (np.asarray(x, dtype=float) - np.asarray(scaler["mean"])) / np.asarray(scaler["scale"])


def principal_components(z: np.ndarray, k: int) -> dict[str, Any]:
    """The first ``k`` principal axes of the finite rows of ``z`` (SVD of the centred rows; each axis signed so that its
    largest absolute loading is positive)."""

    z = np.asarray(z, dtype=float)
    rows = z[np.isfinite(z).all(axis=1)]
    centre = rows.mean(axis=0)
    _, s, vt = np.linalg.svd(rows - centre, full_matrices=False)
    axes = vt[:k]
    signs = np.sign(axes[np.arange(axes.shape[0]), np.argmax(np.abs(axes), axis=1)])
    axes = axes * np.where(signs == 0, 1.0, signs)[:, None]
    var = s ** 2
    return {"centre": [float(v) for v in centre], "components": axes.tolist(), "explained_variance_ratio": [float(v) for v in (var[:k] / var.sum())],
            "n_rows": int(rows.shape[0])}


def pca_scores(z: np.ndarray, pca: Mapping[str, Any]) -> np.ndarray:
    return (np.asarray(z, dtype=float) - np.asarray(pca["centre"])) @ np.asarray(pca["components"]).T


def random_projections(seed: int, draws: int, input_dim: int, k: int) -> np.ndarray:
    """``draws`` standard-normal ``input_dim x k`` matrices drawn in order from ``numpy.random.default_rng(seed)``."""

    rng = np.random.default_rng(int(seed))
    return np.stack([rng.standard_normal((int(input_dim), int(k))) for _ in range(int(draws))]) if draws else np.zeros((0, input_dim, k))


def fit_link(x: np.ndarray, y: np.ndarray, pcfg: Mapping[str, Any]) -> Any:
    """The decision-link model of B18: the audited-predictor settings on the rows with finite inputs and a decision."""

    from ...classification import fit_formal_predictor

    fit_rows = np.isfinite(np.asarray(x, dtype=float)).all(axis=1) & pd.Series(np.asarray(y, dtype=object)).notna().to_numpy()
    return fit_formal_predictor(np.asarray(x, dtype=float), np.asarray(y, dtype=object), dict(pcfg), fit_mask=fit_rows)


def write_predictors(models: list[Any], path: Path, pctx: PostfreezeContext) -> Path:
    return write_json([m.to_dict() for m in models], path, pctx)


def read_predictors(path: Path) -> list[Any]:
    from ...classification import FormalPredictor

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [FormalPredictor.from_dict(d) for d in (payload if isinstance(payload, list) else [payload])]


def agreement(pred: pd.Series, decision: pd.Series) -> tuple[int, int]:
    """(agreements, cases with a prediction and a decision)."""

    d = decision.reindex(pred.index)
    rows = pred.notna() & d.notna()
    return int((pred[rows].astype(str) == d[rows].astype(str)).sum()), int(rows.sum())


def null_summary(fidelities: Iterable[float]) -> dict[str, Any]:
    """Median and 95th percentile (numpy linear interpolation) of a null distribution of fidelities."""

    v = np.asarray([f for f in fidelities if f is not None and np.isfinite(f)], dtype=float)
    if not v.size:
        return {"n_draws": 0, "median": None, "p95": None, "min": None, "max": None, "quantile_method": "numpy_linear"}
    return {"n_draws": int(v.size), "median": float(np.median(v)), "p95": float(np.quantile(v, 0.95)), "min": float(v.min()), "max": float(v.max()),
            "quantile_method": "numpy_linear"}


# --- reproduction checks of recorded values (analyses.B24.reconstruction_check)

RECONSTRUCTION_TOLERANCE = 1e-6


def check_reconstruction(recomputed: pd.DataFrame, recorded: pd.DataFrame, what: str, tol: float = RECONSTRUCTION_TOLERANCE) -> dict[str, Any]:
    """The recomputed reconstruction (index sample_id) must reproduce the recorded one: every recorded row recomputed,
    every recomputed column recorded, the same missing cells and a maximum absolute difference of at most ``tol``;
    otherwise BundleError (exit 7).  No value is reported, only the counts and the maximum difference."""

    from ...stages import BundleError

    recorded = recorded.copy()
    recorded.index = recorded.index.astype(str)
    cols = [str(c) for c in recomputed.columns]
    problems, diff = [], None
    absent_cols = [c for c in cols if c not in recorded.columns]
    absent_rows = recorded.index.difference(recomputed.index.astype(str))
    if absent_cols:
        problems.append(f"{len(absent_cols)} recomputed columns are not recorded (e.g. {absent_cols[:3]})")
    if len(absent_rows):
        problems.append(f"{len(absent_rows)} recorded rows are not recomputed")
    if not problems:
        a = recomputed.reindex(recorded.index)[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        b = recorded[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        missing = int((np.isnan(a) != np.isnan(b)).sum())
        both = np.isfinite(a) & np.isfinite(b)
        diff = float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0
        if missing:
            problems.append(f"{missing} cells missing in one of the two")
        if not diff <= tol:
            problems.append(f"maximum absolute difference above {tol:g}")
    if problems:
        raise BundleError(f"{COMMAND}: {what}: the recomputed reconstruction does not reproduce the recorded one ({'; '.join(problems)}); refused")
    return {"reproduced": True, "n_recorded_rows": int(len(recorded)), "n_recomputed_rows": int(len(recomputed)), "n_columns": len(cols),
            "max_abs_difference": diff, "tolerance": tol}


def compare_recorded(recorded: Any, recomputed: Any, tol: float = RECONSTRUCTION_TOLERANCE, path: str = "") -> list[str]:
    """Paths at which a recomputed JSON block differs from the recorded one: the same keys and list lengths, numbers
    within ``tol`` (absolute), everything else equal.  The messages name paths and types, never values."""

    where = path or "<root>"
    if isinstance(recorded, dict) and isinstance(recomputed, dict):
        out = [f"{where}: keys differ ({sorted(set(map(str, recorded)) ^ set(map(str, recomputed)))[:5]})"] if set(recorded) != set(recomputed) else []
        for k in sorted(set(recorded) & set(recomputed), key=str):
            out += compare_recorded(recorded[k], recomputed[k], tol, f"{path}.{k}" if path else str(k))
        return out
    if isinstance(recorded, list) and isinstance(recomputed, list):
        if len(recorded) != len(recomputed):
            return [f"{where}: {len(recorded)} recorded and {len(recomputed)} recomputed items"]
        return [d for i, (a, b) in enumerate(zip(recorded, recomputed)) for d in compare_recorded(a, b, tol, f"{where}[{i}]")]

    def number(v: Any) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    if number(recorded) and number(recomputed):
        same = (np.isnan(recorded) and np.isnan(recomputed)) if np.isnan(recorded) or np.isnan(recomputed) else abs(recorded - recomputed) <= tol
        return [] if same else [f"{where}: differs by more than {tol:g}"]
    return [] if recorded == recomputed else [f"{where}: differs ({type(recorded).__name__} recorded, {type(recomputed).__name__} recomputed)"]


def paired_difference(pred_a: pd.Series, pred_b: pd.Series, decision: pd.Series, clusters: pd.Series, boot: Mapping[str, Any],
                      min_cases: int) -> dict[str, Any]:
    """fidelity(a) - fidelity(b) on the cases both predict, with the paired cluster bootstrap of the lane (the same
    resampled clusters for both arms; ``SeedSequence(seed).spawn(n)[r]`` as every v6 cluster bootstrap)."""

    from ..alt_bundles import bootstrap_statistic

    d = decision.reindex(pred_a.index)
    common = pred_a.notna() & pred_b.reindex(pred_a.index).notna() & d.notna()
    ids = pred_a.index[common.to_numpy()]
    hit_a = (pred_a.loc[ids].astype(str) == d.loc[ids].astype(str)).to_numpy(dtype=float)
    hit_b = (pred_b.loc[ids].astype(str) == d.loc[ids].astype(str)).to_numpy(dtype=float)
    base = {"n_common": int(len(ids)), "fidelity_a_common": float(hit_a.mean()) if len(ids) else None,
            "fidelity_b_common": float(hit_b.mean()) if len(ids) else None}
    if not len(ids):
        return {**base, "estimate": None, "ci_lower": None, "ci_upper": None, "status": "NE", "reason": "NO_CASES"}
    below = len(ids) < int(min_cases)
    r = bootstrap_statistic(lambda rows: float(hit_a[rows].mean() - hit_b[rows].mean()), len(ids),
                            clusters.reindex(ids).astype(str).to_numpy(dtype=object), n_boot=0 if below else int(boot["replicates"]),
                            seed=int(boot["seed"]), level=float(boot.get("level", 0.95)), min_valid_fraction=0.0)
    if below:
        r.update({"ci_lower": None, "ci_upper": None, "status": "NE", "reason": BELOW_MIN_CASES})
    else:
        r.setdefault("status", "EVALUATED")
        r.setdefault("reason", None)
    return {**base, **{k: r.get(k) for k in ("estimate", "ci_lower", "ci_upper", "status", "reason")}}


def fidelity_row(k: int, n: int) -> dict[str, Any]:
    w = wilson(k, n)
    return {"fidelity": w["estimate"], "fidelity_ci_lower": w["ci_lower"], "fidelity_ci_upper": w["ci_upper"], "fidelity_k": k, "fidelity_n": n}
