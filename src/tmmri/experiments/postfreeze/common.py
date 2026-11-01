"""Shared pieces of the post-freeze exploratory block B10-B22 (protocol-v6.0.2 amendment; SPECIFICATION.md 6.0.2,
Sections 0.4 (D37-D41) and 13.7; ``configs/exploratory_postfreeze.yaml``).

``run-postfreeze --analysis B<k> --source-run <run_id> --run-id v6-postfreeze-<YYYYMMDD>-<HHMM>`` runs one analysis in a
new run root.  The source run (the frozen ``v6-replication-20261003-1241``) is read-only: every file read from it is
checked against the SHA-256 recorded by the stage that wrote it (or by ``release_manifest.json``), a mismatch refuses,
and nothing is ever written below it.  Outputs go to ``<run_root>/postfreeze/<B#>/`` beside a ``provenance.json``
labelled ``EXPLORATORY_POST_HOC``; an existing output refuses (a repeat needs a new run id and is reported as a repeat,
never as a replacement).  Nothing runs before the signed amendment tag ``protocol-v6.0.2`` is contained in HEAD;
``--allow-unfrozen-synthetic`` admits synthetic fixtures only.

:class:`PostfreezeContext` is the contract of every analysis module (``run(pctx) -> dict``): ``source_root``,
``run_root``, ``staged_data_root``, ``study``, ``lane_cfg``, ``analysis_cfg``, ``out_dir``, ``verified``,
``write_provenance`` and ``require_new_output``.  B10-B20 live in this package; B21 and B22 are imported lazily.
"""

from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable, Mapping

import numpy as np
import pandas as pd

from ...paths import ARTIFACT_ROOT, DATA_ROOT, PROJECT_ROOT
from ...runs import RunRoot, atomic_write_json, code_version, hash_path, portable_path, sha256_json
from ...secondary_stages import (
    lane_config,
    lane_config_path,
    require_partition_unlocked,
)

CONFIG_KEY = "exploratory_postfreeze"
LABEL = "EXPLORATORY_POST_HOC"
SPECIFICATION_VERSION = "6.0.2"
COMMAND = "run-postfreeze"
OUTPUT_ROOT = "postfreeze"
ANALYSES = tuple(f"B{k}" for k in range(10, 23))
DECLARED = ("protocol-v6.0.2 amendment (SPECIFICATION.md 6.0.2 Section 13.7, Decision D37): post-freeze exploratory block "
            "B10-B22, declared post hoc (explains recorded results, never confirms them; changes no recorded decision)")
LOCAL_MODULES = {"B10": "b10_analytical_ef", "B11": "b11_descriptor_families", "B12": "b12_geometry", "B13": "b13_median_null",
                 "B14": "b14_sesoi_sweep", "B15": "b15_rule_ledger", "B16": "b16_coverage_fidelity", "B17": "b17_concept_reliability",
                 "B18": "b18_decision_link", "B19": "b19_majority_class", "B20": "b20_worked_example"}
EXTERNAL_MODULES = {"B21": "tmmri.experiments.postfreeze.b21_frozen_image", "B22": "tmmri.experiments.postfreeze.b22_f9m"}
DATASET = {"acdc_dev": "ACDC", "acdc_test": "ACDC", "mms_external": "MMS", "mms2_external": "MMS2"}
EVALUATED = ("acdc_test", "mms_external", "mms2_external")
NOT_EXECUTED = "NOT_EXECUTED"

# Producing stage of each source-run file (its stage record holds the SHA-256); unmatched files fall back to
# release_manifest.json.  ``{p}`` is an evaluated partition.
_P = r"(acdc_test|mms_external|mms2_external)"
_STAGE_RULES: tuple[tuple[str, str], ...] = (
    (r"^formal/A_(ACDC|MMS|MMS2)(_p6_128|_extractor|_failures)?\.(parquet|json)$", "extract-formal:{0}"),
    (r"^semantics/B_(ACDC|MMS|MMS2)\.parquet$", "build-semantics:{0}"),
    (r"^contracts/(cohort_MMS2|split_manifest_MMS2|cohort_validation_MMS2|mms2_duplicates)\.(parquet|json)$", "validate-cohort:MMS2"),
    (r"^contracts/", "validate-cohort"),
    (r"^oof/inner_oof_", "gate-rules"),
    (r"^(oof/bhat_oof_|transitions/(?!refit_arm/))", "fit-transition"),
    (rf"^reconstructed/(?:bhat|stability_[a-z_]+?)_{_P}(?:_decoded)?\.parquet$", "evaluate:{0}"),
    (rf"^inference/(?:d_model|routes_[a-z_]+?)_{_P}\.parquet$", "evaluate:{0}"),
    (rf"^inference/(?:audited_predictor|measured_b_comparator)/{_P}/", "summarize-predictor-extended:{0}"),
    (r"^metrics/(?:fixed_sequence|primary_rho_mms_external_never_analysed|strata_[a-z_]+_mms_external)\.(?:json|parquet)$", "evaluate:mms_external"),
    (r"^metrics/h_primary_3\.json$", "evaluate:mms2_external"),
    (rf"^metrics/(?:primary_rho|per_target|decoder|routes_[a-z_]+?)_{_P}\.(?:json|parquet)$", "evaluate:{0}"),
    (rf"^metrics/confirmatory_h4/h4_{_P}\.json$", "evaluate-h4:{0}"),
    (r"^metrics/predictor_mms2_external\.json$", "summarize-predictor:mms2_external"),
    (r"^metrics/predictor_(?:acdc_dev_oof|acdc_test|mms_external|summary|confusion)\.(?:json|parquet)$", "summarize-predictor"),
    (r"^metrics/predictor_extended/(?:bland_altman_)?(acdc_dev_oof|acdc_test|mms_external|mms2_external)\.(?:json|parquet)$",
     "summarize-predictor-extended:{0}"),
    (r"^metrics/ablations/", "run-ablations"),
    (r"^formal/(?:d_model_|audited_predictor/|measured_b_comparator/)", "fit-predictor"),
    (r"^(?:thresholds|granules)/(?!refit_arm/)", "discretize"),
    (r"^rules/(?:semantic_eligibility|rule_anchor_manifest)\.", "check-semantic-eligibility"),
    (r"^rules/(?:d_model|dataset_label|measured_oracle)/rulebook_induced\.json$", "induce-rules"),
    (r"^rules/(?:d_model|dataset_label|measured_oracle)/", "gate-rules"),
    (rf"^comparators/cart_f8/evaluation/[a-z_]+?_{_P}\.(?:json|parquet)$", "evaluate-cart-surrogate:{0}"),
    (r"^comparators/cart_f8/", "fit-cart-surrogate"),
    (rf"^trust/{_P}/", "evaluate-trust:{0}"),
    (r"^trust/", "fit-trust"),
)


def stage_of(rel: str) -> str | None:
    """The source-run stage whose record carries the hash of ``rel`` (run-relative POSIX path), else ``None``."""

    for pattern, stage in _STAGE_RULES:
        match = re.match(pattern, rel)
        if match:
            return stage.format(*match.groups())
    return None


def _string_keys(value: Any) -> Any:
    """YAML reads the bare key ``null`` (analyses.B13) as None; every lane key is used as a string."""

    if isinstance(value, Mapping):
        return {("null" if k is None else str(k)): _string_keys(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_string_keys(v) for v in value]
    return value


def config(study: Any) -> dict[str, Any]:
    cfg = _string_keys(lane_config(study, CONFIG_KEY))
    if str(cfg.get("label")) != LABEL:
        raise ValueError(f"configs/{CONFIG_KEY}.yaml label must be {LABEL}")
    if str(cfg.get("specification_version")) != SPECIFICATION_VERSION:
        raise ValueError(f"configs/{CONFIG_KEY}.yaml specification_version must be {SPECIFICATION_VERSION}")
    return cfg


def amendment_tag(lane_cfg: Mapping[str, Any]) -> str:
    """``protocol-v<specification_version>``: the signed tag that unlocks the block (``protocol-v6.0.2``)."""

    return f"protocol-v{lane_cfg['specification_version']}"


def run_id_pattern(lane_cfg: Mapping[str, Any]) -> re.Pattern[str]:
    template = str(lane_cfg["output"]["run_id_template"])
    return re.compile(re.escape(template).replace(r"\{YYYYMMDD\}", r"\d{8}").replace(r"\{HHMM\}", r"\d{4}"))


def _inside(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


@dataclass
class PostfreezeContext:
    """The source run (read-only, hash-checked), the new run root and the lane configuration of one analysis call."""

    source_root: Path
    run_root: Path
    staged_data_root: Path
    study: Any
    lane_cfg: dict[str, Any]
    stage_ctx: Any = None                      # StageContext of the source run (read-only RunRoot): v6 read helpers, locks
    n_jobs: int | None = None
    tag_state: str | None = None               # protocol_frozen | synthetic_override (set by run_postfreeze)
    verified_log: dict[str, Path] = field(default_factory=dict)
    _hashes: dict[str, str | None] = field(default_factory=dict, repr=False)
    _release: dict[str, Any] | None = field(default=None, repr=False)
    # The lane of this context (a later block subclasses with its own lane file; lane_v603.py).
    config_key: ClassVar[str] = CONFIG_KEY
    analyses: ClassVar[tuple[str, ...]] = ANALYSES
    declared: ClassVar[str] = DECLARED

    # ------------------------------------------------------------------------------------------------ the contract
    def analysis_cfg(self, analysis: str) -> dict[str, Any]:
        analysis = str(analysis).upper()
        if analysis not in self.analyses or analysis not in (self.lane_cfg.get("analyses") or {}):
            raise ValueError(f"{COMMAND} --analysis must be one of {list(self.analyses)} (configs/{self.config_key}.yaml analyses)")
        return dict(self.lane_cfg["analyses"][analysis])

    def out_dir(self, analysis: str) -> Path:
        """``run_root/postfreeze/<B#>/`` (created; never below the source run)."""

        path = self.run_root / OUTPUT_ROOT / str(analysis).upper()
        self._writable(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def verified(self, path: Path, stage: str | None) -> Path:
        """``path`` (absolute, or relative to the source run) after checking its SHA-256 against the record of ``stage``
        (outputs, else inputs: a staging manifest is a recorded input of ``build-semantics``) or, for a source-run file,
        ``release_manifest.json``.  An unrecorded or changed file refuses (BundleError, exit 7)."""

        from ...stages import BundleError

        p = Path(path)
        p = (self.source_root / p) if not p.is_absolute() else p
        prefix, rel = self._locate(p)
        expected, where = None, None
        record = self._source_run().read_record(stage) if stage else None
        if record is not None:
            keys = (f"{prefix}:{rel}", portable_path(p, self.source_root), p.resolve().as_posix())
            for table, kind in ((record.outputs, "outputs"), (record.inputs, "inputs")):
                hit = next((k for k in keys if k in table), None)
                if hit is not None:
                    expected, where = table[hit], f".stages/{self._source_run().record_path(stage).name} {kind}"
                    break
        if expected is None and prefix == "run":
            expected = (self._release_manifest().get("files") or {}).get(rel)
            where = "release_manifest.json" if expected is not None else None
        if expected is None:
            raise BundleError(f"{COMMAND}: {prefix}:{rel} is recorded neither by {stage or 'any stage'} nor in release_manifest.json of "
                              f"{self.source_root.name}; an unrecorded input is refused")
        key = p.resolve().as_posix()
        if key not in self._hashes:
            self._hashes[key] = hash_path(p)
        if self._hashes[key] != expected:
            raise BundleError(f"{COMMAND}: {prefix}:{rel} differs from the SHA-256 recorded in {where} of {self.source_root.name}; refused")
        self.verified_log[f"{'source' if prefix == 'run' else prefix}:{rel}"] = p
        return p

    def write_provenance(self, analysis: str, inputs: Mapping[str, Path], outputs: list[Path], summary: Mapping[str, Any]) -> Path:
        """``postfreeze/<B#>/provenance.json`` labelled ``EXPLORATORY_POST_HOC`` (input and output hashes, the lane
        configuration digest and settings, the code version, the source run and whether this is a repeat)."""

        analysis = str(analysis).upper()
        acfg = self.analysis_cfg(analysis)
        path = self.out_dir(analysis) / "provenance.json"
        self.require_new_output(path)
        missing = [str(p) for p in outputs if not Path(p).exists()]
        if missing:
            raise RuntimeError(f"{COMMAND} {analysis}: declared outputs missing {missing}")
        code = code_version()
        payload = {
            "schema": "tmmri.postfreeze_provenance/1", "label": LABEL, "analysis": analysis, "title": acfg.get("title"),
            "answers": acfg.get("answers"), "declared": self.declared, "role": "exploratory_post_hoc", "confirmatory": False,
            "inference": "effect sizes with descriptive intervals only; no p-values, no significance language",
            "specification_version": str(self.lane_cfg["specification_version"]), "protocol_tag": amendment_tag(self.lane_cfg),
            "protocol_tag_state": self.tag_state, "run_id": self.run_root.name,
            "source_run": self._source_record(),
            "repeat_of": self.prior_runs(analysis), "repeat_policy": self.lane_cfg["output"].get("repeat_policy"),
            "inputs": {label: {"path": self.portable(p), "sha256": hash_path(p)} for label, p in sorted(inputs.items())},
            "config": {"file": f"configs/{self.config_key}.yaml", "sha256": sha256_json(self.lane_cfg),
                       "file_sha256": hash_path(lane_config_path(self.study, self.config_key)), "analysis_settings": acfg,
                       "statistics": self.lane_cfg.get("statistics"), "policy": self.lane_cfg.get("policy")},
            "outputs": {self.portable(p): hash_path(p) for p in outputs},
            "code_commit": code.get("commit"), "code_sha256": code.get("code_sha256"), "code_dirty": code.get("dirty"),
            "summary": dict(summary),
            **self._provenance_extra(),
        }
        payload["repeat"] = bool(payload["repeat_of"])
        return atomic_write_json(path, payload, allow_nan=True)

    def require_new_output(self, path: Path) -> None:
        """Refuse an output that already exists (a repeat needs a new run id) or that lies below the source run."""

        from ...stages import BundleError

        self._writable(path)
        if Path(path).exists():
            raise BundleError(f"{COMMAND}: {self.portable(path)} exists in {self.run_root.name}; a repeat needs a new run id "
                              "(reported as a repeat, never as a replacement)")

    def require_unlocked(self) -> str:
        """The lock of this lane (protocol_frozen | synthetic_override; refuses with ProtocolFreezeError)."""

        return require_postfreeze_unlocked(self)

    def runner(self, analysis: str) -> Callable[["PostfreezeContext"], dict[str, Any]]:
        return analysis_runner(analysis)

    # ------------------------------------------------------------------------------------------------ helpers
    @property
    def boot(self) -> dict[str, Any]:
        return dict(self.lane_cfg["statistics"]["bootstrap"])

    @property
    def jobs(self) -> int:
        from ...runs import resolve_n_jobs

        return resolve_n_jobs(self.n_jobs)

    def source_file(self, rel: str, stage: str | None = None) -> Path:
        """The verified source-run file ``rel`` (stage from :func:`stage_of` unless given)."""

        return self.verified(self.source_root / rel, stage if stage is not None else stage_of(rel))

    def read_parquet(self, rel: str, stage: str | None = None) -> pd.DataFrame:
        return pd.read_parquet(self.source_file(rel, stage))

    def read_json(self, rel: str, stage: str | None = None) -> Any:
        return json.loads(self.source_file(rel, stage).read_text(encoding="utf-8"))

    def exists(self, rel: str) -> bool:
        return (self.source_root / rel).is_file()

    def portable(self, path: str | Path) -> str:
        p = Path(path)
        for prefix, root in (("run", self.run_root), ("source", self.source_root), ("data", self.staged_data_root), ("repo", PROJECT_ROOT)):
            if _inside(p, root):
                return f"{prefix}:{p.resolve().relative_to(Path(root).resolve()).as_posix()}"
        return p.resolve().as_posix()

    def partition_status(self, split: str, requester: str) -> str | None:
        """``None`` when ``split`` may be read (acdc_dev; an evaluated partition whose ``evaluate:<split>`` record exists
        and whose M&Ms lock passes: protocol-v6.0.0 for M&Ms-1, the signed protocol-v6.0.1 lock and the passed
        ``validate-cohort --dataset mms2`` record for M&Ms-2), else the reason it is not executed.  A failed lock
        refuses (ProtocolFreezeError, exit 3)."""

        from ...stages import require_mms2_validated

        if split == "acdc_dev":
            return None
        if split not in EVALUATED:
            raise ValueError(f"{requester}: unknown partition {split!r}")
        record = self._source_run().read_record(f"evaluate:{split}")
        if record is None or record.status != "EVALUATED":
            return f"evaluate:{split} not recorded in {self.source_root.name}"
        require_partition_unlocked(self.stage_ctx, split, f"{requester} ({split})")
        if split.startswith("mms2"):
            self.verified(require_mms2_validated(self.stage_ctx, f"{requester} ({split})"), None)
        return None

    def prior_runs(self, analysis: str) -> list[str]:
        """Earlier post-freeze run roots beside this one with a provenance record of ``analysis`` from the same source run."""

        out = []
        for prov in sorted(self.run_root.parent.glob(f"*/{OUTPUT_ROOT}/{analysis}/provenance.json")):
            run_id = prov.parents[2].name
            if run_id == self.run_root.name or not run_id_pattern(self.lane_cfg).fullmatch(run_id):
                continue
            try:
                source = json.loads(prov.read_text(encoding="utf-8")).get("source_run", {}).get("run_id")
            except (OSError, ValueError):
                source = None
            if source == self.source_root.name:
                out.append(run_id)
        return out

    def _writable(self, path: Path) -> None:
        for root in self._read_only_roots():
            if _inside(path, root):
                raise PermissionError(f"{COMMAND}: refusing to write below the source run {root.name}: {path}")
        if not _inside(path, self.run_root):
            raise PermissionError(f"{COMMAND}: outputs go below the new run root {self.run_root}: {path}")

    def _locate(self, p: Path) -> tuple[str, str]:
        from ...stages import BundleError

        for prefix, root in (("run", self.source_root), ("data", self.staged_data_root)):
            if _inside(p, root):
                return prefix, p.resolve().relative_to(Path(root).resolve()).as_posix()
        raise BundleError(f"{COMMAND}: {p} is neither in the source run {self.source_root.name} nor in the staged data root")

    def _read_only_roots(self) -> list[Path]:
        return [self.source_root]

    def _source_record(self) -> dict[str, Any]:
        return {"run_id": self.source_root.name, "release_manifest_sha256": hash_path(self.source_root / "release_manifest.json"),
                "files_sha256": self._release_manifest().get("files_sha256"), "frozen_bundle_sha256": self._frozen_bundle()}

    def _provenance_extra(self) -> dict[str, Any]:
        return {}

    def _source_run(self) -> RunRoot:
        return self.stage_ctx.run

    def _release_manifest(self) -> dict[str, Any]:
        if self._release is None:
            path = self.source_root / "release_manifest.json"
            self._release = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        return self._release

    def _frozen_bundle(self) -> str | None:
        path = self.source_root / ".stages" / "FROZEN.json"
        return json.loads(path.read_text(encoding="utf-8")).get("bundle_sha256") if path.is_file() else None


# ----------------------------------------------------------------------------------------------- context and guards


def make_postfreeze_context(study_path: str | Path, source_run: str, run_id: str, *, artifact_root: str | Path | None = None,
                            data_root: str | Path | None = None, n_jobs: int | None = None, allow_unfrozen_synthetic: bool = False,
                            sources: Mapping[str, Path] | None = None) -> PostfreezeContext:
    """The context of one ``run-postfreeze`` call (nothing is created yet; the run root appears with the first output)."""

    from ...config import load_study
    from ...stages import StageContext

    study = load_study(study_path)
    lane = config(study)
    if not run_id_pattern(lane).fullmatch(str(run_id)):
        raise ValueError(f"{COMMAND} --run-id {run_id!r} must match {lane['output']['run_id_template']} (configs/{CONFIG_KEY}.yaml output)")
    if str(run_id) == str(source_run):
        raise ValueError(f"{COMMAND}: --run-id must be a new run root, not the source run {source_run}")
    artifacts = Path(artifact_root) if artifact_root is not None else ARTIFACT_ROOT
    source = RunRoot(str(source_run), artifacts, create=False)
    if not source.path.is_dir() or not source.stages_dir.is_dir():
        raise FileNotFoundError(f"{COMMAND}: source run {source.path} not found (no .stages/)")
    run_root = artifacts / str(run_id)
    if _inside(run_root, source.path):
        raise ValueError(f"{COMMAND}: the new run root may not lie below the source run")
    data = Path(data_root) if data_root is not None else DATA_ROOT
    stage_ctx = StageContext(study=study, run=source, data_root=data, n_jobs=n_jobs, allow_unfrozen_synthetic=allow_unfrozen_synthetic,
                             sources=dict(sources or {}))
    return PostfreezeContext(source_root=source.path, run_root=run_root, staged_data_root=data, study=study, lane_cfg=lane,
                             stage_ctx=stage_ctx, n_jobs=n_jobs)


def synthetic_source(pctx: PostfreezeContext) -> bool:
    """True only when the source run declares synthetic sources (``release_manifest.json``) and the staged ACDC data come
    from a synthetic fixture (staging manifest ``synthetic_fixture``)."""

    from ...stages import _staging_manifest

    manifest = _staging_manifest(pctx.stage_ctx, "ACDC") or {}
    return pctx._release_manifest().get("synthetic_fixture") is True and manifest.get("synthetic_fixture") is True


def require_postfreeze_unlocked(pctx: PostfreezeContext) -> str:
    """The block's lock (Decision D37): the signed annotated tag ``protocol-v6.0.2`` contained in HEAD, and the source run
    named in ``configs/exploratory_postfreeze.yaml``; ``--allow-unfrozen-synthetic`` admits synthetic fixtures only.
    Refuses with ProtocolFreezeError (exit 3)."""

    from ...stages import ProtocolFreezeError, protocol_frozen

    tag = amendment_tag(pctx.lane_cfg)
    declared = str(pctx.lane_cfg["source_run"]["run_id"])
    if pctx.stage_ctx.allow_unfrozen_synthetic:
        if synthetic_source(pctx):
            return "synthetic_override"
        raise ProtocolFreezeError(f"{COMMAND}: --allow-unfrozen-synthetic is accepted only for a synthetic source run (release_manifest.json "
                                  "synthetic_fixture and a synthetic ACDC staging manifest)")
    if pctx.source_root.name != declared:
        raise ProtocolFreezeError(f"{COMMAND}: the source run is fixed to {declared} (configs/{CONFIG_KEY}.yaml source_run); got {pctx.source_root.name}")
    if not protocol_frozen(tag, signed=True):
        raise ProtocolFreezeError(f"{COMMAND}: nothing in the post-freeze block runs on cohort data before the owner signs {tag} (signed annotated "
                                  "tag absent, unsigned, or HEAD does not descend from it)")
    return "protocol_frozen"


def analysis_runner(analysis: str) -> Callable[[PostfreezeContext], dict[str, Any]]:
    """``run(pctx)`` of an analysis: B10-B20 from this package, B21 and B22 imported lazily (a missing module is
    NotImplementedError, exit 4)."""

    if analysis in LOCAL_MODULES:
        return importlib.import_module(f"{__package__}.{LOCAL_MODULES[analysis]}").run
    name = EXTERNAL_MODULES[analysis]
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        raise NotImplementedError(f"{COMMAND} --analysis {analysis}: {name} is not part of this build") from exc
    return module.run


def run_postfreeze(pctx: PostfreezeContext, analysis: str) -> dict[str, Any]:
    """One analysis of the block: lock, repeat refusal, ``run(pctx)``, then a check that the provenance record exists."""

    analysis = str(analysis).upper()
    pctx.analysis_cfg(analysis)
    pctx.tag_state = pctx.require_unlocked()
    pctx.require_new_output(pctx.run_root / OUTPUT_ROOT / analysis / "provenance.json")
    runner = pctx.runner(analysis)
    result = runner(pctx)
    prov = pctx.run_root / OUTPUT_ROOT / analysis / "provenance.json"
    if not prov.is_file():
        raise RuntimeError(f"{COMMAND} {analysis}: the analysis wrote no provenance.json")
    record = json.loads(prov.read_text(encoding="utf-8"))
    return {"analysis": analysis, "label": LABEL, "run_id": pctx.run_root.name, "source_run": pctx.source_root.name,
            "protocol_tag_state": pctx.tag_state, "repeat": record.get("repeat"), "provenance": pctx.portable(prov),
            "outputs": sorted(record.get("outputs", {})), "result": result}


# ----------------------------------------------------------------------------------------------- recorded data


@dataclass
class Partition:
    """The non-critical-QC rows of one partition with their recorded B, B-hat (out-of-fold for acdc_dev) and clusters."""

    split: str
    dataset: str
    ids: list[str]
    b: pd.DataFrame                  # measured B (index sample_id)
    bhat: pd.DataFrame               # recorded raw B-hat (acdc_dev: nested out-of-fold)
    clusters: pd.Series              # bootstrap cluster_id
    units: pd.Series                 # weighting unit (patient)
    n_excluded_critical_qc: int
    role: str                        # development | evaluation

    def observed(self, targets: Iterable[str]) -> pd.DataFrame:
        from ..exploratory.common import observed

        return pd.DataFrame({t: observed(self.b, t) for t in targets}, index=self.b.index)


def split_manifest(pctx: PostfreezeContext, split: str) -> pd.DataFrame:
    rel = "contracts/split_manifest_MMS2.parquet" if split.startswith("mms2") else "contracts/split_manifest.parquet"
    return pctx.read_parquet(rel)


def load_partition(pctx: PostfreezeContext, split: str) -> Partition:
    """``acdc_dev``: the rows of the recorded nested out-of-fold B-hat; an evaluated partition: the rows of its recorded
    B-hat.  Critical-QC rows are excluded (as every v6 fit and evaluation); only OBSERVED cells of B are scored."""

    from ...stages import _cluster_ids, critical_qc_mask

    ds = DATASET[split]
    b = pctx.read_parquet(f"semantics/B_{ds}.parquet").set_index("sample_id")
    rel = "oof/bhat_oof_ACDC_dev.parquet" if split == "acdc_dev" else f"reconstructed/bhat_{split}.parquet"
    bhat = pctx.read_parquet(rel).set_index("sample_id")
    manifest = split_manifest(pctx, split)
    members = set(manifest.loc[manifest["partition"].astype(str) == split, "sample_id"].astype(str))
    keep = sorted(s for s in bhat.index.astype(str) if s in members and s in b.index)
    critical = critical_qc_mask(b.loc[keep], pctx.study).to_numpy(dtype=bool)
    ids = [s for s, c in zip(keep, critical) if not c]
    units = pd.Series([s.split(":", 1)[-1] for s in ids], index=ids, dtype=object)
    return Partition(split=split, dataset=ds, ids=ids, b=b.loc[ids], bhat=bhat.loc[ids], clusters=_cluster_ids(manifest, ids),
                     units=units, n_excluded_critical_qc=int(critical.sum()), role="development" if split == "acdc_dev" else "evaluation")


def dev_reference(pctx: PostfreezeContext) -> tuple[pd.Series, pd.Series]:
    """(SD s_j, mean) of the transition targets on the acdc_dev fit rows (exactly the recorded ``_dev_reference``)."""

    from ...stages import _dev_reference

    pctx.source_file("semantics/B_ACDC.parquet")
    pctx.source_file("contracts/split_manifest.parquet")
    return _dev_reference(pctx.stage_ctx)


def formal_a(pctx: PostfreezeContext, ds: str) -> pd.DataFrame:
    return pctx.read_parquet(f"formal/A_{ds}.parquet").set_index("sample_id")


def reference_labels(pctx: PostfreezeContext, split: str) -> pd.Series:
    """Evaluation-only class reference: the ACDC class, or the M&Ms crosswalk class of same-definition cases only."""

    from ...stages import reference_labels as v6_reference

    ds = DATASET[split]
    cohort = pctx.read_parquet(f"contracts/cohort_{ds}.parquet").set_index("sample_id")
    return v6_reference(cohort, ds)


def included_partitions(pctx: PostfreezeContext, wanted: Iterable[str], requester: str) -> tuple[list[str], dict[str, str]]:
    """``(included, {split: reason})`` over ``wanted`` in order (locks enforced; a missing evaluation is not executed)."""

    included, skipped = [], {}
    for split in wanted:
        reason = pctx.partition_status(split, requester)
        if reason is None:
            included.append(split)
        else:
            skipped[split] = reason
    return included, skipped


def write_table(frame: pd.DataFrame, path: Path, pctx: PostfreezeContext) -> Path:
    """A new parquet output (refuses an existing file)."""

    from ...stages import _write_table

    pctx.require_new_output(path)
    return _write_table(frame.reset_index(drop=True), path)


def write_json(value: Any, path: Path, pctx: PostfreezeContext) -> Path:
    pctx.require_new_output(path)
    return atomic_write_json(path, value, allow_nan=True)


def clean(value: Any) -> Any:
    """JSON-safe scalars (NaN and infinities become None; numpy scalars become Python numbers)."""

    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if value is pd.NA:
        return None
    return value
