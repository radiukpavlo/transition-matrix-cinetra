"""The third post-freeze exploratory block B28-B31 (protocol-v6.0.4 amendment; SPECIFICATION.md 6.0.4, Sections 0.6
(D48-D52) and 13.9; ``configs/exploratory_postfreeze_v604.yaml``).

``run-postfreeze --analysis B28..B31 --source-run v6-replication-20261003-1241 --run-id v6-postfreeze-<YYYYMMDD>-<HHMM>``
runs one analysis in a new run root with :class:`PostfreezeV604Context`: the context of B10-B22 (``common.py``) with

* its own lane file, ``exploratory_postfreeze_v604.yaml`` (``specification_version`` 6.0.4, else refused);
* its own lock (D48, D52): the signed annotated tag ``protocol-v6.0.4`` contained in HEAD, the specification check of
  D47 against that tag (it must be the newest signed protocol tag HEAD descends from and ``SPECIFICATION.md`` must equal
  its blob), the lane file and ``src/`` equal to their copies at the tag (no untracked file below them), ``--study``
  inside the project root with the project's own lane file, and the source runs named in the lane (``source_runs``);
  ``--allow-unfrozen-synthetic`` admits synthetic fixtures only;
* a third read-only source run, ``v6-postfreeze-20261028-2135`` (B23-B27).  Every file read from it is checked against
  the SHA-256 in the provenance record of the analysis that wrote it (``postfreeze/<B#>/provenance.json``), directly or
  through a ``FROZEN.json`` that the record lists and that is verified first; that record must carry the
  EXPLORATORY_POST_HOC label, specification version 6.0.3, its own run id, the frozen source run with the SHA-256 of its
  ``release_manifest.json``, the B10-B22 run of the lane (``source_runs.postfreeze_v602``) and, under the signed tag,
  ``protocol_tag_state`` protocol_frozen.  The B10-B22 run is read by no analysis of this block (B28 reaches the recorded
  B18 values through B23); it is a read-only root all the same.  Nothing is written below any source run.

Shared pieces of B28-B31 live here: Cohen's kappa (unweighted) with the cluster bootstrap of the lane, Wilson rows with
the half-up rounding of values derived from exact counts (BLD-1), the partition labels of the block (``acdc_dev_oof``
is the recorded out-of-fold development reference) and the identifier check of every output (``policy.identifiers``).
"""

from __future__ import annotations

import importlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable, Mapping

import numpy as np
import pandas as pd

from ...paths import ARTIFACT_ROOT, DATA_ROOT, PROJECT_ROOT
from ...runs import RunRoot, hash_path
from ...secondary_stages import lane_config, lane_config_path
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
    write_table,
)
from .estimates import wilson

CONFIG_KEY = "exploratory_postfreeze_v604"
SPECIFICATION_VERSION = "6.0.4"
ANALYSES = ("B28", "B29", "B30", "B31")
MODULES = {"B28": "b28_kappa_ceilings", "B29": "b29_ef_categories", "B30": "b30_reference_categories", "B31": "b31_plausibility"}
THIRD = "postfreeze_v603"                       # lane key of the third source run (v6-postfreeze-20261028-2135, B23-B27)
THIRD_SPECIFICATION_VERSION = "6.0.3"
V602 = "postfreeze_v602"                        # lane key of the B10-B22 run (read by no analysis of this block)
DEVELOPMENT = "acdc_dev_oof"                    # policy.development_reference: the recorded out-of-fold reconstructions
DECLARED = ("protocol-v6.0.4 amendment (SPECIFICATION.md 6.0.4 Section 13.9, Decisions D48-D52): third post-freeze exploratory block "
            "B28-B31, declared post hoc (descriptive, designed after the B10-B27 results; no analysis "
            "has a reading; changes no recorded decision)")
IDENTIFIER_COLUMNS = ("sample_id", "case_id", "patient_id", "subject_id", "cluster_id", "duplicate_pair_id", "anchor")
ROUNDING_DIGITS = 3                             # BLD-1: values derived from exact counts, rounded half-up
TAG_PINNED = (f"configs/{CONFIG_KEY}.yaml", "src/")    # must equal their copies at the tag under protocol_frozen


def config(study: Any) -> dict[str, Any]:
    cfg = _string_keys(lane_config(study, CONFIG_KEY))
    if str(cfg.get("label")) != LABEL:
        raise ValueError(f"configs/{CONFIG_KEY}.yaml label must be {LABEL}")
    if str(cfg.get("specification_version")) != SPECIFICATION_VERSION:
        raise ValueError(f"configs/{CONFIG_KEY}.yaml specification_version must be {SPECIFICATION_VERSION}")
    return cfg


@dataclass
class PostfreezeV604Context(PostfreezeContext):
    """The context of B10-B22 with the v6.0.4 lane, its lock and the third read-only source run."""

    third_root: Path | None = None
    v602_root: Path | None = None
    study_path: Path | None = None
    specification_check: dict[str, Any] | None = None
    identifier_tokens: set[str] = field(default_factory=set, repr=False)    # every case id an output of this call was checked against
    _third_records: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    config_key: ClassVar[str] = CONFIG_KEY
    analyses: ClassVar[tuple[str, ...]] = ANALYSES
    declared: ClassVar[str] = DECLARED

    def require_unlocked(self) -> str:
        return require_v604_unlocked(self)

    def runner(self, analysis: str) -> Callable[[PostfreezeContext], dict[str, Any]]:
        return importlib.import_module(f"{__package__}.{MODULES[analysis]}").run

    def write_provenance(self, analysis: str, inputs: Mapping[str, Path], outputs: list[Path], summary: Mapping[str, Any]) -> Path:
        """The provenance record of the shared context, checked for case and subject identifiers (``policy.identifiers``)
        before it is written (inputs, outputs, summary) and as written (the whole record)."""

        what = f"{OUTPUT_ROOT}/{str(analysis).upper()}/provenance.json"
        require_no_identifiers({"inputs": {k: self.portable(p) for k, p in inputs.items()}, "outputs": [self.portable(p) for p in outputs],
                                "summary": dict(summary)}, self.identifier_tokens, what)
        path = super().write_provenance(analysis, inputs, outputs, summary)
        require_no_identifiers(json.loads(path.read_text(encoding="utf-8")), self.identifier_tokens, what)
        return path

    # ------------------------------------------------------------------------------------------------ third source run
    def third_record(self, analysis: str) -> dict[str, Any]:
        """The provenance record ``postfreeze/<analysis>/provenance.json`` of the third source run after checking that it
        belongs to B23-B27 over the frozen source run and the B10-B22 run of the lane (BundleError, exit 7, otherwise)."""

        from ...stages import BundleError

        analysis = str(analysis).upper()
        if analysis not in self._third_records:
            path = self.third_root / OUTPUT_ROOT / analysis / "provenance.json"
            if not path.is_file():
                raise FileNotFoundError(f"{COMMAND}: {THIRD}:{OUTPUT_ROOT}/{analysis}/provenance.json not found in {self.third_root}")
            record = json.loads(path.read_text(encoding="utf-8"))
            source = record.get("source_run") or {}
            v602 = ((record.get("source_runs") or {}).get(V602) or {}).get("run_id")
            checks = {"label": record.get("label") == LABEL, "analysis": record.get("analysis") == analysis,
                      "specification_version": str(record.get("specification_version")) == THIRD_SPECIFICATION_VERSION,
                      "run_id": record.get("run_id") == self.third_root.name, "source_run": source.get("run_id") == self.source_root.name,
                      "release_manifest_sha256": source.get("release_manifest_sha256") == hash_path(self.source_root / "release_manifest.json"),
                      V602: v602 == str(self.lane_cfg["source_runs"][V602]["run_id"])}
            if self.tag_state == "protocol_frozen":
                checks["protocol_tag_state"] = record.get("protocol_tag_state") == "protocol_frozen"
            failed = [k for k, ok in checks.items() if not ok]
            if failed:
                raise BundleError(f"{COMMAND}: {THIRD}:{OUTPUT_ROOT}/{analysis}/provenance.json of {self.third_root.name} does not belong to "
                                  f"B23-B27 over {self.source_root.name} ({', '.join(failed)}); refused")
            self._third_records[analysis] = record
            self.verified_log[f"{THIRD}:{OUTPUT_ROOT}/{analysis}/provenance.json"] = path
        return self._third_records[analysis]

    def third_file(self, rel: str) -> Path:
        """The third-run file ``rel`` (``postfreeze/<B#>/...``) after checking its SHA-256 against the provenance record of
        ``<B#>`` or a ``FROZEN.json`` listed there; an unrecorded or changed file refuses (BundleError, exit 7)."""

        from ...stages import BundleError

        parts = str(rel).split("/")
        if len(parts) < 3 or parts[0] != OUTPUT_ROOT:
            raise BundleError(f"{COMMAND}: {THIRD}:{rel} is not an output of a post-freeze analysis")
        outputs = self.third_record(parts[1]).get("outputs") or {}
        expected, anchor = outputs.get(f"run:{rel}"), f"{OUTPUT_ROOT}/{parts[1]}/provenance.json"
        if expected is None:
            expected, anchor = self._frozen_anchor(parts, outputs)
        if expected is None:
            raise BundleError(f"{COMMAND}: {THIRD}:{rel} is recorded neither in {OUTPUT_ROOT}/{parts[1]}/provenance.json of {self.third_root.name} "
                              "nor in a FROZEN.json listed there; an unrecorded input is refused")
        path = self.third_root / rel
        key = path.resolve().as_posix()
        if key not in self._hashes:
            self._hashes[key] = hash_path(path) if path.is_file() else None
        if self._hashes[key] != expected:
            raise BundleError(f"{COMMAND}: {THIRD}:{rel} differs from the SHA-256 recorded in {anchor} of {self.third_root.name}; refused")
        self.verified_log[f"{THIRD}:{rel}"] = path
        return path

    def _frozen_anchor(self, parts: list[str], outputs: Mapping[str, Any]) -> tuple[str | None, str | None]:
        """(SHA-256, anchor) of a file listed in the ``files`` of the ``FROZEN.json`` of an enclosing directory that the
        provenance record lists and that is verified first."""

        for depth in range(len(parts) - 1, 1, -1):
            frozen_rel = "/".join([*parts[:depth], "FROZEN.json"])
            if f"run:{frozen_rel}" not in outputs:
                continue
            files = json.loads(self.third_file(frozen_rel).read_text(encoding="utf-8")).get("files") or {}
            name = "/".join(parts[depth:])
            if name in files:
                return str(files[name]), f"{frozen_rel} (listed in {OUTPUT_ROOT}/{parts[1]}/provenance.json)"
        return None, None

    def read_third_parquet(self, rel: str) -> pd.DataFrame:
        return pd.read_parquet(self.third_file(rel))

    def read_third_json(self, rel: str) -> Any:
        return json.loads(self.third_file(rel).read_text(encoding="utf-8"))

    def portable(self, path: str | Path) -> str:
        for key, root in ((THIRD, self.third_root), (V602, self.v602_root)):
            if root is not None and _inside(Path(path), root):
                return f"{key}:{Path(path).resolve().relative_to(root.resolve()).as_posix()}"
        return super().portable(path)

    def _read_only_roots(self) -> list[Path]:
        return [self.source_root, *[r for r in (self.third_root, self.v602_root) if r is not None]]

    def _provenance_extra(self) -> dict[str, Any]:
        third = {"run_id": self.third_root.name,
                 "provenance_sha256": {a: hash_path(self.third_root / OUTPUT_ROOT / a / "provenance.json") for a in sorted(self._third_records)}}
        v602 = {"run_id": str(self.lane_cfg["source_runs"][V602]["run_id"]), "read": False,
                "used_by": self.lane_cfg["source_runs"][V602].get("used_by")}
        return {"source_runs": {"frozen": self._source_record(), V602: v602, THIRD: third}, "specification_check": self.specification_check}


def make_v604_context(study_path: str | Path, source_run: str, run_id: str, *, artifact_root: str | Path | None = None,
                      data_root: str | Path | None = None, n_jobs: int | None = None, allow_unfrozen_synthetic: bool = False,
                      sources: Mapping[str, Path] | None = None) -> PostfreezeV604Context:
    """The context of one ``run-postfreeze --analysis B28..B31`` call (nothing is created yet)."""

    from ...config import load_study
    from ...stages import StageContext

    study = load_study(study_path)
    lane = config(study)
    if not run_id_pattern(lane).fullmatch(str(run_id)):
        raise ValueError(f"{COMMAND} --run-id {run_id!r} must match {lane['output']['run_id_template']} (configs/{CONFIG_KEY}.yaml output)")
    third_id, v602_id = str(lane["source_runs"][THIRD]["run_id"]), str(lane["source_runs"][V602]["run_id"])
    if str(run_id) in (str(source_run), third_id, v602_id):
        raise ValueError(f"{COMMAND}: --run-id must be a new run root, not a source run ({source_run}, {v602_id}, {third_id})")
    artifacts = Path(artifact_root) if artifact_root is not None else ARTIFACT_ROOT
    source = RunRoot(str(source_run), artifacts, create=False)
    if not source.path.is_dir() or not source.stages_dir.is_dir():
        raise FileNotFoundError(f"{COMMAND}: source run {source.path} not found (no .stages/)")
    third, v602 = artifacts / third_id, artifacts / v602_id
    if not (third / OUTPUT_ROOT).is_dir():
        raise FileNotFoundError(f"{COMMAND}: third source run {third} not found (no {OUTPUT_ROOT}/; configs/{CONFIG_KEY}.yaml source_runs)")
    run_root = artifacts / str(run_id)
    if any(_inside(run_root, root) for root in (source.path, third, v602)):
        raise ValueError(f"{COMMAND}: the new run root may not lie below a source run")
    data = Path(data_root) if data_root is not None else DATA_ROOT
    stage_ctx = StageContext(study=study, run=source, data_root=data, n_jobs=n_jobs, allow_unfrozen_synthetic=allow_unfrozen_synthetic,
                             sources=dict(sources or {}))
    return PostfreezeV604Context(source_root=source.path, run_root=run_root, staged_data_root=data, study=study, lane_cfg=lane,
                                 stage_ctx=stage_ctx, n_jobs=n_jobs, third_root=third, v602_root=v602, study_path=Path(study_path))


def specification_check(tag: str, root: Path | None = None) -> dict[str, Any]:
    """The specification check of D47 against ``tag`` (D52): the newest signed protocol tag HEAD descends from
    (:func:`tmmri.manuscript_tables.specification_tag`) must be ``tag`` and the working-tree ``SPECIFICATION.md`` must
    equal its blob (:func:`tmmri.manuscript_tables.spec_matches_tag`)."""

    from ... import manuscript_tables as mt

    root = Path(root) if root is not None else PROJECT_ROOT
    compared = mt.specification_tag(root)
    matches = mt.spec_matches_tag(root / "SPECIFICATION.md", tag) if compared == tag else None
    return {"required_tag": tag, "compared_tag": compared, "matches": matches, "passed": compared == tag and matches is True,
            "rule": "D47 (D52): SPECIFICATION.md equals the newest signed protocol tag HEAD descends from, which must be the tag of this block"}


def working_tree_check(tag: str, root: Path | None = None, paths: Iterable[str] = TAG_PINNED) -> dict[str, Any]:
    """Whether ``paths`` of the working tree equal their copies at ``tag`` (``git diff --quiet <tag> -- <paths>``) with
    no untracked file below them (``git ls-files --others --exclude-standard``); ``equal`` is None when git cannot
    answer."""

    root = Path(root) if root is not None else PROJECT_ROOT
    paths = [str(p) for p in paths]
    base = {"tag": tag, "paths": paths}

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "--no-optional-locks", *args], cwd=root, capture_output=True, text=True, timeout=300)

    try:
        diff = git("diff", "--quiet", tag, "--", *paths)
        untracked = git("ls-files", "--others", "--exclude-standard", "--", *paths)
    except (OSError, subprocess.SubprocessError):
        return {**base, "differs_from_tag": None, "n_untracked": None, "equal": None, "reason": "git unavailable"}
    if diff.returncode not in (0, 1) or untracked.returncode != 0:
        return {**base, "differs_from_tag": None, "n_untracked": None, "equal": None, "reason": "git cannot compare with the tag"}
    n_untracked = sum(1 for line in untracked.stdout.splitlines() if line.strip())
    return {**base, "differs_from_tag": diff.returncode == 1, "n_untracked": n_untracked, "equal": diff.returncode == 0 and not n_untracked}


def study_check(pctx: PostfreezeV604Context, root: Path | None = None) -> dict[str, Any]:
    """``--study`` must lie inside the project root, and the lane file it resolves must be the project's own
    ``configs/exploratory_postfreeze_v604.yaml`` (the file the tag pins)."""

    root = Path(root) if root is not None else PROJECT_ROOT
    lane_file = Path(lane_config_path(pctx.study, CONFIG_KEY))
    inside = pctx.study_path is not None and _inside(Path(pctx.study_path), root)
    own = lane_file.resolve() == (root / "configs" / f"{CONFIG_KEY}.yaml").resolve()
    return {"study_inside_project_root": bool(inside), "lane_file_is_project_file": bool(own), "passed": bool(inside and own)}


def require_v604_unlocked(pctx: PostfreezeV604Context) -> str:
    """The lock of B28-B31 (Decisions D48 and D52): the signed annotated tag ``protocol-v6.0.4`` contained in HEAD, the
    specification check of D47 against it, the lane file and ``src/`` equal to their copies at the tag, ``--study``
    inside the project root, and the source runs named in ``configs/exploratory_postfreeze_v604.yaml``;
    ``--allow-unfrozen-synthetic`` admits synthetic fixtures only.  Refuses with ProtocolFreezeError (exit 3)."""

    from ...stages import ProtocolFreezeError, protocol_frozen

    tag = amendment_tag(pctx.lane_cfg)
    runs = pctx.lane_cfg["source_runs"]
    frozen, third = str(runs["frozen"]["run_id"]), str(runs[THIRD]["run_id"])
    if pctx.stage_ctx.allow_unfrozen_synthetic:
        if synthetic_source(pctx):
            return "synthetic_override"
        raise ProtocolFreezeError(f"{COMMAND}: --allow-unfrozen-synthetic is accepted only for a synthetic source run (release_manifest.json "
                                  "synthetic_fixture and a synthetic ACDC staging manifest)")
    if pctx.source_root.name != frozen:
        raise ProtocolFreezeError(f"{COMMAND}: the source run is fixed to {frozen} (configs/{CONFIG_KEY}.yaml source_runs.frozen); "
                                  f"got {pctx.source_root.name}")
    if pctx.third_root is None or pctx.third_root.name != third:
        raise ProtocolFreezeError(f"{COMMAND}: the third source run is fixed to {third} (configs/{CONFIG_KEY}.yaml source_runs.{THIRD})")
    if not protocol_frozen(tag, signed=True):
        raise ProtocolFreezeError(f"{COMMAND}: nothing in the block B28-B31 runs on cohort data before the owner signs {tag} (signed "
                                  "annotated tag absent, unsigned, or HEAD does not descend from it)")
    check = specification_check(tag)
    if not check["passed"]:
        raise ProtocolFreezeError(f"{COMMAND}: the specification check of D47 does not pass against {tag} (newest signed protocol tag "
                                  f"{check['compared_tag']}, SPECIFICATION.md equal to its blob: {check['matches']}); refused")
    tree = working_tree_check(tag)
    if tree["equal"] is not True:
        raise ProtocolFreezeError(f"{COMMAND}: {' and '.join(tree['paths'])} must equal their copies at {tag} (differs from the tag: "
                                  f"{tree['differs_from_tag']}, untracked files: {tree['n_untracked']}); refused")
    study = study_check(pctx)
    if not study["passed"]:
        raise ProtocolFreezeError(f"{COMMAND}: --study must lie inside the project root {PROJECT_ROOT} and resolve its own "
                                  f"configs/{CONFIG_KEY}.yaml ({study}); refused")
    pctx.specification_check = {**check, "working_tree": tree, "study": study, "passed": True}
    return "protocol_frozen"


# ----------------------------------------------------------------------------------------------- shared helpers


def require_registered(pctx: PostfreezeContext, analysis: str, estimator_ids: Iterable[str]) -> None:
    registry = {str(e["id"]): e for e in pctx.lane_cfg["estimators"]}
    missing = [e for e in estimator_ids if e not in registry or str(registry[e].get("analysis")) != analysis]
    if missing:
        raise ValueError(f"{analysis}: estimators {missing} are not registered for {analysis} in configs/{CONFIG_KEY}.yaml estimators")


def block_partitions(pctx: PostfreezeContext, wanted: Iterable[str], requester: str) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """``([(label, split), ...], {label: reason})`` over ``wanted`` in order: ``acdc_dev_oof`` reads ``acdc_dev`` (its
    recorded out-of-fold reconstructions); an evaluated partition is read only when its locks pass."""

    included, skipped = [], {}
    for label in [str(p) for p in wanted]:
        split = "acdc_dev" if label == DEVELOPMENT else label
        reason = pctx.partition_status(split, requester)
        if reason is None:
            included.append((label, split))
        else:
            skipped[label] = reason
    return included, skipped


def half_up(num: int, den: int, digits: int = ROUNDING_DIGITS) -> str | None:
    """``num / den`` rounded half-up (ties away from zero) to ``digits`` decimals by integer arithmetic (BLD-1: values
    derived from exact counts, never binary floating-point rounding); None when ``den`` is 0."""

    num, den = int(num), int(den)
    if den == 0:
        return None
    negative = (num < 0) != (den < 0) and num != 0
    q = (2 * abs(num) * 10 ** digits + abs(den)) // (2 * abs(den))
    text = f"{q // 10 ** digits}.{q % 10 ** digits:0{digits}d}"
    return f"-{text}" if negative and q else text


def proportion(prefix: str, k: int, n: int, level: float = 0.95) -> dict[str, Any]:
    """A Wilson row ``<prefix>``, ``_k``, ``_n``, ``_ci_lower``, ``_ci_upper`` and ``_half_up`` (BLD-1)."""

    w = wilson(int(k), int(n), level)
    return {prefix: w["estimate"], f"{prefix}_k": int(k), f"{prefix}_n": int(n), f"{prefix}_ci_lower": w["ci_lower"], f"{prefix}_ci_upper": w["ci_upper"],
            f"{prefix}_half_up": half_up(k, n)}


def kappa_counts(a: Iterable[Any], b: Iterable[Any]) -> tuple[int, int]:
    """Cohen's kappa (unweighted) of two label sequences as the exact fraction (numerator, denominator) of integers:
    kappa = (n * agreements - sum_c n_a(c) n_b(c)) / (n^2 - sum_c n_a(c) n_b(c)) over the categories either uses."""

    a, b = np.asarray(list(a), dtype=object).astype(str), np.asarray(list(b), dtype=object).astype(str)
    if a.shape != b.shape:
        raise ValueError("kappa: the two label sequences differ in length")
    n = int(a.size)
    agree = int((a == b).sum())
    chance = sum(int((a == c).sum()) * int((b == c).sum()) for c in sorted(set(a) | set(b)))
    return n * agree - chance, n * n - chance


def cohen_kappa(a: Iterable[Any], b: Iterable[Any]) -> float | None:
    """Cohen's kappa (unweighted); None when undefined (no case, or both sequences use one and the same category)."""

    num, den = kappa_counts(a, b)
    return num / den if den else None


def kappa_row(pred: pd.Series, reference: pd.Series, clusters: pd.Series, boot: Mapping[str, Any], prefix: str = "kappa") -> dict[str, Any]:
    """Cohen's kappa of ``pred`` against ``reference`` on the cases where both exist, with the cluster bootstrap of the
    lane (``SeedSequence(seed).spawn(n)[r]``, unit cluster_id, percentile interval; replicates where kappa is undefined
    are dropped and counted) and the half-up value of the exact point estimate (BLD-1)."""

    from ..alt_bundles import bootstrap_statistic

    ref = reference.reindex(pred.index)
    rows = (pred.notna() & ref.notna()).to_numpy()
    ids = pred.index[rows]
    base = {f"{prefix}_n": int(len(ids))}
    if not len(ids):
        return {**base, prefix: None, f"{prefix}_ci_lower": None, f"{prefix}_ci_upper": None, f"{prefix}_half_up": None, f"{prefix}_status": "NE",
                f"{prefix}_reason": "NO_CASES"}
    a = pred.loc[ids].astype(str).to_numpy(dtype=object)
    b = ref.loc[ids].astype(str).to_numpy(dtype=object)

    def statistic(r: np.ndarray) -> float:
        value = cohen_kappa(a[r], b[r])
        return np.nan if value is None else value

    r = bootstrap_statistic(statistic, len(ids), clusters.reindex(ids).astype(str).to_numpy(dtype=object), n_boot=int(boot["replicates"]),
                            seed=int(boot["seed"]), level=float(boot.get("level", 0.95)), min_valid_fraction=0.0)
    num, den = kappa_counts(a, b)
    return {**base, prefix: r.get("estimate"), f"{prefix}_ci_lower": r.get("ci_lower"), f"{prefix}_ci_upper": r.get("ci_upper"),
            f"{prefix}_half_up": half_up(num, den), f"{prefix}_status": r.get("status", "EVALUATED"), f"{prefix}_reason": r.get("reason"),
            f"{prefix}_n_boot_valid": r.get("n_boot_valid")}


def quantiles(values: Iterable[float], qs: Mapping[str, float]) -> dict[str, Any]:
    """Named quantiles (numpy linear interpolation) of the finite ``values`` and their count."""

    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    return {"n": int(v.size), **{k: (float(np.quantile(v, q)) if v.size else None) for k, q in qs.items()}, "quantile_method": "numpy_linear"}


# --- policy.identifiers: no case or subject identifier leaves the run root


def identifier_tokens(ids: Iterable[str]) -> set[str]:
    """Every sample_id and the subject part after its dataset prefix."""

    return {t for i in ids for t in (str(i), str(i).split(":", 1)[-1]) if t}


def find_identifiers(value: Any, tokens: set[str]) -> list[str]:
    """Where an identifier column or a case or subject identifier occurs in a table or a JSON value: column names,
    index labels and text cells (counts and other numbers are never identifiers); the messages carry no value."""

    if isinstance(value, pd.DataFrame):
        hits = [f"column {c}" for c in value.columns if str(c) in IDENTIFIER_COLUMNS or str(c) in tokens]
        if any(str(i) in tokens for i in value.index if isinstance(i, str)):
            hits.append("index")
        for c in value.columns:
            col = value[c]
            if col.dtype == object or pd.api.types.is_string_dtype(col):
                if any(isinstance(v, str) and v in tokens for v in col.to_numpy()):
                    hits.append(f"cells of {c}")
        return hits
    if isinstance(value, Mapping):
        return [h for k, v in value.items() for h in ([f"key {k}"] if str(k) in tokens or str(k) in IDENTIFIER_COLUMNS else []) + find_identifiers(v, tokens)]
    if isinstance(value, (list, tuple)):
        return [h for v in value for h in find_identifiers(v, tokens)]
    return ["a text value"] if isinstance(value, str) and value in tokens else []


def require_no_identifiers(value: Any, tokens: set[str], what: str) -> None:
    hits = find_identifiers(value, tokens)
    if hits:
        raise RuntimeError(f"{COMMAND}: {what} would carry a case or subject identifier out of the run root ({', '.join(sorted(set(hits))[:5])}); refused")


def _remember(pctx: PostfreezeContext, tokens: set[str]) -> None:
    if isinstance(getattr(pctx, "identifier_tokens", None), set):
        pctx.identifier_tokens |= tokens                      # the provenance record is checked against them as well


def write_checked_table(frame: pd.DataFrame, path: Path, pctx: PostfreezeContext, tokens: set[str]) -> Path:
    _remember(pctx, tokens)
    require_no_identifiers(frame, tokens, pctx.portable(path))
    return write_table(frame, path, pctx)


def write_checked_json(value: Any, path: Path, pctx: PostfreezeContext, tokens: set[str]) -> Path:
    _remember(pctx, tokens)
    require_no_identifiers(value, tokens, pctx.portable(path))
    return write_json(value, path, pctx)
