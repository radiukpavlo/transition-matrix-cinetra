"""Manuscript Tables 1-10 of SPECIFICATION.md Section 16 from recorded run artifacts (stage ``build-manuscript-tables``).

Every table is written to ``<run>/manuscript_tables/`` as a CSV (one row per table row, first column ``panel``) and a
LaTeX booktabs fragment (one ``tabular`` per panel), and ``provenance.json`` maps every table and every column to the
artifact paths, the keys used and the SHA-256 of each source file.  Nothing is fitted, re-evaluated or hand-typed:
Table 1 is the reconciliation table of the frozen ``SPECIFICATION.md`` Section 2.2, Table 3 the frozen registry, and
Tables 2 and 4-10 come from the run's recorded contracts, metrics, rules and stage outputs.  Values the run did not
produce are ``NE`` with their reason.  Nothing is written to ``manuscript/``.

Licence (Section 14.5): the tables hold aggregates only.  No case identifier appears in any table or in the
provenance (checked against every identifier of the split manifest before anything is written), and no per-case
M&Ms content is copied.  A stratum with fewer cases than ``subgroup_reporting_thresholds.min_cases_per_stratum``
shows its n and ``NE (n < threshold)`` only (no point estimate); the internal metrics keep theirs.  The descriptive
demographic strata (sex, age band) are marked descriptive.

Protocol v6.0.1 (lane L2).  The manuscript-2.1 tables of :data:`TABLES_V601` (Table 3 H-primary-1 to H-primary-3
including ``mms2_external``, and the H4 family of Decision D36 from ``metrics/confirmatory_h4/``; Table 5 vendor and centre strata including M&Ms-2; Table 6 audited predictor beside the
measured-B comparator; supplementary Bland-Altman) are written beside Tables 1-10 with ``provenance_v6_0_1.json``
under the same rules (small strata counted in clusters).  Tables 1-10 and ``provenance.json`` are byte-identical with or
without the v6.0.1 summaries (``metrics/predictor_extended/`` is not read by them).  Absent inputs give
``NOT_RECORDED`` rows; a decision is read from a recorded fixed-sequence result, never derived.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .paths import PROJECT_ROOT
from .runs import RunRoot, atomic_write_bytes, atomic_write_json, hash_path

TABLES: tuple[tuple[str, str], ...] = (
    ("table01_methodological_reconciliation", "Methodological reconciliation relative to P1-P6 and v5 (SPECIFICATION.md 2.2)"),
    ("table02_cohort_flow", "Cohort flow: ACDC and M&Ms-1, exclusions, partitions, strata and QC exclusions"),
    ("table03_semantic_registry", "Final semantic registry: supported and unsupported features, tiers, transforms, harmonization"),
    ("table04_primary_reconstruction_acdc_test", "Primary reconstruction on acdc_test: per target versus comparators, rho and H-primary-1"),
    ("table05_external_reconstruction_mms_external", "External reconstruction on mms_external: per target, rho, H-primary-2 and strata"),
    ("table06_audited_predictor", "Audited predictor: OOF, acdc_test, crosswalk-evaluable M&Ms, IHD, off-ontology, semantic comparator"),
    ("table07_rulebook_metrics", "Rulebook metrics for the primary, companion and oracle rulebooks; routes and reason codes"),
    ("table08_refit_arm_mms_refit_eval", "Refit arm on mms_refit_eval, seen versus unseen vendors, paired with the frozen bundle"),
    ("table09_ablations", "Ablations: formal A 57 vs 128, typed vs identity transforms, discretizers, lambda policy, rule policy"),
    ("table10_failures_and_non_estimable", "Failures and non-estimable results"),
)
TABLE_NAMES = tuple(name for name, _ in TABLES)
PROVENANCE = "provenance.json"
NE = "NE"
# Protocol v6.0.1 (lane L2): the manuscript-2.1 tables are written beside Tables 1-10 (unchanged byte for byte) with
# their own provenance file; numbering follows docs/MANUSCRIPT_PROTOCOL_V2_2026-10-02.md Section K.
TABLES_V601: tuple[tuple[str, str], ...] = (
    ("v601_table03_hypotheses", "Manuscript Table 3: H-primary-1 to H-primary-3 and the H4 family (H4a, H4b) on acdc_test, mms_external and the prospective mms2_external"),
    ("v601_table05_vendor_strata", "Manuscript Table 5: vendor and centre strata of mms_external and mms2_external (rho, predictors, shift audit)"),
    ("v601_table06_predictors", "Manuscript Table 6: audited predictor beside the measured-B comparator; development OOF probabilities, ECE, Brier"),
    ("v601_tableS1_bland_altman", "Supplementary table: Bland-Altman agreement of B-hat with measured B per primary target"),
)
TABLE_NAMES_V601 = tuple(name for name, _ in TABLES_V601)
PROVENANCE_V601 = "provenance_v6_0_1.json"
EXTENDED_METRICS_DIR = "predictor_extended"     # metrics/predictor_extended/ (v6.0.1 summaries; not read by Tables 1-10)
H4_METRICS_DIR = "confirmatory_h4"              # metrics/confirmatory_h4/ (Decision D36 records; v6.0.1 Table 3 only)


class TableBuildError(RuntimeError):
    """A table cannot be built from the recorded artifacts (or would leak a case identifier)."""


# ----------------------------------------------------------------------------- sources and cells


@dataclass(frozen=True)
class Cell:
    """A table value with the source artifact (portable path) and the key it was read with."""

    value: Any
    path: str | None = None
    key: str | None = None


class Sources:
    """Read-only access to the declared source artifacts; every file read is hashed once (SHA-256)."""

    def __init__(self, run: RunRoot, allowed: Iterable[Path]) -> None:
        self.run = run
        self.allowed = {Path(p).resolve() for p in allowed}
        self.hashes: dict[str, str | None] = {}
        self._cache: dict[str, Any] = {}

    def label(self, path: Path) -> str:
        return self.run.portable(path)

    def _resolve(self, path: str | Path) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.run.path / p

    def exists(self, path: str | Path) -> bool:
        p = self._resolve(path)
        return p.resolve() in self.allowed and p.is_file()

    def _load(self, path: str | Path, reader) -> tuple[Any, str]:
        p = self._resolve(path)
        if p.resolve() not in self.allowed:
            raise TableBuildError(f"{p} is not a declared input of build-manuscript-tables")
        label = self.label(p)
        if label not in self._cache:
            self._cache[label] = reader(p)
            self.hashes[label] = hash_path(p)
        return self._cache[label], label

    def json(self, path: str | Path) -> tuple[Any, str] | tuple[None, str]:
        if not self.exists(path):
            return None, self.label(self._resolve(path))
        return self._load(path, lambda p: json.loads(p.read_text(encoding="utf-8")))

    def parquet(self, path: str | Path) -> tuple[pd.DataFrame | None, str]:
        if not self.exists(path):
            return None, self.label(self._resolve(path))
        return self._load(path, pd.read_parquet)

    def text(self, path: str | Path) -> tuple[str, str]:
        return self._load(path, lambda p: p.read_text(encoding="utf-8"))


class Table:
    """Rows of one manuscript table with per-column provenance (``panel`` is the first CSV column)."""

    def __init__(self, name: str, title: str, columns: list[str], notes: list[str] | None = None) -> None:
        self.name, self.title, self.columns = name, title, list(columns)
        self.notes = list(notes or [])
        self.rows: list[dict[str, Any]] = []
        self.provenance: dict[str, list[dict[str, str]]] = {}
        self.panels: list[str] = []

    def add(self, panel: str, **cells: Any) -> None:
        unknown = sorted(set(cells) - set(self.columns))
        if unknown:
            raise KeyError(f"{self.name}: unknown columns {unknown}")
        if panel not in self.panels:
            self.panels.append(panel)
        row: dict[str, Any] = {"panel": panel}
        for col, value in cells.items():
            if isinstance(value, Cell):
                row[col] = _plain(value.value)
                if value.path is not None:
                    entry = {"path": value.path, "key": str(value.key)}
                    bucket = self.provenance.setdefault(col, [])
                    if entry not in bucket:
                        bucket.append(entry)
            else:
                row[col] = _plain(value)
        self.rows.append(row)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["panel", *self.columns], dtype=object)


def _plain(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (list, tuple)):
        return "; ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return value


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _get(payload: Any, dotted: str) -> Any:
    value = payload
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


# ----------------------------------------------------------------------------- LaTeX


_LATEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                  "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "<": r"\textless{}", ">": r"\textgreater{}"}
_LATEX_UNICODE = {"≈": r"$\approx$", "ℝ": r"$\mathbb{R}$", "λ": r"$\lambda$", "α": r"$\alpha$", "β": r"$\beta$", "τ": r"$\tau$",
                  "ρ": r"$\rho$", "Σ": r"$\Sigma$", "×": r"$\times$", "≥": r"$\geq$", "≤": r"$\leq$", "→": r"$\rightarrow$", "±": r"$\pm$",
                  "≠": r"$\neq$", "∈": r"$\in$", "·": r"$\cdot$", "−": "-", "–": "--", "—": "---", "’": "'", "‘": "`", "“": "``", "”": "''",
                  "≥": r"$\geq$", "≤": r"$\leq$", "ε": r"$\varepsilon$", "σ": r"$\sigma$", "μ": r"$\mu$", " ": "~"}


def latex_escape(text: str) -> str:
    """Escape LaTeX specials; Markdown code spans become ``\\texttt``, bold markers are dropped, common Unicode math
    symbols map to LaTeX (anything else stays UTF-8)."""

    out: list[str] = []
    for i, segment in enumerate(str(text).replace("**", "").split("`")):
        buf: list[str] = []
        for ch in segment:
            if ch == "̂" and buf:          # combining circumflex: X-hat
                base = buf.pop()
                buf.append(rf"$\hat{{{base}}}$")
                continue
            buf.append(_LATEX_SPECIAL.get(ch) or _LATEX_UNICODE.get(ch) or ch)
        piece = "".join(buf)
        out.append(rf"\texttt{{{piece}}}" if i % 2 == 1 else piece)
    return "".join(out)


def _format_latex(value: Any, column: str) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if column.startswith(("p_value", "q_value")):
            return f"{value:.3g}"
        if value.is_integer() and (column == "n" or column.startswith("n_")):
            return str(int(value))
        if (value != 0 and abs(value) < 1e-3) or abs(value) >= 1e5:
            return latex_escape(f"{value:.2e}")
        return f"{value:.3f}"
    return latex_escape(str(value))


def render_latex(table: Table, run_id: str) -> str:
    lines = [f"% {table.name}: {table.title}",
             f"% Generated by tmmri build-manuscript-tables from run {run_id} (recorded artifacts only; see provenance.json). "
             "Requires booktabs; UTF-8 source.", *[f"% Note: {n}" for n in table.notes]]
    for panel in table.panels:
        part = [r for r in table.rows if r["panel"] == panel]
        cols = [c for c in table.columns if any(r.get(c) is not None for r in part)]
        if not cols:
            continue
        spec = []
        for c in cols:
            values = [r.get(c) for r in part if r.get(c) is not None]
            numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
            longest = max((len(str(v)) for v in values), default=0)
            spec.append("r" if numeric else ("p{4.2cm}" if longest > 40 else "l"))
        lines += [f"% Panel: {panel}", r"\begin{tabular}{" + "".join(spec) + "}", r"\toprule",
                  " & ".join(latex_escape(c) for c in cols) + r" \\", r"\midrule"]
        for row in part:
            lines.append(" & ".join(_format_latex(row.get(c), c) for c in cols) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- helpers


class Context:
    """Everything the table builders share: sources, configuration and the reporting threshold."""

    def __init__(self, run: RunRoot, study: Any, sources: Sources, *, include_mms: bool, registry_path: Path, include_mms2: bool = False) -> None:
        self.run, self.study, self.src = run, study, sources
        self.include_mms = include_mms
        self.include_mms2 = include_mms2
        self.registry_path = registry_path
        _, self.registry_label = sources.text(registry_path)      # hashed once; the parsed registry is the study's (same file)
        thresholds = study.bootstrap["subgroup_reporting_thresholds"]
        self.min_cases = int(thresholds["min_cases_per_stratum"])
        self.descriptive = [str(s) for s in thresholds.get("descriptive_strata", [])]
        self.small = f"NE (n < {self.min_cases})"
        self.features = study.features()

    def unit(self, target: str) -> str:
        return str(self.features.get(target, {}).get("unit") or "")

    def below(self, n: Any) -> bool:
        value = _num(n)
        return value is not None and value < self.min_cases


def _not_recorded(table: Table, panel: str, label: str, first_column: str) -> None:
    table.add(panel, **{first_column: Cell(f"NOT_RECORDED ({label})", label, "file absent")})


# ----------------------------------------------------------------------------- Table 1


def _reconciliation_rows(text: str) -> list[list[str]]:
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith("### 2.2"))
    except StopIteration as exc:
        raise TableBuildError("SPECIFICATION.md has no Section 2.2 reconciliation table") from exc
    rows = []
    for line in lines[start + 1:]:
        if line.startswith("## ") or line.startswith("### "):
            break
        if line.strip().startswith("|"):
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    if len(rows) < 3 or any(len(r) != len(rows[0]) for r in rows):
        raise TableBuildError("SPECIFICATION.md Section 2.2 reconciliation table is malformed")
    return rows


def spec_matches_tag(spec_path: Path, tag: str) -> bool | None:
    """True when the working-tree SPECIFICATION.md equals the blob of the protocol tag (line endings normalized);
    ``None`` when git cannot answer."""

    try:
        proc = subprocess.run(["git", "--no-optional-locks", "show", f"{tag}:SPECIFICATION.md"], cwd=spec_path.parent, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.replace(b"\r\n", b"\n") == spec_path.read_bytes().replace(b"\r\n", b"\n")


# --- protocol-v6.0.1 (owner decision F): the protocol tag a run was executed under

V601_STAGES = frozenset({
    "summarize-predictor-extended", "fit-cart-surrogate", "evaluate-cart-surrogate", "run-exploratory", "verify-torch-weights",
    "build-tensor-cache", "extract-deep-features", "fit-deep-encoder", "infer-f9", "plan-f9", "train-f9", "select-f9",
    "import-f9-fits", "freeze-f9", "upload-cache", "fit-probes", "evaluate-alt-bundles", "fit-trust", "evaluate-trust",
    "build-comparison-tables", "run-data-efficiency", "run-simulated-shift", "extract-mms2", "evaluate-h4"})


def is_v601_stage(name: str) -> bool:
    """A stage record of the protocol-v6.0.1 amendment (its stages, or any M&Ms-2 record)."""

    return str(name).split(":", 1)[0] in V601_STAGES or "mms2" in str(name).lower()


def run_protocol_tag(run: RunRoot, study: Any) -> str:
    """The protocol tag ``run`` was executed under (owner decision F).

    ``FROZEN.json`` ``protocol_version_tag`` when it is the newest configured tag (the run was frozen after that tag was
    signed; freeze-analysis-bundle records only a tag that exists with HEAD descending from it); otherwise the newest
    protocol tag of the configuration once the run holds a record of a protocol-v6.0.1 stage (for example
    ``evaluate:mms2_external`` on the frozen v6.0.0 bundle); otherwise the recorded ``protocol_version_tag`` or freeze
    tag (``protocol-v6.0.0`` for v6.0.0 runs).  Table 1 is checked against ``SPECIFICATION.md`` of that tag."""

    frozen: dict[str, Any] = {}
    if run.frozen_path.is_file():
        frozen = json.loads(run.frozen_path.read_text(encoding="utf-8"))
    recorded = frozen.get("protocol_version_tag")
    if recorded and str(recorded) == str(study.current_protocol_tag):
        return str(recorded)
    stages_dir = Path(run.stages_dir)
    names = [p.stem.replace("__", ":") for p in stages_dir.glob("*.json")] if stages_dir.is_dir() else []
    if any(is_v601_stage(n) for n in names):
        return str(study.current_protocol_tag)
    return str(recorded or frozen.get("protocol_tag") or study.freeze_tag)
# --- end protocol-v6.0.1


def table01(cx: Context, spec_path: Path) -> Table:
    t = Table(*TABLES[0], columns=["issue", "source_definitions", "binding_decision", "status"],
              notes=["Verbatim rows of the reconciliation table of the frozen SPECIFICATION.md Section 2.2 (protocol 6.0.0)."])
    text, label = cx.src.text(spec_path)
    rows = _reconciliation_rows(text)
    for i, row in enumerate(rows[2:], start=1):
        key = f"Section 2.2 reconciliation table, row {i}"
        t.add("reconciliation", **{c: Cell(v, label, f"{key}, column {j + 1}") for j, (c, v) in enumerate(zip(t.columns, row))})
    return t


# ----------------------------------------------------------------------------- Table 2


def table02(cx: Context) -> Table:
    from .stages import critical_qc_mask

    t = Table(*TABLES[1], columns=["cohort", "block", "group", "n", "n_clusters", "note"],
              notes=["Counts only (no per-case content); strata of mms_external by vendor, centre, release split, pathology, MMS_100 membership "
                     "and the never-analysed flag.", "QC exclusions: critical QC reason (QC_ABSTAIN) or non-finite formal A."])
    split, split_label = cx.src.parquet("contracts/split_manifest.parquet")
    clusters = None if split is None else split.set_index("sample_id")["cluster_id"].astype(str)

    def count(cohort: str, block: str, group: str, frame: pd.DataFrame, mask: pd.Series, label: str, key: str, with_clusters: bool = False) -> None:
        ids = frame.loc[mask, "sample_id"].astype(str)
        cells = {"cohort": cohort, "block": block, "group": group, "n": Cell(int(mask.sum()), label, f"count(sample_id) where {key}")}
        if with_clusters and clusters is not None:
            found = [c if isinstance(c, str) and c and c != "nan" else s for s, c in zip(ids, clusters.reindex(ids.to_numpy()).tolist())]
            cells["n_clusters"] = Cell(len(set(found)), split_label, f"nunique(cluster_id) where {key}")
        t.add(block, **cells)

    for ds, name in (("ACDC", "ACDC"), ("MMS", "M&Ms-1")):
        if ds == "MMS" and not cx.include_mms:
            t.add("release", cohort=name, block="release", group="all", note="NOT_RECORDED (evaluate:mms_external not recorded; M&Ms lock)")
            continue
        cohort, label = cx.src.parquet(f"contracts/cohort_{ds}.parquet")
        if cohort is None:
            _not_recorded(t, "release", label, "cohort")
            continue
        count(name, "release", "all cases", cohort, pd.Series(True, index=cohort.index), label, "all rows")
        for value in sorted(cohort["release_split"].astype(str).unique()):
            count(name, "release_split", value, cohort, cohort["release_split"].astype(str) == value, label, f"release_split == {value!r}")
        eligible = cohort["eligible_v6"].astype(bool)
        count(name, "eligibility", "eligible (v6)", cohort, eligible, label, "eligible_v6")
        reasons = cohort.loc[~eligible, "exclusion_reason"].astype(str)
        for value in sorted(reasons.unique()):
            count(name, "exclusion", value, cohort, (~eligible) & (cohort["exclusion_reason"].astype(str) == value), label,
                  f"not eligible_v6 and exclusion_reason == {value!r}")
        if ds == "ACDC":
            for part in ("acdc_dev", "acdc_test"):
                count(name, "partition", part, cohort, eligible & (cohort["partition"].astype(str) == part), label,
                      f"eligible_v6 and partition == {part!r}", with_clusters=True)
            partitions = {"acdc_dev": None, "acdc_test": None}
        else:
            ext = cohort["mms_external"].astype(bool)
            count(name, "partition", "mms_external", cohort, ext, label, "mms_external", with_clusters=True)
            for var in ("vendor", "centre", "release_split", "pathology", "mms100_member", "never_analysed"):
                values = cohort.loc[ext, var].map(_label)
                for value in sorted(values.unique()):
                    count(name, f"stratum:{var}", value, cohort, ext & (cohort[var].map(_label) == value), label,
                          f"mms_external and {var} == {value!r}", with_clusters=True)
            compat = cohort["crosswalk_compatibility"].astype(str)
            for value in sorted(compat[ext].unique()):
                count(name, "crosswalk", value, cohort, ext & (compat == value), label, f"mms_external and crosswalk_compatibility == {value!r}")
            for part in ("mms_refit_train", "mms_refit_eval"):
                count(name, "refit_partition", part, cohort, cohort["refit_partition"].astype(str) == part, label, f"refit_partition == {part!r}",
                      with_clusters=True)
            partitions = {"mms_external": None}
        b, b_label = cx.src.parquet(f"semantics/B_{ds}.parquet")
        a, a_label = cx.src.parquet(f"formal/A_{ds}.parquet")
        if split is None or b is None or a is None:
            continue
        from .formal_features import formal_descriptor_names

        crit = pd.Series(critical_qc_mask(b, cx.study).to_numpy(dtype=bool), index=b["sample_id"].astype(str))
        names = formal_descriptor_names()
        nonfinite = pd.Series(~np.isfinite(a[names].to_numpy(dtype=float)).all(axis=1), index=a["sample_id"].astype(str))
        for part in partitions:
            ids = split.loc[split["partition"] == part, "sample_id"].astype(str)
            t.add("qc_exclusion", cohort=name, block="qc_exclusion", group=f"{part}: critical QC (QC_ABSTAIN)",
                  n=Cell(int(crit.reindex(ids).fillna(False).astype(bool).sum()), b_label, f"count critical qc_failure_reason (cohorts.qc_severity.critical) in {part}"))
            t.add("qc_exclusion", cohort=name, block="qc_exclusion", group=f"{part}: non-finite formal A (INPUT_NONFINITE)",
                  n=Cell(int(nonfinite.reindex(ids).fillna(False).astype(bool).sum()), a_label, f"count rows with a non-finite descriptor in {part}"))
    return t


def _label(value: Any) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "<missing>"
    if isinstance(value, float) and float(value).is_integer():
        return str(int(value))
    return str(value)


# ----------------------------------------------------------------------------- Table 3


def table03(cx: Context) -> Table:
    t = Table(*TABLES[2], columns=["feature", "family", "support", "unit", "evidence_acdc", "evidence_mms", "target_transform", "transition_role",
                                   "rule_role", "discretization", "harmonization_mms", "primary_family", "n"],
              notes=["From the frozen registry/semantic_registry.yaml (registry version 6.0.0)."])
    label = cx.registry_label
    registry = cx.study.registry

    def evidence(value: Any) -> str:
        return "+".join(str(v) for v in value) if isinstance(value, (list, tuple)) else str(value)

    for i, f in enumerate(registry.get("features", [])):
        family = str(f.get("family"))
        support = ("UNSUPPORTED (reserved)" if family == "reserved_unsupported" else
                   "NOT_VALID (geometric approximation)" if family == "geometric_approximation_not_valid" else "SUPPORTED")
        key = f"features[{i}] (name {f.get('name')})"
        ev = f.get("evidence_sources") or {}
        t.add("features", feature=Cell(f.get("name"), label, f"{key}.name"), family=Cell(family, label, f"{key}.family"),
              support=Cell(support, label, f"{key}.family"), unit=Cell(f.get("unit") or "", label, f"{key}.unit"),
              evidence_acdc=Cell(evidence(ev.get("acdc")), label, f"{key}.evidence_sources.acdc"),
              evidence_mms=Cell(evidence(ev.get("mms")), label, f"{key}.evidence_sources.mms"),
              target_transform=Cell(f.get("target_transform"), label, f"{key}.target_transform"),
              transition_role=Cell(f.get("transition_role"), label, f"{key}.transition_role"),
              rule_role=Cell(f.get("rule_role"), label, f"{key}.rule_role"),
              discretization=Cell(f.get("discretization"), label, f"{key}.discretization"),
              harmonization_mms=Cell((f.get("harmonization") or {}).get("mms"), label, f"{key}.harmonization.mms"),
              primary_family=Cell(f.get("primary_family"), label, f"{key}.primary_family"))
    for name, value in (registry.get("counts") or {}).items():
        if isinstance(value, dict):
            for sub, n in value.items():
                t.add("counts", feature=f"{name}.{sub}", n=Cell(int(n), label, f"counts.{name}.{sub}"))
        else:
            t.add("counts", feature=name, n=Cell(int(value), label, f"counts.{name}"))
    return t


# ----------------------------------------------------------------------------- Tables 4 and 5 (reconstruction)


RECON_COLUMNS = ["target", "unit", "model", "role", "stratum_variable", "stratum", "n", "n_clusters", "mae", "bias", "nmae_fit_sd", "rho_j",
                 "rho_j_ci_lower", "rho_j_ci_upper", "delta_mae_mean_minus_ridge", "p_value", "q_value", "mae_pseudoinverse_p1",
                 "mae_rbf_kernel_ridge", "rho", "ci_lower", "ci_upper", "decision", "status", "note"]


def _sequence(cx: Context, hypothesis: str) -> tuple[Cell, dict | None]:
    payload, label = cx.src.json("metrics/fixed_sequence.json")
    if payload is None:
        return Cell("NOT_RECORDED (metrics/fixed_sequence.json)", label, "file absent"), None
    for i, item in enumerate(payload.get("results", [])):
        if item.get("hypothesis_id") == hypothesis:
            return Cell(item.get("status"), label, f"results[{i}].status (hypothesis_id {hypothesis})"), item
    return Cell("NOT_RECORDED", label, f"results with hypothesis_id {hypothesis}"), None


def _per_target_panel(cx: Context, t: Table, split: str, family: str) -> None:
    errors, e_label = cx.src.parquet(f"metrics/per_target_{split}.parquet")
    rho, r_label = cx.src.json(f"metrics/primary_rho_{split}.json")
    tests, m_label = cx.src.parquet(f"metrics/multiplicity_{family}.parquet")
    variants, v_label = cx.src.parquet("metrics/ablations/transition_variants_per_target.parquet")
    if errors is None:
        _not_recorded(t, "per_target", e_label, "target")
        return
    overall = None
    if tests is not None:
        overall = tests[(tests["stratum_variable"].astype(str) == "all") & (tests["partition"].astype(str) == split)].set_index("target")
    for _, row in errors.iterrows():
        target = str(row["target"])
        key = f"target == {target!r}"
        cells: dict[str, Any] = {"target": Cell(target, e_label, key), "unit": Cell(cx.unit(target), cx.registry_label, f"features[name == {target!r}].unit"),
                                 "n": Cell(row.get("n"), e_label, f"{key}: n"), "mae": Cell(row.get("mae"), e_label, f"{key}: mae"),
                                 "bias": Cell(row.get("bias"), e_label, f"{key}: bias"),
                                 "nmae_fit_sd": Cell(row.get("nmae_fit_sd"), e_label, f"{key}: nmae_fit_sd"),
                                 "status": Cell(row.get("status"), e_label, f"{key}: status")}
        if isinstance(rho, dict) and target in (rho.get("rho_j") or {}):
            ci = (rho.get("rho_j_ci") or {}).get(target) or [None, None]
            cells.update(rho_j=Cell(rho["rho_j"][target], r_label, f"rho_j.{target}"), rho_j_ci_lower=Cell(ci[0], r_label, f"rho_j_ci.{target}[0]"),
                         rho_j_ci_upper=Cell(ci[1], r_label, f"rho_j_ci.{target}[1]"))
        if overall is not None and target in overall.index:
            test = overall.loc[target]
            tkey = f"stratum_variable == 'all' and target == {target!r}"
            cells.update(delta_mae_mean_minus_ridge=Cell(test.get("effect"), m_label, f"{tkey}: effect"),
                         p_value=Cell(test.get("p_value"), m_label, f"{tkey}: p_value"), q_value=Cell(test.get("q_value"), m_label, f"{tkey}: q_value"))
        if split == "acdc_test" and variants is not None:
            for variant in ("pseudoinverse_p1", "rbf_kernel_ridge"):
                hit = variants[(variants["variant"] == variant) & (variants["partition"] == split) & (variants["target"] == target)]
                if len(hit):
                    cells[f"mae_{variant}"] = Cell(hit["mae"].iloc[0], v_label, f"variant == {variant!r} and partition == {split!r} and {key}: mae")
        t.add("per_target", **cells)


def _rho_row(cx: Context, t: Table, panel: str, entry: dict | None, label: str, key: str, *, model: str, role: str,
             decision: Cell | None = None, stratum: tuple[str, str] | None = None, suppress: bool = False, note: str | None = None) -> None:
    """One rho row from the recorded ``entry`` (found at ``key`` of the artifact ``label``); a stratum below the
    reporting threshold keeps only its n (``suppress``)."""

    cells: dict[str, Any] = {"model": model, "role": role}
    if stratum is not None:
        cells.update(stratum_variable=stratum[0], stratum=stratum[1])
    if not isinstance(entry, dict):
        cells.update(status="NOT_RECORDED", note=f"{label} {key or ''}".strip())
        t.add(panel, **cells)
        return
    prefix = f"{key}." if key else ""
    n = entry.get("n_cases", entry.get("n_units"))
    cells.update(n=Cell(n, label, f"{prefix}n_cases"), n_clusters=Cell(entry.get("n_clusters"), label, f"{prefix}n_clusters"))
    if suppress and cx.below(n):
        cells.update(rho=cx.small, ci_lower=cx.small, ci_upper=cx.small, status=cx.small, note="STRATUM_BELOW_MIN_CASES: point estimate withheld")
    else:
        cells.update(rho=Cell(entry.get("rho"), label, f"{prefix}rho"), ci_lower=Cell(entry.get("ci_lower"), label, f"{prefix}ci_lower"),
                     ci_upper=Cell(entry.get("ci_upper"), label, f"{prefix}ci_upper"), status=Cell(entry.get("status"), label, f"{prefix}status"))
        if entry.get("reason"):
            cells["note"] = Cell(entry.get("reason"), label, f"{prefix}reason")
    if decision is not None:
        cells["decision"] = decision
    if note and "note" not in cells:
        cells["note"] = note
    t.add(panel, **cells)


def table04(cx: Context) -> Table:
    t = Table(*TABLES[3], columns=RECON_COLUMNS,
              notes=["Per-target MAE and bias in physical units, nMAE divided by the fit-partition SD; rho_j with percentile 95 % cluster-bootstrap "
                     "intervals; F1 Wilcoxon p and BH q within F1.", "Comparators refitted on acdc_dev only (run-ablations, secondary)."])
    _per_target_panel(cx, t, "acdc_test", "F1")
    rho, label = cx.src.json("metrics/primary_rho_acdc_test.json")
    decision, _ = _sequence(cx, "H-primary-1")
    _rho_row(cx, t, "family", rho, label, "", model="ridge_per_target (frozen bundle)", role="primary; H-primary-1", decision=decision)
    variants, v_label = cx.src.json("metrics/ablations/transition_variants.json")
    for name in ("pseudoinverse_p1", "rbf_kernel_ridge"):
        _rho_row(cx, t, "family", _get(variants, f"acdc_test.{name}"), v_label, f"acdc_test.{name}", model=name, role="secondary comparator (run-ablations)")
    return t


def table05(cx: Context) -> Table:
    t = Table(*TABLES[4], columns=RECON_COLUMNS,
              notes=[f"Strata with fewer than {cx.min_cases} cases show their n and {cx.small} only.",
                     "Descriptive demographic strata (sex, age band) are descriptive: counts, point estimates and intervals, never tested."])
    if not cx.include_mms:
        t.add("per_target", status="NOT_RECORDED", note="evaluate:mms_external not recorded")
        return t
    _per_target_panel(cx, t, "mms_external", "F3")
    rho, label = cx.src.json("metrics/primary_rho_mms_external.json")
    decision, _ = _sequence(cx, "H-primary-2")
    _rho_row(cx, t, "family", rho, label, "", model="ridge_per_target (frozen bundle)", role="primary; H-primary-2 (fixed sequence)", decision=decision)
    never, n_label = cx.src.json("metrics/primary_rho_mms_external_never_analysed.json")
    decision, _ = _sequence(cx, "H-primary-2-never-analysed")
    _rho_row(cx, t, "family", never, n_label, "", model="ridge_per_target (frozen bundle)", role="220 never-analysed cases; sensitivity, no gatekeeping",
             decision=decision, stratum=("never_analysed", "True"))
    strata, s_label = cx.src.json("metrics/strata_rho_mms_external.json")
    if strata is None:
        _not_recorded(t, "strata", s_label, "stratum_variable")
        return t
    groups = strata.get("strata", {}) or {}
    _rho_row(cx, t, "family", (groups.get("mms100_member") or {}).get("True"), s_label, "strata.mms100_member.True", model="ridge_per_target (frozen bundle)",
             role="100 previously analysed cases (MMS_100; previously unblinded)", stratum=("mms100_member", "True"), suppress=True)
    descriptive = set(strata.get("descriptive_strata") or cx.descriptive)
    for var in sorted(groups):
        for value in sorted(groups[var]):
            entry = groups[var][value]
            role = "descriptive demographic stratum (descriptive)" if var in descriptive or entry.get("stratum_role") != "prespecified" else "prespecified stratum"
            _rho_row(cx, t, "strata", entry, s_label, f"strata[{var!r}][{value!r}]", model="ridge_per_target (frozen bundle)", role=role,
                     stratum=(var, value), suppress=True)
    return t


# ----------------------------------------------------------------------------- Table 6


PREDICTOR_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "macro_ovr_auroc", "ece", "log_loss")


def _predictor_metric_rows(cx: Context, t: Table, panel: str, rel: str, partition: str) -> dict | None:
    payload, label = cx.src.json(rel)
    if payload is None:
        t.add(panel, partition=partition, status="NOT_RECORDED", note=f"{label} (run summarize-predictor)")
        return None
    counts = payload.get("counts", {})
    for m in PREDICTOR_METRICS:
        entry = (payload.get("metrics") or {}).get(m) or {}
        key = f"metrics.{m}"
        cells = {"partition": partition, "metric": m, "n": Cell(counts.get("n_evaluable"), label, "counts.n_evaluable"),
                 "n_clusters": Cell(counts.get("n_clusters"), label, "counts.n_clusters"),
                 "status": Cell(entry.get("status"), label, f"{key}.status")}
        if entry.get("status") == NE:
            cells.update(estimate=NE, note=Cell(entry.get("reason"), label, f"{key}.reason"))
        else:
            cells.update(estimate=Cell(entry.get("estimate"), label, f"{key}.estimate"))
            if "ci_lower" in entry:
                cells.update(ci_lower=Cell(entry.get("ci_lower"), label, f"{key}.ci_lower"), ci_upper=Cell(entry.get("ci_upper"), label, f"{key}.ci_upper"))
            if m == "ece":
                cells["note"] = f"{payload.get('ece_bins')} equal-mass bins; realized counts in {label}"
        t.add(panel, **cells)
    return payload


def table06(cx: Context) -> Table:
    t = Table(*TABLES[5], columns=["partition", "metric", "group", "class", "n", "n_clusters", "estimate", "ci_lower", "ci_upper", "status", "note"],
              notes=["Percentile 95 % intervals of the fixed-model cluster bootstrap; ECE with 10 equal-mass bins.",
                     "mms_external: crosswalk same-definition cases only; IHD -> MINF (HARMONIZABLE_NOT_EXACT) is reported separately and never pooled; "
                     "off-ontology cases as the distribution of d_model only.",
                     f"Groups with fewer than {cx.min_cases} cases show their n and {cx.small} only."])
    _predictor_metric_rows(cx, t, "metrics", "metrics/predictor_acdc_dev_oof.json", "acdc_dev (cross-fitted OOF)")
    _predictor_metric_rows(cx, t, "metrics", "metrics/predictor_acdc_test.json", "acdc_test")
    if cx.include_mms:
        ext = _predictor_metric_rows(cx, t, "metrics", "metrics/predictor_mms_external.json", "mms_external (crosswalk same-definition)")
        if ext is not None:
            _, label = cx.src.json("metrics/predictor_mms_external.json")
            found = _get(ext, "crosswalk.found") or {}
            for group, n in sorted(found.items()):
                excluded = _get(ext, f"crosswalk.excluded_critical_qc.{group}")
                t.add("crosswalk", group=group, n=Cell(n, label, f"crosswalk.found.{group}"),
                      note=Cell(f"excluded for critical QC: {excluded}", label, f"crosswalk.excluded_critical_qc.{group}"))
            t.add("crosswalk", group="evaluable (same definition, QC passed)", n=Cell(_get(ext, "crosswalk.evaluable_same_definition"), label,
                                                                                     "crosswalk.evaluable_same_definition"))
            for block, panel, title in (("ihd_not_exact", "ihd_not_exact", "IHD -> MINF (HARMONIZABLE_NOT_EXACT)"),
                                        ("off_ontology", "off_ontology", "off-ontology (OFF_ONTOLOGY)")):
                entry = ext.get(block) or {}
                n = entry.get("n")
                if cx.below(n):
                    t.add(panel, group=title, n=Cell(n, label, f"{block}.n"), estimate=cx.small, status=cx.small,
                          note="STRATUM_BELOW_MIN_CASES: distribution withheld")
                    continue
                t.add(panel, group=title, n=Cell(n, label, f"{block}.n"), status="REPORTED_SEPARATELY" if block == "ihd_not_exact" else "DISTRIBUTION_ONLY")
                for cls, k in (entry.get("d_model_distribution") or {}).items():
                    t.add(panel, group=title, **{"class": f"d_model = {cls}"}, n=Cell(k, label, f"{block}.d_model_distribution.{cls}"))
                if block == "ihd_not_exact":
                    t.add(panel, group=title, metric="proportion d_model = mapped class", n=Cell(n, label, f"{block}.n"),
                          estimate=Cell(entry.get("proportion_d_model_equals_mapped_class"), label, f"{block}.proportion_d_model_equals_mapped_class"),
                          ci_lower=Cell((entry.get("wilson_95") or [None])[0], label, f"{block}.wilson_95[0]"),
                          ci_upper=Cell((entry.get("wilson_95") or [None, None])[1], label, f"{block}.wilson_95[1]"), note="Wilson 95 % interval; not pooled")
    else:
        t.add("metrics", partition="mms_external", status="NOT_RECORDED", note="evaluate:mms_external not recorded")
    summary, s_label = cx.src.json("metrics/predictor_summary.json")
    comparator = (summary or {}).get("semantic_comparator")
    if isinstance(comparator, dict):
        t.add("semantic_comparator", partition=Cell(comparator.get("name"), s_label, "semantic_comparator.name"), estimate=NE,
              status=Cell(comparator.get("status"), s_label, "semantic_comparator.status"), note=Cell(comparator.get("reason"), s_label, "semantic_comparator.reason"))
    else:
        name = str((cx.study.formal.get("comparator_predictor") or {}).get("name", "semantic comparator"))
        t.add("semantic_comparator", partition=name, estimate=NE, status=NE, note="COMPARATOR_NOT_FITTED_IN_RUN")
    return t


# ----------------------------------------------------------------------------- Table 7


ROUTE_RATES = ("activation", "accepted_coverage", "abstention", "covered_fidelity", "faithful_coverage", "covered_reference_accuracy", "conflict_rate",
               "conflict_encounter_rate")
ROUTE_DENOMINATORS = ("n_attempted", "n_qc_abstain", "n_processed", "n_accepted", "n_accepted_reference_known")


def _rulebooks(cx: Context) -> list[tuple[str, str, str]]:
    return [(str(v["name"]), str(role), str(v.get("substrate", ""))) for role, v in cx.study.rule_policy["rulebooks"].items()]


def _route_rows(cx: Context, t: Table, rel: str, rulebook: str, role: str, partition: str, eligible: Cell | None, book: Cell | None) -> None:
    payload, label = cx.src.json(rel)
    base = {"rulebook": rulebook, "role": role, "partition": partition}
    if payload is None:
        t.add("denominators", **base, note=f"NOT_RECORDED ({label})")
        return
    t.add("denominators", **base, rulebook_file=book, eligible_rules=eligible,
          **{k: Cell(payload.get(k), label, k) for k in ROUTE_DENOMINATORS}, n_clusters=Cell(payload.get("n_clusters"), label, "n_clusters"))
    rates: dict[str, Any] = {}
    for k in ROUTE_RATES:
        value = payload.get(k)
        rates[k] = Cell(value, label, k) if value is not None else Cell(NE, label, f"{k} (null: no accepted output)")
    tax = payload.get("audit_tax")
    rates["audit_tax"] = Cell(tax.get("difference_rule_minus_base"), label, "audit_tax.difference_rule_minus_base") if isinstance(tax, dict) else \
        Cell(NE, label, "audit_tax (null: no accepted reference-known output)")
    t.add("rates", **base, **rates)
    for route, n in sorted((payload.get("routes") or {}).items()):
        t.add("routes", **base, code=route, n_code=Cell(n, label, f"routes.{route}"))
    for code, entry in sorted((payload.get("abstention_by_reason") or {}).items()):
        t.add("reason_codes", **base, code=code, n_code=Cell(entry.get("n"), label, f"abstention_by_reason.{code}.n"),
              rate=Cell(entry.get("rate"), label, f"abstention_by_reason.{code}.rate"))


ROUTE_COLUMNS = ["rulebook", "role", "partition", "rulebook_file", "eligible_rules", *ROUTE_DENOMINATORS, "n_clusters", *ROUTE_RATES, "audit_tax",
                 "code", "n_code", "rate", "note"]


def table07(cx: Context) -> Table:
    t = Table(*TABLES[6], columns=ROUTE_COLUMNS,
              notes=["Denominator reconciliation: attempted inputs, QC exclusions (QC_ABSTAIN), processed inputs, accepted outputs and "
                     "label-evaluable accepted outputs.", "Covered metrics are NE when no output is accepted (coverage 0, abstention 1).",
                     "mms_external reference: crosswalk same-definition cases only."])
    partitions = ["acdc_test"] + (["mms_external"] if cx.include_mms else [])
    for rb, role, _ in _rulebooks(cx):
        pointer, p_label = cx.src.json(f"rules/{rb}/rulebook_or_empty.json")
        book = Cell(pointer.get("file"), p_label, "file") if isinstance(pointer, dict) else None
        eligible = Cell(pointer.get("eligible_rules"), p_label, "eligible_rules") if isinstance(pointer, dict) else None
        for part in partitions:
            _route_rows(cx, t, f"metrics/routes_{rb}_{part}.json", rb, role, part, eligible, book)
    return t


# ----------------------------------------------------------------------------- Table 8


def table08(cx: Context) -> Table:
    t = Table(*TABLES[7], columns=["arm", "stratum", "rulebook", "metric", "n", "n_clusters", "estimate", "ci_lower", "ci_upper", "accepted_coverage",
                                   "covered_fidelity", "covered_reference_accuracy", "conflict_rate", "status", "note"],
              notes=["Secondary ceiling (Decision D17): never pooled with the primary arm; identical case order and bootstrap seed for both arms.",
                     f"Vendor groups with fewer than {cx.min_cases} cases show their n and {cx.small} only."])
    if not cx.include_mms:
        t.add("rho", status="NOT_RECORDED", note="evaluate:mms_external not recorded")
        return t
    rho, label = cx.src.json("metrics/refit_arm/primary_rho_mms_refit_eval.json")
    if rho is None:
        t.add("rho", status="NOT_RECORDED", note=f"{label} (fit-refit-arm)")
        return t
    for arm in ("refit_arm", "frozen_acdc_bundle"):
        for stratum in ("all", "seen_vendors", "unseen_vendors"):
            entry = _get(rho, f"{arm}.{stratum}")
            key = f"{arm}.{stratum}"
            if not isinstance(entry, dict):
                t.add("rho", arm=arm, stratum=stratum, status="NOT_RECORDED")
                continue
            cells = {"arm": arm, "stratum": stratum, "metric": "rho", "n": Cell(entry.get("n_cases"), label, f"{key}.n_cases"),
                     "n_clusters": Cell(entry.get("n_clusters"), label, f"{key}.n_clusters")}
            if stratum != "all" and cx.below(entry.get("n_cases")):
                cells.update(estimate=cx.small, ci_lower=cx.small, ci_upper=cx.small, status=cx.small, note="STRATUM_BELOW_MIN_CASES: point estimate withheld")
            else:
                cells.update(estimate=Cell(entry.get("rho"), label, f"{key}.rho"), ci_lower=Cell(entry.get("ci_lower"), label, f"{key}.ci_lower"),
                             ci_upper=Cell(entry.get("ci_upper"), label, f"{key}.ci_upper"), status=Cell(entry.get("status"), label, f"{key}.status"))
            t.add("rho", **cells)
    for rb, _, _ in _rulebooks(cx):
        payload, r_label = cx.src.json(f"metrics/refit_arm/routes_{rb}_mms_refit_eval.json")
        if payload is None:
            t.add("routes", arm="refit_arm", rulebook=rb, status="NOT_RECORDED")
            continue
        cells = {k: (Cell(payload.get(k), r_label, k) if payload.get(k) is not None else Cell(NE, r_label, f"{k} (null)"))
                 for k in ("accepted_coverage", "covered_fidelity", "covered_reference_accuracy", "conflict_rate")}
        t.add("routes", arm="refit_arm", stratum="all", rulebook=rb, n=Cell(payload.get("n_attempted"), r_label, "n_attempted"), **cells)
    predictor, p_label = cx.src.json("metrics/refit_arm/predictor_mms_refit_eval.json")
    if predictor is not None:
        for m in PREDICTOR_METRICS:
            entry = (predictor.get("metrics") or {}).get(m) or {}
            cells = {"arm": "refit_arm", "stratum": "all", "metric": f"predictor {m}", "n": Cell(_get(predictor, "counts.n_evaluable"), p_label, "counts.n_evaluable"),
                     "status": Cell(entry.get("status"), p_label, f"metrics.{m}.status")}
            if entry.get("status") == NE:
                cells.update(estimate=NE, note=Cell(entry.get("reason"), p_label, f"metrics.{m}.reason"))
            else:
                cells.update(estimate=Cell(entry.get("estimate"), p_label, f"metrics.{m}.estimate"))
                if "ci_lower" in entry:
                    cells.update(ci_lower=Cell(entry.get("ci_lower"), p_label, f"metrics.{m}.ci_lower"), ci_upper=Cell(entry.get("ci_upper"), p_label, f"metrics.{m}.ci_upper"))
            t.add("predictor", **cells)
    summary, s_label = cx.src.json("metrics/refit_arm/refit_arm_summary.json")
    if summary is not None:
        for k in ("n_train_partition", "n_train_fit_rows", "n_eval"):
            t.add("summary", arm="refit_arm", metric=k, n=Cell(summary.get(k), s_label, k))
        t.add("summary", arm="refit_arm", metric="predictor_classes", note=Cell(summary.get("predictor_classes"), s_label, "predictor_classes"))
    return t


# ----------------------------------------------------------------------------- Table 9


VARIANT_AXES = {"ridge_typed_formal57": "primary: typed ridge on 57 descriptors", "ridge_typed_p6_128": "formal A: 128-d P6 projection",
                "ridge_identity_transform": "target transforms: identity", "pseudoinverse_p1": "transition: pseudoinverse (P1)",
                "ridge_global_lambda": "lambda policy: one global lambda", "rbf_kernel_ridge": "transition: RBF kernel ridge"}


def _tests_by_variant(cx: Context, family: str) -> tuple[pd.DataFrame | None, str]:
    frame, label = cx.src.parquet(f"metrics/ablations/multiplicity_{family}.parquet")
    return (None if frame is None else frame.set_index("variant")), label


def _route_cells(payload: dict, label: str, prefix: str) -> dict[str, Cell]:
    routes = payload.get("acdc_test_routes") or {}
    out = {}
    for k in ("accepted_coverage", "covered_fidelity"):
        value = routes.get(k)
        out[k] = Cell(value if value is not None else NE, label, f"{prefix}.acdc_test_routes.{k}")
    return out


def table09(cx: Context) -> Table:
    t = Table(*TABLES[8], columns=["axis", "variant", "partition", "n", "n_clusters", "rho", "ci_lower", "ci_upper", "value", "n_eligible_features",
                                   "induced_rules", "eligible_rules", "mean_cut_recurrence", "accepted_coverage", "covered_fidelity", "p_value", "q_value",
                                   "status", "note"],
              notes=["Secondary ablations (Section 13.3): no post-test best-configuration claim; BH q within F2, F4 and F5 only."])
    variants, v_label = cx.src.json("metrics/ablations/transition_variants.json")
    if variants is None:
        t.add("transition", status="NOT_RECORDED", note=f"{v_label} (run-ablations)")
    else:
        for part in ("acdc_dev_oof", "acdc_test"):
            for name in VARIANT_AXES:
                entry = _get(variants, f"{part}.{name}")
                if not isinstance(entry, dict):
                    continue
                key = f"{part}.{name}"
                t.add("transition", axis=VARIANT_AXES[name], variant=name, partition=part, n=Cell(entry.get("n_cases"), v_label, f"{key}.n_cases"),
                      n_clusters=Cell(entry.get("n_clusters"), v_label, f"{key}.n_clusters"), rho=Cell(entry.get("rho"), v_label, f"{key}.rho"),
                      ci_lower=Cell(entry.get("ci_lower"), v_label, f"{key}.ci_lower"), ci_upper=Cell(entry.get("ci_upper"), v_label, f"{key}.ci_upper"),
                      status=Cell(entry.get("status"), v_label, f"{key}.status"))
        for i, entry in enumerate(variants.get("global_lambda") or []):
            t.add("lambda_policy", axis="lambda policy: one global lambda", variant=f"global ({entry.get('scope')})",
                  value=Cell(entry.get("selected_lambda"), v_label, f"global_lambda[{i}].selected_lambda"),
                  status=Cell("LAMBDA_AT_GRID_EDGE" if entry.get("lambda_at_grid_edge") else "INTERIOR", v_label, f"global_lambda[{i}].lambda_at_grid_edge"))
    lam, l_label = cx.src.json("transitions/lambda_selection.json")
    if isinstance(lam, dict):
        flags = lam.get("edge_flags") or {}
        t.add("lambda_policy", axis="lambda policy: per-target inner CV (primary)", variant="per target",
              value=Cell(int(sum(bool(v) for v in flags.values())), l_label, "count(edge_flags == true)"),
              n=Cell(len(flags), l_label, "count(edge_flags)"), note="value = targets whose final lambda lies at the grid edge")
    lin, lin_label = cx.src.json("metrics/ablations/linearity_criterion.json")
    if isinstance(lin, dict):
        t.add("lambda_policy", axis="linearity criterion (Section 9.4)", variant=str(lin.get("comparator")),
              value=Cell(lin.get("folds_meeting"), lin_label, "folds_meeting"), n=Cell(lin.get("folds_required"), lin_label, "folds_required"),
              status=Cell("NONLINEAR_DECLARED_BETTER" if lin.get("nonlinear_declared_better") else "AFFINE_RETAINED", lin_label, "nonlinear_declared_better"),
              note="value = folds meeting the criterion; n = folds required; the executed audit stays affine")
    f4, f4_label = _tests_by_variant(cx, "F4")
    disc, d_label = cx.src.json("metrics/ablations/discretizers.json")
    if disc is None:
        t.add("discretizers", status="NOT_RECORDED", note=f"{d_label} (run-ablations)")
    for name, entry in sorted(((disc or {}).get("profiles") or {}).items()):
        _discretizer_row(t, entry, d_label, f"profiles.{name}", "discretizer", name, f4, f4_label, f"discretizer:{name}")
    wedd, w_label = cx.src.json("metrics/ablations/wedd_settings.json")
    for name, entry in sorted(((wedd or {}).get("variants") or {}).items()):
        _discretizer_row(t, entry, w_label, f"variants.{name}", "WEDD settings (one factor at a time)", name, f4, f4_label, f"wedd_settings:{name}")
    f5, f5_label = _tests_by_variant(cx, "F5")
    policy, p_label = cx.src.json("metrics/ablations/rule_policy.json")
    for name, entry in sorted(((policy or {}).get("variants") or {}).items()):
        cells = _route_cells(entry, p_label, f"variants.{name}")
        cells.update(_test_cells(f5, f5_label, f"rule_policy:{name}"))
        t.add("rule_policy", axis="rule policy", variant=name, induced_rules=Cell(entry.get("induced_rules"), p_label, f"variants.{name}.induced_rules"),
              eligible_rules=Cell(entry.get("eligible_rules"), p_label, f"variants.{name}.eligible_rules"),
              value=Cell(entry.get("beta"), p_label, f"variants.{name}.beta"), n=Cell(entry.get("max_length"), p_label, f"variants.{name}.max_length"),
              note="value = beta; n = maximum rule length", **cells)
    gates, g_label = cx.src.json("metrics/ablations/gate_thresholds.json")
    for name, entry in sorted(((gates or {}).get("thresholds") or {}).items()):
        cells = _route_cells(entry, g_label, f"thresholds.{name}")
        cells.update(_test_cells(f5, f5_label, f"gate_lower_bound:{name}"))
        t.add("rule_policy", axis="gate Wilson lower bound", variant=name, value=Cell(_num(name), g_label, f"thresholds.{name} (key)"),
              eligible_rules=Cell(entry.get("eligible_rules"), g_label, f"thresholds.{name}.eligible_rules"), **cells)
    return t


def _test_cells(tests: pd.DataFrame | None, label: str, variant: str) -> dict[str, Cell]:
    if tests is None or variant not in tests.index:
        return {}
    row = tests.loc[variant]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return {"p_value": Cell(row.get("p_value"), label, f"variant == {variant!r}: p_value"), "q_value": Cell(row.get("q_value"), label, f"variant == {variant!r}: q_value")}


def _discretizer_row(t: Table, entry: dict, label: str, key: str, axis: str, name: str, tests: pd.DataFrame | None, t_label: str, variant: str) -> None:
    cells = _route_cells(entry, label, key)
    cells.update(_test_cells(tests, t_label, variant))
    t.add("discretizers", axis=axis, variant=name, n_eligible_features=Cell(len(entry.get("eligible_features") or []), label, f"{key}.eligible_features (count)"),
          induced_rules=Cell(entry.get("induced_rules"), label, f"{key}.induced_rules"), eligible_rules=Cell(entry.get("eligible_rules"), label, f"{key}.eligible_rules"),
          mean_cut_recurrence=Cell(entry.get("mean_cut_recurrence"), label, f"{key}.mean_cut_recurrence"),
          note=Cell(entry.get("source"), label, f"{key}.source"), **cells)


# ----------------------------------------------------------------------------- Table 10


def _ne_nodes(payload: Any, path: str = "") -> Iterable[tuple[str, str | None, str]]:
    if isinstance(payload, dict):
        status = payload.get("status")
        if isinstance(status, str) and status in {"NE", "NOT_EVALUATED"}:
            yield path or "<root>", payload.get("reason"), status
        for k, v in payload.items():
            yield from _ne_nodes(v, f"{path}.{k}" if path else str(k))
    elif isinstance(payload, list):
        for i, v in enumerate(payload):
            yield from _ne_nodes(v, f"{path}[{i}]")


def table10(cx: Context, metric_files: list[Path], profile: str) -> Table:
    t = Table(*TABLES[9], columns=["block", "rulebook", "item", "artifact", "key", "n", "value", "oof_mae", "flag", "reason"],
              notes=["Every NE of the recorded metrics (JSON nodes individually; tables aggregated by reason).",
                     "threshold_within_error (Section 13.4): a WEDD cut within the feature's OOF MAE of a neighbouring cut; derived here from the "
                     "recorded cuts and the recorded OOF MAE of the primary transition."])
    eligibility, e_label = cx.src.json("rules/semantic_eligibility.json")
    for rb, spec in sorted(((eligibility or {}).get("rulebooks") or {}).items()):
        for feature, reason in sorted((spec.get("ineligible") or {}).items()):
            t.add("ineligible_features", block="ineligible feature", rulebook=rb, item=feature, reason=Cell(reason, e_label, f"rulebooks.{rb}.ineligible.{feature}"))
        for cls, entry in sorted((spec.get("consequent_classes") or {}).items()):
            if isinstance(entry, dict) and not entry.get("eligible", True):
                t.add("ineligible_classes", block="ineligible class", rulebook=rb, item=cls, n=Cell(entry.get("fit_patients"), e_label, f"rulebooks.{rb}.consequent_classes.{cls}.fit_patients"),
                      reason=Cell(entry.get("reason_code"), e_label, f"rulebooks.{rb}.consequent_classes.{cls}.reason_code"))
    for arm, prefix in (("primary", "rules"), ("refit_arm", "rules/refit_arm")):
        if arm == "refit_arm" and not cx.include_mms:
            continue
        for rb, _, _ in _rulebooks(cx):
            empty, x_label = cx.src.json(f"{prefix}/{rb}/EMPTY.json")
            if isinstance(empty, dict):
                t.add("empty_rulebooks", block=f"empty rulebook ({arm})", rulebook=rb, item="EMPTY", artifact=x_label,
                      reason=Cell(empty.get("reason_code"), x_label, "reason_code"), value=Cell(empty.get("failing_gates"), x_label, "failing_gates"))
    for path in metric_files:
        rel = path.relative_to(cx.run.path).as_posix()
        if path.suffix == ".json":
            payload, label = cx.src.json(rel)
            for key, reason, status in _ne_nodes(payload):
                t.add("ne_metrics", block=status, artifact=label, key=Cell(key, label, key), reason=Cell(reason, label, f"{key}.reason"))
            if isinstance(payload, dict) and payload.get("n_accepted") == 0 and "covered_fidelity" in payload:
                t.add("ne_metrics", block=NE, artifact=label, key=Cell("covered_fidelity, covered_reference_accuracy, audit_tax", label, "n_accepted == 0"),
                      reason="NO_ACCEPTED_OUTPUT")
        elif path.suffix == ".parquet":
            frame, label = cx.src.parquet(rel)
            for col in ("status", "inference_status"):
                if frame is None or col not in frame.columns:
                    continue
                reason_col = "reason" if col == "status" else "inference_reason"
                ne = frame[frame[col].astype(str) == NE]
                if ne.empty:
                    continue
                reasons = ne[reason_col].astype(object).where(ne[reason_col].notna(), None) if reason_col in ne.columns else pd.Series([None] * len(ne))
                for reason, count in sorted(reasons.map(lambda v: "" if v is None else str(v)).value_counts().items()):
                    t.add("ne_metrics", block=f"{NE} ({col})", artifact=label, key=Cell(f"{col} == 'NE' and {reason_col} == {reason!r}", label, f"{col}, {reason_col}"),
                          n=Cell(int(count), label, f"count({col} == 'NE' and {reason_col} == {reason!r})"), reason=reason or None)
    lam, l_label = cx.src.json("transitions/lambda_selection.json")
    per_fold, f_label = cx.src.parquet("transitions/lambda_selection.parquet")
    if isinstance(lam, dict):
        folds = None
        if per_fold is not None and "lambda_at_grid_edge" in per_fold.columns:
            outer = per_fold[per_fold.get("scope", pd.Series("outer_fold", index=per_fold.index)).astype(str) == "outer_fold"]
            folds = outer.groupby("target")["lambda_at_grid_edge"].agg(lambda s: int(s.astype(bool).sum()))
            n_folds = outer.groupby("target")["lambda_at_grid_edge"].size()
        for target in sorted(lam.get("edge_flags") or {}):
            final = bool(lam["edge_flags"][target])
            at_edge = int(folds.get(target, 0)) if folds is not None else 0
            if not final and not at_edge:
                continue
            cells = {"block": "grid-edge lambda", "item": target, "value": Cell(_get(lam, f"lambdas.{target}"), l_label, f"lambdas.{target}"),
                     "flag": Cell(final, l_label, f"edge_flags.{target}"), "reason": "LAMBDA_AT_GRID_EDGE" if final else "LAMBDA_AT_GRID_EDGE (outer folds only)"}
            if folds is not None:
                cells["n"] = Cell(at_edge, f_label, f"count(scope == 'outer_fold' and target == {target!r} and lambda_at_grid_edge)")
                cells["key"] = Cell(f"outer folds at edge of {int(n_folds.get(target, 0))}", f_label, f"count(scope == 'outer_fold' and target == {target!r})")
            t.add("grid_edge_lambda", **cells)
    oof, o_label = cx.src.parquet("metrics/ablations/transition_variants_per_target.parquet")
    mae = None
    if oof is not None:
        primary = oof[(oof["variant"] == "ridge_typed_formal57") & (oof["partition"] == "acdc_dev_oof")].set_index("target")["mae"]
        mae = primary
    for rb, _, substrate in _rulebooks(cx):
        if substrate == "measured_B":
            continue
        cuts, c_label = cx.src.json(f"thresholds/{rb}_{profile}.json")
        if not isinstance(cuts, dict):
            continue
        for feature, entry in sorted((cuts.get("features") or {}).items()):
            values = sorted(float(c) for c in entry.get("cuts") or [])
            cells: dict[str, Any] = {"block": "threshold within error", "rulebook": rb, "item": feature,
                                     "n": Cell(len(values), c_label, f"features.{feature}.cuts (count)")}
            m = None if mae is None or feature not in mae.index else _num(mae.loc[feature])
            if m is not None:
                cells["oof_mae"] = Cell(m, o_label, f"variant == 'ridge_typed_formal57' and partition == 'acdc_dev_oof' and target == {feature!r}: mae")
            if len(values) < 2:
                cells.update(flag=False, reason="NOT_APPLICABLE (fewer than two cuts)")
            elif m is None:
                cells.update(flag=NE, reason="OOF_MAE_NOT_RECORDED")
            else:
                gap = float(np.min(np.diff(values)))
                cells.update(value=Cell(gap, c_label, f"min adjacent gap of features.{feature}.cuts"), flag=bool(gap < m),
                             reason="threshold_within_error" if gap < m else None)
            t.add("thresholds_within_error", **cells)
    return t


# ----------------------------------------------------------------------------- protocol v6.0.1 tables (lane L2)


EXTERNAL_PARTITIONS_V601 = ("mms_external", "mms2_external")
V601_PREDICTOR_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "macro_ovr_auroc", "ece", "ece_equal_width", "brier", "log_loss")
V601_MODELS = (("audited_predictor", "formal A (57 descriptors)"), ("measured_b_comparator", "measured B (registry primary family)"))
V601_EVIDENCE = {"acdc_dev_oof": "development (cross-fitted OOF)", "acdc_test": "registered after unblinding",
                 "mms_external": "registered after unblinding", "mms2_external": "prospective"}


def _included(cx: Context, partition: str) -> bool:
    return partition in ("acdc_dev_oof", "acdc_test") or (partition == "mms_external" and cx.include_mms) or (
        partition == "mms2_external" and getattr(cx, "include_mms2", False))


def _small_clusters(cx: Context, entry: dict | None, n_key: str = "n_cases") -> bool:
    """A stratum below the reporting threshold, counted in clusters when recorded (else cases)."""

    if not isinstance(entry, dict):
        return False
    n = entry.get("n_clusters", entry.get(n_key))
    return cx.below(n)


def _recorded_decision(cx: Context, hypothesis: str) -> Cell:
    """The recorded fixed-sequence status of ``hypothesis``: ``metrics/fixed_sequence.json`` first, then any other
    declared top-level metrics JSON with a result entry for it (``hypothesis_id`` and ``tested``); never derived here."""

    cell, item = _sequence(cx, hypothesis)
    if item is not None:
        return cell
    base = cx.run.path / "metrics"
    for path in sorted(q for q in base.glob("*.json") if q.resolve() in cx.src.allowed and q.name != "fixed_sequence.json"):
        payload, label = cx.src.json(path)
        results = payload.get("results") if isinstance(payload, dict) else None
        listed = isinstance(results, list)
        for i, entry in enumerate(results if listed else [payload]):
            if isinstance(entry, dict) and entry.get("hypothesis_id") == hypothesis and "tested" in entry:
                return Cell(entry.get("status"), label, f"results[{i}].status (hypothesis_id {hypothesis})" if listed else "status")
    return Cell("NOT_RECORDED", None, None)


def v601_table03(cx: Context) -> Table:
    from .stages import hypothesis_id

    primary = cx.study.bootstrap["primary_test"]
    criterion = str(primary.get("success", f"95 % lower bound of rho > {primary.get('smallest_effect_of_interest')}"))
    t = Table(*TABLES_V601[0], columns=["hypothesis", "partition", "evidence", "family", "estimand", "n", "n_clusters", "rho", "ci_lower", "ci_upper",
                                        "criterion", "decision", "status", "note"],
              notes=[f"Success criterion (configs/bootstrap.yaml primary_test.success): {criterion}; percentile 95 % paired cluster-bootstrap intervals.",
                     "H-primary-1 and H-primary-2 form the recorded v6.0.0 fixed sequence; H-primary-2-never-analysed is its sensitivity (no "
                     "gatekeeping); H-primary-3 is its own one-test family on the prospective mms2_external (protocol-v6.0.1).",
                     "H4 (Decision D36): H4a (rho over the EF targets) and H4b (delta-rho of the S3 H-a harmonization against the frozen bundle; "
                     "the rho column holds the difference) form a second confirmatory family on mms2_external with Bonferroni within the "
                     "family (two-sided 97.5 % intervals); on acdc_test and mms_external they are registered after unblinding (DESCRIPTIVE).",
                     "Decisions are read from recorded results; a decision that is not recorded is NOT_RECORDED, never derived here."])
    h3 = hypothesis_id(cx.study, "mms2_external") or "H-primary-3"
    rows = (("H-primary-1", "acdc_test", "metrics/primary_rho_acdc_test.json", "fixed sequence (v6.0.0)", "evaluate:acdc_test"),
            ("H-primary-2", "mms_external", "metrics/primary_rho_mms_external.json", "fixed sequence (v6.0.0)", "evaluate:mms_external"),
            ("H-primary-2-never-analysed", "mms_external_never_analysed", "metrics/primary_rho_mms_external_never_analysed.json",
             "sensitivity of H-primary-2 (no gatekeeping)", "evaluate:mms_external"),
            (h3, "mms2_external", "metrics/primary_rho_mms2_external.json", "own one-test family (protocol-v6.0.1)", "evaluate:mms2_external"))
    evidence = {"acdc_test": "confirmatory (v6.0.0; recorded)", "mms_external": "confirmatory (v6.0.0; recorded)",
                "mms_external_never_analysed": "sensitivity (v6.0.0; recorded)", "mms2_external": "confirmatory; prospective (protocol-v6.0.1)"}
    for hid, part, rel, family, stage in rows:
        cells: dict[str, Any] = {"hypothesis": hid, "partition": part, "evidence": evidence[part], "family": family, "estimand": "rho", "criterion": criterion}
        if not _included(cx, "mms_external" if part.startswith("mms_external") else part):
            t.add("hypotheses", **cells, status="NOT_RECORDED", note=f"{stage} not recorded")
            continue
        entry, label = cx.src.json(rel)
        if entry is None:
            t.add("hypotheses", **cells, status="NOT_RECORDED", note=f"{label} absent")
            continue
        cells.update(n=Cell(entry.get("n_cases"), label, "n_cases"), n_clusters=Cell(entry.get("n_clusters"), label, "n_clusters"),
                     rho=Cell(entry.get("rho"), label, "rho"), ci_lower=Cell(entry.get("ci_lower"), label, "ci_lower"),
                     ci_upper=Cell(entry.get("ci_upper"), label, "ci_upper"), status=Cell(entry.get("status"), label, "status"),
                     decision=_recorded_decision(cx, hid))
        if entry.get("reason"):
            cells["note"] = Cell(entry.get("reason"), label, "reason")
        t.add("hypotheses", **cells)
    _h4_rows(cx, t)
    return t


def _h4_rows(cx: Context, t: Table) -> None:
    """H4a and H4b per partition from the once-only records ``metrics/confirmatory_h4/h4_<partition>.json`` (Decision D36)."""

    cfg = ((cx.study.cohorts.get("mms2") or {}).get("endpoints") or {}).get("h4") or {}
    hyps = [dict(h) for h in (cfg.get("hypotheses") or {}).values()]
    if not hyps:
        return
    confirmatory = str(cfg.get("partition", "mms2_external"))
    partitions = [*[str(p) for p in cfg.get("descriptive_partitions", [])], confirmatory]
    mult = cfg.get("multiplicity") or {}
    level = 100.0 * (1.0 - float(mult.get("family_alpha", 0.05)) / int(mult.get("n_tests", len(hyps))))
    for part in partitions:
        rel = f"metrics/{H4_METRICS_DIR}/h4_{part}.json"
        payload, label = cx.src.json(rel) if _included(cx, part) else (None, rel)
        results = {str(r.get("hypothesis_id")): (i, r) for i, r in enumerate((payload or {}).get("results") or []) if isinstance(r, dict)}
        for hyp in hyps:
            hid = str(hyp["id"])
            family = f"H4, Bonferroni within the family ({int(mult.get('n_tests', len(hyps)))} tests; protocol-v6.0.1)"
            cells: dict[str, Any] = {"hypothesis": hid, "partition": part, "family": family, "estimand": str(hyp.get("estimand")),
                                     "evidence": "confirmatory; prospective (protocol-v6.0.1)" if part == confirmatory else "registered after unblinding",
                                     "criterion": f"two-sided {level:g} % lower bound of {hyp.get('estimand')} > {float(hyp['threshold']):g}"}
            if not _included(cx, part):
                t.add("hypotheses", **cells, status="NOT_RECORDED", note=f"evaluate:{part} not recorded")
                continue
            if payload is None or hid not in results:
                t.add("hypotheses", **cells, status="NOT_RECORDED", note=f"{label} absent" if payload is None else f"{hid} absent from {label}")
                continue
            i, r = results[hid]
            key = f"results[{i}]"
            cells.update(n=Cell(r.get("n_cases"), label, f"{key}.n_cases"), n_clusters=Cell(r.get("n_clusters"), label, f"{key}.n_clusters"),
                         rho=Cell(r.get("estimate"), label, f"{key}.estimate"), ci_lower=Cell(r.get("ci_lower"), label, f"{key}.ci_lower"),
                         ci_upper=Cell(r.get("ci_upper"), label, f"{key}.ci_upper"), decision=Cell(r.get("status"), label, f"{key}.status"),
                         status=Cell("NE" if r.get("status") == "NE" else "EVALUATED", label, f"{key}.status"))
            if r.get("reason"):
                cells["note"] = Cell(r.get("reason"), label, f"{key}.reason")
            t.add("hypotheses", **cells)


def _metric_cells(cx: Context, entry: dict | None, label: str, key: str, prefix: str = "", suppress: bool = False) -> dict[str, Any]:
    """estimate / ci cells of one recorded metric entry (``NE`` with its reason; withheld when ``suppress``)."""

    names = {"estimate": f"{prefix}estimate" if prefix else "estimate", "ci_lower": f"{prefix}ci_lower", "ci_upper": f"{prefix}ci_upper"}
    if not isinstance(entry, dict):
        return {names["estimate"]: "NOT_RECORDED"}
    if suppress:
        return {names["estimate"]: cx.small, names["ci_lower"]: cx.small, names["ci_upper"]: cx.small}
    if entry.get("estimate") is None:
        return {names["estimate"]: Cell(NE, label, f"{key}.status ({entry.get('reason')})")}
    return {names["estimate"]: Cell(entry.get("estimate"), label, f"{key}.estimate"), names["ci_lower"]: Cell(entry.get("ci_lower"), label, f"{key}.ci_lower"),
            names["ci_upper"]: Cell(entry.get("ci_upper"), label, f"{key}.ci_upper")}


def v601_table05(cx: Context) -> Table:
    t = Table(*TABLES_V601[1], columns=["partition", "evidence", "stratum_variable", "stratum", "n", "n_clusters", "rho", "ci_lower", "ci_upper",
                                        "n_predictor", "accuracy", "accuracy_ci_lower", "accuracy_ci_upper", "macro_f1", "comparator_accuracy",
                                        "domain_auroc", "domain_auroc_ci_lower", "domain_auroc_ci_upper", "status", "note"],
              notes=[f"Strata with fewer than {cx.min_cases} clusters show their n and {cx.small} only (STRATUM_BELOW_MIN_CASES).",
                     "rho: frozen bundle per stratum (evaluate); predictor columns: crosswalk same-definition cases of the stratum (summarize-predictor, "
                     "v6.0.1 extension; comparator = measured-B comparator); domain_auroc: label-free shift audit per vendor (audit-shift).",
                     "mms_external results are registered after unblinding; mms2_external results are prospective."])
    for part in EXTERNAL_PARTITIONS_V601:
        base = {"partition": part, "evidence": V601_EVIDENCE[part]}
        if not _included(cx, part):
            t.add("strata", **base, status="NOT_RECORDED", note=f"evaluate:{part} not recorded")
            continue
        strata, s_label = cx.src.json(f"metrics/strata_rho_{part}.json")
        pred, p_label = cx.src.json(f"metrics/{EXTENDED_METRICS_DIR}/{part}.json")
        shift, a_label = cx.src.json(f"metrics/shift_audit_{part}.json")
        if strata is None and pred is None:
            t.add("strata", **base, status="NOT_RECORDED", note=f"{s_label} and {p_label} absent")
            continue
        groups = (strata or {}).get("strata") or {}
        pstrata = (pred or {}).get("strata") or {}
        audit = {}
        for i, r in enumerate((shift or {}).get("results") or []):
            if r.get("comparison") == "vendor":
                audit[str(r.get("group"))] = (i, r)
        for var in ("vendor", "centre"):
            pvals = pstrata.get(var) if isinstance(pstrata.get(var), dict) else {}
            values = sorted(set((groups.get(var) or {}).keys()) | {k for k, v in pvals.items() if isinstance(v, dict)})
            if not values:
                t.add("strata", **base, stratum_variable=var, status="NOT_RECORDED", note=f"no {var} strata recorded")
                continue
            for value in values:
                cells: dict[str, Any] = {**base, "stratum_variable": var, "stratum": value}
                rho = (groups.get(var) or {}).get(value)
                key = f"strata[{var!r}][{value!r}]"
                if isinstance(rho, dict):
                    small = _small_clusters(cx, rho)
                    cells.update(n=Cell(rho.get("n_cases"), s_label, f"{key}.n_cases"), n_clusters=Cell(rho.get("n_clusters"), s_label, f"{key}.n_clusters"))
                    if small:
                        cells.update(rho=cx.small, ci_lower=cx.small, ci_upper=cx.small, status=cx.small, note="STRATUM_BELOW_MIN_CASES: point estimate withheld")
                    else:
                        cells.update(rho=Cell(rho.get("rho"), s_label, f"{key}.rho"), ci_lower=Cell(rho.get("ci_lower"), s_label, f"{key}.ci_lower"),
                                     ci_upper=Cell(rho.get("ci_upper"), s_label, f"{key}.ci_upper"), status=Cell(rho.get("status"), s_label, f"{key}.status"))
                entry = (pstrata.get(var) or {}).get(value)
                if isinstance(entry, dict):
                    audited = entry.get("audited_predictor") or {}
                    counts = audited.get("counts") or {}
                    pkey = f"strata.{var}.{value}.audited_predictor"
                    cells["n_predictor"] = Cell(counts.get("n_evaluable"), p_label, f"{pkey}.counts.n_evaluable")
                    small_p = cx.below(counts.get("n_clusters"))
                    acc = (audited.get("metrics") or {}).get("accuracy")
                    cells.update(_metric_cells(cx, acc, p_label, f"{pkey}.metrics.accuracy", prefix="accuracy_", suppress=small_p))
                    cells["accuracy"] = cells.pop("accuracy_estimate")
                    f1 = (audited.get("metrics") or {}).get("macro_f1") or {}
                    cells["macro_f1"] = cx.small if small_p else Cell(f1.get("estimate"), p_label, f"{pkey}.metrics.macro_f1.estimate")
                    comp = (entry.get("measured_b_comparator") or {})
                    c_acc = (comp.get("metrics") or {}).get("accuracy") or {}
                    small_c = cx.below((comp.get("counts") or {}).get("n_clusters"))
                    if comp:
                        cells["comparator_accuracy"] = cx.small if small_c else Cell(c_acc.get("estimate"), p_label,
                                                                                       f"strata.{var}.{value}.measured_b_comparator.metrics.accuracy.estimate")
                if var == "vendor" and value in audit:
                    i, r = audit[value]
                    akey = f"results[{i}] (comparison vendor, group {value})"
                    if cx.below(r.get("n_comparison_clusters", r.get("n_comparison"))):
                        cells.update(domain_auroc=cx.small, domain_auroc_ci_lower=cx.small, domain_auroc_ci_upper=cx.small)
                    else:
                        cells.update(domain_auroc=Cell(r.get("auroc"), a_label, f"{akey}.auroc"),
                                     domain_auroc_ci_lower=Cell(r.get("ci_lower"), a_label, f"{akey}.ci_lower"),
                                     domain_auroc_ci_upper=Cell(r.get("ci_upper"), a_label, f"{akey}.ci_upper"))
                t.add("strata", **cells)
    return t


def v601_table06(cx: Context) -> Table:
    t = Table(*TABLES_V601[2], columns=["model", "input", "partition", "evidence", "stratum_variable", "stratum", "metric", "n", "n_clusters", "estimate",
                                        "ci_lower", "ci_upper", "status", "note"],
              notes=["Percentile 95 % intervals of the fixed-model cluster bootstrap; ece: equal-mass bins as frozen (SPECIFICATION.md 13.2); "
                     "ece_equal_width: 10 equal-width bins; brier: multiclass Brier score (protocol v6.0.1 A2).",
                     "acdc_dev: out-of-fold probabilities of the stored outer-fold predictors (A2). measured_b_comparator: the audited-predictor "
                     "settings on measured B of the registry primary family (A4), frozen with the bundle, evaluated once per partition.",
                     "External partitions: crosswalk same-definition cases only.",
                     f"Strata with fewer than {cx.min_cases} clusters show their n and {cx.small} only."])
    for part in ("acdc_dev_oof", "acdc_test", *EXTERNAL_PARTITIONS_V601):
        if not _included(cx, part):
            t.add("overall", partition=part, evidence=V601_EVIDENCE[part], status="NOT_RECORDED", note=f"evaluate:{part} not recorded")
            continue
        payload, label = cx.src.json(f"metrics/{EXTENDED_METRICS_DIR}/{part}.json")
        if payload is None:
            t.add("overall", partition=part, evidence=V601_EVIDENCE[part], status="NOT_RECORDED", note=f"{label} (run summarize-predictor)")
            continue
        for model_id, source in V601_MODELS:
            entry = (payload.get("models") or {}).get(model_id) or {}
            base = {"model": model_id, "input": source, "partition": part, "evidence": V601_EVIDENCE[part]}
            if entry.get("status") != "EVALUATED":
                t.add("overall", **base, status=Cell(entry.get("status", "NOT_RECORDED"), label, f"models.{model_id}.status"),
                      note=Cell(entry.get("reason"), label, f"models.{model_id}.reason"))
                continue
            counts = entry.get("counts") or {}
            for m in V601_PREDICTOR_METRICS:
                me = (entry.get("metrics") or {}).get(m) or {}
                key = f"models.{model_id}.metrics.{m}"
                cells = {**base, "metric": m, "n": Cell(counts.get("n_evaluable"), label, f"models.{model_id}.counts.n_evaluable"),
                         "n_clusters": Cell(counts.get("n_clusters"), label, f"models.{model_id}.counts.n_clusters"),
                         "status": Cell(me.get("status"), label, f"{key}.status")}
                cells.update(_metric_cells(cx, me, label, key))
                if me.get("status") == NE and me.get("reason"):
                    cells["note"] = Cell(me.get("reason"), label, f"{key}.reason")
                t.add("overall", **cells)
        for var, values in sorted(((payload.get("strata") or {}).items())):
            if not isinstance(values, dict) or "status" in values:
                continue
            for value, entry in sorted(values.items()):
                if not isinstance(entry, dict):
                    continue
                for model_id, source in V601_MODELS:
                    model = entry.get(model_id)
                    if not isinstance(model, dict) or not model.get("metrics"):
                        continue
                    counts = model.get("counts") or {}
                    small = cx.below(counts.get("n_clusters"))
                    for m in ("accuracy", "macro_f1"):
                        key = f"strata.{var}.{value}.{model_id}.metrics.{m}"
                        me = model["metrics"].get(m)
                        cells = {"model": model_id, "input": source, "partition": part, "evidence": V601_EVIDENCE[part], "stratum_variable": var,
                                 "stratum": value, "metric": m, "n": Cell(counts.get("n_evaluable"), label, f"strata.{var}.{value}.{model_id}.counts.n_evaluable"),
                                 "n_clusters": Cell(counts.get("n_clusters"), label, f"strata.{var}.{value}.{model_id}.counts.n_clusters")}
                        cells.update(_metric_cells(cx, me, label, key, suppress=small))
                        cells["status"] = cx.small if small else Cell((me or {}).get("status"), label, f"{key}.status")
                        t.add("strata", **cells)
    return t


BA_COLUMNS = ["partition", "stratum_variable", "stratum", "target", "unit", "n", "n_clusters", "bias", "bias_ci_lower", "bias_ci_upper", "loa_lower",
              "loa_lower_ci_lower", "loa_lower_ci_upper", "loa_upper", "loa_upper_ci_lower", "loa_upper_ci_upper", "status", "note"]


def _ba_cells(cx: Context, e: dict, label: str, key: str, suppress: bool) -> dict[str, Any]:
    cells: dict[str, Any] = {"n": Cell(e.get("n"), label, f"{key}.n"), "n_clusters": Cell(e.get("n_clusters"), label, f"{key}.n_clusters")}
    if suppress:
        cells.update({c: cx.small for c in ("bias", "bias_ci_lower", "bias_ci_upper", "loa_lower", "loa_lower_ci_lower", "loa_lower_ci_upper",
                                            "loa_upper", "loa_upper_ci_lower", "loa_upper_ci_upper")})
        cells.update(status=cx.small, note="STRATUM_BELOW_MIN_CASES: point estimates withheld")
        return cells
    ci = e.get("ci") or {}
    for name in ("bias", "loa_lower", "loa_upper"):
        cells[name] = Cell(e.get(name), label, f"{key}.{name}")
        lo, hi = ci.get(name) or [None, None]
        cells[f"{name}_ci_lower"] = Cell(lo, label, f"{key}.ci.{name}[0]")
        cells[f"{name}_ci_upper"] = Cell(hi, label, f"{key}.ci.{name}[1]")
    cells["status"] = Cell(e.get("status"), label, f"{key}.status")
    if e.get("reason"):
        cells["note"] = Cell(e.get("reason"), label, f"{key}.reason")
    return cells


def v601_table_s1(cx: Context) -> Table:
    t = Table(*TABLES_V601[3], columns=BA_COLUMNS,
              notes=["Difference = B-hat minus measured B in physical units; limits of agreement = bias +- 1.96 SD (ddof 1); percentile 95 % "
                     "cluster-bootstrap intervals (one draw per replicate shared by all targets).",
                     "acdc_dev: nested out-of-fold B-hat of fit-transition; test and external partitions: the recorded raw B-hat of evaluate.",
                     f"Strata with fewer than {cx.min_cases} clusters show their n and {cx.small} only."])
    for part in ("acdc_dev_oof", "acdc_test", *EXTERNAL_PARTITIONS_V601):
        if not _included(cx, part):
            t.add("all", partition=part, status="NOT_RECORDED", note=f"evaluate:{part} not recorded")
            continue
        payload, label = cx.src.json(f"metrics/{EXTENDED_METRICS_DIR}/{part}.json")
        ba = (payload or {}).get("bland_altman")
        if not isinstance(ba, dict) or "targets" not in ba:
            t.add("all", partition=part, status="NOT_RECORDED" if payload is None else Cell((ba or {}).get("status", NE), label, "bland_altman.status"),
                  note=f"{label} absent (run summarize-predictor)" if payload is None else Cell((ba or {}).get("reason"), label, "bland_altman.reason"))
            continue
        for target, e in ba["targets"].items():
            t.add("all", partition=part, stratum_variable="all", stratum="all", target=target, unit=cx.unit(target),
                  **_ba_cells(cx, e, label, f"bland_altman.targets.{target}", suppress=False))
        for var, values in sorted((ba.get("strata") or {}).items()):
            for value, targets in sorted(values.items()):
                for target, e in targets.items():
                    t.add("strata", partition=part, stratum_variable=var, stratum=value, target=target, unit=cx.unit(target),
                          **_ba_cells(cx, e, label, f"bland_altman.strata.{var}.{value}.{target}", suppress=_small_clusters(cx, e, "n")))
    return t


# ----------------------------------------------------------------------------- orchestration


def source_paths(run: RunRoot, study: Any, *, include_mms: bool, spec_path: Path, registry_path: Path, include_mms2: bool = False) -> list[Path]:
    """Every artifact the tables may read (the stage's declared inputs); M&Ms artifacts only when the external
    evaluation is recorded (M&Ms-2 artifacts only when ``evaluate:mms2_external`` is)."""

    p = run.path
    profile = str(study.wedd["primary_profile"])
    books = [str(v["name"]) for v in study.rule_policy["rulebooks"].values()]
    paths = [spec_path, registry_path, p / "contracts" / "cohort_ACDC.parquet", p / "contracts" / "split_manifest.parquet",
             p / "semantics" / "B_ACDC.parquet", p / "formal" / "A_ACDC.parquet", p / "rules" / "semantic_eligibility.json",
             p / "transitions" / "lambda_selection.json", p / "transitions" / "lambda_selection.parquet"]
    if include_mms:
        paths += [p / "contracts" / "cohort_MMS.parquet", p / "semantics" / "B_MMS.parquet", p / "formal" / "A_MMS.parquet"]
    for rb in books:
        paths += [p / "rules" / rb / "rulebook_or_empty.json", p / "rules" / rb / "EMPTY.json", p / "thresholds" / f"{rb}_{profile}.json"]
        if include_mms:
            paths += [p / "rules" / "refit_arm" / rb / "EMPTY.json"]
    paths += metric_paths(run, include_mms=include_mms)
    paths += extended_source_paths(run, include_mms=include_mms, include_mms2=include_mms2)
    return sorted({q for q in paths if q.is_file()}, key=lambda q: q.as_posix())


def metric_paths(run: RunRoot, *, include_mms: bool) -> list[Path]:
    base = run.path / "metrics"
    out = [q for q in sorted(base.rglob("*")) if q.is_file() and q.suffix in (".json", ".parquet") and not q.name.startswith(".")]
    out = [q for q in out if not {EXTENDED_METRICS_DIR, H4_METRICS_DIR} & set(q.relative_to(base).parts)]   # v6.0.1 records: the v6.0.1 tables only
    if not include_mms:
        out = [q for q in out if "mms" not in q.name and "refit_arm" not in q.relative_to(base).parts]
    return out


def extended_source_paths(run: RunRoot, *, include_mms: bool, include_mms2: bool) -> list[Path]:
    """Inputs of the v6.0.1 tables beyond :func:`metric_paths`: ``metrics/predictor_extended/`` and the M&Ms-2 metrics
    (lane L1 names: ``metrics/*mms2_external*``), each M&Ms file only when its evaluation is recorded."""

    base = run.path / "metrics"
    out = [q for q in sorted((base / EXTENDED_METRICS_DIR).glob("*")) if q.is_file() and q.suffix in (".json", ".parquet")]
    out += [q for q in sorted(base.glob("*mms2*")) if q.is_file() and q.suffix in (".json", ".parquet")]
    out += [q for q in sorted((base / H4_METRICS_DIR).glob("h4_*.json")) if q.is_file()]   # Decision D36 (gated by partition name below)

    def allowed(q: Path) -> bool:
        if "mms2" in q.name:
            return include_mms2
        return include_mms or "mms" not in q.name

    return [q for q in out if allowed(q) and not q.name.startswith(".")]


def case_identifiers(split: pd.DataFrame | None) -> set[str]:
    if split is None:
        return set()
    if isinstance(split, (list, tuple)):          # protocol-v6.0.1: several split manifests (v6.0.0 and M&Ms-2)
        return set().union(*(case_identifiers(s) for s in split))
    ids: set[str] = set()
    for col in ("case_id", "patient_id"):
        if col in split.columns:
            ids |= {str(v) for v in split[col].dropna().astype(str) if v}
    ids |= {str(v).split(":", 1)[-1] for v in split.get("sample_id", pd.Series(dtype=object)).dropna().astype(str)}
    return ids


def assert_no_case_identifiers(texts: dict[str, str], identifiers: set[str]) -> None:
    """Refuse to write any table or provenance text that contains a case identifier as a token (Section 14.5)."""

    plain = {i for i in identifiers if re.fullmatch(r"[A-Za-z0-9]+", i)}
    joined = [i for i in identifiers if i not in plain and i]   # protocol-v6.0.1: prefixed ids such as MMS2_001 (decision C)
    for name, text in texts.items():
        hits = sorted(set(re.findall(r"[A-Za-z0-9]+", text)) & plain)
        hits += [i for i in joined if re.search(rf"(?<![A-Za-z0-9_]){re.escape(i)}(?![A-Za-z0-9_])", text)]
        if hits:
            raise TableBuildError(f"{name} would contain {len(hits)} case identifier(s); nothing written (licence, SPECIFICATION.md 14.5)")


def _table_provenance(table: Table, csv: bytes, tex: bytes, sources: Sources) -> dict[str, Any]:
    columns: dict[str, list[dict[str, Any]]] = {}
    for col in table.columns:
        grouped: dict[str, list[str]] = {}
        for entry in table.provenance.get(col, []):
            grouped.setdefault(entry["path"], []).append(entry["key"])
        columns[col] = [{"path": path, "sha256": sources.hashes.get(path), "keys": keys} for path, keys in sorted(grouped.items())]
    used = sorted({e["path"] for entries in table.provenance.values() for e in entries})
    return {"title": table.title, "files": {"csv": f"{table.name}.csv", "tex": f"{table.name}.tex"},
            "sha256": {"csv": _sha256(csv), "tex": _sha256(tex)}, "rows": len(table.rows), "panels": table.panels,
            "sources": [{"path": path, "sha256": sources.hashes.get(path)} for path in sorted(set(used))],
            "columns": columns, "numeric_columns": sorted(c for c in table.columns if _numeric_column(table, c)), "notes": table.notes}


def build_tables(run: RunRoot, study: Any, *, include_mms: bool, out_dir: Path, spec_path: Path | None = None,
                 registry_path: Path | None = None, include_mms2: bool = False) -> dict[str, Any]:
    """Build Tables 1-10 and ``provenance.json`` (see the module docstring), and the protocol-v6.0.1 tables of
    :data:`TABLES_V601` with ``provenance_v6_0_1.json``; returns a summary."""

    spec_path = Path(spec_path) if spec_path is not None else PROJECT_ROOT / "SPECIFICATION.md"
    registry_path = Path(registry_path) if registry_path is not None else registry_file(study)
    run_tag = run_protocol_tag(run, study)                   # protocol-v6.0.1 decision F (protocol-v6.0.0 for v6.0.0 runs)
    matches = spec_matches_tag(spec_path, run_tag)
    if matches is False:
        raise TableBuildError(f"{spec_path} differs from {run_tag}:SPECIFICATION.md, the protocol run {run.run_id} was executed under; "
                              "Table 1 must come from that frozen protocol")
    allowed = source_paths(run, study, include_mms=include_mms, spec_path=spec_path, registry_path=registry_path, include_mms2=include_mms2)
    sources = Sources(run, allowed)
    cx = Context(run, study, sources, include_mms=include_mms, registry_path=registry_path, include_mms2=include_mms2)
    profile = str(study.wedd["primary_profile"])
    metric_files = [q for q in metric_paths(run, include_mms=include_mms) if q.resolve() in sources.allowed]
    tables = [table01(cx, spec_path), table02(cx), table03(cx), table04(cx), table05(cx), table06(cx), table07(cx), table08(cx),
              table09(cx), table10(cx, metric_files, profile)]
    tables_v601 = [v601_table03(cx), v601_table05(cx), v601_table06(cx), v601_table_s1(cx)]
    split, _ = sources.parquet("contracts/split_manifest.parquet")
    mms2_split = run.path / "contracts" / "split_manifest_MMS2.parquet"     # protocol-v6.0.1: M&Ms-2 ids are guarded too
    identifiers = case_identifiers([split, pd.read_parquet(mms2_split) if mms2_split.is_file() else None])
    rendered: dict[str, tuple[bytes, bytes]] = {}
    for table in tables + tables_v601:
        csv = table.frame().to_csv(index=False, lineterminator="\n").encode("utf-8")
        tex = render_latex(table, run.run_id).encode("utf-8")
        rendered[table.name] = (csv, tex)
    provenance = {
        "schema": "tmmri.manuscript_tables_provenance/1", "run_id": run.run_id, "protocol_tag": study.freeze_tag,
        "specification": {"path": sources.label(spec_path), "sha256": hash_path(spec_path), "matches_protocol_tag": matches},
        "min_cases_per_stratum": cx.min_cases, "small_stratum_rule": f"n shown; estimates {cx.small}", "descriptive_strata": cx.descriptive,
        "licence": "aggregates only; no case identifiers; no per-case M&Ms content (SPECIFICATION.md 14.5)", "manuscript_directory_written": False,
        "tables": {},
    }
    for table in tables:
        provenance["tables"][table.name] = _table_provenance(table, *rendered[table.name], sources)
    provenance_v601 = {
        "schema": "tmmri.manuscript_tables_provenance/1", "run_id": run.run_id, "protocol_tag": study.freeze_tag,
        "run_protocol_tag": run_tag, "specification_matches_run_protocol_tag": matches,
        "amendment": "protocol-v6.0.1 (docs/MANUSCRIPT_PROTOCOL_V2_2026-10-02.md Section K numbering)", "include_mms": include_mms,
        "include_mms2": include_mms2, "min_cases_per_stratum": cx.min_cases, "small_stratum_rule": f"n shown; estimates {cx.small} (clusters)",
        "licence": "aggregates only; no case identifiers; no per-case M&Ms content (SPECIFICATION.md 14.5)", "manuscript_directory_written": False,
        "tables": {table.name: _table_provenance(table, *rendered[table.name], sources) for table in tables_v601},
    }
    prov_text = json.dumps(provenance, sort_keys=True, indent=2, ensure_ascii=True, default=str) + "\n"
    texts = {f"{name}.csv": csv.decode("utf-8") for name, (csv, _) in rendered.items()}
    texts.update({f"{name}.tex": tex.decode("utf-8") for name, (_, tex) in rendered.items()})
    texts[PROVENANCE] = prov_text
    texts[PROVENANCE_V601] = json.dumps(provenance_v601, sort_keys=True, indent=2, ensure_ascii=True, default=str) + "\n"
    assert_no_case_identifiers(texts, identifiers)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (csv, tex) in rendered.items():
        atomic_write_bytes(out_dir / f"{name}.csv", csv)
        atomic_write_bytes(out_dir / f"{name}.tex", tex)
    atomic_write_json(out_dir / PROVENANCE, provenance)
    atomic_write_json(out_dir / PROVENANCE_V601, provenance_v601)
    return {"tables": list(rendered), "sources": len(sources.hashes), "specification_matches_tag": matches}


def _numeric_column(table: Table, column: str) -> bool:
    values = [r.get(column) for r in table.rows if r.get(column) is not None]
    return any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)


def _sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def registry_file(study: Any) -> Path:
    for rel in study.study.get("registry_files", []):
        if Path(rel).stem == "semantic_registry":
            return Path(study.root) / rel
    return Path(study.root) / "registry" / "semantic_registry.yaml"
