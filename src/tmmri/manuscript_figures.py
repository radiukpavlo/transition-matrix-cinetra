"""Manuscript figures from recorded run artifacts only (protocol v6.0.1 A3; stage ``build-manuscript-figures``).

Figures (``<run>/figures/``; numbering of docs/MANUSCRIPT_PROTOCOL_V2_2026-10-02.md Section K and the v6.0.1 task list):
Figure 2 partition flow (ACDC, M&Ms-1 and the M&Ms-2 split), Figure 3 forest plot of the primary estimand rho by
partition and representation, Figure 4 per-target rho_j heatmap (internal test, external partitions and vendor
strata), Figure 5 representation panel (``build-comparison-tables`` output under ``probes/`` or ``trust/``), Figure 6
gate-power curves (``exploratory/B5/``), Figure 7 risk-coverage with label-free OOD flags (``trust/``) and Figure 8
label-free harmonization and simulated acquisition shift (``trust/``, ``simulated_shift/``).  Each figure is written
as a vector PDF and a 300-dpi PNG in the house style of :mod:`tmmri.clinical.figures`, with a provenance record
``<figure>.provenance.json`` (every source path with its SHA-256 and the keys read, every drawn or omitted panel with
its reason) and the index ``provenance.json``.

Same rules as ``build-manuscript-tables`` (Section 14.5): nothing is fitted or re-evaluated; only tabular artifacts
(``.json``, ``.parquet``, ``.csv``) are read, never an image or mask, so no M&Ms image can appear; no case identifier
appears in any figure text or provenance (checked against the split manifest before anything is written); an estimate
of a group with fewer than ``min_cases_per_stratum`` clusters is not drawn (counted in the provenance).  M&Ms rows
are used only when ``evaluate:mms_external`` is recorded, M&Ms-2 rows only when ``evaluate:mms2_external`` is.  A
panel whose inputs are absent, or not in the expected form, is omitted and the reason is written to the figure's
provenance record; a figure with no input at all is recorded as omitted and not drawn.

Input conventions for other lanes' artifacts (long tables, ``.csv``/``.parquet``, or ``.json`` holding a list of rows or
``{"rows" | "records" | "results" | "table": [...]}``), matched case-insensitively by the column synonyms of
:data:`SYNONYMS`: estimates (``estimate``/``rho``/``delta_rho``/``value``) with optional ``ci_lower``/``ci_upper`` and
``n_clusters``/``n``; grouping by ``partition``, ``representation``/``model``, ``metric``, ``variant``, ``vendor``/``group``;
curves by ``coverage`` and ``risk``; gate power by ``agreement``, ``support``, ``p_pass``/``power`` and ``gate``.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from .manuscript_tables import Sources, TableBuildError, assert_no_case_identifiers, case_identifiers
from .runs import RunRoot, atomic_write_bytes, atomic_write_json, hash_path

FIGURE_DPI = 300
PROVENANCE_INDEX = "provenance.json"
TABULAR = (".json", ".parquet", ".csv")
PARTITION_ORDER = ("acdc_dev_oof", "acdc_dev", "acdc_test", "mms_external", "mms_external_never_analysed", "mms2_external")
REPRESENTATION_ORDER = ("descriptors", "formal_a", "d20", "f9b", "f9")
SYNONYMS: dict[str, tuple[str, ...]] = {
    "estimate": ("estimate", "rho", "delta_rho", "value", "delta", "mean", "point"),
    "ci_lower": ("ci_lower", "lower", "ci_lo", "lo", "lower_95"),
    "ci_upper": ("ci_upper", "upper", "ci_hi", "hi", "upper_95"),
    "n": ("n_clusters", "n", "n_cases", "n_evaluable"),
    "partition": ("partition", "split"),
    "representation": ("representation", "model", "model_id", "predictor", "pipeline", "encoder"),
    "metric": ("metric", "estimand", "measure"),
    "variant": ("variant", "harmonization", "condition", "arm", "setting"),
    "group": ("vendor", "group", "stratum", "target_geometry", "geometry"),
    "coverage": ("coverage",),
    "risk": ("risk", "selective_risk", "error_rate", "error"),
    "flagged": ("proportion_flagged", "flagged_fraction", "fraction_flagged", "flag_rate", "prop_flagged", "proportion"),
    "agreement": ("agreement", "true_agreement", "planted_agreement", "rule_agreement"),
    "support": ("support", "rule_support", "n_support"),
    "p_pass": ("p_pass", "pass_probability", "power", "p_true_rule_passes", "pass_rate"),
    "gate": ("gate", "criterion", "gate_name", "rule_gate"),
    "false_pass": ("false_pass_rate", "p_false_pass", "false_pass"),
}


class PanelUnavailable(Exception):
    """The panel's inputs are absent or not in the expected form; the panel is omitted with this reason."""


# ----------------------------------------------------------------------------- sources and context


class FigureSources(Sources):
    """:class:`tmmri.manuscript_tables.Sources` plus CSV and long-table reads; ``reads`` logs every access so each
    panel's provenance lists exactly the files it used."""

    def __init__(self, run: RunRoot, allowed: Iterable[Path]) -> None:
        super().__init__(run, allowed)
        self.reads: list[str] = []

    def _load(self, path, reader):
        value, label = super()._load(path, reader)
        if label not in self.reads:
            self.reads.append(label)
        return value, label

    def table(self, path: str | Path) -> tuple[pd.DataFrame | None, str]:
        p = self._resolve(path)
        if not self.exists(p):
            return None, self.label(p)
        if p.suffix == ".parquet":
            return self._load(p, pd.read_parquet)
        if p.suffix == ".csv":
            return self._load(p, lambda q: pd.read_csv(q))
        payload, label = self.json(p)
        return _rows_from_json(payload), label


def _rows_from_json(payload: Any) -> pd.DataFrame | None:
    if isinstance(payload, list) and all(isinstance(r, dict) for r in payload):
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        for key in ("rows", "records", "results", "table"):
            if isinstance(payload.get(key), list) and all(isinstance(r, dict) for r in payload[key]):
                return pd.DataFrame(payload[key])
        if any(k in payload for k in SYNONYMS["estimate"]):
            return pd.DataFrame([{k: v for k, v in payload.items() if not isinstance(v, (dict, list))}])
    return None


@dataclass
class FigureContext:
    run: RunRoot
    study: Any
    src: FigureSources
    include_mms: bool
    include_mms2: bool

    @property
    def min_cases(self) -> int:
        return int(self.study.bootstrap["subgroup_reporting_thresholds"]["min_cases_per_stratum"])

    @property
    def sesoi(self) -> float:
        return float(self.study.bootstrap["primary_test"]["smallest_effect_of_interest"])

    def partition_allowed(self, value: Any) -> bool:
        text = str(value).lower()
        if "mms2" in text:
            return self.include_mms2
        if "mms" in text:
            return self.include_mms
        return True

    def below(self, n: Any) -> bool:
        try:
            value = float(n)
        except (TypeError, ValueError):
            return False
        return math.isfinite(value) and value < self.min_cases


def _column(frame: pd.DataFrame, role: str) -> str | None:
    lower = {str(c).lower(): c for c in frame.columns}
    for name in SYNONYMS[role]:
        if name in lower:
            return lower[name]
    return None


def normalize(frame: pd.DataFrame | None, label: str, required: Iterable[str], optional: Iterable[str] = ()) -> pd.DataFrame:
    """Columns renamed to their roles (:data:`SYNONYMS`); :class:`PanelUnavailable` naming the missing roles."""

    if frame is None or frame.empty:
        raise PanelUnavailable(f"{label}: no rows")
    out = {}
    missing = []
    for role in required:
        col = _column(frame, role)
        if col is None:
            missing.append(role)
        else:
            out[role] = frame[col]
    if missing:
        raise PanelUnavailable(f"{label}: columns for {missing} not found (synonyms {[SYNONYMS[m] for m in missing]})")
    for role in optional:
        col = _column(frame, role)
        if col is not None:
            out[role] = frame[col]
    return pd.DataFrame(out).reset_index(drop=True)


def _gate_and_suppress(fx: FigureContext, frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows of partitions whose evaluation is not recorded, and estimates of groups below the threshold."""

    if "partition" in frame.columns:
        frame = frame[frame["partition"].map(fx.partition_allowed)]
    suppressed = 0
    if "n" in frame.columns:
        small = frame["n"].map(fx.below).astype(bool)
        suppressed = int(small.sum())
        frame = frame[~small]
    for col in ("estimate", "ci_lower", "ci_upper", "coverage", "risk", "flagged", "agreement", "support", "p_pass", "false_pass"):
        if col in frame.columns:
            frame = frame.assign(**{col: pd.to_numeric(frame[col], errors="coerce")})
    return frame.reset_index(drop=True), suppressed


def _tables(fx: FigureContext, paths: list[Path], required: Iterable[str], optional: Iterable[str] = ()) -> tuple[pd.DataFrame, list[str]]:
    """Normalized rows of every parseable input; unparseable inputs are listed in the notes."""

    frames, notes = [], []
    for path in paths:
        frame, label = fx.src.table(path)
        try:
            norm = normalize(frame, label, required, optional)
        except PanelUnavailable as exc:
            notes.append(str(exc))
            continue
        norm["source"] = label
        frames.append(norm)
    if not frames:
        raise PanelUnavailable("; ".join(notes) or "no input")
    return pd.concat(frames, ignore_index=True), notes


# ----------------------------------------------------------------------------- house style


def _style():
    import matplotlib

    matplotlib.use("Agg")
    from .clinical.figures import NATURE_PALETTE, configure_nature_style

    configure_nature_style()
    return NATURE_PALETTE


def _colors(n: int) -> list[str]:
    from .clinical.figures import NATURE_PALETTE

    cycle = [NATURE_PALETTE[k] for k in ("blue", "amber", "teal", "red", "purple", "green", "navy", "cyan", "grey")]
    return [cycle[i % len(cycle)] for i in range(n)]


def _clean(ax) -> None:
    from .clinical.figures import _clean_axes

    _clean_axes(ax)


def _order(values: Iterable[Any], preferred: tuple[str, ...]) -> list[str]:
    """Preferred values first, then the rest in natural order (numbers numerically, then text)."""

    def natural(v: str) -> tuple:
        try:
            return (0, float(v), v)
        except ValueError:
            return (1, 0.0, v)

    seen = list(dict.fromkeys(str(v) for v in values))
    head = [p for p in preferred if p in seen]
    return head + sorted((v for v in seen if v not in head), key=natural)


def _interval_plot(ax, frame: pd.DataFrame, y_key: str, hue_key: str | None, *, y_order: tuple[str, ...] = (), hue_order: tuple[str, ...] = (),
                   reference: Iterable[float] = (0.0,), xlabel: str = "") -> None:
    """Dot-and-interval plot: one row per ``y_key`` value, offset per ``hue_key`` value."""

    ys = _order(frame[y_key], y_order)
    hues = _order(frame[hue_key], hue_order) if hue_key else ["all"]
    colors = dict(zip(hues, _colors(len(hues))))
    step = 0.7 / max(len(hues), 1)
    for h_i, hue in enumerate(hues):
        part = frame if hue_key is None else frame[frame[hue_key].astype(str) == hue]
        for _, row in part.iterrows():
            y = ys.index(str(row[y_key])) + (h_i - (len(hues) - 1) / 2) * step
            est = row.get("estimate")
            if est is None or not np.isfinite(est):
                continue
            lo, hi = row.get("ci_lower"), row.get("ci_upper")
            if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
                ax.plot([lo, hi], [y, y], color=colors[hue], lw=1.0)
            ax.plot([est], [y], "o", color=colors[hue], ms=3.5, label=hue if hue_key else None)
    for x in reference:
        ax.axvline(x, color="#6B7280", lw=0.6, ls="--" if x else "-")
    ax.set_yticks(range(len(ys)))
    ax.set_yticklabels(ys)
    ax.set_ylim(len(ys) - 0.5, -0.5)                 # first row on top; offsets stay within their row
    ax.set_xlabel(xlabel)
    if hue_key:
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), frameon=False, loc="best")
    _clean(ax)


# ----------------------------------------------------------------------------- panels


def _flow(fig, spec, fx: FigureContext, paths: list[Path]) -> dict[str, Any]:
    from .clinical.figures import NATURE_PALETTE

    split, label = fx.src.table("contracts/split_manifest.parquet")
    if split is None:
        raise PanelUnavailable(f"{label} absent")
    dataset = split["dataset"].astype(str) if "dataset" in split.columns else split["sample_id"].astype(str).str.split(":").str[0]
    columns = [("ACDC", "ACDC", True), ("MMS", "M&Ms-1", fx.include_mms), ("MMS2", "M&Ms-2", fx.include_mms2)]
    sizes = dict(fx.study.study.get("partition_sizes") or {})
    ax = fig.add_subplot(spec)
    ax.set_axis_off()
    ax.set_xlim(0, 3)
    ax.set_ylim(0.2, 1)
    notes = []
    box = dict(boxstyle="round,pad=0.35", fc=NATURE_PALETTE["near_white"], ec=NATURE_PALETTE["navy"], lw=0.6)
    grey = dict(boxstyle="round,pad=0.3", fc="white", ec=NATURE_PALETTE["grey"], lw=0.5, ls="--")
    for i, (ds, name, included) in enumerate(columns):
        x = i + 0.5
        rows = split[dataset == ds]
        if not included:
            declared = {k: v for k, v in sizes.items() if str(k).startswith("mms2" if ds == "MMS2" else "mms_")} if ds != "ACDC" else {}
            text = f"{name}\n" + ("\n".join(f"{k}: n = {v} (declared)" for k, v in sorted(declared.items())) if declared else "not shown")
            ax.text(x, 0.55, text + f"\nevaluate:{'mms2' if ds == 'MMS2' else 'mms'}_external not recorded", ha="center", va="center", bbox=grey, fontsize=6)
            notes.append(f"{name}: evaluation not recorded; counts withheld (declared sizes only)")
            continue
        cohort, _ = fx.src.table(f"contracts/cohort_{ds}.parquet")
        declared = sorted((k, v) for k, v in sizes.items() if ds == "MMS2" and str(k).startswith("mms2"))
        if cohort is None and rows.empty:
            text = f"{name}\nnot in the split manifest" + "".join(f"\n{k}: n = {v} (declared)" for k, v in declared)
            ax.text(x, 0.6, text, ha="center", va="center", bbox=grey, fontsize=6)
            notes.append(f"{name}: evaluation recorded but no {ds} rows in contracts/split_manifest.parquet (declared sizes only)")
            continue
        release = int(len(cohort)) if cohort is not None else int(len(rows))
        ax.text(x, 0.88, f"{name}\nrelease: n = {release}", ha="center", va="center", bbox=box, fontweight="bold")
        excluded = rows[rows["partition"].astype(str) == "excluded"] if "partition" in rows.columns else rows.iloc[0:0]
        n_excl = int(len(excluded))
        reasons = ""
        if cohort is not None and {"exclusion_reason", "eligible_v6"} <= set(cohort.columns):
            r = cohort.loc[~cohort["eligible_v6"].astype(bool), "exclusion_reason"].map(lambda v: "unspecified" if v is None or pd.isna(v) or not str(v).strip()
                                                                                       else str(v).split(":", 1)[0].strip())
            counts = r.value_counts().sort_index()
            n_excl = max(n_excl, int(counts.sum()))
            reasons = "\n".join(f"{k}: {v}" for k, v in counts.items())
        ax.annotate("", xy=(x, 0.53), xytext=(x, 0.79), arrowprops=dict(arrowstyle="-|>", lw=0.6, color=NATURE_PALETTE["navy"]))
        ax.text(x + 0.06, 0.66, f"excluded: n = {n_excl}" + (f"\n{reasons}" if reasons else ""), ha="left", va="center", fontsize=5.5, bbox=grey)
        parts = [p for p in _order(rows["partition"].astype(str), PARTITION_ORDER) if p != "excluded"] if "partition" in rows.columns else []
        clusters = rows["cluster_id"].astype(str) if "cluster_id" in rows.columns else rows["sample_id"].astype(str)
        lines = []
        for part in parts:
            sel = rows["partition"].astype(str) == part
            lines.append(f"{part}: n = {int(sel.sum())} ({int(clusters[sel].nunique())} clusters)")
            if part == "mms_external" and "never_analysed" in rows.columns:
                lines.append(f"of which never analysed: n = {int((sel & rows['never_analysed'].fillna(False).astype(bool)).sum())}")
        lines += [f"{k}: n = {v} (declared; never extracted under v6)" for k, v in declared if k not in parts]
        ax.text(x, 0.50, "\n".join(lines) or "no partitions", ha="center", va="top", bbox=box, fontsize=6)
    return {"axes": [ax], "keys": ["count(sample_id) and nunique(cluster_id) by dataset and partition", "configs/study_v6.yaml partition_sizes"],
            "note": "; ".join(notes) or None}


def _rho_entries(fx: FigureContext, paths: list[Path]) -> tuple[pd.DataFrame, list[str]]:
    rows, keys = [], []
    variants, v_label = fx.src.json("metrics/ablations/transition_variants.json")
    entry = ((variants or {}).get("acdc_dev_oof") or {}).get("ridge_typed_formal57")
    if isinstance(entry, dict):
        rows.append({"partition": "acdc_dev_oof", "representation": "descriptors", "estimate": entry.get("rho"), "ci_lower": entry.get("ci_lower"),
                     "ci_upper": entry.get("ci_upper"), "n": entry.get("n_clusters", entry.get("n_cases")), "source": v_label})
        keys.append(f"{v_label}: acdc_dev_oof.ridge_typed_formal57.rho")
    for path in paths:
        payload, label = fx.src.json(path)
        if not isinstance(payload, dict) or "rho" not in payload:
            continue
        rel = path.relative_to(fx.run.path)
        if rel.parts[0] == "metrics":
            representation = "descriptors"
            partition = payload.get("partition") or path.stem.replace("primary_rho_", "")
        else:
            representation = str(payload.get("representation") or path.parent.name)
            partition = payload.get("partition") or path.stem.split("rho_", 1)[-1]
        rows.append({"partition": str(partition), "representation": representation, "estimate": payload.get("rho"), "ci_lower": payload.get("ci_lower"),
                     "ci_upper": payload.get("ci_upper"), "n": payload.get("n_clusters", payload.get("n_cases")), "source": label})
        keys.append(f"{label}: rho, ci_lower, ci_upper, n_clusters")
    return pd.DataFrame(rows), keys


def _forest(fig, spec, fx: FigureContext, paths: list[Path]) -> dict[str, Any]:
    frame, keys = _rho_entries(fx, paths)
    if frame.empty:
        raise PanelUnavailable("no recorded rho payload (metrics/primary_rho_*.json, probes/**)")
    frame, suppressed = _gate_and_suppress(fx, frame)
    if frame.empty:
        raise PanelUnavailable("every rho estimate is withheld (partition not evaluated or n below the threshold)")
    ax = fig.add_subplot(spec)
    _interval_plot(ax, frame, "partition", "representation", y_order=PARTITION_ORDER, hue_order=REPRESENTATION_ORDER, reference=(0.0, fx.sesoi),
                   xlabel="ρ (family-mean relative MAE reduction vs the mean; 95 % cluster-bootstrap CI)")
    return {"axes": [ax], "keys": keys, "suppressed": suppressed}


def _heatmap(fig, spec, fx: FigureContext, paths: list[Path]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    targets = list(fx.study.primary_family())
    cols, values, small, keys = [], [], [], []
    for part in ("acdc_test", "mms_external", "mms2_external"):
        if not fx.partition_allowed(part):
            continue
        payload, label = fx.src.json(f"metrics/primary_rho_{part}.json")
        if isinstance(payload, dict) and isinstance(payload.get("rho_j"), dict):
            cols.append(part)
            values.append([payload["rho_j"].get(t) for t in targets])
            small.append(False)
            keys.append(f"{label}: rho_j")
        strata, s_label = fx.src.json(f"metrics/strata_rho_{part}.json")
        for vendor, entry in sorted((((strata or {}).get("strata") or {}).get("vendor") or {}).items()):
            if not isinstance(entry, dict):
                continue
            cols.append(f"{part}\nvendor {vendor}")
            below = fx.below(entry.get("n_clusters", entry.get("n_cases")))
            values.append([None if below else (entry.get("rho_j") or {}).get(t) for t in targets])
            small.append(below)
            keys.append(f"{s_label}: strata.vendor.{vendor}.rho_j")
    if not cols:
        raise PanelUnavailable("no recorded rho_j (metrics/primary_rho_<partition>.json)")
    m = np.array([[np.nan if v is None else float(v) for v in col] for col in values]).T
    ax = fig.add_subplot(spec)
    image = ax.imshow(np.ma.masked_invalid(m), cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_facecolor("#E8EDF2")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            text = "n<{}".format(fx.min_cases) if small[j] else ("NE" if not np.isfinite(m[i, j]) else f"{m[i, j]:.2f}")
            ax.text(j, i, text, ha="center", va="center", fontsize=5.5, color="black")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=0, fontsize=5.5)
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets)
    cbar = plt.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("ρ_j")
    return {"axes": [ax], "keys": keys, "suppressed": int(sum(small))}


def _representations(fig, spec, fx: FigureContext, paths: list[Path]) -> dict[str, Any]:
    frame, notes = _tables(fx, paths, ("estimate", "representation"), ("metric", "partition", "ci_lower", "ci_upper", "n"))
    if "metric" not in frame.columns:
        frame["metric"] = "rho"
    if "partition" not in frame.columns:
        frame["partition"] = "all"
    frame, suppressed = _gate_and_suppress(fx, frame)
    preferred = ("rho", "macro_f1", "accuracy", "macro_ovr_auroc", "ece", "brier", "log_loss")
    metrics = _order(frame["metric"].dropna().astype(str), preferred)[:4]
    if not metrics:
        raise PanelUnavailable("no estimates after gating")
    sub = spec.subgridspec(1, len(metrics), wspace=0.6)
    axes = []
    for k, metric in enumerate(metrics):
        ax = fig.add_subplot(sub[0, k])
        _interval_plot(ax, frame[frame["metric"].astype(str) == metric], "partition", "representation", y_order=PARTITION_ORDER,
                       hue_order=REPRESENTATION_ORDER, reference=(0.0,) if metric == "rho" else (), xlabel=metric)
        if k:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
        axes.append(ax)
    return {"axes": axes, "keys": ["estimate, ci_lower, ci_upper by representation, metric and partition"], "suppressed": suppressed,
            "note": "; ".join(notes) or None}


def _gate_power(fig, spec, fx: FigureContext, paths: list[Path]) -> dict[str, Any]:
    frame, notes = _tables(fx, paths, ("agreement", "p_pass"), ("support", "gate", "false_pass"))
    frame, _ = _gate_and_suppress(fx, frame)
    if "gate" not in frame.columns:
        frame["gate"] = "v6 gate"
    if "support" not in frame.columns:
        frame["support"] = "all"
    gates = _order(frame["gate"].astype(str), ("v6",))[:3]
    extra = "false_pass" in frame.columns and frame["false_pass"].notna().any()
    sub = spec.subgridspec(1, len(gates) + int(extra), wspace=0.45)
    axes = []
    supports = _order(frame["support"].astype(str), ())
    colors = dict(zip(supports, _colors(len(supports))))
    for k, gate in enumerate(gates):
        ax = fig.add_subplot(sub[0, k])
        part = frame[frame["gate"].astype(str) == gate]
        for support in supports:
            s = part[part["support"].astype(str) == support].groupby("agreement", as_index=False)["p_pass"].mean().sort_values("agreement")
            if len(s):
                ax.plot(s["agreement"], s["p_pass"], "-o", color=colors[support], ms=2.5, label=f"support {support}")
        ax.set_ylim(0, 1)
        ax.set_xlabel("planted rule agreement")
        ax.set_ylabel("P(true rule passes)" if k == 0 else "")
        ax.text(0.02, 0.98, str(gate), transform=ax.transAxes, va="top", fontsize=6)
        if k == 0:
            ax.legend(frameon=False, loc="lower right")
        _clean(ax)
        axes.append(ax)
    if extra:
        ax = fig.add_subplot(sub[0, len(gates)])
        fp = frame.dropna(subset=["false_pass"]).groupby("gate", as_index=False)["false_pass"].mean()
        ax.bar(range(len(fp)), fp["false_pass"], color=_colors(len(fp)))
        ax.set_xticks(range(len(fp)))
        ax.set_xticklabels(fp["gate"].astype(str), rotation=30, ha="right")
        ax.set_ylabel("false-pass rate")
        _clean(ax)
        axes.append(ax)
    return {"axes": axes, "keys": ["agreement, support, p_pass, gate, false_pass (EXPLORATORY_POST_HOC, B5)"], "note": "; ".join(notes) or None,
            "label": "EXPLORATORY_POST_HOC"}


def _risk_coverage(fig, spec, fx: FigureContext, paths: list[Path]) -> dict[str, Any]:
    frame, notes = _tables(fx, paths, ("coverage", "risk"), ("partition", "representation", "n"))
    frame, suppressed = _gate_and_suppress(fx, frame)
    if frame.empty:
        raise PanelUnavailable("no rows after gating")
    for col, default in (("partition", "all"), ("representation", "model")):
        if col not in frame.columns:
            frame[col] = default
    ax = fig.add_subplot(spec)
    parts = _order(frame["partition"].astype(str), PARTITION_ORDER)
    colors = dict(zip(parts, _colors(len(parts))))
    styles = ["-", "--", ":", "-."]
    reps = _order(frame["representation"].astype(str), REPRESENTATION_ORDER)
    for part in parts:
        for r_i, rep in enumerate(reps):
            curve = frame[(frame["partition"].astype(str) == part) & (frame["representation"].astype(str) == rep)].sort_values("coverage")
            if len(curve):
                ax.plot(curve["coverage"], curve["risk"], styles[r_i % len(styles)], color=colors[part], lw=1.0, label=f"{part} · {rep}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("coverage")
    ax.set_ylabel("risk (error rate among accepted)")
    ax.legend(frameon=False, fontsize=5.5)
    _clean(ax)
    return {"axes": [ax], "keys": ["coverage, risk by partition and representation (S1)"], "suppressed": suppressed, "note": "; ".join(notes) or None}


def _ood(fig, spec, fx: FigureContext, paths: list[Path]) -> dict[str, Any]:
    frame, notes = _tables(fx, paths, ("flagged",), ("partition", "group", "representation", "ci_lower", "ci_upper", "n"))
    frame, suppressed = _gate_and_suppress(fx, frame)
    if "partition" not in frame.columns and "group" not in frame.columns:
        raise PanelUnavailable("OOD table has neither partition nor group columns")
    frame["label"] = frame.get("partition", pd.Series("all", index=frame.index)).astype(str)
    if "group" in frame.columns:
        frame["label"] = frame["label"] + np.where(frame["group"].notna(), " · " + frame["group"].astype(str), "")
    frame = frame.rename(columns={"flagged": "estimate"})
    if frame.empty:
        raise PanelUnavailable("no rows after gating")
    ax = fig.add_subplot(spec)
    _interval_plot(ax, frame, "label", "representation" if "representation" in frame.columns else None, y_order=PARTITION_ORDER,
                   hue_order=REPRESENTATION_ORDER, reference=(0.05,), xlabel="proportion flagged (label-free OOD; dev 95th percentile)")
    return {"axes": [ax], "keys": ["proportion flagged by partition, group and representation (S2)"], "suppressed": suppressed, "note": "; ".join(notes) or None}


def _harmonization(fig, spec, fx: FigureContext, paths: list[Path]) -> dict[str, Any]:
    frame, notes = _tables(fx, paths, ("estimate", "variant"), ("partition", "metric", "ci_lower", "ci_upper", "n"))
    if "metric" in frame.columns:
        rho = frame[frame["metric"].astype(str).str.contains("rho", case=False, na=False)]
        frame = rho if len(rho) else frame
    if "partition" not in frame.columns:
        frame["partition"] = "all"
    frame, suppressed = _gate_and_suppress(fx, frame)
    if frame.empty:
        raise PanelUnavailable("no rows after gating")
    ax = fig.add_subplot(spec)
    _interval_plot(ax, frame, "partition", "variant", y_order=PARTITION_ORDER, hue_order=("frozen", "H-a", "H-b"), reference=(0.0,),
                   xlabel="ρ under label-free harmonization (S3; transductive)")
    return {"axes": [ax], "keys": ["estimate, ci by variant and partition (S3)"], "suppressed": suppressed, "note": "; ".join(notes) or None}


def _simulated_shift(fig, spec, fx: FigureContext, paths: list[Path]) -> dict[str, Any]:
    frame, notes = _tables(fx, paths, ("estimate", "group"), ("representation", "metric", "partition", "ci_lower", "ci_upper", "n"))
    if "metric" in frame.columns:
        rho = frame[frame["metric"].astype(str).str.contains("rho", case=False, na=False)]
        frame = rho if len(rho) else frame
    frame, suppressed = _gate_and_suppress(fx, frame)
    if frame.empty:
        raise PanelUnavailable("no rows after gating")
    ax = fig.add_subplot(spec)
    _interval_plot(ax, frame, "group", "representation" if "representation" in frame.columns else None, hue_order=REPRESENTATION_ORDER,
                   reference=(0.0,), xlabel="Δρ under simulated acquisition geometry (S6; acdc_test)")
    return {"axes": [ax], "keys": ["estimate, ci by simulated vendor geometry and representation (S6)"], "suppressed": suppressed,
            "note": "; ".join(notes) or None}


# ----------------------------------------------------------------------------- figure specifications


@dataclass(frozen=True)
class PanelSpec:
    key: str
    description: str
    patterns: tuple[str, ...]
    draw: Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class FigureSpec:
    name: str
    title: str
    panels: tuple[PanelSpec, ...]
    height_mm: float = 75.0


def _tab(*stems: str) -> tuple[str, ...]:
    return tuple(f"{stem}{ext}" for stem in stems for ext in TABULAR)


FIGURES: tuple[FigureSpec, ...] = (
    FigureSpec("figure02_partition_flow", "Partition flow: ACDC, M&Ms-1 and the M&Ms-2 split",
               (PanelSpec("flow", "cases and clusters per dataset and partition", ("contracts/split_manifest.parquet", "contracts/cohort_*.parquet"), _flow),),
               height_mm=60.0),
    FigureSpec("figure03_rho_forest", "Primary estimand rho by partition and representation",
               (PanelSpec("forest", "rho with 95 % cluster-bootstrap CI", ("metrics/primary_rho_*.json", "metrics/ablations/transition_variants.json",
                                                                            "probes/**/*rho*.json"), _forest),)),
    FigureSpec("figure04_external_target_heatmap", "Per-target rho_j: internal test, external partitions and vendor strata",
               (PanelSpec("heatmap", "rho_j per primary target", ("metrics/primary_rho_*.json", "metrics/strata_rho_*.json"), _heatmap),), height_mm=80.0),
    FigureSpec("figure05_representation_panel", "Representations head to head (descriptors, D20, F9/F9b)",
               (PanelSpec("comparison", "build-comparison-tables estimates", _tab("probes/**/*comparison*", "trust/**/*comparison*"), _representations),)),
    FigureSpec("figure06_gate_power", "Gate power of planted rules (exploratory B5)",
               (PanelSpec("gate_power", "P(true rule passes) by agreement and support", _tab("exploratory/B5/**/*"), _gate_power),)),
    FigureSpec("figure07_risk_coverage_ood", "Risk-coverage and label-free OOD flags (S1, S2)",
               (PanelSpec("risk_coverage", "risk-coverage curves", _tab("trust/**/*risk_coverage*"), _risk_coverage),
                PanelSpec("ood_flags", "proportion flagged by partition", _tab("trust/**/*ood*"), _ood))),
    FigureSpec("figure08_harmonization_simulated_shift", "Label-free harmonization (S3) and simulated acquisition shift (S6)",
               (PanelSpec("harmonization", "rho under H-a and H-b against the frozen bundle", _tab("trust/**/*harmoni*"), _harmonization),
                PanelSpec("simulated_shift", "delta rho under simulated vendor geometry", _tab("simulated_shift/**/*"), _simulated_shift))),
)
FIGURE_NAMES = tuple(f.name for f in FIGURES)


# ----------------------------------------------------------------------------- planning, rendering, stage


def _candidates(run: RunRoot, patterns: Iterable[str], include_mms: bool, include_mms2: bool) -> list[Path]:
    found: set[Path] = set()
    for pattern in patterns:
        for path in run.path.glob(pattern):
            name = path.name.lower()
            if not path.is_file() or path.suffix not in TABULAR or name.startswith(".") or "provenance" in name or ".stages" in path.parts:
                continue
            if "mms2" in name and not include_mms2:
                continue
            if "mms" in name and "mms2" not in name and not include_mms:
                continue
            found.add(path)
    return sorted(found, key=lambda p: p.as_posix())


def plan_figures(run: RunRoot, *, include_mms: bool, include_mms2: bool) -> dict[str, dict[str, list[Path]]]:
    """Candidate inputs of every panel (existing tabular files only); a figure with none is omitted."""

    return {f.name: {p.key: _candidates(run, p.patterns, include_mms, include_mms2) for p in f.panels} for f in FIGURES}


def figure_outputs(fdir: Path, plan: dict[str, dict[str, list[Path]]]) -> list[Path]:
    out = [fdir / PROVENANCE_INDEX] + [fdir / f"{name}.provenance.json" for name in FIGURE_NAMES]
    for name in FIGURE_NAMES:
        if any(plan[name].values()):
            out += [fdir / f"{name}.pdf", fdir / f"{name}.png"]
    return out


def _render(fig) -> tuple[bytes, bytes, list[str]]:
    import matplotlib.text

    fig.canvas.draw()
    texts = [t.get_text() for t in fig.findobj(matplotlib.text.Text) if t.get_text()]
    pdf, png = io.BytesIO(), io.BytesIO()
    fig.savefig(pdf, format="pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(png, format="png", dpi=FIGURE_DPI)
    return pdf.getvalue(), png.getvalue(), texts


def build_figures(run: RunRoot, study: Any, *, include_mms: bool, include_mms2: bool, out_dir: Path) -> dict[str, Any]:
    """Draw every planned figure (see the module docstring) and write the figures and their provenance records."""

    import matplotlib.pyplot as plt

    from .clinical.figures import DOUBLE_COLUMN_MM, _figsize

    _style()
    plan = plan_figures(run, include_mms=include_mms, include_mms2=include_mms2)
    allowed = {p for panels in plan.values() for paths in panels.values() for p in paths} | {run.path / "contracts" / "split_manifest.parquet"}
    fx = FigureContext(run, study, FigureSources(run, allowed), include_mms, include_mms2)
    split, _ = fx.src.table("contracts/split_manifest.parquet")
    mms2_split = run.path / "contracts" / "split_manifest_MMS2.parquet"     # protocol-v6.0.1: M&Ms-2 ids are guarded too
    identifiers = case_identifiers([split, pd.read_parquet(mms2_split) if mms2_split.is_file() else None])
    rendered: dict[str, tuple[bytes, bytes]] = {}
    records: dict[str, dict[str, Any]] = {}
    for spec in FIGURES:
        panels_plan = plan[spec.name]
        record: dict[str, Any] = {"schema": "tmmri.manuscript_figure_provenance/1", "figure": spec.name, "title": spec.title, "run_id": run.run_id,
                                  "protocol_tag": study.freeze_tag, "include_mms": include_mms, "include_mms2": include_mms2,
                                  "min_cases_per_stratum": fx.min_cases, "dpi": FIGURE_DPI, "formats": ["pdf (vector)", f"png ({FIGURE_DPI} dpi)"],
                                  "licence": "aggregates only; tabular artifacts only (no image or mask read, no M&Ms image); no case identifiers; "
                                             f"groups with fewer than {fx.min_cases} clusters are not drawn (SPECIFICATION.md 14.5)",
                                  "panels": []}
        if not any(panels_plan.values()):
            record.update({"status": "OMITTED", "files": {}, "note": "no recorded input for any panel"})
            record["panels"] = [{"panel": p.key, "description": p.description, "status": "OMITTED", "sources": [],
                                 "note": f"inputs absent (patterns {list(p.patterns)})"} for p in spec.panels]
            records[spec.name] = record
            continue
        planned = [p for p in spec.panels if panels_plan[p.key]]
        fig = plt.figure(figsize=_figsize(DOUBLE_COLUMN_MM, spec.height_mm))
        grid = fig.add_gridspec(1, len(planned), wspace=0.55)
        drawn = 0
        letters = iter("abcdefgh")
        for p in spec.panels:
            entry: dict[str, Any] = {"panel": p.key, "description": p.description}
            if p not in planned:
                entry.update({"status": "OMITTED", "sources": [], "note": f"inputs absent (patterns {list(p.patterns)})"})
                record["panels"].append(entry)
                continue
            fx.src.reads = []
            cell = grid[0, planned.index(p)]
            try:
                result = p.draw(fig, cell, fx, panels_plan[p.key])
            except PanelUnavailable as exc:
                for ax in list(fig.axes):
                    if ax.get_subplotspec() is not None and ax.get_subplotspec().get_topmost_subplotspec() == cell.get_topmost_subplotspec():
                        fig.delaxes(ax)
                entry.update({"status": "OMITTED", "sources": [{"path": label, "sha256": fx.src.hashes.get(label)} for label in fx.src.reads],
                              "note": str(exc)})
                record["panels"].append(entry)
                continue
            letter = next(letters)
            axes = result["axes"]
            axes[0].text(-0.08, 1.06, letter, transform=axes[0].transAxes, fontweight="bold", fontsize=8, va="bottom", ha="right")
            drawn += 1
            entry.update({"status": "DRAWN", "letter": letter, "sources": [{"path": label, "sha256": fx.src.hashes.get(label)} for label in fx.src.reads],
                          "keys": result.get("keys", []), "suppressed_below_min_cases": int(result.get("suppressed", 0)), "note": result.get("note"),
                          "label": result.get("label")})
            record["panels"].append(entry)
        if not drawn:
            ax = fig.add_subplot(grid[0, 0])
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No panel could be drawn from the recorded inputs (see the provenance record)", ha="center", va="center")
        pdf, png, texts = _render(fig)
        plt.close(fig)
        assert_no_case_identifiers({f"{spec.name} (figure text)": "\n".join(texts)}, identifiers)
        rendered[spec.name] = (pdf, png)
        record.update({"status": "DRAWN" if drawn else "PLACEHOLDER", "files": {
            "pdf": {"path": f"figures/{spec.name}.pdf", "sha256": _sha256(pdf)}, "png": {"path": f"figures/{spec.name}.png", "sha256": _sha256(png)}}})
        records[spec.name] = record
    index = {"schema": "tmmri.manuscript_figures_index/1", "run_id": run.run_id, "protocol_tag": study.freeze_tag, "include_mms": include_mms,
             "include_mms2": include_mms2, "figures": {name: {"status": r["status"], "provenance": f"{name}.provenance.json", "files": r["files"]}
                                                       for name, r in records.items()},
             "sources": {label: digest for label, digest in sorted(fx.src.hashes.items())}, "manuscript_directory_written": False}
    texts = {f"{name}.provenance.json": json.dumps(r, sort_keys=True, default=str) for name, r in records.items()}
    texts[PROVENANCE_INDEX] = json.dumps(index, sort_keys=True, default=str)
    assert_no_case_identifiers(texts, identifiers)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (pdf, png) in rendered.items():
        atomic_write_bytes(out_dir / f"{name}.pdf", pdf)
        atomic_write_bytes(out_dir / f"{name}.png", png)
    for name, record in records.items():
        atomic_write_json(out_dir / f"{name}.provenance.json", record, allow_nan=True)
    atomic_write_json(out_dir / PROVENANCE_INDEX, index, allow_nan=True)
    return {"figures": {name: r["status"] for name, r in records.items()}}


def _sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def build_manuscript_figures(ctx: Any):
    """Stage ``build-manuscript-figures``: the figures of :data:`FIGURES` from recorded artifacts (read-only), under the
    rules of ``build-manuscript-tables``: requires ``.stages/FROZEN.json`` and ``evaluate:acdc_test``; M&Ms inputs only
    with a recorded ``evaluate:mms_external`` (and :func:`tmmri.stages.require_mms_unlocked`), M&Ms-2 inputs only with a
    recorded ``evaluate:mms2_external`` (and its lock)."""

    from . import stages
    from .predictor_reporting import evaluated, require_partition_unlocked

    run = ctx.require_run("build-manuscript-figures")
    if not run.frozen_path.is_file():
        raise stages.BundleError(f"build-manuscript-figures: run {run.run_id} has no frozen analysis bundle (run freeze-analysis-bundle first)")
    stages._require_evaluated(run, "evaluate:acdc_test", "build-manuscript-figures")
    include_mms = evaluated(run, "evaluate:mms_external")
    include_mms2 = evaluated(run, "evaluate:mms2_external")
    if include_mms:
        require_partition_unlocked(ctx, "mms_external", "build-manuscript-figures")
    if include_mms2:
        require_partition_unlocked(ctx, "mms2_external", "build-manuscript-figures")
    fdir = run.sub("figures")
    plan = plan_figures(run, include_mms=include_mms, include_mms2=include_mms2)
    inputs = sorted({run.frozen_path, run.record_path("evaluate:acdc_test"), run.path / "contracts" / "split_manifest.parquet",
                     *[p for panels in plan.values() for paths in panels.values() for p in paths]}, key=lambda q: q.as_posix())
    outputs = figure_outputs(fdir, plan)

    def compute() -> dict[str, Any]:
        try:
            return build_figures(run, ctx.study, include_mms=include_mms, include_mms2=include_mms2, out_dir=fdir)
        except TableBuildError as exc:
            raise stages.BundleError(f"build-manuscript-figures: {exc}") from exc

    return stages._stage(ctx, "build-manuscript-figures", compute, inputs, outputs,
                         {"figures": [n for n in FIGURE_NAMES if any(plan[n].values())], "include_mms": include_mms, "include_mms2": include_mms2})
