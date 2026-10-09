"""Table 4 (representation x partition) and the tidy Figure-5 parquet from the recorded alternative-bundle evaluations.

Sources are ``probes/evaluations/<bundle>/<partition>.json`` (stage ``evaluate-alt-bundles``); nothing is fitted or
re-evaluated.  Each Table 4 row pairs a representation with its own predictor (configs/probes.yaml ``pairing``):
descriptors with the audited predictor, D20 with the D20 predictor, F9b with F9 (primary seed).  Cells are
``estimate [lower, upper]`` (paired cluster bootstrap, 2,000 replicates); a missing evaluation is ``NE`` with its
reason, and a cell resting on fewer than ``min_clusters`` clusters is ``NE (n < 20)``.  The Figure-5 parquet holds one
row per (representation or predictor, partition, metric, kind) with ``kind`` in {estimate, difference_vs_descriptors}
and every F9 seed (the primary flagged).  Aggregates only; no case identifier is written (checked).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..manuscript_tables import Cell, Table, assert_no_case_identifiers, case_identifiers, render_latex
from ..runs import RunRoot, atomic_write_bytes, atomic_write_json, hash_path

TABLE4_NAME = "table04_representation_comparison"
FIGURE5_NAME = "figure05_representation_panel.parquet"
COLUMNS = ["representation", "predictor", "evidence", "n_clusters_rho", "rho", "rho_LVEF", "rho_LVEDV", "n_clusters_class", "accuracy",
           "macro_F1", "AUROC", "ECE", "status"]
PREDICTOR_COLUMNS = {"accuracy": "accuracy", "macro_F1": "macro_f1", "AUROC": "macro_ovr_auroc", "ECE": "ece"}
FIGURE_COLUMNS = ["representation", "predictor", "partition", "evidence", "seed", "primary_seed", "metric", "kind", "estimate", "ci_lower", "ci_upper",
                  "status", "reason", "n_clusters", "source"]
LABELS = {"descriptors": "57 mask descriptors (formal A)", "d20": "D20 frozen ResNet-50 (2048-d)", "f9b": "F9b trained CNN probe (64-d)",
          "audited_predictor": "audited predictor", "d20_predictor": "D20 predictor", "f9": "F9 trained CNN"}


def _fmt(value: Any) -> str:
    return f"{float(value):.3f}"


def _cell(estimate: Any, lower: Any, upper: Any, *, status: str | None, reason: str | None, n_clusters: int | None, min_clusters: int,
          path: str | None, key: str) -> Cell:
    if n_clusters is not None and n_clusters < min_clusters:
        return Cell(f"NE (n < {min_clusters})", path, key)
    if estimate is None:
        return Cell(f"NE ({reason or 'not estimable'})", path, key)
    text = _fmt(estimate)
    if lower is not None and upper is not None:
        text += f" [{_fmt(lower)}, {_fmt(upper)}]"
    elif status == "NE":
        text += f" (CI NE: {reason})"
    return Cell(text, path, key)


def _locate(docs: dict[tuple[str, str], tuple[dict | None, str]], bundles: dict[str, Any], kind: str, name: str, partition: str):
    """(entry, source label) of a representation or predictor in the evaluation of its bundle on ``partition``."""

    for bundle, spec in bundles.items():
        if name in (spec.get(kind) or []):
            doc, label = docs[(bundle, partition)]
            if doc is None:
                return None, label, f"NOT_EVALUATED (evaluate-alt-bundles --bundle {bundle} --split {partition} not recorded)"
            entry = doc.get(kind, {}).get(name)
            return entry, label, None if entry is not None else f"{name} absent from {label}"
    return None, None, f"{name} is not part of any configured bundle"


def _doc_label(docs: dict[tuple[str, str], tuple[dict | None, str]], bundles: dict[str, Any], kind: str, name: str, partition: str) -> str | None:
    """The evidence label recorded by the evaluation that holds ``name`` on ``partition`` (for example ``post hoc`` on
    ``mms2_external`` when the bundle was frozen after the M&Ms-2 extraction; SPECIFICATION.md 14.6)."""

    for bundle, spec in bundles.items():
        if name in (spec.get(kind) or []):
            doc, _ = docs[(bundle, partition)]
            return None if doc is None else doc.get("label")
    return None


def _descriptor_doc(docs, bundles, partition):
    for bundle in bundles:
        doc, label = docs[(bundle, partition)]
        if doc is not None and doc.get("descriptor_pipeline"):
            return doc["descriptor_pipeline"], label, None
    return None, None, "NOT_EVALUATED (no alternative-bundle evaluation of this partition is recorded)"


def build_comparison(run: RunRoot, pcfg: dict[str, Any], sources: list[Path], out_dir: Path, *, min_clusters: int) -> dict[str, Any]:
    t4 = pcfg["table4"]
    bundles = pcfg["bundles"]
    pairing = pcfg["pairing"]
    partitions = [str(p) for p in t4["partitions"]]
    primary_seed = str(int(pcfg["f9b"]["seeds"][0]))
    targets = dict(t4["rho_targets"])
    docs: dict[tuple[str, str], tuple[dict | None, str]] = {}
    hashes: dict[str, str | None] = {}
    for bundle in bundles:
        for partition in partitions:
            path = run.path / "probes" / "evaluations" / bundle / f"{partition}.json"
            label = run.portable(path)
            if path in sources and path.is_file():
                docs[(bundle, partition)] = (json.loads(path.read_text(encoding="utf-8")), label)
                hashes[label] = hash_path(path)
            else:
                docs[(bundle, partition)] = (None, label)
    table = Table(TABLE4_NAME, "Three representations head to head: rho, rho_j and the paired predictor's metrics per partition",
                  COLUMNS, notes=["estimate [95 % paired cluster-bootstrap CI]; estimation only, no significance language",
                                  f"cells resting on fewer than {min_clusters} clusters are NE",
                                  "M&Ms class metrics: same-definition crosswalk subsets only; F9: primary seed (3-seed range in the Figure-5 data)"])
    figure: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for partition in partitions:
        evidence = str(pcfg["partitions"][partition]["label"])
        panel = f"{partition} ({evidence})"
        for rep in t4["rows"]:
            predictor = str(pairing[rep])
            recorded = {_doc_label(docs, bundles, "representations", rep, partition), _doc_label(docs, bundles, "predictors", predictor, partition)}
            row_evidence = "post hoc" if rep != "descriptors" and "post hoc" in recorded else evidence
            if rep == "descriptors":
                desc, src, why = _descriptor_doc(docs, bundles, partition)
                rep_entry = None if desc is None else desc["representation"]["rho"]
                pred_entry = None if desc is None else desc["predictor"]["metrics"]
                rep_src = pred_src = src
                rep_why = pred_why = why
                rep_paired = pred_paired = None
            else:
                r_entry, rep_src, rep_why = _locate(docs, bundles, "representations", rep, partition)
                seed_key = primary_seed if rep == "f9b" else "-1"
                rep_block = None if r_entry is None else r_entry.get("seeds", {}).get(seed_key)
                rep_entry = None if rep_block is None else rep_block.get("rho")
                rep_paired = None if rep_block is None else rep_block.get("paired_vs_descriptors")
                p_entry, pred_src, pred_why = _locate(docs, bundles, "predictors", predictor, partition)
                pseed = primary_seed if predictor == "f9" else "-1"
                pred_block = None if p_entry is None else p_entry.get("seeds", {}).get(pseed)
                pred_entry = None if pred_block is None else pred_block.get("metrics")
                pred_paired = None if pred_block is None else pred_block.get("paired_vs_audited_predictor")
                if p_entry is not None:
                    _figure_seeds(figure, p_entry, rep, predictor, partition, row_evidence, primary_seed, pred_src)
            if rep_entry is None:
                missing.append({"partition": partition, "representation": rep, "reason": rep_why or "not recorded"})
            if pred_entry is None:
                missing.append({"partition": partition, "predictor": predictor, "reason": pred_why or "not recorded"})
            row: dict[str, Any] = {"representation": LABELS.get(rep, rep), "predictor": LABELS.get(predictor, predictor), "evidence": row_evidence}
            n_rho = None if rep_entry is None else rep_entry.get("n_clusters")
            row["n_clusters_rho"] = Cell(n_rho, rep_src, "rho.n_clusters") if n_rho is not None else Cell(f"NE ({rep_why})", rep_src, "rho")
            if rep_entry is None:
                for col in ("rho", "rho_LVEF", "rho_LVEDV"):
                    row[col] = Cell(f"NE ({rep_why})", rep_src, "rho")
            else:
                row["rho"] = _cell(rep_entry.get("rho"), rep_entry.get("ci_lower"), rep_entry.get("ci_upper"), status=rep_entry.get("status"),
                                   reason=rep_entry.get("reason"), n_clusters=n_rho, min_clusters=min_clusters, path=rep_src, key="rho")
                for col, target in targets.items():
                    ci = (rep_entry.get("rho_j_ci") or {}).get(target) or (None, None)
                    row[col] = _cell((rep_entry.get("rho_j") or {}).get(target), ci[0], ci[1], status=rep_entry.get("status"),
                                     reason="target not estimable", n_clusters=n_rho, min_clusters=min_clusters, path=rep_src, key=f"rho_j.{target}")
                _figure_rho(figure, rep_entry, rep_paired, rep, predictor, partition, row_evidence, primary_seed if rep == "f9b" else None, rep_src, targets)
            n_cls = None if pred_entry is None else (pred_entry.get("counts") or {}).get("n_clusters")
            row["n_clusters_class"] = Cell(n_cls, pred_src, "counts.n_clusters") if n_cls is not None else Cell(f"NE ({pred_why})", pred_src, "metrics")
            for col, metric in PREDICTOR_COLUMNS.items():
                m = None if pred_entry is None else (pred_entry.get("metrics") or {}).get(metric)
                if m is None:
                    row[col] = Cell(f"NE ({pred_why or 'not recorded'})", pred_src, f"metrics.{metric}")
                else:
                    row[col] = _cell(m.get("estimate"), m.get("ci_lower"), m.get("ci_upper"), status=m.get("status"), reason=m.get("reason"),
                                     n_clusters=n_cls, min_clusters=min_clusters, path=pred_src, key=f"metrics.{metric}")
            if rep == "descriptors" and pred_entry is not None:
                _figure_predictor(figure, pred_entry, None, rep, predictor, partition, row_evidence, None, True, pred_src)
            elif pred_paired is not None and predictor != "f9":
                _figure_predictor(figure, pred_entry, pred_paired, rep, predictor, partition, row_evidence, None, True, pred_src)
            row["status"] = "EVALUATED" if rep_entry is not None and pred_entry is not None else "NE (partial)" if (rep_entry or pred_entry) else "NE"
            table.add(panel, **row)
    frame = pd.DataFrame(figure, columns=FIGURE_COLUMNS)
    csv = table.frame().to_csv(index=False, lineterminator="\n").encode("utf-8")
    tex = render_latex(table, run.run_id).encode("utf-8")
    provenance = {"schema": "tmmri.comparison_tables_provenance/1", "run_id": run.run_id, "min_clusters": int(min_clusters),
                  "sources": {k: v for k, v in sorted(hashes.items())}, "not_evaluated": missing,
                  "table": {"name": TABLE4_NAME, "rows": len(table.rows), "panels": table.panels, "columns": table.columns,
                            "column_sources": {c: table.provenance.get(c, []) for c in table.columns}, "notes": table.notes},
                  "figure05": {"file": FIGURE5_NAME, "rows": int(len(frame)), "columns": FIGURE_COLUMNS},
                  "licence": "aggregates only; no case identifiers"}
    split_path = run.path / "contracts" / "split_manifest.parquet"
    split = pd.read_parquet(split_path) if split_path.is_file() else None
    mms2_path = run.path / "contracts" / "split_manifest_MMS2.parquet"     # M&Ms-2 ids (MMS2_<code>) are guarded too
    mms2_split = pd.read_parquet(mms2_path) if mms2_path.is_file() else None
    assert_no_case_identifiers({f"{TABLE4_NAME}.csv": csv.decode("utf-8"), f"{TABLE4_NAME}.tex": tex.decode("utf-8"),
                                "provenance.json": json.dumps(provenance, sort_keys=True, default=str)}, case_identifiers([split, mms2_split]))
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(out_dir / f"{TABLE4_NAME}.csv", csv)
    atomic_write_bytes(out_dir / f"{TABLE4_NAME}.tex", tex)
    tmp = out_dir / f".{FIGURE5_NAME}.tmp"
    frame.to_parquet(tmp, engine="pyarrow", index=False)
    tmp.replace(out_dir / FIGURE5_NAME)
    provenance["sha256"] = {"csv": hash_path(out_dir / f"{TABLE4_NAME}.csv"), "tex": hash_path(out_dir / f"{TABLE4_NAME}.tex"),
                            "figure05": hash_path(out_dir / FIGURE5_NAME)}
    atomic_write_json(out_dir / "provenance.json", provenance)
    return {"rows": len(table.rows), "figure_rows": int(len(frame)), "not_evaluated": len(missing)}


def _figure_rho(rows: list[dict[str, Any]], entry: dict[str, Any], paired: dict[str, Any] | None, rep: str, predictor: str, partition: str,
                evidence: str, seed: str | None, source: str | None, targets: dict[str, str]) -> None:
    base = {"representation": rep, "predictor": predictor, "partition": partition, "evidence": evidence, "seed": int(seed) if seed else -1,
            "primary_seed": True, "source": source}
    rows.append({**base, "metric": "rho", "kind": "estimate", "estimate": entry.get("rho"), "ci_lower": entry.get("ci_lower"), "ci_upper": entry.get("ci_upper"),
                 "status": entry.get("status"), "reason": entry.get("reason"), "n_clusters": entry.get("n_clusters")})
    for target, value in (entry.get("rho_j") or {}).items():
        ci = (entry.get("rho_j_ci") or {}).get(target) or (None, None)
        rows.append({**base, "metric": f"rho_j:{target}", "kind": "estimate", "estimate": value, "ci_lower": ci[0], "ci_upper": ci[1],
                     "status": entry.get("status"), "reason": None, "n_clusters": entry.get("n_clusters")})
    if paired is not None:
        rows.append({**base, "metric": "rho", "kind": "difference_vs_descriptors", "estimate": paired.get("difference"), "ci_lower": paired.get("ci_lower"),
                     "ci_upper": paired.get("ci_upper"), "status": paired.get("status"), "reason": paired.get("reason"), "n_clusters": paired.get("n_clusters")})
        for target, value in (paired.get("rho_j") or {}).items():
            rows.append({**base, "metric": f"rho_j:{target}", "kind": "difference_vs_descriptors", "estimate": value.get("difference"),
                         "ci_lower": value.get("ci_lower"), "ci_upper": value.get("ci_upper"), "status": value.get("status"), "reason": value.get("reason"),
                         "n_clusters": paired.get("n_clusters")})


def _figure_predictor(rows: list[dict[str, Any]], summary: dict[str, Any] | None, paired: dict[str, Any] | None, rep: str, predictor: str,
                      partition: str, evidence: str, seed: str | None, primary: bool, source: str | None) -> None:
    base = {"representation": rep, "predictor": predictor, "partition": partition, "evidence": evidence, "seed": int(seed) if seed else -1,
            "primary_seed": bool(primary), "source": source}
    n_cl = (summary or {}).get("counts", {}).get("n_clusters")
    for metric, m in ((summary or {}).get("metrics") or {}).items():
        if not isinstance(m, dict):
            continue
        rows.append({**base, "metric": metric, "kind": "estimate", "estimate": m.get("estimate"), "ci_lower": m.get("ci_lower"), "ci_upper": m.get("ci_upper"),
                     "status": m.get("status"), "reason": m.get("reason"), "n_clusters": n_cl})
    for metric, m in ((paired or {}).get("metrics") or {}).items():
        rows.append({**base, "metric": metric, "kind": "difference_vs_descriptors", "estimate": m.get("difference"), "ci_lower": m.get("ci_lower"),
                     "ci_upper": m.get("ci_upper"), "status": m.get("status"), "reason": m.get("reason"), "n_clusters": (paired or {}).get("n_clusters")})


def _figure_seeds(rows: list[dict[str, Any]], entry: dict[str, Any], rep: str, predictor: str, partition: str, evidence: str, primary_seed: str,
                  source: str | None) -> None:
    if predictor != "f9":
        return
    for seed, block in sorted((entry.get("seeds") or {}).items()):
        _figure_predictor(rows, block.get("metrics"), block.get("paired_vs_audited_predictor"), rep, predictor, partition, evidence, seed,
                          seed == primary_seed, source)
