from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from ..ingest.geometry import select_frame
from ..io import resolve_path, utc_now_iso, write_table_pair
from ..ingest.labels import normalize_labels


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def select_visual_qc_cases(
    b_acdc: pd.DataFrame,
    b_mms: pd.DataFrame,
    max_total_panels: int = 120,
    random_seed: int = 20260611,
    random_cases_per_dataset: int = 10,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    frames: list[pd.DataFrame] = []
    for name, df in [("ACDC", b_acdc), ("MMS_100", b_mms)]:
        if df.empty:
            continue
        chosen: set[int] = set()
        manual_col = df.get("qc_manual_review_required", pd.Series([False] * len(df), index=df.index)).map(_as_bool)
        criticalish = df[manual_col].index.tolist()
        chosen.update(criticalish)
        by_label = []
        if "disease_label" in df.columns:
            for _, group in df.groupby("disease_label", dropna=False):
                if len(group):
                    by_label.append(int(group.index[0]))
        chosen.update(by_label)
        remaining = [int(i) for i in df.index if i not in chosen]
        take = min(random_cases_per_dataset, len(remaining))
        if take:
            chosen.update(int(i) for i in rng.choice(remaining, size=take, replace=False))
        subset = df.loc[sorted(chosen)].copy()
        subset["selection_reason"] = np.where(subset.index.isin(criticalish), "manual_review_or_qc_flag", "random_or_label_coverage")
        frames.append(subset)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if len(out) > max_total_panels:
        priority = out.get("qc_manual_review_required", pd.Series([False] * len(out))).map(_as_bool).astype(int)
        out = out.assign(_priority=priority)
        parts = []
        datasets = sorted(out["dataset_source"].astype(str).unique()) if "dataset_source" in out.columns else ["all"]
        base_quota = max(1, max_total_panels // max(len(datasets), 1))
        used_ids: set[str] = set()
        for ds in datasets:
            ds_df = out[out["dataset_source"].astype(str).eq(ds)] if "dataset_source" in out.columns else out
            chosen = ds_df.sort_values(["_priority", "sample_id"], ascending=[False, True]).head(base_quota)
            parts.append(chosen)
            used_ids.update(chosen["sample_id"].astype(str).tolist())
        selected = pd.concat(parts, ignore_index=True) if parts else out.head(0)
        remaining_slots = max_total_panels - len(selected)
        if remaining_slots > 0:
            rest = out[~out["sample_id"].astype(str).isin(used_ids)].sort_values(["_priority", "dataset_source", "sample_id"], ascending=[False, True, True]).head(remaining_slots)
            selected = pd.concat([selected, rest], ignore_index=True)
        out = selected.drop(columns=["_priority"], errors="ignore")
    return out.reset_index(drop=True)


def _resolve_existing_path(path_value: object) -> Path | None:
    if path_value is None or (isinstance(path_value, float) and pd.isna(path_value)):
        return None
    text = str(path_value)
    candidates = [Path(text)]
    # The archived artifacts may contain Windows absolute paths from the generating
    # workstation. Try reconstructing them under this project root.
    normalized = text.replace("\\", "/")
    marker = "/data/"
    if marker in normalized:
        rel = normalized.split(marker, 1)[1]
        candidates.append(resolve_path("data") / rel)
    candidates.append(resolve_path(text))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def foreground_slices(mask: np.ndarray) -> list[int]:
    if mask.ndim == 2:
        return [0] if np.any(mask) else []
    axes = tuple(range(mask.ndim - 1))
    return [int(i) for i in np.where(mask.any(axis=axes))[0]]


def select_basal_mid_apical(mask: np.ndarray) -> list[int]:
    slices = foreground_slices(mask)
    if not slices:
        if mask.ndim >= 3:
            return [0, mask.shape[2] // 2, max(mask.shape[2] - 1, 0)]
        return [0, 0, 0]
    basal = slices[0]
    mid = slices[len(slices) // 2]
    apical = slices[-1]
    return [basal, mid, apical]


def _metadata_panel(row: pd.Series, out_path: Path, reason: str) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)
    ax.axis("off")
    title = f"Visual QC metadata panel - {row.get('dataset_source')} {row.get('case_id')}"
    fields = [
        ("sample_id", row.get("sample_id")),
        ("split", row.get("split")),
        ("label", row.get("disease_label")),
        ("vendor", row.get("scanner_vendor")),
        ("center", row.get("center_id")),
        ("ED/ES frames", f"{row.get('ed_frame_index')} / {row.get('es_frame_index')}"),
        ("LV EDV/ESV/EF", f"{row.get('lv_edv_ml')} / {row.get('lv_esv_ml')} / {row.get('lv_ef_percent')}"),
        ("RV EDV/ESV/EF", f"{row.get('rv_edv_ml')} / {row.get('rv_esv_ml')} / {row.get('rv_ef_percent')}"),
        ("QC reasons", row.get("qc_failure_reason")),
        ("Image status", reason),
    ]
    text = title + "\n\n" + "\n".join(f"{k}: {v}" for k, v in fields)
    wrapped = "\n".join(textwrap.wrap(text, width=110, replace_whitespace=False, break_long_words=False))
    ax.text(0.03, 0.95, wrapped, ha="left", va="top", fontsize=10, family="monospace")
    ax.text(
        0.03,
        0.04,
        "Research use only. Not clinically validated for diagnosis, triage, treatment selection, or patient management.",
        ha="left",
        va="bottom",
        fontsize=9,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _overlay_panel(row: pd.Series, out_path: Path) -> str:
    # A fully image-based panel is implemented for environments where the raw NIfTI
    # files are present. The packaged project often ships only B matrices; in that case
    # generate a metadata panel instead and record the reason.
    dataset = str(row.get("dataset_source", "ACDC"))
    ed_image_path = _resolve_existing_path(row.get("raw_ed_image_path"))
    es_image_path = _resolve_existing_path(row.get("raw_es_image_path"))
    ed_mask_path = _resolve_existing_path(row.get("raw_ed_mask_path"))
    es_mask_path = _resolve_existing_path(row.get("raw_es_mask_path"))
    image_path = _resolve_existing_path(row.get("raw_image_path"))
    mask_path = _resolve_existing_path(row.get("raw_mask_path"))
    if not (all(p is not None for p in [ed_image_path, es_image_path, ed_mask_path, es_mask_path]) or (image_path is not None and mask_path is not None)):
        reason = "native_image_or_mask_unavailable_in_packaged_archive"
        _metadata_panel(row, out_path, reason)
        return reason
    try:
        import matplotlib.pyplot as plt
        import nibabel as nib
    except Exception as exc:
        reason = f"visual_dependencies_unavailable:{exc}"
        _metadata_panel(row, out_path, reason)
        return reason
    try:
        if all(p is not None for p in [ed_image_path, es_image_path, ed_mask_path, es_mask_path]):
            ed = int(row.get("ed_frame_index", 0))
            es = int(row.get("es_frame_index", ed))
            image_ed_full = np.asanyarray(nib.load(str(ed_image_path)).dataobj)
            image_es_full = np.asanyarray(nib.load(str(es_image_path)).dataobj)
            mask_ed_full = np.rint(np.asanyarray(nib.load(str(ed_mask_path)).dataobj)).astype(int)
            mask_es_full = np.rint(np.asanyarray(nib.load(str(es_mask_path)).dataobj)).astype(int)
            image_ed = select_frame(image_ed_full, ed) if image_ed_full.ndim == 4 else image_ed_full
            image_es = select_frame(image_es_full, es) if image_es_full.ndim == 4 else image_es_full
            mask_ed_raw = select_frame(mask_ed_full, ed) if mask_ed_full.ndim == 4 else mask_ed_full
            mask_es_raw = select_frame(mask_es_full, es) if mask_es_full.ndim == 4 else mask_es_full
            mask_ed, unexpected_ed = normalize_labels(mask_ed_raw, dataset)
            mask_es, unexpected_es = normalize_labels(mask_es_raw, dataset)
            if unexpected_ed or unexpected_es:
                raise ValueError(f"unexpected labels: {sorted(unexpected_ed.union(unexpected_es))}")
        else:
            image = np.asanyarray(nib.load(str(image_path)).dataobj)
            mask = np.asanyarray(nib.load(str(mask_path)).dataobj)
            ed = int(row.get("ed_frame_index", 0))
            es = int(row.get("es_frame_index", ed))
            image_ed = select_frame(image, ed) if image.ndim == 4 else image
            image_es = select_frame(image, es) if image.ndim == 4 else image
            mask_ed_raw = select_frame(mask, ed) if mask.ndim == 4 else mask
            mask_es_raw = select_frame(mask, es) if mask.ndim == 4 else mask
            mask_ed, unexpected_ed = normalize_labels(np.rint(mask_ed_raw).astype(int), dataset)
            mask_es, unexpected_es = normalize_labels(np.rint(mask_es_raw).astype(int), dataset)
            if unexpected_ed or unexpected_es:
                raise ValueError(f"unexpected labels: {sorted(unexpected_ed.union(unexpected_es))}")
        slices = select_basal_mid_apical((mask_ed > 0) | (mask_es > 0))
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        for r, (phase, img, msk) in enumerate([("ED", image_ed, mask_ed), ("ES", image_es, mask_es)]):
            for c, z in enumerate(slices):
                ax = axes[r, c]
                z = int(np.clip(z, 0, img.shape[2] - 1)) if img.ndim == 3 else 0
                plane = img[:, :, z] if img.ndim == 3 else img
                mplane = msk[:, :, z] if msk.ndim == 3 else msk
                lo, hi = np.nanpercentile(plane, [1, 99])
                if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                    lo, hi = float(np.nanmin(plane)), float(np.nanmax(plane) + 1.0)
                ax.imshow(plane.T, cmap="gray", origin="lower", vmin=lo, vmax=hi)
                overlay = np.ma.masked_where(mplane.T == 0, mplane.T)
                ax.imshow(overlay, cmap="tab10", origin="lower", alpha=0.35, vmin=0, vmax=3)
                ax.set_title(f"{phase} z={z}")
                ax.axis("off")
        fig.suptitle(f"{row.get('dataset_source')} {row.get('case_id')} | {row.get('split')} | {row.get('disease_label')} | QC: {row.get('qc_failure_reason')}", fontsize=10)
        fig.text(0.02, 0.02, "Research use only. Not clinically validated.", fontsize=8)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        return "native_overlay_panel_generated"
    except Exception as exc:
        reason = f"native_panel_generation_failed:{type(exc).__name__}:{exc}"
        _metadata_panel(row, out_path, reason)
        return reason


def generate_visual_qc_panels(
    b_acdc: pd.DataFrame,
    b_mms: pd.DataFrame,
    output_dir: str | Path = "artifacts/reports/qc/panels",
    max_total_panels: int = 120,
    random_seed: int = 20260611,
    random_cases_per_dataset: int = 10,
    suffix: str = "v3",
) -> tuple[pd.DataFrame, list[Path]]:
    selected = select_visual_qc_cases(b_acdc, b_mms, max_total_panels, random_seed, random_cases_per_dataset)
    rows: list[dict[str, object]] = []
    out_dir = resolve_path(output_dir)
    for _, row in selected.iterrows():
        dataset = str(row.get("dataset_source", "dataset"))
        case_id = str(row.get("case_id", row.get("sample_id", "case"))).replace("/", "_").replace("\\", "_")
        out_path = out_dir / f"{dataset}_{case_id}.png"
        status = _overlay_panel(row, out_path)
        rows.append(
            {
                "sample_id": row.get("sample_id"),
                "case_id": row.get("case_id"),
                "patient_id": row.get("patient_id"),
                "dataset_source": row.get("dataset_source"),
                "split": row.get("split"),
                "panel_path": str(out_path),
                "selection_reason": row.get("selection_reason"),
                "panel_status": status,
                "qc_manual_review_required": row.get("qc_manual_review_required"),
                "qc_failure_reason": row.get("qc_failure_reason"),
                "lv_edv_ml": row.get("lv_edv_ml"),
                "rv_edv_ml": row.get("rv_edv_ml"),
                "lv_ef_percent": row.get("lv_ef_percent"),
                "rv_ef_percent": row.get("rv_ef_percent"),
                "generated_at_utc": utc_now_iso(),
            }
        )
    manifest = pd.DataFrame(rows)
    artifacts: list[Path] = []
    if len(manifest):
        artifacts.extend(write_table_pair(manifest, f"artifacts/reports/qc/visual_qc_manifest_{suffix}.parquet", f"artifacts/reports/qc/visual_qc_manifest_{suffix}.csv"))
    else:
        artifacts.extend(write_table_pair(manifest, f"artifacts/reports/qc/visual_qc_manifest_{suffix}.parquet", f"artifacts/reports/qc/visual_qc_manifest_{suffix}.csv"))
    md = resolve_path("artifacts/reports/qc") / f"visual_qc_summary_{suffix}.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    status_counts = manifest["panel_status"].value_counts().to_dict() if len(manifest) else {}
    lines = [
        "# Visual QC Panel Summary",
        "",
        "Clinical status: research use only. This project is not clinically validated for diagnosis, triage, treatment selection, or patient management.",
        "",
        f"Panels requested/generated: {len(manifest)}",
        "",
        "The panel generator creates native image/mask overlays when NIfTI inputs are available. When the packaged archive contains only derived feature matrices, it creates metadata-only QC PNGs and records that limitation in the manifest.",
        "",
        "## Panel status counts",
        "",
    ]
    if status_counts:
        for status, count in status_counts.items():
            lines.append(f"- {status}: {int(count)}")
    else:
        lines.append("- none: 0")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifacts.append(md)
    artifacts.extend(Path(p).resolve() if Path(p).is_absolute() else Path(p) for p in manifest.get("panel_path", pd.Series(dtype=str)).tolist())
    return manifest, artifacts


# ===============================================================================================================
# Protocol 6.0.0 QC panels (SPECIFICATION.md 12.4; configs/cohorts.yaml qc_panels).  Selection is deterministic:
# every critical or high QC case of a partition plus ``random_per_partition`` further cases ranked by
# SHA256(random_selection_salt + sample_id).  Panels render the ED and ES mid-ventricular slices of the staged case
# with the mask overlay and registry values.  Automated QC is computational triage, not expert adjudication.
# ===============================================================================================================
PANEL_COLUMNS = ["sample_id", "case_id", "partition", "selection_reason", "max_severity", "selection_digest", "selection_rank"]


def select_qc_panels_v6(table: pd.DataFrame, cohorts: dict, reason_codes: dict | None, partitions: list[str]) -> pd.DataFrame:
    """The panel cases of each partition (``split`` column of the typed B table); independent of row order."""
    import hashlib

    from .qc import assess_qc_v6, parse_reason_tokens

    cfg = cohorts["qc_panels"]
    n_random = int(cfg["random_per_partition"])
    salt = str(cfg["random_selection_salt"])
    include = bool(cfg.get("include_all_high_and_critical", True))
    rows = []
    for part in partitions:
        sub = table[table["split"].astype(str) == str(part)]
        flagged, rest = [], []
        for _, r in sub.iterrows():
            sid = str(r["sample_id"])
            severity = assess_qc_v6(parse_reason_tokens(r.get("qc_failure_reason")), cohorts, reason_codes).max_severity
            digest = hashlib.sha256((salt + sid).encode("utf-8")).hexdigest()
            entry = {"sample_id": sid, "case_id": r.get("case_id"), "partition": str(part), "max_severity": severity, "selection_digest": digest}
            (flagged if include and severity in ("critical", "high") else rest).append(entry)
        for e in flagged:
            rows.append({**e, "selection_reason": f"qc_{e['max_severity']}", "selection_rank": None})
        for rank, e in enumerate(sorted(rest, key=lambda e: (e["selection_digest"], e["sample_id"]))[:n_random]):
            rows.append({**e, "selection_reason": "hash_random", "selection_rank": rank})
    out = pd.DataFrame(rows, columns=PANEL_COLUMNS)
    return out.sort_values(["partition", "sample_id"], kind="mergesort").reset_index(drop=True)


def _mid_slice(mask: np.ndarray) -> int:
    slices = foreground_slices(mask > 0) if mask.ndim == 3 else []
    return int(slices[len(slices) // 2]) if slices else (mask.shape[2] // 2 if mask.ndim == 3 else 0)


def render_qc_panel_v6(case, row: dict, out_path: str | Path, value_columns: list[str]) -> Path:
    """ED and ES mid-ventricular short-axis slices with mask overlays and the registry values of ``value_columns``.

    Uses the object-oriented Agg canvas (no pyplot state, no display) and strips the software metadata, so the PNG
    bytes depend only on the inputs and the matplotlib rendering."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(9.0, 4.8), dpi=100)
    for i, (phase, image, mask) in enumerate((("ED", case.ed_image, case.ed_mask), ("ES", case.es_image, case.es_mask))):
        ax = fig.add_subplot(1, 3, i + 1)
        z = _mid_slice(np.asarray(mask))
        plane = np.asarray(image)[:, :, z] if np.asarray(image).ndim == 3 else np.asarray(image)
        mplane = np.asarray(mask)[:, :, z] if np.asarray(mask).ndim == 3 else np.asarray(mask)
        lo, hi = np.nanpercentile(plane, [1, 99]) if np.isfinite(plane).any() else (0.0, 1.0)
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            lo, hi = float(np.nanmin(plane)), float(np.nanmin(plane)) + 1.0
        ax.imshow(plane.T, cmap="gray", origin="lower", vmin=lo, vmax=hi)
        ax.imshow(np.ma.masked_where(mplane.T == 0, mplane.T), cmap="tab10", origin="lower", alpha=0.35, vmin=0, vmax=3)
        ax.set_title(f"{phase} z={z}", fontsize=9)
        ax.axis("off")
    ax = fig.add_subplot(1, 3, 3)
    ax.axis("off")

    def fmt(v: object) -> str:
        try:
            return "NA" if v is None or not np.isfinite(float(v)) else f"{float(v):.1f}"
        except (TypeError, ValueError):
            return str(v)

    lines = [f"{row.get('sample_id')}", f"partition: {row.get('split')}", f"geometry: {getattr(case, 'geometry_method', '')}", ""]
    lines += [f"{c}: {fmt(row.get(c))} [{row.get(c + '__state', '')}]" for c in value_columns]
    lines += ["", "QC: " + "\n    ".join(textwrap.wrap(str(row.get("qc_failure_reason") or "none"), width=34))]
    ax.text(0.0, 1.0, "\n".join(lines), ha="left", va="top", fontsize=7, family="monospace", transform=ax.transAxes)
    fig.text(0.01, 0.01, "Research use only. Computational triage, not expert adjudication.", fontsize=7)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="png", metadata={"Software": None})
    return out
