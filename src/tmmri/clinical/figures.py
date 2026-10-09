from __future__ import annotations

import html
import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from ..formal_features import _resolve_existing_path
from ..ingest.geometry import select_frame
from ..io import resolve_path, utc_now_iso, write_table_pair
from ..ingest.labels import normalize_labels


MM_PER_INCH = 25.4
DOUBLE_COLUMN_MM = 183.0
SINGLE_COLUMN_MM = 89.0
NATURE_PALETTE = {
    "navy": "#16324F",
    "blue": "#2F6F9F",
    "cyan": "#52B5D2",
    "teal": "#238A8D",
    "green": "#3B8F5A",
    "amber": "#D89B2B",
    "red": "#C94B40",
    "purple": "#6D5AA6",
    "grey": "#6B7280",
    "light_grey": "#E8EDF2",
    "near_white": "#F7F9FB",
    "black": "#1F2933",
}


@dataclass(frozen=True)
class ImageTransformRecord:
    reader_case_id: str
    dataset_source: str
    asset_type: str
    output_path: str
    source_image_paths: str
    source_mask_paths: str
    intensity_transform: str
    crop_policy: str
    rotation_policy: str
    slice_selection: str
    overlay_policy: str
    pseudonymization: str
    annotation_policy: str
    generated_at_utc: str
    status: str
    error: str


def configure_nature_style() -> None:
    """Apply a compact Nature-compatible plotting style without embedded titles."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _figsize(width_mm: float, height_mm: float) -> tuple[float, float]:
    return width_mm / MM_PER_INCH, height_mm / MM_PER_INCH


def _clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def export_nature_figure(
    fig: plt.Figure,
    output_stem: str | Path,
    *,
    dpi: int = 600,
) -> list[Path]:
    """Export editable and raster deliverables from the same Python figure."""
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix, kwargs in (
        (".svg", {}),
        (".pdf", {}),
        (".tiff", {"dpi": dpi, "pil_kwargs": {"compression": "tiff_lzw"}}),
        (".png", {"dpi": dpi}),
    ):
        path = stem.with_suffix(suffix)
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def _existing_path(value: object) -> Path | None:
    # Restricted manifests record source-machine absolute paths; re-root them
    # at ``/data/`` exactly as Matrix A extraction does so a relocated or
    # non-Windows checkout still finds the same raw files.
    return _resolve_existing_path(value)


def _load_nifti(path: Path) -> np.ndarray:
    import nibabel as nib

    return np.asanyarray(nib.load(str(path)).dataobj)


def _load_phase_arrays(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Path], list[Path]]:
    image_paths = [_existing_path(row.get("raw_ed_image_path")), _existing_path(row.get("raw_es_image_path"))]
    mask_paths = [_existing_path(row.get("raw_ed_mask_path")), _existing_path(row.get("raw_es_mask_path"))]
    if any(path is None for path in image_paths + mask_paths):
        raise FileNotFoundError("ED/ES native image or mask is unavailable")
    ed_frame = int(row.get("ed_frame_index", 0) or 0)
    es_frame = int(row.get("es_frame_index", ed_frame) or ed_frame)
    image_ed_raw = _load_nifti(image_paths[0])
    image_es_raw = _load_nifti(image_paths[1])
    mask_ed_raw = _load_nifti(mask_paths[0])
    mask_es_raw = _load_nifti(mask_paths[1])
    image_ed = select_frame(image_ed_raw, ed_frame) if image_ed_raw.ndim == 4 else image_ed_raw
    image_es = select_frame(image_es_raw, es_frame) if image_es_raw.ndim == 4 else image_es_raw
    mask_ed_frame = select_frame(mask_ed_raw, ed_frame) if mask_ed_raw.ndim == 4 else mask_ed_raw
    mask_es_frame = select_frame(mask_es_raw, es_frame) if mask_es_raw.ndim == 4 else mask_es_raw
    dataset = str(row.get("dataset_source", "ACDC"))
    mask_ed, unexpected_ed = normalize_labels(np.rint(mask_ed_frame).astype(int), dataset)
    mask_es, unexpected_es = normalize_labels(np.rint(mask_es_frame).astype(int), dataset)
    unexpected = unexpected_ed.union(unexpected_es)
    if unexpected:
        raise ValueError(f"unexpected segmentation labels: {sorted(unexpected)}")
    return image_ed, image_es, mask_ed, mask_es, [p for p in image_paths if p], [p for p in mask_paths if p]


def _global_limits(arrays: Sequence[np.ndarray]) -> tuple[float, float]:
    finite_parts: list[np.ndarray] = []
    for array in arrays:
        flattened = np.asarray(array).reshape(-1)
        stride = max(1, int(math.ceil(flattened.size / 1_000_000)))
        sampled = flattened[::stride].astype(float, copy=False)
        finite_parts.append(sampled[np.isfinite(sampled)])
    finite_parts = [part for part in finite_parts if part.size]
    if not finite_parts:
        return 0.0, 1.0
    values = np.concatenate(finite_parts)
    lo, hi = np.nanpercentile(values, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def _foreground_slices(mask: np.ndarray) -> list[int]:
    if mask.ndim == 2:
        return [0] if np.any(mask) else []
    axes = tuple(range(mask.ndim - 1))
    return [int(i) for i in np.where(mask.any(axis=axes))[0]]


def _basal_mid_apical(mask: np.ndarray) -> list[int]:
    slices = _foreground_slices(mask)
    if slices:
        return [slices[0], slices[len(slices) // 2], slices[-1]]
    depth = int(mask.shape[2]) if mask.ndim >= 3 else 1
    return [0, max(depth // 2, 0), max(depth - 1, 0)]


def _plane(volume: np.ndarray, z_index: int) -> np.ndarray:
    if volume.ndim == 2:
        return volume
    z = int(np.clip(z_index, 0, volume.shape[2] - 1))
    return volume[:, :, z]


def _display_plane(ax: plt.Axes, image: np.ndarray, lo: float, hi: float, mask: np.ndarray | None = None) -> None:
    ax.imshow(image.T, cmap="gray", origin="lower", vmin=lo, vmax=hi, interpolation="nearest")
    if mask is not None:
        colours = matplotlib.colors.ListedColormap(["#000000", "#50C7C7", "#E66AA2", "#F0C44E"])
        overlay = np.ma.masked_where(mask.T == 0, mask.T)
        ax.imshow(overlay, cmap=colours, origin="lower", vmin=0, vmax=3, alpha=0.38, interpolation="nearest")
    ax.set_axis_off()


def render_case_montage(
    row: pd.Series,
    output_path: str | Path,
    *,
    assisted: bool,
    dpi: int = 220,
) -> ImageTransformRecord:
    """Render a label-free 2 x 3 ED/ES montage with no in-pixel text."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    reader_case_id = str(row.get("reader_case_id", "UNKNOWN"))
    dataset = str(row.get("dataset_source", "UNKNOWN"))
    image_sources: list[Path] = []
    mask_sources: list[Path] = []
    status = "generated"
    error = ""
    try:
        image_ed, image_es, mask_ed, mask_es, image_sources, mask_sources = _load_phase_arrays(row)
        union_mask = (mask_ed > 0) | (mask_es > 0)
        slices = _basal_mid_apical(union_mask)
        lo, hi = _global_limits([image_ed, image_es])
        fig, axes = plt.subplots(2, 3, figsize=_figsize(89, 54), facecolor="black")
        for phase_index, (image_volume, mask_volume) in enumerate(((image_ed, mask_ed), (image_es, mask_es))):
            for slice_index, z_index in enumerate(slices):
                _display_plane(
                    axes[phase_index, slice_index],
                    _plane(image_volume, z_index),
                    lo,
                    hi,
                    _plane(mask_volume, z_index) if assisted else None,
                )
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.015, hspace=0.015)
        fig.savefig(
            output,
            dpi=dpi,
            facecolor="black",
            edgecolor="none",
            bbox_inches=None,
            pad_inches=0,
            metadata={"Title": "", "Author": "", "Subject": "label-free CMR review asset"},
        )
        plt.close(fig)
    except Exception as exc:  # fail visibly in the manifest; never synthesize a metadata panel
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    return ImageTransformRecord(
        reader_case_id=reader_case_id,
        dataset_source=dataset,
        asset_type="assisted_segmentation_overlay" if assisted else "unassisted_raw_cine_montage",
        output_path=str(output),
        source_image_paths=";".join(str(path) for path in image_sources),
        source_mask_paths=";".join(str(path) for path in mask_sources) if assisted else "withheld",
        intensity_transform="case-level pooled ED/ES deterministic whole-volume sample, 1st-99th percentile clipping; fixed across all displayed slices",
        crop_policy="none",
        rotation_policy="transpose for display; origin lower; identical for every tile",
        slice_selection="deterministic basal/mid/apical from union of ED and ES foreground masks",
        overlay_policy="fixed label colours and alpha=0.38" if assisted else "none; segmentation withheld",
        pseudonymization="reader_case_id used only in filename; no raw identifier written into pixels",
        annotation_policy="no title, diagnosis, route, QC reason, phase label, slice label, or panel letter in pixels",
        generated_at_utc=utc_now_iso(),
        status=status,
        error=error,
    )


def _resolve_cine_path(row: pd.Series) -> Path | None:
    source = _existing_path(row.get("raw_image_path"))
    if source is not None:
        try:
            import nibabel as nib

            if len(nib.load(str(source)).shape) == 4:
                return source
        except Exception:
            pass
    if source is not None and str(row.get("dataset_source")) == "ACDC":
        case_id = str(row.get("case_id", ""))
        candidate = source.parent / f"{case_id}_4d.nii.gz"
        if candidate.exists():
            return candidate
    return None


def render_cine_viewer(
    row: pd.Series,
    output_dir: str | Path,
    *,
    frame_count: int = 8,
) -> ImageTransformRecord:
    """Create a self-contained label-free HTML cine viewer and lossless frames."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reader_case_id = str(row.get("reader_case_id", "UNKNOWN"))
    dataset = str(row.get("dataset_source", "UNKNOWN"))
    source = _resolve_cine_path(row)
    status = "generated"
    error = ""
    generated_frames: list[str] = []
    if source is None:
        status = "unavailable"
        error = "native four-dimensional cine volume unavailable"
    else:
        try:
            import nibabel as nib

            for stale_frame in output.glob("frame_*.png"):
                stale_frame.unlink()
            cine_image = nib.load(str(source))
            cine_shape = cine_image.shape
            if len(cine_shape) != 4:
                raise ValueError(f"expected 4D cine, received shape {cine_shape}")
            _, _, mask_ed, mask_es, _, _ = _load_phase_arrays(row)
            z_index = _basal_mid_apical((mask_ed > 0) | (mask_es > 0))[1]
            total = int(np.prod(cine_shape))
            spatial_stride = max(1, int(math.ceil((total / 1_000_000) ** (1 / 3))))
            sampled_cine = np.asanyarray(
                cine_image.dataobj[::spatial_stride, ::spatial_stride, ::spatial_stride, :]
            )
            lo, hi = _global_limits([sampled_cine])
            indices = np.linspace(0, cine_shape[3] - 1, num=min(frame_count, cine_shape[3]), dtype=int)
            display_stride = max(1, int(math.ceil(max(cine_shape[0], cine_shape[1]) / 320)))
            for output_index, frame_index in enumerate(indices):
                plane = np.asanyarray(
                    cine_image.dataobj[::display_stride, ::display_stride, z_index, int(frame_index)]
                ).T
                frame_path = output / f"frame_{output_index:02d}.png"
                mpimg.imsave(frame_path, plane, cmap="gray", vmin=lo, vmax=hi, origin="lower")
                generated_frames.append(frame_path.name)
            frame_json = json.dumps(generated_frames)
            document = f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<style>body{{margin:0;background:#111;color:#eee;font:14px Arial,sans-serif;display:grid;place-items:center;min-height:100vh}}main{{width:min(94vw,720px)}}img{{display:block;width:100%;height:auto;background:#000}}input{{width:100%;margin-top:12px}}p{{line-height:1.4}}</style></head>
<body><main data-reader-case-id=\"{html.escape(reader_case_id)}\"><img id=\"cine\" alt=\"Pseudonymous short-axis cine frame\"><input id=\"frame\" type=\"range\" min=\"0\" max=\"{max(len(generated_frames)-1,0)}\" value=\"0\" step=\"1\"><p>Research use only. Raw cine display; no benchmark label or system output is included.</p></main>
<script>const frames={frame_json};const image=document.getElementById('cine');const slider=document.getElementById('frame');function show(){{image.src=frames[Number(slider.value)]}}slider.addEventListener('input',show);show();let i=0;setInterval(()=>{{if(!slider.matches(':active')){{i=(i+1)%frames.length;slider.value=i;show()}}}},160);</script></body></html>"""
            (output / "index.html").write_text(document, encoding="utf-8")
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
    return ImageTransformRecord(
        reader_case_id=reader_case_id,
        dataset_source=dataset,
        asset_type="unassisted_raw_cine_viewer",
        output_path=str(output / "index.html"),
        source_image_paths=str(source or ""),
        source_mask_paths="used only for deterministic mid-ventricular slice selection; not displayed",
        intensity_transform="case-level deterministic whole-cine sample, 1st-99th percentile clipping; fixed across all frames",
        crop_policy="none",
        rotation_policy="transpose for display; origin lower; identical for every frame",
        slice_selection="deterministic mid-ventricular slice from union of ED and ES foreground masks",
        overlay_policy="none",
        pseudonymization="reader_case_id in HTML data attribute; raw identifiers absent",
        annotation_policy="no diagnosis, route, QC reason, phase label, slice label, title, or panel letter in pixels",
        generated_at_utc=utc_now_iso(),
        status=status,
        error=error,
    )


def generate_reader_image_assets(
    manifest: pd.DataFrame,
    output_root: str | Path = "clinical_validation/figures/reader_assets",
) -> tuple[pd.DataFrame, list[Path]]:
    """Generate complete unassisted/assisted montages and cine viewers."""
    configure_nature_style()
    root = Path(output_root)
    records: list[ImageTransformRecord] = []
    outputs: list[Path] = []
    required = {"reader_case_id", "dataset_source"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"reader image manifest is missing columns: {sorted(missing)}")
    if manifest["reader_case_id"].duplicated().any():
        raise ValueError("reader_case_id must be unique before image generation")
    for _, row in manifest.sort_values("reader_case_id").iterrows():
        case_id = str(row["reader_case_id"])
        unassisted_path = root / "unassisted" / f"{case_id}.png"
        assisted_path = root / "assisted" / f"{case_id}.png"
        records.append(render_case_montage(row, unassisted_path, assisted=False))
        records.append(render_case_montage(row, assisted_path, assisted=True))
        records.append(render_cine_viewer(row, root / "cine" / case_id))
        outputs.extend([unassisted_path, assisted_path, root / "cine" / case_id / "index.html"])
    integrity = pd.DataFrame([record.__dict__ for record in records])
    manifest_paths = write_table_pair(
        integrity,
        root / "image_integrity_manifest.parquet",
        root / "image_integrity_manifest.csv",
    )
    outputs.extend(manifest_paths)
    failed = integrity[integrity["status"].eq("failed")]
    if len(failed):
        summary = failed[["reader_case_id", "asset_type", "error"]].to_dict("records")[:10]
        raise RuntimeError(f"reader image generation failed for {len(failed)} assets: {summary}")
    return integrity, outputs


def generate_retest_image_assets(
    restricted_retest_manifest: pd.DataFrame,
    output_root: str | Path = "clinical_validation/figures/reader_assets",
) -> tuple[pd.DataFrame, list[Path]]:
    """Generate retest-only assets without exposing the original pseudonym linkage."""
    configure_nature_style()
    required = {"retest_case_id", "raw_ed_image_path", "raw_es_image_path", "raw_ed_mask_path", "raw_es_mask_path"}
    missing = required.difference(restricted_retest_manifest.columns)
    if missing:
        raise ValueError(f"restricted retest manifest is missing columns: {sorted(missing)}")
    if restricted_retest_manifest["retest_case_id"].duplicated().any():
        raise ValueError("retest_case_id must be unique before image generation")
    root = Path(output_root)
    records: list[ImageTransformRecord] = []
    outputs: list[Path] = []
    for _, source_row in restricted_retest_manifest.sort_values("retest_case_id").iterrows():
        row = source_row.copy()
        retest_id = str(row["retest_case_id"])
        row["reader_case_id"] = retest_id
        montage_path = root / "retest" / f"{retest_id}.png"
        records.append(render_case_montage(row, montage_path, assisted=False))
        records.append(render_cine_viewer(row, root / "retest_cine" / retest_id))
        outputs.extend([montage_path, root / "retest_cine" / retest_id / "index.html"])
    integrity = pd.DataFrame([record.__dict__ for record in records])
    manifest_paths = write_table_pair(
        integrity,
        root / "retest_image_integrity_manifest.parquet",
        root / "retest_image_integrity_manifest.csv",
    )
    outputs.extend(manifest_paths)
    failed = integrity[integrity["status"].eq("failed")]
    if len(failed):
        summary = failed[["reader_case_id", "asset_type", "error"]].to_dict("records")[:10]
        raise RuntimeError(f"retest image generation failed for {len(failed)} assets: {summary}")
    return integrity, outputs


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str = "none",
    textcolor: str = "white",
    fontsize: float = 6.6,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.7,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", color=textcolor, fontsize=fontsize)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color: str | None = None) -> None:
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8, linewidth=0.8, color=color or NATURE_PALETTE["grey"])
    ax.add_patch(arrow)


def build_architecture_figure(output_stem: str | Path) -> list[Path]:
    configure_nature_style()
    fig, ax = plt.subplots(figsize=_figsize(DOUBLE_COLUMN_MM, 68))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    labels = [
        ("Cine CMR\n+ masks", NATURE_PALETTE["navy"]),
        ("Formal\nfeatures A", NATURE_PALETTE["blue"]),
        ("Measured\nsemantics B", NATURE_PALETTE["teal"]),
        ("Transition\nreconstruction B̂", NATURE_PALETTE["purple"]),
        ("Strict rule\ntrace", NATURE_PALETTE["amber"]),
    ]
    x_positions = np.linspace(0.02, 0.78, len(labels))
    for index, ((label, colour), x_value) in enumerate(zip(labels, x_positions)):
        _box(ax, (float(x_value), 0.58), 0.16, 0.22, label, facecolor=colour)
        if index:
            _arrow(ax, (float(x_positions[index - 1] + 0.16), 0.69), (float(x_value), 0.69))
    route_labels = [
        ("Supported\nexplanation", NATURE_PALETTE["green"]),
        ("Conflict / low fidelity\nmanual review", NATURE_PALETTE["amber"]),
        ("Unsupported / off ontology\nreview or abstention", NATURE_PALETTE["red"]),
    ]
    for idx, (label, colour) in enumerate(route_labels):
        x_value = 0.08 + idx * 0.31
        _box(ax, (x_value, 0.15), 0.25, 0.18, label, facecolor=colour)
        _arrow(ax, (0.86, 0.58), (x_value + 0.125, 0.33), color=colour)
    ax.text(0.5, 0.045, "One physician: blinded assessment → frozen responses → assisted audit", ha="center", va="center", color=NATURE_PALETTE["black"], fontsize=7.2)
    return export_nature_figure(fig, output_stem)


def build_cohort_flow_figure(
    manifest: pd.DataFrame,
    output_stem: str | Path,
    *,
    acdc_split_counts: Mapping[str, int] | None = None,
    detailed_measurement_cases: int = 40,
    retest_cases: int = 24,
) -> list[Path]:
    configure_nature_style()
    required = {"dataset_source", "primary_reader_eligible", "reader_case_id"}
    if missing := required.difference(manifest.columns):
        raise ValueError(f"cohort manifest missing: {sorted(missing)}")
    primary = manifest[manifest["primary_reader_eligible"].astype(bool)].copy()
    counts = primary["dataset_source"].value_counts().to_dict()
    if len(primary) != 150 or counts.get("ACDC", 0) != 50 or counts.get("MMS_100", 0) != 100:
        raise ValueError(f"unexpected primary cohort counts: total={len(primary)}, by_dataset={counts}")
    split_counts = dict(acdc_split_counts or {"train": 85, "val": 15, "test": 50})
    if sum(int(value) for value in split_counts.values()) != 150 or int(split_counts.get("test", 0)) != 50:
        raise ValueError(f"unexpected ACDC split counts: {split_counts}")
    fig, ax = plt.subplots(figsize=_figsize(DOUBLE_COLUMN_MM, 92))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _box(ax, (0.02, 0.72), 0.23, 0.18, "ACDC inventory\n150 patients", facecolor=NATURE_PALETTE["navy"])
    _box(ax, (0.29, 0.72), 0.23, 0.18, "Fixed M&Ms subset\n100 patients", facecolor=NATURE_PALETTE["blue"])
    _box(
        ax,
        (0.02, 0.43),
        0.23,
        0.18,
        f"Patient-level split\n{split_counts['train']} train / {split_counts['val']} validation\n{split_counts['test']} held-out test",
        facecolor=NATURE_PALETTE["teal"],
        fontsize=6.5,
    )
    _box(ax, (0.29, 0.43), 0.23, 0.18, "External M&Ms\n100 patients", facecolor=NATURE_PALETTE["purple"])
    _box(ax, (0.60, 0.55), 0.36, 0.22, "Primary reader cohort\n50 ACDC test + 100 M&Ms\n150 unique cases; all routes retained", facecolor=NATURE_PALETTE["green"], fontsize=6.7)
    _box(ax, (0.60, 0.20), 0.17, 0.16, f"Detailed audit\n{int(detailed_measurement_cases)} cases", facecolor=NATURE_PALETTE["cyan"], fontsize=6.5)
    _box(ax, (0.79, 0.20), 0.17, 0.16, f"Hidden retest\n{int(retest_cases)} duplicates", facecolor=NATURE_PALETTE["amber"], fontsize=6.5)
    _arrow(ax, (0.135, 0.72), (0.135, 0.61))
    _arrow(ax, (0.405, 0.72), (0.405, 0.61))
    _arrow(ax, (0.25, 0.52), (0.60, 0.64))
    _arrow(ax, (0.52, 0.52), (0.60, 0.64))
    _arrow(ax, (0.72, 0.55), (0.685, 0.36))
    _arrow(ax, (0.84, 0.55), (0.875, 0.36))
    ax.text(0.39, 0.10, "Pilot: 10 development-only cases, excluded from primary endpoints", ha="center", va="center", fontsize=6.5, color=NATURE_PALETTE["black"])
    return export_nature_figure(fig, output_stem)


def build_interface_preview_figure(
    manifest: pd.DataFrame,
    montage_root: str | Path,
    output_stem: str | Path,
) -> list[Path]:
    configure_nature_style()
    candidates = manifest[manifest.get("system_route", pd.Series(index=manifest.index, dtype=str)).isin(["conflict_review", "supported_rule_with_caveat", "exact_supported_rule"])]
    if candidates.empty:
        candidates = manifest
    row = candidates.sort_values("reader_case_id").iloc[0]
    montage_path = Path(montage_root) / "assisted" / f"{row['reader_case_id']}.png"
    if not montage_path.exists():
        raise FileNotFoundError(montage_path)
    image = mpimg.imread(montage_path)
    fig = plt.figure(figsize=_figsize(DOUBLE_COLUMN_MM, 94))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1, 0.78], wspace=0.12, hspace=0.14)
    image_ax = fig.add_subplot(grid[:, 0])
    image_ax.imshow(image)
    image_ax.axis("off")
    evidence_ax = fig.add_subplot(grid[0, 1])
    evidence_ax.axis("off")
    evidence_ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=NATURE_PALETTE["near_white"], edgecolor=NATURE_PALETTE["light_grey"], linewidth=0.8))
    evidence_ax.text(0.06, 0.91, "Measured cine evidence", ha="left", va="top", weight="bold", fontsize=7.2, color=NATURE_PALETTE["navy"])
    features = [
        ("LV ejection fraction", row.get("lv_ef_percent"), "%"),
        ("RV ejection fraction", row.get("rv_ef_percent"), "%"),
        ("LV end-diastolic volume", row.get("lv_edv_ml"), "mL"),
        ("RV end-diastolic volume", row.get("rv_edv_ml"), "mL"),
        ("LV myocardial mass", row.get("lv_myocardial_mass_ed_g"), "g"),
    ]
    for index, (label, value, unit) in enumerate(features):
        y = 0.76 - index * 0.14
        formatted = "missing" if pd.isna(value) else f"{float(value):.1f} {unit}"
        evidence_ax.text(0.06, y, label, ha="left", va="center", color=NATURE_PALETTE["grey"], fontsize=6.4)
        evidence_ax.text(0.94, y, formatted, ha="right", va="center", color=NATURE_PALETTE["black"], fontsize=6.7)
        evidence_ax.plot([0.06, 0.94], [y - 0.055, y - 0.055], color=NATURE_PALETTE["light_grey"], linewidth=0.5)
    route_ax = fig.add_subplot(grid[1, 1])
    route_ax.axis("off")
    route = str(row.get("system_route", "review"))
    route_colour = NATURE_PALETTE["green"] if route == "exact_supported_rule" else NATURE_PALETTE["amber"]
    _box(route_ax, (0.03, 0.60), 0.94, 0.30, route.replace("_", "\n"), facecolor=route_colour, fontsize=7.0)
    winner_value = row.get("winner_rule_id", "none")
    winner = "none (conflict or abstention)" if pd.isna(winner_value) else str(winner_value)
    fired = int(row.get("fired_rule_count", 0) or 0)
    route_ax.text(0.05, 0.42, f"Winning trace: {winner}", ha="left", va="center", fontsize=6.5, color=NATURE_PALETTE["black"])
    route_ax.text(0.05, 0.28, f"Strictly fired rules: {fired}", ha="left", va="center", fontsize=6.5, color=NATURE_PALETTE["black"])
    route_ax.text(0.05, 0.10, "Physician records utility, confidence, defects, and deferral reasons.", ha="left", va="center", fontsize=6.2, color=NATURE_PALETTE["grey"], wrap=True)
    return export_nature_figure(fig, output_stem)


def build_transition_fidelity_figure(
    transition_metrics: pd.DataFrame,
    fidelity_gate: pd.DataFrame,
    output_stem: str | Path,
) -> list[Path]:
    configure_nature_style()
    required_metrics = {"dataset_source", "split", "feature_name", "pearson_r"}
    required_gate = {"feature_name", "fidelity_status"}
    if missing := required_metrics.difference(transition_metrics.columns):
        raise ValueError(f"transition metrics missing: {sorted(missing)}")
    normalized_error_column = (
        "normalized_rmse_by_train_sd"
        if "normalized_rmse_by_train_sd" in transition_metrics.columns
        else "normalized_rmse_by_split_sd"
        if "normalized_rmse_by_split_sd" in transition_metrics.columns
        else None
    )
    if normalized_error_column is None:
        raise ValueError("transition metrics missing a normalized RMSE field")
    if missing := required_gate.difference(fidelity_gate.columns):
        raise ValueError(f"fidelity gate missing: {sorted(missing)}")
    selected = transition_metrics[
        ((transition_metrics["dataset_source"] == "ACDC") & (transition_metrics["split"] == "test"))
        | (transition_metrics["dataset_source"].astype(str).str.contains("MMS", case=False))
    ].copy()
    if "experiment_id" in selected.columns:
        selected = selected.loc[selected["experiment_id"].astype(str).eq("primary_A1_all_targets")].copy()
    selected["normalized_rmse"] = pd.to_numeric(selected[normalized_error_column], errors="coerce")
    selected = selected.merge(fidelity_gate[list(required_gate)], on="feature_name", how="left", validate="many_to_one")
    selected["fidelity_status"] = selected["fidelity_status"].fillna("red_not_for_case_level_use")
    colour_map = {
        "green_reliable_for_explanation": NATURE_PALETTE["green"],
        "amber_display_with_residual_warning": NATURE_PALETTE["amber"],
        "amber_display_with_external_instability_warning": NATURE_PALETTE["amber"],
        "red_not_for_case_level_use": NATURE_PALETTE["red"],
    }
    legend_labels = {
        "green_reliable_for_explanation": "green: explanation eligible",
        "amber_display_with_residual_warning": "amber: residual warning",
        "amber_display_with_external_instability_warning": "amber: external instability",
        "red_not_for_case_level_use": "red: case-level use suppressed",
    }
    status_order = list(legend_labels)
    fig, axes = plt.subplots(1, 2, figsize=_figsize(DOUBLE_COLUMN_MM, 75), sharex=False, sharey=False)
    for ax, (dataset, group) in zip(axes, selected.groupby("dataset_source", sort=True)):
        for status in status_order:
            status_group = group.loc[group["fidelity_status"].eq(status)]
            if status_group.empty:
                continue
            ax.scatter(
                status_group["normalized_rmse"],
                status_group["pearson_r"],
                color=colour_map.get(status, NATURE_PALETTE["grey"]),
                s=18,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.35,
                label=legend_labels.get(status, status.replace("_", " ")),
            )
        ax.axvline(1.0, color=NATURE_PALETTE["grey"], linewidth=0.6, linestyle="--")
        ax.axhline(0.5, color=NATURE_PALETTE["grey"], linewidth=0.6, linestyle="--")
        dataset_label = "ACDC held-out test" if dataset == "ACDC" else "M&Ms external"
        ax.set_xlabel(f"Normalized RMSE — {dataset_label}")
        ax.set_ylabel("Pearson r")
        _clean_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=min(3, len(handles)), frameon=False)
    fig.subplots_adjust(bottom=0.20, wspace=0.34)
    return export_nature_figure(fig, output_stem)


def build_bland_altman_figure(
    residuals: pd.DataFrame,
    output_stem: str | Path,
) -> list[Path]:
    """Plot measured-versus-reconstructed agreement without embedded titles or panel letters."""
    configure_nature_style()
    required = {"dataset_source", "feature_name", "observed_value", "predicted_value"}
    if missing := required.difference(residuals.columns):
        raise ValueError(f"transition residuals missing: {sorted(missing)}")
    major = [
        ("lv_edv_ml", "LV EDV", "mL"),
        ("lv_esv_ml", "LV ESV", "mL"),
        ("lv_ef_percent", "LVEF", "%"),
        ("rv_edv_ml", "RV EDV", "mL"),
        ("rv_esv_ml", "RV ESV", "mL"),
        ("rv_ef_percent", "RVEF", "%"),
    ]
    available = [(feature, label, unit) for feature, label, unit in major if residuals["feature_name"].astype(str).eq(feature).any()]
    if not available:
        raise ValueError("No prespecified major ventricular feature exists in the residual table")
    fig, axes = plt.subplots(2, 3, figsize=_figsize(DOUBLE_COLUMN_MM, 112))
    axes_flat = np.asarray(axes).reshape(-1)
    dataset_style = {
        "ACDC": (NATURE_PALETTE["blue"], "ACDC"),
        "MMS_100": (NATURE_PALETTE["purple"], "M&Ms"),
    }
    for ax, (feature, display, unit) in zip(axes_flat, available):
        group = residuals.loc[residuals["feature_name"].astype(str).eq(feature)].copy()
        group["observed_value"] = pd.to_numeric(group["observed_value"], errors="coerce")
        group["predicted_value"] = pd.to_numeric(group["predicted_value"], errors="coerce")
        group = group.dropna(subset=["observed_value", "predicted_value"])
        group["mean_value"] = (group["observed_value"] + group["predicted_value"]) / 2.0
        group["difference"] = group["predicted_value"] - group["observed_value"]
        for dataset, dataset_group in group.groupby("dataset_source", sort=True):
            colour, label = dataset_style.get(str(dataset), (NATURE_PALETTE["grey"], str(dataset)))
            ax.scatter(dataset_group["mean_value"], dataset_group["difference"], s=9, alpha=0.58, color=colour, edgecolor="none", label=label)
        difference = group["difference"].to_numpy(dtype=float)
        bias = float(np.mean(difference)) if len(difference) else np.nan
        sd = float(np.std(difference, ddof=1)) if len(difference) > 1 else np.nan
        ax.axhline(0.0, color=NATURE_PALETTE["light_grey"], linewidth=0.6)
        if np.isfinite(bias):
            ax.axhline(bias, color=NATURE_PALETTE["black"], linewidth=0.8)
        if np.isfinite(sd):
            ax.axhline(bias + 1.96 * sd, color=NATURE_PALETTE["red"], linewidth=0.6, linestyle="--")
            ax.axhline(bias - 1.96 * sd, color=NATURE_PALETTE["red"], linewidth=0.6, linestyle="--")
        ax.set_xlabel(f"Mean {display} ({unit})")
        ax.set_ylabel(f"Reconstructed − measured ({unit})")
        ax.text(0.02, 0.96, f"n={len(group)}", transform=ax.transAxes, ha="left", va="top", fontsize=6.2, color=NATURE_PALETTE["grey"])
        _clean_axes(ax)
    for ax in axes_flat[len(available) :]:
        ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=len(handles), frameon=False)
    fig.subplots_adjust(left=0.09, right=0.99, hspace=0.50, wspace=0.48, bottom=0.13)
    return export_nature_figure(fig, output_stem)


def build_route_composition_figure(manifest: pd.DataFrame, output_stem: str | Path) -> list[Path]:
    configure_nature_style()
    required = {"dataset_source", "system_route", "reader_case_id", "disease_label", "scanner_vendor", "center_id"}
    if missing := required.difference(manifest.columns):
        raise ValueError(f"route manifest missing: {sorted(missing)}")
    manifest = manifest.loc[manifest["dataset_source"].astype(str).eq("MMS_100")].copy()
    if len(manifest) != 100:
        raise ValueError(f"M&Ms route figure requires exactly 100 cases, observed {len(manifest)}")
    route_order = [
        "exact_supported_rule",
        "supported_rule_with_caveat",
        "conflict_review",
        "low_fidelity_review",
        "unsupported_feature_review",
        "off_ontology_review",
        "no_rule_abstention",
        "data_quality_exclusion",
    ]
    route_colours = {
        "exact_supported_rule": NATURE_PALETTE["green"],
        "supported_rule_with_caveat": NATURE_PALETTE["teal"],
        "conflict_review": NATURE_PALETTE["amber"],
        "low_fidelity_review": "#E8B05A",
        "unsupported_feature_review": NATURE_PALETTE["red"],
        "off_ontology_review": NATURE_PALETTE["purple"],
        "no_rule_abstention": NATURE_PALETTE["grey"],
        "data_quality_exclusion": NATURE_PALETTE["black"],
    }
    fig, axes = plt.subplots(3, 1, figsize=_figsize(DOUBLE_COLUMN_MM, 140))
    strata = [
        ("disease_label", "M&Ms benchmark family"),
        ("scanner_vendor", "Scanner vendor code"),
        ("center_id", "Center code"),
    ]
    legend_handles: list[object] = []
    legend_labels: list[str] = []
    for axis_index, (ax, (column, x_label)) in enumerate(zip(axes, strata)):
        working = manifest.assign(_stratum=manifest[column].fillna("<missing>").astype(str))
        counts = working.groupby(["_stratum", "system_route"]).size().unstack(fill_value=0).sort_index()
        for route in route_order:
            if route not in counts:
                counts[route] = 0
        counts = counts[route_order]
        fractions = counts.div(counts.sum(axis=1), axis=0)
        bottom = np.zeros(len(fractions))
        x_values = np.arange(len(fractions))
        for route in route_order:
            values = fractions[route].to_numpy(dtype=float)
            bars = ax.bar(x_values, values, bottom=bottom, color=route_colours[route], width=0.72, label=route.replace("_", " "))
            if axis_index == 0:
                legend_handles.append(bars[0])
                legend_labels.append(route.replace("_", " "))
            bottom += values
        denominators = counts.sum(axis=1).astype(int)
        tick_labels = [f"{value}\nn={denominators.loc[value]}" for value in fractions.index]
        ax.set_ylim(0, 1)
        ax.set_ylabel("Route proportion")
        ax.set_xlabel(x_label)
        ax.set_xticks(x_values, tick_labels)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        _clean_axes(ax)
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=4, frameon=False)
    fig.subplots_adjust(hspace=0.62, bottom=0.17)
    return export_nature_figure(fig, output_stem)


def build_representative_cases_figure(
    manifest: pd.DataFrame,
    montage_root: str | Path,
    output_stem: str | Path,
) -> list[Path]:
    configure_nature_style()
    route_groups = [
        ("exact supported", ["exact_supported_rule"], NATURE_PALETTE["green"]),
        ("meaningful conflict", ["conflict_review"], NATURE_PALETTE["amber"]),
        ("appropriate abstention", ["low_fidelity_review", "no_rule_abstention", "data_quality_exclusion"], NATURE_PALETTE["grey"]),
        ("off ontology", ["off_ontology_review"], NATURE_PALETTE["purple"]),
    ]
    selected: list[tuple[str, pd.Series | None, str]] = []
    for display, routes, colour in route_groups:
        candidates = manifest.loc[manifest["system_route"].astype(str).isin(routes)].sort_values("reader_case_id", kind="mergesort")
        if candidates.empty:
            selected.append((display, None, colour))
        else:
            selected.append((display, candidates.iloc[0], colour))
    fig = plt.figure(figsize=_figsize(DOUBLE_COLUMN_MM, 176))
    grid = fig.add_gridspec(4, 2, width_ratios=[1.55, 1.0], hspace=0.16, wspace=0.08)
    for row_index, (display, row, colour) in enumerate(selected):
        if row is None:
            image_ax = fig.add_subplot(grid[row_index, 0])
            image_ax.axis("off")
            image_ax.add_patch(
                Rectangle(
                    (0, 0),
                    1,
                    1,
                    facecolor=NATURE_PALETTE["near_white"],
                    edgecolor=NATURE_PALETTE["light_grey"],
                    linewidth=0.8,
                )
            )
            image_ax.text(
                0.5,
                0.5,
                "No case entered this system route",
                ha="center",
                va="center",
                fontsize=7.0,
                color=NATURE_PALETTE["grey"],
            )
            evidence_ax = fig.add_subplot(grid[row_index, 1])
            evidence_ax.axis("off")
            evidence_ax.add_patch(
                Rectangle(
                    (0, 0),
                    1,
                    1,
                    facecolor=NATURE_PALETTE["near_white"],
                    edgecolor=colour,
                    linewidth=1.0,
                )
            )
            evidence_ax.text(0.05, 0.86, display, ha="left", va="top", fontsize=7.2, weight="bold", color=colour)
            evidence_ax.text(
                0.05,
                0.55,
                "zero system-routed cases\nreference-stratum performance\nremains a separate\nrestricted audit",
                ha="left",
                va="center",
                fontsize=5.8,
                color=NATURE_PALETTE["black"],
                wrap=True,
            )
            continue
        image_path = Path(montage_root) / "assisted" / f"{row['reader_case_id']}.png"
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        image_ax = fig.add_subplot(grid[row_index, 0])
        image_ax.imshow(mpimg.imread(image_path))
        image_ax.axis("off")
        evidence_ax = fig.add_subplot(grid[row_index, 1])
        evidence_ax.axis("off")
        evidence_ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=NATURE_PALETTE["near_white"], edgecolor=colour, linewidth=1.0))
        evidence_ax.text(0.05, 0.86, display, ha="left", va="top", fontsize=7.2, weight="bold", color=colour)
        evidence_ax.text(0.05, 0.67, str(row["reader_case_id"]), ha="left", va="top", fontsize=6.5, color=NATURE_PALETTE["black"])
        evidence_ax.text(0.05, 0.50, str(row["system_route"]).replace("_", " "), ha="left", va="top", fontsize=6.4, color=NATURE_PALETTE["black"], wrap=True)
        winner = row.get("winner_rule_id")
        evidence_ax.text(0.05, 0.32, f"winner: {'none' if pd.isna(winner) else winner}", ha="left", va="top", fontsize=6.2, color=NATURE_PALETTE["grey"])
        caveat = str(row.get("missing_evidence_caveat") or row.get("abstention_reason") or "No additional caveat")
        caveat = caveat.replace("_", " ").replace(";", "; ")
        caveat = textwrap.shorten(caveat, width=165, placeholder=" …")
        caveat = "\n".join(textwrap.wrap(caveat, width=46, break_long_words=False))
        evidence_ax.text(0.05, 0.16, caveat, ha="left", va="center", fontsize=5.4, color=NATURE_PALETTE["grey"])
    return export_nature_figure(fig, output_stem)


def generate_all_possible_figures(
    *,
    primary_manifest_path: str | Path = "clinical_validation/restricted/primary_reader_manifest_restricted.parquet",
    transition_metrics_path: str | Path = "legacy/v5/artifacts/reports/transition/transition_metrics_v5_native_image_dss_100.parquet",
    fidelity_gate_path: str | Path = "clinical_validation/rules/transition_fidelity_gate.parquet",
    transition_residuals_acdc_path: str | Path = "legacy/v5/artifacts/models/transition/transition_residuals_ACDC_v5_native_image_dss_100.parquet",
    transition_residuals_mms_path: str | Path = "legacy/v5/artifacts/models/transition/transition_residuals_MMS_100_v5_native_image_dss_100.parquet",
    restricted_retest_manifest_path: str | Path = "clinical_validation/restricted/restricted_retest_map.parquet",
    output_root: str | Path = "clinical_validation/figures",
    manuscript_output_root: str | Path = "manuscript/article/figures/clinical_validation",
    manuscript_report_root: str | Path = "manuscript/reports/clinical_validation",
    regenerate_image_assets: bool = True,
) -> dict[str, object]:
    """Generate all figures supportable before genuine physician responses exist."""
    root = resolve_path(output_root)
    reader_root = root / "reader_assets"
    manuscript_root = resolve_path(manuscript_output_root)
    report_root = resolve_path(manuscript_report_root)
    primary = pd.read_parquet(resolve_path(primary_manifest_path))
    metrics = pd.read_parquet(resolve_path(transition_metrics_path))
    gate = pd.read_parquet(resolve_path(fidelity_gate_path))
    residuals = pd.concat(
        [
            pd.read_parquet(resolve_path(transition_residuals_acdc_path)),
            pd.read_parquet(resolve_path(transition_residuals_mms_path)),
        ],
        ignore_index=True,
    )
    restricted_retest = pd.read_parquet(resolve_path(restricted_retest_manifest_path))
    if regenerate_image_assets:
        integrity, reader_outputs = generate_reader_image_assets(primary, reader_root)
        retest_integrity, retest_outputs = generate_retest_image_assets(
            restricted_retest, reader_root
        )
    else:
        integrity_path = reader_root / "image_integrity_manifest.parquet"
        retest_integrity_path = reader_root / "retest_image_integrity_manifest.parquet"
        if not integrity_path.is_file() or not retest_integrity_path.is_file():
            raise FileNotFoundError("Verified reader/retest image-integrity manifests are required for asset reuse")
        integrity = pd.read_parquet(integrity_path)
        retest_integrity = pd.read_parquet(retest_integrity_path)
        if set(integrity["reader_case_id"].astype(str)) != set(primary["reader_case_id"].astype(str)):
            raise ValueError("Reader asset manifest does not match the frozen primary case population")
        if set(retest_integrity["reader_case_id"].astype(str)) != set(
            restricted_retest["retest_case_id"].astype(str)
        ):
            raise ValueError("Retest asset manifest does not match the frozen retest population")
        for frame, context in ((integrity, "reader"), (retest_integrity, "retest")):
            if not frame["status"].astype(str).isin({"generated", "ok"}).all():
                raise ValueError(f"{context} asset reuse refused because an integrity row is not successful")
            missing_outputs = [path for path in frame["output_path"].astype(str) if not Path(path).is_file()]
            if missing_outputs:
                raise FileNotFoundError(f"{context} asset reuse is missing outputs: {missing_outputs[:5]}")
        reader_outputs = [Path(path) for path in integrity["output_path"].astype(str)]
        reader_outputs.extend([integrity_path, integrity_path.with_suffix(".csv")])
        retest_outputs = [Path(path) for path in retest_integrity["output_path"].astype(str)]
        retest_outputs.extend([retest_integrity_path, retest_integrity_path.with_suffix(".csv")])
    outputs: list[Path] = list(reader_outputs)
    outputs.extend(retest_outputs)
    outputs.extend(build_architecture_figure(manuscript_root / "figure_1_architecture_and_human_workflow"))
    outputs.extend(build_cohort_flow_figure(primary, manuscript_root / "figure_2_cohort_flow"))
    outputs.extend(build_interface_preview_figure(primary, reader_root, manuscript_root / "figure_3_physician_interface_preview"))
    outputs.extend(build_bland_altman_figure(residuals, manuscript_root / "figure_7_bland_altman_preclinical"))
    outputs.extend(build_route_composition_figure(primary, manuscript_root / "figure_8_route_composition_preclinical"))
    outputs.extend(build_representative_cases_figure(primary, reader_root, manuscript_root / "figure_9_representative_cases_preclinical"))
    outputs.extend(build_transition_fidelity_figure(metrics, gate, manuscript_root / "figure_s1_transition_fidelity_preclinical"))
    off_ontology_count = int(
        primary.get("system_route", pd.Series("", index=primary.index))
        .astype(str)
        .eq("off_ontology_review")
        .sum()
    )
    representative_caption = (
        "**Figure 9. Representative preclinical routes.** Four pseudonymous, label-safe cases illustrate an exact supported route, a meaningful conflict, an appropriate QC/uncertainty abstention, and an off-ontology route with its missing-evidence warning. Benchmark labels and physician responses are not displayed."
        if off_ontology_count
        else "**Figure 9. Representative preclinical routes.** Pseudonymous, label-safe cases illustrate an exact supported route, a meaningful conflict, and an appropriate QC/uncertainty abstention. The fourth slot transparently records that zero cases entered the system off-ontology route; reference-stratum performance remains a separate restricted audit. Benchmark labels and physician responses are not displayed."
    )
    captions = report_root / "figure_captions_pre_response.md"
    captions.parent.mkdir(parents=True, exist_ok=True)
    captions.write_text(
        "\n".join(
            [
                "# Pre-response figure captions",
                "",
                "All figures are research-use-only pre-study artifacts. No physician endpoint, ethics outcome, or deployment claim is shown.",
                "",
                "**Figure 1. Auditable transition-matrix workflow and human-review routes.** Native short-axis cine CMR and masks are transformed into deterministic formal features, measured semantics, fidelity-gated reconstructions, and strict rule traces. Every case retains a terminal supported, conflict/low-fidelity, or unsupported/off-ontology review route.",
                "",
                "**Figure 2. Prespecified reader-cohort assembly.** The complete ACDC inventory contains 150 patients split 85/15/50 at patient level; the fixed external M&Ms subset contains 100 patients. The primary reader cohort contains all 50 held-out ACDC test patients and all 100 M&Ms cases (150 unique cases), with a 40-case detailed audit and 24 hidden retest duplicates outside the primary endpoint denominator.",
                "",
                "**Figure 3. Assisted physician-interface preview.** Label-safe image overlays are presented beside measured cine evidence and the strict rule/review route. Benchmark labels are not shown. The interface records utility, confidence, misleading content, and appropriate deferral without changing the frozen unassisted response.",
                "",
                "**Figure 7. Measured-versus-reconstructed agreement.** Bland–Altman plots show reconstructed minus segmentation-measured values against their mean for six prespecified major ventricular features. Points include all finite ACDC and M&Ms observations; solid black lines show pooled bias and dashed red lines show pooled 95% limits of agreement. Physician remeasurement is unavailable before the reader study and is not simulated.",
                "",
                "**Figure 8. External route composition.** Route proportions include all 100 fixed M&Ms cases and are stratified by benchmark family, scanner-vendor code, and center code. Tick labels show each denominator. Conflict, low-fidelity, unsupported, off-ontology, no-rule, and data-quality routes are retained.",
                "",
                representative_caption,
                "",
                "**Supplementary Figure 1. Preclinical transition-fidelity audit.** Held-out ACDC and external M&Ms reconstruction performance is shown for all semantic targets using correlation and normalized error. Green, amber, and red points correspond to validation-frozen computational gates; red targets are suppressed from case-level reconstructed evidence.",
                "",
                "Figures 4-6 are intentionally withheld until signed and frozen physician responses exist; the generator fails closed rather than populating them from mock data.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    outputs.append(captions)
    summary = {
        "generated_at_utc": utc_now_iso(),
        "primary_case_count": int(primary["reader_case_id"].nunique()),
        "image_asset_rows": int(len(integrity)),
        "image_asset_status_counts": integrity["status"].value_counts().to_dict(),
        "retest_image_asset_rows": int(len(retest_integrity)),
        "retest_image_asset_status_counts": retest_integrity["status"].value_counts().to_dict(),
        "figure_stems": sorted(
            {
                str(path.with_suffix(""))
                for path in outputs
                if path.suffix in {".svg", ".pdf", ".tiff", ".png"}
                and path.is_relative_to(manuscript_root)
            }
        ),
        "deferred_figures": [
            "figure_4_rule_acceptance",
            "figure_5_risk_coverage_and_abstention_concordance",
            "figure_6_paired_reader_change",
        ],
    }
    summary_path = report_root / "figure_generation_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
