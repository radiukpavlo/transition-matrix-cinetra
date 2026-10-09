"""M&Ms-2 ingest and lock for the protocol-v6.0.1 amendment (``configs/cohorts.yaml`` ``mms2``).

Source: the M&Ms-2 release archive ``MnM2.rar`` (size and SHA-256 in ``registry/sources/mms2.yaml``), held unopened
until the annotated tag ``protocol-v6.0.1`` exists and HEAD descends from it.

* **Split.**  Subject ids ``001``-``360`` are ranked by SHA-256 of ``"mms2-split-v6.0.1|" + subject_id`` (hex digest
  ascending, ties by id; rank 1-360).  Ranks 1-180 form ``mms2_external`` (prospective external test), ranks
  181-360 ``mms2_sealed`` (reserved for v7; never extracted, decompressed or written under v6).  The committed
  ``registry/sources/mms2_split_manifest.csv`` must reproduce the rule (:func:`load_split_manifest`).
* **Extraction** (:func:`extract_external`, CLI ``extract-mms2``, host only).  The lock comes first (no byte of the
  archive, not even for its SHA-256, is read before the tag), then the archive size and SHA-256; the archive is listed (``tar -tvf``) and exactly the 720 short-axis ED/ES image and mask entries of
  the 180 ``mms2_external`` subjects are extracted with ``tar -xf <archive> -C <dest> -T <list>`` (Windows
  ``tar.exe`` is bsdtar/libarchive, which reads RAR; GNU tar cannot).  The extracted files must equal the list and
  their sizes the listing.  Long-axis and cine files are never listed for extraction.
* **Release CSV.**  ``dataset_information.csv`` is read only here, into memory (``tar -xO``), checked against the
  declared column contract (``mms2.info_csv``: exactly ``SUBJECT_CODE, DISEASE, VENDOR, SCANNER, FIELD``, the header
  line read before the tag with the owner's approval; any unknown or missing column, a malformed or duplicate subject
  code, or a ``DISEASE`` value outside the declared synonym table fails closed with :class:`HumanGateError`, exit 8),
  and its ``mms2_sealed`` rows are dropped before anything is written: only ``MMS2_INFO_EXTERNAL.csv`` (the external
  rows, original header and strings) reaches the source area.  The values of sealed rows are never parsed.
  ``VENDOR``, ``SCANNER`` and ``FIELD`` are data-driven descriptive strata: every value is accepted and recorded
  verbatim (``VENDOR`` is also normalized to Siemens, GE or Philips when it names exactly one of them).
* **Identifiers.**  The split hash uses the raw zero-padded subject code (``001``-``360``); every downstream
  identifier is prefixed (``case_id`` = ``patient_id`` = ``MMS2_<code>``, ``sample_id`` = ``MMS2:MMS2_<code>``;
  :func:`case_key`), so no purely numeric token is ever a case identifier.
* **Ingest contract.**  Inputs ``<id>_SA_ED.nii.gz``, ``<id>_SA_ED_gt.nii.gz``, ``<id>_SA_ES.nii.gz``,
  ``<id>_SA_ES_gt.nii.gz`` only; phases from the file names; labels identity (1 LV, 2 MYO, 3 RV); geometry by the
  v6 rule (:func:`check_geometry`): the affine is voxel-scaled (column norms equal the header pixdim) or unit-scaled
  (then the header pixdim is the voxel size); anything else, or image/mask or ED/ES headers that disagree, is
  rejected with ``INPUT_GEOMETRY_INCONSISTENT``.
* **File manifest.**  ``MMS2_EXTRACTION_MANIFEST.csv`` in the source area lists every extracted file with its role,
  size and SHA-256 (and the external CSV); staging verifies against it.  ``MMS2_EXTRACTION_RECEIPT.json`` records the
  archive hash, the listing and list-file hashes, the byte checks and the tar version.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from ..paths import SOURCE_REGISTRY_DIR
from .sources import SYNTHETIC_MARKER, SourceError, SourceFile, is_synthetic_fixture, read_verified_bytes, sample_id, sha256_bytes, sha256_file

DATASET = "MMS2"
COHORT_KEY = "mms2"
SPLIT_SALT = "mms2-split-v6.0.1|"
N_SUBJECTS = 360
N_EXTERNAL = 180
EXTERNAL = "mms2_external"
SEALED = "mms2_sealed"
SUBJECT_IDS = tuple(f"{i:03d}" for i in range(1, N_SUBJECTS + 1))
SPLIT_MANIFEST = SOURCE_REGISTRY_DIR / "mms2_split_manifest.csv"
RECEIPT = SOURCE_REGISTRY_DIR / "mms2.yaml"
SPLIT_COLUMNS = ["subject_id", "rank", "sha256", "partition"]
ROLE_SUFFIX = {"ed_image": "SA_ED", "ed_mask": "SA_ED_gt", "es_image": "SA_ES", "es_mask": "SA_ES_gt"}
CASE_ROLES = tuple(ROLE_SUFFIX)
INFO_ROLE = "info_csv"
INFO_CASE = "_release_metadata"
FILE_MANIFEST = "MMS2_EXTRACTION_MANIFEST.csv"
RECEIPT_JSON = "MMS2_EXTRACTION_RECEIPT.json"
INFO_EXTERNAL = "MMS2_INFO_EXTERNAL.csv"
FILE_MANIFEST_COLUMNS = ["case_id", "role", "relpath", "bytes", "sha256", "archive_member"]
SYNTHETIC_ARCHIVE_DESCRIPTOR = f"{SYNTHETIC_MARKER}.archive.json"   # expected size/SHA-256 of a synthetic test archive
DEFAULT_TAR = "C:/Windows/System32/tar.exe"
GEOMETRY_REASON = "INPUT_GEOMETRY_INCONSISTENT"
HUMAN_GATE = "HUMAN_GATE"
EXTRACTION_SCHEMA = "tmmri.mms2_extraction/1"
SA_MEMBER = re.compile(r"(?:^|/)(\d{3})_SA_(ED|ES)(_gt)?\.nii\.gz$")
CASE_PREFIX = "MMS2_"                 # protocol-v6.0.1 owner decision C: prefixed downstream identifiers
VENDOR_PATTERNS = {"Siemens": re.compile(r"siemens", re.I), "Philips": re.compile(r"philips", re.I),
                   "GE": re.compile(r"(?<![a-z])ge(?![a-z])|general\s+electric", re.I)}   # owner decision B


class HumanGateError(SourceError):
    """A release fact differs from the declared assumption (layout, column, value): fail closed for a human decision."""

    reason_code = HUMAN_GATE


class GeometryInconsistent(ValueError):
    """The headers of a case violate the v6 geometry rule (reason code ``INPUT_GEOMETRY_INCONSISTENT``)."""

    reason_code = GEOMETRY_REASON


# ----------------------------------------------------------------------------- identifiers


def case_key(subject_id: str) -> str:
    """``"001"`` -> ``"MMS2_001"``: the downstream ``case_id`` (= ``patient_id``) of a subject (owner decision C).
    The split hash keeps the raw zero-padded code; nothing downstream uses a purely numeric identifier."""

    sid = str(subject_id)
    if sid not in SUBJECT_IDS:
        raise SourceError(f"M&Ms-2 subject code {sid!r} is not one of 001-{N_SUBJECTS:03d}")
    return f"{CASE_PREFIX}{sid}"


def subject_of(case_id: str) -> str:
    """Inverse of :func:`case_key` (``"MMS2_001"`` -> ``"001"``); anything else fails closed."""

    text = str(case_id)
    if not text.startswith(CASE_PREFIX) or text[len(CASE_PREFIX):] not in SUBJECT_IDS:
        raise SourceError(f"{text!r} is not an M&Ms-2 case id ({CASE_PREFIX}001-{CASE_PREFIX}{N_SUBJECTS:03d})")
    return text[len(CASE_PREFIX):]


# ----------------------------------------------------------------------------- split manifest


def split_digest(subject_id: str, salt: str = SPLIT_SALT) -> str:
    return hashlib.sha256(f"{salt}{subject_id}".encode("utf-8")).hexdigest()


def build_split_manifest(salt: str = SPLIT_SALT, n_external: int = N_EXTERNAL, subjects: Iterable[str] = SUBJECT_IDS) -> pd.DataFrame:
    """The split by the registered rule: one row per subject, sorted by ``rank`` (1-based)."""

    digests = {str(s): split_digest(str(s), salt) for s in subjects}
    order = sorted(digests, key=lambda s: (digests[s], s))
    return pd.DataFrame({"subject_id": order, "rank": np.arange(1, len(order) + 1, dtype=np.int64),
                         "sha256": [digests[s] for s in order],
                         "partition": [EXTERNAL if r <= n_external else SEALED for r in range(1, len(order) + 1)]})


def write_split_manifest(path: str | Path = SPLIT_MANIFEST) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(build_split_manifest().to_csv(index=False, lineterminator="\n").encode("utf-8"))
    return path


def lf_sha256(path: str | Path) -> str:
    """SHA-256 of a tracked text file with CRLF normalized to LF (Windows checkouts and the container agree)."""

    return sha256_bytes(Path(path).read_bytes().replace(b"\r\n", b"\n"))


def load_split_manifest(path: str | Path = SPLIT_MANIFEST, *, verify: bool = True) -> pd.DataFrame:
    """The committed split manifest; with ``verify`` it must equal :func:`build_split_manifest` (else SourceError)."""

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(frame.columns) != SPLIT_COLUMNS:
        raise SourceError(f"{path}: expected columns {SPLIT_COLUMNS}, found {list(frame.columns)}")
    frame["rank"] = frame["rank"].astype(np.int64)
    if verify:
        expected = build_split_manifest()
        if not frame.reset_index(drop=True).equals(expected):
            raise SourceError(f"{path} does not reproduce the registered split rule (salt {SPLIT_SALT!r}); never edit it by hand")
    return frame


def external_ids(split: pd.DataFrame | None = None) -> list[str]:
    split = load_split_manifest() if split is None else split
    return sorted(split.loc[split["partition"] == EXTERNAL, "subject_id"].astype(str))


def sealed_ids(split: pd.DataFrame | None = None) -> list[str]:
    split = load_split_manifest() if split is None else split
    return sorted(split.loc[split["partition"] == SEALED, "subject_id"].astype(str))


def load_receipt(path: str | Path = RECEIPT) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def default_cohort_config() -> dict[str, Any]:
    """``configs/cohorts.yaml`` ``mms2`` of the repository (callers with a StudyConfig pass ``study.cohorts["mms2"]``)."""

    import yaml

    from ..paths import CONFIGS_DIR

    cohorts = yaml.safe_load((CONFIGS_DIR / "cohorts.yaml").read_text(encoding="utf-8")) or {}
    if "mms2" not in cohorts:
        raise SourceError("configs/cohorts.yaml has no mms2 section (protocol-v6.0.1 amendment)")
    return cohorts["mms2"]


# ----------------------------------------------------------------------------- file names and the archive listing


def case_file_names(subject_id: str) -> dict[str, str]:
    """``{role: file name}`` of the four short-axis ED/ES inputs of a subject (phases from the file names)."""

    return {role: f"{subject_id}_{suffix}.nii.gz" for role, suffix in ROLE_SUFFIX.items()}


def extraction_names(subjects: Iterable[str]) -> list[str]:
    """The sorted file names to extract (four per subject; 720 for the 180 ``mms2_external`` subjects)."""

    return sorted(name for s in subjects for name in case_file_names(str(s)).values())


@dataclass(frozen=True)
class ArchiveEntry:
    member: str          # the name exactly as the archive listing prints it (passed back to tar unchanged)
    size: int
    is_file: bool

    @property
    def path(self) -> str:
        return self.member.replace("\\", "/")

    @property
    def basename(self) -> str:
        return self.path.rsplit("/", 1)[-1]


# bsdtar:  -rw-rw-rw-  0 0      0      123456 Oct  2 21:51 MnM2/dataset/001/001_SA_ED.nii.gz   (month may print as ???)
# GNU tar: -rw-r--r-- user/group 123456 2026-10-02 21:51 MnM2/dataset/001/001_SA_ED.nii.gz
_GNU_LINE = re.compile(r"^(?P<mode>\S{10})\s+\S+/\S+\s+(?P<size>\d+)\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?\s(?P<name>.+)$")
_BSD_LINE = re.compile(r"^(?P<mode>\S{10})\s+\d+\s+\S+\s+\S+\s+(?P<size>\d+)\s+\S+\s+\S+\s+\S+\s(?P<name>.+)$")


def parse_listing(text: str) -> list[ArchiveEntry]:
    """Parse a verbose ``tar -tvf`` listing (bsdtar or GNU tar); an unparsed line fails closed."""

    entries = []
    for line in text.splitlines():
        if not line.strip():
            continue
        match = _GNU_LINE.match(line) or _BSD_LINE.match(line)
        if match is None:
            raise HumanGateError(f"unparsed archive listing line: {line[:200]!r}")
        mode, name = match["mode"], match["name"]
        if mode.startswith("l") and " -> " in name:
            name = name.split(" -> ", 1)[0]
        entries.append(ArchiveEntry(member=name, size=int(match["size"]), is_file=mode.startswith("-")))
    return entries


def select_members(entries: Iterable[ArchiveEntry], subjects: Iterable[str], *, info_csv_name: str) -> dict[str, Any]:
    """Map the archive entries to the extraction list (exactly four per subject) and locate the release CSV.

    The short-axis members must name exactly the subjects 001-360, and each of the four short-axis ED/ES file names of
    every subject (``mms2_sealed`` included) must occur exactly once among the file entries (else
    :class:`HumanGateError`).  Only file names are inspected; nothing is read from the archive here.
    """

    entries = list(entries)
    files = [e for e in entries if e.is_file]
    by_name: dict[str, list[ArchiveEntry]] = {}
    for e in files:
        by_name.setdefault(e.basename, []).append(e)
    sa = [m for m in (SA_MEMBER.search(e.path) for e in files) if m]
    listed = {m.group(1) for m in sa}
    if listed != set(SUBJECT_IDS):                # SPECIFICATION 4.6: the listing holds exactly the subjects 001-360 (names only)
        raise HumanGateError(f"archive short-axis members cover {len(listed)} subjects, not exactly 001-{N_SUBJECTS:03d} "
                             f"(missing {sorted(set(SUBJECT_IDS) - listed)[:5]}, unexpected {sorted(listed - set(SUBJECT_IDS))[:5]})")
    selected: list[dict[str, Any]] = []
    problems: list[str] = []
    wanted = {str(s) for s in subjects}
    # SPECIFICATION 4.6: every subject 001-360 (mms2_sealed included, names only) has each of its four short-axis ED/ES
    # members exactly once; only the requested subjects are selected for extraction.
    for sid in sorted(set(SUBJECT_IDS) | wanted):
        for role, name in case_file_names(sid).items():
            hits = by_name.get(name, [])
            if len(hits) != 1:
                problems.append(f"{name}: {len(hits)} archive file entries")
                continue
            if sid in wanted:
                selected.append({"case_id": sid, "role": role, "member": hits[0].member, "relpath": hits[0].path, "bytes": hits[0].size})
    info = by_name.get(info_csv_name, [])
    if len(info) != 1:
        problems.append(f"{info_csv_name}: {len(info)} archive file entries")
    if problems:
        raise HumanGateError(f"archive layout differs from the declared M&Ms-2 layout ({len(problems)} problems): {problems[:5]}")
    return {"members": sorted(selected, key=lambda r: r["member"]), "info_csv": info[0],
            "counts": {"archive_entries": len(entries), "archive_files": len(files), "short_axis_ed_es_files": len(sa),
                       "short_axis_subjects": len({m.group(1) for m in sa}), "selected": len(selected)}}


def layout_report(selection: Mapping[str, Any], layout: Mapping[str, Any] | None) -> dict[str, Any]:
    """Compare the selected members with the declared release layout (``mms2.inputs.archive_layout``: member names
    read before the tag with the owner's approval).  Informational only: the archive SHA-256 already fixes every
    member, and the selection itself is by unique file name, so a different spelling of the folders never blocks."""

    layout = dict(layout or {})
    template = str(layout.get("case_member", "MnM2/dataset/<ID>/<FILE>"))
    info_expected = layout.get("info_csv")
    differ = [m["relpath"] for m in selection["members"]
              if m["relpath"].lstrip("./") != template.replace("<ID>", m["case_id"]).replace("<FILE>", m["relpath"].rsplit("/", 1)[-1])]
    entries = layout.get("listing_entries")
    info_path = selection["info_csv"].path.lstrip("./")
    return {"declared": layout, "members_matching_template": len(selection["members"]) - len(differ), "members_differing": differ[:10],
            "info_csv_matches": None if info_expected is None else info_path == str(info_expected),
            "listing_entries_declared": entries, "listing_entries_observed": selection["counts"]["archive_entries"],
            "listing_entries_match": None if entries is None else int(entries) == int(selection["counts"]["archive_entries"])}


# ----------------------------------------------------------------------------- tar (bsdtar) helpers


def _tar(tar: str | Path, args: list[str], *, timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([str(tar), *args], capture_output=True, timeout=timeout)


def tar_version(tar: str | Path) -> str:
    try:
        out = _tar(tar, ["--version"], timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SourceError(f"tar executable unusable: {tar} ({exc})") from exc
    return out.stdout.decode("utf-8", errors="replace").splitlines()[0].strip() if out.stdout else ""


def list_archive(archive: str | Path, tar: str | Path = DEFAULT_TAR) -> tuple[list[ArchiveEntry], str]:
    """(entries, SHA-256 of the raw listing) of ``tar -tvf``; a non-zero exit fails."""

    out = _tar(tar, ["-tvf", str(archive)])
    if out.returncode != 0:
        raise SourceError(f"tar -tvf failed ({out.returncode}): {out.stderr.decode('utf-8', errors='replace')[:400]}")
    return parse_listing(out.stdout.decode("utf-8", errors="replace")), sha256_bytes(out.stdout)


def read_member_bytes(archive: str | Path, member: str, tar: str | Path = DEFAULT_TAR) -> tuple[bytes, str]:
    """One file member into memory only: ``tar -xOf <archive> <member>`` (stdout).  There is no fallback to a file:
    SPECIFICATION.md 4.6 reads the release CSV into memory and drops its sealed rows before any write, so a non-zero
    exit or empty output fails closed (:class:`HumanGateError`, exit 8) and nothing is written."""

    out = _tar(tar, ["-xOf", str(archive), member])
    if out.returncode != 0 or not out.stdout:
        raise HumanGateError(f"tar -xOf did not return {member} into memory (exit {out.returncode}, {len(out.stdout or b'')} bytes): "
                             f"{(out.stderr or b'').decode('utf-8', errors='replace')[:400]}; the release CSV is never written to a "
                             "file, so extraction stops for a human decision")
    return out.stdout, "stdout"


# ----------------------------------------------------------------------------- release CSV contract


def info_contract(cohort_cfg: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(cohort_cfg.get("info_csv") or {})
    if not cfg.get("columns") or not cfg.get("key"):
        raise SourceError("configs/cohorts.yaml mms2.info_csv must declare key and columns")
    return cfg


def normalize_subject_code(value: object) -> str:
    """``1``/``"001"``/``" 1 "`` -> ``"001"`` (1-360); anything else fails closed."""

    text = str(value).strip()
    if not re.fullmatch(r"\d{1,3}", text) or not 1 <= int(text) <= N_SUBJECTS:
        raise HumanGateError(f"release CSV subject code {text!r} is not an integer 1-{N_SUBJECTS}")
    return f"{int(text):03d}"


def _norm_text(value: object) -> str:
    """Case-insensitive, whitespace-trimmed comparison key (inner runs of white space collapse to one space)."""

    return " ".join(str(value).split()).upper()


def _category(value: str, spec: Mapping[str, Any]) -> str | None:
    """The declared code of ``value``: a code itself or one of its spelled-out synonyms (``aliases``), compared
    case-insensitively after trimming white space; ``None`` when undeclared."""

    key = _norm_text(value)
    aliases = {_norm_text(k): str(v) for k, v in (spec.get("aliases") or {}).items()}
    allowed = {_norm_text(v): str(v) for v in spec.get("values") or []}
    return allowed.get(key) or aliases.get(key)


def normalize_vendor(value: object) -> str | None:
    """Owner decision B: ``Siemens``, ``GE`` or ``Philips`` when the release string names exactly one of them
    (case-insensitive; ``GE`` as a word or ``General Electric``), else the string itself (trimmed); never fails."""

    text = str(value).strip() if value is not None else ""
    if not text:
        return None
    hits = [name for name, pattern in VENDOR_PATTERNS.items() if pattern.search(text)]
    return hits[0] if len(hits) == 1 else text


def parse_release_csv(data: bytes, cohort_cfg: Mapping[str, Any], external: Iterable[str], *,
                      expect_all_subjects: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Check the release CSV against the declared contract and drop every non-external row before anything else.

    Returns ``(external rows with the original header and strings, report)``.  ``expect_all_subjects``: the CSV must
    carry exactly the subject codes 001-360 (the release file); otherwise (the external copy) every code must be an
    external one.  Only the key column is read for sealed rows; their other values are never inspected.
    """

    cfg = info_contract(cohort_cfg)
    columns: dict[str, dict[str, Any]] = {str(k): dict(v or {}) for k, v in cfg["columns"].items()}
    key = str(cfg["key"])
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HumanGateError(f"release CSV is not UTF-8 ({exc})") from exc
    header = next(csv.reader(io.StringIO(text)), [])
    unknown = [c for c in header if c not in columns]
    missing = [c for c in columns if c not in header]
    if unknown or missing or len(set(header)) != len(header):
        raise HumanGateError(f"release CSV columns differ from configs/cohorts.yaml mms2.info_csv: unknown {unknown}, missing {missing}, "
                             f"header {header}")
    frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    # Rows whose every field is empty (separator-only lines such as ",,,,") carry no subject: they are dropped.  A row with
    # a subject code still fails closed on any empty or unknown declared value (format observed before the tag, decision D-2).
    frame = frame[frame.apply(lambda r: any(str(v).strip() for v in r), axis=1)].reset_index(drop=True)
    codes = [normalize_subject_code(v) for v in frame[key]]
    if len(set(codes)) != len(codes):
        raise HumanGateError("release CSV repeats a subject code")
    wanted = {str(s) for s in external}
    if expect_all_subjects and set(codes) != set(SUBJECT_IDS):
        raise HumanGateError(f"release CSV subject codes differ from 001-{N_SUBJECTS:03d} ({len(set(codes))} codes)")
    if not expect_all_subjects and not set(codes) <= wanted:
        raise SourceError(f"{len(set(codes) - wanted)} non-external subject rows in the external CSV copy")
    keep = np.array([c in wanted for c in codes], dtype=bool)
    ext = frame.loc[keep].copy()                              # sealed rows dropped here, before any value is parsed
    ext.insert(0, "__case_id", [c for c, k in zip(codes, keep) if k])
    ext = ext.sort_values("__case_id").reset_index(drop=True)
    problems: list[str] = []
    for name, spec in columns.items():
        kind = str(spec.get("type", "free_text"))
        values = ext[name].astype(str)
        if kind == "category":
            bad = sorted({v for v in values if _category(v, spec) is None and not (v.strip() == "" and spec.get("allow_missing"))})
            if bad:
                problems.append(f"{name}: undeclared values {bad[:10]}")
        elif kind == "number":
            lo, hi = (float(x) for x in spec.get("range", [-np.inf, np.inf]))
            for v in values:
                if v.strip() == "":
                    if not spec.get("allow_missing"):
                        problems.append(f"{name}: missing value")
                        break
                    continue
                try:
                    number = float(v)
                except ValueError:
                    problems.append(f"{name}: non-numeric value {v!r}")
                    break
                if not (np.isfinite(number) and lo <= number <= hi):
                    problems.append(f"{name}: value {v!r} outside {[lo, hi]}")
                    break
        elif kind not in {"subject_code", "free_text", "vendor"}:
            raise SourceError(f"configs/cohorts.yaml mms2.info_csv column {name}: unknown type {kind!r}")
    if problems:
        raise HumanGateError(f"release CSV values differ from configs/cohorts.yaml mms2.info_csv: {problems}")
    report = {"columns": header, "rows": int(len(frame)), "rows_external": int(len(ext)), "rows_dropped_not_external": int((~keep).sum())}
    return ext, report


def external_csv_bytes(ext: pd.DataFrame) -> bytes:
    """The external-only copy: original header and cell strings, sorted by subject code, LF line endings."""

    return ext.drop(columns=["__case_id"]).to_csv(index=False, lineterminator="\n").encode("utf-8")


def info_table(ext: pd.DataFrame, cohort_cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Canonical metadata columns (``role`` of each declared column) per external subject; categories canonicalized."""

    cfg = info_contract(cohort_cfg)
    out = pd.DataFrame({"case_id": ext["__case_id"].astype(str)})
    for name, spec in cfg["columns"].items():
        spec = dict(spec or {})
        role = str(spec.get("role", name))
        if role == "case_id":
            continue
        values = ext[name].astype(str)
        kind = str(spec.get("type", "free_text"))
        if kind == "category":
            out[role] = [_category(v, spec) if v.strip() else None for v in values]
        elif kind == "vendor":                        # data-driven stratum: normalized name plus the verbatim string
            out[role] = [normalize_vendor(v) for v in values]
            out[f"{role}_release"] = [v if v.strip() else None for v in values]
        elif kind == "number":
            out[role] = pd.to_numeric(values.where(values.str.strip() != ""), errors="coerce").astype("float64")
        else:
            out[role] = [v.strip() or None for v in values]
    return out


# ----------------------------------------------------------------------------- geometry rule


def classify_affine(header: Any, *, rtol: float = 1e-3, atol: float = 1e-4) -> str:
    """``voxel_scaled`` (affine column norms equal the pixdim), ``unit_scaled`` (norms 1 and pixdim not all 1; the
    header pixdim is then the voxel size) or ``inconsistent``."""

    affine = np.asarray(header.get_best_affine(), dtype=float)
    zooms = np.asarray(header.get_zooms()[:3], dtype=float)
    if affine.shape != (4, 4) or zooms.size < 3 or not np.isfinite(affine).all():
        return "inconsistent"
    norms = np.linalg.norm(affine[:3, :3], axis=0)
    if np.allclose(norms, zooms, rtol=rtol, atol=atol):
        return "voxel_scaled"
    if np.allclose(norms, 1.0, atol=atol) and not np.allclose(zooms, 1.0, atol=atol):
        return "unit_scaled"
    return "inconsistent"


def check_geometry(headers: Mapping[str, Any]) -> dict[str, Any]:
    """The v6 geometry rule on the four headers of a case (roles of :data:`ROLE_SUFFIX`); raises
    :class:`GeometryInconsistent` with a reason, else returns the classification per role."""

    kinds = {role: classify_affine(h) for role, h in headers.items()}
    problems = [f"{role}: affine column norms disagree with the header pixdim" for role, k in kinds.items() if k == "inconsistent"]

    def same(a: Any, b: Any) -> bool:
        return (tuple(a.get_data_shape()[:3]) == tuple(b.get_data_shape()[:3])
                and np.allclose(a.get_best_affine(), b.get_best_affine(), rtol=0, atol=1e-4)
                and np.allclose(a.get_zooms()[:3], b.get_zooms()[:3], rtol=0, atol=1e-6))

    for left, right in (("ed_image", "ed_mask"), ("es_image", "es_mask"), ("ed_image", "es_image")):
        if left in headers and right in headers and not same(headers[left], headers[right]):
            problems.append(f"{left} and {right}: shape, affine or pixdim differ")
    if len({k for k in kinds.values() if k != "inconsistent"}) > 1:
        problems.append(f"mixed affine scaling {kinds}")
    if problems:
        raise GeometryInconsistent("; ".join(problems))
    return {"affine_scaling": kinds, "rule": "voxel-scaled affine, or unit-scaled affine with the header pixdim as voxel size"}


# ----------------------------------------------------------------------------- file manifest and cohort table


def write_file_manifest(root: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    frame = pd.DataFrame(list(rows), columns=FILE_MANIFEST_COLUMNS).sort_values(["case_id", "role"]).reset_index(drop=True)
    path = Path(root) / FILE_MANIFEST
    path.write_bytes(frame.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    return path


def file_manifest_path(root: str | Path) -> Path:
    return Path(root) / FILE_MANIFEST


def load_file_manifest(root: str | Path, split: pd.DataFrame | None = None) -> pd.DataFrame:
    """The extraction manifest of a source root (one row per file; ``info_csv`` row for the external CSV copy).

    Fails closed (SourceError) on an unknown role, a duplicate (case, role), or any subject outside ``mms2_external``.
    """

    path = file_manifest_path(root)
    if not path.is_file():
        raise SourceError(f"{path} missing: run extract-mms2 first (the M&Ms-2 source area has no extraction manifest)")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(frame.columns) != FILE_MANIFEST_COLUMNS:
        raise SourceError(f"{path}: expected columns {FILE_MANIFEST_COLUMNS}")
    frame["bytes"] = frame["bytes"].astype(np.int64)
    frame["sha256"] = frame["sha256"].str.strip().str.lower()
    if frame.duplicated(["case_id", "role"]).any():
        raise SourceError(f"{path}: duplicate (case_id, role)")
    if not set(frame["role"]) <= set(CASE_ROLES) | {INFO_ROLE}:
        raise SourceError(f"{path}: unknown roles {sorted(set(frame['role']) - set(CASE_ROLES) - {INFO_ROLE})}")
    cases = set(frame.loc[frame["role"] != INFO_ROLE, "case_id"])
    try:
        subjects = {subject_of(c) for c in cases}
    except SourceError as exc:
        raise SourceError(f"{path}: {exc} (case ids are MMS2_<subject code>)") from exc
    outside = subjects - set(external_ids(split))
    if outside:
        raise SourceError(f"{path}: {len(outside)} subjects outside mms2_external (sealed subjects are never extracted)")
    return frame.sort_values(["case_id", "role"]).reset_index(drop=True)


def info_entry(manifest: pd.DataFrame) -> SourceFile:
    rows = manifest[manifest["role"] == INFO_ROLE]
    if len(rows) != 1:
        raise SourceError(f"{FILE_MANIFEST} must list exactly one {INFO_ROLE}")
    r = rows.iloc[0]
    return SourceFile(INFO_ROLE, str(r["relpath"]), str(r["sha256"]), int(r["bytes"]))


def case_files(manifest: pd.DataFrame) -> dict[str, dict[str, dict[str, Any]]]:
    """``{case_id: {role: {relpath, sha256, bytes}}}`` of the image and mask files (case ids ``MMS2_<code>``)."""

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for r in manifest[manifest["role"] != INFO_ROLE].itertuples(index=False):
        out.setdefault(str(r.case_id), {})[str(r.role)] = {"relpath": str(r.relpath), "sha256": str(r.sha256), "bytes": int(r.bytes)}
    return dict(sorted(out.items()))


def crosswalk(pathology: object, cohort_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """ACDC class, compatibility and pathology group of an M&Ms-2 code (``mms2.crosswalk``; evaluation-only)."""

    if pathology is None or (isinstance(pathology, float) and np.isnan(pathology)):
        return {"acdc": None, "compatibility": None, "pathology_group": None}
    entry = (cohort_cfg.get("crosswalk") or {}).get(str(pathology))
    if entry is None:
        raise HumanGateError(f"pathology {pathology!r} has no configs/cohorts.yaml mms2.crosswalk entry")
    return {"acdc": entry.get("acdc"), "compatibility": entry.get("compatibility"), "pathology_group": entry.get("pathology_group")}


def typed_b_registry(registry: Mapping[str, Any], cohort_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The registry with the ``mms2.typed_b_metadata`` overlay of ``metadata_columns`` (contract of MMS2 rows only):
    ``enum_add`` extends an enum, ``required: false`` makes a column nullable.  The frozen registry is never edited."""

    import copy

    out = copy.deepcopy(dict(registry))
    columns = out.setdefault("metadata_columns", {})
    for name, overlay in (cohort_cfg.get("typed_b_metadata") or {}).items():
        spec = dict(columns.get(name) or {})
        if overlay.get("enum_add"):
            spec["enum"] = list(spec.get("enum") or []) + [v for v in overlay["enum_add"] if v not in (spec.get("enum") or [])]
        if "required" in overlay:
            spec["required"] = bool(overlay["required"])
            spec["nullable"] = not bool(overlay["required"])
        columns[name] = spec
    return out


COHORT_COLUMNS = ["sample_id", "dataset", "case_id", "patient_id", "partition", "split_rank", "pathology", "pathology_group",
                  "crosswalk_acdc_class", "crosswalk_compatibility", "vendor", "vendor_release", "vendor_name", "scanner", "field_strength",
                  "eligible_v6", "exclusion_reason", "mms2_external", "duplicate_of", "duplicate_frames", "duplicate_reason_code",
                  "sealed_link", *[f"{role}_{k}" for role in CASE_ROLES for k in ("relpath", "sha256", "bytes")]]


def mms2_cohort_table(source_root: str | Path, manifest: pd.DataFrame | None = None, *, cohort_cfg: Mapping[str, Any],
                      split: pd.DataFrame | None = None, info: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per extracted ``mms2_external`` subject (sorted by ``sample_id``); never a sealed row.

    ``info`` overrides the external CSV of the source area (tests; ``case_id`` holds the raw subject code there, as in
    :func:`info_table`).  Eligibility: the four input files are listed and the release CSV has the subject; duplicate
    exclusions are applied later by ``validate-cohort --dataset mms2``.  ``vendor_name`` is the registry
    ``scanner_vendor`` vocabulary (Siemens, GE or Philips) or null for any other release string.
    """

    split = load_split_manifest() if split is None else split
    manifest = load_file_manifest(source_root, split) if manifest is None else manifest
    if info is None:
        data = read_verified_bytes(source_root, info_entry(manifest))
        ext, _ = parse_release_csv(data, cohort_cfg, external_ids(split), expect_all_subjects=False)
        info = info_table(ext, cohort_cfg)
    info = info.set_index("case_id")
    ranks = split.set_index("subject_id")["rank"]
    files = case_files(manifest)
    registry_vendors = {str(v) for v in cohort_cfg.get("registry_vendor_names") or ("Siemens", "GE", "Philips")}
    rows = []
    for cid, roles in files.items():
        sid = subject_of(cid)
        reason = None
        if set(roles) != set(CASE_ROLES):
            reason = "INPUT_MISSING_ED_ES_FRAME"
        elif sid not in info.index:
            reason = HUMAN_GATE
        meta = info.loc[sid].to_dict() if sid in info.index else {}
        walk = crosswalk(meta.get("pathology"), cohort_cfg)
        vendor = meta.get("vendor")
        row = {"sample_id": sample_id(DATASET, cid), "dataset": DATASET, "case_id": cid, "patient_id": cid, "partition": EXTERNAL,
               "split_rank": int(ranks[sid]), "pathology": meta.get("pathology"), "pathology_group": walk["pathology_group"],
               "crosswalk_acdc_class": walk["acdc"], "crosswalk_compatibility": walk["compatibility"],
               "vendor": vendor, "vendor_release": meta.get("vendor_release"),
               "vendor_name": vendor if vendor in registry_vendors else None,
               "scanner": meta.get("scanner"), "field_strength": meta.get("field_strength"),
               "eligible_v6": reason is None, "exclusion_reason": reason,
               "duplicate_of": None, "duplicate_frames": None, "duplicate_reason_code": None, "sealed_link": None}
        for role in CASE_ROLES:
            entry = roles.get(role, {})
            row.update({f"{role}_relpath": entry.get("relpath"), f"{role}_sha256": entry.get("sha256"), f"{role}_bytes": entry.get("bytes")})
        rows.append(row)
    table = pd.DataFrame(rows, columns=COHORT_COLUMNS)
    table["mms2_external"] = table["eligible_v6"].astype(bool)
    table["split_rank"] = table["split_rank"].astype("Int64")
    return table.sort_values("sample_id").reset_index(drop=True)


# ----------------------------------------------------------------------------- extraction


def archive_expectation(archive: str | Path, receipt: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Expected size and SHA-256: ``registry/sources/mms2.yaml``, or the descriptor beside a synthetic test archive."""

    descriptor = Path(archive).parent / SYNTHETIC_ARCHIVE_DESCRIPTOR
    if is_synthetic_fixture(Path(archive).parent) and descriptor.is_file():
        data = json.loads(descriptor.read_text(encoding="utf-8"))
        return {"sha256": str(data["sha256"]).lower(), "bytes": int(data["bytes"]), "synthetic_fixture": True}
    receipt = load_receipt() if receipt is None else receipt
    arc = receipt["archive"]
    return {"sha256": str(arc["sha256"]).lower(), "bytes": int(arc["bytes"]), "synthetic_fixture": False}


def verify_archive(archive: str | Path, *, sha256: str, size: int) -> dict[str, Any]:
    path = Path(archive)
    if not path.is_file():
        raise SourceError(f"archive not found: {path}")
    observed_size = path.stat().st_size
    if observed_size != int(size):
        raise SourceError(f"archive size {observed_size} != {int(size)}: {path}")
    digest = sha256_file(path)
    if digest != str(sha256).lower():
        raise SourceError(f"archive SHA-256 mismatch: {path}")
    return {"bytes": observed_size, "sha256": digest}


def _relative_files(root: Path) -> dict[str, int]:
    return {p.relative_to(root).as_posix(): p.stat().st_size for p in sorted(root.rglob("*")) if p.is_file()}


def _existing_extraction(dest: Path, expected: Mapping[str, Any], list_sha256: str) -> dict[str, Any] | None:
    try:
        receipt = json.loads((dest / RECEIPT_JSON).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if receipt.get("archive", {}).get("sha256") != expected["sha256"] or receipt.get("list_file", {}).get("sha256") != list_sha256:
        return None
    manifest = pd.read_csv(dest / FILE_MANIFEST, dtype=str, keep_default_na=False)
    for r in manifest.itertuples(index=False):
        path = dest / str(r.relpath)
        if not path.is_file() or path.stat().st_size != int(r.bytes) or sha256_file(path) != str(r.sha256):
            return None
    return receipt


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def extract_external(archive: str | Path, dest: str | Path, *, cohort_cfg: Mapping[str, Any], require_unlocked: Callable[[], Any],
                     tar: str | Path = DEFAULT_TAR, split: pd.DataFrame | None = None, receipt: Mapping[str, Any] | None = None,
                     read_only: bool = True) -> dict[str, Any]:
    """Check the lock, verify the archive, then extract only the ``mms2_external`` short-axis ED/ES files.

    Order: (1) ``require_unlocked()`` (raises before the ``protocol-v6.0.1`` tag; SPECIFICATION.md 4.6: no M&Ms-2
    byte, including the archive hash, is read before the tag); (2) archive size and SHA-256 (``registry/sources/mms2.yaml``); (3) listing and the 720-entry list; (4) release CSV into memory, contract check,
    sealed rows dropped (HUMAN_GATE stops here, before any image is extracted); (5) ``tar -xf -C dest -T list``;
    (6) the extracted files must equal the list and the listing sizes; (7) external CSV copy, file manifest, receipt
    (with ``extraction_started_utc``, taken once the lock has passed, and ``extraction_completed_utc``), read-only
    attributes.  An identical earlier extraction (same archive and list, files verified) is skipped and keeps its times.
    """

    archive, dest = Path(archive), Path(dest)
    lock = require_unlocked()
    started_utc = _utc_now()          # SPECIFICATION 14.6: the dated extraction record that every freeze record must precede
    expected = archive_expectation(archive, receipt)
    observed = verify_archive(archive, sha256=expected["sha256"], size=expected["bytes"])
    split = load_split_manifest() if split is None else split
    external = external_ids(split)
    if len(external) != N_EXTERNAL:
        raise SourceError(f"split manifest lists {len(external)} mms2_external subjects, expected {N_EXTERNAL}")
    entries, listing_sha = list_archive(archive, tar)
    info_name = str(info_contract(cohort_cfg)["name_in_archive"])
    selection = select_members(entries, external, info_csv_name=info_name)
    members = selection["members"]
    if len(members) != 4 * N_EXTERNAL or {m["case_id"] for m in members} != set(external):
        raise SourceError(f"extraction list has {len(members)} entries, expected {4 * N_EXTERNAL}")
    layout = layout_report(selection, (cohort_cfg.get("inputs") or {}).get("archive_layout"))
    list_text = "".join(f"{m['member']}\n" for m in members)
    list_sha = sha256_bytes(list_text.encode("utf-8"))
    if dest.is_dir() and any(dest.iterdir()):
        prior = _existing_extraction(dest, expected, list_sha)
        if prior is not None:
            return {**prior, "action": "SKIPPED"}
        raise SourceError(f"destination {dest} is not empty and holds no matching extraction; it is never overwritten")
    dest.parent.mkdir(parents=True, exist_ok=True)
    info_bytes, read_via = read_member_bytes(archive, selection["info_csv"].member, tar)
    full_csv = {"member": selection["info_csv"].member, "bytes": len(info_bytes), "sha256": sha256_bytes(info_bytes), "read_via": read_via}
    ext, csv_report = parse_release_csv(info_bytes, cohort_cfg, external, expect_all_subjects=True)
    external_copy = external_csv_bytes(ext)
    del info_bytes                                            # the release CSV (with sealed rows) never reaches a file
    dest.mkdir(parents=True, exist_ok=True)
    list_path = dest.parent / f".{dest.name}.extract-list.txt"
    list_path.write_bytes(list_text.encode("utf-8"))
    try:
        done = _tar(tar, ["-xf", str(archive), "-C", str(dest), "-T", str(list_path)])
    finally:
        list_path.unlink(missing_ok=True)
    if done.returncode != 0:
        raise SourceError(f"tar -xf -T failed ({done.returncode}): {done.stderr.decode('utf-8', errors='replace')[:400]}")
    on_disk = _relative_files(dest)
    expected_files = {m["relpath"]: int(m["bytes"]) for m in members}
    extra, absent = sorted(set(on_disk) - set(expected_files)), sorted(set(expected_files) - set(on_disk))
    wrong = sorted(k for k in expected_files if k in on_disk and on_disk[k] != expected_files[k])
    if extra or absent or wrong:
        raise SourceError(f"extraction differs from the list: extra {extra[:5]}, absent {absent[:5]}, size mismatch {wrong[:5]}")
    (dest / INFO_EXTERNAL).write_bytes(external_copy)
    rows = [{"case_id": case_key(m["case_id"]), "role": m["role"], "relpath": m["relpath"], "bytes": int(m["bytes"]),
             "sha256": sha256_file(dest / m["relpath"]), "archive_member": m["member"]} for m in members]
    rows.append({"case_id": INFO_CASE, "role": INFO_ROLE, "relpath": INFO_EXTERNAL, "bytes": len(external_copy),
                 "sha256": sha256_bytes(external_copy), "archive_member": f"{selection['info_csv'].member} (external rows only)"})
    manifest_path = write_file_manifest(dest, rows)
    if expected["synthetic_fixture"]:
        (dest / SYNTHETIC_MARKER).write_text("synthetic fixture for tests; not cohort data\n", encoding="utf-8")
    result = {
        "schema": EXTRACTION_SCHEMA, "dataset": DATASET, "synthetic_fixture": bool(expected["synthetic_fixture"]),
        "archive": {"name": archive.name, **observed}, "lock": lock, "tar": {"path": str(tar), "version": tar_version(tar)},
        "listing": {"sha256": listing_sha, **selection["counts"]}, "layout": layout,
        "list_file": {"entries": len(members), "sha256": list_sha, "rule": "the four short-axis ED/ES files of each mms2_external subject"},
        "byte_check": {"files": len(expected_files), "bytes": int(sum(expected_files.values())), "matches_listing": True},
        "release_csv": {**full_csv, **csv_report, "external_copy": {"relpath": INFO_EXTERNAL, "bytes": len(external_copy),
                                                                     "sha256": sha256_bytes(external_copy)},
                        "sealed_rows": "dropped in memory before any write; their values are never parsed"},
        "sealed": {"partition": SEALED, "subjects": N_SUBJECTS - N_EXTERNAL, "listed_for_extraction": 0,
                   "note": "sealed entries are never named in the list file, so tar never writes them; if the archive is solid, "
                           "libarchive still decodes skipped data internally to advance (nothing is written or returned)"},
        "file_manifest": {"relpath": FILE_MANIFEST, "sha256": sha256_file(manifest_path), "rows": len(rows)},
        "split_manifest_sha256": lf_sha256(SPLIT_MANIFEST) if SPLIT_MANIFEST.is_file() else None,
        "extraction_started_utc": started_utc, "extraction_completed_utc": _utc_now(),
        "action": "RAN",
    }
    (dest / RECEIPT_JSON).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if read_only:
        for path in [dest / r["relpath"] for r in rows] + [manifest_path, dest / RECEIPT_JSON]:
            os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    return result
