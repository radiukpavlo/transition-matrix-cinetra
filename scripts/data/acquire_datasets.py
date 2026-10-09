#!/usr/bin/env python
"""Acquire, materialize, and verify the project's raw datasets on any machine.

This is the portable, standard-library-only entry point for rebuilding
``data/raw`` from scratch.  It reuses the frozen registry records under
``registry/sources`` and never re-selects cases.

Subcommands (run from the repository root; every step is idempotent and can
simply be re-run after an interruption):

* ``acdc``        download the official ACDC release from the public CREATIS
  Girder server file by file (HTTP Range resume, parallel), or extract a local
  ``database.zip``; then verify all 1,056 release files by size and SHA-256
  against ``acdc_release_file_manifest.csv`` (the zip's own digest changes on
  every download and is never used).
* ``acdc-aria2``  write an aria2c input list (per-file URLs + SHA-256) instead.
* ``mms100``      materialize the frozen 100-case M&Ms subset from an
  authorized copy of the official release (directory, ``.zip`` such as the one
  made by ``pack-mms100``, or the organisers' MEGA folder link), stream
  decompressing ``*.nii.gz`` to the ``*.nii`` layout the pipeline expects and
  verifying source and target hashes from the normalization manifest.
* ``pack-mms100`` on the source machine, bundle only the 200 selected source
  files (+ diagnosis CSV, 5.71 GB) into one stored zip for transfer.
* ``pack-transfer`` / ``verify-transfer`` / ``unpack-transfer``  write, check,
  and restore the physical transfer archive (ACDC release + the 200 M&Ms-100
  source files, deterministic stored zips with SHA256SUMS); see
  ``docs/history/experiment_machine_v5/transfer_archive.md`` (v5 runbook).
* ``link-clinical-baseline``  recreate the two raw-data links of the isolated
  clinical baseline run root (junctions on Windows, symlinks elsewhere).
* ``verify`` / ``status``  re-check everything / print a JSON readiness report.
  ``verify --dataset mms_full [--source <path>]`` verifies the complete M&Ms-1
  release (protocol CMR-TRANS-AUDIT 6.0.0, Decision D1/D2: 345 cases, 690
  NIfTI files plus the information CSV) in place, read-only, against the
  tracked ``mms_full_release_manifest.csv`` and
  ``mms_full_release_metadata_manifest.csv``.  Expected counts and bytes come
  from ``configs/source_bindings/mms.json``, never from constants.  The default
  source is the environment variable ``TMMRI_MMS_SOURCE``, then the binding's
  ``host_default``.  ``--dataset v6`` checks ACDC and the full M&Ms release;
  ``--dataset all`` keeps its historical meaning (ACDC and M&Ms-100).
  ``status`` reports ACDC, M&Ms-100 and the full release; the historical
  ``ready`` flag and exit code still cover only ACDC, M&Ms-100 and the
  clinical-baseline links, and ``v6_ready`` covers ACDC and the full release.

The full release is never copied into the repository: v6 runs mount it
read-only (``/sources/mms`` in the container) and verify-and-stage only the
ED/ES frames (``python -m tmmri.cli stage-sources --dataset mms``).  Official
source: https://www.ub.edu/mnms/ (M&Ms Research Use Agreement: a verbatim
personal research copy is allowed; redistribution of any portion is forbidden;
cite Campello et al., IEEE Trans Med Imaging 2021;40(12):3543-3554).  Kaggle or
other third-party mirrors are not used.  The tracked manifest is rebuilt from
the backup drive with ``scripts/data/build_mms_full_release_manifest.py``.

The registry manifests record absolute Windows paths from the source machine
(``D:\\GitHub\\transition-matrix-mri\\...``).  They are re-rooted here at the
``data/raw`` components, so the manifests stay byte-identical provenance while
verification works from any checkout location or operating system.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import BinaryIO, Callable, ContextManager, Iterable, Iterator


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from maintenance.repository import find_project_root


PROJECT_ROOT = Path(os.environ.get("TMMRI_PROJECT_ROOT", find_project_root(__file__))).resolve()
REGISTRY = PROJECT_ROOT / "registry" / "sources"  # tracked source manifests (ADR-0004)
CHUNK_BYTES = 8 * 1024 * 1024
USER_AGENT = "tmmri-acquire/1.1"

# ACDC: public CREATIS Girder collection 637218c173e9f0047faa00fb, folder "database".
GIRDER_API = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1"
ACDC_FOLDER_ID = "637218e573e9f0047faa00fc"
ACDC_ZIP_URL = f"{GIRDER_API}/folder/{ACDC_FOLDER_ID}/download"
ACDC_ROOT = PROJECT_ROOT / "data" / "raw" / "acdc"
ACDC_ARCHIVE = ACDC_ROOT / ".incoming" / "database.zip"
ACDC_ARCHIVE_BYTES = 2_457_022_167
ACDC_EXPECTED_FILES = 1056
ACDC_EXPECTED_BYTES = 2_456_816_223
ACDC_RELEASE_MANIFEST = REGISTRY / "acdc_release_file_manifest.csv"  # all 1,056 release files

MMS_ROOT = PROJECT_ROOT / "data" / "raw" / "mms"
MMS_NORMALIZATION_MANIFEST = REGISTRY / "mms_100_source_normalization_manifest.csv"
MMS_CHECKSUM_MANIFEST = REGISTRY / "mms_native_checksum_manifest.csv"
MMS_SOURCE_CSV = "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"
MMS_METADATA = MMS_ROOT / "211230_MnMs_Dataset_information_diagnosis_opendataset.csv"
MMS_METADATA_BYTES = 14_139
MMS_METADATA_SHA256 = "612b76738262dcca277e2fe9afba2c1bfea869543fb6af82a68fe30097433fbd"
MMS_EXPECTED_FILES = 200
MMS_EXPECTED_SOURCE_BYTES = 5_712_707_295
MMS_EXPECTED_TARGET_BYTES = 23_255_838_720

# Complete M&Ms-1 release (protocol 6.0.0).  Counts are read from the source binding at run time.
MMS_FULL_MANIFEST = REGISTRY / "mms_full_release_manifest.csv"
MMS_FULL_METADATA_MANIFEST = REGISTRY / "mms_full_release_metadata_manifest.csv"
MMS_SOURCE_BINDING = PROJECT_ROOT / "configs" / "source_bindings" / "mms.json"
MMS_OFFICIAL_SOURCE = "https://www.ub.edu/mnms/"

BASELINE_RAW = PROJECT_ROOT / "clinical_validation" / "runs" / "cv_baseline_v1_20260809" / "data" / "raw"

TRANSFER_ACDC = "acdc_database.zip"
TRANSFER_MMS = "mms100_sources.zip"
TRANSFER_SUMS = "SHA256SUMS.txt"
TRANSFER_MANIFEST = "TRANSFER_MANIFEST.json"
TRANSFER_README = "README.txt"
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


class AcquisitionError(RuntimeError):
    """Raised before an unverified file can be promoted into ``data/raw``."""


@dataclass(frozen=True)
class Expected:
    relative: PurePosixPath
    size_bytes: int
    sha256: str


_LOG_LOCK = threading.Lock()


def log(message: str) -> None:
    with _LOG_LOCK:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            hasher.update(chunk)
    return hasher.hexdigest()


def matches(path: Path, size: int, digest: str | None) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    return digest is None or sha256_file(path) == digest


def reroot(recorded: str) -> PurePosixPath:
    """Map a recorded absolute path to a project-relative ``data/raw/...`` path."""

    parts = PureWindowsPath(recorded).parts if "\\" in recorded else PurePosixPath(recorded).parts
    for index in range(len(parts) - 1):
        if parts[index].lower() == "data" and parts[index + 1].lower() == "raw":
            return PurePosixPath(*parts[index:])
    raise AcquisitionError(f"Recorded path has no data/raw component: {recorded}")


def load_checksum_manifest(path: Path) -> list[Expected]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            Expected(reroot(row["path"]), int(row["size_bytes"]), row["sha256"].strip().lower())
            for row in csv.DictReader(handle)
        ]


def load_normalization_manifest() -> list[dict[str, str]]:
    with MMS_NORMALIZATION_MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != MMS_EXPECTED_FILES:
        raise AcquisitionError(f"normalization manifest must list {MMS_EXPECTED_FILES} files")
    if sum(int(row["source_size_bytes"]) for row in rows) != MMS_EXPECTED_SOURCE_BYTES:
        raise AcquisitionError("normalization manifest source bytes differ from the receipt")
    if sum(int(row["target_size_bytes"]) for row in rows) != MMS_EXPECTED_TARGET_BYTES:
        raise AcquisitionError("normalization manifest target bytes differ from the receipt")
    return rows


def check(expected: Expected, quick: bool, root: Path | None = None) -> str | None:
    """Return an error string, or ``None`` when the file matches (``root`` defaults to the project root)."""

    path = (PROJECT_ROOT if root is None else root) / expected.relative
    if not path.is_file():
        return f"missing: {expected.relative}"
    size = path.stat().st_size
    if size != expected.size_bytes:
        return f"size {size} != {expected.size_bytes}: {expected.relative}"
    if not quick and sha256_file(path) != expected.sha256:
        return f"sha256 mismatch: {expected.relative}"
    return None


def verify_all(items: list[Expected], quick: bool, workers: int, label: str, root: Path | None = None) -> list[str]:
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(check, item, quick, root) for item in items]
        for done, future in enumerate(as_completed(futures), start=1):
            if error := future.result():
                errors.append(error)
            if done % 100 == 0 or done == len(items):
                log(f"{label}: checked {done}/{len(items)} ({len(errors)} problems)")
    return sorted(errors)


def http_get(url: str, headers: dict[str, str] | None = None, timeout: int = 300):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    return urllib.request.urlopen(request, timeout=timeout)


def resumable_download(
    url_for_attempt: Callable[[], str], part: Path, size: int, retries: int, label: str
) -> None:
    """Download into ``part`` with HTTP Range resume; restart when a range is refused."""

    part.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        if have == size:
            return
        if have > size:
            part.unlink()
            have = 0
        try:
            with http_get(url_for_attempt(), {"Range": f"bytes={have}-"} if have else None) as response:
                content_range = response.headers.get("Content-Range") or ""
                resumed = bool(have) and response.status == 206 and content_range.startswith(f"bytes {have}-")
                with part.open("ab" if resumed else "wb") as sink:
                    while chunk := response.read(CHUNK_BYTES):
                        sink.write(chunk)
        except urllib.error.HTTPError as exc:
            wait = 900 if exc.code == 509 else min(300, 5 * 2**attempt)  # 509: MEGA transfer quota
            log(f"{label}: HTTP {exc.code} (attempt {attempt}/{retries}); retrying in {wait} s")
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            wait = min(120, 5 * 2**attempt)
            log(f"{label}: {exc} (attempt {attempt}/{retries}); resuming in {wait} s")
            time.sleep(wait)
    if not part.exists() or part.stat().st_size != size:
        raise AcquisitionError(f"{label}: incomplete after {retries} attempts")


# --------------------------------------------------------------------------- ACDC


def girder_json(path: str, retries: int = 6):
    for attempt in range(retries):
        try:
            with http_get(f"{GIRDER_API}/{path}", timeout=120) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == retries - 1:
                raise AcquisitionError(f"Girder API unreachable ({path}): {exc}") from exc
            time.sleep(2**attempt)


def girder_paged(path: str) -> Iterator[dict]:
    offset, limit = 0, 500
    separator = "&" if "?" in path else "?"
    while True:
        page = girder_json(f"{path}{separator}limit={limit}&offset={offset}&sort=name")
        yield from page
        if len(page) < limit:
            return
        offset += limit


def girder_walk(folder_id: str, relative: str) -> Iterator[dict]:
    for item in girder_paged(f"item?folderId={folder_id}"):
        yield {"item_id": item["_id"], "rel": f"{relative}/{item['name']}", "size": int(item["size"])}
    for folder in girder_paged(f"folder?parentType=folder&parentId={folder_id}"):
        yield from girder_walk(folder["_id"], f"{relative}/{folder['name']}")


def acdc_listing(workers: int) -> list[dict]:
    log("listing the ACDC release on the CREATIS Girder server")
    entries = list(girder_walk(ACDC_FOLDER_ID, "database"))
    total = sum(entry["size"] for entry in entries)
    if len(entries) != ACDC_EXPECTED_FILES or total != ACDC_EXPECTED_BYTES:
        raise AcquisitionError(
            f"server listing ({len(entries)} files, {total} B) differs from the registered release "
            f"({ACDC_EXPECTED_FILES} files, {ACDC_EXPECTED_BYTES} B)"
        )

    def file_id(entry: dict) -> str:
        # /item/{id}/download answers Range with a bogus full-body 206, so only
        # the /file/{id}/download endpoint is safe to resume against.
        files = girder_json(f"item/{entry['item_id']}/files")
        if len(files) != 1:
            raise AcquisitionError(f"expected one file in item {entry['rel']}, found {len(files)}")
        return files[0]["_id"]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for entry, fid in zip(entries, pool.map(file_id, entries)):
            entry["url"] = f"{GIRDER_API}/file/{fid}/download"
    log(f"listing complete: {len(entries)} files, {total / 1e9:.2f} GB")
    return entries


def load_acdc_release_manifest() -> list[Expected]:
    with ACDC_RELEASE_MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        items = [
            Expected(PurePosixPath("data/raw/acdc") / row["relative_path"], int(row["size_bytes"]), row["sha256"])
            for row in csv.DictReader(handle)
        ]
    if len(items) != ACDC_EXPECTED_FILES or sum(item.size_bytes for item in items) != ACDC_EXPECTED_BYTES:
        raise AcquisitionError("acdc_release_file_manifest.csv differs from the registered release")
    return items


def acdc_manifest_by_rel() -> dict[str, Expected]:
    return {str(item.relative.relative_to("data/raw/acdc")): item for item in load_acdc_release_manifest()}


def fetch_acdc_entry(entry: dict, digests: dict[str, Expected], retries: int) -> str:
    target = ACDC_ROOT / entry["rel"]
    expected = digests.get(entry["rel"])
    if matches(target, entry["size"], expected.sha256 if expected else None):
        return "present"
    part = target.with_name(target.name + ".part")
    resumable_download(lambda: entry["url"], part, entry["size"], retries, entry["rel"])
    if expected and sha256_file(part) != expected.sha256:
        part.unlink()
        raise AcquisitionError(f"sha256 mismatch after download: {entry['rel']}")
    part.replace(target)
    return "downloaded"


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> int:
    root = destination.resolve()
    extracted = 0
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if root not in target.parents and target != root:
            raise AcquisitionError(f"archive member escapes destination: {member.filename}")
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if target.is_file() and target.stat().st_size == member.file_size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, target.open("wb") as sink:
            shutil.copyfileobj(source, sink, CHUNK_BYTES)
        extracted += 1
    return extracted


def acdc_from_archive(archive: Path, retries: int, download: bool) -> None:
    if download and not matches(archive, ACDC_ARCHIVE_BYTES, None):
        # The folder endpoint streams a zip built on the fly: no Content-Length,
        # Range ignored, member timestamps set to the download time.  It cannot
        # resume and its SHA-256 differs per download; size is stable.
        log(f"downloading the single ACDC zip ({ACDC_ARCHIVE_BYTES / 1e9:.2f} GB, not resumable)")
        part = archive.with_name(archive.name + ".part")
        part.unlink(missing_ok=True)
        resumable_download(lambda: ACDC_ZIP_URL, part, ACDC_ARCHIVE_BYTES, retries, "database.zip")
        part.replace(archive)
    if download and not matches(archive, ACDC_ARCHIVE_BYTES, None):
        raise AcquisitionError(f"downloaded ACDC archive has the wrong size: {archive}")
    if not archive.is_file() or not zipfile.is_zipfile(archive):
        raise AcquisitionError(f"ACDC archive is missing or not a zip file: {archive}")
    with zipfile.ZipFile(archive) as handle:
        files = [member for member in handle.infolist() if not member.is_dir()]
        if len(files) != ACDC_EXPECTED_FILES or any(not m.filename.startswith("database/") for m in files):
            raise AcquisitionError(f"ACDC archive has {len(files)} files; expected {ACDC_EXPECTED_FILES} under database/")
        # Every extracted file is then verified against acdc_release_file_manifest.csv.
        log(f"extracted {safe_extract(handle, ACDC_ROOT)} new files into {ACDC_ROOT}")


def cmd_acdc(args: argparse.Namespace) -> int:
    if args.archive or args.method == "zip":
        archive = Path(args.archive).resolve() if args.archive else ACDC_ARCHIVE
        acdc_from_archive(archive, args.retries, download=not args.archive)
    else:
        entries = acdc_listing(args.workers)
        digests = acdc_manifest_by_rel()
        counts: dict[str, int] = {}
        failures: list[str] = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fetch_acdc_entry, entry, digests, args.retries): entry for entry in entries}
            for done, future in enumerate(as_completed(futures), start=1):
                try:
                    status = future.result()
                    counts[status] = counts.get(status, 0) + 1
                except AcquisitionError as exc:
                    failures.append(str(exc))
                if done % 100 == 0 or done == len(entries):
                    log(f"acdc: {done}/{len(entries)} files {counts} ({len(failures)} failed)")
        if failures:
            raise AcquisitionError("ACDC download incomplete; re-run to resume:\n  " + "\n  ".join(failures[:20]))
    errors = verify_all(load_acdc_release_manifest(), False, args.workers, "acdc")
    if errors:
        raise AcquisitionError("ACDC verification failed:\n  " + "\n  ".join(errors[:20]))
    if args.delete_archive and ACDC_ARCHIVE.is_file():
        ACDC_ARCHIVE.unlink()
        log("deleted data/raw/acdc/.incoming/database.zip (--delete-archive)")
    log(f"ACDC ready: {ACDC_EXPECTED_FILES}/{ACDC_EXPECTED_FILES} release files verified by SHA-256")
    return 0


def cmd_acdc_aria2(args: argparse.Namespace) -> int:
    entries = acdc_listing(args.workers)
    digests = acdc_manifest_by_rel()
    output = Path(args.output)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for entry in entries:
            handle.write(f"{entry['url']}\n  out={entry['rel']}\n")
            if entry["rel"] in digests:
                handle.write(f"  checksum=sha-256={digests[entry['rel']].sha256}\n")
    log(f"wrote {output} ({len(entries)} URLs); run from the repo root:")
    print(
        f"aria2c -i {output} -d data/raw/acdc -j 8 -x 1 -c --auto-file-renaming=false "
        "--allow-overwrite=true --check-integrity=true --max-tries=20 --retry-wait=10"
    )
    return 0


# ------------------------------------------------------------------------ M&Ms-100


class SourceIndex:
    """Locate ``source_relative_path`` members in a directory or a zip archive."""

    def __init__(self, source: Path) -> None:
        self.source = source
        self.zip_members: dict[str, str] | None = None
        if source.is_file() and zipfile.is_zipfile(source):
            with zipfile.ZipFile(source) as handle:
                self.zip_members = {name.replace("\\", "/"): name for name in handle.namelist()}

    def opener(self, relative: str) -> Callable[[], ContextManager[BinaryIO]]:
        if self.zip_members is None:
            for candidate in (self.source / relative, self.source / "OpenDataset" / relative):
                if candidate.is_file():
                    return lambda path=candidate: path.open("rb")
            raise AcquisitionError(f"source file not found under {self.source}: {relative}")
        found = [name for key, name in self.zip_members.items() if key == relative or key.endswith("/" + relative)]
        if len(found) != 1:
            raise AcquisitionError(f"expected one archive member for {relative}, found {len(found)}")
        member = found[0]

        @contextlib.contextmanager
        def open_member() -> Iterator[BinaryIO]:
            with zipfile.ZipFile(self.source) as archive, archive.open(member) as handle:
                yield handle

        return open_member


class HashingReader:
    def __init__(self, raw: BinaryIO) -> None:
        self.raw = raw
        self.hasher = hashlib.sha256()
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self.raw.read(size)
        self.hasher.update(chunk)
        self.size += len(chunk)
        return chunk

    def readable(self) -> bool:
        return True


def materialize(row: dict[str, str], index: SourceIndex, check_source: bool) -> str:
    target = MMS_ROOT / PurePosixPath(row["target_relative_path"])
    target_size, target_sha = int(row["target_size_bytes"]), row["target_sha256"].strip().lower()
    if matches(target, target_size, target_sha):
        return "already_verified"
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    target_hasher = hashlib.sha256()
    with index.opener(row["source_relative_path"])() as raw:
        reader = HashingReader(raw)
        with gzip.GzipFile(fileobj=reader) as stream, part.open("wb") as sink:  # type: ignore[arg-type]
            while chunk := stream.read(CHUNK_BYTES):
                target_hasher.update(chunk)
                sink.write(chunk)
        while reader.read(CHUNK_BYTES):
            pass
    if check_source and (
        reader.size != int(row["source_size_bytes"])
        or reader.hasher.hexdigest() != row["source_sha256"].strip().lower()
    ):
        part.unlink(missing_ok=True)
        raise AcquisitionError(f"source checksum mismatch: {row['source_relative_path']}")
    if part.stat().st_size != target_size or target_hasher.hexdigest() != target_sha:
        part.unlink(missing_ok=True)
        raise AcquisitionError(f"decompressed checksum mismatch: {row['target_relative_path']}")
    part.replace(target)
    return "materialized"


def ensure_mms_metadata(index: SourceIndex | None) -> None:
    """The project CSV is git-tracked; restore it from the source if a checkout altered it."""

    if matches(MMS_METADATA, MMS_METADATA_BYTES, MMS_METADATA_SHA256):
        return
    if index is not None:
        with contextlib.suppress(AcquisitionError):
            with index.opener(MMS_SOURCE_CSV)() as handle:
                content = handle.read()
            if hashlib.sha256(content).hexdigest() == MMS_METADATA_SHA256:
                MMS_METADATA.write_bytes(content)
                log("restored the M&Ms diagnosis CSV from the source release (line endings were altered)")
                return
    raise AcquisitionError(
        f"{MMS_METADATA.name} is missing or modified (a checkout with core.autocrlf=true converts it; "
        "the committed .gitattributes prevents this on fresh clones)"
    )


def unexpected_mms_niftis(rows: list[dict[str, str]]) -> list[str]:
    wanted = {row["target_relative_path"] for row in rows}
    if not MMS_ROOT.exists():
        return []
    return sorted(
        path.relative_to(MMS_ROOT).as_posix()
        for path in MMS_ROOT.rglob("*.nii*")
        if path.relative_to(MMS_ROOT).as_posix() not in wanted
    )


# MEGA public folder links (the M&Ms organisers have distributed the release via
# MEGA).  Only the 200 selected files are fetched; needs `pip install pycryptodome`.


def _b64decode(text: str) -> bytes:
    text = text.replace("-", "+").replace("_", "/")
    return base64.b64decode(text + "=" * (-len(text) % 4))


def _a32(data: bytes) -> list[int]:
    return list(struct.unpack(f">{len(data) // 4}I", data))


def _mega_api(folder: str, payload: list, retries: int = 10):
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                f"https://g.api.mega.co.nz/cs?id={int(time.time())}&n={folder}",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.load(response)
            if isinstance(result, int) or (isinstance(result, list) and isinstance(result[0], int)):
                raise AcquisitionError(f"MEGA API error {result}")
            return result
        except (urllib.error.URLError, TimeoutError, AcquisitionError) as exc:
            if attempt == retries - 1:
                raise AcquisitionError(f"MEGA API failed: {exc}") from exc
            time.sleep(min(300, 5 * 2**attempt))


def mega_index(url: str) -> tuple[str, dict[str, dict]]:
    try:
        from Crypto.Cipher import AES
    except ImportError as exc:
        raise AcquisitionError("--mega-url needs pycryptodome: python -m pip install pycryptodome") from exc
    match = re.search(r"/folder/([^#?/]+)#([^/?]+)", url)
    if not match:
        raise AcquisitionError("--mega-url must look like https://mega.nz/folder/<handle>#<key>")
    folder, master = match.group(1), _b64decode(match.group(2))
    info: dict[str, dict] = {}
    for node in _mega_api(folder, [{"a": "f", "c": 1, "ca": 1, "r": 1}])[0]["f"]:
        key = _a32(AES.new(master, AES.MODE_ECB).decrypt(_b64decode(node["k"].split(":", 1)[1])))
        file_key = [key[0] ^ key[4], key[1] ^ key[5], key[2] ^ key[6], key[3] ^ key[7]] if node["t"] == 0 else key[:4]
        attributes = AES.new(struct.pack(">4I", *file_key), AES.MODE_CBC, b"\0" * 16).decrypt(_b64decode(node["a"]))
        name = json.loads(attributes.rstrip(b"\0")[4:])["n"]
        info[node["h"]] = {"name": name, "parent": node["p"], "type": node["t"], "size": node.get("s", 0),
                           "key": key, "file_key": file_key, "handle": node["h"]}

    def relative(handle: str) -> str:
        parts = []
        while handle in info:
            parts.append(info[handle]["name"])
            handle = info[handle]["parent"]
        return "/".join(reversed(parts[:-1]))  # drop the shared root folder (OpenDataset)

    return folder, {relative(h): node for h, node in info.items() if node["type"] == 0}


def mega_fetch(folder: str, node: dict, destination: Path, retries: int) -> None:
    from Crypto.Cipher import AES
    from Crypto.Util import Counter

    encrypted = destination.with_name(destination.name + ".enc.part")

    def fresh_url() -> str:
        return _mega_api(folder, [{"a": "g", "g": 1, "n": node["handle"], "ssl": 1}])[0]["g"]

    resumable_download(fresh_url, encrypted, node["size"], retries, destination.name)
    key, file_key = node["key"], node["file_key"]
    counter = Counter.new(64, prefix=struct.pack(">2I", key[4], key[5]), initial_value=0)
    cipher = AES.new(struct.pack(">4I", *file_key), AES.MODE_CTR, counter=counter)
    part = destination.with_name(destination.name + ".part")
    with encrypted.open("rb") as source, part.open("wb") as sink:
        while chunk := source.read(CHUNK_BYTES):
            sink.write(cipher.decrypt(chunk))
    part.replace(destination)
    encrypted.unlink()


def mms_download_from_mega(url: str, rows: list[dict[str, str]], download_root: Path, workers: int, retries: int) -> None:
    folder, index = mega_index(url)
    wanted = {row["source_relative_path"]: int(row["source_size_bytes"]) for row in rows}
    wanted[MMS_SOURCE_CSV] = MMS_METADATA_BYTES
    missing = [path for path in wanted if path not in index]
    if missing:
        raise AcquisitionError(f"{len(missing)} selected files are not in the MEGA folder, e.g. {missing[:3]}")
    pending = [path for path, size in wanted.items() if not matches(download_root / path, size, None)]
    log(f"MEGA folder indexed; {len(pending)}/{len(wanted)} selected files still to download")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(mega_fetch, folder, index[path], download_root / path, retries) for path in pending]
        for done, future in enumerate(as_completed(futures), start=1):
            future.result()
            log(f"mega: {done}/{len(pending)} files downloaded")


def cmd_mms100(args: argparse.Namespace) -> int:
    rows = load_normalization_manifest()
    if args.mega_url:
        download_root = Path(args.source).resolve()
        mms_download_from_mega(args.mega_url, rows, download_root, min(args.workers, 2), args.retries)
    source = Path(args.source).resolve()
    if not source.exists():
        raise AcquisitionError(f"--source does not exist: {source}")
    MMS_ROOT.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(MMS_ROOT).free
    if free < MMS_EXPECTED_TARGET_BYTES * 1.05 and not args.dry_run:
        log(f"warning: {free / 1e9:.1f} GB free; the decompressed subset needs {MMS_EXPECTED_TARGET_BYTES / 1e9:.1f} GB")
    index = SourceIndex(source)
    for row in rows:
        index.opener(row["source_relative_path"])  # fail fast on any missing case
    log(f"all {len(rows)} selected source files located in {source}")
    if args.dry_run:
        return 0
    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(materialize, row, index, not args.skip_source_check) for row in rows]
        for done, future in enumerate(as_completed(futures), start=1):
            status = future.result()
            counts[status] = counts.get(status, 0) + 1
            if done % 10 == 0 or done == len(rows):
                log(f"mms100: {done}/{len(rows)} files {counts}")
    ensure_mms_metadata(index)
    if extra := unexpected_mms_niftis(rows):
        raise AcquisitionError(f"unexpected NIfTI files in data/raw/mms (inventory would change): {extra[:5]}")
    log("M&Ms-100 ready: 200/200 NIfTI files and the diagnosis CSV verified")
    return 0


def cmd_pack_mms100(args: argparse.Namespace) -> int:
    rows = load_normalization_manifest()
    index = SourceIndex(Path(args.source).resolve())
    output = Path(args.output).resolve()
    if output.exists():
        raise AcquisitionError(f"refusing to overwrite {output}")
    part = output.with_name(output.name + ".part")
    members = [(row["source_relative_path"], int(row["source_size_bytes"]), row["source_sha256"]) for row in rows]
    members.append((MMS_SOURCE_CSV, MMS_METADATA_BYTES, MMS_METADATA_SHA256))
    with zipfile.ZipFile(part, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for number, (relative, size, digest) in enumerate(members, start=1):
            info = zipfile.ZipInfo(f"OpenDataset/{relative}", date_time=(1980, 1, 1, 0, 0, 0))
            hasher = hashlib.sha256()
            with index.opener(relative)() as source, archive.open(info, "w", force_zip64=True) as sink:
                while chunk := source.read(CHUNK_BYTES):
                    hasher.update(chunk)
                    sink.write(chunk)
            if hasher.hexdigest() != digest.strip().lower() or info.file_size != size:
                raise AcquisitionError(f"source checksum mismatch while packing: {relative}")
            if number % 20 == 0 or number == len(members):
                log(f"packed {number}/{len(members)} files")
    part.replace(output)
    log(f"wrote {output} ({output.stat().st_size / 1e9:.2f} GB); transfer it and run "
        f"`acquire_datasets.py mms100 --source {output.name}` on the target machine")
    return 0


# ----------------------------------------------------------- transfer archive


def pack_acdc(output: Path) -> None:
    """Write the verified ACDC release as a deterministic stored zip (members under database/)."""

    items = sorted(load_acdc_release_manifest(), key=lambda item: str(item.relative))
    part = output.with_name(output.name + ".part")
    with zipfile.ZipFile(part, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for number, item in enumerate(items, start=1):
            info = zipfile.ZipInfo(str(item.relative.relative_to("data/raw/acdc")), date_time=FIXED_ZIP_TIME)
            hasher = hashlib.sha256()
            with (PROJECT_ROOT / item.relative).open("rb") as source, archive.open(info, "w", force_zip64=True) as sink:
                while chunk := source.read(CHUNK_BYTES):
                    hasher.update(chunk)
                    sink.write(chunk)
            if info.file_size != item.size_bytes or hasher.hexdigest() != item.sha256:
                part.unlink(missing_ok=True)
                raise AcquisitionError(f"local ACDC file differs from the release manifest: {item.relative}")
            if number % 200 == 0 or number == len(items):
                log(f"acdc: packed {number}/{len(items)} files")
    part.replace(output)


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def read_sums(folder: Path) -> dict[str, str]:
    sums = {}
    for line in (folder / TRANSFER_SUMS).read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            sums[name.strip().lstrip("*")] = digest.lower()
    return sums


def cmd_pack_transfer(args: argparse.Namespace) -> int:
    folder = Path(args.output).resolve()
    if folder.exists() and any(folder.iterdir()):
        raise AcquisitionError(f"refusing to write into a non-empty folder: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    needed = ACDC_EXPECTED_BYTES + MMS_EXPECTED_SOURCE_BYTES
    if shutil.disk_usage(folder).free < needed * 1.05:
        raise AcquisitionError(f"need about {needed / 1e9:.1f} GB free at {folder}")
    errors = verify_all(load_acdc_release_manifest(), True, 8, "acdc (pre-pack size check)")
    if errors:
        raise AcquisitionError("local ACDC release is incomplete:\n  " + "\n  ".join(errors[:20]))
    log(f"packing ACDC into {TRANSFER_ACDC}")
    pack_acdc(folder / TRANSFER_ACDC)
    log(f"packing the 200 M&Ms-100 source files into {TRANSFER_MMS}")
    cmd_pack_mms100(argparse.Namespace(source=args.mms_source, output=str(folder / TRANSFER_MMS)))
    contents = []
    for name, restores_to, members in (
        (TRANSFER_ACDC, "data/raw/acdc/database/ (extracted, gzipped NIfTI kept as released)", ACDC_EXPECTED_FILES),
        (TRANSFER_MMS, "data/raw/mms/<split>/<CASE>/<CASE>_sa[_gt].nii (stream-decompressed)", MMS_EXPECTED_FILES + 1),
    ):
        path = folder / name
        contents.append({"file": name, "bytes": path.stat().st_size, "sha256": sha256_file(path),
                         "members": members, "restores_to": restores_to})
    (folder / TRANSFER_SUMS).write_text("".join(f"{c['sha256']}  {c['file']}\n" for c in contents),
                                        encoding="utf-8", newline="\n")
    manifest = {
        "archive": "tmmri raw-data transfer archive",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repository_commit": git_head(),
        "contents": contents,
        "integrity_authorities": [
            "registry/sources/acdc_release_file_manifest.csv",
            "registry/sources/mms_100_source_normalization_manifest.csv",
            "registry/sources/mms_native_checksum_manifest.csv",
        ],
        "restore_command": "python scripts/data/acquire_datasets.py unpack-transfer --source <this folder>",
        "licences": {
            "ACDC": "CC BY-NC-SA 4.0; non-commercial research only; cite Bernard et al. 2018",
            "M&Ms": "research use only; no redistribution of any portion (DUA clause 5); cite Campello et al. 2021",
        },
        "not_included": [
            "data/raw/acdc/.incoming/database.zip (content-identical to acdc_database.zip; its digest is download-specific)",
            "the 490 non-selected M&Ms OpenDataset files (not used; the protocol limits the project to the frozen 100 cases)",
            "anything tracked by git, .venv, outputs/, restricted reader-study snapshots",
        ],
    }
    (folder / TRANSFER_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    readme = [
        "TM-MRI raw-data transfer archive",
        f"Created {manifest['created_utc']} from repository commit {manifest['repository_commit']}.",
        "",
        f"{TRANSFER_ACDC}  ACDC release, {ACDC_EXPECTED_FILES} files -> data/raw/acdc/database/",
        f"{TRANSFER_MMS}  frozen M&Ms-100 sources, 200 .nii.gz + diagnosis CSV -> data/raw/mms/ (decompressed)",
        f"{TRANSFER_SUMS}  SHA-256 of both zips;  {TRANSFER_MANIFEST}  machine-readable details",
        "",
        "Restore from the repository root on the experiment machine (do not extract by hand):",
        "  .venv\\Scripts\\python.exe scripts\\data\\acquire_datasets.py unpack-transfer --source <this folder>",
        "Full instructions: docs/history/experiment_machine_v5/transfer_archive.md in the repository.",
        "",
        "Licences: ACDC CC BY-NC-SA 4.0 (non-commercial). M&Ms: research use only, no redistribution.",
        "Keep this archive private: never upload it to a public or shared service.",
    ]
    (folder / TRANSFER_README).write_text("\n".join(readme) + "\n", encoding="utf-8", newline="\n")
    log(f"transfer archive ready in {folder}")
    for entry in contents:
        print(f"  {entry['sha256']}  {entry['file']}  ({entry['bytes'] / 1e9:.2f} GB)")
    return 0


def check_transfer_folder(folder: Path, deep: bool) -> None:
    missing = [n for n in (TRANSFER_ACDC, TRANSFER_MMS, TRANSFER_SUMS) if not (folder / n).is_file()]
    if missing:
        raise AcquisitionError(f"transfer folder {folder} lacks {missing}")
    sums = read_sums(folder)
    for name in (TRANSFER_ACDC, TRANSFER_MMS):
        log(f"hashing {name}")
        if sha256_file(folder / name) != sums.get(name):
            raise AcquisitionError(f"{name} does not match {TRANSFER_SUMS}: the copy is damaged")
    log(f"both zips match {TRANSFER_SUMS}")
    if not deep:
        return
    expected = {str(i.relative.relative_to("data/raw/acdc")): i for i in load_acdc_release_manifest()}
    with zipfile.ZipFile(folder / TRANSFER_ACDC) as archive:
        for member in archive.infolist():
            item = expected.get(member.filename)
            hasher = hashlib.sha256()
            with archive.open(member) as handle:
                while chunk := handle.read(CHUNK_BYTES):
                    hasher.update(chunk)
            if item is None or member.file_size != item.size_bytes or hasher.hexdigest() != item.sha256:
                raise AcquisitionError(f"ACDC member does not match the release manifest: {member.filename}")
    log(f"deep check: all {len(expected)} ACDC members match the release manifest")
    index = SourceIndex(folder / TRANSFER_MMS)
    for number, row in enumerate(load_normalization_manifest(), start=1):
        target_hasher, target_size = hashlib.sha256(), 0
        with index.opener(row["source_relative_path"])() as raw:
            reader = HashingReader(raw)
            with gzip.GzipFile(fileobj=reader) as stream:  # type: ignore[arg-type]
                while chunk := stream.read(CHUNK_BYTES):
                    target_hasher.update(chunk)
                    target_size += len(chunk)
            while reader.read(CHUNK_BYTES):
                pass
        if (reader.hasher.hexdigest() != row["source_sha256"].strip().lower()
                or target_size != int(row["target_size_bytes"])
                or target_hasher.hexdigest() != row["target_sha256"].strip().lower()):
            raise AcquisitionError(f"M&Ms member does not match the normalization manifest: {row['source_relative_path']}")
        if number % 20 == 0:
            log(f"deep check: {number}/200 M&Ms members verified (source and decompressed SHA-256)")
    log("deep check: all 200 M&Ms members verified")


def cmd_verify_transfer(args: argparse.Namespace) -> int:
    check_transfer_folder(Path(args.source).resolve(), args.deep)
    log("transfer archive verified")
    return 0


def cmd_unpack_transfer(args: argparse.Namespace) -> int:
    folder = Path(args.source).resolve()
    check_transfer_folder(folder, deep=False)
    cmd_acdc(argparse.Namespace(archive=str(folder / TRANSFER_ACDC), method="zip", workers=args.workers,
                                retries=1, delete_archive=False))
    cmd_mms100(argparse.Namespace(source=str(folder / TRANSFER_MMS), mega_url=None, workers=min(args.workers, 4),
                                  retries=1, skip_source_check=False, dry_run=False))
    cmd_link_clinical_baseline(argparse.Namespace())
    return cmd_status(argparse.Namespace())


# ------------------------------------------------------------ links/verify/status


def baseline_links() -> list[tuple[Path, Path]]:
    return [(BASELINE_RAW / name, PROJECT_ROOT / "data" / "raw" / name) for name in ("acdc", "mms")]


def link_ok(link: Path, target: Path) -> bool:
    return os.path.lexists(link) and os.path.realpath(link) == os.path.realpath(target)


def cmd_link_clinical_baseline(args: argparse.Namespace) -> int:
    for link, target in baseline_links():
        if link_ok(link, target):
            log(f"ok: {link.relative_to(PROJECT_ROOT)} -> data/raw/{target.name}")
            continue
        if os.path.lexists(link):
            raise AcquisitionError(f"{link} exists but does not point at {target}; inspect it manually")
        target.mkdir(parents=True, exist_ok=True)
        link.parent.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
        else:
            os.symlink(os.path.relpath(target, link.parent), link, target_is_directory=True)
        log(f"linked {link.relative_to(PROJECT_ROOT)} -> data/raw/{target.name}")
    return 0


def dataset_items() -> dict[str, list[Expected]]:
    return {
        "acdc": load_acdc_release_manifest(),
        "mms100": load_checksum_manifest(MMS_CHECKSUM_MANIFEST),
    }


# ------------------------------------------------------ complete M&Ms-1 release


def mms_binding() -> dict:
    return json.loads(MMS_SOURCE_BINDING.read_text(encoding="utf-8"))


def mms_full_source(explicit: str | None = None) -> Path:
    """Resolve the read-only release root: ``--source``, ``TMMRI_MMS_SOURCE``, then the binding's host default."""

    value = explicit or os.environ.get("TMMRI_MMS_SOURCE") or mms_binding()["mount"]["host_default"]
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def load_mms_full_manifest() -> list[Expected]:
    """Every release file (690 NIfTI + the information CSV), relative to the release root.

    The expected file count and byte total come from ``configs/source_bindings/mms.json``
    (``release_files_observed``), so the new path carries no hard-coded counts.
    """

    items: list[Expected] = []
    with MMS_FULL_MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for kind in ("image", "mask"):
            items.append(Expected(PurePosixPath(row[f"{kind}_relpath"]), int(row[f"{kind}_bytes"]),
                                  row[f"{kind}_sha256"].strip().lower()))
    with MMS_FULL_METADATA_MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            items.append(Expected(PurePosixPath(row["relpath"]), int(row["bytes"]), row["sha256"].strip().lower()))
    observed = mms_binding()["release_files_observed"]
    if len(rows) != int(observed["cases"]) or len(items) != int(observed["files"]):
        raise AcquisitionError(f"{MMS_FULL_MANIFEST.name}: {len(rows)} cases / {len(items)} files differ from "
                               f"{MMS_SOURCE_BINDING.name} ({observed['cases']} / {observed['files']})")
    if sum(item.size_bytes for item in items) != int(observed["bytes_total"]):
        raise AcquisitionError(f"{MMS_FULL_MANIFEST.name}: byte total differs from {MMS_SOURCE_BINDING.name}")
    return items


def unexpected_release_niftis(root: Path, items: list[Expected]) -> list[str]:
    wanted = {str(item.relative) for item in items}
    if not root.is_dir():
        return []
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*.nii*")
                  if p.relative_to(root).as_posix() not in wanted)


def cmd_verify(args: argparse.Namespace) -> int:
    requested = {"all": ["acdc", "mms100"], "v6": ["acdc", "mms_full"]}.get(args.dataset, [args.dataset])
    items = dataset_items() if any(name in ("acdc", "mms100") for name in requested) else {}
    mode = "size-only" if args.quick else "size+sha256"
    failed = False
    for name in requested:
        root: Path | None = None
        if name == "mms_full":
            root = mms_full_source(getattr(args, "source", None))
            if not root.is_dir():
                failed = True
                log(f"mms_full: FAILED source folder not found: {root} (set --source or TMMRI_MMS_SOURCE; "
                    f"official source {MMS_OFFICIAL_SOURCE})")
                continue
            expected = load_mms_full_manifest()
            log(f"mms_full: verifying {len(expected)} release files in place under {root} (read-only)")
        else:
            expected = items[name]
        errors = verify_all(expected, args.quick, args.workers, name, root)
        if name == "mms_full":
            errors += [f"unexpected: {path}" for path in unexpected_release_niftis(root, expected)]
        if name == "mms100":
            errors += [f"unexpected: data/raw/mms/{path}" for path in unexpected_mms_niftis(load_normalization_manifest())]
        if errors:
            failed = True
            log(f"{name}: FAILED {len(errors)} problems ({mode})")
            for error in errors[:20]:
                print(f"  {error}")
        else:
            log(f"{name}: OK {len(expected)}/{len(expected)} ({mode})")
    return 1 if failed else 0


def cmd_status(args: argparse.Namespace) -> int:
    report: dict[str, dict] = {}
    for name, items in dataset_items().items():
        present = sum(check(item, quick=True) is None for item in items)
        report[name] = {
            "expected_files": len(items),
            "present_with_expected_size": present,
            "expected_bytes": sum(item.size_bytes for item in items),
            "ready": present == len(items),
        }
    report["acdc"]["archive_present"] = ACDC_ARCHIVE.is_file()
    links = {str(link.relative_to(PROJECT_ROOT).as_posix()): link_ok(link, target) for link, target in baseline_links()}
    ready = all(entry["ready"] for entry in report.values()) and all(links.values())
    # Protocol 6.0.0 sources: the complete M&Ms-1 release is verified in place (never copied into data/raw).
    source = mms_full_source(getattr(args, "source", None))
    full: dict[str, object] = {"source_root": str(source), "source_present": source.is_dir(),
                               "official_source": MMS_OFFICIAL_SOURCE, "redistribution": "forbidden"}
    try:
        items = load_mms_full_manifest()
        present = sum(check(item, quick=True, root=source) is None for item in items) if source.is_dir() else 0
        full.update({"expected_files": len(items), "present_with_expected_size": present,
                     "expected_bytes": sum(item.size_bytes for item in items), "ready": present == len(items)})
    except (OSError, KeyError, ValueError, AcquisitionError) as exc:
        full.update({"ready": False, "error": str(exc)})
    v6_ready = bool(report["acdc"]["ready"] and full["ready"])
    print(json.dumps({"project_root": str(PROJECT_ROOT), "datasets": report, "clinical_baseline_links": links,
                      "ready": ready, "v6_sources": {"acdc": report["acdc"]["ready"], "mms_full": full},
                      "v6_ready": v6_ready}, indent=2))
    return 0 if ready else 3


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    acdc = sub.add_parser("acdc", help="download (per file, resumable) and verify ACDC")
    acdc.add_argument("--method", choices=("files", "zip"), default="files",
                      help="files: resumable per-file Girder download (default); zip: single non-resumable archive")
    acdc.add_argument("--archive", help="extract an existing local database.zip instead of downloading")
    acdc.add_argument("--workers", type=int, default=6)
    acdc.add_argument("--retries", type=int, default=20)
    acdc.add_argument("--delete-archive", action="store_true", help="remove data/raw/acdc/.incoming/database.zip")
    acdc.set_defaults(func=cmd_acdc)

    aria = sub.add_parser("acdc-aria2", help="write an aria2c input list for ACDC instead of downloading")
    aria.add_argument("--output", default="acdc_aria2.txt")
    aria.add_argument("--workers", type=int, default=8)
    aria.set_defaults(func=cmd_acdc_aria2)

    mms = sub.add_parser("mms100", help="materialize the frozen M&Ms-100 subset from an authorized release")
    mms.add_argument("--source", required=True,
                     help="OpenDataset directory or .zip (with --mega-url: directory to download the 200 files into)")
    mms.add_argument("--mega-url", help="organisers' MEGA folder link; downloads only the 200 selected files")
    mms.add_argument("--workers", type=int, default=4)
    mms.add_argument("--retries", type=int, default=40)
    mms.add_argument("--skip-source-check", action="store_true", help="do not verify source .nii.gz hashes")
    mms.add_argument("--dry-run", action="store_true", help="only confirm that all 200 source files exist")
    mms.set_defaults(func=cmd_mms100)

    pack = sub.add_parser("pack-mms100", help="bundle the 200 selected M&Ms source files into one zip for transfer")
    pack.add_argument("--source", required=True, help="OpenDataset directory (e.g. data/raw/tmp)")
    pack.add_argument("--output", required=True, help="new .zip path (about 5.71 GB)")
    pack.set_defaults(func=cmd_pack_mms100)

    pack_t = sub.add_parser("pack-transfer", help="write the physical transfer archive (ACDC + M&Ms-100 sources)")
    pack_t.add_argument("--output", required=True, help="new or empty folder, e.g. D:\\tmmri_transfer_<date>")
    pack_t.add_argument("--mms-source", default=str(PROJECT_ROOT / "data" / "raw" / "tmp"),
                        help="authorized OpenDataset directory or zip (default: data/raw/tmp)")
    pack_t.set_defaults(func=cmd_pack_transfer)

    verify_t = sub.add_parser("verify-transfer", help="check a transfer archive folder without writing data/raw")
    verify_t.add_argument("--source", required=True, help="transfer archive folder")
    verify_t.add_argument("--deep", action="store_true",
                          help="also hash every member and every decompressed M&Ms file (no disk writes)")
    verify_t.set_defaults(func=cmd_verify_transfer)

    unpack_t = sub.add_parser("unpack-transfer", help="restore data/raw from a transfer archive folder and verify")
    unpack_t.add_argument("--source", required=True, help="transfer archive folder")
    unpack_t.add_argument("--workers", type=int, default=6)
    unpack_t.set_defaults(func=cmd_unpack_transfer)

    links = sub.add_parser("link-clinical-baseline", help="recreate cv_baseline_v1_20260809 data/raw links")
    links.set_defaults(func=cmd_link_clinical_baseline)

    verify = sub.add_parser("verify", help="verify data/raw (and the full M&Ms release in place) against the registry manifests")
    verify.add_argument("--dataset", choices=("all", "acdc", "mms100", "mms_full", "v6"), default="all",
                        help="all: ACDC + M&Ms-100 (historical); mms_full: complete M&Ms-1 release; v6: ACDC + mms_full")
    verify.add_argument("--source", default=None,
                        help="mms_full only: release root (default TMMRI_MMS_SOURCE, then configs/source_bindings/mms.json)")
    verify.add_argument("--quick", action="store_true", help="check sizes only")
    verify.add_argument("--workers", type=int, default=int(os.environ.get("TMMRI_SOURCE_READERS", "0") or 0) or 8,
                        help="parallel readers (default TMMRI_SOURCE_READERS or 8)")
    verify.set_defaults(func=cmd_verify)

    status = sub.add_parser("status", help="print a JSON readiness report (exit 3 when incomplete)")
    status.add_argument("--source", default=None, help="full M&Ms release root to report (default TMMRI_MMS_SOURCE)")
    status.set_defaults(func=cmd_status)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return args.func(args)
    except AcquisitionError as exc:
        log(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
