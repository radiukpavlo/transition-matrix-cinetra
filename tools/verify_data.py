"""Check your copies of the three datasets against the registered manifests before running the pipeline.

The datasets are not redistributed here; obtain them from their organizers (docs/REPRODUCTION.md). This script
compares every file of your copies with the size and SHA-256 recorded in the registered manifests
(registry/sources/), which tools/verify_registration.py shows to be byte-identical to the registered commit.

    python tools/verify_data.py --acdc <ACDC 'database' folder>
    python tools/verify_data.py --mms <complete M&Ms-1 release folder>
    python tools/verify_data.py --mms2 <MnM2.rar>
    python tools/verify_data.py --split        # recompute the M&Ms-2 hash split (needs the tmmri package)

Standard library only, except --split. Exit status 0 when every requested check passes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "registry" / "sources"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_files(base: Path, rows: list[tuple[str, int, str]], label: str) -> bool:
    def one(row: tuple[str, int, str]) -> str | None:
        rel, size, digest = row
        path = base / rel
        if not path.is_file():
            return f"missing: {rel}"
        if path.stat().st_size != size:
            return f"size differs: {rel}"
        if sha256_file(path) != digest:
            return f"SHA-256 differs: {rel}"
        return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        problems = [p for p in pool.map(one, rows) if p]
    for p in problems[:20]:
        print(f"  {p}")
    print(f"{label}: {len(rows) - len(problems)} of {len(rows)} files match"
          + ("" if not problems else f"; {len(problems)} problem(s)"))
    return not problems


def acdc(folder: Path) -> bool:
    with (SOURCES / "acdc_release_file_manifest.csv").open(encoding="utf-8", newline="") as fh:
        rows = [(r["relative_path"].removeprefix("database/"), int(r["size_bytes"]), r["sha256"]) for r in csv.DictReader(fh)]
    return check_files(folder, rows, "ACDC")


def mms(folder: Path) -> bool:
    rows: list[tuple[str, int, str]] = []
    with (SOURCES / "mms_full_release_manifest.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            rows += [(r["image_relpath"], int(r["image_bytes"]), r["image_sha256"]),
                     (r["mask_relpath"], int(r["mask_bytes"]), r["mask_sha256"])]
    with (SOURCES / "mms_full_release_metadata_manifest.csv").open(encoding="utf-8", newline="") as fh:
        rows += [(r["relpath"], int(r["bytes"]), r["sha256"]) for r in csv.DictReader(fh)]
    return check_files(folder, rows, "M&Ms-1")


def mms2(archive: Path) -> bool:
    expected = json.loads((ROOT / "configs" / "source_bindings" / "mms2.json").read_text(encoding="utf-8"))["archive"]
    ok = archive.stat().st_size == int(expected["bytes"]) and sha256_file(archive) == expected["sha256"]
    print(f"M&Ms-2: {archive.name} {'matches' if ok else 'DIFFERS from'} the registered archive "
          f"({expected['bytes']} bytes, SHA-256 {expected['sha256'][:16]}...)")
    return ok


def split() -> bool:
    sys.path.insert(0, str(ROOT / "src"))
    from tmmri.ingest.mms2 import build_split_manifest

    rebuilt = build_split_manifest().to_csv(index=False, lineterminator="\n").encode("utf-8")
    ok = rebuilt == (SOURCES / "mms2_split_manifest.csv").read_bytes()
    print(f"M&Ms-2 split: the registered rule {'reproduces' if ok else 'does NOT reproduce'} "
          "registry/sources/mms2_split_manifest.csv byte for byte")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--acdc", type=Path, help="the ACDC 'database' folder (training/, testing/, ...)")
    ap.add_argument("--mms", type=Path, help="the complete M&Ms-1 release folder")
    ap.add_argument("--mms2", type=Path, help="the M&Ms-2 archive MnM2.rar")
    ap.add_argument("--split", action="store_true", help="recompute the M&Ms-2 hash split")
    args = ap.parse_args()
    checks = [(args.acdc, acdc), (args.mms, mms), (args.mms2, mms2)]
    results = [fn(arg) for arg, fn in checks if arg is not None]
    if args.split:
        results.append(split())
    if not results:
        ap.print_help()
        return 2
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
