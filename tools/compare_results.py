"""Compare a re-run of CINETRA with the frozen results published in results/v6.0.1.

Every aggregate result file of the frozen run is matched with the same path in your run root
(artifacts/<run_id>/ inside the container). JSON values, CSV and Parquet tables are compared value by value:
numbers within a relative tolerance, everything else exactly. Provenance fields that differ between any two runs
(run ids, times, hosts, paths, commits and input digests) are reported separately and never count as differences.

    python tools/compare_results.py <your run root> [--rtol 1e-9] [--rtol-neural 1e-3]

F9 is trained on a GPU, so its outputs (neural_f9/, probes/, data_efficiency/f9/ and the F9 rows of shared tables)
are compared with --rtol-neural; the protocol's cross-hardware tolerance is 1e-3 on the validation cross-entropy.
Needs pandas and pyarrow (both in the project environment). Exit status 0 when no value differs.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "results" / "v6.0.1"
PROVENANCE = re.compile(r"(?i)(^|_)(run_id|time|date|utc|created|started|finished|host|duration|elapsed|seconds|path|"
                        r"root|commit|sha256|digest|hash|version|python|platform|git|command|argv|pid|user)(_|$)")
NEURAL = ("neural_f9/", "probes/", "data_efficiency/f9/")


def close(a, b, rtol: float) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=rtol, abs_tol=rtol * 1e-3)
    return a == b


def compare_json(a, b, rtol: float, path: str = "") -> tuple[int, int, list[str]]:
    """(values compared, provenance fields skipped, differences)."""

    if isinstance(a, dict) and isinstance(b, dict):
        n = skipped = 0
        diffs: list[str] = []
        for key in sorted(set(a) | set(b)):
            if PROVENANCE.search(str(key)):
                skipped += 1
                continue
            if key not in a or key not in b:
                diffs.append(f"{path}/{key}: present in one run only")
                continue
            dn, ds, dd = compare_json(a[key], b[key], rtol, f"{path}/{key}")
            n, skipped, diffs = n + dn, skipped + ds, diffs + dd
        return n, skipped, diffs
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return 1, 0, [f"{path}: {len(a)} against {len(b)} items"]
        n = skipped = 0
        diffs = []
        for i, (x, y) in enumerate(zip(a, b)):
            dn, ds, dd = compare_json(x, y, rtol, f"{path}[{i}]")
            n, skipped, diffs = n + dn, skipped + ds, diffs + dd
        return n, skipped, diffs
    return 1, 0, ([] if close(a, b, rtol) else [f"{path}: {a!r} against {b!r}"])


def compare_table(frozen: Path, mine: Path, rtol: float) -> tuple[int, int, list[str]]:
    import pandas as pd

    read = pd.read_parquet if frozen.suffix == ".parquet" else pd.read_csv
    a, b = read(frozen), read(mine)
    keep = [c for c in a.columns if not PROVENANCE.search(str(c))]
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return 1, 0, [f"shape or columns differ: {a.shape} against {b.shape}"]
    n, _, diffs = compare_json(a[keep].to_dict("records"), b[keep].to_dict("records"), rtol)
    return n, len(a.columns) - len(keep), diffs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_root", type=Path)
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--rtol-neural", type=float, default=1e-3)
    args = ap.parse_args()
    counts = {"identical": 0, "within tolerance": 0, "different": 0, "missing in your run": 0}
    report: list[str] = []
    for frozen in sorted(p for p in FROZEN.rglob("*") if p.is_file()):
        rel = frozen.relative_to(FROZEN).as_posix()
        if rel == "release_manifest.public.json":
            continue
        mine = args.run_root / rel
        if not mine.is_file():
            counts["missing in your run"] += 1
            report.append(f"MISSING    {rel}")
            continue
        if frozen.read_bytes() == mine.read_bytes():
            counts["identical"] += 1
            continue
        rtol = args.rtol_neural if rel.startswith(NEURAL) else args.rtol
        try:
            if frozen.suffix in (".json", ".jsonl"):
                load = (lambda p: [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]) \
                    if frozen.suffix == ".jsonl" else (lambda p: json.loads(p.read_text(encoding="utf-8")))
                n, skipped, diffs = compare_json(load(frozen), load(mine), rtol)
            elif frozen.suffix in (".csv", ".parquet"):
                n, skipped, diffs = compare_table(frozen, mine, rtol)
            else:
                n, skipped, diffs = 1, 0, ["binary content differs"]
        except Exception as exc:                              # noqa: BLE001 - report and continue
            n, skipped, diffs = 1, 0, [f"could not compare: {exc}"]
        if diffs:
            counts["different"] += 1
            report.append(f"DIFFERENT  {rel}: {len(diffs)} of {n} values; first: {diffs[0]}")
        else:
            counts["within tolerance"] += 1
    for line in report:
        print(line)
    print("; ".join(f"{k}: {v}" for k, v in counts.items()))
    return 0 if counts["different"] == 0 and counts["missing in your run"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
