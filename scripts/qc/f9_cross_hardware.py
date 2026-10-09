"""Cross-hardware QC of F9 (SPECIFICATION.md 6.0.1, sections 14.6 step 4 and 14.7).

The F9 fits were trained on Modal (L40S); every reported F9 number comes from local inference (GTX 1080).  This check
re-evaluates, on the local GPU, the wave-1 selection fits of every winning candidate (3 per fitting set and seed) on
their own selection-fold validation patients, with the same deterministic D4-averaged evaluation as training, and
compares the validation cross entropy and macro-F1 with the values Modal recorded at the selected epoch.

Prespecified criteria (fixed before the first run on 3 October 2026): |delta cross entropy| <= 1e-3 and |delta
macro-F1| <= 1e-9 for every checked fit.  The result is a QC record; it changes no model and selects nothing.

Run inside ``tmmri-gpu`` (after ``gpu_isolation_check.ps1`` prints ISOLATED):
    python scripts/qc/f9_cross_hardware.py --run-id <run_id> --fits outputs/cloud/runs_f9w1/<run_id>
        --wave2-specs outputs/cloud/wave2_specs.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

TOL_CROSS_ENTROPY = 1e-3
TOL_MACRO_F1 = 1e-9


def _key(spec: dict) -> tuple:
    cand = spec["candidate"]
    return int(spec["seed"]), str(spec["outer_fold"]), str(cand["unfreeze"]), float(cand["lr_backbone"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--artifact-root", default="artifacts")
    ap.add_argument("--fits", required=True, help="folder holding the downloaded wave-1 <fit_id>/ directories")
    ap.add_argument("--wave2-specs", required=True, help="wave2_specs.json written by select-f9 (the winning candidates)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from tmmri.neural.data import load_npz, npz_path, read_manifest
    from tmmri.neural.f9_config import load_f9_config, training_config
    from tmmri.neural.model import load_f9
    from tmmri.neural.numerics import configure_numerics, environment_record
    from tmmri.neural.selection import selection_folds
    from tmmri.neural.train import classification_summary, predict_logits, resolve_device

    configure_numerics()
    dev = resolve_device(args.device)
    cfg = training_config(load_f9_config())
    classes = list(cfg["class_order"])
    run_root = Path(args.artifact_root) / args.run_id
    cache = run_root / "neural_cache" / "acdc_dev"
    manifest = read_manifest(cache / "manifest.csv")
    rows_by_id = {str(r["sample_id"]): r for r in manifest.to_dict("records")}

    winners = {_key(s) for s in json.loads(Path(args.wave2_specs).read_text(encoding="utf-8"))["specs"]}
    checked = []
    for fit_dir in sorted(p for p in Path(args.fits).iterdir() if (p / "record.json").is_file()):
        record = json.loads((fit_dir / "record.json").read_text(encoding="utf-8"))
        spec = record["spec"]
        if spec.get("selection_fold") in (None, "") or _key(spec) not in winners:
            continue
        rows = manifest[manifest["sample_id"].isin(spec["fitting_patients"])]
        folds = selection_folds(rows, n_folds=int(cfg["selection"]["n_folds"]), classes=classes, salt=str(cfg["selection"]["salt"]))
        val_ids = sorted(folds.loc[folds["selection_fold"] == spec["selection_fold"], "sample_id"])
        items = [load_npz(npz_path(cache, rows_by_id[sid])) for sid in val_ids]
        y = np.array([classes.index(str(rows_by_id[sid]["label"])) for sid in val_ids])
        model = load_f9(fit_dir).to(dev)
        logits, _ = predict_logits(model, items, dev)
        local = classification_summary(logits, y, classes)
        epoch = int(record["selected_epoch"])
        modal = next(e for e in record["epochs"] if int(e["epoch"]) == epoch)
        d_ce = float(local["cross_entropy"]) - float(modal["val_cross_entropy"])
        d_f1 = float(local["macro_f1"]) - float(modal["val_macro_f1"])
        checked.append({"fit_id": record["fit_id"], "seed": spec["seed"], "outer_fold": spec["outer_fold"],
                        "selection_fold": spec["selection_fold"], "n_val": len(val_ids), "selected_epoch": epoch,
                        "modal_cross_entropy": modal["val_cross_entropy"], "local_cross_entropy": local["cross_entropy"],
                        "delta_cross_entropy": d_ce, "modal_macro_f1": modal["val_macro_f1"], "local_macro_f1": local["macro_f1"],
                        "delta_macro_f1": d_f1, "modal_gpu": (record.get("environment") or {}).get("gpu_name")})
        print(f"{record['fit_id']} seed={spec['seed']} fold={spec['outer_fold']}/{spec['selection_fold']} "
              f"dCE={d_ce:+.2e} dF1={d_f1:+.2e}", flush=True)

    max_ce = max((abs(c["delta_cross_entropy"]) for c in checked), default=float("nan"))
    max_f1 = max((abs(c["delta_macro_f1"]) for c in checked), default=float("nan"))
    passed = bool(checked) and max_ce <= TOL_CROSS_ENTROPY and max_f1 <= TOL_MACRO_F1
    out = {"schema": "f9_cross_hardware_qc_v1", "run_id": args.run_id, "utc": datetime.now(UTC).isoformat(timespec="seconds"),
           "criteria": {"abs_delta_cross_entropy_max": TOL_CROSS_ENTROPY, "abs_delta_macro_f1_max": TOL_MACRO_F1},
           "n_checked": len(checked), "expected": 3 * len(winners), "max_abs_delta_cross_entropy": max_ce,
           "max_abs_delta_macro_f1": max_f1, "passed": passed, "local_environment": environment_record(dev),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "fits": checked}
    target = run_root / "neural_f9" / "cross_hardware_qc.json"
    target.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("n_checked", "expected", "max_abs_delta_cross_entropy", "max_abs_delta_macro_f1", "passed")}))
    print(f"wrote {target}")
    return 0 if passed and len(checked) == 3 * len(winners) else 1


if __name__ == "__main__":
    raise SystemExit(main())
