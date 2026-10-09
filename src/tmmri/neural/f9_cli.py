"""CLI subcommands of the F9 lane (contract C1): ``plan-f9``, ``train-f9``, ``select-f9``, ``import-f9-fits``,
``freeze-f9`` and ``upload-cache``.  :func:`add_f9_parsers` and :func:`run_f9` are wired into :mod:`tmmri.cli`; the
stage code is imported lazily, so the parser never imports torch.  Exit codes follow the v6 commands (1 usage or
missing input, 5 CPE-1 refusal, 7 frozen bundle or once-only refusal)."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

F9_COMMANDS = ("plan-f9", "train-f9", "select-f9", "import-f9-fits", "freeze-f9", "upload-cache")


def add_f9_parsers(sub: argparse._SubParsersAction, common: Callable[[argparse.ArgumentParser], None]) -> None:
    p = sub.add_parser("plan-f9", help="v6.0.1 F9: wave-1 selection specs (6 fitting sets x 3 seeds x 6 candidates x 3 selection folds = 324)")
    common(p)

    p = sub.add_parser("train-f9", help="v6.0.1 F9: train a spec file sequentially on this machine (local fallback; Modal: scripts/cloud/modal_train.py)")
    p.add_argument("--backend", default="local", choices=["local"], help="local only; the Modal backend runs through scripts/cloud/modal_train.py")
    p.add_argument("--specs", required=True, help="spec file, e.g. neural_f9/plan/wave1_specs.json (relative to the run root)")
    p.add_argument("--device", default="auto", help="auto (cuda when available), cuda or cpu")
    common(p)

    p = sub.add_parser("select-f9", help="v6.0.1 F9: selection rule on the wave-1 records -> 18 wave-2 refit specs")
    p.add_argument("--specs", default="neural_f9/plan/wave1_specs.json", help="wave-1 spec file (relative to the run root)")
    common(p)

    p = sub.add_parser("import-f9-fits", help="v6.0.1 F9: copy and SHA-256-verify downloaded fit directories into neural_f9/checkpoints/")
    p.add_argument("--source", required=True, help="folder holding <fit_id>/ directories downloaded from the runs Volume")
    p.add_argument("--wave", type=int, required=True, choices=[1, 2])
    p.add_argument("--specs", default=None, help="spec file of the wave (default neural_f9/plan/wave<N>_specs.json; S5: data_efficiency/f9/train_wave<N>_specs.json)")
    common(p)

    p = sub.add_parser("freeze-f9", help="v6.0.1 F9: hash every selected checkpoint into neural_f9/f9_bundle.json (seed 20261011 primary; no best-seed choice)")
    common(p)

    p = sub.add_parser("upload-cache", help="v6.0.1 F9: CPE-1 upload of neural_cache/acdc_dev to the Modal Volume tmmri-neural-cache (ledgered)")
    p.add_argument("--file", action="append", default=[], help="upload only this permitted file; repeat as needed (default: the whole cache)")
    p.add_argument("--dry-run", action="store_true", help="check CPE-1 and print the commands; upload and record nothing")
    p.add_argument("--create-volume", action="store_true", help="create the cache Volume first (ledgered)")
    p.add_argument("--record-deletion", default=None, metavar="VOLUME",
                   help="only append a delete_volume row for a Volume the owner deleted (no Modal call)")
    common(p)


def run_f9(args: argparse.Namespace, make_context: Callable[[argparse.Namespace], Any]) -> int:
    from .. import cli, stages
    from . import f9_stages
    from .cloud import CloudPolicyError

    command = args.command
    try:
        ctx = make_context(args)
        if command == "plan-f9":
            result = f9_stages.plan_f9(ctx)
        elif command == "train-f9":
            result = f9_stages.train_f9(ctx, args.specs, backend=args.backend, device=args.device)
        elif command == "select-f9":
            result = f9_stages.select_f9(ctx, args.specs)
        elif command == "import-f9-fits":
            result = f9_stages.import_f9_fits(ctx, args.source, wave=args.wave, specs=args.specs)
        elif command == "freeze-f9":
            result = f9_stages.freeze_f9(ctx)
        elif command == "upload-cache":
            if args.record_deletion:
                result = f9_stages.record_volume_deletion(ctx, args.record_deletion)
            else:
                result = f9_stages.upload_cache(ctx, files=args.file or None, dry_run=args.dry_run, create_volume=args.create_volume)
        else:
            raise SystemExit(f"Unknown command: {command}")
        cli._emit_json(cli._stage_summary(result))
        return 0
    except CloudPolicyError as exc:
        print(f"REFUSED (CPE-1): {exc}")
        return cli.EXIT_CONTRACT_VIOLATION
    except stages.BundleError as exc:
        print(f"BUNDLE: {exc}")
        return cli.EXIT_BUNDLE
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}")
        return cli.EXIT_USAGE_OR_INPUT
