"""CLI subcommands of lane L57 (protocol-v6.0.1 items 11-14), dispatched by :func:`tmmri.cli._run_v6`.

``fit-probes``, ``evaluate-alt-bundles --split``, ``fit-trust``, ``evaluate-trust --split`` and
``build-comparison-tables`` share the v6 options (``--run-id``, roots, ``--force``, ``--allow-unfrozen-synthetic``) and
the v6 exit codes (3 before the protocol freeze for M&Ms, 7 for bundle and once-only refusals).
"""

from __future__ import annotations

import argparse
from typing import Any, Callable

L57_COMMANDS = ("fit-probes", "evaluate-alt-bundles", "fit-trust", "evaluate-trust", "build-comparison-tables")
ALT_SPLITS = ("acdc_dev", "acdc_test", "mms_external", "mms2_external")
TRUST_SPLITS = ("acdc_test", "mms_external", "mms2_external")
BUNDLES = ("d20", "f9", "f9b")


def add_l57_parsers(sub: argparse._SubParsersAction, common: Callable[[argparse.ArgumentParser], None]) -> None:
    p = sub.add_parser("fit-probes", help="v6.0.1 F9b: v6 typed ridge T per outer fold on encoder k's 64-d activations (acdc_dev OOF) and the final T")
    p.add_argument("--seed", type=int, action="append", default=[], help="F9 seed (repeatable; default configs/probes.yaml f9b.seeds, primary first)")
    common(p)

    p = sub.add_parser("evaluate-alt-bundles", help="v6.0.1 D20, F9, F9b once per (bundle, partition): rho, rho_j, predictor metrics, paired "
                                                    "differences against the descriptor pipeline")
    p.add_argument("--split", required=True, choices=list(ALT_SPLITS))
    p.add_argument("--bundle", action="append", default=[], choices=list(BUNDLES), help="bundle (repeatable; default all configured)")
    common(p)

    p = sub.add_parser("fit-trust", help="v6.0.1 S1/S2/S4: thresholds, Mahalanobis references and temperatures on acdc_dev out-of-fold outputs (frozen by hash)")
    p.add_argument("--allow-missing", action="store_true", help="record predictors or representations without recorded inputs as NE (not executed)")
    common(p)

    p = sub.add_parser("evaluate-trust", help="v6.0.1 S1-S4 once per partition with the frozen trust bundle")
    p.add_argument("--split", required=True, choices=list(TRUST_SPLITS))
    p.add_argument("--allow-missing", action="store_true", help="report predictors or representations without recorded inputs as NE")
    common(p)

    p = sub.add_parser("build-comparison-tables", help="v6.0.1 Table 4 and the Figure-5 parquet from the recorded evaluations (read-only; NE with reasons)")
    common(p)


def run_l57_command(ctx: Any, args: argparse.Namespace) -> Any:
    from . import l57_stages as l57

    command = args.command
    if command == "fit-probes":
        return l57.fit_probes(ctx, seeds=list(args.seed) or None)
    if command == "evaluate-alt-bundles":
        return {bundle: _summary(result) for bundle, result in l57.evaluate_alt_bundles(ctx, args.split, bundles=list(args.bundle) or None).items()}
    if command == "fit-trust":
        return l57.fit_trust(ctx, allow_missing=args.allow_missing)
    if command == "evaluate-trust":
        return l57.evaluate_trust(ctx, args.split, allow_missing=args.allow_missing)
    if command == "build-comparison-tables":
        return l57.build_comparison_tables(ctx)
    raise SystemExit(f"Unknown command: {command}")


def _summary(result: Any) -> Any:
    if hasattr(result, "record") and hasattr(result, "action"):
        return {"stage": result.stage, "action": result.action, "outputs": result.record.outputs, "status": result.record.status, "value": result.value}
    return result
