"""S5 data-efficiency subsampler and F9 fit specs (protocol-v6.0.1 item 15; configs/data_centric.yaml data_efficiency).

Synthetic identifiers only (algorithm controls, never cohort data): subsample determinism, class stratification and
fractions, development-only enforcement, NE_INSUFFICIENT_CLASS_SUPPORT, the C5 spec counts 540 + 30 = 570 and the
NEURAL-TM-1.0 4.2 selection rule."""
from __future__ import annotations

import hashlib
import math
import sys
import types

import numpy as np
import pandas as pd
import pytest

from tmmri import cli
from tmmri.config import load_study
from tmmri.experiments import data_efficiency as de
from tmmri.experiments.data_centric import NE_INSUFFICIENT_CLASS_SUPPORT, DevelopmentOnlyError, data_centric_config
from tmmri.runs import canonical_json
from tmmri.splits import hash_rank_folds

CLASSES = ["DCM", "HCM", "MINF", "NOR", "RV"]


@pytest.fixture(scope="module")
def cfg():
    return data_centric_config(load_study())["data_efficiency"]


def synthetic_dev(per_class=(20, 20, 19, 19, 20)) -> pd.DataFrame:
    """98 synthetic development identifiers with the v6 outer-fold rule (hash folds within class)."""
    rows, n = [], 1
    for cls, count in zip(CLASSES, per_class):
        for _ in range(count):
            rows.append({"sample_id": f"ACDC:patient{n:03d}", "patient_id": f"patient{n:03d}", "acdc_class": cls})
            n += 1
    frame = pd.DataFrame(rows)
    frame["outer_fold"] = hash_rank_folds(frame, "patient_id", "acdc_class", "acdc-fold-v6|", 5).astype(int)
    return frame


def test_subsample_hash_rule_is_deterministic_and_order_free(cfg):
    dev = synthetic_dev()
    fitting = dev[dev["outer_fold"] != 1]
    salt = cfg["subsample"]["salt"]
    first = de.class_stratified_subsample(fitting, fraction=0.5, repeat=0, salt=salt)
    again = de.class_stratified_subsample(fitting.sample(frac=1.0, random_state=7), fraction=0.5, repeat=0, salt=salt)
    pd.testing.assert_frame_equal(first, again)
    assert salt == "lc-v6.0.1|" and de.fraction_token(0.5) == "0.5" and de.fraction_token(1) == "1.0"
    row = first.iloc[0]
    assert cfg["subsample"]["key"] == "patient_id"                     # owner decision D: the canonical patient key
    assert row["digest"] == hashlib.sha256(f"lc-v6.0.1|0.5|0|{row['patient_id']}".encode()).hexdigest()
    other = de.class_stratified_subsample(fitting, fraction=0.5, repeat=1, salt=salt)
    assert set(other.loc[other["selected"], "sample_id"]) != set(first.loc[first["selected"], "sample_id"])
    within = first[first["acdc_class"] == "DCM"].sort_values("rank")
    assert list(within["digest"]) == sorted(within["digest"])          # ranked by the hash within class


@pytest.mark.parametrize("fraction", [0.5, 0.75, 1.0])
def test_subsample_is_class_stratified_with_round_half_up_counts(cfg, fraction):
    dev = synthetic_dev()
    table = de.subsample_table(dev, cfg, fractions=[fraction])
    assert sorted(table["repeat"].unique()) == ([0] if fraction == 1.0 else [0, 1, 2])
    for (r, k), group in table.groupby(["repeat", "heldout_fold"]):
        assert not (group["outer_fold"] == k).any()                      # never a held-out row
        assert set(group["sample_id"]) == set(dev.loc[dev["outer_fold"] != k, "sample_id"])
        for cls, members in group.groupby("acdc_class"):
            n = len(members)
            assert members["selected"].sum() == math.floor(fraction * n + 0.5)
    if fraction == 1.0:
        assert table["selected"].all()
    assert de.per_class_count(0.5, 15) == 8 and de.per_class_count(0.75, 15) == 11 and de.per_class_count(0.75, 16) == 12


def test_development_only_refuses_any_other_partition(cfg):
    split = pd.DataFrame({"sample_id": ["ACDC:patient001", "ACDC:patient101", "MMS:A1A1A1"],
                          "partition": ["acdc_dev", "acdc_test", "mms_external"]})
    de.assert_development_only(["ACDC:patient001"], split, cfg)
    for bad in (["ACDC:patient101"], ["MMS:A1A1A1"], ["ACDC:unknown"]):
        with pytest.raises(DevelopmentOnlyError, match="development only"):
            de.assert_development_only(bad, split, cfg)
    assert "acdc_test" in cfg["forbidden_partitions"] and "mms2_external" in cfg["forbidden_partitions"]


def _records(specs, seed=3):
    """Deterministic synthetic FitRecords for wave-1 specs (never real training results)."""
    rng = np.random.default_rng(seed)
    out = {}
    for s in specs:
        out[s["fit_id"]] = {"fit_id": s["fit_id"], "val_metrics": {"macro_f1": float(rng.uniform(0.2, 0.9)), "cross_entropy": float(rng.uniform(0.5, 1.5))},
                            "selected_epoch": int(rng.integers(20, 81))}
    return out


def test_f9_spec_counts_are_540_selection_plus_30_refits(cfg):
    assert cfg["f9"]["expected_fits"] == {"wave1_selection": 540, "wave2_refit": 30, "total": 570}
    dev = synthetic_dev()
    wave1, units, source = de.wave1_specs(dev, cfg)
    assert source == "tmmri.neural.fit_spec" and len(units) == 30 and all(u["ok"] for u in units)   # C5: the F9 lane's FitSpec
    assert len(wave1) == 540 and len({s["fit_id"] for s in wave1}) == 540
    for s in wave1:
        assert set(s) == {"fit_id", *de.SPEC_FIELDS}
        assert s["purpose"] == "s5" and s["seed"] == 20261011 and s["refit_epochs"] is None and s["selection_fold"] in (1, 2, 3)
        assert s["fraction"] in (0.5, 0.75) and s["repeat"] in (0, 1, 2) and s["outer_fold"] in range(1, 6)
        assert s["fitting_patients"] == sorted(s["fitting_patients"])
        assert s["fit_id"] == hashlib.sha256(canonical_json({k: s[k] for k in de.SPEC_FIELDS})).hexdigest()[:16]
        held = set(dev.loc[dev["outer_fold"] == s["outer_fold"], "sample_id"])
        assert not held & set(s["fitting_patients"])
    cands = {(s["candidate"]["unfreeze"], s["candidate"]["lr_backbone"]) for s in wave1}
    assert cands == {(u, lr) for u in ("layer4", "layer3+layer4") for lr in (1e-5, 3e-5, 1e-4)}
    wave2, ledger, _ = de.wave2_specs(wave1, _records(wave1), cfg)
    assert len(wave2) == 30 and len(wave1) + len(wave2) == 570 and len(ledger) == 30
    for s in wave2:
        assert s["selection_fold"] is None and isinstance(s["refit_epochs"], int) and s["refit_epochs"] >= 20
    again, _, _ = de.wave1_specs(dev.sample(frac=1.0, random_state=1), cfg)
    assert [s["fit_id"] for s in again] == [s["fit_id"] for s in wave1]          # deterministic


def test_wave2_refuses_missing_selection_records(cfg):
    wave1, _, _ = de.wave1_specs(synthetic_dev(), cfg)
    records = _records(wave1)
    records.pop(wave1[7]["fit_id"])
    with pytest.raises(FileNotFoundError, match="no usable FitRecord"):
        de.wave2_specs(wave1, records, cfg)


def test_selection_rule_tie_breaks_and_median_epoch_refit():
    def rows(unfreeze, lr, f1s, ces, epochs):
        return [{"candidate": {"unfreeze": unfreeze, "lr_backbone": lr}, "macro_f1": f, "cross_entropy": c, "epoch": e}
                for f, c, e in zip(f1s, ces, epochs)]
    best = de.select_candidate(rows("layer4", 1e-4, [0.8, 0.8, 0.8], [1.0, 1.0, 1.0], [30, 41, 50])
                               + rows("layer3+layer4", 1e-5, [0.9, 0.9, 0.9], [2.0, 2.0, 2.0], [20, 21, 22]))
    assert best["candidate"] == {"unfreeze": "layer3+layer4", "lr_backbone": 1e-5} and best["refit_epochs"] == 21   # macro-F1 first
    tie_f1 = de.select_candidate(rows("layer4", 1e-4, [0.8] * 3, [1.2] * 3, [30, 30, 31]) + rows("layer4", 3e-5, [0.8] * 3, [1.1] * 3, [40, 41, 43]))
    assert tie_f1["candidate"]["lr_backbone"] == 3e-5 and tie_f1["refit_epochs"] == 41                                # then cross entropy
    tie_ce = de.select_candidate(rows("layer3+layer4", 1e-5, [0.8] * 3, [1.0] * 3, [20, 25, 26]) + rows("layer4", 1e-4, [0.8] * 3, [1.0] * 3, [33, 34, 36]))
    assert tie_ce["candidate"] == {"unfreeze": "layer4", "lr_backbone": 1e-4}                                      # then fewer blocks
    tie_blocks = de.select_candidate(rows("layer4", 1e-4, [0.8] * 3, [1.0] * 3, [20, 20, 20]) + rows("layer4", 1e-5, [0.8] * 3, [1.0] * 3, [21, 22, 23]))
    assert tie_blocks["candidate"]["lr_backbone"] == 1e-5 and tie_blocks["refit_epochs"] == 22                    # then lower lr
    assert de.select_candidate(rows("layer4", 1e-5, [0.5] * 3, [1.0] * 3, [20, 21, 30]))["refit_epochs"] == 21
    even = de.select_candidate(rows("layer4", 1e-5, [0.5, 0.5], [1.0, 1.0], [20, 23]))
    assert even["refit_epochs"] == 22                                                                              # ceil(21.5)


def test_insufficient_class_support_is_ne_and_emits_no_specs(cfg):
    dev = synthetic_dev(per_class=(20, 20, 19, 19, 4))            # RV: 3-4 per outer fitting set, 2 at f = 0.5
    wave1, units, _ = de.wave1_specs(dev, cfg)
    half = [u for u in units if u["fraction"] == 0.5]
    assert half and all(u["status"] == "NE" and u["reason"] == NE_INSUFFICIENT_CLASS_SUPPORT and u["short_classes"] == ["RV"] for u in half)
    assert not any(s["fraction"] == 0.5 for s in wave1)
    support = de.class_support(["DCM", "HCM"], CLASSES, 1)
    assert support["reason"] == NE_INSUFFICIENT_CLASS_SUPPORT and set(support["short_classes"]) == {"MINF", "NOR", "RV"}


def test_fit_specs_come_from_the_f9_lane_module_when_importable(monkeypatch):
    module = types.ModuleType("tmmri.neural.fit_spec")

    class FitSpec:
        def __init__(self, **fields):
            self.fields = fields

        @classmethod
        def create(cls, **fields):
            return cls(**fields)

        def to_dict(self):
            return {"fit_id": "f" * 16, **self.fields}

    module.FitSpec = FitSpec
    monkeypatch.setitem(sys.modules, "tmmri.neural.fit_spec", module)
    fields = {"purpose": "s5", "seed": 20261011, "outer_fold": 1, "fitting_patients": ["ACDC:patient001"], "selection_fold": 1,
              "candidate": {"unfreeze": "layer4", "lr_backbone": 1e-5}, "refit_epochs": None, "fraction": 0.5, "repeat": 0}
    spec, source = de.make_fit_spec(fields)
    assert source == "tmmri.neural.fit_spec" and spec["fit_id"] == "f" * 16
    monkeypatch.delitem(sys.modules, "tmmri.neural.fit_spec")
    monkeypatch.setattr(de, "_fit_spec_module", lambda: None)
    spec, source = de.make_fit_spec(fields)
    assert source == "c5_json" and spec["fit_id"] == de.compute_fit_id(fields)


@pytest.mark.parametrize("command", ["run-data-efficiency", "run-simulated-shift"])
def test_data_centric_subcommands_have_help(command, capsys):
    with pytest.raises(SystemExit) as info:
        cli.main([command, "--help"])
    assert info.value.code == 0 and command in capsys.readouterr().out
    assert command in cli.DATA_CENTRIC_COMMANDS and command not in cli.V6_COMMANDS


def test_data_centric_subcommands_exit_1_without_a_validated_run(tmp_path, capsys):
    for argv in (["run-data-efficiency", "--step", "descriptor"], ["run-simulated-shift", "--vendor", "A"]):
        with pytest.raises(SystemExit) as info:
            cli.main(argv + ["--run-id", "v6-datacentric-20261003-0901", "--artifact-root", str(tmp_path / "a"), "--data-root", str(tmp_path / "d")])
        assert info.value.code == cli.EXIT_USAGE_OR_INPUT
    assert "ERROR" in capsys.readouterr().out
