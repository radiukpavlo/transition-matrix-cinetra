"""F9 fit specs, selection folds, the selection rule, refit epochs, class weights and wave plans (protocol-v6.0.1;
configs/f9_cnn.yaml).  Synthetic identifiers only; no torch."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from tmmri.neural.fit_spec import (
    CANDIDATES,
    F9_SEEDS,
    Candidate,
    FitSpec,
    read_spec_file,
    write_spec_file,
)
from tmmri.neural.selection import (
    CLASSES,
    NE_INSUFFICIENT_CLASS_SUPPORT,
    EarlyStopper,
    InsufficientClassSupport,
    class_weights,
    plan_wave1,
    plan_wave2,
    refit_epochs,
    select_candidate,
    selection_folds,
)
from tmmri.splits import hash_rank_folds

CLASS_SIZES = {"DCM": 20, "HCM": 20, "MINF": 19, "NOR": 19, "RV": 20}   # 98, as acdc_dev


def dev_cohort() -> pd.DataFrame:
    rows, k = [], 1
    for cls, n in CLASS_SIZES.items():
        for _ in range(n):
            rows.append({"sample_id": f"synthetic{k:03d}", "cluster_id": f"synthetic{k:03d}", "label": cls})
            k += 1
    frame = pd.DataFrame(rows)
    frame["outer_fold"] = hash_rank_folds(frame, "sample_id", "label", "acdc-fold-v6|", 5).astype(int)
    return frame


def fitting_sets(frame: pd.DataFrame) -> dict:
    sets = {f: sorted(frame.loc[frame["outer_fold"] != f, "sample_id"]) for f in range(1, 6)}
    sets["final"] = sorted(frame["sample_id"])
    return sets


def test_fit_id_is_stable_and_order_independent():
    a = FitSpec(purpose="f9", seed=20261011, outer_fold=1, fitting_patients=("synthetic002", "synthetic001"), selection_fold=1,
                candidate=Candidate("layer4", 0.00001), refit_epochs=None)
    b = FitSpec.from_dict({"purpose": "f9", "seed": 20261011, "outer_fold": 1, "fitting_patients": ["synthetic001", "synthetic002"],
                           "selection_fold": 1, "candidate": {"unfreeze": "layer4", "lr_backbone": 1e-5}, "refit_epochs": None})
    assert a.fit_id == b.fit_id and len(a.fit_id) == 16
    assert a.fit_id == "d5b710548d54a507"   # golden: changing the canonical identity changes every fit id
    assert FitSpec.from_dict(a.to_dict()) == a
    with pytest.raises(ValueError, match="fit_id"):
        FitSpec.from_dict({**a.to_dict(), "fit_id": "0" * 16})
    other = FitSpec(**{**{k: v for k, v in a.identity().items()}, "candidate": Candidate("layer4", 0.00003)})
    assert other.fit_id != a.fit_id


@pytest.mark.parametrize("change", [{"seed": 20261001}, {"selection_fold": 4}, {"outer_fold": 6}, {"fraction": 0.5},
                                    {"candidate": {"unfreeze": "layer2", "lr_backbone": 1e-5}},
                                    {"candidate": {"unfreeze": "layer4", "lr_backbone": 1e-3}}, {"refit_epochs": 30}])
def test_fit_spec_rejects_values_outside_the_protocol(change):
    base = {"purpose": "f9", "seed": 20261011, "outer_fold": 1, "fitting_patients": ["x1"], "selection_fold": 1,
            "candidate": {"unfreeze": "layer4", "lr_backbone": 1e-5}, "refit_epochs": None}
    with pytest.raises(ValueError):
        FitSpec.from_dict({**base, **change})


def test_refit_and_s5_specs():
    refit = FitSpec(purpose="f9", seed=20261012, outer_fold="final", fitting_patients=("a", "b"), selection_fold=None,
                    candidate=CANDIDATES[0], refit_epochs=27)
    assert not refit.is_selection_fit and refit.refit_epochs == 27
    s5 = FitSpec(purpose="s5", seed=20261011, outer_fold=2, fitting_patients=("a",), selection_fold=2, candidate=CANDIDATES[5],
                 refit_epochs=None, fraction=0.5, repeat=2)
    assert s5.fit_id != FitSpec(purpose="s5", seed=20261011, outer_fold=2, fitting_patients=("a",), selection_fold=2,
                                candidate=CANDIDATES[5], refit_epochs=None, fraction=0.5, repeat=1).fit_id
    with pytest.raises(ValueError):
        FitSpec(purpose="s5", seed=20261012, outer_fold=2, fitting_patients=("a",), selection_fold=2, candidate=CANDIDATES[5],
                refit_epochs=None, fraction=0.5, repeat=0)


def test_selection_folds_are_stratified_deterministic_and_keep_clusters_together():
    frame = dev_cohort()
    members = frame[frame["outer_fold"] != 1].copy()
    members.loc[members.index[:2], "cluster_id"] = "pair"          # a declared duplicate pair of one class
    members.loc[members.index[:2], "label"] = members.loc[members.index[0], "label"]
    folds = selection_folds(members)
    again = selection_folds(members.sample(frac=1.0, random_state=0))
    pd.testing.assert_frame_equal(folds, again)
    assert folds.groupby("cluster_id")["selection_fold"].nunique().max() == 1
    for fold in (1, 2, 3):
        assert set(folds.loc[folds["selection_fold"] == fold, "label"]) == set(CLASSES)
    counts = folds.drop_duplicates("cluster_id").groupby(["label", "selection_fold"]).size().unstack()
    assert int((counts.max(axis=1) - counts.min(axis=1)).max()) <= 1   # round-robin within class
    other = selection_folds(frame[frame["outer_fold"] != 2])            # another fitting set has its own salt
    shared = folds.merge(other, on="sample_id", suffixes=("", "_2"))
    assert (shared["selection_fold"] != shared["selection_fold_2"]).any()


def test_selection_folds_declare_insufficient_class_support():
    frame = dev_cohort()
    small = pd.concat([frame[frame["label"] != "RV"], frame[frame["label"] == "RV"].head(2)])
    with pytest.raises(InsufficientClassSupport) as info:
        selection_folds(small)
    assert info.value.code == NE_INSUFFICIENT_CLASS_SUPPORT


def test_class_weights_inverse_frequency_mean_one():
    labels = ["DCM"] * 2 + ["HCM"] * 4 + ["MINF"] * 4 + ["NOR"] * 8 + ["RV"] * 2
    w = class_weights(labels)
    inverse = np.array([1 / 2, 1 / 4, 1 / 4, 1 / 8, 1 / 2])
    np.testing.assert_allclose(w, inverse / inverse.mean())
    assert w.mean() == pytest.approx(1.0)
    with pytest.raises(InsufficientClassSupport):
        class_weights(["DCM", "HCM", "MINF", "NOR"])


def _rows(values: dict) -> list[dict]:
    rows = []
    for cand, (f1, ce, epochs) in values.items():
        for fold in (1, 2, 3):
            rows.append({"candidate": cand, "selection_fold": fold, "macro_f1": f1, "cross_entropy": ce, "selected_epoch": epochs[fold - 1]})
    return rows


def test_selection_rule_and_tie_breaks():
    c = {cand.label: cand for cand in CANDIDATES}
    base = {cand: (0.5, 1.0, (20, 20, 20)) for cand in CANDIDATES}
    winner, table = select_candidate(_rows({**base, c["layer3+layer4@0.0001"]: (0.6, 2.0, (20, 30, 25))}))
    assert winner == c["layer3+layer4@0.0001"] and table.iloc[0]["refit_epochs"] == 25 and table["rank"].tolist() == list(range(1, 7))
    winner, _ = select_candidate(_rows({**base, c["layer3+layer4@3e-05"]: (0.5, 0.9, (20, 20, 20))}))
    assert winner == c["layer3+layer4@3e-05"]                     # F1 tie -> lower cross entropy
    winner, _ = select_candidate(_rows({cand: (0.5, 1.0, (20, 20, 20)) for cand in CANDIDATES if cand.lr_backbone == 0.0001}))
    assert winner == c["layer4@0.0001"]                           # F1 and CE tie -> fewer unfrozen blocks
    winner, _ = select_candidate(_rows(base))
    assert winner == c["layer4@1e-05"]                            # all tie -> fewer blocks, then lower learning rate
    rows = _rows(base)[:-1]
    with pytest.raises(ValueError, match="selection folds"):
        select_candidate(rows)


def test_refit_epoch_rule():
    assert refit_epochs([20, 31, 25]) == 25
    assert refit_epochs([20, 21]) == 21                            # ceil of 20.5
    assert refit_epochs([7, 7, 80]) == 7
    with pytest.raises(ValueError):
        refit_epochs([])


def test_early_stopping_rule():
    stopper = EarlyStopper(min_epoch=20, patience=12, min_delta=1e-4)
    values = [1.0 - 0.01 * e for e in range(1, 9)] + [0.92 - 0.00005] + [0.95] * 30   # epoch 9 improves by less than min_delta
    stops = []
    for epoch, value in enumerate(values, start=1):
        _, stop = stopper.update(epoch, value)
        if stop:
            stops.append(epoch)
            break
    assert stopper.best_epoch == 8 and stops == [20]
    tie = EarlyStopper(min_epoch=1, patience=2, min_delta=0.0)
    for epoch, value in enumerate([0.5, 0.5, 0.5], start=1):
        tie.update(epoch, value)
    assert tie.best_epoch == 1 and tie.stopped                    # ties keep the earlier checkpoint
    restored = EarlyStopper.from_state(stopper.state())
    assert restored.state() == stopper.state()


def test_plan_counts_324_and_18(tmp_path):
    frame = dev_cohort()
    sets = fitting_sets(frame)
    assert [len(sets[f]) for f in (1, 2, 3, 4, 5, "final")] == [78, 78, 78, 78, 80, 98]
    wave1 = plan_wave1(sets, seeds=F9_SEEDS)
    assert len(wave1) == 324 and len({s.fit_id for s in wave1}) == 324
    records = {}
    for s in wave1:
        h = int(hashlib.sha256(s.fit_id.encode()).hexdigest()[:8], 16)
        records[s.fit_id] = {"status": "COMPLETE", "selected_epoch": 20 + h % 30,
                             "val_metrics": {"macro_f1": (h % 997) / 997.0, "cross_entropy": 1.0 + (h % 89) / 89.0}}
    refits, table, not_estimable = plan_wave2(wave1, records)
    assert len(refits) == 18 and not not_estimable and len({s.fit_id for s in refits}) == 18
    assert all(not s.is_selection_fit and s.refit_epochs >= 20 for s in refits)
    assert table["selected"].sum() == 18 and len(table) == 108
    assert sorted({(str(s.outer_fold), s.seed) for s in refits}) == sorted({(str(f), seed) for f in sets for seed in F9_SEEDS})
    records[wave1[0].fit_id] = {"status": NE_INSUFFICIENT_CLASS_SUPPORT, "reason": "synthetic"}
    refits2, _, ne = plan_wave2(wave1, records)
    assert len(refits2) == 17 and ne[0]["status"] == NE_INSUFFICIENT_CLASS_SUPPORT
    spec_file = tmp_path / "wave2_specs.json"
    write_spec_file(spec_file, refits, run_id="v6-test-20261003-0000", wave=2, config={"model_id": "f9"},
                    cache_manifest_sha256="0" * 64, weights={"name": "w", "sha256": "1" * 64, "file": "w.pth"})
    assert [s.fit_id for s in read_spec_file(spec_file)["fit_specs"]] == [s.fit_id for s in refits]
