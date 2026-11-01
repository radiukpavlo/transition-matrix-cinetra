"""B21 of the post-freeze block (D20-P and D20-M; configs/exploratory_postfreeze.yaml B21): the layer-4 path equals the D20
encoder, D20-P recombines to the D20 vector, D20-M equals a direct upsample-and-mask computation, masks follow the image
geometry, an absent structure is NE, the generalized fit equals ``fit_d20_models``, and ``run`` on a tiny synthetic
source run (fake PostfreezeContext, random-weight ResNet-50, synthetic cases; no cohort data) freezes before it
evaluates.  Runs in the GPU image (CPU device)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")

from tmmri import deep_encoder as de
from tmmri.config import load_study
from tmmri.experiments.postfreeze import b21_frozen_image as b21
from tmmri.io import read_yaml
from tmmri.neural import preprocess as pp
from tmmri.neural.dihedral import d4_apply, tta_elements
from tmmri.paths import PROJECT_ROOT
from tmmri.runs import atomic_write_json, sha256_file
from tmmri.stages import BundleError

CLASSES = ["DCM", "HCM", "MINF", "NOR", "RV"]
SIZE = (32, 32)


@pytest.fixture(scope="module")
def encoder():
    torch.manual_seed(20260613)
    return de.build_d20_encoder(model=torchvision.models.resnet50(weights=None))


def synthetic_case(sample_id: str, *, seed: int = 0, shape=(24, 24, 3), directions=None, drop: tuple[str, int] | None = None,
                   strength: float = 1.0) -> tuple[pp.CaseVolumes, dict[str, np.ndarray]]:
    """Concentric LV cavity (1), myocardium (2) and an RV blob (3) per slice; the image is a noisy function of the labels."""

    rng = np.random.default_rng(seed)
    nx, ny, nz = shape
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx))
    masks, images = {}, {}
    for phase, scale in (("ed", 1.0), ("es", 0.75)):
        m = np.zeros(shape, dtype=np.int16)
        for z in range(nz):
            r = (4.0 + 1.5 * strength) * scale * (1.0 - 0.15 * z)
            d = np.hypot(xx - nx / 2, yy - ny / 2)
            m[..., z][d <= r + 2.0] = 2
            m[..., z][d <= r] = 1
            m[..., z][(np.hypot(xx - nx / 2, yy - ny / 2 - r - 4.0) <= 2.5) & (m[..., z] == 0)] = 3
        if drop is not None and drop[0] == phase:
            m[m == drop[1]] = 0
        masks[phase] = m
        images[phase] = (np.choose(m.astype(int), [50.0, 400.0, 150.0, 300.0]) + 20.0 * rng.normal(size=shape)).astype(np.float32)
    geom = pp.geometry_from_parameters(shape, (1.25, 1.25), 8.0, np.eye(3) if directions is None else directions)
    case = pp.CaseVolumes(sample_id=sample_id, ed_image=images["ed"], es_image=images["es"], ed_geometry=geom, es_geometry=geom,
                          cluster_id=sample_id, dataset=sample_id.split(":")[0], case_id=sample_id.split(":")[1])
    return case, masks


def test_layer4_pool_is_exactly_the_d20_encoder_output(encoder):
    x = torch.randn(3, 3, 64, 64, generator=torch.Generator().manual_seed(1))
    with torch.inference_mode():
        assert torch.equal(b21.pooled_vectors(encoder, b21.layer4_maps(encoder, x)), encoder(x))


def test_d20p_recombines_to_the_d20_patient_vector(encoder):
    case, masks = synthetic_case("ACDC:patient001")
    kwargs = {"size": SIZE}
    res = b21.case_vectors(encoder, case, masks, device="cpu", batch_size=4, input_kwargs=kwargs)
    feats, failures = de.extract_d20_features([case], encoder, device="cpu", partition="acdc_dev", batch_size=4, input_kwargs=kwargs)
    assert failures.empty
    np.testing.assert_array_equal(res["d20"], feats[de.feature_columns()].to_numpy(dtype=np.float64)[0])
    n_ed, n_es = res["n_ed"], res["n_es"]
    recombined = (n_ed * res["d20p"][:2048] + n_es * res["d20p"][2048:]) / (n_ed + n_es)
    np.testing.assert_allclose(recombined, res["d20"], rtol=0, atol=1e-12)
    assert res["d20p"].shape == (4096,) and res["d20m"].shape == (12288,) and not res["absent"]


def test_d20m_equals_direct_upsample_and_mask_means(encoder):
    case, masks = synthetic_case("ACDC:patient002", seed=2, shape=(28, 28, 2))
    size = (64, 64)                                               # layer-4 map 2 x 2: the bilinear upsampling varies in space
    # one forward chunk, as below: CPU convolutions round differently for different batch sizes
    res = b21.case_vectors(encoder, case, masks, device="cpu", batch_size=64, input_kwargs={"size": size})
    x, counts = de.d20_input_tensor(case, size=size)
    labels = b21.mask_input_labels(case, masks, size=size)
    n_ed = counts["n_ed"]
    expected = np.zeros((8, 2, 3, 2048))
    for e, g in enumerate(tta_elements("d4_mean_8")):
        with torch.inference_mode():
            maps = b21.layer4_maps(encoder, d4_apply(x, g))
            up = torch.nn.functional.interpolate(maps.double(), size=size, mode="bilinear", align_corners=False).numpy()
        lab = d4_apply(labels, g).numpy()
        for p, sl in enumerate((slice(0, n_ed), slice(n_ed, None))):
            for s, (k, _) in enumerate(b21.STRUCTURES):
                sel = lab[sl] == k
                expected[e, p, s] = up[sl].transpose(1, 0, 2, 3)[:, sel].mean(axis=1)
    # by linearity (Float64 interpolation weights of the mask) against the literal Float64 upsample-then-mask computation;
    # an FP32 upsampled map differs from both by its own rounding (about 1e-5 at these feature magnitudes)
    np.testing.assert_allclose(res["d20m"], expected.mean(axis=0).reshape(-1), rtol=1e-9, atol=1e-9)


def test_mask_labels_follow_the_image_orientation_and_grid():
    flip = np.diag([-1.0, 1.0, 1.0])[[1, 0, 2]]                    # transposed and mirrored in-plane axes
    case, masks = synthetic_case("ACDC:patient003", directions=flip, shape=(32, 32, 2))
    noiseless = pp.CaseVolumes(**{**case.__dict__, "ed_image": masks["ed"].astype(np.float32) * 100.0,
                                  "es_image": masks["es"].astype(np.float32) * 100.0})
    x, _ = de.d20_input_tensor(noiseless, size=(32, 32), mean=(0.0, 0.0, 0.0), sd=(1.0, 1.0, 1.0))
    labels = b21.mask_input_labels(noiseless, masks, size=(32, 32))
    assert torch.equal(torch.round(x[:, 0] * 3).to(torch.uint8), labels)          # same pixels as the image (no resize)
    up = b21.mask_input_labels(noiseless, masks, size=(64, 64))                     # integer upscale: each label pixel 2 x 2
    assert torch.equal(up[:, ::2, ::2], labels) and torch.equal(up[:, 1::2, 1::2], labels)


def test_absent_structure_is_ne_for_d20m_only(encoder):
    case, masks = synthetic_case("ACDC:patient004", drop=("es", 3))
    res = b21.case_vectors(encoder, case, masks, device="cpu", batch_size=8, tta="none", input_kwargs={"size": SIZE})
    assert res["absent"] == [("es", "rv")] and res["d20m"] is None and np.isfinite(res["d20p"]).all()
    rows = pd.DataFrame([{"sample_id": "ACDC:patient004", "cluster_id": "ACDC:patient004", "dataset": "ACDC", "case_id": "patient004"}])
    out = b21.extract_partition(rows, "acdc_dev", lambda row: (case, masks), encoder, device="cpu", batch_size=8, tta="none",
                                input_kwargs={"size": SIZE})
    assert len(out["d20p"]) == 1 and out["d20m"].empty
    assert out["failures"][["representation", "reason_code"]].values.tolist() == [["d20m", b21.STRUCTURE_ABSENT]]


def test_source_stage_and_lane_partitions():
    assert b21.source_stage("semantics/B_MMS2.parquet") == "build-semantics:MMS2"
    assert b21.source_stage("reconstructed/bhat_mms_external.parquet") == "evaluate:mms_external"
    assert b21.source_stage("deep_encoder/features/A_D20_acdc_test.parquet") == "extract-deep-features:acdc_test"
    lane = read_yaml(PROJECT_ROOT / "configs" / "exploratory_postfreeze.yaml")

    class Ctx:
        lane_cfg = lane

    assert b21.lane_partitions(Ctx) == ("acdc_dev", ["acdc_test", "mms_external", "mms2_external"])
    Ctx.lane_cfg = copy.deepcopy(lane)
    Ctx.lane_cfg["policy"]["evaluated_partitions"].append("mms2_sealed")
    with pytest.raises(ValueError):
        b21.lane_partitions(Ctx)


# ----------------------------------------------------------------------------- tiny synthetic source run


def _ids(prefix: str, start: int, n: int) -> list[str]:
    return [f"{prefix}{i:03d}" for i in range(start, start + n)]


class FakeContext:
    """The PostfreezeContext contract (source run read-only and hash-checked; outputs in a new run root)."""

    def __init__(self, source: Path, run_root: Path, staged: Path, study, lane):
        self.source_root, self.run_root, self.staged_data_root, self.study, self.lane_cfg = source, run_root, staged, study, lane
        self.recorded = {p.relative_to(source).as_posix(): sha256_file(p) for p in source.rglob("*") if p.is_file()}
        self.calls: list[tuple[str, str]] = []
        self.provenance: dict = {}

    def analysis_cfg(self, analysis):
        return self.lane_cfg["analyses"][analysis]

    def out_dir(self, analysis):
        d = self.run_root / "postfreeze" / analysis
        d.mkdir(parents=True, exist_ok=True)
        return d

    def verified(self, path, stage):
        p = self.source_root / path
        if sha256_file(p) != self.recorded[str(path)]:
            raise RuntimeError(f"{path}: SHA-256 differs from the source run record")
        self.calls.append((str(path), str(stage)))
        return p

    def write_provenance(self, analysis, inputs, outputs, summary):
        path = self.out_dir(analysis) / "provenance.json"
        self.require_new_output(path)
        assert all(Path(p).exists() for p in outputs) and all(Path(p).is_file() for p in inputs.values())
        self.provenance[analysis] = {"inputs": {k: sha256_file(p) for k, p in inputs.items()},
                                     "outputs": {str(p): sha256_file(p) for p in outputs}, "summary": summary}
        return atomic_write_json(path, {"label": "EXPLORATORY_POST_HOC", **self.provenance[analysis]}, allow_nan=True)

    def require_new_output(self, path):
        if Path(path).exists():
            raise BundleError(f"{path} exists; a repeat needs a new run id")


def make_source(root: Path, study) -> dict[str, list[str]]:
    rng = np.random.default_rng(20260613)
    targets = study.transition_targets()
    critical = sorted(str(c) for c in study.cohorts["qc_severity"]["critical"])[0]
    parts = {"acdc_dev": _ids("ACDC:patient", 1, 30), "acdc_test": _ids("ACDC:patient", 101, 10),
             "mms_external": _ids("MMS:A1A", 1, 10), "mms2_external": _ids("MMS2:MMS2_", 1, 10)}
    cls = {s: CLASSES[i % 5] for p in parts.values() for i, s in enumerate(p)}
    fold = {s: (i // 5) % 5 + 1 for i, s in enumerate(parts["acdc_dev"])}
    (root / "contracts").mkdir(parents=True)

    def split_rows(partitions):
        ids = [s for p in partitions for s in parts[p]]
        return pd.DataFrame({"sample_id": ids, "dataset": [s.split(":")[0] for s in ids], "case_id": [s.split(":")[1] for s in ids],
                             "partition": [p for p in partitions for _ in parts[p]],
                             "outer_fold": pd.array([fold.get(s) for s in ids], dtype="Int64"), "cluster_id": ids})

    split_rows(["acdc_dev", "acdc_test", "mms_external"]).to_parquet(root / "contracts" / "split_manifest.parquet", index=False)
    split_rows(["mms2_external"]).to_parquet(root / "contracts" / "split_manifest_MMS2.parquet", index=False)
    inner = [{"outer_fold_heldout": f, "sample_id": s, "inner_fold": j % 4 + 1}
             for f in range(1, 6) for j, s in enumerate([s for s in parts["acdc_dev"] if fold[s] != f])]
    pd.DataFrame(inner).to_parquet(root / "contracts" / "inner_folds.parquet", index=False)
    acdc = parts["acdc_dev"] + parts["acdc_test"]
    pd.DataFrame({"sample_id": acdc, "acdc_class": [cls[s] for s in acdc]}).to_parquet(root / "contracts" / "cohort_ACDC.parquet", index=False)
    for ds, p in (("MMS", "mms_external"), ("MMS2", "mms2_external")):
        ids = parts[p]
        compat = ["HARMONIZABLE_SAME_DEFINITION"] * (len(ids) - 1) + ["NON_EQUIVALENT"]
        pd.DataFrame({"sample_id": ids, "crosswalk_acdc_class": [cls[s] for s in ids], "crosswalk_compatibility": compat}).to_parquet(
            root / "contracts" / f"cohort_{ds}.parquet", index=False)
    atomic_write_json(root / "contracts" / "cohort_validation_MMS2.json", {"passed": True})
    (root / "semantics").mkdir()
    for ds, ps in (("ACDC", ["acdc_dev", "acdc_test"]), ("MMS", ["mms_external"]), ("MMS2", ["mms2_external"])):
        ids = [s for p in ps for s in parts[p]]
        signal = np.array([CLASSES.index(cls[s]) for s in ids], dtype=float)
        b = pd.DataFrame({t: 60.0 + 8.0 * signal + (j + 1) * rng.normal(size=len(ids)) for j, t in enumerate(targets)})
        b.insert(0, "sample_id", ids)
        b["qc_failure_reason"] = None
        if ds == "ACDC":
            b.loc[[3, 33], "qc_failure_reason"] = critical                      # one critical dev row, one critical test row
        b.to_parquet(root / "semantics" / f"B_{ds}.parquet", index=False)
        for p in ps:
            bhat = b[b["sample_id"].isin(parts[p])][["sample_id", *targets]].copy()
            bhat[targets] = bhat[targets] + 5.0 * rng.normal(size=(len(bhat), len(targets)))
            rel = Path("oof") / "bhat_oof_ACDC_dev.parquet" if p == "acdc_dev" else Path("reconstructed") / f"bhat_{p}.parquet"
            (root / rel).parent.mkdir(exist_ok=True)
            bhat.to_parquet(root / rel, index=False)
    (root / "deep_encoder" / "features").mkdir(parents=True)
    atomic_write_json(root / "deep_encoder" / "FROZEN_D20.json", {"provenance": {"weights_sha256": None}})
    for p, ids in parts.items():
        frame = pd.DataFrame(rng.normal(size=(3, 2048)), columns=de.feature_columns())
        frame.insert(0, "sample_id", ids[:3])
        frame.to_parquet(root / "deep_encoder" / "features" / f"A_D20_{p}.parquet", index=False)
    return parts | {"classes": cls}


def test_run_freezes_before_evaluating_and_reports_every_partition(tmp_path, encoder):
    study = load_study()
    study.configs["deep_encoder"] = {**study.configs["deep_encoder"], "tta": "none"}   # fast; the D4 path is tested above
    lane = read_yaml(PROJECT_ROOT / "configs" / "exploratory_postfreeze.yaml")
    lane["statistics"]["bootstrap"]["replicates"] = 20                    # fast; the lane value is 2,000
    parts = make_source(tmp_path / "source", study)
    ctx = FakeContext(tmp_path / "source", tmp_path / "artifacts" / "v6-postfreeze-20261028-0900", tmp_path / "data", study, lane)
    out = ctx.run_root / "postfreeze" / "B21"
    order: list[tuple[str, bool]] = []

    def loader(row):
        sid = str(row["sample_id"])
        order.append((sid, (out / "d20p" / "FROZEN.json").is_file() and (out / "d20m" / "FROZEN.json").is_file()))
        k = CLASSES.index(parts["classes"][sid])
        drop = ("es", 3) if sid == "ACDC:patient007" else None
        return synthetic_case(sid, seed=int(sid[-3:]), drop=drop, strength=float(k))

    summary = b21.run(ctx, device="cpu", encoder=encoder, loader=loader, input_size=SIZE)
    dev_ids = set(parts["acdc_dev"])
    assert all(not frozen for sid, frozen in order if sid in dev_ids)
    assert all(frozen for sid, frozen in order if sid not in dev_ids)
    assert "ACDC:patient004" not in {sid for sid, _ in order}                 # the critical dev row is never processed
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results["label"] == "EXPLORATORY_POST_HOC"
    for rep in ("d20p", "d20m"):
        b21.verify_frozen(out / rep)
        entry = results["representations"][rep]["partitions"]
        assert set(entry) == {"acdc_dev_oof", "acdc_test", "mms_external", "mms2_external"}
        for p, block in entry.items():
            assert {"rho", "rho_ef", "rho_volumes", "paired_vs_primary_ridge"} <= set(block["representation"])
            rates = block["predictor"]["chance_and_majority_rates"]
            assert rates["chance"] == pytest.approx(0.2) and 0 < rates["majority_rate"] <= 1
            assert {"accuracy", "macro_f1", "macro_ovr_auroc"} <= set(block["predictor"]["metrics"])
        assert entry["acdc_test"]["representation"]["rho"]["status"] == "NE"                  # 9 rows < 20: interval NE
        assert entry["acdc_test"]["counts"]["n_excluded_critical_qc"] == 1
    assert results["failures"]["acdc_dev"]["d20m"] == {b21.STRUCTURE_ABSENT: 1} and results["failures"]["acdc_dev"]["d20p"] == {}
    assert results["representations"]["d20m"]["fit_summary"]["transition_rows"] == 28        # 30 - critical - absent structure
    assert results["representations"]["d20p"]["fit_summary"]["transition_rows"] == 29
    assert ("contracts/cohort_validation_MMS2.json", "validate-cohort:MMS2") in ctx.calls
    assert ctx.provenance["B21"]["summary"]["label"] == "EXPLORATORY_POST_HOC"
    assert set(summary["rho"]) == {"d20p", "d20m"}
    assert "source:semantics/B_MMS2.parquet" in ctx.provenance["B21"]["inputs"]
    with pytest.raises(BundleError):                                        # a repeat needs a new run id
        b21.run(ctx, device="cpu", encoder=encoder, loader=loader, input_size=SIZE)


def test_mms2_refuses_without_the_passed_validation_record(tmp_path):
    study = load_study()
    lane = read_yaml(PROJECT_ROOT / "configs" / "exploratory_postfreeze.yaml")
    make_source(tmp_path / "source", study)
    atomic_write_json(tmp_path / "source" / "contracts" / "cohort_validation_MMS2.json", {"passed": False})
    ctx = FakeContext(tmp_path / "source", tmp_path / "run", tmp_path / "data", study, lane)
    with pytest.raises(PermissionError):
        b21.load_partition(ctx, "mms2_external")


def test_generalized_fit_equals_fit_d20_models(tmp_path):
    study = load_study()
    lane = read_yaml(PROJECT_ROOT / "configs" / "exploratory_postfreeze.yaml")
    make_source(tmp_path / "source", study)
    ctx = FakeContext(tmp_path / "source", tmp_path / "run", tmp_path / "data", study, lane)
    dev = b21.load_partition(ctx, "acdc_dev")
    rng = np.random.default_rng(7)
    signal = np.array([CLASSES.index(c) for c in dev.truth.reindex(dev.rows["sample_id"])], dtype=float)
    x = rng.normal(size=(len(dev.rows), 2048)) + 0.5 * signal[:, None] * (np.arange(2048) < 16)
    feats = pd.DataFrame(x, columns=de.feature_columns(), index=pd.Index(dev.rows["sample_id"], name="sample_id"))
    critical = set(dev.critical.index[dev.critical.to_numpy()])
    common = dict(targets=study.transition_targets(), transition_cfg=study.transition, registry=study.registry,
                  predictor_cfg=study.formal["audited_predictor"], make_inner_fn=lambda ids: b21.inner_fold_fn(ctx, ids), classes=CLASSES)
    split = dev.rows[["sample_id", "outer_fold", "cluster_id"]]
    ours = b21.fit_representation(feats, split, dev.b.reset_index(), dev.truth, critical, representation=de.D20_REPRESENTATION, **common)
    ref = de.fit_d20_models(feats.reset_index(), split, dev.b.reset_index(), dev.truth, critical, **common)
    pd.testing.assert_frame_equal(ours.oof.reset_index(drop=True), ref.oof.reset_index(drop=True))
    probe = rng.normal(size=(4, 2048))
    pd.testing.assert_frame_equal(ours.model.predict(probe), ref.model.predict(probe))
    np.testing.assert_array_equal(ours.predictor.predict_proba(probe).to_numpy(), ref.predictor.predict_proba(probe).to_numpy())
    oof_ref = ref.predictions[ref.predictions["is_oof"]].set_index("sample_id")[[f"prob_{c}" for c in CLASSES]]
    np.testing.assert_array_equal(ours.oof_proba(CLASSES).loc[oof_ref.index].to_numpy(), oof_ref.to_numpy())
