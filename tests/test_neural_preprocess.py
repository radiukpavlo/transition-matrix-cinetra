"""Protocol-v6.0.1 neural tensor cache (contract C3; configs/neural_input.yaml) on synthetic data only."""
from __future__ import annotations

import hashlib
import json

import fixtures_v6 as fx
import numpy as np
import pandas as pd
import pytest

from tmmri import cli, stages
from tmmri.config import load_study
from tmmri.neural import preprocess as pp
from tmmri.neural.image_stages import build_tensor_cache
from tmmri.runs import RunRoot

CFG = pp.load_neural_input_config()


def test_input_block_mirrors_the_design_json(repo_root):
    design = json.loads((repo_root / "protocol" / "neural_encoder_v6.0.1.design.json").read_text(encoding="utf-8"))
    assert CFG["input"] == design["input"]
    assert CFG["cache"]["labels"]["allowed_partitions"] == ["acdc_dev"]
    assert pp.transform_version(CFG).startswith("neural_input_v1:")


def _disk_volume(shape, spacing, radii_mm, value=200.0, background=20.0):
    """(X, Y, Z) volume with one centred disk per slice (radius in mm), constant background."""

    nx, ny, nz = shape
    x = (np.arange(nx) - (nx - 1) / 2.0) * spacing
    y = (np.arange(ny) - (ny - 1) / 2.0) * spacing
    r = np.hypot(*np.meshgrid(x, y, indexing="ij"))
    vol = np.full(shape, background, dtype=np.float32)
    for k in range(nz):
        vol[..., k][r <= radii_mm[k]] = value
    return vol


def _case(sample_id, shape, spacing, radii_mm, *, increment=8.0, directions=None, label="NOR", es_spacing=None, ed=None):
    geom = pp.geometry_from_parameters(shape, (spacing, spacing), increment, directions, cfg=CFG)
    es_geom = geom if es_spacing is None else pp.geometry_from_parameters(shape, (es_spacing, es_spacing), increment, directions, cfg=CFG)
    vol = _disk_volume(shape, spacing, radii_mm) if ed is None else ed
    return pp.CaseVolumes(sample_id=sample_id, ed_image=vol, es_image=vol.copy(), ed_geometry=geom, es_geometry=es_geom,
                          cluster_id=sample_id, label=label, dataset="SYN", case_id=sample_id.split(":")[1])


def test_axis_weights_resample_on_the_physical_grid():
    n, spacing = 40, 1.25
    source = (np.arange(n) - (n - 1) / 2.0) * spacing          # value = physical offset from the FOV centre (mm)
    weights = pp.axis_weights(n, spacing, 1.5, 224)
    out = weights @ source
    offsets = (np.arange(224) - 111.5) * 1.5
    between_centres = np.abs(offsets) <= (n - 1) / 2.0 * spacing
    np.testing.assert_allclose(out[between_centres], offsets[between_centres], atol=1e-12)   # bilinear is exact on a ramp
    inside = np.abs(offsets) <= n / 2.0 * spacing
    np.testing.assert_allclose(weights[inside].sum(axis=1), 1.0)
    assert not weights[~inside].any()                           # zero padding outside the field of view
    assert pp.crop_fraction((300, 300), (1.5, 1.5), 1.5, (224, 224)) == pytest.approx(1 - (224 / 300) ** 2)
    assert pp.crop_fraction((200, 180), (1.25, 1.25), 1.5, (224, 224)) == 0.0


def test_physical_size_is_preserved_across_source_spacings(tmp_path):
    radii = [10.0, 13.0, 16.0, 19.0]
    cases = [_case("SYN:fine", (120, 120, 4), 0.8, radii), _case("SYN:coarse", (40, 40, 4), 2.4, radii)]
    result = pp.build_tensor_cache_from_volumes(cases, tmp_path / "cache", "acdc_dev", CFG)
    assert result["failures"].empty
    mass, areas = {}, {}
    for row in result["manifest"].itertuples(index=False):
        arrays = pp.load_cache_npz(tmp_path / "cache" / row.npz_file)
        assert arrays["ed"].shape == (4, 224, 224) and arrays["ed"].dtype == np.float32
        assert arrays["ed"].min() >= 0.0 and arrays["ed"].max() <= 1.0
        mass[row.sample_id] = arrays["ed"].astype(np.float64).sum(axis=(1, 2)) * 1.5 ** 2     # mm^2 of the scaled disk
        areas[row.sample_id] = (arrays["ed"] > 0.5).sum(axis=(1, 2))
    for case in cases:   # scaled intensities are exactly 0 (background) and 1 (disk): the physical area is preserved
        source_mm2 = (case.ed_image > 100).sum(axis=(0, 1)) * case.ed_geometry.spacing_mm[0] ** 2
        np.testing.assert_allclose(mass[case.sample_id], source_mm2, rtol=0.03, err_msg=case.sample_id)
        assert np.all(np.diff(areas[case.sample_id]) > 0), case.sample_id   # slice order preserved (radius grows with k)
    np.testing.assert_allclose(areas["SYN:fine"], np.pi * (np.asarray(radii) / 1.5) ** 2, rtol=0.08)   # pixelated disk
    crop = result["manifest"].set_index("sample_id")["crop_fraction_ed"]
    assert crop["SYN:fine"] == 0.0 and crop["SYN:coarse"] == 0.0


def test_geometry_scalars_orientation_and_determinism(tmp_path):
    case = _case("SYN:geo", (30, 28, 5), 1.25, [5, 6, 7, 8, 9], increment=9.0)
    first = pp.preprocess_case(case, CFG)
    np.testing.assert_allclose(first.arrays["ed_geom"][:, 0], (np.arange(5) - 2) * 9.0 / 100.0, rtol=0, atol=1e-7)
    np.testing.assert_allclose(first.arrays["es_geom"][:, 1], 0.9, rtol=0, atol=1e-7)
    assert first.arrays["ed_geom"].dtype == np.float32
    assert pp.npz_bytes(first.arrays) == pp.npz_bytes(pp.preprocess_case(case, CFG).arrays)   # byte-identical files
    acdc = pp.geometry_from_parameters((24, 22, 4), (1.5625, 1.5625), 10.0, np.diag([-1.0, -1.0, 1.0]))
    mms = pp.geometry_from_parameters((20, 18, 3), (1.25, 1.25), 8.0, np.eye(3))
    assert pp.canonical_orientation(acdc).code == "rows=+1,cols=+0"
    assert pp.canonical_orientation(mms).code == "rows=-1,cols=-0"
    vol = np.zeros((20, 18, 3), dtype=np.float32)
    vol[0, 17, 1] = 1.0                       # most patient-left (RAS -x) and most anterior (RAS +y) voxel
    slices = pp.oriented_slices(vol, pp.canonical_orientation(mms))
    assert slices.shape == (3, 18, 20) and slices[1, 0, -1] == 1.0   # top row (anterior), last column (patient left)


def test_input_qc_rejections_are_listed_not_cached(tmp_path):
    good = _case("SYN:good", (30, 30, 3), 1.25, [6, 7, 8])
    nonfinite = _case("SYN:nan", (30, 30, 3), 1.25, [6, 7, 8])
    nonfinite.es_image[0, 0, 0] = np.nan
    constant = _case("SYN:flat", (30, 30, 3), 1.25, [6, 7, 8], ed=np.full((30, 30, 3), 5.0, dtype=np.float32))
    mismatch = _case("SYN:pair", (30, 30, 3), 1.25, [6, 7, 8], es_spacing=1.5)
    result = pp.build_tensor_cache_from_volumes([good, nonfinite, constant, mismatch], tmp_path / "c", "acdc_test", CFG)
    assert list(result["manifest"]["sample_id"]) == ["SYN:good"]
    assert result["manifest"]["label"].isna().all()               # no label access outside acdc_dev
    reasons = dict(zip(result["failures"]["sample_id"], result["failures"]["reason_code"]))
    assert reasons == {"SYN:nan": "INPUT_NONFINITE", "SYN:flat": "INPUT_CONSTANT_INTENSITY", "SYN:pair": "INPUT_GEOMETRY_INVALID"}
    assert sorted(p.name for p in (tmp_path / "c").glob("*.npz")) == ["SYN__good.npz"]
    for bad in (dict(spacing_mm=(0.0, 1.0), slice_increment_mm=8.0, directions=None),
                dict(spacing_mm=(1.0, 1.0), slice_increment_mm=None, directions=None),
                dict(spacing_mm=(1.0, 1.0), slice_increment_mm=8.0, directions=[[1, 0, 0], [0.2, 1, 0], [0, 0, 1]]),
                dict(spacing_mm=(1.0, 1.0), slice_increment_mm=8.0, directions=[[1, 0, 0], [0, 1, 0], [0.5, 0, 1]])):
        with pytest.raises(pp.InputRejected) as info:
            pp.geometry_from_parameters((10, 10, 2), cfg=CFG, **bad)
        assert info.value.reason_code == "INPUT_GEOMETRY_INVALID"


def test_staged_header_geometry_and_scaled_affine_mismatch(tmp_path):
    nib = pytest.importorskip("nibabel")
    acdc = fx.write_nifti(tmp_path / "acdc.nii", np.ones((24, 22, 4), np.float32), np.diag([-1.0, -1.0, 1.0, 1.0]),
                          zooms=(1.5625, 1.5625, 10.0))
    geom = pp.phase_geometry(nib.load(str(acdc)), CFG)
    assert geom.spacing_mm == (1.5625, 1.5625) and geom.slice_increment_mm == 10.0   # corrected increment (pixdim)
    assert geom.z_spacing_method == "nifti_header_pixdim_unscaled_affine"
    bad = fx.write_nifti(tmp_path / "bad.nii", np.ones((24, 22, 4), np.float32), np.diag([2.0, 2.0, 8.0, 1.0]), zooms=(1.25, 1.25, 8.0))
    with pytest.raises(pp.InputRejected) as info:
        pp.phase_geometry(nib.load(str(bad)), CFG)
    assert info.value.reason_code == "INPUT_GEOMETRY_INVALID"


@pytest.fixture()
def synthetic_run(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "protocol_frozen", lambda *a, **k: False)
    acdc = fx.make_acdc_source(tmp_path / "src" / "acdc", fx.acdc_patients(1))
    mms = fx.make_mms_source(tmp_path / "src" / "mms")
    study = load_study()
    run = RunRoot("v6-neuralcache-20261003-0900", tmp_path / "artifacts", subfolders=study.study["artifact_subfolders"])
    ctx = stages.StageContext(study=study, run=run, data_root=tmp_path / "data", n_jobs=1, sources={"acdc": acdc, "mms": mms},
                              source_manifests={"ACDC": fx.acdc_manifest(acdc), "MMS": fx.mms_manifest(mms)},
                              mms_metadata=fx.mms_metadata(mms))
    assert stages.stage_sources(ctx, "acdc").ok and stages.stage_sources(ctx, "mms").ok
    stages.validate_cohort(ctx)
    return ctx


def test_build_tensor_cache_round_trip_from_staged_nifti(synthetic_run):
    from scipy.ndimage import map_coordinates

    ctx = synthetic_run
    first = build_tensor_cache(ctx, "acdc_dev")
    assert first.action == "RAN" and build_tensor_cache(ctx, "acdc_dev").action == "SKIPPED"
    cache = ctx.run.path / "neural_cache" / "acdc_dev"
    manifest = pd.read_csv(cache / "manifest.csv")
    assert list(manifest.columns[:len(pp.MANIFEST_COLUMNS)]) == list(pp.MANIFEST_COLUMNS)
    cohort = pd.read_parquet(ctx.run.path / "contracts" / "cohort_ACDC.parquet").set_index("sample_id")
    assert len(manifest) == 5 and (manifest["partition"] == "acdc_dev").all()
    assert dict(zip(manifest["sample_id"], manifest["label"])) == cohort.loc[manifest["sample_id"], "acdc_class"].to_dict()
    assert (manifest["n_ed"] == 4).all() and (manifest["n_es"] == 4).all()
    assert (manifest["crop_fraction_ed"] == 0.0).all() and (manifest["inplane_spacing_mm"] == 1.5625).all()
    assert (manifest["orientation"] == "rows=+1,cols=+0").all() and (manifest["slice_increment_mm"] == 10.0).all()
    row = manifest.iloc[0]
    data = (cache / row["npz_file"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == row["npz_sha256"]
    arrays = pp.load_cache_npz(cache / row["npz_file"])
    np.testing.assert_allclose(arrays["ed_geom"], np.column_stack([(np.arange(4) - 1.5) * 0.1, np.ones(4)]), atol=1e-7)
    # Independent check: scipy bilinear interpolation of the staged image on the 1.5 mm grid (rows from voxel axis 1).
    case = pp.load_staged_volumes(ctx.staged_root, "ACDC", row["sample_id"].split(":")[1], cfg=CFG)
    slices = np.moveaxis(case.ed_image, 2, 0).transpose(0, 2, 1).astype(np.float64)
    lo, hi = np.percentile(slices, [1, 99])
    scaled = np.clip((slices - lo) / (hi - lo), 0, 1)
    offsets = (np.arange(224) - 111.5) * 1.5 / 1.5625
    ur, uc = offsets + (slices.shape[1] - 1) / 2.0, offsets + (slices.shape[2] - 1) / 2.0
    grid_r, grid_c = np.meshgrid(ur, uc, indexing="ij")
    inside = (grid_r >= -0.5) & (grid_r <= slices.shape[1] - 0.5) & (grid_c >= -0.5) & (grid_c <= slices.shape[2] - 0.5)
    for k in range(4):
        expected = map_coordinates(scaled[k], [grid_r, grid_c], order=1, mode="nearest") * inside
        np.testing.assert_allclose(arrays["ed"][k], expected, atol=2e-6)
    record = json.loads((cache / "cache_record.json").read_text(encoding="utf-8"))
    assert record["labels_written"] is True and record["n_cases"] == 5 and record["transform_version"] == row["transform_version"]
    stage_record = json.loads((ctx.run.path / ".stages" / "build-tensor-cache__acdc_dev.json").read_text(encoding="utf-8"))
    assert stage_record["payload"]["params"]["partition"] == "acdc_dev"


def test_mms_cache_refuses_before_the_freeze_and_has_no_labels(synthetic_run):
    ctx = synthetic_run
    with pytest.raises(stages.ProtocolFreezeError):
        build_tensor_cache(ctx, "mms_external")
    ctx.allow_unfrozen_synthetic = True
    build_tensor_cache(ctx, "mms_external")
    manifest = pd.read_csv(ctx.run.path / "neural_cache" / "mms_external" / "manifest.csv")
    assert len(manifest) == 4 and manifest["label"].isna().all()
    assert (manifest["orientation"] == "rows=-1,cols=-0").all()


@pytest.mark.parametrize("command", list(cli.NEURAL_IMAGE_COMMANDS))
def test_neural_image_commands_have_help(command, capsys):
    with pytest.raises(SystemExit) as info:
        cli.main([command, "--help"])
    assert info.value.code == 0 and command in capsys.readouterr().out
    assert not set(cli.NEURAL_IMAGE_COMMANDS) & set(cli.V6_COMMANDS)
