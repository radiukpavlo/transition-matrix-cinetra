"""``preflight`` (src/tmmri/preflight.py) on synthetic run roots only: the PASS/WARN/FAIL/UNKNOWN logic of each check
family, the read-only guarantee (no file is created or changed, an absent run root stays absent) and the exit codes.
No cohort data, no Docker volume, no M&Ms-2 archive and no network: every path lives below pytest's tmp_path, git and
the protocol tags are monkeypatched where a test needs a fixed state."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tmmri import cli, preflight, stages
from tmmri.runs import atomic_write_json, run_stage, sha256_json

RUN_ID = "v6-preflighttest-20261003-1200"
CODE = {"commit": "synthetic", "dirty": False, "code_sha256": "synthetic"}
SECRET = "token-contents-must-never-be-printed"


@pytest.fixture(autouse=True)
def _fixed_environment(monkeypatch, tmp_path):
    """Host side, clean tree, no Modal token (an empty profile folder) and no TORCH_HOME unless a test sets them."""

    monkeypatch.setenv("USERPROFILE", str(tmp_path / "empty_profile"))
    monkeypatch.setattr(preflight, "detect_side", lambda: "host")
    monkeypatch.setattr(preflight, "git_dirty", lambda root=None: (False, []))
    monkeypatch.setattr(preflight, "git_head", lambda root=None: "0" * 40)
    for name in ("MODAL_TOKEN_ID", "TORCH_HOME", "TMMRI_ARTIFACT_ROOT", "TMMRI_DATA_ROOT"):
        monkeypatch.delenv(name, raising=False)


def roots(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "artifacts", tmp_path / "data"


def make_run(tmp_path: Path):
    """A synthetic run root (created by the fixture, never by the preflight)."""

    art, data = roots(tmp_path)
    ctx = stages.make_context("configs/study_v6.yaml", RUN_ID, artifact_root=art, data_root=data)
    return ctx


def record(run, name: str, rel: str, content: str = "synthetic\n", *, status: str = "BUILT") -> Path:
    path = run.path / rel

    def work() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    run_stage(run, name, work, [], [path], config_hash="synthetic", code=CODE, status=status)
    return path


def invoke(tmp_path: Path, step: str, *extra: str, capsys) -> tuple[int, list[str]]:
    art, data = roots(tmp_path)
    argv = ["preflight", "--run-id", RUN_ID, "--step", step, "--artifact-root", str(art), "--data-root", str(data),
            "--mms2-source", str(tmp_path / "mms2_area"), *extra]
    try:
        cli.main(argv)
        code = 0
    except SystemExit as exc:
        code = int(exc.code)
    return code, capsys.readouterr().out.splitlines()


def build(tmp_path: Path, step: str = "f8", **overrides):
    import argparse

    art, data = roots(tmp_path)
    ns = {"run_id": RUN_ID, "step": step, "study": "configs/study_v6.yaml", "split": [], "wave": 1, "specs": None, "source": None,
          "archive": None, "vendor": "all", "hash": False, "min_free_gb": 20.0, "data_root": str(data), "artifact_root": str(art),
          "acdc_source": None, "mms_source": None, "mms2_source": str(tmp_path / "mms2_area")}
    ns.update(overrides)
    return preflight.build_preflight(argparse.Namespace(**ns))


def statuses(pf, name_prefix: str) -> list[str]:
    return [c.status for c in pf.checks if c.name.startswith(name_prefix)]


def snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(root.rglob("*"))}


# ----------------------------------------------------------------------------- CLI surface and exit codes


def test_steps_and_cli_registration(capsys):
    assert preflight.STEPS == ("replication", "f8", "b5", "b6", "tensor-cache", "weights", "d20", "modal-qualify", "upload", "f9-train",
                               "f9-import", "f9-freeze", "f9b", "trust-fit", "eval-historical", "s5", "s6", "tag", "mms2-extract",
                               "mms2-evaluate", "results-freeze")
    assert set(preflight.STEP_FUNCTIONS) == set(preflight.STEPS)
    with pytest.raises(SystemExit) as exc:
        cli.main(["preflight", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--step" in out and "results-freeze" in out and "writes nothing" in out


def test_exit_codes(tmp_path, capsys):
    code, lines = invoke(tmp_path, "b5", capsys=capsys)            # synthetic only: nothing upstream is required
    assert code == 0 and lines[-1].startswith("RESULT  GO")
    code, lines = invoke(tmp_path, "f8", capsys=capsys)            # no run root
    assert code == 1 and lines[-1].startswith("RESULT  NO-GO")
    assert any(line.startswith("FAIL    run root") for line in lines)
    with pytest.raises(SystemExit) as exc:                         # an invalid run id is a FAIL line, not a traceback
        cli.main(["preflight", "--run-id", "Bad Id", "--step", "f8", "--artifact-root", str(tmp_path)])
    assert exc.value.code == 1
    assert capsys.readouterr().out.startswith("FAIL    preflight setup")


def test_every_line_has_a_status(tmp_path, capsys):
    make_run(tmp_path)
    _, lines = invoke(tmp_path, "eval-historical", capsys=capsys)
    assert lines and all(line.split(" ", 1)[0] in {"PASS", "WARN", "FAIL", "UNKNOWN", "RESULT"} for line in lines)


# ----------------------------------------------------------------------------- read-only guarantee


def test_read_only_on_a_populated_run_root(tmp_path, capsys):
    ctx = make_run(tmp_path)
    record(ctx.run, "fit-transition", "oof/bhat_oof_ACDC_dev.parquet")
    record(ctx.run, "evaluate:acdc_test", "reconstructed/bhat_acdc_test.parquet", status="EVALUATED")
    atomic_write_json(ctx.run.frozen_path, {"bundle": {}, "bundle_sha256": sha256_json({})})
    (tmp_path / "mms2_area").mkdir()
    before = snapshot(tmp_path)
    for step in preflight.STEPS:
        invoke(tmp_path, step, capsys=capsys)
    assert snapshot(tmp_path) == before


def test_absent_run_root_is_never_created(tmp_path, capsys):
    art, _ = roots(tmp_path)
    for step in preflight.STEPS:
        invoke(tmp_path, step, capsys=capsys)
    assert not (art / RUN_ID).exists()
    assert not art.exists() or not any(art.iterdir())


# ----------------------------------------------------------------------------- run root, records and hashes


def test_recorded_artifact_pass_changed_and_untracked(tmp_path):
    ctx = make_run(tmp_path)
    path = record(ctx.run, "fit-transition", "oof/bhat_oof_ACDC_dev.parquet")
    (ctx.run.path / "formal").mkdir(exist_ok=True)
    (ctx.run.path / "formal" / "stray.parquet").write_text("x", encoding="utf-8")
    pf = build(tmp_path)
    assert pf.artifact("oof/bhat_oof_ACDC_dev.parquet") and pf.checks[-1].status == "PASS"
    assert not pf.artifact("formal/stray.parquet") and "not a recorded output" in pf.checks[-1].detail
    assert not pf.artifact("formal/absent.parquet") and pf.checks[-1].status == "FAIL"
    path.write_text("changed\n", encoding="utf-8")
    pf = build(tmp_path)
    assert not pf.artifact("oof/bhat_oof_ACDC_dev.parquet")
    assert pf.checks[-1].status == "FAIL" and "changed after fit-transition" in pf.checks[-1].detail


def test_record_status_and_outputs(tmp_path):
    ctx = make_run(tmp_path)
    out = record(ctx.run, "evaluate:acdc_test", "reconstructed/bhat_acdc_test.parquet", status="EVALUATED")
    pf = build(tmp_path)
    assert pf.record("evaluate:acdc_test", status="EVALUATED") is not None
    assert pf.record("evaluate:mms_external") is None and pf.checks[-1].status == "FAIL"
    assert pf.record("evaluate:acdc_test", status="BUILT") is None
    out.unlink()
    pf = build(tmp_path)
    assert pf.record("evaluate:acdc_test") is None and "missing or changed" in pf.checks[-1].detail


def test_once_only_state(tmp_path):
    ctx = make_run(tmp_path)
    record(ctx.run, "fit-cart-surrogate", "comparators/cart_f8/tree.json")
    pf = build(tmp_path)
    assert not pf.once_only("fit-deep-encoder") and pf.checks[-1].status == "PASS"
    assert pf.once_only("fit-cart-surrogate")
    assert pf.checks[-1].status == "WARN" and "RE-RUN" in pf.checks[-1].detail
    atomic_write_json(ctx.run.frozen_path, {"bundle": {}})
    pf = build(tmp_path)
    pf.once_only("fit-cart-surrogate")
    assert pf.checks[-1].status == "WARN" and "REFUSE (once-only guard" in pf.checks[-1].detail and "exit 7" in pf.checks[-1].detail
    (ctx.run.frozen_path).unlink()
    (ctx.run.path / "deep_encoder").mkdir()
    (ctx.run.path / "deep_encoder" / "FROZEN_D20.json").write_text("{}", encoding="utf-8")
    record(ctx.run, "extract-deep-features:acdc_dev", "deep_encoder/features/A_D20_acdc_dev.parquet")
    pf = build(tmp_path)
    pf.once_only("extract-deep-features:acdc_dev", freeze_files=["deep_encoder/FROZEN_D20.json"])   # image_bundle_guard
    assert "FROZEN_D20.json" in pf.checks[-1].detail and "REFUSE" in pf.checks[-1].detail


def test_run_root_visibility(tmp_path, monkeypatch):
    pf = build(tmp_path, artifact_root=None, step="f8")
    assert not pf.run_root() and pf.checks[-1].status == "FAIL" and "run this preflight in the container" in pf.checks[-1].detail
    pf = build(tmp_path, artifact_root=None, step="mms2-extract")
    pf.run_root()
    assert pf.checks[-1].status == "UNKNOWN"
    pf = build(tmp_path, step="f8")          # explicit root: visible, so absent means FAIL
    assert not pf.run_root() and pf.checks[-1].status == "FAIL" and "does not exist" in pf.checks[-1].detail


# ----------------------------------------------------------------------------- freezes


def test_frozen_bundle_verification(tmp_path):
    ctx = make_run(tmp_path)
    path = record(ctx.run, "fit-transition", "transitions/transition_model.json", "{}\n")
    files = {"transitions/transition_model.json": hashlib.sha256(path.read_bytes()).hexdigest()}
    atomic_write_json(ctx.run.frozen_path, {"bundle": files, "bundle_sha256": sha256_json(files),
                                            "bundle_config": stages.bundle_config(ctx.study)})
    pf = build(tmp_path)
    assert pf.frozen_bundle() is not None and pf.checks[-1].status == "PASS"
    path.write_text('{"changed": 1}\n', encoding="utf-8")
    pf = build(tmp_path)
    assert pf.frozen_bundle() is None and "changed after the freeze" in pf.checks[-1].detail
    ctx.run.frozen_path.unlink()
    pf = build(tmp_path)
    assert pf.frozen_bundle() is None and pf.checks[-1].status == "FAIL"


def _f9_bundle(run) -> Path:
    fid = "0123456789abcdef"
    folder = run.path / "neural_f9" / "checkpoints" / fid
    folder.mkdir(parents=True)
    (folder / "model.pt").write_bytes(b"synthetic checkpoint")
    (folder / "record.json").write_text("{}", encoding="utf-8")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()   # noqa: E731
    body = {"schema": "tmmri.f9_bundle/1", "fits": [{"checkpoint": f"neural_f9/checkpoints/{fid}/model.pt", "checkpoint_sha256": sha(folder / "model.pt"),
                                                     "record_sha256": sha(folder / "record.json")}]}
    atomic_write_json(run.path / "neural_f9" / "f9_bundle.json", {**body, "bundle_sha256": sha256_json(body)})
    return folder


def test_freeze_record_rehash(tmp_path):
    ctx = make_run(tmp_path)
    folder = _f9_bundle(ctx.run)
    pf = build(tmp_path)
    assert pf.freeze_file("F9") and pf.checks[-1].status == "PASS"
    assert not pf.freeze_file("D20") and pf.checks[-1].status == "FAIL" and "missing" in pf.checks[-1].detail
    (folder / "model.pt").write_bytes(b"tampered")
    pf = build(tmp_path)
    assert not pf.freeze_file("F9") and "changed after the freeze" in pf.checks[-1].detail
    bundle = ctx.run.path / "neural_f9" / "f9_bundle.json"
    payload = json.loads(bundle.read_text(encoding="utf-8"))
    atomic_write_json(bundle, {**payload, "bundle_sha256": "0" * 64})
    pf = build(tmp_path)
    assert not pf.freeze_file("F9") and "bundle_sha256" in pf.checks[-1].detail


def test_freezes_before_extraction(tmp_path):
    ctx = make_run(tmp_path)
    pf = build(tmp_path, step="mms2-extract")
    pf.freezes_before_extraction()
    assert statuses(pf, "freeze order") == ["FAIL"] * 5
    for name in ("fit-cart-surrogate", "fit-deep-encoder", "freeze-f9", "fit-probes", "fit-trust"):
        record(ctx.run, name, f"freeze_marks/{name}.json")
    pf = build(tmp_path, step="mms2-extract")
    pf.freezes_before_extraction()
    assert statuses(pf, "freeze order") == ["PASS"] * 5


# ----------------------------------------------------------------------------- weights, cache, Modal


def test_weights_registry_and_files(tmp_path, monkeypatch):
    from tmmri.neural import weights

    pf = build(tmp_path, step="weights")
    pf.weights((weights.D20_WEIGHTS,), file_required=True)
    assert statuses(pf, "weights resnet50_imagenet1k_v2 registry") == ["PASS"]
    assert statuses(pf, "weights resnet50_imagenet1k_v2 file") == ["UNKNOWN"]     # TORCH_HOME unset: not visible
    home = tmp_path / "torch"
    target = home / "hub" / "checkpoints" / weights.WEIGHTS[weights.D20_WEIGHTS]["file"]
    target.parent.mkdir(parents=True)
    monkeypatch.setenv("TORCH_HOME", str(home))
    pf = build(tmp_path, step="weights")
    pf.weights((weights.D20_WEIGHTS,), file_required=False)
    assert statuses(pf, "weights resnet50_imagenet1k_v2 file") == ["WARN"]        # verify-torch-weights would download it
    target.write_bytes(b"not the weights")
    pf = build(tmp_path, step="d20", hash=True)
    pf.weights((weights.D20_WEIGHTS,), file_required=True)
    assert statuses(pf, "weights resnet50_imagenet1k_v2 file") == ["FAIL"]
    pending = {"weights": {weights.D20_WEIGHTS: {"file": weights.WEIGHTS[weights.D20_WEIGHTS]["file"], "sha256": weights.PENDING}}}
    monkeypatch.setattr(weights, "load_registry", lambda root: pending)
    pf = build(tmp_path, step="d20")
    pf.weights((weights.D20_WEIGHTS,), file_required=True)
    assert statuses(pf, "weights resnet50_imagenet1k_v2 registry") == ["FAIL"]


def _cache(run, n: int = 3) -> None:
    from tmmri.neural.data import MANIFEST_COLUMNS

    base = run.path / "neural_cache" / "acdc_dev"

    def work() -> None:
        base.mkdir(parents=True, exist_ok=True)
        rows = []
        for i in range(n):
            path = base / f"synthetic{i:03d}.npz"
            np.savez(path, ed=np.full((2, 4, 4), i, np.float32))
            rows.append({c: "" for c in MANIFEST_COLUMNS} | {"sample_id": f"synthetic{i:03d}", "partition": "acdc_dev",
                                                              "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        pd.DataFrame(rows).to_csv(base / "manifest.csv", index=False, lineterminator="\n")

    run_stage(run, "build-tensor-cache:acdc_dev", work, [], [base], config_hash="synthetic", code=CODE)


def test_upload_cpe1_set_pin_and_cache(tmp_path, monkeypatch):
    ctx = make_run(tmp_path)
    _cache(ctx.run)
    real = preflight.Preflight.f9_config

    def unpinned(self):                                          # independent of the repository's pin (1.6.0 since 3 October 2026)
        cfg = real(self)
        cfg["backend"]["modal"]["client_version"] = None
        return cfg

    monkeypatch.setattr(preflight.Preflight, "f9_config", unpinned)
    pf = build(tmp_path, step="upload")
    preflight.step_upload(pf)
    assert statuses(pf, "Modal client pin") == ["FAIL"]
    assert statuses(pf, "CPE-1 upload set") == ["PASS"]
    assert statuses(pf, "artifact neural_cache/acdc_dev") == ["PASS"]

    def pinned(self):
        cfg = real(self)
        cfg["backend"]["modal"]["client_version"] = "1.2.3"
        return cfg

    monkeypatch.setattr(preflight.Preflight, "f9_config", pinned)
    (ctx.run.path / "neural_cache" / "acdc_dev" / "extra.txt").write_text("not permitted", encoding="utf-8")
    pf = build(tmp_path, step="upload")
    preflight.step_upload(pf)
    assert statuses(pf, "Modal client pin") == ["PASS"]
    assert statuses(pf, "CPE-1 upload set") == ["FAIL"]          # cpe1_upload_set refuses any other file
    assert statuses(pf, "artifact neural_cache/acdc_dev") == ["FAIL"]   # the recorded directory hash changed


def test_modal_token_existence_only(tmp_path, monkeypatch, capsys):
    profile = tmp_path / "profile"
    profile.mkdir()
    monkeypatch.setenv("USERPROFILE", str(profile))
    pf = build(tmp_path, step="modal-qualify")
    pf.modal_token()
    assert pf.checks[-1].status == "FAIL"
    (profile / ".modal.toml").write_text(f"token_secret = '{SECRET}'\n", encoding="utf-8")
    pf = build(tmp_path, step="modal-qualify")
    pf.modal_token()
    assert pf.checks[-1].status == "PASS" and SECRET not in pf.checks[-1].detail
    monkeypatch.setattr(preflight, "detect_side", lambda: "container")
    pf = build(tmp_path, step="upload")
    pf.modal_token()
    assert pf.checks[-1].status == "UNKNOWN"                     # the host's .modal.toml is not visible in a container
    monkeypatch.setenv("MODAL_TOKEN_ID", SECRET)
    pf = build(tmp_path, step="upload")
    pf.modal_token()
    assert pf.checks[-1].status == "PASS" and SECRET not in pf.checks[-1].detail
    _, lines = invoke(tmp_path, "modal-qualify", capsys=capsys)
    assert not any(SECRET in line for line in lines)


# ----------------------------------------------------------------------------- disk, git, tags


def test_disk_thresholds(tmp_path, monkeypatch):
    import collections

    usage = collections.namedtuple("usage", "total used free")
    for free_gib, expected in ((1, "FAIL"), (10, "WARN"), (100, "PASS")):
        monkeypatch.setattr(preflight.shutil, "disk_usage", lambda p, f=free_gib: usage(0, 0, f * 2 ** 30))
        pf = build(tmp_path)
        pf.disk(tmp_path)
        assert pf.checks[-1].status == expected


def test_git_clean_strict_and_lenient(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "git_dirty", lambda root=None: (True, [" M src/tmmri/stages.py"]))
    pf = build(tmp_path, step="f8")
    pf.git_clean()
    assert pf.checks[-1].status == "FAIL"
    pf = build(tmp_path, step="upload")
    pf.git_clean()
    assert pf.checks[-1].status == "WARN"
    monkeypatch.setattr(preflight, "git_dirty", lambda root=None: (False, []))
    pf = build(tmp_path, step="f8")
    pf.git_clean()
    assert pf.checks[-1].status == "PASS"


def test_tag_v601_states(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "protocol_frozen", lambda tag="protocol-v6.0.0", root=None, *, signed=False: signed)
    pf = build(tmp_path, step="mms2-evaluate")
    assert pf.tag_v601() and pf.checks[-1].status == "PASS"
    monkeypatch.setattr(stages, "protocol_frozen", lambda tag="protocol-v6.0.0", root=None, *, signed=False: False)
    cases = ((None, False, "absent"), ("commit", False, "lightweight"), ("tag", False, "unsigned"), ("tag", True, "HEAD does not descend"))
    for kind, signed, text in cases:
        monkeypatch.setattr(preflight, "tag_object_type", lambda tag, root=None, k=kind: k)
        monkeypatch.setattr(preflight, "tag_signed", lambda tag, root=None, s=signed: s)
        pf = build(tmp_path, step="mms2-evaluate")
        assert not pf.tag_v601() and pf.checks[-1].status == "FAIL" and text in pf.checks[-1].detail


def test_tag_step_absent_tag_is_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "tag_object_type", lambda tag, root=None: None)
    pf = build(tmp_path, step="tag")
    preflight.step_tag(pf)
    assert [c.status for c in pf.checks if c.name == "tag protocol-v6.0.1"] == ["PASS"]
    assert statuses(pf, "configured amendment tag") == ["PASS"]


# ----------------------------------------------------------------------------- M&Ms-2 archive (synthetic)


def _synthetic_archive(tmp_path: Path, payload: bytes = b"synthetic archive bytes") -> Path:
    folder = tmp_path / "archive_area"
    folder.mkdir()
    (folder / "SYNTHETIC_FIXTURE").write_text("synthetic\n", encoding="utf-8")
    archive = folder / "MnM2.rar"
    archive.write_bytes(payload)
    (folder / "SYNTHETIC_FIXTURE.archive.json").write_text(json.dumps({"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}),
                                                          encoding="utf-8")
    return archive


def test_mms2_archive_size_and_hash_gate(tmp_path, monkeypatch):
    from tmmri.ingest import sources

    archive = _synthetic_archive(tmp_path)
    calls = []
    real_sha = sources.sha256_file
    monkeypatch.setattr(sources, "sha256_file", lambda p: calls.append(p) or real_sha(p))
    monkeypatch.setattr(stages, "protocol_frozen", lambda tag="protocol-v6.0.0", root=None, *, signed=False: False)
    pf = build(tmp_path, step="mms2-extract", archive=str(archive), hash=True)
    preflight.step_mms2_extract(pf)
    assert statuses(pf, "M&Ms-2 archive size") == ["PASS"]
    assert statuses(pf, "M&Ms-2 archive SHA-256") == ["WARN"] and calls == []     # no archive byte before the signed tag
    assert statuses(pf, "tag protocol-v6.0.1") == ["FAIL"]
    monkeypatch.setattr(stages, "protocol_frozen", lambda tag="protocol-v6.0.0", root=None, *, signed=False: True)
    pf = build(tmp_path, step="mms2-extract", archive=str(archive), hash=True)
    preflight.step_mms2_extract(pf)
    assert statuses(pf, "M&Ms-2 archive SHA-256") == ["PASS"] and calls
    archive.write_bytes(b"truncated")
    pf = build(tmp_path, step="mms2-extract", archive=str(archive))
    preflight.step_mms2_extract(pf)
    assert statuses(pf, "M&Ms-2 archive size") == ["FAIL"]


def test_mms2_destination_states(tmp_path):
    archive = _synthetic_archive(tmp_path)
    pf = build(tmp_path, step="mms2-extract", archive=str(archive))
    preflight.step_mms2_extract(pf)
    assert statuses(pf, "destination") == ["PASS"]
    area = tmp_path / "mms2_area"
    area.mkdir()
    (area / "stray.txt").write_text("x", encoding="utf-8")
    pf = build(tmp_path, step="mms2-extract", archive=str(archive))
    preflight.step_mms2_extract(pf)
    assert statuses(pf, "destination") == ["FAIL"]                # extract-mms2 refuses a non-empty area without a receipt
    (area / "MMS2_EXTRACTION_RECEIPT.json").write_text("{}", encoding="utf-8")
    pf = build(tmp_path, step="mms2-extract", archive=str(archive))
    preflight.step_mms2_extract(pf)
    assert statuses(pf, "once-only extraction") == ["WARN"]


# ----------------------------------------------------------------------------- whole steps


def record_many(run, name: str, rels: list[str], *, status: str = "BUILT") -> None:
    paths = [run.path / rel for rel in rels]

    def work() -> None:
        for p in paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f"synthetic {p.name}\n", encoding="utf-8")

    run_stage(run, name, work, [], paths, config_hash="synthetic", code=CODE, status=status)


def test_eval_historical_on_a_synthetic_run(tmp_path, capsys):
    ctx = make_run(tmp_path)
    record_many(ctx.run, "evaluate:acdc_test", ["reconstructed/bhat_acdc_test.parquet", "inference/d_model_acdc_test.parquet"], status="EVALUATED")
    record_many(ctx.run, "build-tensor-cache:acdc_test", ["neural_cache/acdc_test/manifest.csv", "neural_cache/acdc_test/a.npz"])
    code, lines = invoke(tmp_path, "eval-historical", "--split", "acdc_test", capsys=capsys)
    text = "\n".join(lines)
    assert code == 1                                                     # no FROZEN.json
    assert "FAIL    freeze .stages/FROZEN.json: missing" in text
    assert "WARN    freeze F9 (neural_f9/f9_bundle.json): missing" in text   # absent freeze: not executed, does not block the others
    assert "PASS    record evaluate:acdc_test" in text
    assert "PASS    artifact reconstructed/bhat_acdc_test.parquet" in text
    assert "WARN    once-only build-tensor-cache:acdc_test" in text and "RE-RUN" in text
    assert "WARN    inputs evaluate-alt-bundles d20:acdc_test" in text
    assert "PASS    once-only evaluate-alt-bundles:f9:acdc_test" in text
    assert "split mms_external" not in text and "tag protocol-v6.0.0" not in text    # only the requested partition
    _, lines = invoke(tmp_path, "eval-historical", "--split", "mms2_external", capsys=capsys)
    assert any(line.startswith("FAIL    --split") for line in lines)


def test_trust_fit_requires_every_input(tmp_path, capsys):
    ctx = make_run(tmp_path)
    record_many(ctx.run, "validate-cohort", ["contracts/split_manifest.parquet", "contracts/cohort_ACDC.parquet"])
    record_many(ctx.run, "build-semantics:ACDC", ["semantics/B_ACDC.parquet"])
    record_many(ctx.run, "extract-formal:ACDC", ["formal/A_ACDC.parquet"])
    record_many(ctx.run, "fit-predictor", ["formal/d_model_crossfit_ACDC_dev.parquet"])
    code, lines = invoke(tmp_path, "trust-fit", capsys=capsys)
    status = {line.split(":", 1)[0][8:]: line[:7].strip() for line in lines if line.startswith(("PASS", "FAIL", "WARN"))}
    assert code == 1
    assert status["trust input audited_predictor"] == "PASS" and status["trust input descriptors"] == "PASS"
    assert status["trust input d20_predictor"] == "FAIL" and status["trust input f9"] == "FAIL" and status["trust input f9b"] == "FAIL"
    assert status["once-only fit-trust"] == "PASS"


def test_f8_step_on_a_synthetic_run(tmp_path, capsys):
    ctx = make_run(tmp_path)
    for name, rel in (("fit-transition", "oof/bhat_oof_ACDC_dev.parquet"), ("fit-predictor", "formal/d_model_crossfit_ACDC_dev.parquet"),
                      ("build-semantics:ACDC", "semantics/B_ACDC.parquet"), ("validate-cohort", "contracts/split_manifest.parquet")):
        record(ctx.run, name, rel)
    _, lines = invoke(tmp_path, "f8", capsys=capsys)
    text = "\n".join(lines)
    assert "FAIL    freeze .stages/FROZEN.json: missing" in text
    assert "PASS    artifact oof/bhat_oof_ACDC_dev.parquet" in text
    assert "FAIL    artifact oof/inner_oof_ACDC_dev_fold1.parquet: missing" in text   # gate-rules inner OOF not recorded
    assert "PASS    M&Ms-2 not yet extracted or staged" in text
    assert "PASS    once-only fit-cart-surrogate" in text
