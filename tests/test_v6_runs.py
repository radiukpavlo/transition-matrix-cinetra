"""Protocol 6.0.0 run roots and content-addressed stage records."""
from __future__ import annotations

import json

import pytest

from tmmri.config import load_study
from tmmri.runs import DEFAULT_SUBFOLDERS, RunRoot, hash_path, parallel_map, resolve_n_jobs, resolve_source_readers, run_stage

CODE = {"commit": "0" * 40, "dirty": False, "code_sha256": "c" * 64}


def _square(x: int) -> int:
    return x * x


def test_run_root_layout(tmp_path):
    study = load_study()
    run = RunRoot("v6-test-20260928-1200", tmp_path, subfolders=study.study["artifact_subfolders"])
    assert tuple(study.study["artifact_subfolders"]) == DEFAULT_SUBFOLDERS
    assert all((run.path / name).is_dir() for name in DEFAULT_SUBFOLDERS)
    assert json.loads(run.status_path.read_text(encoding="utf-8")) == {"run_id": run.run_id, "stages": {}}
    with pytest.raises(ValueError):
        RunRoot("Bad Run Id", tmp_path)


def test_run_stage_skips_on_identical_inputs_and_reruns_on_changes(tmp_path):
    run = RunRoot("v6-test-20260928-1201", tmp_path)
    source = tmp_path / "input.txt"
    source.write_text("alpha\n", encoding="utf-8")
    target = run.sub("formal") / "out.txt"
    calls: list[str] = []

    def work() -> str:
        calls.append("ran")
        target.write_text(source.read_text(encoding="utf-8").upper(), encoding="utf-8")
        return "ok"

    def go(config_hash: str = "h1", **kwargs):
        return run_stage(run, "extract-formal:ACDC", work, [source], [target], config_hash=config_hash, code=CODE, **kwargs)

    assert go().action == "RAN"
    assert go().action == "SKIPPED" and len(calls) == 1
    assert go(config_hash="h2").action == "RAN"                     # changed config
    assert go(config_hash="h2").action == "SKIPPED"
    source.write_text("beta\n", encoding="utf-8")
    assert go(config_hash="h2").action == "RAN"                     # changed input
    target.unlink()
    assert go(config_hash="h2").action == "RAN"                     # missing output
    assert go(config_hash="h2", params={"profile": "x"}).action == "RAN"
    assert go(config_hash="h2", params={"profile": "x"}, force=True).action == "RAN"
    assert run_stage(run, "extract-formal:ACDC", work, [source], [target], config_hash="h2", params={"profile": "x"},
                     code={**CODE, "code_sha256": "d" * 64}).action == "RAN"   # changed code
    assert len(calls) == 7


def test_stage_record_payload_is_timestamp_free_and_portable(tmp_path):
    run = RunRoot("v6-test-20260928-1202", tmp_path)
    source = tmp_path / "in.bin"
    source.write_bytes(b"x")
    out = run.sub("metrics") / "m.json"
    run_stage(run, "fit-predictor", lambda: out.write_text("{}", encoding="utf-8"), [source], [out], config_hash="h", code=CODE, n_jobs=3)
    record = json.loads(run.record_path("fit-predictor").read_text(encoding="utf-8"))
    payload = json.dumps(record["payload"])
    assert "utc" not in payload and "seconds" not in payload
    assert "started_utc" in record["run_log"]
    assert list(record["payload"]["outputs"]) == ["run:metrics/m.json"]
    assert record["payload"]["n_jobs"] == 3 and record["payload"]["code"] == CODE
    assert {"numpy", "pandas", "nibabel", "joblib"} <= set(record["payload"]["library_versions"])
    status = json.loads(run.status_path.read_text(encoding="utf-8"))
    assert status["stages"]["fit-predictor"]["status"] == "BUILT"


def test_run_stage_rejects_missing_inputs_and_outputs(tmp_path):
    run = RunRoot("v6-test-20260928-1203", tmp_path)
    with pytest.raises(FileNotFoundError):
        run_stage(run, "s", lambda: None, [tmp_path / "absent"], [], config_hash="h", code=CODE)
    source = tmp_path / "in"
    source.write_text("1", encoding="utf-8")
    with pytest.raises(RuntimeError, match="did not produce"):
        run_stage(run, "s", lambda: None, [source], [run.path / "never.txt"], config_hash="h", code=CODE)


def test_hashing_and_parallel_helpers(tmp_path, monkeypatch):
    folder = tmp_path / "d"
    (folder / "sub").mkdir(parents=True)
    (folder / "a.txt").write_text("a", encoding="utf-8")
    (folder / "sub" / "b.txt").write_text("b", encoding="utf-8")
    first = hash_path(folder)
    (folder / ".x.123.tmp").write_text("ignored", encoding="utf-8")
    assert hash_path(folder) == first and hash_path(tmp_path / "none") is None
    assert parallel_map(_square, range(7), n_jobs=2) == [x * x for x in range(7)]
    monkeypatch.setenv("TMMRI_N_JOBS", "5")
    monkeypatch.setenv("TMMRI_SOURCE_READERS", "2")
    assert resolve_n_jobs() == 5 and resolve_n_jobs(3) == 3
    assert resolve_source_readers() == 2
    monkeypatch.delenv("TMMRI_SOURCE_READERS")
    assert resolve_source_readers() == 4


def test_load_study_hashes(tmp_path):
    study = load_study()
    assert "configs/study_v6.yaml" in study.file_hashes and "registry/semantic_registry.yaml" in study.file_hashes
    assert len(study.file_hashes) == len(set(study.study["config_files"]) | set(study.study["registry_files"]))
    assert study.config_hash == load_study().config_hash
    assert len(study.transition_targets()) == study.registry["counts"]["transition_candidates"]
    assert study.get("splits.acdc.acdc_dev.outer_folds.hash_salt") == "acdc-fold-v6|"
    assert study.configs["runtime"]["TMMRI_N_JOBS"] == "12"


def test_code_version_is_cached_per_process_and_uses_no_optional_locks(monkeypatch):
    """``git status`` runs once per process and root, never taking optional index locks (container bind mounts)."""
    import subprocess

    from tmmri import runs

    calls = []
    real = subprocess.run

    def spy(args, *a, **k):
        calls.append(list(args))
        return real(args, *a, **k)

    monkeypatch.setattr(runs.subprocess, "run", spy)
    runs.code_version.cache_clear()
    try:
        first = runs.code_version()
        second = runs.code_version()
    finally:
        runs.code_version.cache_clear()
    assert first == second and set(first) == {"commit", "dirty", "code_sha256"}
    assert first is not second                                   # callers get their own dict
    git = [c for c in calls if c and c[0] == "git"]
    assert len(git) == 2 and all(c[1] == "--no-optional-locks" for c in git)
    assert any("status" in c for c in git) and any("rev-parse" in c for c in git)
