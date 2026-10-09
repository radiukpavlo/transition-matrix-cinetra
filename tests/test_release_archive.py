from __future__ import annotations

import zipfile
from pathlib import Path

import yaml

import tmmri.release as release_module


def test_release_archive_uses_configured_project_root(tmp_path: Path, monkeypatch) -> None:
    artifact = tmp_path / "reports" / "artifact.txt"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("stable\n", encoding="utf-8")
    manifest = tmp_path / "provenance" / "release_manifest_isolated.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        yaml.safe_dump(
            {
                "release_id": "isolated",
                "output_artifacts": [{"path": str(artifact)}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(release_module, "PROJECT_ROOT", tmp_path)

    archive, sidecar = release_module.write_release_archive("isolated", manifest)

    assert archive == tmp_path / "artifacts" / "releases" / "isolated_release_artifacts.zip"
    assert sidecar == tmp_path / "artifacts" / "releases" / "isolated_release_archive_manifest.yaml"
    assert archive.is_file() and sidecar.is_file()
    with zipfile.ZipFile(archive) as package:
        assert {Path(name).name for name in package.namelist()} == {
            "artifact.txt",
            "release_manifest_isolated.yaml",
        }


def test_release_manifest_paths_are_relative_to_the_root_not_the_working_directory(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "root"
    artifact = root / "artifacts" / "reports" / "artifact.txt"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("stable\n", encoding="utf-8")
    legacy_only = root / "legacy" / "v5" / "artifacts" / "reports" / "legacy_only.txt"
    legacy_only.parent.mkdir(parents=True)
    legacy_only.write_text("historical\n", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(release_module, "PROJECT_ROOT", root)

    manifest = release_module.write_release_manifest(
        "isolated",
        "test",
        ["artifacts/reports/artifact.txt", "artifacts/reports/legacy_only.txt"],
    )

    assert manifest == root / "artifacts" / "provenance" / "release_manifest_isolated.yaml"
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    # Root-relative, and a missing output is never completed from legacy/v5.
    assert [item["path"] for item in payload["output_artifacts"]] == ["artifacts/reports/artifact.txt"]

    archive, _ = release_module.write_release_archive("isolated", manifest)
    with zipfile.ZipFile(archive) as package:
        assert sorted(package.namelist()) == [
            "artifacts/provenance/release_manifest_isolated.yaml",
            "artifacts/reports/artifact.txt",
        ]

