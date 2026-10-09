"""Pinned torchvision weights (``verify-torch-weights``): host allow-list, prefix check and hash records; no network."""
from __future__ import annotations

import hashlib
import io
import shutil

import pytest
import yaml

from tmmri.neural import weights as w


class _Response(io.BytesIO):
    def __init__(self, data: bytes, url: str) -> None:
        super().__init__(data)
        self._url = url

    def geturl(self) -> str:
        return self._url


def _opener(payloads):
    def open_url(url, timeout=None):
        return _Response(payloads[url], url)

    return open_url


def test_recorded_weights_match_the_file_name_prefixes(repo_root):
    registry = yaml.safe_load((repo_root / "registry" / "weights.yaml").read_text(encoding="utf-8"))
    assert registry["source_host"] == "download.pytorch.org"
    for key, spec in w.WEIGHTS.items():
        entry = registry["weights"][key]
        assert entry["file"] == spec["file"] and len(entry["sha256"]) == 64
        assert entry["sha256"].startswith(w.file_prefix(key)) and entry["url"].startswith("https://download.pytorch.org/models/")
    deep = yaml.safe_load((repo_root / "configs" / "deep_encoder.yaml").read_text(encoding="utf-8"))
    assert deep["encoder"]["weights_sha256"] == registry["weights"][w.D20_WEIGHTS]["sha256"]
    assert deep["status"] == "implemented" and deep["runtime"]["batch_size"] == 32


def test_only_download_pytorch_org_is_accepted(tmp_path):
    for url in ("http://download.pytorch.org/models/x.pth", "https://example.org/models/resnet50-11ad3fa6.pth",
                "https://download.pytorch.org.evil.test/x.pth"):
        with pytest.raises(w.WeightsError, match="only https://download.pytorch.org"):
            w.download(url, tmp_path / "x.pth", opener=_opener({url: b""}))
    redirected = "https://download.pytorch.org/models/a.pth"

    def bounce(url, timeout=None):
        return _Response(b"x", "https://mirror.example/a.pth")

    with pytest.raises(w.WeightsError):
        w.download(redirected, tmp_path / "a.pth", opener=bounce)


def test_verify_downloads_pins_and_then_refuses_changes(tmp_path, monkeypatch, repo_root):
    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    (root / "registry").mkdir()
    text = (repo_root / "configs" / "deep_encoder.yaml").read_text(encoding="utf-8")
    pending = "\n".join("  weights_sha256: pending_first_use   # recorded at first use" if line.strip().startswith("weights_sha256:")
                        else line for line in text.splitlines()) + "\n"
    (root / "configs" / "deep_encoder.yaml").write_text(pending, encoding="utf-8")
    payloads, table = {}, {}
    for key, arch in (("small", "resnet18"), (w.D20_WEIGHTS, "resnet50")):
        data = f"synthetic {arch} weights".encode()
        sha = hashlib.sha256(data).hexdigest()
        url = f"https://download.pytorch.org/models/{arch}-{sha[:8]}.pth"
        payloads[url] = data
        table[key] = {"torchvision_weights": "synthetic", "architecture": arch, "file": f"{arch}-{sha[:8]}.pth", "url": url, "used_by": "test"}
    monkeypatch.setattr(w, "WEIGHTS", table)
    home = tmp_path / "torch"
    report = w.verify_torch_weights(root, home=home, opener=_opener(payloads))
    assert {v["action"] for v in report["weights"].values()} == {"downloaded"} and report["deep_encoder_config"] == "recorded"
    d20_sha = report["weights"][w.D20_WEIGHTS]["sha256"]
    assert yaml.safe_load((root / "configs" / "deep_encoder.yaml").read_text(encoding="utf-8"))["encoder"]["weights_sha256"] == d20_sha
    again = w.verify_torch_weights(root, home=home, allow_download=False)
    assert {v["registry"] for v in again["weights"].values()} == {"matches"} and again["deep_encoder_config"] == "unchanged"
    path, sha = w.require_weights(w.D20_WEIGHTS, root, home)
    assert sha == d20_sha and path.parent == home / "hub" / "checkpoints"
    path.write_bytes(b"tampered")
    with pytest.raises(w.WeightsError):
        w.require_weights(w.D20_WEIGHTS, root, home)
    with pytest.raises(w.WeightsError, match="does not start with the file-name prefix"):
        w.verify_torch_weights(root, home=home, allow_download=False)
    shutil.rmtree(home)
    with pytest.raises(w.WeightsError, match="missing"):
        w.verify_torch_weights(root, home=home, allow_download=False)
