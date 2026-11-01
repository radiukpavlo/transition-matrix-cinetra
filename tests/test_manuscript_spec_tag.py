"""Protocol-v6.0.3 Decision D47: the specification check of ``build-manuscript-tables`` compares ``SPECIFICATION.md``
with the newest signed protocol tag the checked-out commit descends from (amendment tags included).

A synthetic git repository under tmp_path: signed tags are written with ``git mktag`` and a placeholder SSH signature
block (:func:`tmmri.stages.protocol_frozen` checks for the block, not the key), so no key or real tag is needed.
"""
from __future__ import annotations

import subprocess

import pytest

from tmmri import manuscript_tables as mt

SIGNATURE = "-----BEGIN SSH SIGNATURE-----\nU1NIU0lH\n-----END SSH SIGNATURE-----\n"


def _git(repo, *args, stdin=None) -> str:
    cmd = ["git", "-c", "user.name=Synthetic", "-c", "user.email=synthetic@example.invalid", "-c", "commit.gpgsign=false",
           "-c", "tag.gpgsign=false", "-c", "core.autocrlf=false", *args]
    data = stdin.encode("utf-8") if stdin is not None else None          # bytes: text mode would write CRLF on Windows
    return subprocess.run(cmd, cwd=repo, input=data, capture_output=True, check=True).stdout.decode("utf-8").strip()


def _commit(repo, spec: str) -> str:
    (repo / "SPECIFICATION.md").write_text(spec, encoding="utf-8", newline="\n")
    _git(repo, "add", "SPECIFICATION.md")
    _git(repo, "commit", "-q", "-m", spec.strip())
    return _git(repo, "rev-parse", "HEAD")


def _signed_tag(repo, name: str, commit: str) -> None:
    body = (f"object {commit}\ntype commit\ntag {name}\ntagger Synthetic <synthetic@example.invalid> 1793232000 +0000\n\n"
            f"{name}\n{SIGNATURE}")
    _git(repo, "update-ref", f"refs/tags/{name}", _git(repo, "mktag", stdin=body))


@pytest.fixture
def repo(tmp_path):
    """main: A (v6.0.9 signed) - B (v6.0.10 signed) - C (v6.0.11 annotated, unsigned; HEAD); side: B - D (v6.0.12
    signed, not an ancestor of HEAD).  By text, protocol-v6.0.9 sorts above protocol-v6.0.10."""

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    a = _commit(repo, "spec 9\n")
    _signed_tag(repo, "protocol-v6.0.9", a)
    b = _commit(repo, "spec 10\n")
    _signed_tag(repo, "protocol-v6.0.10", b)
    _git(repo, "checkout", "-q", "-b", "side")
    _signed_tag(repo, "protocol-v6.0.12", _commit(repo, "spec 12\n"))
    _git(repo, "checkout", "-q", "main")
    _commit(repo, "spec 11\n")
    _git(repo, "tag", "-a", "-m", "unsigned", "protocol-v6.0.11")
    _git(repo, "tag", "protocol-v6.0.13", b)                                   # lightweight: never a protocol tag
    return repo


def test_newest_signed_ancestor_tag_is_chosen_by_version(repo):
    assert set(mt.protocol_tag_names(repo)) == {f"protocol-v6.0.{k}" for k in (9, 10, 11, 12, 13)}
    assert mt.specification_tag(repo) == "protocol-v6.0.10"


def test_unsigned_lightweight_and_non_ancestor_tags_are_skipped(repo):
    _git(repo, "tag", "-d", "protocol-v6.0.10")
    assert mt.specification_tag(repo) == "protocol-v6.0.9"                     # v6.0.11 unsigned, v6.0.12 off HEAD, v6.0.13 lightweight
    _git(repo, "checkout", "-q", "side")
    assert mt.specification_tag(repo) == "protocol-v6.0.12"                    # signed and now an ancestor


def test_specification_check_matches_and_mismatches(repo):
    spec = repo / "SPECIFICATION.md"
    tag = mt.specification_tag(repo)
    assert spec.read_text(encoding="utf-8") == "spec 11\n" and mt.spec_matches_tag(spec, tag) is False   # the unsigned draft
    spec.write_bytes(b"spec 10\r\n")                                           # line endings are normalized
    assert mt.spec_matches_tag(spec, tag) is True


def test_no_signed_tag_or_no_git_gives_none(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    assert mt.protocol_tag_names(tmp_path) == [] and mt.specification_tag(tmp_path) is None
