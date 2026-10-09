"""Shared fixtures for the CMR-TRANS-AUDIT acceptance tests (SPECIFICATION.md 6.0.0, Section 15.2).

Fixtures here are opt-in (no autouse, no global side effects), so existing tests are unaffected. Tests that need an
implementation module use ``require_module`` and skip cleanly until it exists; the registry tests run against the
delivered files now. Synthetic fixtures are algorithm controls and must never be presented as cohort results.
"""
from __future__ import annotations

import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SEED_SYNTHETIC = 20260613  # configs/study_v6.yaml seeds.synthetic_controls


@pytest.fixture(scope="session")
def repo_root() -> pathlib.Path:
    return REPO


@pytest.fixture(scope="session")
def registry():
    yaml = pytest.importorskip("yaml", reason="pyyaml required to read registry/semantic_registry.yaml")
    with (REPO / "registry" / "semantic_registry.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="session")
def reason_codes():
    yaml = pytest.importorskip("yaml")
    with (REPO / "registry" / "reason_codes.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def require_module(name: str):
    """Import an implementation module or skip the test with a uniform message."""
    return pytest.importorskip(name, reason=f"{name} not implemented yet (SPECIFICATION.md Section 14.1)")
