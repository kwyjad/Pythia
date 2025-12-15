# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Codex-safe guard for optional yamllint dependency in CI tests."""

from __future__ import annotations

import shutil

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("yamllint") is None,
    reason="yamllint not available in this environment",
)


def test_yamllint_presence() -> None:
    """Assert yamllint is discoverable when present."""
    assert shutil.which("yamllint") is not None
