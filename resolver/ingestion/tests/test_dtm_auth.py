#!/usr/bin/env python3
# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for single-key DTM authentication helper."""

import pytest

from resolver.ingestion.dtm_auth import get_dtm_api_key


def test_missing_key_returns_none(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.delenv("DTM_API_KEY", raising=False)
    with caplog.at_level("DEBUG"):
        key = get_dtm_api_key()
    assert key is None
    assert "not present" in caplog.text or "No DTM API credentials" in caplog.text


def test_key_is_logged_masked(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setenv("DTM_API_KEY", "abc123456789")
    with caplog.at_level("INFO"):
        key = get_dtm_api_key()
    assert key.endswith("6789")
    assert "DTM_API_KEY ending with" in caplog.text or "API key ending with" in caplog.text
    assert "abc123456789" not in caplog.text
