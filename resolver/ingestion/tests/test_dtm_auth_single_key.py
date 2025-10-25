#!/usr/bin/env python3
"""Tests for single-key DTM authentication helper."""

import pytest

from resolver.ingestion.dtm_auth import get_dtm_api_key


def test_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DTM_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        get_dtm_api_key()


def test_key_is_logged_masked(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setenv("DTM_API_KEY", "abc123456789")
    with caplog.at_level("INFO"):
        key = get_dtm_api_key()
    assert key.endswith("6789")
    assert "DTM_API_KEY ending with" in caplog.text
    assert "abc123456789" not in caplog.text
