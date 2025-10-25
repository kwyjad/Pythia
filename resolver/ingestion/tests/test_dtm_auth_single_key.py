#!/usr/bin/env python3
"""Tests for DTM API authentication."""

import os
import pytest

from resolver.ingestion.dtm_auth import check_api_key_configured, get_dtm_api_key


def test_missing_key_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DTM_API_KEY", raising=False)

    with pytest.raises(RuntimeError) as excinfo:
        get_dtm_api_key()

    assert "DTM_API_KEY not set" in str(excinfo.value)


def test_key_is_logged_masked(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setenv("DTM_API_KEY", "abc123456789")
    caplog.set_level("INFO")

    key = get_dtm_api_key()

    assert key == "abc123456789"
    assert "ending with ...6789" in caplog.text
    assert "abc123456789" not in caplog.text


def test_check_api_key_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DTM_API_KEY", raising=False)
    assert check_api_key_configured() is False

    monkeypatch.setenv("DTM_API_KEY", "   ")
    assert check_api_key_configured() is False

    monkeypatch.setenv("DTM_API_KEY", "secret")
    assert check_api_key_configured() is True
