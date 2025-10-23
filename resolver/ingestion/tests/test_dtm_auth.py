#!/usr/bin/env python3
"""Tests for DTM API authentication."""

import os
import pytest

from resolver.ingestion.dtm_auth import (
    get_dtm_api_key,
    get_auth_headers,
    check_api_key_configured,
)


def test_get_dtm_api_key_success(monkeypatch):
    """Test successful API key retrieval."""
    test_key = "test-api-key-12345"
    monkeypatch.setenv("DTM_API_KEY", test_key)

    key = get_dtm_api_key()
    assert key == test_key


def test_get_dtm_api_key_missing(monkeypatch):
    """Test that missing API key returns None."""
    monkeypatch.delenv("DTM_API_KEY", raising=False)

    key = get_dtm_api_key()
    assert key is None


def test_get_dtm_api_key_empty(monkeypatch):
    """Test that empty API key returns None."""
    monkeypatch.setenv("DTM_API_KEY", "   ")

    key = get_dtm_api_key()
    assert key is None


def test_get_auth_headers(monkeypatch):
    """Test auth headers generation."""
    test_key = "test-subscription-key"
    monkeypatch.setenv("DTM_API_KEY", test_key)

    headers = get_auth_headers()

    assert "Ocp-Apim-Subscription-Key" in headers
    assert headers["Ocp-Apim-Subscription-Key"] == test_key


def test_get_auth_headers_missing_key(monkeypatch):
    """Test that missing API key in headers raises RuntimeError."""
    monkeypatch.delenv("DTM_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        get_auth_headers()


def test_check_api_key_configured_true(monkeypatch):
    """Test that check_api_key_configured returns True when key is set."""
    monkeypatch.setenv("DTM_API_KEY", "test-key-123")

    assert check_api_key_configured() is True


def test_check_api_key_configured_false_missing(monkeypatch):
    """Test that check_api_key_configured returns False when key is missing."""
    monkeypatch.delenv("DTM_API_KEY", raising=False)

    assert check_api_key_configured() is False


def test_check_api_key_configured_false_empty(monkeypatch):
    """Test that check_api_key_configured returns False when key is empty."""
    monkeypatch.setenv("DTM_API_KEY", "   ")

    assert check_api_key_configured() is False
