#!/usr/bin/env python3
"""Tests for DTM API authentication."""

import os
import pytest

from resolver.ingestion.dtm_auth import get_dtm_api_key, get_auth_headers


def test_get_dtm_api_key_success(monkeypatch):
    """Test successful API key retrieval."""
    test_key = "test-api-key-12345"
    monkeypatch.setenv("DTM_API_KEY", test_key)

    key = get_dtm_api_key()
    assert key == test_key


def test_get_dtm_api_key_missing(monkeypatch):
    """Test that missing API key raises RuntimeError."""
    monkeypatch.delenv("DTM_API_KEY", raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        get_dtm_api_key()

    assert "DTM_API_KEY environment variable not set" in str(exc_info.value)


def test_get_dtm_api_key_empty(monkeypatch):
    """Test that empty API key raises RuntimeError."""
    monkeypatch.setenv("DTM_API_KEY", "   ")

    with pytest.raises(RuntimeError) as exc_info:
        get_dtm_api_key()

    assert "DTM_API_KEY environment variable not set" in str(exc_info.value)


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
