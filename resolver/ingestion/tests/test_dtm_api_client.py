#!/usr/bin/env python3
"""Tests for DTM API client."""

import json
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

from resolver.ingestion.dtm_client import DTMApiClient


@pytest.fixture
def mock_config():
    """Fixture for DTM API configuration."""
    return {
        "api": {
            "base_url": "https://test.api.example.com/v3",
            "rate_limit_delay": 0,  # No delay for tests
            "timeout": 10,
        }
    }


@pytest.fixture
def mock_api_key(monkeypatch):
    """Fixture to mock DTM API key."""
    monkeypatch.setenv("DTM_API_KEY", "test-api-key-12345")


def test_dtm_api_client_init_success(mock_config, mock_api_key):
    """Test successful DTMApiClient initialization."""
    client = DTMApiClient(mock_config)

    assert client.base_url == "https://test.api.example.com/v3"
    assert client.timeout == 10
    assert client.rate_limit_delay == 0
    assert client.api_key == "test-api-key-12345"


def test_dtm_api_client_init_missing_key(mock_config, monkeypatch):
    """Test DTMApiClient initialization fails with missing API key."""
    monkeypatch.delenv("DTM_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        DTMApiClient(mock_config)


@patch("resolver.ingestion.dtm_client.requests.get")
def test_make_request_success_list_response(mock_get, mock_config, mock_api_key):
    """Test _make_request with list response."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{"id": 1}, {"id": 2}]
    mock_get.return_value = mock_response

    client = DTMApiClient(mock_config)
    result = client._make_request("TestEndpoint")

    assert len(result) == 2
    assert result[0]["id"] == 1
    mock_get.assert_called_once()


@patch("resolver.ingestion.dtm_client.requests.get")
def test_make_request_success_dict_with_data(mock_get, mock_config, mock_api_key):
    """Test _make_request with dict response containing 'data' key."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": [{"id": 3}, {"id": 4}]}
    mock_get.return_value = mock_response

    client = DTMApiClient(mock_config)
    result = client._make_request("TestEndpoint")

    assert len(result) == 2
    assert result[0]["id"] == 3


@patch("resolver.ingestion.dtm_client.requests.get")
def test_make_request_http_counts(mock_get, mock_config, mock_api_key):
    """Test _make_request updates HTTP counts."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = []
    mock_get.return_value = mock_response

    http_counts = {"2xx": 0, "4xx": 0, "5xx": 0, "timeout": 0, "error": 0}
    client = DTMApiClient(mock_config)
    client._make_request("TestEndpoint", http_counts=http_counts)

    assert http_counts["2xx"] == 1
    assert http_counts["last_status"] == 200


@patch("resolver.ingestion.dtm_client.requests.get")
def test_make_request_http_error(mock_get, mock_config, mock_api_key):
    """Test _make_request handles HTTP errors."""
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = requests.HTTPError()
    mock_get.return_value = mock_response

    client = DTMApiClient(mock_config)

    with pytest.raises(requests.HTTPError):
        client._make_request("TestEndpoint")


@patch("resolver.ingestion.dtm_client.requests.get")
def test_get_countries(mock_get, mock_config, mock_api_key):
    """Test get_countries method."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"CountryName": "Ethiopia", "ISO3": "ETH"},
        {"CountryName": "Sudan", "ISO3": "SDN"},
    ]
    mock_get.return_value = mock_response

    client = DTMApiClient(mock_config)
    df = client.get_countries()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "CountryName" in df.columns


@patch("resolver.ingestion.dtm_client.requests.get")
def test_get_idp_admin0_with_params(mock_get, mock_config, mock_api_key):
    """Test get_idp_admin0 with query parameters."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"CountryName": "Ethiopia", "TotalIDPs": 1000}
    ]
    mock_get.return_value = mock_response

    client = DTMApiClient(mock_config)
    df = client.get_idp_admin0(
        country="Ethiopia",
        from_date="2024-01-01",
        to_date="2024-12-31",
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1

    # Verify params were passed correctly
    call_args = mock_get.call_args
    assert call_args[1]["params"]["CountryName"] == "Ethiopia"
    assert call_args[1]["params"]["FromReportingDate"] == "2024-01-01"
    assert call_args[1]["params"]["ToReportingDate"] == "2024-12-31"


@patch("resolver.ingestion.dtm_client.requests.get")
def test_get_idp_admin1(mock_get, mock_config, mock_api_key):
    """Test get_idp_admin1 method."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {
            "CountryName": "Sudan",
            "Admin1Name": "Khartoum",
            "TotalIDPs": 5000,
        }
    ]
    mock_get.return_value = mock_response

    client = DTMApiClient(mock_config)
    df = client.get_idp_admin1(country="Sudan")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "Admin1Name" in df.columns


@patch("resolver.ingestion.dtm_client.requests.get")
def test_get_idp_admin2(mock_get, mock_config, mock_api_key):
    """Test get_idp_admin2 method."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {
            "CountryName": "Lebanon",
            "Admin2Name": "Beirut",
            "TotalIDPs": 2000,
            "Operation": "Displacement due to conflict",
        }
    ]
    mock_get.return_value = mock_response

    client = DTMApiClient(mock_config)
    df = client.get_idp_admin2(
        country="Lebanon", operation="Displacement due to conflict"
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "Admin2Name" in df.columns
