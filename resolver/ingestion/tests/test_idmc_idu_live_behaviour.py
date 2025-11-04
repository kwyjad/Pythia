from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest
import json

from resolver.ingestion.idmc import client, config
from resolver.ingestion.idmc.http import HttpRequestError


@pytest.fixture
def cfg(tmp_path: Path) -> config.IdmcConfig:
    cfg = config.load()
    cfg.cache.dir = str(tmp_path / "idmc_cache")
    cfg.cache.ttl_seconds = 0
    cfg.api.countries = ["AFG", "PAK", "KEN"]
    return cfg


def _http_payload(rows: List[Dict[str, Any]]) -> bytes:
    return json.dumps({"data": rows}).encode("utf-8")


def test_idmc_idu_builds_postgrest_query(monkeypatch, cfg: config.IdmcConfig) -> None:
    captured: Dict[str, Any] = {}

    def fake_http_get(url: str, **kwargs):
        captured["url"] = url
        body = _http_payload([])
        return 200, {}, body, {
            "attempts": 1,
            "retries": 0,
            "duration_s": 0.1,
            "backoff_s": 0.0,
            "wire_bytes": len(body),
            "body_bytes": len(body),
            "attempt_durations_s": [0.1],
        }

    monkeypatch.setattr(client, "http_get", fake_http_get)

    frame, diagnostics = client.fetch_idu_json(
        cfg,
        window_start=None,
        window_end=None,
        chunk_start=pd.Timestamp("2024-01-01").date(),
        chunk_end=pd.Timestamp("2024-01-31").date(),
        only_countries=["AFG", "PAK"],
        network_mode="live",
    )

    assert diagnostics["mode"] == "online"
    url = captured["url"]
    assert "displacement_start_date=gte.2024-01-01" in url
    assert "displacement_end_date=lte.2024-01-31" in url
    assert "select=iso3,figure" in url
    assert "iso3=in.(AFG,PAK)" in url
    assert frame.empty


def test_idmc_idu_batches_iso3s(monkeypatch, cfg: config.IdmcConfig) -> None:
    countries = [f"C{i:02d}" for i in range(60)]
    cfg.api.countries = countries
    seen_urls: List[str] = []

    def fake_http_get(url: str, **kwargs):
        seen_urls.append(url)
        body = _http_payload([])
        return 200, {}, body, {
            "attempts": 1,
            "retries": 0,
            "duration_s": 0.05,
            "backoff_s": 0.0,
            "wire_bytes": len(body),
            "body_bytes": len(body),
            "attempt_durations_s": [0.05],
        }

    monkeypatch.setattr(client, "http_get", fake_http_get)

    client.fetch_idu_json(
        cfg,
        window_start=None,
        window_end=None,
        chunk_start=pd.Timestamp("2024-02-01").date(),
        chunk_end=pd.Timestamp("2024-02-29").date(),
        only_countries=countries,
        network_mode="live",
    )

    expected_batches = (len(countries) + client.ISO3_BATCH_SIZE - 1) // client.ISO3_BATCH_SIZE
    assert len(seen_urls) == expected_batches
    for url in seen_urls:
        assert "iso3=in." in url


def test_idmc_idu_hdx_fallback_on_http_error(monkeypatch, cfg: config.IdmcConfig) -> None:
    monkeypatch.setenv("IDMC_ALLOW_HDX_FALLBACK", "1")

    def fake_http_get(*args, **kwargs):
        raise HttpRequestError("boom", {"status": 502, "attempts": 1, "retries": 0})

    fallback_rows = pd.DataFrame(
        {
            "iso3": ["AFG"],
            "figure": [100],
            "displacement_end_date": ["2024-03-15"],
        }
    )

    def fake_hdx_fetch_latest_csv():
        return fallback_rows, {"dataset": "idu"}

    monkeypatch.setattr(client, "http_get", fake_http_get)
    monkeypatch.setattr(client, "_hdx_fetch_latest_csv", fake_hdx_fetch_latest_csv)

    frame, diagnostics = client.fetch_idu_json(
        cfg,
        window_start=None,
        window_end=None,
        chunk_start=pd.Timestamp("2024-03-01").date(),
        chunk_end=pd.Timestamp("2024-03-31").date(),
        only_countries=["AFG"],
        network_mode="live",
    )

    assert not frame.empty
    assert diagnostics["mode"] == "fallback"
    assert diagnostics["fallback"]["type"] == "hdx"
    assert "dataset" in diagnostics["fallback"]
