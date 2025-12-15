# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests covering IDMC HTTP diagnostics for timeouts and fallbacks."""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from resolver.ingestion.idmc import client, config, http
from resolver.ingestion.idmc.http import HttpRequestError


def _minimal_diag(status: Any) -> Dict[str, Any]:
    return {
        "status": status,
        "attempts": 1,
        "retries": 0,
        "duration_s": 0.05,
        "backoff_s": 0.0,
        "exceptions": [
            {
                "attempt": 1,
                "type": "Timeout",
                "message": "boom",
            }
        ],
        "wire_bytes": 0,
        "body_bytes": 0,
        "streamed_to": None,
        "retry_after_s": [],
        "rate_limit_wait_s": [],
        "planned_sleep_s": [],
        "attempt_durations_s": [0.05],
        "timeout": status == "timeout",
        "exception_type": "Timeout" if status == "timeout" else "Error",
}


def _prepare_cfg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> config.IdmcConfig:
    cache_dir = tmp_path / "cache"
    diag_dir = tmp_path / "diag"
    cache_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("IDMC_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(client, "RAW_DIAG_DIR", str(diag_dir))
    cfg = config.load()
    cfg.cache.dir = str(cache_dir)
    cfg.cache.ttl_seconds = 0
    return cfg


def _raise_timeout():
    raise HttpRequestError("timeout", _minimal_diag("timeout"))


def test_idmc_timeouts_classified(monkeypatch, caplog, tmp_path):
    cfg = _prepare_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr(
        client,
        "_probe_idu_schema",
        lambda *args, **kwargs: ("displacement_date", client.DEFAULT_POSTGREST_COLUMNS, {}),
    )
    monkeypatch.setattr(http, "http_get", lambda *args, **kwargs: _raise_timeout())
    monkeypatch.setattr(client, "http_get", lambda *args, **kwargs: _raise_timeout())
    caplog.set_level(logging.DEBUG)

    data, diagnostics = client.fetch(
        cfg,
        network_mode="live",
        window_start=date(2024, 1, 1),
        window_end=date(2024, 1, 31),
        only_countries=["USA"],
        allow_hdx_fallback=False,
    )

    assert data["monthly_flow"].empty
    assert (diagnostics["http_attempt_summary"].get("timeouts") or 0) >= 1
    attempts = diagnostics.get("attempts") or []
    assert any(entry.get("status") == "timeout" for entry in attempts)
    assert "status=timeout" in caplog.text
    assert diagnostics["filters"].get("zero_rows_reason") in {"timeout", "timeout_fallback_empty"}


def test_idmc_fallback_disabled_by_default(monkeypatch, tmp_path):
    cfg = _prepare_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr(
        client,
        "_probe_idu_schema",
        lambda *args, **kwargs: ("displacement_date", client.DEFAULT_POSTGREST_COLUMNS, {}),
    )
    monkeypatch.setattr(http, "http_get", lambda *args, **kwargs: _raise_timeout())
    monkeypatch.setattr(client, "http_get", lambda *args, **kwargs: _raise_timeout())

    data, diagnostics = client.fetch(
        cfg,
        network_mode="live",
        window_start=date(2024, 1, 1),
        window_end=date(2024, 1, 31),
        only_countries=["USA"],
        allow_hdx_fallback=False,
    )

    assert data["monthly_flow"].empty
    assert not diagnostics.get("fallback_used")
    assert diagnostics.get("fallback") is None or not diagnostics.get("fallback", {}).get("used")


def test_idmc_fallback_enabled_writes_rows(monkeypatch, tmp_path):
    cfg = _prepare_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr(
        client,
        "_probe_idu_schema",
        lambda *args, **kwargs: ("displacement_date", client.DEFAULT_POSTGREST_COLUMNS, {}),
    )
    monkeypatch.setattr(http, "http_get", lambda *args, **kwargs: _raise_timeout())
    monkeypatch.setattr(client, "http_get", lambda *args, **kwargs: _raise_timeout())

    fallback_frame = pd.DataFrame(
        [
            {
                "iso3": "USA",
                "displacement_date": "2024-01-15",
                "figure": 123,
                "displacement_type": "Conflict",
            }
        ]
    )
    monkeypatch.setattr(
        client,
        "_hdx_fetch_latest_csv",
        lambda: (fallback_frame, {"resource_url": "https://example.org/csv"}),
    )

    data, diagnostics = client.fetch(
        cfg,
        network_mode="live",
        window_start=date(2024, 1, 1),
        window_end=date(2024, 1, 31),
        only_countries=["USA"],
        allow_hdx_fallback=True,
    )

    fallback_info = diagnostics.get("fallback") or {}
    assert fallback_info.get("used") is True
    assert (fallback_info.get("rows") or 0) == 1
    attempts = diagnostics.get("attempts") or []
    assert any(entry.get("via") == "hdx_fallback" and entry.get("rows", 0) > 0 for entry in attempts)


def test_idmc_summary_includes_http_counters():
    from resolver.ingestion.idmc.cli import _render_summary

    context = {
        "timestamp": "2024-01-01T00:00:00Z",
        "git_sha": "abcdef0",
        "config_source": "test",
        "config_path": "n/a",
        "mode": "live",
        "token_status": "absent",
        "series_display": "flow",
        "date_window": "start=2024-01-01, end=2024-01-31",
        "countries_count": 1,
        "countries_sample_display": "USA",
        "countries_source_display": "config",
        "endpoints_block": "- api: https://example/api",
        "reachability_block": "- DNS ok",
        "http_attempts_block": "- Planned HTTP requests: 1",
        "attempts_block": "- Requests attempted: 1",
        "chunk_attempts_block": "- chunk=full; via=http_get; status=timeout",
        "performance_block": "- Duration (s): 0",
        "rate_limit_block": "- Rate limit (req/s): 1",
        "dataset_block": "- monthly_flow: 1 rows",
        "staging_block": "- resolver/staging/idmc/flow.csv: missing",
        "fallback_block": "- Used: yes",
        "notes_block": "- Network mode: live",
    }

    summary = _render_summary(context)
    assert "## HTTP attempts" in summary
    assert "Planned HTTP requests: 1" in summary
    assert "## Attempt log" in summary
    assert "chunk=full" in summary
    assert "## Fallback" in summary
    assert "Used: yes" in summary
