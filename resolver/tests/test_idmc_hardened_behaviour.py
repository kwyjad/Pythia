from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import pandas as pd
import pytest

from resolver.ingestion.idmc import cli as idmc_cli
from resolver.ingestion.idmc.client import HttpRequestError, IdmcConfig, fetch


def _stub_config(config_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        api=SimpleNamespace(
            countries=[],
            series=["flow", "stock"],
            date_window=SimpleNamespace(start=None, end=None),
            base_url="https://example.test",
            endpoints={"idus_json": "/data/idus_view_flat"},
            token_env="IDMC_API_TOKEN",
        ),
        cache=SimpleNamespace(force_cache_only=False, dir=".cache"),
        field_aliases=SimpleNamespace(
            value_flow=["value"],
            value_stock=["value"],
            date=["date"],
            iso3=["iso3"],
        ),
        _config_details=None,
        _config_source="ingestion",
        _config_path=config_path,
        _config_warnings=(),
    )


def _empty_fetch_diagnostics() -> Dict[str, Any]:
    return {
        "mode": "live",
        "network_mode": "live",
        "http": {
            "requests": 0,
            "retries": 0,
            "status_last": None,
            "latency_ms": {"p50": 0, "p95": 0, "max": 0},
            "retry_after_events": 0,
            "wire_bytes": 0,
            "body_bytes": 0,
            "status_counts": {"2xx": 0, "4xx": 0, "5xx": 0},
        },
        "cache": {"hits": 0, "misses": 0, "dir": ".cache"},
        "filters": {
            "window_start": None,
            "window_end": None,
            "countries": [],
        },
        "http_status_counts": {"2xx": 0, "4xx": 0, "5xx": 0},
        "http_extended": {
            "status_counts": {
                "2xx": 0,
                "4xx": 0,
                "5xx": 0,
                "other": 0,
                "timeout": 0,
            },
            "timeouts": 0,
            "requests_ok_2xx": 0,
            "requests_4xx": 0,
            "requests_5xx": 0,
            "requests_other": 0,
            "other_exceptions": 0,
            "exceptions_by_type": {},
            "requests_planned": 0,
            "requests_executed": 0,
        },
        "performance": {
            "requests": 0,
            "wire_bytes": 0,
            "body_bytes": 0,
            "duration_s": 0,
            "throughput_kibps": 0,
            "records_per_sec": 0,
        },
        "rate_limit": {
            "req_per_sec": 0,
            "max_concurrency": 1,
            "retries": 0,
            "retry_after_events": 0,
            "retry_after_wait_s": 0,
            "rate_limit_wait_s": 0,
            "planned_wait_s": 0,
        },
        "chunks": {"enabled": False, "count": 0, "by_month": []},
        "http_attempt_summary": {
            "planned": 0,
            "ok_2xx": 0,
            "status_4xx": 0,
            "status_5xx": 0,
            "status_other": 0,
            "timeouts": 0,
            "other_exceptions": 0,
        },
        "requests_planned": 0,
        "requests_executed": 0,
        "fallback_used": False,
        "http_timeouts": {"connect_s": 5, "read_s": 5},
        "attempts": [],
    }


def _normalise_empty(*_args, **_kwargs):
    frame = pd.DataFrame(
        columns=[
            "iso3",
            "as_of_date",
            "metric",
            "value",
            "series_semantics",
            "source",
        ]
    )
    return frame, {}


def _maybe_map_passthrough(frame, *_args, **_kwargs):
    return frame, pd.DataFrame()


def _stub_provenance(**_kwargs):
    return {"ok": True}


def _noop_write_connectors_line(_payload):
    return None


def _noop_write_why_zero(_payload):
    return None


def test_idmc_url_has_no_window_params(monkeypatch, tmp_path):
    cfg = IdmcConfig()
    cfg.api.countries = ["AFG"]
    cfg.api.series = ["flow"]
    cfg.cache.dir = (tmp_path / "cache").as_posix()

    monkeypatch.setattr(
        "resolver.ingestion.idmc.client._probe_idu_schema",
        lambda *args, **kwargs: (
            "displacement_date",
            ["iso3", "figure", "displacement_date"],
            {"status": 200, "columns": ["iso3", "figure", "displacement_date"]},
        ),
    )

    monkeypatch.setattr(
        "resolver.ingestion.idmc.client.probe_reachability",
        lambda options: {
            "base_url": options.base_url,
            "dns": {"ok": True, "elapsed_ms": 1, "records": []},
            "tcp": {"ok": True, "elapsed_ms": 1},
            "tls": {"ok": True, "elapsed_ms": 1},
            "http_head": {"ok": True, "status": 200, "elapsed_ms": 1},
        },
    )

    captured_urls: list[str] = []

    def fake_http_get(url: str, **_kwargs: Any) -> Tuple[int, Dict[str, str], bytes, Dict[str, Any]]:
        captured_urls.append(url)
        payload = json.dumps(
            [
                {
                    "iso3": "AFG",
                    "figure": 1,
                    "displacement_date": "2024-01-15",
                }
            ]
        ).encode("utf-8")
        diagnostics = {
            "attempts": 1,
            "retries": 0,
            "duration_s": 0.01,
            "backoff_s": 0.0,
            "wire_bytes": len(payload),
            "body_bytes": len(payload),
            "attempt_durations_s": [0.01],
        }
        return 200, {}, payload, diagnostics

    monkeypatch.setattr("resolver.ingestion.idmc.client.http_get", fake_http_get)

    data, _ = fetch(
        cfg,
        network_mode="live",
        window_start=date(2024, 1, 1),
        window_end=date(2024, 1, 31),
        chunk_by_month=True,
        only_countries=["AFG"],
    )

    assert not data["monthly_flow"].empty
    assert captured_urls
    for url in captured_urls:
        assert "chunk=" not in url
        assert "window_start" not in url
        assert "window_end" not in url


def test_idmc_fallback_uses_alternate_base(monkeypatch, tmp_path):
    cfg = IdmcConfig()
    cfg.api.base_url = "https://primary.example"
    cfg.api.alternate_base_url = "https://alternate.example"
    cfg.api.countries = ["AFG"]
    cfg.api.series = ["flow"]
    cfg.cache.dir = (tmp_path / "cache").as_posix()

    monkeypatch.setenv("IDMC_CACHE_DIR", cfg.cache.dir)

    monkeypatch.setattr(
        "resolver.ingestion.idmc.client._probe_idu_schema",
        lambda *args, **kwargs: (
            "displacement_date",
            ["iso3", "figure", "displacement_date"],
            {"status": 200},
        ),
    )

    def fake_probe(options):
        base = options.base_url
        if "primary" in base:
            return {
                "base_url": base,
                "dns": {"ok": False, "error": "nxdomain", "elapsed_ms": 5},
                "tcp": {"ok": False, "skipped": True, "elapsed_ms": 0},
                "tls": {"ok": False, "skipped": True, "elapsed_ms": 0},
                "http_head": {"ok": False, "skipped": True, "elapsed_ms": 0},
            }
        return {
            "base_url": base,
            "dns": {"ok": True, "elapsed_ms": 1, "records": []},
            "tcp": {"ok": True, "elapsed_ms": 1},
            "tls": {"ok": True, "elapsed_ms": 1},
            "http_head": {"ok": True, "status": 200, "elapsed_ms": 1},
        }

    monkeypatch.setattr(
        "resolver.ingestion.idmc.client.probe_reachability", fake_probe
    )

    captured_urls: list[str] = []

    def fake_http_get(url: str, **_kwargs: Any):
        captured_urls.append(url)
        payload = json.dumps(
            [{
                "iso3": "AFG",
                "figure": 1,
                "displacement_date": "2024-01-01",
            }]
        ).encode("utf-8")
        diagnostics = {
            "attempts": 1,
            "retries": 0,
            "duration_s": 0.01,
            "backoff_s": 0.0,
            "wire_bytes": len(payload),
            "body_bytes": len(payload),
            "attempt_durations_s": [0.01],
        }
        return 200, {}, payload, diagnostics

    monkeypatch.setattr("resolver.ingestion.idmc.client.http_get", fake_http_get)

    data, diagnostics = fetch(
        cfg,
        network_mode="live",
        window_start=date(2024, 1, 1),
        window_end=date(2024, 1, 31),
    )

    assert captured_urls
    assert all("alternate.example" in url for url in captured_urls)
    assert not data["monthly_flow"].empty

    outcomes = diagnostics.get("endpoint_outcomes") or {}
    assert outcomes.get("primary", {}).get("status") == "fail"
    assert outcomes.get("primary", {}).get("stage") == "dns"
    assert outcomes.get("alternate", {}).get("status") in {"ok", "warn"}
    assert diagnostics.get("network_mode") == "live"
    assert outcomes.get("helix", {}).get("status") == "unused"


def test_idmc_dns_failure_routes_to_helix(monkeypatch, tmp_path):
    cfg = IdmcConfig()
    cfg.api.base_url = "https://primary.example"
    cfg.api.alternate_base_url = "https://alternate.example"
    cfg.api.countries = ["AFG"]
    cfg.api.series = ["flow"]
    cfg.cache.dir = (tmp_path / "cache").as_posix()

    monkeypatch.setenv("IDMC_CACHE_DIR", cfg.cache.dir)
    monkeypatch.setenv("IDMC_HELIX_CLIENT_ID", "client-123")

    monkeypatch.setattr(
        "resolver.ingestion.idmc.client._probe_idu_schema",
        lambda *args, **kwargs: (
            "displacement_date",
            ["iso3", "figure", "displacement_date"],
            {"status": 200},
        ),
    )

    def failing_probe(options):
        base = options.base_url
        return {
            "base_url": base,
            "dns": {"ok": False, "error": "nxdomain", "elapsed_ms": 5},
            "tcp": {"ok": False, "skipped": True, "elapsed_ms": 0},
            "tls": {"ok": False, "skipped": True, "elapsed_ms": 0},
            "http_head": {"ok": False, "skipped": True, "elapsed_ms": 0},
        }

    monkeypatch.setattr(
        "resolver.ingestion.idmc.client.probe_reachability", failing_probe
    )

    def fake_helix_last180(*_args, **_kwargs):
        frame = pd.DataFrame(
            [{
                "iso3": "AFG",
                "displacement_date": "2024-01-01",
                "figure": 5,
            }]
        )
        return frame, {"status": 200, "raw_rows": frame.shape[0]}

    monkeypatch.setattr(
        "resolver.ingestion.idmc.client._fetch_helix_last180",
        fake_helix_last180,
    )

    data, diagnostics = fetch(
        cfg,
        network_mode="live",
        window_start=date(2024, 1, 1),
        window_end=date(2024, 1, 31),
    )

    outcomes = diagnostics.get("endpoint_outcomes") or {}
    assert diagnostics.get("network_mode") == "helix"
    assert diagnostics.get("requested_network_mode") == "live"
    assert diagnostics.get("helix_failover") is True
    assert outcomes.get("primary", {}).get("status") == "fail"
    assert outcomes.get("helix", {}).get("status") == "used"
    assert outcomes.get("helix", {}).get("status_code") == 200
    assert outcomes.get("helix", {}).get("rows") == 1
    assert not data["monthly_flow"].empty


def test_idmc_fallback_gated(monkeypatch, tmp_path):
    cfg = IdmcConfig()
    cfg.api.countries = ["AFG"]
    cfg.api.series = ["flow"]
    cfg.cache.dir = (tmp_path / "cache").as_posix()

    monkeypatch.setattr(
        "resolver.ingestion.idmc.client._probe_idu_schema",
        lambda *args, **kwargs: (
            "displacement_date",
            ["iso3", "figure", "displacement_date"],
            {"status": 200, "columns": ["iso3", "figure", "displacement_date"]},
        ),
    )

    monkeypatch.setattr(
        "resolver.ingestion.idmc.client.probe_reachability",
        lambda options: {
            "base_url": options.base_url,
            "dns": {"ok": True, "elapsed_ms": 1, "records": []},
            "tcp": {"ok": True, "elapsed_ms": 1},
            "tls": {"ok": True, "elapsed_ms": 1},
            "http_head": {"ok": True, "status": 200, "elapsed_ms": 1},
        },
    )

    def failing_http_get(_url: str, **_kwargs: Any):
        diagnostics = {
            "attempts": 1,
            "retries": 0,
            "duration_s": 0.01,
            "backoff_s": 0.0,
            "wire_bytes": 0,
            "body_bytes": 0,
            "status": None,
            "timeout": True,
        }
        raise HttpRequestError("boom", diagnostics, kind="connect_timeout")

    monkeypatch.setattr("resolver.ingestion.idmc.client.http_get", failing_http_get)

    data_no_fallback, diagnostics_no_fallback = fetch(
        cfg,
        network_mode="live",
        window_start=date(2024, 1, 1),
        window_end=date(2024, 1, 31),
        allow_hdx_fallback=False,
    )
    assert diagnostics_no_fallback.get("fallback_used") is False
    assert not diagnostics_no_fallback.get("fallback")
    assert data_no_fallback["monthly_flow"].empty

    fallback_calls: list[pd.DataFrame] = []

    fallback_frame = pd.DataFrame(
        [
            {
                "CountryISO3": "AFG",
                "displacement_date": "2024-01-15",
                "figure": 42,
            }
        ]
    )

    def fake_hdx_fetch():
        fallback_calls.append(fallback_frame)
        return fallback_frame, {"source": "hdx"}

    monkeypatch.setattr(
        "resolver.ingestion.idmc.client._hdx_fetch_latest_csv",
        fake_hdx_fetch,
    )

    data, diagnostics = fetch(
        cfg,
        network_mode="live",
        window_start=date(2024, 1, 1),
        window_end=date(2024, 1, 31),
        allow_hdx_fallback=True,
    )

    assert fallback_calls
    assert not data["monthly_flow"].empty
    assert diagnostics.get("fallback_used") is True


def test_write_empty_header_when_requested(monkeypatch, tmp_path, caplog):
    config_path = tmp_path / "resolver" / "config" / "idmc.yml"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IDMC_API_TOKEN", "token")
    monkeypatch.setenv("EMPTY_POLICY", "allow")
    monkeypatch.setenv("RESOLVER_START_ISO", "2024-01-01")
    monkeypatch.setenv("RESOLVER_END_ISO", "2024-01-31")

    monkeypatch.setattr(idmc_cli, "load", lambda: _stub_config(config_path))
    monkeypatch.setattr(idmc_cli, "normalize_all", _normalise_empty)
    monkeypatch.setattr(idmc_cli, "maybe_map_hazards", _maybe_map_passthrough)
    monkeypatch.setattr(idmc_cli, "write_connectors_line", _noop_write_connectors_line)
    monkeypatch.setattr(idmc_cli, "build_provenance", _stub_provenance)
    monkeypatch.setattr(idmc_cli, "write_why_zero", _noop_write_why_zero)
    monkeypatch.setattr(idmc_cli, "probe_reachability", lambda *_args, **_kwargs: {"ok": True})

    def _stub_fetch(*_args, **_kwargs):
        return {"monthly_flow": pd.DataFrame()}, _empty_fetch_diagnostics()

    monkeypatch.setattr(idmc_cli, "fetch", _stub_fetch)

    caplog.set_level(logging.INFO)
    exit_code = idmc_cli.main(["--write-outputs"])

    assert exit_code == 0
    flow_path = tmp_path / "resolver" / "staging" / "idmc" / "flow.csv"
    assert flow_path.exists()
    lines = [line for line in flow_path.read_text(encoding="utf-8").splitlines() if line]
    assert lines == ["iso3,as_of_date,metric,value,series_semantics,source"]


def test_diagnostics_three_keys(monkeypatch):
    diagnostics = _empty_fetch_diagnostics()
    http_counts = diagnostics["http_status_counts"]
    assert set(http_counts.keys()) == {"2xx", "4xx", "5xx"}
