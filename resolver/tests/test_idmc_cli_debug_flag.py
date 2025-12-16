# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import logging
import os
from types import SimpleNamespace

import pandas as pd

from resolver.ingestion.idmc import cli as idmc_cli


def _stub_config(config_path):
    return SimpleNamespace(
        api=SimpleNamespace(
            countries=[],
            series=["flow"],
            date_window=SimpleNamespace(start="2024-01-01", end="2024-01-31"),
            base_url="https://example.test",
            endpoints={"idus_json": "/data/idus_view_flat"},
            token_env="IDMC_API_TOKEN",
        ),
        cache=SimpleNamespace(force_cache_only=False),
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


def _stub_fetch(*_args, **_kwargs):
    diagnostics = {
        "mode": "fixture",
        "network_mode": "fixture",
        "http": {
            "requests": 0,
            "retries": 0,
            "status_last": None,
            "retry_after_events": 0,
            "cache": {"hits": 0, "misses": 0},
            "latency_ms": {"p50": 0, "p95": 0, "max": 0},
        },
        "cache": {"hits": 0, "misses": 0},
        "filters": {"window_start": None, "window_end": None, "countries": []},
        "http_status_counts": None,
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
            "retry_after_wait_s": 0,
            "rate_limit_wait_s": 0,
            "planned_wait_s": 0,
        },
        "chunks": {"enabled": False, "count": 0, "by_month": []},
    }
    return {}, diagnostics


def _stub_normalize(*_args, **_kwargs):
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


def _stub_maybe_map(frame, *_args, **_kwargs):
    return frame, []


def _noop_write_connectors_line(_payload):
    return None


def _stub_provenance(**_kwargs):
    return {"ok": True}


def test_idmc_cli_debug_flag(monkeypatch, tmp_path, caplog):
    config_path = tmp_path / "resolver" / "config" / "idmc.yml"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IDMC_API_TOKEN", "token")
    monkeypatch.setattr(idmc_cli, "load", lambda: _stub_config(config_path))
    monkeypatch.setattr(idmc_cli, "fetch", _stub_fetch)
    monkeypatch.setattr(idmc_cli, "normalize_all", _stub_normalize)
    monkeypatch.setattr(idmc_cli, "maybe_map_hazards", _stub_maybe_map)
    monkeypatch.setattr(idmc_cli, "write_connectors_line", _noop_write_connectors_line)
    monkeypatch.setattr(idmc_cli, "build_provenance", _stub_provenance)
    monkeypatch.setattr(
        idmc_cli,
        "resolve_countries",
        lambda _values=None, _env_value=None, **_kwargs: ["AAA", "BBB"],
    )

    caplog.set_level(logging.DEBUG)
    exit_code = idmc_cli.main(["--network-mode", "fixture", "--debug"])

    assert exit_code == 0
    assert os.environ.get("RESOLVER_DEBUG") == "1"
    assert any("Debug logging enabled" in record.message for record in caplog.records)
