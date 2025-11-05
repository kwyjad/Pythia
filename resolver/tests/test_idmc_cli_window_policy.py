from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from resolver.ingestion.idmc import cli as idmc_cli


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


def _empty_fetch_diagnostics() -> dict:
    return {
        "mode": "fixture",
        "network_mode": "fixture",
        "http": {
            "requests": 0,
            "retries": 0,
            "status_last": None,
            "latency_ms": {"p50": 0, "p95": 0, "max": 0},
            "retry_after_events": 0,
            "wire_bytes": 0,
            "body_bytes": 0,
        },
        "cache": {"hits": 0, "misses": 0, "dir": ".cache"},
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
            "retry_after_events": 0,
            "retry_after_wait_s": 0,
            "rate_limit_wait_s": 0,
            "planned_wait_s": 0,
        },
        "chunks": {"enabled": False, "count": 0, "by_month": []},
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


def test_idmc_cli_window_optional_allows_zero_rows(monkeypatch, tmp_path, caplog):
    config_path = tmp_path / "resolver" / "config" / "idmc.yml"
    monkeypatch.chdir(tmp_path)
    for name in (
        "RESOLVER_START_ISO",
        "RESOLVER_END_ISO",
        "BACKFILL_START_ISO",
        "BACKFILL_END_ISO",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("IDMC_API_TOKEN", "token")
    monkeypatch.setenv("EMPTY_POLICY", "allow")

    monkeypatch.setattr(idmc_cli, "load", lambda: _stub_config(config_path))
    monkeypatch.setattr(idmc_cli, "normalize_all", _normalise_empty)
    monkeypatch.setattr(idmc_cli, "maybe_map_hazards", _maybe_map_passthrough)
    monkeypatch.setattr(idmc_cli, "write_connectors_line", _noop_write_connectors_line)
    monkeypatch.setattr(idmc_cli, "build_provenance", _stub_provenance)
    monkeypatch.setattr(idmc_cli, "write_why_zero", _noop_write_why_zero)
    monkeypatch.setattr(idmc_cli, "probe_reachability", lambda *_args, **_kwargs: {"ok": True})

    fetch_calls: list[dict] = []

    def _stub_fetch(*_args, **kwargs):
        fetch_calls.append(kwargs)
        return {}, _empty_fetch_diagnostics()

    monkeypatch.setattr(idmc_cli, "fetch", _stub_fetch)

    caplog.set_level(logging.WARNING)
    exit_code = idmc_cli.main([])

    assert exit_code == 0
    assert fetch_calls == []
    assert any("No date window provided" in record.message for record in caplog.records)


def test_idmc_cli_uses_env_window_when_flags_missing(monkeypatch, tmp_path):
    config_path = tmp_path / "resolver" / "config" / "idmc.yml"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RESOLVER_START_ISO", "2024-01-01")
    monkeypatch.setenv("RESOLVER_END_ISO", "2024-01-31")
    monkeypatch.setenv("IDMC_API_TOKEN", "token")
    monkeypatch.setenv("EMPTY_POLICY", "allow")

    monkeypatch.setattr(idmc_cli, "load", lambda: _stub_config(config_path))
    monkeypatch.setattr(idmc_cli, "normalize_all", _normalise_empty)
    monkeypatch.setattr(idmc_cli, "maybe_map_hazards", _maybe_map_passthrough)
    monkeypatch.setattr(idmc_cli, "write_connectors_line", _noop_write_connectors_line)
    monkeypatch.setattr(idmc_cli, "build_provenance", _stub_provenance)
    monkeypatch.setattr(idmc_cli, "write_why_zero", _noop_write_why_zero)
    monkeypatch.setattr(idmc_cli, "probe_reachability", lambda *_args, **_kwargs: {"ok": True})

    received: dict[str, object] = {}

    def _recording_fetch(*_args, **kwargs):
        received.update(kwargs)
        return {}, _empty_fetch_diagnostics()

    monkeypatch.setattr(idmc_cli, "fetch", _recording_fetch)

    exit_code = idmc_cli.main([])

    assert exit_code == 0
    assert received["window_start"].isoformat() == "2024-01-01"
    assert received["window_end"].isoformat() == "2024-01-31"


def test_idmc_export_enabled_writes_csvs(monkeypatch, tmp_path):
    config_path = tmp_path / "resolver" / "config" / "idmc.yml"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RESOLVER_START_ISO", "2024-02-01")
    monkeypatch.setenv("RESOLVER_END_ISO", "2024-02-29")
    monkeypatch.setenv("IDMC_API_TOKEN", "token")
    monkeypatch.setenv("EMPTY_POLICY", "allow")
    monkeypatch.setenv("RESOLVER_EXPORT_ENABLE_FLOW", "1")

    monkeypatch.setattr(idmc_cli, "load", lambda: _stub_config(config_path))
    monkeypatch.setattr(idmc_cli, "maybe_map_hazards", _maybe_map_passthrough)
    monkeypatch.setattr(idmc_cli, "write_connectors_line", _noop_write_connectors_line)
    monkeypatch.setattr(idmc_cli, "build_provenance", _stub_provenance)
    monkeypatch.setattr(idmc_cli, "write_why_zero", _noop_write_why_zero)
    monkeypatch.setattr(idmc_cli, "probe_reachability", lambda *_args, **_kwargs: {"ok": True})

    diagnostics = _empty_fetch_diagnostics()
    diagnostics["network_mode"] = "live"

    monkeypatch.setattr(idmc_cli, "fetch", lambda *_args, **_kwargs: ({}, diagnostics))

    normalized = pd.DataFrame(
        [
            {
                "iso3": "AAA",
                "as_of_date": "2024-02-29",
                "metric": "new_displacements",
                "value": 5,
                "series_semantics": "new",
                "source": "idmc_idu",
            },
            {
                "iso3": "BBB",
                "as_of_date": "2024-02-29",
                "metric": "idp_displacement_stock_idmc",
                "value": 7,
                "series_semantics": "stock",
                "source": "IDMC",
            },
        ]
    )

    monkeypatch.setattr(
        idmc_cli,
        "normalize_all",
        lambda *_args, **_kwargs: (normalized.copy(), {}),
    )

    exit_code = idmc_cli.main([
        "--enable-export",
        "--write-outputs",
        "--series",
        "flow,stock",
    ])

    assert exit_code == 0

    staging_dir = tmp_path / "resolver" / "staging" / "idmc"
    flow_path = staging_dir / "flow.csv"
    stock_path = staging_dir / "stock.csv"

    assert flow_path.exists()
    assert stock_path.exists()

    flow_rows = pd.read_csv(flow_path)
    stock_rows = pd.read_csv(stock_path)

    assert len(flow_rows) == 1
    assert len(stock_rows) == 1
