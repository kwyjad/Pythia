import os
from datetime import date
from pathlib import Path
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


def _normalized_frame():
    return pd.DataFrame(
        [
            {
                "iso3": "AAA",
                "as_of_date": "2024-01-31",
                "metric": "new_displacements",
                "value": 100,
                "series_semantics": "new",
                "source": "idmc_idu",
            }
        ]
    )


def _diagnostics(window_start: str, window_end: str, *, requests: int = 1):
    return {
        "mode": "online",
        "network_mode": "live",
        "http": {
            "requests": requests,
            "retries": 0,
            "status_last": 200,
            "retry_after_events": 0,
            "wire_bytes": 10,
            "body_bytes": 5,
            "latency_ms": {"p50": 10, "p95": 10, "max": 10},
            "attempt_durations_ms": [10],
            "planned_sleep_s": [],
            "rate_limit_wait_s": [],
        },
        "cache": {"hits": 0, "misses": 0},
        "filters": {
            "window_start": window_start,
            "window_end": window_end,
            "countries": [],
            "rows_before": requests,
            "rows_after": requests,
        },
        "performance": {
            "requests": requests,
            "wire_bytes": 10,
            "body_bytes": 5,
            "duration_s": 0.1,
            "throughput_kibps": 0,
            "records_per_sec": 0,
        },
        "rate_limit": {
            "req_per_sec": 1,
            "max_concurrency": 1,
            "retries": 0,
            "retry_after_wait_s": 0,
            "rate_limit_wait_s": 0,
            "planned_wait_s": 0,
        },
        "chunks": {"enabled": False, "count": 1, "entries": []},
        "http_status_counts": {"2xx": requests, "4xx": 0, "5xx": 0},
        "requests_planned": requests,
        "window": {
            "start": window_start,
            "end": window_end,
            "window_days": None,
        },
    }


def test_idmc_cli_accepts_debug_and_window_flags(monkeypatch, tmp_path, caplog):
    config_path = tmp_path / "resolver" / "config" / "idmc.yml"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IDMC_API_TOKEN", "token")
    monkeypatch.setenv("IDMC_OUT_DIR", str(tmp_path / "out"))

    captured = {}

    def _fetch(cfg, **kwargs):
        captured["window_start"] = kwargs.get("window_start")
        captured["window_end"] = kwargs.get("window_end")
        frame = pd.DataFrame(
            [{"displacement_date": "2024-01-15", "figure": 100, "iso3": "AAA"}]
        )
        return {"monthly_flow": frame}, _diagnostics("2024-01-01", "2024-02-29")

    def _normalize(*_args, **_kwargs):
        return _normalized_frame(), {}

    monkeypatch.setattr(idmc_cli, "load", lambda: _stub_config(config_path))
    monkeypatch.setattr(idmc_cli, "fetch", _fetch)
    monkeypatch.setattr(idmc_cli, "normalize_all", _normalize)
    monkeypatch.setattr(
        idmc_cli, "maybe_map_hazards", lambda frame, *_args, **_kwargs: (frame, [])
    )
    monkeypatch.setattr(idmc_cli, "write_connectors_line", lambda payload: payload)
    monkeypatch.setattr(idmc_cli, "build_provenance", lambda **_: {"ok": True})
    monkeypatch.setattr(idmc_cli, "write_json", lambda *_args, **_kwargs: None)
    caplog.set_level("DEBUG")

    exit_code = idmc_cli.main(
        [
            "--network-mode",
            "live",
            "--start",
            "2024-01-01",
            "--end",
            "2024-02-29",
            "--debug",
        ]
    )

    assert exit_code == 0
    assert captured["window_start"] == date(2024, 1, 1)
    assert captured["window_end"] == date(2024, 2, 29)
    assert os.environ.get("RESOLVER_DEBUG") == "1"
    staging_flow = Path("resolver/staging/idmc/flow.csv")
    assert staging_flow.exists()


def test_idmc_window_env_fallback(monkeypatch, tmp_path):
    config_path = tmp_path / "resolver" / "config" / "idmc.yml"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IDMC_API_TOKEN", "token")
    monkeypatch.setenv("RESOLVER_START_ISO", "2024-04-01")
    monkeypatch.setenv("RESOLVER_END_ISO", "2024-04-30")

    captured = {}

    def _fetch(cfg, **kwargs):
        captured["window_start"] = kwargs.get("window_start")
        captured["window_end"] = kwargs.get("window_end")
        frame = pd.DataFrame([{"displacement_date": "2024-04-05", "figure": 10}])
        return {"monthly_flow": frame}, _diagnostics("2024-04-01", "2024-04-30")

    def _normalize(*_args, **_kwargs):
        return pd.DataFrame(columns=_normalized_frame().columns), {}

    why_zero_payload = {}

    def _write_why_zero(payload):
        why_zero_payload.update(payload)
        return "why_zero.json"

    monkeypatch.setattr(idmc_cli, "load", lambda: _stub_config(config_path))
    monkeypatch.setattr(idmc_cli, "fetch", _fetch)
    monkeypatch.setattr(idmc_cli, "normalize_all", _normalize)
    monkeypatch.setattr(
        idmc_cli, "maybe_map_hazards", lambda frame, *_args, **_kwargs: (frame, [])
    )
    monkeypatch.setattr(idmc_cli, "write_connectors_line", lambda payload: payload)
    monkeypatch.setattr(idmc_cli, "build_provenance", lambda **_: {"ok": True})
    monkeypatch.setattr(idmc_cli, "write_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(idmc_cli, "write_why_zero", _write_why_zero)

    exit_code = idmc_cli.main(["--network-mode", "live"])

    assert exit_code == 0
    assert captured["window_start"] == date(2024, 4, 1)
    assert captured["window_end"] == date(2024, 4, 30)
    assert why_zero_payload["window"]["start"] == "2024-04-01"
    assert why_zero_payload["window"]["end"] == "2024-04-30"


def test_idmc_live_mode_emits_http_counters(monkeypatch, tmp_path):
    config_path = tmp_path / "resolver" / "config" / "idmc.yml"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IDMC_API_TOKEN", "token")
    monkeypatch.setenv("IDMC_OUT_DIR", str(tmp_path / "exports"))

    captured_payload = {}

    def _fetch(cfg, **kwargs):
        frame = pd.DataFrame(
            [{"displacement_date": "2024-05-05", "figure": 42, "iso3": "AAA"}]
        )
        return {"monthly_flow": frame}, _diagnostics("2024-05-01", "2024-05-31")

    def _normalize(*_args, **_kwargs):
        return _normalized_frame(), {}

    monkeypatch.setattr(idmc_cli, "load", lambda: _stub_config(config_path))
    monkeypatch.setattr(idmc_cli, "fetch", _fetch)
    monkeypatch.setattr(idmc_cli, "normalize_all", _normalize)
    monkeypatch.setattr(
        idmc_cli, "maybe_map_hazards", lambda frame, *_args, **_kwargs: (frame, [])
    )
    monkeypatch.setattr(idmc_cli, "write_connectors_line", lambda payload: captured_payload.setdefault("last", payload))
    monkeypatch.setattr(idmc_cli, "build_provenance", lambda **_: {"ok": True})
    monkeypatch.setattr(idmc_cli, "write_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(idmc_cli, "write_facts_parquet", lambda *_args, **_kwargs: "")

    exit_code = idmc_cli.main(
        [
            "--network-mode",
            "live",
            "--start",
            "2024-05-01",
            "--end",
            "2024-05-31",
            "--enable-export",
            "--write-outputs",
            "--series",
            "flow,stock",
        ]
    )

    assert exit_code == 0
    payload = captured_payload.get("last")
    assert payload is not None
    assert payload["http_status_counts"]["2xx"] == 1
    assert payload["requests_executed"] == 1
    flow_path = Path("resolver/staging/idmc/flow.csv")
    assert flow_path.exists()
    frame = pd.read_csv(flow_path)
    assert not frame.empty
