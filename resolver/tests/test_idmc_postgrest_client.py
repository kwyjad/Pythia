import json
from datetime import date
from types import SimpleNamespace

import pandas as pd

from resolver.ingestion.idmc import cli as idmc_cli
from resolver.ingestion.idmc.client import (
    IdmcConfig,
    HttpRequestError,
    _batch_iso3,
    _postgrest_filters,
    _probe_idu_schema,
    fetch,
)


def test_idmc_idu_schema_probe_picks_displacement_date(monkeypatch):
    cfg = IdmcConfig()
    payload = json.dumps(
        [
            {
                "iso3": "AAA",
                "figure": 1,
                "displacement_date": "2024-01-01",
                "displacement_type": "conflict",
            }
        ]
    ).encode("utf-8")

    def fake_http_get(url, **_kwargs):  # noqa: D401 - local stub
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

    date_column, select_columns, diagnostics = _probe_idu_schema(cfg)

    assert date_column == "displacement_date"
    assert "displacement_date" in select_columns
    assert diagnostics["status"] == 200


def test_idmc_idu_builds_filters_for_single_date_col():
    params = _postgrest_filters(
        chunk_start=date(2024, 1, 1),
        chunk_end=date(2024, 1, 31),
        window_start=None,
        window_end=None,
        iso_batch=["AAA"],
        offset=0,
        limit=10000,
        date_column="displacement_date",
        select_columns=["iso3", "displacement_date"],
    )

    assert ("displacement_date", "gte.2024-01-01") in params
    assert ("displacement_date", "lte.2024-01-31") in params
    assert all(key != "displacement_start_date" for key, _ in params)


def test_idmc_batches_iso3s_max_20():
    codes = [f"C{i:03d}" for i in range(198)]
    batches = _batch_iso3(codes)
    assert len(batches) == 10
    assert all(len(batch) <= 20 for batch in batches)


def test_idmc_per_chunk_http_error_does_not_abort(monkeypatch):
    cfg = IdmcConfig()
    cfg.api.countries = ["AFG"]

    def fake_probe(*_args, **_kwargs):
        return "displacement_date", ["iso3", "figure", "displacement_date"], {
            "status": 200,
            "columns": ["iso3", "figure", "displacement_date"],
        }

    call_state = {"calls": 0}

    def fake_http_get(url, **_kwargs):
        call_state["calls"] += 1
        if "chunk=2023-01" in url:
            diagnostics = {
                "status": 500,
                "attempts": 1,
                "retries": 0,
                "duration_s": 0.01,
                "backoff_s": 0.0,
                "wire_bytes": 0,
                "body_bytes": 0,
                "attempt_durations_s": [0.01],
                "exceptions": [{"message": "boom"}],
            }
            raise HttpRequestError("boom", diagnostics)
        payload = json.dumps(
            [
                {
                    "iso3": "AFG",
                    "figure": 1,
                    "displacement_date": "2023-02-05",
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

    monkeypatch.setattr("resolver.ingestion.idmc.client._probe_idu_schema", fake_probe)
    monkeypatch.setattr("resolver.ingestion.idmc.client.http_get", fake_http_get)

    data, diagnostics = fetch(
        cfg,
        network_mode="live",
        window_start=date(2023, 1, 1),
        window_end=date(2023, 2, 28),
        chunk_by_month=True,
        only_countries=["AFG"],
    )

    frame = data["monthly_flow"]
    assert len(frame) == 1
    assert diagnostics["chunk_errors"] == 1
    counts = diagnostics["http_status_counts"]
    assert set(counts.keys()) == {"2xx", "4xx", "5xx"}
    assert diagnostics.get("http_status_counts_extended", {}).get("other", 0) >= 0
    assert call_state["calls"] >= 2


def test_idmc_empty_but_not_fail_policy_returns_ok(monkeypatch, tmp_path):
    cfg_path = tmp_path / "resolver" / "config" / "idmc.yml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    def _stub_config(_path=None):
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
            _config_path=cfg_path,
            _config_warnings=(),
        )

    def _empty_normalize(*_args, **_kwargs):
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

    def _maybe_map(frame, *_args, **_kwargs):
        return frame, pd.DataFrame()

    captured = {}

    def _capture(payload):
        captured.update(payload)
        return payload

    diagnostics = {
        "mode": "live",
        "network_mode": "live",
        "http": {
            "requests": 1,
            "retries": 0,
            "status_last": 500,
            "latency_ms": {"p50": 0, "p95": 0, "max": 0},
            "status_counts": {"2xx": 0, "4xx": 0, "5xx": 0},
            "status_counts_extended": {
                "2xx": 0,
                "4xx": 0,
                "5xx": 0,
                "other": 1,
            },
        },
        "cache": {"dir": ".cache", "hits": 0, "misses": 0},
        "filters": {"window_start": None, "window_end": None, "countries": []},
        "chunks": {"enabled": True, "count": 1, "by_month": []},
        "chunk_errors": 1,
        "fetch_errors": [
            {"chunk": "2023-01", "exception": "HttpRequestError", "status": 500}
        ],
        "date_column": "displacement_date",
        "select_columns": ["iso3", "figure", "displacement_date"],
        "http_status_counts": {"2xx": 0, "4xx": 0, "5xx": 0},
        "http_status_counts_extended": {
            "2xx": 0,
            "4xx": 0,
            "5xx": 0,
            "other": 1,
        },
        "performance": {
            "requests": 1,
            "wire_bytes": 0,
            "body_bytes": 0,
            "duration_s": 0.0,
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
    }

    def _stub_fetch(*_args, **_kwargs):
        empty_frame = pd.DataFrame(
            columns=[
                "iso3",
                "figure",
                "displacement_date",
                "displacement_type",
            ]
        )
        return {"monthly_flow": empty_frame}, diagnostics

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IDMC_API_TOKEN", "token")
    monkeypatch.setenv("EMPTY_POLICY", "warn")
    monkeypatch.setenv("RESOLVER_START_ISO", "2023-01-01")
    monkeypatch.setenv("RESOLVER_END_ISO", "2023-01-31")

    monkeypatch.setattr(idmc_cli, "load", lambda: _stub_config(cfg_path))
    monkeypatch.setattr(idmc_cli, "fetch", _stub_fetch)
    monkeypatch.setattr(idmc_cli, "normalize_all", _empty_normalize)
    monkeypatch.setattr(idmc_cli, "maybe_map_hazards", _maybe_map)
    monkeypatch.setattr(idmc_cli, "write_connectors_line", _capture)
    monkeypatch.setattr(idmc_cli, "build_provenance", lambda **_: {"ok": True})
    monkeypatch.setattr(idmc_cli, "write_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(idmc_cli, "write_why_zero", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(idmc_cli, "probe_reachability", lambda *_args, **_kwargs: {"ok": True})

    exit_code = idmc_cli.main(["--network-mode", "live"])

    assert exit_code == 0
    assert captured["chunk_errors"] == 1
    assert captured["date_column"] == "displacement_date"
