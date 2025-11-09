from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from resolver.ingestion.idmc import cli as idmc_cli


def _build_config(config_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        api=SimpleNamespace(
            countries=[],
            series=["flow"],
            date_window=SimpleNamespace(start="2024-01-01", end="2024-03-31"),
            base_url="https://example.invalid",
            endpoints={"idus_json": "/data/idus_view_flat"},
            token_env="IDMC_API_TOKEN",
        ),
        cache=SimpleNamespace(force_cache_only=False),
        field_aliases=SimpleNamespace(
            value_flow=["value"],
            value_stock=["value"],
            date=["as_of_date"],
            iso3=["iso3"],
        ),
        _config_details=None,
        _config_source="ingestion",
        _config_path=config_path,
        _config_warnings=(),
    )


def test_idmc_fallback_metrics(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "resolver" / "config" / "idmc.yml"
    monkeypatch.setattr(idmc_cli, "load", lambda: _build_config(config_path))

    fallback_frame = pd.DataFrame(
        [
            {
                "iso3": "COL",
                "as_of_date": "2024-01-31",
                "metric": idmc_cli.FLOW_METRIC,
                "value": 120,
                "series_semantics": idmc_cli.FLOW_SERIES_SEMANTICS,
                "source": "idmc_gidd",
            },
            {
                "iso3": "ETH",
                "as_of_date": "2024-02-29",
                "metric": idmc_cli.FLOW_METRIC,
                "value": 90,
                "series_semantics": idmc_cli.FLOW_SERIES_SEMANTICS,
                "source": "idmc_gidd",
            },
        ]
    )

    diagnostics = {
        "mode": "fallback",
        "network_mode": "live",
        "fallback_used": True,
        "fallback": {"rows": len(fallback_frame), "type": "helix"},
        "http": {"requests": 0, "retries": 0, "status_last": None, "cache": {"hits": 0, "misses": 0}},
        "cache": {"hits": 0, "misses": 0},
        "filters": {"window_start": "2024-01-01", "window_end": "2024-03-31", "countries": []},
        "performance": {"duration_s": 0, "wire_bytes": 0, "body_bytes": 0, "records_per_sec": 0},
        "rate_limit": {
            "req_per_sec": 0.5,
            "max_concurrency": 1,
            "retries": 0,
            "retry_after_wait_s": 0,
            "rate_limit_wait_s": 0,
            "planned_wait_s": 0,
        },
        "chunks": {"enabled": False, "count": 1},
        "rows_fetched": 0,
        "rows_normalized": 0,
        "rows_written": 0,
    }

    monkeypatch.setattr(idmc_cli._client, "fetch", lambda *_args, **_kwargs: ({"monthly_flow": fallback_frame}, diagnostics))

    def _fake_normalize(*_args, **_kwargs):
        tidy = fallback_frame.copy()
        return tidy, {}

    monkeypatch.setattr(idmc_cli, "normalize_all", _fake_normalize)
    monkeypatch.setattr(idmc_cli, "maybe_map_hazards", lambda frame, *_args, **_kwargs: (frame, []))
    monkeypatch.setattr(idmc_cli, "write_connectors_line", lambda payload: payload)
    monkeypatch.setattr(idmc_cli, "build_provenance", lambda **_kwargs: {"ok": True})
    monkeypatch.setattr(idmc_cli, "write_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(idmc_cli, "write_why_zero", lambda *_args, **_kwargs: None)

    def _fake_parquet(path: str | Path, frame: pd.DataFrame) -> str:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text("parquet placeholder", encoding="utf-8")
        return path_obj.as_posix()

    monkeypatch.setattr(idmc_cli, "write_facts_parquet", _fake_parquet)

    monkeypatch.setenv("RESOLVER_EXPORT_ENABLE_IDMC", "1")
    monkeypatch.setenv("IDMC_WRITE_OUTPUTS", "1")
    monkeypatch.setenv("WRITE_TO_DUCKDB", "1")

    exit_code = idmc_cli.main(["--network-mode", "live", "--strict-empty"])
    assert exit_code == 0

    metrics = idmc_cli._LAST_CLIENT.metrics
    assert metrics.fetched > 0
    assert metrics.normalized > 0
    assert metrics.written > 0
    assert metrics.staged.get("flow.csv", 0) > 0

    flow_path = Path("resolver/staging/idmc/flow.csv")
    parquet_path = Path("artifacts/idmc/idmc_facts_flow.parquet")
    assert flow_path.exists()
    assert parquet_path.exists()
