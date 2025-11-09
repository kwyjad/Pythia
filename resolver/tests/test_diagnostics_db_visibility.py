from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from resolver.ingestion.idmc import cli as idmc_cli


def _config(config_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        api=SimpleNamespace(
            countries=[],
            series=["flow"],
            date_window=SimpleNamespace(start="2024-04-01", end="2024-04-30"),
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


def test_summary_includes_duckdb_section(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "resolver" / "config" / "idmc.yml"
    monkeypatch.setattr(idmc_cli, "load", lambda: _config(config_path))

    frame = pd.DataFrame(
        [
            {
                "iso3": "COL",
                "as_of_date": "2024-04-15",
                "metric": idmc_cli.FLOW_METRIC,
                "value": 88,
                "series_semantics": idmc_cli.FLOW_SERIES_SEMANTICS,
                "source": "idmc_gidd",
            },
            {
                "iso3": "ETH",
                "as_of_date": "2024-04-20",
                "metric": idmc_cli.FLOW_METRIC,
                "value": 42,
                "series_semantics": idmc_cli.FLOW_SERIES_SEMANTICS,
                "source": "idmc_gidd",
            },
            {
                "iso3": "NGA",
                "as_of_date": "2024-04-25",
                "metric": idmc_cli.FLOW_METRIC,
                "value": 35,
                "series_semantics": idmc_cli.FLOW_SERIES_SEMANTICS,
                "source": "idmc_gidd",
            },
        ]
    )

    diagnostics = {
        "mode": "live",
        "network_mode": "live",
        "fallback_used": False,
        "http": {"requests": 1, "retries": 0, "status_last": 200, "cache": {"hits": 0, "misses": 1}},
        "cache": {"hits": 0, "misses": 1},
        "filters": {"window_start": "2024-04-01", "window_end": "2024-04-30", "countries": []},
        "performance": {"duration_s": 1, "wire_bytes": 1024, "body_bytes": 512, "records_per_sec": 3},
        "rate_limit": {
            "req_per_sec": 1,
            "max_concurrency": 1,
            "retries": 0,
            "retry_after_wait_s": 0,
            "rate_limit_wait_s": 0,
            "planned_wait_s": 0,
        },
        "chunks": {"enabled": False, "count": 1},
        "rows_fetched": len(frame),
        "rows_normalized": len(frame),
        "rows_written": len(frame),
    }

    monkeypatch.setattr(idmc_cli._client, "fetch", lambda *_args, **_kwargs: ({"monthly_flow": frame}, diagnostics))

    monkeypatch.setattr(idmc_cli, "normalize_all", lambda *_args, **_kwargs: (frame.copy(), {}))
    monkeypatch.setattr(idmc_cli, "maybe_map_hazards", lambda tidy, *_a, **_k: (tidy, []))
    monkeypatch.setattr(idmc_cli, "write_connectors_line", lambda payload: payload)
    monkeypatch.setattr(idmc_cli, "build_provenance", lambda **_kw: {"ok": True})
    monkeypatch.setattr(idmc_cli, "write_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(idmc_cli, "write_why_zero", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(idmc_cli, "write_facts_parquet", lambda path, df: Path(path).as_posix())

    monkeypatch.setenv("RESOLVER_EXPORT_ENABLE_IDMC", "1")
    monkeypatch.setenv("IDMC_WRITE_OUTPUTS", "1")
    monkeypatch.setenv("WRITE_TO_DUCKDB", "1")
    monkeypatch.setenv("RESOLVER_DB_URL", "duckdb:///tmp/test.duckdb")

    exit_code = idmc_cli.main(["--network-mode", "live"])
    assert exit_code == 0

    summary_path = Path("diagnostics/ingestion/idmc/summary.md")
    assert summary_path.exists()
    content = summary_path.read_text(encoding="utf-8")
    assert "## DuckDB write" in content
    assert "Rows inserted" in content
    assert "Target:" in content
    match = re.search(r"Rows inserted: (\d+)", content)
    assert match and int(match.group(1)) == len(frame)
