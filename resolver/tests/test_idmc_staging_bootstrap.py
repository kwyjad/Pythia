from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from resolver.ingestion.idmc import cli as idmc_cli


def _stub_config(config_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        api=SimpleNamespace(
            countries=["AAA", "BBB"],
            date_window=SimpleNamespace(start="2024-01-01", end="2024-01-31"),
            base_url="https://example.test",
            endpoints={"idus_json": "/data/idus_view_flat"},
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
        _config_warnings=("ingestion=resolver",),
    )


def _stub_fetch(*_args, **_kwargs):
    return {}, {
        "mode": "offline",
        "http": {"requests": 0},
        "cache": {},
        "filters": {},
        "performance": {},
        "rate_limit": {},
        "chunks": {},
    }


def _stub_normalize(*_args, **_kwargs):
    return pd.DataFrame(columns=["iso3", "date", "value"]), {}


def _stub_maybe_map(frame, *_args, **_kwargs):
    return frame, []


def _noop_write_connectors_line(_payload):  # pragma: no cover - exercised via CLI
    return None


def _stub_provenance(**_kwargs):
    return {"ok": True}


def _run_cli(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "resolver" / "config" / "idmc.yml"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IDMC_API_TOKEN", "token")
    monkeypatch.setattr(idmc_cli, "load", lambda: _stub_config(config_path))
    monkeypatch.setattr(idmc_cli, "fetch", _stub_fetch)
    monkeypatch.setattr(idmc_cli, "normalize_all", _stub_normalize)
    monkeypatch.setattr(idmc_cli, "maybe_map_hazards", _stub_maybe_map)
    monkeypatch.setattr(idmc_cli, "write_connectors_line", _noop_write_connectors_line)
    monkeypatch.setattr(idmc_cli, "build_provenance", _stub_provenance)
    exit_code = idmc_cli.main(["--skip-network"])
    assert exit_code == 0


def test_idmc_cli_writes_header_on_zero_rows(monkeypatch, tmp_path):
    _run_cli(monkeypatch, tmp_path)
    flow_path = tmp_path / "resolver" / "staging" / "idmc" / "flow.csv"
    assert flow_path.exists()
    frame = pd.read_csv(flow_path)
    assert list(frame.columns) == [
        "iso3",
        "as_of_date",
        "metric",
        "value",
        "series_semantics",
        "source",
    ]
    assert frame.empty


def test_why_zero_json_includes_config_details(monkeypatch, tmp_path):
    _run_cli(monkeypatch, tmp_path)
    why_zero_path = tmp_path / "diagnostics" / "ingestion" / "idmc" / "why_zero.json"
    assert why_zero_path.exists()
    payload = json.loads(why_zero_path.read_text(encoding="utf-8"))
    assert payload["token_present"] is True
    assert payload["countries_count"] == 2
    assert payload["countries_sample"] == ["AAA", "BBB"]
    assert payload["countries_source"] == "config list"
    assert payload["date_window"] == {"start": "2024-01-01", "end": "2024-01-31"}
    assert payload["window"] == {"start": "2024-01-01", "end": "2024-01-31"}
    assert payload["series"] == ["stock", "flow"]
    assert payload["network_attempted"] is False
    assert payload["config_source"] == "ingestion"
    assert payload["config_path_used"] == str(tmp_path / "resolver" / "config" / "idmc.yml")
    assert payload["loader_warnings"] == ["ingestion=resolver"]
