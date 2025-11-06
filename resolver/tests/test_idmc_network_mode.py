"""Tests covering IDMC network mode selection."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from resolver.ingestion.idmc import cli as idmc_cli


def _stub_config(config_path: Path) -> SimpleNamespace:
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


def _common_stubs(
    monkeypatch,
    tmp_path: Path,
    expected_mode: str,
    http_counts,
    http_requests: int,
    *,
    expected_helix_id: str | None = None,
):
    config_path = tmp_path / "resolver" / "config" / "idmc.yml"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IDMC_API_TOKEN", "token")
    monkeypatch.setattr(idmc_cli, "load", lambda: _stub_config(config_path))

    observed_modes: list[str] = []

    class _FakeClient:
        def __init__(self, *, helix_client_id=None):
            if expected_helix_id is not None:
                assert helix_client_id == expected_helix_id
            else:
                assert helix_client_id is None
            self.helix_client_id = helix_client_id

        def fetch(self, cfg, *, network_mode, **_kwargs):
            assert network_mode == expected_mode
            observed_modes.append(network_mode)
            diagnostics = {
                "mode": "online" if network_mode in {"live", "helix"} else network_mode,
                "network_mode": network_mode,
                "http": {
                    "requests": http_requests,
                    "retries": 0,
                    "status_last": 200 if http_requests else None,
                    "retry_after_events": 0,
                    "cache": {"hits": 0, "misses": 0},
                    "latency_ms": {"p50": 0, "p95": 0, "max": 0},
                },
                "cache": {"hits": 0, "misses": 0},
                "filters": {"window_start": None, "window_end": None, "countries": []},
                "performance": {},
                "rate_limit": {},
                "chunks": {"enabled": False, "count": 0, "by_month": []},
                "http_status_counts": http_counts,
            }
            frame = pd.DataFrame(
                [{"displacement_date": "2024-01-05", "figure": 10, "iso3": "AAA"}]
            )
            return {"monthly_flow": frame}, diagnostics

    monkeypatch.setattr(idmc_cli, "IdmcClient", _FakeClient)
    monkeypatch.setattr(
        idmc_cli,
        "normalize_all",
        lambda *_args, **_kwargs: (
            pd.DataFrame(
                columns=[
                    "iso3",
                    "as_of_date",
                    "metric",
                    "value",
                    "series_semantics",
                    "source",
                ]
            ),
            {},
        ),
    )
    monkeypatch.setattr(
        idmc_cli,
        "maybe_map_hazards",
        lambda frame, *_args, **_kwargs: (frame, pd.DataFrame()),
    )
    monkeypatch.setattr(idmc_cli, "build_provenance", lambda **_kwargs: {"ok": True})
    monkeypatch.setattr(idmc_cli, "probe_reachability", lambda *_args, **_kwargs: {"ok": True})

    captured_payloads: list[dict] = []

    def _capture(payload):
        captured_payloads.append(payload)

    monkeypatch.setattr(idmc_cli, "write_connectors_line", _capture)

    return observed_modes, captured_payloads


def test_idmc_cli_default_live(monkeypatch, tmp_path):
    observed_modes, captured = _common_stubs(
        monkeypatch,
        tmp_path,
        expected_mode="live",
        http_counts={"2xx": 1, "4xx": 0, "5xx": 0},
        http_requests=1,
    )

    exit_code = idmc_cli.main([])
    assert exit_code == 0
    assert observed_modes == ["live"]
    assert captured, "connector diagnostics should be captured"
    payload = captured[-1]
    assert payload["network_mode"] == "live"
    assert payload["http_status_counts"] == {"2xx": 1, "4xx": 0, "5xx": 0}

    why_zero_path = tmp_path / "diagnostics" / "ingestion" / "idmc" / "why_zero.json"
    assert why_zero_path.exists()
    why_zero_payload = json.loads(why_zero_path.read_text(encoding="utf-8"))
    assert why_zero_payload["network_mode"] == "live"
    assert why_zero_payload["http_status_counts"] == {"2xx": 1, "4xx": 0, "5xx": 0}


def test_idmc_network_mode_fixture(monkeypatch, tmp_path):
    observed_modes, captured = _common_stubs(
        monkeypatch,
        tmp_path,
        expected_mode="fixture",
        http_counts=None,
        http_requests=0,
    )

    exit_code = idmc_cli.main(["--network-mode", "fixture"])
    assert exit_code == 0
    assert observed_modes == ["fixture"]
    payload = captured[-1]
    assert payload["network_mode"] == "fixture"
    assert payload["http_status_counts"] == {"2xx": 0, "4xx": 0, "5xx": 0}

    why_zero_path = tmp_path / "diagnostics" / "ingestion" / "idmc" / "why_zero.json"
    assert why_zero_path.exists()
    why_zero_payload = json.loads(why_zero_path.read_text(encoding="utf-8"))
    assert why_zero_payload["network_mode"] == "fixture"
    assert why_zero_payload["http_status_counts"] == {"2xx": 0, "4xx": 0, "5xx": 0}


def test_idmc_network_mode_cache_only(monkeypatch, tmp_path):
    observed_modes, captured = _common_stubs(
        monkeypatch,
        tmp_path,
        expected_mode="cache_only",
        http_counts=None,
        http_requests=0,
    )

    exit_code = idmc_cli.main(["--network-mode", "cache_only"])
    assert exit_code == 0
    assert observed_modes == ["cache_only"]
    payload = captured[-1]
    assert payload["network_mode"] == "cache_only"
    assert payload["http"]["requests"] == 0
    assert payload["http_status_counts"] == {"2xx": 0, "4xx": 0, "5xx": 0}

    why_zero_path = tmp_path / "diagnostics" / "ingestion" / "idmc" / "why_zero.json"
    assert why_zero_path.exists()
    why_zero_payload = json.loads(why_zero_path.read_text(encoding="utf-8"))
    assert why_zero_payload["network_mode"] == "cache_only"
    assert why_zero_payload["http_status_counts"] == {"2xx": 0, "4xx": 0, "5xx": 0}


def test_idmc_cli_accepts_helix_mode(monkeypatch, tmp_path):
    observed_modes, captured = _common_stubs(
        monkeypatch,
        tmp_path,
        expected_mode="helix",
        http_counts={"2xx": 1, "4xx": 0, "5xx": 0},
        http_requests=1,
    )

    exit_code = idmc_cli.main(["--network-mode", "helix"])
    assert exit_code == 0
    assert observed_modes == ["helix"]
    payload = captured[-1]
    assert payload["network_mode"] == "helix"
    assert payload["http_status_counts"] == {"2xx": 1, "4xx": 0, "5xx": 0}


def test_idmc_cli_plumbs_helix_id(monkeypatch, tmp_path):
    helix_id = "client-123"
    monkeypatch.setenv("IDMC_HELIX_CLIENT_ID", helix_id)
    observed_modes, captured = _common_stubs(
        monkeypatch,
        tmp_path,
        expected_mode="live",
        http_counts={"2xx": 1, "4xx": 0, "5xx": 0},
        http_requests=1,
        expected_helix_id=helix_id,
    )

    exit_code = idmc_cli.main([])
    assert exit_code == 0
    assert observed_modes == ["live"]
    payload = captured[-1]
    assert payload["network_mode"] == "live"
    assert payload["http_status_counts"] == {"2xx": 1, "4xx": 0, "5xx": 0}

