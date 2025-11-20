from pathlib import Path

import pytest

from resolver.cli import odp_json_to_duckdb


def test_odp_cli_calls_build_and_write_with_expected_args(tmp_path, monkeypatch):
    db_path = tmp_path / "cli_odp.duckdb"
    called: dict[str, object] = {}

    def fake_build_and_write(*, config_path, normalizers_path, db_url, fetch_html, fetch_json, today):
        called["config_path"] = config_path
        called["normalizers_path"] = normalizers_path
        called["db_url"] = db_url
        called["today"] = today
        return 42

    monkeypatch.setattr(
        "resolver.ingestion.odp_duckdb.build_and_write_odp_series",
        fake_build_and_write,
    )

    exit_code = odp_json_to_duckdb.run(
        [
            "--db",
            str(db_path),
            "--config",
            "my_config.yml",
            "--normalizers",
            "my_norm.yml",
            "--today",
            "2025-01-01",
        ]
    )

    assert exit_code == 0
    assert called["config_path"] == "my_config.yml"
    assert called["normalizers_path"] == "my_norm.yml"
    assert called["db_url"]
    assert isinstance(called["db_url"], str)
    assert called["today"].isoformat() == "2025-01-01"


def test_odp_cli_invalid_today_returns_error(tmp_path, monkeypatch):
    def fail_build(**kwargs):
        pytest.fail("build_and_write_odp_series should not be called on invalid --today")

    monkeypatch.setattr(
        "resolver.ingestion.odp_duckdb.build_and_write_odp_series",
        fail_build,
    )

    exit_code = odp_json_to_duckdb.run(["--db", str(tmp_path / "x.duckdb"), "--today", "not-a-date"])

    assert exit_code == 2


def test_odp_cli_propagates_pipeline_failure(tmp_path, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("ODP failed")

    monkeypatch.setattr(
        "resolver.ingestion.odp_duckdb.build_and_write_odp_series",
        boom,
    )

    code = odp_json_to_duckdb.run(["--db", str(tmp_path / "x.duckdb")])

    assert code == 1
