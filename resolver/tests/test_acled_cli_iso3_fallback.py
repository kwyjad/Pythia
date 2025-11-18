import json
from pathlib import Path

import pandas as pd

from resolver.cli import acled_to_duckdb


def _reset_diag_file(path: Path) -> None:
    if path.exists():
        path.unlink()


def _stub_client_with_rows(rows):
    class _StubClient:
        page_size = 500

        def monthly_fatalities(self, start, end, countries=None):  # noqa: D401,ARG002
            return pd.DataFrame(rows)

    return _StubClient()


def test_iso3_fallback_populates_missing(monkeypatch, capsys):
    rows = [
        {"month": "2025-11-01", "country": "Turkey", "fatalities": 3},
        {"month": "2025-11-08", "country": "Kenya", "fatalities": 1},
        {"month": "2025-11-15", "country": "France", "fatalities": 2},
    ]
    _reset_diag_file(acled_to_duckdb.ACLED_CLI_FRAME_DIAG_PATH)

    stub = _stub_client_with_rows(rows)
    monkeypatch.setattr(acled_to_duckdb, "ACLEDClient", lambda: stub)
    monkeypatch.setattr(acled_to_duckdb.duckdb_io, "DUCKDB_AVAILABLE", True)

    exit_code = acled_to_duckdb.run(
        ["--start", "2025-11-01", "--end", "2025-11-30", "--dry-run"]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "✅ Wrote 0 rows to DuckDB (dry-run)" in captured.out

    diag = json.loads(acled_to_duckdb.ACLED_CLI_FRAME_DIAG_PATH.read_text())
    assert diag["input_rows"] == 3
    assert diag["missing_iso3_before"] > diag["missing_iso3_after"]


def test_iso3_already_present_keeps_counts(monkeypatch, capsys):
    rows = [
        {"month": "2025-12-01", "iso3": "USA", "fatalities": 5},
        {"month": "2025-12-15", "iso3": "MEX", "fatalities": 1},
    ]
    _reset_diag_file(acled_to_duckdb.ACLED_CLI_FRAME_DIAG_PATH)

    stub = _stub_client_with_rows(rows)
    monkeypatch.setattr(acled_to_duckdb, "ACLEDClient", lambda: stub)
    monkeypatch.setattr(acled_to_duckdb.duckdb_io, "DUCKDB_AVAILABLE", True)

    exit_code = acled_to_duckdb.run(
        ["--start", "2025-12-01", "--end", "2025-12-31", "--dry-run"]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "✅ Wrote 0 rows to DuckDB (dry-run)" in captured.out

    diag = json.loads(acled_to_duckdb.ACLED_CLI_FRAME_DIAG_PATH.read_text())
    assert diag["input_rows"] == 2
    assert diag["missing_iso3_after"] == 0
