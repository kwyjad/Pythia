import pandas as pd
import pytest

from resolver.cli import acled_to_duckdb
from resolver.db import duckdb_io


@pytest.mark.duckdb
def test_cli_handles_stub_without_optional_attrs(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    if not duckdb_io.DUCKDB_AVAILABLE:
        pytest.skip("DuckDB module not available")

    frame = pd.DataFrame(
        {
            "iso3": ["KEN"],
            "month": ["2024-01-01"],
            "fatalities": [1],
            "source": ["ACLED"],
            "updated_at": ["2024-01-01T00:00:00Z"],
        }
    )
    frame["month"] = pd.to_datetime(frame["month"], utc=True).dt.tz_convert(None)
    frame["updated_at"] = pd.to_datetime(frame["updated_at"], utc=True)

    class _StubClient:
        def monthly_fatalities(self, *_args, **_kwargs):
            return frame.copy()

    monkeypatch.setattr(acled_to_duckdb, "ACLEDClient", _StubClient)

    diag_dir = tmp_path / "diagnostics" / "acled"
    diag_file = diag_dir / "duckdb_summary.md"
    monkeypatch.setattr(acled_to_duckdb, "ACLED_DIAGNOSTICS_DIR", diag_dir)
    monkeypatch.setattr(acled_to_duckdb, "ACLED_DUCKDB_SUMMARY_PATH", diag_file)

    db_path = tmp_path / "acled.duckdb"
    args = [
        "--start",
        "2024-01-01",
        "--end",
        "2024-01-31",
        "--db",
        str(db_path),
    ]

    exit_code = acled_to_duckdb.run(args)
    assert exit_code == 0
    assert db_path.exists()
    assert diag_file.exists()
    text = diag_file.read_text(encoding="utf-8")
    assert "Page size: 0" in text
    assert "Fields parameter:" in text
