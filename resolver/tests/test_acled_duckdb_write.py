import pandas as pd
import pytest

from resolver.cli import acled_to_duckdb
from resolver.db import duckdb_io


@pytest.mark.duckdb
def test_acled_duckdb_cli_idempotent(tmp_path, monkeypatch, capfd):
    pytest.importorskip("duckdb")
    if not duckdb_io.DUCKDB_AVAILABLE:
        pytest.skip("DuckDB module not available")

    frame = pd.DataFrame(
        {
            "iso3": ["KEN", "ETH"],
            "month": ["2024-01-01", "2024-02-01"],
            "fatalities": [5, 3],
            "source": ["ACLED", "ACLED"],
            "updated_at": ["2024-03-01T00:00:00Z", "2024-03-01T00:00:00Z"],
        }
    )
    frame["month"] = pd.to_datetime(frame["month"], utc=True).dt.tz_convert(None)
    frame["updated_at"] = pd.to_datetime(frame["updated_at"], utc=True)

    class _StubClient:
        def __init__(self, *_, **__):
            pass

        def monthly_fatalities(self, *_args, **_kwargs):
            return frame.copy()

    monkeypatch.setattr(acled_to_duckdb, "ACLEDClient", _StubClient)

    db_path = tmp_path / "acled.duckdb"
    args = [
        "--start",
        "2024-01-01",
        "--end",
        "2024-02-29",
        "--db",
        str(db_path),
    ]

    exit_code_first = acled_to_duckdb.run(args)
    assert exit_code_first == 0
    stdout_first = capfd.readouterr().out.splitlines()
    assert any(line.startswith("✅ Wrote") for line in stdout_first)

    exit_code_second = acled_to_duckdb.run(args)
    assert exit_code_second == 0
    stdout_second = capfd.readouterr().out.splitlines()
    assert any(line.startswith("✅ Wrote") for line in stdout_second)

    db_url = f"duckdb:///{db_path.as_posix()}"
    conn = duckdb_io.get_db(db_url)
    try:
        rows = conn.execute("SELECT COUNT(*) FROM acled_monthly_fatalities").fetchone()[0]
        assert rows == len(frame)

        dupes = conn.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT iso3, month, COUNT(*) AS c
                FROM acled_monthly_fatalities
                GROUP BY 1, 2
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
        assert dupes == 0

        schema_rows = conn.execute("PRAGMA table_info('acled_monthly_fatalities')").fetchall()
        column_types = {name: dtype for (_, name, dtype, *_rest) in schema_rows}
        assert column_types["month"].upper() == "DATE"
        assert column_types["fatalities"].upper() in {"BIGINT", "INTEGER"}

        indexes = conn.execute(
            "SELECT index_name FROM duckdb_indexes() WHERE table_name = 'acled_monthly_fatalities'"
        ).fetchall()
        assert any(name == "ux_acled_monthly_fatalities" for (name,) in indexes)
    finally:
        duckdb_io.close_db(conn)
