# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import pytest

from resolver.cli import emdat_to_duckdb
from resolver.db import duckdb_io


@pytest.mark.duckdb
def test_emdat_cli_offline_smoke(tmp_path, monkeypatch, capfd):
    pytest.importorskip("duckdb")
    if not duckdb_io.DUCKDB_AVAILABLE:
        pytest.skip("DuckDB module not available")

    monkeypatch.delenv("EMDAT_API_KEY", raising=False)

    class _FailingClient:
        def __init__(self, *args, **kwargs):  # noqa: D401 - simple guard
            raise AssertionError("Network client should not be instantiated in offline mode")

    monkeypatch.setattr(emdat_to_duckdb, "EmdatClient", _FailingClient)

    db_path = tmp_path / "emdat.duckdb"
    exit_code = emdat_to_duckdb.run(
        [
            "--from",
            "2021",
            "--to",
            "2021",
            "--db",
            str(db_path),
        ]
    )
    assert exit_code == 0

    stdout = capfd.readouterr().out.splitlines()
    assert any(line.startswith("✅ Wrote") for line in stdout)
    assert any("emdat_pa" in line for line in stdout)

    conn = duckdb_io.get_db(f"duckdb:///{db_path.as_posix()}")
    try:
        rows = conn.execute("SELECT COUNT(*) FROM emdat_pa").fetchone()[0]
    finally:
        duckdb_io.close_db(conn)
    assert rows > 0
