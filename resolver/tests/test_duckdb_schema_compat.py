import logging

from resolver.db import duckdb_io


def test_schema_exception_tuple_constructs():
    exc_tuple = duckdb_io._SCHEMA_EXC_TUPLE  # noqa: SLF001 - test internal compatibility tuple
    assert isinstance(exc_tuple, tuple)
    # Should include base duckdb.Error plus at least catalog + not-implemented variants
    assert len(exc_tuple) >= 3


def test_init_schema_fastpath_emits_skip(caplog, tmp_path):
    db_url = f"duckdb:///{tmp_path/'compat.duckdb'}"
    conn = duckdb_io.get_db(db_url)
    caplog.set_level(logging.DEBUG, logger="resolver.db.duckdb_io")

    duckdb_io.init_schema(conn)
    caplog.clear()
    duckdb_io.init_schema(conn)

    skip_logs = [record.message for record in caplog.records if "skipping DDL execution" in record.message]
    assert skip_logs, "Expected fastpath skip log to be emitted on second init"
