"""Ensure diagnostics stay quiet when RESOLVER_DIAG is unset."""

import importlib
import logging
import sys


def test_imports_without_diag_flag_are_quiet(monkeypatch, caplog):
    monkeypatch.delenv("RESOLVER_DIAG", raising=False)
    caplog.clear()
    modules = [
        "resolver.db.duckdb_io",
        "resolver.cli.resolver_cli",
        "resolver.query.selectors",
    ]
    for name in modules:
        if name in sys.modules:
            del sys.modules[name]
    for name in modules:
        importlib.import_module(name)
    diag_records = [record for record in caplog.records if record.levelno == logging.DEBUG]
    assert not diag_records
