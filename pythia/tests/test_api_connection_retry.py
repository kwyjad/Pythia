# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Unit tests for the read-connection retry-once resilience in _execute.

When a query on the cached read connection raises a *connection-level* error
(closed/invalidated connection), ``pythia.api.core._execute`` must close the
cached connection, reopen from the same DB path, and retry the operation
exactly once. Ordinary query errors (catalog/binder/syntax/constraint) must
never trigger a retry.

Uses stub connection objects only — no real server or DB file required.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
duckdb = pytest.importorskip("duckdb")

from pythia.api import core


class _StubCursor:
    """Stands in for the cursor object DuckDB's execute() returns."""


class _FailingCon:
    """Connection stub whose execute() always raises the given exception."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc
        self.calls: list[tuple[str, object]] = []

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        raise self.exc


class _GoodCon:
    """Connection stub whose execute() succeeds and records calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.cursor_result = _StubCursor()

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        return self.cursor_result


def test_retries_once_on_connection_exception(monkeypatch):
    good = _GoodCon()
    reopen_calls: list[int] = []

    def _fake_reopen():
        reopen_calls.append(1)
        return good

    monkeypatch.setattr(core, "_reopen_read_connection", _fake_reopen)
    failing = _FailingCon(
        duckdb.ConnectionException(
            "Connection Error: Connection has already been closed"
        )
    )

    result = core._execute(failing, "SELECT 1")

    assert result is good.cursor_result
    assert len(failing.calls) == 1
    assert len(reopen_calls) == 1
    assert good.calls == [("SELECT 1", None)]


def test_retry_preserves_positional_params(monkeypatch):
    good = _GoodCon()
    monkeypatch.setattr(core, "_reopen_read_connection", lambda: good)
    failing = _FailingCon(duckdb.ConnectionException("Connection Error: closed"))

    core._execute(failing, "SELECT * FROM t WHERE x = ?", ["ETH"])

    assert good.calls == [("SELECT * FROM t WHERE x = ?", ["ETH"])]


def test_retry_recompiles_named_params(monkeypatch):
    good = _GoodCon()
    monkeypatch.setattr(core, "_reopen_read_connection", lambda: good)
    failing = _FailingCon(duckdb.ConnectionException("Connection Error: closed"))

    core._execute(failing, "SELECT * FROM t WHERE x = :iso3", {"iso3": "ETH"})

    # Named params are compiled to positional form on both attempts.
    assert failing.calls == [("SELECT * FROM t WHERE x = ?", ["ETH"])]
    assert good.calls == [("SELECT * FROM t WHERE x = ?", ["ETH"])]


def test_retries_on_fatal_exception(monkeypatch):
    good = _GoodCon()
    monkeypatch.setattr(core, "_reopen_read_connection", lambda: good)
    failing = _FailingCon(
        duckdb.FatalException(
            "FATAL Error: Failed: database has been invalidated because of a "
            "previous fatal error. The database must be restarted prior to "
            "being used again."
        )
    )

    result = core._execute(failing, "SELECT 1")
    assert result is good.cursor_result


def test_retries_on_generic_error_with_invalidated_message(monkeypatch):
    good = _GoodCon()
    monkeypatch.setattr(core, "_reopen_read_connection", lambda: good)
    failing = _FailingCon(
        duckdb.Error("some wrapper: database has been invalidated, restart needed")
    )

    result = core._execute(failing, "SELECT 1")
    assert result is good.cursor_result


@pytest.mark.parametrize(
    "exc",
    [
        duckdb.CatalogException('Catalog Error: Table "missing" does not exist!'),
        duckdb.BinderException('Binder Error: Referenced column "x" not found'),
        duckdb.ParserException('Parser Error: syntax error at or near "SELEC"'),
        duckdb.ConstraintException("Constraint Error: Duplicate key"),
        duckdb.InvalidInputException("Invalid Input Error: bad parameter count"),
        KeyError("Missing SQL parameter: iso3"),
        ValueError("unrelated"),
    ],
)
def test_no_retry_on_ordinary_query_errors(monkeypatch, exc):
    reopen_calls: list[int] = []

    def _fake_reopen():
        reopen_calls.append(1)
        return _GoodCon()

    monkeypatch.setattr(core, "_reopen_read_connection", _fake_reopen)
    failing = _FailingCon(exc)

    with pytest.raises(type(exc)):
        core._execute(failing, "SELECT 1")

    assert reopen_calls == []
    assert len(failing.calls) == 1


def test_never_loops_when_retry_also_fails(monkeypatch):
    second = _FailingCon(duckdb.ConnectionException("Connection Error: closed"))
    monkeypatch.setattr(core, "_reopen_read_connection", lambda: second)
    first = _FailingCon(duckdb.ConnectionException("Connection Error: closed"))

    with pytest.raises(duckdb.ConnectionException):
        core._execute(first, "SELECT 1")

    # Exactly one retry: the second failure propagates, no further reopen.
    assert len(first.calls) == 1
    assert len(second.calls) == 1


def test_is_connection_level_error_classification():
    assert core._is_connection_level_error(
        duckdb.ConnectionException("Connection Error: closed")
    )
    assert core._is_connection_level_error(
        duckdb.FatalException("database has been invalidated")
    )
    assert core._is_connection_level_error(
        RuntimeError("Connection has already been closed")
    )
    assert not core._is_connection_level_error(
        duckdb.CatalogException("Table does not exist")
    )
    assert not core._is_connection_level_error(
        duckdb.BinderException("column not found")
    )
    assert not core._is_connection_level_error(ValueError("nope"))
    assert not core._is_connection_level_error(KeyError("param"))


def test_execute_rejects_non_connection_objects():
    with pytest.raises(TypeError):
        core._execute(object(), "SELECT 1")


def test_app_reexports_execute_from_core():
    """The backward-compat seam: pythia.api.app._execute is core's _execute."""
    import pythia.api.app as app_mod

    assert app_mod._execute is core._execute
