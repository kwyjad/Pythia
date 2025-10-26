"""CI guardrails ensuring required extras are installed."""

import importlib.util


def test_duckdb_import_available():
    spec = importlib.util.find_spec("duckdb")
    assert spec is not None, "DuckDB module should be installed in CI runs"


def test_pyproject_extras_present():
    spec = importlib.util.find_spec("pytest")
    assert spec is not None, "Pytest should be available when installing the test extra"
