# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from resolver.tools import export_facts


class _StubResult:
    def __init__(self, rows_delta: int = 1, rows_written: int = 1, rows_in: int = 1) -> None:
        self.rows_delta = rows_delta
        self.rows_written = rows_written
        self.rows_in = rows_in

    def to_dict(self) -> dict[str, int]:
        return {
            "rows_delta": int(self.rows_delta),
            "rows_written": int(self.rows_written),
            "rows_in": int(self.rows_in),
        }


@pytest.fixture
def _stub_duckdb(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    def get_db(url: str):  # noqa: D401 - test stub
        calls["url"] = url
        return object()

    def init_schema(conn):  # noqa: D401 - test stub
        calls["init"] = True
        return conn

    def upsert_dataframe(conn, table, frame, *, keys):  # noqa: D401 - test stub
        calls.setdefault("tables", []).append((table, len(frame)))
        return _StubResult(rows_delta=len(frame), rows_written=len(frame), rows_in=len(frame))

    stub = SimpleNamespace(
        get_db=get_db,
        init_schema=init_schema,
        upsert_dataframe=upsert_dataframe,
        FACTS_RESOLVED_KEY_COLUMNS=["iso3", "ym", "metric"],
        FACTS_DELTAS_KEY_COLUMNS=["iso3", "ym", "metric"],
    )

    monkeypatch.setattr(export_facts, "duckdb_io", stub)
    monkeypatch.setattr(
        export_facts,
        "canonicalize_duckdb_target",
        lambda url: (str(Path(url.replace("duckdb:///", ""))), url),
    )

    return calls


def _sample_resolved_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_semantics": ["new"],
            "metric": ["fatalities_battle_month"],
            "value": [5],
            "as_of_date": ["2024-01-31"],
            "publication_date": ["2024-02-01"],
            "ym": ["2024-01"],
        }
    )


def _sample_deltas_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_semantics": ["new"],
            "metric": ["fatalities_battle_month"],
            "value": [5],
            "as_of": ["2024-01-31"],
            "ym": ["2024-01"],
        }
    )


def test_export_facts_db_diagnostics_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _stub_duckdb):
    monkeypatch.chdir(tmp_path)

    export_facts._maybe_write_to_db(
        facts_resolved=_sample_resolved_df(),
        facts_deltas=_sample_deltas_df(),
        db_url="duckdb:///example.duckdb",
        write_db=True,
        fail_on_error=True,
    )

    diag_path = tmp_path / "diagnostics" / "ingestion" / "export_facts_db.json"
    assert diag_path.is_file()
    payload = json.loads(diag_path.read_text(encoding="utf-8"))
    assert payload["facts_resolved_rows"] == 1
    assert payload["facts_deltas_rows"] == 1
    assert payload["facts_resolved_semantics"] == {"new": 1}

    summary_path = tmp_path / "diagnostics" / "ingestion" / "summary.md"
    summary = summary_path.read_text(encoding="utf-8")
    assert "## DuckDB — Export Facts write" in summary
    assert "facts_resolved semantics" in summary
    assert "new: 1" in summary


def test_export_facts_db_diagnostics_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)

    def failing_upsert(*_args, **_kwargs):  # noqa: D401 - test stub
        raise RuntimeError("boom")

    stub = SimpleNamespace(
        get_db=lambda url: object(),
        init_schema=lambda conn: conn,
        upsert_dataframe=failing_upsert,
        FACTS_RESOLVED_KEY_COLUMNS=["iso3", "ym", "metric"],
        FACTS_DELTAS_KEY_COLUMNS=["iso3", "ym", "metric"],
    )
    monkeypatch.setattr(export_facts, "duckdb_io", stub)
    monkeypatch.setattr(export_facts, "canonicalize_duckdb_target", lambda url: (url, url))

    with pytest.raises(export_facts.DuckDBWriteError):
        export_facts._maybe_write_to_db(
            facts_resolved=_sample_resolved_df(),
            facts_deltas=_sample_deltas_df(),
            db_url="duckdb:///example.duckdb",
            write_db=True,
            fail_on_error=True,
        )

    diag_path = tmp_path / "diagnostics" / "ingestion" / "export_facts_db.json"
    assert diag_path.is_file()
    payload = json.loads(diag_path.read_text(encoding="utf-8"))
    assert payload["facts_resolved_rows"] == 1

    summary_path = tmp_path / "diagnostics" / "ingestion" / "summary.md"
    summary = summary_path.read_text(encoding="utf-8")
    assert "## DB Write Diagnostics" in summary
    assert "- **Step:** Export Facts" in summary
    assert "boom" in summary
    assert "DuckDB — Export Facts write" not in summary
