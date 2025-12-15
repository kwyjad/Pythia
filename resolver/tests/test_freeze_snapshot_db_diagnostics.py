# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from resolver.tools import freeze_snapshot

pytestmark = [
    pytest.mark.legacy_freeze,
    pytest.mark.xfail(
        reason=(
            "Legacy freeze_snapshot DB diagnostics dropped in favour of DB-backed snapshot builder."
        )
    ),
]


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


def _make_resolved_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "iso3": ["KEN"],
            "metric": ["fatalities_battle_month"],
            "series_semantics": ["new"],
            "value": [5],
            "as_of_date": ["2024-01-31"],
            "publication_date": ["2024-02-01"],
            "ym": ["2024-01"],
        }
    )


def _make_deltas_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "iso3": ["KEN"],
            "metric": ["fatalities_battle_month"],
            "series_semantics": ["new"],
            "value": [5],
            "value_new": [5],
            "as_of": ["2024-01-31"],
            "ym": ["2024-01"],
        }
    )


def _stub_duckdb(monkeypatch: pytest.MonkeyPatch, *, failing: bool = False) -> None:
    class _Conn:
        def close(self):  # noqa: D401 - test stub
            return None

    def get_db(url: str):  # noqa: D401 - test stub
        return _Conn()

    def init_schema(conn):  # noqa: D401 - test stub
        return conn

    if failing:
        def upsert_dataframe(*_args, **_kwargs):  # noqa: D401 - test stub
            raise RuntimeError("freeze boom")
    else:
        def upsert_dataframe(conn, table, frame, *, keys):  # noqa: D401 - test stub
            return _StubResult(rows_delta=len(frame), rows_written=len(frame), rows_in=len(frame))

    stub = SimpleNamespace(
        get_db=get_db,
        init_schema=init_schema,
        upsert_dataframe=upsert_dataframe,
        FACTS_RESOLVED_KEY_COLUMNS=["iso3", "ym", "metric"],
        FACTS_DELTAS_KEY_COLUMNS=["iso3", "ym", "metric"],
    )

    monkeypatch.setattr(freeze_snapshot, "duckdb_io", stub)
    monkeypatch.setattr(freeze_snapshot, "canonicalize_duckdb_target", lambda url: (url, url))


def test_freeze_snapshot_db_diagnostics_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    _stub_duckdb(monkeypatch)

    freeze_snapshot._maybe_write_db(
        ym="2024-01",
        facts_df=_make_resolved_df(),
        validated_facts_df=None,
        preview_df=None,
        resolved_df=_make_resolved_df(),
        deltas_df=_make_deltas_df(),
        manifest={"created_at_utc": "2024-02-01T00:00:00Z"},
        facts_out=tmp_path / "facts.csv",
        deltas_out=tmp_path / "deltas.csv",
        write_db=True,
        db_url="duckdb:///freeze.duckdb",
    )

    diag_path = tmp_path / "diagnostics" / "ingestion" / "freeze_db.json"
    assert diag_path.is_file()
    payload = json.loads(diag_path.read_text(encoding="utf-8"))
    assert payload["facts_resolved_rows"] == 1
    assert payload["facts_deltas_rows"] == 1
    assert payload["facts_resolved_semantics"] == {"new": 1}

    summary_path = tmp_path / "diagnostics" / "ingestion" / "summary.md"
    summary = summary_path.read_text(encoding="utf-8")
    assert "## Freeze Snapshot — DB diagnostics" in summary
    assert "new: 1" in summary
    assert "## DB Write Diagnostics" not in summary


def test_freeze_snapshot_db_diagnostics_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    _stub_duckdb(monkeypatch, failing=True)

    freeze_snapshot._maybe_write_db(
        ym="2024-01",
        facts_df=_make_resolved_df(),
        validated_facts_df=None,
        preview_df=None,
        resolved_df=_make_resolved_df(),
        deltas_df=_make_deltas_df(),
        manifest={"created_at_utc": "2024-02-01T00:00:00Z"},
        facts_out=tmp_path / "facts.csv",
        deltas_out=tmp_path / "deltas.csv",
        write_db=True,
        db_url="duckdb:///freeze.duckdb",
    )

    diag_path = tmp_path / "diagnostics" / "ingestion" / "freeze_db.json"
    assert diag_path.is_file()
    payload = json.loads(diag_path.read_text(encoding="utf-8"))
    assert payload["facts_resolved_rows"] == 1

    summary_path = tmp_path / "diagnostics" / "ingestion" / "summary.md"
    summary = summary_path.read_text(encoding="utf-8")
    assert "## DB Write Diagnostics" in summary
    assert "- **Step:** Freeze Snapshot" in summary
    assert "freeze boom" in summary
    assert "## Freeze Snapshot — DB diagnostics" not in summary
