# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Group C of the run-33946954189 repairs: rows claimed and never attributed.

The reconciliation disagreed for three Phase-1 connectors:

    connector        claimed   touched since run start
    idmc_helix            89                         0
    acled_client         424                         0
    ifrc_go_client       205                         0

718 rows reached ``facts_raw`` — 424 + 205 + 89, exactly — so the staging
files were read and the loader ran. What it did not do was say WHEN it
wrote: ``updated_at`` was added to the facts tables by an ALTER, so it
carries no column default, and a plain INSERT that does not name it writes
NULL. The reconciliation reads ``updated_at`` first, found NULL on every
row, and correctly reported that this run had touched nothing.

The step that does the writing was also the only substantive step in the
workflow with no ``| tee``, so none of that could be told from the artifact.

Network-free.
"""

from __future__ import annotations

import datetime as dt

import duckdb
import pandas as pd
import pytest

from resolver.tools import load_and_derive as lad


@pytest.fixture()
def con(tmp_path):
    connection = duckdb.connect(str(tmp_path / "resolver.duckdb"))
    connection.execute(
        """
        CREATE TABLE facts_resolved (
            ym TEXT, iso3 TEXT, value DOUBLE,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP
        )
        """
    )
    yield connection
    connection.close()


class TestWriteStamp:
    def test_an_insert_stamps_updated_at(self, con):
        before = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        frame = pd.DataFrame({"ym": ["2026-08"], "iso3": ["KEN"], "value": [1.0]})

        assert lad._insert_dataframe(con, "facts_resolved", frame) == 1

        stamp = con.execute("SELECT updated_at FROM facts_resolved").fetchone()[0]
        assert stamp is not None, (
            "an unstamped row cannot be attributed to the run that wrote it"
        )
        assert stamp >= before - dt.timedelta(seconds=5)

    def test_the_row_is_findable_by_a_run_start_query(self, con):
        """The reconciliation's own question, asked of the loader's output."""

        started_at = (
            dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
            - dt.timedelta(minutes=1)
        ).isoformat()
        lad._insert_dataframe(
            con,
            "facts_resolved",
            pd.DataFrame({"ym": ["2026-08"], "iso3": ["KEN"], "value": [1.0]}),
        )

        touched = con.execute(
            "SELECT COUNT(*) FROM facts_resolved "
            "WHERE TRY_CAST(updated_at AS TIMESTAMP) >= TRY_CAST(? AS TIMESTAMP)",
            [started_at],
        ).fetchone()[0]
        assert touched == 1

    def test_a_table_with_no_stamp_column_is_written_unchanged(self, con):
        con.execute("CREATE TABLE facts_raw (ym TEXT, iso3 TEXT, value DOUBLE)")
        frame = pd.DataFrame({"ym": ["2026-08"], "iso3": ["KEN"], "value": [1.0]})

        assert lad._insert_dataframe(con, "facts_raw", frame) == 1
        assert con.execute("SELECT COUNT(*) FROM facts_raw").fetchone()[0] == 1

    def test_the_source_own_dates_are_never_touched(self, con):
        con.execute("ALTER TABLE facts_resolved ADD COLUMN as_of_date TEXT")
        con.execute("ALTER TABLE facts_resolved ADD COLUMN publication_date TEXT")
        frame = pd.DataFrame({
            "ym": ["2026-08"], "iso3": ["KEN"], "value": [1.0],
            "as_of_date": ["2026-08-31"], "publication_date": ["2026-08-04"],
        })

        lad._insert_dataframe(con, "facts_resolved", frame)

        as_of, published = con.execute(
            "SELECT as_of_date, publication_date FROM facts_resolved"
        ).fetchone()
        assert as_of == "2026-08-31"
        assert published == "2026-08-04", (
            "the write stamp says when we wrote; the source's dates say what "
            "the figure is about and when it was published"
        )

    def test_an_empty_frame_writes_nothing(self, con):
        assert lad._insert_dataframe(con, "facts_resolved", pd.DataFrame()) == 0

    def test_the_stamp_columns_are_the_write_time_ones_only(self):
        assert lad._WRITE_STAMP_COLUMNS == ("updated_at", "created_at")
        assert "as_of_date" not in lad._WRITE_STAMP_COLUMNS
        assert "publication_date" not in lad._WRITE_STAMP_COLUMNS


class TestSupersededConnector:
    """C3 — one source must not file two entries telling opposite stories."""

    def test_the_skipped_legacy_entry_is_marked_superseded(self):
        from scripts.build_resolver_debug_bundle import _superseded_connectors

        records = [
            {"connector_id": "idmc", "status": "skipped"},
            {"connector_id": "idmc_helix", "status": "ok"},
        ]

        assert _superseded_connectors(records) == {"idmc": "idmc_helix"}

    def test_nothing_is_superseded_when_the_live_path_did_not_run(self):
        from scripts.build_resolver_debug_bundle import _superseded_connectors

        records = [{"connector_id": "idmc", "status": "ok"}]

        assert _superseded_connectors(records) == {}

    def test_a_skipped_live_path_supersedes_nothing(self):
        from scripts.build_resolver_debug_bundle import _superseded_connectors

        records = [
            {"connector_id": "idmc", "status": "skipped"},
            {"connector_id": "idmc_helix", "status": "skipped"},
        ]

        assert _superseded_connectors(records) == {}


class TestPhase1IsLogged:
    """C1 — the step that writes facts_resolved leaves a log behind."""

    def test_both_invocations_are_teed(self):
        import pathlib

        workflow = pathlib.Path(".github/workflows/resolver_update.yml").read_text()

        assert "diagnostics/phase1_normalize.log" in workflow
        assert "diagnostics/phase1_load_and_derive.log" in workflow
