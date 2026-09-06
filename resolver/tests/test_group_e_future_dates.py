# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Group E of the run-33946954189 repairs: dates in the future.

The repair pass was marked completed and the process was not. It fixed the
five future-dated rows in ``facts_resolved``, then hit ``facts_deltas``,
where clamping a future publication date onto an existing row's key
collided with a row already holding it:

    _duckdb.ConstraintException: Constraint Error: Duplicate key
    "event_id: , iso3: JPN, hazard_code: TC, metric: event_occurrence,
    as_of_date: 2026-09-30, publication_date: 2026-09-05, source_id: ,
    ym: 2026-09" violates unique constraint.

It exited before repairing anything in that table, and 296 future-dated rows
survived — the latest 2027-03-01 — to be found again by the next run.

Network-free.
"""

from __future__ import annotations

import datetime as dt

import duckdb
import pytest

from resolver.tools import repair_publication_dates as repair_mod

TODAY = dt.date(2026, 9, 6)


def _deltas_db(path):
    """``facts_deltas`` in the shape ``duckdb_io`` actually creates it —
    a UNIQUE INDEX that INCLUDES publication_date."""

    con = duckdb.connect(str(path))
    con.execute(
        """
        CREATE TABLE facts_deltas (
            ym TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT,
            value_new DOUBLE, as_of_date VARCHAR, publication_date VARCHAR,
            event_id TEXT, source_id TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    con.execute(
        """
        CREATE UNIQUE INDEX ux_facts_deltas_series ON facts_deltas (
            event_id, iso3, hazard_code, metric, as_of_date,
            publication_date, source_id, ym
        )
        """
    )
    return con


class TestCollisionSafety:
    def test_the_repair_no_longer_raises_on_a_collision(self, tmp_path):
        con = _deltas_db(tmp_path / "d.duckdb")
        # The exact shape from the run: one correct row already at the date
        # the future-dated one is about to be clamped to.
        con.executemany(
            "INSERT INTO facts_deltas VALUES (?,?,?,?,?,?,?,?,?, TIMESTAMP '2026-09-05')",
            [
                ("2026-09", "JPN", "TC", "event_occurrence", 1.0,
                 "2026-09-30", "2026-09-05", "", ""),
                ("2026-09", "JPN", "TC", "event_occurrence", 1.0,
                 "2026-09-30", "2027-03-01", "", ""),
            ],
        )

        counts = repair_mod.repair_table(con, "facts_deltas", TODAY)

        assert counts["future"] == 1
        assert counts["repaired"] == 1
        assert counts["merged_away"] == 1
        rows = con.execute(
            "SELECT publication_date FROM facts_deltas"
        ).fetchall()
        assert rows == [("2026-09-05",)]
        con.close()

    def test_a_non_colliding_row_is_repaired_in_place(self, tmp_path):
        con = _deltas_db(tmp_path / "d.duckdb")
        con.execute(
            "INSERT INTO facts_deltas VALUES ('2026-09','KEN','FL','affected',5.0,"
            "'2026-09-30','2027-03-01','','', TIMESTAMP '2026-09-05')"
        )

        counts = repair_mod.repair_table(con, "facts_deltas", TODAY)

        assert counts["repaired"] == 1
        assert counts["merged_away"] == 0
        assert con.execute(
            "SELECT publication_date FROM facts_deltas"
        ).fetchone()[0] == "2026-09-05"
        con.close()

    def test_two_future_rows_collapsing_onto_one_key_keep_one(self, tmp_path):
        con = _deltas_db(tmp_path / "d.duckdb")
        con.executemany(
            "INSERT INTO facts_deltas VALUES (?,?,?,?,?,?,?,?,?, TIMESTAMP '2026-09-05')",
            [
                ("2026-09", "SDN", "DR", "phase3plus_projection", 3.0,
                 "2027-03-31", "2027-03-01", "", ""),
                ("2026-09", "SDN", "DR", "phase3plus_projection", 3.0,
                 "2027-03-31", "2027-04-01", "", ""),
            ],
        )

        repair_mod.repair_table(con, "facts_deltas", TODAY)

        assert con.execute("SELECT COUNT(*) FROM facts_deltas").fetchone()[0] == 1
        assert repair_mod.count_future(con, today=TODAY)["facts_deltas"] == 0
        con.close()

    def test_the_pass_is_idempotent_over_a_collision(self, tmp_path):
        con = _deltas_db(tmp_path / "d.duckdb")
        con.executemany(
            "INSERT INTO facts_deltas VALUES (?,?,?,?,?,?,?,?,?, TIMESTAMP '2026-09-05')",
            [
                ("2026-09", "JPN", "TC", "event_occurrence", 1.0,
                 "2026-09-30", "2026-09-05", "", ""),
                ("2026-09", "JPN", "TC", "event_occurrence", 1.0,
                 "2026-09-30", "2027-03-01", "", ""),
            ],
        )

        repair_mod.repair_table(con, "facts_deltas", TODAY)
        second = repair_mod.repair_table(con, "facts_deltas", TODAY)

        assert second["future"] == 0 and second["repaired"] == 0
        assert con.execute("SELECT COUNT(*) FROM facts_deltas").fetchone()[0] == 1
        con.close()

    def test_a_row_that_is_not_future_dated_is_never_touched(self, tmp_path):
        con = _deltas_db(tmp_path / "d.duckdb")
        con.execute(
            "INSERT INTO facts_deltas VALUES ('2026-08','KEN','FL','affected',5.0,"
            "'2026-08-31','2026-08-04','','', TIMESTAMP '2026-08-05')"
        )

        repair_mod.repair_table(con, "facts_deltas", TODAY)

        assert con.execute(
            "SELECT publication_date FROM facts_deltas"
        ).fetchone()[0] == "2026-08-04"
        con.close()


class TestUniqueKeyDiscovery:
    def test_the_unique_index_is_read_from_the_database(self, tmp_path):
        con = _deltas_db(tmp_path / "d.duckdb")

        key = repair_mod._unique_key(con, "facts_deltas")

        assert "publication_date" in key
        assert "iso3" in key and "ym" in key
        con.close()

    def test_a_table_with_no_such_key_reports_none(self, tmp_path):
        con = duckdb.connect(str(tmp_path / "x.duckdb"))
        con.execute("CREATE TABLE emdat_pa (iso3 TEXT, publication_date VARCHAR)")

        assert repair_mod._unique_key(con, "emdat_pa") == []
        con.close()


class TestOneTableFailureDoesNotCostTheOthers:
    def test_a_failing_table_is_recorded_and_the_rest_are_repaired(
        self, tmp_path, monkeypatch
    ):
        con = _deltas_db(tmp_path / "d.duckdb")
        con.execute(
            "CREATE TABLE facts_resolved (iso3 TEXT, publication_date VARCHAR, "
            "created_at TIMESTAMP)"
        )
        con.execute(
            "INSERT INTO facts_resolved VALUES ('KEN','2027-03-01', "
            "TIMESTAMP '2026-09-05')"
        )
        con.execute(
            "INSERT INTO facts_deltas VALUES ('2026-09','KEN','FL','affected',5.0,"
            "'2026-09-30','2027-03-01','','', TIMESTAMP '2026-09-05')"
        )

        real = repair_mod.repair_table

        def flaky(connection, table, today, **kwargs):
            if table == "facts_resolved":
                raise RuntimeError("boom")
            return real(connection, table, today, **kwargs)

        monkeypatch.setattr(repair_mod, "repair_table", flaky)
        report = repair_mod.repair(con, today=TODAY)

        assert "boom" in report.tables["facts_resolved"]["error"]
        assert report.tables["facts_deltas"]["repaired"] == 1
        assert repair_mod.count_future(con, today=TODAY)["facts_deltas"] == 0
        con.close()


class TestFreezeDeadlineIsNotAnEvent:
    """E3 — frozen_at holds a DEADLINE, and a future one is correct."""

    def test_the_bundle_expects_a_future_freeze_deadline(self):
        from scripts import build_resolver_debug_bundle as bundle

        assert ("haz_resolutions", "frozen_at") in bundle.BundleBuilder._EXPECTED_FUTURE_PAIRS

    def test_a_period_end_in_the_future_is_expected_too(self):
        from scripts import build_resolver_debug_bundle as bundle

        assert "as_of_date" in bundle.BundleBuilder._EXPECTED_FUTURE_COLUMNS
        assert "publication_date" not in bundle.BundleBuilder._EXPECTED_FUTURE_COLUMNS, (
            "a publication date in the future is a defect, not a period end"
        )
