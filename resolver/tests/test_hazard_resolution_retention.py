# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Guards for raw-cache retention.

The load-bearing test here is the ReliefWeb one. ``record_id`` is the
DOCUMENT id, so the same document is legitimately cached once per cell it was
fetched for; a compaction keyed on ``record_id`` alone would keep one cell's
copy and silently delete every other cell's documents. Everything else in
this file is about not touching the machine's verdicts or its resume ledger.
"""

from __future__ import annotations

import duckdb
import pytest

from resolver.hazard_resolution import retention
from resolver.hazard_resolution.reliefweb_docs import documents_for_country_month
from resolver.hazard_resolution.schema import ensure_haz_schema
from resolver.hazard_resolution.sources import (
    RawRecord,
    load_raw_records,
    store_raw_records,
)


@pytest.fixture()
def con():
    con = duckdb.connect(":memory:")
    ensure_haz_schema(con)
    return con


def _revision(con, source, record_id, payload, *, iso3, ym, hazard, stamp):
    """Append a row directly, bypassing the (now working) dedup."""
    store_raw_records(con, source, [RawRecord(
        record_id=record_id, payload=payload, iso3=iso3, ym=ym, hazard=hazard,
    )])
    con.execute(
        f"UPDATE haz_raw_{source} SET retrieved_at = ? "
        "WHERE record_id = ? AND iso3 IS NOT DISTINCT FROM ? "
        "AND ym IS NOT DISTINCT FROM ? AND retrieved_at = ("
        f"  SELECT MAX(retrieved_at) FROM haz_raw_{source} WHERE record_id = ?"
        "  AND iso3 IS NOT DISTINCT FROM ? AND ym IS NOT DISTINCT FROM ?)",
        [stamp, record_id, iso3, ym, record_id, iso3, ym],
    )


class TestTheCompactionKey:
    """The data-loss trap: record_id is the DOCUMENT, not the cell."""

    def test_one_document_cached_for_two_cells_keeps_both(self, con):
        for iso3, ym in (("PHL", "2024-03"), ("VNM", "2024-04")):
            store_raw_records(con, "reliefweb_docs", [RawRecord(
                record_id="rw-42",
                payload={"doc_id": "42", "iso3": iso3, "ym": ym,
                         "hazard": "flood", "body": f"body for {iso3}",
                         "title": "Floods", "source_rank": 1},
                iso3=iso3, ym=ym, hazard="flood",
            )])
        assert con.execute(
            "SELECT COUNT(*) FROM haz_raw_reliefweb_docs"
        ).fetchone()[0] == 2

        retention.compact_raw_cache(con, "reliefweb_docs", apply=True)

        # Both cells still resolve to their OWN document.
        assert con.execute(
            "SELECT COUNT(*) FROM haz_raw_reliefweb_docs"
        ).fetchone()[0] == 2
        phl = documents_for_country_month(con, "PHL", "2024-03", "flood")
        vnm = documents_for_country_month(con, "VNM", "2024-04", "flood")
        assert phl and vnm
        assert phl[0]["body"] == "body for PHL"
        assert vnm[0]["body"] == "body for VNM"

    def test_revisions_within_one_cell_collapse_to_the_newest(self, con):
        for i, stamp in enumerate(
            ["2026-08-05 20:30:00", "2026-08-06 20:30:00", "2026-08-07 20:30:00"]
        ):
            _revision(
                con, "emdat", "emdat-1", {"total_affected": 100 + i},
                iso3="PHL", ym="2024-03", hazard="FL", stamp=stamp,
            )
        assert con.execute("SELECT COUNT(*) FROM haz_raw_emdat").fetchone()[0] == 3
        before = load_raw_records(con, "emdat")

        retention.compact_raw_cache(con, "emdat", apply=True)

        assert con.execute("SELECT COUNT(*) FROM haz_raw_emdat").fetchone()[0] == 1
        after = load_raw_records(con, "emdat")
        # The reader saw the newest before; it must see the same one after.
        assert [r["total_affected"] for r in after] == \
               [r["total_affected"] for r in before] == [102]

    def test_the_rebuild_keeps_the_unique_constraint(self, con):
        """CREATE TABLE AS SELECT would drop it and re-break the dedup."""
        rec = RawRecord(record_id="g1", payload={"a": 1}, iso3="PHL",
                        ym="2024-03", hazard="FL")
        store_raw_records(con, "gdacs", [rec])
        retention.compact_raw_cache(con, "gdacs", apply=True)
        assert store_raw_records(con, "gdacs", [rec])["inserted"] == 0
        assert con.execute("SELECT COUNT(*) FROM haz_raw_gdacs").fetchone()[0] == 1


class TestDryRunIsTheDefault:
    def test_apply_false_writes_nothing(self, con):
        for stamp in ("2026-08-05 20:30:00", "2026-08-06 20:30:00"):
            _revision(con, "emdat", "e1", {"v": stamp}, iso3="PHL",
                      ym="2024-03", hazard="FL", stamp=stamp)
        before = con.execute("SELECT COUNT(*) FROM haz_raw_emdat").fetchone()[0]

        plan = retention.compact_raw_cache(con, "emdat")

        assert plan["applied"] is False
        assert plan["rows_before"] == before
        assert plan["rows_after"] == 1
        assert plan["removed"] == before - 1
        assert con.execute("SELECT COUNT(*) FROM haz_raw_emdat").fetchone()[0] == before

    def test_the_plan_reports_distinct_cells_for_eyeballing(self, con):
        store_raw_records(con, "emdat", [RawRecord(
            record_id="e1", payload={"v": 1}, iso3="PHL", ym="2024-03", hazard="FL")])
        plan = retention.compact_raw_cache(con, "emdat")
        assert plan["distinct_cells"] == plan["rows_after"] == 1


class TestTheAllowlistIsStructural:
    """Losing haz_backcast_progress would make the backcast re-walk history."""

    @pytest.mark.parametrize("table", [
        "haz_backcast_progress", "haz_resolutions", "haz_revisions",
        "haz_triggers", "haz_impact_candidates", "haz_doc_extractions",
        "haz_base_rates_occurrence", "questions", "facts_resolved",
    ])
    def test_non_raw_tables_are_refused(self, table):
        with pytest.raises(ValueError):
            retention._assert_compactable(table)

    def test_population_is_refused_it_is_not_a_revision_cache(self):
        assert "population" not in retention.COMPACTABLE_SOURCES
        with pytest.raises(ValueError):
            retention._assert_compactable("population")

    def test_compact_all_leaves_the_machines_own_tables_alone(self, con):
        con.execute(
            "INSERT INTO haz_backcast_progress (hazard, ym, status) "
            "VALUES ('FL', '2020-03', 'ok')"
        )
        con.execute(
            "INSERT INTO haz_resolutions "
            "(iso3, hazard, year, month, status, provenance_json, rule_fired) "
            "VALUES ('PHL','FL',2024,3,'RESOLVED_ZERO','{}','z')"
        )
        retention.compact_all(con, apply=True)
        assert con.execute(
            "SELECT COUNT(*) FROM haz_backcast_progress").fetchone()[0] == 1
        assert con.execute(
            "SELECT COUNT(*) FROM haz_resolutions").fetchone()[0] == 1


class TestResilience:
    def test_an_absent_table_is_reported_not_raised(self, con):
        con.execute("DROP TABLE IF EXISTS haz_raw_dfo")
        result = retention.compact_raw_cache(con, "dfo", apply=True)
        assert result["present"] is False

    def test_compact_all_covers_every_compactable_source(self, con):
        results = retention.compact_all(con)
        assert set(results) == set(retention.COMPACTABLE_SOURCES)

    def test_summarize_says_a_copy_is_what_reclaims_space(self, con):
        text = "\n".join(retention.summarize(retention.compact_all(con)))
        assert "TOTAL" in text
        assert "never truncates" in text
