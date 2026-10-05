# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.
"""A monthly flow is not differenced into a delta (Oct 2026).

``_derive_deltas`` treated every ``facts_resolved`` series as a stock, so
IDMC's monthly new displacements became "this month minus last month": the
1 October 2026 Afghanistan prompt read -1,791 new displacements for August.
"""

from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")

from resolver.db.duckdb_io import init_schema
from resolver.tools.load_and_derive import (
    PeriodMonths,
    _derive_deltas,
    repair_flow_deltas,
)


def _seed(con):
    init_schema(con)
    con.execute(
        """
        INSERT INTO facts_resolved
            (ym, iso3, hazard_code, metric, series_semantics, value, unit,
             as_of_date, source_id, event_id)
        VALUES
            ('2024-01', 'AFG', 'IDU', 'new_displacements', 'new', 172443, 'persons', '2024-01-31', 'idmc', 'a'),
            ('2024-02', 'AFG', 'IDU', 'new_displacements', 'new', 10758, 'persons', '2024-02-29', 'idmc', 'b'),
            ('2024-03', 'AFG', 'IDU', 'new_displacements', 'new', 90, 'persons', '2024-03-31', 'idmc', 'c'),
            ('2024-01', 'SDN', 'FL', 'affected', 'stock', 5000, 'persons', '2024-01-31', 'ifrc_go', 'd'),
            ('2024-02', 'SDN', 'FL', 'affected', 'stock', 3000, 'persons', '2024-02-29', 'ifrc_go', 'e')
        """
    )


def _deltas(con, iso3):
    return con.execute(
        "SELECT ym, value_new FROM facts_deltas WHERE iso3 = ? ORDER BY ym", [iso3]
    ).fetchall()


def test_a_flow_keeps_its_own_value():
    con = duckdb.connect(":memory:")
    _seed(con)
    _derive_deltas(con, PeriodMonths.from_label("2024Q1"))
    assert _deltas(con, "AFG") == [("2024-01", 172443.0), ("2024-02", 10758.0), ("2024-03", 90.0)]


def test_a_stock_is_still_differenced():
    con = duckdb.connect(":memory:")
    _seed(con)
    _derive_deltas(con, PeriodMonths.from_label("2024Q1"))
    assert _deltas(con, "SDN") == [("2024-01", 5000.0), ("2024-02", -2000.0)]


def test_repair_rewrites_differenced_flow_rows_once():
    con = duckdb.connect(":memory:")
    _seed(con)
    # What the old writer left behind.
    con.execute(
        """
        INSERT INTO facts_deltas
            (ym, iso3, hazard_code, metric, value_new, value_stock,
             series_semantics, as_of, source_id)
        VALUES
            ('2024-02', 'AFG', 'IDU', 'new_displacements', -161685, 10758, 'new', '2024-02-29', 'idmc'),
            ('2024-03', 'AFG', 'IDU', 'new_displacements', -10668, 90, 'new', '2024-03-31', 'idmc'),
            ('2024-02', 'SDN', 'FL', 'affected', -2000, 3000, 'new', '2024-02-29', 'ifrc_go')
        """
    )
    first = repair_flow_deltas(con)
    second = repair_flow_deltas(con)
    assert first == {"repaired": 2, "negative_repaired": 2}
    assert second == {"repaired": 0, "negative_repaired": 0}
    assert _deltas(con, "AFG") == [("2024-02", 10758.0), ("2024-03", 90.0)]
    # A stock's delta is a difference by design and is left alone.
    assert _deltas(con, "SDN") == [("2024-02", -2000.0)]


def test_all_cause_idmc_rows_are_purged_and_conflict_rows_kept():
    """Oct 2026: every IDMC row was hazard IDU whatever its cause, and was
    read as conflict displacement. The purge removes them from both tables
    and leaves the conflict series (hazard ACE) and other sources alone."""
    from resolver.tools.load_and_derive import purge_all_cause_idmc_rows

    con = duckdb.connect(":memory:")
    _seed(con)
    con.execute(
        """
        INSERT INTO facts_resolved
            (ym, iso3, hazard_code, metric, series_semantics, value, unit,
             as_of_date, source_id, event_id)
        VALUES ('2024-02', 'AFG', 'ACE', 'new_displacements', 'new', 3000,
                'persons', '2024-02-29', 'idmc', 'x')
        """
    )
    _derive_deltas(con, PeriodMonths.from_label("2024Q1"))
    counts = purge_all_cause_idmc_rows(con)
    assert counts == {"facts_resolved": 3, "facts_deltas": 3}
    left = con.execute(
        "SELECT hazard_code, COUNT(*) FROM facts_resolved GROUP BY 1 ORDER BY 1"
    ).fetchall()
    assert left == [("ACE", 1), ("FL", 2)]
    assert con.execute(
        "SELECT COUNT(*) FROM facts_deltas WHERE hazard_code = 'IDU'"
    ).fetchone()[0] == 0
    # Idempotent.
    assert purge_all_cause_idmc_rows(con) == {"facts_resolved": 0, "facts_deltas": 0}
