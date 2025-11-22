from __future__ import annotations

from typing import Any

import pytest

try:  # pragma: no cover - optional dependency resolution
    import duckdb
except ModuleNotFoundError:  # pragma: no cover - handled via skip marker
    duckdb = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(duckdb is None, reason="duckdb not installed")

from resolver.snapshot import PaTrendPoint, get_pa_trend, render_pa_trend_markdown


def _setup_facts_snapshot() -> Any:
    assert duckdb is not None  # for type checkers
    con = duckdb.connect(database=":memory:")
    con.execute(
        """
        CREATE TABLE facts_snapshot (
            ym TEXT,
            iso3 TEXT,
            hazard_code TEXT,
            metric TEXT,
            series_semantics TEXT,
            value DOUBLE,
            source TEXT,
            as_of_date DATE,
            provenance_table TEXT,
            run_id TEXT
        );
        """
    )
    return con


def test_get_pa_trend_aggregates_emdat_dtm_idmc() -> None:
    con = _setup_facts_snapshot()

    # EM-DAT affected: hazard 'drought' in SDN
    con.execute(
        """
        INSERT INTO facts_snapshot VALUES
            ('2025-09', 'SDN', 'drought', 'affected', 'new', 1000.0, 'emdat', DATE '2025-09-30', 'facts_deltas', 'run1'),
            ('2025-10', 'SDN', 'drought', 'affected', 'new', 1500.0, 'emdat', DATE '2025-10-31', 'facts_deltas', 'run1');
        """
    )

    # DTM stock IDPs: hazard 'displacement' in SDN
    con.execute(
        """
        INSERT INTO facts_snapshot VALUES
            ('2025-09', 'SDN', 'displacement', 'idp_displacement_stock_dtm', 'stock', 2000.0, 'IOM DTM', DATE '2025-09-30', 'facts_resolved', 'run1'),
            ('2025-10', 'SDN', 'displacement', 'idp_displacement_stock_dtm', 'stock', 2500.0, 'IOM DTM', DATE '2025-10-31', 'facts_resolved', 'run1');
        """
    )

    # IDMC new displacements: hazard 'displacement' in SDN
    con.execute(
        """
        INSERT INTO facts_snapshot VALUES
            ('2025-10', 'SDN', 'displacement', 'new_displacements', 'new', 300.0, 'IDMC', DATE '2025-10-31', 'facts_deltas', 'run1');
        """
    )

    # For drought, we expect EM-DAT only
    drought_trend = get_pa_trend(con, iso3="SDN", hazard_code="drought", months=36)
    assert len(drought_trend) == 2
    assert [p.ym for p in drought_trend] == ["2025-09", "2025-10"]
    assert [p.pa_value for p in drought_trend] == [1000.0, 1500.0]

    # For displacement, we expect DTM + IDMC combined per month
    disp_trend = get_pa_trend(con, iso3="SDN", hazard_code="displacement", months=36)
    assert len(disp_trend) == 2
    # 2025-09: only DTM stock
    assert disp_trend[0].ym == "2025-09"
    assert disp_trend[0].pa_value == 2000.0
    # 2025-10: DTM stock + IDMC new = 2500 + 300 = 2800
    assert disp_trend[1].ym == "2025-10"
    assert disp_trend[1].pa_value == 2800.0


def test_render_pa_trend_markdown_basic() -> None:
    trend = [
        PaTrendPoint(ym="2025-09", pa_value=1000.0),
        PaTrendPoint(ym="2025-10", pa_value=1500.0),
    ]
    md = render_pa_trend_markdown(trend, iso3="SDN", hazard_code="drought")

    # Basic sanity checks: title and table structure
    assert "SDN" in md
    assert "drought" in md
    assert "| Month | People affected |" in md
    assert "| 2025-09 | 1000 |" in md
    assert "| 2025-10 | 1500 |" in md

    # Empty trend should be handled gracefully
    md_empty = render_pa_trend_markdown([], iso3="SDN", hazard_code="drought")
    assert "No PA history available" in md_empty
