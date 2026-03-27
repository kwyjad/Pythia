# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for FEWS NET IPC Phase 3+ resolution in compute_resolutions."""

from __future__ import annotations

from datetime import datetime

import duckdb
import pytest

from pythia.tools.compute_resolutions import (
    _data_freshness_cutoff,
    _resolve_value,
    _should_default_to_zero,
    _try_phase3plus,
)


def _create_facts_resolved(conn) -> None:
    """Create a minimal facts_resolved table matching the Resolver schema."""
    conn.execute(
        """
        CREATE TABLE facts_resolved (
            iso3 TEXT,
            hazard_code TEXT,
            metric TEXT,
            ym TEXT,
            value DOUBLE,
            created_at TIMESTAMP
        )
        """
    )


def _insert_fact(
    conn,
    iso3: str = "ETH",
    hazard_code: str = "DR",
    metric: str = "phase3plus_in_need",
    ym: str = "2026-01",
    value: float = 12500000.0,
    created_at: str = "2026-02-15 10:00:00",
) -> None:
    """Insert a single row into facts_resolved."""
    conn.execute(
        """
        INSERT INTO facts_resolved (iso3, hazard_code, metric, ym, value, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [iso3, hazard_code, metric, ym, value, created_at],
    )


# ---- Test 1: PHASE3PLUS_IN_NEED resolves from facts_resolved ----


def test_fewsnet_ipc_resolves_value():
    """_try_phase3plus returns (value, created_at) when matching data exists."""
    conn = duckdb.connect(":memory:")
    _create_facts_resolved(conn)
    _insert_fact(conn, iso3="ETH", ym="2026-01", value=12500000.0,
                 created_at="2026-02-15 10:00:00")

    result = _try_phase3plus(conn, "ETH", "DR", "2026-01")

    assert result is not None
    value, source_ts = result
    assert value == 12500000.0
    assert source_ts is not None
    assert "2026-02-15" in source_ts
    conn.close()


def test_fewsnet_ipc_returns_latest_by_created_at():
    """When multiple rows exist, _try_fewsnet_ipc returns the most recent."""
    conn = duckdb.connect(":memory:")
    _create_facts_resolved(conn)
    _insert_fact(conn, iso3="SOM", ym="2026-01", value=3000000.0,
                 created_at="2026-02-01 08:00:00")
    _insert_fact(conn, iso3="SOM", ym="2026-01", value=3200000.0,
                 created_at="2026-02-10 12:00:00")

    result = _try_phase3plus(conn, "SOM", "DR", "2026-01")

    assert result is not None
    value, _ = result
    assert value == 3200000.0
    conn.close()


# ---- Test 2: Returns None when no data ----


def test_fewsnet_ipc_returns_none_when_no_data():
    """_try_phase3plus returns None (NOT zero) when no matching row exists."""
    conn = duckdb.connect(":memory:")
    _create_facts_resolved(conn)
    # Table exists but has no rows for this country/month.

    result = _try_phase3plus(conn, "ETH", "DR", "2026-01")

    assert result is None
    conn.close()


def test_fewsnet_ipc_returns_none_when_table_missing():
    """_try_phase3plus returns None when facts_resolved table does not exist."""
    conn = duckdb.connect(":memory:")

    result = _try_phase3plus(conn, "ETH", "DR", "2026-01")

    assert result is None
    conn.close()


# ---- Test 3: phase3plus_projection rows are NOT used ----


def test_fewsnet_ipc_ignores_projection_rows():
    """Only phase3plus_in_need is matched, NOT phase3plus_projection."""
    conn = duckdb.connect(":memory:")
    _create_facts_resolved(conn)
    # Insert ONLY projection rows.
    _insert_fact(conn, iso3="ETH", metric="phase3plus_projection",
                 ym="2026-01", value=15000000.0)

    result = _try_phase3plus(conn, "ETH", "DR", "2026-01")

    assert result is None
    conn.close()


def test_fewsnet_ipc_selects_in_need_over_projection():
    """When both metrics exist, only phase3plus_in_need is returned."""
    conn = duckdb.connect(":memory:")
    _create_facts_resolved(conn)
    _insert_fact(conn, iso3="ETH", metric="phase3plus_in_need",
                 ym="2026-01", value=10000000.0,
                 created_at="2026-02-10 10:00:00")
    _insert_fact(conn, iso3="ETH", metric="phase3plus_projection",
                 ym="2026-01", value=15000000.0,
                 created_at="2026-02-10 10:00:00")

    result = _try_phase3plus(conn, "ETH", "DR", "2026-01")

    assert result is not None
    value, _ = result
    assert value == 10000000.0
    conn.close()


# ---- Test 4: Hazard guard ----


def test_fewsnet_ipc_returns_none_for_non_dr_hazard():
    """_try_phase3plus returns None for hazards other than DR."""
    conn = duckdb.connect(":memory:")
    _create_facts_resolved(conn)
    _insert_fact(conn, iso3="ETH", hazard_code="DR", ym="2026-01",
                 value=12500000.0)

    for hazard in ("FL", "TC", "ACE", "ACO", "HW", "DI", "CU"):
        result = _try_phase3plus(conn, "ETH", hazard, "2026-01")
        assert result is None, f"Expected None for hazard={hazard}, got {result}"

    conn.close()


# ---- Test 5: _should_default_to_zero is False for PHASE3PLUS_IN_NEED ----


def test_should_default_to_zero_false_for_phase3plus():
    """PHASE3PLUS_IN_NEED must NOT default to zero — absence = unknown."""
    assert _should_default_to_zero("PHASE3PLUS_IN_NEED", "DR") is False


def test_should_default_to_zero_false_for_phase3plus_all_hazards():
    """PHASE3PLUS_IN_NEED never defaults to zero regardless of hazard."""
    for hazard in ("DR", "FL", "TC", "ACE", "ACO", "HW", "DI"):
        assert _should_default_to_zero("PHASE3PLUS_IN_NEED", hazard) is False, (
            f"PHASE3PLUS_IN_NEED should not default to zero for hazard={hazard}"
        )


def test_should_default_to_zero_true_for_fatalities_ace():
    """Cross-check: FATALITIES+ACE does default to zero (sanity check)."""
    assert _should_default_to_zero("FATALITIES", "ACE") is True


def test_should_default_to_zero_true_for_event_occurrence():
    """Cross-check: EVENT_OCCURRENCE+FL defaults to zero (sanity check)."""
    assert _should_default_to_zero("EVENT_OCCURRENCE", "FL") is True


# ---- Test 6: _data_freshness_cutoff handles PHASE3PLUS_IN_NEED ----


def test_data_freshness_cutoff_phase3plus():
    """_data_freshness_cutoff queries facts_resolved for phase3plus_in_need."""
    conn = duckdb.connect(":memory:")
    _create_facts_resolved(conn)
    _insert_fact(conn, iso3="ETH", metric="phase3plus_in_need",
                 ym="2026-01", value=10000000.0)
    _insert_fact(conn, iso3="SOM", metric="phase3plus_in_need",
                 ym="2025-12", value=5000000.0)

    cutoff = _data_freshness_cutoff(conn, "PHASE3PLUS_IN_NEED")

    assert cutoff == "2026-01"
    conn.close()


def test_data_freshness_cutoff_phase3plus_empty():
    """_data_freshness_cutoff returns None when no phase3plus data exists."""
    conn = duckdb.connect(":memory:")
    _create_facts_resolved(conn)
    # Table exists but has no phase3plus_in_need rows.
    _insert_fact(conn, iso3="ETH", metric="fatalities", ym="2026-01",
                 value=100.0)

    cutoff = _data_freshness_cutoff(conn, "PHASE3PLUS_IN_NEED")

    assert cutoff is None
    conn.close()


def test_data_freshness_cutoff_phase3plus_ignores_projection():
    """_data_freshness_cutoff for PHASE3PLUS_IN_NEED ignores projection rows."""
    conn = duckdb.connect(":memory:")
    _create_facts_resolved(conn)
    _insert_fact(conn, iso3="ETH", metric="phase3plus_projection",
                 ym="2026-03", value=15000000.0)
    _insert_fact(conn, iso3="ETH", metric="phase3plus_in_need",
                 ym="2026-01", value=10000000.0)

    cutoff = _data_freshness_cutoff(conn, "PHASE3PLUS_IN_NEED")

    # Should return 2026-01 (in_need max), not 2026-03 (projection).
    assert cutoff == "2026-01"
    conn.close()


# ---- Integration: _resolve_value dispatches correctly ----


def test_resolve_value_dispatches_to_fewsnet():
    """_resolve_value routes PHASE3PLUS_IN_NEED to _try_phase3plus."""
    conn = duckdb.connect(":memory:")
    _create_facts_resolved(conn)
    _insert_fact(conn, iso3="ETH", hazard_code="DR",
                 metric="phase3plus_in_need", ym="2026-01",
                 value=12500000.0, created_at="2026-02-15 10:00:00")

    result = _resolve_value(conn, "ETH", "DR", "2026-01", "PHASE3PLUS_IN_NEED")

    assert result is not None
    value, source_ts = result
    assert value == 12500000.0
    conn.close()


def test_resolve_value_phase3plus_returns_none_no_data():
    """_resolve_value returns None for PHASE3PLUS_IN_NEED when no data."""
    conn = duckdb.connect(":memory:")
    _create_facts_resolved(conn)

    result = _resolve_value(conn, "ETH", "DR", "2026-01", "PHASE3PLUS_IN_NEED")

    assert result is None
    conn.close()
