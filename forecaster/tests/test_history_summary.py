# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")

import forecaster.cli as cli  # type: ignore


def test_build_history_summary_ifrc_uses_facts_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    """_build_history_summary should read IFRC PA from the facts_resolved table."""
    # Prepare an in-memory DuckDB with a minimal facts_resolved table.
    con = duckdb.connect(":memory:")

    con.execute(
        """
        CREATE TABLE facts_resolved (
            iso3 TEXT,
            hazard_code TEXT,
            ym DATE,
            metric TEXT,
            value DOUBLE,
            source_id TEXT
        )
        """
    )
    # Two months of synthetic PA data for ETH/FL.
    con.execute(
        """
        INSERT INTO facts_resolved (iso3, hazard_code, ym, metric, value, source_id) VALUES
            ('ETH', 'FL', DATE '2024-01-01', 'affected', 10.0, 'ifrc'),
            ('ETH', 'FL', DATE '2024-02-01', 'affected', 20.0, 'ifrc')
        """
    )

    # Monkeypatch cli.connect to use our in-memory DB, ignoring read_only.
    def fake_connect(read_only: bool = False):
        return con

    monkeypatch.setattr(cli, "connect", fake_connect)

    summary = cli._build_history_summary("ETH", "FL", "PA")

    assert summary["type"] == "seasonal_profile"
    assert summary["source"] == "IFRC"
    assert summary["years_of_data"] == 1
    # January and February should have observations
    assert summary["months"][1]["n_observations"] == 1
    assert summary["months"][1]["max"] == pytest.approx(10.0)
    assert summary["months"][2]["n_observations"] == 1
    assert summary["months"][2]["max"] == pytest.approx(20.0)
    # Months without data should be zero-filled
    assert summary["months"][3]["n_observations"] == 0


def test_build_history_summary_idmc_conflict_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    """IDMC history should come from conflict flow metrics with hazard filter."""

    con = duckdb.connect(":memory:")

    # Need both tables since _build_conflict_base_rate queries both
    con.execute(
        """
        CREATE TABLE acled_monthly_fatalities (
            iso3 TEXT, month DATE, fatalities BIGINT,
            source TEXT, updated_at TIMESTAMP
        )
        """
    )

    con.execute(
        """
        CREATE TABLE facts_deltas (
            ym DATE,
            iso3 TEXT,
            hazard_code TEXT,
            metric TEXT,
            value_new DOUBLE,
            series_semantics TEXT,
            source_id TEXT
        )
        """
    )

    con.execute(
        """
        INSERT INTO facts_deltas (ym, iso3, hazard_code, metric, value_new, series_semantics, source_id) VALUES
            (DATE '2024-01-01', 'ETH', 'ACE', 'new_displacements', 1000.0, 'new', 'idmc'),
            (DATE '2024-02-01', 'ETH', 'ACE', 'new_displacements', 2000.0, 'new', 'idmc'),
            (DATE '2024-03-01', 'ETH', 'ACE', 'new_displacements', 2000.0, 'new', 'idmc'),
            (DATE '2024-03-01', 'ETH', 'ACE', 'new_displacements', 500.0, 'new', 'idmc')
        """
    )

    def fake_connect(read_only: bool = False):
        return con

    monkeypatch.setattr(cli, "connect", fake_connect)

    summary = cli._build_history_summary("ETH", "ACE", "PA")

    assert summary["type"] == "conflict_trajectory"
    # Displacements should have data
    disp = summary["displacements"]
    assert disp["source"] == "IDMC"
    assert disp["last_month"] is not None
    assert len(disp["last_6m"]) == 3
    assert disp["last_6m"][-1]["value"] == pytest.approx(2500.0)


def test_build_history_summary_di_has_no_base_rate() -> None:
    """DI should explicitly return no Resolver base rate."""

    summary = cli._build_history_summary("ETH", "DI", "PA")

    assert summary["type"] == "no_base_rate"
    assert summary["source"] == "NONE"
    assert "no resolver base rate" in summary["note"].lower()


# ---------------------------------------------------------------------------
# conflict_trajectory rendering (the ACE/PA raw-JSON-dump regression)
# ---------------------------------------------------------------------------


_CONFLICT_SUMMARY = {
    "type": "conflict_trajectory",
    "fatalities": {
        "source": "ACLED",
        "last_month": {"ym": "2026-06", "value": 61},
        "trailing_3m_avg": 52,
        "prior_3m_avg": 21,
        "trend_pct": 147.6,
        "trend_direction": "escalating",
        "last_6m": [],
    },
    "displacements": {
        "source": "IDMC",
        "last_month": {"ym": "2026-06", "value": 9100},
        "trailing_3m_avg": 6100,
        "prior_3m_avg": 1833,
        "trend_pct": 232.8,
        "trend_direction": "escalating",
        "last_6m": [],
    },
}

_FORECAST_KEYS = ["2026-08", "2026-09", "2026-10", "2026-11", "2026-12", "2027-01"]


def _render(metric: str = "") -> str:
    from forecaster.history_loaders import _format_base_rate_for_prompt

    return _format_base_rate_for_prompt(
        _CONFLICT_SUMMARY, _FORECAST_KEYS, "ETH", "ACE", metric
    )


def test_ace_pa_renders_a_curated_block_not_a_json_dump(monkeypatch: pytest.MonkeyPatch) -> None:
    """The regression this guards: ACE/PA used to hit the legacy JSON branch.

    _build_history_summary routed ACE/PA to a loader whose dict has no "type",
    so _format_base_rate_for_prompt fell through to its ```json dump while
    every other hazard/metric got a curated base-rate block.
    """
    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE acled_monthly_fatalities (iso3 TEXT, month DATE, fatalities BIGINT)"
    )
    con.execute(
        """
        CREATE TABLE facts_deltas (
            ym DATE, iso3 TEXT, hazard_code TEXT, metric TEXT,
            value_new DOUBLE, series_semantics TEXT, source_id TEXT
        )
        """
    )
    con.execute(
        """
        INSERT INTO facts_deltas VALUES
            (DATE '2024-01-01', 'ETH', 'ACE', 'new_displacements', 1000.0, 'new', 'idmc'),
            (DATE '2024-02-01', 'ETH', 'ACE', 'new_displacements', 2000.0, 'new', 'idmc')
        """
    )
    monkeypatch.setattr(cli, "connect", lambda read_only=False: con)

    from forecaster.history_loaders import _format_base_rate_for_prompt

    summary = cli._build_history_summary("ETH", "ACE", "PA")
    rendered = _format_base_rate_for_prompt(summary, _FORECAST_KEYS, "ETH", "ACE", "PA")

    assert "```json" not in rendered
    assert rendered.startswith("BASE RATE: ")
    assert "Displacement (IDMC)" in rendered


def test_conflict_render_leads_with_the_questions_own_series() -> None:
    pa = _render("PA")
    fatalities = _render("FATALITIES")

    # PA resolves on displacement; FATALITIES on the ACLED series.
    assert pa.index("Displacement (IDMC)") < pa.index("Fatalities (ACLED)")
    assert "Displacement (IDMC) — THIS QUESTION'S SERIES" in pa
    assert "THIS QUESTION'S SERIES" not in pa.split("Fatalities (ACLED)")[1]

    assert fatalities.index("Fatalities (ACLED)") < fatalities.index("Displacement (IDMC)")
    assert "Fatalities (ACLED) — THIS QUESTION'S SERIES" in fatalities

    # Both series are always present — the non-anchor one is conflict context.
    for text in (pa, fatalities):
        assert "9,100 new displacements" in text
        assert "Last month (2026-06): 61" in text


def test_conflict_render_without_metric_keeps_the_legacy_shape() -> None:
    """Callers that pass no metric (e.g. older code paths) must be unchanged."""
    legacy = _render()

    assert legacy.index("Fatalities (ACLED)") < legacy.index("Displacement (IDMC)")
    assert "THIS QUESTION'S SERIES" not in legacy
