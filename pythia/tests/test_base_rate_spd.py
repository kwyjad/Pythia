# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for pythia.tools.base_rate_spd — the anchor-distribution builder."""

from __future__ import annotations

import duckdb
import pytest

from pythia.buckets import n_buckets_for
from pythia.tools.base_rate_spd import (
    NO_BASE_RATE_SOURCE,
    base_rate_spd,
    forecast_months,
)


@pytest.fixture()
def con():
    c = duckdb.connect(":memory:")
    c.execute(
        """
        CREATE TABLE facts_resolved (
            ym TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT, value DOUBLE
        )
        """
    )
    c.execute(
        """
        CREATE TABLE facts_deltas (
            ym TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT,
            series_semantics TEXT, source_id TEXT, value_new DOUBLE
        )
        """
    )
    c.execute(
        "CREATE TABLE acled_monthly_fatalities (iso3 TEXT, month TEXT, fatalities DOUBLE)"
    )
    yield c
    c.close()


def _assert_valid_spd(probs, metric):
    assert len(probs) == n_buckets_for(metric)
    assert abs(sum(probs) - 1.0) < 1e-9
    assert all(p > 0 for p in probs), "smoothing must keep every bucket non-zero"


class TestNoAnchorPairs:
    def test_cu_pa_has_no_base_rate(self, con):
        probs, source, detail = base_rate_spd(con, "SOM", "CU", "PA", "2026-08")
        assert probs == []
        assert source == NO_BASE_RATE_SOURCE
        assert "reason" in detail

    def test_di_pa_has_no_base_rate(self, con):
        probs, source, _ = base_rate_spd(con, "SOM", "DI", "PA", "2026-08")
        assert probs == []
        assert source == NO_BASE_RATE_SOURCE

    def test_unknown_pair_has_no_base_rate(self, con):
        probs, source, _ = base_rate_spd(con, "SOM", "EC", "FATALITIES", "2026-08")
        assert probs == []
        assert source == NO_BASE_RATE_SOURCE

    def test_empty_tables_yield_no_anchor_not_a_guess(self, con):
        for hz, metric in [
            ("ACE", "FATALITIES"), ("ACE", "PA"),
            ("DR", "PHASE3PLUS_IN_NEED"), ("FL", "PA"),
            ("FL", "EVENT_OCCURRENCE"),
        ]:
            probs, source, _ = base_rate_spd(con, "SOM", hz, metric, "2026-08")
            assert probs == [], f"{hz}/{metric} invented an anchor from empty tables"
            assert source == NO_BASE_RATE_SOURCE


class TestConflictFatalities:
    def test_empirical_distribution_and_window(self, con):
        # 6 complete months before the window; one later month must be ignored.
        for ym, fat in [
            ("2026-02", 0), ("2026-03", 2), ("2026-04", 12),
            ("2026-05", 40), ("2026-06", 250), ("2026-07", 800),
        ]:
            con.execute(
                "INSERT INTO acled_monthly_fatalities VALUES ('SOM', ?, ?)", [ym, fat]
            )
        # Leakage guard: this is inside the forecast window.
        con.execute("INSERT INTO acled_monthly_fatalities VALUES ('SOM', '2026-08', 99999)")

        probs, source, detail = base_rate_spd(con, "SOM", "ACE", "FATALITIES", "2026-08")
        _assert_valid_spd(probs, "FATALITIES")
        assert source.startswith("acled_monthly_fatalities")
        assert detail["score_family"] == "spd"
        assert detail["n_months_used"] == 6
        assert 99999 not in detail["values"]
        # The observed values hit buckets 0..5 but never the >=1000 bucket, so
        # the top bucket must carry only smoothing mass — strictly less than
        # any observed bucket.
        assert probs[-1] < probs[0]


class TestConflictDisplacement:
    @staticmethod
    def _series(con):
        con.execute("ALTER TABLE facts_resolved ADD COLUMN series_semantics TEXT")
        con.execute("ALTER TABLE facts_resolved ADD COLUMN publisher TEXT")

    @staticmethod
    def _row(con, ym, iso3, hazard, value, publisher="IDMC"):
        con.execute(
            "INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value, "
            "series_semantics, publisher) VALUES (?, ?, ?, 'new_displacements', ?, 'new', ?)",
            [ym, iso3, hazard, value, publisher],
        )

    def test_ace_pa_uses_idmc_conflict_series(self, con):
        self._series(con)
        for i, val in enumerate([0, 5000, 20000, 0, 12000, 60000]):
            self._row(con, f"2026-0{i + 1}", "SOM", "ACE", val)
        probs, source, detail = base_rate_spd(con, "SOM", "ACE", "PA", "2026-08")
        _assert_valid_spd(probs, "PA")
        assert source.startswith("idmc_conflict")
        assert detail["score_family"] == "spd"

    def test_all_cause_idu_rows_never_enter_the_anchor(self, con):
        """Oct 2026: China's 7.3 million typhoon evacuees were the ACE/PA
        anchor, because every IDMC row was hazard IDU whatever its cause."""
        self._series(con)
        self._row(con, "2026-07", "CHN", "IDU", 7_305_385)
        self._row(con, "2026-07", "SOM", "ACE", 4000)
        probs, source, detail = base_rate_spd(con, "CHN", "ACE", "PA", "2026-08")
        assert probs == [] and source == "NONE"

    def test_a_quiet_month_is_zero_only_when_settled_regular_and_bracketed(self, con):
        """Oct 2026: a month IDMC had not reported YET was zero-defaulted
        (60 of 64 zeros on the 5 October release); the full rule is pinned in
        test_conflict_displacement_settle.py."""
        from datetime import date

        from pythia.tools.base_rate_spd import conflict_displacement_value

        self._series(con)
        for year in (2024, 2025):
            for m in range(1, 13):
                if (year, m) != (2025, 7):
                    self._row(con, f"{year}-{m:02d}", "SOM", "ACE", 4000)
        later = date(2026, 6, 1)
        assert conflict_displacement_value(con, "SOM", "2025-06", today=later) == (
            4000.0, "facts_resolved:IDMC:conflict_new_displacements"
        )
        # A regular reporter's missing month, bracketed by a later report.
        assert conflict_displacement_value(con, "SOM", "2025-07", today=later) == (0.0, "zero_default")
        # A trailing month (no later report) is unknown, never zero.
        assert conflict_displacement_value(con, "SOM", "2026-01", today=later) is None
        # A month not yet settled is unknown, reported or not.
        assert conflict_displacement_value(con, "SOM", "2025-12", today=date(2026, 1, 15)) is None
        # A country IDMC never reported for is outside its sight.
        assert conflict_displacement_value(con, "ISL", "2025-07", today=later) is None


class TestPhase3History:
    def test_stock_series_bucketised(self, con):
        for i in range(1, 7):
            con.execute(
                "INSERT INTO facts_resolved VALUES (?, 'SOM', 'DR', 'phase3plus_in_need', ?)",
                [f"2026-0{i}", 3_500_000.0],
            )
        probs, source, _ = base_rate_spd(con, "SOM", "DR", "PHASE3PLUS_IN_NEED", "2026-08")
        _assert_valid_spd(probs, "PHASE3PLUS_IN_NEED")
        assert source.startswith("facts_resolved:phase3plus_in_need")
        # All observed mass in the 1M-<5M bucket (index 3).
        assert probs[3] == max(probs)


class TestBinary:
    def test_binary_form_and_monthly_probs(self, con):
        # 3 years of history: events every June, never in December.
        for year in (2023, 2024, 2025):
            for month in range(1, 13):
                occurred = 1.0 if month == 6 else 0.0
                con.execute(
                    "INSERT INTO facts_resolved VALUES (?, 'PHL', 'TC', 'event_occurrence', ?)",
                    [f"{year:04d}-{month:02d}", occurred],
                )
        probs, source, detail = base_rate_spd(con, "PHL", "TC", "EVENT_OCCURRENCE", "2026-04")
        assert len(probs) == 2
        assert abs(sum(probs) - 1.0) < 1e-9
        assert detail["score_family"] == "binary"
        assert source == "facts_resolved:event_occurrence"
        by_month = detail["probs_by_month"]
        assert set(by_month) == set(forecast_months("2026-04"))
        # June (inside the window) is high; the other window months are low.
        assert by_month["2026-06"][0] > 0.5
        assert by_month["2026-04"][0] < 0.5

    def test_leakage_window_months_excluded(self, con):
        # Only history AT/AFTER as_of: must produce NO anchor, not a leaked one.
        con.execute(
            "INSERT INTO facts_resolved VALUES ('2026-08', 'PHL', 'TC', 'event_occurrence', 1.0)"
        )
        probs, source, _ = base_rate_spd(con, "PHL", "TC", "EVENT_OCCURRENCE", "2026-08")
        assert probs == []
        assert source == NO_BASE_RATE_SOURCE

    def test_non_gdacs_hazard_refused(self, con):
        probs, source, _ = base_rate_spd(con, "SOM", "ACE", "EVENT_OCCURRENCE", "2026-08")
        assert probs == []
        assert source == NO_BASE_RATE_SOURCE


class TestSeasonalPa:
    def _seed(self, con):
        # GDACS occurrence: events in ~half the observed Julys/Augusts.
        for year in (2022, 2023, 2024, 2025):
            for month in range(1, 13):
                occurred = 1.0 if (month in (7, 8) and year % 2 == 0) else 0.0
                con.execute(
                    "INSERT INTO facts_resolved VALUES (?, 'BGD', 'FL', 'event_occurrence', ?)",
                    [f"{year:04d}-{month:02d}", occurred],
                )
        # Reported impact for the event months.
        con.execute(
            "INSERT INTO facts_resolved VALUES ('2022-07', 'BGD', 'FL', 'affected', 120000)"
        )
        con.execute(
            "INSERT INTO facts_resolved VALUES ('2024-08', 'BGD', 'FL', 'affected', 30000)"
        )

    def test_occurrence_times_severity_mixture(self, con):
        self._seed(con)
        probs, source, detail = base_rate_spd(con, "BGD", "FL", "PA", "2026-06")
        _assert_valid_spd(probs, "PA")
        assert "gdacs_occurrence" in source
        assert detail["occurrence_method"] == "gdacs_seasonal_event_rate"
        # Most months are quiet: the "0" bucket must dominate.
        assert probs[0] == max(probs)
        # Observed severities (30k, 120k) land in buckets 3 and 4; those must
        # outweigh the unobserved non-zero buckets.
        assert probs[2] > probs[1]
        assert probs[3] > probs[1]

    def test_in_need_rows_never_enter_severity(self, con):
        self._seed(con)
        # GDACS modelled exposure must not enter the PA severity distribution
        # (the montandon_assessment class of bug).
        con.execute(
            "INSERT INTO facts_resolved VALUES ('2024-07', 'BGD', 'FL', 'in_need', 8400000)"
        )
        probs, _, _ = base_rate_spd(con, "BGD", "FL", "PA", "2026-06")
        _assert_valid_spd(probs, "PA")
        # 8.4M would land in the top bucket; only smoothing mass may be there.
        assert probs[-1] < probs[3]


class TestInputHandling:
    def test_bad_as_of_is_reported_not_raised(self, con):
        probs, source, detail = base_rate_spd(con, "SOM", "ACE", "FATALITIES", "not-a-month")
        assert probs == []
        assert source == NO_BASE_RATE_SOURCE
        assert "as_of" in detail["reason"]

    def test_date_and_full_date_string_accepted(self, con):
        con.execute("INSERT INTO acled_monthly_fatalities VALUES ('SOM', '2026-07', 10)")
        p1, _, _ = base_rate_spd(con, "SOM", "ACE", "FATALITIES", "2026-08-01")
        from datetime import date

        p2, _, _ = base_rate_spd(con, "SOM", "ACE", "FATALITIES", date(2026, 8, 1))
        assert p1 == p2
        assert p1
