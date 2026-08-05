# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Phase-5 tests for the base rates computed from the backcast.

Two things these tests care about above all:

* **the denominator counts years ASSESSED**, not years in the calendar —
  counting an unassessed month as a quiet one manufactures a low base rate
  out of an ingestion gap; and
* **the provenance mix travels with the quantiles**, because a severity
  distribution made of displacement lower bounds is a different quantity
  from one made of EM-DAT assessments, and the numbers alone cannot say
  which it is.

Network-free: every input is seeded straight into the ``haz_*`` tables.
"""

from __future__ import annotations

import datetime as dt
import json

import duckdb
import pytest

from resolver.hazard_resolution import base_rates as br
from resolver.hazard_resolution.schema import ensure_haz_schema
from resolver.tests.hazard_resolution_utils import (
    make_rulebook,
    seed_resolution,
    seed_trigger,
)

TODAY = dt.date(2026, 8, 5)


@pytest.fixture()
def rulebook():
    return make_rulebook()


@pytest.fixture()
def con():
    con = duckdb.connect(":memory:")
    ensure_haz_schema(con)
    return con


# ---------------------------------------------------------------------------
# Windows and quantile arithmetic
# ---------------------------------------------------------------------------


def test_last_frozen_month_is_the_newest_month_past_its_deadline(rulebook):
    # freeze_days = 60, so on 5 Aug 2026 the newest frozen month is May 2026
    # (ends 31 May, freezes 30 July) — June freezes 29 August, too late.
    assert br.last_frozen_month(rulebook, today=TODAY) == "2026-05"
    assert br.last_frozen_month(rulebook, today=dt.date(2026, 8, 30)) == "2026-06"


def test_backcast_window_starts_at_the_rulebook_year(rulebook):
    assert br.backcast_window("TC", rulebook, today=TODAY) == ("2000-01", "2026-05")
    assert br.backcast_window("FL", rulebook, today=TODAY) == ("2010-01", "2026-05")
    assert br.backcast_window("DR", rulebook, today=TODAY) == ("2017-01", "2026-05")


def test_backcast_window_follows_the_rulebook_not_the_code(rulebook):
    moved = make_rulebook({"backcast": {"flood": 1995}})
    assert br.backcast_window("FL", moved, today=TODAY)[0] == "1995-01"


def test_months_back_window_counts_inclusively():
    assert br.months_back_window("2026-05", 12) == "2025-06"
    assert br.months_back_window("2026-01", 1) == "2026-01"
    assert br.months_back_window("2026-01", 13) == "2025-01"
    with pytest.raises(ValueError):
        br.months_back_window("2026-01", 0)


@pytest.mark.parametrize(
    "values, q, expected",
    [
        ([1, 2, 3, 4, 5], 0.5, 3.0),
        ([1, 2, 3, 4], 0.5, 2.5),
        ([10, 20], 0.9, 19.0),
        ([7], 0.1, 7.0),
        ([5, 1, 3], 0.0, 1.0),
        ([5, 1, 3], 1.0, 5.0),
    ],
)
def test_quantile_is_linear_interpolation(values, q, expected):
    assert br.quantile(list(values), q) == pytest.approx(expected)


def test_quantile_of_nothing_is_none():
    assert br.quantile([], 0.5) is None


# ---------------------------------------------------------------------------
# Occurrence
# ---------------------------------------------------------------------------


def test_occurrence_is_the_share_of_assessed_years_that_triggered(con, rulebook):
    # Six Septembers assessed, two of them triggered -> 2/6.
    for year in range(2018, 2024):
        seed_trigger(
            con, iso3="PHL", ym=f"{year}-09", hazard="FL",
            triggered=year in (2020, 2022),
        )

    br.compute_occurrence(con, rulebook, hazards=["FL"], today=TODAY)

    row = con.execute(
        """
        SELECT p_occurrence, n_years, source_window
        FROM haz_base_rates_occurrence
        WHERE iso3 = 'PHL' AND hazard = 'FL' AND calendar_month = 9
        """
    ).fetchone()
    assert row is not None
    assert row[0] == pytest.approx(2 / 6)
    assert row[1] == 6
    assert row[2] == "2010-01..2026-05"


def test_occurrence_denominator_excludes_years_never_assessed(con, rulebook):
    """A year with no trigger row was never looked at and must not count.

    This is the base-rate scale version of the coverage gate: an ingestion
    gap must never read as a run of quiet months.
    """

    # Ten calendar years in the window, but only four were assessed.
    for year in (2019, 2020, 2021, 2022):
        seed_trigger(con, iso3="MOZ", ym=f"{year}-01", hazard="FL", triggered=year == 2020)

    br.compute_occurrence(con, rulebook, hazards=["FL"], today=TODAY)

    p, n_years = con.execute(
        """
        SELECT p_occurrence, n_years FROM haz_base_rates_occurrence
        WHERE iso3 = 'MOZ' AND hazard = 'FL' AND calendar_month = 1
        """
    ).fetchone()
    assert n_years == 4
    assert p == pytest.approx(0.25)


def test_occurrence_refuses_to_publish_a_rate_from_too_few_years(con, rulebook):
    # min_years is 3 in the shipped rulebook; two observed Julys is not a
    # base rate, and a single triggered year would publish p = 1.0.
    for year in (2020, 2021):
        seed_trigger(con, iso3="FJI", ym=f"{year}-07", hazard="TC", triggered=True)

    run = br.compute_occurrence(con, rulebook, hazards=["TC"], today=TODAY)

    assert run.rows_written == 0
    assert run.rows_skipped_thin == 1
    assert con.execute("SELECT COUNT(*) FROM haz_base_rates_occurrence").fetchone()[0] == 0


def test_occurrence_min_years_comes_from_the_rulebook(con, rulebook):
    for year in (2020, 2021):
        seed_trigger(con, iso3="FJI", ym=f"{year}-07", hazard="TC", triggered=True)

    relaxed = make_rulebook({"base_rates": {"occurrence": {"min_years": 2}}})
    run = br.compute_occurrence(con, relaxed, hazards=["TC"], today=TODAY)

    assert run.rows_written == 1


def test_occurrence_ignores_months_outside_the_backcast_window(con, rulebook):
    # backcast.flood is 2010, so 2008 is outside it entirely.
    for year in (2007, 2008, 2009):
        seed_trigger(con, iso3="BGD", ym=f"{year}-06", hazard="FL", triggered=True)
    for year in (2011, 2012, 2013):
        seed_trigger(con, iso3="BGD", ym=f"{year}-06", hazard="FL", triggered=False)

    br.compute_occurrence(con, rulebook, hazards=["FL"], today=TODAY)

    p, n_years = con.execute(
        """
        SELECT p_occurrence, n_years FROM haz_base_rates_occurrence
        WHERE iso3 = 'BGD' AND calendar_month = 6
        """
    ).fetchone()
    assert n_years == 3  # only the in-window years
    assert p == 0.0


def test_occurrence_stops_at_the_last_frozen_month(con, rulebook):
    """A month that can still change must not enter a base rate."""

    for year in (2023, 2024, 2025):
        seed_trigger(con, iso3="VNM", ym=f"{year}-07", hazard="FL", triggered=True)
    # July 2026 has not frozen on 2026-08-05 (last frozen is 2026-05).
    seed_trigger(con, iso3="VNM", ym="2026-07", hazard="FL", triggered=True)

    br.compute_occurrence(con, rulebook, hazards=["FL"], today=TODAY)

    n_years = con.execute(
        "SELECT n_years FROM haz_base_rates_occurrence WHERE iso3 = 'VNM' AND calendar_month = 7"
    ).fetchone()[0]
    assert n_years == 3


def test_occurrence_recompute_replaces_rather_than_accumulates(con, rulebook):
    for year in range(2018, 2024):
        seed_trigger(con, iso3="PHL", ym=f"{year}-09", hazard="FL", triggered=True)

    br.compute_occurrence(con, rulebook, hazards=["FL"], today=TODAY)
    br.compute_occurrence(con, rulebook, hazards=["FL"], today=TODAY)

    assert con.execute("SELECT COUNT(*) FROM haz_base_rates_occurrence").fetchone()[0] == 1


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------


def test_severity_quantiles_over_resolved_values(con, rulebook):
    values = [100, 200, 300, 400, 500]
    for index, value in enumerate(values):
        seed_resolution(
            con, iso3="PHL", ym=f"{2018 + index}-03", hazard="FL",
            status="RESOLVED_VALUE", value=float(value),
        )

    br.compute_severity(con, rulebook, hazards=["FL"], today=TODAY)

    q10, q50, q90, n = con.execute(
        "SELECT q10, q50, q90, n_events FROM haz_base_rates_severity WHERE iso3 = 'PHL'"
    ).fetchone()
    assert n == 5
    assert q50 == pytest.approx(300.0)
    assert q10 == pytest.approx(140.0)
    assert q90 == pytest.approx(460.0)


def test_severity_excludes_zeros_and_no_data(con, rulebook):
    """Quantiles are of RESOLVED_VALUE amounts.

    A month with no qualifying hazard is not a small event, and letting
    zeros into the distribution would drag every quantile toward nothing.
    """

    seed_resolution(con, iso3="KEN", ym="2019-03", status="RESOLVED_VALUE", value=5000.0)
    seed_resolution(con, iso3="KEN", ym="2019-04", status="RESOLVED_ZERO", value=0.0)
    seed_resolution(con, iso3="KEN", ym="2019-05", status="NO_DATA", value=None)

    br.compute_severity(con, rulebook, hazards=["FL"], today=TODAY)

    q50, n = con.execute(
        "SELECT q50, n_events FROM haz_base_rates_severity WHERE iso3 = 'KEN'"
    ).fetchone()
    assert n == 1
    assert q50 == pytest.approx(5000.0)


def test_severity_window_starts_at_the_rulebook_year(con, rulebook):
    # severity_base_rate_window_start is 2015.
    seed_resolution(con, iso3="MMR", ym="2013-08", status="RESOLVED_VALUE", value=999999.0)
    seed_resolution(con, iso3="MMR", ym="2016-08", status="RESOLVED_VALUE", value=100.0)

    br.compute_severity(con, rulebook, hazards=["FL"], today=TODAY)

    q50, n, start = con.execute(
        "SELECT q50, n_events, window_start FROM haz_base_rates_severity WHERE iso3 = 'MMR'"
    ).fetchone()
    assert (n, start) == (1, 2015)
    assert q50 == pytest.approx(100.0)


def test_severity_records_the_provenance_mix_by_year(con, rulebook):
    seed_resolution(con, iso3="NPL", ym="2019-07", status="RESOLVED_VALUE",
                    value=100.0, source="emdat")
    seed_resolution(con, iso3="NPL", ym="2019-08", status="RESOLVED_VALUE",
                    value=200.0, source="ifrc_go")
    seed_resolution(con, iso3="NPL", ym="2021-07", status="RESOLVED_VALUE",
                    value=300.0, source="emdat")

    br.compute_severity(con, rulebook, hazards=["FL"], today=TODAY)

    mix = json.loads(
        con.execute(
            "SELECT provenance_mix_json FROM haz_base_rates_severity WHERE iso3 = 'NPL'"
        ).fetchone()[0]
    )
    assert mix["by_year"]["2019"] == {"emdat": 0.5, "ifrc_go": 0.5}
    assert mix["by_year"]["2021"] == {"emdat": 1.0}
    # The shares are rounded to 4dp so the stored JSON stays readable.
    assert mix["overall"]["emdat"] == pytest.approx(2 / 3, abs=1e-4)
    assert mix["n_values"] == 3


def test_severity_mix_flags_a_distribution_made_of_lower_bounds(con, rulebook):
    """A displacement figure is a floor, and the mix has to say so.

    Without ``lower_bound_share`` these quantiles read as an estimate of
    people affected when they are an estimate of the smallest it could be.
    """

    seed_resolution(con, iso3="SOM", ym="2019-07", status="RESOLVED_VALUE",
                    value=100.0, source="idmc_idu", lower_bound=True)
    seed_resolution(con, iso3="SOM", ym="2019-09", status="RESOLVED_VALUE",
                    value=300.0, source="idmc_idu", lower_bound=True)
    seed_resolution(con, iso3="SOM", ym="2020-01", status="RESOLVED_VALUE",
                    value=500.0, source="emdat", lower_bound=False)

    br.compute_severity(con, rulebook, hazards=["FL"], today=TODAY)

    mix = json.loads(
        con.execute(
            "SELECT provenance_mix_json FROM haz_base_rates_severity WHERE iso3 = 'SOM'"
        ).fetchone()[0]
    )
    assert mix["n_lower_bound"] == 2
    assert mix["lower_bound_share"] == pytest.approx(2 / 3, abs=1e-4)


def test_severity_hazards_do_not_bleed_into_each_other(con, rulebook):
    seed_resolution(con, iso3="PHL", ym="2019-07", hazard="FL",
                    status="RESOLVED_VALUE", value=100.0)
    seed_resolution(con, iso3="PHL", ym="2019-11", hazard="TC",
                    status="RESOLVED_VALUE", value=900.0)

    br.compute_severity(con, rulebook, hazards=["FL", "TC"], today=TODAY)

    rows = dict(
        con.execute(
            "SELECT hazard, q50 FROM haz_base_rates_severity WHERE iso3 = 'PHL'"
        ).fetchall()
    )
    assert rows == {"FL": pytest.approx(100.0), "TC": pytest.approx(900.0)}


def test_compute_all_rebuilds_both_tables(con, rulebook):
    for year in range(2018, 2024):
        seed_trigger(con, iso3="PHL", ym=f"{year}-09", hazard="FL", triggered=True)
    seed_resolution(con, iso3="PHL", ym="2019-09", status="RESOLVED_VALUE", value=42.0)

    runs = br.compute_all(con, rulebook, hazards=["FL"], today=TODAY)

    assert runs["occurrence"].rows_written == 1
    assert runs["severity"].rows_written == 1
