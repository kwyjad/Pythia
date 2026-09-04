# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The ENSO record is a number first and a word second.

These tests pin the design that replaced the August 2026 failure, in which
a page-scraped classification of "Neutral" was stored through a strong El
Niño and read as current by every drought and cyclone prompt in the run.

Three properties do the work:

* the phase is COMPUTED from an index and no HTML is involved;
* a null Niño 3.4 can never be written beside a stated phase;
* a run in which every numeric source fails carries the last good record
  forward with its age stated, and never writes Neutral.
"""

from __future__ import annotations

import datetime as dt

import pytest

from horizon_scanner.enso import indices as idx


# ---------------------------------------------------------------------------
# Fixture bodies, in the shape each real source publishes.
# ---------------------------------------------------------------------------

# ERDDAP tabledap: a column-name row, then a UNITS row, then data. The units
# row is the trap — a parser that trusts row order rather than parsing the
# date reads "UTC" as a timestamp.
ERDDAP_CSV = """time,NINO3_4,ANOM3_4
UTC,degree_C,degree_C
2026-06-03T00:00:00Z,28.10,1.62
2026-07-01T00:00:00Z,28.44,1.75
2026-08-05T00:00:00Z,28.61,1.88
2026-08-19T00:00:00Z,28.70,1.95
"""

# CPC wksst9120.for (1991-2020 base; the 8110 file froze in Jan 2021): four SST/SSTA pairs per line — Niño 1+2, 3, 3.4, 4 —
# so the Niño 3.4 anomaly is the sixth number after the week label.
CPC_WEEKLY = """ Nino1+2      Nino3        Nino34        Nino4
Week          SST SSTA     SST SSTA     SST SSTA     SST SSTA
03JUN2026     24.20 -0.30    27.10  1.40    28.10  1.62    29.20  0.80
01JUL2026     24.55 -0.10    27.35  1.55    28.44  1.75    29.31  0.85
05AUG2026     24.80  0.10    27.60  1.70    28.61  1.88    29.40  0.90
19AUG2026     24.91  0.18    27.72  1.79    28.70  1.95    29.44  0.93
"""

# CPC oni.ascii.txt: SEAS YR TOTAL ANOM, seasons overlapping by two months.
CPC_ONI = """ SEAS YR TOTAL ANOM
 DJF 1950 24.72 -1.53
 MJJ 2026 27.90  1.55
 JJA 2026 28.05  1.70
 JAS 2026 28.20  1.80
"""

TODAY = dt.date(2026, 9, 1)


def _getter(bodies: dict[str, str]):
    """A transport seam that serves a body per URL fragment, else raises."""

    def get(url: str, timeout: float) -> str:
        for fragment, body in bodies.items():
            if fragment in url:
                return body
        raise RuntimeError(f"no fixture for {url}")

    return get


def _all_dead(url: str, timeout: float) -> str:
    raise RuntimeError("connection refused")


# ---------------------------------------------------------------------------
# Classification is arithmetic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("oni", "phase", "strength"),
    [
        (1.8, "El Niño", "strong"),
        (0.5, "El Niño", "weak"),
        (0.49, "Neutral", ""),
        (1.0, "El Niño", "moderate"),
        (1.5, "El Niño", "strong"),
        (2.4, "El Niño", "very strong"),
        (-0.5, "La Niña", "weak"),
        (-1.7, "La Niña", "strong"),
        (0.0, "Neutral", ""),
    ],
)
def test_classify_oni_follows_the_operational_definition(oni, phase, strength):
    assert idx.classify_oni(oni) == (phase, strength)


def test_an_oni_of_plus_1_8_is_el_nino_strong_with_no_page_parsing(monkeypatch):
    """The August 2026 case, resolved off the number alone.

    The IRI page is not fetched, not parsed, and not present. If any part of
    the phase still depended on HTML this test could not pass.
    """

    import horizon_scanner.enso.enso_module as enso

    def explode(*_a, **_k):  # pragma: no cover - must never be called
        raise AssertionError("the phase must not depend on the IRI page")

    monkeypatch.setattr(enso, "fetch_iri_page", explode)

    record = enso.build_enso_record(
        get=_getter({"erddap": ERDDAP_CSV}), today=TODAY, fetch_page=False
    )

    assert record.current_state == "El Niño"
    assert record.strength == "strong"
    assert record.oni == pytest.approx(1.8, abs=0.06)
    assert record.status == "fresh"
    assert record.source_rank_used == 1
    assert enso.describe_phase(record.current_state, record.strength) == "El Niño, strong"


def test_no_number_means_no_phase():
    """``classify_oni(None)`` returns no phase, not Neutral.

    Neutral is a computed result. It is never what the code says when it
    does not know, and this is where that starts.
    """

    assert idx.classify_oni(None) == ("", "")


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def test_erddap_parser_skips_the_units_row():
    observations = idx.parse_erddap_csv(ERDDAP_CSV)
    assert len(observations) == 4
    assert observations[-1].date == dt.date(2026, 8, 19)
    assert observations[-1].anomaly == pytest.approx(1.95)


def test_cpc_weekly_parser_takes_the_nino34_anomaly_not_a_neighbouring_column():
    observations = idx.parse_cpc_weekly(CPC_WEEKLY)
    assert len(observations) == 4
    latest = observations[-1]
    assert latest.date == dt.date(2026, 8, 19)
    # 1.95 is Niño 3.4's anomaly. 1.79 is Niño 3's and 0.93 is Niño 4's;
    # picking either would misclassify the strength band.
    assert latest.anomaly == pytest.approx(1.95)


def test_oni_seasons_are_dated_to_their_centre_month():
    """JAS 2026 describes August 2026, not July and not September.

    Dating a season to its first month would report the ONI as two months
    staler than it is and could trip the staleness bound on a healthy feed.
    """

    observations = idx.parse_cpc_oni(CPC_ONI)
    by_date = {obs.date: obs.anomaly for obs in observations}
    assert by_date[dt.date(2026, 8, 1)] == pytest.approx(1.80)   # JAS
    assert by_date[dt.date(2026, 7, 1)] == pytest.approx(1.70)   # JJA
    assert by_date[dt.date(1950, 1, 1)] == pytest.approx(-1.53)  # DJF 1950


@pytest.mark.parametrize("bad", ["", "n/a", None, "9.9", "-12.0", "nan", "abc"])
def test_an_out_of_range_or_unparseable_value_is_a_parse_failure(bad):
    """Not a reading, and therefore not a phase.

    The observed Niño 3.4 record has never approached ±4 °C, so a value
    beyond it means a column was misread.
    """

    assert idx.valid_anomaly(bad) is None


def test_valid_anomaly_accepts_the_real_range():
    assert idx.valid_anomaly("1.95") == pytest.approx(1.95)
    assert idx.valid_anomaly(-2.2) == pytest.approx(-2.2)


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------

def test_the_ladder_falls_through_to_the_next_rank():
    resolution = idx.resolve_indices(
        get=_getter({"wksst9120": CPC_WEEKLY}), today=TODAY
    )
    assert resolution.resolved
    assert resolution.source_rank_used == 2
    assert resolution.readings[0].ok is False
    assert resolution.readings[0].error


def test_rank_three_is_the_published_oni_and_says_so():
    resolution = idx.resolve_indices(get=_getter({"oni.ascii": CPC_ONI}), today=TODAY)
    assert resolution.source_rank_used == 3
    assert resolution.oni_basis == idx.BASIS_ONI_TABLE
    assert resolution.oni == pytest.approx(1.80)


def test_a_weekly_series_yields_a_labelled_proxy_not_a_published_oni():
    resolution = idx.resolve_indices(get=_getter({"erddap": ERDDAP_CSV}), today=TODAY)
    assert resolution.oni_basis == idx.BASIS_WEEKLY_MEAN
    # Every observation inside the trailing 90 days of the newest one.
    assert resolution.n_observations == 4
    assert resolution.oni == pytest.approx((1.62 + 1.75 + 1.88 + 1.95) / 4, abs=1e-3)
    # The newest weekly value is kept separately from the mean: they are
    # different quantities and the classification uses the mean.
    assert resolution.nino34 == pytest.approx(1.95)


def test_a_stale_series_is_refused_rather_than_read_as_current():
    """An archived copy of last year's file is not a reading about this month."""

    resolution = idx.resolve_indices(
        get=_getter({"erddap": ERDDAP_CSV}), today=dt.date(2027, 6, 1)
    )
    assert not resolution.resolved
    assert any("days old" in (r.error or "") for r in resolution.readings)


def test_every_source_failing_resolves_nothing():
    resolution = idx.resolve_indices(get=_all_dead, today=TODAY)
    assert not resolution.resolved
    assert resolution.nino34 is None
    assert resolution.oni is None
    assert len(resolution.readings) == 3
    assert all(not r.ok for r in resolution.readings)
