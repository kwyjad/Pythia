# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The faults run 33841370196 left in ``enso_state``, and the passes that end them.

Four rows stated ``Neutral`` beside a null ONI and a null Niño 3.4; the
backfill stamped 919 rows of the 1950-2026 ONI table ``fresh`` with an age
of zero; rank 1 answered HTTP 400 and rank 2 served a file frozen in 2021,
so the corroboration machinery had nothing to compare; and the seasonal TC
module discarded a 65-day-old ONI as stale. Each test here pins the repair
by the data it leaves in the table, not by a log line.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from horizon_scanner.enso import indices as idx
from horizon_scanner.tests.test_enso_indices import (
    CPC_ONI,
    CPC_WEEKLY,
    ERDDAP_CSV,
    TODAY,
    _all_dead,
    _getter,
)


@pytest.fixture()
def db(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    from pythia.db.schema import ensure_schema

    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{tmp_path / 'enso.duckdb'}")
    ensure_schema()
    return tmp_path / "enso.duckdb"


def _con():
    from pythia.db.schema import connect

    return connect(read_only=False)


def _row(fetch_date: str) -> dict:
    con = _con()
    try:
        cur = con.execute("SELECT * FROM enso_state WHERE fetch_date = ?", [fetch_date])
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    finally:
        con.close()
    return dict(zip(cols, rows[0])) if rows else {}


#: The ONI table as the backfill would have seen it on 2026-09-04: the
#: newest published season is JJA 2026 (centre July), ONI +1.80.
ONI_TABLE = """ SEAS YR TOTAL ANOM
 DJF 1950 24.72 -1.53
 AMJ 2026 27.60  1.20
 MJJ 2026 27.90  1.55
 JJA 2026 28.05  1.80
"""


def _insert_unbacked(fetch_dates: list[str]) -> None:
    """The pre-fix rows: a scraped word, nothing behind it."""

    con = _con()
    try:
        for day in fetch_dates:
            con.execute(
                "INSERT INTO enso_state (fetch_date, enso_phase, raw_context) "
                "VALUES (?, 'Neutral', 'Current state: Neutral')",
                [day],
            )
    finally:
        con.close()


def _unbacked_count() -> int:
    con = _con()
    try:
        return con.execute(
            "SELECT COUNT(*) FROM enso_state WHERE enso_phase IS NOT NULL "
            "AND (oni IS NULL OR nino34_anomaly IS NULL)"
        ).fetchone()[0]
    finally:
        con.close()


# ---------------------------------------------------------------------------
# A1: the repair pass
# ---------------------------------------------------------------------------

def test_the_four_unbacked_rows_are_rebuilt_from_the_oni_history(db):
    """The acceptance query of the fix, run against the shape of the fault."""

    import horizon_scanner.enso.enso_module as enso

    _insert_unbacked(["2026-07-09", "2026-07-12", "2026-07-15", "2026-08-28"])
    assert enso.backfill_oni_history(get=_getter({"oni.ascii": ONI_TABLE})) == 4
    assert _unbacked_count() == 4

    summary = enso.repair_unbacked_rows(today=dt.date(2026, 9, 4))

    assert summary == {"repaired": 4, "deleted": 0, "untouched": 4}
    assert _unbacked_count() == 0

    row = _row("2026-08-28")
    assert row["enso_phase"] == "El Niño"
    assert row["enso_strength"] == "strong"
    assert row["oni"] == pytest.approx(1.80)
    assert row["nino34_anomaly"] == pytest.approx(1.80)
    assert row["nino34_weekly"] is None           # a seasonal table has no weekly value
    assert str(row["observation_date"]) == "2026-07-01"
    assert row["age_days"] == 58
    assert row["oni_basis"] == "oni_table"
    assert row["nino34_source"] == "cpc_oni_ascii"
    assert row["source_rank_used"] == 3
    assert row["row_kind"] == "repaired"
    assert row["status"] == "carried_forward"
    assert row["scraped_phase"] == "Neutral"      # the original word, kept as what the page said
    warnings = json.loads(row["warnings_json"])
    assert any("repaired on 2026-09-04" in w and "Neutral" in w for w in warnings)
    assert "El Niño, strong" in row["raw_context"]
    assert "STALE READING" in row["raw_context"]

    # The July rows take the newest observation dated at or before them —
    # the JJA season is centred on 1 July — and state the gap as their age.
    july = _row("2026-07-09")
    assert str(july["observation_date"]) == "2026-07-01"
    assert july["oni"] == pytest.approx(1.80)
    assert july["age_days"] == 8


def test_a_row_no_observation_predates_is_deleted_not_defaulted(db):
    import horizon_scanner.enso.enso_module as enso

    _insert_unbacked(["1949-06-01", "2026-08-28"])
    enso.backfill_oni_history(get=_getter({"oni.ascii": ONI_TABLE}))

    summary = enso.repair_unbacked_rows(today=dt.date(2026, 9, 4))

    assert summary["deleted"] == 1
    assert summary["repaired"] == 1
    assert _row("1949-06-01") == {}
    assert _unbacked_count() == 0


def test_the_repair_pass_is_idempotent_and_counts_what_it_left_alone(db):
    import horizon_scanner.enso.enso_module as enso

    _insert_unbacked(["2026-08-28"])
    enso.backfill_oni_history(get=_getter({"oni.ascii": ONI_TABLE}))
    first = enso.repair_unbacked_rows(today=dt.date(2026, 9, 4))
    second = enso.repair_unbacked_rows(today=dt.date(2026, 9, 5))

    assert first["repaired"] == 1
    assert second == {"repaired": 0, "deleted": 0, "untouched": 5}
    assert _row("2026-08-28")["age_days"] == 58   # untouched by the second pass


def test_the_repair_runs_at_the_end_of_every_fetch_and_store(db):
    """Whether the run's own fetch succeeded or every source was dead."""

    import horizon_scanner.enso.enso_module as enso

    _insert_unbacked(["2026-08-28"])
    enso.backfill_oni_history(get=_getter({"oni.ascii": ONI_TABLE}))

    # Every numeric source dead: the run carries forward AND repairs.
    assert enso.fetch_and_store_enso(get=_all_dead, today=dt.date(2026, 9, 4), fetch_page=False)
    assert _unbacked_count() == 0
    assert _row("2026-08-28")["row_kind"] == "repaired"


# ---------------------------------------------------------------------------
# A2: three kinds of row
# ---------------------------------------------------------------------------

def test_backfilled_rows_are_historical_not_fresh(db):
    import horizon_scanner.enso.enso_module as enso

    enso.backfill_oni_history(get=_getter({"oni.ascii": ONI_TABLE}))
    row = _row("1950-01-01")
    assert row["row_kind"] == "historical"
    assert row["status"] == "historical"
    assert row["raw_context"].startswith("## ENSO State (CPC ONI table)\nENSO state for 1950-01:")
    assert "Current state" not in row["raw_context"]


def test_pre_column_rows_are_classified_by_migration(db):
    """The 919 rows the first backfill stamped 'fresh' with age 0."""

    import horizon_scanner.enso.enso_module as enso

    con = _con()
    try:
        # A backfilled row of the old shape, and a live row from a weekly rank.
        con.execute(
            "INSERT INTO enso_state (fetch_date, enso_phase, nino34_anomaly, oni, "
            "oni_basis, observation_date, source_rank_used, nino34_source, status, "
            "age_days) VALUES ('1950-01-01', 'La Niña', -1.53, -1.53, 'oni_table', "
            "'1950-01-01', 3, 'cpc_oni_ascii', 'fresh', 0)"
        )
        con.execute(
            "INSERT INTO enso_state (fetch_date, enso_phase, nino34_anomaly, oni, "
            "oni_basis, observation_date, source_rank_used, nino34_source, status, "
            "age_days) VALUES ('2026-09-01', 'El Niño', 1.95, 1.80, "
            "'weekly_3month_mean', '2026-08-19', 1, 'noaa_erddap_ncepNinoSSTwk', "
            "'fresh', 0)"
        )
    finally:
        con.close()

    assert enso.classify_row_kinds() == {"historical": 1, "live": 1}
    assert _row("1950-01-01")["row_kind"] == "historical"
    assert _row("1950-01-01")["status"] == "historical"
    assert _row("2026-09-01")["row_kind"] == "live"
    assert _row("2026-09-01")["status"] == "fresh"
    # Idempotent: a second pass finds nothing unclassified.
    assert enso.classify_row_kinds() == {"historical": 0, "live": 0}


def test_the_state_as_of_today_never_reads_a_historical_row(db, monkeypatch):
    """A 1950 row must never be the newest thing a caller sees."""

    import horizon_scanner.enso.enso_module as enso

    # Only history in the table: no live row, so no "state as of today".
    enso.backfill_oni_history(get=_getter({"oni.ascii": ONI_TABLE}))
    assert enso.load_enso_state_from_db(max_age_days=100_000) is None

    # Then one live run; the loader returns it, not a season from the table.
    assert enso.fetch_and_store_enso(
        get=_getter({"erddap": ERDDAP_CSV}), today=TODAY, fetch_page=False
    )
    loaded = enso.load_enso_state_from_db(max_age_days=100_000)
    assert loaded is not None
    assert loaded.row_kind == "live"
    assert loaded.fetch_date == TODAY.isoformat()


def test_the_backfill_never_overwrites_a_live_row_on_the_same_date(db):
    import horizon_scanner.enso.enso_module as enso

    # A live run on the first of a month, which is also a season's centre date.
    assert enso.fetch_and_store_enso(
        get=_getter({"erddap": ERDDAP_CSV}), today=dt.date(2026, 7, 1), fetch_page=False
    )
    before = _row("2026-07-01")
    assert before["row_kind"] == "live"

    written = enso.backfill_oni_history(get=_getter({"oni.ascii": ONI_TABLE}))
    assert written == 3                       # four seasons, one date held by a live row
    after = _row("2026-07-01")
    assert after["row_kind"] == "live"
    assert after["nino34_source"] == before["nino34_source"]


# ---------------------------------------------------------------------------
# A3: two working index sources
# ---------------------------------------------------------------------------

#: ERDDAP under the column names the live dataset's info page uses. The
#: request asks for every column, so the parser must find the anomaly by
#: pattern and must not mistake the SST column for it.
ERDDAP_CSV_LIVE_NAMES = """time,Nino12_sst,Nino12_ssta,Nino3_sst,Nino3_ssta,Nino34_sst,Nino34_ssta,Nino4_sst,Nino4_ssta
UTC,degree_C,degree_C,degree_C,degree_C,degree_C,degree_C,degree_C,degree_C
2026-08-05T00:00:00Z,24.80,0.10,27.60,1.70,28.61,1.88,29.40,0.90
2026-08-19T00:00:00Z,24.91,0.18,27.72,1.79,28.70,1.95,29.44,0.93
"""


def test_erddap_is_asked_for_every_column_and_the_anomaly_is_found_by_pattern():
    url = idx.source_ladder(TODAY)[0]["url"]
    assert "NINO3_4" not in url and "ANOM3_4" not in url
    assert url.startswith(idx.ERDDAP_URL + "&time%3E=")

    observations = idx.parse_erddap_csv(ERDDAP_CSV_LIVE_NAMES)
    assert [o.anomaly for o in observations] == [1.88, 1.95]
    # The historical spelling still parses.
    assert [o.anomaly for o in idx.parse_erddap_csv(ERDDAP_CSV)][-1] == 1.95


def test_an_erddap_body_with_no_anomaly_column_names_the_columns_it_saw():
    with pytest.raises(ValueError) as excinfo:
        idx.parse_erddap_csv("time,Nino34_sst\nUTC,degree_C\n2026-08-19T00:00:00Z,28.70\n")
    assert "Nino34_sst" in str(excinfo.value)


def test_rank_two_is_the_live_weekly_file_not_the_frozen_one():
    spec = idx.source_ladder(TODAY)[1]
    assert spec["url"].endswith("wksst9120.for")
    assert idx.CPC_WEEKLY_URL_FROZEN.endswith("wksst8110.for")
    assert all("wksst8110" not in s["url"] for s in idx.source_ladder(TODAY))


def test_every_rank_is_read_and_recorded_even_when_rank_one_answers():
    resolution = idx.resolve_indices(
        get=_getter({"erddap": ERDDAP_CSV, "wksst9120": CPC_WEEKLY, "oni.ascii": CPC_ONI}),
        today=TODAY,
    )
    assert resolution.source_rank_used == 1
    assert [r.ok for r in resolution.readings] == [True, True, True]
    evidence = resolution.as_evidence()
    assert evidence["ranks_ok"] == [1, 2, 3]
    assert evidence["readings"][2]["newest_observation"] == "2026-08-01"
    assert evidence["readings"][2]["oni"] == pytest.approx(1.80)


def test_a_seasonal_only_run_stores_the_weekly_reading_as_absent(db):
    import horizon_scanner.enso.enso_module as enso

    resolution = idx.resolve_indices(get=_getter({"oni.ascii": CPC_ONI}), today=TODAY)
    assert resolution.oni == pytest.approx(1.80)
    assert resolution.nino34 == pytest.approx(1.80)     # the index the phase rests on
    assert resolution.nino34_weekly is None               # not the ONI under a weekly label

    assert enso.fetch_and_store_enso(
        get=_getter({"oni.ascii": CPC_ONI}), today=TODAY, fetch_page=False
    )
    row = _row(TODAY.isoformat())
    assert row["oni"] == pytest.approx(1.80)
    assert row["nino34_anomaly"] == pytest.approx(1.80)
    assert row["nino34_weekly"] is None
    assert "No independent weekly Niño 3.4 reading this run" in row["raw_context"]
    assert "Latest weekly" not in row["raw_context"]


def test_a_weekly_run_stores_the_weekly_reading(db):
    import horizon_scanner.enso.enso_module as enso

    assert enso.fetch_and_store_enso(
        get=_getter({"erddap": ERDDAP_CSV}), today=TODAY, fetch_page=False
    )
    row = _row(TODAY.isoformat())
    assert row["nino34_weekly"] == pytest.approx(1.95)
    assert row["nino34_anomaly"] == pytest.approx(1.95)
    assert "Latest weekly Niño 3.4: +1.9°C" in row["raw_context"] or \
        "Latest weekly Niño 3.4: +2.0°C" in row["raw_context"]


def test_corroboration_uses_the_ranks_already_read_this_run():
    calls: list[str] = []
    both = _getter({"erddap": ERDDAP_CSV, "oni.ascii": CPC_ONI})

    def counting(url, timeout):
        calls.append(url)
        return both(url, timeout)

    resolution = idx.resolve_indices(get=counting, today=TODAY)
    n_after_resolve = len(calls)
    other = idx.corroborate(resolution, get=counting, today=TODAY)
    assert other is not None and other.source_rank_used == 3
    assert len(calls) == n_after_resolve      # no second request was needed


# ---------------------------------------------------------------------------
# A4: the DB loader's age bound
# ---------------------------------------------------------------------------

def test_a_65_day_old_oni_is_served_not_discarded(db, monkeypatch):
    """phase4_seasonal_tc.log, 2026-09-04: '65 days old (max 30); treating as stale'."""

    import horizon_scanner.enso.enso_module as enso

    assert enso.DB_MAX_OBSERVATION_AGE_DAYS == 100
    assert enso.fetch_and_store_enso(
        get=_getter({"oni.ascii": ONI_TABLE}), today=dt.date(2026, 9, 4), fetch_page=False
    )
    # The stored observation is 2026-07-01; make "now" 65 days later.
    class _Clock:
        @staticmethod
        def now(tz=None):
            return dt.datetime(2026, 9, 4, tzinfo=tz)

        @staticmethod
        def strptime(*args, **kwargs):
            return dt.datetime.strptime(*args, **kwargs)

        @staticmethod
        def fromisoformat(text):
            return dt.datetime.fromisoformat(text)

    monkeypatch.setattr(enso, "datetime", _Clock)
    loaded = enso.load_enso_state_from_db()
    assert loaded is not None
    assert loaded.current_state == "El Niño"
    assert loaded.strength == "strong"

    from horizon_scanner.seasonal_tc import imd_nio_scraper

    assert imd_nio_scraper._load_enso_context() == "El Niño"
