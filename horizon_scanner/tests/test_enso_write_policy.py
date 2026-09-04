# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""What may be written to ``enso_state``, and what happens when nothing can be.

The August 2026 row stated a phase with no index behind it. One rule
forecloses that: a null Niño 3.4 can never accompany a stated phase. The
other half of the design is what a failed run does instead of defaulting —
it carries the last good reading forward with its age on the row, so a
prompt reads a stale reading as stale.
"""

from __future__ import annotations

import datetime as dt

import pytest

from horizon_scanner.tests.test_enso_indices import (
    CPC_ONI,
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


def _rows():
    from pythia.db.schema import connect

    con = connect(read_only=False)
    try:
        return con.execute(
            """
            SELECT fetch_date, enso_phase, nino34_anomaly, oni, enso_strength,
                   observation_date, status, age_days, nino34_source
            FROM enso_state ORDER BY fetch_date
            """
        ).fetchall()
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_a_phase_with_no_index_is_refused(db):
    """The August 2026 row, offered to the writer and turned away."""

    import horizon_scanner.enso.enso_module as enso

    record = enso.ENSOForecast()
    record.current_state = "Neutral"      # what the page said
    record.nino34_latest_weekly = None    # what the machine actually had
    record.oni = None

    problems = enso.validation_problems(record)
    assert problems
    assert any("no valid Niño 3.4" in p for p in problems)
    assert enso.store_enso_state(record) is False
    assert _rows() == []


def test_an_out_of_range_index_is_refused(db):
    import horizon_scanner.enso.enso_module as enso

    record = enso.ENSOForecast()
    record.current_state = "El Niño"
    record.nino34_latest_weekly = 12.0    # a misread column, not an ocean
    record.oni = 12.0

    assert enso.validation_problems(record)
    assert enso.store_enso_state(record) is False
    assert _rows() == []


def test_a_complete_record_is_written_with_its_computed_phase(db):
    import horizon_scanner.enso.enso_module as enso

    assert enso.fetch_and_store_enso(
        get=_getter({"erddap": ERDDAP_CSV}), today=TODAY, fetch_page=False
    )

    (row,) = _rows()
    _fetch, phase, nino34, oni, strength, observed, status, age, source = row
    assert phase == "El Niño"
    assert strength == "strong"
    assert oni == pytest.approx(1.8, abs=0.06)
    assert nino34 == pytest.approx(1.95)
    assert str(observed) == "2026-08-19"
    assert status == "fresh"
    assert age == 0
    assert source == "noaa_erddap_ncepNinoSSTwk"


# ---------------------------------------------------------------------------
# Carry forward
# ---------------------------------------------------------------------------

def test_every_source_failing_carries_forward_and_never_writes_neutral(db):
    """The run that would previously have stored a scraped word.

    It writes a row — the table gets one every run — but that row is the
    last good reading, under its own observation date, with a non-zero age.
    """

    import horizon_scanner.enso.enso_module as enso

    assert enso.fetch_and_store_enso(
        get=_getter({"erddap": ERDDAP_CSV}), today=TODAY, fetch_page=False
    )

    later = dt.date(2026, 9, 28)
    assert enso.fetch_and_store_enso(get=_all_dead, today=later, fetch_page=False)

    rows = _rows()
    assert len(rows) == 2
    carried = [r for r in rows if r[6] == "carried_forward"]
    assert len(carried) == 1
    (_fetch, phase, _n34, oni, strength, observed, status, age, _src) = carried[0]

    assert phase == "El Niño"          # never Neutral, never blank
    assert strength == "strong"
    assert oni == pytest.approx(1.8, abs=0.06)
    assert str(observed) == "2026-08-19"   # the ORIGINAL observation date
    assert status == "carried_forward"
    assert age == (later - dt.date(2026, 8, 19)).days
    assert age > 0                          # a non-zero age, stated on the row


def test_a_carried_forward_record_labels_itself_stale_in_the_prompt(db):
    import horizon_scanner.enso.enso_module as enso

    enso.fetch_and_store_enso(
        get=_getter({"erddap": ERDDAP_CSV}), today=TODAY, fetch_page=False
    )
    previous = enso.load_last_good_record()
    carried = enso.carry_forward(previous, today=dt.date(2026, 9, 28))

    text = carried.to_prompt_context()
    assert "STALE READING" in text
    assert "40 days old" in text
    assert "2026-08-19" in text
    # The state is still there — a stale reading is better than none, so
    # long as the reader is told which it is.
    assert "El Niño, strong" in text


def test_with_no_previous_record_a_failed_run_writes_nothing(db):
    """An absent row is honest. A defaulted Neutral is not."""

    import horizon_scanner.enso.enso_module as enso

    assert enso.fetch_and_store_enso(get=_all_dead, today=TODAY, fetch_page=False) is False
    assert _rows() == []


# ---------------------------------------------------------------------------
# Continuity
# ---------------------------------------------------------------------------

def _record(oni: float, observed: str, state: str = "", strength: str = ""):
    import horizon_scanner.enso.enso_module as enso
    from horizon_scanner.enso.indices import classify_oni

    f = enso.ENSOForecast()
    f.oni = oni
    f.nino34_latest_weekly = oni
    f.observation_date = observed
    f.current_state, f.strength = (state, strength) if state else classify_oni(oni)
    return f


def test_a_large_oni_step_is_flagged():
    import horizon_scanner.enso.enso_module as enso

    problems = enso.continuity_problems(
        _record(0.2, "2026-07-01"), _record(1.9, "2026-08-01")
    )
    assert problems and "over the 1.0" in problems[0]


def test_a_phase_transition_skipping_neutral_is_flagged():
    import horizon_scanner.enso.enso_module as enso

    problems = enso.continuity_problems(
        _record(-1.6, "2026-07-01"), _record(1.6, "2026-08-01")
    )
    assert any("without passing through Neutral" in p for p in problems)


def test_an_ordinary_move_is_not_flagged():
    import horizon_scanner.enso.enso_module as enso

    assert enso.continuity_problems(
        _record(1.5, "2026-07-01"), _record(1.8, "2026-08-01")
    ) == []


def test_an_uncorroborated_jump_carries_forward_instead_of_being_believed(db):
    """Two explanations are open; the conservative one is chosen.

    A move this large is physically implausible in a month, so the cheapest
    explanation is that a column changed. With no second source to confirm
    it, the previous reading stands and the run says why.
    """

    import horizon_scanner.enso.enso_module as enso

    # A quiet baseline: ONI -0.1 in July.
    quiet = "\n".join([" SEAS YR TOTAL ANOM", " MJJ 2026 26.90 -0.10"])
    assert enso.fetch_and_store_enso(
        get=_getter({"oni.ascii": quiet}), today=dt.date(2026, 7, 15), fetch_page=False
    )

    # Then a single source claims +1.95, and nothing else answers.
    assert enso.fetch_and_store_enso(
        get=_getter({"erddap": ERDDAP_CSV}), today=TODAY, fetch_page=False
    )

    rows = _rows()
    latest = rows[-1]
    assert latest[6] == "carried_forward"
    assert latest[3] == pytest.approx(-0.10)   # the quiet reading stood


def test_a_corroborated_jump_is_believed(db):
    """The same jump, confirmed by a second numeric source, is written."""

    import horizon_scanner.enso.enso_module as enso

    quiet = "\n".join([" SEAS YR TOTAL ANOM", " MJJ 2026 26.90 -0.10"])
    enso.fetch_and_store_enso(
        get=_getter({"oni.ascii": quiet}), today=dt.date(2026, 7, 15), fetch_page=False
    )

    both = _getter({"erddap": ERDDAP_CSV, "oni.ascii": CPC_ONI})
    assert enso.fetch_and_store_enso(get=both, today=TODAY, fetch_page=False)

    latest = _rows()[-1]
    assert latest[6] == "fresh"
    assert latest[1] == "El Niño"
    assert latest[4] == "strong"


# ---------------------------------------------------------------------------
# The page is decoration
# ---------------------------------------------------------------------------

def test_a_scraped_phase_that_disagrees_loses_and_is_recorded(db):
    """August 2026 in miniature: the page says Neutral, the index says otherwise."""

    import horizon_scanner.enso.enso_module as enso

    html = "<html><body><p>ENSO-neutral conditions are present.</p></body></html>"
    record = enso.build_enso_record(
        html=html, get=_getter({"erddap": ERDDAP_CSV}), today=TODAY, fetch_page=False
    )

    assert record.scraped_state == "Neutral"
    assert record.current_state == "El Niño"    # the computed phase wins
    assert record.state_disagreement
    assert "Neutral" in record.state_disagreement
    assert record.warnings


def test_an_unreachable_page_costs_only_the_page(db, monkeypatch):
    """A missing probability table does not cost the phase."""

    import horizon_scanner.enso.enso_module as enso

    def dead_page():
        raise RuntimeError("iri.columbia.edu timed out")

    monkeypatch.setattr(enso, "fetch_iri_page", dead_page)
    record = enso.build_enso_record(get=_getter({"erddap": ERDDAP_CSV}), today=TODAY)

    assert record.current_state == "El Niño"
    assert record.probability_forecast == []
    assert record.iod_state == ""


def test_backfill_seeds_history_from_the_published_oni_table(db):
    import horizon_scanner.enso.enso_module as enso

    written = enso.backfill_oni_history(get=_getter({"oni.ascii": CPC_ONI}))
    assert written == 4

    rows = _rows()
    assert len(rows) == 4
    by_observed = {str(r[5]): r for r in rows}
    assert by_observed["1950-01-01"][1] == "La Niña"
    assert by_observed["1950-01-01"][4] == "strong"
    assert by_observed["2026-08-01"][1] == "El Niño"
    # The published table is the record's past, never a reading a run took.
    assert all(r[6] == "historical" for r in rows)


def test_recompute_overwrites_a_stored_row_from_the_oni_table(db):
    """The correction path for the August 2026 record.

    A run on 2026-08-28 could only have known the JJA figure — the ONI is a
    three-month mean and JAS is not published until September — so the
    recompute takes the newest observation at or before the row's date and
    never a later one.
    """

    import horizon_scanner.enso.enso_module as enso
    from pythia.db.schema import connect

    con = connect(read_only=False)
    try:
        con.execute(
            "INSERT INTO enso_state (fetch_date, enso_phase, raw_context) VALUES "
            "('2026-08-28', 'Neutral', 'the wrong record')"
        )
    finally:
        con.close()

    record = enso.recompute_record("2026-08-28", get=_getter({"oni.ascii": CPC_ONI}))
    assert record is not None
    assert record.current_state == "El Niño"
    assert record.strength == "strong"
    assert record.oni == pytest.approx(1.80)
    assert record.observation_date == "2026-08-01"

    (row,) = [r for r in _rows() if str(r[0]) == "2026-08-28"]
    assert row[1] == "El Niño"
    assert row[4] == "strong"
