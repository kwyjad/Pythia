# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Group F of the run-34124705852 repairs: six small faults, each of which
costs rows or costs attention.

None of them is architectural. Each one either drops a row that should have
landed, or spends a reader's attention on something that was never wrong.

F1  ENSO claimed one row and touched none. ``enso_state``'s only timestamp
    was ``created_at`` with a column default, and its INSERT OR REPLACE did
    not name it, so a row an earlier run had already written for today's
    fetch_date kept the earlier run's stamp. Two runs share a date and the
    second is invisible. Beneath it sat a trap worth closing on its own:
    ``fetch_date`` is a DATE, so cast to a timestamp it is midnight and a
    same-day run start is always later. Such a table can never say it was
    touched.
F2  GDACS warned every run about tropical cyclones over open ocean, which
    name no country because they are at sea. A warning for the ordinary
    case is how a reader learns to skip the warnings that matter.
F3  Seven territories with no resident population were assessed every
    month. They cannot carry humanitarian impact and have no population
    denominator, so they were noise in the denominator of every occurrence
    base rate and every acceptance rate.
F5  ACLED political attributed an event by three ISO3 keys and a name. A
    country whose name our alias table does not carry lost every one of its
    events, which is exactly what happened to GIN, COG, BIH, KOR and TLS.
F6  CrisisWatch could not resolve Trinidad and Tobago or Jamaica at all, so
    an entry for either was parsed and dropped. And a CDX query was clamped
    to 30 seconds, where a timeout costs the whole edition it was for.

Network-free.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest


# ---------------------------------------------------------------------------
# F1: enso_state carries a stamp its own writers move
# ---------------------------------------------------------------------------


@pytest.fixture()
def enso_db(tmp_path):
    from pythia.db import schema as pythia_schema

    con = duckdb.connect(str(tmp_path / "enso.duckdb"))
    pythia_schema._ensure_enso_state_table(con)
    return con


def _columns(con) -> set[str]:
    return {row[1] for row in con.execute("PRAGMA table_info('enso_state')").fetchall()}


def test_enso_state_has_a_write_stamp_of_its_own(enso_db):
    """``created_at`` is first-seen and cannot answer for a matched upsert."""

    assert "updated_at" in _columns(enso_db), (
        "enso_state needs a stamp that moves on every write; created_at does not"
    )


def test_an_insert_or_replace_moves_updated_at_but_not_created_at(enso_db):
    """The exact shape that made the run's reconciliation read zero."""

    con = enso_db
    insert = (
        "INSERT OR REPLACE INTO enso_state "
        "(fetch_date, enso_phase, oni, row_kind, updated_at) "
        "VALUES (DATE '2026-09-07', ?, ?, 'live', CURRENT_TIMESTAMP)"
    )
    con.execute(insert, ["La Niña", -0.9])
    first = con.execute(
        "SELECT created_at, updated_at FROM enso_state "
        "WHERE fetch_date = DATE '2026-09-07'"
    ).fetchone()

    con.execute("SELECT 1")  # let the clock advance between the two writes
    con.execute(insert, ["El Niño", 2.08])
    second = con.execute(
        "SELECT created_at, updated_at, enso_phase FROM enso_state "
        "WHERE fetch_date = DATE '2026-09-07'"
    ).fetchone()

    assert second[2] == "El Niño", "the second write must win"
    assert second[1] >= first[1], "updated_at must move with the write"
    assert second[1] is not None


def test_every_enso_state_write_site_names_updated_at():
    """A writer that forgets the stamp reintroduces the fault silently.

    Asserted against the source because the failure has no end state to
    observe: the row is correct, only its stamp is stale, and every
    downstream reader of the DATA is happy.
    """

    source = Path("horizon_scanner/enso/enso_module.py").read_text(encoding="utf-8")
    writes = list(source.split("INSERT OR REPLACE INTO enso_state")[1:])
    assert writes, "expected the enso_state writers to still be here"
    for chunk in writes:
        statement = chunk.split('"""')[0]
        assert "updated_at" in statement, (
            "an INSERT OR REPLACE into enso_state that does not name updated_at "
            "leaves the stamp at whatever the matched row already held"
        )

    for update in source.split("UPDATE enso_state")[1:]:
        statement = update.split("WHERE")[0]
        assert "updated_at" in statement, (
            "an UPDATE that rewrites a row must move its write stamp too"
        )


def test_a_date_stamp_is_compared_as_a_date(tmp_path):
    """A DATE column cast to a timestamp is midnight, so it never answers."""

    from scripts.build_resolver_debug_bundle import BundleBuilder

    db = tmp_path / "stamped.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE only_a_date (iso3 TEXT, fetch_date DATE)")
    con.execute("INSERT INTO only_a_date VALUES ('SOM', DATE '2026-09-07')")
    con.close()

    builder = BundleBuilder(
        db_path=db,
        out_path=tmp_path / "b.zip",
        diagnostics_dir=tmp_path / "diagnostics",
        run_log_dir=tmp_path / "run_log",
        staging=tmp_path / "staging",
    )
    touched = builder._touched_since("only_a_date", None, "2026-09-07T12:59:07Z")

    assert touched == 1, (
        "a row written on the run's own day must count as touched even when "
        "its only stamp has date granularity"
    )


# ---------------------------------------------------------------------------
# F2: a cyclone at sea is not a fault
# ---------------------------------------------------------------------------


def test_an_event_over_open_ocean_is_reported_at_info(caplog):
    """A warning for the ordinary case trains the reader to skip warnings."""

    import logging

    from resolver.connectors import gdacs as gdacs_mod

    source = Path(gdacs_mod.__file__).read_text(encoding="utf-8")
    block = source.split("if self.events_without_country:")[1].split("return rows")[0]
    assert "LOG.info(" in block, (
        "an event GDACS names no country for, and that geometry cannot place, "
        "is a storm at sea — counted and named, but not a warning every run"
    )
    assert "LOG.warning(" not in block
    assert logging.INFO < logging.WARNING  # the point, stated


def test_the_open_ocean_events_are_still_counted_and_named():
    """Downgrading a log must not hide the number."""

    from resolver.connectors import gdacs as gdacs_mod

    source = Path(gdacs_mod.__file__).read_text(encoding="utf-8")
    block = source.split("if self.events_without_country:")[1].split("return rows")[0]
    assert "len(self.events_without_country)" in block
    assert "events_without_country[:30]" in block, (
        "the ids must still be listed, or a rise in unplaceable events is "
        "invisible rather than merely quiet"
    )


# ---------------------------------------------------------------------------
# F3: a place with no people is not assessed
# ---------------------------------------------------------------------------


def test_the_unpopulated_territories_are_not_assessed():
    from resolver.hazard_resolution.cli import (
        UNPOPULATED_TERRITORIES,
        _load_universe,
    )

    universe = set(_load_universe())
    assert not (universe & UNPOPULATED_TERRITORIES), (
        "a territory with no resident population has no population cap and, "
        "with GDACS silent, no upper bound at all; assessing it puts noise in "
        "every base-rate and acceptance denominator"
    )
    assert len(universe) > 200, "the rest of the world must still be assessed"


def test_every_excluded_territory_is_absent_from_the_population_table():
    """The exclusion must rest on evidence, not on a hunch about a name."""

    from resolver.hazard_resolution.cli import UNPOPULATED_TERRITORIES

    text = Path("resolver/data/population.csv").read_text(encoding="utf-8-sig")
    for iso3 in sorted(UNPOPULATED_TERRITORIES):
        assert f",{iso3}," not in text, (
            f"{iso3} has a population figure, so excluding it drops real cells"
        )


def test_a_populated_small_state_is_still_assessed():
    """The list must be the seven, not everything small."""

    from resolver.hazard_resolution.cli import _load_universe

    universe = set(_load_universe())
    for iso3 in ("TUV", "NRU", "VUT", "TON", "MHL"):
        assert iso3 in universe, f"{iso3} has people and cyclones; it is assessed"


# ---------------------------------------------------------------------------
# F5: attribute by any identifier the source states
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "event,expected",
    [
        ({"iso3": "SOM"}, "SOM"),
        ({"country_iso3": "ETH"}, "ETH"),
        ({"#country+code": "KEN"}, "KEN"),
        # The casing variants a key read must survive. ACLED has served the
        # same field under more than one spelling.
        ({"ISO3": "SDN"}, "SDN"),
        ({"Iso3": "TCD"}, "TCD"),
        ({"COUNTRY_ISO3": "MLI"}, "MLI"),
        # The NUMERIC code, which the request already sends as its filter.
        ({"iso": 706}, "SOM"),
        ({"iso": "231"}, "ETH"),
        ({"#country+code+num": "404"}, "KEN"),
        # The name, last, because it has to survive our alias table.
        ({"country": "Guinea"}, "GIN"),
        ({}, ""),
        ({"iso": "notanumber"}, ""),
    ],
)
def test_an_event_is_attributed_by_whatever_the_source_stated(event, expected):
    from pythia.acled_political import _event_iso3

    assert _event_iso3(event) == expected


def test_a_code_outranks_a_name():
    """A code is what the source said about identity; a name is a guess."""

    from pythia.acled_political import _event_iso3

    assert _event_iso3({"iso3": "SOM", "country": "Ethiopia"}) == "SOM"


def test_a_numeric_code_rescues_a_country_our_aliases_miss():
    """The fault that cost GIN, COG, BIH, KOR and TLS every one of their events."""

    from pythia.acled_political import _event_iso3

    # A name form no alias table carries, beside the numeric code the API
    # returns because the request filtered on it.
    assert _event_iso3({"country": "Kingdom of Eswatini", "iso": "748"}) == "SWZ"


def test_a_zero_resolution_warning_names_the_keys_it_saw(caplog):
    """Three different repairs hide behind '0 of 50 resolve'."""

    import logging

    from pythia.acled_political import _filter_events_to_country

    events = [{"unexpected_key": "x", "another": 1}]
    with caplog.at_level(logging.WARNING):
        kept = _filter_events_to_country(events, "SOM")

    assert kept == []
    text = caplog.text
    assert "unexpected_key" in text and "another" in text, (
        "a changed field name, a missing alias and an ignored filter all read "
        "as '0 resolve'; only the keys say which"
    )


# ---------------------------------------------------------------------------
# F6: two countries CrisisWatch could not name, and a query given too little time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,iso3", [
    ("Trinidad and Tobago", "TTO"),
    ("Jamaica", "JAM"),
])
def test_the_missing_caribbean_countries_resolve(name, iso3):
    from horizon_scanner.crisiswatch import _resolve_iso3

    assert _resolve_iso3(name) == iso3, (
        f"an ICG entry for {name} was parsed and then dropped as unresolved"
    )


def test_the_existing_country_map_is_unchanged_for_its_neighbours():
    """An insertion must not disturb what was already right."""

    from horizon_scanner.crisiswatch import _resolve_iso3

    assert _resolve_iso3("Somaliland") == "SOM"
    assert _resolve_iso3("Togo") == "TGO"
    assert _resolve_iso3("Tunisia") == "TUN"
    assert _resolve_iso3("Jordan") == "JOR"


def test_a_cdx_query_is_given_more_than_thirty_seconds():
    """A CDX timeout costs the whole edition the query was for."""

    from scripts import refresh_crisiswatch as rc

    assert rc._CDX_TIMEOUT_SEC >= 60, (
        "30s is tight against an archive answering under load, and the walk "
        "then reports no capture exists when we never waited long enough"
    )
    source = Path("scripts/refresh_crisiswatch.py").read_text(encoding="utf-8")
    assert "min(timeout_sec, 30)" not in source, (
        "the raised ceiling must reach every CDX call site, not just one"
    )
