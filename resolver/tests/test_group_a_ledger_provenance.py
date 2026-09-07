# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Group A of the run-34124705852 repairs: a ledger that forgets its rules.

A one-character delimiter bug froze ten years of drought behind a wall of
``status='ok'``. The ASAP fix landed, and not one of the 114 affected
months would ever re-walk to use it, because the resume ledger records
only THAT a month finished — never what decided it. Clearing it needed a
human to remember a flag.

The ledger now records the rulebook fingerprint that governed the hazard
and the commit that walked it. A month whose stored fingerprint differs
from the rulebook in force is re-walked on its own. The commit is recorded
and deliberately NOT compared: a bare code change must not invalidate the
ledger, or every merge would trigger a full re-walk of every hazard.

Beside it, the NMME coverage question. ``seasonal_forecasts`` rows carry an
ISSUE date and a lead, and the row is ABOUT issue + lead. The span check
compared the raw issue date, so a table whose earliest vintage was issued
in July at lead 1 — and therefore speaks for August onwards — looked as
though it covered July. A month with no possible vintage was reported as a
fault, once per month, for every backcast month before the ingest began.

Network-free.
"""

from __future__ import annotations

import duckdb
import pytest

from resolver.hazard_resolution import backcast as bc
from resolver.hazard_resolution import drought_indicators as di
from resolver.hazard_resolution.rulebook import load_rulebook
from resolver.hazard_resolution.schema import ensure_haz_schema


@pytest.fixture()
def con(tmp_path):
    connection = duckdb.connect(str(tmp_path / "haz.duckdb"))
    ensure_haz_schema(connection)
    return connection


def _trigger_row(con, hazard: str, ym: str) -> None:
    """A trigger row, so completed_months' database check is satisfied."""

    year, month = int(ym[:4]), int(ym[5:7])
    con.execute(
        "INSERT INTO haz_triggers (iso3, hazard, year, month, triggered, "
        "trigger_source) VALUES ('SOM', ?, ?, ?, FALSE, 'test')",
        [hazard, year, month],
    )


# ---------------------------------------------------------------------------
# A1: the ledger records what decided a month
# ---------------------------------------------------------------------------


def test_the_ledger_carries_the_rulebook_that_decided_each_month(con):
    bc.record_month(
        con, hazard="DR", ym="2020-05", status="ok", counts={"cells": 12},
        walked_by_commit="a" * 40, rulebook_hash="feedfacecafe0001",
    )
    row = con.execute(
        "SELECT walked_by_commit, rulebook_hash FROM haz_backcast_progress "
        "WHERE hazard = 'DR' AND ym = '2020-05'"
    ).fetchone()
    assert row == ("a" * 40, "feedfacecafe0001")


def test_a_changed_rulebook_re_walks_the_month(con):
    """The ASAP case: a fix lands and the ledger must not hide it."""

    _trigger_row(con, "DR", "2020-05")
    bc.record_month(
        con, hazard="DR", ym="2020-05", status="ok", counts={"cells": 12},
        rulebook_hash="before_the_fix",
    )
    assert bc.completed_months(con, "DR", "before_the_fix") == {"2020-05"}
    assert bc.completed_months(con, "DR", "after_the_fix") == set(), (
        "a month decided under rules since changed must be walked again, "
        "without an operator remembering --no-resume"
    )


def test_a_bare_commit_change_does_not_re_walk_anything(con):
    """Otherwise every merge triggers a full re-walk of every hazard."""

    _trigger_row(con, "DR", "2020-05")
    bc.record_month(
        con, hazard="DR", ym="2020-05", status="ok", counts={"cells": 12},
        walked_by_commit="a" * 40, rulebook_hash="unchanged",
    )
    # A later run on a different commit, same rulebook.
    assert bc.completed_months(con, "DR", "unchanged") == {"2020-05"}


def test_a_month_with_no_recorded_rulebook_is_re_walked_once(con):
    """A pre-column row recorded nothing about what decided it."""

    _trigger_row(con, "DR", "2020-05")
    bc.record_month(con, hazard="DR", ym="2020-05", status="ok", counts={"cells": 12})
    assert bc.completed_months(con, "DR", "any_hash") == set()
    # And with no fingerprint to compare, behaviour is exactly as before.
    assert bc.completed_months(con, "DR") == {"2020-05"}


def test_the_database_still_overrules_the_ledger(con):
    """The existing guard must survive: cells assessed, no trigger rows."""

    bc.record_month(
        con, hazard="DR", ym="2020-05", status="ok", counts={"cells": 12},
        rulebook_hash="current",
    )
    assert bc.completed_months(con, "DR", "current") == set()


def test_a_hazards_fingerprint_moves_only_with_its_own_rules():
    """A digest over the whole file would re-walk all three on any edit."""

    rb = load_rulebook()
    drought = rb.hazard_fingerprint("drought")
    flood = rb.hazard_fingerprint("flood")
    assert drought != flood
    assert drought == rb.hazard_fingerprint("drought"), "must be stable"
    assert len(drought) == 16


def test_the_fingerprint_ignores_key_order():
    """A reordered YAML must not look like a rule change."""

    from pathlib import Path

    from resolver.hazard_resolution.rulebook import Rulebook

    a = Rulebook({"drought": {"x": 1, "y": 2}, "ladder": ["emdat"]}, Path("a"))
    b = Rulebook({"ladder": ["emdat"], "drought": {"y": 2, "x": 1}}, Path("b"))
    assert a.hazard_fingerprint("drought") == b.hazard_fingerprint("drought")


def test_a_shared_section_change_moves_every_hazard():
    """The ladder decides a flood answer as surely as the flood section."""

    from pathlib import Path

    from resolver.hazard_resolution.rulebook import Rulebook

    a = Rulebook({"flood": {"x": 1}, "ladder": ["emdat", "reliefweb"]}, Path("a"))
    b = Rulebook({"flood": {"x": 1}, "ladder": ["emdat"]}, Path("b"))
    assert a.hazard_fingerprint("flood") != b.hazard_fingerprint("flood")


# ---------------------------------------------------------------------------
# A2: a forecast table's span is the months it SPEAKS FOR
# ---------------------------------------------------------------------------


@pytest.fixture()
def nmme(tmp_path):
    """seasonal_forecasts in the shape run 34124705852 actually held."""

    con = duckdb.connect(str(tmp_path / "pythia.duckdb"))
    con.execute(
        "CREATE TABLE seasonal_forecasts (iso3 TEXT, variable TEXT, value DOUBLE, "
        "forecast_issue_date DATE, lead_months INTEGER)"
    )
    rows = [
        ("SOM", "prate", -1.4, issue, lead)
        for issue in ("2026-07-08", "2026-08-08")
        for lead in range(1, 8)
    ]
    con.executemany("INSERT INTO seasonal_forecasts VALUES (?,?,?,?,?)", rows)
    return con


_NMME_ENTRY = {
    "table": "seasonal_forecasts",
    "iso3_column": "iso3",
    "value_column": "value",
    "date_column": "forecast_issue_date",
    "date_offset_column": "lead_months",
    "where": "variable = 'prate'",
}


def test_the_span_is_the_months_the_table_speaks_for(nmme):
    """Issued July at lead 1 is ABOUT August; July is not covered."""

    span = di._table_date_span(nmme, _NMME_ENTRY)
    assert span is not None
    assert span[0] == "2026-08", (
        "the earliest month a vintage speaks for is issue + lead, not the "
        "issue date — comparing the issue date made July look covered"
    )
    assert span[1] == "2027-03"


def test_a_month_before_the_first_vintage_is_structural_not_a_fault(nmme):
    """2026-07 predates coverage, so it is INFO and its own reason code."""

    span = di._table_date_span(nmme, _NMME_ENTRY)
    assert span is not None and "2026-07" < span[0], (
        "July must read as predating coverage so the gate records "
        "predates_table_coverage rather than warning once per month"
    )
    assert not ("2026-08" < span[0]), "August is covered and must not be excused"


def test_a_table_with_no_lead_column_keeps_its_own_dates(nmme):
    """The offset is applied only where the entry declares one."""

    entry = {k: v for k, v in _NMME_ENTRY.items() if k != "date_offset_column"}
    assert di._table_date_span(nmme, entry) == ("2026-07", "2026-08")


def test_the_span_never_raises_on_a_missing_table(tmp_path):
    con = duckdb.connect(str(tmp_path / "empty.duckdb"))
    assert di._table_date_span(con, _NMME_ENTRY) is None


# ---------------------------------------------------------------------------
# A2: the NMME vintage backfill
# ---------------------------------------------------------------------------


def test_earlier_issue_months_walk_back_oldest_first():
    from resolver.tools.ingest_nmme import _earlier_issue_months

    assert _earlier_issue_months("2026-07-08", 3) == ["202604", "202605", "202606"]


def test_earlier_issue_months_cross_the_year_boundary():
    from resolver.tools.ingest_nmme import _earlier_issue_months

    assert _earlier_issue_months("2026-01-08", 2) == ["202511", "202512"]


def test_a_vintage_the_archive_lacks_is_named_not_raised(monkeypatch):
    """CPC's realtime_anom is a rolling window; an absent month is ordinary."""

    from resolver.tools import ingest_nmme

    def _fetch(year_month, max_leads):
        if year_month == "202606":
            raise FileNotFoundError(f"NMME directory {year_month}0800/ not found")
        raise FileNotFoundError("not held")

    monkeypatch.setattr("resolver.ingestion.nmme.fetch_and_process", _fetch)
    out = ingest_nmme._backfill_earlier_issues(
        con=None, months=2, newest_issue="2026-07-08", max_leads=7
    )
    assert out["recovered"] == []
    assert set(out["absent"]) == {"202605", "202606"}
    assert out["rows_written"] == 0
    assert out["failed"] == {}
