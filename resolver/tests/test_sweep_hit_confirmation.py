# Pythia / Copyright (c) 2025 Kevin Wyjad
"""A ReliefWeb sweep hit confirmed by the ladder is an occurrence.

The 17 Sept 2026 fix stopped a bare sweep hit from TRIGGERING a cell, which
was right: it had become 95% of every trigger. But it left the cells whose
reports DID state a figure about the country and month undecided too, and 94
live cells of run 36401252026 kept rows written under the old rule beside
trigger rows saying they were undecided. Now the ladder is walked for a
sweep-hit cell: an admissible figure promotes it, anything less writes
nothing and retracts an unfrozen row nothing supports.
"""

from __future__ import annotations

import datetime as dt
import json

import duckdb
import pytest

from resolver.hazard_resolution import candidates as cand_mod
from resolver.hazard_resolution import detect as detect_mod
from resolver.hazard_resolution import impact as impact_mod
from resolver.hazard_resolution.schema import ensure_haz_schema
from resolver.tests.hazard_resolution_utils import (
    make_candidate,
    make_rulebook,
    seed_resolution,
    seed_trigger,
)

YM = "2026-07"
BEFORE_FREEZE = dt.date(2026, 8, 15)


@pytest.fixture()
def con():
    con = duckdb.connect(":memory:")
    ensure_haz_schema(con)
    return con


@pytest.fixture()
def rulebook():
    return make_rulebook()


def _sweep_hit(con, iso3, *, silent=False, inconclusive=False):
    seed_trigger(con, iso3=iso3, ym=YM, hazard="FL", triggered=False,
                 trigger_source="none")
    detect_mod.record_sweep_hit(
        con, hazard="FL", iso3=iso3, ym=YM,
        sweep_evidence={"silent": silent, "inconclusive": inconclusive,
                        "total_hits": 0 if silent else 27},
    )


def _stub_candidates(monkeypatch, by_iso3):
    def fake(con, iso3, ym, hazard, rulebook, extracted=None):
        return [
            make_candidate("ifrc_go", value, hazard=hazard, iso3=iso3, ym=ym)
            for value in by_iso3.get(iso3, [])
        ]

    monkeypatch.setattr(cand_mod, "build_candidates", fake)
    monkeypatch.setattr(cand_mod, "write_candidates", lambda *a, **k: None)


def _trigger(con, iso3):
    return con.execute(
        "SELECT triggered, trigger_source, trigger_detail_json FROM haz_triggers "
        "WHERE iso3 = ? AND hazard = 'FL' AND year = 2026 AND month = 7",
        [iso3],
    ).fetchone()


def _resolution(con, iso3):
    return con.execute(
        "SELECT status, value FROM haz_resolutions "
        "WHERE iso3 = ? AND hazard = 'FL' AND year = 2026 AND month = 7",
        [iso3],
    ).fetchone()


def test_only_hits_are_confirmation_candidates(con):
    _sweep_hit(con, "AFG")
    _sweep_hit(con, "ISL", silent=True)
    _sweep_hit(con, "NPL", inconclusive=True)
    seed_trigger(con, iso3="PAK", ym=YM, hazard="FL", triggered=True,
                 trigger_source="gdacs")
    assert detect_mod.sweep_hit_iso3s(con, YM, "FL") == ["AFG"]


def test_a_figure_promotes_the_cell(con, rulebook, monkeypatch):
    _sweep_hit(con, "AFG")
    _stub_candidates(monkeypatch, {"AFG": [55_000.0]})
    run = impact_mod.resolve_triggered_cells(
        con, ym=YM, hazard="FL", iso3s=[], confirm_iso3s=["AFG"],
        rulebook=rulebook, extract=False, today=BEFORE_FREEZE,
    )
    assert run.confirmed == 1 and run.unconfirmed == 0
    triggered, source, detail = _trigger(con, "AFG")
    assert triggered is True
    assert source == detect_mod.TRIGGER_SOURCE_RELIEFWEB_LADDER
    assert json.loads(detail)["ladder_confirmation"]["value"] == 55_000.0
    assert _resolution(con, "AFG") == ("RESOLVED_VALUE", 55_000.0)


def test_no_figure_writes_nothing_and_retracts_a_stale_row(con, rulebook, monkeypatch):
    # The row a pre-fix run wrote when a sweep hit still triggered the cell.
    seed_resolution(con, iso3="AFG", ym=YM, hazard="FL", value=55_000.0,
                    provisional=True, with_trigger=False)
    _sweep_hit(con, "AFG")
    _stub_candidates(monkeypatch, {})
    run = impact_mod.resolve_triggered_cells(
        con, ym=YM, hazard="FL", iso3s=[], confirm_iso3s=["AFG"],
        rulebook=rulebook, extract=False, today=BEFORE_FREEZE,
    )
    assert run.unconfirmed == 1 and run.retracted == 1
    assert _resolution(con, "AFG") is None, "never a NO_DATA for an undecided cell"
    triggered, source, _ = _trigger(con, "AFG")
    assert triggered is False


def test_a_frozen_row_is_never_retracted(con, rulebook, monkeypatch):
    seed_resolution(con, iso3="AFG", ym=YM, hazard="FL", value=55_000.0,
                    frozen_at="2026-09-29 00:00:00", with_trigger=False)
    _sweep_hit(con, "AFG")
    _stub_candidates(monkeypatch, {})
    run = impact_mod.resolve_triggered_cells(
        con, ym=YM, hazard="FL", iso3s=[], confirm_iso3s=["AFG"],
        rulebook=rulebook, extract=False, today=dt.date(2026, 10, 5),
    )
    assert run.retracted == 0
    assert _resolution(con, "AFG") is not None


def test_a_detected_cell_is_walked_as_before(con, rulebook, monkeypatch):
    seed_trigger(con, iso3="PAK", ym=YM, hazard="FL", triggered=True,
                 trigger_source="gdacs")
    _stub_candidates(monkeypatch, {})
    run = impact_mod.resolve_triggered_cells(
        con, ym=YM, hazard="FL", iso3s=["PAK"], confirm_iso3s=[],
        rulebook=rulebook, extract=False, today=dt.date(2026, 12, 1),
    )
    # A detected cell past its freeze with no candidate is still NO_DATA.
    assert run.no_data == 1
    assert _resolution(con, "PAK")[0] == "NO_DATA"


def test_a_confirmed_cell_counts_as_an_occurrence(con, rulebook, monkeypatch):
    """A bare sweep hit leaves both sides of the rate; a confirmed one is a
    triggered year like any other."""

    _sweep_hit(con, "AFG")
    _stub_candidates(monkeypatch, {"AFG": [55_000.0]})
    impact_mod.resolve_triggered_cells(
        con, ym=YM, hazard="FL", iso3s=[], confirm_iso3s=["AFG"],
        rulebook=rulebook, extract=False, today=BEFORE_FREEZE,
    )
    row = con.execute(
        "SELECT COUNT(*) FROM haz_triggers WHERE hazard = 'FL' AND triggered "
        f"AND trigger_source <> '{detect_mod.TRIGGER_SOURCE_RELIEFWEB}'"
    ).fetchone()
    # compute_occurrence counts triggered rows whose source is not the bare
    # sweep; the confirmed cell is one of them.
    assert row[0] == 1


def test_the_switch_is_in_the_shipped_rulebook_and_validated():
    from resolver.hazard_resolution.rulebook import load_rulebook

    rb = load_rulebook()
    assert rb.get("flood.reliefweb_sweep.confirm_hits_with_ladder") is True
    assert rb.get("cyclone.reliefweb_sweep.confirm_hits_with_ladder") is True


def test_a_budget_stop_never_retracts(con, rulebook, monkeypatch):
    """Documents left unread are not a failed confirmation."""

    seed_resolution(con, iso3="AFG", ym=YM, hazard="FL", value=55_000.0,
                    provisional=True, with_trigger=False)
    _sweep_hit(con, "AFG")
    _stub_candidates(monkeypatch, {})
    monkeypatch.setattr(
        impact_mod, "extract_rung_for_cell",
        lambda *a, **k: ([], {"extraction": {"budget_capped": True}}),
    )
    run = impact_mod.resolve_triggered_cells(
        con, ym=YM, hazard="FL", iso3s=[], confirm_iso3s=["AFG"],
        rulebook=rulebook, extract=True, today=BEFORE_FREEZE,
    )
    assert run.unconfirmed == 1 and run.retracted == 0
    assert _resolution(con, "AFG") is not None
