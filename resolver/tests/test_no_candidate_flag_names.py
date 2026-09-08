# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""A flagged row must say WHICH finding flagged it.

``reconcile``'s no-candidate branch has always set ``flagged = TRUE`` and
carried ``FLAG_NO_CANDIDATE`` on the Reconciliation object, but the
provenance it wrote set ``"decision": consulted`` bare — no ``flags`` key —
while the resolved-value branch and ``drought`` both write
``{**consulted, "conflicts": [...], "flags": [...]}``. The flag name reached
nothing durable, so a reader with the row in hand could see the machine
doubted its answer and not which of the four findings it doubted it for.

Two halves, tested here: the writer now names the flag, and the backfill
names it on the rows written before it did.
"""

from __future__ import annotations

import datetime as dt
import json

import duckdb
import pytest

from resolver.hazard_resolution import reconcile as reconcile_mod
from resolver.hazard_resolution import resolutions as res_mod
from resolver.hazard_resolution.schema import ensure_haz_schema
from resolver.tests.hazard_resolution_utils import make_candidate, make_rulebook

YM = "2024-03"
AFTER_FREEZE = dt.date(2024, 8, 1)


@pytest.fixture()
def rulebook():
    return make_rulebook()


@pytest.fixture()
def con():
    con = duckdb.connect(":memory:")
    ensure_haz_schema(con)
    return con


def _reconcile(candidates, rulebook, **kwargs):
    kwargs.setdefault("today", AFTER_FREEZE)
    return reconcile_mod.reconcile(
        iso3="PHL", ym=YM, hazard="FL", candidates=candidates,
        rulebook=rulebook, **kwargs,
    )


def _stored_decision(con, iso3="PHL"):
    row = con.execute(
        "SELECT provenance_json FROM haz_resolutions WHERE iso3 = ?", [iso3]
    ).fetchone()
    assert row is not None, "no resolution row was written"
    return json.loads(row[0])["decision"]


# ---------------------------------------------------------------------------
# The writer names the flag
# ---------------------------------------------------------------------------


def test_no_candidate_row_names_its_flag_in_provenance(con, rulebook):
    """The acceptance test: a stored NO_DATA row says why it is flagged."""
    verdict = _reconcile([], rulebook)
    assert res_mod.write_reconciliation(con, verdict, rulebook, today=AFTER_FREEZE)

    decision = _stored_decision(con)
    assert decision["flags"] == [reconcile_mod.FLAG_NO_CANDIDATE]


def test_no_candidate_row_carries_the_same_shape_as_a_resolved_value(con, rulebook):
    """One shape across the branches, or a reader needs two readers."""
    no_data = _reconcile([], rulebook)
    resolved = _reconcile([make_candidate("emdat", 4_000.0)], rulebook)

    assert set(no_data.provenance["decision"]) == set(resolved.provenance["decision"])
    assert no_data.provenance["decision"]["conflicts"] == []


def test_the_stored_flag_matches_the_reconciliation_object(con, rulebook):
    """The row and the object must not be able to disagree."""
    verdict = _reconcile([], rulebook)
    res_mod.write_reconciliation(con, verdict, rulebook, today=AFTER_FREEZE)
    assert _stored_decision(con)["flags"] == list(verdict.flags)


def test_the_consulted_evidence_survives_the_flags_key(con, rulebook):
    """Adding the flag must not cost the rung accounting beside it."""
    verdict = _reconcile([], rulebook)
    decision = verdict.provenance["decision"]
    assert decision["rungs_populated"] == []
    assert "rungs_empty" in decision


# ---------------------------------------------------------------------------
# The backfill names it on rows written before the fix
# ---------------------------------------------------------------------------


def _insert_legacy_row(con, iso3, *, flagged=True, rule=None, decision=None):
    """A row in the pre-fix shape: flagged, with no `decision.flags`."""
    provenance = {
        "source": "ladder",
        "source_record_ids": [],
        "source_urls": [],
        "retrieved_at": None,
        "rule_fired": rule or reconcile_mod.RULE_NO_CANDIDATE,
        "decision": decision
        if decision is not None
        else {"rungs_populated": [], "rungs_empty": ["emdat", "ifrc_go"]},
        "note": "written before the flag was named",
    }
    con.execute(
        """
        INSERT INTO haz_resolutions
            (iso3, year, month, hazard, status, value, provenance_json,
             rule_fired, flagged, provisional, run_type, frozen_at)
        VALUES (?, 2024, 3, 'FL', 'NO_DATA', NULL, ?, ?, ?, FALSE, 'backcast',
                TIMESTAMP '2024-05-30 00:00:00')
        """,
        [
            iso3,
            json.dumps(provenance),
            rule or reconcile_mod.RULE_NO_CANDIDATE,
            flagged,
        ],
    )


def test_backfill_names_the_flag_on_a_legacy_row(con):
    _insert_legacy_row(con, "PHL")
    assert res_mod.backfill_no_candidate_flags(con) == 1
    assert _stored_decision(con)["flags"] == [reconcile_mod.FLAG_NO_CANDIDATE]
    assert _stored_decision(con)["conflicts"] == []


def test_backfill_leaves_status_value_and_freeze_stamp_alone(con):
    """Provenance only. The answer and its freeze deadline never move."""
    _insert_legacy_row(con, "PHL")
    before = con.execute(
        "SELECT status, value, flagged, provisional, rule_fired, frozen_at, run_type"
        " FROM haz_resolutions"
    ).fetchone()

    res_mod.backfill_no_candidate_flags(con)

    after = con.execute(
        "SELECT status, value, flagged, provisional, rule_fired, frozen_at, run_type"
        " FROM haz_resolutions"
    ).fetchone()
    assert before == after


def test_backfill_keeps_the_rung_accounting_it_found(con):
    """A merge patch, not a replacement: the consulted evidence survives."""
    _insert_legacy_row(con, "PHL")
    res_mod.backfill_no_candidate_flags(con)

    decision = _stored_decision(con)
    assert decision["rungs_empty"] == ["emdat", "ifrc_go"]
    assert json.loads(
        con.execute("SELECT provenance_json FROM haz_resolutions").fetchone()[0]
    )["note"] == "written before the flag was named"


def test_backfill_is_idempotent(con):
    _insert_legacy_row(con, "PHL")
    assert res_mod.backfill_no_candidate_flags(con) == 1
    assert res_mod.backfill_no_candidate_flags(con) == 0
    assert _stored_decision(con)["flags"] == [reconcile_mod.FLAG_NO_CANDIDATE]


def test_backfill_skips_an_unflagged_row(con):
    """`flagged` false means the machine did not doubt it — nothing to name."""
    _insert_legacy_row(con, "PHL", flagged=False)
    assert res_mod.backfill_no_candidate_flags(con) == 0
    assert "flags" not in _stored_decision(con)


def test_backfill_skips_another_rule(con):
    """Scoped to the branch that has the defect, not every flagged row."""
    _insert_legacy_row(con, "PHL", rule="ladder:emdat")
    assert res_mod.backfill_no_candidate_flags(con) == 0
    assert "flags" not in _stored_decision(con)


def test_backfill_never_overwrites_a_flag_already_named(con):
    _insert_legacy_row(
        con,
        "PHL",
        decision={"rungs_populated": [], "conflicts": [], "flags": ["ceiling_exceeded"]},
    )
    assert res_mod.backfill_no_candidate_flags(con) == 0
    assert _stored_decision(con)["flags"] == ["ceiling_exceeded"]


def test_backfill_dry_run_counts_and_writes_nothing(con):
    _insert_legacy_row(con, "PHL")
    assert res_mod.backfill_no_candidate_flags(con, dry_run=True) == 1
    assert "flags" not in _stored_decision(con)


def test_backfill_repairs_every_matching_row(con):
    for i, iso3 in enumerate(("PHL", "IDN", "VNM", "MMR")):
        _insert_legacy_row(con, iso3)
    assert res_mod.backfill_no_candidate_flags(con) == 4
    named = con.execute(
        "SELECT COUNT(*) FROM haz_resolutions"
        " WHERE json_extract(provenance_json, '$.decision.flags') IS NOT NULL"
    ).fetchone()[0]
    assert named == 4


def test_backfill_on_an_empty_table_is_a_no_op(con):
    assert res_mod.backfill_no_candidate_flags(con) == 0


def test_the_base_rate_rebuild_runs_the_backfill(con, monkeypatch):
    """One call site covers the live months and the backcast alike."""
    from resolver.hazard_resolution import base_rates

    _insert_legacy_row(con, "PHL")
    monkeypatch.setattr(base_rates, "compute_occurrence", lambda *a, **k: None)
    monkeypatch.setattr(base_rates, "compute_severity", lambda *a, **k: None)

    base_rates.compute_all(con, make_rulebook(), today=AFTER_FREEZE)

    assert _stored_decision(con)["flags"] == [reconcile_mod.FLAG_NO_CANDIDATE]
