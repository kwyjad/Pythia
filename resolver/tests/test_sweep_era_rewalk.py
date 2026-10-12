# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The one-time retraction of rows written under the pre-2026-09-17 sweep.

Pins the predicate (exactly the defective rows, none of the look-alikes),
that a dry run writes nothing, that apply deletes, resets and frees the
ledger once, that a second apply is a no-op, that the integrity check fails
when an untouched row moves, and that the freeze guard lets the backcast
write the first answer for a re-walked cell without logging a revision.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import replace as dc_replace

import duckdb
import pytest

from resolver.hazard_resolution import reconcile as reconcile_mod
from resolver.hazard_resolution import resolutions as res_mod
from resolver.hazard_resolution import sweep_era_rewalk as sw
from resolver.hazard_resolution.schema import ensure_haz_schema
from resolver.tests.hazard_resolution_utils import make_candidate, make_rulebook

BEFORE_FIX = "2026-08-20 01:00:00"
AFTER_FIX = "2026-09-20 01:00:00"


def _prov(*, populated=(), figures=0, trigger_note="") -> str:
    return json.dumps({
        "source": "ladder",
        "rule_fired": sw.RULE,
        "reliefweb_extraction": {"extraction": {"figures_extracted": figures}},
        "decision": {
            "rungs_populated": list(populated),
            "rungs_empty": ["emdat", "reliefweb_extracted", "ifrc_go", "idmc_idu"],
            "sources_unavailable": ["emdat"],
            "flags": ["no_candidate_past_freeze"],
            "note": trigger_note,
        },
    })


def _res(con, iso3, year, month, hazard, *, status="NO_DATA", rule=sw.RULE,
         created=BEFORE_FIX, prov=None, value=None):
    con.execute(
        """
        INSERT INTO haz_resolutions
            (iso3, year, month, hazard, status, value, provenance_json, rule_fired,
             flagged, provisional, run_type, frozen_at, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, TRUE, FALSE, 'backcast', ?, ?)
        """,
        [iso3, year, month, hazard, status, value, prov or _prov(), rule,
         dt.datetime(year, month, 28) + dt.timedelta(days=60), created],
    )


def _trig(con, iso3, year, month, hazard, *, triggered=False, source="none"):
    con.execute(
        """
        INSERT INTO haz_triggers
            (iso3, year, month, hazard, triggered, trigger_source,
             trigger_detail_json, run_type, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'backcast', ?)
        """,
        [iso3, year, month, hazard, triggered, source,
         json.dumps({"kept": "yes"}), AFTER_FIX],
    )


def _ledger(con, hazard, ym):
    con.execute(
        "INSERT INTO haz_backcast_progress (hazard, ym, status, cells) VALUES (?, ?, 'ok', 5)",
        [hazard, ym],
    )


@pytest.fixture()
def con():
    con = duckdb.connect(":memory:")
    ensure_haz_schema(con)
    # The two defective shapes.
    _res(con, "AFG", 2019, 3, "FL"); _trig(con, "AFG", 2019, 3, "FL")
    _res(con, "COD", 2015, 6, "TC")
    _trig(con, "COD", 2015, 6, "TC", triggered=True, source="reliefweb_sweep")
    # Look-alikes, none of which may be selected.
    _res(con, "PAK", 2019, 3, "FL", prov=_prov(figures=4))           # read figures
    _trig(con, "PAK", 2019, 3, "FL")
    _res(con, "BGD", 2019, 3, "FL", prov=_prov(populated=["ifrc_go"]))  # rung populated
    _trig(con, "BGD", 2019, 3, "FL")
    _res(con, "IND", 2019, 3, "FL"); _trig(con, "IND", 2019, 3, "FL", triggered=True, source="gdacs")
    _res(con, "PHL", 2015, 6, "TC"); _trig(con, "PHL", 2015, 6, "TC", triggered=True, source="ibtracs")
    _res(con, "VNM", 2019, 3, "FL", created=AFTER_FIX); _trig(con, "VNM", 2019, 3, "FL")
    _res(con, "MOZ", 2019, 3, "FL", status="RESOLVED_ZERO",
         rule="flood_zero:no_gdacs_trigger+reliefweb_silent", value=0.0)
    _trig(con, "MOZ", 2019, 3, "FL")
    _res(con, "MWI", 2019, 3, "FL", status="RESOLVED_VALUE", rule="ladder:ifrc_go", value=900.0)
    _trig(con, "MWI", 2019, 3, "FL")
    _res(con, "ETH", 2019, 3, "DR"); _trig(con, "ETH", 2019, 3, "DR")
    # Ledger: the two affected months plus one that must not be freed.
    _ledger(con, "FL", "2019-03"); _ledger(con, "TC", "2015-06"); _ledger(con, "FL", "2019-04")
    # One revision, which must survive whole.
    con.execute(
        "INSERT INTO haz_revisions (iso3, year, month, hazard, source, source_ref, "
        "old_value, new_value, rule_fired) VALUES ('AFG', 2019, 3, 'FL', 'ladder', 'x', NULL, NULL, ?)",
        [sw.RULE],
    )
    return con


def _selected(con):
    sw.stage_selection(con)
    return sorted(
        (r[0], r[3]) for r in con.execute(f"SELECT * FROM {sw.KEYS_TABLE}").fetchall()
    )


def _snapshot(con):
    return {
        t: con.execute(f"SELECT * FROM {t} ORDER BY ALL").fetchall()
        for t in ("haz_resolutions", "haz_triggers", "haz_backcast_progress",
                  "haz_revisions", "haz_feed_restale")
    }


def test_the_selector_takes_exactly_the_defective_rows(con):
    assert _selected(con) == [("AFG", "FL"), ("COD", "TC")]


def test_the_cutoff_is_the_merge_of_the_scoped_sweep():
    assert sw.SWEEP_FIX_CUTOFF == "2026-09-17 08:51:01"
    assert "e97a136a" in sw.SWEEP_FIX_COMMIT


def test_a_dry_run_writes_nothing(con, tmp_path):
    before = _snapshot(con)
    sw.stage_selection(con)
    plan = sw.selection_counts(con)
    sw.budget_estimate(con, make_rulebook())
    sw.export_audit(con, tmp_path / "audit")
    assert plan["rows"] == 2 and plan["ledger_rows_to_free"] == 2
    assert _snapshot(con) == before
    assert (tmp_path / "audit" / "haz_resolutions.csv").read_text().count("\n") == 3


def test_apply_deletes_resets_and_frees_the_ledger_once(con):
    sw.stage_selection(con)
    before = sw.integrity_report(con)
    result = sw.apply_retraction(con)
    after = sw.integrity_report(con)
    deleted = {tuple(k.split("/", 1)): v for k, v in result["deleted"].items()}
    assert sw.compare_reports(before, after, deleted) == []
    assert result["applied"] and deleted == {("FL", "NO_DATA"): 1, ("TC", "NO_DATA"): 1}
    left = {r[0] for r in con.execute(
        "SELECT iso3 FROM haz_resolutions WHERE status = 'NO_DATA'").fetchall()}
    assert "AFG" not in left and "COD" not in left
    assert {"PAK", "BGD", "IND", "PHL", "VNM", "ETH"} <= left
    trig = con.execute(
        "SELECT triggered, trigger_source, trigger_detail_json FROM haz_triggers "
        "WHERE iso3 = 'COD' AND hazard = 'TC'").fetchone()
    assert trig[0] is False and trig[1] == "none"
    detail = json.loads(trig[2])
    assert detail["no_row_reason"] == sw.PENDING_REASON and detail["kept"] == "yes"
    assert json.loads(con.execute(
        "SELECT trigger_detail_json FROM haz_triggers WHERE iso3 = 'AFG' AND hazard = 'FL'"
    ).fetchone()[0])["no_row_reason"] == sw.PENDING_REASON
    assert con.execute(
        "SELECT hazard, ym FROM haz_backcast_progress ORDER BY ALL").fetchall() == [("FL", "2019-04")]
    assert con.execute(
        "SELECT feed, hazard, token, months, rows_deleted FROM haz_feed_restale ORDER BY hazard"
    ).fetchall() == [
        (sw.FEED, "FL", sw.TOKEN, "2019-03", 1),
        (sw.FEED, "TC", sw.TOKEN, "2015-06", 1),
    ]
    assert con.execute("SELECT COUNT(*) FROM haz_revisions").fetchone()[0] == 1

    # A month the backcast re-walks gets a ledger row back; a second apply
    # must never free it again.
    _ledger(con, "FL", "2019-03")
    snapshot = _snapshot(con)
    sw.stage_selection(con)
    again = sw.apply_retraction(con)
    assert again["applied"] is False
    assert _snapshot(con) == snapshot


def test_the_token_blocks_a_second_apply_even_if_rows_reappear(con):
    sw.stage_selection(con)
    sw.apply_retraction(con)
    # A row matching the predicate again (impossible in production: a new
    # row is created after the cutoff) is not retracted under the same token.
    _res(con, "AFG", 2019, 3, "FL")
    sw.stage_selection(con)
    result = sw.apply_retraction(con)
    assert result["applied"] is False and "already applied" in result["reason"]


def test_the_integrity_check_fails_when_an_untouched_row_moves(con):
    sw.stage_selection(con)
    before = sw.integrity_report(con)
    result = sw.apply_retraction(con)
    con.execute("UPDATE haz_resolutions SET value = 1.0 WHERE iso3 = 'MWI'")
    after = sw.integrity_report(con)
    deleted = {tuple(k.split("/", 1)): v for k, v in result["deleted"].items()}
    problems = sw.compare_reports(before, after, deleted)
    assert any("haz_resolutions_untouched" in p for p in problems)


def test_the_integrity_check_fails_when_a_count_moves_unexplained(con):
    sw.stage_selection(con)
    before = sw.integrity_report(con)
    con.execute("DELETE FROM haz_backcast_progress WHERE ym = '2019-04'")
    after = sw.integrity_report(con)
    assert sw.compare_reports(before, after, {})


def test_the_cli_dry_run_and_apply(tmp_path):
    path = tmp_path / "db.duckdb"
    c = duckdb.connect(str(path))
    ensure_haz_schema(c)
    _res(c, "AFG", 2019, 3, "FL"); _trig(c, "AFG", 2019, 3, "FL"); _ledger(c, "FL", "2019-03")
    c.close()
    assert sw.main(["--db", str(path), "--audit-dir", str(tmp_path / "a"),
                    "--json-out", str(tmp_path / "p.json")]) == 0
    c = duckdb.connect(str(path))
    assert c.execute("SELECT COUNT(*) FROM haz_resolutions").fetchone()[0] == 1
    c.close()
    assert sw.main(["--db", str(path), "--apply", "--json-out", str(tmp_path / "x.json")]) == 0
    c = duckdb.connect(str(path))
    assert c.execute("SELECT COUNT(*) FROM haz_resolutions").fetchone()[0] == 0
    assert c.execute("SELECT COUNT(*) FROM haz_backcast_progress").fetchone()[0] == 0
    c.close()
    payload = json.loads((tmp_path / "x.json").read_text())
    assert payload["problems"] == [] and payload["plan"]["rows"] == 1


# ---------------------------------------------------------------------------
# The freeze guard: a re-walked cell has no row, so the backcast writes one
# ---------------------------------------------------------------------------

AFTER_FREEZE = dt.date(2026, 10, 12)


def test_a_rewalked_cell_gets_a_new_frozen_zero_and_no_revision(con):
    rb = make_rulebook()
    sw.stage_selection(con)
    sw.apply_retraction(con)
    revisions = con.execute("SELECT COUNT(*) FROM haz_revisions").fetchone()[0]
    outcome = res_mod.write_zero_resolution(
        con, iso3="AFG", year=2019, month=3, hazard="FL",
        evidence_of_absence={"reliefweb": {"queries": []}}, rulebook=rb,
        today=AFTER_FREEZE, run_type="backcast",
    )
    assert outcome == res_mod.WRITE_WRITTEN
    row = con.execute(
        "SELECT status, provisional, frozen_at FROM haz_resolutions "
        "WHERE iso3 = 'AFG' AND year = 2019 AND month = 3 AND hazard = 'FL'"
    ).fetchone()
    assert row[0] == "RESOLVED_ZERO" and row[1] is False
    assert row[2].date() < AFTER_FREEZE
    assert con.execute("SELECT COUNT(*) FROM haz_revisions").fetchone()[0] == revisions
    # The new row is frozen in its turn: the next attempt is a revision.
    again = res_mod.write_zero_resolution(
        con, iso3="AFG", year=2019, month=3, hazard="FL",
        evidence_of_absence={"reliefweb": {"queries": []}}, rulebook=rb,
        today=AFTER_FREEZE, run_type="backcast",
    )
    assert again == res_mod.WRITE_FROZEN_SKIP


def test_a_rewalked_cell_the_ladder_answers_is_written_without_a_revision(con):
    rb = make_rulebook()
    sw.stage_selection(con)
    sw.apply_retraction(con)
    revisions = con.execute("SELECT COUNT(*) FROM haz_revisions").fetchone()[0]
    cand = dc_replace(make_candidate("ifrc_go", 4_000), iso3="COD", ym="2015-06", hazard="TC")
    verdict = reconcile_mod.reconcile(
        iso3="COD", ym="2015-06", hazard="TC", candidates=[cand],
        rulebook=rb, today=AFTER_FREEZE,
    )
    assert res_mod.write_reconciliation(
        con, verdict, rb, today=AFTER_FREEZE, run_type="backcast"
    ) == res_mod.WRITE_WRITTEN
    assert con.execute("SELECT COUNT(*) FROM haz_revisions").fetchone()[0] == revisions


# ---------------------------------------------------------------------------
# The workflow mode
# ---------------------------------------------------------------------------


def test_the_workflow_mode_audits_always_and_retracts_before_the_guarded_upload():
    import pathlib

    import yaml

    path = pathlib.Path(__file__).resolve().parents[2] / ".github/workflows/compact_resolver_db.yml"
    data = yaml.safe_load(path.read_text())
    on = data.get("on") or data.get(True)
    assert "sweep_era_rewalk" in on["workflow_dispatch"]["inputs"]["maintenance"]["options"]
    gate_if = data["jobs"]["gate"]["if"]
    assert "sweep_era_rewalk" in gate_if
    job = data["jobs"]["compact"]
    assert job["concurrency"]["group"] == "pythia-resolver-db"
    names = [s.get("name", "") for s in job["steps"]]
    audit = job["steps"][names.index("Sweep-era re-walk: upload the audit copy")]
    assert audit["with"]["name"] == "pythia-audit-sweep-era-rows"
    assert audit["with"]["retention-days"] == 90
    assert "APPLY" not in audit["if"]
    plan = job["steps"][names.index("Sweep-era re-walk: plan, estimate and audit copy")]
    assert "--audit-only" in plan["run"] and "APPLY" not in plan["if"]
    apply_idx = names.index("Sweep-era re-walk: retract the rows and free the ledger months")
    assert "env.APPLY == 'true'" in job["steps"][apply_idx]["if"]
    assert apply_idx < names.index("Refuse to upload over a newer canonical DB")
    assert names.index("Refuse to upload over a newer canonical DB") < names.index(
        "Upload compacted canonical DB")
