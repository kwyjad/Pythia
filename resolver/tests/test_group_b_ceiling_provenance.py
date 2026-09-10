# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Group B of the run-34124705852 repairs: a flag that names nothing.

The run resolved 4,999 values and flagged 2,089 of them. Every one of those
rendered as the single word ``True``. ``flagged`` covers four findings — a
figure above the GDACS exposure ceiling, one above the national population,
an order-of-magnitude disagreement between adjacent rungs, and no candidate
past the freeze deadline — and each wants a different repair. Which of them
fired was recorded nowhere a reader looks.

The ceiling was worse. ``decision.ceiling.source`` held a rulebook constant
naming the RULE, never the event that supplied the number, and there was no
event id and no field name at all. So a ceiling breach could not be told
apart from a GDACS enrichment failure, which is precisely the distinction
the extraction path has carried since Sept 2026 and the ladder had not.

And 59,967 NO_DATA rows carried a blank ``rungs_unavailable`` column while
EM-DAT rejected the key on every call of the run. A rung that could not be
READ is not a rung that was empty, and only the second can justify a
NO_DATA. The distinction existed in the run stream, which the nightly
backcast does not write, so for a backcast month it existed nowhere.

None of this lands a row. It is admitted because the GDACS repair cannot be
measured without it: the way to tell whether that work succeeded is whether
``ceiling_basis`` moves off ``no_usable_gdacs_exposure``, and today the
bundle cannot say.

Network-free.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import replace as dc_replace

import duckdb
import pytest

from resolver.hazard_resolution import reconcile as reconcile_mod
from resolver.hazard_resolution.candidates import CEILING_FIELD
from resolver.tests.hazard_resolution_utils import make_candidate, make_rulebook

YM = "2024-03"
AFTER_FREEZE = __import__("datetime").date(2024, 8, 1)


@pytest.fixture()
def rulebook():
    return make_rulebook()


#: These cases pin how a GDACS ceiling is RECORDED, which needs a hazard
#: whose GDACS figure really is a population exposure. Flood's is not — see
#: rules.NO_POPULATION_EXPOSURE_HAZARDS — and that is a fact about the feed,
#: not about the provenance machinery under test here.
CEILING_HAZARD = "TC"


def _reconcile(candidates, rulebook, hazard="FL", **kwargs):
    kwargs.setdefault("today", AFTER_FREEZE)
    candidates = [dc_replace(c, hazard=hazard) for c in candidates]
    return reconcile_mod.reconcile(
        iso3="PHL", ym=YM, hazard=hazard, candidates=candidates,
        rulebook=rulebook, **kwargs,
    )


def _ceiling(verdict) -> dict:
    return verdict.provenance["decision"]["ceiling"]


# ---------------------------------------------------------------------------
# B1: the ceiling names the event that supplied it
# ---------------------------------------------------------------------------


def test_the_ceiling_names_the_event_that_supplied_it(rulebook):
    """A ceiling of two and a ceiling of two million want different repairs."""

    verdict = _reconcile(
        [
            make_candidate("emdat", 40_000),
            make_candidate("gdacs", 900_000, source_ref="gdacs-event-1104004"),
        ],
        rulebook, hazard=CEILING_HAZARD,
    )
    ceiling = _ceiling(verdict)
    assert ceiling["basis"] == "gdacs_exposed"
    assert ceiling["exposed_population"] == 900_000
    assert ceiling["source"] == "gdacs", (
        "the ceiling's source must be the thing that supplied the number, "
        "not the rulebook key that says a ceiling applies"
    )
    assert ceiling["source_ref"] == "gdacs-event-1104004", (
        "without the event id a reader auditing a ceiling breach cannot go "
        "and look at the event"
    )
    assert ceiling["field"] == CEILING_FIELD


def test_the_largest_usable_exposure_binds_and_is_the_one_named(rulebook):
    """Several overlapping events bound the month; say which one won."""

    verdict = _reconcile(
        [
            make_candidate("emdat", 1_000),
            make_candidate("gdacs", 50_000, source_ref="small"),
            make_candidate("gdacs", 800_000, source_ref="large"),
        ],
        rulebook, hazard=CEILING_HAZARD,
    )
    ceiling = _ceiling(verdict)
    assert ceiling["exposed_population"] == 800_000
    assert ceiling["source_ref"] == "large"
    assert ceiling["n_events"] == 2
    assert ceiling["n_events_with_exposure"] == 2


def test_a_blank_ceiling_says_why_it_is_blank(rulebook):
    """GDACS listed no event, or listed events and described none of them.

    Only the second is an enrichment failure with a repair, and run
    34124705852 rendered both as the same empty column across 1,280 rows.
    """

    nothing = _reconcile([make_candidate("emdat", 1_000)], rulebook)
    assert _ceiling(nothing)["n_events"] == 0
    assert _ceiling(nothing)["source"] is None
    assert _ceiling(nothing)["field"] is None, (
        "a field named beside a blank ceiling reads as a ceiling that was "
        "evaluated and came out empty, which is a different fault"
    )

    # GDACS listed an event and could not describe it: a zero exposure is
    # GDACS declining to say, and it is the enrichment failure to chase.
    described_none = _reconcile(
        [make_candidate("emdat", 1_000), make_candidate("gdacs", 0.0)], rulebook
    )
    assert _ceiling(described_none)["n_events"] == 1
    assert _ceiling(described_none)["n_events_with_exposure"] == 0
    assert _ceiling(described_none)["exposed_population"] is None


def test_an_exposure_below_the_plausibility_floor_is_counted_as_such(rulebook):
    """Five people is a parse failure, not a national monthly flood total."""

    verdict = _reconcile(
        [make_candidate("emdat", 1_000), make_candidate("gdacs", 5.0)], rulebook
    )
    ceiling = _ceiling(verdict)
    assert ceiling["exposed_population"] is None
    assert ceiling["n_events_below_plausible_floor"] == 1


def test_a_population_share_ceiling_keeps_the_gdacs_counts(rulebook):
    """A share standing in for six silent events is not the same report
    as a share standing in for no events at all."""

    verdict = _reconcile(
        [make_candidate("emdat", 1_000), make_candidate("gdacs", 0.0)],
        rulebook,
        national_population=10_000_000,
    )
    ceiling = _ceiling(verdict)
    assert ceiling["basis"] == "population_share"
    assert ceiling["source"] == "haz_raw_population"
    assert ceiling["n_events"] == 1, (
        "the GDACS counts must survive the fallback, or a reader cannot see "
        "that GDACS had something to say about this cell and did not say it"
    )
    assert ceiling["population_share"] == rulebook.get(
        "sanity.population_fallback_share"
    )


def test_the_rulebook_key_keeps_its_own_name(rulebook):
    """It is still recorded; it simply no longer occupies `source`."""

    verdict = _reconcile([make_candidate("emdat", 1_000)], rulebook)
    assert _ceiling(verdict)["rule_source"] == rulebook.get("sanity.ceiling_source")


def test_effective_ceiling_keeps_its_two_value_contract(rulebook):
    """Existing callers want the number and the word, and still get them."""

    value, basis = reconcile_mod.effective_ceiling(
        [make_candidate("gdacs", 700_000, hazard=CEILING_HAZARD)], rulebook, None
    )
    assert (value, basis) == (700_000, "gdacs_exposed")


# ---------------------------------------------------------------------------
# B2: an unread rung is not an empty one, on the stored row
# ---------------------------------------------------------------------------


def test_an_unreadable_rung_is_named_on_the_row(rulebook):
    """EM-DAT rejected the key on every call of the run; every NO_DATA row
    from that run said nothing about it."""

    verdict = _reconcile([], rulebook, sources_unavailable=["emdat"])
    decision = verdict.provenance["decision"]
    assert decision["rungs_unavailable"] == ["emdat"]
    assert "emdat" in decision["rungs_empty"]
    assert "emdat" not in decision["rungs_empty_and_readable"], (
        "a NO_DATA rests on the rungs that were consulted and had nothing; "
        "an unread rung is in neither category"
    )


def test_a_readable_empty_rung_stays_in_both_lists(rulebook):
    verdict = _reconcile([], rulebook)
    decision = verdict.provenance["decision"]
    assert decision["rungs_unavailable"] == []
    assert decision["rungs_empty"] == decision["rungs_empty_and_readable"]


def test_the_unavailable_rungs_reach_a_resolved_row_too(rulebook):
    """A value resolved on rung 3 while rung 1 was unreadable is a weaker
    claim than the same value with rung 1 consulted and empty."""

    verdict = _reconcile(
        [make_candidate("idmc_idu", 12_000)], rulebook, sources_unavailable=["emdat"]
    )
    assert verdict.status == "RESOLVED_VALUE"
    assert verdict.provenance["decision"]["rungs_unavailable"] == ["emdat"]


# ---------------------------------------------------------------------------
# B3: the bundle reads the provenance the rows carry
# ---------------------------------------------------------------------------


def _bundle():
    import scripts.build_resolver_debug_bundle as bundle

    return bundle


def test_the_bundle_flattens_a_stored_provenance_into_columns():
    bundle = _bundle()
    provenance = json.dumps(
        {
            "winning_rung": "idmc_idu",
            "decision": {
                "flags": ["ceiling_exceeded", "population_cap_exceeded"],
                "rungs_empty": ["emdat", "ifrc_go"],
                "rungs_unavailable": ["emdat"],
                "ceiling": {
                    "basis": "gdacs_exposed",
                    "exposed_population": 900_000.0,
                    "source": "gdacs",
                    "source_ref": "gdacs-event-1104004",
                    "national_population": 115_000_000.0,
                    "n_events": 3,
                    "n_events_with_exposure": 1,
                },
            },
        }
    )
    columns = bundle._provenance_columns(provenance)
    assert columns["answering_rung"] == "idmc_idu"
    assert columns["flags"] == "ceiling_exceeded|population_cap_exceeded"
    assert columns["ceiling"] == 900_000.0
    assert columns["ceiling_basis"] == "gdacs_exposed"
    assert columns["ceiling_source_ref"] == "gdacs-event-1104004"
    assert columns["national_population"] == 115_000_000.0
    assert columns["rungs_unavailable"] == "emdat"
    assert columns["rungs_empty"] == "emdat|ifrc_go"
    assert columns["ceiling_events_seen"] == 3
    assert columns["ceiling_events_with_exposure"] == 1


def test_the_bundle_reads_the_legacy_sources_unavailable_key():
    """Rows stored before the parameter existed carry the other name."""

    bundle = _bundle()
    columns = bundle._provenance_columns(
        json.dumps({"decision": {"sources_unavailable": ["emdat", "ifrc_go"]}})
    )
    assert columns["rungs_unavailable"] == "emdat|ifrc_go"


def test_absent_or_broken_provenance_yields_blanks_not_an_exception():
    bundle = _bundle()
    for payload in (None, "", "not json", json.dumps([1, 2, 3]), json.dumps({})):
        columns = bundle._provenance_columns(payload)
        assert set(columns) == set(bundle.PROVENANCE_COLUMNS)
        assert all(v == "" for v in columns.values())


# ---------------------------------------------------------------------------
# B4: a flag that names nothing fails the bundle
# ---------------------------------------------------------------------------


def _bundle_over(tmp_path, rows):
    """A builder over a database holding exactly these resolutions."""

    bundle = _bundle()
    db = tmp_path / "resolver.duckdb"
    con = duckdb.connect(str(db))
    con.execute(
        "CREATE TABLE haz_resolutions (iso3 TEXT, hazard TEXT, year INTEGER, "
        "month INTEGER, status TEXT, value DOUBLE, flagged BOOLEAN, "
        "provenance_json TEXT)"
    )
    for row in rows:
        con.execute(
            "INSERT INTO haz_resolutions VALUES (?,?,?,?,?,?,?,?)", row
        )
    con.close()
    builder = bundle.BundleBuilder(
        db_path=db,
        out_path=tmp_path / "bundle.zip",
        diagnostics_dir=tmp_path / "diagnostics",
        run_log_dir=tmp_path / "runlog",
        staging=tmp_path / "staging",
    )
    return builder


def _verdict(builder, name):
    return next(c for c in builder.checks if c["name"] == name)


CHECK = "every_flagged_resolution_names_the_flag_it_raised"


def test_a_flagged_row_that_names_no_flag_fails(tmp_path):
    builder = _bundle_over(
        tmp_path,
        [("PHL", "FL", 2024, 3, "RESOLVED_VALUE", 40_000.0, True,
          json.dumps({"decision": {}}))],
    )
    builder._check_flagged_resolutions_name_their_flag()
    check = _verdict(builder, CHECK)
    assert check["verdict"] == "FAIL"
    assert "PHL/FL/2024-03" in check["detail"]


def test_a_ceiling_breach_with_no_ceiling_recorded_fails(tmp_path):
    """It says a bound was exceeded and declines to say what the bound was."""

    builder = _bundle_over(
        tmp_path,
        [("PHL", "FL", 2024, 3, "RESOLVED_VALUE", 40_000.0, True,
          json.dumps({"decision": {"flags": ["ceiling_exceeded"], "ceiling": {}}}))],
    )
    builder._check_flagged_resolutions_name_their_flag()
    assert _verdict(builder, CHECK)["verdict"] == "FAIL"


def test_a_flagged_row_that_names_its_flag_and_bound_passes(tmp_path):
    builder = _bundle_over(
        tmp_path,
        [("PHL", "FL", 2024, 3, "RESOLVED_VALUE", 40_000.0, True,
          json.dumps({
              "decision": {
                  "flags": ["ceiling_exceeded"],
                  "ceiling": {"exposed_population": 900.0, "basis": "gdacs_exposed"},
              }
          }))],
    )
    builder._check_flagged_resolutions_name_their_flag()
    assert _verdict(builder, CHECK)["verdict"] == "PASS"


def test_the_scoped_note_says_the_count_is_almost_all_flood(tmp_path):
    """A raw ceiling_exceeded count reads as the machine distrusting its own
    record, and 94% of it is flood, where the ceiling was a different
    quantity until the commit the note names.
    """

    ceiling = {"exposed_population": 900.0, "basis": "gdacs_exposed",
               "source": "gdacs:population"}
    builder = _bundle_over(
        tmp_path,
        [("PHL", "FL", 2024, 3, "RESOLVED_VALUE", 40_000.0, True,
          json.dumps({"decision": {"flags": ["ceiling_exceeded"], "ceiling": ceiling}})),
         ("BGD", "FL", 2024, 4, "RESOLVED_VALUE", 90_000.0, True,
          json.dumps({"decision": {"flags": ["ceiling_exceeded"], "ceiling": ceiling}})),
         ("VNM", "TC", 2024, 5, "RESOLVED_VALUE", 12_000.0, True,
          json.dumps({"decision": {"flags": ["ceiling_exceeded"], "ceiling": ceiling}}))],
    )
    builder._check_flagged_resolutions_name_their_flag()
    detail = _verdict(builder, CHECK)["detail"]

    from resolver.hazard_resolution.rules import CEILING_EXCEEDED_FIX_COMMIT

    assert "2 of 3 are flood" in detail
    assert "1 cyclone" in detail
    assert CEILING_EXCEEDED_FIX_COMMIT in detail
    # And the BOUND each breach was measured against, because a breach
    # against gdacs_exposed and one against a population share are different
    # statements and only the basis says which.
    assert "gdacs_exposed 3" in detail
    # The note LEADS. Appended after a passing check's detail it was the last
    # line anybody would read, and it is the one that changes what they
    # conclude from the count.
    assert detail.startswith("`ceiling_exceeded` is scoped")


def test_the_flood_rate_reaches_the_register_as_a_measurement(tmp_path):
    """A RATE, not a count. A count also falls when fewer flood cells
    resolve, so it cannot say whether the fix worked."""

    builder = _bundle_over(
        tmp_path,
        [("PHL", "FL", 2024, 3, "RESOLVED_VALUE", 40_000.0, True,
          json.dumps({"decision": {
              "flags": ["ceiling_exceeded"],
              "ceiling": {"exposed_population": 900.0, "basis": "population_share"},
          }})),
         ("BGD", "FL", 2024, 4, "RESOLVED_VALUE", 90_000.0, False, "{}"),
         ("IDN", "FL", 2024, 5, "RESOLVED_VALUE", 1_000.0, False, "{}"),
         ("VNM", "FL", 2024, 6, "RESOLVED_VALUE", 2_000.0, False, "{}")],
    )
    builder._check_flagged_resolutions_name_their_flag()
    issue = next(
        i for i in builder.extra_issues if i.id == "flood_ceiling_exceeded_rate"
    )
    # One flagged of four resolved flood values, none of them pre-fix, so
    # the live residual is the whole 25%.
    assert issue.cost == 25.0
    assert "% of resolved flood values" in issue.cost_unit
    assert "1 of 4" in issue.title and "25.0%" in issue.title


def test_the_pre_fix_share_is_named_apart_from_the_live_residual(tmp_path):
    """A breach citing gdacs_exposed is pre-fix by construction — flood takes
    no GDACS ceiling at any size now — and it is frozen, so it can never
    fall by anything clearing it. Only the residual can still move, and only
    the residual answers "did the fix work"."""

    def _flagged(iso3, month, basis):
        return (iso3, "FL", 2024, month, "RESOLVED_VALUE", 40_000.0, True,
                json.dumps({"decision": {
                    "flags": ["ceiling_exceeded"],
                    "ceiling": {"exposed_population": 900.0, "basis": basis},
                }}))

    builder = _bundle_over(
        tmp_path,
        [_flagged("PHL", 3, "gdacs_exposed"),
         _flagged("BGD", 4, "gdacs_exposed"),
         _flagged("IDN", 5, "gdacs_exposed"),
         _flagged("VNM", 6, "population_share"),
         ("THA", "FL", 2024, 7, "RESOLVED_VALUE", 1_000.0, False, "{}")],
    )
    builder._check_flagged_resolutions_name_their_flag()
    detail = _verdict(builder, CHECK)["detail"]
    assert "4 of 5 flood values resolved (80.0%)" in detail
    assert "3 were" in detail and "predate the fix" in detail
    assert "live residual of 1 (20.0%)" in detail
    issue = next(
        i for i in builder.extra_issues if i.id == "flood_ceiling_exceeded_rate"
    )
    assert issue.cost == 20.0


def test_a_flood_rate_with_no_denominator_says_so_rather_than_dividing(tmp_path):
    """Every flood row flagged and none resolved to a value is possible on a
    scoped run. Printing a rate there would be inventing one."""

    builder = _bundle_over(
        tmp_path,
        [("PHL", "FL", 2024, 3, "NO_DATA", None, True,
          json.dumps({"decision": {
              "flags": ["ceiling_exceeded"],
              "ceiling": {"exposed_population": 900.0, "basis": "gdacs_exposed"},
          }}))],
    )
    builder._check_flagged_resolutions_name_their_flag()
    detail = _verdict(builder, CHECK)["detail"]
    assert "no denominator" in detail or "no flood value resolved" in detail
    issue = next(
        i for i in builder.extra_issues if i.id == "flood_ceiling_exceeded_rate"
    )
    assert issue.cost is None


def test_a_flag_that_is_not_a_ceiling_breach_gets_no_ceiling_note(tmp_path):
    """The note is scoped to the flag it is about."""

    builder = _bundle_over(
        tmp_path,
        [("PHL", "FL", 2024, 3, "NO_DATA", None, True,
          json.dumps({"decision": {"flags": ["no_candidate_past_freeze"]}}))],
    )
    builder._check_flagged_resolutions_name_their_flag()
    detail = _verdict(builder, CHECK)["detail"]
    assert "ceiling_exceeded` is scoped" not in detail
    assert not builder.extra_issues


def test_no_flagged_rows_is_a_pass_not_a_skip(tmp_path):
    builder = _bundle_over(tmp_path, [])
    builder._check_flagged_resolutions_name_their_flag()
    assert _verdict(builder, CHECK)["verdict"] == "PASS"


# ---------------------------------------------------------------------------
# B5: the figures ledger renders why a blank ceiling is blank
# ---------------------------------------------------------------------------


def test_the_figures_ledger_columns_carry_the_ceiling_counts(tmp_path):
    """1,280 rows read no_usable_gdacs_exposure and no reader could tell
    whether GDACS had listed nothing or had listed and said nothing."""

    bundle = _bundle()
    builder = bundle.BundleBuilder(
        db_path=tmp_path / "absent.duckdb",
        out_path=tmp_path / "bundle.zip",
        diagnostics_dir=tmp_path / "diagnostics",
        run_log_dir=tmp_path / "runlog",
        staging=tmp_path / "staging",
    )
    stream = tmp_path / "runlog" / "figures_ledger.jsonl"
    stream.parent.mkdir(parents=True, exist_ok=True)
    stream.write_text(
        json.dumps({
            "iso3": "PHL", "hazard": "FL", "ym": "2024-03", "outcome": "accepted",
            "value": 40_000, "ceiling": None,
            "ceiling_basis": "no_usable_gdacs_exposure",
            "detail": {"ceiling_events_seen": 6, "ceiling_events_with_exposure": 0},
        }) + "\n",
        encoding="utf-8",
    )
    dest = tmp_path / "out"
    dest.mkdir()
    builder._figures_ledger(dest)
    body = "".join(
        line for line in (dest / "figures_ledger.csv").read_text(encoding="utf-8").splitlines(True)
        if not line.startswith("#")
    )
    row = next(iter(csv.DictReader(io.StringIO(body))))
    assert row["ceiling_events_seen"] == "6"
    assert row["ceiling_events_with_exposure"] == "0", (
        "GDACS listed six events for this cell and described none of them; "
        "that is an enrichment failure to chase, and it must be readable "
        "from the table rather than from a JSON blob"
    )
