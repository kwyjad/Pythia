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


def _reconcile(candidates, rulebook, **kwargs):
    kwargs.setdefault("today", AFTER_FREEZE)
    return reconcile_mod.reconcile(
        iso3="PHL", ym=YM, hazard="FL", candidates=candidates,
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
        rulebook,
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
        rulebook,
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
        [make_candidate("gdacs", 700_000)], rulebook, None
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
