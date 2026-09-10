# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Standing EM-DAT down must not cost the record of what it cost us.

There is no working EM-DAT credential and there will not be one for a while:
the key was issued to UNICEF for UNICEF work and Pythia is a personal
project. A request you know will be refused is one you should not make — it
was six rejections per ladder run.

The constraint that makes this more than a one-line change is the LEDGER. A
NO_DATA cell records ``rungs_unavailable: emdat``, and that marking is the
only thing that will ever say which cells were decided without their top
rung, and therefore which to re-walk when a credential exists. If standing
the source down made those cells record a clean NO_DATA with no note, the
audit trail would be gone and recovery would be guesswork.

So: **skipped for want of credentials and attempted-then-refused must
produce the same ledger marking. Zero requests, identical record.** That is
what most of this file asserts.
"""

from __future__ import annotations

import json

import duckdb
import pytest

from resolver.diagnostics import issue_sources, issues as issues_mod
from resolver.hazard_resolution import emdat, impact
from resolver.hazard_resolution import rulebook as rulebook_mod
from resolver.hazard_resolution.rulebook import load_rulebook
from resolver.hazard_resolution.schema import ensure_haz_schema

SOURCE = "emdat"


def _con():
    con = duckdb.connect(":memory:")
    ensure_haz_schema(con)
    return con


def _rulebook(reason: str | None):
    rb = load_rulebook()
    section = dict(rb.get("emdat"))
    if reason is None:
        section.pop("unavailable_reason", None)
    else:
        section["unavailable_reason"] = reason
    rb._data["emdat"] = section
    return rb


def _refuse(*_args, **_kwargs):
    """What EM-DAT actually did on every call of run 34222175003."""

    raise RuntimeError(
        "HTTP 500: Invalid key passed or insufficient user access"
    )


def _explode(*_args, **_kwargs):
    raise AssertionError("a request was made for a source marked unavailable")


# -- the constraint that matters ---------------------------------------------


def test_the_stand_down_makes_no_request(monkeypatch):
    monkeypatch.setenv("EMDAT_API_KEY", "a-key-that-exists-and-is-rejected")
    outcome = emdat.fetch_emdat(
        _con(), "2024-03", "FL", _rulebook("no_credentials"), post=_explode
    )
    assert outcome.ok is False
    assert outcome.detail["failure_class"] == emdat.FAILURE_UNAVAILABLE
    assert outcome.detail["unavailable_reason"] == "no_credentials"


def test_the_stand_down_does_not_read_the_credential(monkeypatch):
    """Stood down BEFORE the key is looked at, so a present-but-rejected key
    changes nothing and an absent one is not what is being reported."""

    monkeypatch.delenv("EMDAT_API_KEY", raising=False)
    without = emdat.fetch_emdat(
        _con(), "2024-03", "FL", _rulebook("no_credentials"), post=_explode
    )
    monkeypatch.setenv("EMDAT_API_KEY", "present")
    with_key = emdat.fetch_emdat(
        _con(), "2024-03", "FL", _rulebook("no_credentials"), post=_explode
    )
    assert (
        without.detail["failure_class"]
        == with_key.detail["failure_class"]
        == emdat.FAILURE_UNAVAILABLE
    )


def test_stood_down_and_refused_produce_the_same_ledger_marking(monkeypatch):
    """The test this whole change exists to satisfy.

    A cell walked with EM-DAT switched off must carry
    ``rungs_unavailable: emdat``, indistinguishable in the ledger from a cell
    walked with EM-DAT attempting and failing.
    """

    monkeypatch.setenv("EMDAT_API_KEY", "a-key-that-exists-and-is-rejected")

    refused = emdat.fetch_emdat(_con(), "2024-03", "FL", _rulebook(None), post=_refuse)
    stood_down = emdat.fetch_emdat(
        _con(), "2024-03", "FL", _rulebook("no_credentials"), post=_explode
    )

    # `unavailable_sources` is what becomes `rungs_unavailable`: the names of
    # the rungs whose fetch outcome is not ok.
    for outcome in (refused, stood_down):
        run = impact.LadderRun(hazard="FL", ym="2024-03")
        run.fetches = {SOURCE: outcome.as_provenance()}
        assert run.unavailable_sources == [SOURCE], (
            "a cell decided without this rung must say so whether the rung "
            "refused us or we declined to ask"
        )


def test_the_stand_down_still_serves_a_populated_cache(monkeypatch):
    """A stand-down is not a reason to ignore what we already hold.

    Routing through the same cache fallback a refusal takes is what makes the
    two records identical by construction rather than by coincidence: a
    populated cache answers as a STALE rung either way.
    """

    monkeypatch.delenv("EMDAT_API_KEY", raising=False)
    con = _con()
    rb = _rulebook("no_credentials")
    start, end = emdat.fetch_window("2024-03", rb, "emdat")
    con.execute(
        """
        INSERT INTO haz_raw_emdat
            (record_id, iso3, hazard, ym, payload_json, content_hash,
             retrieved_at)
        VALUES ('e1', 'PHL', 'FL', '2024-03', ?, 'h1', now())
        """,
        [json.dumps({
            "iso3": "PHL", "hazard": "FL",
            "months_overlapped": ["2024-03"],
            "affected": 1000.0,
            "start_date": start.isoformat(), "end_date": end.isoformat(),
        })],
    )
    outcome = emdat.fetch_emdat(con, "2024-03", "FL", rb, post=_explode)
    assert outcome.ok is True
    assert outcome.detail.get("served_from_cache") is True
    assert outcome.detail.get("unavailable_reason") == "no_credentials"


def test_without_the_switch_the_connector_is_unchanged(monkeypatch):
    """The switch is opt-in. A source with no declared state behaves exactly
    as it did before this existed."""

    monkeypatch.setenv("EMDAT_API_KEY", "present")
    outcome = emdat.fetch_emdat(
        _con(), "2024-03", "FL", _rulebook(None), post=_refuse
    )
    assert outcome.detail["failure_class"] == emdat.FAILURE_AUTH


# -- the reason, and the rulebook -------------------------------------------


def test_the_state_is_a_reason_not_a_boolean():
    """A boolean would record that a rung is down and lose the only part a
    reader can act on."""

    assert (
        rulebook_mod.source_unavailable_reason(load_rulebook(), "emdat")
        == "no_credentials"
    )
    assert "no_credentials" in rulebook_mod.KNOWN_SOURCE_UNAVAILABLE_REASONS
    assert len(rulebook_mod.KNOWN_SOURCE_UNAVAILABLE_REASONS) > 1, (
        "one reason is a boolean wearing a string's clothes"
    )


def test_an_unknown_reason_fails_validation():
    """Silently reading as "available" would leave a source we meant to stand
    down making requests we know will be refused."""

    rb = load_rulebook()
    data = {**rb._data, "emdat": {**rb.get("emdat"), "unavailable_reason": "dunno"}}
    problems = rulebook_mod.validate_rulebook(data)
    assert any("emdat.unavailable_reason" in p for p in problems)


def test_a_source_with_no_switch_is_available():
    rb = load_rulebook()
    for source in ("ifrc_go", "idmc_idu"):
        assert rulebook_mod.source_unavailable_reason(rb, source) == ""


def test_gdacs_cannot_be_stood_down():
    """GDACS is the detector AND the ceiling, so standing it down would
    silently remove every flood trigger rather than one rung's figures."""

    assert "gdacs" not in rulebook_mod.UNAVAILABLE_SWITCH_SOURCES
    rb = load_rulebook()
    rb._data["gdacs"] = {"unavailable_reason": "no_credentials"}
    assert rulebook_mod.source_unavailable_reason(rb, "gdacs") == ""


def test_the_switch_moves_no_hazard_fingerprint():
    """Which is exactly why recovery needs a targeted restale, and why the
    register entry has to say so.

    ``emdat`` is in no fingerprinted section, so flipping this switch leaves
    every ledger row's fingerprint intact, ``completed_months`` returns every
    month already marked ``ok``, and nothing re-walks.
    """

    off = _rulebook("no_credentials")
    on = _rulebook(None)
    for hazard in ("flood", "cyclone", "drought"):
        assert off.hazard_fingerprint(hazard) == on.hazard_fingerprint(hazard)


# -- the register ------------------------------------------------------------


def test_the_register_fires_from_config_state_with_nothing_attempted():
    """If the register only raised on a failed fetch, switching the connector
    off would make the issue vanish — and a rung being down would stop being
    visible at the moment it became permanent."""

    found = issue_sources.issues_from_source_state({"emdat": "no_credentials"})
    assert [i.id for i in found] == ["emdat_auth_rejected"]
    assert found[0].recovers_on_rerun is False
    assert "no_credentials" in found[0].evidence


def test_an_available_source_says_nothing():
    assert issue_sources.issues_from_source_state({}) == []
    assert issue_sources.issues_from_source_state(None) == []
    assert issue_sources.issues_from_source_state({"emdat": ""}) == []


def test_the_config_issue_and_a_failed_fetch_merge_into_one_record():
    """Two ids for one fault is the thing the register exists to prevent."""

    register = issues_mod.IssueRegister()
    register.extend(issue_sources.issues_from_source_state({"emdat": "no_credentials"}))
    register.extend(issue_sources.issues_from_source_fetches([
        {"source": "emdat", "ok": False, "failure_class": "marked_unavailable",
         "error": "marked unavailable"},
    ]))
    assert [i.id for i in register.issues] == ["emdat_auth_rejected"]


def test_a_stood_down_source_is_registered_and_therefore_quiet():
    """The point of the register: an owned, diagnosed fault reports once, at
    `known`, rather than raising an error annotation every run."""

    register = issues_mod.IssueRegister(known=issues_mod.KnownIssues.load())
    register.extend(issue_sources.issues_from_source_state({"emdat": "no_credentials"}))
    issue = register.issues[0]
    assert issue.severity == issues_mod.KNOWN
    assert issue.owner == "pythia"
    assert "restale" in issue.note


def test_the_entry_states_the_restale_requirement_and_not_the_inflated_cost():
    """61,413 reads as recoverable rows and is not. 61,355 of those cells had
    no readable rung at all, and EM-DAT is sparse by design."""

    entry = issues_mod.KnownIssues.load().entries["emdat_auth_rejected"]
    note = entry["note"]
    assert "restale" in note
    assert "61,413" in note and "61,355" in note
    assert "58" in note
    assert "unknown until a credential exists" in note
    assert entry["owner"] == "pythia"
    # PyYAML parses an unquoted ISO date into a date object.
    assert str(entry["review_by"]) == "2027-03-10"


def test_marked_unavailable_is_a_permanent_failure_class():
    """No re-run clears it, and nobody fixes it by writing code."""

    found = issue_sources.issues_from_source_fetches([
        {"source": "emdat", "ok": False, "failure_class": "marked_unavailable"},
    ])
    assert found[0].recovers_on_rerun is False
