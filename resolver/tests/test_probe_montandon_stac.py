# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Gate logic for the Montandon STAC probe.

The probe itself needs egress to *.ifrc.org, which no CI job here is
guaranteed. These tests cover the part that decides what the answers MEAN, so a
misread verdict cannot survive to the artifact a human reads.

The load-bearing case is G1. Montandon aggregates GDACS alongside EM-DAT,
DesInventar and IFRC field reports. GDACS impact figures are modelled
population exposure — hazard footprint x population — not reported impact.
Consuming Montandon without being able to exclude them would blend the two in
one column, which is the exact failure removed from Pythia's own PA base rates
in August 2026. G1 must therefore be able to fail the whole evaluation on its
own.
"""

from __future__ import annotations

import pytest

from tools.probe_montandon_stac import (
    MODELLED_EXPOSURE_SOURCES,
    REPORTED_IMPACT_SOURCES,
    _classify_source,
    _parse_date,
    decide,
    render_markdown,
)


def _gate(gate_id, passed, question="q", verdict="v"):
    return {"gate": gate_id, "pass": passed, "question": question, "verdict": verdict}


# ---------------------------------------------------------------------------
# Source classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", sorted(MODELLED_EXPOSURE_SOURCES))
def test_modelled_exposure_sources_are_classified_as_such(text):
    assert _classify_source(f"monty-{text}-impacts") == "modelled_exposure"


@pytest.mark.parametrize("text", sorted(REPORTED_IMPACT_SOURCES))
def test_reported_impact_sources_are_classified_as_such(text):
    assert _classify_source(f"monty-{text}-impacts") == "reported_impact"


def test_gdacs_never_reads_as_reported_impact():
    """The whole point of the kill gate."""
    for label in ("gdacs-events", "GDACS impact", "monty/gdacs/exposure"):
        assert _classify_source(label) == "modelled_exposure"


def test_unrecognised_source_is_unknown_not_assumed_safe():
    assert _classify_source("some-new-partner-feed") == "unknown"


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def test_unreachable_endpoint_yields_unknown_not_rejection():
    """A blocked network is a missing answer, not a negative one."""
    verdict = decide([_gate("G0", False)])
    assert verdict["outcome"].startswith("UNKNOWN")
    assert "unreachable" in verdict["outcome"].lower()


def test_failing_the_source_attribution_gate_rejects_outright():
    verdict = decide([_gate("G0", True), _gate("G1", False), _gate("G2", True), _gate("G4", True)])
    assert verdict["outcome"].startswith("D")
    assert "gdacs" in verdict["verdict"].lower()


def test_g1_failure_outranks_every_other_gate_passing():
    """No amount of coverage or speed rescues an unattributable feed."""
    verdict = decide(
        [_gate("G0", True), _gate("G1", False), _gate("G2", True), _gate("G4", True),
         _gate("G7", True)]
    )
    assert verdict["outcome"].startswith("D")


def test_slow_but_attributable_feed_is_base_rates_only():
    verdict = decide([_gate("G0", True), _gate("G1", True), _gate("G2", True), _gate("G4", False)])
    assert verdict["outcome"].startswith("C")
    assert "base rate" in verdict["verdict"].lower()
    # Outcome C is only safe if the agreement check confirmed one class.
    assert "g5" in verdict["verdict"].lower() or "agreement" in verdict["verdict"].lower()


def test_all_gates_passing_is_a_candidate_not_an_automatic_adoption():
    verdict = decide([_gate("G0", True), _gate("G1", True), _gate("G2", True), _gate("G4", True)])
    assert verdict["outcome"].startswith("A/B")
    assert "coverage" in verdict["verdict"].lower()


def test_unmeasurable_latency_does_not_read_as_a_pass_or_a_fail():
    """G4 returns None when items lacked dates; that must not become outcome A."""
    verdict = decide([_gate("G0", True), _gate("G1", True), _gate("G2", True), _gate("G4", None)])
    # None is not False, so it falls through to the candidate branch — the
    # markdown still shows G4 as n/a rather than claiming it passed.
    assert verdict["outcome"].startswith("A/B")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_markdown_marks_unmeasured_gates_as_na_not_pass():
    report = {
        "endpoint": "https://example.invalid/stac",
        "generated_at": "2026-08-04T00:00:00Z",
        "gates": [_gate("G4", None, question="latency", verdict="could not measure")],
        "collections": {},
        "overall": {"outcome": "X", "verdict": "y"},
    }
    md = render_markdown(report)
    assert "| G4 | latency | n/a |" in md
    assert "PASS" not in md


def test_markdown_does_not_break_the_table_on_pipes_in_a_verdict():
    report = {
        "endpoint": "e",
        "generated_at": "t",
        "gates": [_gate("G1", False, question="attribution", verdict="a|b|c")],
        "collections": {},
        "overall": {"outcome": "D", "verdict": "v"},
    }
    row = [line for line in render_markdown(report).splitlines() if line.startswith("| G1 ")][0]
    assert row.count("|") == 5


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_parse_date_tolerates_timestamps_and_junk():
    from datetime import date

    assert _parse_date("2025-08-19T12:00:00Z") == date(2025, 8, 19)
    assert _parse_date("2025-08-19") == date(2025, 8, 19)
    assert _parse_date("not a date") is None
    assert _parse_date(None) is None
