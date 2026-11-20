# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""inject_health.csv's GDACS row says what the prompt SHOWED.

Before Nov 2026 the row was rebuilt from the database as it stood at bundle
time, so it read clean while the stored prompts repeated months and printed
"0 events" over four months. These tests pin the columns parsed from the
stored prompt beside the rebuild.
"""

from __future__ import annotations

from scripts.ai_bundle import error_attribution as ea

META = {"iso3": "NER", "hazard_code": "DR", "metric": "EVENT_OCCURRENCE", "track": 1}
INJECT = {
    "gdacs_history": {
        "applicable": True, "available": True, "window_start": "2017-07",
        "window_end": "2026-08", "total_months": 110, "event_months": 4, "reason": None,
    },
}

SHORT_ZERO = """\
GDACS EVENT HISTORY (NER — drought):
Source: GDACS (Global Disaster Alert and Coordination System)
Coverage: 2026-06 to 2026-09 (4 months)
Overall event rate: 0 events in 4 months (0%)

Recent event timeline:
  2026-06: -- | 2026-07: -- | 2026-08: --
  2026-09: --

INTERPRETATION: x
"""

REPEATED = """\
GDACS EVENT HISTORY (NER — drought):
Coverage: 2017-07 to 2026-08 (110 months)
Overall event rate: 4 events in 110 months (4%)

Recent event timeline:
  2026-07: -- | 2026-07: -- | 2026-08: --
  2026-08: --

INTERPRETATION: x
"""

CURRENT = SHORT_ZERO.replace(
    "Coverage: 2026-06 to 2026-09 (4 months)",
    "Coverage: 2017-07 to 2026-08 (110 calendar months; a month with no alert counts as no event)",
).replace("0 events in 4 months (0%)", "4 events in 110 months (4%)")


def _gdacs_row(prompt):
    rows = ea.inject_health_rows("NER_DR_EVENT_OCCURRENCE_2026-10", META, INJECT, prompt_text=prompt)
    return next(r for r in rows if r["inject"] == "gdacs_history")


def test_a_short_window_shown_as_zero_is_invalid_and_lower_than_the_rebuild():
    r = _gdacs_row(SHORT_ZERO)
    assert r["prompt_total_months"] == 4 and r["prompt_event_months"] == 0
    assert r["prompt_vs_rebuild"] == "prompt_lower"
    assert r["invalid"] is True and r["prompt_history_unavailable"] is False
    # The rebuild columns are unchanged.
    assert r["present"] is True and r["age_months"] == 110 and r["stale"] is False


def test_repeated_months_are_counted_and_invalid():
    r = _gdacs_row(REPEATED)
    assert r["prompt_repeated_months"] == 2
    assert r["invalid"] is True and r["prompt_vs_rebuild"] == "match"


def test_a_correct_block_matches():
    r = _gdacs_row(CURRENT)
    assert r["invalid"] is False and r["prompt_vs_rebuild"] == "match"
    assert r["prompt_repeated_months"] == 0


def test_a_question_with_no_block():
    r = _gdacs_row("A prompt with nothing about GDACS.")
    assert r["prompt_vs_rebuild"] == "no_block"
    assert r["invalid"] is None and r.get("prompt_total_months") is None
    # Other rows carry no prompt columns at all.
    rows = ea.inject_health_rows("q", META, INJECT, prompt_text=None)
    assert all(set(ea.INJECT_HEALTH_COLUMNS) >= set(row) for row in rows)


def test_the_digest_names_the_faults_per_hazard():
    rows = [_gdacs_row(SHORT_ZERO), _gdacs_row(REPEATED), _gdacs_row(CURRENT)]
    text = "\n".join(ea.inject_digest_lines(rows))
    assert (
        "DR: GDACS block as shown: 1 repeated-month, 1 zero-where-DB-has-events, 0 unavailable"
        in text
    )
