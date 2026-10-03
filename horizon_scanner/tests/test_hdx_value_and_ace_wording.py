# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.
"""Two wording repairs from the 1 Oct 2026 prompts.

* HDX Signals printed "Indicator value: 171615.00000000003".
* ACE/FATALITIES questions resolve on ACLED deaths over ALL event types, and
  the question template still said "battle-related fatalities".
"""

from __future__ import annotations

from horizon_scanner import db_writer
from horizon_scanner.hdx_signals import format_indicator_value


def test_indicator_values_print_as_a_reader_writes_them():
    assert format_indicator_value("171615.00000000003") == "171,615"
    assert format_indicator_value("0.4250000001") == "0.43"
    assert format_indicator_value("12.5") == "12.5"
    assert format_indicator_value("not a number") == "not a number"


def test_new_fatalities_questions_name_all_event_types():
    text = db_writer.DISPLACEMENT_TEMPLATES["acled_fatalities"]
    assert "battle-related" not in text
    assert "all ACLED event types" in text
