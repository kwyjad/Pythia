# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.
"""Tests for scripts/decks/check_deck_sync.py (stdlib only)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.decks.check_deck_sync import CHANGES_HEADING, decide, touched_sections

MANIFEST = json.loads((Path(__file__).resolve().parents[3] / "docs/decks/manifest.json").read_text())

OVERVIEW = "\n".join([
    "## Changes in this Version",   # 1
    "- something new",              # 2
    "## 12b. Sibyl: the parallel deep-research track",  # 3
    "Sibyl takes 15 questions.",    # 4
    "### Cost control",             # 5
    "The cap is $25.",              # 6
])


def test_touched_sections_follow_the_diff():
    diff = "@@ -2,0 +2,1 @@\n+- something new\n@@ -6 +6 @@\n-The cap is $60.\n+The cap is $25.\n"
    assert touched_sections(OVERVIEW, diff) == {CHANGES_HEADING, "12b. Sibyl: the parallel deep-research track"}


def test_no_warning_without_a_change_log_entry():
    assert decide(MANIFEST, ["sibyl/run.py"], {"12b. Sibyl"}) is None


def test_no_warning_when_a_deck_changed():
    assert decide(MANIFEST, ["docs/decks/4_sibyl.pptx"], {CHANGES_HEADING}) is None


def test_warning_names_decks_from_sections_and_code_paths():
    msg = decide(MANIFEST, ["docs/fred_overview.md", "interpreter/pdf.py"],
                 {CHANGES_HEADING, "12b. Sibyl: the parallel deep-research track"})
    assert "4 Forecast: Sibyl Track" in msg and "7 The Interpreter" in msg
    assert "0 Introductory" not in msg


def test_warning_when_nothing_maps():
    msg = decide(MANIFEST, ["docs/fred_overview.md"], {CHANGES_HEADING})
    assert "maps no deck" in msg
