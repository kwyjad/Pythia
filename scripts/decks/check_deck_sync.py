# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.
"""Warn when a PR records a change in fred_overview.md and touches no deck.

A change big enough for a "Changes in this Version" entry in
docs/fred_overview.md may have made a slide in docs/decks/ wrong. This
check reads the PR's diff and, when that section changed and nothing under
docs/decks/ did, prints a GitHub ``::warning`` naming the decks
docs/decks/manifest.json maps to the overview sections and code paths the
PR touched. It never fails: a deck may legitimately need no change, and
the rule in CLAUDE.md leaves that judgement to whoever makes the change.

Usage::

    python -m scripts.decks.check_deck_sync --base origin/main
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

OVERVIEW = "docs/fred_overview.md"
DECKS_DIR = "docs/decks/"
MANIFEST = Path("docs/decks/manifest.json")
CHANGES_HEADING = "Changes in this Version"


def headings_by_line(text: str) -> list[str]:
    """For each line (1-based index into the list, index 0 unused), the
    ``##``-level heading it sits under."""
    out = [""]
    current = ""
    for line in text.splitlines():
        m = re.match(r"^##\s+(.*)$", line)
        if m:
            current = m.group(1).strip()
        out.append(current)
    return out


def changed_new_lines(diff: str) -> list[int]:
    """New-side line numbers a unified diff (``-U0``) adds or changes; a pure
    deletion counts as the line it sits before."""
    lines = []
    for m in re.finditer(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", diff, re.M):
        start, count = int(m.group(1)), int(m.group(2) or 1)
        lines.extend(range(start, start + count) if count else [max(start, 1)])
    return lines


def touched_sections(overview_text: str, diff: str) -> set[str]:
    heads = headings_by_line(overview_text)
    return {heads[n] for n in changed_new_lines(diff) if 0 < n < len(heads) and heads[n]}


def decks_for(manifest: dict, sections: set[str], files: list[str]) -> list[dict]:
    hits = []
    for deck in manifest.get("decks", []):
        sec = any(s.startswith(p) for s in sections for p in deck.get("overview_sections", []))
        code = any(f == p or f.startswith(p.rstrip("/") + "/")
                   for f in files for p in deck.get("code_paths", []))
        if sec or code:
            hits.append(deck)
    return hits


def decide(manifest: dict, files: list[str], sections: set[str]) -> str | None:
    """The warning to print, or None."""
    if CHANGES_HEADING not in sections:
        return None
    if any(f.startswith(DECKS_DIR) for f in files):
        return None
    decks = decks_for(manifest, sections - {CHANGES_HEADING}, files)
    if decks:
        names = ", ".join(f"{d['number']} {d['title']} ({d['file']})" for d in decks)
        return ("This PR adds to 'Changes in this Version' in docs/fred_overview.md but "
                f"changes nothing under docs/decks/. Check these decks: {names}. "
                "Fix wrong slides with scripts/decks/edit_deck.py and update "
                "docs/decks/manifest.json (see CLAUDE.md, 'Deck editing').")
    return ("This PR adds to 'Changes in this Version' in docs/fred_overview.md but "
            "changes nothing under docs/decks/, and docs/decks/manifest.json maps no deck "
            "to what it touched. Check whether any deck describes this change.")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          check=True).stdout


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default="origin/main")
    args = ap.parse_args(argv)
    try:
        files = [f for f in _git("diff", "--name-only", f"{args.base}...HEAD").splitlines() if f]
        diff = _git("diff", "-U0", f"{args.base}...HEAD", "--", OVERVIEW)
        overview = Path(OVERVIEW).read_text(encoding="utf-8") if Path(OVERVIEW).exists() else ""
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - a warning check must never fail a PR
        print(f"::warning title=Deck check skipped::{exc}")
        return 0
    message = decide(manifest, files, touched_sections(overview, diff))
    if message:
        print(f"::warning title=Slide decks may be out of date::{message}")
    else:
        print("Deck check: nothing to flag.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
