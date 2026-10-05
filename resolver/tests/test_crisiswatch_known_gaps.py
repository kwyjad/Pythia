# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The twelve-edition CrisisWatch check honours a registered, dated gap.

May 2026 cannot be recovered (no capture exists anywhere a runner can reach),
so it is registered in horizon_scanner/crisiswatch_known_gaps.py with a review
date. The check must still fail for any other missing month, and fail again
for the registered one once its review date has passed.
"""

from __future__ import annotations

import datetime as _dt

from scripts import build_resolver_debug_bundle as bundle


def _crisiswatch_builder(tmp_path, missing):
    import duckdb

    tmp_path.mkdir(parents=True, exist_ok=True)
    db = tmp_path / "cw.duckdb"
    con = duckdb.connect(str(db))
    con.execute(
        "CREATE TABLE crisiswatch_entries (iso3 TEXT, month INTEGER, year INTEGER, "
        "arrow TEXT, alert_type TEXT, summary TEXT, country_name TEXT, "
        "fetched_at TIMESTAMP, content_hash TEXT)"
    )
    for (yy, mm) in bundle.expected_crisiswatch_window(_dt.date.today()):
        if (yy, mm) not in missing:
            con.execute(
                "INSERT INTO crisiswatch_entries VALUES ('SOM', ?, ?, '', '', '', 'Somalia', now(), 'h')",
                [mm, yy],
            )
    con.close()
    return bundle.BundleBuilder(
        out_path=tmp_path / "b.zip", db_path=db, diagnostics_dir=tmp_path / "diag",
        run_log_dir=tmp_path / "none", staging=tmp_path / "staging",
        max_bytes=bundle.DEFAULT_MAX_BYTES, environ={},
    )


def test_a_registered_crisiswatch_gap_passes_and_any_other_still_fails(tmp_path, monkeypatch):
    """May 2026 is unrecoverable and registered; another missing month is not."""

    from horizon_scanner import crisiswatch_known_gaps as gaps

    window = bundle.expected_crisiswatch_window(_dt.date.today())
    known, other = window[3], window[7]
    known_label = f"{known[0]}-{known[1]:02d}"
    other_label = f"{other[0]}-{other[1]:02d}"
    review = (_dt.date.today() + _dt.timedelta(days=60)).isoformat()
    monkeypatch.setattr(gaps, "KNOWN_MISSING_EDITIONS",
                        {known_label: {"reason": "no capture", "review_by": review}})

    builder = _crisiswatch_builder(tmp_path, {known})
    builder._check_crisiswatch_holds_the_last_twelve_editions()
    check = builder.checks[-1]
    assert check["verdict"] == "PASS" and "known gap" in check["detail"]

    builder = _crisiswatch_builder(tmp_path / "b", {known, other})
    builder._check_crisiswatch_holds_the_last_twelve_editions()
    check = builder.checks[-1]
    assert check["verdict"] == "FAIL"
    assert [i["id"] for i in check["issues"]] == [f"crisiswatch_edition_missing_{other_label}"]

    # A gap past its review date fails again.
    monkeypatch.setattr(gaps, "KNOWN_MISSING_EDITIONS",
                        {known_label: {"reason": "no capture", "review_by": "2000-01-01"}})
    builder = _crisiswatch_builder(tmp_path / "c", {known})
    builder._check_crisiswatch_holds_the_last_twelve_editions()
    assert builder.checks[-1]["verdict"] == "FAIL"


def test_may_2026_is_the_registered_crisiswatch_gap():
    from horizon_scanner.crisiswatch_known_gaps import KNOWN_MISSING_EDITIONS

    assert set(KNOWN_MISSING_EDITIONS) == {"2026-05"}
    assert KNOWN_MISSING_EDITIONS["2026-05"]["review_by"]
