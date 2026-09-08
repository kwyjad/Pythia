# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""A worker count is not a rule, and must not re-derive ten years of history.

The resume ledger stamps each month with a digest of the rulebook that
decided it, so a fix reaches the historical record without an operator
remembering a flag. That mechanism was right and its scope was wrong:
EVERY key in a covered section entered the digest, including the ones
that govern how fast the machine asks a source.

Cyclone borrows ``flood.gdacs``, so a September 2026 pacing edit there put
307 cyclone months back on the queue. At the rates that run measured, that
is roughly six months of nightly work and about $30 to reproduce answers
that could not have moved, because a request rate decides nothing about a
flood.

The asymmetry decides the bar. Excluding a key that DOES decide freezes
history under stale rules, silently, which is the fault the fingerprint
exists to end; including one that does not costs an expensive but harmless
re-walk. So the exclusion list is short, explicit and audited here.

Network-free.
"""

from __future__ import annotations

import copy
from pathlib import Path

import duckdb
import pytest

from resolver.hazard_resolution import backcast as bc
from resolver.hazard_resolution.rulebook import Rulebook, load_rulebook
from resolver.hazard_resolution.schema import ensure_haz_schema

HAZARDS = ("flood", "cyclone", "drought")

#: Exactly the keys the digest must ignore. Written out rather than
#: imported, so a silent widening of the module's own set fails here.
PACING = (
    "flood.gdacs.request_delay_sec",
    "flood.gdacs.enrich_workers",
    "flood.gdacs.enrich_min_interval_sec",
    "reliefweb.documents.request_delay_sec",
    "flood.gdacs.enrich_max_seconds",
    "reliefweb.documents.request_timeout_sec",
    "extraction.request_timeout_sec",
    "extraction.max_calls_per_month",
    "extraction.live_reserve_calls",
    "extraction.backcast_max_calls_per_month",
)

#: Keys that decide an answer and must keep moving the digest.
DECIDING = (
    "flood.gdacs.exposure_refresh_days",
    "flood.gdacs.coverage_grace_days",
    "flood.gdacs.lookback_months",
    "flood.gdacs.lookahead_months",
    "reliefweb.documents.max_docs_per_cell",
    "reliefweb.documents.candidate_pool_size",
    "reliefweb.documents.body_char_limit",
    "reliefweb.documents.publication_pad_days",
    "reliefweb.household_conversion.default_multiplier",
    "extraction.prompt_version",
    "extraction.max_output_tokens",
    "extraction.enabled",
    "extraction.skip_when_higher_rung_populated",
    "sanity.ceiling_multiplier",
    "sanity.min_plausible_exposure",
    "conflict_detection.order_of_magnitude_factor",
    "freeze_days",
    "raw_cache.keep_revisions_per_record",
)


def _set(data: dict, dotted: str, value) -> dict:
    out = copy.deepcopy(data)
    node = out
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value
    return out


@pytest.fixture(scope="module")
def shipped() -> dict:
    return copy.deepcopy(load_rulebook()._data)


def _rb(data: dict) -> Rulebook:
    return Rulebook(data, Path("test"))


# ---------------------------------------------------------------------------
# What the digest must ignore
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", PACING)
@pytest.mark.parametrize("hazard", HAZARDS)
def test_a_pacing_key_does_not_re_walk_history(shipped, hazard, key):
    before = _rb(shipped).hazard_fingerprint(hazard)
    after = _rb(_set(shipped, key, 999)).hazard_fingerprint(hazard)
    assert before == after, (
        f"changing {key} moved {hazard}'s digest, so the next run would "
        "re-derive every month of it — and that key cannot change a single "
        "answer"
    )


def test_the_whole_compaction_subtree_is_housekeeping(shipped):
    changed = copy.deepcopy(shipped)
    changed["raw_cache"]["compaction"]["file_bloat_ratio"] = 9.9
    changed["raw_cache"]["compaction"]["something_new"] = True
    for hazard in HAZARDS:
        assert _rb(shipped).hazard_fingerprint(hazard) == _rb(
            changed
        ).hazard_fingerprint(hazard)


def test_the_cyclone_borrowed_block_is_pruned_too(shipped):
    """The point of the exercise: cyclone reads flood.gdacs and must not
    re-walk when the GDACS pace changes."""

    changed = _set(shipped, "flood.gdacs.enrich_workers", 6)
    assert _rb(shipped).hazard_fingerprint("cyclone") == _rb(
        changed
    ).hazard_fingerprint("cyclone")
    assert _rb(shipped).hazard_fingerprint("flood") == _rb(
        changed
    ).hazard_fingerprint("flood")


# ---------------------------------------------------------------------------
# What the digest must still notice
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", DECIDING)
def test_a_deciding_key_still_re_walks_history(shipped, key):
    """Narrowing must not become deafness. Each of these moves an answer."""

    current = _rb(shipped).get(key)
    replacement = (not current) if isinstance(current, bool) else "moved"
    changed = _set(shipped, key, replacement)
    moved = {
        hazard
        for hazard in HAZARDS
        if _rb(shipped).hazard_fingerprint(hazard)
        != _rb(changed).hazard_fingerprint(hazard)
    }
    assert moved, f"changing {key} moved no hazard's digest — history would freeze"


def test_the_ladder_still_moves_every_hazard(shipped):
    changed = copy.deepcopy(shipped)
    changed["ladder"] = ["emdat"]
    for hazard in HAZARDS:
        assert _rb(shipped).hazard_fingerprint(hazard) != _rb(
            changed
        ).hazard_fingerprint(hazard)


def test_a_new_key_beside_an_excluded_one_is_deciding_until_listed(shipped):
    """The exclusion list is a list of keys, never a prefix or a section.

    A key added next to a pacing knob has not been assessed by anyone, so
    it counts until someone says otherwise.
    """

    changed = copy.deepcopy(shipped)
    changed["flood"]["gdacs"]["some_new_threshold"] = 4
    assert _rb(shipped).hazard_fingerprint("flood") != _rb(
        changed
    ).hazard_fingerprint("flood")
    assert _rb(shipped).hazard_fingerprint("cyclone") != _rb(
        changed
    ).hazard_fingerprint("cyclone")


def test_drought_is_untouched_by_the_gdacs_block(shipped):
    """It has no ladder and no ceiling, so nothing there reaches it."""

    changed = _set(shipped, "flood.gdacs.exposure_refresh_days", 99)
    assert _rb(shipped).hazard_fingerprint("drought") == _rb(
        changed
    ).hazard_fingerprint("drought")


# ---------------------------------------------------------------------------
# The exclusion list is audited, not assumed
# ---------------------------------------------------------------------------


def test_every_excluded_key_exists_in_the_shipped_rulebook(shipped):
    """A typo or a rename silently un-excludes a key.

    The symptom is the fault this change removes coming back: an operator
    trims a timeout and 300 months re-walk. Nothing else would notice.
    """

    rb = _rb(shipped)
    for key in Rulebook._NON_DECIDING_KEYS:
        assert rb.get(key, None) is not None, (
            f"{key} is excluded from the fingerprint but is not in the "
            "rulebook — it was renamed or mistyped, and whatever it is "
            "called now is deciding again"
        )


def test_the_exclusion_list_is_exactly_what_this_file_states(shipped):
    assert Rulebook._NON_DECIDING_KEYS == frozenset(PACING) | {
        "raw_cache.compaction"
    }


def test_pruning_never_empties_a_covered_section(shipped):
    """A fat-fingered prefix would remove a whole section from the digest."""

    rb = _rb(shipped)
    for section in Rulebook._SHARED_SECTIONS + ("flood", "cyclone", "drought"):
        value = rb.get(section, None)
        if not isinstance(value, dict):
            continue
        pruned = rb._prune_non_deciding(value, section)
        assert pruned, f"pruning emptied {section} — the digest now covers nothing"


def test_the_digest_is_still_stable_and_still_sixteen_hex(shipped):
    rb = _rb(shipped)
    for hazard in HAZARDS:
        digest = rb.hazard_fingerprint(hazard)
        assert digest == rb.hazard_fingerprint(hazard)
        assert len(digest) == 16
    assert len({rb.hazard_fingerprint(h) for h in HAZARDS}) == 3


def test_key_order_still_cannot_move_the_digest():
    a = Rulebook({"drought": {"x": 1, "y": 2}, "ladder": ["emdat"]}, Path("a"))
    b = Rulebook({"ladder": ["emdat"], "drought": {"y": 2, "x": 1}}, Path("b"))
    assert a.hazard_fingerprint("drought") == b.hazard_fingerprint("drought")


# ---------------------------------------------------------------------------
# Narrowing must not itself cost a re-walk
# ---------------------------------------------------------------------------


@pytest.fixture()
def con(tmp_path):
    connection = duckdb.connect(str(tmp_path / "haz.duckdb"))
    ensure_haz_schema(connection)
    return connection


def _record(con, hazard: str, ym: str, digest: str | None) -> None:
    year, month = int(ym[:4]), int(ym[5:7])
    con.execute(
        "INSERT INTO haz_triggers (iso3, hazard, year, month, triggered, "
        "trigger_source) VALUES ('SOM', ?, ?, ?, FALSE, 'test')",
        [hazard, year, month],
    )
    bc.record_month(
        con, hazard=hazard, ym=ym, status="ok", counts={"cells": 5},
        rulebook_hash=digest,
    )


def test_the_legacy_digest_is_what_the_ledger_actually_holds():
    """The cyclone ledger's stored value on 2026-09-07 was 29ccb0bbbe515dad.

    That is the pre-narrowing digest of this same rulebook, so those months
    were decided under today's deciding values and must not be walked
    again. If this ever fails, a deciding key moved and the re-walk is
    correct — do not reach for the re-stamp.
    """

    assert load_rulebook().legacy_hazard_fingerprint("cyclone") == (
        "29ccb0bbbe515dad"
    )


def test_months_decided_under_this_rulebook_are_re_stamped_not_re_walked(con):
    rb = load_rulebook()
    legacy = rb.legacy_hazard_fingerprint("cyclone")
    current = rb.hazard_fingerprint("cyclone")
    assert legacy != current, "the narrowing must have moved the digest"

    _record(con, "TC", "2026-05", legacy)
    _record(con, "TC", "2026-04", legacy)
    assert bc.completed_months(con, "TC", current) == set(), (
        "without the re-stamp, narrowing costs the re-walk it prevents"
    )

    moved = bc.restamp_equivalent_fingerprints(
        con, "TC", current=current, legacy=legacy
    )
    assert moved == 2
    assert bc.completed_months(con, "TC", current) == {"2026-05", "2026-04"}


def test_the_re_stamp_blesses_nothing_from_an_older_rulebook(con):
    rb = load_rulebook()
    _record(con, "TC", "2026-03", "a_rulebook_since_changed")
    bc.restamp_equivalent_fingerprints(
        con, "TC",
        current=rb.hazard_fingerprint("cyclone"),
        legacy=rb.legacy_hazard_fingerprint("cyclone"),
    )
    assert bc.completed_months(con, "TC", rb.hazard_fingerprint("cyclone")) == set()


def test_the_re_stamp_is_idempotent_and_a_no_op_once_converged(con):
    rb = load_rulebook()
    legacy = rb.legacy_hazard_fingerprint("cyclone")
    current = rb.hazard_fingerprint("cyclone")
    _record(con, "TC", "2026-05", legacy)
    assert bc.restamp_equivalent_fingerprints(
        con, "TC", current=current, legacy=legacy
    ) == 1
    assert bc.restamp_equivalent_fingerprints(
        con, "TC", current=current, legacy=legacy
    ) == 0


def test_a_row_with_no_digest_is_still_re_walked_once(con):
    rb = load_rulebook()
    _record(con, "TC", "2026-05", None)
    bc.restamp_equivalent_fingerprints(
        con, "TC",
        current=rb.hazard_fingerprint("cyclone"),
        legacy=rb.legacy_hazard_fingerprint("cyclone"),
    )
    assert bc.completed_months(con, "TC", rb.hazard_fingerprint("cyclone")) == set()


def test_the_re_stamp_touches_only_its_own_hazard(con):
    rb = load_rulebook()
    legacy_tc = rb.legacy_hazard_fingerprint("cyclone")
    _record(con, "TC", "2026-05", legacy_tc)
    _record(con, "FL", "2026-05", legacy_tc)
    bc.restamp_equivalent_fingerprints(
        con, "TC",
        current=rb.hazard_fingerprint("cyclone"), legacy=legacy_tc,
    )
    stored = con.execute(
        "SELECT rulebook_hash FROM haz_backcast_progress WHERE hazard = 'FL'"
    ).fetchone()
    assert stored[0] == legacy_tc, "the flood ledger is not this hazard's to bless"
