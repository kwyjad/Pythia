# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""One fault, one record — even when four log shapes describe it.

Run 34081262443 reported the EM-DAT lockout five times: once as `known`,
because the register carries it, and four more as `degraded` with owner
`pythia`, each raising an error annotation. Four of that run's seven degraded
issues were that one fault. The cause is that an id from the ERROR histogram
is minted from the log NAME and the normalised line, so it can never equal a
registered fault's id, while ``KnownIssues.apply`` matches on the id alone.

Matching now decides SEVERITY and GROUPING and never visibility: a matched
shape is folded into the record that owns it, as an evidence line carrying
its own count. And an entry absorbs only the signatures it lists explicitly,
because a register that swallowed everything from a source would silence the
next, unrelated fault from it — which is the failure the register exists to
prevent rather than to commit.
"""

from __future__ import annotations

from resolver.diagnostics import issue_sources
from resolver.diagnostics.issues import (
    DEGRADED,
    KNOWN,
    Issue,
    IssueRegister,
    KnownIssues,
    render_markdown,
    render_text,
)

#: A register with one entry that lists its shapes, shaped like the shipped
#: `emdat_auth_rejected` entry.
def _known() -> KnownIssues:
    return KnownIssues(entries={
        "emdat_auth_rejected": {
            "id": "emdat_auth_rejected",
            "owner": "external",
            "review_by": "2099-01-01",
            "first_seen": "2026-08-05",
            "note": "EM-DAT rejects the key; the account tier has to change.",
            "log_connector": "[emdat]",
            "log_signatures": [
                "[emdat] fetch failed for",
                "unavailable and the cache holds nothing",
                "invalid key passed or insufficient user access",
            ],
        },
    })


def _log_issues(*shapes):
    return issue_sources.issues_from_log_histogram(
        [(log, f"ERROR: {message}", count) for log, message, count in shapes]
    )


# ---------------------------------------------------------------------------
# Absorption
# ---------------------------------------------------------------------------


def test_four_matched_log_shapes_become_one_record():
    register = IssueRegister(known=_known())
    register.add(Issue(
        id="emdat_auth_rejected", severity=DEGRADED,
        title="emdat could not be read.", cost=6, cost_unit="failed fetches",
    ))
    register.extend(_log_issues(
        ("phase25_haz_flood.log", "[emdat] fetch failed for flood <YM>", 252),
        ("phase25_haz_cyclone.log", "[emdat] fetch failed for cyclone <YM>", 118),
        ("phase25_haz_flood.log",
         "[emdat] <ISO3> <YM> unavailable and the cache holds nothing", 4),
        ("phase25_haz_drought.log",
         "[emdat] Invalid key passed or insufficient user access", 9),
    ))

    assert [i.id for i in register.issues] == ["emdat_auth_rejected"]
    assert register.counts()[DEGRADED] == 0
    assert register.counts()[KNOWN] == 1


def test_the_absorbed_shapes_are_printed_with_their_counts():
    """Grouped, not hidden. A record that swallowed four shapes has to name
    them, or the register has quietened the evidence with the noise."""

    register = IssueRegister(known=_known())
    register.add(Issue(id="emdat_auth_rejected", severity=DEGRADED, title="unread"))
    register.extend(_log_issues(
        ("phase25_haz_flood.log", "[emdat] fetch failed for flood <YM>", 252),
    ))

    text = render_text(register)
    assert "log shapes under this fault (1)" in text
    assert "[emdat] fetch failed for flood <YM>" in text
    assert "252 log lines" in text

    markdown = render_markdown(register)
    assert "log shapes folded under this fault (1)" in markdown


def test_an_absorbed_shape_raises_a_notice_not_an_error():
    """The whole point of registering a fault is that it stops shouting."""

    from resolver.diagnostics.issues import render_annotations

    register = IssueRegister(known=_known())
    register.add(Issue(id="emdat_auth_rejected", severity=DEGRADED, title="unread"))
    register.extend(_log_issues(
        ("phase25_haz_flood.log", "[emdat] fetch failed for flood <YM>", 252),
    ))
    annotations = render_annotations(register)
    assert not [a for a in annotations if a.startswith("::error")]
    assert [a for a in annotations if a.startswith("::notice")]


def test_a_matched_shape_arriving_before_its_owner_is_not_lost():
    """The histogram runs last in practice; this is the ordering accident."""

    register = IssueRegister(known=_known())
    register.extend(_log_issues(
        ("phase25_haz_flood.log", "[emdat] fetch failed for flood <YM>", 252),
    ))
    assert [i.id for i in register.issues] == ["emdat_auth_rejected"]
    assert register.issues[0].placeholder is True

    register.add(Issue(
        id="emdat_auth_rejected", severity=DEGRADED,
        title="emdat could not be read, so every cell resolved without it.",
    ))
    owner = register.issues[0]
    assert owner.placeholder is False
    assert "could not be read" in owner.title
    assert owner.absorbed  # the evidence survived the ordering


# ---------------------------------------------------------------------------
# An entry absorbs only what it lists
# ---------------------------------------------------------------------------


def test_an_unlisted_shape_from_a_registered_connector_still_reports():
    """A new EM-DAT fault must not be silenced by the old one being known."""

    register = IssueRegister(known=_known())
    register.extend(_log_issues(
        ("phase25_haz_flood.log", "[emdat] the response schema has changed", 3),
    ))

    issues = register.issues
    assert len(issues) == 1
    assert issues[0].id.startswith("log_error_")
    assert issues[0].severity == DEGRADED


def test_an_unlisted_shape_says_the_connector_has_a_registered_issue():
    """Otherwise the reader assumes it is the known one again."""

    register = IssueRegister(known=_known())
    register.extend(_log_issues(
        ("phase25_haz_flood.log", "[emdat] the response schema has changed", 3),
    ))
    evidence = register.issues[0].evidence
    assert "emdat_auth_rejected" in evidence
    assert "does not cover this signature" in evidence


def test_a_shape_from_an_unregistered_connector_is_untouched():
    register = IssueRegister(known=_known())
    register.extend(_log_issues(
        ("phase3_nmme.log", "[nmme] the ftp route answered 404", 2),
    ))
    issue = register.issues[0]
    assert issue.severity == DEGRADED
    assert "registered issue" not in issue.evidence


def test_a_registered_entry_with_no_signatures_absorbs_nothing():
    """The two vintage entries list none, on purpose: their fault is found by
    a contradiction check and logged as a WARNING, so nothing can match."""

    known = KnownIssues(entries={
        "views_stale_vintage": {"id": "views_stale_vintage", "owner": "external"},
    })
    register = IssueRegister(known=known)
    register.extend(_log_issues(
        ("phase3_views.log", "[views] something failed", 1),
    ))
    assert register.issues[0].id.startswith("log_error_")


def test_warnings_are_still_left_in_the_log_index():
    """A register carrying every WARNING is a log with a border round it."""

    register = IssueRegister(known=_known())
    register.extend(issue_sources.issues_from_log_histogram(
        [("phase25_haz_flood.log", "WARNING: [emdat] fetch failed for flood", 40)]
    ))
    assert register.issues == []


def test_matching_is_case_insensitive_over_the_normalised_line():
    register = IssueRegister(known=_known())
    register.extend(_log_issues(
        ("phase25_haz_flood.log", "[EMDAT] FETCH FAILED FOR flood <YM>", 1),
    ))
    assert [i.id for i in register.issues] == ["emdat_auth_rejected"]


def test_the_shipped_register_lists_the_shapes_emdat_actually_logs():
    """Written out by hand, so a test has to hold them to the source."""

    from pathlib import Path

    known = KnownIssues.load()
    assert not known.problems, known.problems
    entry = known.entries["emdat_auth_rejected"]
    assert entry["log_connector"] == "[emdat]"

    emdat_source = (
        Path(__file__).resolve().parents[2]
        / "resolver" / "hazard_resolution" / "emdat.py"
    ).read_text("utf-8").lower()
    for signature in entry["log_signatures"]:
        if signature == "invalid key passed or insufficient user access":
            # EM-DAT's own words, quoted back in the response body rather
            # than written by this repo.
            continue
        assert signature.lower() in emdat_source, (
            f"{signature!r} is listed in known_issues.yml but emdat.py logs "
            "nothing of the sort"
        )


def test_a_blocking_issue_is_never_absorbed_or_demoted():
    """A run that could not write its output is not something a register
    entry can make acceptable."""

    from resolver.diagnostics.issues import BLOCKING

    register = IssueRegister(known=_known())
    issue = register.add(Issue(
        id="emdat_auth_rejected", severity=BLOCKING, title="the run wrote nothing",
    ))
    assert issue.severity == BLOCKING
