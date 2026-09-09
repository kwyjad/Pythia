# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The run issue register: does a registered fault stop shouting, and only it?

Run 34222175003 went green carrying two failed contradiction checks, an
unread EM-DAT rung and a broken flood ceiling. The register exists so that
never happens quietly again, and it earns its place only if two things hold
at once: a fault already diagnosed and owned reports as `known`, once, and
changes no exit code; and a fault of exactly the same shape that nobody has
registered reports as `degraded` and is impossible to miss.

If the first fails the register is noise. If the second fails it is a
blindfold.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import duckdb
import pytest

from resolver.diagnostics import issue_sources, issues as mod

pytestmark = pytest.mark.usefixtures()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _register_file(tmp_path: Path, *ids: str) -> Path:
    """A known-issues file naming ``ids``, each owned and reviewable."""

    lines = ["issues:"]
    for issue_id in ids:
        lines += [
            f"  - id: {issue_id}",
            "    owner: external",
            "    first_seen: 2026-08-05",
            "    review_by: 2099-01-01",
            "    recovers_on_rerun: false",
            "    note: >-",
            f"      {issue_id} is upstream and already escalated.",
        ]
    path = tmp_path / "known_issues.yml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _unread_source_records(source: str, n: int = 2) -> list[dict]:
    """``n`` failed fetches of one source — the shape EM-DAT produced."""

    return [
        {
            "source": source, "hazard": "flood", "ym": "2026-08", "ok": False,
            "failure_class": "auth_rejected",
            "error": "Invalid key passed or insufficient user access",
        }
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# The two halves of the contract
# ---------------------------------------------------------------------------


class TestSeededKnownIssue:
    """A registered fault: reported at `known`, once, and never fatal."""

    def test_a_registered_issue_reports_at_known(self, tmp_path):
        known = mod.KnownIssues.load(_register_file(tmp_path, "emdat_auth_rejected"))
        register = mod.IssueRegister(known=known)
        register.extend(issue_sources.issues_from_source_fetches(
            _unread_source_records("emdat")
        ))

        found = [i for i in register.issues if i.id == "emdat_auth_rejected"]
        assert len(found) == 1
        assert found[0].severity == mod.KNOWN
        assert found[0].owner == "external"
        assert found[0].recovers_on_rerun is False
        assert "escalated" in found[0].note

    def test_it_prints_once_however_many_times_it_happened(self, tmp_path):
        """Two failed fetches are ONE fault, not two lines.

        This is the whole point of a register rather than a log: an issue is
        a distinct fault, and repeating it per occurrence is how the report
        becomes the thing it was meant to replace.
        """

        known = mod.KnownIssues.load(_register_file(tmp_path, "emdat_auth_rejected"))
        register = mod.IssueRegister(known=known)
        register.extend(issue_sources.issues_from_source_fetches(
            _unread_source_records("emdat", n=7)
        ))

        text = mod.render_text(register)
        assert text.count("[emdat_auth_rejected]") == 1
        # The cost is still the full count — quieter, not less honest.
        assert register.issues[0].cost == 7
        assert "7 failed fetches" in text

    def test_a_known_issue_is_a_notice_not_an_error(self, tmp_path):
        """The annotation level is what actually quietens the run."""

        known = mod.KnownIssues.load(_register_file(tmp_path, "emdat_auth_rejected"))
        register = mod.IssueRegister(known=known)
        register.extend(issue_sources.issues_from_source_fetches(
            _unread_source_records("emdat")
        ))

        annotations = mod.render_annotations(register)
        assert len(annotations) == 1
        assert annotations[0].startswith("::notice ")
        assert "::error" not in annotations[0]

    def test_a_known_issue_does_not_change_the_exit_code(self, tmp_path, monkeypatch):
        """A registered fault the run cannot repair must not fail the run.

        The register step exits 0 on purpose. Two failed checks throwing
        away 2,887 successful ReliefWeb fetches is a worse outcome than a
        green run with a loud report in it.
        """

        from scripts.ci import report_run_issues

        payload = {
            "counts": {"blocking": 0, "degraded": 0, "known": 1, "info": 0},
            "notes": [],
            "issues": [{
                "id": "emdat_auth_rejected", "severity": "known",
                "title": "EM-DAT could not be read.", "cost": 2,
                "cost_unit": "failed fetches", "owner": "external",
                "recovers_on_rerun": False,
            }],
        }
        written = tmp_path / "issues.json"
        written.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        code = report_run_issues.main([
            "--from-json", str(written), "--diagnostics-dir", str(tmp_path),
        ])
        assert code == 0

    def test_an_overdue_entry_still_reports_but_says_it_is_overdue(self, tmp_path):
        """A suppression that never expires is an invisible fault."""

        path = tmp_path / "known_issues.yml"
        path.write_text(
            "issues:\n"
            "  - id: emdat_auth_rejected\n"
            "    owner: external\n"
            "    review_by: 2020-01-01\n"
            "    note: chased with EM-DAT\n",
            encoding="utf-8",
        )
        register = mod.IssueRegister(known=mod.KnownIssues.load(path))
        register.extend(issue_sources.issues_from_source_fetches(
            _unread_source_records("emdat")
        ))

        issue = register.issues[0]
        assert issue.severity == mod.KNOWN
        assert issue.overdue is True
        assert "OVERDUE" in mod.render_text(register)
        assert "overdue for review" in mod.render_markdown(register)

    def test_an_entry_with_no_review_date_is_overdue_from_the_start(self, tmp_path):
        path = tmp_path / "known_issues.yml"
        path.write_text(
            "issues:\n  - id: emdat_auth_rejected\n    owner: external\n",
            encoding="utf-8",
        )
        register = mod.IssueRegister(known=mod.KnownIssues.load(path))
        register.extend(issue_sources.issues_from_source_fetches(
            _unread_source_records("emdat")
        ))
        assert register.issues[0].overdue is True


class TestUnregisteredIssue:
    """The same fault, unregistered: `degraded`, and impossible to miss."""

    def test_the_same_shape_unregistered_reports_at_degraded(self, tmp_path):
        # The register names EM-DAT and nothing else; ReliefWeb's failure is
        # identical in shape and nobody has claimed it.
        known = mod.KnownIssues.load(_register_file(tmp_path, "emdat_auth_rejected"))
        register = mod.IssueRegister(known=known)
        register.extend(issue_sources.issues_from_source_fetches(
            _unread_source_records("emdat") + _unread_source_records("reliefweb")
        ))

        by_id = {i.id: i for i in register.issues}
        assert by_id["emdat_auth_rejected"].severity == mod.KNOWN
        assert by_id["source_unread_reliefweb"].severity == mod.DEGRADED

    def test_an_unregistered_issue_is_an_error_annotation(self, tmp_path):
        register = mod.IssueRegister(known=mod.KnownIssues(entries={}))
        register.extend(issue_sources.issues_from_source_fetches(
            _unread_source_records("reliefweb")
        ))
        assert mod.render_annotations(register)[0].startswith("::error ")

    def test_worst_first(self, tmp_path):
        register = mod.IssueRegister(known=mod.KnownIssues(entries={}))
        register.add(mod.Issue(id="c", severity=mod.INFO, title="a measurement"))
        register.add(mod.Issue(id="b", severity=mod.KNOWN, title="an owned fault"))
        register.add(mod.Issue(id="a", severity=mod.DEGRADED, title="a live fault"))
        register.add(mod.Issue(id="z", severity=mod.BLOCKING, title="no output"))
        assert [i.id for i in register.issues] == ["z", "a", "b", "c"]

    def test_blocking_is_never_demoted_by_the_register(self, tmp_path):
        """A run that produced no output is not made acceptable by a note."""

        known = mod.KnownIssues.load(_register_file(tmp_path, "emdat_auth_rejected"))
        register = mod.IssueRegister(known=known)
        register.add(mod.Issue(
            id="emdat_auth_rejected", severity=mod.BLOCKING,
            title="the run wrote no database",
        ))
        assert register.issues[0].severity == mod.BLOCKING


# ---------------------------------------------------------------------------
# Collectors: one issue per fault, from evidence that already exists
# ---------------------------------------------------------------------------


class TestCollectors:
    def test_a_failed_check_becomes_one_issue_named_after_the_check(self):
        issues = issue_sources.issues_from_checks([
            {"name": "no_assessed_cell_lacks_a_reason_for_having_no_row",
             "verdict": "FAIL", "left": "29,035 unexplained", "right": "0",
             "detail": "29,035 cells produced no row and named no reason"},
            {"name": "something_fine", "verdict": "PASS", "left": "0", "right": "0",
             "detail": ""},
        ])
        assert [i.id for i in issues] == [
            "no_assessed_cell_lacks_a_reason_for_having_no_row"
        ]
        # A detail that opens with a count is stating its own cost.
        assert issues[0].cost == 29035

    def test_a_check_may_name_the_faults_it_found(self):
        """One live source must not stand for two dead ones.

        The vintage check finds several stale sources at once; its own name
        can only ever say "something is stale", so it names them.
        """

        issues = issue_sources.issues_from_checks([{
            "name": "no_stored_forecast_vintage_past_its_threshold",
            "verdict": "FAIL", "left": "2 of 3 sources", "right": "at most 45 days",
            "detail": "acled_cast latest=2025-12-01 (281d); views latest=2026-07-01 (69d)",
            "issues": [
                {"id": "acled_cast_stale_vintage", "title": "ACLED CAST is 281 days stale",
                 "cost": 281, "cost_unit": "days stale", "owner": "external"},
                {"id": "views_stale_vintage", "title": "VIEWS is 69 days stale",
                 "cost": 69, "cost_unit": "days stale", "owner": "external"},
            ],
        }])
        assert sorted(i.id for i in issues) == [
            "acled_cast_stale_vintage", "views_stale_vintage"
        ]

    def test_a_check_that_could_not_run_is_an_issue_too(self):
        """A check that errored verified nothing, which is not the same as passing."""

        issues = issue_sources.issues_from_checks([
            {"name": "a_check", "verdict": "ERROR", "left": "", "right": "",
             "detail": "BinderException: no such column"},
        ])
        assert len(issues) == 1
        assert "could not run" in issues[0].title

    def test_a_cache_served_rung_is_info_not_a_fault(self):
        """A STALE rung still answers. Only an UNREAD one costs the ladder a rung."""

        issues = issue_sources.issues_from_source_fetches([
            {"source": "emdat", "ok": True, "served_from_cache": True},
        ])
        assert [i.severity for i in issues] == [mod.INFO]

    def test_a_connector_that_claimed_rows_its_table_cannot_show(self):
        issues = issue_sources.issues_from_reconciliation([
            {"connector": "nmme", "status": "ok", "claimed": 2408,
             "touched": 0, "table": "seasonal_forecasts", "agrees": "**NO**"},
            {"connector": "acled", "status": "ok", "claimed": 424,
             "touched": 424, "table": "facts_resolved", "agrees": "yes"},
            {"connector": "crisiswatch", "status": "ok", "claimed": 78,
             "touched": 0, "table": "crisiswatch_entries",
             "agrees": "unchanged on purpose"},
        ])
        assert [i.id for i in issues] == ["connector_wrote_nothing_nmme"]
        assert issues[0].cost == 2408

    def test_warnings_stay_in_the_log_index(self):
        """A run legitimately warns dozens of times.

        A register carrying every warning is a log with a border round it,
        and the reader stops opening it — which is the failure this whole
        thing exists to prevent.
        """

        issues = issue_sources.issues_from_log_histogram([
            ("phase3_nmme.log", "ERROR: could not read <NUM> vintages", 3),
            ("phase3_nmme.log", "WARNING: skipped <ISO3>", 252),
        ])
        assert len(issues) == 1
        assert issues[0].cost == 3


class TestMergingOneFaultFoundTwice:
    """One fault, two collectors, and a cost that must not lie.

    EM-DAT's lockout arrives twice in a real run: as failed fetches from the
    source stream, and as a failed contradiction check counting rows. Adding
    them gave "29,041 failed fetches" — a number wearing the unit of a
    different measurement, which is the one thing a cost column may never
    do.
    """

    def test_costs_in_the_same_unit_are_added(self):
        register = mod.IssueRegister(known=mod.KnownIssues(entries={}))
        register.add(mod.Issue(id="x", severity=mod.DEGRADED, title="t",
                               cost=3, cost_unit="failed fetches"))
        register.add(mod.Issue(id="x", severity=mod.DEGRADED, title="t",
                               cost=4, cost_unit="failed fetches"))
        assert register.issues[0].cost_text() == "7 failed fetches"

    def test_costs_in_different_units_are_never_added(self):
        register = mod.IssueRegister(known=mod.KnownIssues(entries={}))
        register.add(mod.Issue(id="x", severity=mod.DEGRADED, title="t",
                               cost=6, cost_unit="failed fetches"))
        register.add(mod.Issue(id="x", severity=mod.DEGRADED, title="t",
                               cost=29035, cost_unit="rows"))
        issue = register.issues[0]
        assert issue.cost_text() == "6 failed fetches"
        assert "also 29,035 rows" in issue.evidence

    def test_a_measured_cost_fills_an_unmeasured_one(self):
        register = mod.IssueRegister(known=mod.KnownIssues(entries={}))
        register.add(mod.Issue(id="x", severity=mod.DEGRADED, title="t"))
        register.add(mod.Issue(id="x", severity=mod.DEGRADED, title="t",
                               cost=12, cost_unit="rows"))
        assert register.issues[0].cost_text() == "12 rows"

    def test_the_worse_severity_wins(self):
        register = mod.IssueRegister(known=mod.KnownIssues(entries={}))
        register.add(mod.Issue(id="x", severity=mod.INFO, title="t"))
        register.add(mod.Issue(id="x", severity=mod.DEGRADED, title="t"))
        assert register.issues[0].severity == mod.DEGRADED


class TestRendering:
    def test_an_unmeasured_cost_says_so_rather_than_printing_zero(self):
        """Zero is a claim that the fault cost nothing. Silence is not."""

        issue = mod.Issue(id="x", severity=mod.DEGRADED, title="t")
        assert issue.cost_text() == "not measured"
        assert mod.Issue(id="x", severity=mod.INFO, title="t", cost=0,
                         cost_unit="rows").cost_text() == "0 rows"

    def test_a_clean_run_says_so_plainly(self):
        register = mod.IssueRegister(known=mod.KnownIssues(entries={}))
        text = mod.render_text(register)
        assert "Nothing to report" in text
        assert "0 blocking, 0 degraded, 0 known, 0 info" in text

    def test_markdown_survives_a_pipe_in_the_evidence(self):
        register = mod.IssueRegister(known=mod.KnownIssues(entries={}))
        register.add(mod.Issue(id="x", severity=mod.DEGRADED,
                               title="a | b", evidence="c | d\ne"))
        table = mod.render_markdown(register).splitlines()
        row = [l for l in table if l.startswith("| degraded ")][0]
        assert row.count("|") == 7

    def test_a_round_trip_through_json_keeps_the_severities(self, tmp_path):
        """The printed register and the bundled one must not disagree.

        The bundle copies `diagnostics/issues.json` rather than rebuilding,
        and re-applying a register that may have moved since would be a way
        for the two to say different things about one run.
        """

        register = mod.IssueRegister(known=mod.KnownIssues(entries={}))
        register.add(mod.Issue(id="a", severity=mod.KNOWN, title="owned",
                               owner="external", note="chased"))
        register.add(mod.Issue(id="b", severity=mod.DEGRADED, title="live"))
        path = tmp_path / "issues.json"
        mod.write_outputs(register, json_path=path, markdown_path=tmp_path / "issues.md")

        again = mod.read_register(path)
        assert {i.id: i.severity for i in again.issues} == {
            "a": mod.KNOWN, "b": mod.DEGRADED,
        }

    def test_history_makes_a_six_week_fault_read_differently(self):
        register = mod.IssueRegister(known=mod.KnownIssues(entries={}))
        register.add(mod.Issue(id="a", severity=mod.DEGRADED, title="t"))
        register.add(mod.Issue(id="b", severity=mod.DEGRADED, title="t"))
        register.apply_history({"a": {"first_seen": "2026-07-28", "runs_seen": 5}})

        by_id = {i.id: i for i in register.issues}
        assert by_id["a"].age_text() == "seen in 6 runs since 2026-07-28"
        assert by_id["b"].age_text() == "first seen in this run"


class TestTheShippedRegister:
    """The three seeded entries, held to what the run actually observed."""

    def test_the_three_confirmed_external_faults_are_registered(self):
        known = mod.KnownIssues.load()
        assert known.problems == []
        assert set(known.entries) == {
            "emdat_auth_rejected",
            "acled_cast_stale_vintage",
            "views_stale_vintage",
        }

    def test_every_entry_names_an_owner_a_note_and_a_review_date(self):
        known = mod.KnownIssues.load()
        for issue_id, entry in known.entries.items():
            assert entry.get("owner") in ("external", "pythia"), issue_id
            assert entry.get("note"), issue_id
            assert entry.get("review_by"), issue_id

    def test_the_review_dates_are_chosen_per_fault_not_stamped_uniformly(self):
        """One date on every entry is a placeholder wearing a date's clothes.

        Three faults with different remedies — a credential the owner
        renews, a vendor support request, an escalation nobody has sent —
        cannot honestly share one review interval. When they do, the date
        was not chosen; a quarter was picked so nothing would fire.
        """

        known = mod.KnownIssues.load()
        dates = {str(e.get("review_by")) for e in known.entries.values()}
        assert len(dates) == len(known.entries), (
            f"every entry carries the same review date: {dates}"
        )

    def test_no_review_date_is_parked_where_it_never_fires(self):
        """A date years out is a suppression with no expiry, in disguise.

        Anchoring the horizon to `first_seen` would be wrong: a fault can be
        legitimately old and only recently escalated, which is exactly the
        ACLED CAST case. What cannot be right is one entry parked far beyond
        the others, so the test is the SPREAD — it stays meaningful when
        somebody renews a date, and still catches a 2099 left in the file.
        """

        known = mod.KnownIssues.load()
        due = sorted(
            dt.date.fromisoformat(str(e["review_by"]))
            for e in known.entries.values()
        )
        for issue_id, entry in known.entries.items():
            first = dt.date.fromisoformat(str(entry["first_seen"]))
            assert dt.date.fromisoformat(str(entry["review_by"])) > first, issue_id
        assert (due[-1] - due[0]).days <= 180, (
            f"review dates span {(due[-1] - due[0]).days} days: "
            f"{due[0]} to {due[-1]}"
        )

    def test_every_review_date_says_why_that_interval(self):
        """A date with no reasoning beside it is a date nobody can revise.

        The next person to reach one of these has to decide whether to chase
        it or move it, and cannot without knowing what the interval was for.
        """

        text = mod.KNOWN_ISSUES_PATH.read_text(encoding="utf-8")
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if not line.strip().startswith("review_by:"):
                continue
            preceding = lines[i - 1].strip() if i else ""
            assert preceding.startswith("#"), (
                f"line {i + 1} sets a review date with no reason beside it"
            )

    def test_the_overdue_line_fires_on_the_shipped_register(self):
        """Held to a pinned date, so it is a property and not the calendar.

        On 2026-09-09 the EM-DAT entry is a month past its review date and
        the other two are not. That is what a live run prints, and it is the
        difference between a register that expires and one that does not.
        """

        known = mod.KnownIssues.load()
        today = dt.date(2026, 9, 9)
        overdue = set()
        for issue_id in known.entries:
            issue = mod.Issue(id=issue_id, severity=mod.DEGRADED,
                              title="t", evidence="e")
            if known.apply(issue, today=today).overdue:
                overdue.add(issue_id)
        assert overdue == {"emdat_auth_rejected"}, overdue

    def test_the_seeded_ids_match_the_ids_the_collectors_emit(self):
        """A register keyed on an id nothing emits suppresses nothing."""

        known = mod.KnownIssues.load()
        emitted = (
            set(issue_sources.CHECK_ISSUE_IDS.values())
            | set(issue_sources.SOURCE_ISSUE_IDS.values())
            | set(issue_sources.VINTAGE_ISSUE_IDS.values())
        )
        assert set(known.entries) <= emitted

    def test_a_missing_register_suppresses_nothing_rather_than_everything(self, tmp_path):
        """Failing to read the register must never hide a fault."""

        known = mod.KnownIssues.load(tmp_path / "does_not_exist.yml")
        assert known.entries == {}
        assert known.problems
        register = mod.IssueRegister(known=known)
        register.extend(issue_sources.issues_from_source_fetches(
            _unread_source_records("emdat")
        ))
        assert register.issues[0].severity == mod.DEGRADED


class TestTheAnnotationSaysWhenAnEntryIsOverdue:
    """The Actions annotations are where a reader looks first.

    An overdue entry is the one that most needs looking at — it is a
    suppression nobody has re-read — and a quiet `::notice::` that says
    nothing about it reads as settled.
    """

    def _register(self, tmp_path: Path, review_by: str):
        path = tmp_path / "known.yml"
        path.write_text(
            "issues:\n"
            "  - id: some_fault\n"
            "    owner: external\n"
            "    note: somebody is chasing it\n"
            f"    review_by: {review_by}\n",
            encoding="utf-8",
        )
        register = mod.IssueRegister(known=mod.KnownIssues.load(path))
        register.add(mod.Issue(id="some_fault", severity=mod.DEGRADED,
                               title="a fault", evidence="e"))
        return register

    def test_an_overdue_entry_says_so_in_the_annotation(self, tmp_path):
        register = self._register(tmp_path, "2020-01-01")
        line = mod.render_annotations(register)[0]
        assert "::notice" in line
        assert "overdue" in line.lower()
        assert "2020-01-01" in line

    def test_an_entry_within_its_review_window_stays_quiet(self, tmp_path):
        register = self._register(tmp_path, "2099-01-01")
        line = mod.render_annotations(register)[0]
        assert "overdue" not in line.lower()


class TestTheHistorySurvivesBetweenRuns:
    """`first_seen` and `runs_seen` need state carried between runs.

    There is only one store that travels: the canonical database, because
    nothing in CI may push to `main`. So the claim these fields rest on is
    that a row written before the canonical upload is IN the uploaded file
    — and the upload copies `data/resolver.duckdb` alone, so a write left
    in the WAL is uploaded as if it never happened.

    These drive real files rather than `:memory:`, because an in-memory
    database cannot fail the way a WAL can.
    """

    def _issue(self, issue_id="some_fault", severity=None):
        return mod.Issue(
            id=issue_id, severity=severity or mod.DEGRADED,
            title="a fault", evidence="e",
        )

    def _run(self, path, issue_id="some_fault", today=None):
        con = duckdb.connect(str(path))
        try:
            register = mod.IssueRegister()
            register.apply_history(mod.load_history(con))
            register.add(self._issue(issue_id))
            # apply_history runs before add in the real path too; re-apply so
            # the freshly added issue picks the prior run up.
            register.apply_history(mod.load_history(con))
            mod.save_history(con, register, today=today)
            con.execute("CHECKPOINT")
            return register
        finally:
            con.close()

    def test_a_second_run_counts_the_first(self, tmp_path):
        db = tmp_path / "resolver.duckdb"
        first = self._run(db, today=dt.date(2026, 9, 1))
        assert first.issues[0].runs_seen == 1, (
            "the first run has seen the fault once — itself"
        )
        second = self._run(db, today=dt.date(2026, 9, 9))
        assert second.issues[0].runs_seen == 2

    def test_first_seen_is_the_first_run_and_never_moves(self, tmp_path):
        db = tmp_path / "resolver.duckdb"
        self._run(db, today=dt.date(2026, 8, 1))
        third = None
        for day in (dt.date(2026, 8, 15), dt.date(2026, 9, 9)):
            third = self._run(db, today=day)
        assert third.issues[0].first_seen == "2026-08-01"
        assert third.issues[0].runs_seen == 3

    def test_the_history_is_in_the_file_the_upload_copies(self, tmp_path):
        """The sharp one. The canonical upload copies the .duckdb file and
        NOT its .wal, so an uncheckpointed write is uploaded as if it had
        never happened — and the counter silently resets to 1 forever."""

        db = tmp_path / "resolver.duckdb"
        self._run(db, today=dt.date(2026, 9, 1))
        uploaded = tmp_path / "uploaded.duckdb"
        uploaded.write_bytes(db.read_bytes())  # the file alone, no .wal

        con = duckdb.connect(str(uploaded), read_only=True)
        try:
            rows = con.execute(
                f"SELECT issue_id, runs_seen FROM {mod.HISTORY_TABLE}"
            ).fetchall()
        finally:
            con.close()
        assert rows == [("some_fault", 1)]

    def test_a_run_that_finds_nothing_else_still_records_the_table(self, tmp_path):
        """A run writing no new rows must not lose the count.

        The table's own CREATE is a write too, which is why the checkpoint
        runs whether or not a row was written.
        """

        db = tmp_path / "resolver.duckdb"
        con = duckdb.connect(str(db))
        try:
            mod.save_history(con, mod.IssueRegister())  # no issues at all
            con.execute("CHECKPOINT")
        finally:
            con.close()
        con = duckdb.connect(str(db), read_only=True)
        try:
            tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        finally:
            con.close()
        assert mod.HISTORY_TABLE in tables

    def test_a_history_write_that_fails_costs_the_counter_and_nothing_else(self):
        class Broken:
            def execute(self, *a, **k):
                raise RuntimeError("disk full")

        assert mod.save_history(Broken(), mod.IssueRegister()) == 0
        assert mod.load_history(Broken()) == {}
