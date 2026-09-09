# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Which resources were refused, and whether they are the same ones.

A refusal rate says how often a source said no. It cannot say what kind of
no. The same events refused every run means the refusal belongs to those
events; different events each run means it belongs to the asking. The two
want opposite responses, and this is the direct test — over the run's own
HTTP stream, at the cost of no request at all.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import duckdb
import pytest

from resolver.diagnostics import refusals, run_log

from scripts import build_resolver_debug_bundle as bundle


def _record(url: str, status: int, connector: str = "resolver.connectors.gdacs",
            probe: str | None = None) -> dict:
    return {"connector": connector, "url": url, "status": status, "probe": probe}


def _gdacs(event_id: int) -> str:
    return (
        "https://www.gdacs.org/datareport/resources/FL/"
        f"{event_id}/rss_{event_id}.xml"
    )


class TestResourceKey:
    def test_the_same_event_asked_twice_is_one_resource(self):
        a = refusals.resource_key(_gdacs(1001273))
        b = refusals.resource_key(_gdacs(1001273))
        assert a == b and a is not None

    def test_two_events_are_two_resources(self):
        assert refusals.resource_key(_gdacs(1)) != refusals.resource_key(_gdacs(2))

    def test_the_event_id_survives_when_it_lives_in_the_query(self):
        # GDACS puts the id in the path on one route and the query on the
        # other. A key that dropped the query would call every geteventdata
        # call the same resource.
        one = refusals.resource_key(
            "https://www.gdacs.org/gdacsapi/api/events/geteventdata"
            "?eventtype=FL&eventid=1001273"
        )
        two = refusals.resource_key(
            "https://www.gdacs.org/gdacsapi/api/events/geteventdata"
            "?eventtype=FL&eventid=1001999"
        )
        assert one != two
        assert "1001273" in one

    def test_query_order_does_not_make_two_resources(self):
        a = refusals.resource_key("https://x/y?b=2&a=1")
        b = refusals.resource_key("https://x/y?a=1&b=2")
        assert a == b

    def test_a_cache_buster_is_not_part_of_the_identity(self):
        a = refusals.resource_key("https://x/y?id=7&_=1757000000")
        b = refusals.resource_key("https://x/y?id=7&_=1757000999")
        assert a == b

    def test_a_scheme_change_is_not_a_different_resource(self):
        assert refusals.resource_key("http://x/y") == refusals.resource_key("https://x/y")

    def test_nonsense_yields_no_key_rather_than_a_bad_one(self):
        assert refusals.resource_key("") is None
        assert refusals.resource_key("not a url") is None


class TestReadingTheStream:
    def test_a_404_is_not_a_refusal(self):
        # "It does not exist" and "not for you" are different statements,
        # and a connector walking a sparse archive would otherwise read as
        # throttled.
        got = refusals.refusals_from_records([_record(_gdacs(1), 404)])
        assert got["resolver.connectors.gdacs"].refused == set()
        assert len(got["resolver.connectors.gdacs"].asked) == 1

    @pytest.mark.parametrize("status", sorted(refusals.REFUSAL_STATUSES))
    def test_every_refusal_status_counts(self, status):
        got = refusals.refusals_from_records([_record(_gdacs(1), status)])
        assert len(got["resolver.connectors.gdacs"].refused) == 1

    def test_a_probes_refusal_is_not_a_connectors_refusal(self):
        # A diagnostic asks questions no connector would. One of this
        # repository's checks has already been fooled by counting a probe's
        # 405 as connector traffic.
        got = refusals.refusals_from_records(
            [_record(_gdacs(1), 403, probe="acled_auth")]
        )
        assert got == {}

    def test_connectors_are_kept_apart(self):
        got = refusals.refusals_from_records([
            _record(_gdacs(1), 403),
            _record("https://acleddata.com/x", 403, connector="acled"),
        ])
        assert set(got) == {"resolver.connectors.gdacs", "acled"}

    def test_the_scope_filter_keeps_only_the_named_connectors(self):
        got = refusals.refusals_from_records(
            [_record(_gdacs(1), 403), _record("https://a/b", 403, connector="acled")],
            connectors=["acled"],
        )
        assert set(got) == {"acled"}

    def test_a_run_that_asks_for_too_many_says_so(self):
        records = [_record(f"https://x/{i}", 403) for i in range(refusals.MAX_RESOURCES_PER_RUN + 50)]
        got = refusals.refusals_from_records(records)["resolver.connectors.gdacs"]
        assert got.truncated is True
        assert len(got.asked) == refusals.MAX_RESOURCES_PER_RUN


class TestTheVerdict:
    def _entry(self, refused, asked=None):
        entry = refusals.ConnectorRefusals(connector="gdacs")
        entry.asked = set(asked if asked is not None else refused)
        entry.refused = set(refused)
        return entry

    def test_the_same_resources_refused_twice_reads_as_the_resources(self):
        ids = {f"r{i}" for i in range(40)}
        got = refusals.compare("gdacs", self._entry(ids), "prev", ids, ids)
        assert got.verdict == refusals.VERDICT_SAME
        assert got.overlap == pytest.approx(1.0)

    def test_different_resources_each_run_reads_as_the_asking(self):
        now = {f"a{i}" for i in range(40)}
        prev_refused = {f"b{i}" for i in range(40)}
        prev_asked = now | prev_refused
        got = refusals.compare("gdacs", self._entry(now), "prev", prev_asked, prev_refused)
        assert got.verdict == refusals.VERDICT_DIFFERENT
        assert got.overlap == pytest.approx(0.0)

    def test_a_half_and_half_overlap_claims_neither(self):
        now = {f"r{i}" for i in range(40)}
        prev_refused = {f"r{i}" for i in range(20)}
        got = refusals.compare("gdacs", self._entry(now), "prev", now, prev_refused)
        assert got.verdict == refusals.VERDICT_MIXED

    def test_only_resources_the_previous_run_asked_about_are_comparable(self):
        # The heart of it. A resource the previous run never requested says
        # nothing about whether it would have been refused, and counting it
        # as "not refused then" manufactures a throttling verdict out of a
        # changed work list.
        now = {f"r{i}" for i in range(40)}
        prev_asked = {f"r{i}" for i in range(25)}
        prev_refused = set(prev_asked)
        got = refusals.compare("gdacs", self._entry(now), "prev", prev_asked, prev_refused)
        assert got.n_comparable == 25
        assert got.overlap == pytest.approx(1.0)
        assert got.verdict == refusals.VERDICT_SAME

    def test_too_few_comparable_resources_is_inconclusive_not_a_number(self):
        now = {"r1", "r2"}
        got = refusals.compare("gdacs", self._entry(now), "prev", now, now)
        assert got.verdict == refusals.VERDICT_INCONCLUSIVE

    def test_no_comparable_resources_has_no_overlap_at_all(self):
        got = refusals.compare("gdacs", self._entry({"a"}), "prev", set(), set())
        assert got.overlap is None
        assert got.verdict == refusals.VERDICT_INCONCLUSIVE


class TestTheIssue:
    def _comparison(self, verdict_ids):
        return refusals.compare(
            "resolver.connectors.gdacs",
            TestTheVerdict()._entry(verdict_ids),
            "34222175003", verdict_ids, verdict_ids,
        )

    def test_the_finding_is_info_because_nothing_is_broken(self):
        issue = refusals.issue_for(self._comparison({f"r{i}" for i in range(40)}))
        assert issue.severity == refusals.INFO

    def test_the_title_names_the_pattern_and_the_share(self):
        issue = refusals.issue_for(self._comparison({f"r{i}" for i in range(40)}))
        assert refusals.VERDICT_SAME in issue.title
        assert "100%" in issue.title
        assert "40 comparable" in issue.title

    def test_an_inconclusive_comparison_prints_no_share(self):
        issue = refusals.issue_for(refusals.compare(
            "gdacs", TestTheVerdict()._entry({"a"}), "prev", set(), set()))
        assert "not comparable" in issue.title

    def test_the_evidence_says_no_extra_request_was_made(self):
        issue = refusals.issue_for(self._comparison({f"r{i}" for i in range(40)}))
        assert "no extra request" in issue.evidence

    def test_the_cost_unit_is_resources_not_requests(self):
        # One event can cost several requests. Saying "40 refused requests"
        # when 40 events were refused is the unit fault this repository has
        # already been bitten by.
        issue = refusals.issue_for(self._comparison({f"r{i}" for i in range(40)}))
        assert issue.cost_unit == "refused resources"


class TestStateThatTravels:
    @pytest.fixture()
    def con(self):
        con = duckdb.connect(":memory:")
        refusals.ensure_refusal_table(con)
        yield con
        con.close()

    def _entry(self, asked, refused):
        entry = refusals.ConnectorRefusals(connector="gdacs")
        entry.asked, entry.refused = set(asked), set(refused)
        return entry

    def test_the_first_run_has_no_previous_run_and_says_so(self, con):
        refusals.save_run(con, "run1", self._entry({"a", "b"}, {"a"}))
        assert refusals.previous_run(con, "gdacs", "run1") is None

    def test_the_second_run_reads_the_first(self, con):
        refusals.save_run(con, "run1", self._entry({"a", "b"}, {"a"}),
                          today=dt.date(2026, 9, 1))
        got = refusals.previous_run(con, "gdacs", "run2")
        assert got is not None
        prev_id, asked, refused = got
        assert prev_id == "run1"
        assert asked == {"a", "b"} and refused == {"a"}

    def test_a_run_never_reads_itself(self, con):
        refusals.save_run(con, "run1", self._entry({"a"}, {"a"}))
        refusals.save_run(con, "run2", self._entry({"a"}, {"a"}),
                          today=dt.date(2026, 9, 2))
        prev_id, _, _ = refusals.previous_run(con, "gdacs", "run2")
        assert prev_id == "run1"

    def test_two_runs_on_one_day_are_ordered_by_run_id(self, con):
        """Ordinary: a scoped verification run beside a full one.

        recorded_at is a DATE and cannot separate them, so the previous run
        would otherwise be whichever the table happened to yield.
        """

        day = dt.date(2026, 9, 9)
        refusals.save_run(con, "34222175003", self._entry({"a"}, {"a"}), today=day)
        refusals.save_run(con, "34374766009", self._entry({"b"}, {"b"}), today=day)
        prev_id, asked, _ = refusals.previous_run(con, "gdacs", "34400000000")
        assert prev_id == "34374766009"
        assert asked == {"b"}

    def test_a_non_numeric_run_id_still_orders(self, con):
        """A local run or a test has no GitHub id, and must not sort to the
        front of every listing by being unparseable."""

        day = dt.date(2026, 9, 9)
        refusals.save_run(con, "local-a", self._entry({"a"}, {"a"}), today=day)
        refusals.save_run(con, "34374766009", self._entry({"b"}, {"b"}), today=day)
        prev_id, _, _ = refusals.previous_run(con, "gdacs", "later")
        assert prev_id == "34374766009"

    def test_one_connectors_history_is_not_anothers(self, con):
        refusals.save_run(con, "run1", self._entry({"a"}, {"a"}))
        other = refusals.ConnectorRefusals(connector="acled")
        other.asked, other.refused = {"z"}, {"z"}
        refusals.save_run(con, "run1", other)
        _, asked, _ = refusals.previous_run(con, "gdacs", "run2")
        assert asked == {"a"}

    def test_saving_the_same_run_twice_is_a_no_op(self, con):
        entry = self._entry({"a", "b"}, {"a"})
        refusals.save_run(con, "run1", entry)
        refusals.save_run(con, "run1", entry)
        n = con.execute(
            f"SELECT COUNT(*) FROM {refusals.REFUSAL_TABLE}").fetchone()[0]
        assert n == 2

    def test_old_runs_are_pruned_so_the_table_is_not_a_log(self, con):
        for i in range(refusals.KEEP_RUNS + 3):
            refusals.save_run(
                con, f"run{i:02d}", self._entry({"a"}, {"a"}),
                today=dt.date(2026, 9, 1) + dt.timedelta(days=i),
            )
        refusals.prune(con)
        runs = con.execute(
            f"SELECT COUNT(DISTINCT run_id) FROM {refusals.REFUSAL_TABLE}"
        ).fetchone()[0]
        assert runs == refusals.KEEP_RUNS

    def test_pruning_keeps_the_newest_runs(self, con):
        for i in range(refusals.KEEP_RUNS + 2):
            refusals.save_run(
                con, f"run{i:02d}", self._entry({"a"}, {"a"}),
                today=dt.date(2026, 9, 1) + dt.timedelta(days=i),
            )
        refusals.prune(con)
        kept = {r[0] for r in con.execute(
            f"SELECT DISTINCT run_id FROM {refusals.REFUSAL_TABLE}").fetchall()}
        assert f"run{refusals.KEEP_RUNS + 1:02d}" in kept
        assert "run00" not in kept

    def test_a_secret_in_a_url_is_fingerprinted_on_the_way_in(self, con):
        # This table travels in a public artifact. Capture-time redaction is
        # a first line and never the guarantee.
        refusals.save_run(
            con, "run1",
            self._entry({"host/x?key=s3cr3tvalue"}, {"host/x?key=s3cr3tvalue"}),
            secrets=["s3cr3tvalue"],
        )
        stored = con.execute(
            f"SELECT resource FROM {refusals.REFUSAL_TABLE}").fetchone()[0]
        assert "s3cr3tvalue" not in stored

    def test_an_empty_run_writes_nothing(self, con):
        assert refusals.save_run(con, "run1", self._entry(set(), set())) == 0


# ---------------------------------------------------------------------------
# The whole path, rendered
# ---------------------------------------------------------------------------

def _db(path: Path) -> Path:
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE facts_resolved (iso3 TEXT, ym TEXT, value DOUBLE)")
    con.close()
    return path


def _stream(tmp_path: Path, refused_ids, served_ids) -> Path:
    streams = tmp_path / "runlog"
    streams.mkdir(exist_ok=True)
    with open(streams / f"{run_log.STREAM_HTTP}.jsonl", "w", encoding="utf-8") as fh:
        for event_id in refused_ids:
            fh.write(json.dumps({
                "connector": "resolver.connectors.gdacs",
                "url": _gdacs(event_id), "status": 403, "elapsed_ms": 100.0,
            }) + "\n")
        for event_id in served_ids:
            fh.write(json.dumps({
                "connector": "resolver.connectors.gdacs",
                "url": _gdacs(event_id), "status": 200, "elapsed_ms": 100.0,
            }) + "\n")
    return streams


def _run(tmp_path: Path, name: str, run_id: str, db: Path, refused, served):
    root = tmp_path / name
    root.mkdir()
    diagnostics = root / "diagnostics"
    diagnostics.mkdir()
    return bundle.build_register(
        db_path=db, diagnostics_dir=diagnostics,
        run_log_dir=_stream(root, refused, served),
        staging=root / "reg", environ={"GITHUB_RUN_ID": run_id},
        write_history=False,
    )


class TestTheWholePathRendersIt:
    """Unit tests on a comparison function proved nothing last time.

    The GDACS refusal RATE passed every unit test it had and still never
    reached the printed register, because the early path did not run the
    section that computes it. So this drives the real entry point, twice,
    against a database that carries state from the first run to the second.
    """

    def test_the_first_run_reports_no_pattern_and_records_the_set(self, tmp_path):
        db = _db(tmp_path / "resolver.duckdb")
        register = _run(tmp_path, "one", "111", db, range(40), range(100, 140))
        ids = {i.id for i in register.issues}
        # The rate fires; the pattern cannot, because there is no earlier run.
        assert any(i.startswith("refused_requests_") for i in ids)
        assert not any(i.startswith("refusal_pattern_") for i in ids)
        con = duckdb.connect(str(db))
        try:
            n = con.execute(
                f"SELECT COUNT(*) FROM {refusals.REFUSAL_TABLE}").fetchone()[0]
        finally:
            con.close()
        assert n == 80

    def test_the_second_run_reports_the_pattern_in_the_printed_register(self, tmp_path):
        db = _db(tmp_path / "resolver.duckdb")
        _run(tmp_path, "one", "111", db, range(40), range(100, 140))
        register = _run(tmp_path, "two", "222", db, range(40), range(100, 140))
        pattern = [i for i in register.issues if i.id.startswith("refusal_pattern_")]
        assert pattern, [i.id for i in register.issues]
        assert refusals.VERDICT_SAME in pattern[0].title
        assert "100%" in pattern[0].title

    def test_a_run_refused_different_events_reads_as_the_asking(self, tmp_path):
        db = _db(tmp_path / "resolver.duckdb")
        _run(tmp_path, "one", "111", db, range(40), range(40, 120))
        register = _run(tmp_path, "two", "222", db, range(40, 80), range(0, 40))
        pattern = [i for i in register.issues if i.id.startswith("refusal_pattern_")]
        assert pattern
        assert refusals.VERDICT_DIFFERENT in pattern[0].title

    def test_the_refused_resources_are_written_where_a_person_can_read_them(self, tmp_path):
        db = _db(tmp_path / "resolver.duckdb")
        _run(tmp_path, "one", "111", db, range(40), range(100, 140))
        csv = tmp_path / "one" / "reg" / "http" / "refused_resources.csv"
        assert csv.is_file()
        text = csv.read_text(encoding="utf-8")
        assert "rss_1.xml" in text
        assert "rss_100.xml" not in text  # a served event is not a refusal

    def test_a_quiet_connector_records_nothing(self, tmp_path):
        # Storing every URL of every connector would put an append-only
        # cache back into a database this repository has already compacted
        # once. Only a connector the rate alarm named is recorded.
        db = _db(tmp_path / "resolver.duckdb")
        _run(tmp_path, "one", "111", db, [], range(100, 200))
        con = duckdb.connect(str(db))
        try:
            tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        finally:
            con.close()
        assert refusals.REFUSAL_TABLE not in tables
