# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Three smaller faults from run 34222175003, reported rather than papered over.

Each is a measurement the run already made and then dropped: the share of
GDACS requests that were refused, whether the CrisisWatch backfill's budget
bought anything, and which workflow is supposed to fill an empty table. None
of them is a code fault the run could repair; all three are things a reader
has to be told.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from resolver.diagnostics import run_log

duckdb = pytest.importorskip("duckdb")

from scripts import build_resolver_debug_bundle as bundle  # noqa: E402


def _minimal_db(path: Path) -> Path:
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE facts_resolved (iso3 TEXT, ym TEXT, value DOUBLE)")
    con.close()
    return path


def _register_before_the_upload(tmp_path: Path, streams: Path | None = None):
    """The register as the pre-upload step builds it, which is what prints.

    Separate from the bundle path on purpose. Any collector the early path
    does not run reaches the zip and never reaches the person reading the
    Actions log, which is the fault this whole thing exists to end — and it
    happened once already, to the GDACS refusal rate.
    """

    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    return bundle.build_register(
        db_path=_minimal_db(tmp_path / "resolver.duckdb"),
        diagnostics_dir=diagnostics, run_log_dir=streams,
        staging=tmp_path / "reg", environ={}, write_history=False,
    )


def _register(tmp_path: Path, streams: Path | None = None):
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    bundle.build_bundle(
        out_path=tmp_path / "b.zip",
        db_path=_minimal_db(tmp_path / "resolver.duckdb"),
        diagnostics_dir=diagnostics, run_log_dir=streams,
        staging=tmp_path / "stg", environ={},
    )
    return bundle.read_register_from(tmp_path / "stg")


# ---------------------------------------------------------------------------
# GDACS refusals
# ---------------------------------------------------------------------------


class TestTheRefusalRateIsReported:
    """108 of 243 datareport fetches answered 403 and nothing said so."""

    def _http_stream(self, tmp_path: Path, n_403: int, n_200: int) -> Path:
        streams = tmp_path / "runlog"
        streams.mkdir(exist_ok=True)
        with open(streams / f"{run_log.STREAM_HTTP}.jsonl", "w", encoding="utf-8") as fh:
            for i in range(n_403):
                fh.write(json.dumps({
                    "connector": "resolver.connectors.gdacs",
                    "url": f"https://www.gdacs.org/datareport/resources/FL/{i}/rss_{i}.xml",
                    "status": 403, "elapsed_ms": 120.0, "response_bytes": 500,
                }) + "\n")
            for i in range(n_200):
                fh.write(json.dumps({
                    "connector": "resolver.connectors.gdacs",
                    "url": "https://www.gdacs.org/xml/rss_fl_3m.xml",
                    "status": 200, "elapsed_ms": 300.0, "response_bytes": 90000,
                }) + "\n")
        return streams

    def test_a_refused_connector_is_named_with_its_rate(self, tmp_path):
        register = _register(tmp_path, self._http_stream(tmp_path, 108, 135))
        issue = next(
            i for i in register.issues
            if i.id.startswith("refused_requests_resolver_connectors_gdacs")
        )
        assert issue.severity == "degraded"
        assert "44%" in issue.title
        assert "108 of 243" in issue.title
        assert issue.cost == 108

    def test_the_report_says_it_is_reported_not_retried(self, tmp_path):
        """The retry question was measured twice and settled.

        Cutting GDACS volume by 73% moved the refusal rate 79.6% to 76%,
        and 153 of 291 events were served on the FIRST request while 138
        were refused on all four attempts. A rate limit eases when the rate
        falls. So a 403 is asked once, and the missing half was telling
        anyone about it.
        """

        register = _register(tmp_path, self._http_stream(tmp_path, 108, 135))
        issue = next(i for i in register.issues if i.id.startswith("refused_requests_"))
        assert "not retried" in issue.evidence

    def test_a_healthy_connector_raises_nothing(self, tmp_path):
        register = _register(tmp_path, self._http_stream(tmp_path, 2, 200))
        assert not [i for i in register.issues if i.id.startswith("refused_requests_")]

    def test_a_handful_of_requests_is_not_a_rate(self, tmp_path):
        """Three refusals out of four is arithmetic on noise."""

        register = _register(tmp_path, self._http_stream(tmp_path, 3, 1))
        assert not [i for i in register.issues if i.id.startswith("refused_requests_")]


# ---------------------------------------------------------------------------
# CrisisWatch backfill
# ---------------------------------------------------------------------------


class TestTheBackfillRecoveryRate:
    def _backfill_stream(self, tmp_path: Path, payload: dict) -> Path:
        streams = tmp_path / "runlog"
        streams.mkdir(exist_ok=True)
        name = bundle.BundleBuilder.CRISISWATCH_BACKFILL_STREAM
        with open(streams / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
        return streams

    def test_the_writer_and_the_reader_name_the_same_stream(self):
        """Two literals for one contract is a stream nobody reads.

        The scraper runs as its own process and the bundle cannot import it
        (bs4 is not in the bundle's dependency set), so the name is written
        twice on purpose. Twice written means once checked.
        """

        from scripts import refresh_crisiswatch

        assert (
            refresh_crisiswatch.BACKFILL_STREAM
            == bundle.BundleBuilder.CRISISWATCH_BACKFILL_STREAM
        )

    def test_a_budget_spent_for_nothing_is_reported_as_such(self, tmp_path):
        """915 seconds, 40 downloads, 0 of 1 editions. The run's own numbers."""

        register = _register(tmp_path, self._backfill_stream(tmp_path, {
            "wanted": ["2026-05"], "recovered": [], "still_missing": ["2026-05"],
            "snapshots_tried": 40, "snapshots_downloaded": 40,
            "stopped_early": "download budget (40)",
        }))
        issue = next(
            i for i in register.issues
            if i.id == "crisiswatch_backfill_spends_its_budget_for_nothing"
        )
        assert issue.severity == "degraded"
        assert issue.recovers_on_rerun is False
        assert issue.cost == 40
        # The budget is NOT the thing to raise, and the report says so.
        assert "larger budget" in issue.title

    def test_a_successful_backfill_reports_its_rate_at_info(self, tmp_path):
        register = _register(tmp_path, self._backfill_stream(tmp_path, {
            "wanted": ["2026-03", "2026-05", "2026-06"],
            "recovered": ["2026-03", "2026-06"],
            "still_missing": ["2026-05"],
            "snapshots_tried": 12, "snapshots_downloaded": 8, "stopped_early": "",
        }))
        issue = next(
            i for i in register.issues
            if i.id == "crisiswatch_backfill_recovery_rate"
        )
        assert issue.severity == "info"
        assert "67%" in issue.title
        assert issue.cost == 1

    def test_no_backfill_pass_means_no_issue(self, tmp_path):
        register = _register(tmp_path)
        assert not [
            i for i in register.issues if i.id.startswith("crisiswatch_backfill")
        ]

    def test_the_scraper_records_its_accounting_where_the_bundle_reads_it(
        self, tmp_path, monkeypatch
    ):
        """Otherwise the next run spends the same 15 minutes learning the same thing."""

        from scripts import refresh_crisiswatch

        monkeypatch.setenv(run_log.ENV_DIR, str(tmp_path))
        run_log.reset_for_tests()
        refresh_crisiswatch._record_backfill_accounting({
            "wanted": ["2026-05"], "recovered": [], "still_missing": ["2026-05"],
            "snapshots_tried": 40, "snapshots_downloaded": 40,
            "stopped_early": "download budget (40)",
        })
        written = list(run_log.read_stream(tmp_path / "crisiswatch_backfill.jsonl"))
        assert written and written[0]["still_missing"] == ["2026-05"]
        run_log.reset_for_tests()


# ---------------------------------------------------------------------------
# An empty table is a claim about a writer
# ---------------------------------------------------------------------------


class TestTheDfoTableNamesItsWriter:
    def test_haz_raw_dfo_is_declared_with_the_workflow_that_fills_it(self, tmp_path):
        """It was in neither registry, so the check said nothing about it.

        The nightly backcast fetches the archive; a Resolver Update never
        does. An empty table here is a statement about the chain, and the
        check that reads it has to know the writer or it cannot say so.
        """

        assert "haz_raw_dfo" not in bundle.BundleBuilder.RESOLVER_UPDATE_WRITES
        writer = bundle.BundleBuilder.CARRIED_TABLES["haz_raw_dfo"]
        assert "haz_backcast" in writer

    def test_an_empty_dfo_table_is_reported_and_never_failed(self, tmp_path):
        db = tmp_path / "resolver.duckdb"
        con = duckdb.connect(str(db))
        con.execute("CREATE TABLE facts_resolved (iso3 TEXT, ym TEXT, value DOUBLE)")
        con.execute("INSERT INTO facts_resolved VALUES ('PHL','2026-08',1.0)")
        con.execute("CREATE TABLE haz_raw_dfo (record_id TEXT, payload_json TEXT)")
        con.close()
        diagnostics = tmp_path / "diagnostics"
        diagnostics.mkdir()
        manifest = bundle.build_bundle(
            out_path=tmp_path / "b.zip", db_path=db, diagnostics_dir=diagnostics,
            run_log_dir=None, staging=tmp_path / "stg", environ={},
        )
        check = {c["name"]: c for c in manifest["checks"]}[
            "no_declared_active_table_is_empty_after_the_run"
        ]
        carried = check["detail"].split("carried from another workflow", 1)
        assert len(carried) == 2, check["detail"]
        # Named, with its writer, under "not this run's to fill" — never
        # among the tables this workflow is held responsible for.
        assert "haz_raw_dfo" in carried[1]
        assert "haz_backcast" in carried[1]
        assert "haz_raw_dfo" not in carried[0]

    def test_the_nightly_workflow_really_does_fetch_it(self):
        """The writer named in the registry has to be the writer.

        haz_raw_dfo held no rows in run 34222175003, and the first question
        is always whether anything ever tried. It does: the step is there,
        continue-on-error, every night.
        """

        workflow = (
            Path(__file__).resolve().parents[2]
            / ".github" / "workflows" / "haz_backcast.yml"
        ).read_text(encoding="utf-8")
        assert "resolver.hazard_resolution.dfo" in workflow


class TestBothPathsSeeTheSameIssues:
    """The printed register and the bundled one are one register.

    They are built by two entry points -- one before the canonical upload,
    where the history has to land, and one after, where the zip is
    assembled. A collector wired into only the second is a fault that
    reaches the artifact and never reaches the reader.
    """

    def _streams(self, tmp_path: Path) -> Path:
        streams = tmp_path / "runlog"
        streams.mkdir(exist_ok=True)
        with open(streams / f"{run_log.STREAM_HTTP}.jsonl", "w", encoding="utf-8") as fh:
            for i in range(30):
                fh.write(json.dumps({
                    "connector": "resolver.connectors.gdacs",
                    "url": f"https://www.gdacs.org/datareport/resources/FL/{i}/rss_{i}.xml",
                    "status": 403, "elapsed_ms": 100.0,
                }) + "\n")
        with open(streams / "source_fetches.jsonl", "w", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "source": "emdat", "ok": False, "failure_class": "auth_rejected",
                "error": "Invalid key",
            }) + "\n")
        name = bundle.BundleBuilder.CRISISWATCH_BACKFILL_STREAM
        with open(streams / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "wanted": ["2026-05"], "recovered": [], "still_missing": ["2026-05"],
                "snapshots_tried": 40, "snapshots_downloaded": 40,
                "stopped_early": "download budget (40)",
            }) + "\n")
        return streams

    def test_the_early_path_carries_every_issue_the_bundle_path_does(self, tmp_path):
        streams = self._streams(tmp_path)
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        early = _register_before_the_upload(tmp_path / "a", streams)
        late = _register(tmp_path / "b", streams)

        early_ids = {i.id for i in early.issues}
        late_ids = {i.id for i in late.issues}
        assert late_ids - early_ids == set(), (
            "these issues reach the zip and never reach the reader: "
            f"{sorted(late_ids - early_ids)}"
        )
        # And the three sources are all present, so the assertion above is
        # comparing something rather than two empty sets.
        assert "emdat_auth_rejected" in early_ids
        assert "crisiswatch_backfill_spends_its_budget_for_nothing" in early_ids
        assert any(i.startswith("refused_requests_") for i in early_ids)
