# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Guards for the published-DB builder.

Two incidents shaped this. On 2026-08-06 an unvalidated ``--clobber`` upload
deleted the working asset before GitHub rejected the oversized replacement; the
size guard is the answer to that. On 2026-09-01 the guard fired correctly but
the strip was too narrow to get under the limit -- only ``haz_raw_`` was
dropped, so the machine's working tables still shipped and the release copy was
5.1 GB. Hence the whole ``haz_`` family, and the per-table size report that
makes the next overrun attributable instead of a guess.
"""

from __future__ import annotations

import duckdb
import pytest

from scripts.ci import build_release_db as brd


def _src(tmp_path, *, with_machine_tables=True):
    path = tmp_path / "canonical.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE questions (question_id TEXT, iso3 TEXT)")
    con.execute("INSERT INTO questions VALUES ('SOM_ACE_PA_2026-08', 'SOM')")
    con.execute("CREATE TABLE forecasts_ensemble (question_id TEXT, p DOUBLE)")
    con.execute("INSERT INTO forecasts_ensemble VALUES ('SOM_ACE_PA_2026-08', 0.4)")
    con.execute("CREATE TABLE interpretations (interpretation_id TEXT)")
    con.execute("INSERT INTO interpretations VALUES ('int_combined_x_v1')")
    if with_machine_tables:
        # The bulky caches: raw source payloads the API never reads.
        con.execute("CREATE TABLE haz_raw_reliefweb_docs (doc_id TEXT, body TEXT)")
        con.execute(
            "INSERT INTO haz_raw_reliefweb_docs SELECT 'd' || i, repeat('x', 4000) "
            "FROM range(400) t(i)"
        )
        con.execute("CREATE TABLE haz_raw_gdacs (event_id TEXT, payload_json TEXT)")
        con.execute("INSERT INTO haz_raw_gdacs VALUES ('g1', '{}')")
        # Also excluded since 2026-09-01: the machine's working and verdict
        # tables. haz_triggers carries one evidence_of_absence_json per
        # ASSESSED cell -- ~200 countries x 3 hazards x ~360 backcast months --
        # and no API or dashboard code reads any of them.
        con.execute("CREATE TABLE haz_resolutions (iso3 TEXT, status TEXT)")
        con.execute("INSERT INTO haz_resolutions VALUES ('SOM', 'RESOLVED_ZERO')")
        con.execute("CREATE TABLE haz_triggers (iso3 TEXT, evidence_of_absence_json TEXT)")
        con.execute(
            "INSERT INTO haz_triggers SELECT 'SOM', repeat('e', 2000) FROM range(200)"
        )
        con.execute("CREATE TABLE haz_base_rates_occurrence (iso3 TEXT, p DOUBLE)")
        con.execute("INSERT INTO haz_base_rates_occurrence VALUES ('SOM', 0.2)")
    con.close()
    return path


class TestExclusion:
    def test_machine_tables_dropped_served_tables_untouched(self, tmp_path):
        src = _src(tmp_path)
        out = tmp_path / "release.duckdb"
        stats = brd.build_release_db(str(src), str(out))
        assert sorted(stats["dropped_tables"]) == [
            "haz_base_rates_occurrence", "haz_raw_gdacs", "haz_raw_reliefweb_docs",
            "haz_resolutions", "haz_triggers",
        ]
        con = duckdb.connect(str(out), read_only=True)
        assert con.execute("SELECT COUNT(*) FROM questions").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM forecasts_ensemble").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM interpretations").fetchone()[0] == 1
        names = {r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='main'").fetchall()}
        assert not any(n.startswith("haz_") for n in names)
        con.close()

    def test_the_machines_working_tables_are_gone_not_just_the_raw_caches(self, tmp_path):
        """The 2026-09-01 abort: haz_triggers alone was multi-GB and shipped."""
        src = _src(tmp_path)
        out = tmp_path / "release.duckdb"
        stats = brd.build_release_db(str(src), str(out))
        assert "haz_triggers" in stats["dropped_tables"]
        assert brd.is_excluded("haz_triggers")
        assert brd.is_excluded("haz_doc_extractions")
        assert brd.is_excluded("haz_impact_candidates")
        assert brd.is_excluded("haz_revisions")
        assert brd.is_excluded("haz_backcast_progress")

    def test_keep_all_drops_nothing(self, tmp_path):
        src = _src(tmp_path)
        out = tmp_path / "full.duckdb"
        stats = brd.build_release_db(str(src), str(out), keep_all=True)
        assert stats["dropped_tables"] == []
        con = duckdb.connect(str(out), read_only=True)
        assert con.execute("SELECT COUNT(*) FROM haz_raw_gdacs").fetchone()[0] == 1
        con.close()

    def test_excluding_shrinks_the_file(self, tmp_path):
        src = _src(tmp_path)
        out = tmp_path / "release.duckdb"
        stats = brd.build_release_db(str(src), str(out))
        assert stats["out_bytes"] < stats["src_bytes"]

    def test_a_db_without_machine_tables_still_builds(self, tmp_path):
        # The pre-PA-machine shape (e.g. the 2026-08-01 release): nothing to
        # drop, compaction only, no crash.
        src = _src(tmp_path, with_machine_tables=False)
        out = tmp_path / "release.duckdb"
        stats = brd.build_release_db(str(src), str(out))
        assert stats["dropped_tables"] == []
        con = duckdb.connect(str(out), read_only=True)
        assert con.execute("SELECT COUNT(*) FROM questions").fetchone()[0] == 1
        con.close()


class TestCompactDatabase:
    """The primitive the retention workflow shares with the release path."""

    def test_deleting_rows_does_not_shrink_but_copying_does(self, tmp_path):
        """DuckDB reuses freed blocks and never truncates the file.

        This is why retention needs two steps: collapsing the rows stops the
        growth, and only the copy gives the bytes back.
        """
        src = tmp_path / "bloat.duckdb"
        con = duckdb.connect(str(src))
        con.execute("CREATE TABLE t (a TEXT)")
        con.execute(
            "INSERT INTO t SELECT repeat(md5(i::TEXT), 40) FROM range(20000) t(i)"
        )
        con.execute("CHECKPOINT")
        full = src.stat().st_size
        con.execute("DELETE FROM t WHERE rowid % 10 != 0")
        con.execute("CHECKPOINT")
        con.close()
        assert src.stat().st_size == full, "delete is expected NOT to shrink the file"

        stats = brd.compact_database(str(src), str(tmp_path / "small.duckdb"))
        assert stats["out_bytes"] < full / 2
        assert stats["saved_bytes"] > 0

    def test_the_copy_preserves_every_row(self, tmp_path):
        src = _src(tmp_path)
        out = tmp_path / "compact.duckdb"
        brd.compact_database(str(src), str(out))
        con = duckdb.connect(str(out), read_only=True)
        # Nothing is dropped by compaction -- that is the strip's job.
        assert con.execute("SELECT COUNT(*) FROM questions").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM haz_triggers").fetchone()[0] == 200
        con.close()

    def test_missing_source_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            brd.compact_database(str(tmp_path / "nope.duckdb"),
                                 str(tmp_path / "out.duckdb"))


class TestSizeReport:
    """The 2026-09-01 abort could not be attributed to any table."""

    def test_sizes_are_reported_before_and_after(self, tmp_path):
        src = _src(tmp_path)
        stats = brd.build_release_db(str(src), str(tmp_path / "r.duckdb"))
        assert stats["sizes_before"] and stats["sizes_after"]
        assert any("haz_triggers" in line for line in stats["sizes_before"])
        assert not any("haz_triggers" in line for line in stats["sizes_after"])

    def test_the_error_log_carries_the_attribution(self, tmp_path, caplog):
        src = _src(tmp_path)
        with caplog.at_level("INFO"):
            brd.main(["--src", str(src), "--out", str(tmp_path / "r.duckdb"),
                      "--max-bytes", "1024"])
        assert "canonical |" in caplog.text
        assert "questions" in caplog.text

    def test_report_top_zero_disables_it(self, tmp_path):
        src = _src(tmp_path)
        stats = brd.build_release_db(
            str(src), str(tmp_path / "r.duckdb"), report_top=0
        )
        assert stats["sizes_before"] == [] and stats["sizes_after"] == []


class TestSizeGuard:
    """The guard whose absence deleted the release asset on 2026-08-06."""

    def test_over_limit_exits_nonzero_so_publish_stops(self, tmp_path, capsys):
        src = _src(tmp_path)
        out = tmp_path / "release.duckdb"
        rc = brd.main([
            "--src", str(src), "--out", str(out), "--max-bytes", "1024",
        ])
        assert rc == 1
        captured = capsys.readouterr()
        assert "::error::" in captured.out
        assert "exceeds" in captured.out
        # The message must say the publish was aborted BEFORE clobbering.
        assert "clobber" in captured.out.lower()

    def test_under_limit_exits_zero_and_reports_size(self, tmp_path, capsys):
        src = _src(tmp_path)
        out = tmp_path / "release.duckdb"
        rc = brd.main(["--src", str(src), "--out", str(out)])
        assert rc == 0
        assert "RELEASE_DB_BYTES=" in capsys.readouterr().out

    def test_default_limit_is_githubs_two_gib(self):
        assert brd.GITHUB_ASSET_LIMIT_BYTES == 2147483648

    def test_missing_source_raises_rather_than_publishing_nothing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            brd.build_release_db(str(tmp_path / "nope.duckdb"),
                                 str(tmp_path / "out.duckdb"))


class TestExclusionListMatchesTheDocumentedSet:
    def test_the_whole_haz_family_is_excluded(self):
        # Widening this list is a decision about what the dashboard can ever
        # show, so it stays deliberate. It was widened on 2026-09-01 because
        # `grep haz_ pythia/api/ web/src/` returns nothing -- the API and
        # dashboard read no haz_* table at all -- while the machine's working
        # tables were pushing the release copy to 5.1 GB against a 2 GB cap.
        # The machine's own state lives in the canonical artifact, which this
        # script never touches. If a future dashboard view wants the verdicts,
        # narrow this back to what that view needs and say so here.
        assert brd.EXCLUDED_TABLE_PREFIXES == ("haz_",)

    @pytest.mark.parametrize("table", [
        "questions", "forecasts_ensemble", "forecasts_raw", "scores",
        "resolutions", "interpretations", "forecast_deviation", "hs_triage",
        "hs_runs", "facts_resolved", "facts_deltas", "sibyl_runs",
        "sibyl_forecasts", "calibration_weights", "calibration_advice",
        "llm_calls", "question_research", "bucket_centroids",
    ])
    def test_served_tables_are_never_excluded(self, table):
        assert not brd.is_excluded(table)

    def test_a_hazard_named_column_elsewhere_is_not_confused_for_the_prefix(self):
        """`haz_` is a table-name prefix, not a substring match."""
        assert not brd.is_excluded("hazard_lookup")
        assert not brd.is_excluded("questions_haz_raw_notes")


# ---------------------------------------------------------------------------
# Oct 2026 security audit: licensed and internal tables, and credentials.
# ---------------------------------------------------------------------------

GEMINI_KEY = "AIzaSyLEAKED0123456789abcdefghij"
ANTHROPIC_KEY = "sk-ant-api03-leakedleakedleaked"


def _src_with_secrets(tmp_path):
    path = _src(tmp_path, with_machine_tables=False)
    con = duckdb.connect(str(path))
    for table in sorted(brd.EXCLUDED_TABLES):
        con.execute(f"CREATE TABLE {table} (x TEXT)")
        con.execute(f"INSERT INTO {table} VALUES ('row')")
    con.execute(
        "CREATE TABLE llm_calls (call_id TEXT, prompt_text TEXT, error_text TEXT, usage_json JSON)"
    )
    con.execute(
        "INSERT INTO llm_calls VALUES "
        "('c1', 'assess the risk-assessment-framework', "
        f"'ConnectionError for url: https://g/v1beta/m:generateContent?key={GEMINI_KEY}', "
        f"'{{\"note\": \"{ANTHROPIC_KEY}\"}}'), "
        "('c2', 'a clean prompt', NULL, '{\"prompt_tokens\": 10}')"
    )
    con.execute("CREATE TABLE hs_country_reports (iso3 TEXT, grounding_debug_json TEXT)")
    con.execute(
        "INSERT INTO hs_country_reports VALUES "
        "('SOM', '{\"provider_error_message\": \"GET https://api.ipcinfo.org/population?key=deadbeef0123456789abcdef01234567\"}')"
    )
    con.close()
    return path


class TestLicensedAndInternalTables:
    @pytest.mark.parametrize("table", sorted(brd.EXCLUDED_TABLES))
    def test_each_listed_table_is_dropped(self, tmp_path, table):
        out = tmp_path / "r.duckdb"
        stats = brd.build_release_db(str(_src_with_secrets(tmp_path)), str(out))
        assert table in stats["dropped_tables"]
        con = duckdb.connect(str(out))
        try:
            assert table not in brd.tables_in(con, "r")
        finally:
            con.close()

    def test_the_licence_list_is_the_documented_one(self):
        assert brd.EXCLUDED_TABLES == {
            "acled_political_events", "emdat_pa", "facts_raw",
            "acaps_risk_radar", "acaps_daily_monitoring", "acaps_humanitarian_access",
            "llm_batches", "llm_batch_requests", "ui_runs",
        }

    def test_aggregates_and_the_inform_index_still_ship(self):
        for table in ("acled_monthly_fatalities", "acaps_inform_severity",
                      "acaps_inform_severity_trend", "facts_resolved"):
            assert not brd.is_excluded(table)


class TestCredentialScrub:
    def _build(self, tmp_path, *extra):
        out = tmp_path / "r.duckdb"
        stats = brd.build_release_db(str(_src_with_secrets(tmp_path)), str(out), *extra)
        con = duckdb.connect(str(out))
        try:
            calls = con.execute(
                "SELECT call_id, prompt_text, error_text, CAST(usage_json AS VARCHAR) "
                "FROM llm_calls ORDER BY call_id"
            ).fetchall()
            reports = con.execute("SELECT grounding_debug_json FROM hs_country_reports").fetchall()
        finally:
            con.close()
        return stats, calls, reports

    def test_planted_keys_are_removed_and_counted(self, tmp_path):
        stats, calls, reports = self._build(tmp_path)
        published = repr(calls) + repr(reports)
        for secret in (GEMINI_KEY, ANTHROPIC_KEY, "deadbeef0123456789abcdef01234567"):
            assert secret not in published
        assert stats["scrubbed"] == {
            "hs_country_reports.grounding_debug_json": 1,
            "llm_calls.error_text": 1,
            "llm_calls.usage_json": 1,
        }
        assert stats["residual"] == {}

    def test_text_without_a_credential_is_byte_identical(self, tmp_path):
        _, calls, _ = self._build(tmp_path)
        assert calls[0][1] == "assess the risk-assessment-framework"
        assert calls[1][1:] == ("a clean prompt", None, '{"prompt_tokens": 10}')

    def test_keep_all_still_scrubs(self, tmp_path):
        out = tmp_path / "r.duckdb"
        stats = brd.build_release_db(
            str(_src_with_secrets(tmp_path)), str(out), keep_all=True
        )
        assert stats["dropped_tables"] == []
        assert stats["scrubbed"]

    def test_a_scrub_warning_is_printed_so_the_key_gets_rotated(self, tmp_path, capsys):
        rc = brd.main(["--src", str(_src_with_secrets(tmp_path)),
                       "--out", str(tmp_path / "r.duckdb"), "--report-top", "0"])
        assert rc == 0
        assert "::warning title=Credentials scrubbed from release::" in capsys.readouterr().out

    def test_a_residual_match_stops_the_publish(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(
            brd, "scrub_database",
            lambda con, alias: {"scrubbed": {}, "residual": {"llm_calls.error_text": 1}},
        )
        rc = brd.main(["--src", str(_src_with_secrets(tmp_path)),
                       "--out", str(tmp_path / "r.duckdb"), "--report-top", "0"])
        assert rc == 2
        out = capsys.readouterr().out
        assert "::error::" in out and "clobber" in out
