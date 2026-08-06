# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Guards for the published-DB builder (the 2026-08-06 release incident)."""

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
        # NOT excluded: the machine's verdict tables are small and are the
        # ones a future dashboard view would actually want.
        con.execute("CREATE TABLE haz_resolutions (iso3 TEXT, status TEXT)")
        con.execute("INSERT INTO haz_resolutions VALUES ('SOM', 'RESOLVED_ZERO')")
    con.close()
    return path


class TestExclusion:
    def test_raw_caches_dropped_served_tables_untouched(self, tmp_path):
        src = _src(tmp_path)
        out = tmp_path / "release.duckdb"
        stats = brd.build_release_db(str(src), str(out))
        assert sorted(stats["dropped_tables"]) == [
            "haz_raw_gdacs", "haz_raw_reliefweb_docs",
        ]
        con = duckdb.connect(str(out), read_only=True)
        assert con.execute("SELECT COUNT(*) FROM questions").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM forecasts_ensemble").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM interpretations").fetchone()[0] == 1
        # Verdict tables survive — only the RAW caches are stripped.
        assert con.execute("SELECT COUNT(*) FROM haz_resolutions").fetchone()[0] == 1
        names = {r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='main'").fetchall()}
        assert not any(n.startswith("haz_raw_") for n in names)
        con.close()

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


class TestExclusionListIsNarrow:
    def test_only_raw_caches_are_excluded(self):
        # Widening this list is a decision about what the dashboard can ever
        # show. Keep it deliberate: haz_resolutions / haz_triggers /
        # haz_base_rates_* stay in the published copy.
        assert brd.EXCLUDED_TABLE_PREFIXES == ("haz_raw_",)

    @pytest.mark.parametrize("table", [
        "questions", "forecasts_ensemble", "forecasts_raw", "scores",
        "resolutions", "interpretations", "forecast_deviation", "hs_triage",
        "hs_runs", "facts_resolved", "facts_deltas", "sibyl_runs",
        "calibration_weights", "llm_calls", "haz_resolutions",
        "haz_base_rates_occurrence",
    ])
    def test_served_tables_are_never_excluded(self, table):
        assert not brd.is_excluded(table)
