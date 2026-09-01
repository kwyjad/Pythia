# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Guards for the per-table size report.

The report exists because the 2026-09-01 publish abort could not be
attributed: 5.1 GB of release DB and nothing in the repo could say what it
was. So the property that matters most here is not precision -- it is that
the numbers ADD UP and say so when they don't.
"""

from __future__ import annotations

import duckdb
import pytest

from scripts.ci import db_table_sizes as dts


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "probe.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE big (payload TEXT)")
    con.execute(
        "INSERT INTO big SELECT repeat(md5(i::TEXT), 40) FROM range(8000) t(i)"
    )
    con.execute("CREATE TABLE small (n INTEGER)")
    con.execute("INSERT INTO small SELECT * FROM range(50)")
    con.execute("CREATE TABLE empty_table (n INTEGER)")
    con.execute("CHECKPOINT")
    con.close()
    return path


@pytest.fixture()
def con(db):
    con = duckdb.connect()
    con.execute(f"ATTACH '{db}' AS probe (READ_ONLY)")
    yield con
    con.close()


class TestAttribution:
    def test_the_big_table_is_ranked_first(self, con):
        report = dts.size_report(con, catalog="probe")
        assert report[0].table == "big"
        assert report[0].est_bytes > 0

    def test_every_table_appears_including_the_empty_one(self, con):
        names = {t.table for t in dts.size_report(con, catalog="probe")}
        assert names == {"big", "small", "empty_table"}

    def test_rows_are_counted(self, con):
        by_name = {t.table: t for t in dts.size_report(con, catalog="probe")}
        assert by_name["big"].rows == 8000
        assert by_name["empty_table"].rows == 0

    def test_attributed_bytes_never_exceed_the_file(self, con):
        """The remainder must be non-negative, or the report is nonsense."""
        report = dts.size_report(con, catalog="probe")
        info = dts.whole_file(con, "probe")
        assert sum(t.est_bytes for t in report) <= info["file_bytes"]

    def test_attribution_covers_most_of_the_file(self, con):
        """Sanity: catalog overhead is small, so the tables should dominate."""
        report = dts.size_report(con, catalog="probe")
        info = dts.whole_file(con, "probe")
        attributed = sum(t.est_bytes for t in report)
        assert attributed >= 0.8 * info["file_bytes"]

    def test_top_truncates_but_keeps_the_largest(self, con):
        top = dts.size_report(con, catalog="probe", top=1)
        assert len(top) == 1 and top[0].table == "big"


class TestHonesty:
    """A size report whose numbers do not add up is worse than none."""

    def test_markdown_always_shows_the_unattributed_remainder(self, con):
        report = dts.size_report(con, catalog="probe")
        info = dts.whole_file(con, "probe")
        out = dts.format_markdown(report, info)
        assert "unattributed" in out
        assert "free list" in out

    def test_markdown_remainder_is_present_even_when_truncated(self, con):
        """With --top, the omitted tables must land in the remainder, not vanish."""
        report = dts.size_report(con, catalog="probe", top=1)
        info = dts.whole_file(con, "probe")
        out = dts.format_markdown(report, info)
        assert "unattributed" in out

    def test_log_lines_carry_the_remainder_too(self, con):
        report = dts.size_report(con, catalog="probe")
        info = dts.whole_file(con, "probe")
        assert any("unattributed" in line for line in dts.report_lines(report, info))

    def test_free_list_is_reported(self, con):
        info = dts.whole_file(con, "probe")
        assert "free_blocks" in info and "free_bytes" in info


class TestCatalogResolution:
    """duckdb.connect(path) names the catalog after the FILE, not "main".

    Guessing wrong makes every size read zero, which looks like an empty
    database rather than a bad catalog name -- so both connection styles are
    pinned here. The inspect report connects directly; the release builder
    attaches.
    """

    def test_a_direct_file_connection_still_measures(self, db):
        con = duckdb.connect(str(db), read_only=True)
        try:
            report = dts.size_report(con)          # default catalog "main"
            info = dts.whole_file(con)
            assert info["file_bytes"] > 0
            assert report and report[0].table == "big"
            assert report[0].est_bytes > 0
        finally:
            con.close()

    def test_an_attached_alias_is_used_as_given(self, db):
        con = duckdb.connect()
        try:
            con.execute(f"ATTACH '{db}' AS someplace (READ_ONLY)")
            assert dts.resolve_catalog(con, "someplace") == "someplace"
            assert dts.whole_file(con, "someplace")["file_bytes"] > 0
        finally:
            con.close()


class TestSafety:
    def test_an_odd_table_name_is_refused_not_interpolated(self):
        with pytest.raises(ValueError):
            dts._quote("t; DROP TABLE questions")

    def test_missing_db_exits_nonzero(self, tmp_path, capsys):
        rc = dts.main(["--db", str(tmp_path / "nope.duckdb")])
        assert rc == 1
        assert "::error::" in capsys.readouterr().out

    def test_cli_runs_against_a_real_file(self, db, capsys):
        assert dts.main(["--db", str(db), "--top", "5"]) == 0
        assert "big" in capsys.readouterr().out


class TestLogicalBytes:
    def test_logical_bytes_measures_text_columns(self, con):
        assert dts.logical_bytes(con, "big", catalog="probe") == 8000 * 1280

    def test_a_table_with_no_text_columns_is_zero(self, con):
        assert dts.logical_bytes(con, "small", catalog="probe") == 0
