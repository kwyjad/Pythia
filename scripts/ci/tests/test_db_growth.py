# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""Disk room for the canonical DB, and the bytes each run adds."""

from __future__ import annotations

import collections

import pytest

duckdb = pytest.importorskip("duckdb")

from scripts.ci import db_growth  # noqa: E402

GIB = db_growth.GIB
Usage = collections.namedtuple("Usage", "total used free")


def _db(tmp_path, size: int):
    path = tmp_path / "resolver.duckdb"
    path.write_bytes(b"\0" * size)
    return path


def test_room_is_the_db_times_the_factor_plus_headroom():
    ok, msg = db_growth.check_room(30 * GIB, 33 * GIB, 1.0)
    assert ok and "32.00 GB" in msg
    ok, _ = db_growth.check_room(30 * GIB, 31 * GIB, 1.0)
    assert not ok


def test_too_little_disk_stops_the_job_straight_after_the_download(tmp_path, monkeypatch, capsys):
    path = _db(tmp_path, 1024)
    monkeypatch.setattr(db_growth.shutil, "disk_usage", lambda p: Usage(10 * GIB, 9 * GIB, GIB))
    env = tmp_path / "env"
    monkeypatch.setenv("GITHUB_ENV", str(env))
    assert db_growth.main(["downloaded", "--db", str(path)]) == 2
    out = capsys.readouterr().out
    assert "::error title=Not enough disk" in out
    assert "CANONICAL_DB_BYTES_AT_DOWNLOAD=1024" in env.read_text()


def test_enough_disk_passes(tmp_path, monkeypatch):
    path = _db(tmp_path, 1024)
    monkeypatch.setattr(db_growth.shutil, "disk_usage", lambda p: Usage(100 * GIB, 0, 50 * GIB))
    assert db_growth.main(["downloaded", "--db", str(path), "--factor", "1.2"]) == 0


def test_before_upload_records_what_the_run_added(tmp_path, monkeypatch, capsys):
    path = tmp_path / "resolver.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE haz_revisions (observed_at TIMESTAMP, detail_json TEXT)")
    con.execute("INSERT INTO haz_revisions VALUES (CURRENT_TIMESTAMP, '{\"x\": 1}')")
    con.close()
    monkeypatch.setenv("CANONICAL_DB_BYTES_AT_DOWNLOAD", "100")
    monkeypatch.setenv("GITHUB_WORKFLOW", "Hazard Backcast")
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    assert db_growth.main(["before-upload", "--db", str(path)]) == 0
    con = duckdb.connect(str(path), read_only=True)
    row = con.execute(
        "SELECT workflow, run_id, bytes_at_download, bytes_added > 0 FROM db_growth_log"
    ).fetchone()
    report = db_growth.growth_report(con)
    con.close()
    assert row == ("Hazard Backcast", "42", 100, True)
    assert report["by_workflow"][0]["workflow"] == "Hazard Backcast"
    assert report["revisions_total"]["rows"] == 1
    assert "this run added" in capsys.readouterr().out


def test_before_upload_never_fails(tmp_path, monkeypatch):
    path = _db(tmp_path, 16)  # not a DuckDB file: the record fails, the upload must not
    monkeypatch.delenv("CANONICAL_DB_BYTES_AT_DOWNLOAD", raising=False)
    assert db_growth.main(["before-upload", "--db", str(path)]) == 0
