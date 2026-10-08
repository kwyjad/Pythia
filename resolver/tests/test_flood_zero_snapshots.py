# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""A flood zero cites a snapshot of the GDACS cache, never the cache itself.

On 8 Oct 2026 the 20,048 flood zero rows carried every GDACS event URL in
the cache twice (4.86 GB of a 9.2 GB canonical DB), and each month's rows
were larger than the last. These tests pin the new shape for rows written
from now on, and the rewrite of the rows already written: counts and every
column but provenance_json unchanged, every old URL recoverable, a row that
moved under the rewrite left alone.
"""

from __future__ import annotations

import json

import duckdb
import pytest

from resolver.hazard_resolution import cli as cli_mod
from resolver.hazard_resolution import evidence_snapshots as snap
from resolver.hazard_resolution import resolutions as res_mod
from resolver.hazard_resolution.schema import ensure_haz_schema
from resolver.tests.hazard_resolution_utils import (
    make_rulebook,
    seed_gdacs_event,
    seed_population,
    silent_sweep_evidence,
)


@pytest.fixture()
def rulebook():
    return make_rulebook()


@pytest.fixture()
def con():
    con = duckdb.connect(":memory:")
    ensure_haz_schema(con)
    seed_population(con)
    return con


def _seed_cache(con, n_other: int) -> None:
    # VNM has one Green event in March 2024 (below the trigger level, so the
    # cell can still be a zero); IDN has nothing; n_other events elsewhere.
    seed_gdacs_event(con, iso3="VNM", alert_level="Green", event_id="900")
    for i in range(n_other):
        seed_gdacs_event(con, iso3="PHL", alert_level="Green", event_id=str(1000 + i),
                         start_date="2024-01-05", end_date="2024-01-09")


def _run_flood(con, rulebook, monkeypatch):
    monkeypatch.setattr(cli_mod, "_load_universe", lambda: ["VNM", "IDN"])
    monkeypatch.setattr(
        "resolver.hazard_resolution.reliefweb_sweep.sweep_country_month",
        lambda iso3, ym, rulebook, **kw: silent_sweep_evidence(iso3, ym),
    )
    monkeypatch.setattr(
        "resolver.hazard_resolution.gdacs.coverage", lambda *a, **k: (True, "ok (test)")
    )
    return cli_mod.run_flood_month(
        ym="2024-03", db_url=None, countries_filter=None, skip_fetch=True,
        no_sweep=False, dry_run=False, no_ladder=True, rulebook=rulebook, con=con,
    )


def _prov(con, iso3):
    return json.loads(con.execute(
        "SELECT provenance_json FROM haz_resolutions WHERE iso3=? AND hazard='FL'", [iso3]
    ).fetchone()[0])


def test_a_new_zero_cites_a_snapshot_and_its_own_window(con, rulebook, monkeypatch):
    _seed_cache(con, n_other=40)
    _run_flood(con, rulebook, monkeypatch)

    prov = _prov(con, "VNM")
    gdacs = prov["evidence_of_absence"]["gdacs"]
    assert "source_urls" not in gdacs
    cited = gdacs["snapshot"]
    assert cited["n_urls"] == 41
    assert cited["snapshot_id"].startswith("gdacs:FL:")
    # The zero names the one event it was weighed against, and no other.
    assert [e["event_id"] for e in gdacs["events_in_window"]] == ["900"]
    assert len([u for u in prov["source_urls"] if "gdacs" in u]) == 1
    # The whole listing is recoverable from the snapshot table.
    urls = snap.snapshot_urls(con, cited["snapshot_id"])
    assert len(urls) == 41
    # IDN has no events at all: the window is empty, the citation is the same.
    idn = _prov(con, "IDN")["evidence_of_absence"]["gdacs"]
    assert idn["events_in_window"] == []
    assert idn["snapshot"]["snapshot_id"] == cited["snapshot_id"]
    assert con.execute("SELECT COUNT(*) FROM haz_evidence_snapshots").fetchone()[0] == 1


def test_a_zero_does_not_grow_with_the_cache(rulebook, monkeypatch):
    sizes = []
    for n_other in (2, 200):
        con = duckdb.connect(":memory:")
        ensure_haz_schema(con)
        seed_population(con)
        _seed_cache(con, n_other=n_other)
        _run_flood(con, rulebook, monkeypatch)
        sizes.append(len(json.dumps(_prov(con, "IDN"))))
    # Before Oct 2026 the 200-event cache added ~200 URLs twice to every zero.
    assert abs(sizes[1] - sizes[0]) < 64, sizes


def _old_style_row(con, iso3, urls, *, extra_top=()):
    """A zero as written before Oct 2026: the listing twice."""
    evidence = {
        "gdacs": {
            "source": "gdacs", "total_records": len(urls),
            "last_retrieved_at": "2026-09-01 03:00:00",
            "source_urls": list(urls),
            "query": {"ym": "2024-03"},
        },
        "reliefweb": silent_sweep_evidence(iso3, "2024-03"),
        "retrieved_at": "2026-01-15T00:00:00+00:00",
    }
    provenance = {
        "source": "absence",
        "source_record_ids": [],
        "source_urls": res_mod._collect_urls(evidence) + list(extra_top),
        "retrieved_at": "2026-01-15T00:00:00+00:00",
        "rule_fired": "zero",
        "evidence_of_absence": evidence,
    }
    con.execute(
        """
        INSERT INTO haz_resolutions
            (iso3, year, month, hazard, status, value, provenance_json, rule_fired,
             flagged, provisional, run_type, frozen_at)
        VALUES (?, 2024, 3, 'FL', 'RESOLVED_ZERO', 0.0, ?, 'zero', FALSE, FALSE,
                'backcast', TIMESTAMP '2024-05-30 00:00:00')
        """,
        [iso3, json.dumps(provenance)],
    )
    return provenance


def _cache_urls(con):
    return [r[0] for r in con.execute(
        "SELECT DISTINCT source_url FROM haz_raw_gdacs WHERE hazard='FL' ORDER BY 1"
    ).fetchall()]


def test_the_rewrite_keeps_counts_and_columns_and_every_url(con):
    _seed_cache(con, n_other=30)
    urls = _cache_urls(con)
    for iso3 in ("VNM", "IDN", "THA"):
        _old_style_row(con, iso3, urls)
    con.execute(
        "INSERT INTO haz_resolutions (iso3, year, month, hazard, status, value, "
        "provenance_json, rule_fired) VALUES ('PHL', 2024, 1, 'FL', 'RESOLVED_VALUE', "
        "5.0, '{\"source\": \"emdat\"}', 'ladder')"
    )
    before = snap.integrity_report(con)

    out = snap.rewrite_zero_rows(con, apply=True, batch_rows=2)
    after = snap.integrity_report(con)

    assert out["rows_targeted"] == 3 and out["rows_rewritten"] == 3
    assert out["batches"] == 2
    assert snap.compare_reports(before, after) == []
    assert after["provenance_json_bytes"] < before["provenance_json_bytes"]

    vnm = _prov(con, "VNM")["evidence_of_absence"]["gdacs"]
    assert "source_urls" not in vnm
    assert vnm["events_in_window_basis"] == "reconstructed_at_rewrite"
    assert [e["event_id"] for e in vnm["events_in_window"]] == ["900"]
    assert snap.snapshot_urls(con, vnm["snapshot"]["snapshot_id"]) == sorted(urls)
    # Three rows, one listing: stored once.
    assert con.execute("SELECT COUNT(*) FROM haz_evidence_snapshots").fetchone()[0] == 1

    again = snap.rewrite_zero_rows(con, apply=True)
    assert again["rows_targeted"] == 0


def test_the_dry_run_writes_nothing(con):
    _seed_cache(con, n_other=5)
    _old_style_row(con, "VNM", _cache_urls(con))
    before = con.execute("SELECT provenance_json FROM haz_resolutions").fetchone()[0]
    out = snap.rewrite_zero_rows(con, apply=False)
    assert out["rows_targeted"] == 1 and out["bytes_after"] < out["bytes_before"]
    assert con.execute("SELECT provenance_json FROM haz_resolutions").fetchone()[0] == before
    assert con.execute("SELECT COUNT(*) FROM haz_evidence_snapshots").fetchone()[0] == 0


def test_a_row_that_moved_under_the_rewrite_is_left_alone(con, monkeypatch):
    _seed_cache(con, n_other=5)
    _old_style_row(con, "VNM", _cache_urls(con))
    real = snap.slim_provenance

    def slim_while_another_writer_lands(c, provenance, **kw):
        c.execute(
            "UPDATE haz_resolutions SET provenance_json = '{\"moved\": true}' "
            "WHERE iso3 = 'VNM'"
        )
        return real(c, provenance, **kw)

    monkeypatch.setattr(snap, "slim_provenance", slim_while_another_writer_lands)
    out = snap.rewrite_zero_rows(con, apply=True)
    assert out["rows_cas_skipped"] == 1 and out["rows_rewritten"] == 0
    assert con.execute("SELECT provenance_json FROM haz_resolutions").fetchone()[0] == '{"moved": true}'


def test_a_row_whose_urls_would_be_lost_is_refused(con):
    _seed_cache(con, n_other=5)
    _old_style_row(con, "VNM", _cache_urls(con), extra_top=["https://example.test/only-here"])
    before = con.execute("SELECT provenance_json FROM haz_resolutions").fetchone()[0]
    out = snap.rewrite_zero_rows(con, apply=True)
    assert out["rows_refused"] == 1 and out["rows_rewritten"] == 0
    assert con.execute("SELECT provenance_json FROM haz_resolutions").fetchone()[0] == before


def test_a_snapshot_that_does_not_match_its_hash_is_refused(con):
    cited = snap.store_snapshot(con, source="gdacs", hazard="FL",
                                urls=["https://a", "https://b"], snapshot_date="2026-10-08")
    assert cited["snapshot_id"].startswith("gdacs:FL:2026-10-08:2:")
    con.execute("UPDATE haz_evidence_snapshots SET urls_json = '[\"https://a\"]'")
    with pytest.raises(ValueError):
        snap.snapshot_urls(con, cited["snapshot_id"])
