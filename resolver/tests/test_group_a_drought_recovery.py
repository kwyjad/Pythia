# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Group A of the run-33946954189 repairs: drought resolved nothing.

The acceptance report put DR at 0.0% of 3,024 assessed cells and the cell
ledger held 28,728 DR cells that wrote no row. Upstream of that number sat
four separate faults, one of them the root cause:

* the JRC hotspot archive is semicolon-delimited and was read as
  comma-delimited, so every one of its 10,123 records failed to resolve to a
  country and the indicator reported a changed shape for a feed that had not
  changed;
* HDX Signals wrote 2,808 rows to a table that gained nothing and kept the
  previous day's ``fetched_at``, because DuckDB's INSERT OR REPLACE leaves
  every column the statement does not name;
* NMME reached 176 of 252 countries, the other 76 absent from the mask
  entirely rather than skipped; and
* the occurrence base rates were published from a hazard path that had
  resolved not one cell.

Network-free.
"""

from __future__ import annotations

import duckdb
import pytest

from resolver.hazard_resolution import drought_indicators as di

# The header of the real JRC hotspots_ts.csv, verbatim from the run log.
ASAP_HEADER = (
    "asap0_id;asap0_name;date;hs_code;hs_name;comment;"
    "g1_w_crop;g1_w_range;g1_w_any"
)
ASAP_BODY = (
    f"{ASAP_HEADER}\n"
    "12;Kenya;2026-07-01;2;Drought hotspot;;1;0;1\n"
    "31;Somalia;2026-07-01;1;Watch;;0;0;0\n"
    "48;Ethiopia;2026-06-01;0;None;;0;0;0\n"
)


class TestDelimiterSniff:
    """A1 — the feed parses, so its records reach the drought gate."""

    def test_a_semicolon_feed_resolves_to_countries(self):
        rows = di._rows_from_text(ASAP_BODY, "text/csv")

        assert len(rows) == 3
        # The tell of the bug was ONE column whose name was the whole header.
        assert len(rows[0]) == 9
        assert [di._record_iso3(r) for r in rows] == ["KEN", "SOM", "ETH"]

    def test_a_comma_feed_is_unchanged(self):
        rows = di._rows_from_text("iso3,value\nKEN,1.2\nSOM,-0.4\n", "text/csv")

        assert rows == [
            {"iso3": "KEN", "value": "1.2"},
            {"iso3": "SOM", "value": "-0.4"},
        ]

    @pytest.mark.parametrize("delimiter", [",", ";", "\t", "|"])
    def test_every_candidate_delimiter_is_read(self, delimiter):
        body = delimiter.join(["iso3", "value"]) + "\n" + delimiter.join(["KEN", "1.2"])

        assert di._rows_from_text(body, "text/csv") == [{"iso3": "KEN", "value": "1.2"}]

    def test_a_single_column_file_still_parses(self):
        assert di._rows_from_text("iso3\nKEN\nSOM\n", "text/csv") == [
            {"iso3": "KEN"},
            {"iso3": "SOM"},
        ]

    def test_a_semicolon_series_becomes_one_snapshot_per_month(self):
        entry = {
            "name": "asap_hotspots",
            "provider": di.PROVIDER_ASAP,
            "match": "classes",
            "drought_classes": ["1", "2"],
            "time_series": True,
        }
        snapshots = di._parse_feed_series(entry, ASAP_BODY, "text/csv", "2026-09")

        by_month = {s["observed_ym"]: s for s in snapshots}
        assert set(by_month) == {"2026-06", "2026-07"}
        assert by_month["2026-07"]["values"] == {"KEN": "2", "SOM": "1"}
        assert by_month["2026-07"]["n_unresolved"] == 0


class TestHdxSignalsWriteStamp:
    """A2 — the write reaches the table AND says which run wrote it."""

    def test_insert_or_replace_leaves_an_unnamed_column_alone(self):
        """The mechanism, pinned. This is why the guard could not see the write."""

        con = duckdb.connect()
        con.execute(
            "CREATE TABLE t (k VARCHAR NOT NULL, v VARCHAR, "
            "ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (k))"
        )
        con.execute("INSERT INTO t (k, v) VALUES ('a', '1')")
        con.execute("UPDATE t SET ts = TIMESTAMP '2020-01-01 00:00:00'")
        con.execute("INSERT OR REPLACE INTO t (k, v) VALUES ('a', '2')")

        v, ts = con.execute("SELECT v, ts FROM t").fetchone()
        assert v == "2"
        assert str(ts).startswith("2020-01-01")

    def test_the_writer_stamps_fetched_at_and_counts_what_it_stamped(
        self, tmp_path, monkeypatch
    ):
        db = tmp_path / "pythia.duckdb"
        monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{db}")

        from pythia.db.schema import connect
        from horizon_scanner import hdx_signals as hs

        signals = [
            {
                "iso3": "KEN",
                "indicator_id": "jrc_agricultural_hotspots",
                "campaign_date": "2026-08-01",
                "alert_level": "High concern",
                "value": "3",
            },
            {
                "iso3": "SOM",
                "indicator_id": "acled_conflict",
                "campaign_date": "2026-08-01",
                "alert_level": "Medium concern",
                "value": "2",
            },
        ]
        first = hs.store_hdx_signals(signals)
        assert first > 0

        con = connect(read_only=True)
        try:
            before = con.execute("SELECT MAX(fetched_at) FROM hdx_signals").fetchone()[0]
        finally:
            con.close()

        # A second identical write is what a monthly cycle actually does.
        second = hs.store_hdx_signals(signals)
        con = connect(read_only=True)
        try:
            after, total = con.execute(
                "SELECT MAX(fetched_at), COUNT(*) FROM hdx_signals"
            ).fetchone()
        finally:
            con.close()

        assert second == first
        assert after > before, "a re-write must move the stamp, or no run can claim it"
        assert total == first

    def test_a_multi_hazard_indicator_keeps_every_hazard(self, tmp_path, monkeypatch):
        """jrc_agricultural_hotspots maps to {DR, HW}. Before Sept 2026 the key
        excluded the hazard, so the two rows collided and the DR prompt or the
        HW prompt — whichever lost — never saw the signal at all."""

        db = tmp_path / "pythia.duckdb"
        monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{db}")

        from pythia.db.schema import connect
        from horizon_scanner import hdx_signals as hs

        hs.store_hdx_signals([
            {
                "iso3": "KEN",
                "indicator_id": "jrc_agricultural_hotspots",
                "campaign_date": "2026-08-01",
                "alert_level": "High concern",
            }
        ])

        con = connect(read_only=True)
        try:
            hazards = [
                r[0]
                for r in con.execute(
                    "SELECT hazard_code FROM hdx_signals ORDER BY hazard_code"
                ).fetchall()
            ]
        finally:
            con.close()
        assert hazards == ["DR", "HW"]

    def test_an_unmapped_indicator_is_kept_under_the_empty_hazard(
        self, tmp_path, monkeypatch
    ):
        db = tmp_path / "pythia.duckdb"
        monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{db}")

        from pythia.db.schema import connect
        from horizon_scanner import hdx_signals as hs

        hs.store_hdx_signals([
            {
                "iso3": "KEN",
                "indicator_id": "an_indicator_we_do_not_map",
                "campaign_date": "2026-08-01",
                "alert_level": "High concern",
            }
        ])

        con = connect(read_only=True)
        try:
            assert con.execute(
                "SELECT hazard_code FROM hdx_signals"
            ).fetchone()[0] == ""
        finally:
            con.close()

    def test_a_legacy_table_is_rebuilt_on_the_hazard_bearing_key(self, tmp_path):
        """The pre-Sept-2026 shape carries its rows across, NULL becoming ''."""

        from pythia.db import schema as sch

        db = tmp_path / "legacy.duckdb"
        con = duckdb.connect(str(db))
        con.execute(
            """
            CREATE TABLE hdx_signals (
                iso3 VARCHAR NOT NULL, hazard_code VARCHAR,
                indicator VARCHAR NOT NULL, concern_level VARCHAR,
                indicator_value DOUBLE, description VARCHAR,
                source_url VARCHAR, signal_date VARCHAR,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (iso3, indicator, signal_date)
            )
            """
        )
        con.execute(
            "INSERT INTO hdx_signals (iso3, hazard_code, indicator, signal_date) "
            "VALUES ('KEN', 'DR', 'jrc_agricultural_hotspots', '2026-08-01'), "
            "       ('SOM', NULL, 'unmapped', '2026-08-01')"
        )

        sch._ensure_hdx_signals_table(con)

        assert "hazard_code" in sch._primary_key_columns(con, "hdx_signals")
        rows = con.execute(
            "SELECT iso3, hazard_code FROM hdx_signals ORDER BY iso3"
        ).fetchall()
        assert rows == [("KEN", "DR"), ("SOM", "")]
        # The rebuild is idempotent and must not re-run on an already-keyed table.
        sch._ensure_hdx_signals_table(con)
        assert con.execute("SELECT COUNT(*) FROM hdx_signals").fetchone()[0] == 2
        con.close()


class TestNmmeNearestCellSampling:
    """A4 — a country smaller than a grid cell is absent, not skipped."""

    def test_a_country_the_mask_never_named_is_sampled(self):
        xr = pytest.importorskip("xarray")
        np = pytest.importorskip("numpy")

        from resolver.ingestion import nmme

        da = xr.DataArray(
            np.arange(9.0).reshape(3, 3),
            coords={"lat": [-1.0, 0.0, 1.0], "lon": [-1.0, 0.0, 1.0]},
            dims=("lat", "lon"),
        )

        class _Region:
            def __init__(self, abbrev, name, point):
                self.abbrev = abbrev
                self.name = name
                self.centroid = point

        class _Regions:
            numbers = [0, 1]
            _by_number = {
                0: _Region("KEN", "Kenya", (0.0, 0.0)),
                1: _Region("MDV", "Maldives", (1.0, 1.0)),
            }

            def __getitem__(self, number):
                return self._by_number[number]

        rows, n_sampled, n_unsampled = nmme._sample_regions_absent_from_mask(
            da, _Regions(), [0], {"KEN"}
        )

        assert n_unsampled == 0
        assert rows == [{"iso3": "MDV", "anomaly_value": 8.0}]
        assert n_sampled == 1

    def test_a_masked_country_is_never_sampled_twice(self):
        xr = pytest.importorskip("xarray")
        np = pytest.importorskip("numpy")

        from resolver.ingestion import nmme

        da = xr.DataArray(
            np.zeros((2, 2)),
            coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]},
            dims=("lat", "lon"),
        )

        class _Region:
            abbrev = "KEN"
            name = "Kenya"
            centroid = (0.0, 0.0)

        class _Regions:
            numbers = [0]

            def __getitem__(self, number):
                return _Region()

        rows, n_sampled, _ = nmme._sample_regions_absent_from_mask(
            da, _Regions(), [], {"KEN"}
        )

        assert rows == []
        assert n_sampled == 0
