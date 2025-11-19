"""Offline tests for the ODP normalization and DuckDB writer."""

from __future__ import annotations

from datetime import date
import json
import re

import pandas as pd
import pytest

from resolver.ingestion import odp_discovery
from resolver.ingestion import odp_duckdb
from resolver.ingestion import odp_series


SAMPLE_PAYLOAD = [
    {"date": "2025-01-15", "value": 100, "iso3": "UGA"},
    {"date": "2025-01-31", "value": 150, "iso3": "UGA"},
]


@pytest.fixture()
def sample_spec() -> odp_series.NormalizerSpec:
    return odp_series.NormalizerSpec(
        id="refugees",
        label_regex=re.compile(r"refugees", re.IGNORECASE),
        metric="refugees_total",
        series_semantics="stock",
        frequency="monthly",
        value_field="value",
        date_field="date",
        date_format="%Y-%m-%d",
        iso3_field="iso3",
        origin_iso3_field=None,
        admin_name_field=None,
        required_fields=["date", "value", "iso3"],
        unit="persons",
    )


@pytest.fixture()
def sample_discovery() -> odp_discovery.PageDiscovery:
    return odp_discovery.PageDiscovery(
        page_id="page-1",
        page_url="https://example.org/page",
        links=[
            odp_discovery.DiscoveredLink(
                href="https://example.org/widget?format=json",
                text="Refugees by origin",
            )
        ],
    )


def test_odp_normalize_basic_monthly_series(sample_spec, sample_discovery):
    def fake_fetch_json(url: str):  # noqa: ARG001 - signature parity for clarity
        assert url == "https://example.org/widget?format=json"
        return SAMPLE_PAYLOAD

    frame = odp_series.normalize_all([sample_discovery], fake_fetch_json, [sample_spec], today=date(2025, 2, 1))
    assert len(frame) == 2
    assert frame["iso3"].unique().tolist() == ["UGA"]
    assert frame["ym"].unique().tolist() == ["2025-01"]
    assert frame["as_of_date"].unique().tolist() == [date(2025, 1, 31)]
    assert frame["metric"].unique().tolist() == ["refugees_total"]
    assert frame["series_semantics"].unique().tolist() == ["stock"]
    assert frame["value"].tolist() == [100.0, 150.0]


def test_odp_normalize_skips_unmatched_widget(sample_spec):
    discovery = odp_discovery.PageDiscovery(
        page_id="page-1",
        page_url="https://example.org/page",
        links=[
            odp_discovery.DiscoveredLink(
                href="https://example.org/widget?format=json",
                text="Cases by status",
            )
        ],
    )

    frame = odp_series.normalize_all([discovery], lambda url: SAMPLE_PAYLOAD, [sample_spec])
    assert frame.empty


def test_odp_duckdb_writer_idempotent(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    frame = pd.DataFrame(
        [
            {
                "source_id": "refugees",
                "iso3": "UGA",
                "origin_iso3": None,
                "admin_name": None,
                "ym": "2025-01",
                "as_of_date": date(2025, 1, 31),
                "metric": "refugees_total",
                "series_semantics": "stock",
                "value": 100.0,
                "unit": "persons",
                "extra": json.dumps({"page_url": "https://example.org", "json_url": "https://example.org/widget"}),
            }
        ]
    )
    db_path = tmp_path / "odp.duckdb"
    db_url = f"duckdb:///{db_path}"

    odp_duckdb.write_odp_timeseries(frame, db_url)
    odp_duckdb.write_odp_timeseries(frame, db_url)

    conn = duckdb.connect(db_path)
    try:
        table_info = conn.execute("PRAGMA table_info('odp_timeseries_raw')").fetchall()
        assert table_info, "Expected odp_timeseries_raw to exist"
        count = conn.execute("SELECT COUNT(*) FROM odp_timeseries_raw").fetchone()[0]
        assert count == 1
        stored = conn.execute("SELECT value FROM odp_timeseries_raw").fetchone()[0]
        assert stored == 100.0
    finally:
        conn.close()


def test_build_and_write_odp_series_with_stubs(tmp_path, monkeypatch):
    duckdb = pytest.importorskip("duckdb")
    config_path = tmp_path / "odp.yml"
    config_path.write_text("pages:\n  - id: demo\n    url: https://example.org/demo\n", encoding="utf-8")

    normalizers_path = tmp_path / "normalizers.yml"
    normalizers_path.write_text(
        """
defaults:
  unit: persons
series:
  - id: refugees_by_origin
    label_regex: "(?i)refugees by origin"
    metric: refugees_total
    series_semantics: stock
    frequency: monthly
    value_field: value
    date_field: date
    date_format: "%Y-%m-%d"
    iso3_field: iso3
    required_fields: [date, value, iso3]
""".strip(),
        encoding="utf-8",
    )

    discovery = odp_discovery.PageDiscovery(
        page_id="demo",
        page_url="https://example.org/demo",
        links=[
            odp_discovery.DiscoveredLink(
                href="https://example.org/widget?format=json",
                text="Refugees by origin",
            )
        ],
    )
    monkeypatch.setattr(odp_discovery, "discover_pages", lambda config, fetch_html=None: [discovery])

    def fake_fetch_json(url: str):  # noqa: ARG001
        return [{"date": "2025-01-15", "value": 200, "iso3": "UGA"}]

    db_path = tmp_path / "odp.duckdb"
    db_url = f"duckdb:///{db_path}"

    rows = odp_duckdb.build_and_write_odp_series(
        config_path=config_path,
        normalizers_path=normalizers_path,
        db_url=db_url,
        fetch_json=fake_fetch_json,
    )
    assert rows == 1

    conn = duckdb.connect(db_path)
    try:
        table_info = conn.execute("PRAGMA table_info('odp_timeseries_raw')").fetchall()
        assert table_info, "Expected odp_timeseries_raw to exist"
        stored = conn.execute("SELECT source_id, value, ym FROM odp_timeseries_raw").fetchall()
        assert stored == [("refugees_by_origin", 200.0, "2025-01")]
    finally:
        conn.close()
