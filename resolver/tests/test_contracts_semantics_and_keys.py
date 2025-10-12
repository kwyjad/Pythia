"""Data contract tests for snapshot semantics and natural keys."""

from __future__ import annotations

import uuid

import pandas as pd
import pytest

from resolver.db import duckdb_io


def _db_url(tmp_path) -> str:
    db_path = tmp_path / f"contract_{uuid.uuid4().hex}.duckdb"
    return f"duckdb:///{db_path}"


def _resolved_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ym": "2024-01",
                "iso3": "PHL",
                "hazard_code": "TC",
                "metric": "in_need",
                "series_semantics": "new",
                "value": 1000,
                "as_of_date": "2024-01-15",
                "publication_date": "2024-01-16",
                "hazard_label": "Tropical Cyclone",
                "hazard_class": "Cyclone",
                "publisher": "OCHA",
                "unit": "persons",
            },
            {
                "ym": "2024-01",
                "iso3": "PHL",
                "hazard_code": "EQ",
                "metric": "affected",
                "series_semantics": "",
                "value": 250,
                "as_of_date": "2024-01-20",
                "publication_date": "2024-01-21",
                "hazard_label": "Earthquake",
                "hazard_class": "Geophysical",
                "publisher": "OCHA",
                "unit": "persons",
            },
        ]
    )


def _deltas_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ym": "2024-01",
                "iso3": "PHL",
                "hazard_code": "TC",
                "metric": "in_need",
                "series_semantics": "stock",
                "value_new": 300,
                "source_name": "resolver",
                "as_of": "2024-01-31",
            },
            {
                "ym": "2024-01",
                "iso3": "PHL",
                "hazard_code": "EQ",
                "metric": "affected",
                "series_semantics": "",
                "value_new": 120,
                "source_name": "resolver",
                "as_of": "2024-01-31",
            },
        ]
    )


def test_facts_resolved_semantics_stock_only(tmp_path) -> None:
    conn = duckdb_io.get_db(_db_url(tmp_path))
    try:
        duckdb_io.write_snapshot(
            conn,
            ym="2024-01",
            facts_resolved=_resolved_frame(),
            facts_deltas=None,
            manifests=None,
            meta=None,
        )
        values = {
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT series_semantics FROM facts_resolved"
            ).fetchall()
        }
        assert values == {"stock"}
    finally:
        conn.close()


def test_facts_deltas_semantics_new_only(tmp_path) -> None:
    conn = duckdb_io.get_db(_db_url(tmp_path))
    try:
        duckdb_io.write_snapshot(
            conn,
            ym="2024-01",
            facts_resolved=None,
            facts_deltas=_deltas_frame(),
            manifests=None,
            meta=None,
        )
        values = {
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT series_semantics FROM facts_deltas"
            ).fetchall()
        }
        assert values == {"new"}
    finally:
        conn.close()


def test_composite_keys_non_null_and_unique(tmp_path) -> None:
    conn = duckdb_io.get_db(_db_url(tmp_path))
    try:
        base_resolved = _resolved_frame()
        base_deltas = _deltas_frame()
        resolved_dupes = pd.concat(
            [base_resolved, base_resolved.iloc[[0]].assign(value=9999)],
            ignore_index=True,
        )
        deltas_dupes = pd.concat(
            [base_deltas, base_deltas.iloc[[0]].assign(value_new=777)],
            ignore_index=True,
        )
        duckdb_io.write_snapshot(
            conn,
            ym="2024-01",
            facts_resolved=base_resolved,
            facts_deltas=base_deltas,
            manifests=None,
            meta=None,
        )
        duckdb_io.write_snapshot(
            conn,
            ym="2024-01",
            facts_resolved=resolved_dupes,
            facts_deltas=deltas_dupes,
            manifests=None,
            meta=None,
        )
        resolved_df = conn.execute(
            "SELECT ym, iso3, hazard_code, metric, series_semantics, value FROM facts_resolved"
        ).df()
        key_cols_resolved = duckdb_io.FACTS_RESOLVED_KEY_COLUMNS
        for column in key_cols_resolved:
            assert resolved_df[column].notna().all(), f"Expected non-null {column}"
            series = resolved_df[column].astype(str).str.strip()
            assert series.ne("").all(), f"Expected non-empty {column}"
        assert len(resolved_df) == len(resolved_df.drop_duplicates(subset=key_cols_resolved))
        target_resolved = resolved_df[
            (resolved_df["ym"] == "2024-01")
            & (resolved_df["iso3"] == "PHL")
            & (resolved_df["hazard_code"] == "TC")
            & (resolved_df["metric"] == "in_need")
            & (resolved_df["series_semantics"] == "stock")
        ]
        assert not target_resolved.empty
        assert pytest.approx(9999.0) == target_resolved["value"].iloc[0]

        deltas_df = conn.execute(
            "SELECT ym, iso3, hazard_code, metric, series_semantics, value_new FROM facts_deltas"
        ).df()
        key_cols_deltas = duckdb_io.FACTS_DELTAS_KEY_COLUMNS
        for column in key_cols_deltas:
            assert deltas_df[column].notna().all(), f"Expected non-null {column}"
            series = deltas_df[column].astype(str).str.strip()
            assert series.ne("").all(), f"Expected non-empty {column}"
        assert len(deltas_df) == len(deltas_df.drop_duplicates(subset=key_cols_deltas))
        target_delta = deltas_df[
            (deltas_df["ym"] == "2024-01")
            & (deltas_df["iso3"] == "PHL")
            & (deltas_df["hazard_code"] == "TC")
            & (deltas_df["metric"] == "in_need")
        ]
        assert not target_delta.empty
        assert pytest.approx(777.0) == target_delta["value_new"].iloc[0]
    finally:
        conn.close()
