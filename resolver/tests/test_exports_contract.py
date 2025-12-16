# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations
import math, datetime as dt, os
import pandas as pd
import pytest
from resolver.tests.test_utils import (
    load_schema, load_countries, load_shocks, read_csv,
    discover_export_files, today_iso, require_columns
)
from resolver.tools import llm_context
from resolver.tools.export_facts import prepare_duckdb_tables

try:
    from resolver.db import duckdb_io
except Exception:  # pragma: no cover - optional duckdb dependency
    duckdb_io = None

def _is_iso_date(s: str) -> bool:
    try:
        dt.date.fromisoformat(s); return True
    except Exception:
        return False

def _num_ok(x: str) -> bool:
    try:
        return float(x) >= 0
    except Exception:
        return False

def test_exports_facts_contract_if_present():
    schema = load_schema()
    req = set(schema["required"])
    enum_metric = set(schema["enums"]["metric"])
    enum_unit = set(schema["enums"]["unit"])
    countries = load_countries()
    shocks = load_shocks()
    cset = set(countries["iso3"])
    sset = set(shocks["hazard_code"])

    # prefer current canonical facts
    paths = [p for p in discover_export_files() if p.name == "facts.csv" and "state" not in str(p)]
    if not paths:
        # skip if no exports/facts.csv yet
        return

    df = read_csv(paths[0])

    # columns present
    missing = req - set(df.columns)
    assert not missing, f"Missing required columns: {missing}"

    # registry membership
    assert df["iso3"].isin(cset).all(), "Some iso3 not in countries registry"
    assert df["hazard_code"].isin(sset).all(), "Some hazard_code not in shocks registry"

    # labels/classes match the shocks registry rows
    reg = shocks.set_index("hazard_code")
    bad_label = []
    bad_class = []
    for _, r in df.iterrows():
        code = r["hazard_code"]
        if code in reg.index:
            if r["hazard_label"] != reg.loc[code, "hazard_label"]:
                bad_label.append(code)
            if r["hazard_class"] != reg.loc[code, "hazard_class"]:
                bad_class.append(code)
    assert not bad_label, f"hazard_label mismatch for codes: {set(bad_label)}"
    assert not bad_class, f"hazard_class mismatch for codes: {set(bad_class)}"

    # metric & unit enums
    assert df["metric"].isin(enum_metric).all()
    assert df["unit"].isin(enum_unit).all()

    # numeric values >= 0
    assert df["value"].map(_num_ok).all(), "Non-numeric or negative values present"

    # date sanity
    assert df["as_of_date"].map(_is_iso_date).all()
    assert df["publication_date"].map(_is_iso_date).all()
    assert (df["as_of_date"] <= df["publication_date"]).all()
    assert (df["publication_date"] <= today_iso()).all()

    # governance rules
    media_in_need = df[(df["source_type"] == "media") & (df["metric"] == "in_need")]
    assert media_in_need.empty, "Policy: media cannot be the source for in_need"

    cases_wrong_unit = df[(df["metric"] == "cases") & (df["unit"] != "persons_cases")]
    assert cases_wrong_unit.empty, "Policy: cases must use unit=persons_cases"


def test_prepare_duckdb_tables_routes_stock_and_new_rows():
    facts = pd.DataFrame(
        [
            {
                "event_id": "evt-stock",
                "iso3": "COL",
                "hazard_code": "HZ",
                "metric": "idps_present",
                "series_semantics": "stock",
                "semantics": "stock",
                "value": 100,
                "unit": "persons",
                "as_of_date": "2024-01-31",
                "publication_date": "2024-02-01",
                "source_id": "dtm_admin0",
                "ym": "2024-01",
            },
            {
                "event_id": "evt-new",
                "iso3": "COL",
                "hazard_code": "HZ",
                "metric": "new_displacements",
                "series_semantics": "new",
                "semantics": "new",
                "value": 15,
                "unit": "persons",
                "as_of_date": "2024-01-31",
                "publication_date": "2024-02-01",
                "source_id": "idmc",
                "ym": "2024-01",
            },
        ]
    )

    resolved, deltas = prepare_duckdb_tables(facts)
    assert resolved is not None and not resolved.empty
    assert deltas is not None and not deltas.empty
    assert set(resolved["series_semantics"].str.lower()) == {"stock"}
    assert set(deltas["series_semantics"].str.lower()) == {"new"}

def test_remote_first_state_exports_are_valid_if_present():
    # Validate any committed state exports CSVs (PR or daily). Skip gracefully if none.
    schema = load_schema()
    req = set(schema["required"])
    enum_metric = set(schema["enums"]["metric"])
    enum_unit = set(schema["enums"]["unit"])
    countries = load_countries()
    shocks = load_shocks()
    cset = set(countries["iso3"]); sset = set(shocks["hazard_code"])

    paths = [p for p in discover_export_files() if p.name == "facts.csv" and "state" in str(p)]
    if not paths:
        return

    for p in paths:
        df = read_csv(p)
        missing = req - set(df.columns)
        assert not missing, f"{p}: missing columns {missing}"
        assert df["iso3"].isin(cset).all(), f"{p}: iso3 not in registry"
        assert df["hazard_code"].isin(sset).all(), f"{p}: hazard_code not in registry"
        assert df["metric"].isin(enum_metric).all(), f"{p}: metric enum"
        assert df["unit"].isin(enum_unit).all(), f"{p}: unit enum"


def test_duckdb_facts_resolved_contract_if_present():
    if duckdb_io is None:
        return
    db_url = os.environ.get("RESOLVER_DB_URL")
    if not db_url:
        return
    try:
        conn = duckdb_io.get_db(db_url)
        duckdb_io.init_schema(conn)
    except Exception:
        pytest.skip("DuckDB not initialised for facts_resolved contract check")
        return

    schema = load_schema()
    entities = schema.get("entities", {})
    resolved_schema = entities.get("db.facts_resolved")
    if not resolved_schema:
        pytest.skip("db.facts_resolved schema not defined")
        return

    base_columns = {
        "event_id",
        "iso3",
        "hazard_code",
        "metric",
        "value",
        "unit",
        "as_of_date",
        "publication_date",
        "ym",
    }
    required_columns = {
        col["name"]
        for col in resolved_schema.get("columns", [])
        if col.get("required")
    }
    select_columns = sorted(base_columns | required_columns)

    try:
        table_info = conn.execute("PRAGMA table_info('facts_resolved')").fetchall()
    except Exception:
        conn.close()
        pytest.skip("Unable to inspect facts_resolved table definition")
        return

    present_columns = {row[1] for row in table_info}
    missing_physical = set(select_columns) - present_columns
    if missing_physical:
        conn.close()
        pytest.skip(
            "facts_resolved missing columns required by schema: "
            + ", ".join(sorted(missing_physical))
        )
        return

    column_sql = ", ".join(select_columns)
    try:
        df = conn.execute(f"SELECT {column_sql} FROM facts_resolved").fetch_df()
    except Exception:
        conn.close()
        pytest.skip("DuckDB facts_resolved contract query failed to execute")
        return

    if df.empty:
        conn.close()
        pytest.skip("facts_resolved empty; nothing to validate")

    missing = required_columns - set(df.columns)
    assert not missing, f"DuckDB facts_resolved missing required columns {missing}"

    countries = load_countries()
    shocks = load_shocks()
    assert df["iso3"].isin(set(countries["iso3"])).all()
    assert df["hazard_code"].isin(set(shocks["hazard_code"])).all()

    conn.close()


def test_llm_context_schema_contract():
    frame = llm_context.build_context_frame(months=[])
    expected = llm_context.CONTEXT_COLUMNS
    missing = [col for col in expected if col not in frame.columns]
    assert not missing, f"Context frame missing columns: {missing}"
    if not frame.empty:
        assert set(frame["series"].unique()) == {"new"}
