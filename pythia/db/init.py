from __future__ import annotations
import pathlib, duckdb

SCHEMA_PATH = pathlib.Path(__file__).with_name("schema.sql")

def init(db_url: str):
    con = duckdb.connect(db_url.replace("duckdb:///", ""))
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        con.execute(f.read())
    con.close()
