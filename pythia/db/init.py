# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations
import pathlib, duckdb

SCHEMA_PATH = pathlib.Path(__file__).with_name("schema.sql")

def init(db_url: str):
    con = duckdb.connect(db_url.replace("duckdb:///", ""))
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        con.execute(f.read())
    con.close()
