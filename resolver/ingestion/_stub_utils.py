# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import datetime as dt
import pandas as pd
from pathlib import Path

from resolver.ingestion.utils.io import resolve_output_path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STAGING = ROOT / "staging"

COUNTRIES = DATA / "countries.csv"
SHOCKS = DATA / "shocks.csv"

REQUIRED_COLUMNS = [
    "event_id","country_name","iso3",
    "hazard_code","hazard_label","hazard_class",
    "metric","value","unit",
    "as_of_date","publication_date",
    "publisher","source_type","source_url","doc_title",
    "definition_text","method","confidence",
    "revision","ingested_at"
]

def load_registries():
    c = pd.read_csv(COUNTRIES, dtype=str).fillna("")
    s = pd.read_csv(SHOCKS, dtype=str).fillna("")
    return c, s

def now_dates():
    today = dt.date.today()
    as_of = today.strftime("%Y-%m-%d")
    pub = today.strftime("%Y-%m-%d")
    ing = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    return as_of, pub, ing

def write_staging(rows, out_path: Path, *, series_semantics: str = "stock"):
    resolved_path = resolve_output_path(out_path)
    df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    insert_pos = df.columns.get_loc("metric") + 1 if "metric" in df.columns else len(df.columns)
    semantics_value = str(series_semantics or "stock").strip().lower() or "stock"
    df.insert(insert_pos, "series_semantics", semantics_value)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(resolved_path, index=False)
    return resolved_path
