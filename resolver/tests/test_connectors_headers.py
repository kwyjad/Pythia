# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from pathlib import Path
import csv
import importlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "staging"

CANON = [
    "event_id","country_name","iso3",
    "hazard_code","hazard_label","hazard_class",
    "metric","series_semantics","value","unit",
    "as_of_date","publication_date",
    "publisher","source_type","source_url","doc_title",
    "definition_text","method","confidence",
    "revision","ingested_at"
]

def _assert_header(csv_path: Path):
    assert csv_path.exists(), f"missing {csv_path}"
    df = pd.read_csv(csv_path, dtype=str)
    assert list(df.columns) == CANON, f"{csv_path} columns differ: {list(df.columns)}"

def test_ifrc_go_header(tmp_path, monkeypatch):
    # Hermetic: skip network and force header-only CSV if needed
    monkeypatch.setenv("RESOLVER_SKIP_IFRCGO", "1")
    mod = importlib.import_module("resolver.ingestion.ifrc_go_client")
    mod.main()
    _assert_header(STAGING / "ifrc_go.csv")


def test_acled_header_written(tmp_path, monkeypatch):
    monkeypatch.setenv("RESOLVER_SKIP_ACLED", "1")
    from resolver.ingestion import acled_client

    acled_client.OUT_PATH = Path(tmp_path) / "acled.csv"
    assert acled_client.main() is False

    with open(acled_client.OUT_PATH, newline="", encoding="utf-8") as f:
        row = next(csv.reader(f))
    assert row == acled_client.CANONICAL_HEADERS
