# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Offline tests for the IDMC facts export wiring."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from resolver.ingestion.idmc.exporter import to_facts, write_facts_csv


def test_export_adapter_writes_facts_csv(tmp_path: Path) -> None:
    fixture_path = Path("resolver/ingestion/tests/fixtures/idmc_normalized_small.csv")
    df = pd.read_csv(fixture_path, comment="#")

    facts = to_facts(df)
    assert len(facts) == 2

    sdn = facts[(facts["iso3"] == "SDN") & (facts["as_of_date"] == "2024-02-29")]
    assert len(sdn) == 1
    assert float(sdn["value"].iloc[0]) == 800.0

    out_dir = tmp_path / "idmc"
    path = write_facts_csv(facts, str(out_dir))
    assert os.path.exists(path)

    cols = ["iso3", "as_of_date", "metric", "value", "series_semantics", "source"]
    written = pd.read_csv(path)
    assert list(written.columns) == cols
