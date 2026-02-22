# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

from resolver.tests.test_utils import SNAPS, read_parquet

def test_any_snapshot_parquet_reads_and_has_core_columns():
    if not SNAPS.exists():
        return
    # take any new-style snapshot first, fall back to legacy naming
    paths = list(SNAPS.glob("*/facts_resolved.parquet"))
    if not paths:
        paths = list(SNAPS.glob("*/facts.parquet"))
    if not paths:
        return
    df = read_parquet(paths[0])
    core = {"iso3","hazard_code","metric","value","as_of_date","publication_date"}
    assert core.issubset(set(df.columns))
