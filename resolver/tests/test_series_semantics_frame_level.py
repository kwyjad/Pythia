# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Frame-level series_semantics canonicalisation tests."""

from __future__ import annotations

import pandas as pd
import pytest

from resolver.db import duckdb_io


@pytest.fixture()
def frame_with_unknown_semantics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_semantics": [
                "new",
                "Random text",
                None,
                "stock_estimate",
                "  STOCK  ",
            ]
        }
    )


def test_unknowns_become_default_for_deltas(frame_with_unknown_semantics):
    canonical, _ = duckdb_io._canonicalize_semantics(
        frame_with_unknown_semantics.copy(),
        table_name="facts_deltas",
        default_target="new",
    )
    assert canonical["series_semantics"].tolist() == ["new"] * 5
    duckdb_io._assert_semantics_required(canonical, "facts_deltas")


def test_unknowns_become_default_for_snapshots(frame_with_unknown_semantics):
    canonical, _ = duckdb_io._canonicalize_semantics(
        frame_with_unknown_semantics.copy(),
        table_name="facts_resolved",
        default_target="stock",
    )
    assert canonical["series_semantics"].tolist() == ["stock"] * 5
    duckdb_io._assert_semantics_required(canonical, "facts_resolved")


@pytest.mark.parametrize("default_target", ["new", "stock"])
def test_idempotent_semantics_canonicalisation(
    frame_with_unknown_semantics, default_target
):
    table = "facts_deltas" if default_target == "new" else "facts_resolved"
    canonical, _ = duckdb_io._canonicalize_semantics(
        frame_with_unknown_semantics.copy(),
        table_name=table,
        default_target=default_target,
    )
    rerun, _ = duckdb_io._canonicalize_semantics(
        canonical.copy(),
        table_name=table,
        default_target=default_target,
    )
    assert rerun["series_semantics"].tolist() == canonical["series_semantics"].tolist()
