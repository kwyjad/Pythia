# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

from pathlib import Path

import pytest

from resolver.io import files_locator
from resolver.query.selectors import resolve_point

TEST_DATA = Path(__file__).resolve().parent / "data"


@pytest.fixture()
def _files_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESOLVER_SNAPSHOTS_DIR", str(TEST_DATA))


def test_discover_files_root_prefers_env(_files_env: None) -> None:
    root = files_locator.discover_files_root()
    assert root == TEST_DATA


def test_load_table_finds_facts_deltas(_files_env: None) -> None:
    root = files_locator.discover_files_root()
    df = files_locator.load_table(root, "facts_deltas")
    assert not df.empty
    required = {
        "ym": "2024-01",
        "iso3": "PHL",
        "hazard_code": "TC",
        "metric": "in_need",
    }
    for column, expected in required.items():
        assert column in df.columns
        assert expected in set(df[column].astype(str))


@pytest.mark.parametrize("backend", ["files", "csv"])
def test_resolve_point_files_backend_returns_row(_files_env: None, backend: str) -> None:
    result = resolve_point(
        iso3="PHL",
        hazard_code="TC",
        cutoff="2024-01-31",
        series="new",
        metric="in_need",
        backend=backend,
    )
    assert result is not None
    assert result.get("value") == 1500
    assert result.get("series_returned") == "new"
