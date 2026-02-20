# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Unit tests for the IDMC normalize adapter."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from resolver.transform.adapters.idmc import IDMCAdapter
from resolver.transform.adapters.base import CANONICAL_COLUMNS


@pytest.fixture()
def idmc_staging_dir(tmp_path: Path) -> Path:
    """Create a staging directory with an idmc/flow.csv file."""

    idmc_dir = tmp_path / "idmc"
    idmc_dir.mkdir()
    csv_text = textwrap.dedent("""\
        iso3,as_of_date,metric,value,series_semantics,source
        SDN,2024-02-29,new_displacements,800,new,idmc_idu
        COD,2024-01-31,new_displacements,500,new,idmc_idu
        SDN,2024-02-29,new_displacements,700,new,idmc_idu
    """)
    (idmc_dir / "flow.csv").write_text(csv_text, encoding="utf-8")
    return tmp_path


@pytest.fixture()
def idmc_flat_file(tmp_path: Path) -> Path:
    """Create a staging directory with a flat idmc.csv (fallback)."""

    csv_text = textwrap.dedent("""\
        iso3,as_of_date,metric,value,series_semantics,source
        ETH,2024-03-31,new_displacements,1200,new,idmc_idu
    """)
    (tmp_path / "idmc.csv").write_text(csv_text, encoding="utf-8")
    return tmp_path


@pytest.fixture()
def adapter() -> IDMCAdapter:
    return IDMCAdapter("idmc")


class TestIDMCAdapterResolveRawPath:
    def test_finds_subdirectory_flow_csv(
        self, adapter: IDMCAdapter, idmc_staging_dir: Path
    ) -> None:
        path = adapter.resolve_raw_path(idmc_staging_dir)
        assert path.name == "flow.csv"
        assert path.parent.name == "idmc"

    def test_fallback_to_flat_file(
        self, adapter: IDMCAdapter, idmc_flat_file: Path
    ) -> None:
        path = adapter.resolve_raw_path(idmc_flat_file)
        assert path.name == "idmc.csv"

    def test_raises_when_not_found(
        self, adapter: IDMCAdapter, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError, match="No IDMC staging CSV"):
            adapter.resolve_raw_path(tmp_path)


class TestIDMCAdapterMap:
    def test_produces_canonical_columns(
        self, adapter: IDMCAdapter, idmc_staging_dir: Path
    ) -> None:
        raw_path = adapter.resolve_raw_path(idmc_staging_dir)
        raw = adapter.load(raw_path)
        canonical = adapter.map(raw)
        assert list(canonical.columns) == CANONICAL_COLUMNS

    def test_source_is_idmc(
        self, adapter: IDMCAdapter, idmc_staging_dir: Path
    ) -> None:
        raw_path = adapter.resolve_raw_path(idmc_staging_dir)
        raw = adapter.load(raw_path)
        canonical = adapter.map(raw)
        assert (canonical["source"] == "idmc").all()

    def test_hazard_code_is_idu(
        self, adapter: IDMCAdapter, idmc_staging_dir: Path
    ) -> None:
        raw_path = adapter.resolve_raw_path(idmc_staging_dir)
        raw = adapter.load(raw_path)
        canonical = adapter.map(raw)
        assert (canonical["hazard_code"] == "IDU").all()

    def test_hazard_label_populated(
        self, adapter: IDMCAdapter, idmc_staging_dir: Path
    ) -> None:
        raw_path = adapter.resolve_raw_path(idmc_staging_dir)
        raw = adapter.load(raw_path)
        canonical = adapter.map(raw)
        assert (canonical["hazard_label"] == "Internal Displacement").all()

    def test_unit_is_persons(
        self, adapter: IDMCAdapter, idmc_staging_dir: Path
    ) -> None:
        raw_path = adapter.resolve_raw_path(idmc_staging_dir)
        raw = adapter.load(raw_path)
        canonical = adapter.map(raw)
        assert (canonical["unit"] == "persons").all()

    def test_series_semantics_preserved(
        self, adapter: IDMCAdapter, idmc_staging_dir: Path
    ) -> None:
        raw_path = adapter.resolve_raw_path(idmc_staging_dir)
        raw = adapter.load(raw_path)
        canonical = adapter.map(raw)
        assert (canonical["series_semantics"] == "new").all()

    def test_metric_preserved(
        self, adapter: IDMCAdapter, idmc_staging_dir: Path
    ) -> None:
        raw_path = adapter.resolve_raw_path(idmc_staging_dir)
        raw = adapter.load(raw_path)
        canonical = adapter.map(raw)
        assert (canonical["metric"] == "new_displacements").all()

    def test_value_is_numeric(
        self, adapter: IDMCAdapter, idmc_staging_dir: Path
    ) -> None:
        raw_path = adapter.resolve_raw_path(idmc_staging_dir)
        raw = adapter.load(raw_path)
        canonical = adapter.map(raw)
        assert canonical["value"].dtype == float

    def test_as_of_date_formatted(
        self, adapter: IDMCAdapter, idmc_staging_dir: Path
    ) -> None:
        raw_path = adapter.resolve_raw_path(idmc_staging_dir)
        raw = adapter.load(raw_path)
        canonical = adapter.map(raw)
        # Already YYYY-MM-DD from IDMC; verify kept as-is
        sdn_rows = canonical[canonical["iso3"] == "SDN"]
        assert (sdn_rows["as_of_date"] == "2024-02-29").all()

    def test_empty_input_returns_empty_canonical(
        self, adapter: IDMCAdapter
    ) -> None:
        empty = pd.DataFrame(columns=[
            "iso3", "as_of_date", "metric", "value",
            "series_semantics", "source",
        ])
        result = adapter.map(empty)
        assert result.empty
        assert list(result.columns) == CANONICAL_COLUMNS


class TestIDMCAdapterNormalize:
    def test_normalize_end_to_end(
        self, adapter: IDMCAdapter, idmc_staging_dir: Path
    ) -> None:
        canonical = adapter.normalize(idmc_staging_dir)
        assert len(canonical) == 3
        assert list(canonical.columns) == CANONICAL_COLUMNS
