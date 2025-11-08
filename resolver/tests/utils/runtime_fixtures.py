"""Helpers that build small IDMC staging trees at runtime for DuckDB tests."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass
class IdmcRuntimeFixture:
    """Paths and metadata for the generated IDMC staging assets."""

    staging_dir: Path
    flow_csv: Path
    parquet_path: Optional[Path]
    stock_csv: Optional[Path]
    parquet_available: bool
    flow_rows: int


def _build_flow_dataframe() -> pd.DataFrame:
    data = [
        {
            "iso3": "COL",
            "as_of_date": "2024-01-31",
            "metric": "new_displacements",
            "value": 100,
            "series_semantics": "new",
            "source": "IDMC Internal Displacement Monitoring Centre",
        },
        {
            "iso3": "COL",
            "as_of_date": "2024-02-29",
            "metric": "new_displacements",
            "value": 120,
            "series_semantics": "new",
            "source": "IDMC Internal Displacement Monitoring Centre",
        },
        {
            "iso3": "COL",
            "as_of_date": "2024-02-29",
            "metric": "new_displacements",
            "value": 150,
            "series_semantics": "new",
            "source": "IDMC Update Duplicate",
        },
        {
            "iso3": "ETH",
            "as_of_date": "2024-01-31",
            "metric": "new_displacements",
            "value": 90,
            "series_semantics": "new",
            "source": "IDMC Internal Displacement Monitoring Centre",
        },
        {
            "iso3": "ETH",
            "as_of_date": "2024-02-29",
            "metric": "new_displacements",
            "value": 110,
            "series_semantics": "new",
            "source": "IDMC Internal Displacement Monitoring Centre",
        },
        {
            "iso3": "NGA",
            "as_of_date": "2024-01-31",
            "metric": "new_displacements",
            "value": 70,
            "series_semantics": "new",
            "source": "IDMC Internal Displacement Monitoring Centre",
        },
    ]
    frame = pd.DataFrame(data)
    LOGGER.info("runtime_fixture.flow | rows=%s", len(frame))
    return frame


def _write_stock_csv(path: Path, empty: bool) -> None:
    columns = [
        "iso3",
        "as_of_date",
        "metric",
        "value",
        "series_semantics",
        "source",
    ]
    if empty:
        df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(
            [
                {
                    "iso3": "COL",
                    "as_of_date": "2024-01-31",
                    "metric": "idp_displacement_stock_idmc",
                    "value": 200,
                    "series_semantics": "stock",
                    "source": "IDMC",
                },
                {
                    "iso3": "ETH",
                    "as_of_date": "2024-02-29",
                    "metric": "idp_displacement_stock_idmc",
                    "value": 250,
                    "series_semantics": "stock",
                    "source": "IDMC",
                },
            ]
        )
    df.to_csv(path, index=False)
    LOGGER.info("runtime_fixture.stock | path=%s rows=%s", path, len(df))


def _maybe_write_parquet(parquet_path: Path, flow_frame: pd.DataFrame) -> tuple[Optional[Path], bool]:
    if parquet_path.exists():
        parquet_path.unlink()
    try:
        df = flow_frame.assign(
            ym=pd.to_datetime(flow_frame["as_of_date"]).dt.strftime("%Y-%m"),
            semantics=flow_frame["series_semantics"],
        )[
            [
                "iso3",
                "ym",
                "as_of_date",
                "metric",
                "value",
                "semantics",
                "source",
            ]
        ]
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)
    except (ImportError, RuntimeError, ValueError) as exc:
        LOGGER.warning("runtime_fixture.parquet.skip | reason=%s", exc)
        return None, False
    else:
        LOGGER.info("runtime_fixture.parquet | path=%s rows=%s", parquet_path, len(df))
        return parquet_path, True


def create_idmc_runtime_fixture(
    base_dir: Path,
    *,
    include_stock: bool = True,
    empty_stock: bool = False,
    write_parquet: bool = True,
) -> IdmcRuntimeFixture:
    """Build a temporary IDMC staging tree under ``base_dir``."""

    staging_dir = base_dir / "resolver" / "staging" / "idmc"
    staging_dir.mkdir(parents=True, exist_ok=True)

    flow_frame = _build_flow_dataframe()
    flow_path = staging_dir / "flow.csv"
    flow_frame.to_csv(flow_path, index=False)
    LOGGER.info("runtime_fixture.flow.write | path=%s", flow_path)

    parquet_path: Optional[Path] = None
    parquet_available = False
    if write_parquet:
        parquet_candidate = staging_dir / "idmc_facts_flow.parquet"
        parquet_path, parquet_available = _maybe_write_parquet(parquet_candidate, flow_frame)

    stock_path: Optional[Path] = None
    if include_stock:
        stock_path = staging_dir / "stock.csv"
        _write_stock_csv(stock_path, empty_stock)

    return IdmcRuntimeFixture(
        staging_dir=staging_dir,
        flow_csv=flow_path,
        parquet_path=parquet_path,
        stock_csv=stock_path,
        parquet_available=parquet_available,
        flow_rows=len(flow_frame),
    )
