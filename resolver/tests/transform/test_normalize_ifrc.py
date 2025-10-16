"""Tests for IFRC canonical normalization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from resolver.transform import normalize
from resolver.transform.adapters.base import CANONICAL_COLUMNS


def _make_raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "ETH-FL-ifrcgo-1",
                "country_name": "Ethiopia",
                "iso3": "eth",
                "hazard_code": "fl",
                "hazard_label": "Flood",
                "hazard_class": "natural",
                "metric": "affected",
                "unit": "persons",
                "series_semantics": "new",
                "value": "1,234",
                "as_of_date": "2025-07-12",
            },
            {
                "event_id": "ETH-DR-ifrcgo-2",
                "country_name": "Ethiopia",
                "iso3": "eth",
                "hazard_code": "dr",
                "hazard_label": "Drought",
                "hazard_class": "natural",
                "metric": "in_need",
                "unit": "persons",
                "series_semantics": "stock",
                "value": "2500 ",
                "as_of_date": "2025-08-01",
            },
            {
                "event_id": "ETH-CY-ifrcgo-3",
                "country_name": "Ethiopia",
                "iso3": "ETH",
                "hazard_code": "CY",
                "hazard_label": "Cyclone",
                "hazard_class": "natural",
                "metric": "affected",
                "unit": "persons",
                "series_semantics": "new",
                "value": "3000",
                "as_of_date": "2025-09-18",
            },
            {  # Outside resolver window
                "event_id": "ETH-FL-ifrcgo-4",
                "country_name": "Ethiopia",
                "iso3": "ETH",
                "hazard_code": "FL",
                "hazard_label": "Flood",
                "hazard_class": "natural",
                "metric": "affected",
                "unit": "persons",
                "series_semantics": "new",
                "value": "4100",
                "as_of_date": "2025-10-01",
            },
            {  # Non-numeric value should drop
                "event_id": "ETH-FL-ifrcgo-5",
                "country_name": "Ethiopia",
                "iso3": "ETH",
                "hazard_code": "FL",
                "hazard_label": "Flood",
                "hazard_class": "natural",
                "metric": "affected",
                "unit": "persons",
                "series_semantics": "new",
                "value": "unknown",
                "as_of_date": "2025-08-15",
            },
        ]
    )


def test_ifrc_normalization_creates_canonical_csv(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "canonical"
    raw_dir.mkdir()
    frame = _make_raw_frame()
    frame.to_csv(raw_dir / "ifrc_go.csv", index=False)

    monkeypatch.setenv("RESOLVER_START_ISO", "2025-07-01")
    monkeypatch.setenv("RESOLVER_END_ISO", "2025-09-30")

    exit_code = normalize.main(
        [
            "--in",
            str(raw_dir),
            "--out",
            str(out_dir),
            "--period",
            "2025Q3",
            "--sources",
            "ifrc_go",
        ]
    )
    assert exit_code == 0

    canonical_path = out_dir / "ifrc_go.csv"
    assert canonical_path.exists()

    canonical = pd.read_csv(canonical_path)
    assert canonical.columns.tolist() == CANONICAL_COLUMNS
    assert len(canonical) == 3

    expected_dates = ["2025-07-31", "2025-08-31", "2025-09-30"]
    assert canonical["as_of_date"].tolist() == expected_dates

    values = canonical["value"].tolist()
    assert values == [1234.0, 2500.0, 3000.0]

    assert canonical["source"].unique().tolist() == ["ifrc_go"]
    assert canonical["series_semantics"].tolist() == ["new", "stock", "new"]
