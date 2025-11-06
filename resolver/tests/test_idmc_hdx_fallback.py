from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from resolver.ingestion.idmc.client import _hdx_dataset_slug, fetch
from resolver.ingestion.idmc.config import IdmcConfig
from resolver.ingestion.idmc.http import HttpRequestError


def test_idmc_hdx_slug_default() -> None:
    assert _hdx_dataset_slug() == "preliminary-internal-displacement-updates"


def test_idmc_single_shot_fallback_filters_per_chunk(tmp_path: Path, monkeypatch) -> None:
    cfg = IdmcConfig()
    cfg.api.countries = ["AFG"]
    cfg.api.series = ["flow"]
    cfg.cache.dir = (tmp_path / "cache").as_posix()
    Path(cfg.cache.dir).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "resolver.ingestion.idmc.client._probe_idu_schema",
        lambda *args, **kwargs: (
            "displacement_date",
            ["iso3", "figure", "displacement_date"],
            {"status": 200},
        ),
    )

    def _failing_http_get(*_args, **_kwargs):
        diagnostics = {
            "attempts": 1,
            "retries": 0,
            "duration_s": 0.01,
            "backoff_s": 0.0,
            "wire_bytes": 0,
            "body_bytes": 0,
            "status": None,
        }
        raise HttpRequestError("boom", diagnostics, kind="dns_error")

    monkeypatch.setattr(
        "resolver.ingestion.idmc.client.http_get",
        _failing_http_get,
    )

    fallback_frame = pd.DataFrame(
        [
            {"CountryISO3": "AFG", "displacement_date": "2024-01-15", "figure": 10},
            {"CountryISO3": "AFG", "displacement_date": "2024-02-10", "figure": 20},
            {"CountryISO3": "BDI", "displacement_date": "2024-01-20", "figure": 99},
            {"CountryISO3": "AFG", "displacement_date": "2023-12-31", "figure": 5},
        ]
    )
    fallback_diag = {
        "dataset": "preliminary-internal-displacement-updates",
        "resource_url": "https://data.humdata.org/idus_view_flat.csv",
        "resource_status_code": 200,
    }
    fallback_fetch = MagicMock(return_value=(fallback_frame, fallback_diag))
    monkeypatch.setattr(
        "resolver.ingestion.idmc.client._hdx_fetch_latest_csv",
        fallback_fetch,
    )

    data, diagnostics = fetch(
        cfg,
        network_mode="live",
        window_start=date(2024, 1, 1),
        window_end=date(2024, 2, 29),
        chunk_by_month=True,
        allow_hdx_fallback=True,
    )

    flow = data["monthly_flow"]
    assert len(flow) == 2
    assert set(flow.columns) == {
        "iso3",
        "as_of_date",
        "metric",
        "value",
        "series_semantics",
        "source",
        "__hdx_preaggregated__",
    }
    as_of_dates = sorted(flow["as_of_date"].astype(str).tolist())
    assert as_of_dates == ["2024-01-31", "2024-02-29"]
    values = sorted(flow["value"].tolist())
    assert values == [10, 20]
    assert flow["metric"].unique().tolist() == ["new_displacements"]

    attempts = diagnostics.get("attempts") or []
    chunk_rows = [entry.get("rows") for entry in attempts if entry.get("via") == "hdx_fallback"]
    assert chunk_rows == [1, 1]

    fallback_info = diagnostics.get("fallback") or {}
    assert fallback_info.get("rows") == 2
    assert fallback_info.get("resource_url") == fallback_diag["resource_url"]

    assert fallback_fetch.call_count == 1
