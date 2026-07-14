# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Fixture tests for the NIO climatology context builder (no network).

IMD/RSMC New Delhi publishes no seasonal TC count outlook, so the NIO block
is climatology-first: it must ALWAYS render, live enrichment failures must
be contained, and Optional fields must never be string-formatted."""

from __future__ import annotations

from horizon_scanner.seasonal_tc import imd_nio_scraper as nio
from horizon_scanner.seasonal_tc.imd_nio_scraper import (
    SeasonalForecast,
    _parse_ero_filename_date,
    build_nio_context,
)


def test_climatology_always_renders():
    forecasts = build_nio_context(fetch_live=False)
    assert len(forecasts) == 1
    f = forecasts[0]
    assert f.basin == "NIO"
    ctx = f.to_prompt_context()
    assert "Bay of Bengal" in ctx
    assert "Arabian Sea" in ctx
    assert "pre-monsoon" in f.season and "post-monsoon" in f.season
    assert "not a" in ctx and "seasonal count forecast" in ctx


def test_live_failure_falls_back_to_climatology(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(nio, "fetch_latest_ero_pdf_url", boom)
    monkeypatch.setattr(nio, "_load_enso_context", boom)
    forecasts = build_nio_context(fetch_live=True)
    assert len(forecasts) == 1
    f = forecasts[0]
    assert f.extended_range_note is None
    ctx = f.to_prompt_context()
    assert "Bay of Bengal" in ctx


def test_optional_fields_none_safe():
    f = SeasonalForecast(
        climatology_note="note", extended_range_note=None, enso_context=None
    )
    ctx = f.to_prompt_context()  # must not raise or render None
    assert "None" not in ctx


def test_enrichments_render_when_present():
    f = SeasonalForecast(
        climatology_note="note",
        extended_range_note="issued 2026-07-09 (next ~2 weeks): Bay of Bengal cyclogenesis probability: low.",
        enso_context="Neutral",
    )
    ctx = f.to_prompt_context()
    assert "extended-range cyclogenesis outlook" in ctx
    assert "ENSO context: Neutral" in ctx


def test_ero_filename_date_parsing():
    d = _parse_ero_filename_date(
        "/uploads/archive/24/24_d9c866_Extended_Range_Outlook_11June2026.pdf"
    )
    assert d is not None and (d.year, d.month, d.day) == (2026, 6, 11)
    d = _parse_ero_filename_date(
        "/uploads/archive/24/24_b009b9_Extended%20Range%20Outlook_05Sep2024.pdf"
    )
    assert d is not None and (d.year, d.month, d.day) == (2024, 9, 5)
    assert _parse_ero_filename_date("/uploads/whatever.pdf") is None


def test_nio_basin_fanout_contract():
    """COUNTRY_TO_BASINS must still map the NIO/SWI countries this feature
    exists for — the runner fan-out keys on basin membership."""
    from horizon_scanner.seasonal_tc import COUNTRY_TO_BASINS

    assert "NIO" in COUNTRY_TO_BASINS.get("BGD", [])
    assert "NIO" in COUNTRY_TO_BASINS.get("SOM", [])
    assert "SWI" in COUNTRY_TO_BASINS.get("MDG", [])
    assert "SWI" in COUNTRY_TO_BASINS.get("MOZ", [])
