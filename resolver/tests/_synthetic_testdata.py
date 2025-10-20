"""Synthetic fixtures for running resolver tests without large datasets.

This module builds tiny, schema-compliant DataFrames and writes them to a
filesystem layout that mirrors the real resolver exports.  The helpers are
intentionally lightweight so they can run inside Codex without the external
fixtures that GitHub Actions uses.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

_FACTS_RESOLVED_COLUMNS = [
    "event_id",
    "country_name",
    "iso3",
    "hazard_code",
    "hazard_label",
    "hazard_class",
    "metric",
    "value",
    "unit",
    "as_of_date",
    "publication_date",
    "publisher",
    "source_type",
    "source_url",
    "doc_title",
    "definition_text",
    "method",
    "confidence",
    "provenance_source",
    "provenance_rank",
    "revision",
    "ingested_at",
    "ym",
    "series_semantics",
]

_FACTS_DELTAS_COLUMNS = [
    "ym",
    "iso3",
    "hazard_code",
    "metric",
    "value_new",
    "value_stock",
    "series_semantics",
    "series_semantics_out",
    "rebase_flag",
    "first_observation",
    "delta_negative_clamped",
    "as_of",
    "source_id",
    "source_url",
    "definition_text",
]

_COUNTRIES_ROWS = [
    {"country_name": "Philippines", "iso3": "PHL"},
    {"country_name": "Ethiopia", "iso3": "ETH"},
]

_HAZARDS_ROWS = [
    {
        "hazard_label": "Tropical Cyclone",
        "hazard_code": "TC",
        "hazard_class": "natural",
    },
    {
        "hazard_label": "Drought",
        "hazard_code": "DR",
        "hazard_class": "natural",
    },
    {
        "hazard_label": "Conflict Onset",
        "hazard_code": "ACO",
        "hazard_class": "human-induced",
    },
    {
        "hazard_label": "Conflict Escalation",
        "hazard_code": "ACE",
        "hazard_class": "human-induced",
    },
]


def make_facts_resolved_df() -> pd.DataFrame:
    """Return a tiny facts_resolved-style DataFrame.

    The values are deliberately simple but respect the canonical schema so
    downstream validation tests exercise the real code paths.
    """

    rows = [
        {
            "event_id": "EVT-1",
            "country_name": "Philippines",
            "iso3": "PHL",
            "hazard_code": "TC",
            "hazard_label": "Tropical Cyclone",
            "hazard_class": "natural",
            "metric": "in_need",
            "value": "1500",
            "unit": "persons",
            "as_of_date": "2024-01-31",
            "publication_date": "2024-02-02",
            "publisher": "OCHA",
            "source_type": "situation_report",
            "source_url": "https://example.org/report-a",
            "doc_title": "Synthetic SitRep",
            "definition_text": "People in need",
            "method": "reported",
            "confidence": "medium",
            "provenance_source": "OCHA",
            "provenance_rank": 1,
            "revision": "1",
            "ingested_at": "2024-02-02T00:00:00Z",
            "ym": "2024-01",
            "series_semantics": "stock",
        },
        {
            "event_id": "EVT-2",
            "country_name": "Ethiopia",
            "iso3": "ETH",
            "hazard_code": "DR",
            "hazard_label": "Drought",
            "hazard_class": "natural",
            "metric": "affected",
            "value": "500",
            "unit": "persons",
            "as_of_date": "2024-03-10",
            "publication_date": "2024-03-12",
            "publisher": "OCHA",
            "source_type": "situation_report",
            "source_url": "https://example.org/report-b",
            "doc_title": "Synthetic Report",
            "definition_text": "People affected",
            "method": "reported",
            "confidence": "high",
            "provenance_source": "OCHA",
            "provenance_rank": 1,
            "revision": "1",
            "ingested_at": "2024-03-13T00:00:00Z",
            "ym": "2024-03",
            "series_semantics": "stock",
        },
    ]
    df = pd.DataFrame(rows, columns=_FACTS_RESOLVED_COLUMNS)
    assert list(df.columns) == _FACTS_RESOLVED_COLUMNS, "facts_resolved columns drifted"
    return df


def make_facts_deltas_df() -> pd.DataFrame:
    """Return a tiny facts_deltas-style DataFrame."""

    rows = [
        {
            "ym": "2024-01",
            "iso3": "PHL",
            "hazard_code": "TC",
            "metric": "in_need",
            "value_new": "500",
            "value_stock": "1500",
            "series_semantics": "new",
            "series_semantics_out": "new",
            "rebase_flag": 0,
            "first_observation": 1,
            "delta_negative_clamped": 0,
            "as_of": "2024-01-31",
            "source_id": "OCHA",
            "source_url": "https://example.org/report-a",
            "definition_text": "Synthetic delta",
        },
        {
            "ym": "2024-03",
            "iso3": "ETH",
            "hazard_code": "DR",
            "metric": "affected",
            "value_new": "500",
            "value_stock": "500",
            "series_semantics": "new",
            "series_semantics_out": "new",
            "rebase_flag": 0,
            "first_observation": 0,
            "delta_negative_clamped": 0,
            "as_of": "2024-03-10",
            "source_id": "OCHA",
            "source_url": "https://example.org/report-b",
            "definition_text": "Synthetic delta",
        },
    ]
    df = pd.DataFrame(rows, columns=_FACTS_DELTAS_COLUMNS)
    assert list(df.columns) == _FACTS_DELTAS_COLUMNS, "facts_deltas columns drifted"
    return df


def write_csv(dirpath: Path) -> Dict[str, Path]:
    """Write the synthetic fixtures to ``dirpath`` and return a manifest.

    The manifest maps a short descriptive key to the file that was written.
    This doubles as a tiny self-check so tests can assert the files exist.
    """

    dirpath.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Path] = {}

    resolved = make_facts_resolved_df()
    deltas = make_facts_deltas_df()

    resolved_path = dirpath / "facts_resolved_min.csv"
    deltas_path = dirpath / "facts_deltas_min.csv"
    resolved.to_csv(resolved_path, index=False)
    deltas.to_csv(deltas_path, index=False)
    manifest["facts_resolved_csv"] = resolved_path
    manifest["facts_deltas_csv"] = deltas_path

    exports_dir = dirpath / "exports"
    exports_dir.mkdir(exist_ok=True)
    resolved.to_csv(exports_dir / "facts.csv", index=False)
    resolved.to_csv(exports_dir / "resolved.csv", index=False)
    deltas.to_csv(exports_dir / "deltas.csv", index=False)
    manifest["exports_dir"] = exports_dir

    data_dir = dirpath / "data"
    data_dir.mkdir(exist_ok=True)
    countries = pd.DataFrame(_COUNTRIES_ROWS, columns=["country_name", "iso3"])
    hazards = pd.DataFrame(
        _HAZARDS_ROWS, columns=["hazard_label", "hazard_code", "hazard_class"]
    )
    countries_path = data_dir / "countries.csv"
    hazards_path = data_dir / "shocks.csv"
    countries.to_csv(countries_path, index=False)
    hazards.to_csv(hazards_path, index=False)
    manifest["countries_csv"] = countries_path
    manifest["shocks_csv"] = hazards_path

    # Stub directories that parity/state tests probe. They may remain empty but
    # the presence of the directories keeps path discovery logic satisfied.
    (dirpath / "snapshots").mkdir(exist_ok=True)
    (dirpath / "state").mkdir(exist_ok=True)
    (dirpath / "review").mkdir(exist_ok=True)

    return manifest

