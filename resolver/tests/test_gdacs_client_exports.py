# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Ensure the public API exposed by ``gdacs_client`` stays stable."""

from resolver.ingestion.gdacs_client import (
    GDACSEvent,
    _aggregate_final_rows,
    allocate_event,
    allocate_value,
    dedupe_monthly_rows,
    hazard_from_key,
    load_config,
    map_hazard,
)


def test_gdacs_client_exports_are_available():
    assert callable(allocate_event)
    assert callable(allocate_value)
    assert callable(hazard_from_key)
    assert callable(map_hazard)
    assert callable(load_config)
    assert callable(dedupe_monthly_rows)
    assert callable(_aggregate_final_rows)
    assert GDACSEvent.__name__ == "GDACSEvent"
