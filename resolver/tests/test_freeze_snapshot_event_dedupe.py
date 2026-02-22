# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import pandas as pd
import pytest

from resolver.tools import freeze_snapshot

pytestmark = [
    pytest.mark.legacy_freeze,
]


def test_dedupe_preserves_event_level_rows():
    base_rows = [
        {
            "ym": "2024-02",
            "iso3": "CCC",
            "hazard_code": "FL",
            "metric": "affected",
            "event_id": f"ev-{idx}",
            "value": 10 * idx,
            "as_of_date": "2024-02-29",
            "publication_date": "2024-03-05",
            "source_id": "test",
        }
        for idx in range(1, 5)
    ]
    frame = pd.DataFrame(base_rows)

    deduped, diag = freeze_snapshot._dedupe_snapshot_frame(  # type: ignore[attr-defined]
        frame,
        table="facts_deltas",
        base_keys=["iso3", "ym", "hazard_code", "metric"],
    )

    assert deduped is not None
    assert len(deduped) == 4
    assert diag is not None
    assert diag.event_id_included is True
    assert "event_id" in diag.keys_used
    assert diag.rows_before == 4
    assert diag.rows_after == 4
