# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import json
from pathlib import Path

import pandas as pd

from resolver.ingestion.idmc.export import build_resolution_ready_facts


def test_drop_missing_ym_writes_diagnostics(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    normalized = pd.DataFrame(
        [
            {
                "iso3": "AFG",
                "as_of_date": "",
                "metric": "new_displacements",
                "value": 5,
                "series_semantics": "new",
                "source": "idmc_idu",
                "ym": "",
                "record_id": "rec-1",
            }
        ]
    )

    facts = build_resolution_ready_facts(normalized)

    assert facts.empty
    diag_path = Path("diagnostics/idmc_month_drop.json")
    assert diag_path.exists()
    payload = json.loads(diag_path.read_text(encoding="utf-8"))
    assert payload["dropped_count"] == 1
    assert payload["examples"]


def test_valid_date_derives_ym(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    normalized = pd.DataFrame(
        [
            {
                "iso3": "AFG",
                "as_of_date": "2024-02-29",
                "metric": "new_displacements",
                "value": 10,
                "series_semantics": "new",
                "source": "idmc_idu",
                "ym": "",
                "record_id": "",
            }
        ]
    )

    facts = build_resolution_ready_facts(normalized)

    assert facts["ym"].tolist() == ["2024-02"]
    assert facts["record_id"].astype(str).str.strip().tolist()[0]
