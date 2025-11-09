import json
from datetime import date
from pathlib import Path
from typing import Any, Dict

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

from resolver.ingestion.idmc import client, config


@pytest.fixture
def cfg(tmp_path: Path) -> config.IdmcConfig:
    cfg_obj = config.load()
    cfg_obj.cache.dir = str(tmp_path / "cache")
    cfg_obj.cache.ttl_seconds = 0
    cfg_obj.api.countries = ["AFG", "UKR"]
    return cfg_obj


def _load_last180_fixture() -> pd.DataFrame:
    fixture_path = (
        Path(__file__).resolve().parent.parent
        / "idmc"
        / "fixtures"
        / "helix_last180.json"
    )
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(payload)
    if "value" not in frame.columns and "new_displacements" in frame.columns:
        frame = frame.rename(columns={"new_displacements": "value"})
    return frame


def test_gidd_404_then_idus_last180_200_writes_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idus_frame = _load_last180_fixture()

    def fake_gidd(*args: Any, **kwargs: Any) -> tuple[pd.DataFrame, Dict[str, Any]]:
        return pd.DataFrame(), {
            "status": 404,
            "raw_rows": 0,
            "zero_rows_reason": "helix_http_error",
        }

    def fake_idus(*args: Any, **kwargs: Any) -> tuple[pd.DataFrame, Dict[str, Any]]:
        return idus_frame.copy(), {"status": 200, "bytes": 1024}

    monkeypatch.setattr(client, "_fetch_helix_last180", fake_gidd)
    monkeypatch.setattr(client, "_fetch_helix_idus_last180", fake_idus)

    frame, diagnostics = client._fetch_helix_chain(
        "demo-client",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 31),
        iso3_list=["AFG", "UKR"],
    )

    assert diagnostics.get("helix_endpoint") == "idus_last180"
    assert diagnostics.get("raw_rows") == int(idus_frame.shape[0])
    assert diagnostics.get("rows") == int(frame.shape[0])
    assert diagnostics.get("normalized_rows") == int(frame.shape[0])
    assert not diagnostics.get("zero_rows_reason")
    attempts = diagnostics.get("helix_attempts") or {}
    assert attempts.get("gidd", {}).get("status") == 404
    assert attempts.get("idus_last180", {}).get("status") == 200
    assert {code.upper() for code in frame["iso3"].unique()} <= {"AFG", "UKR"}


def test_helix_mode_disables_chunking(
    monkeypatch: pytest.MonkeyPatch, cfg: config.IdmcConfig
) -> None:
    helix_frame = pd.DataFrame(
        {
            "iso3": ["AFG", "AFG"],
            "displacement_date": ["2024-01-15", "2024-01-31"],
            "figure": [50, 75],
        }
    )

    def fake_split(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("split_by_month should not be used in helix mode")

    def fake_chain(*args: Any, **kwargs: Any) -> tuple[pd.DataFrame, Dict[str, Any]]:
        return helix_frame.copy(), {
            "status": 200,
            "rows": helix_frame.shape[0],
            "raw_rows": helix_frame.shape[0],
            "helix_endpoint": "gidd",
        }

    monkeypatch.setattr(client, "split_by_month", fake_split)
    monkeypatch.setattr(client, "_fetch_helix_chain", fake_chain)

    data, diagnostics = client.fetch(
        cfg,
        network_mode="helix",
        window_start=date(2024, 1, 1),
        window_end=date(2024, 1, 31),
        chunk_by_month=True,
    )

    assert diagnostics.get("helix_endpoint") == "gidd"
    chunks_block = diagnostics.get("chunks") or {}
    assert not chunks_block.get("enabled")
    assert chunks_block.get("count") == 1
    assert int(diagnostics.get("rows_fetched", 0)) >= helix_frame.shape[0]
    assert data["monthly_flow"].shape[0] == helix_frame.shape[0]


def test_helix_both_empty_sets_zero_rows_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_frame = pd.DataFrame(columns=["iso3", "displacement_date", "figure"])

    def fake_gidd(*args: Any, **kwargs: Any) -> tuple[pd.DataFrame, Dict[str, Any]]:
        return empty_frame.copy(), {
            "status": 200,
            "raw_rows": 0,
            "rows": 0,
        }

    def fake_idus(*args: Any, **kwargs: Any) -> tuple[pd.DataFrame, Dict[str, Any]]:
        return empty_frame.copy(), {"status": 200, "bytes": 0}

    monkeypatch.setattr(client, "_fetch_helix_last180", fake_gidd)
    monkeypatch.setattr(client, "_fetch_helix_idus_last180", fake_idus)

    frame, diagnostics = client._fetch_helix_chain(
        "demo-client",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        iso3_list=["AFG"],
    )

    assert frame.empty
    assert diagnostics.get("helix_endpoint") == "idus_last180"
    assert diagnostics.get("zero_rows_reason") == "helix_empty"
    assert diagnostics.get("normalized_rows") == 0
    attempts = diagnostics.get("helix_attempts") or {}
    assert "gidd" in attempts and "idus_last180" in attempts
