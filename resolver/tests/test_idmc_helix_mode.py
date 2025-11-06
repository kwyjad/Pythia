import json
from datetime import date

import pandas as pd
import pytest

from resolver.ingestion.idmc.client import IdmcClient, IdmcConfig


@pytest.fixture
def helix_fetch(monkeypatch, tmp_path):
    payload = [
        {
            "iso3": "AFG",
            "displacement_date": "2024-01-15",
            "figure": 10,
        },
        {
            "iso3": "AGO",
            "displacement_date": "2024-03-20",
            "figure": 5,
        },
    ]
    body = json.dumps(payload).encode("utf-8")
    calls = {"count": 0}

    def fake_http_get(url, **_kwargs):  # noqa: D401 - test stub
        calls["count"] += 1
        diagnostics = {
            "attempts": 1,
            "retries": 0,
            "duration_s": 0.05,
            "backoff_s": 0.0,
            "wire_bytes": len(body),
            "body_bytes": len(body),
            "attempt_durations_s": [0.05],
            "retry_after_s": [],
            "rate_limit_wait_s": [],
            "planned_sleep_s": [],
        }
        return 200, {"Content-Type": "application/json"}, body, diagnostics

    monkeypatch.setattr(
        "resolver.ingestion.idmc.client.http_get", fake_http_get
    )

    cfg = IdmcConfig()
    cfg.api.countries = ["AFG", "AGO"]
    cfg.cache.dir = tmp_path.as_posix()

    client = IdmcClient(helix_client_id="client123")
    data, diagnostics = client.fetch(
        cfg,
        network_mode="helix",
        window_start=date(2024, 1, 1),
        window_end=date(2024, 3, 31),
        chunk_by_month=True,
    )

    return data, diagnostics, calls


def test_idmc_helix_fetch_and_chunking(helix_fetch):
    data, diagnostics, calls = helix_fetch

    assert calls["count"] == 1
    frame = data.get("monthly_flow")
    assert isinstance(frame, pd.DataFrame)
    assert sorted(frame["iso3"].tolist()) == ["AFG", "AGO"]
    assert diagnostics["network_mode"] == "helix"
    assert diagnostics["http"]["requests"] == 1
    assert diagnostics["helix"]["status"] == 200
    assert diagnostics["http_status_counts"] == {"2xx": 1, "4xx": 0, "5xx": 0}
    assert "client123" not in diagnostics["helix"].get("url", "")


def test_idmc_no_hdx_residuals(helix_fetch):
    _data, diagnostics, _calls = helix_fetch
    serialized = json.dumps(diagnostics, default=str).lower()
    assert "\"type\": \"hdx\"" not in serialized
    assert "hdx_attempt" not in serialized


def test_idmc_three_key_http_counts(helix_fetch):
    _data, diagnostics, _calls = helix_fetch
    assert set(diagnostics["http_status_counts"].keys()) == {"2xx", "4xx", "5xx"}
