# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Offline cache behaviour for the IDMC connector."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from resolver.ingestion.idmc import cache, client, config


def test_fetch_uses_cache_when_forced(monkeypatch, tmp_path):
    cache_dir = tmp_path / "idmc_cache"
    raw_dir = tmp_path / "diagnostics"
    monkeypatch.setenv("IDMC_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("IDMC_FORCE_CACHE_ONLY", "1")
    monkeypatch.setenv("IDMC_CACHE_TTL_S", "0")
    monkeypatch.setattr(client, "RAW_DIAG_DIR", str(raw_dir))

    cfg = config.load()
    cfg.cache.dir = str(cache_dir)
    cfg.cache.ttl_seconds = 0
    cfg.cache.force_cache_only = True
    endpoint = cfg.api.endpoints["idus_json"]
    url = f"{cfg.api.base_url.rstrip('/')}{endpoint}"
    key = cache.cache_key(url, params={"chunk": "full"})

    fixture_path = Path(client.FIXTURES_DIR) / "idus_view_flat.json"
    payload_bytes = fixture_path.read_bytes()
    cache.cache_put(str(cache_dir), key, payload_bytes, {"source": "fixture"})

    data, diagnostics = client.fetch(
        cfg,
        network_mode="cache_only",
        window_days=None,
        only_countries=["SDN"],
    )

    frame = data["monthly_flow"]
    assert isinstance(frame, pd.DataFrame)
    assert list(frame["iso3"]) == ["SDN"]
    assert frame.iloc[0]["figure"] == 800

    assert diagnostics["mode"] == "cache"
    assert diagnostics["network_mode"] == "cache_only"
    assert diagnostics["http"]["requests"] == 0
    assert diagnostics["http_status_counts"] is None
    assert diagnostics["cache"]["hits"] == 1
    assert diagnostics["cache"]["misses"] == 0
    assert diagnostics["filters"]["countries"] == ["SDN"]
    assert diagnostics["filters"]["rows_after"] == 1
    assert Path(diagnostics["raw_path"]).is_file()
