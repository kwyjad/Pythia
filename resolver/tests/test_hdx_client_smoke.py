# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import csv

import pytest

from resolver.ingestion import hdx_client


@pytest.mark.allow_network
def test_hdx_client_writes_header_only_csv(tmp_path, monkeypatch):
    staging_root = tmp_path / "staging"
    monkeypatch.setenv("RESOLVER_STAGING_DIR", str(staging_root))
    monkeypatch.setenv("RESOLVER_SKIP_HDX", "1")
    monkeypatch.delenv("RESOLVER_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("RESOLVER_OUTPUT_PATH", raising=False)

    hdx_client.main()

    expected_csv = staging_root / "unknown" / "raw" / "hdx.csv"
    assert expected_csv.exists(), f"missing CSV at {expected_csv}"  # pragma: no cover

    with expected_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    assert rows, "expected header row"  # pragma: no cover
    assert rows[0] == hdx_client.COLUMNS
    assert len(rows) == 1, "expected header-only CSV"
