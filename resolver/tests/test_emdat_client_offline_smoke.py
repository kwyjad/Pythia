# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

from unittest import mock

import pytest

from resolver.ingestion import emdat_stub
from resolver.ingestion.emdat_client import EmdatClient, OfflineRequested


EXPECTED_COLUMNS = {
    "disno",
    "classif_key",
    "type",
    "subtype",
    "iso",
    "country",
    "start_year",
    "start_month",
    "start_day",
    "end_year",
    "end_month",
    "end_day",
    "total_affected",
    "entry_date",
    "last_update",
}


def test_offline_defaults_use_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EMDAT_API_KEY", raising=False)
    monkeypatch.delenv("EMDAT_NETWORK", raising=False)

    client = EmdatClient()

    with mock.patch.object(client.session, "post", side_effect=AssertionError("network should be disabled")):
        with pytest.raises(OfflineRequested):
            client.fetch_raw(2018, 2019)

    stub_frame = emdat_stub.fetch_raw(2018, 2019)
    assert not stub_frame.empty
    assert EXPECTED_COLUMNS.issubset(set(stub_frame.columns))
