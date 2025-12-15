# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from resolver.ingestion.idmc.diagnostics import serialize_http_status_counts


def test_http_status_counts_shape_live_only_three_keys():
    counts = {
        "2xx": 3,
        "4xx": 1,
        "5xx": 2,
        "other": 4,
        "timeouts": 5,
    }

    serialized = serialize_http_status_counts(counts)

    assert serialized == {"2xx": 3, "4xx": 1, "5xx": 2}
    assert set(serialized.keys()) == {"2xx", "4xx", "5xx"}


def test_http_status_counts_shape_fixture_cache_only_zeroes():
    assert serialize_http_status_counts(None) == {"2xx": 0, "4xx": 0, "5xx": 0}

    source = {"2xx": None, "4xx": "", "5xx": "7"}
    serialized = serialize_http_status_counts(source)

    assert serialized == {"2xx": 0, "4xx": 0, "5xx": 7}
