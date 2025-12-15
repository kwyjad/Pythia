# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import pandas as pd

from resolver.helpers.series_semantics import normalize_series_semantics


def test_series_semantics_normalization_preserves_known_variants():
    frame = pd.DataFrame(
        {
            "series_semantics": [
                "stock estimate",
                "Stock",
                " new  ",
                "stock_est.",
                "unknown",
            ]
        }
    )

    normalised = normalize_series_semantics(frame)
    assert normalised["series_semantics"].tolist() == [
        "stock_estimate",
        "stock",
        "new",
        "stock_estimate",
        "unknown",
    ]


def test_series_semantics_normalization_is_copy():
    frame = pd.DataFrame({"series_semantics": ["stock"]})
    _ = normalize_series_semantics(frame)
    assert frame["series_semantics"].tolist() == ["stock"]
