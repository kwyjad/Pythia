from __future__ import annotations

import pandas as pd

from resolver.tools import llm_context


def test_build_context_frame_empty_months_returns_dataframe_with_schema() -> None:
    df = llm_context.build_context_frame(months=[])
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == llm_context.CONTEXT_COLUMNS
    assert df.empty


def test_build_context_frame_minimal_shape(monkeypatch) -> None:
    class _Selectors:
        @staticmethod
        def load_series_for_month(ym, requested_series, backend, is_current_month):  # noqa: D401
            return pd.DataFrame(
                {
                    "iso3": ["PHL", "PHL"],
                    "hazard_code": ["TC", "TC"],
                    "metric": ["in_need", "affected"],
                    "unit": ["persons", "persons"],
                    "value": [100, 250],
                }
            )

    monkeypatch.setattr(llm_context, "selectors", _Selectors)
    df = llm_context.build_context_frame(months=["2025-01"])
    assert not df.empty
    assert list(df.columns) == llm_context.CONTEXT_COLUMNS
    assert set(df["series"].unique()) == {"new"}
    assert set(df["ym"].unique()) == {"2025-01"}
