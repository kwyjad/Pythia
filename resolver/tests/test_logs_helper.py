from __future__ import annotations
import pandas as pd

from resolver.common import get_logger, dict_counts, df_schema


def test_get_logger_is_cached():
    logger_a = get_logger("resolver.test")
    logger_b = get_logger("resolver.test")
    assert logger_a is logger_b
    handler_count = len(logger_a.handlers)
    # Re-fetch to ensure handlers are not duplicated
    logger_c = get_logger("resolver.test")
    assert len(logger_c.handlers) == handler_count
    assert logger_c.level == logger_a.level


def test_dict_counts_handles_edge_cases():
    assert dict_counts(None) == {}
    empty_series = pd.Series([], dtype=object)
    assert dict_counts(empty_series) == {}
    series = pd.Series(["stock", "", None])
    counts = dict_counts(series)
    assert counts.get("stock") == 1
    assert counts.get("") == 2


def test_df_schema_metadata():
    df = pd.DataFrame({"value": [1, 2], "label": ["a", "b"]})
    schema = df_schema(df)
    assert schema["rows"] == 2
    assert schema["columns"] == ["value", "label"]
    assert schema["dtypes"]["value"] in {"int64", "Int64"}
