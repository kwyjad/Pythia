from resolver.common import compute_series_semantics


def test_stock_metric_overrides_blank_existing():
    assert compute_series_semantics("in_need", "") == "stock"
    assert compute_series_semantics("in_need", None) == "stock"


def test_existing_value_preserved():
    assert compute_series_semantics("affected", "incident") == "incident"


def test_blank_metric_defaults_to_empty_string():
    assert compute_series_semantics(None, None) == ""
    assert compute_series_semantics("", "  ") == ""
