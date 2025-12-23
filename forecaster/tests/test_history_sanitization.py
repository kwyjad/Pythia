import pytest

duckdb = pytest.importorskip("duckdb")

from forecaster import cli


def test_sanitize_month_series_drops_future_and_keeps_valid():
    cleaned, dropped, unparseable = cli._sanitize_month_series(
        {
            "2030-01": 1,
            "2025-12": 2,
            "2025-12-15": 3,
            "bad-key": 4,
            "": 5,
        }
    )

    assert "2030-01" in dropped
    assert {"bad-key", ""}.issubset(set(unparseable))
    assert "2025-12" in cleaned
    assert "2025-12-15" in cleaned
    assert "2030-01" not in cleaned
    assert "bad-key" not in cleaned
    assert "" not in cleaned


def test_sanitize_month_series_skips_unparseable_but_reports():
    cleaned, dropped, unparseable = cli._sanitize_month_series({"not-a-month": 1, "2024-05": 2})

    assert cleaned == {"2024-05": 2}
    assert dropped == []
    assert unparseable == ["not-a-month"]
