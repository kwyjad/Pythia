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
        }
    )

    assert "2030-01" in dropped
    assert "bad-key" in unparseable
    assert "2025-12" in cleaned
    assert "2025-12-15" in cleaned
    assert "2030-01" not in cleaned
