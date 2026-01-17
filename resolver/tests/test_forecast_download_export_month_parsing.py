import pandas as pd

from resolver.query.downloads import _extract_year_month


def test_extract_year_month_handles_timestamps() -> None:
    series = pd.Series(
        [
            "2024-01",
            "2024-01-01",
            "2024-01-01 00:00:00",
            "not-a-date",
            None,
        ]
    )
    result = _extract_year_month(series)

    assert result.iloc[0] == "2024-01"
    assert result.iloc[1] == "2024-01"
    assert result.iloc[2] == "2024-01"
    assert pd.isna(result.iloc[3])
    assert pd.isna(result.iloc[4])
