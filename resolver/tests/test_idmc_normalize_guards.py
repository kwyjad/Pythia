"""Guard-rail tests for the IDMC normalizer."""

import pandas as pd

from resolver.ingestion.idmc.normalize import _normalize_monthly_flow


def _aliases() -> dict:
    return {"value_flow": [], "value_stock": [], "date": [], "iso3": ["iso3"]}


def test_idmc_normalize_flow_empty_input_safe(caplog):
    frame = pd.DataFrame(columns=["iso3", "figure", "displacement_date"])

    with caplog.at_level("DEBUG"):
        normalized, drops = _normalize_monthly_flow(
            frame,
            _aliases(),
            {"start": "2020-01-01", "end": "2020-12-31"},
        )

    assert normalized.empty
    assert drops["date_parse_failed"] == 0
    assert "normalize_flow: empty input" in caplog.text


def test_idmc_normalize_flow_all_nat_dates_safe(caplog):
    frame = pd.DataFrame(
        {
            "iso3": ["COL", "PER"],
            "figure": [10, 20],
            "displacement_date": ["not-a-date", "still-not-a-date"],
        }
    )

    with caplog.at_level("DEBUG"):
        normalized, drops = _normalize_monthly_flow(
            frame,
            _aliases(),
            {"start": "2020-01-01", "end": "2020-12-31"},
        )

    assert normalized.empty
    assert drops["date_parse_failed"] == 2
    assert "normalize_flow: all dates NaT after coercion" in caplog.text


def test_idmc_normalize_flow_datetime_comparison_ok():
    frame = pd.DataFrame(
        {
            "iso3": ["COL", "PER"],
            "figure": [10, 20],
            "displacement_date": ["2024-01-10", "2024-04-01"],
        }
    )

    normalized, drops = _normalize_monthly_flow(
        frame,
        _aliases(),
        {"start": "2024-02-01", "end": "2024-12-31"},
    )

    assert list(normalized["iso3"]) == ["PER"]
    assert drops["date_out_of_window"] == 1
    assert pd.api.types.is_datetime64_any_dtype(normalized["as_of_date"])  # noqa: PD013


def test_idmc_as_of_date_is_datetime_after_group():
    frame = pd.DataFrame(
        {
            "iso3": ["COL", "COL"],
            "figure": [5, 7],
            "displacement_date": ["2024-01-02", "2024-01-15"],
        }
    )

    normalized, drops = _normalize_monthly_flow(
        frame,
        _aliases(),
        {"start": "2023-12-01", "end": "2024-12-31"},
    )

    assert drops["date_out_of_window"] == 0
    assert len(normalized) == 1
    assert pd.api.types.is_datetime64_any_dtype(normalized["as_of_date"])  # noqa: PD013
    assert normalized.loc[0, "as_of_date"] == pd.Timestamp("2024-01-31")


def test_idmc_as_of_date_dtype_survives_noop_window():
    frame = pd.DataFrame(
        {
            "iso3": ["COL", "PER"],
            "figure": [11, 13],
            "displacement_date": ["2024-05-01", "2024-06-15"],
        }
    )

    normalized, _ = _normalize_monthly_flow(
        frame,
        _aliases(),
        {"start": "2024-01-01", "end": "2024-12-31"},
    )

    assert not normalized.empty
    assert pd.api.types.is_datetime64_any_dtype(normalized["as_of_date"])  # noqa: PD013
    assert list(normalized["as_of_date"]) == [pd.Timestamp("2024-05-31"), pd.Timestamp("2024-06-30")]
