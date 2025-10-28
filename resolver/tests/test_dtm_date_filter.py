"""Unit tests for DTM date filter helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from resolver.ingestion import dtm_client


def test_pick_reporting_date_column_is_case_insensitive() -> None:
    frame_primary = pd.DataFrame({"ReportingDate": ["2024-05-01"]})
    assert dtm_client._pick_reporting_date_column(frame_primary) == "ReportingDate"

    frame_camel = pd.DataFrame({"reportingDate": ["2024-05-01"]})
    assert dtm_client._pick_reporting_date_column(frame_camel) == "reportingDate"

    frame_snake = pd.DataFrame({"reporting_date": ["2024-05-01"]})
    assert dtm_client._pick_reporting_date_column(frame_snake) == "reporting_date"

    frame_custom = pd.DataFrame({"ReportDate": ["2024-05-01"]})
    assert (
        dtm_client._pick_reporting_date_column(frame_custom, preferred=("ReportDate",))
        == "ReportDate"
    )


def test_apply_date_window_inclusive_bounds(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "ReportingDate": ["2024-05-01", "2024-05-31", "2024-06-01"],
            "value": [1, 2, 3],
        }
    )
    extras: dict[str, object] = {}

    kept, diag = dtm_client._apply_date_window(
        frame,
        "2024-05-01",
        "2024-05-31",
        disable=False,
        diag_out_dir=tmp_path,
        extras=extras,
        preferred_columns=("ReportingDate",),
    )

    assert len(kept) == 2
    assert diag["inside"] == 2
    assert diag["outside"] == 1
    aggregate = extras.get("date_filter")
    assert isinstance(aggregate, dict)
    assert aggregate["inside"] == 2
    assert aggregate["outside"] == 1
    assert aggregate["parsed_total"] == 3
    assert aggregate["parsed_ok"] == 3


def test_apply_date_window_skipped_when_flag(tmp_path: Path) -> None:
    frame = pd.DataFrame({"ReportingDate": ["2024-05-01"], "value": [10]})
    extras: dict[str, object] = {}

    kept, diag = dtm_client._apply_date_window(
        frame,
        "2024-05-01",
        "2024-05-31",
        disable=True,
        diag_out_dir=tmp_path,
        extras=extras,
        preferred_columns=("ReportingDate",),
    )

    assert kept.equals(frame)
    assert diag["skipped"] is True
    aggregate = extras.get("date_filter")
    assert isinstance(aggregate, dict)
    assert aggregate["skipped"] is True
    assert aggregate["inside"] == 1
    assert aggregate["outside"] == 0


def test_date_diag_counts(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "reportingDate": ["2024-05-15", "2024-06-10", "not a date"],
            "value": [1, 2, 3],
        }
    )
    extras: dict[str, object] = {}

    kept, diag = dtm_client._apply_date_window(
        frame,
        "2024-05-01",
        "2024-05-31",
        disable=False,
        diag_out_dir=tmp_path,
        extras=extras,
        preferred_columns=("reportingDate",),
    )

    assert len(kept) == 1
    assert diag["inside"] == 1
    assert diag["outside"] == 2
    assert diag["parse_failed"] == 1
    assert diag["drop_counts"]["date_out_of_window"] == 1
    assert diag["drop_counts"]["date_parse_failed"] == 1

    sample_path = tmp_path / "normalize_drops.csv"
    assert sample_path.exists()
    sample = pd.read_csv(sample_path)
    assert set(sample["_drop_reason"]) == {"date_out_of_window", "date_parse_failed"}
    assert sample.shape[0] == 2

    aggregate = extras.get("date_filter")
    assert isinstance(aggregate, dict)
    assert aggregate["outside"] >= 2
    assert aggregate["parse_failed"] >= 1
    assert aggregate.get("sample_path") == str(sample_path)
    assert aggregate.get("date_column_used") == "reportingDate"
