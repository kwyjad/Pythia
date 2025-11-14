"""Unit tests for resolver.ingestion.dtm_client.CANONICAL_COLUMNS."""

from resolver.ingestion.dtm_client import CANONICAL_COLUMNS


def test_canonical_columns_non_empty_and_strings() -> None:
    assert isinstance(CANONICAL_COLUMNS, list)
    assert CANONICAL_COLUMNS, "CANONICAL_COLUMNS should not be empty"
    for column in CANONICAL_COLUMNS:
        assert isinstance(column, str), "Each canonical column should be a string"
        assert column, "Canonical column names should be non-empty"


def test_canonical_columns_core_fields_present() -> None:
    required = {"iso3", "as_of_date", "ym", "metric", "value"}
    assert required.issubset(set(CANONICAL_COLUMNS))
