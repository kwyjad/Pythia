"""Unit tests for `_is_candidate_newer` helper."""

from resolver.ingestion import dtm_client


def test_candidate_with_newer_as_of_wins() -> None:
    assert dtm_client._is_candidate_newer("2024-01-01", "2024-02-01") is True
    assert dtm_client._is_candidate_newer("2024-02-01", "2024-01-01") is False


def test_empty_existing_accepts_candidate() -> None:
    assert dtm_client._is_candidate_newer("", "2024-03-01") is True
    assert dtm_client._is_candidate_newer("", "") is False


def test_equal_as_of_does_not_replace() -> None:
    assert dtm_client._is_candidate_newer("2024-03-01", "2024-03-01") is False
