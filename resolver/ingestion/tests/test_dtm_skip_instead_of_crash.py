from __future__ import annotations

from resolver.ingestion import dtm_client


def test_read_source_missing_id_returns_skip() -> None:
    result = dtm_client._read_source(
        {"name": "no-id"},
        {},
        no_date_filter=False,
        window_start=None,
        window_end=None,
    )
    assert isinstance(result, dtm_client.SourceResult)
    assert result.status == "skipped"
    assert result.skip_reason == "missing id_or_path"
    assert result.rows == 0
    assert result.http_counts == {
        "2xx": 0,
        "4xx": 0,
        "5xx": 0,
        "timeout": 0,
        "error": 0,
    }
