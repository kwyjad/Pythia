from pathlib import Path

import pytest

from resolver.ingestion.idmc import cli as idmc_cli
from resolver.ingestion.idmc.diagnostics import serialize_http_status_counts


def _summary_context(**overrides):
    base = {
        "timestamp": "2024-01-01T00:00:00+00:00",
        "git_sha": "abc123",
        "config_source": "test",
        "config_path": "/tmp/config.yml",
        "mode": "live",
        "token_status": "present",
        "series_display": "flow",
        "date_window": "start=2023-01-01, end=2023-12-31",
        "countries_count": 3,
        "countries_sample_display": "AAA, BBB, CCC",
        "countries_source_display": "config",
        "endpoints_block": "- gidd: https://example",
        "reachability_block": "- Base URL reachable",
        "http_attempts_block": "- Requests attempted: 1",
        "attempts_block": "- Retry-after events: 0",
        "chunk_attempts_block": "- chunk=full; status=200",
        "performance_block": "- Rows fetched: 10\n- Rows normalized: 10\n- Rows written: 10",
        "rate_limit_block": "- Rate limit (req/s): 0.5",
        "fallback_block": "- (none)",
        "zero_rows_reason": "n/a",
        "dataset_block": "- idmc_flow.csv",
        "duckdb_block": "- DuckDB write: not requested",
        "outputs_block": "- Staged flow.csv: rows=10",
        "staging_block": "- resolver/staging/idmc/flow.csv: present",
        "notes_block": "- Network mode: live",
        "helix_block": None,
        "helix_last180": None,
        "fallback_details": None,
        "rows_fetched_total": 10,
        "rows_normalized_total": 10,
        "rows_written_total": 10,
        "staged_counts": {"flow.csv": 10},
        "export_details": {"enabled": False},
    }
    base.update(overrides)
    return base


def test_serialize_http_status_counts_clamps_buckets():
    counts = serialize_http_status_counts({"2xx": 3, "4xx": 1, "timeout": 7, "other": 9})

    assert counts == {"2xx": 3, "4xx": 1, "5xx": 0}


def test_serialize_http_status_counts_handles_none():
    counts = serialize_http_status_counts(None)

    assert counts == {"2xx": 0, "4xx": 0, "5xx": 0}


def test_render_summary_includes_helix_rows():
    context = _summary_context(
        helix_block={
            "url": "https://helix.example/",
            "status": 200,
            "bytes": 1024,
            "raw_rows": 260,
        },
        performance_block="- Rows fetched: 260\n- Rows normalized: 260\n- Rows written: 260",
        rows_fetched_total=260,
        rows_normalized_total=260,
        rows_written_total=260,
    )

    rendered = idmc_cli._render_summary(context)

    assert "Helix (IDU) Reachability" in rendered
    assert "Rows fetched: 260" in rendered
    assert "Raw rows: 260" in rendered


def test_write_summary_falls_back_to_minimal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def fail_render(_context):
        raise RuntimeError("boom")

    monkeypatch.setattr(idmc_cli, "_render_summary", fail_render)
    monkeypatch.setattr(idmc_cli, "diagnostics_dir", lambda: tmp_path.as_posix())

    context = _summary_context()
    summary_path = idmc_cli._write_summary(context)

    assert summary_path is not None
    text = Path(summary_path).read_text(encoding="utf-8")
    assert "IDMC ingestion summary (fallback)" in text
    assert "## Row counts" in text
