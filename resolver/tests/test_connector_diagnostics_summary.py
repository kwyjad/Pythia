from __future__ import annotations

from pathlib import Path

from resolver.ingestion.diagnostics_emitter import append_jsonl, finalize_run, start_run
from scripts.ci import summarize_connectors


def _write_result(
    report: Path,
    connector_id: str,
    mode: str,
    status: str,
    *,
    reason: str | None = None,
    http: dict | None = None,
    counts: dict | None = None,
    coverage: dict | None = None,
    samples: dict | None = None,
    extras: dict | None = None,
) -> None:
    ctx = start_run(connector_id, mode if mode in {"real", "stub"} else "real")
    result = finalize_run(
        ctx,
        status=status,
        reason=reason,
        http=http,
        counts=counts,
        coverage=coverage,
        samples=samples,
        extras=extras,
    )
    append_jsonl(report, result)


def test_schema_round_trip(tmp_path: Path) -> None:
    report = tmp_path / "connectors_report.jsonl"
    _write_result(
        report,
        connector_id="acled_client",
        mode="real",
        status="ok",
        http={"2xx": 12, "4xx": 0, "5xx": 0, "rate_limit_remaining": 40, "last_status": 200},
        counts={"fetched": 15, "normalized": 12, "written": 12},
        coverage={
            "ym_min": "2023-01",
            "ym_max": "2023-02",
            "as_of_min": "2023-02-01",
            "as_of_max": "2023-02-20",
        },
        samples={"top_iso3": [("KEN", 5), ("UGA", 3)], "top_hazard": [("conflict", 4)]},
        extras={"rows_method": "manifest"},
    )
    entries = summarize_connectors.load_report(report)
    assert entries[0]["connector_id"] == "acled_client"
    assert entries[0]["counts"]["written"] == 12
    markdown = summarize_connectors.build_markdown(entries)
    assert "| acled_client | real | ok | — |" in markdown
    assert "`KEN` (5)" in markdown


def test_redaction_hides_sensitive_extras(tmp_path: Path) -> None:
    report = tmp_path / "connectors_report.jsonl"
    _write_result(
        report,
        connector_id="gdacs_client",
        mode="real",
        status="ok",
        counts={"fetched": 1, "normalized": 1, "written": 1},
        extras={"authorization": "Bearer token", "note": "safe"},
    )
    markdown = summarize_connectors.build_markdown(summarize_connectors.load_report(report))
    assert "Bearer" not in markdown
    assert "***" in markdown


def test_aggregates_reflect_status_and_reason_histograms(tmp_path: Path) -> None:
    report = tmp_path / "connectors_report.jsonl"
    _write_result(
        report,
        connector_id="reliefweb_client",
        mode="real",
        status="ok",
        counts={"fetched": 10, "normalized": 10, "written": 9},
    )
    _write_result(
        report,
        connector_id="ifrc_go_client",
        mode="real",
        status="ok",
        counts={"fetched": 5, "normalized": 5, "written": 5},
    )
    _write_result(
        report,
        connector_id="worldpop_client",
        mode="real",
        status="skipped",
        reason="disabled: config",
    )
    _write_result(
        report,
        connector_id="gdacs_client",
        mode="real",
        status="error",
        reason="upstream-502",
    )
    markdown = summarize_connectors.build_markdown(summarize_connectors.load_report(report))
    assert "Status counts:** error=1, ok=2, skipped=1" in markdown
    assert "Reason histogram:** disabled: config=1, upstream-502=1" in markdown


def test_coverage_formats_missing_values(tmp_path: Path) -> None:
    report = tmp_path / "connectors_report.jsonl"
    _write_result(
        report,
        connector_id="fews_stub",
        mode="stub",
        status="skipped",
        reason="missing secret",
    )
    markdown = summarize_connectors.build_markdown(summarize_connectors.load_report(report))
    assert "| fews_stub | stub | skipped | missing secret | — | 0/0/0 (0) | 0/0/0 | — | — | — |" in markdown
