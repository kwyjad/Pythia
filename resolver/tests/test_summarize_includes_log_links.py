from __future__ import annotations

from pathlib import Path

from resolver.ingestion.diagnostics_emitter import append_jsonl, finalize_run, start_run
from scripts.ci import summarize_connectors


def test_summary_includes_log_links(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    logs_dir = tmp_path / "diagnostics" / "ingestion" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "alpha_client.log").write_text("alpha\n", encoding="utf-8")

    report = tmp_path / "diagnostics" / "ingestion" / "connectors_report.jsonl"
    ctx = start_run("alpha_client", "real")
    append_jsonl(report, finalize_run(ctx, status="ok"))
    ctx = start_run("beta_client", "real")
    append_jsonl(report, finalize_run(ctx, status="error", reason="boom"))

    entries = summarize_connectors.load_report(report)
    markdown = summarize_connectors.build_markdown(entries)

    assert "| Logs |" in markdown

    table_lines = [line for line in markdown.splitlines() if line.startswith("| ")]
    alpha_line = next(line for line in table_lines if line.startswith("| alpha_client |"))
    beta_line = next(line for line in table_lines if line.startswith("| beta_client |"))

    assert alpha_line.endswith("| diagnostics/ingestion/logs/alpha_client.log |")
    assert beta_line.endswith("| — |")
