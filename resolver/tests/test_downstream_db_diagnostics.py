import json
from pathlib import Path

import pandas as pd
import pytest

from resolver.tools import export_facts
from resolver.tools import freeze_snapshot
from scripts.ci import append_error_to_summary
from scripts.ci import build_llm_context


@pytest.fixture(autouse=True)
def _chdir_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def capture_append(monkeypatch: pytest.MonkeyPatch):
    calls: list[list[str]] = []

    def _capture(cmd: list[str], check: bool = False, **_: object) -> None:
        calls.append(list(cmd))
        # The helper is invoked via ``python -m``; execute the same logic inline so the
        # summary file is populated for assertions.
        append_error_to_summary.main(cmd[3:])

    monkeypatch.setattr(export_facts.subprocess, "run", _capture)
    monkeypatch.setattr(freeze_snapshot.subprocess, "run", _capture)
    monkeypatch.setattr(build_llm_context.subprocess, "run", _capture)
    return calls


def test_export_append_error_on_db_failure(monkeypatch: pytest.MonkeyPatch, capture_append: list[list[str]]):
    dummy_df = pd.DataFrame(
        {
            "iso3": ["COL"],
            "as_of_date": ["2024-01-31"],
            "metric": ["test_metric"],
            "value": [1.0],
            "series_semantics": ["new"],
        }
    )

    def _boom(*_: object, **__: object) -> None:
        raise RuntimeError("db offline")

    monkeypatch.setattr(export_facts.duckdb_io, "get_db", _boom)

    export_facts._maybe_write_to_db(
        facts_resolved=dummy_df,
        facts_deltas=None,
        db_url="duckdb:///test.duckdb",
        write_db=True,
        fail_on_error=False,
    )

    summary_path = Path("diagnostics/ingestion/summary.md")
    assert summary_path.exists()
    content = summary_path.read_text(encoding="utf-8")
    assert "## Export Facts — DB write" in content
    assert "db offline" in content
    assert capture_append, "expected append helper to be invoked"


def test_context_append_error_on_failure(monkeypatch: pytest.MonkeyPatch, capture_append: list[list[str]]):
    monkeypatch.setenv("RESOLVER_DB_URL", "duckdb:///missing.duckdb")
    monkeypatch.setenv("CONTEXT_MONTHS", "3")

    def _boom() -> int:
        raise RuntimeError("context failure")

    monkeypatch.setattr(build_llm_context, "_run", _boom)

    with pytest.raises(RuntimeError):
        build_llm_context.main()

    summary_path = Path("diagnostics/ingestion/summary.md")
    assert summary_path.exists()
    content = summary_path.read_text(encoding="utf-8")
    assert "## LLM Context — build error" in content
    payload_line = next(
        line for line in content.splitlines() if line.startswith("- **Context:**")
    )
    payload = json.loads(payload_line.split("`", 2)[1])
    assert payload["db_url"] == "duckdb:///missing.duckdb"
    assert payload["context_months"] == "3"
    assert capture_append, "expected append helper to be invoked"
