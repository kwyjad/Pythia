import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from resolver.tools import freeze_snapshot


def _make_resolved_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "iso3": ["KEN"],
            "metric": ["fatalities_battle_month"],
            "series_semantics": ["new"],
            "value": [5],
            "as_of_date": ["2024-01-31"],
            "publication_date": ["2024-02-01"],
            "ym": ["2024-01"],
        }
    )


def _make_deltas_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "iso3": ["KEN"],
            "metric": ["fatalities_battle_month"],
            "series_semantics": ["new"],
            "value": [5],
            "value_new": [5],
            "as_of": ["2024-01-31"],
            "ym": ["2024-01"],
        }
    )


def _stub_duckdb(monkeypatch: pytest.MonkeyPatch, *, failing: bool = False) -> None:
    class _Conn:
        pass

    def get_db(url: str):  # noqa: D401 - test stub
        return _Conn()

    def write_snapshot(
        _conn,
        *,
        ym: str,
        facts_resolved: pd.DataFrame | None,
        facts_deltas: pd.DataFrame | None,
        manifests,
        meta,
    ) -> None:  # noqa: D401 - test stub
        if failing:
            raise RuntimeError("freeze boom")
        assert ym == "2024-01"
        assert meta["facts_path"].endswith("facts.csv")
        assert facts_resolved is None or len(facts_resolved) >= 1

    stub = SimpleNamespace(
        get_db=get_db,
        write_snapshot=write_snapshot,
        close_db=lambda _conn: None,
    )

    monkeypatch.setattr(freeze_snapshot, "duckdb_io", stub)
    monkeypatch.setattr(freeze_snapshot, "canonicalize_duckdb_target", lambda url: (url, url))


def test_freeze_snapshot_db_diagnostics_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    _stub_duckdb(monkeypatch)

    resolved = _make_resolved_df()
    deltas = _make_deltas_df()
    facts_csv = tmp_path / "facts.csv"
    resolved_csv = tmp_path / "resolved.csv"
    deltas_csv = tmp_path / "deltas.csv"
    manifest_path = tmp_path / "manifest.json"

    resolved.to_csv(facts_csv, index=False)
    resolved.to_csv(resolved_csv, index=False)
    deltas.to_csv(deltas_csv, index=False)
    manifest_path.write_text(json.dumps({"created_at_utc": "2024-02-01T00:00:00Z"}), encoding="utf-8")

    freeze_snapshot._maybe_write_db(
        facts_path=facts_csv,
        resolved_path=resolved_csv,
        deltas_path=deltas_csv,
        manifest_path=manifest_path,
        month="2024-01",
        write_db=True,
        db_url="duckdb:///freeze.duckdb",
    )

    diag_path = tmp_path / "diagnostics" / "ingestion" / "freeze_db.json"
    assert diag_path.is_file()
    payload = json.loads(diag_path.read_text(encoding="utf-8"))
    assert payload["facts_resolved_rows"] == 1
    assert payload["facts_deltas_rows"] == 1
    assert payload["facts_resolved_semantics"] == {"new": 1}

    summary_path = tmp_path / "diagnostics" / "ingestion" / "summary.md"
    summary = summary_path.read_text(encoding="utf-8")
    assert "## Freeze Snapshot — DB diagnostics" in summary
    assert "new: 1" in summary
    assert "## DB Write Diagnostics" not in summary


def test_freeze_snapshot_db_diagnostics_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    _stub_duckdb(monkeypatch, failing=True)

    resolved = _make_resolved_df()
    deltas = _make_deltas_df()
    facts_csv = tmp_path / "facts.csv"
    resolved_csv = tmp_path / "resolved.csv"
    deltas_csv = tmp_path / "deltas.csv"
    manifest_path = tmp_path / "manifest.json"

    resolved.to_csv(facts_csv, index=False)
    resolved.to_csv(resolved_csv, index=False)
    deltas.to_csv(deltas_csv, index=False)
    manifest_path.write_text(json.dumps({"created_at_utc": "2024-02-01T00:00:00Z"}), encoding="utf-8")

    freeze_snapshot._maybe_write_db(
        facts_path=facts_csv,
        resolved_path=resolved_csv,
        deltas_path=deltas_csv,
        manifest_path=manifest_path,
        month="2024-01",
        write_db=True,
        db_url="duckdb:///freeze.duckdb",
    )

    diag_path = tmp_path / "diagnostics" / "ingestion" / "freeze_db.json"
    assert diag_path.is_file()
    payload = json.loads(diag_path.read_text(encoding="utf-8"))
    assert payload["facts_resolved_rows"] == 1

    summary_path = tmp_path / "diagnostics" / "ingestion" / "summary.md"
    summary = summary_path.read_text(encoding="utf-8")
    assert "## DB Write Diagnostics" in summary
    assert "- **Step:** Freeze Snapshot" in summary
    assert "freeze boom" in summary
    assert "## Freeze Snapshot — DB diagnostics" not in summary
