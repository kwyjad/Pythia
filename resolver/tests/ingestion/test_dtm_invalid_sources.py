from __future__ import annotations

import json
from pathlib import Path

from resolver.ingestion import dtm_client
from resolver.tests.ingestion.test_dtm_ok_empty_status import _setup_paths


def test_dtm_records_invalid_sources(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RESOLVER_SKIP_DTM", raising=False)
    monkeypatch.delenv("DTM_STRICT", raising=False)
    monkeypatch.delenv("DTM_STRICT_EMPTY", raising=False)
    monkeypatch.setenv("RESOLVER_START_ISO", "2023-01-01")
    monkeypatch.setenv("RESOLVER_END_ISO", "2023-12-31")

    _setup_paths(monkeypatch, tmp_path)

    source_dir = tmp_path / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    valid_csv = source_dir / "valid.csv"
    valid_csv.write_text(
        "country_iso3,admin1,date,value\n"
        "ETH,Addis Ababa,2023-03-01,10\n",
        encoding="utf-8",
    )

    config = {
        "enabled": True,
        "sources": [
            {"id_or_path": str(valid_csv)},
            {"name": "missing-id"},
            {
                "id_or_path": str(valid_csv),
                "name": "missing-date",
                "date_column": "not_real",
            },
        ],
    }

    monkeypatch.setattr(dtm_client, "load_config", lambda: config)

    exit_code = dtm_client.main([])
    assert exit_code == 0

    run_payload = json.loads(dtm_client.RUN_DETAILS_PATH.read_text(encoding="utf-8"))
    invalid_entries = run_payload["sources"]["invalid"]
    reasons = {entry.get("reason") or entry.get("skip_reason") for entry in invalid_entries}
    assert "missing id_or_path" in reasons
    assert "missing date_column" in reasons
    totals = run_payload["totals"]
    assert totals["invalid_sources"] == 2
    extras = json.loads(dtm_client.CONNECTORS_REPORT.read_text(encoding="utf-8").splitlines()[0])
    assert extras["extras"]["invalid_sources"] == 2
