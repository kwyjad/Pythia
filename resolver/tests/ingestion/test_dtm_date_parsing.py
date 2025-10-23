from __future__ import annotations

import json
from pathlib import Path

import pytest

from resolver.ingestion import dtm_client
from resolver.tests.ingestion.test_dtm_ok_empty_status import _setup_paths


@pytest.mark.parametrize(
    "date_format,dates,expected_min,expected_max,expected_errors",
    [
        (None, ["2023-01-01", "2023-01-05", ""], "2023-01-01", "2023-01-05", 1),
        ("%d/%m/%Y", ["01/02/2023", "03/02/2023", "bad"], "2023-02-01", "2023-02-03", 1),
    ],
)
def test_dtm_date_parsing(monkeypatch, tmp_path: Path, date_format, dates, expected_min, expected_max, expected_errors) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RESOLVER_SKIP_DTM", raising=False)
    monkeypatch.delenv("DTM_STRICT", raising=False)
    monkeypatch.delenv("DTM_STRICT_EMPTY", raising=False)
    monkeypatch.setenv("RESOLVER_START_ISO", "2023-01-01")
    monkeypatch.setenv("RESOLVER_END_ISO", "2023-12-31")

    out_path = _setup_paths(monkeypatch, tmp_path)

    source_dir = tmp_path / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    csv_path = source_dir / "sample.csv"
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write("country_iso3,admin1,date,value\n")
        for index, value in enumerate(dates, start=1):
            handle.write(f"KEN,Nairobi,{value},{index}\n")

    source_cfg = {
        "id_or_path": str(csv_path),
        "country_column": "country_iso3",
        "admin1_column": "admin1",
        "date_column": "date",
        "value_column": "value",
    }
    if date_format:
        source_cfg["date_format"] = date_format

    monkeypatch.setattr(
        dtm_client,
        "load_config",
        lambda: {"enabled": True, "sources": [source_cfg]},
    )

    exit_code = dtm_client.main([])
    assert exit_code == 0

    run_payload = json.loads(dtm_client.RUN_DETAILS_PATH.read_text(encoding="utf-8"))
    valid_sources = run_payload["sources"]["valid"]
    assert len(valid_sources) == 1
    valid = valid_sources[0]
    assert valid["parse_errors"] == expected_errors
    assert valid["min"] == expected_min
    assert valid["max"] == expected_max

    totals = run_payload["totals"]
    assert totals["parse_errors"] == expected_errors
    assert totals["kept"] >= 1
    assert totals["rows_written"] == dtm_client.count_csv_rows(out_path)


