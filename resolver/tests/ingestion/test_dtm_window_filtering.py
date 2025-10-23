from __future__ import annotations

import json
from pathlib import Path

from resolver.ingestion import dtm_client
from resolver.tests.ingestion.test_dtm_ok_empty_status import _setup_paths


def _prepare_config(monkeypatch, tmp_path: Path, dates: list[str]) -> None:
    out_path = _setup_paths(monkeypatch, tmp_path)
    source_dir = tmp_path / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    csv_path = source_dir / "window.csv"
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write("country_iso3,admin1,date,value\n")
        for index, value in enumerate(dates, start=1):
            handle.write(f"UGA,Central,{value},{index}\n")

    source_cfg = {
        "id_or_path": str(csv_path),
        "country_column": "country_iso3",
        "admin1_column": "admin1",
        "date_column": "date",
        "value_column": "value",
    }

    monkeypatch.setattr(
        dtm_client,
        "load_config",
        lambda: {"enabled": True, "sources": [source_cfg]},
    )

    return out_path


def test_dtm_window_filters_rows(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RESOLVER_SKIP_DTM", raising=False)
    monkeypatch.delenv("DTM_STRICT", raising=False)
    monkeypatch.delenv("DTM_STRICT_EMPTY", raising=False)
    monkeypatch.setenv("RESOLVER_START_ISO", "2023-01-01")
    monkeypatch.setenv("RESOLVER_END_ISO", "2023-01-31")

    _prepare_config(monkeypatch, tmp_path, ["2023-01-05", "2023-02-05"])

    exit_code = dtm_client.main([])
    assert exit_code == 0

    run_payload = json.loads(dtm_client.RUN_DETAILS_PATH.read_text(encoding="utf-8"))
    totals = run_payload["totals"]
    assert totals["kept"] == 1
    assert totals["dropped"] == 1
    assert totals["invalid_sources"] == 0


def test_dtm_no_date_filter_keeps_all_rows(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RESOLVER_SKIP_DTM", raising=False)
    monkeypatch.delenv("DTM_STRICT", raising=False)
    monkeypatch.delenv("DTM_STRICT_EMPTY", raising=False)
    monkeypatch.setenv("RESOLVER_START_ISO", "2023-01-01")
    monkeypatch.setenv("RESOLVER_END_ISO", "2023-01-31")

    out_path = _prepare_config(monkeypatch, tmp_path, ["2023-01-05", "2023-02-05"])

    exit_code = dtm_client.main(["--no-date-filter"])
    assert exit_code == 0

    run_payload = json.loads(dtm_client.RUN_DETAILS_PATH.read_text(encoding="utf-8"))
    totals = run_payload["totals"]
    assert totals["kept"] == 2
    assert totals["dropped"] == 0
    assert totals["invalid_sources"] == 0
    assert dtm_client.count_csv_rows(out_path) > 0


