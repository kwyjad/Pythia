import json
from pathlib import Path

import pandas as pd

from resolver.tools.export_facts import DEFAULT_CONFIG, export_facts


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_export_skips_meta_json_and_uses_csv(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    csv_path = staging_dir / "dtm_displacement.csv"
    _write_csv(
        csv_path,
        [
            {"CountryISO3": "nga", "ReportingDate": "2024-01-01", "idp_count": 100},
            {"CountryISO3": "ner", "ReportingDate": "2024-01-05", "idp_count": 50},
        ],
    )
    meta_path = staging_dir / "dtm_displacement.csv.meta.json"
    meta_path.write_text("", encoding="utf-8")

    out_dir = tmp_path / "exports"
    result = export_facts(
        inp=staging_dir,
        config_path=DEFAULT_CONFIG,
        out_dir=out_dir,
        write_db="0",
    )

    assert result.rows == 2
    assert result.csv_path.exists()

    meta_entries = [detail for detail in result.sources if detail.path == meta_path]
    assert meta_entries, "meta json should be recorded as skipped"
    assert meta_entries[0].strategy == "meta-skip"
    assert meta_entries[0].rows_out == 0

    used_paths = {detail.path for detail in result.sources if detail.rows_out > 0}
    assert csv_path in used_paths


def test_export_report_written(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    csv_path = staging_dir / "dtm_displacement.csv"
    _write_csv(
        csv_path,
        [
            {"CountryISO3": "uga", "ReportingDate": "2024-01-10", "idp_count": 25},
        ],
    )
    bad_json = staging_dir / "bad.json"
    bad_json.write_text("not json", encoding="utf-8")

    out_dir = tmp_path / "exports"
    result = export_facts(
        inp=staging_dir,
        config_path=DEFAULT_CONFIG,
        out_dir=out_dir,
        write_db="0",
    )

    report_json = out_dir / "export_report.json"
    report_md = out_dir / "export_report.md"

    assert result.rows == 1
    assert report_json.exists()
    assert report_md.exists()

    report_data = json.loads(report_json.read_text(encoding="utf-8"))
    assert report_data["rows_exported"] == 1
    assert str(csv_path) in report_data["inputs_used"]
    skipped = {entry["path"]: entry for entry in report_data["inputs_skipped"]}
    assert str(bad_json) in skipped
    assert skipped[str(bad_json)]["strategy"] in {"empty-input", "read-failed"}
    assert report_data["sample_head"]

    md_contents = report_md.read_text(encoding="utf-8")
    assert "Export Report" in md_contents
    assert "bad.json" in md_contents


def test_dtm_source_mapping_applies(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    csv_path = staging_dir / "dtm_displacement.csv"
    _write_csv(
        csv_path,
        [
            {"CountryISO3": "nga", "ReportingDate": "2024-01-05", "idp_count": 100},
            {"CountryISO3": "NGA", "ReportingDate": "2024-01-05", "idp_count": 120},
            {"CountryISO3": "ner", "ReportingDate": "2024-01-07", "idp_count": 40},
        ],
    )

    out_dir = tmp_path / "exports"
    result = export_facts(
        inp=staging_dir,
        config_path=DEFAULT_CONFIG,
        out_dir=out_dir,
        write_db="0",
    )

    df = result.dataframe
    assert {"NGA", "NER"} == set(df["iso3"])
    nga_row = df[df["iso3"] == "NGA"].iloc[0]
    assert nga_row["value"] == "120"
    assert nga_row["metric"] == "idps"
    assert nga_row["ym"] == "2024-01"
    assert nga_row["semantics"] == "stock"

    assert len(df[df["iso3"] == "NGA"]) == 1
    assert (out_dir / "facts.csv").exists()
