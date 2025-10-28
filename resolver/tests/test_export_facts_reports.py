import json
from pathlib import Path

import pandas as pd

from resolver.tools.export_facts import DEFAULT_CONFIG, export_facts


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_export_source_specific_dtm_mapping(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    csv_path = staging_dir / "dtm_displacement.csv"
    _write_csv(
        csv_path,
        [
            {"CountryISO3": "nga", "ReportingDate": "2024-01-01", "idp_count": 100},
            {"CountryISO3": "Ner", "ReportingDate": "2024-01-05", "idp_count": 50},
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
    assert result.rows == 2
    assert {
        "iso3",
        "as_of_date",
        "ym",
        "metric",
        "value",
        "semantics",
        "series_semantics",
        "source",
    }.issubset(
        set(df.columns)
    )
    assert set(df["iso3"]) == {"NGA", "NER"}
    assert set(df["metric"]) == {"idp_displacement_stock_dtm"}
    assert set(df["semantics"]) == {"stock"}
    assert set(df["source"]) == {"IOM DTM"}
    assert set(df["as_of_date"]) == {"2024-01-31"}

    matched = result.report["matched_files"]
    assert len(matched) == 1
    assert matched[0]["source"] == "dtm_displacement_admin0"
    assert matched[0]["rows_in"] == 2
    assert matched[0]["rows_after_aggregate"] == 2
    assert matched[0]["aggregate"] == {
        "funcs": {"value": "max"},
        "keys": ["iso3", "as_of_date", "metric"],
        "rows_after": 2,
        "rows_before": 2,
    }
    mapping = matched[0]["mapping"]
    assert mapping["metric"]["const"] == "idp_displacement_stock_dtm"
    assert mapping["iso3"]["source"] == "CountryISO3"
    assert result.report["monthly_summary"] == [{"month": "2024-01", "rows": 2}]


def test_export_filters_and_dedupe(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    csv_path = staging_dir / "dtm_displacement.csv"
    _write_csv(
        csv_path,
        [
            {"CountryISO3": "uga", "ReportingDate": "2024-02-01", "idp_count": 5},
            {"CountryISO3": "uga", "ReportingDate": "2024-02-01", "idp_count": 10},
            {"CountryISO3": "uga", "ReportingDate": "2024-02-02", "idp_count": 0},
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
    assert result.rows == 1
    assert df.iloc[0]["value"] == "10"
    assert result.report["dropped_by_filter"] == {
        "keep_if_not_null": 0,
        "keep_if_positive": 1,
    }
    assert result.report["dedupe_keys"] == ["iso3", "as_of_date", "metric"]
    assert result.report["dedupe_keep"] == ["max"]

    matched = result.report["matched_files"][0]
    assert matched["rows_in"] == 3
    assert matched["rows_after_filters"] == 2
    assert matched["rows_after_aggregate"] == 1
    assert matched["rows_after_dedupe"] == 1
    assert matched["aggregate"]["rows_before"] == 2
    assert matched["aggregate"]["rows_after"] == 1
    assert matched["aggregate"]["funcs"] == {"value": "max"}
    assert result.report["monthly_summary"] == [{"month": "2024-02", "rows": 1}]


def test_export_report_written_and_appended(tmp_path, monkeypatch):
    gh_summary = tmp_path / "gh_summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(gh_summary))
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    csv_path = staging_dir / "dtm_displacement.csv"
    _write_csv(
        csv_path,
        [
            {"CountryISO3": "cod", "ReportingDate": "2024-03-15", "idp_count": 75},
        ],
    )

    out_dir = tmp_path / "exports"
    summary_path = tmp_path / "summary.md"
    result = export_facts(
        inp=staging_dir,
        config_path=DEFAULT_CONFIG,
        out_dir=out_dir,
        write_db="0",
        append_summary_path=summary_path,
    )

    report_json = out_dir / "export_report.json"
    report_md = out_dir / "export_report.md"

    assert report_json.exists()
    assert report_md.exists()
    assert summary_path.exists()
    assert gh_summary.exists()

    report_data = json.loads(report_json.read_text(encoding="utf-8"))
    assert report_data["rows_exported"] == 1
    assert report_data["matched_files"][0]["path"].endswith("dtm_displacement.csv")
    assert report_data["filters_applied"] == ["keep_if_not_null", "keep_if_positive"]
    assert report_data["monthly_summary"] == [{"month": "2024-03", "rows": 1}]
    assert "rows_after_aggregate" in report_data["matched_files"][0]

    md_contents = report_md.read_text(encoding="utf-8")
    assert "## Export Facts" in md_contents
    assert "Matched files" in md_contents
    assert "dtm_displacement.csv" in md_contents
    assert "Mapping & aggregation" in md_contents
    assert "Monthly summary" in md_contents

    summary_contents = summary_path.read_text(encoding="utf-8")
    assert md_contents.strip() in summary_contents
    gh_contents = gh_summary.read_text(encoding="utf-8")
    assert "## Export Facts" in gh_contents


def test_export_handles_unmatched_files_gracefully(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    unknown_path = staging_dir / "unknown.csv"
    _write_csv(unknown_path, [{"foo": "bar"}])

    out_dir = tmp_path / "exports"
    result = export_facts(
        inp=staging_dir,
        config_path=DEFAULT_CONFIG,
        out_dir=out_dir,
        write_db="0",
    )

    assert result.rows == 0
    assert result.report["rows_exported"] == 0
    assert result.report["matched_files"] == []
    assert any(unknown_path.name in path for path in result.report["unmatched_files"])
    assert (out_dir / "facts.csv").exists()
