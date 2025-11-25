import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("duckdb")

from resolver.db import duckdb_io


def test_exporter_dual_write_matches_csv(tmp_path, monkeypatch):
    staging = tmp_path / "staging.csv"
    data = pd.DataFrame(
        [
            {
                "event_id": "E1",
                "country_name": "Philippines",
                "iso3": "PHL",
                "hazard_code": "TC",
                "hazard_label": "Tropical Cyclone",
                "hazard_class": "Cyclone",
                "metric": "in_need",
                "value": "1000",
                "unit": "persons",
                "as_of_date": "2024-01-15",
                "publication_date": "2024-01-16",
                "publisher": "OCHA",
                "source_type": "situation_report",
                "source_url": "https://example.org/tc",
                "doc_title": "Report TC",
                "definition_text": "People in need",
                "method": "reported",
                "confidence": "medium",
                "revision": "1",
                "ingested_at": "2024-01-16T00:00:00Z",
            },
            {
                "event_id": "E2",
                "country_name": "Philippines",
                "iso3": "PHL",
                "hazard_code": "EQ",
                "hazard_label": "Earthquake",
                "hazard_class": "Geophysical",
                "metric": "affected",
                "value": "500",
                "unit": "persons",
                "as_of_date": "2024-01-10",
                "publication_date": "2024-01-11",
                "publisher": "OCHA",
                "source_type": "situation_report",
                "source_url": "https://example.org/eq",
                "doc_title": "Report EQ",
                "definition_text": "People affected",
                "method": "reported",
                "confidence": "high",
                "revision": "1",
                "ingested_at": "2024-01-11T00:00:00Z",
            },
        ]
    )
    data.to_csv(staging, index=False)

    mapping = {column: [column] for column in data.columns}
    config = {"mapping": mapping, "constants": {}}
    config_path = tmp_path / "config.yml"
    config_path.write_text(json.dumps(config))

    out_dir = tmp_path / "exports"
    out_dir.mkdir()

    db_path = tmp_path / "resolver.duckdb"
    monkeypatch.setenv("RESOLVER_DB_URL", f"duckdb:///{db_path}")

    module = __import__("resolver.tools.export_facts", fromlist=["main"])
    argv = [
        "export_facts",
        "--in",
        str(staging),
        "--config",
        str(config_path),
        "--out",
        str(out_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()

    csv_df = pd.read_csv(out_dir / "facts.csv")
    conn = duckdb_io.get_db(f"duckdb:///{db_path}")
    try:
        rows = conn.execute(
            "SELECT event_id, iso3, hazard_code, metric FROM facts_resolved ORDER BY event_id"
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == len(csv_df)
    assert sorted(row[2] for row in rows) == sorted(csv_df["hazard_code"].tolist())


def test_exporter_cli_write_db_flag(monkeypatch, tmp_path):
    staging = tmp_path / "staging.csv"
    data = pd.DataFrame(
        [
            {
                "event_id": "E100",
                "country_name": "Philippines",
                "iso3": "PHL",
                "hazard_code": "TC",
                "hazard_label": "Tropical Cyclone",
                "hazard_class": "Cyclone",
                "metric": "in_need",
                "value": "1000",
                "unit": "persons",
                "as_of_date": "2024-01-15",
                "publication_date": "2024-01-16",
                "publisher": "OCHA",
                "source_type": "situation_report",
                "source_url": "https://example.org/tc",
                "doc_title": "Report TC",
                "definition_text": "People in need",
                "method": "reported",
                "confidence": "medium",
                "revision": "1",
                "ingested_at": "2024-01-16T00:00:00Z",
            }
        ]
    )
    data.to_csv(staging, index=False)

    mapping = {column: [column] for column in data.columns}
    config = {"mapping": mapping, "constants": {}}
    config_path = tmp_path / "config.yml"
    config_path.write_text(json.dumps(config))

    out_dir = tmp_path / "exports"
    out_dir.mkdir()

    db_path = tmp_path / "resolver.duckdb"
    db_url = f"duckdb:///{db_path}"
    module = __import__("resolver.tools.export_facts", fromlist=["main"])

    calls = []
    original = module._maybe_write_to_db

    def spy(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(module, "_maybe_write_to_db", spy)

    argv = [
        "export_facts",
        "--in",
        str(staging),
        "--config",
        str(config_path),
        "--out",
        str(out_dir),
        "--write-db",
        "1",
        "--db-url",
        db_url,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()

    assert calls, "_maybe_write_to_db should be invoked when --write-db is set"
    assert calls[0]["db_url"] == db_url

    conn = duckdb_io.get_db(db_url)
    try:
        rows = conn.execute("SELECT COUNT(*) FROM facts_resolved").fetchone()[0]
    finally:
        conn.close()

    assert rows == len(data)


def test_exporter_cli_duckdb_integration_subprocess(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]

    staging = tmp_path / "staging.csv"
    data = pd.DataFrame(
        [
            {
                "event_id": "E200",
                "country_name": "Philippines",
                "iso3": "PHL",
                "hazard_code": "EQ",
                "hazard_label": "Earthquake",
                "hazard_class": "Geophysical",
                "metric": "affected",
                "value": "1200",
                "unit": "persons",
                "as_of_date": "2024-01-20",
                "publication_date": "2024-01-21",
                "publisher": "OCHA",
                "source_type": "situation_report",
                "source_url": "https://example.org/eq",
                "doc_title": "EQ update",
                "definition_text": "People affected",
                "method": "reported",
                "confidence": "medium",
                "revision": "1",
                "ingested_at": "2024-01-21T00:00:00Z",
            }
        ]
    )
    data.to_csv(staging, index=False)

    mapping = {column: [column] for column in data.columns}
    config = {"mapping": mapping, "constants": {}}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    out_dir = tmp_path / "exports"
    out_dir.mkdir()

    db_path = tmp_path / "subprocess.duckdb"
    db_url = f"duckdb:///{db_path}"

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(repo_root)
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "resolver.tools.export_facts",
            "--in",
            str(staging),
            "--out",
            str(out_dir),
            "--config",
            str(config_path),
            "--write-db",
            "1",
            "--db-url",
            db_url,
        ],
        check=True,
        cwd=repo_root,
        env=env,
    )

    conn = duckdb_io.get_db(db_url)
    try:
        rows = conn.execute("SELECT COUNT(*) FROM facts_resolved").fetchone()[0]
    finally:
        conn.close()

    assert rows == len(data)


def test_verify_duckdb_counts_writes_markdown(monkeypatch, tmp_path):
    import duckdb

    db_path = tmp_path / "resolver.duckdb"
    con = duckdb.connect(db_path)
    con.execute(
        """
        CREATE TABLE facts_resolved (
            source VARCHAR,
            metric VARCHAR,
            series_semantics VARCHAR,
            value INTEGER
        )
        """
    )
    con.execute(
        "INSERT INTO facts_resolved VALUES (?, ?, ?, ?)",
        ("idmc", "affected", "stock", 10),
    )
    con.execute(
        "INSERT INTO facts_resolved VALUES (?, ?, ?, ?)",
        ("idmc", "in_need", "new", 5),
    )
    con.close()

    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repo_root))
    monkeypatch.chdir(tmp_path)

    diagnostics = Path("diagnostics/ingestion")
    diagnostics.mkdir(parents=True, exist_ok=True)
    summary_file = diagnostics / "summary.md"
    summary_file.write_text("## Existing summary\n", encoding="utf-8")

    step_summary = tmp_path / "step-summary.md"

    monkeypatch.setenv("RESOLVER_DB_URL", f"duckdb:///{db_path}")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(step_summary))

    subprocess.run(
        [sys.executable, "-m", "scripts.ci.verify_duckdb_counts"],
        check=True,
    )

    counts_path = diagnostics / "duckdb_counts.md"
    assert counts_path.exists(), "verification markdown should be created"
    contents = counts_path.read_text(encoding="utf-8")
    assert "## DuckDB write verification" in contents
    assert "facts_resolved rows: 2" in contents
    assert "| table | rows |" in contents
    assert "| facts_resolved | 2 |" in contents
    assert "| idmc |" in contents
    assert "Missing tables" not in contents

    combined_summary = summary_file.read_text(encoding="utf-8")
    assert combined_summary.count("DuckDB write verification") == 1

    summary_echo = step_summary.read_text(encoding="utf-8")
    assert "DuckDB write verification" in summary_echo


def test_verify_duckdb_counts_allow_missing(monkeypatch, tmp_path):
    import duckdb

    db_path = tmp_path / "resolver.duckdb"
    con = duckdb.connect(db_path)
    con.execute(
        "CREATE TABLE acled_monthly_fatalities (country TEXT, deaths INTEGER)"
    )
    con.execute(
        "INSERT INTO acled_monthly_fatalities VALUES (?, ?)",
        ("KEN", 12),
    )
    con.close()

    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repo_root))
    monkeypatch.chdir(tmp_path)

    diagnostics = Path("diagnostics/ingestion")
    diagnostics.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("RESOLVER_DB_URL", f"duckdb:///{db_path}")

    rc = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.ci.verify_duckdb_counts",
            str(db_path),
            "--tables",
            "acled_monthly_fatalities",
        ],
        capture_output=True,
        text=True,
    )
    assert rc.returncode == 1
    assert "ERROR:" in rc.stdout

    rc_allow = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.ci.verify_duckdb_counts",
            str(db_path),
            "--tables",
            "acled_monthly_fatalities",
            "--allow-missing",
        ],
        capture_output=True,
        text=True,
    )
    assert rc_allow.returncode == 0
    assert "WARNING:" in rc_allow.stdout

    counts_path = diagnostics / "duckdb_counts.md"
    assert counts_path.exists()
    contents = counts_path.read_text(encoding="utf-8")
    assert "facts_resolved rows: 0" in contents
    assert "| acled_monthly_fatalities | 1 |" in contents
    assert "**Missing tables**" in contents
    assert "- facts_resolved" in contents
