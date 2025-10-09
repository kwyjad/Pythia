from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

import pytest

from resolver.tests.test_utils import SNAPS, read_parquet

def test_any_snapshot_parquet_reads_and_has_core_columns():
    if not SNAPS.exists():
        return
    # take any new-style snapshot first, fall back to legacy naming
    paths = list(SNAPS.glob("*/facts_resolved.parquet"))
    if not paths:
        paths = list(SNAPS.glob("*/facts.parquet"))
    if not paths:
        return
    df = read_parquet(paths[0])
    core = {"iso3","hazard_code","metric","value","as_of_date","publication_date"}
    assert core.issubset(set(df.columns))


def test_snapshot_cli_creates_monthly_artifacts(tmp_path, monkeypatch):
    monkeypatch.delenv("RESOLVER_DB_URL", raising=False)

    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    sample_csv = staging_dir / "sample.csv"
    sample_rows = [
        "event_id,country_name,iso3,hazard_code,hazard_label,hazard_class,metric,value,as_of_date,publication_date,publisher,source_type,source_url,doc_title,definition_text,method,confidence,revision,ingested_at",
        "evt-1,Ethiopia,ETH,DR,Drought,natural,in_need,100,2024-01-15,2024-01-16,Relief Org,agency,http://example.com/report-a,Report A,Definition text,api,med,1,2024-01-17T00:00:00Z",
        "evt-2,Ethiopia,ETH,DR,Drought,natural,affected,200,2024-01-14,2024-01-15,Relief Org,agency,http://example.com/report-b,Report B,Definition text,api,med,1,2024-01-15T00:00:00Z",
    ]
    sample_csv.write_text("\n".join(sample_rows) + "\n", encoding="utf-8")

    exports_dir = tmp_path / "exports"
    out_dir = tmp_path / "snapshots"

    cmd = [
        sys.executable,
        "-m",
        "resolver.cli.snapshot_cli",
        "make-monthly",
        "--ym",
        "2024-01",
        "--staging",
        str(staging_dir),
        "--exports-dir",
        str(exports_dir),
        "--outdir",
        str(out_dir),
        "--export-config",
        str(Path("resolver/tools/export_config.yml")),
        "--overwrite",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"snapshot_cli failed: {result.stderr}\n{result.stdout}")

    snapshot_month_dir = out_dir / "2024-01"
    assert snapshot_month_dir.exists()
    resolved_csv = snapshot_month_dir / "facts_resolved.csv"
    resolved_parquet = snapshot_month_dir / "facts_resolved.parquet"
    deltas_csv = snapshot_month_dir / "facts_deltas.csv"
    deltas_parquet = snapshot_month_dir / "facts_deltas.parquet"
    manifest = snapshot_month_dir / "manifest.json"

    assert resolved_csv.exists()
    assert resolved_parquet.exists()
    assert deltas_csv.exists()
    assert deltas_parquet.exists()
    assert manifest.exists()

    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_data.get("target_month") == "2024-01"
    assert manifest_data.get("resolved_rows", 0) > 0
