"""Tests for the IDMC provenance manifest."""
import json
from pathlib import Path

from resolver.ingestion.idmc import cli


def test_idmc_manifest_shape_and_redaction(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IDMC_API_TOKEN", "supersecret-token")

    exit_code = cli.main(["--skip-network"])
    assert exit_code == 0

    manifest_path = Path("diagnostics/ingestion/idmc/manifest.json")
    assert manifest_path.exists(), "manifest.json should be written"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for key in [
        "schema_version",
        "generated_at_utc",
        "source_system",
        "series",
        "attribution",
        "run",
        "reachability",
        "http",
        "cache",
        "normalize",
        "export",
        "notes",
    ]:
        assert key in manifest, f"missing {key} in manifest"

    assert manifest["schema_version"] == 1
    assert manifest["source_system"] == "IDMC"
    assert manifest["series"] == "IDU"

    run_block = manifest["run"]
    assert run_block["cmd"] == "resolver.ingestion.idmc.cli"
    assert run_block["args"]["skip_network"] is True

    env_block = run_block.get("env", {})
    assert "IDMC_API_TOKEN" in env_block
    assert env_block["IDMC_API_TOKEN"] == "***REDACTED***"
    assert "IDMC_BASE_URL" in env_block

    normalize_block = manifest["normalize"]
    assert "rows_normalized" in normalize_block
    assert normalize_block["rows_normalized"] >= 0
