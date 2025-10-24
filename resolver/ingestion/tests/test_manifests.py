from __future__ import annotations

import json
import csv
from pathlib import Path

import pytest

from resolver.ingestion._manifest import (
    count_csv_rows,
    ensure_manifest_for_csv,
    load_manifest,
    manifest_path_for,
)
from resolver.ingestion import run_all_stubs
from resolver.ingestion.utils.io import resolve_output_path, resolve_period_label


def test_manifest_creation_and_repair(tmp_path: Path) -> None:
    csv_path = tmp_path / "example.csv"
    csv_path.write_text("a,b\n", encoding="utf-8")

    manifest = ensure_manifest_for_csv(csv_path)
    manifest_path = manifest_path_for(csv_path)

    assert manifest_path.exists(), "manifest file should be created"
    assert manifest["format"] == "csv"
    assert manifest["row_count"] == 0
    assert manifest["data_path"] == csv_path.name
    assert "sha256" in manifest
    assert "generated_at" in manifest

    csv_path.write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")
    updated = ensure_manifest_for_csv(csv_path)

    assert updated["row_count"] == 3
    assert updated.get("sha256") != manifest.get("sha256")

    corrupt = load_manifest(manifest_path) or {}
    corrupt["row_count"] = 999
    manifest_path.write_text(json.dumps(corrupt), encoding="utf-8")

    recount = count_csv_rows(csv_path)
    reloaded = load_manifest(manifest_path)
    assert reloaded is not None
    assert reloaded["row_count"] != recount

    repaired = ensure_manifest_for_csv(csv_path)
    assert repaired["row_count"] == recount
    fixed = load_manifest(manifest_path)
    assert fixed is not None
    assert fixed["row_count"] == recount


@pytest.mark.usefixtures("monkeypatch")
def test_runner_writes_to_staging_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_dir = tmp_path
    monkeypatch.setenv("RESOLVER_STAGING_DIR", str(base_dir))
    monkeypatch.setenv("RESOLVER_START_ISO", "2025-07-01")
    monkeypatch.setenv("RESOLVER_END_ISO", "2025-09-30")

    connector_names = ["ifrc_go_client.py", "ipc_client.py"]
    specs: list[run_all_stubs.ConnectorSpec] = []
    writer_map: dict[str, tuple[Path, list[list[str]]]] = {}

    for name in connector_names:
        default_filename = run_all_stubs.CONNECTOR_OUTPUTS[name]
        default_path = run_all_stubs.STAGING / default_filename
        output_path = resolve_output_path(default_path)
        metadata = {"output_path": str(output_path), "default_filename": default_filename}
        spec = run_all_stubs.ConnectorSpec(
            filename=name,
            path=Path(f"/tmp/{name}"),
            kind="real",
            output_path=output_path,
            summary=None,
            skip_reason=None,
            metadata=metadata,
            config_path=None,
            config={"enabled": True},
            canonical_name=name,
            origin="real_list",
            authoritatively_selected=True,
        )
        specs.append(spec)

    writer_map[specs[0].path.name] = (specs[0].output_path, [["r1", "v1"], ["r2", "v2"]])
    writer_map[specs[1].path.name] = (specs[1].output_path, [])

    def fake_build_specs(
        real,
        stubs,
        selected,
        run_real,
        run_stubs,
        *,
        real_authoritative=False,
        stub_authoritative=False,
    ):
        return list(specs)

    def fake_invoke(path: Path, *, logger=None) -> None:  # type: ignore[override]
        output_path, rows = writer_map[path.name]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["col_a", "col_b"])
            writer.writerows(rows)

    monkeypatch.setattr(run_all_stubs, "_build_specs", fake_build_specs)
    monkeypatch.setattr(run_all_stubs, "_invoke_connector", fake_invoke)

    exit_code = run_all_stubs.main([])
    assert exit_code == 0

    period = resolve_period_label()
    expected_dir = base_dir / period / "raw"
    for spec in specs:
        assert spec.output_path.parent == expected_dir
        assert spec.output_path.exists()
    with specs[0].output_path.open("r", encoding="utf-8") as handle:
        assert sum(1 for _ in handle) == 3
    with specs[1].output_path.open("r", encoding="utf-8") as handle:
        assert sum(1 for _ in handle) == 1
