# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pytest
import yaml

from resolver.ingestion._shared import config_loader
from resolver.ingestion._shared.run_io import attach_config_source


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


@pytest.fixture()
def loader_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Tuple[Path, Path, Path]:
    repo_root = tmp_path / "repo"
    resolver_root = repo_root / "resolver"
    ingestion_dir = resolver_root / "ingestion" / "config"
    fallback_dir = resolver_root / "config"
    monkeypatch.setattr(config_loader, "INGESTION_CONFIG_ROOT", ingestion_dir)
    monkeypatch.setattr(config_loader, "LEGACY_CONFIG_ROOT", fallback_dir)
    monkeypatch.setattr(config_loader, "RESOLVER_ROOT", resolver_root)
    monkeypatch.setattr(config_loader, "REPO_ROOT", repo_root)
    monkeypatch.setattr(config_loader, "_LAST_RESULTS", {})
    return resolver_root, ingestion_dir, fallback_dir


def test_loader_prefers_resolver(loader_environment: Tuple[Path, Path, Path]) -> None:
    _, _, legacy_dir = loader_environment
    data = {"enabled": False, "api": {"token_env": "EXAMPLE"}}
    _write_yaml(legacy_dir / "idmc.yml", data)

    cfg, source, chosen_path, warnings = config_loader.load_connector_config("idmc")
    assert source == "resolver"
    assert cfg == data
    assert chosen_path == (legacy_dir / "idmc.yml").resolve()
    assert not warnings

    details = config_loader.get_config_details("idmc")
    assert details is not None
    assert details.path == (legacy_dir / "idmc.yml").resolve()
    assert not details.warnings


def test_loader_ingestion_fallback_warns(
    loader_environment: Tuple[Path, Path, Path]
) -> None:
    _, ingestion_dir, legacy_dir = loader_environment
    fallback_payload = {"enabled": True}
    _write_yaml(ingestion_dir / "idmc.yml", fallback_payload)

    cfg, source, chosen_path, warnings = config_loader.load_connector_config("idmc")
    assert source == "ingestion"
    assert cfg == fallback_payload
    assert chosen_path == (ingestion_dir / "idmc.yml").resolve()
    assert warnings
    assert any("resolver/ingestion/config/idmc.yml" in message for message in warnings)

    diagnostics = attach_config_source({}, "idmc")
    assert diagnostics["config_source"] == "ingestion"
    assert any(
        "resolver/ingestion/config/idmc.yml" in message
        for message in diagnostics.get("warnings", [])
    )
    assert diagnostics["config"]["config_path_used"].endswith(
        "resolver/ingestion/config/idmc.yml"
    )
    assert diagnostics["config"]["config_source_label"] == "ingestion"
    # Ensure helper respected canonical paths from loader
    assert diagnostics["config"]["legacy_config_path"].endswith(
        "resolver/config/idmc.yml"
    )
    assert diagnostics["config"]["ingestion_config_path"].endswith(
        "resolver/ingestion/config/idmc.yml"
    )


def test_loader_duplicate_equal(loader_environment: Tuple[Path, Path, Path]) -> None:
    _, ingestion_dir, legacy_dir = loader_environment
    payload = {"enabled": True, "cache": {"dir": ".cache"}}
    _write_yaml(ingestion_dir / "idmc.yml", payload)
    _write_yaml(legacy_dir / "idmc.yml", payload)

    cfg, source, chosen_path, warnings = config_loader.load_connector_config("idmc")
    assert source == "resolver (dup-equal)"
    assert cfg == payload
    assert chosen_path == (legacy_dir / "idmc.yml").resolve()
    assert not config_loader.get_config_warnings("idmc")
    assert not warnings


def test_loader_duplicate_mismatch(loader_environment: Tuple[Path, Path, Path]) -> None:
    _, ingestion_dir, legacy_dir = loader_environment
    _write_yaml(ingestion_dir / "idmc.yml", {"enabled": True, "api": {"countries": ["AAA"]}})
    legacy_payload = {"enabled": False, "cache": {"ttl_seconds": 3600}}
    _write_yaml(legacy_dir / "idmc.yml", legacy_payload)

    cfg, source, chosen_path, warnings = config_loader.load_connector_config("idmc")
    assert source == "resolver"
    assert cfg == legacy_payload
    assert chosen_path == (legacy_dir / "idmc.yml").resolve()
    mismatch_warnings = config_loader.get_config_warnings("idmc")
    assert mismatch_warnings and "duplicate-mismatch" in mismatch_warnings[0]
    assert warnings and "duplicate-mismatch" in warnings[0]
    with pytest.raises(ValueError) as excinfo:
        config_loader.load_connector_config("idmc", strict_mismatch=True)
    message = str(excinfo.value)
    assert "resolver/config/idmc.yml" in message
    assert "resolver/ingestion/config/idmc.yml" in message


