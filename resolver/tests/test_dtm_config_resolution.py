from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from resolver.ingestion import dtm_client


@pytest.fixture(autouse=True)
def clear_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DTM_CONFIG_PATH", raising=False)


def _prepare_repo(monkeypatch: pytest.MonkeyPatch, repo_root: Path) -> None:
    resolver_root = repo_root / "resolver"
    ingestion_config = resolver_root / "ingestion" / "config"
    ingestion_config.mkdir(parents=True, exist_ok=True)
    (resolver_root / "config").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(dtm_client, "_REPO_ROOT", repo_root, raising=False)
    monkeypatch.setattr(dtm_client, "REPO_ROOT", repo_root, raising=False)
    monkeypatch.setattr(dtm_client, "RESOLVER_ROOT", resolver_root, raising=False)
    ingestion_default = (ingestion_config / "dtm.yml").resolve()
    monkeypatch.setattr(dtm_client, "LEGACY_CONFIG_PATH", ingestion_default, raising=False)
    monkeypatch.setattr(dtm_client, "CONFIG_PATH", ingestion_default, raising=False)


def test_load_config_honours_search_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _prepare_repo(monkeypatch, repo_root)

    custom_config = tmp_path / "custom" / "dtm.yml"
    custom_config.parent.mkdir(parents=True, exist_ok=True)
    custom_config.write_text("enabled: false\napi:\n  countries: []\n", encoding="utf-8")

    resolver_config = repo_root / "resolver" / "config" / "dtm.yml"
    resolver_config.write_text("enabled: true\n", encoding="utf-8")

    ingestion_config = repo_root / "resolver" / "ingestion" / "config" / "dtm.yml"
    ingestion_config.write_text("enabled: true\napi:\n  countries: []\n", encoding="utf-8")

    monkeypatch.setenv("DTM_CONFIG_PATH", str(custom_config))
    cfg = dtm_client.load_config()
    assert Path(getattr(cfg, "_source_path")).resolve() == custom_config.resolve()
    assert getattr(cfg, "_source_exists") is True
    expected_sha = hashlib.sha256(custom_config.read_bytes()).hexdigest()[:12]
    assert getattr(cfg, "_source_sha256") == expected_sha
    assert cfg.get("enabled") is False

    monkeypatch.delenv("DTM_CONFIG_PATH", raising=False)
    cfg = dtm_client.load_config()
    assert Path(getattr(cfg, "_source_path")).resolve() == ingestion_config.resolve()
    assert getattr(cfg, "_source_exists") is True

    resolver_config.unlink()
    cfg = dtm_client.load_config()
    assert Path(getattr(cfg, "_source_path")).resolve() == ingestion_config.resolve()
    assert getattr(cfg, "_source_exists") is True

    ingestion_config.unlink()
    cfg = dtm_client.load_config()
    assert Path(getattr(cfg, "_source_path")).resolve() == resolver_config.resolve()
    assert getattr(cfg, "_source_exists") is False
    assert cfg.get("api") == {}
