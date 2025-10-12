from __future__ import annotations

import importlib
import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Iterable

import pytest

from . import _synthetic_testdata

CI_ENV_VAR = "GITHUB_ACTIONS"
REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORTS_DIR = REPO_ROOT / "exports"
STAGING_DIR = REPO_ROOT / "staging"
STATE_DIR = REPO_ROOT / "state"
SNAPSHOTS_DIR = REPO_ROOT / "snapshots"


def _running_on_ci() -> bool:
    return os.environ.get(CI_ENV_VAR, "").lower() == "true"


def _flag_enabled(var: str, default: str = "1") -> bool:
    value = os.environ.get(var, default)
    return value.lower() not in {"0", "false", "no"}


@lru_cache(maxsize=1)
def _exports_available() -> bool:
    env_dir = os.environ.get("RESOLVER_TEST_DATA_DIR")
    if env_dir:
        exports = Path(env_dir) / "exports"
        if exports.exists() and any(exports.glob("*.csv")):
            return True
    if EXPORTS_DIR.exists() and any(EXPORTS_DIR.glob("*.csv")):
        return True
    if STATE_DIR.exists() and any(STATE_DIR.glob("**/exports/*.csv")):
        return True
    return False


@lru_cache(maxsize=1)
def _staging_available() -> bool:
    return STAGING_DIR.exists() and any(STAGING_DIR.glob("*.csv"))


def _should_skip_node(nodeid: str) -> bool:
    nodeid = nodeid.lower()
    if "test_staging_schema_all.py" in nodeid:
        return True
    if "staging_schema" in nodeid:
        return not _staging_available()
    if "schema_parity" in nodeid or "export_parity" in nodeid:
        return not _exports_available()
    if "db_parity" in nodeid and "exporter_db_parity_smoke" not in nodeid:
        return not _exports_available()
    if "test_db_query_contract.py" in nodeid:
        return not _exports_available()
    if "test_exports_contract.py" in nodeid:
        return not _exports_available()
    if "test_duckdb_idempotency.py::test_semantics_canonicalisation" in nodeid:
        return not _exports_available()
    return False


def _patch_test_utils(root: Path) -> None:
    try:
        module = importlib.import_module("resolver.tests.test_utils")
    except ImportError:
        return

    module.EXPORTS = root / "exports"
    module.REVIEW = root / "review"
    module.SNAPS = root / "snapshots"
    module.STATE = root / "state"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _running_on_ci():
        return

    if _exports_available():
        return

    skip_reason = pytest.mark.skip(
        reason="Skipping fixture-dependent test: fixture files not present (local/Codex mode)"
    )

    for item in items:
        nodeid = item.nodeid
        if _should_skip_node(nodeid):
            item.add_marker(skip_reason)


@pytest.fixture(scope="session")
def synthetic_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("resolver_synthetic_fixtures")
    manifest = _synthetic_testdata.write_csv(base)
    missing: Iterable[str] = [key for key, path in manifest.items() if not Path(path).exists()]
    if missing:
        raise RuntimeError(f"Synthetic fixture generation failed for keys: {sorted(missing)}")
    return base


@pytest.fixture(scope="session", autouse=True)
def _configure_synthetic_fixtures(request: pytest.FixtureRequest) -> Iterable[Path | None]:
    if _running_on_ci():
        yield None
        return

    if _exports_available() or not _flag_enabled("RESOLVER_ALLOW_SYNTHETIC_FIXTURES", "1"):
        yield None
        return

    base: Path = request.getfixturevalue("synthetic_data_dir")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("RESOLVER_TEST_DATA_DIR", str(base))
    _patch_test_utils(base)
    print(f"[resolver-tests] Using synthetic fixtures from {base}")
    try:
        yield base
    finally:
        monkeypatch.undo()


@lru_cache(maxsize=1)
def _maybe_conn_shared() -> ModuleType | None:
    """Return ``resolver.db.conn_shared`` if DuckDB is available."""

    if importlib.util.find_spec("duckdb") is None:
        return None
    from resolver.db import conn_shared as module

    return module


@pytest.fixture(scope="session")
def clear_duckdb_cache():
    """Return a helper that clears a specific DuckDB cache entry by URL."""

    module = _maybe_conn_shared()
    if module is None:
        pytest.skip("duckdb optional dependency is not installed")

    def _clear(db_url: str) -> None:
        module.clear_cached_connection(db_url)

    return _clear


@pytest.fixture(autouse=True)
def _duckdb_cache_hygiene():
    """Ensure all DuckDB caches are cleared around each test invocation."""

    module = _maybe_conn_shared()
    if module is None:
        yield
        return

    module.clear_all_cached_connections()
    yield
    module.clear_all_cached_connections()

