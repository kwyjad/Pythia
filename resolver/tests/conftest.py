from __future__ import annotations

import importlib
import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Iterable, Optional

import pytest

from . import _synthetic_testdata
from resolver.tests.fixtures.bootstrap_fast_exports import FastExports, build_fast_exports


def pytest_sessionstart(session: pytest.Session) -> None:
    """Set default skip flag for DTM tests unless the runner overrides it."""
    os.environ.setdefault("RESOLVER_SKIP_DTM", "1")

CI_ENV_VAR = "GITHUB_ACTIONS"
REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORTS_DIR = REPO_ROOT / "exports"
STAGING_DIR = REPO_ROOT / "staging"
STATE_DIR = REPO_ROOT / "state"
SNAPSHOTS_DIR = REPO_ROOT / "snapshots"

_FAST_EXPORTS: Optional[FastExports] = None


def _running_on_ci() -> bool:
    return os.environ.get(CI_ENV_VAR, "").lower() == "true"


def _flag_enabled(var: str, default: str = "1") -> bool:
    value = os.environ.get(var, default)
    return value.lower() not in {"0", "false", "no"}


@lru_cache(maxsize=1)
def _ensure_fast_exports() -> Optional[FastExports]:
    global _FAST_EXPORTS
    if _FAST_EXPORTS is not None:
        return _FAST_EXPORTS

    try:
        bundle = build_fast_exports()
    except Exception as exc:  # pragma: no cover - defensive guard
        print(f"[resolver-tests] fast exports bootstrap failed: {exc}")
        return None

    _FAST_EXPORTS = bundle
    os.environ.setdefault("RESOLVER_TEST_DATA_DIR", str(bundle.base_dir))
    os.environ.setdefault("RESOLVER_SNAPSHOTS_DIR", str(bundle.snapshots_root))
    os.environ.setdefault("RESOLVER_DB_PATH", str(bundle.db_path))
    _patch_test_utils(bundle.base_dir)
    return bundle


@lru_cache(maxsize=1)
def _exports_available() -> bool:
    env_dir = os.environ.get("RESOLVER_TEST_DATA_DIR")
    if env_dir:
        exports = Path(env_dir) / "exports"
        if exports.exists() and any(exports.glob("*.csv")):
            return True
    bundle = _ensure_fast_exports()
    if bundle is not None:
        exports_dir = bundle.exports_dir
        if exports_dir.exists() and any(exports_dir.glob("*.csv")):
            return True
    if EXPORTS_DIR.exists() and any(EXPORTS_DIR.glob("*.csv")):
        return True
    if STATE_DIR.exists() and any(STATE_DIR.glob("**/exports/*.csv")):
        return True
    return False


@lru_cache(maxsize=1)
def _staging_available() -> bool:
    bundle = _ensure_fast_exports()
    if bundle is not None and any(bundle.canonical_dir.glob("*.csv")):
        return True
    return STAGING_DIR.exists() and any(STAGING_DIR.glob("*.csv"))


def _should_skip_node(nodeid: str) -> bool:
    nodeid = nodeid.lower()
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


@pytest.fixture(scope="session")
def fast_exports() -> FastExports:
    bundle = _ensure_fast_exports()
    if bundle is None:
        pytest.skip("fast exports fixture unavailable")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("RESOLVER_TEST_DATA_DIR", str(bundle.base_dir))
    monkeypatch.setenv("RESOLVER_SNAPSHOTS_DIR", str(bundle.snapshots_root))
    monkeypatch.setenv("RESOLVER_DB_PATH", str(bundle.db_path))
    monkeypatch.setenv("RESOLVER_DB_URL", f"duckdb:///{bundle.db_path}")
    try:
        yield bundle
    finally:
        monkeypatch.undo()


@pytest.fixture(scope="session")
def fast_staging_dir() -> Path:
    staging_dir = REPO_ROOT / "tests" / "fixtures" / "staging" / "minimal" / "staging"
    if not staging_dir.exists():
        pytest.skip("staging fixtures unavailable")
    return staging_dir


if os.environ.get("RUN_EXPORTS_TESTS") == "1":

    @pytest.fixture(scope="session", autouse=True)
    def _opt_in_exports(tmp_path_factory: pytest.TempPathFactory) -> Path:
        """Provide a tiny exports directory for opt-in contract tests."""

        base = tmp_path_factory.mktemp("resolver_opt_in_exports")
        exports_dir = base / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)

        facts_path = exports_dir / "facts.csv"
        facts_path.write_text(
            "event_id,iso3,hazard_code,metric,value,unit,as_of_date,publication_date,ym,"
            "hazard_label,hazard_class,publisher,source_type\n"
            "EVT-OPT,PHL,TC,affected,10,persons,2024-01-02,2024-01-03,2024-01,"
            "Typhoon,Storm,ReliefWeb,agency\n",
            encoding="utf-8",
        )

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("RESOLVER_TEST_DATA_DIR", str(base))
        _patch_test_utils(base)
        try:
            yield exports_dir
        finally:
            monkeypatch.undo()

