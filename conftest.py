from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

try:
    import pytest  # noqa: F401  # ensure pytest picks up hooks
except Exception:  # pragma: no cover - pytest will import this module regardless
    pytest = None  # type: ignore


DUCKDB_SCHEME = "duckdb:///"


def _derive_worker_path(base_path: Path, workerid: Optional[str]) -> Path:
    if not workerid or workerid in {"master"}:
        return base_path
    stemmed = base_path.with_stem(f"{base_path.stem}-{workerid}")
    return stemmed


def _prepare_worker_db(base_url: str | None, workerid: Optional[str]) -> str | None:
    if not base_url or not base_url.startswith(DUCKDB_SCHEME):
        return base_url

    base_path = Path(base_url.replace(DUCKDB_SCHEME, "", 1))
    worker_path = _derive_worker_path(base_path, workerid)
    worker_path.parent.mkdir(parents=True, exist_ok=True)

    if worker_path != base_path and base_path.exists():
        try:
            if worker_path.exists():
                worker_path.unlink()
        except OSError:
            pass
        try:
            shutil.copy2(base_path, worker_path)
        except Exception:
            # Fall back to empty file if copy fails; tests will create tables as needed.
            worker_path.touch(exist_ok=True)
    elif not worker_path.exists():
        worker_path.touch(exist_ok=True)

    return f"{DUCKDB_SCHEME}{worker_path}"


def pytest_configure_node(node):  # pragma: no cover - xdist only
    base_url = os.environ.get("RESOLVER_DB_URL")
    node.workerinput["resolver_db_url_base"] = base_url


def pytest_configure(config):  # pragma: no cover - runtime hook
    workerinput = getattr(config, "workerinput", None)
    workerid = None
    base_url = os.environ.get("RESOLVER_DB_URL")
    if workerinput is not None:
        workerid = workerinput.get("workerid")
        base_url = workerinput.get("resolver_db_url_base", base_url)
    new_url = _prepare_worker_db(base_url, workerid)
    if new_url:
        os.environ["RESOLVER_DB_URL"] = new_url
    elif base_url:
        # Preserve empty strings to avoid inheriting stale URLs.
        os.environ.pop("RESOLVER_DB_URL", None)
