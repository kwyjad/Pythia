from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Optional

try:
    import pytest  # noqa: F401  # ensure pytest picks up hooks
except Exception:  # pragma: no cover - pytest will import this module regardless
    pytest = None  # type: ignore


DUCKDB_SCHEME = "duckdb:///"


def _has_xdist() -> bool:
    try:
        import xdist  # noqa: F401

        return True
    except Exception:
        return False


def _strip_xdist_tokens(values: list[str]) -> list[str]:
    stripped: list[str] = []
    skip_next = False
    for token in values:
        if skip_next:
            skip_next = False
            continue
        if token == "-n":
            skip_next = True
            continue
        if token.startswith("-n"):
            # covers -nauto or -n=auto
            continue
        if token == "--numprocesses":
            skip_next = True
            continue
        if token.startswith("--numprocesses"):
            continue
        stripped.append(token)
    return stripped


XDIST_AVAILABLE = _has_xdist()
XDIST_NOTICE_EMITTED = False


if not XDIST_AVAILABLE:
    addopts = os.environ.get("PYTEST_ADDOPTS")
    if addopts:
        tokens = addopts.split()
        filtered = _strip_xdist_tokens(tokens)
        if filtered != tokens:
            new_value = " ".join(filtered)
            if new_value:
                os.environ["PYTEST_ADDOPTS"] = new_value
            else:
                os.environ.pop("PYTEST_ADDOPTS", None)
            if not XDIST_NOTICE_EMITTED:
                print("[note] pytest-xdist not available; running single-process")
                XDIST_NOTICE_EMITTED = True
    argv_tokens = _strip_xdist_tokens(sys.argv[1:])
    if argv_tokens != sys.argv[1:]:
        if not XDIST_NOTICE_EMITTED:
            print("[note] pytest-xdist not available; running single-process")
            XDIST_NOTICE_EMITTED = True
        sys.argv[1:] = argv_tokens


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


if XDIST_AVAILABLE:

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
