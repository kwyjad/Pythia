from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd

from resolver.db.runtime_flags import USE_DUCKDB
from resolver.ingestion._fast_fixtures import resolve_fast_fixtures_mode
from resolver.tests.utils import run as run_proc

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:  # pragma: no cover - keep logging quiet in libraries
    LOGGER.addHandler(logging.NullHandler())

_NOOP_BOOTSTRAP_DONE = False

PERIOD_LABEL = "2024Q1"


@dataclass(frozen=True)
class FastExports:
    base_dir: Path
    staging_root: Path
    canonical_dir: Path
    snapshots_root: Path
    snapshots_period_dir: Path
    exports_dir: Path
    db_path: Path
    period: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _copy_canonical(source_dir: Path, canonical_dir: Path) -> None:
    canonical_dir.mkdir(parents=True, exist_ok=True)
    if not source_dir.exists():
        raise FileNotFoundError(f"Canonical fixture directory missing: {source_dir}")
    for path in source_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, canonical_dir / path.name)


def _write_csv_exports(parquet_dir: Path, exports_dir: Path) -> None:
    exports_dir.mkdir(parents=True, exist_ok=True)
    for parquet_path in parquet_dir.glob("*.parquet"):
        try:
            frame = pd.read_parquet(parquet_path)
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to read {parquet_path}: {exc}") from exc
        target = exports_dir / f"{parquet_path.stem}.csv"
        frame.to_csv(target, index=False)


def _run_load_and_derive(staging_root: Path, snapshots_root: Path, db_path: Path) -> None:
    repo = _repo_root()
    env = os.environ.copy()
    pythonpath_entries: Iterable[str] = [str(repo), env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join(filter(None, pythonpath_entries))
    cmd = [
        sys.executable,
        "-m",
        "resolver.tools.load_and_derive",
        "--period",
        PERIOD_LABEL,
        "--staging-root",
        str(staging_root),
        "--snapshots-root",
        str(snapshots_root),
        "--db",
        str(db_path),
        "--allow-negatives",
        "1",
    ]
    run_proc(cmd, check=True, env=env)


@lru_cache(maxsize=1)
def build_fast_exports() -> FastExports:
    mode, auto_fallback, reason = resolve_fast_fixtures_mode()
    if not USE_DUCKDB:
        if auto_fallback:
            extra = f" (missing DuckDB: {reason})" if reason else ""
            LOGGER.warning(
                "Fast fixtures disabled%s; running offline-smoke fallback only",
                extra,
            )
        else:
            LOGGER.info(
                "Fast fixtures disabled via RESOLVER_FAST_FIXTURES_MODE=noop; running offline-smoke fallback"
            )
        _run_offline_smoke_fallback()
        raise RuntimeError(
            "Fast fixtures bootstrap is unavailable in noop mode. Install DuckDB and "
            "set RESOLVER_FAST_FIXTURES_MODE=duckdb to enable full fixtures."
        )

    repo = _repo_root()
    canonical_source = (
        repo
        / "resolver"
        / "tests"
        / "fixtures"
        / "staging"
        / "minimal"
        / "canonical"
    )
    base_dir = Path(tempfile.mkdtemp(prefix="resolver-fast-fixtures-"))
    staging_root = base_dir / "data" / "staging"
    snapshots_root = base_dir / "data" / "snapshots"
    canonical_dir = staging_root / PERIOD_LABEL / "canonical"

    _copy_canonical(canonical_source, canonical_dir)

    db_path = base_dir / "resolver.duckdb"
    _run_load_and_derive(staging_root, snapshots_root, db_path)

    snapshots_period_dir = snapshots_root / PERIOD_LABEL
    exports_dir = base_dir / "exports"
    _write_csv_exports(snapshots_period_dir, exports_dir)

    snapshots_stub_dir = base_dir / "snapshots" / PERIOD_LABEL
    snapshots_stub_dir.mkdir(parents=True, exist_ok=True)
    for parquet_path in snapshots_period_dir.glob("*.parquet"):
        shutil.copy2(parquet_path, snapshots_stub_dir / parquet_path.name)

    (base_dir / "review").mkdir(parents=True, exist_ok=True)
    (base_dir / "state").mkdir(parents=True, exist_ok=True)

    return FastExports(
        base_dir=base_dir,
        staging_root=staging_root,
        canonical_dir=canonical_dir,
        snapshots_root=snapshots_root,
        snapshots_period_dir=snapshots_period_dir,
        exports_dir=exports_dir,
        db_path=db_path,
        period=PERIOD_LABEL,
    )


__all__ = ["FastExports", "build_fast_exports"]


def _run_offline_smoke_fallback() -> None:
    """Ensure the offline-smoke bootstrap has been executed once."""

    global _NOOP_BOOTSTRAP_DONE
    if _NOOP_BOOTSTRAP_DONE:
        return

    from resolver.ingestion import dtm_client

    exit_code = dtm_client.main(["--offline-smoke"])
    if exit_code != 0:
        raise RuntimeError(
            f"Offline smoke fallback failed with exit code {exit_code}"
        )
    _NOOP_BOOTSTRAP_DONE = True
