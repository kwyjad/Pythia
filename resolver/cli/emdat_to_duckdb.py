"""EM-DAT → DuckDB helper CLI for People Affected (PA) writes."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

from resolver.db import duckdb_io
from resolver.ingestion import emdat_stub
from resolver.ingestion.emdat_client import EmdatClient, OfflineRequested
from resolver.ingestion.emdat_normalize import normalize_emdat_pa, write_emdat_pa_to_duckdb

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)

DEFAULT_DB_PATH = Path("./resolver_data/resolver.duckdb")


def _normalize_db_url_arg(raw: str | None) -> tuple[str, str]:
    """Return (DuckDB URL, filesystem path) for ``raw`` input."""

    candidate = (raw or "").strip() or str(DEFAULT_DB_PATH)
    if "://" in candidate and not candidate.lower().startswith("duckdb://"):
        return candidate, candidate
    if candidate.lower().startswith("duckdb://"):
        if candidate.lower().startswith("duckdb:///"):
            fs_part = candidate[len("duckdb:///") :]
            fs_path = Path(fs_part).expanduser().resolve()
            return f"duckdb:///{fs_path.as_posix()}", str(fs_path)
        return candidate, candidate
    fs_path = Path(candidate).expanduser().resolve()
    return f"duckdb:///{fs_path.as_posix()}", str(fs_path)


def _parse_countries(raw: str | None) -> list[str]:
    if not raw:
        return []
    values: list[str] = []
    for item in raw.split(","):
        value = item.strip().upper()
        if value:
            values.append(value)
    return values


def _to_iterable(value: Sequence[str] | None) -> Iterable[str] | None:
    if not value:
        return None
    return list(value)


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="from_year", type=int, required=True, help="Inclusive start year")
    parser.add_argument("--to", dest="to_year", type=int, required=True, help="Inclusive end year")
    parser.add_argument(
        "--countries",
        dest="countries",
        default=None,
        help="Optional comma-separated ISO3 country codes",
    )
    parser.add_argument(
        "--db",
        dest="db",
        default=str(DEFAULT_DB_PATH),
        help="DuckDB URL or filesystem path (default: %(default)s)",
    )
    parser.add_argument(
        "--network",
        action="store_true",
        help="Enable live EM-DAT requests (requires EMDAT_API_KEY)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit override for debugging",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("RESOLVER_LOG_LEVEL", "INFO"),
        help="Set logging level (default: %(default)s)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    LOGGER.setLevel(level)

    duckdb_url, duckdb_path = _normalize_db_url_arg(args.db)
    if duckdb_path and "://" not in duckdb_path:
        Path(duckdb_path).parent.mkdir(parents=True, exist_ok=True)
    elif duckdb_url.lower().startswith("duckdb:///"):
        fs_part = duckdb_url[len("duckdb:///") :]
        Path(fs_part).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    if not duckdb_io.DUCKDB_AVAILABLE:
        reason = duckdb_io.duckdb_unavailable_reason()
        print(f"DuckDB is unavailable: {reason}", file=sys.stderr)
        return 2

    countries = _parse_countries(args.countries)
    LOGGER.info(
        "emdat_to_duckdb.start | from=%s to=%s countries=%s network=%s limit=%s db_url=%s",
        args.from_year,
        args.to_year,
        countries,
        bool(args.network),
        args.limit if args.limit is not None else -1,
        duckdb_url,
    )

    fetch_start = time.perf_counter()
    probe_info = None

    try:
        if args.network:
            client = EmdatClient(network=True)
            probe = client.probe(
                from_year=args.from_year,
                to_year=args.to_year,
                iso=_to_iterable(countries),
            )
            probe_info = probe.get("info") if isinstance(probe, dict) else None
            if not probe.get("ok"):
                message = probe.get("error") or "probe failed"
                LOGGER.error("emdat_to_duckdb.probe_failed | reason=%s", message)
                print(f"Probe failed: {message}", file=sys.stderr)
                return 2
            LOGGER.info(
                "emdat_to_duckdb.probe_ok | api_version=%s total_available=%s",
                probe.get("api_version"),
                probe.get("total_available"),
            )
            raw = client.fetch_raw(
                args.from_year,
                args.to_year,
                iso=_to_iterable(countries),
                include_hist=False,
                limit=args.limit,
            )
        else:
            raw = emdat_stub.fetch_raw(
                args.from_year,
                args.to_year,
                iso=_to_iterable(countries),
                limit=args.limit,
            )
    except OfflineRequested as exc:
        LOGGER.error("emdat_to_duckdb.network_disabled | error=%s", exc)
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.exception("emdat_to_duckdb.fetch_failed")
        print(f"Fetch failed: {exc}", file=sys.stderr)
        return 3

    fetch_elapsed = (time.perf_counter() - fetch_start) * 1000
    LOGGER.info("emdat_to_duckdb.fetch_done | rows=%s elapsed_ms=%.2f", len(raw), fetch_elapsed)

    normalize_start = time.perf_counter()
    normalized = normalize_emdat_pa(raw, info=probe_info)
    normalize_elapsed = (time.perf_counter() - normalize_start) * 1000
    LOGGER.info(
        "emdat_to_duckdb.normalize_done | rows=%s elapsed_ms=%.2f",
        len(normalized),
        normalize_elapsed,
    )

    conn = duckdb_io.get_db(duckdb_url)
    try:
        write_start = time.perf_counter()
        result = write_emdat_pa_to_duckdb(conn, normalized)
        write_elapsed = (time.perf_counter() - write_start) * 1000
        LOGGER.info(
            "emdat_to_duckdb.write_done | rows_in=%s rows_written=%s rows_delta=%s elapsed_ms=%.2f",
            result.rows_in,
            result.rows_written,
            result.rows_delta,
            write_elapsed,
        )
    finally:
        duckdb_io.close_db(conn)

    print(f"✅ Wrote {result.rows_written} rows to DuckDB")
    print(f" - emdat_pa Δ={result.rows_delta} total={result.rows_after}")

    LOGGER.info("emdat_to_duckdb.finished | rows_after=%s", result.rows_after)
    return 0


def main() -> None:
    sys.exit(run())


def run_emdat_pa_backfill(
    from_year: int,
    to_year: int,
    *,
    db_url: str,
    iso3_list: Optional[Iterable[str]] = None,
    network: bool = True,
    limit: Optional[int] = None,
) -> int:
    """Programmatic wrapper around the EM-DAT PA backfill.

    This leaves CLI behaviour unchanged while enabling tests/CI to run the
    backfill with explicit parameters.

    Returns the same integer status code as ``run(argv)``.
    """

    argv: list[str] = ["--from", str(from_year), "--to", str(to_year), "--db", db_url]

    if iso3_list:
        iso_csv = ",".join(sorted({value.upper() for value in iso3_list if value}))
        if iso_csv:
            argv.extend(["--countries", iso_csv])

    if network:
        argv.append("--network")

    if limit is not None:
        argv.extend(["--limit", str(limit)])

    return run(argv)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
