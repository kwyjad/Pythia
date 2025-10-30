"""Command line entrypoint for the IDMC connector."""
from __future__ import annotations

import argparse
import io
import os
from typing import List

from .client import fetch
from .config import load
from .diagnostics import (
    debug_block,
    tick,
    timings_block,
    to_ms,
    write_connectors_line,
    write_drop_reasons,
    write_sample_preview,
    zero_rows_rescue,
)
from .normalize import normalize_all
from .probe import ProbeOptions, probe_reachability


def _parse_csv(value: str | None, *, transform=None) -> List[str]:
    if not value:
        return []
    transform = transform or (lambda item: item)
    return [
        transform(part.strip())
        for part in value.split(",")
        if part and part.strip()
    ]


def _env_truthy(value: str | None) -> bool | None:
    if value is None:
        return None
    return value.strip().lower() not in {"", "0", "false", "no"}


def _env_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("idmc")
    parser.add_argument("--skip-network", action="store_true", help="Force offline mode")
    parser.add_argument(
        "--strict-empty",
        action="store_true",
        help="Exit non-zero if zero rows were written",
    )
    parser.add_argument(
        "--no-date-filter", action="store_true", help="Disable date filtering"
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=None,
        help="Client-side window in days for IDU records (default: 30)",
    )
    parser.add_argument(
        "--only-countries",
        type=str,
        default=None,
        help="Comma-separated ISO3 codes to keep (client-side filter)",
    )
    parser.add_argument(
        "--series",
        type=str,
        default=None,
        help="Comma-separated series to normalise (default: flow)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Override the IDU base URL for this run",
    )
    parser.add_argument(
        "--cache-ttl",
        type=int,
        default=None,
        help="Override cache TTL in seconds for this run",
    )
    args = parser.parse_args(argv)

    cfg = load()

    env_force_cache_only = _env_truthy(os.getenv("IDMC_FORCE_CACHE_ONLY"))
    env_no_date_filter = _env_truthy(os.getenv("IDMC_NO_DATE_FILTER"))
    env_window_days = _env_int(os.getenv("IDMC_WINDOW_DAYS"))
    env_countries = _parse_csv(os.getenv("IDMC_ONLY_COUNTRIES"), transform=str.upper)
    env_series = _parse_csv(os.getenv("IDMC_SERIES"), transform=lambda value: value.lower())

    cli_countries = _parse_csv(args.only_countries, transform=str.upper)
    cli_series = _parse_csv(args.series, transform=lambda value: value.lower())

    selected_countries: List[str] = cli_countries or env_countries or [
        country.upper() for country in cfg.api.countries
    ]
    if not selected_countries:
        selected_countries = []
    else:
        selected_countries = list(dict.fromkeys(selected_countries))

    selected_series: List[str] = cli_series or env_series or ["flow"]
    if not selected_series:
        selected_series = ["flow"]
    else:
        selected_series = list(dict.fromkeys(selected_series))

    effective_no_date_filter = bool(args.no_date_filter)
    if env_no_date_filter is not None:
        effective_no_date_filter = effective_no_date_filter or bool(env_no_date_filter)

    window_days = None
    if not effective_no_date_filter:
        if args.window_days is not None:
            window_days = args.window_days
        elif env_window_days is not None:
            window_days = env_window_days
        else:
            window_days = 30
    overall_start = tick()

    probe_result = None
    probe_start = tick()
    force_cache_only = cfg.cache.force_cache_only
    if env_force_cache_only is not None:
        force_cache_only = bool(env_force_cache_only)

    cfg.cache.force_cache_only = force_cache_only

    if not args.skip_network and not force_cache_only:
        try:
            probe_result = probe_reachability(
                ProbeOptions(base_url=args.base_url or cfg.api.base_url)
            )
        except Exception as exc:  # pragma: no cover - defensive
            probe_result = {"error": str(exc)}
    probe_ms = to_ms(tick() - probe_start)

    fetch_start = tick()
    data, diagnostics = fetch(
        cfg,
        skip_network=bool(args.skip_network),
        soft_timeouts=True,
        window_days=window_days,
        only_countries=selected_countries,
        base_url=args.base_url,
        cache_ttl=args.cache_ttl,
    )
    fetch_ms = to_ms(tick() - fetch_start)

    date_window = {
        "start": cfg.api.date_window.start,
        "end": cfg.api.date_window.end,
    }
    if args.no_date_filter:
        date_window = {"start": None, "end": None}

    normalize_start = tick()
    tidy, drops = normalize_all(
        data,
        {
            "value_flow": cfg.field_aliases.value_flow,
            "value_stock": cfg.field_aliases.value_stock,
            "date": cfg.field_aliases.date,
            "iso3": cfg.field_aliases.iso3,
        },
        date_window,
        selected_series,
    )
    normalize_ms = to_ms(tick() - normalize_start)

    rows = len(tidy)
    buffer = io.StringIO()
    tidy.head(10).to_csv(buffer, index=False)
    preview_path = write_sample_preview("normalized", buffer.getvalue())
    drop_path = write_drop_reasons(drops)

    status = "ok"
    reason = None
    if rows == 0 and args.strict_empty:
        status = "error"
        reason = "strict-empty-0-rows"

    samples = {"normalized_preview": preview_path, "drop_reasons": drop_path}
    if diagnostics.get("raw_path"):
        samples["raw_snapshot"] = diagnostics["raw_path"]

    selectors = {
        "only_countries": selected_countries,
        "window_days": window_days,
        "series": selected_series,
    }
    zero_rows = None
    if rows == 0:
        zero_rows = zero_rows_rescue(
            selectors,
            "No rows after filters. See drop_reasons.",
        )

    timings = timings_block(
        probe_ms=probe_ms,
        fetch_ms=fetch_ms,
        normalize_ms=normalize_ms,
        total_ms=to_ms(tick() - overall_start),
    )

    run_flags = {
        "skip_network": bool(args.skip_network),
        "strict_empty": bool(args.strict_empty),
        "no_date_filter": effective_no_date_filter,
        "window_days": window_days,
        "only_countries": selected_countries,
        "series": selected_series,
        "force_cache_only": force_cache_only,
    }

    debug = debug_block(
        selected_series=selected_series,
        selected_countries_count=len(selected_countries),
        cache_mode=diagnostics.get("mode", "offline"),
    )

    write_connectors_line(
        {
            "status": status,
            "reason": reason,
            "mode": diagnostics.get("mode", "offline"),
            "http": diagnostics.get("http", {}),
            "cache": diagnostics.get("cache"),
            "filters": diagnostics.get("filters"),
            "probe": probe_result,
            "timings": timings,
            "rows_fetched": sum(len(frame) for frame in data.values()),
            "rows_normalized": rows,
            "rows_written": rows,
            "drop_reasons": drops,
            "samples": samples,
            "zero_rows": zero_rows,
            "run_flags": run_flags,
            "debug": debug,
        }
    )

    if status == "error":
        raise SystemExit(2)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
