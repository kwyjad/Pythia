"""Command line entrypoint for the IDMC connector skeleton."""
from __future__ import annotations

import argparse
import io

from .client import fetch
from .config import load
from .diagnostics import write_connectors_line, write_sample_preview
from .normalize import normalize_all


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
        default=30,
        help="Client-side window in days for IDU records (default: 30)",
    )
    parser.add_argument(
        "--only-countries",
        type=str,
        default="",
        help="Comma-separated ISO3 codes to keep (client-side filter)",
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
    cli_countries = [part.strip() for part in args.only_countries.split(",") if part.strip()]

    window_days = None if args.no_date_filter else args.window_days
    data, diagnostics = fetch(
        cfg,
        skip_network=bool(args.skip_network),
        soft_timeouts=True,
        window_days=window_days,
        only_countries=cli_countries,
        base_url=args.base_url,
        cache_ttl=args.cache_ttl,
    )

    date_window = {
        "start": cfg.api.date_window.start,
        "end": cfg.api.date_window.end,
    }
    if args.no_date_filter:
        date_window = {"start": None, "end": None}

    tidy, drops = normalize_all(
        data,
        {
            "value_flow": cfg.field_aliases.value_flow,
            "value_stock": cfg.field_aliases.value_stock,
            "date": cfg.field_aliases.date,
            "iso3": cfg.field_aliases.iso3,
        },
        date_window,
    )

    rows = len(tidy)
    buffer = io.StringIO()
    tidy.head(10).to_csv(buffer, index=False)
    preview_path = write_sample_preview("normalized", buffer.getvalue())

    status = "ok"
    reason = None
    if rows == 0 and args.strict_empty:
        status = "error"
        reason = "strict-empty-0-rows"

    samples = {"normalized_preview": preview_path}
    if diagnostics.get("raw_path"):
        samples["raw_snapshot"] = diagnostics["raw_path"]

    write_connectors_line(
        {
            "status": status,
            "reason": reason,
            "mode": diagnostics.get("mode", "offline"),
            "http": diagnostics.get("http", {}),
            "cache": diagnostics.get("cache"),
            "filters": diagnostics.get("filters"),
            "probe": diagnostics.get("probe"),
            "rows_fetched": sum(len(frame) for frame in data.values()),
            "rows_normalized": rows,
            "rows_written": rows,
            "drop_reasons": drops,
            "samples": samples,
        }
    )

    if status == "error":
        raise SystemExit(2)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
