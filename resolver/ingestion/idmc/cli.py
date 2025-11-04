"""Command line entrypoint for the IDMC connector."""
from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, cast

from resolver.ingestion.utils.country_utils import (
    read_countries_override_from_env,
    resolve_countries,
)

from .client import NETWORK_MODES, NetworkMode, fetch
from .config import load
from .diagnostics import (
    debug_block,
    diagnostics_dir,
    tick,
    timings_block,
    to_ms,
    write_connectors_line,
    write_drop_reasons,
    write_sample_preview,
    write_unmapped_hazards_preview,
    zero_rows_rescue,
)
from .export import build_resolution_ready_facts, summarise_facts
from .exporter import to_facts, write_facts_csv, write_facts_parquet
from .normalize import maybe_map_hazards, normalize_all
from .probe import ProbeOptions, probe_reachability
from .provenance import build_provenance, write_json
from .staging import ensure_staging, write_header_if_empty

LOGGER = logging.getLogger(__name__)
SUMMARY_TEMPLATE_PATH = Path("resolver/diagnostics/templates/idmc_summary.md.j2")
SUMMARY_TEMPLATE_FALLBACK = """# IDMC ingestion summary

Run at: {timestamp}
Git SHA: {git_sha}
Config source: {config_source}
Config path: {config_path}

Mode: {mode}
Token: {token_status}
Series: {series_display}
Date window: {date_window}

## Countries
- Resolved count: {countries_count}
- Sample: {countries_sample_display}
- Source: {countries_source_display}

## Endpoints
{endpoints_block}

## Attempts
{attempts_block}

## Performance
{performance_block}

## Rate limiting
{rate_limit_block}

## Datasets
{dataset_block}

## Staging
{staging_block}

Notes:
{notes_block}
"""


def _format_bullets(items: Iterable[str]) -> str:
    entries = [str(item).strip() for item in items if str(item).strip()]
    if not entries:
        return "- (none)"
    return "\n".join(f"- {entry}" for entry in entries)


def _count_csv_rows(path: Path) -> int | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            rows = sum(1 for _ in reader)
    except FileNotFoundError:
        return None
    except OSError as exc:  # pragma: no cover - defensive logging only
        LOGGER.debug("Failed to inspect %s: %s", path, exc)
        return None
    return max(rows - 1, 0)


def _resolve_git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
            )
            .strip()
        )
    except Exception:  # pragma: no cover - informational only
        return "unknown"


def _render_summary(context: Dict[str, Any]) -> str:
    template = SUMMARY_TEMPLATE_FALLBACK
    if SUMMARY_TEMPLATE_PATH.exists():
        try:
            template = SUMMARY_TEMPLATE_PATH.read_text(encoding="utf-8")
        except OSError as exc:  # pragma: no cover - defensive logging only
            LOGGER.debug("Failed to read summary template %s: %s", SUMMARY_TEMPLATE_PATH, exc)
    return template.format(**context)


def _write_summary(context: Dict[str, Any]) -> Path | None:
    try:
        diag_path = Path(diagnostics_dir())
        summary_path = diag_path / "summary.md"
        summary_text = _render_summary(context)
        summary_path.write_text(summary_text, encoding="utf-8")
        LOGGER.debug("Wrote IDMC summary to %s", summary_path)
        return summary_path
    except Exception as exc:  # pragma: no cover - diagnostics should not fail run
        LOGGER.warning("Failed to write IDMC summary: %s", exc)
        return None


def _enable_debug_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    root.setLevel(logging.DEBUG)
    for handler in root.handlers:
        handler.setLevel(logging.DEBUG)
    LOGGER.debug("Debug logging enabled for IDMC CLI")
from .why_zero import write_why_zero


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
        "--network-mode",
        choices=NETWORK_MODES,
        default=None,
        help="Network behaviour: live (default), cache_only, or fixture",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging and diagnostics",
    )
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
    parser.add_argument(
        "--rate",
        type=float,
        default=None,
        help="Requests per second cap (token bucket); 0 disables limiting",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum concurrent fetches (default: 1)",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        help="Stream responses larger than this many bytes to disk",
    )
    parser.add_argument(
        "--chunk-by-month",
        action="store_true",
        help="Partition long windows into month chunks",
    )
    parser.add_argument(
        "--enable-export",
        action="store_true",
        help="Enable resolution-ready facts export preview",
    )
    parser.add_argument(
        "--write-outputs",
        action="store_true",
        help="Write facts-ready outputs (CSV/Parquet) after normalize",
    )
    parser.add_argument(
        "--write-candidates",
        action="store_true",
        help="Write precedence candidate CSV after normalize",
    )
    parser.add_argument(
        "--candidates-out",
        default=os.getenv(
            "IDMC_CANDIDATES_OUT", "diagnostics/ingestion/idmc/idmc_candidates.csv"
        ),
        help="Destination for precedence candidates CSV",
    )
    parser.add_argument(
        "--run-precedence",
        action="store_true",
        help="Run precedence selection using generated candidates",
    )
    parser.add_argument(
        "--precedence-config",
        default=os.getenv("IDMC_PRECEDENCE_CONFIG", "tools/precedence_config.yml"),
        help="Path to precedence config when running --run-precedence",
    )
    parser.add_argument(
        "--precedence-out",
        default=os.getenv(
            "IDMC_PRECEDENCE_OUT", "diagnostics/ingestion/idmc/idmc_selected.csv"
        ),
        help="Destination CSV for precedence selection output",
    )
    parser.add_argument(
        "--map-hazards",
        action="store_true",
        help="Append hazard_code/label/class columns",
    )
    parser.add_argument(
        "--out-dir",
        default=os.getenv("IDMC_OUT_DIR", "artifacts/idmc"),
        help="Directory to write facts files (default artifacts/idmc)",
    )
    args = parser.parse_args(argv)

    if args.debug:
        _enable_debug_logging()
        os.environ["RESOLVER_DEBUG"] = "1"
    else:
        LOGGER.debug("Debug flag not set for this run")

    cfg = load()
    config_details = getattr(cfg, "_config_details", None)
    config_source_label = str(getattr(cfg, "_config_source", "ingestion"))
    config_path_used = getattr(cfg, "_config_path", None)
    config_warnings = [
        str(item)
        for item in getattr(cfg, "_config_warnings", ())
        if str(item)
    ]

    ensure_staging()

    env_force_cache_only = _env_truthy(os.getenv("IDMC_FORCE_CACHE_ONLY"))
    env_no_date_filter = _env_truthy(os.getenv("IDMC_NO_DATE_FILTER"))
    env_window_days = _env_int(os.getenv("IDMC_WINDOW_DAYS"))
    env_countries = _parse_csv(os.getenv("IDMC_ONLY_COUNTRIES"), transform=str.upper)
    env_series = _parse_csv(os.getenv("IDMC_SERIES"), transform=lambda value: value.lower())
    env_enable_export = _env_truthy(os.getenv("IDMC_ENABLE_EXPORT"))
    env_write_outputs = _env_truthy(os.getenv("IDMC_WRITE_OUTPUTS"))
    env_write_candidates = _env_truthy(os.getenv("IDMC_WRITE_CANDIDATES"))
    env_run_precedence = _env_truthy(os.getenv("IDMC_RUN_PRECEDENCE"))
    env_map_hazards = _env_truthy(os.getenv("IDMC_MAP_HAZARDS"))
    env_rate = os.getenv("IDMC_REQ_PER_SEC")
    env_max_concurrency = _env_int(os.getenv("IDMC_MAX_CONCURRENCY"))
    env_max_bytes = _env_int(os.getenv("IDMC_MAX_BYTES"))
    env_chunk_by_month = _env_truthy(os.getenv("IDMC_CHUNK_BY_MONTH"))

    cli_countries = _parse_csv(args.only_countries, transform=str.upper)
    cli_series = _parse_csv(args.series, transform=lambda value: value.lower())

    raw_network_mode = args.network_mode or os.getenv("IDMC_NETWORK_MODE")
    network_mode: NetworkMode
    if raw_network_mode:
        candidate = str(raw_network_mode).strip().lower()
        if candidate not in NETWORK_MODES:
            parser.error(
                f"Invalid network mode '{raw_network_mode}'. Choose from {', '.join(NETWORK_MODES)}."
            )
        network_mode = cast(NetworkMode, candidate)
    elif args.skip_network:
        network_mode = cast(NetworkMode, "fixture")
    elif env_force_cache_only:
        network_mode = cast(NetworkMode, "cache_only")
    elif env_force_cache_only is None and cfg.cache.force_cache_only:
        network_mode = cast(NetworkMode, "cache_only")
    else:
        network_mode = cast(NetworkMode, "live")

    cfg.cache.force_cache_only = network_mode == "cache_only"

    LOGGER.info("Effective network mode: %s", network_mode)

    config_countries_raw = list(getattr(cfg.api, "countries", []))
    override_raw = read_countries_override_from_env()
    config_countries = resolve_countries(config_countries_raw, override_raw)

    countries_source = "config list"
    if override_raw:
        countries_source = "env(IDMC_COUNTRIES)"
    elif not config_countries_raw:
        countries_source = "config (all countries)"

    selected_countries = list(config_countries)

    if env_countries:
        env_override = resolve_countries(env_countries)
        if env_override:
            selected_countries = env_override
            countries_source = "env(IDMC_ONLY_COUNTRIES)"
        else:
            LOGGER.warning(
                "Environment filter IDMC_ONLY_COUNTRIES did not match any ISO3 codes; retaining %d configured countries",
                len(selected_countries),
            )

    if cli_countries:
        cli_override = resolve_countries(cli_countries)
        if cli_override:
            selected_countries = cli_override
            countries_source = "cli(--only-countries)"
        else:
            LOGGER.warning(
                "CLI --only-countries override did not match any ISO3 codes; retaining %d configured countries",
                len(selected_countries),
            )

    config_series = [
        str(value).strip().lower()
        for value in getattr(cfg.api, "series", [])
        if str(value).strip()
    ]
    if not config_series:
        config_series = ["stock", "flow"]
    series_source = "config"
    selected_series_raw: List[str]
    if cli_series:
        series_source = "cli"
        selected_series_raw = cli_series
    elif env_series:
        series_source = "env"
        selected_series_raw = env_series
    else:
        selected_series_raw = config_series
    selected_series = list(dict.fromkeys(selected_series_raw or ["flow"]))
    if not selected_series:
        selected_series = ["flow"]

    LOGGER.debug(
        "Country scope source=%s override_env=%s resolved=%d sample=%s",
        countries_source,
        "yes" if override_raw else "no",
        len(selected_countries),
        ", ".join(selected_countries[:10]),
    )
    LOGGER.debug(
        "Series selection source=%s values=%s",
        series_source,
        ", ".join(selected_series),
    )

    enable_export = bool(args.enable_export)
    if env_enable_export is not None:
        enable_export = enable_export or bool(env_enable_export)

    write_outputs = bool(args.write_outputs)
    if env_write_outputs is not None:
        write_outputs = write_outputs or bool(env_write_outputs)

    write_candidates = bool(args.write_candidates)
    if env_write_candidates is not None:
        write_candidates = write_candidates or bool(env_write_candidates)

    run_precedence = bool(args.run_precedence)
    if env_run_precedence is not None:
        run_precedence = run_precedence or bool(env_run_precedence)

    if run_precedence:
        write_candidates = True

    map_hazards = bool(args.map_hazards)
    if env_map_hazards is not None:
        map_hazards = map_hazards or bool(env_map_hazards)

    rate_limit = args.rate
    if rate_limit is None and env_rate is not None:
        try:
            rate_limit = float(env_rate)
        except ValueError:  # pragma: no cover - defensive
            rate_limit = None
    if rate_limit is None:
        try:
            rate_limit = float(os.getenv("IDMC_REQ_PER_SEC", "0.5"))
        except ValueError:  # pragma: no cover - defensive
            rate_limit = 0.5

    max_concurrency = args.max_concurrency
    if max_concurrency is None and env_max_concurrency is not None:
        max_concurrency = env_max_concurrency
    if max_concurrency is None or max_concurrency <= 0:
        max_concurrency = 1

    max_bytes = args.max_bytes
    if max_bytes is None and env_max_bytes is not None:
        max_bytes = env_max_bytes
    if max_bytes is None or max_bytes <= 0:
        max_bytes = 10 * 1024 * 1024

    chunk_by_month = bool(args.chunk_by_month)
    if env_chunk_by_month is not None:
        chunk_by_month = chunk_by_month or bool(env_chunk_by_month)

    export_feature_flags = {
        "RESOLVER_EXPORT_ENABLE_IDMC": _env_truthy(
            os.getenv("RESOLVER_EXPORT_ENABLE_IDMC")
        ),
        "RESOLVER_EXPORT_ENABLE_FLOW": _env_truthy(
            os.getenv("RESOLVER_EXPORT_ENABLE_FLOW")
        ),
    }
    export_feature_enabled = any(value for value in export_feature_flags.values())

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

    if network_mode == "live":
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
        network_mode=network_mode,
        soft_timeouts=True,
        window_days=window_days,
        only_countries=selected_countries,
        base_url=args.base_url,
        cache_ttl=args.cache_ttl,
        rate_per_sec=rate_limit,
        max_concurrency=max_concurrency,
        max_bytes=max_bytes,
        chunk_by_month=chunk_by_month,
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
        map_hazards=map_hazards,
    )
    tidy, unmapped_hazards = maybe_map_hazards(tidy, map_hazards)
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

    hazards_payload = {"enabled": bool(map_hazards), "unmapped_hazard": 0}
    if map_hazards:
        unmapped_count = int(len(unmapped_hazards))
        hazards_payload["unmapped_hazard"] = unmapped_count
        if unmapped_count:
            hazards_payload["samples_unmapped"] = int(min(unmapped_count, 3))
            preview = write_unmapped_hazards_preview(unmapped_hazards)
            if preview:
                samples["hazard_unmapped_preview"] = preview

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
        write_header_if_empty()

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
        "countries_source": countries_source,
        "series_source": series_source,
        "network_mode": network_mode,
        "enable_export": enable_export,
        "write_outputs": write_outputs,
        "write_candidates": write_candidates,
        "out_dir": args.out_dir,
        "candidates_out": args.candidates_out,
        "run_precedence": run_precedence,
        "precedence_config": args.precedence_config,
        "precedence_out": args.precedence_out,
        "export_feature_enabled": export_feature_enabled,
        "map_hazards": map_hazards,
        "rate": rate_limit,
        "max_concurrency": max_concurrency,
        "max_bytes": max_bytes,
        "chunk_by_month": chunk_by_month,
        "debug": bool(args.debug),
    }

    feature_flag_states = {
        key: bool(value) if value is not None else False
        for key, value in export_feature_flags.items()
    }
    export_details = {
        "enabled": False,
        "rows": 0,
        "paths": {},
        "feature_flags": feature_flag_states,
    }

    if write_outputs and not export_feature_enabled:
        export_details["reason"] = "feature-flag-disabled"

    if write_outputs and export_feature_enabled:
        facts_frame = to_facts(tidy)
        csv_path = write_facts_csv(facts_frame, args.out_dir)
        parquet_path = write_facts_parquet(facts_frame, args.out_dir)
        export_details.update(
            {
                "enabled": True,
                "rows": len(facts_frame),
                "paths": {
                    "csv": csv_path,
                    "parquet": parquet_path or None,
                },
            }
        )

    debug = debug_block(
        selected_series=selected_series,
        selected_countries_count=len(selected_countries),
        countries_source=countries_source,
        series_source=series_source,
        cache_mode=diagnostics.get("mode", "offline"),
    )

    exports_payload = {}
    if enable_export:
        facts_frame = build_resolution_ready_facts(tidy)
        facts_summary = summarise_facts(facts_frame)
        buffer = io.StringIO()
        facts_frame.head(10).to_csv(buffer, index=False)
        facts_preview_path = write_sample_preview("facts", buffer.getvalue())
        samples["facts_preview"] = facts_preview_path
        exports_payload["facts"] = {**facts_summary, "preview": facts_preview_path}

    candidates_frame = None
    if write_candidates:
        from .candidates import to_candidates_from_normalized

        candidates_frame = to_candidates_from_normalized(tidy)
        candidates_path = Path(args.candidates_out)
        candidates_path.parent.mkdir(parents=True, exist_ok=True)
        candidates_frame.to_csv(candidates_path, index=False)
        samples["precedence_candidates"] = str(candidates_path)
        exports_payload["precedence_candidates"] = {
            "rows": int(len(candidates_frame)),
            "path": str(candidates_path),
        }

    if run_precedence:
        if candidates_frame is None:
            from .candidates import to_candidates_from_normalized

            candidates_frame = to_candidates_from_normalized(tidy)

        import yaml

        from tools.precedence_engine import apply_precedence

        config_path = Path(args.precedence_config)
        precedence_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        selected_frame = apply_precedence(candidates_frame, precedence_config)
        precedence_path = Path(args.precedence_out)
        precedence_path.parent.mkdir(parents=True, exist_ok=True)
        selected_frame.to_csv(precedence_path, index=False)
        samples["precedence_selected"] = str(precedence_path)
        exports_payload["precedence_selected"] = {
            "rows": int(len(selected_frame)),
            "path": str(precedence_path),
            "config": str(config_path),
        }

    rows_fetched = sum(len(frame) for frame in data.values())
    http_rollup = diagnostics.get("http") or {}
    cache_stats = diagnostics.get("cache") or {}
    effective_network_mode = str(diagnostics.get("network_mode", network_mode))
    http_status_counts = diagnostics.get("http_status_counts")

    diagnostics_payload = {
        "status": status,
        "reason": reason,
        "mode": diagnostics.get("mode", "offline"),
        "network_mode": effective_network_mode,
        "http_status_counts": http_status_counts,
        "http": http_rollup,
        "cache": cache_stats,
        "filters": diagnostics.get("filters"),
        "probe": probe_result,
        "timings": timings,
        "rows_fetched": rows_fetched,
        "rows_normalized": rows,
        "rows_written": rows,
        "drop_reasons": drops,
        "samples": samples,
        "zero_rows": zero_rows,
        "run_flags": run_flags,
        "debug": debug,
        "export": export_details,
        "exports": exports_payload or None,
        "hazards": hazards_payload,
        "performance": diagnostics.get("performance"),
        "rate_limit": diagnostics.get("rate_limit"),
        "chunks": diagnostics.get("chunks"),
    }

    config_payload = {
        "config_source_label": config_source_label,
        "config_path_used": str(config_path_used) if config_path_used else None,
    }
    if config_details is not None:
        ingestion_path = getattr(config_details, "ingestion_path", None)
        fallback_path = getattr(config_details, "fallback_path", None)
        if isinstance(ingestion_path, Path):
            config_payload["ingestion_config_path"] = ingestion_path.as_posix()
        if isinstance(fallback_path, Path):
            config_payload["legacy_config_path"] = fallback_path.as_posix()
    if config_warnings:
        config_payload["config_warnings"] = config_warnings
    diagnostics_payload["config"] = config_payload
    diagnostics_payload["config_source"] = config_source_label
    if config_warnings:
        diagnostics_payload["warnings"] = config_warnings

    base_url_used = (args.base_url or cfg.api.base_url or "").rstrip("/")
    endpoints_used = {
        name: (
            f"{base_url_used}{endpoint}" if base_url_used else str(endpoint)
        )
        for name, endpoint in cfg.api.endpoints.items()
    }
    token_env_name = getattr(cfg.api, "token_env", "IDMC_API_TOKEN")
    token_present = bool(os.getenv(token_env_name, "").strip())
    token_status = "present" if token_present else "absent"

    diag_filters = diagnostics.get("filters") or {}
    filter_start = diag_filters.get("window_start")
    filter_end = diag_filters.get("window_end")
    window_parts = [
        f"start={filter_start or '∅'}",
        f"end={filter_end or '∅'}",
    ]
    if window_days is not None:
        window_parts.append(f"window_days={window_days}")
    if effective_no_date_filter:
        window_parts.append("filter=disabled")
    date_window_display = ", ".join(window_parts)

    dataset_lines = [
        f"{name}: {int(len(frame))} rows"
        for name, frame in data.items()
    ]
    staging_dir = Path("resolver/staging/idmc")
    staged_counts: Dict[str, str] = {}
    staging_entries: List[str] = []
    for filename in ("flow.csv", "stock.csv"):
        path_candidate = staging_dir / filename
        row_count = _count_csv_rows(path_candidate)
        if row_count is None:
            staging_entries.append(f"{path_candidate.as_posix()}: missing")
            staged_counts[filename] = "missing"
        else:
            staging_entries.append(
                f"{path_candidate.as_posix()}: present (rows={row_count})"
            )
            staged_counts[filename] = str(row_count)

    cache_info = http_rollup.get("cache") or {}
    latency = http_rollup.get("latency_ms") or {}
    attempts_lines = [
        f"Requests attempted: {int(http_rollup.get('requests', 0) or 0)}",
        f"Retries: {int(http_rollup.get('retries', 0) or 0)}",
        f"Last status: {http_rollup.get('status_last') or 'n/a'}",
        f"Cache hits/misses: {int(cache_info.get('hits', 0) or 0)}/{int(cache_info.get('misses', 0) or 0)}",
        f"Retry-after events: {int(http_rollup.get('retry_after_events', 0) or 0)}",
    ]
    if latency:
        attempts_lines.append(
            "Latency ms (p50/p95/max): {p50}/{p95}/{max}".format(
                p50=int(latency.get("p50", 0) or 0),
                p95=int(latency.get("p95", 0) or 0),
                max=int(latency.get("max", 0) or 0),
            )
        )
    if http_status_counts:
        attempts_lines.append(
            "HTTP status counts: 2xx={two} 4xx={four} 5xx={five}".format(
                two=int(http_status_counts.get("2xx", 0) or 0),
                four=int(http_status_counts.get("4xx", 0) or 0),
                five=int(http_status_counts.get("5xx", 0) or 0),
            )
        )
    elif effective_network_mode != "live":
        attempts_lines.append("HTTP status counts: n/a in non-live mode")

    performance = diagnostics.get("performance") or {}
    performance_lines = [
        f"Duration (s): {performance.get('duration_s', 0)}",
        f"Wire bytes: {performance.get('wire_bytes', 0)}",
        f"Body bytes: {performance.get('body_bytes', 0)}",
        f"Throughput (KiB/s): {performance.get('throughput_kibps', 0)}",
        f"Records/sec: {performance.get('records_per_sec', 0)}",
        f"Rows fetched: {rows_fetched}",
        f"Rows normalized: {rows}",
        f"Rows written: {rows}",
    ]

    rate_limit_info = diagnostics.get("rate_limit") or {}
    rate_limit_lines = [
        f"Rate limit (req/s): {rate_limit_info.get('req_per_sec', 0)}",
        f"Max concurrency: {rate_limit_info.get('max_concurrency', 0)}",
        f"Retries: {rate_limit_info.get('retries', 0)}",
        f"Retry-after wait (s): {rate_limit_info.get('retry_after_wait_s', 0)}",
        f"Rate-limit wait (s): {rate_limit_info.get('rate_limit_wait_s', 0)}",
        f"Planned wait (s): {rate_limit_info.get('planned_wait_s', 0)}",
    ]

    response_mode = diagnostics.get("mode", "offline")

    notes_lines = [
        f"Network mode: {effective_network_mode}",
        f"Debug flag: {'on' if args.debug else 'off'}",
        f"Countries source: {countries_source}",
        f"Series source: {series_source}",
        f"Token env: {token_env_name}",
        "Staged rows: "
        + ", ".join(f"{name}={value}" for name, value in staged_counts.items()),
    ]
    if response_mode != effective_network_mode:
        notes_lines.append(f"Response mode: {response_mode}")
    if status != "ok":
        note_reason = f" ({reason})" if reason else ""
        notes_lines.append(f"Run status: {status}{note_reason}")
    if rows == 0:
        notes_lines.append("0 normalized rows (inspect drop reasons)")
    if config_warnings:
        notes_lines.append("Config warnings: " + "; ".join(config_warnings))
    chunks_info = diagnostics.get("chunks") or {}
    if chunks_info:
        notes_lines.append(
            f"Chunks enabled: {bool(chunks_info.get('enabled'))} (count={chunks_info.get('count', 0)})"
        )

    countries_sample_display = ", ".join(selected_countries[:10]) or "(none)"

    summary_context = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": _resolve_git_sha(),
        "config_source": config_source_label,
        "config_path": str(config_path_used) if config_path_used else "n/a",
        "mode": effective_network_mode,
        "token_status": token_status,
        "series_display": ", ".join(selected_series) or "(none)",
        "date_window": date_window_display,
        "countries_count": len(selected_countries),
        "countries_sample_display": countries_sample_display,
        "countries_source_display": countries_source,
        "endpoints_block": _format_bullets(
            f"{name}: {url}" for name, url in endpoints_used.items()
        ),
        "attempts_block": _format_bullets(attempts_lines),
        "performance_block": _format_bullets(performance_lines),
        "rate_limit_block": _format_bullets(rate_limit_lines),
        "dataset_block": _format_bullets(dataset_lines),
        "staging_block": _format_bullets(staging_entries),
        "notes_block": _format_bullets(notes_lines),
    }
    summary_path = _write_summary(summary_context)
    if summary_path:
        samples["summary"] = summary_path.as_posix()

    run_meta = {
        "cmd": "resolver.ingestion.idmc.cli",
        "args": vars(args),
        "env": {
            key: os.getenv(key)
            for key in [
                "IDMC_API_TOKEN",
                "IDMC_BASE_URL",
                "IDMC_REQ_PER_SEC",
                "IDMC_MAX_CONCURRENCY",
                "IDMC_FORCE_CACHE_ONLY",
                "IDMC_NETWORK_MODE",
            ]
        },
        "egress_ip": (probe_result or {}).get("egress_ip") if isinstance(probe_result, dict) else None,
        "base_url": base_url_used or None,
        "endpoints": endpoints_used,
        "timings_ms": timings,
        "mode": response_mode,
        "network_mode": effective_network_mode,
    }
    run_meta["config_source"] = config_source_label
    if config_path_used:
        run_meta["config_path"] = str(config_path_used)
    diagnostics_payload["run_env"] = run_meta["env"]
    drop_hist = {key: int(value) for key, value in (drops or {}).items()}
    normalize_stats = {
        "rows_fetched": int(rows_fetched),
        "rows_normalized": int(rows),
        "drop_reasons": drop_hist,
    }
    export_info = export_details.copy()
    export_paths = export_info.get("paths", {}) if isinstance(export_info, dict) else {}
    if not isinstance(export_paths, dict):
        export_paths = {}
    csv_export_path = export_paths.get("csv")
    csv_manifest_path = None
    if export_info.get("enabled") and csv_export_path:
        csv_manifest_path = f"{csv_export_path}.manifest.json"
        export_info.setdefault("manifests", {})["csv"] = csv_manifest_path
    notes = {"zero_rows": zero_rows} if zero_rows else {}
    should_write_why_zero = rows == 0 or bool(args.debug)
    if should_write_why_zero:
        window_start = date_window.get("start")
        window_end = date_window.get("end")
        requests_raw = http_rollup.get("requests", 0)
        try:
            requests_count = int(requests_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):  # pragma: no cover - defensive
            requests_count = 0
        countries_sample = selected_countries[:10]
        http_counts_zero = None
        if isinstance(http_status_counts, dict):
            http_counts_zero = {
                "2xx": int(http_status_counts.get("2xx", 0) or 0),
                "4xx": int(http_status_counts.get("4xx", 0) or 0),
                "5xx": int(http_status_counts.get("5xx", 0) or 0),
            }
        why_zero_payload = {
            "network_mode": effective_network_mode,
            "http_status_counts": http_counts_zero,
            "token_present": bool(token_present),
            "countries_count": len(selected_countries),
            "countries_sample": countries_sample,
            "countries_source": countries_source,
            "series": list(selected_series),
            "window": {
                "start": str(window_start) if window_start is not None else None,
                "end": str(window_end) if window_end is not None else None,
            },
            "date_window": {
                "start": str(window_start) if window_start is not None else None,
                "end": str(window_end) if window_end is not None else None,
            },
            "filters": {
                "date_out_of_window": int(drop_hist.get("date_out_of_window", 0)),
                "no_iso3": int(drop_hist.get("no_iso3", 0)),
                "no_value_col": int(drop_hist.get("no_value_col", 0)),
            },
            "network_attempted": requests_count > 0,
            "requests_attempted": requests_count,
            "config_source": config_source_label,
            "config_path_used": str(config_path_used) if config_path_used else None,
            "loader_warnings": config_warnings,
        }
        if override_raw and countries_source != "cli(--only-countries)" and countries_source != "env(IDMC_ONLY_COUNTRIES)":
            why_zero_payload["countries_override"] = "env(IDMC_COUNTRIES)"
        write_why_zero(why_zero_payload)
        diagnostics_payload["why_zero"] = why_zero_payload
    provenance = build_provenance(
        run_meta=run_meta,
        reachability=probe_result or {},
        http_rollup=http_rollup,
        cache_info=cache_stats,
        normalize_stats=normalize_stats,
        export_info=export_info,
        notes=notes,
    )
    manifest_path = os.path.join("diagnostics", "ingestion", "idmc", "manifest.json")
    write_json(manifest_path, provenance)
    diagnostics_payload["provenance"] = {"path": manifest_path}
    diagnostics_payload["export"] = export_info

    if csv_manifest_path:
        write_json(csv_manifest_path, provenance)

    write_connectors_line(diagnostics_payload)

    if status == "error":
        raise SystemExit(2)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
