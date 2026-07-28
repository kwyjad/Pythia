# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

"""Quick, stable post-run diagnostics for Pythia pipeline outputs."""

import argparse
import os
from typing import Iterable

import duckdb

DEFAULT_DB_URL = "duckdb:///data/resolver.duckdb"


def _print_count(conn, table: str) -> None:
    try:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"Table {table}: {int(count)} rows")
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"[warn] Could not count table {table}: {type(exc).__name__}: {exc}")


def _print_forecast_breakdown(conn) -> None:
    try:
        n_ens = conn.execute("SELECT COUNT(*) FROM forecasts_ensemble").fetchone()[0]
        n_raw = conn.execute("SELECT COUNT(*) FROM forecasts_raw").fetchone()[0]
        print(f"Table forecasts_ensemble: {int(n_ens)} rows")
        print(f"Table forecasts_raw: {int(n_raw)} rows")
        n_run_ids = conn.execute("SELECT COUNT(DISTINCT run_id) FROM forecasts_ensemble").fetchone()[0]
        print(f"Distinct run_id values in forecasts_ensemble: {int(n_run_ids)}")
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"[warn] Could not inspect forecasts tables: {type(exc).__name__}: {exc}")
        return

    try:
        rows: Iterable[tuple[str, int]] = conn.execute(
            """
            SELECT question_id, COUNT(*) AS n_rows
            FROM forecasts_ensemble
            GROUP BY question_id
            ORDER BY question_id
            LIMIT 20
            """
        ).fetchall()
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"[warn] Could not summarize forecasts_ensemble: {type(exc).__name__}: {exc}")
        return

    rows = list(rows)
    if not rows:
        print("No rows in forecasts_ensemble for this run.")
        return

    print("Forecasts per question_id (first 20):")
    for question_id, n_rows in rows:
        print(f"  {question_id}: {int(n_rows)} rows")


def _print_prompt_cache_stats(conn) -> None:
    """Per-provider prompt-cache read totals from llm_calls.usage_json.

    This is the proof that prompt caching is actually connected: with
    PYTHIA_PROMPT_CACHE_ENABLED=1 and a healthy wiring, Anthropic calls show
    cache_read_input_tokens > 0 and OpenAI calls show cached_tokens > 0
    after the first question. Zero reads across many calls means the flag is
    on but no call site is passing cache_segments/prompt_cache_key — the
    exact silent failure the July 2026 audit found in production.
    """

    try:
        rows = conn.execute(
            """
            SELECT provider,
                   COUNT(*) AS calls,
                   SUM(COALESCE(TRY_CAST(json_extract_string(usage_json, '$.cache_read_input_tokens') AS BIGINT), 0)) AS anth_reads,
                   SUM(COALESCE(TRY_CAST(json_extract_string(usage_json, '$.cached_tokens') AS BIGINT), 0)) AS oai_reads,
                   SUM(CASE WHEN json_extract_string(usage_json, '$.service_tier') = 'batch' THEN 1 ELSE 0 END) AS batch_calls
            FROM llm_calls
            WHERE usage_json IS NOT NULL
              AND timestamp > now() - INTERVAL 2 DAY
            GROUP BY provider
            ORDER BY provider
            """
        ).fetchall()
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"[warn] Could not compute prompt-cache stats: {type(exc).__name__}: {exc}")
        return

    cache_flag_on = os.getenv("PYTHIA_PROMPT_CACHE_ENABLED", "0").strip().lower() in ("1", "true", "yes")
    print("Prompt cache (last 2 days of llm_calls):")
    for provider, calls, anth_reads, oai_reads, batch_calls in rows:
        reads = int(anth_reads or 0) + int(oai_reads or 0)
        print(
            f"  {provider or '?'}: calls={int(calls)} cache_read_tokens={reads} "
            f"batch_tier_calls={int(batch_calls or 0)}"
        )
        if (
            cache_flag_on
            and int(calls) >= 10
            and reads == 0
            and (provider or "").lower() in ("anthropic", "openai")
        ):
            print(
                f"::warning::prompt cache produced ZERO reads for {provider} across "
                f"{int(calls)} calls despite PYTHIA_PROMPT_CACHE_ENABLED=1 — "
                "check that the call sites pass cache_segments/prompt_cache_key"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_URL,
        help="DuckDB path or duckdb:/// URL (default: duckdb:///data/resolver.duckdb)",
    )
    args = parser.parse_args(argv)

    print("Python post-run diagnostic script starting...")

    db_arg = args.db
    db_path = db_arg[len("duckdb:///") :] if db_arg.startswith("duckdb:///") else db_arg
    if not os.path.exists(db_path):
        print(f"[warn] DuckDB database not found at {db_path}")
        return 0

    try:
        conn = duckdb.connect(db_path, read_only=True)
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"[warn] Could not open DuckDB at {db_path}: {type(exc).__name__}: {exc}")
        return 0

    try:
        _print_count(conn, "hs_runs")
        _print_count(conn, "hs_scenarios")
        _print_count(conn, "questions")
        _print_forecast_breakdown(conn)
        _print_prompt_cache_stats(conn)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
