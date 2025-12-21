# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

from resolver.db import duckdb_io


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assert HS outputs are non-empty and emit run-scoped diagnostics."
    )
    parser.add_argument(
        "--db",
        required=True,
        help="DuckDB URL or filesystem path, e.g. duckdb:///data/resolver.duckdb",
    )
    parser.add_argument(
        "--hs-run-id",
        required=True,
        help="HS run id to validate (e.g. hs_20240101T000000).",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=["triage", "questions"],
        help="Pipeline stage to validate.",
    )
    return parser.parse_args()


def _excerpt(text: str, limit: int = 400) -> str:
    snippet = (text or "")[:limit]
    if len(text or "") > limit:
        snippet += "…"
    return snippet


def _escape_md_cell(text: str) -> str:
    return (text or "").replace("|", "\\|")


def _load_hs_manifest(
    con,
    hs_run_id: str,
) -> tuple[list[str], list[str]]:
    try:
        row = con.execute(
            """
            SELECT countries_json, requested_countries_json
            FROM hs_runs
            WHERE hs_run_id = ?
            ORDER BY COALESCE(generated_at, CURRENT_TIMESTAMP) DESC
            LIMIT 1
            """,
            [hs_run_id],
        ).fetchone()
    except Exception:
        row = None

    countries: list[str] = []
    requested: list[str] = []
    if row:
        countries_json, requested_json = row
        try:
            countries = json.loads(countries_json or "[]")
        except Exception:
            countries = []
        try:
            requested = json.loads(requested_json or "[]")
        except Exception:
            requested = []

    return countries, requested


def _write_header(lines: list[str], stage: str, hs_run_id: str) -> None:
    lines.append(f"# Horizon Scanner assertions ({stage})")
    lines.append("")
    lines.append(f"- HS run id: `{hs_run_id}`")
    lines.append(f"- HS model id: `{os.getenv('HS_MODEL_ID', '')}`")


def _render_llm_rows(rows: Iterable[tuple]) -> list[str]:
    lines: list[str] = []
    lines.append("| iso3 | error_text | elapsed_ms | provider | model_id | response_excerpt |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    found = False
    for iso3, error_text, elapsed_ms, provider, model_id, response_text in rows:
        found = True
        excerpt = _excerpt(response_text or "", 800)
        excerpt = _escape_md_cell(excerpt)
        error_cell = _escape_md_cell(error_text)
        lines.append(
            f"| {iso3 or ''} | {error_cell} | {elapsed_ms or ''} | "
            f"{provider or ''} | {model_id or ''} | {excerpt} |"
        )
    if not found:
        lines.append("| (none) | (none) | (none) | (none) | (none) | (none) |")
    return lines


def _render_triage_failure(
    con,
    hs_run_id: str,
    diag_path: Path,
) -> int:
    countries, requested = _load_hs_manifest(con, hs_run_id)
    resolved_env = os.getenv("HS_RESOLVED_ISO3S", "")
    resolved_from_env = [c for c in resolved_env.split(",") if c]

    try:
        llm_rows = con.execute(
            """
            SELECT iso3, error_text, elapsed_ms, provider, model_id, response_text
            FROM llm_calls
            WHERE hs_run_id = ?
              AND (phase = 'hs_triage' OR call_type = 'hs_triage')
            ORDER BY COALESCE(timestamp, CURRENT_TIMESTAMP) DESC
            LIMIT 25
            """,
            [hs_run_id],
        ).fetchall()
    except Exception:
        llm_rows = []

    lines: list[str] = []
    _write_header(lines, "triage", hs_run_id)
    lines.append("- hs_triage count: 0")
    lines.append(
        "- Resolved ISO3s: "
        + (", ".join(sorted({*(countries or []), *resolved_from_env})) or "(none)")
    )
    lines.append("- Requested countries (workflow input): " + (", ".join(requested) or "(none)"))
    lines.append("")
    lines.append("## hs_triage LLM call diagnostics (latest first)")
    lines.extend(_render_llm_rows(llm_rows))

    diag_path.parent.mkdir(parents=True, exist_ok=True)
    diag_path.write_text("\n".join(lines), encoding="utf-8")
    return 2


def _render_questions_failure(
    con,
    hs_run_id: str,
    diag_path: Path,
) -> int:
    try:
        triage_rows = con.execute(
            """
            SELECT iso3, hazard_code, tier, triage_score, need_full_spd
            FROM hs_triage
            WHERE run_id = ?
            ORDER BY triage_score DESC, iso3, hazard_code
            LIMIT 25
            """,
            [hs_run_id],
        ).fetchall()
    except Exception:
        triage_rows = []

    lines: list[str] = []
    if diag_path.exists():
        lines.append(diag_path.read_text(encoding="utf-8"))
        lines.append("")
        lines.append("---")
        lines.append("")

    _write_header(lines, "questions", hs_run_id)
    lines.append("- questions count: 0")
    lines.append("")
    lines.append("## Top hs_triage rows by triage_score (if any)")
    lines.append("| iso3 | hazard_code | tier | triage_score | need_full_spd |")
    lines.append("| --- | --- | --- | --- | --- |")
    if triage_rows:
        for iso3, hazard_code, tier, triage_score, need_full_spd in triage_rows:
            lines.append(
                f"| {iso3 or ''} | {hazard_code or ''} | {tier or ''} | "
                f"{triage_score if triage_score is not None else ''} | "
                f"{need_full_spd} |"
            )
    else:
        lines.append("| (none) | (none) | (none) | (none) | (none) |")

    diag_path.parent.mkdir(parents=True, exist_ok=True)
    diag_path.write_text("\n".join(lines), encoding="utf-8")
    return 3


def _render_success(
    stage: str,
    hs_run_id: str,
    diag_path: Path,
    triage_count: int,
    question_count: int,
) -> int:
    lines: list[str] = []
    _write_header(lines, stage, hs_run_id)
    lines.append(f"- hs_triage rows: {triage_count}")
    lines.append(f"- questions rows: {question_count}")
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    diag_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"HS assertion OK for stage={stage} hs_run_id={hs_run_id}")
    return 0


def run_assertion(db_url: str, hs_run_id: str, stage: str) -> int:
    diag_path = Path("diagnostics/hs_assertion.md")

    con = duckdb_io.get_db(db_url)
    try:
        try:
            triage_count_row = con.execute(
                "SELECT COUNT(*) FROM hs_triage WHERE run_id = ?", [hs_run_id]
            ).fetchone()
            triage_count = triage_count_row[0] if triage_count_row else 0
        except Exception:
            triage_count = 0

        try:
            question_count_row = con.execute(
                "SELECT COUNT(*) FROM questions WHERE hs_run_id = ?", [hs_run_id]
            ).fetchone()
            question_count = question_count_row[0] if question_count_row else 0
        except Exception:
            question_count = 0

        if stage == "triage" and triage_count == 0:
            return _render_triage_failure(con, hs_run_id, diag_path)

        if stage == "questions" and question_count == 0:
            return _render_questions_failure(con, hs_run_id, diag_path)

        return _render_success(stage, hs_run_id, diag_path, int(triage_count), int(question_count))
    finally:
        duckdb_io.close_db(con)


def main() -> None:
    args = _parse_args()
    exit_code = run_assertion(args.db, args.hs_run_id, args.stage)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
