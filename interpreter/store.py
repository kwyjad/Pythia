# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""DB writes for interpreter reports (the ``interpretations`` table).

The table's authoritative definition lives in pythia/db/schema.py; the DDL
here is the same idempotent CREATE the compute tools carry so the runner
works against a DB that predates ensure_schema's latest run.

``is_test`` is None-and-inherit (pythia.test_mode.is_test_mode) — never a
hardcoded False.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

LOGGER = logging.getLogger(__name__)

INTERPRETATIONS_DDL = """
CREATE TABLE IF NOT EXISTS interpretations (
    interpretation_id TEXT,
    kind TEXT,
    run_id TEXT,
    hs_run_id TEXT,
    scored_run_id TEXT,
    version INTEGER,
    template_version TEXT,
    model_name TEXT,
    thinking_level TEXT,
    prompt_hash TEXT,
    pack_hash TEXT,
    content_json TEXT,
    content_md TEXT,
    figures_json TEXT,
    status TEXT,
    validation_json TEXT,
    cost_usd DOUBLE,
    input_tokens INTEGER,
    output_tokens INTEGER,
    created_at TIMESTAMP DEFAULT now(),
    is_test BOOLEAN DEFAULT FALSE
)
"""


def ensure_table(con) -> None:
    con.execute(INTERPRETATIONS_DDL)
    # Older DBs created before a column landed: idempotent adds, mirroring
    # pythia/db/schema.py's migration dict.
    existing = {
        str(r[1]).lower()
        for r in con.execute("PRAGMA table_info('interpretations')").fetchall()
    }
    for column, decl in (("figures_json", "TEXT"), ("is_test", "BOOLEAN DEFAULT FALSE")):
        if column not in existing:
            con.execute(f"ALTER TABLE interpretations ADD COLUMN {column} {decl}")


def _run_key(kind: str, run_id: str | None, scored_run_id: str | None) -> str:
    return (run_id if kind in ("current", "combined") else scored_run_id) or ""


def next_version(con, kind: str, run_id: str | None, scored_run_id: str | None) -> int:
    ensure_table(con)
    key = _run_key(kind, run_id, scored_run_id)
    column = "run_id" if kind in ("current", "combined") else "scored_run_id"
    row = con.execute(
        f"SELECT MAX(version) FROM interpretations WHERE kind = ? "
        f"AND COALESCE({column}, '') = ?",
        [kind, key],
    ).fetchone()
    return int(row[0] or 0) + 1


def existing_ok_version(
    con, kind: str, run_id: str | None, scored_run_id: str | None
) -> int | None:
    """Highest status='ok' version for this (kind, run key), else None."""
    ensure_table(con)
    key = _run_key(kind, run_id, scored_run_id)
    column = "run_id" if kind in ("current", "combined") else "scored_run_id"
    row = con.execute(
        f"SELECT MAX(version) FROM interpretations WHERE kind = ? "
        f"AND COALESCE({column}, '') = ? AND status = 'ok'",
        [kind, key],
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else None


def save_interpretation(
    con,
    *,
    kind: str,
    run_id: str | None,
    hs_run_id: str | None,
    scored_run_id: str | None,
    template_version: str,
    model_name: str,
    thinking_level: str | None,
    prompt_hash: str,
    pack_hash: str,
    content: dict[str, Any] | None,
    content_md: str | None,
    figures: dict[str, Any] | None,
    status: str,
    validation: dict[str, Any] | None,
    cost_usd: float | None,
    input_tokens: int | None,
    output_tokens: int | None,
    is_test: bool | None = None,
) -> tuple[str, int]:
    """Insert a new versioned row; returns (interpretation_id, version)."""
    ensure_table(con)
    if is_test is None:
        try:
            from pythia.test_mode import is_test_mode

            is_test = is_test_mode()
        except Exception:  # noqa: BLE001
            is_test = False
    version = next_version(con, kind, run_id, scored_run_id)
    key = _run_key(kind, run_id, scored_run_id) or "none"
    interpretation_id = f"int_{kind}_{key}_v{version}"
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    con.execute(
        """
        INSERT INTO interpretations
            (interpretation_id, kind, run_id, hs_run_id, scored_run_id,
             version, template_version, model_name, thinking_level,
             prompt_hash, pack_hash, content_json, content_md, figures_json,
             status, validation_json, cost_usd, input_tokens, output_tokens,
             created_at, is_test)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            interpretation_id, kind, run_id, hs_run_id, scored_run_id,
            version, template_version, model_name, thinking_level,
            prompt_hash, pack_hash,
            json.dumps(content, ensure_ascii=False, default=str) if content is not None else None,
            content_md,
            json.dumps(figures, ensure_ascii=False, default=str) if figures is not None else None,
            status,
            json.dumps(validation, ensure_ascii=False, default=str) if validation is not None else None,
            cost_usd, input_tokens, output_tokens, now, bool(is_test),
        ],
    )
    return interpretation_id, version


def get_interpretation(con, interpretation_id: str) -> dict[str, Any] | None:
    """One stored report by id, with its content parsed.

    Written for the side-by-side model harness, which needs to read back what
    each model actually produced rather than trust the runner's result dict.
    Returns None rather than raising: a reader must never fail a run.
    """
    try:
        ensure_table(con)
        rows = con.execute(
            """
            SELECT interpretation_id, kind, version, status, model_name,
                   content_json, content_md, cost_usd, created_at
            FROM interpretations
            WHERE interpretation_id = ?
            ORDER BY version DESC LIMIT 1
            """,
            [interpretation_id],
        ).fetchall()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("get_interpretation(%s) failed: %s", interpretation_id, exc)
        return None
    if not rows:
        return None
    r = rows[0]
    out: dict[str, Any] = {
        "interpretation_id": r[0], "kind": r[1], "version": r[2],
        "status": r[3], "model_name": r[4], "content": None,
        "content_md": r[6], "cost_usd": r[7], "created_at": r[8],
    }
    try:
        out["content"] = json.loads(r[5]) if r[5] else None
    except Exception:  # noqa: BLE001
        pass
    return out


def latest_scored(con, *, include_test: bool = False) -> dict[str, Any] | None:
    """The newest status='ok' scored interpretation (for combined assembly).

    Threads the test filter like every other "latest" selection in the repo.
    """
    try:
        ensure_table(con)
        test_clause = "" if include_test else " AND COALESCE(is_test, FALSE) = FALSE"
        rows = con.execute(
            f"""
            SELECT interpretation_id, scored_run_id, version, content_json,
                   figures_json, created_at
            FROM interpretations
            WHERE kind = 'scored' AND status = 'ok'{test_clause}
            ORDER BY created_at DESC, version DESC
            LIMIT 1
            """
        ).fetchall()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("latest_scored lookup failed: %s", exc)
        return None
    if not rows:
        return None
    r = rows[0]
    out = {
        "interpretation_id": r[0],
        "scored_run_id": r[1],
        "version": r[2],
        "content": None,
        "figures": None,
        "created_at": r[5],
    }
    try:
        out["content"] = json.loads(r[3]) if r[3] else None
    except Exception:  # noqa: BLE001
        pass
    try:
        out["figures"] = json.loads(r[4]) if r[4] else None
    except Exception:  # noqa: BLE001
        pass
    return out
