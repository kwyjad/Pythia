# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Resolve the identifiers the Sibyl-terminus bundles need, and fetch the
forecaster stage's context.

run_sibyl.yml builds the operational debug bundle and the forecast
attribution bundle since Sept 2026. Dispatched from fc_collect_finalize it
receives pipeline_id, hs_run_id and forecaster_run_id as inputs; triggered
by workflow_run from the legacy Horizon Scanner Triage workflow it receives
nothing but the DB. This helper resolves whatever is missing from the
canonical DB — the same discovery Sibyl uses for its own hs_run_id — so the
resolution lives in one place rather than in inline shell.

pipeline_id matters: the bundle's workflow-log collector discovers the
pipeline's stage runs through the Actions API by that id. Without it the
bundle captures the Sibyl job's logs and nothing else.

It also downloads the ``pythia-stage-context`` artifact the stage uploaded
into the bundle's ``stage_context/`` directory, and writes a stub there
when the artifact is missing, so a reader never has to guess whether a
collector failed or a file was never produced.

Usage:
    python -m scripts.ci.sibyl_bundle_context --db "$RESOLVER_DB_URL" \
        --pipeline-id "" --hs-run-id "" --forecaster-run-id "" \
        --stage-context-run-id "<actions run id>" \
        --stage-context-dir debug/stage_context \
        --github-env "$GITHUB_ENV"

Never fails. Exit 0.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STAGE_CONTEXT_ARTIFACT = "pythia-stage-context"


def _db_path(db: str) -> str:
    raw = (db or "").strip()
    return raw[len("duckdb:///"):] if raw.startswith("duckdb:///") else raw


def _column_exists(con, table: str, column: str) -> bool:
    try:
        rows = con.execute(f"PRAGMA table_info('{table}')").fetchall()
    except Exception:  # noqa: BLE001
        return False
    # PRAGMA table_info yields (cid, name, ...): the NAME is column 1.
    return any(str(r[1]).lower() == column.lower() for r in rows)


def _table_exists(con, table: str) -> bool:
    try:
        row = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [table]
        ).fetchone()
        return bool(row and row[0])
    except Exception:  # noqa: BLE001
        return False


def resolve_identifiers(
    con,
    *,
    pipeline_id: str | None,
    hs_run_id: str | None,
    forecaster_run_id: str | None,
) -> dict[str, Any]:
    """Fill in whatever was not supplied, from the DB. Supplied values win."""
    out: dict[str, Any] = {
        "pipeline_id": (pipeline_id or "").strip() or None,
        "hs_run_id": (hs_run_id or "").strip() or None,
        "forecaster_run_id": (forecaster_run_id or "").strip() or None,
        "resolved_from": {},
    }

    if not out["hs_run_id"]:
        try:
            from sibyl.select_questions import latest_hs_run_id  # noqa: PLC0415

            out["hs_run_id"] = latest_hs_run_id(con)
            out["resolved_from"]["hs_run_id"] = "sibyl.select_questions.latest_hs_run_id"
        except Exception as exc:  # noqa: BLE001
            out["resolved_from"]["hs_run_id"] = f"failed: {type(exc).__name__}: {exc}"
        if not out["hs_run_id"] and _table_exists(con, "questions"):
            # Sibyl's query joins hs_runs; a DB without that table (a
            # partial fixture, a very old artifact) still names its runs
            # on the questions themselves.
            try:
                row = con.execute(
                    "SELECT hs_run_id FROM questions WHERE status = 'active' AND hs_run_id IS NOT NULL "
                    "GROUP BY hs_run_id ORDER BY hs_run_id DESC LIMIT 1"
                ).fetchone()
                if row and row[0]:
                    out["hs_run_id"] = str(row[0])
                    out["resolved_from"]["hs_run_id"] = "questions (latest hs_run_id with active questions)"
            except Exception as exc:  # noqa: BLE001
                out["resolved_from"]["hs_run_id"] = f"failed: {type(exc).__name__}: {exc}"

    if not out["forecaster_run_id"] and _table_exists(con, "forecasts_raw"):
        try:
            row = None
            if out["hs_run_id"] and _table_exists(con, "questions"):
                row = con.execute(
                    "SELECT MAX(fr.run_id) FROM forecasts_raw fr "
                    "JOIN questions q ON q.question_id = fr.question_id "
                    "WHERE q.hs_run_id = ? AND fr.run_id IS NOT NULL AND fr.run_id <> ''",
                    [out["hs_run_id"]],
                ).fetchone()
            if not (row and row[0]):
                row = con.execute(
                    "SELECT MAX(run_id) FROM forecasts_raw WHERE run_id IS NOT NULL AND run_id <> ''"
                ).fetchone()
                out["resolved_from"]["forecaster_run_id"] = "forecasts_raw (latest run, any epoch)"
            else:
                out["resolved_from"]["forecaster_run_id"] = "forecasts_raw joined on the HS run's questions"
            out["forecaster_run_id"] = str(row[0]) if row and row[0] else None
        except Exception as exc:  # noqa: BLE001
            out["resolved_from"]["forecaster_run_id"] = f"failed: {type(exc).__name__}: {exc}"

    if (
        not out["pipeline_id"]
        and _table_exists(con, "llm_batches")
        and _column_exists(con, "llm_batches", "pipeline_id")
    ):
        try:
            row = None
            if out["forecaster_run_id"] and _column_exists(con, "llm_batches", "run_id"):
                row = con.execute(
                    "SELECT pipeline_id FROM llm_batches WHERE run_id = ? AND pipeline_id IS NOT NULL "
                    "ORDER BY submitted_at DESC LIMIT 1",
                    [out["forecaster_run_id"]],
                ).fetchone()
            if not (row and row[0]) and out["hs_run_id"] and _column_exists(con, "llm_batches", "hs_run_id"):
                row = con.execute(
                    "SELECT pipeline_id FROM llm_batches WHERE hs_run_id = ? AND pipeline_id IS NOT NULL "
                    "ORDER BY submitted_at DESC LIMIT 1",
                    [out["hs_run_id"]],
                ).fetchone()
            out["pipeline_id"] = str(row[0]) if row and row[0] else None
            out["resolved_from"]["pipeline_id"] = (
                "llm_batches by run id" if out["pipeline_id"] else "no batch row for this run (sync pipeline?)"
            )
        except Exception as exc:  # noqa: BLE001
            out["resolved_from"]["pipeline_id"] = f"failed: {type(exc).__name__}: {exc}"
    return out


def _write_stub(target: Path, reason: str, **extra: Any) -> None:
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "missing": True,
        "artifact": STAGE_CONTEXT_ARTIFACT,
        "reason": reason,
        "written_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    (target / "MISSING.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def fetch_stage_context(
    *,
    run_id: str | None,
    target: Path,
    repo: str | None = None,
    runner=None,
) -> dict[str, Any]:
    """Download the stage's context artifact into ``target``; stub if absent.

    ``runner`` is the subprocess seam for tests (defaults to gh via
    subprocess.run). Never raises.
    """
    run_id = (run_id or "").strip()
    if not run_id:
        _write_stub(target, "no source run id was available to download the artifact from")
        return {"downloaded": False, "reason": "no run id"}
    if runner is None:
        if shutil.which("gh") is None:
            _write_stub(target, "gh CLI not available in this job", run_id=run_id)
            return {"downloaded": False, "reason": "no gh"}

        def runner(cmd: list[str]) -> tuple[int, str]:  # type: ignore[misc]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            return proc.returncode, (proc.stdout + proc.stderr)[-2000:]

    cmd = ["gh", "run", "download", run_id, "-n", STAGE_CONTEXT_ARTIFACT, "-D", str(target)]
    if repo:
        cmd += ["-R", repo]
    try:
        rc, output = runner(cmd)
    except Exception as exc:  # noqa: BLE001
        rc, output = 1, f"{type(exc).__name__}: {exc}"
    files = [p.name for p in target.glob("*")] if target.exists() else []
    if rc != 0 or not any(f != "MISSING.json" for f in files):
        _write_stub(
            target,
            f"artifact {STAGE_CONTEXT_ARTIFACT} not downloadable from run {run_id}",
            run_id=run_id,
            gh_output=output.strip(),
        )
        return {"downloaded": False, "reason": output.strip()[-500:], "run_id": run_id}
    return {"downloaded": True, "run_id": run_id, "files": sorted(files)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--pipeline-id", default="")
    parser.add_argument("--hs-run-id", default="")
    parser.add_argument("--forecaster-run-id", default="")
    parser.add_argument("--stage-context-run-id", default="",
                        help="Actions run id that uploaded pythia-stage-context (blank = write a stub)")
    parser.add_argument("--stage-context-dir", default="debug/stage_context")
    parser.add_argument("--github-env", default=os.getenv("GITHUB_ENV", ""),
                        help="File to append PIPELINE_ID/HS_RUN_ID/FORECASTER_RUN_ID to")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args(argv)

    resolved: dict[str, Any] = {"pipeline_id": None, "hs_run_id": None, "forecaster_run_id": None}
    try:
        import duckdb  # noqa: PLC0415

        con = duckdb.connect(_db_path(args.db), read_only=True)
        try:
            resolved = resolve_identifiers(
                con,
                pipeline_id=args.pipeline_id,
                hs_run_id=args.hs_run_id,
                forecaster_run_id=args.forecaster_run_id,
            )
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001
        resolved["error"] = f"{type(exc).__name__}: {exc}"
        for key, value in (("pipeline_id", args.pipeline_id), ("hs_run_id", args.hs_run_id),
                           ("forecaster_run_id", args.forecaster_run_id)):
            resolved[key] = (value or "").strip() or None
    print("bundle context: " + json.dumps(resolved, default=str))

    if args.github_env:
        try:
            with open(args.github_env, "a", encoding="utf-8") as fh:
                fh.write(f"PIPELINE_ID={resolved.get('pipeline_id') or ''}\n")
                fh.write(f"HS_RUN_ID={resolved.get('hs_run_id') or ''}\n")
                fh.write(f"FORECASTER_RUN_ID={resolved.get('forecaster_run_id') or ''}\n")
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: could not write {args.github_env}: {exc}")

    if not args.skip_download:
        result = fetch_stage_context(
            run_id=args.stage_context_run_id,
            target=Path(args.stage_context_dir),
            repo=os.getenv("GITHUB_REPOSITORY") or None,
        )
        print("stage context: " + json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
