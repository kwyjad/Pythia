#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render a consolidated CI diagnostics summary without fragile f-strings."""

from __future__ import annotations

import json
import os
import sys
import textwrap
import traceback
from datetime import datetime, UTC
from pathlib import Path
from string import Template
from typing import Iterable, Mapping, MutableMapping

DEFAULT_ART_DIR = ".ci/diagnostics"


def _resolve_art_dir(argv: Iterable[str]) -> Path:
    args = list(argv)
    if len(args) >= 2 and args[1]:
        target = Path(args[1])
    else:
        target = Path(os.environ.get("ART_DIR", DEFAULT_ART_DIR))
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    return target


def _read_text(path: Path, default: str = "(no output captured)") -> str:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        pass
    return default


def _try_junit_totals(junit_path: Path) -> Mapping[str, object]:
    try:
        import xml.etree.ElementTree as ET
    except Exception as exc:  # pragma: no cover - extremely unlikely on CI images
        return {"status": f"xml unavailable: {exc}"}

    if not junit_path.exists():
        return {"status": "absent"}

    try:
        root = ET.fromstring(junit_path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        return {"status": f"unreadable: {exc}"}

    suites = []
    if root.tag == "testsuite":
        suites = [root]
    elif root.tag == "testsuites":
        suites = list(root.findall("testsuite"))

    totals: MutableMapping[str, int | float] = {
        "tests": 0,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "time": 0.0,
    }
    for suite in suites:
        totals["tests"] += int(suite.attrib.get("tests", 0))
        totals["failures"] += int(suite.attrib.get("failures", 0))
        totals["errors"] += int(suite.attrib.get("errors", 0))
        totals["skipped"] += int(suite.attrib.get("skipped", 0))
        totals["time"] += float(suite.attrib.get("time", 0.0))

    return {"status": "present", **totals}


def _duckdb_inspect(db_path: Path) -> Mapping[str, object]:
    try:
        import duckdb
    except Exception as exc:
        return {"error": f"duckdb import failed: {exc}"}

    if not db_path.exists():
        return {"error": f"database not found: {db_path}"}

    try:
        con = duckdb.connect(str(db_path))
    except Exception as exc:
        return {"error": f"duckdb connect failed: {exc}"}

    try:
        tables_relation = con.sql(
            "SELECT name FROM duckdb_tables() WHERE database_name IS NULL"
        )
        table_names = {row[0] for row in tables_relation.fetchall()}

        report: MutableMapping[str, Mapping[str, object]] = {}
        targets = [
            "facts_raw",
            "facts_resolved",
            "facts_deltas",
            "facts_monthly_deltas",
            "manifests",
            "meta_runs",
            "snapshots",
        ]
        for table in targets:
            exists = table in table_names
            table_report: MutableMapping[str, object] = {"exists": exists}
            if exists:
                count_relation = con.sql(f"SELECT COUNT(*) AS cnt FROM {table}")
                table_report["rows"] = count_relation.fetchone()[0]

                sample_relation = con.sql(f"SELECT * FROM {table} LIMIT 4")
                columns = sample_relation.columns
                sample_rows = sample_relation.fetchall()
                table_report["sample"] = [
                    dict(zip(columns, row)) for row in sample_rows
                ]

                schema_relation = con.sql(f"PRAGMA table_info('{table}')")
                schema_columns = schema_relation.columns
                schema_rows = schema_relation.fetchall()
                table_report["schema"] = [
                    dict(zip(schema_columns, row)) for row in schema_rows
                ]
            else:
                table_report["rows"] = "n/a"
                table_report["sample"] = []
                table_report["schema"] = []
            report[table] = table_report

        return report
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        try:
            con.close()
        except Exception:
            pass


def _render_summary(env: Mapping[str, str], junit: Mapping[str, object], *, lda_log: str, normalize_log: str, pip_freeze: str, duckdb_report: Mapping[str, object]) -> str:
    pytest_summary = "JUnit report: unavailable"
    if junit:
        status = str(junit.get("status", "unknown"))
        if status == "present":
            pytest_summary = (
                "tests={tests} failures={failures} errors={errors} skipped={skipped} "
                "time={time}".format(**{k: junit.get(k, 0) for k in ("tests", "failures", "errors", "skipped", "time")})
            )
        else:
            pytest_summary = f"JUnit report: {status}"

    tmpl = Template(
        textwrap.dedent(
            """
            # CI Diagnostics Summary

            ## Run Metadata
            Commit: $commit  
            Ref: $ref  
            Workflow: $workflow  
            Job: $job  
            Run ID: $run_id  Attempt: $attempt  
            Timestamp (UTC): $ts  
            Python: $python  OS: $os

            ## Pytest Summary
            $pytest_summary

            ## Command Tails
            ### lda-all
            $lda_log

            ### normalize
            $normalize_log

            ## DuckDB Inspection (key tables)
            ```json
            $duckdb_json
            ```

            ## pip freeze
            ```
            $pip_freeze
            ```
            """
        ).strip()
    )

    duckdb_json = json.dumps(duckdb_report, indent=2, ensure_ascii=False)

    return tmpl.substitute(
        commit=env.get("commit", "(unknown)"),
        ref=env.get("ref", "(unknown)"),
        workflow=env.get("workflow", "(unknown)"),
        job=env.get("job", "(unknown)"),
        run_id=env.get("run_id", "(unknown)"),
        attempt=env.get("attempt", "(unknown)"),
        ts=env.get("ts", "(unknown)"),
        python=env.get("python", "(unknown)"),
        os=env.get("os", "(unknown)"),
        pytest_summary=pytest_summary,
        lda_log=lda_log,
        normalize_log=normalize_log,
        duckdb_json=duckdb_json,
        pip_freeze=pip_freeze,
    )


def _gather_env() -> Mapping[str, str]:
    uname = os.uname().sysname if hasattr(os, "uname") else "(unknown)"
    return {
        "commit": os.environ.get("GITHUB_SHA") or os.environ.get("GIT_COMMIT", "(unknown)"),
        "ref": os.environ.get("GITHUB_REF", "(unknown)"),
        "workflow": os.environ.get("GITHUB_WORKFLOW", "(unknown)"),
        "job": os.environ.get("GITHUB_JOB", "(unknown)"),
        "run_id": os.environ.get("GITHUB_RUN_ID", "(unknown)"),
        "attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "(unknown)"),
        "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "python": sys.version.split()[0],
        "os": uname,
    }


def _write_summary_files(art_dir: Path, content: str) -> None:
    summary_path = art_dir / "SUMMARY.md"
    summary_path.write_text(content, encoding="utf-8")
    legacy = art_dir / "summary.md"
    legacy.write_text(content, encoding="utf-8")


def _append_step_summary(content: str) -> None:
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not step_summary:
        return
    try:
        with open(step_summary, "a", encoding="utf-8") as handle:
            handle.write(content)
            handle.write("\n")
    except Exception:
        pass


def main(argv: Iterable[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    art_dir = _resolve_art_dir(argv)

    try:
        env = _gather_env()
        junit = _try_junit_totals(art_dir / "pytest-junit.xml")
        pip_freeze = _read_text(art_dir / "pip-freeze.txt")
        lda_log = _read_text(art_dir / "lda-all.log")
        normalize_log = _read_text(art_dir / "normalize.log")
        duckdb_report = _duckdb_inspect(Path("data/resolver.duckdb"))

        summary_text = _render_summary(
            env,
            junit,
            lda_log=lda_log,
            normalize_log=normalize_log,
            pip_freeze=pip_freeze,
            duckdb_report=duckdb_report,
        )

        _write_summary_files(art_dir, summary_text)
        _append_step_summary(summary_text)

        print(f"Wrote diagnostics summary to {art_dir / 'SUMMARY.md'}")
        return 0
    except Exception:
        tb = traceback.format_exc()
        fallback = "# CI Diagnostics Summary (fallback)\n\n````\n" + tb + "\n````\n"
        _write_summary_files(art_dir, fallback)
        _append_step_summary(fallback)
        print("collect_diagnostics.py encountered an error but wrote fallback summary", file=sys.stderr)
        return 0


if __name__ == "__main__":
    sys.exit(main())
