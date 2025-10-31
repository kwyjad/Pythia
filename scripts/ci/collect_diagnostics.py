#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render a consolidated CI diagnostics summary without fragile f-strings."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import textwrap
import traceback
from datetime import datetime, UTC
from pathlib import Path
from string import Template
from typing import Iterable, List, Mapping, MutableMapping, Optional

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
        "failures_detail": [],
    }
    details: List[Mapping[str, object]] = []
    for suite in suites:
        totals["tests"] += int(suite.attrib.get("tests", 0))
        totals["failures"] += int(suite.attrib.get("failures", 0))
        totals["errors"] += int(suite.attrib.get("errors", 0))
        totals["skipped"] += int(suite.attrib.get("skipped", 0))
        totals["time"] += float(suite.attrib.get("time", 0.0))
        for case in suite.findall("testcase"):
            duration = float(case.attrib.get("time", 0.0))
            testcase_name = case.attrib.get("name", "")
            classname = case.attrib.get("classname", "")
            full_name = ".".join(filter(None, (classname, testcase_name))) or testcase_name
            for failure_tag in ("failure", "error"):
                for node in case.findall(failure_tag):
                    message = (node.attrib.get("message") or "").strip()
                    text = (node.text or "").strip()
                    trace_tail = "\n".join(text.splitlines()[-12:]) if text else ""
                    details.append(
                        {
                            "name": full_name or testcase_name or "(unknown)",
                            "message": message or failure_tag,
                            "trace": trace_tail,
                            "time": duration,
                            "type": failure_tag,
                        }
                    )
    totals["failures_detail"] = details

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


def _format_failures(junit: Mapping[str, object]) -> str:
    detail = junit.get("failures_detail") if isinstance(junit, Mapping) else None
    if not detail or not isinstance(detail, list):
        return "No test failures recorded."
    lines: List[str] = []
    for entry in list(detail)[:5]:
        name = str(entry.get("name", "(unknown)")) if isinstance(entry, Mapping) else "(unknown)"
        message = str(entry.get("message", "")) if isinstance(entry, Mapping) else ""
        duration = entry.get("time", 0.0) if isinstance(entry, Mapping) else 0.0
        try:
            duration_val = float(duration)
        except (TypeError, ValueError):
            duration_val = 0.0
        summary_line = f"- **{name}** ({duration_val:.3f}s)"
        if message:
            summary_line += f" — {message}"
        lines.append(summary_line)
        trace = str(entry.get("trace", "")) if isinstance(entry, Mapping) else ""
        if trace.strip():
            lines.append("  ```")
            lines.extend(f"  {line}" for line in trace.strip().splitlines())
            lines.append("  ```")
    if len(detail) > 5:
        lines.append(f"- … {len(detail) - 5} additional failure(s) omitted")
    return "\n".join(lines) or "No test failures recorded."


def _selected_env_block() -> str:
    interesting = [
        "GITHUB_WORKFLOW",
        "GITHUB_JOB",
        "GITHUB_RUN_ID",
        "GITHUB_RUN_ATTEMPT",
        "RESOLVER_DB_URL",
        "RESOLVER_API_BACKEND",
        "PYTHONPATH",
    ]
    return "\n".join(f"{key}={os.environ.get(key, '')}" for key in interesting)


def _pip_freeze_output(art_dir: Path) -> str:
    freeze_path = art_dir / "pip-freeze.txt"
    if freeze_path.exists():
        return freeze_path.read_text(encoding="utf-8", errors="replace")
    try:
        output = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    except Exception as exc:
        return f"pip freeze failed: {exc}"
    try:
        freeze_path.write_text(output, encoding="utf-8")
    except Exception:
        pass
    return output


def _find_run_details_path() -> Optional[Path]:
    candidates: List[Path] = []
    override = os.environ.get("RUN_DETAILS_PATH") or os.environ.get(
        "RESOLVER_RUN_DETAILS_PATH"
    )
    if override:
        candidates.append(Path(override))
    candidates.extend(
        [
            Path("diagnostics/ingestion/dtm/dtm_run.json"),
            Path("resolver/diagnostics/ingestion/dtm/dtm_run.json"),
        ]
    )
    for candidate in candidates:
        try:
            expanded = candidate.expanduser()
            if not expanded.is_absolute():
                expanded = (Path.cwd() / expanded).resolve()
        except Exception:
            continue
        if expanded.exists():
            return expanded
    return None


def _collect_run_details(art_dir: Path) -> str:
    path = _find_run_details_path()
    if not path:
        return ""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    target = art_dir / path.name
    try:
        if path.resolve() != target.resolve():
            target.write_text(content, encoding="utf-8")
    except Exception:
        pass
    return content


def _render_summary(
    env: Mapping[str, str],
    junit: Mapping[str, object],
    *,
    pytest_failures: str,
    env_block: str,
    lda_log: str,
    normalize_log: str,
    pip_freeze: str,
    duckdb_report: Mapping[str, object],
    run_details: str,
    pytest_tail: str,
) -> str:
    pytest_summary = "JUnit report: unavailable"
    if junit:
        status = str(junit.get("status", "unknown"))
        if status == "present":
            totals = {
                key: junit.get(key, 0)
                for key in ("tests", "failures", "errors", "skipped", "time")
            }
            try:
                totals["time"] = float(totals.get("time", 0.0))
            except (TypeError, ValueError):
                totals["time"] = 0.0
            pytest_summary = (
                "tests={tests} failures={failures} errors={errors} skipped={skipped} time={time:.2f}s"
            ).format(**totals)
        else:
            pytest_summary = f"JUnit report: {status}"

    duckdb_json = json.dumps(duckdb_report, indent=2, ensure_ascii=False)
    env_text = env_block.strip() or "(no additional environment variables captured)"
    pip_text = pip_freeze.strip() or "(pip freeze unavailable)"
    run_details_text = run_details.strip() or "{}"
    pytest_tail_text = pytest_tail.strip() or "(no pytest output captured)"

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
            Platform: $platform
            Python executable: $python_executable

            ## Pytest Summary
            $pytest_summary

            ## Pytest Failures (top entries)
            $pytest_failures

            ## Environment (selected)
            ```
            $env_block
            ```

            ## pip freeze
            ```
            $pip_freeze
            ```

            ## DTM run-details
            ```json
            $run_details
            ```

            ## Pytest output tail
            ```
            $pytest_tail
            ```

            ## Command Tails
            ### lda-all
            $lda_log

            ### normalize
            $normalize_log

            ## DuckDB Inspection (key tables)
            ```json
            $duckdb_json
            ```
            """
        ).strip()
    )

    return tmpl.safe_substitute(
        commit=env.get("commit", "(unknown)"),
        ref=env.get("ref", "(unknown)"),
        workflow=env.get("workflow", "(unknown)"),
        job=env.get("job", "(unknown)"),
        run_id=env.get("run_id", "(unknown)"),
        attempt=env.get("attempt", "(unknown)"),
        ts=env.get("ts", "(unknown)"),
        python=env.get("python", "(unknown)"),
        os=env.get("os", "(unknown)"),
        platform=env.get("platform", platform.platform()),
        python_executable=env.get("python_executable", sys.executable),
        pytest_summary=pytest_summary,
        pytest_failures=pytest_failures,
        env_block=env_text,
        pip_freeze=pip_text,
        run_details=run_details_text,
        pytest_tail=pytest_tail_text,
        lda_log=lda_log,
        normalize_log=normalize_log,
        duckdb_json=duckdb_json,
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
        "platform": platform.platform(),
        "python_executable": sys.executable,
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
        junit_path = art_dir / "pytest-junit.xml"
        if not junit_path.exists() and Path("pytest-junit.xml").exists():
            try:
                junit_path.write_text(Path("pytest-junit.xml").read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
            except Exception:
                pass
        junit = _try_junit_totals(junit_path)
        pip_freeze = _pip_freeze_output(art_dir)
        lda_log = _read_text(art_dir / "lda-all.log")
        normalize_log = _read_text(art_dir / "normalize.log")
        duckdb_report = _duckdb_inspect(Path("data/resolver.duckdb"))
        run_details = _collect_run_details(art_dir)

        pytest_log_path = Path("pytest-Linux.log")
        pytest_tail = ""
        if pytest_log_path.exists():
            try:
                tail_lines = pytest_log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                tail_lines = []
            if tail_lines:
                pytest_tail = "\n".join(tail_lines[-400:])

        summary_text = _render_summary(
            env,
            junit,
            pytest_failures=_format_failures(junit),
            env_block=_selected_env_block(),
            lda_log=lda_log,
            normalize_log=normalize_log,
            pip_freeze=pip_freeze,
            duckdb_report=duckdb_report,
            run_details=run_details,
            pytest_tail=pytest_tail,
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
