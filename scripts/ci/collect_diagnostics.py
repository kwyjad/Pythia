#!/usr/bin/env python3
# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

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
NORMALIZE_EXPECTED_KEYS = [
    "no_iso3",
    "no_value_col",
    "bad_iso",
    "unknown_country",
    "missing_value",
    "non_positive_value",
    "non_integer_value",
    "missing_as_of",
    "future_as_of",
    "invalid_semantics",
    "other",
]


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


def _preview_validator_section(repo_root: Path) -> str:
    candidates = [
        repo_root / "diagnostics" / "ingestion" / "preview_validator.stderr.txt",
        repo_root
        / "diagnostics"
        / "ingestion"
        / "export_preview"
        / "validator_stderr.txt",
    ]

    content = ""
    for path in candidates:
        try:
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            continue
        if text:
            content = text
            break

    if not content:
        return ""

    tail_lines = content.splitlines()[-200:]
    tail = "\n".join(tail_lines)
    return "\n".join(
        [
            "### Preview Validator (stderr)",
            "```",
            tail,
            "```",
            "",
        ]
    )


def _collect_additional_diagnostics() -> str:
    lines: List[str] = []
    heading_added = False

    def _ensure_heading() -> None:
        nonlocal heading_added
        if not heading_added:
            lines.append("\n\n---\n## Additional Diagnostics\n")
            heading_added = True

    summary_path = Path("diagnostics") / "summary.md"
    try:
        summary_text = summary_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        summary_text = ""
    if summary_text.strip():
        _ensure_heading()
        lines.append(summary_text.strip())

    candidate_files = [
        (
            Path("diagnostics") / "ingestion" / "preview_validator.stderr.txt",
            "Preview validator stderr (tail)",
        ),
        (
            Path("diagnostics") / "ingestion" / "export_preview" / "validator_stderr.txt",
            "Preview validator stderr (tail)",
        ),
        (
            Path("diagnostics") / "ingestion" / "preview_validator.stdout.txt",
            "Preview validator stdout (tail)",
        ),
        (
            Path("diagnostics")
            / "ingestion"
            / "export_preview"
            / "validator_stdout.txt",
            "Preview validator stdout (tail)",
        ),
    ]
    for candidate, title in candidate_files:
        try:
            if not candidate.exists():
                continue
            text = candidate.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        tail_lines = text.splitlines()[-200:]
        tail = "\n".join(tail_lines).strip()
        if not tail:
            continue
        _ensure_heading()
        lines.append(
            "\n\n### "
            + title
            + "\n\n```\n"
            + tail
            + "\n```"
        )

    return "".join(lines)


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


def _format_pytest_overview(junit: Mapping[str, object]) -> str:
    if not isinstance(junit, Mapping) or junit.get("status") != "present":
        return ""
    totals = {
        "tests": junit.get("tests", 0),
        "failures": junit.get("failures", 0),
        "errors": junit.get("errors", 0),
        "skipped": junit.get("skipped", 0),
        "time": junit.get("time", 0.0),
    }
    try:
        totals["time"] = float(totals.get("time", 0.0))
    except (TypeError, ValueError):
        totals["time"] = 0.0
    return "\n".join(
        [
            "| Metric | Count |",
            "| --- | ---: |",
            f"| Tests | {totals['tests']} |",
            f"| Failures | {totals['failures']} |",
            f"| Errors | {totals['errors']} |",
            f"| Skipped | {totals['skipped']} |",
            f"| Time (s) | {totals['time']:.2f} |",
        ]
    )


def _dtm_debug_block(run_details_text: str) -> str:
    if not run_details_text.strip():
        return "Run-details JSON not captured."
    try:
        payload = json.loads(run_details_text)
    except Exception as exc:  # pragma: no cover - defensive guard
        return f"Failed to parse run-details: {exc}"
    if not isinstance(payload, Mapping):
        return "Run-details payload is not a mapping."

    extras = payload.get("extras") if isinstance(payload.get("extras"), Mapping) else {}
    normalize_block = extras.get("normalize") if isinstance(extras, Mapping) else {}
    if not isinstance(normalize_block, Mapping):
        normalize_block = {}

    drop_reasons = normalize_block.get("drop_reasons")
    if not isinstance(drop_reasons, Mapping):
        drop_reasons = {}
    chosen_columns = normalize_block.get("chosen_value_columns")
    if not isinstance(chosen_columns, list):
        chosen_columns = []
    config_block = normalize_block.get("config")
    if not isinstance(config_block, Mapping):
        config_block = extras.get("config") if isinstance(extras, Mapping) else {}
        if not isinstance(config_block, Mapping):
            config_block = {}

    skip_flags = extras.get("skip_flags") if isinstance(extras, Mapping) else {}
    if not isinstance(skip_flags, Mapping):
        skip_flags = {}

    env_flags = {
        "RESOLVER_SKIP_DTM": os.environ.get("RESOLVER_SKIP_DTM", ""),
        "DTM_FORCE_RUN": os.environ.get("DTM_FORCE_RUN", ""),
        "DTM_CONFIG_PATH": os.environ.get("DTM_CONFIG_PATH", ""),
    }

    status = str(payload.get("status", "unknown"))
    reason = payload.get("reason") or payload.get("zero_rows_reason")
    rows_written = normalize_block.get("rows_written")
    if rows_written is None:
        rows_written = (
            payload.get("rows", {}).get("written")
            if isinstance(payload.get("rows"), Mapping)
            else None
        )

    overview = {
        "status": status,
        "reason": reason,
        "rows_written": rows_written,
        "totals": payload.get("totals", {}),
        "rows": payload.get("rows", {}),
    }

    lines: List[str] = []
    lines.append("#### Run overview")
    lines.append("```json")
    lines.append(json.dumps(overview, indent=2, ensure_ascii=False))
    lines.append("```")

    lines.append("")
    lines.append("#### Skip & environment flags")
    lines.append("| Flag | Value | Source |")
    lines.append("| --- | --- | --- |")
    for key, value in sorted(skip_flags.items()):
        lines.append(f"| `{key}` | `{bool(value)}` | run-details |")
    for key, value in env_flags.items():
        display = value if value else ""
        lines.append(f"| `{key}` | `{display}` | env |")

    countries_mode = config_block.get("countries_mode")
    admin_levels = config_block.get("admin_levels")
    if not isinstance(admin_levels, (list, tuple)):
        admin_levels = []
    lines.append("")
    lines.append("#### Config summary")
    lines.append(f"- countries_mode: `{countries_mode if countries_mode is not None else 'n/a'}`")
    levels_text = ", ".join(str(level) for level in admin_levels) if admin_levels else "(none)"
    lines.append(f"- admin_levels: {levels_text}")

    lines.append("")
    lines.append("#### Normalize drop reasons")
    if drop_reasons:
        lines.append("| Reason | Count |")
        lines.append("| --- | ---: |")
        for key in sorted(drop_reasons):
            value = drop_reasons.get(key, 0)
            try:
                count_val = int(value)
            except (TypeError, ValueError):
                count_val = value
            lines.append(f"| `{key}` | {count_val} |")
    else:
        lines.append("No drop reasons recorded.")
    missing_keys = [key for key in NORMALIZE_EXPECTED_KEYS if key not in drop_reasons]
    if missing_keys:
        lines.append(
            f"_Missing expected key(s): {', '.join(sorted(missing_keys))}_"
        )

    lines.append("")
    lines.append("#### Value column usage")
    if chosen_columns:
        lines.append("| Column | Count |")
        lines.append("| --- | ---: |")
        for entry in chosen_columns:
            if not isinstance(entry, Mapping):
                continue
            column = entry.get("column")
            count = entry.get("count", 0)
            lines.append(f"| `{column}` | {count} |")
    else:
        lines.append("No value column diagnostics recorded.")

    fetch_block = extras.get("fetch") if isinstance(extras, Mapping) else {}
    fetch_levels = fetch_block.get("levels") if isinstance(fetch_block, Mapping) else []
    if fetch_levels:
        lines.append("")
        lines.append("#### Admin-level fetch summary")
        lines.append("| Level | Calls | Rows | Elapsed (ms) |")
        lines.append("| --- | ---: | ---: | ---: |")
        for entry in fetch_levels:
            if not isinstance(entry, Mapping):
                continue
            level = entry.get("level") or entry.get("name")
            calls = entry.get("pages")
            rows = entry.get("rows")
            elapsed = entry.get("elapsed_ms") or entry.get("elapsed")
            lines.append(f"| `{level}` | {calls} | {rows} | {elapsed} |")

    return "\n".join(line for line in lines if line is not None).strip()

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
    pytest_overview: str,
    env_block: str,
    lda_log: str,
    normalize_log: str,
    pip_freeze: str,
    duckdb_report: Mapping[str, object],
    run_details: str,
    dtm_debug: str,
    pytest_tail: str,
    preview_validator_section: str,
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
    pytest_overview_text = pytest_overview.strip()
    dtm_debug_text = dtm_debug.strip() or "(No DTM diagnostics captured)"

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

            $pytest_overview

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

            ## DTM Debug
            $dtm_debug

            $preview_validator_section

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
        pytest_overview=pytest_overview_text,
        env_block=env_text,
        pip_freeze=pip_text,
        run_details=run_details_text,
        dtm_debug=dtm_debug_text,
        pytest_tail=pytest_tail_text,
        lda_log=lda_log,
        normalize_log=normalize_log,
        duckdb_json=duckdb_json,
        preview_validator_section=preview_validator_section,
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

        pytest_failures_text = _format_failures(junit)
        pytest_overview_text = _format_pytest_overview(junit)
        dtm_debug_text = _dtm_debug_block(run_details)
        preview_validator_text = _preview_validator_section(Path.cwd())

        summary_text = _render_summary(
            env,
            junit,
            pytest_failures=pytest_failures_text,
            pytest_overview=pytest_overview_text,
            env_block=_selected_env_block(),
            lda_log=lda_log,
            normalize_log=normalize_log,
            pip_freeze=pip_freeze,
            duckdb_report=duckdb_report,
            run_details=run_details,
            dtm_debug=dtm_debug_text,
            pytest_tail=pytest_tail,
            preview_validator_section=preview_validator_text,
        )

        additional = _collect_additional_diagnostics()
        if additional:
            summary_text = summary_text.rstrip() + additional

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
