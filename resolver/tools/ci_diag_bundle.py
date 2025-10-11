"""Create a consolidated CI diagnostics bundle for resolver runs."""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE_NAME = "diagnostics.zip"
FILES_TO_STAMP: tuple[tuple[str, str], ...] = (
    ("resolver/db/duckdb_io.py", "resolver.db.duckdb_io"),
    ("resolver/tools/export_facts.py", "resolver.tools.export_facts"),
    ("resolver/query/db_reader.py", "resolver.query.db_reader"),
    ("resolver/query/selectors.py", "resolver.query.selectors"),
    ("resolver/cli/resolver_cli.py", "resolver.cli.resolver_cli"),
    ("schema.sql", "schema"),
)


def _run(cmd: Iterable[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(cmd), capture_output=True, text=True, check=check)


def _safe_text(process: subprocess.CompletedProcess[str]) -> str:
    data = process.stdout.strip()
    if data:
        return data
    return process.stderr.strip()


def _gather_git_meta() -> Mapping[str, str | None]:
    head = _safe_text(_run(["git", "rev-parse", "HEAD"]))
    describe = _safe_text(
        _run(["git", "show", "-s", "--date=iso", "--format=%H %cd %s"])
    )
    return {
        "git_head": head or None,
        "git_show": describe or None,
        "github_sha": os.environ.get("GITHUB_SHA"),
        "github_ref": os.environ.get("GITHUB_REF"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "pr_head_sha": os.environ.get("PR_HEAD_SHA")
        or os.environ.get("GITHUB_HEAD_REF"),
    }


def _pip_duckdb_listing() -> str | None:
    result = _run([sys.executable, "-m", "pip", "list", "--format=columns"])
    lines = [line for line in result.stdout.splitlines() if "duckdb" in line.lower()]
    return "\n".join(lines).strip() or None


def _resolve_import_path(import_path: str) -> str | None:
    if not import_path or import_path == "schema":
        return None
    try:
        module = importlib.import_module(import_path)
    except Exception:  # pragma: no cover - diagnostics only
        return None
    file_path = getattr(module, "__file__", None)
    if not file_path:
        return None
    return str(Path(file_path).resolve())


def _stamp_file(rel_path: str, import_path: str) -> Mapping[str, object]:
    path = ROOT / rel_path
    payload: MutableMapping[str, object] = {
        "path": str(path),
        "exists": path.exists(),
        "import_path": import_path,
    }
    resolved = _resolve_import_path(import_path)
    if resolved:
        payload["import_resolved_path"] = resolved
    if path.exists():
        payload.update(
            {
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return payload


def _write_json(zip_file: ZipFile, arcname: str, payload: Mapping[str, object]) -> None:
    zip_file.writestr(
        arcname,
        json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n",
    )


def _parse_index_columns(sql: str | None) -> list[str] | None:
    if not sql:
        return None
    match = re.search(r"\((.*)\)", sql)
    if not match:
        return None
    columns = [part.strip().strip('"') for part in match.group(1).split(",")]
    return [column for column in columns if column]


def _record_db_section(
    zip_file: ZipFile, db_url: str | None, duckdb_version: str | None
) -> Mapping[str, object]:
    summary: MutableMapping[str, object] = {"tables": []}
    if not db_url:
        message = {"error": "db_url not provided"}
        _write_json(zip_file, "diagnostics/db/db.json", message)
        summary["error"] = message["error"]
        return summary

    try:
        from resolver.db import duckdb_io
    except Exception as exc:  # pragma: no cover - diagnostics only
        message = {"error": f"duckdb import failed: {exc}"}
        _write_json(zip_file, "diagnostics/db/db.json", message)
        summary["error"] = message["error"]
        return summary

    try:
        conn = duckdb_io.get_db(db_url)
    except Exception as exc:  # pragma: no cover - diagnostics only
        message = {"error": f"get_db failed: {exc}"}
        _write_json(zip_file, "diagnostics/db/db.json", message)
        summary["error"] = message["error"]
        return summary

    resolved_path = getattr(conn, "_path", None) or getattr(conn, "database", None)
    meta = {
        "db_url": db_url,
        "resolved_path": resolved_path,
        "duckdb_version": duckdb_version,
    }

    db_root = "diagnostics/db"
    _write_json(zip_file, f"{db_root}/meta.json", meta)
    summary["resolved_path"] = resolved_path

    try:
        show_tables = conn.execute("PRAGMA show_tables").fetchall()
    except Exception as exc:  # pragma: no cover - diagnostics only
        message = {"error": f"show_tables failed: {exc}"}
        _write_json(zip_file, f"{db_root}/show_tables.json", message)
        summary["error"] = message["error"]
        return summary

    _write_json(zip_file, f"{db_root}/show_tables.json", {"tables": show_tables})

    tables = [row[0] for row in show_tables if row]
    focus_tables = [t for t in ("facts_resolved", "facts_deltas") if t in tables]
    ym_env = next(
        (os.environ[key] for key in ("RESOLVER_TARGET_YM", "RESOLVER_SNAPSHOT_YM", "TARGET_YM") if os.environ.get(key)),
        None,
    )

    for table in focus_tables:
        prefix = f"{db_root}/{table}"
        table_summary: MutableMapping[str, object] = {"name": table}
        try:
            table_info = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        except Exception as exc:  # pragma: no cover - diagnostics only
            payload = {"error": repr(exc)}
            _write_json(zip_file, f"{prefix}.table_info.json", payload)
            table_summary["table_info_error"] = payload["error"]
        else:
            _write_json(zip_file, f"{prefix}.table_info.json", {"columns": table_info})
            table_summary["columns"] = [row[1] for row in table_info]

        try:
            indexes = conn.execute(f"PRAGMA indexes('{table}')").fetchall()
        except Exception as exc:  # pragma: no cover - diagnostics only
            payload = {"error": repr(exc)}
            _write_json(zip_file, f"{prefix}.indexes.json", payload)
            table_summary["indexes_error"] = payload["error"]
            indexes = []
        else:
            _write_json(zip_file, f"{prefix}.indexes.json", {"indexes": indexes})

        try:
            duckdb_indexes = conn.execute(
                "SELECT index_name, sql FROM duckdb_indexes() WHERE table_name = ?",
                [table],
            ).fetchall()
        except Exception as exc:  # pragma: no cover - diagnostics only
            payload = {"error": repr(exc)}
            _write_json(zip_file, f"{prefix}.duckdb_indexes.json", payload)
            table_summary["duckdb_indexes_error"] = payload["error"]
            duckdb_indexes = []
        else:
            _write_json(
                zip_file,
                f"{prefix}.duckdb_indexes.json",
                {"indexes": duckdb_indexes},
            )

        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except Exception as exc:  # pragma: no cover - diagnostics only
            payload = {"error": repr(exc)}
            _write_json(zip_file, f"{prefix}.count.json", payload)
            table_summary["row_count_error"] = payload["error"]
        else:
            count_int = int(count)
            _write_json(zip_file, f"{prefix}.count.json", {"count": count_int})
            table_summary["row_count"] = count_int

        if ym_env:
            try:
                filtered = conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE ym = ?",
                    [ym_env],
                ).fetchone()[0]
            except Exception as exc:  # pragma: no cover - diagnostics only
                payload = {"ym": ym_env, "error": repr(exc)}
                _write_json(zip_file, f"{prefix}.count_ym.json", payload)
                table_summary["row_count_ym_error"] = payload["error"]
            else:
                filtered_int = int(filtered)
                _write_json(
                    zip_file,
                    f"{prefix}.count_ym.json",
                    {"ym": ym_env, "count": filtered_int},
                )
                table_summary.setdefault("filters", {})["ym"] = {
                    "value": ym_env,
                    "count": filtered_int,
                }

        if duckdb_indexes:
            formatted = []
            for index_name, sql in duckdb_indexes:
                formatted.append(
                    {
                        "index_name": index_name,
                        "sql": sql,
                        "columns": _parse_index_columns(sql),
                    }
                )
            table_summary["indexes"] = formatted
        elif indexes:
            table_summary["indexes"] = [list(row) for row in indexes]

        summary["tables"].append(table_summary)

    if not summary["tables"]:
        summary.pop("tables")
    return summary


def _record_tests(zip_file: ZipFile) -> Mapping[str, object]:
    junit_path = ROOT / "pytest-junit.xml"
    summary: MutableMapping[str, object] = {"failures": []}
    if not junit_path.exists():
        return summary

    zip_file.write(junit_path, arcname="diagnostics/tests/junit.xml")

    try:
        tree = ET.parse(junit_path)
    except ET.ParseError as exc:  # pragma: no cover - diagnostics only
        payload = {"error": f"parse error: {exc}"}
        _write_json(zip_file, "diagnostics/tests/failures.json", payload)
        summary["error"] = payload["error"]
        return summary

    failures: list[Mapping[str, object]] = []
    for case in tree.iterfind(".//testcase"):
        failure = case.find("failure") or case.find("error")
        if failure is None:
            continue
        nodeid = "::".join(
            part for part in (case.get("classname", ""), case.get("name", "")) if part
        )
        text = (failure.text or failure.get("message") or "").strip()
        first_line = text.splitlines()[0] if text else "(no message)"
        failures.append(
            {
                "nodeid": nodeid,
                "message": first_line,
                "file": case.get("file"),
                "line": case.get("line"),
            }
        )

    if failures:
        _write_json(zip_file, "diagnostics/tests/failures.json", failures)
    else:
        _write_json(zip_file, "diagnostics/tests/failures.json", [])

    summary["failures"] = failures

    stdout_candidate = ROOT / "pytest-stdout.txt"
    if stdout_candidate.exists():
        zip_file.write(stdout_candidate, arcname="diagnostics/tests/pytest-stdout.txt")
        summary["pytest_stdout"] = str(stdout_candidate)

    return summary


def _build_env_snapshot(
    duckdb_version: str | None, suite: str | None, timestamp: str
) -> Mapping[str, object]:
    env_subset = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("GITHUB_")
        or key.startswith("MATRIX_")
        or key.startswith("RESOLVER_")
        or key in {"PYTHONPATH", "VIRTUAL_ENV"}
    }
    env_subset["PYTHON_EXECUTABLE"] = sys.executable
    env_subset["PYTHON_VERSION"] = sys.version
    env_subset["DUCKDB_VERSION"] = duckdb_version
    env_subset["SUITE"] = suite
    env_subset["TIMESTAMP_UTC"] = timestamp
    pip_listing = _pip_duckdb_listing()
    if pip_listing:
        env_subset["PIP_DUCKDB"] = pip_listing
    return env_subset


def _render_summary(
    *,
    suite: str | None,
    duckdb_version: str | None,
    timestamp: str,
    git_meta: Mapping[str, object],
    files_info: Mapping[str, Mapping[str, object]],
    db_summary: Mapping[str, object],
    test_summary: Mapping[str, object],
) -> str:
    lines: list[str] = []
    lines.append("# Resolver CI Diagnostics Summary")
    lines.append("")
    lines.append(f"* Generated: {timestamp}")
    lines.append(f"* Suite: {suite or '(unknown suite)'}")
    lines.append(f"* DuckDB version: {duckdb_version or 'unknown'}")
    for key in ("git_head", "pr_head_sha", "github_sha"):
        value = git_meta.get(key)
        if value:
            lines.append(f"* {key}: {value}")

    duckdb_stamp = files_info.get("duckdb_io.py")
    if duckdb_stamp:
        resolved = duckdb_stamp.get("import_resolved_path") or duckdb_stamp.get("path")
        sha = duckdb_stamp.get("sha256", "missing")
        lines.append("")
        lines.append("## resolver.db.duckdb_io")
        lines.append(f"- Path: {resolved}")
        lines.append(f"- SHA256: {sha}")

    if db_summary.get("error"):
        lines.append("")
        lines.append("## Database Snapshot")
        lines.append(f"- Error: {db_summary['error']}")
    elif db_summary.get("tables"):
        lines.append("")
        lines.append("## Database Snapshot")
        resolved_path = db_summary.get("resolved_path")
        if resolved_path:
            lines.append(f"- Resolved path: {resolved_path}")
        for table in db_summary["tables"]:
            lines.append("")
            lines.append(f"### Table `{table['name']}`")
            if table.get("row_count") is not None:
                lines.append(f"- Row count: {table['row_count']}")
            if table.get("filters"):
                for key, payload in table["filters"].items():
                    lines.append(
                        f"- {key}={payload['value']} count={payload['count']}"
                    )
            if table.get("columns"):
                joined = ", ".join(table["columns"])
                lines.append(f"- Columns: {joined}")
            if table.get("indexes"):
                lines.append("- Indexes:")
                for index in table["indexes"]:
                    if isinstance(index, dict):
                        cols = ", ".join(index.get("columns") or [])
                        sql = index.get("sql")
                        lines.append(
                            f"  - {index.get('index_name')}: ({cols}) :: {sql}"
                        )
                    else:
                        lines.append(f"  - {index}")
            for key in (
                "table_info_error",
                "indexes_error",
                "duckdb_indexes_error",
                "row_count_error",
                "row_count_ym_error",
            ):
                if table.get(key):
                    lines.append(f"- {key.replace('_', ' ')}: {table[key]}")
    else:
        lines.append("")
        lines.append("## Database Snapshot")
        lines.append("- No focus tables present")

    failures = test_summary.get("failures") or []
    lines.append("")
    lines.append("## Test Failures")
    if failures:
        for entry in failures:
            nodeid = entry.get("nodeid", "<unknown>")
            message = entry.get("message", "")
            lines.append(f"- {nodeid}: {message}")
    elif test_summary.get("error"):
        lines.append(f"- Error: {test_summary['error']}")
    else:
        lines.append("- None")

    return "\n".join(lines) + "\n"


def build_bundle(
    *,
    out_path: Path,
    db_url: str | None,
    suite: str | None,
    duckdb_version: str | None,
) -> Path:
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = _dt.datetime.utcnow().isoformat() + "Z"
    git_meta = _gather_git_meta()
    env_snapshot = _build_env_snapshot(duckdb_version, suite, timestamp)

    files_info: dict[str, Mapping[str, object]] = {}

    with ZipFile(out_path, "w", ZIP_DEFLATED) as zip_file:
        meta_lines = [
            f"created_at={timestamp}",
            *(f"{key}={value}" for key, value in git_meta.items()),
        ]
        if suite:
            meta_lines.append(f"suite={suite}")
        if duckdb_version:
            meta_lines.append(f"duckdb_version={duckdb_version}")
        if db_url:
            meta_lines.append(f"db_url={db_url}")
        zip_file.writestr("diagnostics/meta.txt", "\n".join(meta_lines) + "\n")

        _write_json(zip_file, "diagnostics/env.json", env_snapshot)

        files_root = "diagnostics/files"
        for rel_path, import_path in FILES_TO_STAMP:
            payload = _stamp_file(rel_path, import_path)
            files_info[Path(rel_path).name] = payload
            arcname = f"{files_root}/{Path(rel_path).name}.json"
            _write_json(zip_file, arcname, payload)

        db_summary = _record_db_section(zip_file, db_url, duckdb_version)
        test_summary = _record_tests(zip_file)

        summary_text = _render_summary(
            suite=suite,
            duckdb_version=duckdb_version,
            timestamp=timestamp,
            git_meta=git_meta,
            files_info=files_info,
            db_summary=db_summary,
            test_summary=test_summary,
        )
        zip_file.writestr("diagnostics/SUMMARY.md", summary_text)

    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=DEFAULT_BUNDLE_NAME,
        help="Path to the output diagnostics zip (default: diagnostics.zip)",
    )
    parser.add_argument("--db-url", default=os.environ.get("RESOLVER_DB_URL"))
    parser.add_argument("--suite", default=None)
    parser.add_argument("--duckdb-version", default=None)

    args = parser.parse_args(argv)
    out_path = Path(args.out)

    build_bundle(
        out_path=out_path,
        db_url=args.db_url,
        suite=args.suite,
        duckdb_version=args.duckdb_version,
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
