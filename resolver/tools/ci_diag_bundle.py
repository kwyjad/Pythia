"""Create an exhaustive CI diagnostics bundle with a dream SUMMARY.md."""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib
import json
import os
import platform
import re
import sys
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from zipfile import ZIP_DEFLATED, ZipFile

try:
    import duckdb  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency for diagnostics
    duckdb = None  # type: ignore[assignment]
    DUCKDB_AVAILABLE = False
    _DUCKDB_IMPORT_ERROR: Exception | None = exc
else:  # pragma: no branch - cache import result
    DUCKDB_AVAILABLE = True
    _DUCKDB_IMPORT_ERROR = None

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE_NAME = "diagnostics.zip"
FILES_TO_STAMP: tuple[tuple[str, str], ...] = (
    ("resolver/db/duckdb_io.py", "resolver.db.duckdb_io"),
    ("resolver/tools/export_facts.py", "resolver.tools.export_facts"),
    ("resolver/query/db_reader.py", "resolver.query.db_reader"),
    ("resolver/query/selectors.py", "resolver.query.selectors"),
    ("resolver/cli/resolver_cli.py", "resolver.cli.resolver_cli"),
    ("resolver/db/schema.sql", "resolver.db.schema"),
)
DEFAULT_EXPECTED_KEYS = {
    "facts_resolved": ["ym", "iso3", "hazard_code", "metric", "series_semantics"],
    "facts_deltas": ["ym", "iso3", "hazard_code", "metric"],
}
MAX_DB_SCAN_RESULTS = 200


def _now_utc_iso() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat()


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


@dataclass
class FileStamp:
    rel_path: str
    import_path: str
    exists: bool
    absolute_path: str | None
    sha256: str | None
    size: int | None
    import_resolved_path: str | None

    @classmethod
    def create(cls, rel_path: str, import_path: str) -> "FileStamp":
        path = ROOT / rel_path
        exists = path.exists()
        absolute_path = str(path.resolve()) if exists else str(path)
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest() if exists else None
        size = path.stat().st_size if exists else None
        import_resolved = _resolve_import_path(import_path)
        return cls(
            rel_path=rel_path,
            import_path=import_path,
            exists=exists,
            absolute_path=absolute_path,
            sha256=sha256,
            size=size,
            import_resolved_path=import_resolved,
        )


def _write_text(zip_file: ZipFile, arcname: str, text: str) -> None:
    if not text.endswith("\n"):
        text += "\n"
    zip_file.writestr(arcname, text)


def _write_json(zip_file: ZipFile, arcname: str, payload: Mapping[str, object]) -> None:
    _write_text(zip_file, arcname, json.dumps(payload, sort_keys=True, indent=2, default=str))


def _safe_env_var(key: str) -> str | None:
    value = os.environ.get(key)
    return value if value else None


def _gather_git_meta() -> Mapping[str, str | None]:
    def _run(cmd: Iterable[str]) -> str | None:
        try:
            out = sys.modules.get("subprocess")
            if out is None:
                import subprocess as _subprocess  # pragma: no cover - fallback
            else:
                _subprocess = sys.modules["subprocess"]
            proc = _subprocess.run(list(cmd), capture_output=True, text=True, check=False)
        except Exception:  # pragma: no cover - diagnostics only
            return None
        for stream in (proc.stdout, proc.stderr):
            if stream:
                data = stream.strip()
                if data:
                    return data
        return None

    return {
        "git_head": _run(["git", "rev-parse", "HEAD"]),
        "git_show": _run(["git", "show", "-s", "--date=iso", "--format=%H %cd %s"]),
        "github_sha": _safe_env_var("GITHUB_SHA"),
        "github_ref": _safe_env_var("GITHUB_REF"),
        "github_run_id": _safe_env_var("GITHUB_RUN_ID"),
        "github_run_attempt": _safe_env_var("GITHUB_RUN_ATTEMPT"),
        "github_actor": _safe_env_var("GITHUB_ACTOR"),
        "pr_head_sha": _safe_env_var("PR_HEAD_SHA") or _safe_env_var("GITHUB_HEAD_REF"),
    }


def _pip_duckdb_listing() -> str | None:
    try:
        import subprocess

        proc = subprocess.run(
            [sys.executable, "-m", "pip", "show", "duckdb"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:  # pragma: no cover - diagnostics only
        return None
    data = proc.stdout.strip() or proc.stderr.strip()
    return data or None


def _default_scan_paths() -> list[Path]:
    workspace = Path.cwd()
    candidates = [workspace, workspace / ".ci"]
    runner_temp = os.environ.get("RUNNER_TEMP")
    if runner_temp:
        candidates.append(Path(runner_temp))
    candidates.extend([Path("/tmp/pytest-of-runner"), Path("/tmp")])
    paths: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except FileNotFoundError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        paths.append(resolved)
    return paths


def _normalize_duckdb_url(url: str | None) -> Path | None:
    if not url:
        return None
    if url.startswith("duckdb:///"):
        # duckdb:///path.duckdb or duckdb:///:memory:
        rest = url[len("duckdb:///") :]
        if rest in {":memory:", "memory", ""}:
            return None
        return Path(rest).expanduser().resolve()
    if url.startswith("duckdb://"):
        rest = url[len("duckdb://") :]
        if rest.startswith("/"):
            return Path(rest).expanduser().resolve()
    if url.endswith(".duckdb"):
        return Path(url).expanduser().resolve()
    return None


def _discover_db_paths(db_url: str | None, scan_paths: Sequence[Path]) -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()

    direct = _normalize_duckdb_url(db_url)
    if direct and direct.exists():
        seen.add(direct)
        discovered.append(direct)

    for root in scan_paths:
        if len(discovered) >= MAX_DB_SCAN_RESULTS:
            break
        if not root.exists():
            continue
        try:
            iterator = root.rglob("*.duckdb")
        except PermissionError:  # pragma: no cover - diagnostics only
            continue
        for path in iterator:
            if len(discovered) >= MAX_DB_SCAN_RESULTS:
                break
            try:
                resolved = path.resolve()
            except FileNotFoundError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            discovered.append(resolved)
    return discovered


def _parse_expression_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(part).strip().strip("'\"") for part in value]
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [part.strip().strip("'\"") for part in inner.split(",")]
    return [text.strip().strip("'\"")]


def _format_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    if not rows:
        return "(no rows)"
    header_line = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body_lines = [
        "| " + " | ".join("" if value is None else str(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join([header_line, divider, *body_lines])


@dataclass
class TableReport:
    name: str
    columns: list[str] = field(default_factory=list)
    row_count: int | None = None
    indexes: list[dict[str, object]] = field(default_factory=list)
    constraints: list[dict[str, object]] = field(default_factory=list)
    key_check: dict[str, object] | None = None
    ym_counts: list[Sequence[object]] = field(default_factory=list)
    as_of_check: dict[str, object] | None = None
    preview_columns: list[str] = field(default_factory=list)
    preview_rows: list[Sequence[object]] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)


@dataclass
class DBReport:
    path: Path
    size: int | None = None
    sha256: str | None = None
    error: str | None = None
    tables: list[TableReport] = field(default_factory=list)


def _inspect_db(path: Path, expected_keys: Mapping[str, Sequence[str]], zip_file: ZipFile) -> DBReport:
    report = DBReport(path=path)
    if not path.exists():
        report.error = "file not found"
        return report

    if not DUCKDB_AVAILABLE or duckdb is None:
        reason = "duckdb import failed"
        if _DUCKDB_IMPORT_ERROR is not None:
            reason = f"duckdb import failed: {_DUCKDB_IMPORT_ERROR}"
        report.error = reason
        return report

    try:
        conn = duckdb.connect(database=str(path), read_only=True)
    except Exception as exc:  # pragma: no cover - diagnostics only
        report.error = f"connect failed: {exc}"
        return report

    try:
        report.size = path.stat().st_size
        report.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:  # pragma: no cover - diagnostics only
        pass

    db_dir = Path("diagnostics/db") / _safe_db_dirname(path)

    try:
        show_tables = conn.execute("PRAGMA show_tables").fetchall()
        _write_text(
            zip_file,
            str(db_dir / "show_tables.txt"),
            json.dumps(show_tables, indent=2, default=str),
        )
    except Exception as exc:  # pragma: no cover - diagnostics only
        report.error = f"PRAGMA show_tables failed: {exc}"
        conn.close()
        return report

    table_names = [row[0] for row in show_tables if row]

    # Gather indexes/constraints once for reuse.
    indexes_data: dict[str, list[dict[str, object]]] = {}
    try:
        rows = conn.execute(
            "SELECT table_name, index_name, is_unique, is_primary, expressions, sql "
            "FROM duckdb_indexes()"
        ).fetchall()
        for row in rows:
            table_name = row[0]
            indexes_data.setdefault(table_name, []).append(
                {
                    "index_name": row[1],
                    "is_unique": bool(row[2]),
                    "is_primary": bool(row[3]),
                    "expressions": _parse_expression_list(row[4]),
                    "sql": row[5],
                    "columns_ordered": [],
                }
            )
    except Exception as exc:  # pragma: no cover - diagnostics only
        indexes_error = str(exc)
        indexes_data["__error__"] = ["duckdb_indexes failed: " + indexes_error]
        _write_text(
            zip_file,
            str(db_dir / "duckdb_indexes.error.txt"),
            indexes_error,
        )

    constraints_data: dict[str, list[dict[str, object]]] = {}
    try:
        rows = conn.execute(
            "SELECT table_name, constraint_name, constraint_type, constraint_column_names "
            "FROM duckdb_constraints()"
        ).fetchall()
        for row in rows:
            table_name = row[0]
            constraints_data.setdefault(table_name, []).append(
                {
                    "constraint_name": row[1],
                    "constraint_type": row[2],
                    "columns": _parse_expression_list(row[3]),
                }
            )
    except Exception as exc:  # pragma: no cover - diagnostics only
        constraints_error = str(exc)
        constraints_data["__error__"] = ["duckdb_constraints failed: " + constraints_error]
        _write_text(
            zip_file,
            str(db_dir / "duckdb_constraints.error.txt"),
            constraints_error,
        )

    for table in table_names:
        table_dir = db_dir / table
        table_report = TableReport(name=table)

        try:
            table_info = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
            _write_text(
                zip_file,
                str(table_dir.with_suffix(".table_info.txt")),
                json.dumps(table_info, indent=2, default=str),
            )
            table_report.columns = [row[1] for row in table_info]
        except Exception as exc:  # pragma: no cover - diagnostics only
            table_report.errors["table_info"] = str(exc)

        try:
            indexes = conn.execute(f"PRAGMA indexes('{table}')").fetchall()
            _write_text(
                zip_file,
                str(table_dir.with_suffix(".indexes.txt")),
                json.dumps(indexes, indent=2, default=str),
            )
        except Exception as exc:  # pragma: no cover - diagnostics only
            table_report.errors["indexes"] = str(exc)

        table_report.indexes = indexes_data.get(table, [])
        _write_text(
            zip_file,
            str(table_dir.with_suffix(".duckdb_indexes.txt")),
            json.dumps(table_report.indexes, indent=2, default=str),
        )
        for idx in table_report.indexes:
            idx_name = idx.get("index_name")
            if not idx_name:
                continue
            safe_name = _safe_index_filename(str(idx_name))
            try:
                info_rows = conn.execute(
                    f"PRAGMA index_info('{idx_name}')"
                ).fetchall()
            except Exception as exc:  # pragma: no cover - diagnostics only
                idx["index_info_error"] = str(exc)
                _write_text(
                    zip_file,
                    str(table_dir / f"{safe_name}.index_info.error.txt"),
                    str(exc),
                )
                continue
            serialisable = [tuple(row) for row in info_rows]
            ordered_columns: list[str] = []
            for row in info_rows:
                if len(row) >= 3 and row[2] is not None:
                    ordered_columns.append(str(row[2]))
                elif len(row) >= 2 and row[1] is not None:
                    ordered_columns.append(str(row[1]))
            idx["columns_ordered"] = ordered_columns
            _write_text(
                zip_file,
                str(table_dir / f"{safe_name}.index_info.txt"),
                json.dumps(serialisable, indent=2, default=str),
            )

        table_report.constraints = constraints_data.get(table, [])
        _write_text(
            zip_file,
            str(table_dir.with_suffix(".constraints.txt")),
            json.dumps(table_report.constraints, indent=2, default=str),
        )

        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            table_report.row_count = int(count)
            _write_text(
                zip_file,
                str(table_dir.with_suffix(".count.txt")),
                json.dumps({"count": table_report.row_count}),
            )
        except Exception as exc:  # pragma: no cover - diagnostics only
            table_report.errors["count"] = str(exc)

        if table_report.columns:
            order_cols = list(range(1, min(len(table_report.columns), 4) + 1))
            order_clause = "ORDER BY " + ", ".join(str(i) for i in order_cols) if order_cols else ""
            query = f"SELECT * FROM {table} {order_clause} LIMIT 5"
            try:
                rows = conn.execute(query).fetchall()
                table_report.preview_columns = table_report.columns
                table_report.preview_rows = rows
                _write_text(
                    zip_file,
                    str(table_dir.with_suffix(".preview.txt")),
                    json.dumps(
                        {
                            "columns": table_report.preview_columns,
                            "rows": rows,
                        },
                        indent=2,
                        default=str,
                    ),
                )
            except Exception as exc:  # pragma: no cover - diagnostics only
                table_report.errors["preview"] = str(exc)

        expected = expected_keys.get(table)
        if expected:
            match = None
            details = []
            for idx in table_report.indexes:
                if not (idx.get("is_unique") or idx.get("is_primary")):
                    continue
                columns_pref = idx.get("columns_ordered") or idx.get("expressions") or []
                cols = list(columns_pref)
                details.append(
                    f"index {idx.get('index_name')} columns={cols} unique={idx.get('is_unique')} primary={idx.get('is_primary')}"
                )
                if cols == list(expected):
                    match = {
                        "type": "index",
                        "name": idx.get("index_name"),
                        "columns": cols,
                    }
                    break
            if not match:
                for constraint in table_report.constraints:
                    ctype = (constraint.get("constraint_type") or "").upper()
                    if ctype not in {"UNIQUE", "PRIMARY KEY"}:
                        continue
                    cols = list(constraint.get("columns") or [])
                    details.append(
                        f"constraint {constraint.get('constraint_name')} type={ctype} columns={cols}"
                    )
                    if cols == list(expected):
                        match = {
                            "type": "constraint",
                            "name": constraint.get("constraint_name"),
                            "columns": cols,
                        }
                        break
            table_report.key_check = {
                "expected": list(expected),
                "status": "PASS" if match else "FAIL",
                "matched": match,
                "details": details,
            }

        if table in {"facts_resolved", "facts_deltas"}:
            try:
                ym_rows = conn.execute(
                    f"SELECT ym, COUNT(*) AS n FROM {table} GROUP BY 1 ORDER BY 1 LIMIT 20"
                ).fetchall()
                table_report.ym_counts = ym_rows
                _write_text(
                    zip_file,
                    str(table_dir.with_suffix(".ym_counts.txt")),
                    json.dumps(ym_rows, indent=2, default=str),
                )
            except Exception:
                pass

        if table == "facts_deltas" and table_report.columns and "as_of" in table_report.columns:
            try:
                as_of_check = conn.execute(
                    "SELECT COUNT(*) FROM facts_deltas WHERE as_of = (SELECT MAX(as_of) FROM facts_deltas)"
                ).fetchone()[0]
                total = table_report.row_count or 0
                table_report.as_of_check = {
                    "rows_at_latest_as_of": int(as_of_check),
                    "total": total,
                }
                _write_text(
                    zip_file,
                    str(table_dir.with_suffix(".as_of_check.txt")),
                    json.dumps(table_report.as_of_check, indent=2, default=str),
                )
            except Exception:
                pass

        if table_report.errors:
            _write_json(zip_file, str(table_dir.with_suffix(".errors.json")), table_report.errors)

        report.tables.append(table_report)

    conn.close()
    return report


def _safe_db_dirname(path: Path) -> str:
    stem = path.name or "database"
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    return f"{stem}-{digest}"


def _safe_index_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return safe or "index"


def _build_env_snapshot(
    *,
    duckdb_version: str | None,
    suite: str | None,
    timestamp: str,
    scan_paths: Sequence[Path],
) -> Mapping[str, object]:
    env_subset = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("GITHUB_")
        or key.startswith("MATRIX_")
        or key.startswith("RESOLVER_")
        or key in {"PYTHONPATH", "VIRTUAL_ENV", "PR_HEAD_SHA", "RUNNER_TEMP"}
    }
    env_subset["PYTHON_EXECUTABLE"] = sys.executable
    env_subset["PYTHON_VERSION"] = sys.version
    env_subset["OS"] = platform.platform()
    env_subset["DUCKDB_VERSION"] = duckdb_version
    env_subset["SUITE"] = suite
    env_subset["TIMESTAMP_UTC"] = timestamp
    env_subset["SCAN_PATHS"] = [str(path) for path in scan_paths]
    pip_listing = _pip_duckdb_listing()
    if pip_listing:
        env_subset["PIP_DUCKDB"] = pip_listing
    return env_subset


def _parse_junit(junit_path: Path, zip_file: ZipFile) -> Mapping[str, object]:
    summary: dict[str, object] = {
        "total": 0,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "failure_details": [],
    }
    if not junit_path.exists():
        return summary

    import xml.etree.ElementTree as ET

    try:
        tree = ET.parse(junit_path)
    except ET.ParseError as exc:  # pragma: no cover - diagnostics only
        summary["error"] = f"failed to parse junit xml: {exc}"
        _write_json(zip_file, "diagnostics/tests/failures.json", summary)
        return summary

    root = tree.getroot()
    suites = list(root.iter("testsuite")) if root.tag != "testsuite" else [root]
    for suite in suites:
        summary["total"] += int(suite.get("tests", "0"))
        summary["failures"] += int(suite.get("failures", "0"))
        summary["errors"] += int(suite.get("errors", "0"))
        summary["skipped"] += int(suite.get("skipped", "0"))

    details = []
    for case in tree.iterfind(".//testcase"):
        failure = case.find("failure") or case.find("error")
        if failure is None:
            continue
        nodeid_parts = [part for part in (case.get("classname"), case.get("name")) if part]
        nodeid = "::".join(nodeid_parts) or case.get("name") or "<unknown>"
        text = (failure.text or failure.get("message") or "").strip()
        lines = text.splitlines()
        first_line = lines[0] if lines else "(no message)"
        short_trace = "\n".join(lines[:8])
        details.append(
            {
                "nodeid": nodeid,
                "message": first_line,
                "short_trace": short_trace,
                "file": case.get("file"),
                "line": case.get("line"),
            }
        )

    summary["failure_details"] = details
    _write_json(zip_file, "diagnostics/tests/failures.json", details)
    zip_file.write(junit_path, arcname="diagnostics/tests/junit.xml")
    return summary


def _render_summary(
    *,
    timestamp: str,
    suite: str | None,
    duckdb_version: str | None,
    git_meta: Mapping[str, object],
    env_snapshot: Mapping[str, object],
    file_stamps: Sequence[FileStamp],
    db_reports: Sequence[DBReport],
    junit_summary: Mapping[str, object],
    pytest_stdout_path: Path | None,
) -> str:
    lines: list[str] = []
    lines.append("# Resolver CI Dream Diagnostics Summary")
    lines.append("")
    lines.append("Paste this entire SUMMARY.md into chat; no other logs are needed.")
    lines.append("")
    lines.append(f"* Generated: {timestamp}")
    lines.append(f"* Suite: {suite or '(unknown suite)'}")
    lines.append(f"* DuckDB version: {duckdb_version or 'unknown'}")
    lines.append(f"* Python: {env_snapshot.get('PYTHON_VERSION', 'unknown')}")
    lines.append(f"* OS: {env_snapshot.get('OS', 'unknown')}")
    lines.append("")
    lines.append("## Git & Environment")
    for key in (
        "git_head",
        "git_show",
        "pr_head_sha",
        "github_sha",
        "github_ref",
        "github_run_id",
        "github_run_attempt",
        "github_actor",
    ):
        value = git_meta.get(key)
        if value:
            lines.append(f"- {key}: {value}")
    lines.append(f"- Python executable: {env_snapshot.get('PYTHON_EXECUTABLE')}")
    lines.append(f"- RESOLVER_DB_URL: {os.environ.get('RESOLVER_DB_URL', '(unset)')}")
    lines.append(f"- Scan paths: {', '.join(env_snapshot.get('SCAN_PATHS', []))}")
    pip_duckdb = env_snapshot.get("PIP_DUCKDB")
    if pip_duckdb:
        lines.append("- pip show duckdb:\n````\n" + str(pip_duckdb) + "\n````")

    lines.append("")
    lines.append("## Files Stamped")
    if file_stamps:
        lines.append("| Module | Absolute Path | SHA256 | Exists |")
        lines.append("| --- | --- | --- | --- |")
        for stamp in file_stamps:
            path = stamp.import_resolved_path or stamp.absolute_path or stamp.rel_path
            sha = stamp.sha256 or "<missing>"
            exists = "yes" if stamp.exists else "no"
            lines.append(f"| `{stamp.import_path}` | `{path}` | `{sha}` | {exists} |")
    else:
        lines.append("(no files stamped)")

    lines.append("")
    lines.append("## DuckDB Databases Discovered")
    if not db_reports:
        lines.append("- No *.duckdb files were discovered in the configured scan paths.")
    for report in db_reports:
        lines.append("")
        lines.append(f"### `{report.path}`")
        if report.error:
            lines.append(f"- Error: {report.error}")
            continue
        lines.append(f"- Size: {report.size} bytes")
        lines.append(f"- SHA256: `{report.sha256}`")
        if not report.tables:
            lines.append("- No tables found")
            continue
        for table in report.tables:
            lines.append("")
            lines.append(f"#### Table `{table.name}`")
            if table.row_count is not None:
                lines.append(f"- Row count: {table.row_count}")
            if table.columns:
                lines.append(f"- Columns: {', '.join(table.columns)}")
            if table.key_check:
                status = table.key_check.get("status")
                emoji = "✅" if status == "PASS" else "❌"
                expected = table.key_check.get("expected")
                if table.key_check.get("matched"):
                    matched = table.key_check["matched"]
                    lines.append(
                        f"- Key check: {emoji} {status} via {matched.get('type')} `{matched.get('name')}` with columns {matched.get('columns')}"
                    )
                else:
                    lines.append(
                        f"- Key check: {emoji} {status} (expected {expected}; see diagnostics for details)"
                    )
            if table.indexes:
                lines.append("- Indexes:")
                for idx in table.indexes:
                    ordered = idx.get("columns_ordered") or []
                    expressions = idx.get("expressions") or []
                    cols = ", ".join(ordered or expressions)
                    unique = "unique" if idx.get("is_unique") else "non-unique"
                    primary = " primary" if idx.get("is_primary") else ""
                    lines.append(
                        f"  - `{idx.get('index_name')}` ({unique}{primary}) columns: [{cols}]"
                    )
                    if ordered and expressions and ordered != expressions:
                        lines.append(
                            f"    expressions order (from duckdb_indexes): [{', '.join(expressions)}]"
                        )
            if table.constraints:
                lines.append("- Constraints:")
                for constraint in table.constraints:
                    cols = ", ".join(constraint.get("columns") or [])
                    lines.append(
                        f"  - `{constraint.get('constraint_name')}` ({constraint.get('constraint_type')}) columns: [{cols}]"
                    )
            if table.ym_counts:
                lines.append("- Rows per ym (up to 20):")
                lines.append(_format_markdown_table(["ym", "count"], table.ym_counts))
            if table.as_of_check:
                lines.append(
                    f"- Rows at latest as_of: {table.as_of_check['rows_at_latest_as_of']} of {table.as_of_check['total']}"
                )
            if table.preview_rows:
                lines.append("- Preview (first up to 5 rows):")
                lines.append(
                    _format_markdown_table(table.preview_columns, table.preview_rows)
                )
            if table.errors:
                lines.append("- Errors:")
                for key, value in table.errors.items():
                    lines.append(f"  - {key}: {value}")

    lines.append("")
    lines.append("## Test Results")
    totals_line = (
        f"- Total: {junit_summary.get('total', 0)} | "
        f"Failures: {junit_summary.get('failures', 0)} | "
        f"Errors: {junit_summary.get('errors', 0)} | "
        f"Skipped: {junit_summary.get('skipped', 0)}"
    )
    lines.append(totals_line)
    details = junit_summary.get("failure_details") or []
    if details:
        lines.append("")
        lines.append("### Failure Digest")
        lines.append("| Test | Message | Trace (first lines) |")
        lines.append("| --- | --- | --- |")
        for detail in details:
            trace = detail.get("short_trace", "").replace("|", "\\|")
            message = detail.get("message", "").replace("|", "\\|")
            lines.append(
                f"| `{detail.get('nodeid')}` | {message} | ````\n{trace}\n```` |"
            )
    else:
        lines.append("- No test failures")

    if pytest_stdout_path and pytest_stdout_path.exists():
        lines.append("")
        lines.append("## Pytest Stdout (last 200 lines)")
        try:
            content = pytest_stdout_path.read_text().splitlines()
            tail = list(islice(reversed(content), 0, 200))
            tail.reverse()
            lines.append("````")
            lines.extend(tail)
            lines.append("````")
        except Exception:  # pragma: no cover - diagnostics only
            lines.append("(failed to read pytest stdout)")

    return "\n".join(lines) + "\n"


def build_bundle(
    *,
    out_path: Path,
    db_url: str | None,
    suite: str | None,
    duckdb_version: str | None,
    scan_paths: Sequence[Path],
    pytest_stdout: Path | None,
    expected_keys: Mapping[str, Sequence[str]],
) -> Path:
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = _now_utc_iso()
    git_meta = _gather_git_meta()
    env_snapshot = _build_env_snapshot(
        duckdb_version=duckdb_version,
        suite=suite,
        timestamp=timestamp,
        scan_paths=scan_paths,
    )

    file_stamps = [FileStamp.create(rel, imp) for rel, imp in FILES_TO_STAMP]
    discovered_paths = _discover_db_paths(db_url, scan_paths)

    with ZipFile(out_path, "w", ZIP_DEFLATED) as zip_file:
        meta_lines = [
            f"created_at={timestamp}",
            *(f"{key}={value}" for key, value in git_meta.items() if value),
        ]
        if suite:
            meta_lines.append(f"suite={suite}")
        if duckdb_version:
            meta_lines.append(f"duckdb_version={duckdb_version}")
        if db_url:
            meta_lines.append(f"db_url={db_url}")
        _write_text(zip_file, "diagnostics/meta.txt", "\n".join(meta_lines))

        _write_json(zip_file, "diagnostics/env.json", env_snapshot)

        files_root = Path("diagnostics/files")
        for stamp in file_stamps:
            base = Path(stamp.rel_path).name
            _write_text(zip_file, str(files_root / f"{base}.path"), stamp.import_resolved_path or (stamp.absolute_path or stamp.rel_path))
            _write_text(zip_file, str(files_root / f"{base}.sha256"), stamp.sha256 or "<missing>")
            _write_json(
                zip_file,
                str(files_root / f"{base}.json"),
                {
                    "rel_path": stamp.rel_path,
                    "import_path": stamp.import_path,
                    "absolute_path": stamp.absolute_path,
                    "import_resolved_path": stamp.import_resolved_path,
                    "sha256": stamp.sha256,
                    "size": stamp.size,
                    "exists": stamp.exists,
                },
            )

        db_reports = [
            _inspect_db(path, expected_keys, zip_file)
            for path in discovered_paths
        ]

        junit_summary = _parse_junit(ROOT / "pytest-junit.xml", zip_file)

        if pytest_stdout and pytest_stdout.exists():
            zip_file.write(pytest_stdout, arcname="diagnostics/tests/pytest-stdout.txt")

        summary_text = _render_summary(
            timestamp=timestamp,
            suite=suite,
            duckdb_version=duckdb_version,
            git_meta=git_meta,
            env_snapshot=env_snapshot,
            file_stamps=file_stamps,
            db_reports=db_reports,
            junit_summary=junit_summary,
            pytest_stdout_path=pytest_stdout,
        )
        _write_text(zip_file, "diagnostics/SUMMARY.md", summary_text)

    return out_path


def _load_expected_keys(path: Path | None) -> Mapping[str, Sequence[str]]:
    if not path:
        return DEFAULT_EXPECTED_KEYS
    try:
        data = json.loads(path.read_text())
    except Exception:  # pragma: no cover - diagnostics only
        return DEFAULT_EXPECTED_KEYS
    parsed: dict[str, Sequence[str]] = {}
    for key, value in data.items():
        if isinstance(value, (list, tuple)):
            parsed[key] = [str(item) for item in value]
    return {**DEFAULT_EXPECTED_KEYS, **parsed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_BUNDLE_NAME)
    parser.add_argument("--db-url", default=os.environ.get("RESOLVER_DB_URL"))
    parser.add_argument("--suite", default=None)
    parser.add_argument("--duckdb-version", default=None)
    parser.add_argument(
        "--scan-path",
        dest="scan_paths",
        action="append",
        default=None,
        help="Directory to scan for *.duckdb files (may be passed multiple times)",
    )
    parser.add_argument(
        "--pytest-stdout",
        dest="pytest_stdout",
        default=None,
        help="Path to pytest stdout capture to embed in the bundle",
    )
    parser.add_argument(
        "--expected-keys-json",
        dest="expected_keys_json",
        default=None,
        help="Optional JSON file mapping table names to expected unique key column lists",
    )

    args = parser.parse_args(argv)

    if args.scan_paths:
        scan_paths = []
        for item in args.scan_paths:
            try:
                scan_paths.append(Path(item).resolve())
            except FileNotFoundError:
                continue
    else:
        scan_paths = _default_scan_paths()

    pytest_stdout = Path(args.pytest_stdout).resolve() if args.pytest_stdout else None
    expected_keys = _load_expected_keys(Path(args.expected_keys_json)) if args.expected_keys_json else DEFAULT_EXPECTED_KEYS

    out_path = Path(args.out)
    build_bundle(
        out_path=out_path,
        db_url=args.db_url,
        suite=args.suite,
        duckdb_version=args.duckdb_version,
        scan_paths=scan_paths,
        pytest_stdout=pytest_stdout,
        expected_keys=expected_keys,
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
