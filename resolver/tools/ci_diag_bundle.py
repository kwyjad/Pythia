"""Create a consolidated CI diagnostics bundle for resolver runs."""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Mapping
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


def _stamp_file(path: Path) -> Mapping[str, str | int]:
    payload = {
        "path": str(path),
        "exists": path.exists(),
    }
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


def _record_db_section(zip_file: ZipFile, db_url: str | None, duckdb_version: str | None) -> None:
    if not db_url:
        _write_json(
            zip_file,
            "diagnostics/db/db.json",
            {"error": "db_url not provided"},
        )
        return

    try:
        from resolver.db import duckdb_io
    except Exception as exc:  # pragma: no cover - diagnostics only
        _write_json(
            zip_file,
            "diagnostics/db/db.json",
            {"error": f"duckdb import failed: {exc}"},
        )
        return

    try:
        conn = duckdb_io.get_db(db_url)
    except Exception as exc:  # pragma: no cover - diagnostics only
        _write_json(
            zip_file,
            "diagnostics/db/db.json",
            {"error": f"get_db failed: {exc}"},
        )
        return

    resolved_path = getattr(conn, "_path", None) or getattr(conn, "database", None)
    meta = {
        "db_url": db_url,
        "resolved_path": resolved_path,
        "duckdb_version": duckdb_version,
    }

    db_root = "diagnostics/db"
    _write_json(zip_file, f"{db_root}/meta.json", meta)

    try:
        show_tables = conn.execute("PRAGMA show_tables").fetchall()
    except Exception as exc:  # pragma: no cover - diagnostics only
        _write_json(
            zip_file,
            f"{db_root}/show_tables.json",
            {"error": f"show_tables failed: {exc}"},
        )
        return

    _write_json(
        zip_file,
        f"{db_root}/show_tables.json",
        {"tables": show_tables},
    )

    tables = [row[0] for row in show_tables if row]
    focus_tables = [t for t in ("facts_resolved", "facts_deltas") if t in tables]
    for table in focus_tables:
        prefix = f"{db_root}/{table}"
        try:
            table_info = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        except Exception as exc:  # pragma: no cover - diagnostics only
            _write_json(zip_file, f"{prefix}.table_info.json", {"error": repr(exc)})
        else:
            _write_json(
                zip_file,
                f"{prefix}.table_info.json",
                {"columns": table_info},
            )
        try:
            indexes = conn.execute(f"PRAGMA indexes('{table}')").fetchall()
        except Exception as exc:  # pragma: no cover - diagnostics only
            _write_json(zip_file, f"{prefix}.indexes.json", {"error": repr(exc)})
        else:
            _write_json(zip_file, f"{prefix}.indexes.json", {"indexes": indexes})
        try:
            duckdb_indexes = conn.execute(
                "SELECT index_name, sql FROM duckdb_indexes() WHERE table_name = ?",
                [table],
            ).fetchall()
        except Exception as exc:  # pragma: no cover - diagnostics only
            _write_json(
                zip_file,
                f"{prefix}.duckdb_indexes.json",
                {"error": repr(exc)},
            )
        else:
            _write_json(
                zip_file,
                f"{prefix}.duckdb_indexes.json",
                {"indexes": duckdb_indexes},
            )
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except Exception as exc:  # pragma: no cover - diagnostics only
            _write_json(zip_file, f"{prefix}.count.json", {"error": repr(exc)})
        else:
            _write_json(zip_file, f"{prefix}.count.json", {"count": int(count)})

        ym_env = None
        for key in ("RESOLVER_TARGET_YM", "RESOLVER_SNAPSHOT_YM", "TARGET_YM"):
            if key in os.environ and os.environ[key]:
                ym_env = os.environ[key]
                break
        if ym_env:
            try:
                filtered = conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE ym = ?",
                    [ym_env],
                ).fetchone()[0]
            except Exception as exc:  # pragma: no cover - diagnostics only
                _write_json(
                    zip_file,
                    f"{prefix}.count_ym.json",
                    {"ym": ym_env, "error": repr(exc)},
                )
            else:
                _write_json(
                    zip_file,
                    f"{prefix}.count_ym.json",
                    {"ym": ym_env, "count": int(filtered)},
                )


def _record_tests(zip_file: ZipFile) -> None:
    junit_path = ROOT / "pytest-junit.xml"
    if not junit_path.exists():
        return

    zip_file.write(junit_path, arcname="diagnostics/tests/junit.xml")

    try:
        tree = ET.parse(junit_path)
    except ET.ParseError as exc:  # pragma: no cover - diagnostics only
        _write_json(
            zip_file,
            "diagnostics/tests/failures.json",
            {"error": f"parse error: {exc}"},
        )
        return

    failures: dict[str, str] = {}
    for case in tree.iterfind(".//testcase"):
        nodeid = case.get("classname", "") + "::" + case.get("name", "")
        failure = case.find("failure") or case.find("error")
        if failure is None:
            continue
        text = (failure.text or failure.get("message") or "").strip()
        first_line = text.splitlines()[0] if text else "(no message)"
        failures[nodeid] = first_line

    if failures:
        _write_json(
            zip_file,
            "diagnostics/tests/failures.json",
            failures,
        )


def _build_env_snapshot(duckdb_version: str | None, suite: str | None) -> Mapping[str, object]:
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
    pip_listing = _pip_duckdb_listing()
    if pip_listing:
        env_subset["PIP_DUCKDB"] = pip_listing
    return env_subset


def build_bundle(
    *,
    out_path: Path,
    db_url: str | None,
    suite: str | None,
    duckdb_version: str | None,
) -> Path:
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    git_meta = _gather_git_meta()
    env_snapshot = _build_env_snapshot(duckdb_version, suite)
    timestamp = _dt.datetime.utcnow().isoformat() + "Z"

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
            path = ROOT / rel_path
            payload = _stamp_file(path)
            payload["import_path"] = import_path
            arcname = f"{files_root}/{Path(rel_path).name}.sha256"
            _write_json(zip_file, arcname, payload)

        _record_db_section(zip_file, db_url, duckdb_version)
        _record_tests(zip_file)

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
