from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


TABLES = ("facts_raw", "facts_resolved", "facts_deltas")


def format_header(title: str) -> str:
    return f"== {title} =="


def write_report(out_path: Path, lines: list[str], payload: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text_lines = lines + ["", json.dumps(payload, indent=2, sort_keys=True)]
    out_path.write_text("\n".join(text_lines), encoding="utf-8")


def table_exists(con: Any, table: str) -> bool:
    try:
        result = con.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = ?
            """,
            [table],
        ).fetchone()
    except Exception:
        return False
    return bool(result and result[0])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit DuckDB table counts for diagnostics")
    parser.add_argument("--db", required=True, help="Path to the DuckDB file")
    parser.add_argument("--out", required=True, help="Where to write the diagnostics report")
    args = parser.parse_args(argv)

    db_path = Path(args.db)
    out_path = Path(args.out)

    lines: list[str] = [format_header("DuckDB table counts"), f"database: {db_path}"]
    payload: dict[str, Any] = {
        "database": str(db_path),
        "exists": db_path.exists(),
        "tables": {},
    }

    if not db_path.exists():
        lines.append("database not found; skipping counts")
        write_report(out_path, lines, payload)
        return 0

    try:
        import duckdb  # type: ignore
    except ModuleNotFoundError:
        lines.append("duckdb module not available; cannot inspect tables")
        payload["error"] = "duckdb module missing"
        write_report(out_path, lines, payload)
        return 0

    try:
        con = duckdb.connect(str(db_path))
    except Exception as exc:  # pragma: no cover - best effort diagnostics
        lines.append(f"failed to connect: {exc}")
        payload["error"] = f"connect failed: {exc}"
        write_report(out_path, lines, payload)
        return 0

    try:
        for table in TABLES:
            info: dict[str, Any] = {"exists": False, "rows": None}
            if not table_exists(con, table):
                lines.append(f"{table}: missing")
                payload["tables"][table] = info
                continue
            info["exists"] = True
            try:
                count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except Exception as exc:  # pragma: no cover - diagnostics only
                lines.append(f"{table}: failed to count rows ({exc})")
                info["error"] = str(exc)
            else:
                lines.append(f"{table}: {count} rows")
                info["rows"] = int(count)
            payload["tables"][table] = info
    finally:
        try:
            con.close()
        except Exception:
            pass

    write_report(out_path, lines, payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
