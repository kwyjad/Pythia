from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

# Core tables we expect; we'll also include any other facts_* we find.
KEY_TABLES = ("facts_raw", "facts_resolved", "facts_monthly_deltas")


def format_header(title: str) -> str:
    return f"== {title} =="


def write_report(out_path: Path, lines: list[str], payload: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text_lines = lines + ["", json.dumps(payload, indent=2, sort_keys=True)]
    out_path.write_text("\n".join(text_lines), encoding="utf-8")


def enumerate_tables(con: Any) -> tuple[list[str], str | None]:
    try:
        rows = con.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
            """
        ).fetchall()
    except Exception as exc:
        return [], str(exc)
    names = sorted({row[0] for row in rows if row and row[0]})
    return names, None


def select_tables(all_tables: Iterable[str]) -> list[str]:
    targets = list(KEY_TABLES)
    for name in all_tables:
        if name.startswith("facts_") and name not in targets:
            targets.append(name)
    # de-dupe, keep order-ish
    seen: dict[str, None] = {}
    return [t for t in targets if not (t in seen or seen.setdefault(t, None))]


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
        "all_tables": [],
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

    catalog_exc = getattr(duckdb, "CatalogException", Exception)

    try:
        con = duckdb.connect(str(db_path))
    except Exception as exc:
        lines.append(f"failed to connect: {exc}")
        payload["error"] = f"connect failed: {exc}"
        write_report(out_path, lines, payload)
        return 0

    try:
        try:
            con.execute("PRAGMA show_indexes")
        except catalog_exc:
            pass
        except Exception:
            pass

        table_names, table_err = enumerate_tables(con)
        if table_err:
            lines.append(f"failed to enumerate tables: {table_err}")
            payload["error"] = f"enumerate tables failed: {table_err}"
            write_report(out_path, lines, payload)
            return 0
        payload["all_tables"] = table_names

        targets = select_tables(table_names)
        if not targets:
            lines.append("no matching tables (facts_*) found")
            write_report(out_path, lines, payload)
            return 0

        for table in targets:
            info: dict[str, Any] = {"exists": False, "rows": None}
            if table not in table_names:
                lines.append(f"{table}: missing")
                payload["tables"][table] = info
                continue
            info["exists"] = True
            try:
                count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                rows = int(count)
                lines.append(f"{table}: {rows} rows")
                info["rows"] = rows
            except Exception as exc:
                lines.append(f"{table}: failed to count rows ({exc})")
                info["error"] = str(exc)
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
