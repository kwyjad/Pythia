from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any, Iterable

KEY_TABLES = ("facts_raw", "facts_resolved", "facts_monthly_deltas")

def header(s: str) -> str: return f"== {s} =="

def write(out: Path, lines: list[str], payload: dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines + ["", json.dumps(payload, indent=2, sort_keys=True)]), encoding="utf-8")

def list_tables(con: Any) -> tuple[list[str], str | None]:
    try:
        rows = con.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
        """).fetchall()
    except Exception as exc:
        return [], str(exc)
    return sorted({r[0] for r in rows if r and r[0]}), None

def choose_targets(all_names: Iterable[str]) -> list[str]:
    targets = list(KEY_TABLES)
    for name in all_names:
        if name.startswith("facts_") and name not in targets:
            targets.append(name)
    return sorted(dict.fromkeys(targets))

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Emit DuckDB table counts for diagnostics")
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    db, out = Path(args.db), Path(args.out)
    lines = [header("DuckDB table counts"), f"database: {db}"]
    payload: dict[str, Any] = {"database": str(db), "exists": db.exists(), "tables": {}, "all_tables": []}

    if not db.exists():
        lines.append("database not found; skipping counts")
        write(out, lines, payload); return 0

    try:
        import duckdb
    except ModuleNotFoundError:
        lines.append("duckdb module not available; cannot inspect tables")
        payload["error"] = "duckdb module missing"
        write(out, lines, payload); return 0

    try:
        con = duckdb.connect(str(db))
    except Exception as exc:
        lines.append(f"failed to connect: {exc}")
        payload["error"] = f"connect failed: {exc}"
        write(out, lines, payload); return 0

    try:
        names, err = list_tables(con)
        if err:
            lines.append(f"failed to enumerate tables: {err}")
            payload["error"] = f"enumerate tables failed: {err}"
            write(out, lines, payload); return 0
        payload["all_tables"] = names

        for table in choose_targets(names):
            info: dict[str, Any] = {"exists": False, "rows": None}
            if table not in names:
                lines.append(f"{table}: missing")
                payload["tables"][table] = info
                continue
            info["exists"] = True
            try:
                rows = int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                lines.append(f"{table}: {rows} rows")
                info["rows"] = rows
            except Exception as exc:
                lines.append(f"{table}: failed to count rows ({exc})")
                info["error"] = str(exc)
            payload["tables"][table] = info
    finally:
        try: con.close()
        except Exception: pass

    write(out, lines, payload)
    return 0

if __name__ == "__main__":
    sys.exit(main())
