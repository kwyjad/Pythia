from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Iterable

SUPPORTED_SUFFIXES = {".csv", ".parquet", ".parq"}

def count_rows_csv(p: Path) -> tuple[int | None, str | None]:
    try:
        total = 0
        with p.open("r", encoding="utf-8", errors="ignore") as h:
            for total, _ in enumerate(h, start=1):
                pass
        return (max(total - 1, 0) if total else 0), None
    except Exception as exc:
        return None, f"failed to count CSV rows ({exc})"

def count_rows_parquet(p: Path) -> tuple[int | None, str | None]:
    errors: list[str] = []
    try:
        import duckdb
        try:
            con = duckdb.connect()
            rows = con.execute("SELECT COUNT(*) FROM read_parquet(?)", [str(p)]).fetchone()[0]
            con.close()
            return int(rows), None
        except Exception as exc:
            errors.append(f"duckdb failed ({exc})")
    except ModuleNotFoundError as exc:
        errors.append(f"duckdb missing ({exc})")
    try:
        import pyarrow.parquet as pq
        try:
            pf = pq.ParquetFile(str(p))
            if pf.metadata is not None:
                return int(pf.metadata.num_rows), None
        except Exception as exc:
            errors.append(f"pyarrow failed ({exc})")
    except ModuleNotFoundError as exc:
        errors.append(f"pyarrow missing ({exc})")
    try:
        import pandas as pd
        try:
            frame = pd.read_parquet(p)
            return int(frame.shape[0]), None
        except Exception as exc:
            errors.append(f"pandas failed ({exc})")
    except ModuleNotFoundError as exc:
        errors.append(f"pandas missing ({exc})")
    return None, ("; ".join(errors) if errors else "unknown failure")

def iter_canonical_files(d: Path) -> Iterable[Path]:
    for path in sorted(d.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path

def collect(d: Path) -> tuple[list[str], list[dict[str, Any]], int, int]:
    lines, entries = [], []
    total_known, unknown = 0, 0
    for path in iter_canonical_files(d):
        rel = path.relative_to(d)
        ext = path.suffix.lower()
        rows, err = (count_rows_csv(path) if ext == ".csv" else count_rows_parquet(path))
        size = path.stat().st_size
        item: dict[str, Any] = {"path": str(rel), "rows": rows, "bytes": size, "type": ext.lstrip(".")}
        if err: item["error"] = err
        if rows is None:
            unknown += 1; row_disp = "rows=?"
        else:
            total_known += rows; row_disp = f"rows={rows}"
        line = f"{rel} ({item['type']}): {row_disp} bytes={size}" + (f" error={err}" if err else "")
        lines.append(line); entries.append(item)
    return lines, entries, total_known, unknown

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Summarize canonical CSV/Parquet files for diagnostics")
    ap.add_argument("--dir", dest="directory", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    d = Path(args.directory)
    out = Path(args.out)

    header = "== Canonical directory listing =="
    lines: list[str] = [header, f"directory: {d}"]
    payload: dict[str, Any] = {"directory": str(d), "exists": d.is_dir(), "files": []}

    if not d.exists():
        lines.append("canonical directory missing; nothing to list")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines + ["", json.dumps(payload, indent=2, sort_keys=True)]), encoding="utf-8")
        return 0
    if not d.is_dir():
        lines.append("canonical path exists but is not a directory")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines + ["", json.dumps(payload, indent=2, sort_keys=True)]), encoding="utf-8")
        return 0

    file_lines, entries, total_rows, unknown_counts = collect(d)
    payload["files"] = entries
    payload["total_rows"] = total_rows
    payload["unknown_row_counts"] = unknown_counts

    if not file_lines:
        lines.append("no CSV/Parquet files found")
    else:
        lines.append(f"files: {len(file_lines)}")
        lines.extend(file_lines)
        lines.append(f"total rows (excluding headers where applicable): {total_rows}")
        if unknown_counts:
            lines.append(f"files with unknown row count: {unknown_counts}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines + ["", json.dumps(payload, indent=2, sort_keys=True)]), encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
